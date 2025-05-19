import argparse
import os
from typing import Tuple

import matplotlib
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA  # Import PCA here
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from sklearn.metrics import silhouette_score
matplotlib.use('Agg')  # Use Agg backend for saving files


# --- Helper Functions ---
def load_fits_data(channel_dir: str) -> Tuple[np.ndarray, dict]:
    """Load FITS data and metadata from a single channel directory.

    Loads only the *first* FITS file in the directory.
    """
    fits_files = [
        f for f in os.listdir(channel_dir) if f.endswith(".fits")
    ]
    if not fits_files:
        raise FileNotFoundError(
            f"No FITS files found in directory: {channel_dir}"
        )

    fits_path = os.path.join(
        channel_dir, fits_files[0]
    )  # Load the first FITS file
    aia_map = sunpy.map.Map(fits_path)
    return aia_map.data, aia_map.meta


def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates a circular mask for the solar disk based on metadata."""
    ny, nx = data.shape
    x_center, y_center = nx // 2, ny // 2
    cdelt1 = metadata.get("cdelt1", 1.0)  # Arcsec/pixel in X
    solar_radius_arcsec = metadata.get("rsun_obs", 960.0)  # Solar radius arcsec
    solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))

    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    return distance_from_center <= solar_radius_pixels


def preprocess_image(
    data: np.ndarray, mask: np.ndarray, size: int = 512
) -> np.ndarray:
    """Resizes the image and applies the mask."""
    resized_data = resize(
        data, (size, size), mode='reflect', anti_aliasing=True
    )
    resized_mask = resize(
        mask, (size, size), mode='reflect', anti_aliasing=False
    ) > 0.5
    masked_data = resized_data.copy()
    masked_data[~resized_mask] = np.nan  # Apply mask
    return masked_data


# --- Data Preparation and Anomaly Detection ---
def prepare_data_concatenated(
    masked_data_list: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data, handles NaNs, and scales the data."""
    stacked_data = np.stack(
        masked_data_list, axis=-1
    )  # (height, width, channels)
    reshaped_data = stacked_data.reshape(
        (-1, len(masked_data_list))
    )  # (pixels, channels)
    nan_mask = np.isnan(reshaped_data).any(axis=1)  # Rows with any NaN
    cleaned_data = reshaped_data[~nan_mask]  # Remove rows with NaNs
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask


def detect_anomalies_isolation_forest(
    data: np.ndarray, contamination: float, random_state: int
) -> np.ndarray:
    """Detects anomalies using Isolation Forest."""
    iso_forest = IsolationForest(
        contamination=contamination, random_state=random_state  # Added random_state
    )
    iso_forest.fit(data)
    return iso_forest.decision_function(data)  # Anomaly scores


def reduce_dimensionality(data: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Applies PCA to reduce the dimensionality of the dataset."""
    n_components = min(n_components, data.shape[1])  # Ensure valid n_components
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(data)


# --- Clustering Methods ---
def perform_gmm_clustering_with_search(data: np.ndarray, _: int, random_state: int = 42) -> np.ndarray:
    """Performs GMM with selection of best n_components using Silhouette Score and plots
    the scores."""
    best_score = -1
    best_labels = None
    best_n = None
    silhouette_scores = []
    n_components_range = range(2, 11)  # probar entre 2 y 10 clusters

    print("    Starting GMM hyperparameter search...")
    for n_components in n_components_range:
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=random_state)
            labels = gmm.fit_predict(data)

            if len(np.unique(labels)) <= 1:
                silhouette_scores.append(np.nan)
                continue

            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
            print(f"    n_components={n_components}, silhouette={score:.3f}")

            if score > best_score:
                best_score = score
                best_labels = labels
                best_n = n_components

        except Exception as e:
            print(f"    Skipping n_components={n_components} due to error: {e}")
            silhouette_scores.append(np.nan)
            continue

    if best_labels is None:
        print("    No valid GMM clustering found, returning noise.")
        return np.full(shape=(len(data),), fill_value=-1)

    # --- Plot Silhouette Scores ---
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(n_components_range, silhouette_scores, marker='o', linestyle='-',
                 label='Silhouette Score', color='green')
        plt.axvline(best_n, linestyle='--', color='darkgreen', label=f'Best k={best_n}')
        plt.title('GMM Silhouette Scores by Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Silhouette Score')
        plt.xticks(n_components_range)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output_clustering/gmm_silhouette_scores.png")
        plt.close()
        print("    Silhouette score plot saved to: output_clustering/gmm_silhouette_scores.png")
    except Exception as e:
        print(f"    Failed to plot silhouette scores: {e}")

    print(f"    Best GMM n_components={best_n}, silhouette={best_score:.3f}")
    return best_labels


def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray,
    image_size: int
) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels."""

    cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])
    cluster_patches = []

    if len(labels) > 0:
        n_clusters = len(np.unique(labels))
        print(f"  Number of clusters: {n_clusters}")
        # 5 Vivid cluster colors, manually selected for contrast with pastel green
        cluster_colors = [
            'magenta',      # Vivid Magenta (contrasts green well)
            'orangered',    # Vivid Orange-Red (contrasts green well)
            'blue',         # Vivid Blue (contrasts green well)
            'yellow',       # Vivid Yellow (contrasts green well)
            'cyan'          # Vivid Cyan (contrasts green well)
        ]
        cluster_cmap = matplotlib.colors.ListedColormap(
            # cap at 5 clusters for consistent color mapping
            cluster_colors[:min(n_clusters, 5)]
        )

        # Create 2D masks for easier indexing
        valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))
        anomaly_pixels_indices = np.argwhere(anomaly_mask)  # 2D indices anomalies

        # Map 2D anomaly pixel indices to rows in prepared data (as before)
        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {
            tuple(index_2d): i for i, index_2d in enumerate(valid_pixel_indices_2d)
        }

        valid_anomaly_pixel_indices = []
        for anomaly_pixel_index_2d in anomaly_pixels_indices:
            if tuple(anomaly_pixel_index_2d) in pixel_index_map:  # Ensure valid
                valid_anomaly_pixel_indices.append(anomaly_pixel_index_2d)
        valid_anomaly_pixel_indices = np.array(valid_anomaly_pixel_indices)

        if len(valid_anomaly_pixel_indices) > 0:
            for cluster_idx in range(n_clusters):
                # Get indices of pixels belonging to current cluster
                cluster_pixel_indices = valid_anomaly_pixel_indices[
                    labels == cluster_idx
                ]
                if len(cluster_pixel_indices) > 0:
                    # Assign cluster label to corresponding pixels in 2D mask
                    cluster_mask[tuple(cluster_pixel_indices.T)] = (
                        cluster_idx + 1
                    )  # +1 to avoid 0 (background)
                    cluster_color = cluster_cmap(
                        cluster_idx / n_clusters
                    )  # Get color for cluster
                    cluster_patches.append(
                        mpatches.Patch(
                            color=cluster_color,
                            label=f'Cluster {cluster_idx + 1}'
                        )
                    )

    return cluster_mask, cluster_cmap, cluster_patches, n_clusters


# --- Plotting ---

def plot_results(
    masked_data_list: list,
    cluster_mask_global: np.ndarray,
    cluster_cmap_global: matplotlib.colors.ListedColormap,
    n_clusters_global: int,
    cluster_patches_global: list,
    channel_names: list,
    anomaly_threshold: float,
    output_dir: str,
    clustering_method_name: str,  # Added clustering method name
    total_pixels: int,  # Added total_pixels
    anomaly_pixels_count: int  # Added anomaly_pixels_count
):
    """Plots and saves the results, overlaying global clusters."""
    num_rows, num_cols = 3, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), dpi=100)
    axes = axes.flatten()

    anomaly_percentage = (anomaly_pixels_count / total_pixels) * \
        100 if total_pixels > 0 else 0  # Calculate percentage
    fig.suptitle(
        f'Anomaly Detection with {
            clustering_method_name} Clustering (Max 5 Clusters)\n'  # Method in title
        f'Anomaly Threshold: {anomaly_threshold:.2f} | '
        f'Anomalous Pixels: {anomaly_pixels_count} / {total_pixels} '
        f'({anomaly_percentage:.2f}%)',  # Added anomaly info to title
        fontsize=18
    )

    for i, (masked_data, channel) in enumerate(
        zip(masked_data_list, channel_names)
    ):
        if i < num_rows * num_cols:  # Prevent index out of bounds
            ax = axes[i]

            # Display the original image with pastel green colormap
            ax.imshow(
                masked_data,
                origin='lower',
                cmap='Greens',  # Pastel green for the sun
                vmin=np.nanpercentile(masked_data, 2),
                vmax=np.nanpercentile(masked_data, 98),
                alpha=0.4  # Adjusted alpha for pastel effect
            )

            # Overlay the GLOBAL cluster mask (only color cluster areas)
            if n_clusters_global > 0:
                # Loop up to max 5 clusters
                for cluster_index in range(1, min(n_clusters_global, 5) + 1):
                    cluster_area_mask = cluster_mask_global == cluster_index
                    cluster_color = cluster_cmap_global(
                        # Corrected cmap index
                        (cluster_index - 1) / min(n_clusters_global, 5)
                    )
                    ax.imshow(
                        np.ma.masked_where(
                            ~cluster_area_mask, cluster_mask_global
                        ),
                        origin='lower',
                        cmap=matplotlib.colors.ListedColormap([cluster_color]),
                        alpha=0.7,  # Adjusted alpha for clusters
                        vmin=cluster_index - 0.5,
                        vmax=cluster_index + 0.5
                    )

            ax.set_title(
                # Clusters capped at 5 in title
                f'AIA {channel} Å (Clusters: {min(n_clusters_global, 5)})',
                color='black',
                fontsize=14,
                pad=10
            )
            ax.text(
                0.5, -0.18,
                f'Anomaly Threshold: {anomaly_threshold:.2f}',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
                color='dimgray'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add legend (only to the last used subplot)
            if (i == len(channel_names) - 1) and cluster_patches_global:

                # Modify legend handles to show max 5 clusters
                legend_handles = cluster_patches_global[:min(n_clusters_global, 5)]
                ax.legend(
                    handles=legend_handles,
                    loc='upper right',
                    fontsize='small'
                )

    # Remove any unused subplots
    for j in range(len(channel_names), num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout(
        # Adjusted rect to make space for title and legend
        rect=[0, 0, 1, 0.90], w_pad=0.1, h_pad=0.1
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    filename = os.path.join(output_dir,
                            # Method in filename
                            f"anomaly_detection_{clustering_method_name}_threshold_{anomaly_threshold:.2f}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {filename}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="SDO/AIA Anomaly Detection with Multiple Clustering Methods")
    parser.add_argument("--data_dir", type=str, default="Data/sdo_data",
                        help="Path to directory containing SDO/AIA data.")  # Updated data_dir
    parser.add_argument("--output_dir", type=str, default="./output_clustering",
                        help="Directory to save output figures.")
    parser.add_argument("--anomaly_thresholds", type=float, nargs='+',
                        # thresholds to iterate over
                        default=[0.1, 0.15, 0.2, 0.25, 0.3],
                        help="Threshold(s) for anomaly detection.")
    parser.add_argument("--n_clusters", type=int, default=5,
                        help="Number of clusters for methods requiring it.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Size to resize images to.")  # Added image_size argument
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Contamination parameter for Isolation Forest.")  # Added contamination
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")  # ADDED random_state argument
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.image_size = 512  # Define image_size

    # Load FITS data
    channels = [d for d in os.listdir(args.data_dir) if os.path.isdir(
        os.path.join(args.data_dir, d))]
    masked_data_list = []
    channel_names = []
    for channel_dir in channels:
        channel_names.append(channel_dir.split("_")[1])  # Append channel name
        data, metadata = load_fits_data(os.path.join(args.data_dir, channel_dir))
        mask = create_circular_mask(data, metadata)
        # Pass mask and image_size to preprocess
        masked_data = preprocess_image(data, mask, args.image_size)
        masked_data_list.append(masked_data)

    # Prepare Data
    prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(
        masked_data_list)
    reduced_data = reduce_dimensionality(prepared_data)

    clustering_methods = {
        "GMM": perform_gmm_clustering_with_search,
    }


    # Loop through anomaly thresholds
    for anomaly_threshold in args.anomaly_thresholds:
        print(f"Processing with anomaly threshold: {anomaly_threshold:.2f}")

        # Anomaly Detection (Isolation Forest)
        anomaly_scores = detect_anomalies_isolation_forest(
            prepared_data, args.contamination, args.random_state)  # Pass random_state

        # Create a 2D anomaly map
        anomaly_map_2d = np.full((args.image_size, args.image_size), np.nan)
        anomaly_map_2d[valid_pixel_mask.reshape(
            (args.image_size, args.image_size))] = anomaly_scores
        anomaly_mask_global = anomaly_map_2d < anomaly_threshold

        anomaly_pixels_count = np.sum(anomaly_mask_global)  # Count anomaly pixels
        total_pixels = args.image_size * args.image_size  # Calculate total pixels
        print(f"  np.sum(anomaly_mask_global) in main loop: {anomaly_pixels_count}")
        print(f"  Total pixels in image: {total_pixels}")
        anomaly_data = reduced_data[anomaly_mask_global[valid_pixel_mask.reshape(
            (args.image_size, args.image_size))].flatten()]

        # Loop through clustering methods
        for method_name, clustering_func in clustering_methods.items():
            print(f"  Applying {method_name} Clustering...")

            if method_name in ["GMM"]:  # Methods that need n_clusters parameter
                cluster_labels = clustering_func(
                    anomaly_data, args.n_clusters, args.random_state)  # Pass random_state
                # Limit clusters to max 5 for GMM
                n_clusters_to_plot = min(args.n_clusters, 5)
            elif method_name == "DBSCAN" or method_name == "MeanShift":  # Methods that DO NOT need n_clusters parameter
                cluster_labels = clustering_func(anomaly_data)
                # DBSCAN determines clusters, limit to 5
                n_clusters_to_plot = min(
                    len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0), 5)
            else:
                print(f"  Unknown clustering method: {method_name}")
                continue  # Skip to the next method

            # Create cluster mask and plot results
            cluster_mask_global, cluster_cmap_global, cluster_patches_global, n_clusters_global = create_cluster_mask(
                anomaly_mask_global, cluster_labels, valid_pixel_mask, args.image_size
            )

            plot_results(
                masked_data_list, cluster_mask_global, cluster_cmap_global, n_clusters_global,
                cluster_patches_global, channel_names, anomaly_threshold, args.output_dir, method_name, total_pixels, anomaly_pixels_count  # Pass pixel counts
            )

    print("All anomaly detection and clustering plots saved to:", args.output_dir)


if __name__ == "__main__":
    main()
