import argparse
import os
from typing import List, Tuple

import matplotlib
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

matplotlib.use('Agg')  # Use Agg backend for saving files


# --- Helper Functions ---
def load_fits_data(channel_dir: str) -> Tuple[np.ndarray, dict]:
    """Loads FITS data and metadata from a single channel directory."""
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in: {channel_dir}")
    fits_path = os.path.join(channel_dir, fits_files[0])
    aia_map = sunpy.map.Map(fits_path)
    return aia_map.data, aia_map.meta


def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates a circular mask for the solar disk based on metadata."""
    ny, nx = data.shape
    x_center, y_center = nx // 2, ny // 2
    cdelt1 = metadata.get("cdelt1", 1.0)
    solar_radius_arcsec = metadata.get("rsun_obs", 960.0)
    solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    return distance_from_center <= solar_radius_pixels


def preprocess_image(
    data: np.ndarray, mask: np.ndarray, size: int = 512
) -> np.ndarray:
    """Resizes the image and applies the mask."""
    resized_data = resize(data, (size, size), mode='reflect', anti_aliasing=True)
    resized_mask = resize(mask, (size, size), mode='reflect', anti_aliasing=False) > 0.5
    masked_data = resized_data.copy()
    masked_data[~resized_mask] = np.nan
    return masked_data


# --- Data Preparation ---
def prepare_data_concatenated(
    masked_data_list: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data, handles NaNs, and scales the data."""
    stacked_data = np.stack(masked_data_list, axis=-1)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    nan_mask = np.isnan(reshaped_data).any(axis=1)
    cleaned_data = reshaped_data[~nan_mask]
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask


# --- Anomaly Detection ---
def detect_anomalies_isolation_forest(
    data: np.ndarray, contamination: float
) -> np.ndarray:
    """Detects anomalies using Isolation Forest."""
    iso_forest = IsolationForest(
        contamination=contamination, random_state=42
    )
    iso_forest.fit(data)
    return iso_forest.decision_function(data)


# --- Clustering ---
def perform_kmeans_clustering(
    data: np.ndarray, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """Performs K-Means clustering."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )
    kmeans.fit(data)
    return kmeans.labels_, kmeans.inertia_


def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray,
    image_size: int
) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels."""
    print("-" * 20)
    print("Inside create_cluster_mask:")
    print(f"anomaly_mask.shape: {anomaly_mask.shape}")
    print(f"np.sum(anomaly_mask): {np.sum(anomaly_mask)}")

    cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])
    cluster_patches = []

    if len(labels) > 0:
        n_clusters = len(np.unique(labels))
        print(f"Number of clusters (n_clusters): {n_clusters}")
        cluster_colors = [
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f'
        ]
        cluster_cmap = matplotlib.colors.ListedColormap(
            cluster_colors[:n_clusters]
        )

        valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))
        anomaly_pixels_indices = np.argwhere(anomaly_mask)
        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {
            tuple(index_2d): i for i, index_2d in enumerate(valid_pixel_indices_2d)
        }

        valid_anomaly_pixel_indices = [
            idx for idx in anomaly_pixels_indices if tuple(idx) in pixel_index_map
        ]
        valid_anomaly_pixel_indices = np.array(valid_anomaly_pixel_indices)

        if len(valid_anomaly_pixel_indices) > 0:
            for cluster_idx in range(n_clusters):
                cluster_pixel_indices = valid_anomaly_pixel_indices[labels == cluster_idx]
                cluster_mask[tuple(cluster_pixel_indices.T)] = cluster_idx + 1
                cluster_color = cluster_cmap(cluster_idx / n_clusters)
                cluster_patches.append(
                    mpatches.Patch(
                        color=cluster_color,
                        label=f'Cluster {cluster_idx + 1}'
                    )
                )

    print("-" * 20)
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
    total_pixels: int,
    anomaly_pixels_count: int,
    cluster_pixels_counts: List[int],
    cluster_anomaly_percentages: List[float],
    clustering_method_name: str = "K-Means",
):
    """Plots and saves anomaly detection and clustering results."""
    num_rows, num_cols = 3, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), dpi=100)
    axes = axes.flatten()

    anomaly_percentage = (anomaly_pixels_count / total_pixels) * \
        100 if total_pixels > 0 else 0
    fig.suptitle(
        f'{clustering_method_name} Anomaly Clusters in SDO/AIA EUV Channels\n'
        f'Anomaly Threshold: {anomaly_threshold:.2f} | Anomalous Pixels: {
            anomaly_pixels_count}/{total_pixels} ({anomaly_percentage:.2f}%)',
        fontsize=16, y=0.98
    )

    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        if i >= num_rows * num_cols:
            continue

        ax = axes[i]
        ax.imshow(
            masked_data, cmap='YlOrBr', origin='lower',  # Reverted to original colormap 'YlOrBr'
            vmin=np.nanpercentile(masked_data, 2),
            vmax=np.nanpercentile(masked_data, 98),
            alpha=0.5  # Reverted to original alpha 0.5
        )

        if n_clusters_global > 0:
            for cluster_index in range(1, n_clusters_global + 1):
                cluster_area_mask = cluster_mask_global == cluster_index
                cluster_color = cluster_cmap_global(
                    (cluster_index - 1) / n_clusters_global)
                ax.imshow(
                    np.ma.masked_where(~cluster_area_mask, cluster_mask_global),
                    cmap=matplotlib.colors.ListedColormap([cluster_color]),
                    alpha=0.6, origin='lower',  # Reverted to original alpha 0.6
                    vmin=cluster_index - 0.5, vmax=cluster_index + 0.5
                )

        title_lines = [f'AIA {channel} Å']
        if cluster_pixels_counts and cluster_anomaly_percentages and cluster_index <= len(
                cluster_pixels_counts):
            cluster_pixels = cluster_pixels_counts[cluster_index - 1]
            cluster_percentage = cluster_anomaly_percentages[cluster_index - 1]
            title_lines.append(
                f'Cluster {cluster_index + 1}: {cluster_pixels} Pixels ({cluster_percentage:.2f}%)')

        ax.set_title(
            "\n".join(title_lines),
            color='black', fontsize=14, pad=10  # Reverted to original fontsize 14
        )
        ax.axis('off')

    if cluster_patches_global:
        fig.legend(
            handles=cluster_patches_global, loc='upper right',
            bbox_to_anchor=(0.95, 0.95),
            fontsize='small', framealpha=0.8
        )

    for j in range(len(channel_names), num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 0.93, 0.95], w_pad=0.1, h_pad=0.1)
    filename = os.path.join(
        output_dir, f"kmeans_anomaly_detection_threshold_{
            anomaly_threshold:.2f}_global_clusters.png"  # Reverted to original filename
    )
    plt.savefig(filename, bbox_inches='tight', dpi=100)  # Reverted to original dpi 100
    plt.close(fig)
    print(f"Figure saved to: {filename}")


# --- Main Execution ---
def main():
    """Main function to execute SDO/AIA anomaly detection pipeline."""
    parser = argparse.ArgumentParser(
        description="SDO/AIA Anomaly Detection using Isolation Forest and K-Means"
    )
    parser.add_argument("--data_dir", type=str, default="Data/sdo_data",
                        help="Path to SDO/AIA data directory.")
    parser.add_argument("--channels", type=str, nargs='+',
                        default=None, help="AIA channels (e.g., '94', '131').")
    parser.add_argument("--anomaly_thresholds", type=float, nargs='+',
                        default=[0.1], help="Anomaly threshold(s).")
    parser.add_argument("--output_dir", type=str,
                        default="./output_figures", help="Output directory for figures.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize image size.")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Isolation Forest contamination parameter.")
    parser.add_argument("--n_clusters", type=int, default=7,
                        help="Number of clusters for KMeans.")
    parser.add_argument("--max_k", type=int, default=10,
                        help="Max clusters for Elbow method.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    channels_arg = ['94', '131', '171', '193', '211', '233',
                    '304', '335', '700']  # Specify all 9 channels
    channels = args.channels if args.channels else [f"aia_{c}" for c in channels_arg]

    if not channels:
        print("No channels found. Exiting.")
        return

    masked_data_list, channel_names = [], []
    for channel_dir in channels:
        try:
            channel = channel_dir.split("_")[1]
            channel_names.append(channel)
            channel_path = os.path.join(args.data_dir, channel_dir)
            data, metadata = load_fits_data(channel_path)
            mask = create_circular_mask(data, metadata)
            masked_data_list.append(preprocess_image(data, mask, args.image_size))
        except Exception as e:
            print(f"Error processing {channel_dir}: {e}")

    if not masked_data_list:
        print("No data loaded. Exiting.")
        return

    prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(
        masked_data_list)

    anomaly_scores = detect_anomalies_isolation_forest(
        prepared_data, args.contamination
    )

    anomaly_map_2d = np.full((args.image_size, args.image_size), np.nan)
    anomaly_map_2d[
        valid_pixel_mask.reshape((args.image_size, args.image_size))
    ] = anomaly_scores

    # Initialize anomaly_mask_global to None here, before the loop
    anomaly_mask_global = None

    for anomaly_threshold in args.anomaly_thresholds:
        print(f"Processing with anomaly threshold: {anomaly_threshold}")
        print(f"np.sum(anomaly_mask_global) in main loop: {
              np.sum(anomaly_mask_global)}")
        anomaly_mask_global = anomaly_map_2d < anomaly_threshold
        anomaly_pixels_count = np.sum(anomaly_mask_global)
        total_pixels = args.image_size * args.image_size
        print(f"Total pixels in image: {total_pixels}")
        print(f"Using user-specified n_clusters = {args.n_clusters}")

        anomaly_pixels_indices = np.argwhere(anomaly_mask_global)
        valid_pixel_mask_2d = ~nan_mask.reshape((args.image_size, args.image_size))
        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {
            tuple(idx): i for i, idx in enumerate(valid_pixel_indices_2d)}

        anomaly_intensity_features = np.array([
            prepared_data[pixel_index_map[tuple(idx)]] for idx in anomaly_pixels_indices if tuple(idx) in pixel_index_map
        ])

        cluster_pixels_counts: List[int] = []
        cluster_anomaly_percentages: List[float] = []

        if len(anomaly_intensity_features) > 0:
            cluster_labels, _ = perform_kmeans_clustering(
                anomaly_intensity_features, args.n_clusters, args.random_state
            )
            cluster_mask_global, cluster_cmap_global, cluster_patches_global, n_clusters_global = create_cluster_mask(
                anomaly_mask_global, cluster_labels, valid_pixel_mask, args.image_size
            )
            for cluster_index in range(n_clusters_global):
                cluster_pixel_count = np.sum(cluster_mask_global == (cluster_index + 1))
                cluster_pixels_counts.append(cluster_pixel_count)
                cluster_percentage = (
                    cluster_pixel_count / anomaly_pixels_count) * 100 if anomaly_pixels_count else 0
                cluster_anomaly_percentages.append(cluster_percentage)
                print(f"  Cluster {cluster_index + 1}: {cluster_pixel_count} pixels"
                      f" ({cluster_percentage:.2f}%) of total anomalies")

        plot_results(
            masked_data_list,
            cluster_mask_global,
            cluster_cmap_global,
            n_clusters_global,
            cluster_patches_global,
            channel_names,
            anomaly_threshold,
            args.output_dir,
            total_pixels,  # Pass total pixels
            anomaly_pixels_count,  # Pass anomaly pixel count
            cluster_pixels_counts,  # Pass cluster pixel counts
            cluster_anomaly_percentages,  # Pass cluster anomaly percentages
            clustering_method_name="K-Means"  # Pass clustering method name
        )

    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
