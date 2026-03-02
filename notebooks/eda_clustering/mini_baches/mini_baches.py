import argparse
import os
import time

import matplotlib
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans  # Changed from KMeans to MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")  # Use Agg backend for saving files


# --- Helper Functions ---
def load_fits_data(channel_dir: str) -> tuple[np.ndarray, dict]:
    """Load FITS data and metadata from a single channel directory. Loads only the
    *first* FITS file in the directory.

    Args:
        channel_dir (str): Path to the channel directory.

    Returns:
        Tuple[np.ndarray, dict]: Data and metadata from the FITS file.

    Raises:
        FileNotFoundError: If no FITS files are found in the directory.
    """
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in directory: {channel_dir}")

    fits_path = os.path.join(channel_dir, fits_files[0])  # Load the first FITS file
    aia_map = sunpy.map.Map(fits_path)
    return aia_map.data, aia_map.meta


def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates a circular mask for the solar disk based on metadata.

    Args:
        data (np.ndarray): Image data.
        metadata (dict): FITS metadata containing header info.

    Returns:
        np.ndarray: Boolean mask, True for pixels inside the solar disk.
    """
    ny, nx = data.shape
    x_center, y_center = nx // 2, ny // 2
    cdelt1 = metadata.get("cdelt1", 1.0)  # Arcsec/pixel in X
    solar_radius_arcsec = metadata.get("rsun_obs", 960.0)  # Solar radius arcsec
    solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))

    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    return distance_from_center <= solar_radius_pixels


def preprocess_image(data: np.ndarray, mask: np.ndarray, size: int = None) -> np.ndarray:
    """Resizes the image and applies the mask, setting masked areas to NaN.

    Args:
        data (np.ndarray): Input image data.
        mask (np.ndarray): Boolean mask for the solar disk.
        size (int, optional): Desired size of the resized image. If None, no resize. Defaults to None.

    Returns:
        np.ndarray: Preprocessed image data with mask applied and resized.
    """
    if size is not None:
        resized_data = resize(data, (size, size), mode="reflect", anti_aliasing=True)
        resized_mask = resize(mask, (size, size), mode="reflect", anti_aliasing=False) > 0.5
        masked_data = resized_data.copy()
        masked_data[~resized_mask] = np.nan  # Apply mask
        return masked_data
    else:
        masked_data = data.copy()
        masked_data[~mask] = np.nan  # Apply original mask
        return masked_data


# --- Data Preparation and Anomaly Detection ---
def prepare_data_concatenated(masked_data_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data from multiple channels, handles NaNs, and scales the
    data.

    Args:
        masked_data_list (list): List of masked data arrays (one per channel).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - scaled_data: Scaled data, reshaped to
              (pixels_without_nan, num_channels).
            - valid_pixel_mask: 1D boolean mask (True for valid pixels).
            - nan_mask: 1D boolean mask (True for pixels with NaN in any channel).
    """
    stacked_data = np.stack(masked_data_list, axis=-1)  # (height, width, channels)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))  # (pixels, channels)
    nan_mask = np.isnan(reshaped_data).any(axis=1)  # Rows with any NaN
    cleaned_data = reshaped_data[~nan_mask]  # Remove rows with NaNs
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask


def detect_anomalies_isolation_forest(data: np.ndarray, contamination: float, random_state: int) -> np.ndarray:
    """Detects anomalies using Isolation Forest and returns anomaly scores.

    Args:
        data (np.ndarray): Input data for anomaly detection.
        contamination (float): Expected proportion of anomalies in the data.
        random_state (int): Random seed for reproducibility. # Added random_state

    Returns:
        np.ndarray: Anomaly scores for each data point.
                     Lower scores indicate higher anomaly.
    """
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,  # Use random_state here
    )
    iso_forest.fit(data)
    return iso_forest.decision_function(data)  # Return anomaly *scores*


# --- Clustering ---
def perform_MiniBatch_K_Means_clustering(  # Function name remains the same but now uses MiniBatchKMeans
    data: np.ndarray, n_clusters: int, random_state: int = 42
) -> tuple[np.ndarray, float]:
    """Performs MiniBatch K-Means clustering. # Updated docstring to reflect
    MiniBatchKMeans.

    Args:
        data (np.ndarray): Data to be clustered.
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[np.ndarray, float]:
            - labels: Cluster labels for each data point.
            - inertia: Sum of squared distances to closest centroid.
    """
    kmeans = MiniBatchKMeans(  # Changed KMeans to MiniBatchKMeans
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # Reduced n_init for MiniBatchKMeans as it's faster
        batch_size=256,  # Added batch_size for MiniBatchKMeans
    )
    kmeans.fit(data)
    return kmeans.labels_, kmeans.inertia_


def determine_optimal_k_elbow(data: np.ndarray, max_k: int = 10, random_state: int = 42) -> int:
    """Applies the Elbow Method to suggest an optimal number of clusters (k).

    Args:
        data (np.ndarray): Data to be clustered.
        max_k (int): Maximum number of clusters to test (default: 10).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        int: Suggested optimal number of clusters (k).
    """
    inertias = []
    for k in range(1, max_k + 1):
        _, inertia = perform_MiniBatch_K_Means_clustering(data, k, random_state)
        inertias.append(inertia)

    # Plot the Elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    elbow_plot_path = "elbow_plot.png"
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow plot saved to: {elbow_plot_path}")

    # Find the "elbow" (simplistic approach - could be improved)
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(diffs2) + 2  # +2 because of two diff operations

    return optimal_k


def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray,
    image_size: int,  # image_size is still passed but not strictly used anymore
) -> tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels.

    Args:
        anomaly_mask (np.ndarray): 2D boolean mask of anomaly pixels.
        labels (np.ndarray): Cluster labels for each anomaly pixel.
        valid_pixel_mask (np.ndarray): 1D boolean mask of valid pixels.
        image_size (int): Size of the image (height/width). - Not strictly used anymore, derived from anomaly_mask

    Returns:
        Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
            - cluster_mask: 2D array representing cluster assignments.
            - cluster_cmap: Colormap used for clusters.
            - cluster_patches: Legend patches for the clusters.
            - n_clusters: Number of clusters.
    """
    print("-" * 20)
    print("Inside create_cluster_mask:")
    print(f"anomaly_mask.shape: {anomaly_mask.shape}")
    print(f"np.sum(anomaly_mask): {np.sum(anomaly_mask)}")
    # Number of initially detected anomalies

    cluster_mask = np.zeros_like(anomaly_mask, dtype=int)  # Initialize zeros
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])  # Initialize empty cmap
    cluster_patches = []

    if len(labels) > 0:
        n_clusters = len(np.unique(labels))
        print(f"Number of clusters (n_clusters): {n_clusters}")
        # Modified color palette, removed dark blue, added more distinct colors
        cluster_colors = [
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",  # No blue, added grey
        ]
        cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors[:n_clusters])

        # Create 2D masks for easier indexing
        current_image_size = anomaly_mask.shape[0]  # Get image size from anomaly mask
        valid_pixel_mask_2d = valid_pixel_mask.reshape((current_image_size, current_image_size))
        anomaly_pixels_indices = np.argwhere(anomaly_mask)  # 2D anomaly indices
        print(f"len(anomaly_pixels_indices): {len(anomaly_pixels_indices)}")
        # Should be same as np.sum(anomaly_mask)

        # Map 2D anomaly pixel indices to rows in prepared data
        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {tuple(index_2d): i for i, index_2d in enumerate(valid_pixel_indices_2d)}

        valid_anomaly_pixel_indices = []
        for anomaly_pixel_index_2d in anomaly_pixels_indices:
            if tuple(anomaly_pixel_index_2d) in pixel_index_map:  # Check valid
                valid_anomaly_pixel_indices.append(anomaly_pixel_index_2d)
        valid_anomaly_pixel_indices = np.array(valid_anomaly_pixel_indices)
        print(f"len(valid_anomaly_pixel_indices): {len(valid_anomaly_pixel_indices)}")  # Anomalies after valid check
        print(f"labels.shape: {labels.shape}")

        if len(valid_anomaly_pixel_indices) > 0:
            for cluster_idx in range(n_clusters):
                # Get indices of pixels belonging to current cluster
                cluster_pixel_indices = valid_anomaly_pixel_indices[labels == cluster_idx]
                if len(cluster_pixel_indices) > 0:
                    # Assign cluster label to corresponding pixels in 2D mask
                    cluster_mask[tuple(cluster_pixel_indices.T)] = cluster_idx + 1  # +1 to avoid 0 (background)
                    cluster_color = cluster_cmap(cluster_idx / n_clusters)  # Get color for cluster
                    cluster_patches.append(mpatches.Patch(color=cluster_color, label=f"Cluster {cluster_idx + 1}"))

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
    total_pixels: int,  # Added total_pixels
    anomaly_pixels_count: int,  # Added anomaly_pixels_count
    elapsed_time: float,
    # Added clustering_method_name with default "K-Means"
    clustering_method_name: str = "mini_batch_kmeans",
):
    """Plots and saves the results, overlaying global clusters on each channel.

    Args:
        masked_data_list (list): List of masked image data arrays.
        cluster_mask_global (np.ndarray): Global cluster mask.
        cluster_cmap_global (matplotlib.colors.ListedColormap): Colormap for clusters.
        n_clusters_global (int): Number of global clusters.
        cluster_patches_global (list): Legend patches for global clusters.
        channel_names (list): List of channel names (wavelengths).
        anomaly_threshold (float): Anomaly threshold used.
        output_dir (str): Directory to save output figures.
        total_pixels (int): Total number of pixels in the image. # Added
        anomaly_pixels_count (int): Number of anomaly pixels detected. # Added
        clustering_method_name (str): Name of the clustering method used. # Added
    """
    num_rows, num_cols = 3, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), dpi=100)
    axes = axes.flatten()

    total_pixels = cluster_mask_global.size
    anomalous_pixels = np.count_nonzero(cluster_mask_global)
    anomaly_percentage = 100 * anomalous_pixels / total_pixels

    anomaly_percentage = (anomaly_pixels_count / total_pixels) * 100 if total_pixels > 0 else 0  # Calculate percentage
    fig.suptitle(
        f"minibatch k-means Anomaly Clusters in SDO/AIA JP2 Channels\n"
        f"Anomaly Threshold: {anomaly_threshold:.2f} | "
        f"Anomalous Pixels: {anomalous_pixels:,}/{total_pixels:,} "
        f"({anomaly_percentage:.2f}%) | Exec Time: {elapsed_time:.2f}s",
        fontsize=16,
    )

    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        if i < num_rows * num_cols:  # Prevent index out of bounds
            ax = axes[i]

            # Display the original image with pastel yellow colormap
            ax.imshow(
                masked_data,
                origin="lower",
                cmap="YlOrBr",  # Pastel yellow/orange for the sun
                vmin=np.nanpercentile(masked_data, 2),
                vmax=np.nanpercentile(masked_data, 98),
                alpha=0.5,
            )

            # Overlay the GLOBAL cluster mask (only color cluster areas)
            if n_clusters_global > 0:
                for cluster_index in range(1, n_clusters_global + 1):
                    cluster_area_mask = cluster_mask_global == cluster_index
                    cluster_color = cluster_cmap_global((cluster_index - 1) / n_clusters_global)
                    ax.imshow(
                        np.ma.masked_where(~cluster_area_mask, cluster_mask_global),
                        origin="lower",
                        cmap=matplotlib.colors.ListedColormap([cluster_color]),
                        alpha=0.6,
                        vmin=cluster_index - 0.5,
                        vmax=cluster_index + 0.5,
                    )

            ax.set_title(f"AIA {channel} Å (Global Clusters: {n_clusters_global})", color="black", fontsize=14, pad=10)
            ax.text(
                0.5,
                -0.18,
                f"Anomaly Threshold: {anomaly_threshold:.2f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="dimgray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add legend (only to the last used subplot)
            if (i == len(channel_names) - 1) and cluster_patches_global:
                ax.legend(handles=cluster_patches_global, loc="upper right", fontsize="small")

    # Remove any unused subplots
    for j in range(len(channel_names), num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.92], w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    filename = os.path.join(
        output_dir,
        f"minibatch_kmeans_anomaly_detection_threshold_{anomaly_threshold:.2f}"  # Changed filename to minibatch_kmeans
        "_global_clusters.png",
    )
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {filename}")


# --- Main Execution ---
def main():
    """Main function to execute SDO/AIA anomaly detection pipeline."""
    parser = argparse.ArgumentParser(
        description="SDO/AIA Anomaly Detection using Isolation Forest"
        " and MiniBatch K-Means Clustering"  # Updated description
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data/sdo_data",  # Changed default path here
        help="Path to the directory containing SDO/AIA data.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=None,
        help="List of AIA channels (e.g., '94' '131'). If None, use all except 1600 and 1700.",
    )
    parser.add_argument(
        "--anomaly_thresholds",
        type=float,
        nargs="+",
        default=[0.1],
        help="Threshold(s) for anomaly detection. Lower values more sensitive.",
    )
    parser.add_argument("--output_dir", type=str, default="./output_figures", help="Directory to save output figures.")
    parser.add_argument(
        "--image_size", type=int, default=512, help="Size to resize images to. Ignored if --no_resize is used."
    )
    parser.add_argument("--contamination", type=float, default=0.05, help="Estimated proportion of anomalies.")
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=7,  # Changed default to 7 clusters
        help="Number of clusters for minibatch K-Means. If None, use Elbow method.",
    )
    parser.add_argument(
        "--max_k", type=int, default=10, help="Maximum number of clusters to test with Elbow method."
    )  # Added max_k
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_resize", action="store_true", help="Do not resize images, use original size.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Data Loading and Channel Selection ---
    if args.channels:
        channels = [f"aia_{c}" for c in args.channels]  # Prefix with "aia_"
    else:
        all_dirs = os.listdir(args.data_dir)
        channels = [
            d
            for d in all_dirs
            if os.path.isdir(os.path.join(args.data_dir, d))
            and not d.startswith("aia_1600")
            and not d.startswith("aia_1700")
        ]

    if not channels:
        print("No channels found. Exiting.")
        return

    # --- 2. Load and Preprocess Data for All Channels ---
    masked_data_list = []
    channel_names = []
    # Determine size, None if no_resize
    image_size_for_processing = args.image_size if not args.no_resize else None
    # Initialize, will be updated if no_resize
    current_image_size = args.image_size if not args.no_resize else None
    original_image_size = None  # To store original image size if not resizing

    for channel_dir in channels:
        try:
            channel = channel_dir.split("_")[1]
            channel_names.append(channel)  # Store channel names
            channel_path = os.path.join(args.data_dir, channel_dir)
            data, metadata = load_fits_data(channel_path)
            mask = create_circular_mask(data, metadata)
            masked_data = preprocess_image(
                data,
                mask,
                image_size_for_processing,  # Pass size, can be None
            )
            masked_data_list.append(masked_data)
            if args.no_resize:
                # Update to original size if no resize
                current_image_size = data.shape[0]
        except Exception as e:
            print(f"Error processing {channel_dir}: {e}")

    if not masked_data_list:
        print("No data loaded. Exiting.")
        return

    # --- 3. Prepare Data for Anomaly Detection ---
    prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(masked_data_list)

    # --- 4. Anomaly Detection (One-time calculation) ---
    anomaly_scores = detect_anomalies_isolation_forest(prepared_data, args.contamination, args.random_state)

    # Create a 2D anomaly map (for visualization)
    # Use current_image_size
    anomaly_map_2d = np.full((current_image_size, current_image_size), np.nan)
    valid_pixel_mask_2d_reshaped = valid_pixel_mask.reshape(
        (current_image_size, current_image_size)
    )  # Use current_image_size
    anomaly_map_2d[valid_pixel_mask_2d_reshaped] = anomaly_scores

    # time tracking
    start_time = time.time()

    # --- 5. Loop Through Anomaly Thresholds ---
    for anomaly_threshold in args.anomaly_thresholds:
        print(f"Processing with anomaly threshold: {anomaly_threshold}")
        anomaly_mask_global = anomaly_map_2d < anomaly_threshold
        # Global anomaly mask
        anomaly_pixels_count = np.sum(anomaly_mask_global)  # Count anomaly pixels
        total_pixels = current_image_size * current_image_size  # Calculate total pixels
        print(f"np.sum(anomaly_mask_global) in main loop: {anomaly_pixels_count}")
        print(f"Total pixels in image: {total_pixels}")

        # --- 6. Clustering (MiniBatch K-Means) --- # Changed comment to reflect MiniBatchKMeans
        anomaly_pixels_indices = np.argwhere(anomaly_mask_global)

        # Map anomaly pixel indices to rows in prepared data (critical step)
        valid_pixel_mask_2d = ~nan_mask.reshape((current_image_size, current_image_size))  # Use current_image_size
        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {tuple(index_2d): i for i, index_2d in enumerate(valid_pixel_indices_2d)}

        anomaly_intensity_features = []
        valid_anomaly_pixel_indices = []
        for anomaly_pixel_index_2d in anomaly_pixels_indices:
            if tuple(anomaly_pixel_index_2d) in pixel_index_map:  # Ensure valid
                prepared_data_row_index = pixel_index_map[tuple(anomaly_pixel_index_2d)]
                anomaly_intensity_features.append(prepared_data[prepared_data_row_index])
                valid_anomaly_pixel_indices.append(anomaly_pixel_index_2d)
        anomaly_intensity_features = np.array(anomaly_intensity_features)
        valid_anomaly_pixel_indices = np.array(valid_anomaly_pixel_indices)

        if len(anomaly_intensity_features) > 0:  # Proceed if valid anomalies
            n_clusters_to_use = args.n_clusters  # Use n_clusters (default 7)
            print(f"Using user-specified n_clusters = {n_clusters_to_use}")

            # Perform MiniBatch K-Means Clustering # Changed comment to reflect
            # MiniBatchKMeans
            cluster_labels, _ = (
                perform_MiniBatch_K_Means_clustering(  # Function name is still perform_kmeans_clustering but now uses MiniBatchKMeans
                    anomaly_intensity_features, n_clusters_to_use, args.random_state
                )
            )

            # Create cluster mask, colormap, and legend patches
            cluster_mask_global, cluster_cmap_global, cluster_patches_global, n_clusters_global = create_cluster_mask(
                anomaly_mask_global,
                cluster_labels,
                valid_pixel_mask,
                current_image_size,  # Pass current image size
            )

        else:  # Handle case where no anomalies are detected
            print(f"No anomalies found for threshold {anomaly_threshold}.")
            cluster_mask_global = np.zeros_like(anomaly_mask_global, dtype=int)
            cluster_cmap_global = matplotlib.colors.ListedColormap([])
            cluster_patches_global = []
            n_clusters_global = 0

        elapsed_time = time.time() - start_time
        print(f"Execution time for threshold {anomaly_threshold:.2f}: {elapsed_time:.2f} seconds")

        # --- 7. Plotting and Saving Results ---
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
            # Added clustering_method_name for plot title
            clustering_method_name="MiniBatch K-Means",
            elapsed_time=elapsed_time,
        )


if __name__ == "__main__":
    main()
