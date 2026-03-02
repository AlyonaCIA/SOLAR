import argparse
import os

import imageio.v2 as imageio  # <-- Added imageio import
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

matplotlib.use("Agg")  # Use Agg backend for saving files


# --- Helper Functions ---


# --- FITS Specific ---
def load_fits_data(channel_dir: str) -> tuple[np.ndarray, dict]:
    """Loads FITS data and metadata from a single channel directory."""
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith((".fits", ".fit"))]  # Allow .fit too
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in: {channel_dir}")
    fits_path = os.path.join(channel_dir, fits_files[0])
    print(f"Loading FITS: {fits_path}")
    aia_map = sunpy.map.Map(fits_path)
    print(f"  Loaded FITS data shape: {aia_map.data.shape}")
    return aia_map.data, aia_map.meta


def create_circular_mask_fits(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates a circular mask for the solar disk based on FITS metadata."""
    if data is None or metadata is None:
        raise ValueError("Data and metadata must be provided for FITS mask creation.")
    ny, nx = data.shape
    # Use metadata center and radius calculation
    cdelt1 = metadata.get("cdelt1", 0.0)
    # Handle potential case where cdelt1 is missing or zero
    if abs(cdelt1) < 1e-9:
        print("Warning: CDELT1 missing or zero in FITS header. Cannot calculate radius from metadata.")
        # Fallback: estimate radius based on image size (similar to JP2 logic)
        # This is less ideal for FITS but provides a fallback.
        print("Falling back to estimated radius based on image dimensions.")
        solar_radius_pixels = int(min(nx, ny) * 0.48)  # Approx half the smallest dim
        x_center, y_center = nx / 2.0, ny / 2.0  # Assume centered if no CRPIX
    else:
        solar_radius_arcsec = metadata.get("rsun_obs", 960.0)  # Solar radius in arcsec
        solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))
        # Use CRPIX for center (adjusting from 1-based FITS to 0-based numpy)
        crpix1 = metadata.get("crpix1", (nx + 1) / 2.0)  # Default to center if missing
        crpix2 = metadata.get("crpix2", (ny + 1) / 2.0)  # Default to center if missing
        x_center = crpix1 - 1
        y_center = crpix2 - 1

    print(f"Creating FITS mask: Center=({x_center:.1f}, {y_center:.1f}), Radius={solar_radius_pixels} pix")
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    mask = distance_from_center <= solar_radius_pixels
    print(f"  Generated FITS mask shape: {mask.shape}, Sum: {np.sum(mask)}")
    return mask


# --- JP2 Specific ---
def load_jp2_data_imageio(channel_dir: str) -> tuple[np.ndarray | None, dict | None]:
    """Loads JP2 data using Imageio from a single channel directory.

    Metadata will always be None.
    """
    jp2_files = [f for f in os.listdir(channel_dir) if f.endswith(".jp2")]
    if not jp2_files:
        print(f"Warning: No JP2 files found in: {channel_dir}")
        return None, None
    jp2_path = os.path.join(channel_dir, jp2_files[0])
    print(f"Attempting to load JP2 with Imageio: {jp2_path}")
    data = None
    try:
        data = imageio.imread(jp2_path)
        print(f"  Loaded JP2 data shape: {data.shape}, dtype: {data.dtype}")
    except Exception as e_imgio:
        print(f"Imageio failed to load {jp2_path}: {e_imgio}")
        return None, None
    # Return data and None for metadata, matching the FITS function signature
    return data, None


def create_circular_mask_jp2(data: np.ndarray, fixed_radius_pixels: int) -> np.ndarray:
    """Creates a circular mask for JP2 images using a fixed radius, assuming a centered
    Sun."""
    if data is None:
        raise ValueError("Input data cannot be None for JP2 mask creation.")
    ny, nx = data.shape
    print(f"Creating JP2 mask for image size {ny}x{nx} using fixed radius: {fixed_radius_pixels}")
    x_center, y_center = nx // 2, ny // 2  # Assume centered
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    mask = distance_from_center <= fixed_radius_pixels
    print(f"  Generated JP2 mask shape: {mask.shape}, Sum: {np.sum(mask)}")
    return mask


# --- Common Preprocessing ---
def preprocess_image(data: np.ndarray, mask: np.ndarray, size: int = 512) -> np.ndarray:
    """Resizes the image and the mask, then applies the resized mask."""
    if data is None or mask is None:
        raise ValueError("Data and mask must be provided for preprocessing.")
    print(f"Preprocessing: Resizing data ({data.shape}) and mask ({mask.shape}) to {size}x{size}")
    # Resize data using anti-aliasing
    resized_data = resize(data, (size, size), mode="reflect", anti_aliasing=True)
    # Resize mask using nearest neighbor (order=0) or similar to avoid intermediate values
    # Using anti_aliasing=False and thresholding is generally okay for masks
    resized_mask = resize(mask.astype(float), (size, size), mode="reflect", anti_aliasing=False) > 0.5
    print(f"  Resized mask shape: {resized_mask.shape}, Sum: {np.sum(resized_mask)}")

    # Apply the *resized* mask
    masked_data = resized_data.copy()
    masked_data[~resized_mask] = np.nan  # Set pixels outside the resized mask to NaN
    print(f"  Final masked data shape: {masked_data.shape}, Non-NaN count: {np.sum(~np.isnan(masked_data))}")
    return masked_data


# --- Data Preparation --- (No changes needed here)
def prepare_data_concatenated(masked_data_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data, handles NaNs, and scales the data."""
    stacked_data = np.stack(masked_data_list, axis=-1)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    nan_mask = np.isnan(reshaped_data).any(axis=1)
    # Ensure we don't try to scale if all pixels are NaN after masking/concatenation
    if np.all(nan_mask):
        print("Warning: All pixels are NaN after concatenation. Cannot scale.")
        return np.array([]), np.array([]), nan_mask  # Return empty arrays and the mask

    cleaned_data = reshaped_data[~nan_mask]
    # Handle case where cleaned_data might be empty after removing NaNs
    if cleaned_data.shape[0] == 0:
        print("Warning: No valid (non-NaN) pixels left after masking.")
        return np.array([]), np.array([]), nan_mask  # Return empty arrays and the mask

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask  # Return scaled data, valid pixel mask, and NaN mask


# --- Anomaly Detection --- (No changes needed here)
def detect_anomalies_isolation_forest(data: np.ndarray, contamination: float) -> np.ndarray:
    """Detects anomalies using Isolation Forest."""
    if data.shape[0] == 0:  # Handle empty input data
        print("Warning: Cannot detect anomalies on empty dataset.")
        return np.array([])
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,  # Use n_jobs=-1 for potential speedup
    )
    print(f"Fitting Isolation Forest (contamination={contamination}) on data shape: {data.shape}")
    iso_forest.fit(data)
    return iso_forest.decision_function(data)


# --- Clustering --- (Minor print adjustments, no logic change)
def perform_kmeans_clustering(data: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, float]:
    """Performs K-Means clustering."""
    if data.shape[0] == 0:  # Handle empty input data
        print("Warning: Cannot perform K-Means on empty dataset.")
        return np.array([]), np.inf  # Return empty labels and infinite inertia
    print(f"Performing K-Means (k={n_clusters}) on data shape: {data.shape}")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",  # Use 'auto' for default (10 in recent sklearn)
    )
    kmeans.fit(data)
    print(f"  K-Means inertia: {kmeans.inertia_:.2f}")
    return kmeans.labels_, kmeans.inertia_


def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray,  # This mask corresponds to the *flattened* scaled data
    image_size: int,
) -> tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels."""
    print("-" * 20)
    print("Inside create_cluster_mask:")
    # anomaly_mask is the 2D boolean mask from thresholding anomaly scores
    # labels correspond to the *anomalous* pixels only
    # valid_pixel_mask is the 1D boolean mask where True means the pixel was valid *before* anomaly detection

    if anomaly_mask is None or labels is None or valid_pixel_mask is None:
        print("Warning: Inputs to create_cluster_mask are invalid. Returning empty cluster map.")
        return np.zeros((image_size, image_size), dtype=int), matplotlib.colors.ListedColormap([]), [], 0

    print(f"Input anomaly_mask shape: {anomaly_mask.shape} (Sum: {np.sum(anomaly_mask)})")
    print(f"Input labels length: {len(labels)}")
    print(f"Input valid_pixel_mask shape: {valid_pixel_mask.shape} (Sum: {np.sum(valid_pixel_mask)})")

    cluster_mask_2d = np.zeros((image_size, image_size), dtype=int)
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])
    cluster_patches = []

    # Check if there are any anomaly labels to process
    if len(labels) > 0 and np.any(anomaly_mask):
        # Ensure labels correspond only to the anomalous pixels
        np.sum(anomaly_mask)

        # We need to map the labels back to the 2D grid.
        # The labels correspond to the `anomaly_intensity_features` which were derived
        # from `prepared_data[anomaly_indices]`.
        # `prepared_data` corresponds to `reshaped_data[valid_pixel_mask]`.

        # Create a full-size array for labels, initially zero or a placeholder
        # Map 1D valid_pixel_mask to 2D
        valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))

        # Find the 2D indices of the pixels that were both valid AND anomalous
        valid_and_anomalous_indices_2d = np.argwhere(valid_pixel_mask_2d & anomaly_mask)

        if len(valid_and_anomalous_indices_2d) != len(labels):
            print(
                f"Warning: Mismatch between number of valid+anomalous pixels ({len(valid_and_anomalous_indices_2d)}) and number of labels ({len(labels)}). Check logic."
            )
            # Attempt to proceed if possible, otherwise return empty
            if len(valid_and_anomalous_indices_2d) == 0:
                return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters

        n_clusters = len(np.unique(labels))
        print(f"Number of unique cluster labels found: {n_clusters}")

        # Define colors
        cluster_colors = [
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#1f77b4",  # Added more colors
        ]
        # Handle case where n_clusters might exceed defined colors
        if n_clusters > len(cluster_colors):
            print(
                f"Warning: Number of clusters ({n_clusters}) exceeds defined colors ({len(cluster_colors)}). Repeating colors."
            )
            cluster_colors = (cluster_colors * (n_clusters // len(cluster_colors) + 1))[:n_clusters]

        cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors)

        # Assign cluster labels to the corresponding 2D positions
        # Add 1 to labels so cluster indices start from 1 (0 means no cluster)
        cluster_mask_2d[tuple(valid_and_anomalous_indices_2d.T)] = labels + 1

        # Create legend patches
        for cluster_idx in range(n_clusters):
            cluster_color = cluster_cmap(
                cluster_idx / (n_clusters - 1 if n_clusters > 1 else 1)
            )  # Normalize index correctly
            cluster_patches.append(mpatches.Patch(color=cluster_color, label=f"Cluster {cluster_idx + 1}"))
    else:
        print("No anomaly labels or no anomalous pixels in mask. No clusters to map.")

    print(f"Final cluster_mask_2d shape: {cluster_mask_2d.shape}, Max value: {np.max(cluster_mask_2d)}")
    print("-" * 20)
    return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters


# --- Plotting --- (Added file type to title/filename, check for empty clusters)
def plot_results(
    masked_data_list: list,
    cluster_mask_global: np.ndarray,
    cluster_cmap_global: matplotlib.colors.ListedColormap,
    n_clusters_global: int,
    cluster_patches_global: list,
    channel_names: list,
    anomaly_threshold: float,
    output_dir: str,
    total_pixels_resized: int,  # Renamed for clarity
    anomaly_pixels_count: int,
    cluster_pixels_counts: list[int],
    cluster_anomaly_percentages: list[float],
    file_type: str,  # Added file type
    clustering_method_name: str = "K-Means",
):
    """Plots and saves anomaly detection and clustering results."""
    if not masked_data_list:
        print("No data to plot.")
        return

    # Determine plot grid size dynamically (e.g., up to 3x3)
    num_images = len(masked_data_list)
    num_cols = min(3, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=100, squeeze=False
    )  # Use squeeze=False for consistent 2D array
    axes = axes.flatten()

    anomaly_percentage = (anomaly_pixels_count / total_pixels_resized) * 100 if total_pixels_resized > 0 else 0
    fig.suptitle(
        f"{clustering_method_name} Anomaly Clusters in SDO/AIA {file_type.upper()} Channels\n"
        f"Anomaly Threshold: {anomaly_threshold:.2f} | Anomalous Pixels (resized): {anomaly_pixels_count}/{
            total_pixels_resized
        } ({anomaly_percentage:.2f}%)",
        fontsize=16,
        y=0.98 if num_rows > 1 else 1.02,  # Adjust title position
    )

    base_cmap_name = "sdoaia{channel}"  # Base colormap name template
    fallback_cmap = "viridis"  # Fallback if specific AIA map isn't found

    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        if i >= len(axes):
            continue  # Should not happen with dynamic grid, but safe check

        ax = axes[i]

        # Try channel-specific colormap, else fallback
        try:
            cmap_name = base_cmap_name.format(channel=channel)
            img_cmap = plt.get_cmap(cmap_name)
        except ValueError:
            print(f"Colormap {cmap_name} not found, using {fallback_cmap}.")
            img_cmap = fallback_cmap

        # Plot base image (masked data)
        # Handle cases where all data might be NaN
        valid_data = masked_data[~np.isnan(masked_data)]
        if valid_data.size > 0:
            vmin = np.percentile(valid_data, 2)
            vmax = np.percentile(valid_data, 98)
        else:
            vmin, vmax = 0, 1  # Default if no valid data

        ax.imshow(
            masked_data,
            cmap=img_cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,  # Slightly more transparent base
        )

        # Overlay clusters if they exist
        if n_clusters_global > 0 and cluster_mask_global is not None and cluster_mask_global.shape == masked_data.shape:
            # Plot each cluster individually to control color and legend precisely
            for cluster_index in range(1, n_clusters_global + 1):
                cluster_area_mask = cluster_mask_global == cluster_index
                if np.any(cluster_area_mask):  # Only plot if pixels exist for this cluster
                    # Use the provided global cmap, ensure index is correct
                    # cmap expects normalized value 0..1
                    cluster_color_norm = (cluster_index - 1) / (n_clusters_global - 1 if n_clusters_global > 1 else 1)
                    cluster_color = cluster_cmap_global(cluster_color_norm)
                    # Create a single-color map for this cluster
                    single_color_cmap = matplotlib.colors.ListedColormap([cluster_color])

                    # Mask everything *except* the current cluster
                    overlay = np.ma.masked_where(~cluster_area_mask, cluster_mask_global)
                    ax.imshow(
                        overlay,
                        cmap=single_color_cmap,  # Use the single color map
                        origin="lower",
                        alpha=0.8,  # Make clusters slightly more opaque
                        vmin=cluster_index - 0.5,
                        vmax=cluster_index + 0.5,  # Center vmin/vmax on index
                    )

        # Set title
        title_lines = [f"AIA {channel} Å"]
        # Add cluster summary to title (optional, can get long)
        # if cluster_pixels_counts and cluster_anomaly_percentages and cluster_index <= len(cluster_pixels_counts):
        #     cluster_pixels = cluster_pixels_counts[cluster_index-1]
        #     cluster_percentage = cluster_anomaly_percentages[cluster_index-1]
        #     title_lines.append(f'Cluster {cluster_index}: {cluster_pixels} Pix ({cluster_percentage:.2f}%)')

        ax.set_title("\n".join(title_lines), color="black", fontsize=12, pad=5)  # Slightly smaller font
        ax.axis("off")

    # Add legend if clusters were found
    if cluster_patches_global:
        # Place legend outside the plot area to avoid overlap
        fig.legend(
            handles=cluster_patches_global,
            loc="center right",  # Position relative to the figure
            bbox_to_anchor=(1.0, 0.5),  # Adjust anchor to be outside
            fontsize="medium",  # Slightly larger legend font
            framealpha=0.9,
        )

    # Remove empty subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust rect to make space for legend if needed
    filename = os.path.join(
        output_dir,
        f"{file_type}_kmeans_anomaly_detection_thresh_{anomaly_threshold:.2f}.png",  # Updated filename
    )
    plt.savefig(filename, bbox_inches="tight", dpi=150)  # Increase DPI slightly
    plt.close(fig)
    print(f"Figure saved to: {filename}")


# --- Main Execution ---
def main():
    """Main function to execute SDO/AIA anomaly detection pipeline."""
    parser = argparse.ArgumentParser(description="SDO/AIA Anomaly Detection using Isolation Forest and K-Means")
    # --- Input/Output Arguments ---
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,  # Make data_dir required
        help="Path to parent directory containing AIA channel subdirectories.",
    )
    parser.add_argument("--output_dir", type=str, default="./output_figures", help="Output directory for figures.")
    parser.add_argument(
        "--file_type",
        type=str,
        default="fits",
        choices=["fits", "jp2"],  # Choice argument
        help="Type of input image files ('fits' or 'jp2').",
    )

    # --- Data Selection/Processing Arguments ---
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=None,
        help="AIA channels numbers (e.g., '94' '131' '171'). If None, uses default set.",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Resize image size (square).")
    parser.add_argument(
        "--jp2_mask_radius",
        type=int,
        default=1600,  # JP2 specific argument
        help="Fixed radius in pixels for JP2 mask (used if file_type is 'jp2'). Assumes 4096 original size.",
    )

    # --- Algorithm Parameters ---
    parser.add_argument(
        "--anomaly_thresholds",
        type=float,
        nargs="+",
        default=[-0.1, 0.0, 0.1],  # Example thresholds
        help="Anomaly score threshold(s) for Isolation Forest (lower scores are more anomalous).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.02,  # Adjusted default
        help="Estimated proportion of anomalies in the dataset for Isolation Forest.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,  # Adjusted default
        help="Number of clusters for K-Means.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    # parser.add_argument("--max_k", type=int, default=10, help="Max clusters for Elbow method.") # Elbow method not implemented here

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Determine Channels ---
    default_channels_num = ["94", "131", "171", "193", "211", "304", "335"]  # Common EUV + 304
    # Add other channels like '1600', '1700' if needed and available in data_dir
    # default_channels_num = ['94', '131', '171', '193', '211', '304', '335', '1600', '1700']

    if args.channels:
        channels_to_process = [f"aia_{c}" for c in args.channels]
    else:
        channels_to_process = [f"aia_{c}" for c in default_channels_num]

    print(f"Processing file type: {args.file_type.upper()}")
    print(f"Target channels: {', '.join(c.split('_')[1] for c in channels_to_process)}")

    # --- Load, Mask, and Preprocess Data ---
    masked_data_list, channel_names_loaded = [], []
    original_shapes = {}  # Store original shapes if needed later
    metadata_list = {}  # Store metadata if FITS

    for channel_dir_name in channels_to_process:
        channel_num = channel_dir_name.split("_")[1]
        channel_path = os.path.join(args.data_dir, channel_dir_name)

        if not os.path.isdir(channel_path):
            print(f"Warning: Directory not found for channel {channel_num}: {channel_path}. Skipping.")
            continue

        data, mask = None, None
        try:
            if args.file_type == "fits":
                data, metadata = load_fits_data(channel_path)
                if data is not None and metadata:
                    original_shapes[channel_num] = data.shape
                    metadata_list[channel_num] = metadata
                    mask = create_circular_mask_fits(data, metadata)
                else:
                    print(f"Failed to load FITS data or metadata for {channel_num}.")
            elif args.file_type == "jp2":
                data, _ = load_jp2_data_imageio(channel_path)  # Metadata is None
                if data is not None:
                    original_shapes[channel_num] = data.shape
                    # Check if jp2_mask_radius needs adjustment for non-4k images
                    if data.shape != (4096, 4096) and args.jp2_mask_radius == 1600:
                        # Simple scaling based on shortest dimension relative to 4096
                        scale_factor = min(data.shape) / 4096.0
                        scaled_radius = int(1600 * scale_factor)
                        print(
                            f"Warning: JP2 image shape {data.shape} is not 4096x4096. Adjusting mask radius from {args.jp2_mask_radius} to {scaled_radius}."
                        )
                        mask_radius_to_use = scaled_radius
                    else:
                        mask_radius_to_use = args.jp2_mask_radius
                    mask = create_circular_mask_jp2(data, mask_radius_to_use)
                else:
                    print(f"Failed to load JP2 data for {channel_num}.")

            # Proceed to preprocess if data and mask were successfully created
            if data is not None and mask is not None:
                masked_data = preprocess_image(data, mask, args.image_size)
                masked_data_list.append(masked_data)
                channel_names_loaded.append(channel_num)
            else:
                print(f"Skipping channel {channel_num} due to loading or masking errors.")

        except FileNotFoundError as e:
            print(f"Error processing {channel_dir_name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing {channel_dir_name}: {e}")
            # Optionally re-raise if debugging: raise e

    if not masked_data_list:
        print("No data successfully loaded and preprocessed. Exiting.")
        return

    print(f"Successfully processed {len(masked_data_list)} channels: {', '.join(channel_names_loaded)}")

    # --- Prepare Data for ML ---
    prepared_data, valid_pixel_mask_1d, nan_mask_1d = prepare_data_concatenated(masked_data_list)

    if prepared_data.shape[0] == 0:
        print("No valid pixels found after concatenation. Cannot proceed with anomaly detection.")
        return

    print(f"Prepared data shape for ML: {prepared_data.shape}")  # (n_valid_pixels, n_channels)
    print(
        f"Valid pixel mask (1D) shape: {valid_pixel_mask_1d.shape}, Sum: {np.sum(valid_pixel_mask_1d)}"
    )  # (n_total_pixels,)

    # --- Anomaly Detection ---
    anomaly_scores = detect_anomalies_isolation_forest(prepared_data, args.contamination)

    if anomaly_scores.size == 0:
        print("Anomaly detection returned no scores. Exiting.")
        return

    print(f"Anomaly scores shape: {anomaly_scores.shape}")  # (n_valid_pixels,)

    # Create a 2D map of anomaly scores (NaN where pixels were masked out initially)
    # Map the scores back to the original resized image grid
    anomaly_map_2d = np.full((args.image_size, args.image_size), np.nan)
    # Ensure valid_pixel_mask_1d is boolean
    valid_pixel_mask_1d = valid_pixel_mask_1d.astype(bool)
    # Reshape the 1D valid mask to 2D
    valid_pixel_mask_2d = valid_pixel_mask_1d.reshape((args.image_size, args.image_size))
    # Place scores in the valid locations
    anomaly_map_2d[valid_pixel_mask_2d] = anomaly_scores
    print(f"Anomaly map (2D) shape: {anomaly_map_2d.shape}, Non-NaN count: {np.sum(~np.isnan(anomaly_map_2d))}")

    # --- Loop Through Thresholds for Clustering and Plotting ---
    total_pixels_resized = args.image_size * args.image_size

    for anomaly_threshold in args.anomaly_thresholds:
        print(f"\n===== Processing with Anomaly Threshold: {anomaly_threshold} =====")

        # Determine anomalous pixels based on the threshold
        # Apply threshold only where scores exist (i.e., where anomaly_map_2d is not NaN)
        anomaly_mask_global_2d = np.full((args.image_size, args.image_size), False)  # Initialize with False
        valid_score_mask = ~np.isnan(anomaly_map_2d)
        anomaly_mask_global_2d[valid_score_mask] = anomaly_map_2d[valid_score_mask] < anomaly_threshold

        anomaly_pixels_count = np.sum(anomaly_mask_global_2d)
        anomaly_percentage = (
            (anomaly_pixels_count / np.sum(valid_score_mask)) * 100 if np.sum(valid_score_mask) > 0 else 0
        )

        print(f"Anomaly threshold: {anomaly_threshold}")
        print(f"Pixels considered anomalous: {anomaly_pixels_count} ({anomaly_percentage:.2f}% of valid pixels)")

        # --- Prepare data for Clustering (only anomalous pixels) ---
        # Find the 1D indices corresponding to the anomalous pixels in the *original* `prepared_data`
        # We need to select rows from `prepared_data` where the corresponding pixel in 2D is anomalous

        # Method 1: Use the 2D anomaly mask and the 2D valid mask
        # Find where pixels are BOTH valid (originally) AND anomalous (now)
        valid_and_anomalous_mask_2d = valid_pixel_mask_2d & anomaly_mask_global_2d
        # Get the 1D indices within the flattened array where this is true
        valid_and_anomalous_indices_flat = np.where(valid_and_anomalous_mask_2d.flatten())[0]

        # Now, we need the indices relative to `prepared_data`.
        # `prepared_data` corresponds to `flattened_data[valid_pixel_mask_1d]`.
        # Create a mapping from full flat index to prepared_data index
        np.arange(total_pixels_resized)
        prepared_data_indices = np.full(total_pixels_resized, -1, dtype=int)
        prepared_data_indices[valid_pixel_mask_1d] = np.arange(prepared_data.shape[0])

        # Get the indices into `prepared_data` for the anomalous pixels
        indices_for_clustering = prepared_data_indices[valid_and_anomalous_indices_flat]
        # Filter out any potential -1s (shouldn't happen if logic is right)
        indices_for_clustering = indices_for_clustering[indices_for_clustering != -1]

        if len(indices_for_clustering) == 0:
            print("No anomalous pixels found for clustering at this threshold.")
            anomaly_intensity_features = np.array([])  # Empty array
        else:
            anomaly_intensity_features = prepared_data[indices_for_clustering]

        print(f"Data shape for clustering: {anomaly_intensity_features.shape}")

        # --- Perform Clustering ---
        cluster_labels = np.array([])
        cluster_mask_final = np.zeros((args.image_size, args.image_size), dtype=int)
        cluster_cmap_final = matplotlib.colors.ListedColormap([])
        cluster_patches_final = []
        n_clusters_final = 0
        cluster_pixels_counts: list[int] = []
        cluster_anomaly_percentages: list[float] = []

        if anomaly_intensity_features.shape[0] > 0 and anomaly_intensity_features.shape[0] >= args.n_clusters:
            cluster_labels, _ = perform_kmeans_clustering(
                anomaly_intensity_features, args.n_clusters, args.random_state
            )

            # Create the 2D cluster map using the labels and the global anomaly mask
            cluster_mask_final, cluster_cmap_final, cluster_patches_final, n_clusters_final = create_cluster_mask(
                anomaly_mask_global_2d,  # Pass the 2D anomaly mask
                cluster_labels,
                valid_pixel_mask_1d,  # Pass the 1D mask of valid pixels before anomaly detection
                args.image_size,
            )

            # Calculate stats per cluster
            if n_clusters_final > 0:
                for cluster_index in range(1, n_clusters_final + 1):
                    cluster_pixel_count = np.sum(cluster_mask_final == cluster_index)
                    cluster_pixels_counts.append(cluster_pixel_count)
                    cluster_percentage = (
                        (cluster_pixel_count / anomaly_pixels_count) * 100 if anomaly_pixels_count > 0 else 0
                    )
                    cluster_anomaly_percentages.append(cluster_percentage)
                    print(
                        f"  Cluster {cluster_index}: {cluster_pixel_count} pixels ({cluster_percentage:.2f}% of total anomalies)"
                    )
        elif anomaly_intensity_features.shape[0] > 0:
            print(
                f"Not enough anomalous samples ({anomaly_intensity_features.shape[0]}) to form {args.n_clusters} clusters. Skipping clustering."
            )
        else:
            print("No data points for clustering. Skipping.")

        # --- Plot Results ---
        plot_results(
            masked_data_list=masked_data_list,
            cluster_mask_global=cluster_mask_final,
            cluster_cmap_global=cluster_cmap_final,
            n_clusters_global=n_clusters_final,
            cluster_patches_global=cluster_patches_final,
            channel_names=channel_names_loaded,
            anomaly_threshold=anomaly_threshold,
            output_dir=args.output_dir,
            total_pixels_resized=np.sum(valid_score_mask),  # Base total on valid pixels in resized img
            anomaly_pixels_count=anomaly_pixels_count,
            cluster_pixels_counts=cluster_pixels_counts,
            cluster_anomaly_percentages=cluster_anomaly_percentages,
            file_type=args.file_type,  # Pass file type for filename/title
            clustering_method_name="K-Means",
        )

    print(f"\nPipeline finished. Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
