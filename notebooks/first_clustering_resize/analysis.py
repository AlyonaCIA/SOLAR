import argparse  # Import argparse for command-line arguments
import os
from typing import Tuple

import matplotlib
import matplotlib.colors  # Import for creating colormaps
import matplotlib.patches as mpatches  # Import for legend patches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.cluster import MeanShift  # Import clustering tools
from sklearn.cluster import estimate_bandwidth
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

matplotlib.use('Agg')


# --- Helper Functions (unchanged) ---


def load_fits_data(channel_dir: str) -> Tuple[np.ndarray, dict]:
    """Load FITS data and metadata."""
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in directory: {channel_dir}")

    # Load ONLY the first FITS file
    fits_path = os.path.join(channel_dir, fits_files[0])
    aia_map = sunpy.map.Map(fits_path)
    return aia_map.data, aia_map.meta


def create_circular_mask(data, metadata):
    """Creates a circular mask for the solar disk."""
    ny, nx = data.shape
    x_center = nx // 2
    y_center = ny // 2

    cdelt1 = metadata.get("cdelt1", 1.0)  # Arcsec/píxel en X
    solar_radius_arcsec = metadata.get("rsun_obs", 960.0)  # Radio solar en arcsec
    solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))

    print(f"Metadata for channel: CDELT1={cdelt1}, RSUN_OBS={solar_radius_arcsec}")
    print(f"Solar radius in pixels: {solar_radius_pixels}")

    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    mask = distance_from_center <= solar_radius_pixels
    return mask


def preprocess_image(data: np.ndarray, mask: np.ndarray, size: int = 512) -> np.ndarray:
    """Resize and mask the image."""
    resized_data = resize(data, (size, size), mode="reflect", anti_aliasing=True)
    resized_mask = resize(mask, (size, size), mode="reflect", anti_aliasing=False) > 0.5
    masked_data = resized_data.copy()
    masked_data[~resized_mask] = np.nan
    return masked_data


# --- Modified Functions ---

# Modified return type
def prepare_data_concatenated(
        masked_data_list: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepares the data for anomaly detection by concatenating channels and
    normalizing.

    Args:
        masked_data_list (list): List of masked image data (2D arrays) for each channel.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: # Modified return description
            - Data reshaped to (pixels_without_nan, num_channels) and normalized.
            - A boolean mask indicating the positions of valid (non-NaN) pixels in the original 512x512 grid (1D reshaped).
            - The nan_mask (1D boolean array) itself. # ADDED nan_mask to the return
    """

    # Stack the masked data along a new axis (channels)
    # Shape: (512, 512, num_channels)
    stacked_data = np.stack(masked_data_list, axis=-1)
    print(f"Shape of stacked_data (before NaN removal): {stacked_data.shape}")

    # Reshape to (512*512, num_channels)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    print(f"Shape of reshaped_data (before NaN removal): {reshaped_data.shape}")

    # Identify rows containing NaNs
    # boolean mask: True for rows with NaN, False otherwise
    nan_mask = np.isnan(reshaped_data).any(axis=1)
    print(f"Number of rows with NaN: {np.sum(nan_mask)}")

    # Remove rows containing NaNs
    reshaped_data_cleaned = reshaped_data[~nan_mask]  # Select rows WITHOUT NaNs
    print(f"Shape of reshaped_data_cleaned (after NaN removal): {
          reshaped_data_cleaned.shape}")

    # Robust scaling
    scaler = RobustScaler()  # initialize
    reshaped_data_scaled = scaler.fit_transform(
        reshaped_data_cleaned)  # Fit and transform on cleaned data
    print(f"Shape of reshaped_data_scaled (after scaling): {
          reshaped_data_scaled.shape}")

    # Return scaled data, valid pixel mask, AND nan_mask
    return reshaped_data_scaled, ~nan_mask, nan_mask


def detect_anomalies_isolation_forest_decision(
        data: np.ndarray,
        contamination: float = 0.05) -> np.ndarray:
    """Detects anomalies using Isolation Forest's decision_function.

    Args:
        data (np.ndarray):  Prepared data (samples x features).
        contamination (float): Expected proportion of anomalies.

    Returns:
        np.ndarray: Anomaly scores (higher = more normal, lower = more anomalous).
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(data)  # Fit the model
    anomaly_scores = iso_forest.decision_function(data)  # Get anomaly scores
    print(f"Shape of anomaly_scores: {anomaly_scores.shape}")
    return anomaly_scores


def plot_isolation_forest_anomalies(
        ax, masked_data, anomaly_mask_channel, channel_name, threshold):
    """Plots the original image and overlays the anomaly mask from Isolation Forest
    (pre-clustering).

    Args:
        ax (plt.Axes): Matplotlib Axes object for plotting.
        masked_data (np.ndarray): Original masked image data for the channel.
        anomaly_mask_channel (np.ndarray): Anomaly mask for the CURRENT channel based on Isolation Forest.
        channel_name (str): Name of the channel.
        threshold (float): Anomaly threshold used.
    """

    # Display the original image in grayscale
    ax.imshow(masked_data, origin='lower', cmap='gray',
              vmin=np.nanpercentile(masked_data, 2),
              vmax=np.nanpercentile(masked_data, 98))

    # Overlay the anomaly mask in red
    ax.imshow(np.ma.masked_where(~anomaly_mask_channel, anomaly_mask_channel), origin='lower',
              # Red overlay for anomalies
              cmap=matplotlib.colors.ListedColormap(['red']), alpha=0.4, vmin=0.5, vmax=1.5)

    ax.set_title(f'AIA {channel_name} Å - Isolation Forest Anomalies',
                 color='black', fontsize=12, pad=10)
    ax.text(0.5, -0.18, f'Anomaly Threshold: {threshold:.2f}', ha='center',
            va='center', transform=ax.transAxes, fontsize=10, color='dimgray')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_clustered_anomalies(ax, masked_data, cluster_mask_global, cluster_cmap_global,
                             n_clusters_global, cluster_patches_global, channel_name, threshold):
    """Plots the original image and overlays GLOBAL cluster mask (only cluster areas
    colored).

    Args:
        ax (plt.Axes): Matplotlib Axes object for plotting.
        masked_data (np.ndarray): Original masked image data for the channel.
        cluster_mask_global (np.ndarray): GLOBAL cluster mask calculated across all channels.
        cluster_cmap_global (matplotlib.colors.ListedColormap): Colormap for GLOBAL clusters.
        n_clusters_global (int): Number of GLOBAL clusters.
        cluster_patches_global (list): List of legend patches for global clusters.
        channel_name (str): Name of the channel.
        threshold (float): Anomaly threshold used.
    """
    # Display the original image - now we will OVERLAY clusters on top of this
    ax.imshow(masked_data, origin='lower', cmap='gray',  # Use gray colormap
              # Adjust vmin and vmax for better contrast
              vmin=np.nanpercentile(masked_data, 5),
              vmax=np.nanpercentile(masked_data, 95))

    # Overlay the GLOBAL cluster mask - only color the CLUSTER areas, leave
    # background grayscale
    if n_clusters_global > 0:
        # Iterate through each cluster index (1, 2, 3...)
        for cluster_index in range(1, n_clusters_global + 1):
            # Mask for pixels belonging to THIS cluster
            cluster_area_mask = cluster_mask_global == cluster_index
            cluster_color = cluster_cmap_global(
                (cluster_index - 1) / n_clusters_global)  # Get color for this cluster
            ax.imshow(np.ma.masked_where(~cluster_area_mask, cluster_mask_global), origin='lower', cmap=matplotlib.colors.ListedColormap(
                # Overlay ONLY cluster area
                [cluster_color]), alpha=0.6, vmin=cluster_index - 0.5, vmax=cluster_index + 0.5)

    ax.set_title(f'AIA {channel_name} Å - Clustered Anomalies', color='black',
                 fontsize=12, pad=10)  # Title with GLOBAL cluster count
    ax.text(0.5, -0.18, f'Anomaly Threshold: {threshold:.2f}', ha='center', va='center',
            # Adjusted subtitle position and fontsize
            transform=ax.transAxes, fontsize=10, color='dimgray')

    ax.set_xticks([])  # Remove x ticks for cleaner visualization
    ax.set_yticks([])  # Remove y ticks for cleaner visualization
    ax.spines['top'].set_visible(False)  # Remove spines for cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if cluster_patches_global:  # Add legend only if there are patches to show
        ax.legend(handles=cluster_patches_global,
                  loc='upper right', fontsize='small')  # Add legend


def plot_comparison_decision(ax, masked_data_list, anomaly_mask_channel, cluster_mask_global,
                             cluster_cmap_global, n_clusters_global, cluster_patches_global, channel_names, threshold, i):
    """Plots the original image, and overlays GLOBAL cluster mask (only cluster areas
    colored).

    Args:
        ax (plt.Axes): Matplotlib Axes object for plotting.
        masked_data_list (list): List of original masked image data (for shape reference).
        anomaly_mask_channel (np.ndarray): Anomaly mask for the CURRENT channel (for individual channel anomaly overlay if needed).
        cluster_mask_global (np.ndarray): GLOBAL cluster mask calculated across all channels.
        cluster_cmap_global (matplotlib.colors.ListedColormap): Colormap for GLOBAL clusters.
        n_clusters_global (int): Number of GLOBAL clusters.
        cluster_patches_global (list): List of legend patches for global clusters.
        channel_names (list): List of channel names.
        threshold (float): Threshold to classify anomalies based on anomaly scores.
        i (int): Channel's index
    """
    masked_data = masked_data_list[i]
    channel = channel_names[i]

    # Display the original image - now we will OVERLAY clusters on top of this
    ax.imshow(masked_data, origin='lower', cmap='gray',  # Use gray colormap
              # Adjust vmin and vmax for better contrast
              vmin=np.nanpercentile(masked_data, 2),
              vmax=np.nanpercentile(masked_data, 98))

    # Overlay the GLOBAL cluster mask - only color the CLUSTER areas, leave
    # background grayscale
    if n_clusters_global > 0:
        # Iterate through each cluster index (1, 2, 3...)
        for cluster_index in range(1, n_clusters_global + 1):
            # Mask for pixels belonging to THIS cluster
            cluster_area_mask = cluster_mask_global == cluster_index
            cluster_color = cluster_cmap_global(
                (cluster_index - 1) / n_clusters_global)  # Get color for this cluster
            ax.imshow(np.ma.masked_where(~cluster_area_mask, cluster_mask_global), origin='lower', cmap=matplotlib.colors.ListedColormap(
                # Overlay ONLY cluster area
                [cluster_color]), alpha=0.6, vmin=cluster_index - 0.5, vmax=cluster_index + 0.5)

    # else: # No clusters to overlay - original grayscale image is shown as is

    ax.set_title(f'AIA {channel} Å (Global Clusters: {n_clusters_global})',
                 color='black', fontsize=14, pad=10)  # Title with GLOBAL cluster count
    ax.text(0.5, -0.18, f'Anomaly Threshold: {threshold:.2f}', ha='center', va='center',
            # Adjusted subtitle position and fontsize
            transform=ax.transAxes, fontsize=12, color='dimgray')

    ax.set_xticks([])  # Remove x ticks for cleaner visualization
    ax.set_yticks([])  # Remove y ticks for cleaner visualization
    ax.spines['top'].set_visible(False)  # Remove spines for cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if cluster_patches_global:  # Add legend only if there are patches to show
        ax.legend(handles=cluster_patches_global,
                  loc='upper right', fontsize='small')  # Add legend


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDO/AIA Anomaly Detection using Isolation Forest")
    parser.add_argument(
        "--anomaly_thresholds",
        type=float,
        nargs='+',  # Allow multiple values for anomaly_thresholds
        default=[-0.1],  # Example default thresholds - let's focus on one for now
        help="Threshold(s) for anomaly detection (lower values are more sensitive). Can provide multiple thresholds separated by spaces.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_figures",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--bandwidth_quantile",
        type=float,
        default=0.2,  # Default quantile as before
        help="Quantile parameter for bandwidth estimation in MeanShift clustering (lower values = smaller bandwidth = more clusters).",
    )

    args = parser.parse_args()
    anomaly_thresholds = args.anomaly_thresholds
    output_dir = args.output_dir
    bandwidth_quantile = args.bandwidth_quantile  # Get quantile from arguments

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Update this path if necessary
    data_dir = "/mnt/c/Users/juague/Documents/Personal/Other/Alyona/SOLAR_old/tmp/sdo_data"
    print(f"Data directory is set to: {data_dir}")
    print(f"Contents of data directory: {os.listdir(data_dir)}")

    channels = []
    all_dirs = os.listdir(data_dir)
    print(f"All directories found in data_dir: {all_dirs}")  # Print all directories

    for d in all_dirs:
        full_path = os.path.join(data_dir, d)
        is_dir = os.path.isdir(full_path)
        excludes_1600 = not d.startswith("aia_1600")
        excludes_1700 = not d.startswith("aia_1700")

        print(f"Checking directory: {d}, is_dir: {is_dir}, excludes_1600: {
              excludes_1600}, excludes_1700: {excludes_1700}")  # Detailed check

        if is_dir and excludes_1600 and excludes_1700:
            channels.append(d)
            print(f"  Adding channel: {d}")  # Confirm channel is added

    print(f"Channels list after filtering: {channels}")
    num_channels = len(channels)
    print(f"Number of channels to process: {num_channels}")

    # Parameters
    image_size = 512  # Use the actual processed image size, previously was hardcoded to 512, but now is 2048 after resizing to original
    contamination = 0.05  # Adjust this value

    # 1. Load and Preprocess Images (Do this ONCE outside the threshold loop)
    masked_data_list = []
    channel_names = []
    for channel_dir in channels:  # iterate over channels
        try:
            # get channel name, example aia_171 -> 171
            channel = channel_dir.split("_")[1]

            channel_path = os.path.join(data_dir, channel_dir)
            print(f"Loading data for channel: {channel}")
            data, metadata = load_fits_data(channel_path)
            print(f"Shape of original data for channel {channel}: {data.shape}")
            mask = create_circular_mask(data, metadata)
            # Use the correct image_size
            masked_data = preprocess_image(data, mask, image_size)
            print(f"Shape of masked data for channel {channel}: {masked_data.shape}")
            masked_data_list.append(masked_data)
            channel_names.append(channel)

        except Exception as e:
            print(f"Error processing {channel_dir}: {e}")

    # 2. Prepare data concatenated for Isolation Forest (Do this ONCE outside
    # the threshold loop)
    if masked_data_list:  # Check if masked_data_list is not empty before proceeding
        prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(
            masked_data_list)

        # 3. Detect Anomalies (Do this ONCE outside the threshold loop - anomaly
        # scores are independent of threshold)
        anomaly_scores = detect_anomalies_isolation_forest_decision(
            prepared_data, contamination)

        # Reshape anomaly scores back to 2048x2048 for anomaly map creation (use
        # correct image_size)
        anomaly_map_2d = np.full((image_size, image_size), np.nan)
        anomaly_map_2d[valid_pixel_mask.reshape(
            (image_size, image_size))] = anomaly_scores

        # 4. Loop through thresholds and Visualize & Save Anomalies
        for anomaly_threshold in anomaly_thresholds:
            # GLOBAL anomaly mask based on threshold
            anomaly_mask_global = anomaly_map_2d < anomaly_threshold

            # --- Clustering GLOBAL Anomalies using MeanShift (applied to GLOBAL anomaly mask) ---
            anomaly_pixels_indices_global = np.argwhere(
                anomaly_mask_global)  # Get indices of GLOBAL anomaly pixels
            cluster_patches_global = []  # List for GLOBAL legend patches
            cluster_mask_global = np.zeros_like(
                anomaly_mask_global, dtype=int)  # GLOBAL cluster mask

            n_clusters_global = 0  # Initialize GLOBAL cluster count
            cluster_cmap_global = matplotlib.colors.ListedColormap(
                [])  # Initialize GLOBAL colormap

            if len(anomaly_pixels_indices_global) > 0:  # Only cluster if there are GLOBAL anomalies
                # --- MODIFICATION START ---
                # 1. Create a mapping from 2D pixel indices to prepared_data row indices
                valid_pixel_mask_2d = ~nan_mask.reshape((image_size, image_size))
                valid_pixel_indices_2d = np.argwhere(
                    valid_pixel_mask_2d)  # Get 2D indices of valid pixels
                pixel_index_map = {}  # Dictionary to map 2D index to prepared_data row index
                for i, index_2d in enumerate(valid_pixel_indices_2d):
                    # Map (row, col) to row index in prepared_data
                    pixel_index_map[tuple(index_2d)] = i

                # 2. Extract intensity features for anomalous pixels using the index map
                anomaly_intensity_features = []
                # Keep track of anomaly pixels that are actually in prepared_data
                valid_anomaly_pixel_indices_global = []
                for anomaly_pixel_index_2d in anomaly_pixels_indices_global:
                    # Check if anomaly pixel is in valid pixel set
                    if tuple(anomaly_pixel_index_2d) in pixel_index_map:
                        prepared_data_row_index = pixel_index_map[tuple(
                            anomaly_pixel_index_2d)]
                        anomaly_intensity_features.append(
                            prepared_data[prepared_data_row_index])
                        valid_anomaly_pixel_indices_global.append(
                            anomaly_pixel_index_2d)  # Store valid anomaly pixel indices

                anomaly_intensity_features = np.array(
                    anomaly_intensity_features)  # Convert to numpy array
                valid_anomaly_pixel_indices_global = np.array(
                    valid_anomaly_pixel_indices_global)  # Convert to numpy array

                if anomaly_intensity_features.size > 0:  # Proceed only if there are valid anomaly pixels after NaN removal

                    # Estimate bandwidth for MeanShift based on INTENSITY FEATURES
                    bandwidth_global = estimate_bandwidth(anomaly_intensity_features, quantile=bandwidth_quantile, n_samples=min(
                        500, len(anomaly_intensity_features)))  # Use intensity features

                    if bandwidth_global > 0:  # Bandwidth must be positive for MeanShift
                        ms_global = MeanShift(
                            bandwidth=bandwidth_global, bin_seeding=True)
                        # FIT on INTENSITY FEATURES
                        ms_global.fit(anomaly_intensity_features)
                        labels_global = ms_global.labels_
                        cluster_labels_global = np.unique(labels_global)
                        n_clusters_global = len(cluster_labels_global)
                        print(f"  Global Clustering (Threshold {anomaly_threshold:.2f}, Quantile {bandwidth_quantile}): Estimated {
                              n_clusters_global} clusters, Bandwidth: {bandwidth_global:.2f}")  # Print bandwidth

                        # Create a colormap for GLOBAL clusters
                        cluster_cmap_global = matplotlib.colors.ListedColormap(plt.cm.tab20(
                            # Use tab10 instead of viridis
                            np.linspace(0, 1, n_clusters_global)))

                        # Reset cluster mask here to ensure correct size
                        cluster_mask_global = np.zeros_like(
                            anomaly_mask_global, dtype=int)

                        for cluster_idx in range(n_clusters_global):
                            # Get pixel indices for pixels belonging to this intensity
                            # cluster
                            cluster_intensity_indices_in_anomaly = np.argwhere(
                                # Indices in anomaly_intensity_features
                                labels_global == cluster_idx).flatten()
                            # Get the actual pixel indices from valid anomaly pixel set
                            cluster_pixels_indices = valid_anomaly_pixel_indices_global[
                                cluster_intensity_indices_in_anomaly]

                            if len(cluster_pixels_indices) > 0:
                                # Assign cluster label in pixel space
                                cluster_mask_global[tuple(
                                    cluster_pixels_indices.T)] = cluster_idx + 1
                                cluster_color = cluster_cmap_global(
                                    cluster_idx / n_clusters_global)
                                cluster_patches_global.append(mpatches.Patch(
                                    color=cluster_color, label=f'Cluster {cluster_idx + 1}'))

                    else:  # Bandwidth too small for GLOBAL clustering
                        print(f"  Global Clustering (Threshold {anomaly_threshold:.2f}, Quantile {
                              bandwidth_quantile}): Bandwidth too small, no global clusters estimated.")
                        cluster_cmap_global = matplotlib.colors.ListedColormap(
                            ['Reds'])  # Fallback colormap
                        cluster_patches_global.append(mpatches.Patch(
                            color='red', label='Global Anomalies (No Clusters)'))
                else:
                    print(f"  Global Clustering (Threshold {anomaly_threshold:.2f}, Quantile {
                          bandwidth_quantile}): No valid anomaly pixels after NaN removal. No clustering performed.")
                    cluster_patches_global.append(mpatches.Patch(
                        color='none', label='No Valid Global Anomalies for Clustering'))

            else:  # No GLOBAL anomalies
                print(f"  Global Clustering (Threshold {anomaly_threshold:.2f}, Quantile {
                      bandwidth_quantile}): No global anomalies detected.")
                cluster_patches_global.append(mpatches.Patch(
                    color='none', label='No Global Anomalies'))

            num_rows = 3
            num_cols = 3
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(
                18, 15), dpi=100)  # Increased figsize and dpi
            axes = axes.flatten()

            fig.suptitle(f'Anomaly Detection in SDO/AIA EUV Channels using Isolation Forest\nAnomaly Threshold: {
                         anomaly_threshold:.2f}\nBandwidth Quantile: {bandwidth_quantile}', fontsize=18)  # Main title with threshold and quantile

            for i in range(num_channels):
                # Create anomaly mask for each channel individually for visualization purposes (if needed)
                # Example: using original masked data as anomaly score for each channel
                # - ADJUST if needed
                anomaly_mask_channel = masked_data_list[i] < anomaly_threshold

                plot_comparison_decision(axes[i], masked_data_list, anomaly_mask_channel, cluster_mask_global, cluster_cmap_global, n_clusters_global,
                                         # Pass GLOBAL cluster info  <--- CORRECTED
                                         # FUNCTION CALL HERE
                                         cluster_patches_global, channel_names, anomaly_threshold, i)

                if i >= num_channels:  # hide unused subplots if num_channels < num_rows * num_cols
                    axes[i].axis('off')

            # Add GLOBAL legend to the last subplot if there are clusters
            if cluster_patches_global and num_channels < num_rows * \
                    num_cols:  # Add legend to the last used subplot
                axes[num_channels - 1].legend(handles=cluster_patches_global,
                                              loc='upper right', fontsize='small')
            elif cluster_patches_global and num_channels == num_rows * num_cols:  # Add to the very last subplot if all are used
                axes[-1].legend(handles=cluster_patches_global,
                                loc='upper right', fontsize='small')

            # Remove any extra subplots if fewer channels than grid spots.
            for j in range(num_channels, num_rows * num_cols):
                fig.delaxes(axes[j])

            # Adjusted layout parameters: reduced w_pad and h_pad for spacing, adjust
            # rect for suptitle
            plt.tight_layout(rect=[0, 0, 1, 0.92], w_pad=0.1, h_pad=0.1)
            # Added subplots_adjust for finer control
            plt.subplots_adjust(wspace=0.1, hspace=0.3)

            # Save the figure instead of showing it
            filename = os.path.join(output_dir, f"anomaly_detection_threshold_{
                                    anomaly_threshold:.2f}_quantile_{bandwidth_quantile:.2f}_global_clusters.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig)  # Close figure to free memory

            print(f"Figure saved to: {filename}")

    else:
        print("No channels were processed. Please check the data directory and channel folders.")
