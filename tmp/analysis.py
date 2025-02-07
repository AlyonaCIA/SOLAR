import os
import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from typing import Tuple
import argparse  # Import argparse for command-line arguments

# --- Helper Functions (unchanged) ---

def load_fits_data(channel_dir: str) -> Tuple[np.ndarray, dict]:
    """Load FITS data and metadata."""
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in directory: {channel_dir}")

    fits_path = os.path.join(channel_dir, fits_files[0]) # Load ONLY the first FITS file
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

def prepare_data_concatenated(masked_data_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares the data for anomaly detection by concatenating channels and normalizing.

    Args:
        masked_data_list (list): List of masked image data (2D arrays) for each channel.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Data reshaped to (pixels_without_nan, num_channels) and normalized.
            - A boolean mask indicating the positions of valid (non-NaN) pixels in the original 512x512 grid.
    """

    # Stack the masked data along a new axis (channels)
    stacked_data = np.stack(masked_data_list, axis=-1) # Shape: (512, 512, num_channels)
    print(f"Shape of stacked_data (before NaN removal): {stacked_data.shape}")

    # Reshape to (512*512, num_channels)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    print(f"Shape of reshaped_data (before NaN removal): {reshaped_data.shape}")

    # Identify rows containing NaNs
    nan_mask = np.isnan(reshaped_data).any(axis=1) # boolean mask: True for rows with NaN, False otherwise
    print(f"Number of rows with NaN: {np.sum(nan_mask)}")

    # Remove rows containing NaNs
    reshaped_data_cleaned = reshaped_data[~nan_mask] # Select rows WITHOUT NaNs
    print(f"Shape of reshaped_data_cleaned (after NaN removal): {reshaped_data_cleaned.shape}")

    #Robust scaling
    scaler = RobustScaler() #initialize
    reshaped_data_scaled = scaler.fit_transform(reshaped_data_cleaned) # Fit and transform on cleaned data
    print(f"Shape of reshaped_data_scaled (after scaling): {reshaped_data_scaled.shape}")


    return reshaped_data_scaled, ~nan_mask # Return scaled data and the *inverse* of nan_mask (valid pixel mask)

def detect_anomalies_isolation_forest_decision(
    data: np.ndarray,
    contamination: float = 0.05) -> np.ndarray:
    """
    Detects anomalies using Isolation Forest's decision_function.

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

def plot_comparison_decision(ax, masked_data_list, anomaly_scores, channel_names, threshold, i, valid_pixel_mask):
    """
    Plots the original image and overlays a mask highlighting anomalies based on the decision function.

    Args:
        ax (plt.Axes): Matplotlib Axes object for plotting.
        masked_data_list (list): List of original masked image data (for shape reference).
        anomaly_scores (np.ndarray): Anomaly scores from Isolation Forest's decision_function.
        channel_names (list): List of channel names.
        threshold (float): Threshold to classify anomalies based on anomaly scores.
        i (int): Channel's index
        valid_pixel_mask (np.ndarray): Boolean mask of valid pixels.
    """
    masked_data = masked_data_list[i]
    channel = channel_names[i]

    # Initialize anomaly_map with NaNs in the shape of the original image
    anomaly_map = np.full(masked_data.shape, np.nan)

    # Place the anomaly scores into the anomaly map at the positions indicated by valid_pixel_mask
    anomaly_map[valid_pixel_mask.reshape(masked_data.shape)] = anomaly_scores

    # Create a mask for anomalies based on the threshold
    anomaly_mask = anomaly_map < threshold

    # Display the original image with better contrast and colormap
    im = ax.imshow(masked_data, origin='lower', cmap='gray',  # Use gray colormap
                   vmin=np.nanpercentile(masked_data, 5), # Adjust vmin and vmax for better contrast
                   vmax=np.nanpercentile(masked_data, 95))

    # Overlay the anomaly mask with a different colormap and transparency
    ax.imshow(anomaly_mask, origin='lower', cmap='YlOrRd', alpha=0.6) # Use YlOrRd for anomalies, adjust alpha
    ax.set_title(f'AIA {channel} Å', color='black', fontsize=14, pad=10) # Increased fontsize and padding for title
    ax.text(0.5, -0.18, f'Anomaly Threshold: {threshold:.2f}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='dimgray') # Adjusted subtitle position and fontsize

    ax.set_xticks([]) # Remove x ticks for cleaner visualization
    ax.set_yticks([]) # Remove y ticks for cleaner visualization
    ax.spines['top'].set_visible(False) # Remove spines for cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDO/AIA Anomaly Detection using Isolation Forest")
    parser.add_argument(
        "--anomaly_thresholds",
        type=float,
        nargs='+',  # Allow multiple values for anomaly_thresholds
        default=[-0.2, -0.1, 0.0, 0.1], # Example default thresholds
        help="Threshold(s) for anomaly detection (lower values are more sensitive). Can provide multiple thresholds separated by spaces.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_figures",
        help="Directory to save output figures.",
    )
    args = parser.parse_args()
    anomaly_thresholds = args.anomaly_thresholds
    output_dir = args.output_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    data_dir = "./sdo_data/"  # Update this path if necessary
    print(f"Data directory is set to: {data_dir}")
    print(f"Contents of data directory: {os.listdir(data_dir)}")

    channels = []
    all_dirs = os.listdir(data_dir)
    print(f"All directories found in data_dir: {all_dirs}") # Print all directories

    for d in all_dirs:
        full_path = os.path.join(data_dir, d)
        is_dir = os.path.isdir(full_path)
        excludes_1600 = not d.startswith("aia_1600")
        excludes_1700 = not d.startswith("aia_1700")

        print(f"Checking directory: {d}, is_dir: {is_dir}, excludes_1600: {excludes_1600}, excludes_1700: {excludes_1700}") # Detailed check

        if is_dir and excludes_1600 and excludes_1700:
            channels.append(d)
            print(f"  Adding channel: {d}") # Confirm channel is added


    print(f"Channels list after filtering: {channels}")
    num_channels = len(channels)
    print(f"Number of channels to process: {num_channels}")

    # Parameters
    image_size = 512
    contamination = 0.05  #Adjust this value

    #1. Load and Preprocess Images (Do this ONCE outside the threshold loop)
    masked_data_list = []
    channel_names = []
    for channel_dir in channels: #iterate over channels
      try:
        channel = channel_dir.split("_")[1] #get channel name, example aia_171 -> 171

        channel_path = os.path.join(data_dir, channel_dir)
        print(f"Loading data for channel: {channel}")
        data, metadata = load_fits_data(channel_path)
        print(f"Shape of original data for channel {channel}: {data.shape}")
        mask = create_circular_mask(data, metadata)
        masked_data = preprocess_image(data, mask, image_size)
        print(f"Shape of masked data for channel {channel}: {masked_data.shape}")
        masked_data_list.append(masked_data)
        channel_names.append(channel)

      except Exception as e:
        print(f"Error processing {channel_dir}: {e}")

    #2. Prepare data concatenated for Isolation Forest (Do this ONCE outside the threshold loop)
    if masked_data_list: # Check if masked_data_list is not empty before proceeding
        prepared_data, valid_pixel_mask = prepare_data_concatenated(masked_data_list)

        #3. Detect Anomalies (Do this ONCE outside the threshold loop - anomaly scores are independent of threshold)
        anomaly_scores = detect_anomalies_isolation_forest_decision(prepared_data, contamination)


        #4. Loop through thresholds and Visualize & Save Anomalies
        for anomaly_threshold in anomaly_thresholds:
            num_rows = 3
            num_cols = 3
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), dpi=100) # Increased figsize and dpi
            axes = axes.flatten()

            fig.suptitle(f'Anomaly Detection in SDO/AIA EUV Channels using Isolation Forest\nAnomaly Threshold: {anomaly_threshold:.2f}', fontsize=18) # Main title with threshold

            for i in range(num_channels):
                plot_comparison_decision(axes[i], masked_data_list, anomaly_scores, channel_names, anomaly_threshold, i, valid_pixel_mask)
                if i >= num_channels: # hide unused subplots if num_channels < num_rows * num_cols
                    axes[i].axis('off')

            # Remove any extra subplots if fewer channels than grid spots.
            for j in range(num_channels, num_rows * num_cols):
                fig.delaxes(axes[j])


            plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.1, h_pad=0.1) # Adjusted layout parameters: reduced w_pad and h_pad for spacing, adjust rect for suptitle
            plt.subplots_adjust(wspace=0.1, hspace=0.3) # Added subplots_adjust for finer control

            # Save the figure instead of showing it
            filename = os.path.join(output_dir, f"anomaly_detection_threshold_{anomaly_threshold:.2f}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close(fig) # Close figure to free memory

            print(f"Figure saved to: {filename}")


    else:
        print("No channels were processed. Please check the data directory and channel folders.")