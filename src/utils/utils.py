"""Module to track the time or progress in our model using a progress bar and formatting
time, som utilities here"."""
# src/solar/utils.py

import logging
import os

import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)


def load_fits_data(channel_dir: str) -> tuple[np.ndarray, dict]:
    """Load FITS data and metadata from a single channel directory. Loads only the
    *first* FITS file in the directory.

    Args:
        channel_dir (str): Path to the channel directory.

    Returns:
        Tuple[np.ndarray, dict]: Data and metadata from the FITS file.

    Raises:
        FileNotFoundError: If no FITS files are found in the directory.
        Exception: For errors during FITS file reading.
    """
    log.debug(f"Searching for FITS files in: {channel_dir}")
    # Use lower() in case extensions are capitalized (.FITS)
    fits_files = [f for f in os.listdir(channel_dir) if f.lower().endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in directory: {channel_dir}")

    # Load the first FITS file alphabetically for consistency
    fits_files.sort()
    fits_path = os.path.join(channel_dir, fits_files[0])
    log.info(f"Loading FITS file: {fits_path}")
    try:
        aia_map = sunpy.map.Map(fits_path)
        # Convert to float32 early to avoid type issues later
        data = np.array(aia_map.data).astype(np.float32)
        return data, aia_map.meta
    except Exception as e:
        log.error(f"Error loading FITS file {fits_path}: {e}")
        raise  # Re-raise the exception


def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates a circular mask for the solar disk based on metadata.

    Args:
        data (np.ndarray): Image data.
        metadata (dict): FITS metadata containing header info.

    Returns:
        np.ndarray: Boolean mask, True for pixels inside the solar disk.
    """
    log.debug("Creating circular solar disk mask.")
    ny, nx = data.shape
    try:
        # Use header keywords for center and radius (CRPIX is 1-based)
        x_center_fits = metadata.get("CRPIX1", nx / 2.0 + 0.5)
        y_center_fits = metadata.get("CRPIX2", ny / 2.0 + 0.5)
        x_center = x_center_fits - 1.0  # Convert to 0-based index
        y_center = y_center_fits - 1.0

        cdelt1 = abs(metadata.get("CDELT1", 1.0))  # Pixels per arcsec, ensure positive
        if cdelt1 == 0:  # Avoid division by zero
            log.warning("CDELT1 is zero, using 1.0.")
            cdelt1 = 1.0

        solar_radius_arcsec = metadata.get("RSUN_OBS", metadata.get("R_SUN", 960.0))  # Observed or nominal radius
        if solar_radius_arcsec is None:  # Handle case where keys exist but value is None
            log.warning("RSUN_OBS/R_SUN is None, using 960.0 arcsec.")
            solar_radius_arcsec = 960.0

        solar_radius_pixels = solar_radius_arcsec / cdelt1
        log.debug(f"Disk mask params: Center=({x_center:.2f}, {y_center:.2f}), Radius={solar_radius_pixels:.2f} pix")
    except Exception as e:
        log.error(
            f"Error reading metadata for mask creation: {e}. Using fallback to image center and min dimension / 2.",
            exc_info=True,
        )
        x_center, y_center = nx // 2, ny // 2
        solar_radius_pixels = min(nx, ny) / 2.0

    y_coords, x_coords = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)
    mask = distance_from_center <= solar_radius_pixels
    log.debug(f"Generated mask shape: {mask.shape}, Masked pixels (inside circle): {np.sum(mask)}")
    return mask


def preprocess_image(data: np.ndarray, mask: np.ndarray, target_size: int | None = None) -> np.ndarray:
    """Resizes the image and applies the mask, setting masked areas to NaN.

    Args:
        data (np.ndarray): Input image data (preferably float32).
        mask (np.ndarray): Boolean mask for the solar disk (same shape as data).
        target_size (int, optional): Desired size of the resized image (square). If None, no resize. Defaults to None.

    Returns:
        np.ndarray: Preprocessed image data with mask applied and optionally resized.
    """
    log.debug(f"Preprocessing single image. Original shape: {data.shape}, Target size: {target_size}")

    if data.shape != mask.shape:
        raise ValueError(f"Data shape {data.shape} and mask shape {mask.shape} mismatch.")

    if target_size is not None:
        target_shape = (target_size, target_size)
        if data.shape == target_shape:
            log.debug("Image already at target size, skipping resize.")
            resized_data = data
            resized_mask = mask  # Assuming mask is already correct size if data is
        else:
            log.debug(f"Resizing image and mask to {target_shape}")
            # Ensure both data and mask are resized if needed
            # Use float32 data for resize, it's common practice
            resized_data = resize(data.astype(np.float32), target_shape, mode="reflect", anti_aliasing=True)
            # Mask resize should be order 0 (nearest neighbor) to keep boolean nature
            resized_mask = resize(mask, target_shape, mode="reflect", anti_aliasing=False, order=0) > 0.5
    else:
        log.debug("No resizing requested, using original size.")
        resized_data = data
        resized_mask = mask  # Use original mask

    log.debug(f"Applying mask. Mask shape: {resized_mask.shape}, True values: {np.sum(resized_mask)}")
    masked_data = resized_data.copy()  # Work on a copy after resize/using original
    masked_data[~resized_mask] = np.nan  # Apply mask by setting outside pixels to NaN
    log.debug(f"Preprocessing complete for single image. Output shape: {masked_data.shape}")
    return masked_data


def prepare_data_concatenated(
    masked_data_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Concatenates masked data from multiple channels, handles NaNs, and scales the
    data.

    Args:
        masked_data_list (list): List of masked data arrays (one per channel).
                                 Assumes all arrays have the same 2D shape.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
            - scaled_data: Scaled data, reshaped to (pixels_without_nan, num_channels).
            - valid_pixel_mask_flat: 1D boolean mask (True for valid pixels in flattened array).
            - nan_mask_flat: 1D boolean mask (True for pixels with NaN in any channel, flattened).
            - image_shape: The 2D shape (height, width) of the original/resized images.
    """
    if not masked_data_list:
        raise ValueError("Masked data list is empty.")

    image_shape = masked_data_list[0].shape
    log.info(f"Preparing data for model from {len(masked_data_list)} channels. Image shape: {image_shape}")

    # Check that all images have the same shape
    for i, img in enumerate(masked_data_list):
        if img.shape != image_shape:
            raise ValueError(f"Image shape inconsistency: Channel {i} has shape {img.shape}, expected {image_shape}.")

    stacked_data = np.stack(masked_data_list, axis=-1)  # Shape (H, W, C)
    n_pixels_flattened = stacked_data.shape[0] * stacked_data.shape[1]
    n_channels = stacked_data.shape[2]

    reshaped_data = stacked_data.reshape((n_pixels_flattened, n_channels))  # Shape (H*W, C)

    # Identify pixels that are NaN in *any* channel (these are outside the
    # solar disk mask)
    nan_mask_flat = np.isnan(reshaped_data).any(axis=1)  # Shape (H*W,)
    valid_pixel_mask_flat = ~nan_mask_flat  # Shape (H*W,)

    # Count valid pixels (which are inside the solar disk and not NaN in any channel)
    total_valid_pixels = int(np.sum(valid_pixel_mask_flat))
    log.info(f"Found {total_valid_pixels} valid pixels out of {n_pixels_flattened} total after removing NaNs.")

    if total_valid_pixels == 0:
        raise ValueError("No valid pixels found after handling NaNs. Cannot proceed with modeling.")

    # Select only the data rows corresponding to valid pixels
    cleaned_data = reshaped_data[valid_pixel_mask_flat]  # Shape (num_valid_pixels, C)

    # Scale the valid data
    scaler = RobustScaler()  # RobustScaler handles outliers well
    log.info(f"Applying RobustScaler to {cleaned_data.shape[0]} valid pixels...")
    scaled_data = scaler.fit_transform(cleaned_data)  # Shape (num_valid_pixels, C)
    log.info(f"Scaling complete. Scaled data shape: {scaled_data.shape}")

    return scaled_data, valid_pixel_mask_flat, nan_mask_flat, image_shape
