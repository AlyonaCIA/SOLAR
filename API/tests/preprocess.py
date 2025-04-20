import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List


def create_circular_mask(image: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Create a circular mask for the solar disk based on FITS metadata.

    Args:
        image (np.ndarray): 2D image data array.
        metadata (dict): FITS header metadata containing spatial info.

    Returns:
        np.ndarray: Boolean mask where True indicates pixels inside the solar disk.
    """
    height, width = image.shape
    x_center, y_center = width // 2, height // 2
    cdelt1 = metadata.get("cdelt1", 1.0)  # Arcsec per pixel in X-axis
    solar_radius_arcsec = metadata.get("rsun_obs", 960.0)  # Solar radius in arcseconds
    solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - x_center)**2 + (y - y_center)**2)

    return distance <= solar_radius_pixels


def preprocess_image(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int = 512
) -> np.ndarray:
    """
    Resize an image and apply a binary mask, setting masked areas to NaN.

    Args:
        image (np.ndarray): 2D input image array.
        mask (np.ndarray): Boolean mask array where True indicates valid pixels.
        output_size (int): Target image size after resizing (default is 512x512).

    Returns:
        np.ndarray: Preprocessed image with masked regions as NaN.
    """
    resized_image = resize(
        image, (output_size, output_size), mode='reflect', anti_aliasing=True
    )
    resized_mask = resize(
        mask, (output_size, output_size), mode='reflect', anti_aliasing=False
    ) > 0.5

    masked_image = resized_image.copy()
    masked_image[~resized_mask] = np.nan
    return masked_image


def prepare_data_concatenated(
    masked_images: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate masked images from multiple channels, remove NaNs,
    and apply robust scaling.

    Args:
        masked_images (List[np.ndarray]): List of 2D masked images (one per channel).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - scaled_data: 2D array of shape (valid_pixels, num_channels).
            - valid_pixel_mask: 1D boolean array indicating valid pixels.
            - nan_mask: 1D boolean array indicating pixels with NaN in any channel.
    """
    stacked = np.stack(masked_images, axis=-1)  # Shape: (H, W, C)
    reshaped = stacked.reshape(-1, len(masked_images))  # Shape: (pixels, channels)
    nan_mask = np.isnan(reshaped).any(axis=1)  # Pixels with NaN in any channel
    valid_data = reshaped[~nan_mask]  # Remove invalid pixels

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(valid_data)

    return scaled_data, ~nan_mask, nan_mask
