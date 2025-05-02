from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler


def prepare_data_concatenated(
    masked_images: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate masked images from multiple channels, remove NaNs, and apply robust
    scaling.

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
