from typing import List, Tuple
import numpy as np
from sklearn.preprocessing import RobustScaler

def prepare_data_concatenated(masked_data_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data from multiple channels, handles NaNs, and normalizes data.
    
    Args:
        masked_data_list: List of masked data arrays (one per channel).
    
    Returns:
        Tuple containing:
        - valid_data: Normalized data array
        - valid_pixel_mask_1d: Boolean mask of valid pixels
        - nan_mask_1d: Boolean mask of NaN pixels
    """
    if not masked_data_list:
        raise ValueError("Empty masked data list provided")

    # Get shape info from first image
    img_shape = masked_data_list[0].shape
    n_pixels = img_shape[0] * img_shape[1]
    n_channels = len(masked_data_list)
    
    print(f"Preparing data from {n_channels} channels with shape {img_shape}")
    
    # Create a 2D array (pixels x channels)
    all_data_flat = np.zeros((n_pixels, n_channels))
    
    # Flatten each image and put in corresponding column
    for i, img in enumerate(masked_data_list):
        if img.shape != img_shape:
            raise ValueError(f"Image shape mismatch: Expected {img_shape}, got {img.shape}")
        all_data_flat[:, i] = img.flatten()
    
    # Create a mask where any pixel is valid in at least one channel
    valid_pixel_mask_1d = ~np.all(np.isnan(all_data_flat), axis=1)
    
    # Check if there are any valid pixels
    if not np.any(valid_pixel_mask_1d):
        print("Warning: No valid pixels found in any channel. Attempting to recover data...")
        
        # Try a more lenient approach: consider a pixel valid if it's not NaN in ANY channel
        for i, img in enumerate(masked_data_list):
            # Replace NaNs with a fill value just for mask creation
            filled_img = np.nan_to_num(img, nan=0.0)
            if i == 0:
                valid_pixel_mask_2d = filled_img > 0
            else:
                valid_pixel_mask_2d |= filled_img > 0
        
        valid_pixel_mask_1d = valid_pixel_mask_2d.flatten()
        
        # If still no valid pixels, use all pixels
        if not np.any(valid_pixel_mask_1d):
            print("Warning: Still no valid pixels. Using all pixels...")
            valid_pixel_mask_1d = np.ones(n_pixels, dtype=bool)
            valid_pixel_mask_2d = np.ones(img_shape, dtype=bool)
    else:
        # Reshape the 1D mask back to 2D for visualization
        valid_pixel_mask_2d = valid_pixel_mask_1d.reshape(img_shape)
    
    # Extract only valid pixels
    valid_data = all_data_flat[valid_pixel_mask_1d, :]
    
    # Replace any remaining NaNs with channel mean
    for col in range(valid_data.shape[1]):
        col_data = valid_data[:, col]
        col_nans = np.isnan(col_data)
        
        if np.all(col_nans):
            # If all values in the column are NaN, replace with zeros
            valid_data[:, col] = 0.0
        elif np.any(col_nans):
            # If some values are NaN, replace with column mean
            col_mean = np.nanmean(col_data)
            valid_data[col_nans, col] = col_mean
    
    # Normalize each channel to zero mean and unit variance
    for col in range(valid_data.shape[1]):
        mean = np.mean(valid_data[:, col])
        std = np.std(valid_data[:, col])
        
        if std == 0:  # If std is zero, set it to a small value to avoid division by zero
            std = 1e-6
            print(f"Warning: Zero standard deviation for channel {col}. Using small value.")
        
        valid_data[:, col] = (valid_data[:, col] - mean) / std
    
    # Create a NaN mask
    nan_mask_1d = np.isnan(all_data_flat).any(axis=1)
    
    return valid_data, valid_pixel_mask_1d, nan_mask_1d