from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler



# --- Data Preparation --- (No changes needed here)
def prepare_data_concatenated(
    masked_data_list: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data, handles NaNs, and scales the data."""
    stacked_data = np.stack(masked_data_list, axis=-1)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    nan_mask = np.isnan(reshaped_data).any(axis=1)
    # Ensure we don't try to scale if all pixels are NaN after masking/concatenation
    if np.all(nan_mask):
        print("Warning: All pixels are NaN after concatenation. Cannot scale.")
        return np.array([]), np.array([]), nan_mask # Return empty arrays and the mask

    cleaned_data = reshaped_data[~nan_mask]
    # Handle case where cleaned_data might be empty after removing NaNs
    if cleaned_data.shape[0] == 0:
         print("Warning: No valid (non-NaN) pixels left after masking.")
         return np.array([]), np.array([]), nan_mask # Return empty arrays and the mask

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask # Return scaled data, valid pixel mask, and NaN mask

