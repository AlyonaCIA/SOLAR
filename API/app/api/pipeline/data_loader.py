import os
import re
from typing import List, Set, Tuple, Union

import numpy as np
import imageio
from skimage.transform import resize


def load_masked_channel_data_jp2(
    data_dir: str,
    image_size: Union[int, Tuple[int, int]],
    excluded_channels: Set[int] = {1600, 1700},
    mask_radius: int = 512 // 2  # default, can be overwritten per image
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    masked_data, channel_names, jp2_paths = [], [], []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".jp2"):
                continue

            match = re.search(r"AIA_(\d+)\.jp2$", file)
            if not match:
                continue

            channel = int(match.group(1))
            if channel in excluded_channels:
                continue

            jp2_path = os.path.join(root, file)
            print(f"Attempting to load JP2 with Imageio: {jp2_path}")
            try:
                data = imageio.v2.imread(jp2_path)
                print(f"  Loaded shape: {data.shape}, dtype: {data.dtype}")

                # --- Ajuste dinámico del radio si la imagen no es 4096x4096 ---
                if data is not None:
                    if data.shape != (4096, 4096) and mask_radius == 1600:
                        scale_factor = min(data.shape) / 4096.0
                        scaled_radius = int(1600 * scale_factor)
                        print(f"Warning: JP2 image shape {data.shape} is not 4096x4096. "
                              f"Adjusting mask radius from {mask_radius} to {scaled_radius}.")
                        mask_radius_to_use = scaled_radius
                    else:
                        mask_radius_to_use = mask_radius

                    mask = create_circular_mask_jp2(data, mask_radius_to_use)
                    masked = preprocess_image(data, mask, image_size)

                    masked_data.append(masked)
                    channel_names.append(str(channel))
                    jp2_paths.append(jp2_path)

            except Exception as e:
                print(f"Error processing {jp2_path}: {e}")

    if not masked_data:
        print("No JP2 data loaded. Exiting.")

    return masked_data, channel_names, jp2_paths


def create_circular_mask_jp2(data: np.ndarray, fixed_radius_pixels: int) -> np.ndarray:
    if data is None:
        raise ValueError("Input data cannot be None for JP2 mask creation.")
    ny, nx = data.shape
    print(f"Creating JP2 mask for image size {ny}x{nx} using fixed radius: {fixed_radius_pixels}")
    x_center, y_center = nx // 2, ny // 2
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    mask = distance_from_center <= fixed_radius_pixels
    print(f"  Generated JP2 mask shape: {mask.shape}, Sum: {np.sum(mask)}")
    return mask


def preprocess_image(
    data: np.ndarray, mask: np.ndarray, size: int = 512
) -> np.ndarray:
    if data is None or mask is None:
        raise ValueError("Data and mask must be provided for preprocessing.")
    print(f"Preprocessing: Resizing data ({data.shape}) and mask ({mask.shape}) to {size}x{size}")
    resized_data = resize(data, (size, size), mode='reflect', anti_aliasing=True)
    resized_mask = resize(mask.astype(float), (size, size), mode='reflect', anti_aliasing=False) > 0.5
    print(f"  Resized mask shape: {resized_mask.shape}, Sum: {np.sum(resized_mask)}")

    masked_data = resized_data.copy()
    masked_data[~resized_mask] = np.nan
    print(f"  Final masked data shape: {masked_data.shape}, Non-NaN count: {np.sum(~np.isnan(masked_data))}")
    return masked_data
