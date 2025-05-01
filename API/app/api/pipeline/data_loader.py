import os
import re
from typing import List, Set, Tuple, Union

import numpy as np
import sunpy.map
from skimage.transform import resize


def load_masked_channel_data(
    data_dir: str,
    image_size: Union[int, Tuple[int, int]],
    excluded_channels: Set[int] = {1600, 1700}
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Detect valid AIA channels from FITS filenames in the given directory, and load
    and preprocess the corresponding image data.

    Parameters:
        data_dir (str): Directory containing the .image.fits files.
        image_size (int or tuple): Target size for resizing the images.
        excluded_channels (Set[int], optional): Set of channel numbers to exclude
        from processing.

    Returns:
        Tuple[List[np.ndarray], List[str], List[str]]:
            - List of masked and resized image arrays (NaNs outside solar disk).
            - List of channel numbers (as strings) corresponding to each image.
            - List of filenames used in the process.
    """
    channel_pattern = re.compile(r"\.(\d+)\.image\.fits$")
    masked_data_list = []
    channel_names = []
    valid_files = []
    channels_set = set()

    for fname in os.listdir(data_dir):
        match = channel_pattern.search(fname)
        if not match:
            continue

        channel = int(match.group(1))
        if channel in excluded_channels:
            continue

        channel_names.append(str(channel))
        channels_set.add(f"aia_{channel}")
        valid_files.append(fname)

        fits_path = os.path.join(data_dir, fname)
        try:
            aia_map = sunpy.map.Map(fits_path)
            data, metadata = aia_map.data, aia_map.meta
            mask = create_circular_mask(data, metadata)
            masked = preprocess_image(data, mask, image_size)
            masked_data_list.append(masked)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if not channels_set:
        print("No valid channels found in the data. Exiting.")
    if not masked_data_list:
        print("No data loaded. Exiting.")

    return masked_data_list, channel_names, valid_files


def create_circular_mask(image: np.ndarray, metadata: dict) -> np.ndarray:
    """Generate a circular mask for the solar disk using FITS metadata.

    Parameters:
        image (np.ndarray): 2D array of image data.
        metadata (dict): FITS header metadata with spatial scale and solar radius.

    Returns:
        np.ndarray: Boolean mask (True = inside solar disk, False = outside).
    """
    height, width = image.shape
    x_center, y_center = width // 2, height // 2
    cdelt1 = metadata.get("cdelt1", 1.0)  # Arcseconds per pixel
    rsun_arcsec = metadata.get("rsun_obs", 960.0)  # Solar radius in arcseconds
    rsun_pixels = int(rsun_arcsec / abs(cdelt1))

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    return distance <= rsun_pixels


def preprocess_image(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: Union[int, Tuple[int, int]] = 512
) -> np.ndarray:
    """Resize an image to the desired shape and apply a mask, replacing masked regions
    with NaNs.

    Parameters:
        image (np.ndarray): 2D input image array.
        mask (np.ndarray): Boolean array where True indicates valid (unmasked) pixels.
        output_size (int or tuple): Output dimensions for the resized image.

    Returns:
        np.ndarray: Preprocessed image array with masked areas set to NaN.
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
