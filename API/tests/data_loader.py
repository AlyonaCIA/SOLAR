import os
from typing import Tuple

import numpy as np
import sunpy.map


def load_fits_data(channel_directory: str) -> Tuple[np.ndarray, dict]:
    """
    Load FITS image data and metadata from a specified directory.

    This function reads the first available FITS file found in the given
    channel directory using SunPy and returns its image array and header metadata.

    Args:
        channel_directory (str): Path to the directory containing FITS files.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing:
            - A 2D NumPy array with the image data.
            - A dictionary with the FITS header metadata.

    Raises:
        FileNotFoundError: If no FITS files are found in the specified directory.
    """
    fits_files = [f for f in os.listdir(channel_directory) if f.endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(
            f"No FITS files were found in the directory: {channel_directory}"
        )

    fits_path = os.path.join(channel_directory, fits_files[0])  # Load the first FITS file
    aia_map = sunpy.map.Map(fits_path)

    return aia_map.data, aia_map.meta
