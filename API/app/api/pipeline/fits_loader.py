import glob
import os

# Import astropy units explicitly
import astropy.units as u
import numpy as np
import sunpy.map
from skimage.transform import resize


def load_fits_data(channel_dir: str) -> tuple[np.ndarray, dict]:
    """Load FITS files from a directory and return data and metadata."""
    fits_pattern = os.path.join(channel_dir, "*.fits")
    fits_files = glob.glob(fits_pattern)

    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {channel_dir}")

    # Sort files to get the most recent one
    fits_files.sort()
    latest_fits = fits_files[-1]

    print(f"Loading FITS file: {latest_fits}")
    sunpy_map = sunpy.map.Map(latest_fits)

    # Extract data and metadata
    data = sunpy_map.data
    metadata = {
        "header": sunpy_map.meta,
        "dimensions": sunpy_map.dimensions,
        "center": sunpy_map.center,
        "radius": sunpy_map.rsun_obs,
    }

    return data, metadata


def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Create a circular mask for FITS data based on the solar radius."""
    ny, nx = data.shape

    # Get image center and radius from metadata
    if "header" in metadata and "CRPIX1" in metadata["header"]:
        x_center = metadata["header"]["CRPIX1"] - 1  # FITS is 1-indexed
        y_center = metadata["header"]["CRPIX2"] - 1
    else:
        x_center, y_center = nx // 2, ny // 2

    # Get solar radius in pixels - ensure correct unit conversion
    if "radius" in metadata and metadata["radius"] is not None:
        # Convert radius from arcsec to pixels
        if isinstance(metadata["radius"], u.Quantity):
            radius_arcsec = metadata["radius"].value
        else:
            radius_arcsec = metadata["radius"]

        if "header" in metadata and "CDELT1" in metadata["header"]:
            cdelt = abs(metadata["header"]["CDELT1"])  # arcsec/pixel
            radius_pixels = radius_arcsec / cdelt
        else:
            # Default to 95% of half the smaller dimension
            radius_pixels = min(nx, ny) * 0.475
    else:
        # Default to 95% of half the smaller dimension
        radius_pixels = min(nx, ny) * 0.475

    print(f"Creating circular mask with center ({x_center}, {y_center}) and radius {radius_pixels} pixels")

    # Create the mask using pixel coordinates
    y_indices, x_indices = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x_indices - x_center) ** 2 + (y_indices - y_center) ** 2)
    mask = distance_from_center <= radius_pixels

    return mask


def preprocess_image(data: np.ndarray, mask: np.ndarray, size: int | None = None) -> np.ndarray:
    """Preprocess a FITS image by applying a mask and optional resizing."""
    # Apply the mask
    masked_data = data.copy()
    masked_data[~mask] = np.nan

    # Resize if needed
    if size is not None and (data.shape[0] != size or data.shape[1] != size):
        # Keep aspect ratio
        if data.shape[0] != data.shape[1]:
            # Pad to square before resize
            max_dim = max(data.shape)
            padded = np.full((max_dim, max_dim), np.nan)
            y_offset = (max_dim - data.shape[0]) // 2
            x_offset = (max_dim - data.shape[1]) // 2
            padded[y_offset : y_offset + data.shape[0], x_offset : x_offset + data.shape[1]] = masked_data
            masked_data = padded

        # Preserve NaN values during resize
        mask_valid = ~np.isnan(masked_data)
        masked_data_valid = np.where(mask_valid, masked_data, 0)

        # Resize valid values
        resized_valid = resize(masked_data_valid, (size, size), order=1, anti_aliasing=True)

        # Resize mask and apply it
        resized_mask = resize(mask_valid.astype(float), (size, size), order=0) > 0.5
        resized_data = np.where(resized_mask, resized_valid, np.nan)

        print(f"Resized image from {data.shape} to {resized_data.shape}")
        return resized_data

    return masked_data
