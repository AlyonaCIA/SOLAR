import os
import glob
import re
from typing import List, Set, Tuple, Union, Optional

import numpy as np
import imageio
import astropy.units as u
from skimage.transform import resize

def load_fits_data(channel_dir: str) -> Tuple[np.ndarray, dict]:
    """Load FITS files from a directory and return data and metadata."""
    # Change the pattern to match AIA files from VSO (*.fits instead of *image*.fits)
    fits_pattern = os.path.join(channel_dir, "*.fits")
    fits_files = glob.glob(fits_pattern)
    
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {channel_dir}")
    
    # Sort files to get the most recent one
    fits_files.sort()
    latest_fits = fits_files[-1]
    
    print(f"Loading FITS file: {latest_fits}")
    
    # Use sunpy.map.Map to load the FITS file
    import sunpy.map
    sunpy_map = sunpy.map.Map(latest_fits)
    
    # Extract data and metadata
    data = sunpy_map.data
    metadata = {
        "header": sunpy_map.meta,
        "dimensions": sunpy_map.dimensions,
        "center": sunpy_map.center,
        "radius": sunpy_map.rsun_obs
    }
    
    return data, metadata

def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Create a circular mask for FITS data based on the solar radius."""
    # Check for multi-dimensional data and flatten if needed
    original_shape = data.shape
    if len(original_shape) > 2:
        # For 3D data, select the first non-empty plane
        for i in range(original_shape[0]):
            if np.any(data[i]):
                data_2d = data[i]
                print(f"Selected plane {i} for masking from shape {original_shape}")
                break
        else:
            # If no non-empty plane found, use the first one
            data_2d = data[0]
            print(f"Using first plane for masking from shape {original_shape}")
    else:
        data_2d = data
    
    ny, nx = data_2d.shape
    
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
    
    # Create the mask using pixel coordinates for the 2D plane
    y_indices, x_indices = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
    mask_2d = distance_from_center <= radius_pixels
    
    # If the original data was 3D, create a 3D mask
    if len(original_shape) > 2:
        mask = np.zeros(original_shape, dtype=bool)
        for i in range(original_shape[0]):
            mask[i] = mask_2d
    else:
        mask = mask_2d
    
    print(f"Created mask with shape {mask.shape}, Sum of True: {np.sum(mask)}")
    return mask

def preprocess_image(data: np.ndarray, mask: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """Preprocess a FITS image by applying a mask and optional resizing."""
    # Handle different data shapes
    if len(data.shape) > 2:
        # Find the first non-empty plane in 3D data
        for i in range(data.shape[0]):
            if np.any(data[i]):
                print(f"Selected plane {i} from {data.shape} for preprocessing")
                data_2d = data[i].copy()
                mask_2d = mask[i] if len(mask.shape) > 2 else mask
                break
        else:
            # If no non-empty plane found, use the first one
            print(f"Using first plane from {data.shape} for preprocessing")
            data_2d = data[0].copy()
            mask_2d = mask[0] if len(mask.shape) > 2 else mask
    else:
        data_2d = data.copy()
        mask_2d = mask
    
    # Handle mask dimensions
    if mask_2d.shape != data_2d.shape:
        print(f"Warning: Mask shape {mask_2d.shape} doesn't match data shape {data_2d.shape}")
        # Create a new mask with correct shape
        y, x = np.ogrid[:data_2d.shape[0], :data_2d.shape[1]]
        center_y, center_x = data_2d.shape[0] // 2, data_2d.shape[1] // 2
        radius = min(data_2d.shape) // 2 * 0.95
        mask_2d = ((y - center_y)**2 + (x - center_x)**2) <= radius**2
    
    # Apply mask
    masked_data = data_2d.copy()
    masked_data[~mask_2d] = np.nan
    
    # Check if we have any valid data after masking
    if np.all(np.isnan(masked_data)):
        print("Warning: All pixels are NaN after masking. Using simple threshold mask...")
        # Try using a simple threshold mask instead
        threshold = np.percentile(data_2d, 5)  # Use bottom 5% as background
        simple_mask = data_2d > threshold
        masked_data = data_2d.copy()
        masked_data[~simple_mask] = np.nan
        
        # If still all NaN, try a more aggressive approach
        if np.all(np.isnan(masked_data)):
            print("Warning: Still all NaN. Using full image data...")
            masked_data = data_2d.copy()
    
    # Resize if needed
    if size is not None and (masked_data.shape[0] != size or masked_data.shape[1] != size):
        print(f"Resizing image from {masked_data.shape} to ({size}, {size})")
        
        # Before resizing, fill NaNs with zeros temporarily to avoid resize issues
        has_nans = np.any(np.isnan(masked_data))
        if has_nans:
            nan_mask = np.isnan(masked_data)
            masked_data_filled = np.where(nan_mask, 0, masked_data)
        else:
            masked_data_filled = masked_data
        
        # Resize the image
        resized_data = resize(masked_data_filled, (size, size), order=1, anti_aliasing=True, preserve_range=True)
        
        # If we had NaNs, we need to also resize the NaN mask and reapply it
        if has_nans:
            resized_nan_mask = resize(nan_mask.astype(float), (size, size), order=0) > 0.5
            resized_data = np.where(resized_nan_mask, np.nan, resized_data)
        
        print(f"Resized image: valid pixels: {np.sum(~np.isnan(resized_data))}/{size*size}")
        return resized_data
    
    return masked_data

def create_circular_mask_jp2(data: np.ndarray, fixed_radius_pixels: int) -> np.ndarray:
    """Create a circular mask for JP2 data based on a fixed radius."""
    # Get image dimensions
    ny, nx = data.shape
    
    # Center of the image
    x_center = nx // 2
    y_center = ny // 2
    
    # Create the mask
    y_indices, x_indices = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)
    mask = distance_from_center <= fixed_radius_pixels
    
    print(f"Created JP2 circular mask with radius {fixed_radius_pixels}, shape {mask.shape}, Sum of True: {np.sum(mask)}")
    return mask

def load_masked_channel_data_jp2(
    data_dir: str,
    image_size: Union[int, Tuple[int, int]],
    excluded_channels: Set[int] = {1600, 1700},
    mask_radius: int = 512 // 2  # default, can be overwritten per image
) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load JP2 images from a directory, create masks, and preprocess."""
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
            try:
                data = imageio.v2.imread(jp2_path)
                print(f"\tLoaded shape: {data.shape}, dtype: {data.dtype}")

                # --- Dynamic radius adjustment if image is not 4096x4096 ---
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