#!/usr/bin/env python
"""
This script downloads 30 minutes of SDO/AIA data for multiple EUV channels
(94, 131, 171, 193, 211, 304, and 335 Å) from a quiet period (e.g., April 1, 2019,
00:00–00:30 UT). The data for each channel is saved in its own subdirectory.
After downloading the data, you can use DVC to track the data without pushing the
FITS files to your GitHub repository.

Usage:
    python download_sdo_data.py

DVC Tracking (do these steps in PowerShell after running the script):

    1. Initialize DVC in your repo (if not done already):
       > dvc init
       > git add .dvc .gitignore
       > git commit -m "Initialize DVC"

    2. Add the data directory to DVC tracking:
       > dvc add src/data_prep/sdo_data/
       > git add src/data_prep/sdo_data.dvc .gitignore
       > git commit -m "Track SDO data with DVC"
       
    3. (Optional) Set up a remote storage and push your data:
       > dvc remote add -d myremote s3://my-bucket/path/to/store/data
       > dvc push
"""

import os
from datetime import datetime
import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net import jsoc

# ----------------------------
# Functions for constructing the SDO query
# ----------------------------

def validate_parameters(item, duration: u.Quantity) -> None:
    """
    Validates the input parameters for the SDO query.
    
    Parameters:
        item (Union[int, str]): Data item (e.g., wavelength in Angstrom, 'hmi', or 'dopplergram').
        duration (astropy.units.Quantity): Sampling interval for the data.
        
    Raises:
        ValueError: If an unsupported item is provided or if the sample duration is too short.
    """
    aia_cad_12 = [94, 131, 171, 193, 211, 304, 335]
    aia_cad_24 = [1600, 1700]
    valid_items = aia_cad_12 + aia_cad_24 + ['hmi', 'dopplergram']
    
    if isinstance(item, str):
        if item.lower() not in map(str.lower, valid_items):
            raise ValueError("Supported items are AIA wavelengths, 'hmi', and 'dopplergram' only")
    elif isinstance(item, int):
        if item not in valid_items:
            raise ValueError("Supported items are AIA wavelengths, 'hmi', and 'dopplergram' only")
    else:
        raise ValueError("Item must be an integer or a string")
    
    if isinstance(item, str) and item.lower() in ['hmi', 'dopplergram'] and duration < 45 * u.s:
        raise ValueError("The selected sample duration is lower than the instrument cadence for HMI or Dopplergram")
    if item in aia_cad_12 and duration < 12 * u.s:
        raise ValueError("The selected sample duration is lower than the instrument cadence for AIA EUV 12s")
    if item in aia_cad_24 and duration < 24 * u.s:
        raise ValueError("The selected sample duration is lower than the instrument cadence for AIA UV 24s")

def construct_query(item, bottom_left: SkyCoord, top_right: SkyCoord,
                    start_time: str, end_time: str, email: str,
                    duration: u.Quantity, tracking: bool):
    """
    Constructs a query for the SDO data.
    
    Parameters:
        item (Union[int, str]): Data item to search for.
        bottom_left (SkyCoord): Bottom left coordinate for the search area.
        top_right (SkyCoord): Top right coordinate for the search area.
        start_time (str): Start time in ISO format.
        end_time (str): End time in ISO format.
        email (str): Contact email.
        duration (astropy.units.Quantity): Sampling interval.
        tracking (bool): Whether to enable tracking.
    
    Returns:
        Query object from Fido.search.
    """
    aia_cad_12 = [94, 131, 171, 193, 211, 304, 335]
    aia_cad_24 = [1600, 1700]
    cutout = jsoc.Cutout(bottom_left, top_right=top_right, tracking=tracking)
    
    if item in aia_cad_12:
        return Fido.search(
            a.Time(start_time, end_time),
            a.Wavelength(item * u.angstrom),
            a.Sample(duration),
            jsoc.Series.aia_lev1_euv_12s,
            jsoc.Notify(email),
            jsoc.Segment.image,
            cutout
        )
    if item in aia_cad_24:
        return Fido.search(
            a.Time(start_time, end_time),
            a.Wavelength(item * u.angstrom),
            a.Sample(duration),
            jsoc.Series.aia_lev1_uv_24s, # Corrected Series name
            jsoc.Notify(email),
            jsoc.Segment.image,
            cutout
        )
    if str(item).lower() == 'hmi':
        return Fido.search(
            a.Time(start_time, end_time),
            a.Sample(duration),
            jsoc.Series('hmi.M_45s'),
            jsoc.Notify(email),
            jsoc.Segment.magnetogram,
            cutout
        )
    if str(item).lower() == 'dopplergram':
        return Fido.search(
            a.Time(start_time, end_time),
            a.Sample(duration),
            jsoc.Series('hmi.v_45s'),
            jsoc.Notify(email),
            jsoc.Segment.dopplergram,
            cutout
        )
    raise ValueError("Unsupported item provided")

def get_query_sdo(item, bottom_left: SkyCoord, top_right: SkyCoord,
                  start_time: str, end_time: str, email: str,
                  duration: u.Quantity, tracking: bool = False):
    """
    Constructs a query to search for SDO data based on specified parameters.
    
    Returns:
        Query object from Fido.search or None if parameters are invalid.
    """
    if any(param is None for param in [item, bottom_left, top_right, start_time, end_time, email, duration]):
        return None
    validate_parameters(item, duration)
    return construct_query(item, bottom_left, top_right, start_time, end_time, email, duration, tracking)

# ----------------------------
# Main Script
# ----------------------------

def main():
    # --- Set the date/time and region parameters ---
    # Choose a quiet date and a 30-minute interval (example: April 1, 2019)
    start_date_str = '2019-04-01T00:00:00.00'
    end_date_str   = '2019-04-01T00:30:00.00'
    date_format = "%Y-%m-%dT%H:%M:%S.%f"
    start_time_global = datetime.strptime(start_date_str, date_format)
    end_time_global   = datetime.strptime(end_date_str, date_format)
    
    # Define the center and square size for the image cutout
    center_x, center_y = 0, 0  # in arcseconds (modify if needed)
    half_square_size = 1210   # in arcseconds
    top_right_coord = SkyCoord(
        (center_x + half_square_size) * u.arcsec,
        (center_y + half_square_size) * u.arcsec,
        obstime=start_time_global, observer="earth", frame="helioprojective"
    )
    bottom_left_coord = SkyCoord(
        (center_x - half_square_size) * u.arcsec,
        (center_y - half_square_size) * u.arcsec,
        obstime=start_time_global, observer="earth", frame="helioprojective"
    )
    
    # --- Set additional parameters ---
    sample_interval = 24 * u.second  #  AIA UV channels (24-second cadence)
    contact_email = "j.c.g.gomez@astro.uio.no"  # Change to your email if needed
    
    # List of AIA EUV channels (in Ångstroms)
    # aia_euv_channels = [131, 171, 193, 211, 304, 335, 1600, 1700]
    aia_euv_channels = [1600, 1700]

    
    # Base directory for saving downloaded data
    download_base_path = './sdo_data/'
    os.makedirs(download_base_path, exist_ok=True)
    
    # --- Loop over channels and download data ---
    for channel in aia_euv_channels:
        print(f"\nDownloading AIA {channel} Å data...")
        if channel in [1600, 1700]:
            sample_interval = 24 * u.second
        else:
            sample_interval = 12 * u.second
        query_result = get_query_sdo(
            channel,
            bottom_left_coord,
            top_right_coord,
            start_date_str,
            end_date_str,
            contact_email,
            sample_interval,
            tracking=False
        )
        
        if query_result:
            # Create a subdirectory for the channel (e.g., src/data_prep/sdo_data/aia_171)
            channel_path = os.path.join(download_base_path, f"aia_{channel}")
            os.makedirs(channel_path, exist_ok=True)
            try:
                files_downloaded = Fido.fetch(query_result, path=os.path.join(channel_path, "{file}"))
                print(f"Downloaded files for channel {channel}:")
                for f in files_downloaded:
                    print(f"  - {f}")
            except Exception as e:
                print(f"Error downloading channel {channel}: {e}")
                continue
            
            # # Optionally, display the first downloaded FITS file using sunpy
            # fits_files = [f for f in os.listdir(channel_path) if f.endswith('.fits')]
            # if fits_files:
            #     first_file = os.path.join(channel_path, fits_files[0])
            #     try:
            #         solar_map = sunpy.map.Map(first_file)
            #         plt.figure()
            #         solar_map.plot()
            #         plt.colorbar()
            #         plt.title(f"AIA {channel} Å")
            #         plt.show()
            #     except Exception as e:
            #         print(f"Error displaying file {first_file}: {e}")
            # else:
            #     print(f"No FITS files found in {channel_path}")
        else:
            print(f"Query result is None for channel {channel}. Please check input parameters.")

if __name__ == "__main__":
    main()