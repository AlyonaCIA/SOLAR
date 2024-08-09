"""This module is used to download data from the Solar Dynamics Observatory (SDO)
telescope."""

# Routine created by F. Javier Ordonez-Araujo improved by Alyona C. Ivanova-Araujo
# Standard library imports
import os
from datetime import datetime
from typing import Union

# Third-party imports
import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.map
from astropy.coordinates import SkyCoord
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net import jsoc

# Define the date range and format
start_date_str = '2017-09-10T14:30:00.00'
end_date_str = '2017-09-10T17:30:00.00'
date_format = "%Y-%m-%dT%H:%M:%S.%f"

# Define the center coordinates and the square size in arcseconds
center_x, center_y = 0, 0  # arcseconds
half_square_size = 1210  # arcseconds

# Parse the start and end times
start_time_global = datetime.strptime(start_date_str, date_format)
end_time_global = datetime.strptime(end_date_str, date_format)

# Define the top right and bottom left coordinates of the square
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

# Define other parameters
wavelength_channel = 171  # AIA wavelength channel
sample_interval = 12 * u.second  # Cadence to sample data (12 seconds)
contact_email = "Example@astro.uio.no"  # Contact email for data request


def validate_parameters(item: Union[int, str], duration: u.Quantity) -> None:
    """Validates the input parameters for the SDO query.

    Parameters:
        item (Union[int, str]): The data item to search for.
        duration (astropy.units.Quantity): Sampling interval for the data.

    Raises:
        ValueError: If the item or sample duration is not valid.
    """
    aia_cad_12 = [94, 131, 171, 193, 211, 304, 335]
    aia_cad_24 = [1600, 1700]

    valid_items = aia_cad_12 + aia_cad_24 + ['hmi', 'dopplergram']

    if isinstance(item, str):
        if item.lower() not in map(str.lower, valid_items):
            raise ValueError(
                "Supported items are AIA wavelengths, 'hmi', and 'dopplergram' only")

    elif isinstance(item, int):
        if item not in valid_items:
            raise ValueError(
                "Supported items are AIA wavelengths, 'hmi', and 'dopplergram' only")
    else:
        raise ValueError("Item must be an integer or a string")

    if isinstance(item, str) and item.lower() in ['hmi', 'dopplergram'] \
            and duration < 45 * u.s:
        raise ValueError(
            "The selected sample duration is lower than the instrument cadence \
                for HMI or Dopplergram")

    if item in aia_cad_12 and duration < 12 * u.s:
        raise ValueError(
            "The selected sample duration is lower than the instrument cadence \
                for AIA EUV 12s")

    if item in aia_cad_24 and duration < 24 * u.s:
        raise ValueError(
            "The selected sample duration is lower than the instrument cadence \
                for AIA UV 24s")


def construct_query(
        item: Union[int, str],
        bottom_left: SkyCoord,
        top_right: SkyCoord,
        start_time: str,
        end_time: str,
        email: str,
        duration: u.Quantity,
        tracking: bool) -> Fido.search:
    """Constructs a query for the SDO data.

    Parameters:
        item (Union[int, str]): The data item to search for.
        bottom_left (SkyCoord): Coordinates for the bottom left corner of the search
        area.
        top_right (SkyCoord): Coordinates for the top right corner of the search area.
        start_time (str): Start time of the data search period.
        end_time (str): End time of the data search period.
        email (str): Email address for notification.
        duration (astropy.units.Quantity): Sampling interval for the data.
        tracking (bool): Whether to enable tracking.

    Returns:
        Fido.search: Query object for Fido search.
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
            jsoc.Series('aia.lev1_uv_24s'),
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


def get_query_sdo(
        item: Union[int, str],
        bottom_left: SkyCoord,
        top_right: SkyCoord,
        start_time: str,
        end_time: str,
        email: str,
        duration: u.Quantity,
        tracking: bool = False) -> Union[Fido.search, None]:
    """Constructs a query to search for Solar Dynamics Observatory (SDO) data based on
    specified parameters.

    Parameters:
        item (Union[int, str]): The data item to search for (e.g., wavelength in
        Angstrom, 'hmi', 'dopplergram').
        bottom_left (SkyCoord): Coordinates for the bottom left corner
        of the search area.
        top_right (SkyCoord): Coordinates for the top right corner of the search area.
        start_time (str): Start time of the data search period.
        end_time (str): End time of the data search period.
        email (str): Email address for notification.
        duration (astropy.units.Quantity): Sampling interval for the data.
        tracking (bool): Whether to enable tracking (default is False).

    Returns:
        Fido.search: Query object for Fido search or None if parameters are invalid.
    """
    if any(param is None for param in [item, bottom_left, top_right, start_time,
                                       end_time, email, duration]):
        return None

    validate_parameters(item, duration)
    return construct_query(item, bottom_left, top_right, start_time,
                           end_time, email, duration, tracking)


# Example usage
query_result = get_query_sdo(
    wavelength_channel, bottom_left_coord, top_right_coord,
    start_time_global, end_time_global, contact_email, sample_interval,
    tracking=False
)

if query_result:
    file_download = Fido.fetch(query_result, path='src/data_prep/sdo_data/')

    # Find the most recent FITS file in the download directory
    fits_files = [f for f in os.listdir(
        'src/data_prep/sdo_data/') if f.endswith('.fits')]
    if fits_files:
        for fits_file in fits_files:
            solar_map = sunpy.map.Map(
                os.path.join('src/data_prep/sdo_data/', fits_file)
            )

            # Display the image using matplotlib
            plt.figure()
            solar_map.plot()
            plt.colorbar()
            plt.show()

            print(solar_map)
    else:
        print("No FITS files found in the download directory.")
else:
    print("Query result is None. Please check the input parameters.")
