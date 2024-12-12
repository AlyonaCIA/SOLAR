"""This module is used to download data from the Solar Dynamics Observatory (SDO).

with all wavelength channels
"""

# Standard library imports
import os
import matplotlib.pyplot as plt

# Third-party imports
import astropy.units as u
from sunpy.net import Fido, attrs as a

# Define the date range for data retrieval
start_date = '2020-09-01T00:00:00'
end_date = '2020-09-01T01:00:00'

# Define the wavelength channels of interest (in Ångstroms)
wavelengths = [
    # 94 * u.angstrom,
    131 * u.angstrom,
    171 * u.angstrom,
    193 * u.angstrom,
    211 * u.angstrom,
    304 * u.angstrom,
    335 * u.angstrom,
    1600 * u.angstrom,
    1700 * u.angstrom,
    4500 * u.angstrom,

]

# Directory to save downloaded data
download_dir = 'Data/multichannel_SDO_AIA/'


def ensure_directory_exists(directory: str):
    """Ensures the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_and_plot_sdo_data(start_time: str,
                               end_time: str,
                               wavelengths: list,
                               download_dir: str):
    """Downloads solar data from SDO for multiple wavelengths using SunPy and plots the
    retrieved data.

    Parameters:
        start_time (str): Start time of the data retrieval period in ISO format.
        end_time (str): End time of the data retrieval period in ISO format.
        wavelengths (list): List of wavelengths (astropy.units.Quantity) to download.
        download_dir (str): Directory to save the downloaded data.
    """
    # Ensure the download directory exists
    ensure_directory_exists(download_dir)

    for wavelength in wavelengths:
        try:
            # Perform the data search for the current wavelength
            print(
                f"Searching for data from {start_time} to {end_time} at {wavelength}.")
            query = Fido.search(
                a.Time(start_time, end_time),
                a.Instrument('AIA'),
                a.Wavelength(wavelength)
            )

            # Download the data
            print(f"Downloading data for wavelength {wavelength}...")
            files = Fido.fetch(query, path=os.path.join(
                download_dir, f'{wavelength.value}Å/{{file}}'))

            if not files:
                print(
                    f"No files downloaded for wavelength {wavelength}.\
                        Please check the query parameters.")
                continue

            # Process and plot the downloaded data
            # for file in files:
            #     print(f"Processing file: {file}")
            #     solar_map = sunpy.map.Map(file)

            #     # Plot the data using matplotlib
            #     plt.figure()
            #     solar_map.plot()
            #     plt.colorbar()
            #     plt.title(f"SDO AIA {wavelength} - {solar_map.date}")
            #     plt.show()

            #     # Print metadata for the map
            #     print(solar_map)

        except Exception as e:
            print(f"An error occurred for wavelength {wavelength}: {e}")


# Main execution
if __name__ == "__main__":
    download_and_plot_sdo_data(start_date, end_date, wavelengths, download_dir)
