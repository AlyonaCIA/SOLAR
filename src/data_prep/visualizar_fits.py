"""Module to Display Data."""
import os
import sunpy.map
import matplotlib.pyplot as plt

# Obtén la ruta del directorio donde está este archivo
script_dir = os.path.dirname(os.path.abspath(__file__))

# path for our file to vizualice.
fits_file_path = os.path.join(
    script_dir, 'aia.lev1_euv_12s.2017-09-10T142959Z.171.image.fits')

# check ig this file exits
if os.path.exists(fits_file_path):
    # Cargar la imagen como un mapa solar de SunPy
    solar_map = sunpy.map.Map(fits_file_path)

    # Visualizar la imagen
    plt.figure()
    solar_map.plot()
    plt.colorbar()
    plt.show()
else:
    print(f"El archivo {fits_file_path} no existe.")
