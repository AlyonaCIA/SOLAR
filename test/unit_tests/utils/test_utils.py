# test/Unit_Tests/utils/test_utils.py

import numpy as np
import pytest
from astropy.io import fits  # Para crear FITS de prueba
from skimage.transform import resize  # Para comparar resize si es necesario

# Importar las funciones a testear desde src.utils.utils
# CORRECCIÓN IMPORT: Asumiendo que utils.py está directamente en src/utils/
from src.utils import utils

# --- Fixtures para datos de prueba ---


@pytest.fixture(scope="module")
def dummy_fits_file(tmp_path_factory):
    """Crea un archivo FITS de prueba válido pero pequeño."""
    base_temp_dir = tmp_path_factory.getbasetemp()
    fits_dir = base_temp_dir / "fits_test_data"
    fits_dir.mkdir(exist_ok=True)
    fits_path = fits_dir / "test.fits"

    data = np.arange(25, dtype=np.float32).reshape((5, 5))
    header = fits.Header()
    header["SIMPLE"] = True
    header["BITPIX"] = -32
    header["NAXIS"] = 2
    header["NAXIS1"] = 5
    header["NAXIS2"] = 5
    header["CRPIX1"] = 3.0
    header["CRPIX2"] = 3.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["RSUN_OBS"] = 4.5
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["DATE-OBS"] = "2024-01-01T00:00:00.000"
    # Añadir CTYPE para evitar warnings (opcional pero recomendado)
    header["CTYPE1"] = "HPLN-TAN"  # Ejemplo común
    header["CTYPE2"] = "HPLT-TAN"  # Ejemplo común
    header["TELESCOP"] = "TestScope"
    header["INSTRUME"] = "TestCam"

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_path, overwrite=True)
    return fits_dir


@pytest.fixture
def sample_image_data():
    """Genera un array numpy simple para usar como imagen."""
    return np.arange(100, dtype=np.float32).reshape((10, 10))


@pytest.fixture
def sample_metadata():
    """Genera un diccionario de metadata simple."""
    return {
        "CRPIX1": 5.5,
        "CRPIX2": 5.5,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "CUNIT1": "arcsec",
        "CUNIT2": "arcsec",
        "RSUN_OBS": 4.0,
        "NAXIS1": 10,
        "NAXIS2": 10,
    }


@pytest.fixture
def sample_mask(sample_image_data, sample_metadata):
    """Genera una máscara simple basada en sample_image_data y sample_metadata."""
    ny, nx = sample_image_data.shape
    x_center = sample_metadata["CRPIX1"] - 1.0
    y_center = sample_metadata["CRPIX2"] - 1.0
    cdelt1 = sample_metadata.get("CDELT1", 1.0)
    if cdelt1 == 0:
        cdelt1 = 1.0
    radius = sample_metadata["RSUN_OBS"] / cdelt1
    y, x = np.ogrid[:ny, :nx]
    dist_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    mask = dist_from_center <= radius
    return mask


# --- Tests para load_fits_data ---


def test_load_fits_data_success(dummy_fits_file):
    """Verifica que se cargue un archivo FITS válido."""
    fits_dir = dummy_fits_file
    data, meta = utils.load_fits_data(str(fits_dir))
    assert isinstance(data, np.ndarray)
    assert data.shape == (5, 5)
    assert data.dtype == np.float32
    assert isinstance(meta, dict)
    assert "crpix1" in meta
    assert meta.get("telescop") == "TestScope"


def test_load_fits_data_no_files(tmp_path):
    """Verifica que lance FileNotFoundError si no hay FITS."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No FITS files found"):
        utils.load_fits_data(str(empty_dir))


def test_load_fits_data_directory_not_found(tmp_path):
    """Verifica el comportamiento si el directorio no existe."""
    non_existent_dir = tmp_path / "i_dont_exist"
    with pytest.raises(FileNotFoundError):
        utils.load_fits_data(str(non_existent_dir))


# --- Tests para create_circular_mask ---


def test_create_circular_mask_valid(sample_image_data, sample_metadata):
    """Verifica la creación de máscara con metadata válida."""
    mask = utils.create_circular_mask(sample_image_data, sample_metadata)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == sample_image_data.shape
    center_x_idx, center_y_idx = 4, 4
    assert mask[center_y_idx, center_x_idx]
    assert not mask[0, 0]
    assert not mask[9, 9]


def test_create_circular_mask_fallback(sample_image_data):
    """Verifica la creación de máscara cuando falta metadata clave.

    Utiliza los valores de fallback de la función original (radio 960).
    """
    bad_metadata = {"NAXIS1": 10, "NAXIS2": 10}  # No contiene 'cdelt1' ni 'rsun_obs'
    mask = utils.create_circular_mask(sample_image_data, bad_metadata)

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == sample_image_data.shape

    # El fallback usa: centro = (5, 5), radio_pixels = 960
    center_x_fallback, center_y_fallback = 5, 5
    assert mask[center_y_fallback, center_x_fallback]  # Centro (dist 0) <= 960

    # --- CORRECCIÓN DE LA ASERCIÓN ---
    # Verificar esquina (0,0). Distancia a (5,5) es sqrt(50) ~ 7.07. Radio es 960.
    # 7.07 <= 960 es VERDADERO. La esquina DEBE estar DENTRO de la máscara.
    assert mask[0, 0]  # <-- CORREGIDO: Esperamos True
    # --- FIN DE CORRECCIÓN ---

    # Verificación adicional: toda la máscara 10x10 debería ser True con radio 960
    assert mask.all()


# --- Tests para preprocess_image ---


def test_preprocess_image_no_resize(sample_image_data, sample_mask):
    """Verifica preprocesamiento sin redimensionado."""
    processed = utils.preprocess_image(sample_image_data, sample_mask, target_size=None)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == sample_image_data.shape
    assert np.isnan(processed[~sample_mask]).all()
    assert not np.isnan(processed[sample_mask]).any()
    np.testing.assert_array_equal(processed[sample_mask], sample_image_data[sample_mask])


def test_preprocess_image_with_resize(sample_image_data, sample_mask):
    """Verifica preprocesamiento con redimensionado."""
    target_size = 5
    processed = utils.preprocess_image(sample_image_data, sample_mask, target_size=target_size)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (target_size, target_size)
    resized_mask_manual = resize(sample_mask, (target_size, target_size), order=0, anti_aliasing=False) > 0.5
    assert np.isnan(processed[~resized_mask_manual]).all()
    assert not np.isnan(processed[resized_mask_manual]).any()


def test_preprocess_image_shape_mismatch(sample_image_data):
    """Verifica que lance ValueError si las formas no coinciden."""
    wrong_shape_mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="Data shape .* and mask shape .* mismatch"):
        utils.preprocess_image(sample_image_data, wrong_shape_mask, target_size=None)


# --- Tests para prepare_data_concatenated ---


@pytest.fixture
def sample_masked_list_data():
    """Crea una lista de arrays 2D (simulando canales) con algunos NaNs."""
    img1 = np.array([[1, 2, np.nan], [4, 5, 6], [7, np.nan, 9]], dtype=np.float32)
    img2 = np.array([[10, 11, 12], [13, np.nan, 15], [16, 17, 18]], dtype=np.float32)
    return [img1, img2]


def test_prepare_data_concatenated_success(sample_masked_list_data):
    """Verifica la preparación y escalado correctos."""
    masked_list = sample_masked_list_data
    img_shape = masked_list[0].shape
    total_pixels = img_shape[0] * img_shape[1]
    scaled_data, valid_mask_flat, nan_mask_flat, image_shape = utils.prepare_data_concatenated(masked_list)

    expected_valid_pixels = 6
    assert image_shape == img_shape
    assert valid_mask_flat.shape == (total_pixels,)
    assert valid_mask_flat.dtype == bool
    assert nan_mask_flat.shape == (total_pixels,)
    assert nan_mask_flat.dtype == bool
    assert np.sum(valid_mask_flat) == expected_valid_pixels
    assert np.sum(nan_mask_flat) == total_pixels - expected_valid_pixels
    expected_nan_mask = np.array([False, False, True, False, True, False, False, True, False])
    assert np.array_equal(nan_mask_flat, expected_nan_mask)
    assert np.array_equal(valid_mask_flat, ~expected_nan_mask)
    assert scaled_data.shape == (expected_valid_pixels, len(masked_list))
    assert not np.isnan(scaled_data).any()


def test_prepare_data_concatenated_empty_list():
    """Verifica que lance ValueError con lista vacía."""
    with pytest.raises(ValueError, match="Masked data list is empty"):
        utils.prepare_data_concatenated([])


def test_prepare_data_concatenated_shape_mismatch():
    """Verifica que lance ValueError si las formas en la lista no coinciden."""
    img1 = np.zeros((3, 3))
    img2 = np.zeros((4, 4))
    with pytest.raises(ValueError, match="Image shape inconsistency"):
        utils.prepare_data_concatenated([img1, img2])


def test_prepare_data_concatenated_no_valid_pixels():
    """Verifica ValueError si no quedan píxeles válidos."""
    img1 = np.array([[1, np.nan], [np.nan, 4]], dtype=np.float32)
    img2 = np.array([[np.nan, 2], [3, np.nan]], dtype=np.float32)
    with pytest.raises(ValueError, match="No valid pixels found after handling NaNs"):
        utils.prepare_data_concatenated([img1, img2])
