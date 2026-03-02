# test/unit_test/solar/test_solar_pipeline.py # O donde coloques tus tests

import shutil
from pathlib import Path

import pytest

# Asume que tienes FITS de prueba aquí. ¡Necesitas crearlos!
# Estos deben ser archivos FITS válidos que tu utils.load_fits_data pueda leer.
TEST_ASSETS_DIR = Path(__file__).parent / "assets" / "Data"
TEST_FITS_FILES = {
    "aia_94": "test_aia_94.fits",
    "aia_171": "test_aia_171.fits",
}

# Importa la clase a testear
from src.solar.pipeline import SolarAnomalyPipeline

# Importa utils si necesitas verificar llamadas específicas (opcional con mocking)
# import src.solar.utils as solar_utils


# --- Fixture para configurar el entorno de prueba ---
@pytest.fixture
def solar_pipeline_fixture(tmp_path):
    """Sets up a temporary environment for testing SolarAnomalyPipeline.

    - Creates temporary data_dir and output_dir.
    - Copies dummy FITS files into data_dir.
    - Instantiates the pipeline.
    - Yields the pipeline instance and paths.
    - Cleans up implicitly via tmp_path.
    """
    data_dir = tmp_path / "test_data"
    output_dir = tmp_path / "test_output"
    data_dir.mkdir()
    output_dir.mkdir()

    # --- Crear o copiar datos FITS de prueba ---
    # Necesitas tener estos archivos en TEST_ASSETS_DIR
    test_channels_present = []
    for channel_dir_name, fits_filename in TEST_FITS_FILES.items():
        asset_file_path = TEST_ASSETS_DIR / fits_filename
        if not asset_file_path.is_file():
            pytest.skip(
                f"Test FITS file not found: {asset_file_path}. \
                 Skipping integration test."
            )

        channel_subdir = data_dir / channel_dir_name
        channel_subdir.mkdir()
        # Copiar el archivo FITS de prueba al directorio temporal del canal
        shutil.copy(asset_file_path, channel_subdir / fits_filename)
        test_channels_present.append(channel_dir_name.split("_")[1])

    if not test_channels_present:
        pytest.fail("No test FITS files were available or copied.")

    # --- Instanciar el Pipeline ---
    pipeline = SolarAnomalyPipeline(
        data_dir=str(data_dir),  # Pipeline espera strings
        output_dir=str(output_dir),
        channels=test_channels_present,  # Usar los canales para los que tenemos datos
        image_size=64,  # Usar tamaño pequeño para acelerar test
        contamination=0.1,  # Un valor razonable para prueba
        n_clusters=3,
        cluster_method="KMeans",
        random_state=42,
    )

    # yield permite usar el setup y luego continuar con el teardown
    # (implícito con tmp_path)
    yield pipeline, output_dir

    # tmp_path se limpia automáticamente por pytest


# --- Tests ---


def test_pipeline_run_successful_execution(solar_pipeline_fixture):
    """Tests if the pipeline's run() method executes without raising errors and returns
    a 'success' status with expected structure."""
    pipeline, output_dir = solar_pipeline_fixture
    thresholds = [0.0, -0.1]  # Probar algunos umbrales

    # Ejecutar el pipeline
    results = pipeline.run(anomaly_thresholds=thresholds)

    # --- Asserts ---
    assert isinstance(results, dict)
    assert results.get("status") == "success", (
        f"Pipeline failed \
        with message: {results.get('message')}"
    )
    assert "message" in results

    # Verificar estructura para cada umbral procesado
    for threshold in thresholds:
        assert threshold in results
        threshold_data = results[threshold]
        assert isinstance(threshold_data, dict)
        assert threshold_data.get("anomaly_threshold") == threshold
        assert "total_pixels_in_image_grid" in threshold_data
        assert "total_valid_pixels_after_masking" in threshold_data
        assert "anomalous_pixels_count" in threshold_data
        assert "anomaly_percentage_of_total" in threshold_data
        assert "anomaly_percentage_of_valid" in threshold_data
        assert "n_clusters_attempted" in threshold_data
        assert "n_clusters_found" in threshold_data  # Importante verificar que existe
        assert "cluster_method" in threshold_data
        assert "plot_path" in threshold_data
        assert "cluster_stats" in threshold_data
        assert isinstance(threshold_data["cluster_stats"], list)

        # Verificar que el plot fue creado (si se encontraron anomalías)
        plot_path_str = threshold_data.get("plot_path")
        anomalies_found = threshold_data.get("anomalous_pixels_count", 0) > 0
        n_clusters_found = threshold_data.get("n_clusters_found", 0)

        if anomalies_found and n_clusters_found > 0:
            assert plot_path_str is not None, (
                f"Plot path is None for\
                 threshold {threshold} despite anomalies."
            )
            plot_path = Path(plot_path_str)
            assert plot_path.exists(), f"Plot file was not created at {plot_path}"
            assert plot_path.is_file()
            assert plot_path.parent == output_dir  # Asegurar que está en el dir correcto
        elif plot_path_str is not None:
            # Si no hubo anomalías, el path podría ser None o el plot no crearse
            print(
                f"Warning: Plot path {plot_path_str} exists for \
                 threshold {threshold} but no anomalies/clusters were expected/found."
            )


def test_pipeline_run_handles_no_data(tmp_path):
    """Tests pipeline behavior when the data directory is empty or missing channels."""
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    output_dir = tmp_path / "output_nodata"
    output_dir.mkdir()

    pipeline = SolarAnomalyPipeline(
        data_dir=str(empty_data_dir),
        output_dir=str(output_dir),
        channels=["94", "171"],  # Pedir canales que no existen
        image_size=64,
        contamination=0.1,
        n_clusters=3,
        random_state=42,
    )

    # Ejecutar esperando un fallo controlado durante la carga/procesamiento
    results = pipeline.run(anomaly_thresholds=[0.0])

    assert results.get("status") == "error"
    # La excepción exacta podría variar, pero debería indicar fallo de datos
    assert (
        "No data was successfully loaded" in results.get("message", "")
        or "Pipeline failed due to data or \
               configuration issue"
        in results.get("message", "")
    )


def test_pipeline_init_invalid_config():
    """Tests __init__ validation checks."""
    with pytest.raises(ValueError, match="Channel list cannot be empty"):
        SolarAnomalyPipeline(data_dir="dummy", output_dir="dummy", channels=[])

    with pytest.raises(ValueError, match="Invalid cluster_method"):
        SolarAnomalyPipeline(data_dir="dummy", output_dir="dummy", channels=["94"], cluster_method="InvalidMethod")

    with pytest.raises(ValueError, match="Number of clusters must be positive"):
        SolarAnomalyPipeline(data_dir="dummy", output_dir="dummy", channels=["94"], n_clusters=0)


# --- Unit Test Example (Opcional) ---
def test_pipeline_validate_params_passes():
    """Unit test for _validate_params success case (requires temp dir)"""
    # Need actual directories for the check inside _validate_params
    with pytest.MonkeyPatch.context():
        # Mock os.path.isdir to return True to bypass actual dir check if needed
        # mp.setattr(os.path, "isdir", lambda x: True)
        # Or create dummy dirs
        dummy_data = Path("./dummy_data_val")
        dummy_out = Path("./dummy_out_val")
        dummy_data.mkdir(exist_ok=True)
        dummy_out.mkdir(exist_ok=True)
        try:
            SolarAnomalyPipeline(data_dir=str(dummy_data), output_dir=str(dummy_out), channels=["94"])
            # If it reaches here without error, validation (as implemented) passed
            assert True
        finally:
            # Clean up dummy dirs
            if dummy_data.exists():
                shutil.rmtree(dummy_data)
            if dummy_out.exists():
                shutil.rmtree(dummy_out)
