# Changelog — SOLAR

**Solar Observer Learning Anomaly Recognition**

All notable changes to this project will be documented in this file.

This format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Comprehensive English README with architecture diagrams, full project structure, API reference, and contributor credits.
- Mermaid diagrams: high-level system overview, ML pipeline flow, and API request sequence.
- `LICENSE` file with dual licensing model (Academic Free / Commercial Paid).
- Badges for Python, FastAPI, scikit-learn, SunPy, DVC, and license model.
- CLI argument reference table and programmatic usage examples in README.
- Technology stack summary table in README.

### Changed

- Rewrote README entirely in English with standardized structure.
- Updated CHANGELOG to English; converted meeting notes to English summaries.
- Improved documentation organization for clearer onboarding.

## [0.1.3] - 2025-05

### Added

- FastAPI REST service (`API/`) with background job support for FITS analysis.
- Helioviewer JP2 image retrieval endpoints.
- JSOC/SDO data query and download via Fido/VSO.
- Background job manager with polling-based status tracking.
- API pipeline modules: `executor.py`, `job_manager.py`, `background_job.py`, `data_loader.py`, `fits_loader.py`, `preprocess.py`, `model.py`, `visualization.py`.
- Google Cloud Storage integration with local storage fallback.
- Parallel upload support for analysis results.
- Scheduled processing service (`scheduled_processing/`) — Flask-based daily cron for automated SDO analysis.
- Cloud Build CI/CD pipelines (`cloudbuild.yaml`).
- Docker configurations for API and scheduled processing services.
- Pre-commit hooks for code quality enforcement.
- GitHub Actions workflows for CI.

### Changed

- Migrated from notebook-only experiments to a production-ready pipeline architecture.

## [0.1.0] - 2024-07-02

### Added

- Initial project launch with repository structure and documentation.
- `SolarAnomalyPipeline` class in `src/solar/pipeline.py` — end-to-end pipeline for multi-channel anomaly detection and clustering.
- `run_kmeans_pipeline.py` CLI script with full argument parsing.
- Utility modules: `utils.py` (FITS I/O, circular masking, preprocessing, RobustScaler) and `plotting.py`.
- Data preparation scripts: `download_sdo_data.py`, `visualizar_fits.py`.
- Exploratory notebooks for SDO/AIA EDA, clustering comparisons (DBSCAN, GMM, KMeans, MiniBatchKMeans), and model experiments (Isolation Forest, LOF, Normalizing Flows).
- DVC data versioning with Google Cloud Storage remote backend.
- `setup.py` for editable package installation.
- `pytest.ini` and initial test suite under `test/unit_tests/`.

---

## Meeting Notes

### Meeting 1 — 2024-06-05

**Participants:** Alyona Carolina Ivanova Araujo, Prof. Luis Felipe Giraldo Trujillo, Carlos José Díaz Baso, Juan Camilo Guevara Gomez

#### Topics Discussed

1. **Atypical events** — Infrequent events occurring at specific spatial and temporal locations across a combination of channels. Discussion of model explainability: black-box vs. interpretable approaches, post-hoc interpretability via gradient studies, and using interpretability to motivate focused studies in specific solar regions.

2. **Types of atypical events** — Different classes of anomalies with varying levels of scientific interest.

3. **Literature review** (Prof. Luis Felipe) — Anomaly detection, one-class classification, extreme events / extreme value theory, outlier detection, and self-supervised learning applied to this domain and in general.

4. **Problem clarification** — For this project, atypical events are the desired output (not noise to remove). Scope includes real-time detection.

5. **Data selection strategy** — Start with a small data subset, then scale incrementally.

6. **Computational resources** — Assess available compute: personal hardware, cloud, and other options.

7. **Open source considerations** — Decide on open-source strategy for code and model weights.

---

## Contributors

- **Alyona Carolina Ivanova Araujo** — Principal Investigator
- **Carlos José Díaz Baso** — Supervisor
- **Juan Camilo Guevara Gomez** — Co-Supervisor
- Andres Forero — AI/ML Engineering
- Santiago Calderón López — AI/ML Engineering
- Camilo Matson Hernandez — AI/ML Engineering
- Andres Vega — AI/ML Engineering
