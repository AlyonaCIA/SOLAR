# src/solar/pipeline.py

import argparse  # Needed if the if __name__ == "__main__" block remains
import logging
import os
import sys  # Potentially needed if modifying path, but prefer install
from pathlib import Path  # Potentially needed if modifying path
from typing import Any

import matplotlib.colors  # Need explicit import
import matplotlib.patches as mpatches  # Need explicit import
import matplotlib.pyplot as plt  # Need explicit import
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest

# --- CORRECCIÓN IMPORT ---
# Importar funciones específicas directamente desde su ubicación absoluta
# (asumiendo que 'src' está en PYTHONPATH o el proyecto está instalado)
try:
    from src.utils.utils import (
        create_circular_mask,
        load_fits_data,
        prepare_data_concatenated,
        preprocess_image,
    )
    # Si también usas funciones de plotting.py:
    # from src.utils.plotting import plot_results # O el nombre correcto de la función
except ImportError:
    # Intenta añadir el directorio padre de src a sys.path como fallback
    # (esto es menos ideal que instalar el paquete)
    SRC_DIR = Path(__file__).resolve().parent.parent  # Va de solar -> src
    PROJECT_ROOT = SRC_DIR.parent  # Va de src -> SOLAR
    if str(PROJECT_ROOT) not in sys.path:
        print(f"[WARN] Adding project root to sys.path for import: {PROJECT_ROOT}")
        sys.path.insert(0, str(PROJECT_ROOT))
        # Reintentar la importación
        from src.utils.utils import (
            create_circular_mask,
            load_fits_data,
            prepare_data_concatenated,
            preprocess_image,
        )
# --- FIN CORRECCIÓN IMPORT ---


# Configure logging for the module
log = logging.getLogger(__name__)


class SolarAnomalyPipeline:
    """Encapsulates the entire pipeline for detecting and clustering anomalies in
    SDO/AIA multi-channel solar imagery."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        channels: list[str],
        image_size: int | None = 512,
        contamination: float = 0.05,
        n_clusters: int = 7,
        cluster_method: str = "KMeans",
        random_state: int = 42,
    ):
        """Initializes the pipeline with configuration parameters."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.channels = channels
        self.image_size = image_size
        self.contamination = contamination
        self.n_clusters = n_clusters
        # Asegurar que el método sea minúsculas para comparación
        self.cluster_method = cluster_method.lower()
        self.random_state = random_state
        self.log = logging.getLogger(self.__class__.__name__)

        os.makedirs(self.output_dir, exist_ok=True)

        # Internal state variables
        self._masked_data_list: list[np.ndarray] = []
        self._loaded_channel_names: list[str] = []
        self._prepared_data: np.ndarray | None = None
        self._valid_pixel_mask_flat: np.ndarray | None = None
        self._nan_mask_flat: np.ndarray | None = None
        self._image_shape: tuple[int, int] | None = None
        self._total_pixels_final_shape: int = 0
        self._total_valid_pixels: int = 0
        self._anomaly_scores_valid: np.ndarray | None = None

        self._validate_params()

    def _validate_params(self):
        """Basic validation of initialization parameters."""
        if not os.path.isdir(self.data_dir):
            # Es mejor lanzar un error si el directorio no existe
            # en lugar de solo un warning.
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.channels:
            raise ValueError("Channel list cannot be empty.")
        if self.cluster_method not in ["kmeans", "minibatchkmeans"]:
            raise ValueError(f"Invalid cluster_method: {self.cluster_method}. Choose 'KMeans' or 'MiniBatchKMeans'.")
        if self.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive.")
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError("Image size must be positive if specified.")
        self.log.info("Pipeline initialized with valid parameters.")

    # --------------------------------------------------------------------------
    # Private Step Methods
    # --------------------------------------------------------------------------

    def _load_and_preprocess_all(self) -> None:
        """Loads and preprocesses data for all specified channels."""
        self.log.info(f"--- Starting Data Loading/Preprocessing for: {self.channels} ---")
        self._masked_data_list = []
        self._loaded_channel_names = []
        self._image_shape = None
        self._total_pixels_final_shape = 0

        channels_to_process_dirs = [f"aia_{c}" for c in self.channels]

        for channel_dir_name in channels_to_process_dirs:
            channel = channel_dir_name.split("_")[1]
            channel_path = os.path.join(self.data_dir, channel_dir_name)
            self.log.debug(f"Attempting to process directory: {channel_path}")

            if not os.path.isdir(channel_path):
                self.log.warning(f"Channel dir not found: {channel_path}. Skipping {channel}.")
                continue

            try:
                # --- CORRECCIÓN LLAMADA: Usar funciones importadas directamente ---
                data, metadata = load_fits_data(channel_path)
                mask = create_circular_mask(data, metadata)
                preprocessed_img = preprocess_image(data, mask, self.image_size)
                # --- FIN CORRECCIÓN LLAMADA ---

                self._masked_data_list.append(preprocessed_img)
                self._loaded_channel_names.append(channel)

                if self._image_shape is None:
                    self._image_shape = preprocessed_img.shape
                    self._total_pixels_final_shape = self._image_shape[0] * self._image_shape[1]
                    self.log.info(
                        f"First image processed. Final shape: "
                        f"{self._image_shape}, Total pixels: "
                        f"{self._total_pixels_final_shape}"
                    )
                elif preprocessed_img.shape != self._image_shape:
                    raise ValueError(
                        f"Inconsistent shapes. Expected {self._image_shape}, "
                        f"got {preprocessed_img.shape} for channel {channel}."
                    )

            except FileNotFoundError as e:
                self.log.error(f"FITS file/dir missing in {channel_path}: {e}")
                continue
            except ValueError as e:
                self.log.error(f"Data error processing channel {channel}: {e}")
                raise
            except Exception as e:
                self.log.error(f"Unexpected error processing {channel}: {e}", exc_info=True)
                continue

        if not self._masked_data_list:
            raise ValueError("No data successfully loaded/processed for any channel.")

        self.log.info(f"Loaded {len(self._masked_data_list)} channels: {self._loaded_channel_names}.")

        try:
            # --- CORRECCIÓN LLAMADA: Usar función importada directamente ---
            (self._prepared_data, self._valid_pixel_mask_flat, self._nan_mask_flat, _) = prepare_data_concatenated(
                self._masked_data_list
            )
            # --- FIN CORRECCIÓN LLAMADA ---
            self._total_valid_pixels = int(np.sum(self._valid_pixel_mask_flat))
            self.log.info(f"Data prepared. Valid pixels for modeling: {self._total_valid_pixels}")

        except ValueError as e:
            self.log.critical(f"Error during final data prep: {e}")
            raise

        self.log.info("--- Data Loading/Preprocessing Complete ---")

    def _detect_anomalies(self) -> None:
        """Runs Isolation Forest on the prepared data."""
        if self._prepared_data is None:
            raise RuntimeError("Prepared data missing for anomaly detection.")

        self.log.info(
            f"Running Isolation Forest ({self._prepared_data.shape[0]} valid pixels, "
            f"cont={self.contamination}, state={self.random_state})"
        )
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso_forest.fit(self._prepared_data)
        self._anomaly_scores_valid = iso_forest.decision_function(self._prepared_data)
        self.log.info("Isolation Forest scoring complete.")

    def _cluster_anomalies(self, anomaly_features: np.ndarray) -> tuple[np.ndarray, float]:  # Inertia is float
        """Performs clustering on the features of anomalous pixels."""
        if anomaly_features.ndim != 2 or anomaly_features.shape[0] == 0:
            self.log.warning("No anomaly features for clustering.")
            return np.array([], dtype=int), 0.0  # Return float inertia

        n_anomalies = anomaly_features.shape[0]
        # Adjust n_clusters if fewer anomalies than requested clusters
        actual_n_clusters = min(self.n_clusters, n_anomalies)
        if actual_n_clusters < self.n_clusters:
            self.log.warning(
                f"Reducing n_clusters from {self.n_clusters} to "
                f"{actual_n_clusters} due to only {n_anomalies} anomalies."
            )
        elif actual_n_clusters <= 1:
            self.log.warning(
                f"Only {n_anomalies} anomaly/anomalies found. Assigning all to cluster 0. Clustering skipped."
            )
            # Assign all to cluster 0, return 0 inertia
            return np.zeros(n_anomalies, dtype=int), 0.0

        self.log.info(
            f"Running {self.cluster_method} ({n_anomalies} features, k={actual_n_clusters}, state={self.random_state})"
        )

        try:
            if self.cluster_method == "kmeans":
                model = KMeans(n_clusters=actual_n_clusters, random_state=self.random_state, n_init="auto")
            elif self.cluster_method == "minibatchkmeans":
                batch_size = min(1024, n_anomalies)
                model = MiniBatchKMeans(
                    n_clusters=actual_n_clusters,
                    random_state=self.random_state,
                    batch_size=batch_size,
                    n_init="auto",
                )
            else:
                raise ValueError(f"Unsupported cluster_method: {self.cluster_method}")

            model.fit(anomaly_features)
            self.log.info("Clustering complete.")
            inertia = getattr(model, "inertia_", 0.0)  # Use getattr for safety
            return model.labels_.astype(int), float(inertia)

        except Exception as e:
            self.log.error(f"Error during clustering: {e}", exc_info=True)
            raise

    def _create_cluster_mask_2d(
        self,
        anomaly_mask_valid_flat: np.ndarray,  # Mask for valid pixels
        cluster_labels: np.ndarray,  # Labels for anomalies within valid set
    ) -> tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
        """Creates the 2D cluster mask, colormap, and legend patches."""
        self.log.debug("Creating 2D cluster mask...")
        if self._image_shape is None or self._valid_pixel_mask_flat is None:
            raise RuntimeError("Image shape or valid pixel mask not available.")

        n_total_pixels_grid = self._image_shape[0] * self._image_shape[1]
        cluster_mask_flat_full_grid = np.zeros(n_total_pixels_grid, dtype=int)

        original_valid_flat_indices = np.where(self._valid_pixel_mask_flat)[0]

        if cluster_labels.size == 0 or anomaly_mask_valid_flat.size == 0:
            self.log.warning("No anomaly labels/mask, returning empty cluster mask.")
            return cluster_mask_flat_full_grid.reshape(self._image_shape), matplotlib.colors.ListedColormap([]), [], 0

        if anomaly_mask_valid_flat.shape[0] != self._total_valid_pixels:
            raise ValueError(
                f"Internal mismatch: anomaly_mask_valid_flat len "
                f"({anomaly_mask_valid_flat.shape[0]}) != "
                f"total_valid_pixels ({self._total_valid_pixels})."
            )

        anomalous_indices_in_valid_set = np.where(anomaly_mask_valid_flat)[0]
        n_anomalies_found = len(anomalous_indices_in_valid_set)

        if cluster_labels.size != n_anomalies_found:
            # Check this condition carefully, especially if clustering was skipped
            if n_anomalies_found > 0 and cluster_labels.size == 0:
                # This might happen if clustering failed silently or was skipped unexpectedly
                self.log.error(f"Mismatch: {n_anomalies_found} anomalies found, but cluster_labels array is empty.")
                # Handle gracefully: return empty mask or raise error?
                # For now, let's return empty as if no clusters were labelled.
                return (
                    cluster_mask_flat_full_grid.reshape(self._image_shape),
                    matplotlib.colors.ListedColormap([]),
                    [],
                    0,
                )
            else:
                # The length mismatch is unexpected otherwise
                raise ValueError(
                    f"Internal mismatch: Cluster labels len ({cluster_labels.size}) != "
                    f"anomalous valid pixels count ({n_anomalies_found})."
                )

        original_anomalous_flat_indices = original_valid_flat_indices[anomalous_indices_in_valid_set]

        # Assign labels (1-based) to the full grid
        # Ensure cluster_labels has content before indexing
        if cluster_labels.size > 0:
            cluster_mask_flat_full_grid[original_anomalous_flat_indices] = cluster_labels + 1

        cluster_mask_2d = cluster_mask_flat_full_grid.reshape(self._image_shape)

        # Create colormap and patches
        # Use np.unique on the assigned labels (which are 1-based or 0)
        unique_mask_values = np.unique(cluster_mask_flat_full_grid)
        # Filter out background (0) and get unique cluster IDs (1, 2, ...)
        unique_cluster_ids = sorted([val for val in unique_mask_values if val > 0])
        n_clusters_found = len(unique_cluster_ids)

        if n_clusters_found == 0:
            self.log.warning("No unique cluster IDs found in the final mask.")
            return cluster_mask_2d, matplotlib.colors.ListedColormap([]), [], 0

        self.log.info(f"Creating colormap/patches for {n_clusters_found} clusters.")
        cluster_colors = [  # Default colors
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
        if n_clusters_found > len(cluster_colors):
            self.log.warning(
                f"Clusters ({n_clusters_found}) exceed defined colors ({len(cluster_colors)}). Using 'tab10'."
            )
            # Create a colormap suitable for the number found
            # Need N distinct colors for N clusters.
            cmap_tab10 = plt.cm.get_cmap("tab10", n_clusters_found)
            # Generate colors for the number of clusters found
            actual_colors = [cmap_tab10(i) for i in range(n_clusters_found)]
            cluster_cmap = matplotlib.colors.ListedColormap(actual_colors)
            cluster_patches = [
                mpatches.Patch(color=actual_colors[i], label=f"Cluster {cluster_id}")
                for i, cluster_id in enumerate(unique_cluster_ids)  # Use the 1-based IDs
            ]
        else:
            # Use the first n_clusters_found colors
            actual_colors = cluster_colors[:n_clusters_found]
            cluster_cmap = matplotlib.colors.ListedColormap(actual_colors)
            cluster_patches = [
                mpatches.Patch(color=actual_colors[i], label=f"Cluster {cluster_id}")
                for i, cluster_id in enumerate(unique_cluster_ids)  # Use the 1-based IDs
            ]

        self.log.debug("Cluster mask creation complete.")
        return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters_found

    def _plot_results(
        self,
        anomaly_threshold: float,
        cluster_mask_2d: np.ndarray,
        cluster_cmap: matplotlib.colors.ListedColormap,
        cluster_patches: list,
        n_clusters_found: int,
        anomalous_pixels_count: int,
        cluster_pixels_counts: dict[int, int],  # Use dict: {cluster_id: count}
        cluster_anomaly_percentages: dict[int, float],  # Use dict: {cluster_id: percent}
    ):
        """Generates and saves the result plot for a given threshold."""
        if not self._masked_data_list or not self._loaded_channel_names:
            self.log.warning("No masked data/channels available for plotting.")
            return None  # Return None if plotting fails

        self.log.info(f"Plotting results for threshold {anomaly_threshold:.2f}...")
        num_channels = len(self._loaded_channel_names)
        num_cols = 3
        num_rows = max(1, int(np.ceil(num_channels / num_cols)))  # Ensure at least 1 row

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4.5), dpi=120, squeeze=False)
        axes = axes.flatten()

        # Figure Title
        anomaly_perc = (
            anomalous_pixels_count / self._total_pixels_final_shape * 100 if self._total_pixels_final_shape > 0 else 0
        )
        suptitle = (
            f"{self.cluster_method.capitalize()} Clusters (SDO/AIA EUV)\n"
            f"Thresh: {anomaly_threshold:.2f} | Anomalies: "
            f"{anomalous_pixels_count}/{self._total_pixels_final_shape} ({anomaly_perc:.2f}%)"
        )
        fig.suptitle(suptitle, fontsize=16, y=0.99)

        # Global Vmin/Vmax for consistent scaling
        all_valid = (
            np.concatenate([d[~np.isnan(d)].flatten() for d in self._masked_data_list if np.any(~np.isnan(d))])
            if self._masked_data_list
            else np.array([])
        )

        if all_valid.size > 0:
            vmin_global = np.percentile(all_valid, 2)
            vmax_global = np.percentile(all_valid, 98)
            if vmin_global >= vmax_global:  # Handle flat data cases
                vmin_global = np.nanmin(all_valid) - (1 if np.nanmin(all_valid) == np.nanmax(all_valid) else 0)
                vmax_global = np.nanmax(all_valid) + (1 if np.nanmin(all_valid) == np.nanmax(all_valid) else 0)
                if vmin_global >= vmax_global:  # Final fallback
                    vmin_global, vmax_global = 0, 1
        else:
            vmin_global, vmax_global = 0, 1  # Default if no valid data
        self.log.debug(f"Global vmin/vmax for plotting: {vmin_global:.2f}/{vmax_global:.2f}")

        # Plot Each Channel
        for i, (masked_data, channel) in enumerate(zip(self._masked_data_list, self._loaded_channel_names)):
            if i >= len(axes):
                break  # Safety break
            ax = axes[i]

            # Base image
            ax.imshow(masked_data, cmap="YlOrBr", origin="lower", vmin=vmin_global, vmax=vmax_global, alpha=0.5)

            # Overlay clusters if they exist and shapes match
            if n_clusters_found > 0 and cluster_mask_2d.shape == masked_data.shape:
                cluster_overlay = np.ma.masked_where(cluster_mask_2d == 0, cluster_mask_2d)
                # Ensure vmin/vmax covers the actual cluster IDs present (1 to N)
                cluster_ids_present = np.unique(cluster_overlay.compressed())  # Get non-masked values
                vmin_overlay = np.min(cluster_ids_present) if cluster_ids_present.size > 0 else 1
                vmax_overlay = np.max(cluster_ids_present) if cluster_ids_present.size > 0 else n_clusters_found

                ax.imshow(
                    cluster_overlay,
                    cmap=cluster_cmap,
                    alpha=0.6,
                    origin="lower",
                    interpolation="nearest",
                    vmin=vmin_overlay,
                    vmax=vmax_overlay,
                )
            elif n_clusters_found > 0:
                log.error(f"Shape mismatch: Mask {cluster_mask_2d.shape}, Img {masked_data.shape}")

            # Subplot Title with Stats
            title_lines = [f"AIA {channel} Å"]
            # Use the dicts for stats, ensuring cluster_id matches patch label
            for patch in cluster_patches:  # Patches are ordered by cluster_id
                try:
                    cluster_id = int(patch.get_label().split(" ")[1])  # 1-based ID
                    pixels = cluster_pixels_counts.get(cluster_id, 0)
                    percentage = cluster_anomaly_percentages.get(cluster_id, 0.0)
                    if pixels > 0 and len(title_lines) < 5:  # Limit lines
                        title_lines.append(f" C{cluster_id}: {pixels}px ({percentage:.1f}%)")
                    elif pixels > 0 and len(title_lines) == 5:
                        title_lines.append(" ...")
                        break
                except (IndexError, ValueError) as e:
                    log.debug(f"Could not parse patch label {patch.get_label()}: {e}")

            ax.set_title("\n".join(title_lines), fontsize=10, pad=5)
            ax.axis("off")

        # Turn off unused axes
        for j in range(num_channels, len(axes)):
            axes[j].axis("off")

        # Add legend if there are patches
        if cluster_patches:
            fig.legend(
                handles=cluster_patches,
                loc="upper right",
                bbox_to_anchor=(0.98, 0.95),
                fontsize="small",
                frameon=True,
                framealpha=0.9,
                title="Anomaly Clusters",
            )

        plt.tight_layout(rect=[0, 0, 0.90, 0.95])  # Adjust for legend

        # Save figure
        filename = os.path.join(
            self.output_dir, f"{self.cluster_method}_anomalies_thresh_{anomaly_threshold:.2f}_k{self.n_clusters}.png"
        )
        try:
            plt.savefig(filename, bbox_inches="tight", dpi=150)
            plt.close(fig)  # Close figure to free memory
            self.log.info(f"Result plot saved to: {filename}")
            return filename
        except Exception as e:
            plt.close(fig)  # Ensure figure is closed even on error
            self.log.error(f"Error saving figure {filename}: {e}", exc_info=True)
            return None

    # --------------------------------------------------------------------------
    # Public Execution Method
    # --------------------------------------------------------------------------

    def run(self, anomaly_thresholds: list[float]) -> dict[str, Any]:
        """Executes the full anomaly detection pipeline."""
        results = {}
        self.log.info(f"--- Starting pipeline execution for thresholds: {anomaly_thresholds} ---")

        try:
            # 1. Load and Preprocess Data
            self._load_and_preprocess_all()
            if self._image_shape is None:
                raise RuntimeError("Image shape not determined during preprocessing.")
            self.log.info(f"Data processed. Shape: {self._image_shape}, Valid px: {self._total_valid_pixels}")

            # 2. Anomaly Detection (once for all thresholds)
            self._detect_anomalies()
            if self._anomaly_scores_valid is None or self._prepared_data is None:
                raise RuntimeError("Anomaly scores/prepared data missing.")

            # 3. Process each threshold
            for threshold in sorted(anomaly_thresholds):  # Process in order
                self.log.info(f"--- Processing Threshold: {threshold:.2f} ---")

                anomaly_mask_valid = self._anomaly_scores_valid < threshold
                anomaly_count = int(np.sum(anomaly_mask_valid))

                results[threshold] = {
                    "anomaly_threshold": threshold,
                    "total_pixels_in_image_grid": self._total_pixels_final_shape,
                    "total_valid_pixels_after_masking": self._total_valid_pixels,
                    "anomalous_pixels_count": anomaly_count,
                    "anomaly_percentage_of_total": (
                        anomaly_count / self._total_pixels_final_shape * 100
                        if self._total_pixels_final_shape > 0
                        else 0
                    ),
                    "anomaly_percentage_of_valid": (
                        anomaly_count / self._total_valid_pixels * 100 if self._total_valid_pixels > 0 else 0
                    ),
                    "n_clusters_attempted": self.n_clusters,
                    "n_clusters_found": 0,  # Default
                    "cluster_method": self.cluster_method,
                    "clustering_inertia": None,  # Default
                    "plot_path": None,
                    "cluster_stats": [],
                }

                if anomaly_count == 0:
                    self.log.warning(f"No anomalies found for threshold {threshold:.2f}.")
                    continue  # Skip clustering/plotting

                self.log.info(f"Found {anomaly_count} anomalies for threshold {threshold:.2f}.")

                # Extract features for anomalies
                anomaly_features = self._prepared_data[anomaly_mask_valid]

                # Cluster (handles case with < n_clusters anomalies)
                cluster_labels, inertia = self._cluster_anomalies(anomaly_features)
                results[threshold]["clustering_inertia"] = inertia

                # Create 2D mask and get info for plotting
                cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters_found = self._create_cluster_mask_2d(
                    anomaly_mask_valid, cluster_labels
                )
                results[threshold]["n_clusters_found"] = n_clusters_found

                # Calculate cluster stats
                cluster_pixels_counts = {}
                cluster_anomaly_percentages = {}
                if n_clusters_found > 0:
                    # Use unique cluster IDs (1-based) found in the mask
                    cluster_ids_in_mask = np.unique(cluster_mask_2d)
                    for cluster_id in cluster_ids_in_mask:
                        if cluster_id == 0:
                            continue  # Skip background
                        count = int(np.sum(cluster_mask_2d == cluster_id))
                        cluster_pixels_counts[cluster_id] = count
                        percentage = (count / anomaly_count * 100) if anomaly_count > 0 else 0
                        cluster_anomaly_percentages[cluster_id] = float(percentage)
                        results[threshold]["cluster_stats"].append(
                            {
                                "cluster_index": cluster_id,
                                "pixel_count": count,
                                "percentage_of_anomalies": float(percentage),
                            }
                        )
                        self.log.info(f"  Cluster {cluster_id}: {count} pixels ({percentage:.2f}%)")

                # Plot results
                plot_path = self._plot_results(
                    anomaly_threshold=threshold,
                    cluster_mask_2d=cluster_mask_2d,
                    cluster_cmap=cluster_cmap,
                    cluster_patches=cluster_patches,
                    n_clusters_found=n_clusters_found,
                    anomalous_pixels_count=anomaly_count,
                    cluster_pixels_counts=cluster_pixels_counts,  # Pass dict
                    cluster_anomaly_percentages=cluster_anomaly_percentages,  # Pass dict
                )
                results[threshold]["plot_path"] = plot_path

            self.log.info("--- Pipeline Execution Finished Successfully ---")
            results["status"] = "success"
            results["message"] = "Pipeline executed successfully."

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            self.log.critical(f"Pipeline failed: Data/Config error: {e}", exc_info=False)  # No need for traceback here
            results["status"] = "error"
            results["message"] = f"Pipeline failed: {e}"
        except Exception as e:
            self.log.critical(f"Unexpected error during pipeline execution: {e}", exc_info=True)  # Log traceback here
            results["status"] = "error"
            results["message"] = f"Pipeline failed unexpectedly: {e}"

        self.log.info("--- Pipeline Run Method Finished ---")
        return results


# --- Example Usage Guard (if running this file directly) ---
# Generally, pipeline classes shouldn't have __main__ blocks,
# but kept here if it was from original code for standalone testing.
if __name__ == "__main__":
    # This block should ideally be in a separate script like run_kmeans_pipeline.py
    # If kept, it needs its own imports like argparse
    parser = argparse.ArgumentParser(description="Run Solar Anomaly Pipeline (Standalone Test)")
    # Add arguments similar to the run_kmeans_pipeline.py script
    parser.add_argument("--data_dir", default="Data/sdo_data")
    parser.add_argument("--output_dir", default="./pipeline_standalone_output")
    parser.add_argument("--channels", nargs="+", default=["94", "171"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0])
    args = parser.parse_args()

    print("Running pipeline directly from pipeline.py (for testing)...")
    pipeline_instance = SolarAnomalyPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        image_size=args.image_size,
        # Add other params like contamination, n_clusters etc. if needed
    )
    run_results = pipeline_instance.run(args.thresholds)
    print("Pipeline finished. Results:")
    import json

    print(json.dumps(run_results, indent=2))
