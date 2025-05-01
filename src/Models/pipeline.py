# src/solar/pipeline.py

import os
import logging
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, MiniBatchKMeans # Allow selection
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches

# Configure logging for the module
# Consider using a more sophisticated logging setup for a larger application
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SolarAnomalyPipeline:
    """
    Encapsulates the entire pipeline for detecting and clustering anomalies
    in SDO/AIA multi-channel solar imagery.

    Designed for modularity and potential API integration.
    """
    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 channels: List[str],
                 image_size: Optional[int] = 512,
                 contamination: float = 0.05,
                 n_clusters: int = 5,
                 cluster_method: str = 'KMeans', # 'KMeans' or 'MiniBatchKMeans'
                 random_state: int = 42):
        """
        Initializes the pipeline with configuration parameters.

        Args:
            data_dir: Path to the root directory containing channel subdirs.
            output_dir: Path where result plots will be saved.
            channels: List of AIA channel wavelengths (as strings, e.g., ['94', '171']).
            image_size: Target square size for image resizing. None for no resize.
            contamination: Expected proportion of anomalies for Isolation Forest.
            n_clusters: Number of clusters (k) for anomaly grouping.
            cluster_method: Clustering algorithm ('KMeans' or 'MiniBatchKMeans').
            random_state: Seed for reproducibility.
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.channels = channels
        self.image_size = image_size
        self.contamination = contamination
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method.lower()
        self.random_state = random_state
        self.log = logging.getLogger(self.__class__.__name__)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Internal state variables (will be populated during run) ---
        self._masked_data_list: List[np.ndarray] = []
        self._loaded_channel_names: List[str] = []
        self._prepared_data: Optional[np.ndarray] = None
        self._valid_pixel_mask_flat: Optional[np.ndarray] = None
        self._nan_mask_flat: Optional[np.ndarray] = None
        self._image_shape: Optional[Tuple[int, int]] = None
        self._total_valid_pixels: int = 0
        self._anomaly_scores_valid: Optional[np.ndarray] = None

        self._validate_params()

    def _validate_params(self):
        """Basic validation of initialization parameters."""
        if not os.path.isdir(self.data_dir):
            self.log.warning(f"Data directory not found: {self.data_dir}")
            # Consider raising error depending on expected behavior
        if not self.channels:
            raise ValueError("Channel list cannot be empty.")
        if self.cluster_method not in ['kmeans', 'minibatchkmeans']:
             raise ValueError(f"Invalid cluster_method: {self.cluster_method}. Choose 'KMeans' or 'MiniBatchKMeans'.")
        self.log.info("Pipeline initialized with valid parameters.")

    # --------------------------------------------------------------------------
    # Private Helper Methods: Data Loading & Preprocessing
    # --------------------------------------------------------------------------

    def _load_fits_data(self, channel_dir: str) -> Tuple[np.ndarray, dict]:
        """Loads FITS data/metadata for a single channel."""
        # (Identical logic as before, now a private method)
        self.log.debug(f"Searching for FITS files in: {channel_dir}")
        fits_files = [f for f in os.listdir(channel_dir) if f.lower().endswith(".fits")]
        if not fits_files:
            raise FileNotFoundError(f"No FITS files found in directory: {channel_dir}")
        fits_files.sort()
        fits_path = os.path.join(channel_dir, fits_files[0])
        self.log.info(f"Loading FITS file: {fits_path}")
        try:
            aia_map = sunpy.map.Map(fits_path)
            data = np.array(aia_map.data) # Ensure data is read into memory
            return data, aia_map.meta
        except Exception as e:
            self.log.error(f"Error loading FITS file {fits_path}: {e}")
            raise

    def _create_circular_mask(self, data: np.ndarray, metadata: dict) -> np.ndarray:
        """Creates a circular solar disk mask."""
        # (Identical logic as before, now a private method)
        self.log.debug("Creating circular solar disk mask.")
        ny, nx = data.shape
        try:
            x_center_fits = metadata.get('CRPIX1', nx / 2.0 + 0.5)
            y_center_fits = metadata.get('CRPIX2', ny / 2.0 + 0.5)
            x_center = x_center_fits - 1.0
            y_center = y_center_fits - 1.0
            cdelt1 = abs(metadata.get('CDELT1', 1.0))
            if cdelt1 == 0: cdelt1 = 1.0
            solar_radius_arcsec = metadata.get('RSUN_OBS', metadata.get('R_SUN', 960.0))
            if solar_radius_arcsec is None: solar_radius_arcsec = 960.0
            solar_radius_pixels = solar_radius_arcsec / cdelt1
            self.log.debug(f"Disk mask params: Center=({x_center:.2f}, {y_center:.2f}), Radius={solar_radius_pixels:.2f} pix")
        except KeyError as e:
            self.log.error(f"Missing metadata key for mask creation: {e}. Using fallback.", exc_info=True)
            x_center, y_center = nx // 2, ny // 2
            solar_radius_pixels = min(nx, ny) * 0.48

        y_coords, x_coords = np.ogrid[:ny, :nx]
        distance_from_center = np.sqrt((x_coords - x_center)**2 + (y_coords - y_center)**2)
        mask = distance_from_center <= solar_radius_pixels
        self.log.debug(f"Generated mask shape: {mask.shape}, Masked pixels: {np.sum(mask)}")
        return mask

    def _preprocess_single_image(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        target_size: Optional[int]
    ) -> np.ndarray:
        """Resizes (optional) and applies mask to a single image."""
        # (Refactored logic from previous version)
        self.log.debug(f"Preprocessing single image. Original shape: {data.shape}, Target size: {target_size}")
        processed_data = data.astype(np.float32).copy()

        if target_size is not None:
            current_shape = (target_size, target_size)
            if data.shape == current_shape:
                self.log.debug("Image already at target size.")
                resized_data = processed_data
                # Ensure mask matches target size even if image didn't need resize
                if mask.shape != current_shape:
                     resized_mask = resize(mask, current_shape, mode='reflect', anti_aliasing=False, order=0) > 0.5
                else:
                     resized_mask = mask
            else:
                 self.log.debug(f"Resizing image and mask to {current_shape}")
                 resized_data = resize(processed_data, current_shape, mode='reflect', anti_aliasing=True)
                 resized_mask = resize(mask, current_shape, mode='reflect', anti_aliasing=False, order=0) > 0.5
        else:
            self.log.debug("No resizing requested.")
            resized_data = processed_data
            resized_mask = mask # Use original mask
            current_shape = data.shape # Store the original shape

        # Store the final shape if not already set (first image processed)
        if self._image_shape is None:
             self._image_shape = current_shape
        elif self._image_shape != current_shape:
            # This check is crucial if target_size is None
            raise ValueError(f"Inconsistent image shapes after processing. Expected {self._image_shape}, got {current_shape}. All images must have the same dimensions.")

        self.log.debug(f"Applying mask. Mask shape: {resized_mask.shape}, True values: {np.sum(resized_mask)}")
        masked_data = resized_data
        masked_data[~resized_mask] = np.nan
        self.log.debug(f"Preprocessing complete. Output shape: {masked_data.shape}")
        return masked_data

    def _prepare_data_for_model(self) -> None:
        """
        Stacks, handles NaNs, reshapes, and scales the preprocessed data list.
        Stores results in instance attributes.
        """
        if not self._masked_data_list:
            raise ValueError("Cannot prepare data for model: No masked data available. Run _load_all_channels first.")
        if self._image_shape is None:
             raise ValueError("Cannot prepare data for model: Image shape not determined.")

        self.log.info(f"Preparing data for model from {len(self._masked_data_list)} channels.")

        stacked_data = np.stack(self._masked_data_list, axis=-1)
        n_pixels = stacked_data.shape[0] * stacked_data.shape[1]
        n_channels = stacked_data.shape[2]
        reshaped_data = stacked_data.reshape((n_pixels, n_channels))

        self._nan_mask_flat = np.isnan(reshaped_data).any(axis=1)
        self._valid_pixel_mask_flat = ~self._nan_mask_flat
        self._total_valid_pixels = int(np.sum(self._valid_pixel_mask_flat)) # Cast to int
        self.log.info(f"Found {self._total_valid_pixels} valid pixels out of {n_pixels} total.")

        if self._total_valid_pixels == 0:
            raise ValueError("No valid pixels found after handling NaNs.")

        cleaned_data = reshaped_data[self._valid_pixel_mask_flat]
        scaler = RobustScaler()
        self.log.info("Applying RobustScaler...")
        self._prepared_data = scaler.fit_transform(cleaned_data)
        self.log.info("Scaling complete.")
        self.log.debug(f"Prepared data shape: {self._prepared_data.shape}")


    def _load_and_preprocess_all(self) -> None:
        """Loads and preprocesses data for all specified channels."""
        self.log.info(f"--- Starting Data Loading and Preprocessing for channels: {self.channels} ---")
        self._masked_data_list = []
        self._loaded_channel_names = []
        self._image_shape = None # Reset shape for this run

        for channel in self.channels:
            channel_dir_name = f"aia_{channel}"
            channel_path = os.path.join(self.data_dir, channel_dir_name)
            self.log.info(f"Processing channel: {channel} from {channel_path}")

            if not os.path.isdir(channel_path):
                self.log.warning(f"Channel directory not found: {channel_path}. Skipping channel {channel}.")
                continue

            try:
                data, metadata = self._load_fits_data(channel_path)
                mask = self._create_circular_mask(data, metadata)
                preprocessed_img = self._preprocess_single_image(data, mask, self.image_size)
                self._masked_data_list.append(preprocessed_img)
                self._loaded_channel_names.append(channel)
            except FileNotFoundError as e:
                self.log.error(f"FITS file not found for channel {channel}: {e}")
                raise # Fail fast if a requested channel cannot be loaded
            except Exception as e:
                self.log.error(f"Error processing channel {channel}: {e}", exc_info=True)
                raise # Fail fast on other errors

        if not self._masked_data_list:
            raise ValueError("No data was successfully loaded for any specified channel.")

        self.log.info(f"Successfully loaded and individually preprocessed {len(self._masked_data_list)} channels: {self._loaded_channel_names}")

        # Final preparation step (stacking, scaling)
        self._prepare_data_for_model()
        self.log.info("--- Data Loading and Preprocessing Complete ---")

    # --------------------------------------------------------------------------
    # Private Helper Methods: Modeling
    # --------------------------------------------------------------------------

    def _detect_anomalies(self) -> None:
        """Runs Isolation Forest on the prepared data."""
        if self._prepared_data is None:
            raise RuntimeError("Prepared data not available. Run _load_and_preprocess_all first.")

        self.log.info(f"Running Isolation Forest with contamination={self.contamination}, random_state={self.random_state}")
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        iso_forest.fit(self._prepared_data)
        self._anomaly_scores_valid = iso_forest.decision_function(self._prepared_data)
        self.log.info("Isolation Forest scoring complete.")

    def _cluster_anomalies(self, anomaly_features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Performs clustering on the features of anomalous pixels."""
        if anomaly_features.ndim != 2 or anomaly_features.shape[0] == 0:
             self.log.warning("No anomaly features provided for clustering.")
             return np.array([]), 0.0 # Return empty results

        self.log.info(f"Running {self.cluster_method} with n_clusters={self.n_clusters}, random_state={self.random_state}")

        if self.cluster_method == 'kmeans':
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init='auto'
            )
        elif self.cluster_method == 'minibatchkmeans':
             # Set batch size appropriately, e.g., based on 
             # expected number of anomalies or memory limits
             batch_size = min(1024, anomaly_features.shape[0])
             model = MiniBatchKMeans(
                 n_clusters=self.n_clusters,
                 random_state=self.random_state,
                 batch_size=batch_size,
                 n_init='auto' # Use 'auto' or default 10 depending on version
             )
        else:
             # Should have been caught in __init__, but defensive check
             raise ValueError(f"Internal error: Unsupported cluster_method '{self.cluster_method}'")

        model.fit(anomaly_features)
        self.log.info("Clustering complete.")
        return model.labels_, model.inertia_ if hasattr(model, 'inertia_') else 0.0

    # --------------------------------------------------------------------------
    # Private Helper Methods: Visualization
    # --------------------------------------------------------------------------

    def _create_cluster_mask_2d(
        self,
        anomaly_mask_valid_flat: np.ndarray, # Mask for anomalies within the valid set
        cluster_labels: np.ndarray           # Labels for those anomalies
    ) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
        """Creates the 2D cluster mask, colormap, and legend patches."""
        # (Uses logic from previous `create_cluster_mask`, now using instance attributes)
        self.log.debug("Creating 2D cluster mask...")
        if self._image_shape is None or self._valid_pixel_mask_flat is None:
             raise RuntimeError("Image shape or valid pixel mask not available.")
        if len(anomaly_mask_valid_flat) != len(cluster_labels):
            raise ValueError(f"Anomaly mask length ({len(anomaly_mask_valid_flat)}) \
                and labels length ({len(cluster_labels)}) mismatch.")

        cluster_mask_2d = np.zeros(self._image_shape, dtype=int)
        n_total_pixels = self._image_shape[0] * self._image_shape[1]

        if len(cluster_labels) == 0:
            self.log.warning("No anomaly labels provided, returning empty cluster mask.")
            return cluster_mask_2d, matplotlib.colors.ListedColormap([]), [], 0

        full_flat_labels = np.full(n_total_pixels, -1, dtype=int) # Use -1 for background
        valid_indices = np.where(self._valid_pixel_mask_flat)[0]
        anomalous_indices_in_valid_set = np.where(anomaly_mask_valid_flat)[0]

        if len(anomalous_indices_in_valid_set) > 0:
            original_anomalous_indices = valid_indices[anomalous_indices_in_valid_set]
            full_flat_labels[original_anomalous_indices] = cluster_labels + 1 # Make 1-based

        cluster_mask_2d = full_flat_labels.reshape(self._image_shape)

        unique_labels = np.unique(cluster_labels) # 0-based original labels
        n_clusters = len(unique_labels)
        if n_clusters == 0: return cluster_mask_2d, \
            matplotlib.colors.ListedColormap([]), [], 0

        self.log.info(f"Creating colormap and patches for {n_clusters} clusters.")
        cluster_colors = plt.cm.get_cmap('tab10', n_clusters).colors
        cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors)
        cluster_patches = [
            mpatches.Patch(color=cluster_cmap(i), label=f'Cluster {label_value + 1}')
            for i, label_value in enumerate(unique_labels)
        ]

        self.log.debug("Cluster mask creation complete.")
        return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters

    def _plot_results(
        self,
        anomaly_threshold: float,
        cluster_mask_2d: np.ndarray,
        cluster_cmap: matplotlib.colors.ListedColormap,
        n_clusters: int,
        cluster_patches: list,
        anomaly_pixels_count: int,
        cluster_pixels_counts: Optional[List[int]] = None,
        cluster_anomaly_percentages: Optional[List[float]] = None,
    ):
        """Generates and saves the result plot for a given threshold."""
        # (Uses logic from previous `plot_results`, now using instance attributes)
        if not self._masked_data_list:
             self.log.warning("No masked data available for plotting.")
             return

        self.log.info(f"Plotting results for threshold {anomaly_threshold:.2f}...")
        num_channels = len(self._loaded_channel_names)
        num_cols = int(np.ceil(np.sqrt(num_channels)))
        num_rows = int(np.ceil(num_channels / num_cols))

        fig, axes = plt.subplots(num_rows,
                                 num_cols,
                                 figsize=(num_cols * 5,
                                          num_rows * 4.5),
                                 dpi=120, squeeze=False)
        axes = axes.flatten()

        anomaly_percentage = (
            anomaly_pixels_count / self._total_valid_pixels) * 100  \
                self._total_valid_pixels > 0 else 0
        title = (f'{self.cluster_method.capitalize()} Anomaly Clusters in SDO/AIA Channels\n'
                 f'Threshold: {anomaly_threshold:.2f} | Anomalous Pixels: {anomaly_pixels_count}/{self._total_valid_pixels} ({anomaly_percentage:.2f}%)')
        fig.suptitle(title, fontsize=16, y=0.99)

        # Calculate vmin/vmax dynamically just once before the loop
        v_ranges = {}
        for masked_data in self._masked_data_list:
            valid_pix = masked_data[~np.isnan(masked_data)]
            v_ranges[id(masked_data)] = (np.percentile(valid_pix, 2), 
                                         np.percentile(valid_pix, 98)) \
                                             if len(valid_pix) > 0 else (0,1)


        for i, (masked_data, channel) in enumerate(zip(self._masked_data_list, self._loaded_channel_names)):
            ax = axes[i]
            vmin, vmax = v_ranges[id(masked_data)]
            ax.imshow(masked_data, cmap='YlOrBr', origin='lower', vmin=vmin, vmax=vmax, alpha=0.6)

            if n_clusters > 0 and np.any(cluster_mask_2d > 0):
                cluster_overlay_data = np.ma.masked_where(cluster_mask_2d == 0, cluster_mask_2d)
                ax.imshow(cluster_overlay_data, cmap=cluster_cmap, alpha=0.7, origin='lower',
                          interpolation='nearest', vmin=1, vmax=n_clusters)

            ax.set_title(f'AIA {channel} Å', color='black', fontsize=12, pad=5)
            ax.axis('off')

        if cluster_patches:
            fig.legend(handles=cluster_patches, loc='upper right',
                       bbox_to_anchor=(0.98, 0.95),
                       fontsize='medium', frameon=True,
                       framealpha=0.9, title="Anomaly Clusters")

        for j in range(num_channels, num_rows * num_cols):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        filename = os.path.join(
            self.output_dir,
            f"{self.cluster_method}_anomalies_thresh_{anomaly_threshold:.2f}_k{self.n_clusters}.png" # Use configured k
        )
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        self.log.info(f"Result plot saved to: {filename}")

    # --------------------------------------------------------------------------
    # Public Execution Method
    # --------------------------------------------------------------------------

    def run(self, anomaly_thresholds: List[float]) -> None:
        """
        Executes the full anomaly detection pipeline for the given thresholds.

        Args:
            anomaly_thresholds: A list of anomaly score thresholds to process.
        """
        try:
            # 1. Load and Preprocess Data (stores results in self.* attributes)
            self._load_and_preprocess_all()

            # 2. Anomaly Detection (stores scores in self._anomaly_scores_valid)
            self._detect_anomalies()

            if self._anomaly_scores_valid is None or self._prepared_data is None:
                 raise RuntimeError("Anomaly \
                     scores or prepared data are missing after initial steps.")

            # 3. Process each threshold
            for threshold in anomaly_thresholds:
                self.log.info(f"--- Processing Threshold: {threshold:.2f} ---")

                # Identify anomalous pixels within the valid set
                anomaly_mask_valid_flat = self._anomaly_scores_valid < threshold
                anomaly_indices_in_valid = np.where(anomaly_mask_valid_flat)[0]
                anomaly_count = len(anomaly_indices_in_valid)

                if anomaly_count == 0:
                    self.log.warning(f"No anomalies found for threshold {threshold:.2f}. Skipping clustering and plotting.")
                    # Optionally plot an empty visualization
                    self._plot_results(
                         anomaly_threshold=threshold,
                         cluster_mask_2d=np.zeros(self._image_shape, dtype=int),
                         cluster_cmap=matplotlib.colors.ListedColormap([]),
                         n_clusters=0,
                         cluster_patches=[],
                         anomaly_pixels_count=0
                    )
                    continue # Skip to the next threshold

                self.log.info(f"Found {anomaly_count} \
                    anomalous pixels (out of {self._total_valid_pixels} valid).")
                anomaly_features = self._prepared_data[anomaly_indices_in_valid]

                # Cluster anomalous pixels
                cluster_labels, _ = self._cluster_anomalies(anomaly_features)

                # Create 2D mask for visualization
                cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters_found = \
                    self._create_cluster_mask_2d(anomaly_mask_valid_flat, cluster_labels)

                # Optional: Calculate stats (can be added if needed for plots)
                # cluster_pixels_counts = ...
                # cluster_anomaly_percentages = ...

                # Plot results for this threshold
                self._plot_results(
                    anomaly_threshold=threshold,
                    cluster_mask_2d=cluster_mask_2d,
                    cluster_cmap=cluster_cmap,
                    n_clusters=n_clusters_found,
                    cluster_patches=cluster_patches,
                    anomaly_pixels_count=anomaly_count
                    # Pass stats if calculated:
                    # cluster_pixels_counts=cluster_pixels_counts,
                    # cluster_anomaly_percentages=cluster_anomaly_percentages,
                )

            self.log.info("--- Pipeline Execution Finished Successfully ---")

        except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
            self.log.critical(f"Pipeline execution failed: {e}", exc_info=True)
            # Optionally re-raise or handle specific exceptions differently
            