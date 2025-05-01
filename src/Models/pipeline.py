# src/solar/pipeline.py

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest

# Import refactored modules
from . import utils

# Configure logging for the module
# Consider using a more sophisticated logging setup
# This is basic, assuming root logger is configured elsewhere or defaults are fine
log = logging.getLogger(__name__)


class SolarAnomalyPipeline:
    """Encapsulates the entire pipeline for detecting and clustering anomalies in
    SDO/AIA multi-channel solar imagery.

    Designed for modularity and potential API integration.
    """

    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 channels: List[str],
                 image_size: Optional[int] = 512,
                 contamination: float = 0.05,
                 n_clusters: int = 7,  # Default 7 for KMeans
                 cluster_method: str = 'KMeans',  # 'KMeans' or 'MiniBatchKMeans'
                 random_state: int = 42):
        """Initializes the pipeline with configuration parameters.

        Args:
            data_dir: Path to the root directory containing channel subdirs (e.g., 'Data/sdo_data').
            output_dir: Path where result plots will be saved.
            channels: List of AIA channel wavelengths as strings (e.g., ['94', '171']).
                      These should correspond to directory names like 'aia_94'.
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
        # Logger specific to this class
        self.log = logging.getLogger(self.__class__.__name__)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Internal state variables (will be populated during run) ---
        self._masked_data_list: List[np.ndarray] = []
        self._loaded_channel_names: List[str] = []
        # Scaled data for model (valid pixels only)
        self._prepared_data: Optional[np.ndarray] = None
        # 1D mask (True for valid pixels in flattened array)
        self._valid_pixel_mask_flat: Optional[np.ndarray] = None
        # 1D mask (True for nan pixels in flattened array)
        self._nan_mask_flat: Optional[np.ndarray] = None
        # Final shape (resized or original)
        self._image_shape: Optional[Tuple[int, int]] = None
        self._total_pixels_final_shape: int = 0  # Total pixels in the final shape
        self._total_valid_pixels: int = 0  # Total pixels after removing NaNs
        # Isolation Forest scores for valid pixels (on prepared data)
        self._anomaly_scores_valid: Optional[np.ndarray] = None

        self._validate_params()

    def _validate_params(self):
        """Basic validation of initialization parameters."""
        if not os.path.isdir(self.data_dir):
            self.log.warning(f"Data directory not found: {self.data_dir}")
            # In a real API, you might want to raise an error here
            # raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.channels:
            raise ValueError("Channel list cannot be empty.")
        if self.cluster_method not in ['kmeans', 'minibatchkmeans']:
            raise ValueError(f"Invalid cluster_method: {
                             self.cluster_method}. Choose 'KMeans' or 'MiniBatchKMeans'.")
        if self.n_clusters <= 0:
            raise ValueError("Number of clusters must be positive.")
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError("Image size must be positive if specified.")

        self.log.info("Pipeline initialized with valid parameters.")

    # --------------------------------------------------------------------------
    # Private Step Methods (Orchestrating Calls to Utils/Models/Plotting)
    # --------------------------------------------------------------------------

    def _load_and_preprocess_all(self) -> None:
        """Loads and preprocesses data for all specified channels."""
        self.log.info(
            f"--- Starting Data Loading and Preprocessing for channels: {self.channels} ---")
        self._masked_data_list = []
        self._loaded_channel_names = []
        self._image_shape = None  # Reset shape for this run
        self._total_pixels_final_shape = 0  # Reset count

        # Construct expected directory names
        channels_to_process_dirs = [f"aia_{c}" for c in self.channels]

        for channel_dir_name in channels_to_process_dirs:
            channel = channel_dir_name.split("_")[1]  # Get just the wavelength part
            channel_path = os.path.join(self.data_dir, channel_dir_name)
            self.log.debug(f"Attempting to process directory: {channel_path}")

            if not os.path.isdir(channel_path):
                self.log.warning(f"Channel directory not found: {
                                 channel_path}. Skipping channel {channel}.")
                continue  # Skip to the next channel if directory doesn't exist

            try:
                # Use the functions from utils.py
                data, metadata = utils.load_fits_data(channel_path)
                mask = utils.create_circular_mask(data, metadata)
                # Pass the mask and target_size to preprocess_image from utils.py
                preprocessed_img = utils.preprocess_image(data, mask, self.image_size)

                self._masked_data_list.append(preprocessed_img)
                self._loaded_channel_names.append(channel)

                # Store the final processed image shape and total pixels in the grid
                if self._image_shape is None:
                    self._image_shape = preprocessed_img.shape
                    self._total_pixels_final_shape = self._image_shape[0] * \
                        self._image_shape[1]
                    self.log.info(f"First image processed. Final image shape: {
                                  self._image_shape}, Total pixels in grid: {self._total_pixels_final_shape}")
                elif preprocessed_img.shape != self._image_shape:
                    # This is a critical check, especially if self.image_size is None
                    raise ValueError(f"Inconsistent image shapes after preprocessing. Expected {self._image_shape}, got {
                                     preprocessed_img.shape}. All processed images must have identical dimensions.")

            except FileNotFoundError as e:
                self.log.error(f"FITS file not found in {
                               channel_path} or directory missing: {e}")
                # In an API, you might want to handle this differently, e.g., return an
                # error for this channel
                continue  # Continue to the next channel if a FITS file is missing or dir is not found
            except ValueError as e:
                self.log.error(
                    f"Data consistency error processing channel {channel}: {e}")
                raise  # Re-raise ValueErrors related to inconsistent shapes
            except Exception as e:
                self.log.error(f"Unexpected error processing channel {
                               channel}: {e}", exc_info=True)
                # Depending on requirements, you might raise here too
                continue  # Continue to the next channel on other unexpected errors

        if not self._masked_data_list:
            # This happens if none of the specified channels could be loaded/processed
            raise ValueError(
                "No data was successfully loaded or processed for any specified channel.")

        self.log.info(f"Successfully loaded and individually preprocessed {
                      len(self._masked_data_list)} channels: {self._loaded_channel_names}.")

        # Final preparation step (stacking, scaling) using utils.py
        try:
            (self._prepared_data,
             self._valid_pixel_mask_flat,
             self._nan_mask_flat,
             _) = utils.prepare_data_concatenated(self._masked_data_list)
            # Recalculate valid pixels after concatenation NaN check
            self._total_valid_pixels = int(np.sum(self._valid_pixel_mask_flat))
            self.log.info(f"Data concatenated and scaled. Total valid pixels for modeling: {
                          self._total_valid_pixels}")

        except ValueError as e:
            self.log.critical(f"Error during data preparation after preprocessing: {e}")
            raise  # Re-raise to be caught by the main run method

        self.log.info("--- Data Loading and Preprocessing Complete ---")

    def _detect_anomalies(self) -> None:
        """Runs Isolation Forest on the prepared data and stores scores."""
        if self._prepared_data is None:
            raise RuntimeError(
                "Prepared data not available. Run _load_and_preprocess_all first.")

        self.log.info(f"Running Isolation Forest on {self._prepared_data.shape[0]} valid pixels with contamination={
                      self.contamination}, random_state={self.random_state}")
        # Call the function directly (or potentially a wrapper in models.py)
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,  # Use configured random_state
            n_jobs=-1  # Use all available CPU cores
        )
        iso_forest.fit(self._prepared_data)
        # Get decision function scores for the valid pixels
        self._anomaly_scores_valid = iso_forest.decision_function(self._prepared_data)

        self.log.info("Isolation Forest scoring complete.")
        # The anomaly mask is determined later based on the threshold in the run loop

    def _cluster_anomalies(
            self, anomaly_features: np.ndarray) -> Tuple[np.ndarray, int]:
        """Performs clustering on the features of anomalous pixels."""
        if anomaly_features.ndim != 2 or anomaly_features.shape[0] == 0:
            self.log.warning("No anomaly features provided for clustering.")
            # Return empty labels (as int) and 0 inertia
            return np.array([], dtype=int), 0

        self.log.info(f"Running {self.cluster_method} on {anomaly_features.shape[0]} anomaly features with n_clusters={
                      self.n_clusters}, random_state={self.random_state}")

        try:
            if self.cluster_method == 'kmeans':
                # Use KMeans from scikit-learn
                model = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,  # Use configured random_state
                    n_init='auto'  # Recommended in recent scikit-learn to avoid warnings
                )
            elif self.cluster_method == 'minibatchkmeans':
                # Use MiniBatchKMeans from scikit-learn
                # Set batch size appropriately
                batch_size = min(1024, anomaly_features.shape[0])
                model = MiniBatchKMeans(
                    n_clusters=self.n_clusters,
                    random_state=self.random_state,  # Use configured random_state
                    batch_size=batch_size,
                    n_init='auto'  # Use 'auto' or default 10 depending on version
                )
            else:
                # Should have been caught in __init__, but defensive check
                self.log.error(f"Unsupported cluster_method '{
                               self.cluster_method}' encountered during execution.")
                raise ValueError(f"Unsupported cluster_method: {self.cluster_method}")

            model.fit(anomaly_features)
            self.log.info("Clustering complete.")
            # Return labels (guaranteed int by KMeans) and inertia
            return model.labels_.astype(int), model.inertia_ if hasattr(
                model, 'inertia_') else 0.0

        except Exception as e:
            self.log.error(f"Error during clustering with {
                           self.cluster_method}: {e}", exc_info=True)
            raise  # Re-raise to be caught by the main run method

    def _create_cluster_mask_2d(
        self,
        # Mask for anomalies *within the valid set* (length = total_valid_pixels)
        anomaly_mask_valid_flat: np.ndarray,
        # Labels for *those* anomalies (length = number of anomalies)
        cluster_labels: np.ndarray
    ) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
        """Creates the 2D cluster mask on the full image grid (final shape), colormap,
        and legend patches."""
        self.log.debug("Creating 2D cluster mask...")
        if self._image_shape is None or self._valid_pixel_mask_flat is None or self._nan_mask_flat is None:
            raise RuntimeError(
                "Image shape, valid pixel mask, or nan mask not available.")

        n_total_pixels_grid = self._image_shape[0] * self._image_shape[1]
        # Initialize flat mask with 0 for background (pixels outside disk or NaN
        # in some channel)
        cluster_mask_flat_full_grid = np.zeros(n_total_pixels_grid, dtype=int)

        # Find the indices in the *original flattened* array that are valid
        original_valid_flat_indices = np.where(self._valid_pixel_mask_flat)[0]

        if len(cluster_labels) == 0 or len(anomaly_mask_valid_flat) == 0:
            self.log.warning(
                "No anomaly labels or mask provided, returning empty cluster mask.")
            return cluster_mask_flat_full_grid.reshape(self._image_shape), \
                matplotlib.colors.ListedColormap([]), [], 0  # Return empty

        # anomaly_mask_valid_flat is a mask *of the valid pixels*.
        # So anomaly_mask_valid_flat.shape[0] should equal self._total_valid_pixels
        if anomaly_mask_valid_flat.shape[0] != self._total_valid_pixels:
            # This check ensures the mask passed here aligns with the valid pixels
            # identified earlier
            self.log.error(f"Internal error: anomaly_mask_valid_flat length ({
                           anomaly_mask_valid_flat.shape[0]}) does not match total valid pixels ({self._total_valid_pixels}).")
            raise ValueError("Internal data mismatch.")

        # Get the indices *within the valid set* that were marked as anomalous
        anomalous_indices_in_valid_set = np.where(anomaly_mask_valid_flat)[0]

        # The cluster_labels are 0-based labels for the pixels *identified as anomalous*
        if len(cluster_labels) != len(anomalous_indices_in_valid_set):
            self.log.error(f"Internal error: Cluster labels length ({len(
                cluster_labels)}) does not match count of anomalous valid pixels ({len(anomalous_indices_in_valid_set)}).")
            raise ValueError("Internal data mismatch.")

        # Get the indices in the *original flattened* grid that correspond to the
        # anomalous valid pixels
        original_anomalous_flat_indices = original_valid_flat_indices[anomalous_indices_in_valid_set]

        # Assign cluster labels (plus 1 to make them 1-based, as 0 is background)
        # to these indices
        cluster_mask_flat_full_grid[original_anomalous_flat_indices] = cluster_labels + 1

        # Reshape the flat mask back to 2D image shape
        cluster_mask_2d = cluster_mask_flat_full_grid.reshape(self._image_shape)

        # Create colormap and patches
        # Get actual labels found (0-based)
        unique_labels_found = np.unique(cluster_labels)
        n_clusters_found = len(unique_labels_found)
        # Ensure we only create patches for labels that were actually assigned (in case n_clusters_attempted > actual clusters)
        # Sort for consistent patch order
        valid_patch_labels = sorted(list(unique_labels_found))

        if n_clusters_found == 0:
            self.log.warning("No unique cluster labels found after clustering.")
            return cluster_mask_2d, matplotlib.colors.ListedColormap([]), [], 0

        self.log.info(f"Creating colormap and patches for {
                      n_clusters_found} found clusters.")
        # Use the original color palette
        cluster_colors = [
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f'
        ]
        # If more clusters than defined colors, cycle or use a standard cmap
        if n_clusters_found > len(cluster_colors):
            self.log.warning(f"Number of clusters ({n_clusters_found}) exceeds defined colors ({
                             len(cluster_colors)}). Using 'tab10' colormap.")
            cluster_cmap = plt.cm.get_cmap('tab10', n_clusters_found)
            cluster_patches = [
                mpatches.Patch(color=cluster_cmap(
                    i), label=f'Cluster {label_value + 1}')
                # Use sorted found labels for patches
                for i, label_value in enumerate(valid_patch_labels)
            ]
        else:
            # Create colormap from the defined colors
            cluster_cmap = matplotlib.colors.ListedColormap(
                cluster_colors[:n_clusters_found])
            # Create patches using the colormap (map 0..N-1 indices to colors)
            cluster_patches = [
                mpatches.Patch(color=cluster_cmap(
                    i), label=f'Cluster {label_value + 1}')
                # Use sorted found labels for patches
                for i, label_value in enumerate(valid_patch_labels)
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
        cluster_pixels_counts: List[int],
        cluster_anomaly_percentages: List[float],
    ):
        """Generates and saves the result plot for a given threshold."""
        if not self._masked_data_list or not self._loaded_channel_names:
            self.log.warning("No masked data or channel names available for plotting.")
            return

        self.log.info(f"Plotting results for threshold {
                      anomaly_threshold:.2f} and {n_clusters_found} clusters...")
        num_channels = len(self._loaded_channel_names)
        # Use a 3x3 grid for consistent layout if possible
        num_cols = 3
        num_rows = int(np.ceil(num_channels / num_cols))
        if num_rows == 0:
            num_rows = 1

        fig, axes = plt.subplots(num_rows,
                                 num_cols,
                                 figsize=(num_cols * 5,
                                          num_rows * 4.5),
                                 dpi=120,
                                 squeeze=False)
        axes = axes.flatten()

        # Figure Suptitle (Main Title)
        anomaly_percentage_total = (anomalous_pixels_count / self._total_pixels_final_shape) * 100 \
            if self._total_pixels_final_shape > 0 else 0
        suptitle_text = (f'{self.cluster_method.capitalize()} Anomaly Clusters in SDO/AIA EUV Channels\n'
                         f'Threshold: {anomaly_threshold:.2f} | Anomalous Pixels: {anomalous_pixels_count}/{self._total_pixels_final_shape} ({anomaly_percentage_total:.2f}%)')
        fig.suptitle(suptitle_text, fontsize=16, y=0.99)

        # Determine Global Vmin/Vmax for Consistent Color Scaling (using all
        # loaded data)
        all_valid_pixels = np.concatenate([
            data[~np.isnan(data)].flatten() for data in self._masked_data_list if np.any(~np.isnan(data))
        ])
        vmin_global = np.percentile(all_valid_pixels, 2) if len(
            all_valid_pixels) > 0 else 0
        vmax_global = np.percentile(all_valid_pixels, 98) if len(
            all_valid_pixels) > 0 else 1
        # Handle edge cases for vmin/vmax
        if vmin_global >= vmax_global:
            log.warning(f"Global vmin ({vmin_global:.2f}) >= vmax ({
                        vmax_global:.2f}), adjusting limits.")
            v_range = np.nanmax(all_valid_pixels) - np.nanmin(all_valid_pixels)
            if v_range > 0:
                vmin_global = np.nanmin(all_valid_pixels)
                vmax_global = np.nanmax(all_valid_pixels)
            else:  # All valid pixels are the same value or all NaNs
                vmin_global, vmax_global = (np.nanmin(all_valid_pixels) - 1, np.nanmax(
                    all_valid_pixels) + 1) if np.any(~np.isnan(all_valid_pixels)) else (0, 1)

        # Plot Each Channel
        for i, (masked_data, channel) in enumerate(
                zip(self._masked_data_list, self._loaded_channel_names)):
            # Use break instead of continue for safety if grid size is miscalculated
            if i >= num_rows * num_cols:
                break

            ax = axes[i]
            # Display the base image with global color scaling and original
            # colormap/alpha
            ax.imshow(masked_data, cmap='YlOrBr', origin='lower',
                      vmin=vmin_global, vmax=vmax_global, alpha=0.5)  # Original alpha

            # Overlay clusters only if there are clusters found and mask matches image
            # shape
            if n_clusters_found > 0 and cluster_mask_2d.shape == masked_data.shape and np.any(
                    cluster_mask_2d > 0):
                # Mask the 2D cluster mask to only show cluster areas (where value > 0)
                cluster_overlay_data = np.ma.masked_where(
                    cluster_mask_2d == 0, cluster_mask_2d)
                ax.imshow(cluster_overlay_data, cmap=cluster_cmap, alpha=0.6, origin='lower',  # Original alpha
                          interpolation='nearest',  # Use nearest for masks
                          # Set vmin/vmax to span cluster labels (1 to N)
                          vmin=1, vmax=n_clusters_found)
            elif np.any(cluster_mask_2d > 0) and cluster_mask_2d.shape != masked_data.shape:
                # This indicates a mismatch, which ideally is caught earlier in
                # _preprocess_single_image
                log.error(f"Internal error: Cluster mask shape {cluster_mask_2d.shape} does not match image shape {
                          masked_data.shape} for channel {channel}. Skipping cluster overlay.")

            # Subplot Title (Channel + Cluster Info)
            title_lines = [f'AIA {channel} Å']
            # Add cluster stats to title if provided and match number of clusters found
            if len(cluster_pixels_counts) == n_clusters_found and len(
                    cluster_anomaly_percentages) == n_clusters_found:
                # We assume stats lists correspond to original cluster labels 0..N-1
                # And cluster_mask_2d uses 1..N-1. The patches correspond to 1..N as displayed.
                # Let's iterate through patches to get the 1-based label for display
                # Find corresponding stat index (0-based)
                for patch in cluster_patches:
                    try:
                        cluster_id = int(patch.get_label().split(
                            ' ')[1])  # Get 1-based ID from label
                        original_label_index = cluster_id - 1  # Convert to 0-based index

                        if 0 <= original_label_index < n_clusters_found:
                            pixels = cluster_pixels_counts[original_label_index]
                            percentage = cluster_anomaly_percentages[original_label_index]
                            # Only add if there are pixels in this cluster
                            if pixels > 0:
                                # Check if adding this line exceeds max lines
                                if len(
                                        title_lines) < 5:  # Limit total lines including channel name
                                    title_lines.append(f'C{cluster_id}: {
                                                       pixels} Pix ({percentage:.1f}%)')
                                else:
                                    # Indicate more clusters exist but are not shown
                                    title_lines.append('...')
                                    break  # Stop adding stats lines for this subplot
                        else:
                            log.debug(f"Cluster ID {cluster_id} from patch label is out of expected range {
                                      1}..{n_clusters_found}.")
                    except Exception as e:
                        log.debug(f"Error parsing cluster patch label {
                                  patch.get_label()}: {e}")
                        # Fallback to just adding the label if stats can't be parsed
                        if len(title_lines) < 5:
                            title_lines.append(patch.get_label())
                        else:
                            title_lines.append('...')
                            break

            ax.set_title(
                "\n".join(title_lines),  # Multiline title
                color='black', fontsize=10, pad=5  # Adjusted fontsize and padding
            )
            ax.axis('off')  # Hide axes ticks and labels

        # Turn off any unused subplots
        for j in range(num_channels, num_rows * num_cols):
            axes[j].axis('off')

        # Add legend outside the subplots
        if cluster_patches:  # Only add legend if there are clusters to show
            # Filter patches to only include those that actually have pixels assigned
            # in the mask
            valid_patches = []
            if cluster_pixels_counts is not None and len(
                    cluster_pixels_counts) == n_clusters_found:
                for patch in cluster_patches:
                    try:
                        cluster_id = int(patch.get_label().split(' ')
                                         [1])  # Get 1-based ID
                        original_label_index = cluster_id - 1  # Convert to 0-based index
                        if 0 <= original_label_index < n_clusters_found and cluster_pixels_counts[
                                original_label_index] > 0:
                            valid_patches.append(patch)
                    except Exception as e:
                        log.debug(f"Error filtering patch {patch.get_label()}: {e}")
                        valid_patches.append(patch)  # Include if parsing fails

            else:  # If counts not available or mismatch, include all patches
                valid_patches = cluster_patches

            if valid_patches:
                fig.legend(handles=valid_patches, loc='upper right',
                           bbox_to_anchor=(0.98, 0.95),  # Position of the legend
                           fontsize='small', frameon=True,  # Add frame for clarity
                           framealpha=0.9, title="Anomaly Clusters")
            else:
                self.log.info("No clusters with pixels assigned, skipping legend.")

        # Final Layout and Saving
        # Adjust layout to make space for legend on the right
        plt.tight_layout(rect=[0, 0, 0.90, 0.95])
        # Use a descriptive filename including cluster method, threshold, and k
        filename = os.path.join(
            self.output_dir,
            f"{self.cluster_method.lower().replace(' ', '_')}_anomalies_threshold_{
                anomaly_threshold:.2f}_k{self.n_clusters}_clusters.png"  # Use configured k
        )
        try:
            # Increased dpi for better quality
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            self.log.info(f"Result plot saved to: {filename}")
            return filename  # Return the path on success
        except Exception as e:
            self.log.error(f"Error saving figure {filename}: {e}", exc_info=True)
            return None  # Return None or raise if saving fails

    # --------------------------------------------------------------------------
    # Public Execution Method
    # --------------------------------------------------------------------------

    def run(self, anomaly_thresholds: List[float]) -> Dict[str, Any]:
        """Executes the full anomaly detection pipeline for the given thresholds.

        Args:
            anomaly_thresholds: A list of anomaly score thresholds to process.

        Returns:
             A dictionary containing results, including plot paths, for each threshold.
        """
        results = {}  # Dictionary to store results for API response
        self.log.info(
            f"--- Starting pipeline execution for thresholds: {anomaly_thresholds} ---")

        try:
            # 1. Load and Preprocess Data (stores results in self.* attributes)
            self._load_and_preprocess_all()
            self.log.info(f"Data loaded and preprocessed. Final image shape: {
                          self._image_shape}, Total valid pixels: {self._total_valid_pixels}")

            if self._image_shape is None:
                raise RuntimeError(
                    "Image shape was not determined during preprocessing.")

            # 2. Anomaly Detection (stores scores in self._anomaly_scores_valid)
            # This is done ONCE for all thresholds as it uses the same Isolation
            # Forest model
            self._detect_anomalies()

            if self._anomaly_scores_valid is None or self._prepared_data is None:
                self.log.error(
                    "Anomaly scores or prepared data are missing after initial steps.")
                # Return error state
                return {"status": "error", "message": "Failed during anomaly detection."}

            # 3. Process each threshold
            for threshold in anomaly_thresholds:
                self.log.info(f"--- Processing Threshold: {threshold:.2f} ---")

                # Identify anomalous pixels *within the valid set* using the threshold
                # self._anomaly_scores_valid has scores for the pixels in self._prepared_data
                # Remember self._anomaly_scores_valid is only for the valid pixels
                # Boolean mask, length = self._total_valid_pixels
                anomaly_mask_valid_flat = self._anomaly_scores_valid < threshold
                # Count anomalies among valid pixels
                anomaly_count = int(np.sum(anomaly_mask_valid_flat))

                # total_pixels_in_image_grid = self._image_shape[0] *
                # self._image_shape[1] if self._image_shape else 0 # Already calculated
                # as self._total_pixels_final_shape

                # Initialize results entry for this threshold
                results[threshold] = {
                    "anomaly_threshold": threshold,
                    "total_pixels_in_image_grid": self._total_pixels_final_shape,
                    "total_valid_pixels_after_masking": self._total_valid_pixels,
                    "anomalous_pixels_count": anomaly_count,
                    "anomaly_percentage_of_total": (anomaly_count / self._total_pixels_final_shape) * 100 if self._total_pixels_final_shape > 0 else 0,
                    "anomaly_percentage_of_valid": (anomaly_count / self._total_valid_pixels) * 100 if self._total_valid_pixels > 0 else 0,
                    "n_clusters_attempted": self.n_clusters,
                    "cluster_method": self.cluster_method,
                    "plot_path": None,  # Will be updated if plot is saved
                    "cluster_stats": []  # List of dicts for cluster stats
                }

                if anomaly_count == 0:
                    self.log.warning(f"No anomalies found for threshold {
                                     threshold:.2f}. Skipping clustering and plotting for this threshold.")
                    continue  # Skip clustering/plotting and move to the next threshold

                self.log.info(f"Found {anomaly_count} anomalous valid pixels for threshold {
                              threshold:.2f}.")

                # Extract the features specifically for the anomalous valid pixels
                # These are the rows in self._prepared_data where anomaly_mask_valid_flat is True
                # Shape (anomaly_count, num_channels)
                anomaly_features = self._prepared_data[anomaly_mask_valid_flat]

                # Cluster anomalous pixels
                cluster_labels, inertia = self._cluster_anomalies(anomaly_features)

                if len(
                        cluster_labels) == 0:  # Should not happen if anomaly_count > 0 and clustering succeeds
                    self.log.warning(f"Clustering returned no labels for threshold {
                                     threshold:.2f}.")
                    continue  # Skip plotting and move to the next threshold

                # Add inertia to results
                results[threshold]["clustering_inertia"] = inertia

                # Create 2D mask for visualization (maps labels back to image grid)
                # anomaly_mask_valid_flat is length = total_valid_pixels
                # cluster_labels is length = anomaly_count
                cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters_found = \
                    self._create_cluster_mask_2d(
                        anomaly_mask_valid_flat, cluster_labels)

                results[threshold]["n_clusters_found"] = n_clusters_found

                # Calculate cluster stats for output and plotting
                cluster_pixels_counts: List[int] = []
                cluster_anomaly_percentages: List[float] = []
                cluster_stats_list: List[Dict[str, Any]] = []

                if n_clusters_found > 0 and anomaly_count > 0:
                    # Ensure cluster indices align correctly (0-based labels to 1-based mask values)
                    # The labels are 0-based and correspond to the order of pixels in anomaly_features
                    # Get actual labels found (e.g., [0, 1, 2])
                    unique_labels = np.unique(cluster_labels)
                    # Exclude noise label if present and sort
                    valid_unique_labels = sorted(
                        [lbl for lbl in unique_labels if lbl >= 0])

                    for label_value in valid_unique_labels:  # Iterate through actual found labels
                        # Find indices in the original anomaly_features array belonging
                        # to this label
                        indices_for_this_label_in_anomaly_features = np.where(
                            cluster_labels == label_value)[0]
                        # Use these indices to get the corresponding indices in the
                        # original valid set
                        indices_in_valid_set = anomalous_indices_in_valid_set[
                            indices_for_this_label_in_anomaly_features]
                        # Get the original flat grid indices
                        original_valid_flat_indices[indices_in_valid_set]

                        # Count how many of these indices are in the 2D cluster mask for this cluster ID (label_value + 1)
                        # This is simpler: just count pixels in the 2D mask equal to the
                        # cluster ID
                        cluster_pixel_count = np.sum(
                            cluster_mask_2d == (label_value + 1))

                        cluster_pixels_counts.append(
                            int(cluster_pixel_count))  # Store as int
                        # Percentage relative to the *total number of anomalies* for
                        # this threshold
                        cluster_percentage_anomalies = (
                            cluster_pixel_count / anomaly_count) * 100 if anomaly_count > 0 else 0
                        cluster_anomaly_percentages.append(
                            float(cluster_percentage_anomalies))  # Store as float
                        self.log.info(f"  Cluster {label_value + 1}: {cluster_pixel_count} pixels ({
                                      cluster_percentage_anomalies:.2f}%) of total anomalies")

                        cluster_stats_list.append({
                            # 1-based index for clarity
                            "cluster_index": int(label_value + 1),
                            "pixel_count": int(cluster_pixel_count),
                            "percentage_of_anomalies": float(cluster_percentage_anomalies)
                        })

                # Add stats to results
                results[threshold]["cluster_stats"] = cluster_stats_list

                # Plot results for this threshold
                plot_path = self._plot_results(
                    anomaly_threshold=threshold,
                    cluster_mask_2d=cluster_mask_2d,
                    cluster_cmap=cluster_cmap,
                    cluster_patches=cluster_patches,
                    n_clusters_found=n_clusters_found,
                    anomalous_pixels_count=anomaly_count,
                    cluster_pixels_counts=cluster_pixels_counts,  # Pass calculated stats
                    cluster_anomaly_percentages=cluster_anomaly_percentages,  # Pass calculated stats
                )
                # Update plot path in results
                results[threshold]["plot_path"] = plot_path

            self.log.info("--- Pipeline Execution Finished Successfully ---")
            results["status"] = "success"
            results["message"] = "Pipeline executed successfully."

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            self.log.critical(f"Pipeline execution failed due to a data or configuration error: {
                              e}", exc_info=True)
            results["status"] = "error"
            results["message"] = f"Pipeline failed due to data or configuration issue: {
                e}"
        except Exception as e:
            self.log.critical(f"An unexpected error occurred during pipeline execution: {
                              e}", exc_info=True)
            results["status"] = "error"
            results["message"] = f"Pipeline failed due to an unexpected error: {e}"

        self.log.info("--- Pipeline Run Method Finished ---")
        return results  # Return the results dictionary


# --- Example Usage Script (Equivalent to your original main) ---
# Save this part as scripts/run_kmeans_pipeline.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SDO/AIA Anomaly Detection Pipeline with KMeans or MiniBatchKMeans"
    )
    parser.add_argument("--data_dir", type=str, default="Data/sdo_data",
                        help="Path to SDO/AIA data directory.")
    # Default channels now match the 9 channels specified
    parser.add_argument("--channels", type=str, nargs='+',
                        default=['94', '131', '171', '193',
                                 '211', '233', '304', '335', '700'],
                        help="AIA channels (e.g., '94', '131'). Default uses all 9 main EUV channels.")
    parser.add_argument("--anomaly_thresholds", type=float, nargs='+',
                        default=[0.1], help="Anomaly threshold(s).")
    parser.add_argument("--output_dir", type=str,
                        default="./output_figures", help="Output directory for figures.")
    # image_size is now optional in the class, but kept as an arg for
    # consistency with original
    parser.add_argument("--image_size", type=int, default=512,
                        help="Resize image size. Use None for original size (Pass None explicitly).")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Isolation Forest contamination parameter.")
    parser.add_argument("--n_clusters", type=int, default=7,
                        help="Number of clusters for clustering.")
    parser.add_argument("--cluster_method", type=str, default="KMeans",
                        choices=["KMeans", "MiniBatchKMeans"], help="Clustering method to use.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    # Removed --no_resize arg as handling None for image_size is the way to do
    # original size

    args = parser.parse_args()

    # Instantiate the pipeline
    pipeline = SolarAnomalyPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        channels=args.channels,
        image_size=args.image_size if args.image_size !=
        - 1 else None,  # Example: use -1 to mean None/original size
        contamination=args.contamination,
        n_clusters=args.n_clusters,
        cluster_method=args.cluster_method,
        random_state=args.random_state
    )

    # Run the pipeline with the specified thresholds
    results = pipeline.run(args.anomaly_thresholds)

    # --- Process the results dictionary ---
    print("\n--- Pipeline Run Summary ---")
    print(f"Status: {results.get('status', 'Unknown')}")
    if results.get("status") == "error":
        print(f"Message: {results.get('message', 'No error message provided.')}")
    else:
        print(f"Data directory: {pipeline.data_dir}")
        print(f"Output directory: {pipeline.output_dir}")
        # Access loaded names
        print(f"Channels processed: {pipeline._loaded_channel_names}")
        print(f"Final image shape: {pipeline._image_shape}")
        print(f"Total valid pixels after masking: {pipeline._total_valid_pixels}")
        print(f"Clustering Method: {pipeline.cluster_method.capitalize()}")
        print(f"N Clusters Attempted: {pipeline.n_clusters}")
        print(f"Random State: {pipeline.random_state}")

        print("\nResults per Anomaly Threshold:")
        for threshold in args.anomaly_thresholds:  # Iterate by requested thresholds
            data = results.get(threshold)  # Get the results for this specific threshold
            if data:  # Check if results exist for this threshold
                print(f"  Threshold {threshold:.2f}:")
                print(f"    Anomalous Pixels: {data.get('anomalous_pixels_count', 'N/A')} / {data.get(
                    'total_pixels_in_image_grid', 'N/A')} ({data.get('anomaly_percentage_of_total', 0.0):.2f}%)")
                n_clusters_found = data.get('n_clusters_found', 'N/A')
                print(f"    Clusters Found: {n_clusters_found}")
                print(f"    Clustering Inertia: {data.get('clustering_inertia', 'N/A'):.2f}" if isinstance(data.get(
                    'clustering_inertia'), (int, float)) else f"    Clustering Inertia: {data.get('clustering_inertia', 'N/A')}")

                if data["cluster_stats"]:
                    print("    Cluster Stats:")
                    for stats in data["cluster_stats"]:
                        print(f"      Cluster {stats.get('cluster_index', 'N/A')}: {stats.get(
                            'pixel_count', 'N/A')} pixels ({stats.get('percentage_of_anomalies', 0.0):.2f}%)")
                else:
                    print("    No cluster stats available (e.g., no anomalies found).")

                plot_path = data.get('plot_path', 'N/A')
                print(f"    Plot saved to: {plot_path}")
            else:
                print(f"  Threshold {threshold:.2f}: No results found (e.g., skipped).")

    print("\n--- Pipeline Run Finished ---")
