
# src/solar/plotting.py

import logging
import os
from typing import List, Optional

import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


def plot_anomaly_clusters(
    masked_data_list: List[np.ndarray],
    channel_names: List[str],
    cluster_mask_2d: np.ndarray,  # The 2D mask on the image grid
    cluster_cmap: matplotlib.colors.ListedColormap,
    cluster_patches: List[mpatches.Patch],
    n_clusters_found: int,
    anomaly_threshold: float,
    output_dir: str,
    total_pixels_image_grid: int,  # Total pixels in the final image shape
    anomalous_pixels_count: int,  # Total anomalous pixels for this threshold
    cluster_pixels_counts: Optional[List[int]] = None,  # Pixels in each cluster
    # % of total anomalies in each cluster
    cluster_anomaly_percentages: Optional[List[float]] = None,
    clustering_method_name: str = "Clustering",
):
    """Generates and saves a multi-panel plot showing anomaly clusters on each channel.

    Args:
        masked_data_list: List of masked image data arrays (same size).
        channel_names: List of channel wavelengths corresponding to masked_data_list.
        cluster_mask_2d: A 2D numpy array on the image grid where values > 0
                         represent cluster assignments (1 to N).
        cluster_cmap: Colormap for clusters.
        cluster_patches: Legend patches for clusters.
        n_clusters_found: Number of clusters found.
        anomaly_threshold: The anomaly score threshold used.
        output_dir: Directory to save the plot.
        total_pixels_image_grid: Total number of pixels in the image grid (H * W).
        anomalous_pixels_count: Total number of pixels identified as anomalous.
        cluster_pixels_counts: Optional list of pixel counts for each cluster.
        cluster_anomaly_percentages: Optional list of percentage of total anomalies for each cluster.
        clustering_method_name: Name of the clustering method (e.g., "K-Means", "GMM").
    """
    if not masked_data_list or not channel_names or len(
            masked_data_list) != len(channel_names):
        log.warning("Invalid input lists for plotting.")
        return

    num_channels = len(channel_names)
    # Determine grid size (e.g., 3x3 for up to 9 channels)
    num_cols = 3
    num_rows = int(np.ceil(num_channels / num_cols))
    if num_rows == 0:
        num_rows = 1  # Handle case with 0 channels (shouldn't happen if we plot)

    fig, axes = plt.subplots(num_rows,
                             num_cols,
                             figsize=(num_cols * 5,
                                      num_rows * 4.5),
                             dpi=120,  # Increased DPI for better quality
                             squeeze=False)  # squeeze=False ensures axes is always 2D
    axes = axes.flatten()  # Flatten to easily iterate

    # --- Figure Suptitle (Main Title) ---
    anomaly_percentage_total = (anomalous_pixels_count / total_pixels_image_grid) * 100 \
        if total_pixels_image_grid > 0 else 0
    suptitle_text = (f'{clustering_method_name} Anomaly Clusters in SDO/AIA EUV Channels\n'
                     f'Threshold: {anomaly_threshold:.2f} | Anomalous Pixels: {anomalous_pixels_count}/{total_pixels_image_grid} ({anomaly_percentage_total:.2f}%)')
    fig.suptitle(suptitle_text, fontsize=16, y=0.99)  # Adjusted y position

    # --- Determine Global Vmin/Vmax for Consistent Color Scaling ---
    # Use percentiles across ALL data in all channels
    all_valid_pixels = np.concatenate([
        data[~np.isnan(data)].flatten() for data in masked_data_list if np.any(~np.isnan(data))
    ])
    vmin_global = np.percentile(all_valid_pixels, 2) if len(all_valid_pixels) > 0 else 0
    vmax_global = np.percentile(all_valid_pixels, 98) if len(
        all_valid_pixels) > 0 else 1
    # Handle cases where percentiles might be the same (flat image) or invalid
    if vmin_global >= vmax_global:
        vmin_global, vmax_global = np.nanmin(
            all_valid_pixels), np.nanmax(all_valid_pixels)
        if vmin_global >= vmax_global:  # Still flat or all NaNs
            vmin_global, vmax_global = 0, 1

    # --- Plot Each Channel ---
    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        # Stop if we run out of axes (shouldn't happen with correct grid size)
        if i >= num_rows * num_cols:
            break

        ax = axes[i]
        # Display the base image with global color scaling
        ax.imshow(masked_data, cmap='YlOrBr', origin='lower',  # Original colormap
                  vmin=vmin_global, vmax=vmax_global, alpha=0.6)  # Use global scaling and alpha

        # Overlay clusters only if there are clusters found
        if n_clusters_found > 0 and cluster_mask_2d.shape == masked_data.shape and np.any(
                cluster_mask_2d > 0):
            # Mask the 2D cluster mask to only show cluster areas (where value > 0)
            cluster_overlay_data = np.ma.masked_where(
                cluster_mask_2d == 0, cluster_mask_2d)
            ax.imshow(cluster_overlay_data, cmap=cluster_cmap, alpha=0.8, origin='lower',  # Slightly increased alpha for clusters
                      interpolation='nearest',  # Use nearest for masks
                      # Set vmin/vmax to span cluster labels (1 to N)
                      vmin=1, vmax=n_clusters_found)
        elif np.any(cluster_mask_2d > 0) and cluster_mask_2d.shape != masked_data.shape:
            log.warning(f"Cluster mask shape {cluster_mask_2d.shape} does not match image shape {
                        masked_data.shape} for channel {channel}. Skipping cluster overlay.")

        # --- Subplot Title (Channel + Cluster Info) ---
        title_lines = [f'AIA {channel} Å']

        # Add cluster stats to title if provided. We need to find the stats for the *current* cluster ID.
        # A simpler way is to loop through the *found* cluster labels (1 to n_clusters_found)
        # and look up their corresponding stats if available.
        # The stats lists `cluster_pixels_counts` and `cluster_anomaly_percentages`
        # are assumed to be ordered by the original 0-based cluster labels.
        # The patch labels usually correspond to unique sorted original labels.
        # For simplicity, let's just add the total number of clusters found to the subplot title
        # and put the detailed cluster stats in the main title or legend if space allows.
        # The request was to add it to the subplot title, so let's try that multiline approach again,
        # ensuring we match the stats to the correct cluster index (1-based).
        # We need the labels that actually exist in `cluster_mask_2d`.
        actual_cluster_ids_in_mask = np.unique(
            cluster_mask_2d[cluster_mask_2d > 0])  # 1-based IDs
        if len(actual_cluster_ids_in_mask) > 0:
            # Just the count of found clusters
            title_lines.append(f'Clusters Found: {len(actual_cluster_ids_in_mask)}')

        # Let's add the per-cluster stats to the subplot title, but only if it fits
        # and make sure we handle the mapping from mask ID (1..N) to stat index (0..N-1)
        # This requires the stats lists to be ordered by cluster ID (0 to N-1).
        if cluster_pixels_counts is not None and cluster_anomaly_percentages is not None \
           and len(cluster_pixels_counts) == n_clusters_found \
           and len(cluster_anomaly_percentages) == n_clusters_found:
            # Assuming the stats lists correspond to original cluster labels 0..N-1
            # And cluster_mask_2d uses 1..N-1
            # The patches are also assumed to correspond to original labels
            # Use patches to get labels (e.g., 'Cluster 1')
            for patch in cluster_patches:
                try:
                    # Extract the cluster number from the label (e.g., 'Cluster 1' -> 1)
                    cluster_id = int(patch.get_label().split(' ')[1])
                    # Find the original label index (0-based) for this ID
                    original_label_index = cluster_id - 1  # Assuming they are sequential 1..N

                    if 0 <= original_label_index < n_clusters_found:
                        pixels = cluster_pixels_counts[original_label_index]
                        percentage = cluster_anomaly_percentages[original_label_index]
                        # Only add if there are pixels in this cluster
                        if pixels > 0:
                            # More concise format
                            title_lines.append(f'C{cluster_id}: {
                                               pixels} Pix ({percentage:.1f}%)')
                except Exception as e:
                    log.debug(f"Error processing cluster patch label {
                              patch.get_label()}: {e}")
                    # Fallback to just adding the label if stats can't be parsed
                    title_lines.append(patch.get_label())

        # Limit the number of lines in the title to prevent excessive length
        max_title_lines = 5  # Limit total lines including channel name
        if len(title_lines) > max_title_lines:
            title_lines = title_lines[:max_title_lines - 1] + [
                # Add ellipsis line
                f'... +{len(title_lines) - max_title_lines + 1} more']

        ax.set_title(
            "\n".join(title_lines),  # Multiline title
            color='black', fontsize=10, pad=5  # Adjusted fontsize and padding again
        )
        ax.axis('off')  # Hide axes ticks and labels

    # --- Turn Off Unused Subplots ---
    for j in range(num_channels, num_rows * num_cols):
        axes[j].axis('off')

    # --- Add Legend ---
    # Legend is now outside the main tight_layout area
    if cluster_patches:  # Only add legend if there are clusters to show
        # Filter patches to only include those that actually have pixels assigned in the mask
        # (i.e., cluster_pixels_counts[patch_cluster_id - 1] > 0)
        valid_patches = []
        if cluster_pixels_counts is not None and len(
                cluster_pixels_counts) == n_clusters_found:
            for patch in cluster_patches:
                try:
                    cluster_id = int(patch.get_label().split(' ')[1])
                    original_label_index = cluster_id - 1
                    if 0 <= original_label_index < n_clusters_found and cluster_pixels_counts[
                            original_label_index] > 0:
                        valid_patches.append(patch)
                except Exception as e:
                    log.debug(f"Error filtering patch {patch.get_label()}: {e}")
                    valid_patches.append(patch)  # Include if parsing fails

        else:  # If counts not available, include all patches
            valid_patches = cluster_patches

        if valid_patches:
            fig.legend(handles=valid_patches, loc='upper right',
                       bbox_to_anchor=(0.98, 0.95),  # Position of the legend
                       fontsize='small', frameon=True,  # Add frame for clarity
                       framealpha=0.9, title="Anomaly Clusters")
        else:
            log.info("No clusters with pixels assigned, skipping legend.")

    # --- Final Layout and Saving ---
    # Adjust layout to make space for legend on the right
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    # Use a descriptive filename including cluster method, threshold, and k
    filename = os.path.join(
        output_dir,
        f"{clustering_method_name.lower().replace(' ', '_')}_anomalies_threshold_{anomaly_threshold:.2f}_k{
            n_clusters_found}_clusters.png"  # Use method name and found k in filename
    )
    try:
        # Increased dpi for better quality
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        log.info(f"Result plot saved to: {filename}")
    except Exception as e:
        log.error(f"Error saving figure {filename}: {e}", exc_info=True)
        # In an API, you might want to raise an exception here

    plt.close(fig)
