import os

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


def plot_results(
    masked_data_list: list,
    cluster_mask_global: np.ndarray,
    cluster_cmap_global: matplotlib.colors.ListedColormap,
    n_clusters_global: int,
    cluster_patches_global: list,
    channel_names: list,
    anomaly_threshold: float,
    output_dir: str
):
    """Plots and saves the results, overlaying global clusters on each channel.

    Args:
        masked_data_list (list): List of masked image data arrays.
        cluster_mask_global (np.ndarray): Global cluster mask.
        cluster_cmap_global (matplotlib.colors.ListedColormap): Colormap for clusters.
        n_clusters_global (int): Number of global clusters.
        cluster_patches_global (list): Legend patches for global clusters.
        channel_names (list): List of channel names (wavelengths).
        anomaly_threshold (float): Anomaly threshold used.
        output_dir (str): Directory to save the output figure.

    Returns:
        None: The function saves the results as an image file in the specified
        directory.
    """
    num_rows, num_cols = 3, 3
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(18, 15), dpi=100
    )
    axes = axes.flatten()

    fig.suptitle(
        f'Anomaly Detection in SDO/AIA EUV Channels (K-Means)\n'
        f'Anomaly Threshold: {anomaly_threshold:.2f}',
        fontsize=18
    )

    for i, (masked_data, channel) in enumerate(
        zip(masked_data_list, channel_names)
    ):
        if i < num_rows * num_cols:
            ax = axes[i]

            ax.imshow(
                masked_data,
                origin='lower',
                cmap='YlOrBr',
                vmin=np.nanpercentile(masked_data, 2),
                vmax=np.nanpercentile(masked_data, 98),
                alpha=0.5
            )

            if n_clusters_global > 0:
                for cluster_index in range(1, n_clusters_global + 1):
                    cluster_area_mask = cluster_mask_global == cluster_index
                    cluster_color = cluster_cmap_global(
                        (cluster_index - 1) / n_clusters_global
                    )
                    ax.imshow(
                        np.ma.masked_where(
                            ~cluster_area_mask, cluster_mask_global
                        ),
                        origin='lower',
                        cmap=matplotlib.colors.ListedColormap([cluster_color]),
                        alpha=0.6,
                        vmin=cluster_index - 0.5,
                        vmax=cluster_index + 0.5
                    )

            ax.set_title(
                f'AIA {channel} Å (Clusters: {n_clusters_global})',
                color='black',
                fontsize=14,
                pad=10
            )
            ax.text(
                0.5, -0.18,
                f'Anomaly Threshold: {anomaly_threshold:.2f}',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=12,
                color='dimgray'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Move the legend here to display in each subplot
            if cluster_patches_global:
                ax.legend(
                    handles=cluster_patches_global,
                    loc='upper right',
                    fontsize='small',
                    frameon=True,
                    handlelength=1,
                    borderpad=0.3
                )

    # Remove unused axes
    for j in range(len(channel_names), num_rows * num_cols):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    filename = os.path.join(
        output_dir,
        f"kmeans_anomaly_detection_threshold_{anomaly_threshold:.2f}"
        "_global_clusters.png"
    )
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {filename}")
