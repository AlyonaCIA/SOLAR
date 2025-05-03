import os
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np


def crop_with_margin(masked_data, cluster_mask=None, margin_fraction=0.1):
    """Recorta la imagen y la máscara de clusters dejando un margen del 10% de píxeles no válidos."""
    # Encuentra las filas y columnas que contienen valores no NaN
    valid_rows = ~np.isnan(masked_data).all(axis=1)  # Fila no completamente NaN
    valid_cols = ~np.isnan(masked_data).all(axis=0)  # Columna no completamente NaN
    
    # Si se proporciona la máscara de clusters, no recortamos las zonas del cluster
    if cluster_mask is not None:
        valid_rows = valid_rows | np.any(cluster_mask, axis=1)
        valid_cols = valid_cols | np.any(cluster_mask, axis=0)

    # Encuentra los índices de la primera y última fila válida
    min_row, max_row = np.where(valid_rows)[0][[0, -1]]
    
    # Encuentra los índices de la primera y última columna válida
    min_col, max_col = np.where(valid_cols)[0][[0, -1]]
    
    # Calculamos el margen a recortar (10%)
    row_margin = int((max_row - min_row + 1) * margin_fraction)
    col_margin = int((max_col - min_col + 1) * margin_fraction)

    # Aplicamos el margen
    min_row = max(min_row - row_margin, 0)
    max_row = min(max_row + row_margin, masked_data.shape[0] - 1)
    min_col = max(min_col - col_margin, 0)
    max_col = min(max_col + col_margin, masked_data.shape[1] - 1)

    # Recortar la imagen según estos índices
    cropped_data = masked_data[min_row:max_row+1, min_col:max_col+1]
    
    # Recortar también la máscara del cluster
    cropped_cluster_mask = cluster_mask[min_row:max_row+1, min_col:max_col+1] if cluster_mask is not None else None
    
    return cropped_data, cropped_cluster_mask, (min_row, max_row, min_col, max_col)


def plot_results(
    masked_data_list: list,
    cluster_mask_global: np.ndarray,
    cluster_cmap_global: matplotlib.colors.ListedColormap,
    n_clusters_global: int,
    cluster_patches_global: list,
    channel_names: list,
    anomaly_threshold: float,
    output_dir: str,
    total_pixels_resized: int,
    anomaly_pixels_count: int,
    file_type: str,
    clustering_method_name: str = "K-Means",
):
    """Plots and saves anomaly detection and clustering results."""
    if not masked_data_list:
        print("No data to plot.")
        return

    # Determine plot grid size dynamically (e.g., up to 3x3)
    num_images = len(masked_data_list)
    num_cols = min(3, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=100, squeeze=False)  # Use squeeze=False for consistent 2D array
    axes = axes.flatten()

    anomaly_percentage = (anomaly_pixels_count / total_pixels_resized) * 100 if total_pixels_resized > 0 else 0
    fig.suptitle(
        f'{clustering_method_name} Anomaly Clusters in SDO/AIA {file_type.upper()} Channels\n'
        f'Anomaly Threshold: {anomaly_threshold:.2f} | Anomalous Pixels (resized): {anomaly_pixels_count}/{total_pixels_resized} ({anomaly_percentage:.2f}%)',
        fontsize=16, y=0.98 if num_rows > 1 else 1.02  # Adjust title position
    )

    base_cmap_name = 'sdoaia{channel}'  # Base colormap name template
    fallback_cmap = 'viridis'  # Fallback if specific AIA map isn't found

    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        if i >= len(axes): continue  # Should not happen with dynamic grid, but safe check

        ax = axes[i]

        # Recortar la imagen y la máscara del cluster con margen
        cropped_data, cropped_cluster_mask, _ = crop_with_margin(masked_data, cluster_mask_global)

        # Try channel-specific colormap, else fallback
        try:
            cmap_name = base_cmap_name.format(channel=channel)
            img_cmap = plt.get_cmap(cmap_name)
        except ValueError:
            print(f"Colormap {cmap_name} not found, using {fallback_cmap}.")
            img_cmap = fallback_cmap

        # Plot base image (masked data)
        # Handle cases where all data might be NaN
        valid_data = cropped_data[~np.isnan(cropped_data)]
        if valid_data.size > 0:
             vmin = np.percentile(valid_data, 2)
             vmax = np.percentile(valid_data, 98)
        else:
             vmin, vmax = 0, 1  # Default if no valid data

        ax.imshow(
            cropped_data, cmap=img_cmap, origin='lower',
            vmin=vmin, vmax=vmax,
            alpha=0.6  # Slightly more transparent base
        )

        # Overlay clusters if they exist
        if n_clusters_global > 0 and cropped_cluster_mask is not None:
            # Plot each cluster individually to control color and legend precisely
            for cluster_index in range(1, n_clusters_global + 1):
                cluster_area_mask = (cropped_cluster_mask == cluster_index)
                if np.any(cluster_area_mask):  # Only plot if pixels exist for this cluster
                    # Use the provided global cmap, ensure index is correct
                    # cmap expects normalized value 0..1
                    cluster_color_norm = (cluster_index - 1) / (n_clusters_global - 1 if n_clusters_global > 1 else 1)
                    cluster_color = cluster_cmap_global(cluster_color_norm)
                    # Create a single-color map for this cluster
                    single_color_cmap = matplotlib.colors.ListedColormap([cluster_color])

                    # Mask everything *except* the current cluster
                    overlay = np.ma.masked_where(~cluster_area_mask, cropped_cluster_mask)
                    ax.imshow(
                        overlay,
                        cmap=single_color_cmap,  # Use the single color map
                        origin='lower',
                        alpha=0.8,  # Make clusters slightly more opaque
                        vmin=cluster_index - 0.5, vmax=cluster_index + 0.5  # Center vmin/vmax on index
                    )

        # Set title
        title_lines = [f'AIA {channel} Å']
        ax.set_title("\n".join(title_lines), color='black', fontsize=12, pad=5)  # Slightly smaller font
        ax.axis('off')

    # Add legend if clusters were found
    if cluster_patches_global:
        # Place legend outside the plot area to avoid overlap
        fig.legend(
            handles=cluster_patches_global,
            loc='center right',  # Position relative to the figure
            bbox_to_anchor=(1.0, 0.5),  # Adjust anchor to be outside
            fontsize='medium',  # Slightly larger legend font
            framealpha=0.9
        )

    # Remove empty subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust rect to make space for legend if needed
    filename = os.path.join(
        output_dir, f"{file_type}_kmeans_anomaly_detection_thresh_{anomaly_threshold:.2f}.png"  # Updated filename
    )
    plt.savefig(filename, bbox_inches='tight', dpi=150)  # Increase DPI slightly
    plt.close(fig)
    print(f"Figure saved to: {filename}")
