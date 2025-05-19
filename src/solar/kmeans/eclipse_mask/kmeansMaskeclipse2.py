import argparse
import os
from typing import List, Tuple, Optional, Dict

import matplotlib
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import imageio.v2 as imageio
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

matplotlib.use('Agg')  # Use Agg backend for saving files


# --- Helper Functions ---

# --- FITS Specific ---
def load_fits_data(channel_dir: str) -> Tuple[np.ndarray, dict]:
    """Loads FITS data and metadata from a single channel directory."""
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith((".fits", ".fit"))]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in: {channel_dir}")
    fits_path = os.path.join(channel_dir, fits_files[0])
    print(f"Loading FITS: {fits_path}")
    aia_map = sunpy.map.Map(fits_path)
    print(f"  Loaded FITS data shape: {aia_map.data.shape}")
    return aia_map.data, aia_map.meta

# *** FUNCIÓN DE MÁSCARA DE OCULTACIÓN PARA FITS ***
def create_occulting_mask_fits(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates an occulting mask (hides solar disk) based on FITS metadata."""
    if data is None or metadata is None:
        raise ValueError("Data and metadata must be provided for FITS mask creation.")
    ny, nx = data.shape
    cdelt1 = metadata.get("cdelt1", 0.0)

    if abs(cdelt1) < 1e-9 :
        print("Warning: CDELT1 missing or zero. Falling back to estimated radius for occulting mask.")
        solar_radius_pixels = int(min(nx, ny) * 0.48)
        x_center, y_center = nx / 2.0, ny / 2.0
    else:
        solar_radius_arcsec = metadata.get("rsun_obs", 960.0)
        solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))
        crpix1 = metadata.get('crpix1', (nx + 1) / 2.0 )
        crpix2 = metadata.get('crpix2', (ny + 1) / 2.0 )
        x_center = crpix1 - 1
        y_center = crpix2 - 1

    print(f"Creating FITS occulting mask: Center=({x_center:.1f}, {y_center:.1f}), Radius={solar_radius_pixels} pix")
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    # --- Lógica invertida: selecciona píxeles FUERA del radio ---
    mask = distance_from_center > solar_radius_pixels
    # -------------------------------------------------------
    print(f"  Generated FITS occulting mask shape: {mask.shape}, Sum (True pixels): {np.sum(mask)}")
    return mask

# --- JP2 Specific ---
def load_jp2_data_imageio(channel_dir: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Loads JP2 data using Imageio from a single channel directory.

    Metadata will always be None.
    """
    jp2_files = [f for f in os.listdir(channel_dir) if f.endswith(".jp2")]
    if not jp2_files:
        print(f"Warning: No JP2 files found in: {channel_dir}")
        return None, None
    jp2_path = os.path.join(channel_dir, jp2_files[0])
    print(f"Attempting to load JP2 with Imageio: {jp2_path}")
    data = None
    try:
        data = imageio.imread(jp2_path)
        print(f"  Loaded JP2 data shape: {data.shape}, dtype: {data.dtype}")
    except Exception as e_imgio:
        print(f"Imageio failed to load {jp2_path}: {e_imgio}")
        return None, None
    return data, None

# *** FUNCIÓN DE MÁSCARA DE OCULTACIÓN PARA JP2 ***
def create_occulting_mask_jp2(data: np.ndarray, fixed_radius_pixels: int) -> np.ndarray:
    """Creates an occulting mask (hides solar disk) for JP2 images using a fixed
    radius."""
    if data is None:
        raise ValueError("Input data cannot be None for JP2 mask creation.")
    ny, nx = data.shape
    print(f"Creating JP2 occulting mask for image size {ny}x{nx} using fixed radius: {fixed_radius_pixels}")
    x_center, y_center = nx // 2, ny // 2
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center)**2 + (y - y_center)**2)
    # --- Lógica invertida: selecciona píxeles FUERA del radio ---
    mask = distance_from_center > fixed_radius_pixels
    # -------------------------------------------------------
    print(f"  Generated JP2 occulting mask shape: {mask.shape}, Sum (True pixels): {np.sum(mask)}")
    return mask

# --- Common Preprocessing ---
def preprocess_image(
    data: np.ndarray, mask: np.ndarray, size: int = 512
) -> np.ndarray:
    """Resizes the image and the mask, then applies the resized mask.

    Assumes mask=True indicates pixels TO KEEP.
    """
    if data is None or mask is None:
        raise ValueError("Data and mask must be provided for preprocessing.")
    print(f"Preprocessing: Resizing data ({data.shape}) and mask ({mask.shape}) to {size}x{size}")
    resized_data = resize(data, (size, size), mode='reflect', anti_aliasing=True)
    resized_mask = resize(mask.astype(float), (size, size), mode='reflect', anti_aliasing=False) > 0.5
    print(f"  Resized mask shape: {resized_mask.shape}, Sum (Pixels to keep): {np.sum(resized_mask)}")

    masked_data = resized_data.copy()
    # Pone NaN donde la máscara redimensionada es False (DENTRO del disco en este caso)
    masked_data[~resized_mask] = np.nan
    print(f"  Final masked data shape: {masked_data.shape}, Non-NaN count (Pixels kept): {np.sum(~np.isnan(masked_data))}")
    return masked_data


# --- Data Preparation --- (Sin cambios)
def prepare_data_concatenated(
    masked_data_list: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data, handles NaNs, and scales the data."""
    stacked_data = np.stack(masked_data_list, axis=-1)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    nan_mask = np.isnan(reshaped_data).any(axis=1)
    if np.all(nan_mask):
        print("Warning: All pixels are NaN after concatenation. Cannot scale.")
        return np.array([]), np.array([]), nan_mask
    cleaned_data = reshaped_data[~nan_mask]
    if cleaned_data.shape[0] == 0:
         print("Warning: No valid (non-NaN) pixels left after masking.")
         return np.array([]), np.array([]), nan_mask
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask


# --- Anomaly Detection --- (Sin cambios)
def detect_anomalies_isolation_forest(
    data: np.ndarray, contamination: float
) -> np.ndarray:
    """Detects anomalies using Isolation Forest."""
    if data.shape[0] == 0:
        print("Warning: Cannot detect anomalies on empty dataset.")
        return np.array([])
    iso_forest = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1
    )
    print(f"Fitting Isolation Forest (contamination={contamination}) on data shape: {data.shape}")
    iso_forest.fit(data)
    return iso_forest.decision_function(data)


# --- Clustering --- (Sin cambios lógicos)
def perform_kmeans_clustering(
    data: np.ndarray, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """Performs K-Means clustering."""
    if data.shape[0] == 0:
        print("Warning: Cannot perform K-Means on empty dataset.")
        return np.array([]), np.inf
    # Asegurarse de que hay suficientes muestras para los clusters
    if data.shape[0] < n_clusters:
        print(f"Warning: Number of samples ({data.shape[0]}) is less than n_clusters ({n_clusters}). Reducing n_clusters.")
        n_clusters = data.shape[0]
    # Evitar n_clusters=0 si data.shape[0] es 0 (ya manejado arriba) o 1
    if n_clusters == 0:
        print("Warning: n_clusters is zero. Cannot perform K-Means.")
        return np.array([]), np.inf

    print(f"Performing K-Means (k={n_clusters}) on data shape: {data.shape}")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'
    )
    kmeans.fit(data)
    print(f"  K-Means inertia: {kmeans.inertia_:.2f}")
    return kmeans.labels_, kmeans.inertia_

def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray,
    image_size: int
) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels."""
    print("-" * 20)
    print("Inside create_cluster_mask:")
    if anomaly_mask is None or labels is None or valid_pixel_mask is None:
         print("Warning: Inputs to create_cluster_mask are invalid. Returning empty cluster map.")
         return np.zeros((image_size, image_size), dtype=int), matplotlib.colors.ListedColormap([]), [], 0

    print(f"Input anomaly_mask shape: {anomaly_mask.shape} (Sum: {np.sum(anomaly_mask)})")
    print(f"Input labels length: {len(labels)}")
    print(f"Input valid_pixel_mask shape: {valid_pixel_mask.shape} (Sum: {np.sum(valid_pixel_mask)})")

    cluster_mask_2d = np.zeros((image_size, image_size), dtype=int)
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])
    cluster_patches = []

    if len(labels) > 0 and np.any(anomaly_mask):
        valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))
        valid_and_anomalous_indices_2d = np.argwhere(valid_pixel_mask_2d & anomaly_mask)

        if len(valid_and_anomalous_indices_2d) != len(labels):
             print(f"Warning: Mismatch between valid+anomalous pixels ({len(valid_and_anomalous_indices_2d)}) and labels ({len(labels)}).")
             # Intentar continuar solo si hay indices válidos
             if len(valid_and_anomalous_indices_2d) == 0:
                  return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters
             # Tomar el mínimo de las longitudes para evitar errores de índice
             min_len = min(len(valid_and_anomalous_indices_2d), len(labels))
             valid_and_anomalous_indices_2d = valid_and_anomalous_indices_2d[:min_len]
             labels = labels[:min_len]


        # Proceder solo si todavía tenemos etiquetas después del ajuste
        if len(labels) > 0:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            print(f"Number of unique cluster labels found: {n_clusters}")

            cluster_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']
            if n_clusters > len(cluster_colors):
                print(f"Warning: Number of clusters ({n_clusters}) exceeds defined colors. Repeating colors.")
                cluster_colors = (cluster_colors * (n_clusters // len(cluster_colors) + 1))[:n_clusters]

            cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors)

            # Asegurarse de que las etiquetas correspondan a los clusters reales
            label_map = {label: i for i, label in enumerate(unique_labels)}
            mapped_labels = np.array([label_map[l] for l in labels])

            # Asignar etiquetas mapeadas + 1 a las posiciones 2D
            cluster_mask_2d[tuple(valid_and_anomalous_indices_2d.T)] = mapped_labels + 1

            # Crear parches de leyenda para los clusters mapeados
            for cluster_idx in range(n_clusters): # Iterar sobre los clusters mapeados 0 a n_clusters-1
                cluster_color = cluster_cmap(cluster_idx / (n_clusters -1 if n_clusters > 1 else 1))
                cluster_patches.append(
                    mpatches.Patch(
                        color=cluster_color,
                        label=f'Cluster {cluster_idx + 1}' # Etiqueta 1 a n_clusters
                    )
                )
        else:
             print("No labels remaining after index mismatch adjustment.")

    else:
        print("No anomaly labels or no anomalous pixels in mask. No clusters to map.")


    print(f"Final cluster_mask_2d shape: {cluster_mask_2d.shape}, Max value: {np.max(cluster_mask_2d)}")
    print("-" * 20)
    return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters


# --- Plotting --- (Título y nombre de archivo actualizados)
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
    cluster_pixels_counts: List[int],
    cluster_anomaly_percentages: List[float],
    file_type: str,
    mask_type: str, # Añadido para título/nombre archivo
    clustering_method_name: str = "K-Means",
):
    """Plots and saves anomaly detection and clustering results."""
    if not masked_data_list:
        print("No data to plot.")
        return

    num_images = len(masked_data_list)
    num_cols = min(3, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows), dpi=100, squeeze=False)
    axes = axes.flatten()

    anomaly_percentage = (anomaly_pixels_count / total_pixels_resized) * 100 if total_pixels_resized > 0 else 0
    # *** Título actualizado ***
    mask_desc = "Off-Limb" if mask_type == 'occulting' else "On-Disk"
    fig.suptitle(
        f'{clustering_method_name} {mask_desc} Anomaly Clusters in SDO/AIA {file_type.upper()} Channels\n'
        f'Anomaly Threshold: {anomaly_threshold:.2f} | Anomalous Pixels (valid area): {
            anomaly_pixels_count}/{total_pixels_resized} ({anomaly_percentage:.2f}%)',
        fontsize=16, y=0.98 if num_rows > 1 else 1.02
    )

    base_cmap_name = 'sdoaia{channel}'
    fallback_cmap = 'viridis'

    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        if i >= len(axes): continue
        ax = axes[i]
        try:
            cmap_name = base_cmap_name.format(channel=channel)
            img_cmap = plt.get_cmap(cmap_name)
        except ValueError:
            print(f"Colormap {cmap_name} not found, using {fallback_cmap}.")
            img_cmap = fallback_cmap

        valid_data = masked_data[~np.isnan(masked_data)]
        # Ajustar Vmin/Vmax para datos fuera del disco (pueden ser más tenues)
        if valid_data.size > 0:
             vmin = np.percentile(valid_data, 5 if mask_type == 'occulting' else 2) # Percentil más alto para off-limb
             vmax = np.percentile(valid_data, 98)
        else:
             vmin, vmax = 0, 1

        ax.imshow(
            masked_data, cmap=img_cmap, origin='lower',
            vmin=vmin, vmax=vmax,
            alpha=0.6
        )

        if n_clusters_global > 0 and cluster_mask_global is not None and cluster_mask_global.shape == masked_data.shape:
             for cluster_index in range(1, n_clusters_global + 1):
                 cluster_area_mask = (cluster_mask_global == cluster_index)
                 if np.any(cluster_area_mask):
                     cluster_color_norm = (cluster_index - 1) / (n_clusters_global -1 if n_clusters_global > 1 else 1)
                     cluster_color = cluster_cmap_global(cluster_color_norm)
                     single_color_cmap = matplotlib.colors.ListedColormap([cluster_color])
                     overlay = np.ma.masked_where(~cluster_area_mask, cluster_mask_global)
                     ax.imshow(
                         overlay,
                         cmap=single_color_cmap,
                         origin='lower',
                         alpha=0.8,
                         vmin=cluster_index - 0.5, vmax=cluster_index + 0.5
                     )

        title_lines = [f'AIA {channel} Å']
        ax.set_title("\n".join(title_lines), color='black', fontsize=12, pad=5)
        ax.axis('off')

    if cluster_patches_global:
        fig.legend(
            handles=cluster_patches_global,
            loc='center right',
            bbox_to_anchor=(1.0, 0.5),
            fontsize='medium',
            framealpha=0.9
        )

    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    # *** Nombre de archivo actualizado ***
    filename = os.path.join(
        output_dir, f"{file_type}_{mask_type}_kmeans_thresh_{anomaly_threshold:.2f}.png"
    )
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {filename}")


# --- Main Execution ---
def main(cli_args=None):
    """Main function to execute SDO/AIA anomaly detection pipeline.

    Returns data for plotting if called as a module.
    """
    if cli_args:
        args = cli_args
    else:
        parser = argparse.ArgumentParser(
            description="SDO/AIA Anomaly Detection using Isolation Forest and K-Means"
        )
        parser.add_argument("--data_dir", type=str, required=True,
                            help="Path to parent directory containing AIA channel subdirectories.")
        parser.add_argument("--output_dir", type=str, default="./output_analysis_data",
                            help="Output directory for data (if any saved by script, e.g. logs).")
        parser.add_argument("--file_type", type=str, default="jp2", choices=['fits', 'jp2'],
                            help="Type of input image files ('fits' or 'jp2').")
        parser.add_argument("--mask_type", type=str, default="occulting", choices=['disk', 'occulting'],
                            help="Type of mask: 'disk' (keep disk) or 'occulting' (hide disk).")
        parser.add_argument("--channels", type=str, nargs='+', default=None,
                            help="AIA channels numbers (e.g., '94' '131' '171'). If None, uses default set.")
        parser.add_argument("--image_size", type=int, default=512,
                            help="Resize image size (square).")
        parser.add_argument("--jp2_mask_radius", type=int, default=1600,
                            help="Fixed radius in pixels for JP2 mask. Assumes 4096 original size.")
        parser.add_argument("--anomaly_thresholds", type=float, nargs='+', default=[-0.1, 0.0, 0.1],
                            help="Anomaly score threshold(s) for Isolation Forest.")
        parser.add_argument("--contamination", type=float, default=0.05,
                            help="Estimated proportion of anomalies for Isolation Forest.")
        parser.add_argument("--n_clusters", type=int, default=5,
                            help="Number of clusters for K-Means.")
        parser.add_argument("--random_state", type=int, default=42,
                            help="Random seed for reproducibility.")
        args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Debug prints for paths ---
    print(f"  DEBUG main: Current working directory: {os.getcwd()}")
    print(f"  DEBUG main: args.data_dir: {args.data_dir}")
    # print(f"  DEBUG main: Absolute path of args.data_dir: {os.path.abspath(args.data_dir)}")
    if not os.path.isdir(os.path.abspath(args.data_dir)):
        print(f"  CRITICAL ERROR main: args.data_dir IS NOT A VALID DIRECTORY: {os.path.abspath(args.data_dir)}")
        if cli_args: return None
        return

    # --- Determinar Canales ---
    default_channels_num = ['94', '131', '171', '193', '211', '304', '335']
    if args.channels:
        channels_to_process = [f"aia_{c}" for c in args.channels]
    else:
        channels_to_process = [f"aia_{c}" for c in default_channels_num]

    print(f"Processing file type: {args.file_type.upper()}")
    print(f"Using mask type: {args.mask_type.upper()}")
    print(f"Target channels: {', '.join(c.split('_')[1] for c in channels_to_process)}")

    # --- Cargar, Enmascarar y Preprocesar Datos ---
    masked_data_list, channel_names_loaded = [], []
    original_shapes = {} # If you need this later
    metadata_list = {}   # If you need this later

    # ********************************************************************
    # ***** THIS IS THE CRITICAL DATA LOADING LOOP THAT WAS MISSING ******
    # ********************************************************************
    for channel_dir_name in channels_to_process:
        channel_num = channel_dir_name.split("_")[1]
        channel_path = os.path.join(args.data_dir, channel_dir_name)

        print(f"\n  PROCESSING CHANNEL: {channel_num}")
        print(f"    Constructed channel_path: '{channel_path}'")
        print(f"    Absolute channel_path: '{os.path.abspath(channel_path)}'")

        if not os.path.isdir(channel_path):
            print(f"    WARNING: Directory not found for channel {channel_num}: '{channel_path}'. Skipping.")
            continue

        data, mask = None, None
        current_metadata = None
        try:
            # Cargar datos
            if args.file_type == 'fits':
                print(f"    Attempting to load FITS for {channel_num} from '{channel_path}'")
                data, current_metadata = load_fits_data(channel_path)
            elif args.file_type == 'jp2':
                print(f"    Attempting to load JP2 for {channel_num} from '{channel_path}'")
                data, current_metadata = load_jp2_data_imageio(channel_path) # metadata is None for JP2

            if data is None:
                 print(f"    Failed to load data for {channel_num} (data is None).")
                 continue

            original_shapes[channel_num] = data.shape
            if current_metadata:
                metadata_list[channel_num] = current_metadata

            # --- Crear máscara según el tipo elegido ---
            if args.mask_type == 'occulting':
                print(f"    Creating OCCULTING mask for channel {channel_num}")
                if args.file_type == 'fits':
                    if current_metadata:
                        print(f"      Calling create_occulting_mask_fits with data shape {data.shape if data is not None else 'None'}")
                        mask = create_occulting_mask_fits(data, current_metadata)
                    else:
                         print(f"    Error: FITS metadata missing for channel {channel_num} for occulting mask. Skipping.")
                         continue
                elif args.file_type == 'jp2':
                    mask_radius_to_use = args.jp2_mask_radius
                    if data.shape != (4096, 4096) and args.jp2_mask_radius == 1600: # Assuming 4096 default for JP2
                         scale_factor = min(data.shape) / 4096.0
                         scaled_radius = int(args.jp2_mask_radius * scale_factor)
                         print(f"    Adjusting JP2 occulting mask radius to {scaled_radius}.")
                         mask_radius_to_use = scaled_radius
                    mask = create_occulting_mask_jp2(data, mask_radius_to_use)

            elif args.mask_type == 'disk':
                 # This part would need implementation for disk masks
                 print(f"    Creating DISK mask for channel {channel_num}")
                 raise NotImplementedError("Mask type 'disk' requires specific disk mask functions.")
            else:
                 raise ValueError(f"Invalid mask type specified: {args.mask_type}")

            # Preprocesar si la máscara se creó
            if mask is not None:
                print(f"    Preprocessing image for {channel_num}. Data shape: {data.shape if data is not None else 'None'}, Mask shape: {mask.shape if mask is not None else 'None'}")
                masked_img_data = preprocess_image(data, mask, args.image_size) # Renamed to avoid conflict
                masked_data_list.append(masked_img_data)
                channel_names_loaded.append(channel_num)
                print(f"    SUCCESSFULLY processed and added channel {channel_num}")
            else:
                print(f"    Skipping channel {channel_num} due to masking error (mask is None).")

        except FileNotFoundError as e:
             print(f"    ERROR (FileNotFoundError) processing {channel_dir_name}: {e}")
        except NotImplementedError as e:
             print(f"    ERROR (NotImplementedError) processing {channel_dir_name}: {e}")
        except Exception as e:
            print(f"    UNEXPECTED ERROR processing {channel_dir_name}: {e}")
            traceback.print_exc()
    # ********************************************************************
    # ***************** END OF DATA LOADING LOOP *************************
    # ********************************************************************

    if not masked_data_list:
        print("No data successfully loaded and preprocessed. Exiting.") # This should now only happen if all channels genuinely fail
        if cli_args:
            return None
        return

    print(f"Successfully processed {len(masked_data_list)} channels: {', '.join(channel_names_loaded)}")

    # --- Prepare Data for ML, Anomaly Detection logic as before ---
    prepared_data, valid_pixel_mask_1d, nan_mask_1d = prepare_data_concatenated(
        masked_data_list
    )
    if prepared_data.shape[0] == 0:
        print("No valid pixels found after concatenation. Cannot proceed.")
        if cli_args: return None
        return

    anomaly_scores = detect_anomalies_isolation_forest(prepared_data, args.contamination)
    if anomaly_scores.size == 0:
        print("Anomaly detection returned no scores. Exiting.")
        if cli_args: return None
        return

    anomaly_map_2d = np.full((args.image_size, args.image_size), np.nan)
    valid_pixel_mask_1d_bool = valid_pixel_mask_1d.astype(bool)
    if valid_pixel_mask_1d_bool.size == args.image_size * args.image_size:
        valid_pixel_mask_2d_for_scores = valid_pixel_mask_1d_bool.reshape((args.image_size, args.image_size))
        anomaly_map_2d[valid_pixel_mask_2d_for_scores] = anomaly_scores
    else:
        temp_full_map_1d = np.full(args.image_size * args.image_size, np.nan)
        temp_full_map_1d[valid_pixel_mask_1d_bool] = anomaly_scores
        anomaly_map_2d = temp_full_map_1d.reshape((args.image_size, args.image_size))

    # --- Loop de Umbrales ---
    results_for_notebook = []
    valid_score_pixel_mask_2d = ~np.isnan(anomaly_map_2d)
    total_valid_pixels_resized = np.sum(valid_score_pixel_mask_2d)

    for anomaly_threshold in args.anomaly_thresholds:
        print(f"\n===== Processing with Anomaly Threshold: {anomaly_threshold} =====")
        anomaly_mask_global_2d = np.full((args.image_size, args.image_size), False, dtype=bool)
        if total_valid_pixels_resized > 0: # Ensure we only operate where scores exist
            anomaly_mask_global_2d[valid_score_pixel_mask_2d] = (anomaly_map_2d[valid_score_pixel_mask_2d] < anomaly_threshold)
        anomaly_pixels_count = np.sum(anomaly_mask_global_2d)

        # --- Preparar datos para Clustering ---
        anomalous_indices_1d_flat = np.where(anomaly_mask_global_2d.flatten())[0]
        map_flat_to_prepared_idx = np.full(args.image_size * args.image_size, -1, dtype=int)
        prepared_data_indices_range = np.arange(prepared_data.shape[0])
        map_flat_to_prepared_idx[valid_pixel_mask_1d_bool] = prepared_data_indices_range
        indices_in_prepared_data_for_clustering = map_flat_to_prepared_idx[anomalous_indices_1d_flat]
        indices_in_prepared_data_for_clustering = indices_in_prepared_data_for_clustering[indices_in_prepared_data_for_clustering != -1]

        anomaly_intensity_features = np.array([])
        if len(indices_in_prepared_data_for_clustering) > 0:
             anomaly_intensity_features = prepared_data[indices_in_prepared_data_for_clustering]
        else:
             print("No anomalous pixels found for clustering at this threshold.")


        # --- Clustering ---
        cluster_labels_for_threshold = np.array([])
        cluster_mask_for_threshold = np.zeros((args.image_size, args.image_size), dtype=int)
        cluster_cmap_for_threshold = matplotlib.colors.ListedColormap([])
        cluster_patches_for_threshold = []
        n_clusters_for_threshold = 0

        if anomaly_intensity_features.shape[0] > 0 :
            num_samples_for_clustering = anomaly_intensity_features.shape[0]
            clusters_to_use = min(args.n_clusters, num_samples_for_clustering) if num_samples_for_clustering > 0 else 0

            if clusters_to_use > 0:
                cluster_labels_for_threshold, _ = perform_kmeans_clustering(
                    anomaly_intensity_features, clusters_to_use, args.random_state
                )
                (
                    cluster_mask_for_threshold,
                    cluster_cmap_for_threshold,
                    cluster_patches_for_threshold,
                    n_clusters_for_threshold,
                ) = create_cluster_mask(
                     anomaly_mask_global_2d,
                     cluster_labels_for_threshold,
                     valid_pixel_mask_1d_bool,
                     args.image_size
                 )
            else:
                print("Not enough samples to form any clusters. Skipping K-Means.")
        else:
             print("No data points for clustering. Skipping K-Means.")


        results_for_notebook.append({
            "anomaly_threshold": anomaly_threshold,
            "masked_data_list": masked_data_list, # This is the full list of processed channel images
            "channel_names_loaded": channel_names_loaded, # Names of successfully loaded channels
            "cluster_mask_global": cluster_mask_for_threshold,
            "cluster_cmap_global": cluster_cmap_for_threshold,
            "n_clusters_global": n_clusters_for_threshold,
            "cluster_patches_global": cluster_patches_for_threshold,
            "total_pixels_resized": total_valid_pixels_resized, # Total valid pixels in the resized, masked area
            "anomaly_pixels_count": anomaly_pixels_count, # Anomalous pixels for this threshold
        })

    print(f"\nScript processing finished. Data for {len(results_for_notebook)} thresholds ready.")

    if cli_args:
        return results_for_notebook

if __name__ == "__main__":
    main()
