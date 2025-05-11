import os
import numpy as np
import matplotlib
from tqdm import tqdm
import datetime

from app.api.pipeline.data_loader import load_masked_channel_data_jp2
from app.api.pipeline.model import (
    create_cluster_mask,
    detect_anomalies_isolation_forest,
    perform_kmeans_clustering
)
from app.api.pipeline.preprocess import prepare_data_concatenated
from app.api.pipeline.visualization import plot_results, plot_single_channel

from app.config.settings import config

# --- Default Configuration ---
# config = {
#     "data_dir": r"D:\OneDrive - Universidad de La Salle\Maestría IA\SOLAR\API\app\api\pipeline\testing_input",
#     "output_dir": "./output_figures",
#     "file_type": "jp2",  
#     "channels": None,  # List of channels like ['94', '131', '171'], or None for all
#     "image_size": 2048,
#     "jp2_mask_radius": 1600,

#     # --- Algorithm Parameters ---
#     "anomaly_thresholds": [0.1],  # Different thresholds for anomaly detection
#     "contamination": 0.02,  # Proportion of outliers in the data
#     "n_clusters": 5,  # Number of clusters for KMeans
#     "random_state": 42,  # Random seed for reproducibility
# }

def run_pipeline(config=config):
    """
    Executes the entire pipeline for anomaly detection and clustering.
    Each step of the process is printed for clarity.
    """

    print("Step 0: Creating output directory...\n")
    os.makedirs(config["output_dir"], exist_ok=True)  # Ensure output directory exists

    # --- Step 1: Load and preprocess masked channel data ---
    print("Step 1: Loading and preprocessing masked channel data...\n")
    masked_data_list, channel_names, jp2_paths = load_masked_channel_data_jp2(
        config["data_dir"],
        config["image_size"]
    )

    # --- Step 2: Prepare data for anomaly detection ---
    print("Step 2: Preparing data for anomaly detection...\n")
    prepared_data, valid_pixel_mask_1d, nan_mask_1d = prepare_data_concatenated(masked_data_list)

    # --- Step 3: Run Isolation Forest for anomaly detection ---
    print("Step 3: Running Isolation Forest for anomaly detection...\n")
    anomaly_scores = detect_anomalies_isolation_forest(prepared_data, config["contamination"])

    # Create anomaly map
    anomaly_map_2d = np.full((config["image_size"], config["image_size"]), np.nan)
    valid_pixel_mask_1d = valid_pixel_mask_1d.astype(bool)
    valid_pixel_mask_2d = valid_pixel_mask_1d.reshape((config["image_size"], config["image_size"]))
    anomaly_map_2d[valid_pixel_mask_2d] = anomaly_scores

    total_pixels_resized = config["image_size"] * config["image_size"]

    # --- Step 4: Thresholding anomalies and clustering ---
    print("Step 4: Thresholding anomalies and clustering...\n")
    for anomaly_threshold in config["anomaly_thresholds"]:
        # Create anomaly mask based on threshold
        anomaly_mask_global_2d = np.full((config["image_size"], config["image_size"]), False)
        valid_score_mask = ~np.isnan(anomaly_map_2d)
        anomaly_mask_global_2d[valid_score_mask] = (anomaly_map_2d[valid_score_mask] < anomaly_threshold)

        anomaly_pixels_count = np.sum(anomaly_mask_global_2d)
        anomaly_percentage = (anomaly_pixels_count / np.sum(valid_score_mask)) * 100 if np.sum(valid_score_mask) > 0 else 0

        valid_and_anomalous_mask_2d = valid_pixel_mask_2d & anomaly_mask_global_2d
        valid_and_anomalous_indices_flat = np.where(valid_and_anomalous_mask_2d.flatten())[0]

        full_indices = np.arange(total_pixels_resized)
        prepared_data_indices = np.full(total_pixels_resized, -1, dtype=int)
        prepared_data_indices[valid_pixel_mask_1d] = np.arange(prepared_data.shape[0])
        indices_for_clustering = prepared_data_indices[valid_and_anomalous_indices_flat]
        indices_for_clustering = indices_for_clustering[indices_for_clustering != -1]

        if len(indices_for_clustering) == 0:
            anomaly_intensity_features = np.array([])  # No anomalous pixels for clustering
        else:
            anomaly_intensity_features = prepared_data[indices_for_clustering]

        # Variables for clustering results
        cluster_labels = np.array([])
        cluster_mask_final = np.zeros((config["image_size"], config["image_size"]), dtype=int)
        cluster_cmap_final = matplotlib.colors.ListedColormap([])
        cluster_patches_final = []
        n_clusters_final = 0
        cluster_pixels_counts = []
        cluster_anomaly_percentages = []

        if anomaly_intensity_features.shape[0] >= config["n_clusters"]:
            # Perform KMeans clustering
            cluster_labels, _ = perform_kmeans_clustering(
                anomaly_intensity_features, config["n_clusters"], config["random_state"]
            )

            # Create the final cluster mask
            cluster_mask_final, cluster_cmap_final, cluster_patches_final, n_clusters_final = create_cluster_mask(
                anomaly_mask_global_2d,
                cluster_labels,
                valid_pixel_mask_1d,
                config["image_size"]
            )

            # Collect statistics for each cluster
            for cluster_index in range(1, n_clusters_final + 1):
                count = np.sum(cluster_mask_final == cluster_index)
                pct = (count / anomaly_pixels_count) * 100 if anomaly_pixels_count > 0 else 0
                cluster_pixels_counts.append(count)
                cluster_anomaly_percentages.append(pct)

        plot_results(
            masked_data_list=masked_data_list,
            cluster_mask_global=cluster_mask_final,
            cluster_cmap_global=cluster_cmap_final,
            n_clusters_global=n_clusters_final,
            cluster_patches_global=cluster_patches_final,
            channel_names=channel_names,
            anomaly_threshold=anomaly_threshold,
            output_dir=config["output_dir"],
            total_pixels_resized=np.sum(valid_score_mask),  # Base total on valid pixels in resized image
            anomaly_pixels_count=anomaly_pixels_count,
            file_type=config["file_type"],  # Pass file type for filename/title
            clustering_method_name="K-Means"
        )

    print("Pipeline execution completed!")



def run_single_channel_pipeline(config=config, timestamp=None):
    """
    Executes the entire pipeline for anomaly detection and clustering.
    Each step of the process is printed for clarity.

    Parameters:
    -----------
    config : dict
        Pipeline configuration dictionary
    timestamp : str, optional
        Timestamp to use in output filenames. If None, current time will be used.    
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Step 0: Creating output directory...\n")
    os.makedirs(config["output_dir"], exist_ok=True)  # Ensure output directory exists

    # --- Step 1: Load and preprocess masked channel data ---
    print("Step 1: Loading and preprocessing masked channel data...\n")
    masked_data_list, channel_names, jp2_paths = load_masked_channel_data_jp2(
        config["data_dir"],
        config["image_size"]
    )

    # --- Step 2: Prepare data for anomaly detection ---
    print("Step 2: Preparing data for anomaly detection...\n")
    prepared_data, valid_pixel_mask_1d, nan_mask_1d = prepare_data_concatenated(masked_data_list)

    # --- Step 3: Run Isolation Forest for anomaly detection ---
    print("Step 3: Running Isolation Forest for anomaly detection...\n")
    anomaly_scores = detect_anomalies_isolation_forest(prepared_data, config["contamination"])

    # Create anomaly map
    anomaly_map_2d = np.full((config["image_size"], config["image_size"]), np.nan)
    valid_pixel_mask_1d = valid_pixel_mask_1d.astype(bool)
    valid_pixel_mask_2d = valid_pixel_mask_1d.reshape((config["image_size"], config["image_size"]))
    anomaly_map_2d[valid_pixel_mask_2d] = anomaly_scores

    total_pixels_resized = config["image_size"] * config["image_size"]

    # --- Step 4: Thresholding anomalies and clustering ---
    print("Step 4: Thresholding anomalies and clustering...\n")
    for anomaly_threshold in config["anomaly_thresholds"]:
        # Create anomaly mask based on threshold
        anomaly_mask_global_2d = np.full((config["image_size"], config["image_size"]), False)
        valid_score_mask = ~np.isnan(anomaly_map_2d)
        anomaly_mask_global_2d[valid_score_mask] = (anomaly_map_2d[valid_score_mask] < anomaly_threshold)

        anomaly_pixels_count = np.sum(anomaly_mask_global_2d)

        # Process clustering
        valid_and_anomalous_mask_2d = valid_pixel_mask_2d & anomaly_mask_global_2d
        valid_and_anomalous_indices_flat = np.where(valid_and_anomalous_mask_2d.flatten())[0]

        prepared_data_indices = np.full(total_pixels_resized, -1, dtype=int)
        prepared_data_indices[valid_pixel_mask_1d] = np.arange(prepared_data.shape[0])
        indices_for_clustering = prepared_data_indices[valid_and_anomalous_indices_flat]
        indices_for_clustering = indices_for_clustering[indices_for_clustering != -1]

        # Initialize clustering variables
        cluster_mask_final = np.zeros((config["image_size"], config["image_size"]), dtype=int)
        cluster_cmap_final = matplotlib.colors.ListedColormap([])
        cluster_patches_final = []
        n_clusters_final = 0

        # Perform clustering if enough anomalous pixels
        if len(indices_for_clustering) >= config["n_clusters"]:
            anomaly_intensity_features = prepared_data[indices_for_clustering]
            cluster_labels, _ = perform_kmeans_clustering(
                anomaly_intensity_features, 
                config["n_clusters"], 
                config["random_state"]
            )

            # Create the final cluster mask
            cluster_mask_final, cluster_cmap_final, cluster_patches_final, n_clusters_final = create_cluster_mask(
                anomaly_mask_global_2d,
                cluster_labels,
                valid_pixel_mask_1d,
                config["image_size"]
            )

        # Plot results separately for each channel
        for idx, (channel_data, channel_name) in enumerate(zip(masked_data_list, channel_names)):
            plot_single_channel(
                masked_data=channel_data,
                cluster_mask_global=cluster_mask_final,
                cluster_cmap_global=cluster_cmap_final,
                n_clusters_global=n_clusters_final,
                cluster_patches_global=cluster_patches_final,
                channel=channel_name,
                anomaly_threshold=anomaly_threshold,
                output_dir=config["output_dir"],
                total_pixels_resized=np.sum(valid_score_mask),
                anomaly_pixels_count=anomaly_pixels_count,
                file_type=config["file_type"],
                clustering_method_name="K-Means",
                timestamp=timestamp
            )

    print("Pipeline execution completed!")