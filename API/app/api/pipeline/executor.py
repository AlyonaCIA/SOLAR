import os

import matplotlib
import numpy as np
from app.api.pipeline.data_loader import load_masked_channel_data
from app.api.pipeline.model import (create_cluster_mask,
                                    detect_anomalies_isolation_forest,
                                    perform_kmeans_clustering)
from app.api.pipeline.preprocess import prepare_data_concatenated
from app.api.pipeline.visualization import plot_results
from tqdm import tqdm

# --- Defalut Configuration ---
config = {
    "data_dir": "testing_input",
    "anomaly_thresholds": [0.1],
    "output_dir": "./output_figures",
    "image_size": 512,
    "contamination": 0.05,
    "n_clusters": 7,
    "max_k": 10,
    "random_state": 42
}


def run_pipeline(config=config):
    """Runs the entire pipeline for anomaly detection and clustering on solar image
    data.

    Steps:
        0. Create output directory
        1. Load and preprocess masked channel data
        2. Prepare concatenated data for anomaly detection
        3. Detect anomalies using Isolation Forest
        4. For each threshold:
            - Create anomaly mask
            - Extract features for valid anomaly pixels
            - Perform clustering (KMeans)
            - Visualize and save clustering results

    Args:
        config (dict): Configuration parameters including:
            - "output_dir": Path to save outputs
            - "data_dir": Directory containing input data
            - "image_size": Dimensions of the square images
            - "anomaly_thresholds": List of thresholds for anomaly detection
            - "contamination": Expected contamination for Isolation Forest
            - "n_clusters": Number of clusters for KMeans
            - "random_state": Random seed for reproducibility

    Returns:
        None
    """
    # --- 0. Create output directory if it doesn't exist ---
    print("Step 0: Creating output directory...")
    os.makedirs(config["output_dir"], exist_ok=True)

    # --- 1. Load and Preprocess Data ---
    print("Step 1: Loading and preprocessing masked channel data...")
    masked_data_list, channel_names, valid_files = load_masked_channel_data(
        config["data_dir"],
        config["image_size"]
    )

    # --- 2. Prepare data for anomaly detection ---
    print("Step 2: Preparing data for anomaly detection...")
    prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(
        masked_data_list)

    # --- 3. Anomaly detection with Isolation Forest ---
    print("Step 3: Running Isolation Forest for anomaly detection...")
    anomaly_map = detect_anomalies_isolation_forest(
        prepared_data,
        config["contamination"],
        config["image_size"],
        valid_pixel_mask
    )

    # --- 4. Anomaly thresholding and clustering ---
    print("Step 4: Thresholding anomalies and clustering...")
    for threshold in tqdm(config["anomaly_thresholds"], desc="Threshold loop"):
        print(f"\n  → Processing with anomaly threshold: {threshold}")

        # Identify pixels considered anomalous
        anomaly_mask = anomaly_map < threshold
        num_anomalies = np.sum(anomaly_mask)
        print(f"    Anomalies detected: {num_anomalies}")

        if num_anomalies == 0:
            print("    No anomalies detected for this threshold.")
            cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
            cluster_cmap = matplotlib.colors.ListedColormap([])
            cluster_patches = []
            n_clusters = 0
        else:
            # Get valid pixel indices and build mapping
            valid_indices = np.argwhere(~nan_mask.reshape(
                (config["image_size"], config["image_size"])))
            index_map = {tuple(idx): i for i, idx in enumerate(valid_indices)}

            # Get valid anomaly pixel indices
            anomaly_indices = np.argwhere(anomaly_mask)
            valid_keys = [tuple(idx)
                          for idx in anomaly_indices if tuple(idx) in index_map]

            if not valid_keys:
                print("    No valid anomaly pixels found within the valid pixel mask.")
                cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
                cluster_cmap = matplotlib.colors.ListedColormap([])
                cluster_patches = []
                n_clusters = 0
            else:
                # Extract feature vectors for valid anomalies
                feature_indices = [index_map[key] for key in valid_keys]
                features = prepared_data[feature_indices]
                np.array(valid_keys)

                # Perform KMeans clustering
                cluster_labels, _ = perform_kmeans_clustering(
                    features,
                    config["n_clusters"],
                    random_state=config["random_state"]
                )
                print(f"    Clusters formed: {np.unique(cluster_labels).size}")

                # Generate cluster mask and colormap
                cluster_mask,  # noqa: F821
                cluster_cmap,  # noqa: F821
                cluster_patches,  # noqa: F821
                n_clusters = create_cluster_mask(
                    anomaly_mask,
                    cluster_labels,
                    valid_pixel_mask,
                    config["image_size"]
                )

        # --- 4.1. Visualization ---
        print("    Saving visualizations...")
        plot_results(
            masked_data_list,
            cluster_mask,
            cluster_cmap,
            n_clusters,
            cluster_patches,
            channel_names,
            threshold,
            config["output_dir"]
        )
