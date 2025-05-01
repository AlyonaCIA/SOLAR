import os

import matplotlib
import numpy as np
from data_loader import load_fits_data
from model import (create_cluster_mask, detect_anomalies_isolation_forest,
                   perform_kmeans_clustering)
from preprocess import (create_circular_mask, prepare_data_concatenated,
                        preprocess_image)
from visualization import plot_results

# --- Configuration ---
config = {
    "data_dir": "sdo_data",
    "channels": ["94", "131", "171", "193", "211", "304", "335", "1600", "1700"],
    "anomaly_thresholds": [0.1],
    "output_dir": "./output_figures",
    "image_size": 512,
    "contamination": 0.05,
    "n_clusters": 7,
    "max_k": 10,
    "random_state": 42
}


def run_pipeline(config):
    """Runs the entire pipeline for anomaly detection and clustering on solar image
    data.

    Args:
        config (dict): A dictionary containing configuration settings, including data
        directory,channels, anomaly thresholds, and other parameters for clustering and
        anomaly detection.

    Returns:
        None: The function saves the results as output files in the specified directory.
    """
    os.makedirs(config["output_dir"], exist_ok=True)

    # --- 1. Channel selection ---
    channels = (
        [f"aia_{c}" for c in config["channels"]]
        if config["channels"] else
        [
            d for d in os.listdir(config["data_dir"])
            if os.path.isdir(os.path.join(config["data_dir"], d))
            and not d.startswith("aia_1600")
            and not d.startswith("aia_1700")
        ]
    )

    if not channels:
        print("No channels found. Exiting.")
        return

    # --- 2. Load and preprocess image data ---
    masked_data_list = []
    channel_names = []

    for channel_dir in channels:
        try:
            channel = channel_dir.split("_")[1]
            channel_names.append(channel)

            path = os.path.join(config["data_dir"], channel_dir)
            data, metadata = load_fits_data(path)
            mask = create_circular_mask(data, metadata)
            masked = preprocess_image(data, mask, config["image_size"])

            masked_data_list.append(masked)
        except Exception as e:
            print(f"Error processing {channel_dir}: {e}")

    if not masked_data_list:
        print("No data loaded. Exiting.")
        return

    # --- 3. Prepare data for anomaly detection ---
    prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(
        masked_data_list)

    # --- 4. Anomaly detection with Isolation Forest ---
    anomaly_scores = detect_anomalies_isolation_forest(
        prepared_data, config["contamination"])

    anomaly_map = np.full((config["image_size"], config["image_size"]), np.nan)
    anomaly_map[valid_pixel_mask.reshape(
        config["image_size"], config["image_size"])] = anomaly_scores

    # --- 5. Anomaly thresholding and clustering ---
    for threshold in config["anomaly_thresholds"]:
        print(f"Processing with anomaly threshold: {threshold}")
        anomaly_mask = anomaly_map < threshold
        print(f"Anomalies detected: {np.sum(anomaly_mask)}")

        anomaly_indices = np.argwhere(anomaly_mask)
        valid_indices = np.argwhere(~nan_mask.reshape(
            (config["image_size"], config["image_size"])))
        index_map = {tuple(idx): i for i, idx in enumerate(valid_indices)}

        features = []
        valid_pixel_positions = []

        for idx_2d in anomaly_indices:
            key = tuple(idx_2d)
            if key in index_map:
                features.append(prepared_data[index_map[key]])
                valid_pixel_positions.append(idx_2d)

        features = np.array(features)
        valid_pixel_positions = np.array(valid_pixel_positions)

        if len(features) > 0:
            cluster_labels, _ = perform_kmeans_clustering(
                features,
                config["n_clusters"],
                random_state=config["random_state"]
            )

            cluster_mask,  # noqa: F821
            cluster_cmap,  # noqa: F821
            cluster_patches,  # noqa: F821
            n_clusters = create_cluster_mask(
                anomaly_mask,
                cluster_labels,
                valid_pixel_mask,
                config["image_size"]
            )
        else:
            print(f"No anomalies found for threshold {threshold}.")
            cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
            cluster_cmap = matplotlib.colors.ListedColormap([])
            cluster_patches = []
            n_clusters = 0

        # --- 6. Visualization ---
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


# --- Run pipeline ---
if __name__ == "__main__":
    run_pipeline(config)
