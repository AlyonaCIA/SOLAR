API_NAME = "API SOLAR"
API_VERSION = "0.1.0"

# Configuration for the anomaly detection pipeline
# --- Default Configuration ---
config = {
    "data_dir": "testing_input",
    "output_dir": "./output_figures",
    "file_type": "jp2",  
    "channels": None,  # List of channels like ['94', '131', '171'], or None for all
    "image_size": 2048,
    "jp2_mask_radius": 1600,

    # --- Algorithm Parameters ---
    # "anomaly_thresholds": [0.0, 0.05, 0.1],  # Different thresholds for anomaly detection
    "anomaly_thresholds": [0.1, 0.2],  # Different thresholds for anomaly detection
    "contamination": 0.02,  # Proportion of outliers in the data
    "n_clusters": 5,  # Number of clusters for KMeans
    "random_state": 42,  # Random seed for reproducibility
}
