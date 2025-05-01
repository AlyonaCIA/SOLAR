API_NAME = "API SOLAR"
API_VERSION = "0.1.0"
API_DESCRIPTION = "API para la gestión de datos solares y meteorológicos"
CONTRAST_FACTOR = 5.0

# Configuration for the anomaly detection pipeline
config = {
    "data_dir": "sdo_data",  # Directory where the input .fits files are stored
    "output_dir": "./output_figures",  # Directory where the output images will be saved
    "anomaly_thresholds": [0.1],  # Thresholds for anomaly detection
    "image_size": 512,  # Size of the images to be processed
    "contamination": 0.05,  # Proportion of anomalies in the dataset
    "n_clusters": 7,  # Number of clusters for KMeans
    "max_k": 10,  # Maximum number of clusters for KMeans
    "random_state": 42  # Random state for reproducibility
}
