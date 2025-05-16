API_NAME = "API SOLAR"
API_VERSION = "0.1.0"

# Configuration for the anomaly detection pipeline
config = {
    "data_dir": "",  # Will be set dynamically
    "output_dir": "",  # Will be set dynamically
    "file_type": "fits",  # Default to JP2, can be changed to "fits"
    "channels": None,  # List of channels like ['94', '131', '171'], or None for all
    "image_size": 512,  # Default size for both JP2 and FITS
    "jp2_mask_radius": 1600,  # JP2-specific setting
    
    # Algorithm parameters
    "contamination": 0.05,
    "anomaly_thresholds": [0.1],
    "n_clusters": 5,
    "random_state": 42,
}
