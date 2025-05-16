import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import datetime

# Import the necessary pipeline modules
from app.api.pipeline.data_loader import load_fits_data, create_circular_mask, preprocess_image
from app.api.pipeline.data_loader import load_masked_channel_data_jp2
from app.api.pipeline.model import (
    detect_anomalies_isolation_forest,
    perform_kmeans_clustering,
    create_cluster_mask
)
from app.api.pipeline.preprocess import prepare_data_concatenated
from app.api.pipeline.visualization import plot_results, plot_single_channel

def run_pipeline(config):
    """Run the anomaly detection pipeline."""
    # Create output directory if it doesn't exist
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\n--- Step 1: Loading and preprocessing data ---\n")
    
    masked_data_list = []
    channel_names = []
    
    if config.get("file_type", "fits").lower() == "fits":
        # Process FITS files
        print("Processing FITS files...")
        
        # Get list of subdirectories for each channel
        data_dir = config["data_dir"]
        channels_to_process = config.get("channels", [])
        
        # If specific channels are defined, use those, otherwise try to find all available ones
        if not channels_to_process:
            all_dirs = os.listdir(data_dir)
            channels_to_process = [d.split("_")[1] if "_" in d else d 
                                  for d in all_dirs 
                                  if os.path.isdir(os.path.join(data_dir, d))]
            
            # Filter out non-EUV channels if needed
            channels_to_process = [c for c in channels_to_process 
                                  if c not in ['1600', '1700']]
        
        print(f"Processing channels: {channels_to_process}")
        
        for channel in channels_to_process:
            try:
                # Construct channel directory path - handle different formats
                if os.path.isdir(os.path.join(data_dir, f"aia_{channel}")):
                    channel_dir = os.path.join(data_dir, f"aia_{channel}")
                elif os.path.isdir(os.path.join(data_dir, channel)):
                    channel_dir = os.path.join(data_dir, channel)
                else:
                    print(f"Warning: Could not find directory for channel {channel}. Skipping.")
                    continue
                
                data, metadata = load_fits_data(channel_dir)

                # --- Ensure we use a 2D slice ---
                if data.ndim == 3:
                    # Use the first non-empty slice
                    for i in range(data.shape[0]):
                        if np.any(data[i]):
                            data2d = data[i]
                            break
                    else:
                        data2d = data[0]
                else:
                    data2d = data

                mask = create_circular_mask(data2d, metadata)
                masked_data = preprocess_image(data2d, mask, config.get("image_size", 512))
                
                # Append processed data
                masked_data_list.append(masked_data)
                channel_names.append(channel)
                
                print(f"Successfully processed channel {channel}. Shape: {masked_data.shape}")
                
            except Exception as e:
                print(f"Error processing channel {channel}: {e}")
                continue
    else:
        # Process JP2 files
        masked_data_list, channel_names, _ = load_masked_channel_data_jp2(
            config["data_dir"],
            config.get("image_size", 512),
            mask_radius=config.get("jp2_mask_radius", 1600)
        )
    
    if not masked_data_list:
        raise ValueError(f"No {config.get('file_type', 'jp2')} files were successfully processed.")
        
    print(f"\nProcessed {len(masked_data_list)} channels: {channel_names}")
    
    # Print statistics about valid pixels per channel
    print("\nChecking masked data before concatenation:")
    for i, (data, channel) in enumerate(zip(masked_data_list, channel_names)):
        valid_count = np.sum(~np.isnan(data))
        total_count = data.size
        print(f"Channel {channel}: {valid_count}/{total_count} valid pixels ({valid_count/total_count*100:.2f}%)")

    # Prepare data for anomaly detection
    prepared_data, valid_pixel_mask_1d, nan_mask_1d = prepare_data_concatenated(
        masked_data_list
    )
    
    # Create 2D mask for visualization
    img_size = config.get("image_size", 512)
    valid_pixel_mask_2d = valid_pixel_mask_1d.reshape((img_size, img_size))
    
    print(f"\nData prepared for modeling: {prepared_data.shape}")
    
    # Step 2: Detect anomalies using Isolation Forest
    print("\n--- Step 2: Detecting anomalies with Isolation Forest ---\n")
    anomaly_scores = detect_anomalies_isolation_forest(
        prepared_data, 
        config.get("contamination", 0.05)
    )

    # Create anomaly map
    anomaly_map_2d = np.full((img_size, img_size), np.nan)
    anomaly_map_2d[valid_pixel_mask_2d] = anomaly_scores

    # Step 3: Perform clustering on the anomalies
    print("\n--- Step 3: Performing clustering on anomalies ---\n")
    
    # Use the specified thresholds or defaults
    anomaly_thresholds = config.get("anomaly_thresholds", [-0.05, -0.1, -0.15])
    
    results = {}
    
    # For each threshold, create cluster masks and visualizations
    for threshold in anomaly_thresholds:
        threshold_str = str(abs(threshold)).replace('.', '_')
        threshold_dir = os.path.join(output_dir, f"threshold_{threshold_str}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        print(f"\nProcessing threshold {threshold}...")
        
        # Apply threshold to get anomaly mask
        anomaly_mask = anomaly_scores <= threshold
        anomaly_mask_2d = np.full((img_size, img_size), False)
        anomaly_mask_2d[valid_pixel_mask_2d] = anomaly_mask
        
        anomaly_count = np.sum(anomaly_mask)
        print(f"Found {anomaly_count} anomalous points ({anomaly_count/prepared_data.shape[0]*100:.2f}% of valid pixels)")
        
        # Skip if no anomalies found
        if not np.any(anomaly_mask):
            print(f"No anomalies found for threshold {threshold}. Skipping clustering.")
            continue
        
        # Prepare for clustering - only cluster the anomalies
        # anomaly_features = prepared_data[anomaly_mask]
        
        # Perform K-means clustering on anomalies
        cluster_labels, inertia = perform_kmeans_clustering(
            prepared_data, 
            anomaly_scores,
            threshold, 
            config.get("n_clusters", 5), 
            config.get("random_state", 42)
        )
        
        # Create cluster mask
        cluster_mask_result = create_cluster_mask(
            anomaly_mask_2d,
            cluster_labels,
            valid_pixel_mask_1d,
            img_size
        )
        
        cluster_mask_2d = cluster_mask_result[0]
        cluster_cmap = cluster_mask_result[1]
        cluster_patches = cluster_mask_result[2]
        n_clusters = cluster_mask_result[3]
        
        # Count pixels per cluster
        cluster_stats = []
        if n_clusters > 0:
            for cluster_id in range(1, n_clusters+1):
                count = np.sum(cluster_mask_2d == cluster_id)
                percentage = count / anomaly_count * 100 if anomaly_count > 0 else 0
                cluster_stats.append({
                    "cluster_id": cluster_id,
                    "pixel_count": int(count),
                    "percentage": float(percentage)
                })
                print(f"  Cluster {cluster_id}: {count} pixels ({percentage:.2f}%)")
        
        results[threshold] = {
            "anomaly_count": int(anomaly_count),
            "n_clusters": n_clusters,
            "cluster_stats": cluster_stats,
            "threshold_dir": threshold_dir
        }
        
        # Plot results - create both overview plot and individual channel plots
        
        # Plot anomaly map
        plt.figure(figsize=(10, 8))
        plt.imshow(anomaly_map_2d, cmap='coolwarm_r')
        plt.colorbar(label='Anomaly Score')
        plt.title(f'Anomaly Map (Threshold: {threshold})')
        plt.axis('off')
        plt.savefig(os.path.join(threshold_dir, 'anomaly_map.png'), bbox_inches='tight', dpi=150)
        plt.close()
        
        # Plot cluster map
        if n_clusters > 0:
            plt.figure(figsize=(10, 8))
            plt.imshow(cluster_mask_2d, cmap=cluster_cmap)
            plt.colorbar(label='Cluster ID')
            plt.title(f'Cluster Map (Threshold: {threshold}, Clusters: {n_clusters})')
            plt.axis('off')
            plt.savefig(os.path.join(threshold_dir, 'cluster_map.png'), bbox_inches='tight', dpi=150)
            plt.close()
        
        # Plot each channel with overlaid clusters
        for i, (data, channel) in enumerate(zip(masked_data_list, channel_names)):
            plot_single_channel(
                data,
                cluster_mask_global=cluster_mask_2d,
                cluster_cmap_global=cluster_cmap,
                n_clusters_global=n_clusters,
                cluster_patches_global=cluster_patches,
                channel=channel,
                anomaly_threshold=threshold,
                output_dir=threshold_dir,
                total_pixels_resized=img_size * img_size,
                anomaly_pixels_count=anomaly_count,
                file_type=config.get("file_type", "jp2"),
                clustering_method_name="K-Means"
            )
    
    # Generate individual channel visualizations
    print("\n--- Step 4: Generating channel visualizations ---\n")
    channel_dir = os.path.join(output_dir, "channel_results")
    os.makedirs(channel_dir, exist_ok=True)
    
    for i, (data, channel) in enumerate(zip(masked_data_list, channel_names)):
        try:
            plot_single_channel(
                data,
                channel=channel,
                output_dir=channel_dir
            )
        except Exception as e:
            print(f"Error visualizing channel {channel}: {e}")
    
    print("\nPipeline completed successfully.")
    return results

def run_single_channel_pipeline(config, timestamp=None):
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
        config["image_size"],
        mask_radius=config.get("jp2_mask_radius", 1600)
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
            # Instead of filtering the data here and passing only anomalous data,
            # we'll pass the full prepared_data and let the function do the filtering
            cluster_labels, _ = perform_kmeans_clustering(
                prepared_data,            # Complete data
                anomaly_scores,           # Complete anomaly scores  
                anomaly_threshold,        # Threshold
                config["n_clusters"],     # n_clusters
                config["random_state"]    # random_state
            )
            
            # Create the final cluster mask
            # cluster_labels already has the correct shape from the function
            cluster_mask_final, cluster_cmap_final, cluster_patches_final, n_clusters_final = create_cluster_mask(
                anomaly_mask_global_2d,
                cluster_labels,
                valid_pixel_mask_1d,
                config["image_size"]
            )

        # Plot results separately for each channel
        for idx, (channel_data, channel_name) in enumerate(zip(masked_data_list, channel_names)):
            # Create threshold-specific output directory
            threshold_dir = os.path.join(config["output_dir"], f"threshold_{str(abs(anomaly_threshold)).replace('.', '_')}")
            os.makedirs(threshold_dir, exist_ok=True)
            
            plot_single_channel(
                masked_data=channel_data,
                cluster_mask_global=cluster_mask_final,
                cluster_cmap_global=cluster_cmap_final,
                n_clusters_global=n_clusters_final,
                cluster_patches_global=cluster_patches_final,
                channel=channel_name,
                anomaly_threshold=anomaly_threshold,
                output_dir=threshold_dir,
                total_pixels_resized=np.sum(valid_score_mask),
                anomaly_pixels_count=anomaly_pixels_count,
                file_type=config["file_type"],
                clustering_method_name="K-Means",
                timestamp=timestamp
            )

    print("Pipeline execution completed!")
    return True

######### TESTING FUNCTIONS #########

def save_and_list_raw_fits(config):
    """
    Lists and returns the paths of all raw FITS files in the input directory.
    No processing or masking is performed.
    """
    data_dir = config["data_dir"]
    channels = config.get("channels", [])
    fits_files = []

    for channel in channels:
        channel_dir = os.path.join(data_dir, f"aia_{channel}")
        if not os.path.isdir(channel_dir):
            continue
        for fname in os.listdir(channel_dir):
            if fname.lower().endswith(".fits"):
                full_path = os.path.abspath(os.path.join(channel_dir, fname))
                fits_files.append({
                    "channel": channel,
                    "filename": fname,
                    "path": full_path,
                    "url": f"file://{full_path}"
                })
    return fits_files