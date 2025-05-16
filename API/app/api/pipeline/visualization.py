import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def plot_single_channel(
    masked_data,
    cluster_mask_global=None,
    cluster_cmap_global=None,
    n_clusters_global=0,
    cluster_patches_global=None,
    channel="Unknown",
    anomaly_threshold=None,
    output_dir="./output",
    total_pixels_resized=0,
    anomaly_pixels_count=0,
    file_type="fits",
    clustering_method_name="K-Means",
    timestamp=None
):
    """Plot a single channel with optional cluster overlay."""
    try:
        plt.figure(figsize=(10, 8))
        
        # Check if we have any valid data
        valid_mask = ~np.isnan(masked_data)
        if np.any(valid_mask):
            # Compute percentiles for better visualization range
            vmin = np.nanpercentile(masked_data, 2)
            vmax = np.nanpercentile(masked_data, 98)
            
            # Plot the original data with proper scaling
            plt.imshow(masked_data, cmap='sdoaia' + str(channel) if channel.isdigit() else 'viridis', 
                      vmin=vmin, vmax=vmax, origin='lower')
            
            # Overlay clusters if provided
            if cluster_mask_global is not None and n_clusters_global > 0:
                for cluster_idx in range(1, n_clusters_global + 1):
                    # Create mask for this specific cluster
                    cluster_area = (cluster_mask_global == cluster_idx)
                    if np.any(cluster_area):
                        # Get color from colormap
                        color_idx = (cluster_idx - 1) / max(1, n_clusters_global - 1)
                        cluster_color = cluster_cmap_global(color_idx)
                        
                        # Create masked array to only show this cluster
                        masked_cluster = np.ma.masked_where(~cluster_area, cluster_mask_global)
                        
                        # Plot the cluster with a single color
                        plt.imshow(
                            masked_cluster,
                            cmap=matplotlib.colors.ListedColormap([cluster_color]),
                            alpha=0.6,  # Semi-transparent
                            vmin=cluster_idx - 0.5,
                            vmax=cluster_idx + 0.5,
                            origin='lower'
                        )
        else:
            # If all data is NaN, show a blank image
            plt.imshow(np.zeros_like(masked_data), cmap='viridis')
            plt.text(masked_data.shape[1]//2, masked_data.shape[0]//2, "No valid data", 
                     ha='center', va='center', fontsize=20)
        
        # Add colorbar and title
        plt.colorbar(label='Intensity')
        
        title = f'Channel {channel}'
        if anomaly_threshold is not None:
            title += f' (Threshold: {anomaly_threshold})'
            if anomaly_pixels_count > 0 and total_pixels_resized > 0:
                percentage = anomaly_pixels_count / total_pixels_resized * 100
                title += f'\nAnomalies: {anomaly_pixels_count}/{total_pixels_resized} ({percentage:.2f}%)'
        
        plt.title(title)
        plt.axis('off')
        
        # Add legend if clusters are present
        if cluster_patches_global and n_clusters_global > 0:
            plt.legend(
                handles=cluster_patches_global,
                loc='upper right',
                fontsize='small',
                framealpha=0.7
            )
        
        # Save the figure with descriptive filename
        filename = f'aia_{channel}'
        if anomaly_threshold is not None:
            threshold_str = str(abs(anomaly_threshold)).replace('.', '_')
            filename += f'_threshold_{threshold_str}'
        filename += '.png'
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Saved visualization for channel {channel} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error in plot_single_channel for {channel}: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(
    masked_data_list,
    channel_names,
    anomaly_map,
    cluster_mask,
    threshold,
    output_dir
):
    """Plot combined results including anomaly map and clusters."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot anomaly map
        plt.figure(figsize=(10, 8))
        plt.imshow(anomaly_map, cmap='coolwarm_r')
        plt.colorbar(label='Anomaly Score')
        plt.title(f'Anomaly Map (Threshold: {threshold})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'anomaly_map.png'), bbox_inches='tight', dpi=150)
        plt.close()
        
        # Plot cluster mask if it exists and has clusters
        if cluster_mask is not None and np.any(cluster_mask > 0):
            # Create a colormap for clusters
            n_clusters = np.max(cluster_mask)
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            cluster_cmap = ListedColormap(colors)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(cluster_mask, cmap=cluster_cmap)
            plt.colorbar(label='Cluster ID')
            plt.title(f'Cluster Map (Threshold: {threshold})')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, 'cluster_map.png'), bbox_inches='tight', dpi=150)
            plt.close()
            
            # Plot each channel with cluster overlay
            for data, channel in zip(masked_data_list, channel_names):
                plt.figure(figsize=(10, 8))
                
                # Plot the channel data
                valid_mask = ~np.isnan(data)
                if np.any(valid_mask):
                    # Normalize channel data
                    norm_data = data.copy()
                    if np.any(np.isnan(norm_data)):
                        min_val = np.nanmin(norm_data)
                        norm_data[np.isnan(norm_data)] = min_val
                    
                    if np.max(norm_data) > np.min(norm_data):
                        norm_data = (norm_data - np.min(norm_data)) / (np.max(norm_data) - np.min(norm_data))
                    
                    plt.imshow(norm_data, cmap='gray', alpha=0.7)
                
                # Overlay clusters with transparency
                cluster_overlay = np.ma.masked_where(cluster_mask == 0, cluster_mask)
                plt.imshow(cluster_overlay, cmap=cluster_cmap, alpha=0.6)
                
                plt.colorbar(label='Cluster ID')
                plt.title(f'Channel {channel} with Clusters (Threshold: {threshold})')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'channel_{channel}_clusters.png'), 
                            bbox_inches='tight', dpi=150)
                plt.close()
        
        print(f"Saved result visualizations to {output_dir}")
    except Exception as e:
        print(f"Error in plot_results: {e}")