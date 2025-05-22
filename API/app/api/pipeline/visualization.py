import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import io

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - use the fastest possible backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import io
import time

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
    """Plot a single channel with optional cluster overlay - optimized version."""
    try:
        start_time = time.time()
        
        # Smaller figure for faster rendering
        plt.figure(figsize=(10, 8), dpi=115)  # Lower DPI set directly on figure creation
        
        # Check if we have any valid data
        valid_mask = ~np.isnan(masked_data)
        
        if np.any(valid_mask):
            # Fast percentile computation using a sample if data is large
            if masked_data.size > 100000:
                # Sample 10% of non-nan values for faster percentile calculation
                valid_indices = np.where(valid_mask.flatten())[0]
                sample_size = min(len(valid_indices), 10000)  # Cap at 10,000 points
                sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
                sample_data = masked_data.flatten()[sample_indices]
                vmin = np.percentile(sample_data, 2)
                vmax = np.percentile(sample_data, 98)
            else:
                # For smaller arrays, use the full data
                vmin = np.nanpercentile(masked_data, 2)
                vmax = np.nanpercentile(masked_data, 98)
            
            # Keep original custom colormaps for each channel
            channel_cmap = 'sdoaia' + str(channel) if channel.isdigit() else 'viridis'
            
            # Plot the original data with proper scaling
            plt.imshow(masked_data, cmap=channel_cmap, 
                      vmin=vmin, vmax=vmax, origin='lower', alpha=0.85)
            
            # Overlay clusters if provided - optimize by plotting all clusters at once
            if cluster_mask_global is not None and n_clusters_global > 0:
                cluster_overlay = np.ma.masked_where(cluster_mask_global == 0, cluster_mask_global)
                plt.imshow(
                    cluster_overlay, 
                    cmap=cluster_cmap_global,
                    alpha=0.6,
                    origin='lower'
                )
                
        else:
            # If all data is NaN, show a minimal blank image
            plt.imshow(np.zeros_like(masked_data), cmap='viridis')
            plt.text(masked_data.shape[1]//2, masked_data.shape[0]//2, "No valid data", 
                    ha='center', va='center', fontsize=12)
        
        # Minimal title for faster rendering
        title = f'Channel {channel}'
        if anomaly_threshold is not None:
            title += f' (T:{anomaly_threshold})'
        
        plt.title(title)
        plt.axis('off')  # Turn off axis for faster rendering
        
        # Add legend if clusters are present
        if cluster_patches_global and n_clusters_global > 0:
            plt.legend(
                handles=cluster_patches_global,
                loc='upper right',
                fontsize='small',
                framealpha=0.7
            )
        
        # Save directly to buffer with optimized settings while keeping image type as .png
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi = 80, pad_inches=0.1)
        plt.close()  # Close the figure to free memory
        buffer.seek(0)  # Reset buffer position to beginning
        
        
        # Reset buffer position to beginning
        buffer.seek(0)
        
        # Create filename for reference (but don't save to disk)
        filename = f'aia_{channel}'
        if anomaly_threshold is not None:
            threshold_str = str(abs(anomaly_threshold)).replace('.', '_')
            filename += f'_threshold_{threshold_str}'
        filename += '.png'  
        
        object_name = os.path.join(output_dir, filename)
        
        elapsed = time.time() - start_time
        print(f"Generated vis for ch{channel} in {elapsed:.2f}s")
        
        # Return buffer and object name instead of file path
        return buffer, object_name
        
    except Exception as e:
        print(f"Error in plot_single_channel for {channel}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
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
                    
                    # Plot the background image (no colorbar)
                    plt.imshow(norm_data, cmap='gray', alpha=0.7)  
                
                # Overlay clusters with transparency
                cluster_overlay = np.ma.masked_where(cluster_mask == 0, cluster_mask)
                clust_img = plt.imshow(cluster_overlay, cmap=cluster_cmap, alpha=0.6)
                
                # Only add colorbar for clusters
                plt.colorbar(clust_img, label='Cluster ID')
                
                plt.title(f'Channel {channel} with Clusters (Threshold: {threshold})')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'channel_{channel}_clusters.png'), 
                            bbox_inches='tight', dpi=150)
                plt.close()
        
        print(f"Saved result visualizations to {output_dir}")
    except Exception as e:
        print(f"Error in plot_results: {e}")