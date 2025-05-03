from typing import Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

matplotlib.use('Agg')


def detect_anomalies_isolation_forest(
    data: np.ndarray, contamination: float
) -> np.ndarray:
    """Detects anomalies using Isolation Forest."""
    if data.shape[0] == 0: # Handle empty input data
        print("Warning: Cannot detect anomalies on empty dataset.")
        return np.array([])
    iso_forest = IsolationForest(
        contamination=contamination, random_state=42, n_jobs=-1 # Use n_jobs=-1 for potential speedup
    )
    print(f"\tFitting Isolation Forest (contamination={contamination}) on data shape: {data.shape}\n")
    iso_forest.fit(data)
    return iso_forest.decision_function(data)








def perform_kmeans_clustering(
    data: np.ndarray, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """Performs K-Means clustering on the given data.

    Parameters
    ----------
    data : np.ndarray
        Data to be clustered of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters to form.
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    Tuple[np.ndarray, float]
        labels : np.ndarray
            Cluster labels assigned to each sample.
        inertia : float
            Sum of squared distances of samples to their closest cluster center.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20
    )
    kmeans.fit(data)
    return kmeans.labels_, kmeans.inertia_

def perform_kmeans_clustering(
    data: np.ndarray, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """Performs K-Means clustering."""
    if data.shape[0] == 0: # Handle empty input data
        print("Warning: Cannot perform K-Means on empty dataset.")
        return np.array([]), np.inf # Return empty labels and infinite inertia
    print(f"Performing K-Means (k={n_clusters}) on data shape: {data.shape}")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto' # Use 'auto' for default (10 in recent sklearn)
    )
    kmeans.fit(data)
    print(f"\tK-Means inertia: {kmeans.inertia_:.2f}")
    return kmeans.labels_, kmeans.inertia_







def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray, # This mask corresponds to the *flattened* scaled data
    image_size: int
) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels."""
    print("\t"+"-" * 20)
    print("\t"+"Inside create_cluster_mask:")
    # anomaly_mask is the 2D boolean mask from thresholding anomaly scores
    # labels correspond to the *anomalous* pixels only
    # valid_pixel_mask is the 1D boolean mask where True means the pixel was valid *before* anomaly detection

    if anomaly_mask is None or labels is None or valid_pixel_mask is None:
         print("Warning: Inputs to create_cluster_mask are invalid. Returning empty cluster map.")
         return np.zeros((image_size, image_size), dtype=int), matplotlib.colors.ListedColormap([]), [], 0

    print(f"\t\tInput anomaly_mask shape: {anomaly_mask.shape} (Sum: {np.sum(anomaly_mask)})")
    print(f"\t\tInput labels length: {len(labels)}")
    print(f"\t\tInput valid_pixel_mask shape: {valid_pixel_mask.shape} (Sum: {np.sum(valid_pixel_mask)})")

    cluster_mask_2d = np.zeros((image_size, image_size), dtype=int)
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])
    cluster_patches = []

    # Check if there are any anomaly labels to process
    if len(labels) > 0 and np.any(anomaly_mask):
        # Ensure labels correspond only to the anomalous pixels
        num_anomalous_pixels_in_anomaly_mask = np.sum(anomaly_mask)

        # We need to map the labels back to the 2D grid.
        # The labels correspond to the `anomaly_intensity_features` which were derived
        # from `prepared_data[anomaly_indices]`.
        # `prepared_data` corresponds to `reshaped_data[valid_pixel_mask]`.

        # Create a full-size array for labels, initially zero or a placeholder
        # Map 1D valid_pixel_mask to 2D
        valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))

        # Find the 2D indices of the pixels that were both valid AND anomalous
        valid_and_anomalous_indices_2d = np.argwhere(valid_pixel_mask_2d & anomaly_mask)

        if len(valid_and_anomalous_indices_2d) != len(labels):
             print(f"Warning: Mismatch between number of valid+anomalous pixels ({len(valid_and_anomalous_indices_2d)}) and number of labels ({len(labels)}). Check logic.")
             # Attempt to proceed if possible, otherwise return empty
             if len(valid_and_anomalous_indices_2d) == 0 :
                  return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters


        n_clusters = len(np.unique(labels))
        print(f"\t\tNumber of unique cluster labels found: {n_clusters}")

        # Define colors
        cluster_colors = [
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4' # Added more colors
        ]
        # Handle case where n_clusters might exceed defined colors
        if n_clusters > len(cluster_colors):
            print(f"Warning: Number of clusters ({n_clusters}) exceeds defined colors ({len(cluster_colors)}). Repeating colors.")
            cluster_colors = (cluster_colors * (n_clusters // len(cluster_colors) + 1))[:n_clusters]

        cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors)

        # Assign cluster labels to the corresponding 2D positions
        # Add 1 to labels so cluster indices start from 1 (0 means no cluster)
        cluster_mask_2d[tuple(valid_and_anomalous_indices_2d.T)] = labels + 1

        # Create legend patches
        for cluster_idx in range(n_clusters):
            cluster_color = cluster_cmap(cluster_idx / (n_clusters -1 if n_clusters > 1 else 1)) # Normalize index correctly
            cluster_patches.append(
                mpatches.Patch(
                    color=cluster_color,
                    label=f'Cluster {cluster_idx + 1}'
                )
            )
    else:
        print("No anomaly labels or no anomalous pixels in mask. No clusters to map.")


    print(f"\t\tFinal cluster_mask_2d shape: {cluster_mask_2d.shape}, Max value: {np.max(cluster_mask_2d)}")
    print("\t"+"-" * 20)
    return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters
