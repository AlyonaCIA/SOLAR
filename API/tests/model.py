import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import RobustScaler
from typing import Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def detect_anomalies_isolation_forest(
    data: np.ndarray, contamination: float
) -> np.ndarray:
    """
    Detects anomalies using Isolation Forest and returns anomaly scores.

    Parameters
    ----------
    data : np.ndarray
        Input data for anomaly detection of shape (n_samples, n_features).
    contamination : float
        The proportion of anomalies in the data.

    Returns
    -------
    np.ndarray
        Anomaly scores. Lower scores indicate higher anomaly likelihood.
    """
    iso_forest = IsolationForest(
        contamination=contamination, random_state=42
    )
    iso_forest.fit(data)
    return iso_forest.decision_function(data)


def perform_kmeans_clustering(
    data: np.ndarray, n_clusters: int, random_state: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Performs K-Means clustering on the given data.

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


def determine_optimal_k_elbow(
    data: np.ndarray, max_k: int = 10, random_state: int = 42
) -> int:
    """
    Uses the Elbow method to determine the optimal number of clusters.

    Parameters
    ----------
    data : np.ndarray
        Data to be clustered of shape (n_samples, n_features).
    max_k : int, optional
        Maximum number of clusters to test, by default 10.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    int
        Estimated optimal number of clusters.
    """
    inertias = []
    for k in range(1, max_k + 1):
        _, inertia = perform_kmeans_clustering(data, k, random_state)
        inertias.append(inertia)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    elbow_plot_path = "elbow_plot.png"
    plt.savefig(elbow_plot_path)
    plt.close()
    print(f"Elbow plot saved to: {elbow_plot_path}")

    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(diffs2) + 2

    return optimal_k


def create_cluster_mask(
    anomaly_mask: np.ndarray,
    labels: np.ndarray,
    valid_pixel_mask: np.ndarray,
    image_size: int
) -> Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """
    Generates a 2D cluster mask showing spatial location of clusters over anomaly pixels.

    Parameters
    ----------
    anomaly_mask : np.ndarray
        2D boolean mask indicating which pixels are considered anomalous.
    labels : np.ndarray
        Cluster labels assigned to each anomaly pixel.
    valid_pixel_mask : np.ndarray
        1D boolean mask indicating which pixels are valid (non-NaN).
    image_size : int
        Size (height/width) of the square image.

    Returns
    -------
    Tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]
        cluster_mask : np.ndarray
            2D array with cluster indices assigned to anomaly pixels.
        cluster_cmap : matplotlib.colors.ListedColormap
            Colormap used to visualize cluster assignments.
        cluster_patches : list
            List of legend patches for labeling clusters in a plot.
        n_clusters : int
            Total number of clusters found.
    """
    print("-" * 20)
    print("Inside create_cluster_mask:")
    print(f"anomaly_mask.shape: {anomaly_mask.shape}")
    print(f"np.sum(anomaly_mask): {np.sum(anomaly_mask)}")

    cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
    n_clusters = 0
    cluster_cmap = matplotlib.colors.ListedColormap([])
    cluster_patches = []

    if len(labels) > 0:
        n_clusters = len(np.unique(labels))
        print(f"Number of clusters (n_clusters): {n_clusters}")

        cluster_colors = [
            '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f'
        ]
        cluster_cmap = matplotlib.colors.ListedColormap(
            cluster_colors[:n_clusters]
        )

        valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))
        anomaly_pixels_indices = np.argwhere(anomaly_mask)
        print(f"len(anomaly_pixels_indices): {len(anomaly_pixels_indices)}")

        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {
            tuple(index_2d): i for i, index_2d in enumerate(valid_pixel_indices_2d)
        }

        valid_anomaly_pixel_indices = []
        for anomaly_pixel_index_2d in anomaly_pixels_indices:
            if tuple(anomaly_pixel_index_2d) in pixel_index_map:
                valid_anomaly_pixel_indices.append(anomaly_pixel_index_2d)
        valid_anomaly_pixel_indices = np.array(valid_anomaly_pixel_indices)
        print(f"len(valid_anomaly_pixel_indices): {len(valid_anomaly_pixel_indices)}")
        print(f"labels.shape: {labels.shape}")

        if len(valid_anomaly_pixel_indices) > 0:
            for cluster_idx in range(n_clusters):
                cluster_pixel_indices = valid_anomaly_pixel_indices[
                    labels == cluster_idx
                ]
                if len(cluster_pixel_indices) > 0:
                    cluster_mask[tuple(cluster_pixel_indices.T)] = (
                        cluster_idx + 1
                    )
                    cluster_color = cluster_cmap(cluster_idx / n_clusters)
                    cluster_patches.append(
                        mpatches.Patch(
                            color=cluster_color,
                            label=f'Cluster {cluster_idx + 1}'
                        )
                    )

    print("-" * 20)
    return cluster_mask, cluster_cmap, cluster_patches, n_clusters
