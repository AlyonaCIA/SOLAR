import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

matplotlib.use("Agg")  # Non-interactive backend for server environment


def detect_anomalies_isolation_forest(data: np.ndarray, contamination: float) -> np.ndarray:
    """Detects anomalies using Isolation Forest.

    Args:
        data: Feature data of shape (n_samples, n_features).
        contamination: Proportion of outliers in the data.

    Returns:
        Anomaly scores. Lower scores indicate more anomalous points.
    """
    print(f"\tFitting Isolation Forest (contamination={contamination}) on data shape: {data.shape}\n")
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_forest.fit(data)
    return iso_forest.decision_function(data)


def perform_kmeans_clustering(
    data: np.ndarray, anomaly_scores: np.ndarray, threshold: float, n_clusters: int, random_state: int = 42
) -> tuple[np.ndarray, float]:
    """Performs K-Means clustering on anomalous data points.

    Args:
        data: Feature data of shape (n_samples, n_features).
        anomaly_scores: Anomaly scores for each sample.
        threshold: Threshold value to determine anomalies.
        n_clusters: Number of clusters to form.
        random_state: Seed for reproducibility.

    Returns:
        Tuple containing:
        - labels: Cluster labels for each sample (-1 for non-anomalies)
        - inertia: Sum of squared distances to closest cluster center
    """
    # Filter data to include only anomalies
    anomaly_mask = anomaly_scores <= threshold
    anomalous_data = data[anomaly_mask]

    if anomalous_data.shape[0] < n_clusters:
        print(f"Warning: Not enough anomalies ({anomalous_data.shape[0]}) for {n_clusters} clusters.")
        if anomalous_data.shape[0] == 0:
            return np.array([]), np.inf
        # Reduce number of clusters if needed
        n_clusters = max(1, min(n_clusters, anomalous_data.shape[0] // 2))
        print(f"Reduced to {n_clusters} clusters.")

    print(f"Performing K-Means (k={n_clusters}) on {anomalous_data.shape[0]} anomalous data points")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",  # Use 'auto' for default in recent sklearn
    )

    # Perform clustering on anomalous data
    kmeans.fit(anomalous_data)

    # Create result array with -1 for non-anomalies
    labels = np.full(data.shape[0], -1)
    labels[anomaly_mask] = kmeans.labels_

    print(f"\tK-Means inertia: {kmeans.inertia_:.2f}")
    return labels, kmeans.inertia_


def create_cluster_mask(
    anomaly_mask: np.ndarray, labels: np.ndarray, valid_pixel_mask: np.ndarray, image_size: int
) -> tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels.

    Args:
        anomaly_mask: 2D boolean mask of anomalies
        labels: Cluster labels for valid pixels (-1 for non-anomalies)
        valid_pixel_mask: 1D boolean mask of valid pixels
        image_size: Size of the square image

    Returns:
        Tuple containing:
        - cluster_mask_2d: 2D mask with cluster labels (0 for no cluster)
        - cluster_cmap: Colormap for visualizing clusters
        - cluster_patches: Legend patches for clusters
        - n_clusters: Number of clusters found
    """
    print("\t" + "-" * 20)
    print("\t" + "Inside create_cluster_mask:")

    if anomaly_mask is None or labels is None or valid_pixel_mask is None:
        print("Warning: Inputs to create_cluster_mask are invalid. Returning empty cluster map.")
        return np.zeros((image_size, image_size), dtype=int), matplotlib.colors.ListedColormap([]), [], 0

    print(f"\t\tInput anomaly_mask shape: {anomaly_mask.shape} (Sum: {np.sum(anomaly_mask)})")
    print(f"\t\tInput labels length: {len(labels)}")
    print(f"\t\tInput valid_pixel_mask shape: {valid_pixel_mask.shape} (Sum: {np.sum(valid_pixel_mask)})")

    cluster_mask_2d = np.zeros((image_size, image_size), dtype=int)

    # Get valid anomaly indices
    anomaly_indices = np.where(labels >= 0)[0]
    if len(anomaly_indices) == 0:
        print("\t\tNo anomalous points found in labels.")
        return cluster_mask_2d, matplotlib.colors.ListedColormap([]), [], 0

    # Create a 2D mask
    valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))
    valid_indices_2d = np.argwhere(valid_pixel_mask_2d)

    # Get positions of anomalous pixels
    anomaly_positions_2d = valid_indices_2d[anomaly_indices]

    # Get unique labels (excluding -1 which means "not an anomaly")
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)
    print(f"\t\tNumber of unique cluster labels found: {n_clusters}")

    # Define colors for clusters
    cluster_colors = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1f77b4",
    ]

    # If we have more clusters than colors, repeat colors
    if n_clusters > len(cluster_colors):
        print(
            f"Warning: Number of clusters ({n_clusters}) exceeds defined colors ({len(cluster_colors)}). Repeating colors."
        )
        cluster_colors = (cluster_colors * (n_clusters // len(cluster_colors) + 1))[:n_clusters]

    # Create colormap
    cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors[:n_clusters])

    # Assign cluster labels to the 2D mask
    # Add 1 so cluster IDs start from 1 (0 means no cluster)
    for pos, label_idx in zip(anomaly_positions_2d, anomaly_indices):
        y, x = pos
        cluster_id = labels[label_idx]
        if cluster_id >= 0:  # Only assign valid clusters
            cluster_mask_2d[y, x] = cluster_id + 1

    # Create legend patches
    cluster_patches = []
    for i, cluster_idx in enumerate(unique_labels):
        color_idx = i / (n_clusters - 1 if n_clusters > 1 else 1)
        cluster_color = cluster_cmap(color_idx)
        cluster_patches.append(mpatches.Patch(color=cluster_color, label=f"Cluster {int(cluster_idx) + 1}"))

    print(f"\t\tFinal cluster_mask_2d shape: {cluster_mask_2d.shape}, Max value: {np.max(cluster_mask_2d)}")
    print("\t" + "-" * 20)
    return cluster_mask_2d, cluster_cmap, cluster_patches, n_clusters
