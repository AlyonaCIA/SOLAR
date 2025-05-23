import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple


def plot_elbow_and_silhouette(
    data: np.ndarray, max_k: int, output_dir: str) -> Tuple[int, int, str]:
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = np.argmax(diffs2) + 3
    silhouette_best_k = k_range[np.argmax(silhouette_scores)]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(k_range, inertias, 'o-', label='Inertia (Elbow)', color='blue')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axvline(elbow_k, color='blue', linestyle='--', label=f'Elbow at k={elbow_k}')

    ax2 = ax1.twinx()
    ax2.plot(k_range, silhouette_scores, 's-', color='green', label='Silhouette Score')
    ax2.set_ylabel('Silhouette Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.axvline(silhouette_best_k, color='green', linestyle='--',
                label=f'Silhouette Best k={silhouette_best_k}')

    fig.suptitle('Elbow Method and Silhouette Scores')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    output_path = os.path.join(output_dir, 'elbow_silhouette_plot.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return elbow_k, silhouette_best_k, output_path
