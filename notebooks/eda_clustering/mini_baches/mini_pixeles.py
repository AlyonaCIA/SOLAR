import argparse
import os

import matplotlib
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")  # Use Agg backend for saving files


# --- Helper Functions ---
def load_fits_data(channel_dir: str) -> tuple[np.ndarray, dict]:
    """Loads FITS data and metadata from a single channel directory."""
    fits_files = [f for f in os.listdir(channel_dir) if f.endswith(".fits")]
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in: {channel_dir}")
    fits_path = os.path.join(channel_dir, fits_files[0])
    aia_map = sunpy.map.Map(fits_path)
    return aia_map.data, aia_map.meta


def create_circular_mask(data: np.ndarray, metadata: dict) -> np.ndarray:
    """Creates a circular mask for the solar disk based on metadata."""
    ny, nx = data.shape
    x_center, y_center = nx // 2, ny // 2
    cdelt1 = metadata.get("cdelt1", 1.0)
    solar_radius_arcsec = metadata.get("rsun_obs", 960.0)
    solar_radius_pixels = int(solar_radius_arcsec / abs(cdelt1))
    y, x = np.ogrid[:ny, :nx]
    distance_from_center = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    return distance_from_center <= solar_radius_pixels


def preprocess_image(data: np.ndarray, mask: np.ndarray, size: int = None) -> np.ndarray:
    """Resizes the image and applies the mask."""
    if size is not None:
        resized_data = resize(data, (size, size), mode="reflect", anti_aliasing=True)
        resized_mask = resize(mask, (size, size), mode="reflect", anti_aliasing=False) > 0.5
        masked_data = resized_data.copy()
        masked_data[~resized_mask] = np.nan
        return masked_data
    masked_data = data.copy()
    masked_data[~mask] = np.nan
    return masked_data


# --- Data Preparation ---
def prepare_data_concatenated(masked_data_list: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenates masked data, handles NaNs, and scales the data."""
    stacked_data = np.stack(masked_data_list, axis=-1)
    reshaped_data = stacked_data.reshape((-1, len(masked_data_list)))
    nan_mask = np.isnan(reshaped_data).any(axis=1)
    cleaned_data = reshaped_data[~nan_mask]
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(cleaned_data)
    return scaled_data, ~nan_mask, nan_mask


# --- Anomaly Detection ---
def detect_anomalies_isolation_forest(data: np.ndarray, contamination: float, random_state: int) -> np.ndarray:
    """Detects anomalies using Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest.fit(data)
    return iso_forest.decision_function(data)


# --- Clustering ---
def perform_kmeans_clustering(data: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[np.ndarray, float]:
    """Performs MiniBatch K-Means clustering."""
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        batch_size=256,
    )
    kmeans.fit(data)
    return kmeans.labels_, kmeans.inertia_


def evaluate_minibatch_kmeans_clustering(
    data: np.ndarray, max_k: int = 10, random_state: int = 42, output_dir: str = "./output_figures", prefix: str = ""
) -> int:
    """Evalúa MiniBatchKMeans para múltiples valores de k usando Inertia (codo) y
    Silhouette Score."""
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)

    print("Evaluando MiniBatchKMeans con método del codo y Silhouette Score...")
    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=10, batch_size=256)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        try:
            sil_score = silhouette_score(data, labels)
            silhouettes.append(sil_score)
        except Exception as e:
            print(f"Silhouette no calculable para k={k}: {e}")
            silhouettes.append(np.nan)

        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

    # Plot Inertia (Elbow)
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("MiniBatchKMeans Elbow Method")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}minibatch_elbow.png"))
    plt.close()

    # Plot Silhouette Score
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, silhouettes, marker="s", color="green")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("MiniBatchKMeans Silhouette Scores")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{prefix}minibatch_silhouette.png"))
    plt.close()

    # Mejor k = el que maximiza Silhouette
    best_k = k_range[np.nanargmax(silhouettes)]
    print(f"\nMejor k según Silhouette Score: {best_k} (score = {np.nanmax(silhouettes):.3f})\n")
    return best_k


def create_cluster_mask(
    anomaly_mask: np.ndarray, labels: np.ndarray, valid_pixel_mask: np.ndarray, image_size: int
) -> tuple[np.ndarray, matplotlib.colors.ListedColormap, list, int]:
    """Creates a 2D cluster mask from anomaly mask and cluster labels."""
    cluster_mask = np.zeros_like(anomaly_mask, dtype=int)
    n_clusters = len(np.unique(labels))
    cluster_colors = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]  # Vivid, distinct colors
    cluster_cmap = matplotlib.colors.ListedColormap(cluster_colors[:n_clusters])
    cluster_patches = [
        mpatches.Patch(color=cluster_cmap(i / n_clusters), label=f"Cluster {i + 1}") for i in range(n_clusters)
    ]

    valid_pixel_mask_2d = valid_pixel_mask.reshape((image_size, image_size))
    anomaly_pixels_indices = np.argwhere(anomaly_mask)
    valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
    pixel_index_map = {tuple(index_2d): i for i, index_2d in enumerate(valid_pixel_indices_2d)}

    valid_anomaly_pixel_indices = [idx for idx in anomaly_pixels_indices if tuple(idx) in pixel_index_map]
    valid_anomaly_pixel_indices = np.array(valid_anomaly_pixel_indices)

    for cluster_idx in range(n_clusters):
        cluster_pixel_indices = valid_anomaly_pixel_indices[labels == cluster_idx]
        cluster_mask[tuple(cluster_pixel_indices.T)] = cluster_idx + 1

    return cluster_mask, cluster_cmap, cluster_patches, n_clusters


# --- Plotting ---
def plot_results(
    masked_data_list: list,
    cluster_mask_global: np.ndarray,
    cluster_cmap_global: matplotlib.colors.ListedColormap,
    n_clusters_global: int,
    cluster_patches_global: list,
    channel_names: list,
    anomaly_threshold: float,
    output_dir: str,
    total_pixels: int,
    anomaly_pixels_count: int,
    cluster_pixels_counts: list[int],
    cluster_anomaly_percentages: list[float],
    clustering_method_name: str = "MiniBatch K-Means",
):
    """Plots and saves anomaly detection and clustering results."""
    num_rows, num_cols = 3, 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 15), dpi=100)
    axes = axes.flatten()

    anomaly_percentage = (anomaly_pixels_count / total_pixels) * 100 if total_pixels > 0 else 0
    fig.suptitle(
        f"{clustering_method_name} Anomaly Clusters in SDO/AIA EUV Channels\n"
        f"Anomaly Threshold: {anomaly_threshold:.2f} | Anomalous Pixels: {anomaly_pixels_count}/{total_pixels} ({
            anomaly_percentage:.2f}%)",
        fontsize=16,
        y=0.98,  # Adjusted suptitle y position slightly up
    )

    for i, (masked_data, channel) in enumerate(zip(masked_data_list, channel_names)):
        if i >= num_rows * num_cols:
            continue

        ax = axes[i]
        ax.imshow(
            masked_data,
            cmap="YlOrBr",
            origin="lower",
            vmin=np.nanpercentile(masked_data, 5),
            vmax=np.nanpercentile(masked_data, 95),
            alpha=0.7,
        )

        if n_clusters_global > 0:
            for cluster_index in range(1, n_clusters_global + 1):
                cluster_mask_channel = cluster_mask_global == cluster_index
                cluster_color = cluster_cmap_global((cluster_index - 1) / n_clusters_global)
                ax.imshow(
                    np.ma.masked_where(~cluster_mask_channel, cluster_mask_global),
                    cmap=matplotlib.colors.ListedColormap([cluster_color]),
                    alpha=0.8,
                    origin="lower",
                    vmin=cluster_index - 0.5,
                    vmax=cluster_index + 0.5,
                )

        title_lines = [f"AIA {channel} Å"]  # Start with channel name
        if cluster_pixels_counts and cluster_anomaly_percentages and cluster_index <= len(cluster_pixels_counts):
            cluster_pixels = cluster_pixels_counts[cluster_index - 1]
            cluster_percentage = cluster_anomaly_percentages[cluster_index - 1]
            # Shorter percentage format
            title_lines.append(f"Cluster {cluster_index + 1}: {cluster_pixels} Pixels ({cluster_percentage:.1f}%)")

        ax.set_title(
            "\n".join(title_lines),  # Multiline title
            color="black",
            fontsize=11,
            pad=8,  # Adjusted fontsize and padding
        )
        ax.axis("off")

    # Legend outside the loop, adjust if needed for multi-channel
    if cluster_patches_global:
        fig.legend(
            handles=cluster_patches_global,
            loc="upper right",
            bbox_to_anchor=(0.95, 0.95),  # Adjusted legend position
            fontsize="small",
            framealpha=0.8,
        )

    for j in range(len(channel_names), num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjusted tight_layout rect and reduced whitespace
    plt.tight_layout(rect=[0, 0, 0.93, 0.95], w_pad=0.1, h_pad=0.1)
    filename = os.path.join(
        output_dir,
        f"minibatch_kmeans_anomalies_threshold_{anomaly_threshold:.2f}_clusters.png",  # More concise filename
    )
    # Increased dpi for better quality
    plt.savefig(filename, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Figure saved to: {filename}")


# --- Main Execution ---
def main():
    """Main function to execute SDO/AIA anomaly detection pipeline."""
    parser = argparse.ArgumentParser(
        description="SDO/AIA Anomaly Detection using Isolation Forest and MiniBatch K-Means"
    )
    parser.add_argument("--data_dir", type=str, default="Data/sdo_data", help="Path to SDO/AIA data directory.")
    parser.add_argument("--channels", type=str, nargs="+", default=None, help="AIA channels (e.g., '94', '131').")
    parser.add_argument("--anomaly_thresholds", type=float, nargs="+", default=[0.1], help="Anomaly threshold(s).")
    parser.add_argument("--output_dir", type=str, default="./output_figures", help="Output directory for figures.")
    parser.add_argument("--image_size", type=int, default=512, help="Resize image size.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Isolation Forest contamination parameter.")
    parser.add_argument("--n_clusters", type=int, default=7, help="Number of clusters for MiniBatchKMeans.")
    parser.add_argument("--max_k", type=int, default=10, help="Max clusters for Elbow method.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no_resize", action="store_true", help="Process images at original size.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    channels = (
        args.channels
        if args.channels
        else [
            d
            for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith("aia_1")
        ]
    )
    if not channels:
        print("No channels found. Exiting.")
        return

    masked_data_list, channel_names = [], []
    image_size_for_processing = args.image_size if not args.no_resize else None
    current_image_size = args.image_size if not args.no_resize else None

    for channel_dir in channels:
        try:
            channel_names.append(channel_dir.split("_")[1])
            data, metadata = load_fits_data(os.path.join(args.data_dir, channel_dir))
            mask = create_circular_mask(data, metadata)
            masked_data = preprocess_image(data, mask, image_size_for_processing)
            masked_data_list.append(masked_data)
            if args.no_resize:
                current_image_size = data.shape[0]
        except Exception as e:
            print(f"Error processing {channel_dir}: {e}")

    if not masked_data_list:
        print("No data loaded. Exiting.")
        return

    prepared_data, valid_pixel_mask, nan_mask = prepare_data_concatenated(masked_data_list)

    anomaly_scores = detect_anomalies_isolation_forest(prepared_data, args.contamination, args.random_state)

    final_image_size = current_image_size or original_image_size  # type: ignore
    anomaly_map_2d = np.full((final_image_size, final_image_size), np.nan)  # type: ignore
    anomaly_map_2d[valid_pixel_mask.reshape((final_image_size, final_image_size))] = anomaly_scores  # type: ignore

    for anomaly_threshold in args.anomaly_thresholds:
        anomaly_mask_global = anomaly_map_2d < anomaly_threshold
        anomaly_pixels_count = np.sum(anomaly_mask_global)
        total_pixels = final_image_size * final_image_size  # type: ignore

        anomaly_pixels_indices = np.argwhere(anomaly_mask_global)
        valid_pixel_mask_2d = ~nan_mask.reshape((final_image_size, final_image_size))  # type: ignore
        valid_pixel_indices_2d = np.argwhere(valid_pixel_mask_2d)
        pixel_index_map = {tuple(idx): i for i, idx in enumerate(valid_pixel_indices_2d)}

        anomaly_intensity_features = np.array(
            [
                prepared_data[pixel_index_map[tuple(idx)]]
                for idx in anomaly_pixels_indices
                if tuple(idx) in pixel_index_map
            ]
        )

        cluster_pixels_counts: list[int] = []
        cluster_anomaly_percentages: list[float] = []

        if len(anomaly_intensity_features):
            best_k = evaluate_minibatch_kmeans_clustering(
                anomaly_intensity_features,
                max_k=args.max_k,
                random_state=args.random_state,
                output_dir=args.output_dir,
                prefix=f"thresh_{anomaly_threshold:.2f}_",
            )

            cluster_labels, _ = perform_kmeans_clustering(anomaly_intensity_features, best_k, args.random_state)

            cluster_mask_global, cluster_cmap_global, cluster_patches_global, n_clusters_global = create_cluster_mask(
                anomaly_mask_global,
                cluster_labels,
                valid_pixel_mask,
                final_image_size,  # type: ignore
            )

            for cluster_index in range(n_clusters_global):
                cluster_pixel_count = np.sum(cluster_mask_global == (cluster_index + 1))
                cluster_pixels_counts.append(cluster_pixel_count)
                cluster_percentage = (cluster_pixel_count / anomaly_pixels_count) * 100 if anomaly_pixels_count else 0
                cluster_anomaly_percentages.append(cluster_percentage)

        plot_results(
            masked_data_list,
            cluster_mask_global,
            cluster_cmap_global,
            n_clusters_global,
            cluster_patches_global,
            channel_names,
            anomaly_threshold,
            args.output_dir,
            total_pixels,
            anomaly_pixels_count,
            cluster_pixels_counts,
            cluster_anomaly_percentages,
            clustering_method_name="MiniBatch K-Means",
        )

    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
