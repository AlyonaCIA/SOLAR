# scripts/run_kmeans_pipeline.py
"""Executes the Solar Anomaly Detection pipeline using KMeans clustering.

This script parses command-line arguments to configure and run the pipeline defined in
src.solar.pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# --- Add project root to sys.path ---
# This allows importing from 'src' when running the script from the root directory.
# Assumes the script is located in project_root/scripts/
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        # Optional: print a debug message
        # print(f"DEBUG: Added '{PROJECT_ROOT}' to sys.path")
except NameError:
    # Handle cases where __file__ might not be defined (e.g., interactive)
    PROJECT_ROOT = Path(".").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging for the script and imported modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --- Import the main pipeline class ---
try:
    from src.solar.pipeline import SolarAnomalyPipeline

    log.info("Successfully imported SolarAnomalyPipeline.")
except ImportError as e:
    log.critical(f"Failed to import SolarAnomalyPipeline: {e}")
    log.critical(f"Current sys.path: {sys.path}")
    log.critical(
        "Ensure you are running this script from the project root directory "
        "(e.g., 'python3 scripts/run_kmeans_pipeline.py') OR that the "
        "project is installed ('pip install -e .')."
    )
    exit(1)  # Exit if the necessary class cannot be imported


def _parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description=("Run SDO/AIA Anomaly Detection Pipeline with K-Means Clustering"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data/sdo_data",
        help="Path to the directory containing SDO/AIA channel subdirectories.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=["94", "131", "171", "193", "211", "233", "304", "335", "700"],
        help="List of AIA channel wavelengths (e.g., '94' '131').",
    )
    parser.add_argument(
        "--anomaly_thresholds",
        type=float,
        nargs="+",
        default=[0.1],
        help="Threshold(s) for anomaly detection scores. Lower is more sensitive.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_figures_kmeans",  # Default specific to this script
        help="Directory to save output figures and results.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Size to resize images (square). Set to -1 for original size.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Estimated proportion of anomalies for Isolation Forest.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=7,
        help="Number of clusters for K-Means clustering.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (Isolation Forest, K-Means).",
    )
    # Optional: max_k if Elbow method integration is planned later
    # parser.add_argument(
    #     "--max_k", type=int, default=10,
    #     help="Max clusters for Elbow method (if used)."
    # )

    return parser.parse_args()


def _get_image_size(args_image_size: int) -> int | None:
    """Determines the image size for the pipeline based on arguments."""
    if args_image_size == -1:
        log.info("Image size set to None, processing images at original size.")
        return None
    elif args_image_size <= 0:
        log.error(f"Invalid image size: {args_image_size}. Must be positive or -1.")
        exit(1)
    else:
        log.info(f"Images will be resized to: {args_image_size}x{args_image_size}")
        return args_image_size


def _print_results_summary(results: dict, pipeline: SolarAnomalyPipeline, thresholds: list):
    """Prints a formatted summary of the pipeline results."""
    print("\n--- Pipeline Run Summary ---")
    status = results.get("status", "Unknown")
    print(f"Status: {status}")
    if status == "error":
        print(f"Message: {results.get('message', 'No error message provided.')}")
        return

    # Safely access potentially protected attributes for info
    loaded_channels = getattr(pipeline, "_loaded_channel_names", [])
    img_shape = getattr(pipeline, "_image_shape", "N/A")
    valid_pixels = getattr(pipeline, "_total_valid_pixels", "N/A")

    print(f"Data directory: {pipeline.data_dir}")
    print(f"Output directory: {pipeline.output_dir}")
    print(f"Channels successfully processed: {loaded_channels}")
    print(f"Final image shape: {img_shape}")
    print(f"Total valid pixels after masking: {valid_pixels}")
    print(f"Clustering Method: {pipeline.cluster_method.capitalize()}")
    print(f"N Clusters Attempted: {pipeline.n_clusters}")
    print(f"Random State: {pipeline.random_state}")

    print("\nResults per Anomaly Threshold:")
    for threshold in thresholds:
        res = results.get(threshold)
        if res:
            print(f"  Threshold {threshold:.2f}:")
            anom_count = res.get("anomalous_pixels_count", "N/A")
            total_pix = res.get("total_pixels_in_image_grid", "N/A")
            perc_total = res.get("anomaly_percentage_of_total", 0.0)
            n_clusters_found = res.get("n_clusters_found", "N/A")
            inertia = res.get("clustering_inertia")
            plot_path = res.get("plot_path", "N/A")
            cluster_stats = res.get("cluster_stats", [])

            print(f"    Anomalous Pixels: {anom_count} / {total_pix} ({perc_total:.2f}%)")
            print(f"    Clusters Found: {n_clusters_found}")
            if inertia is not None:
                print(f"    Clustering Inertia: {inertia:.2f}")

            if cluster_stats:
                print("    Cluster Stats:")
                for stats in cluster_stats:
                    idx = stats.get("cluster_index", "?")
                    count = stats.get("pixel_count", "?")
                    perc = stats.get("percentage_of_anomalies", 0.0)
                    print(f"      Cluster {idx}: {count} pixels ({perc:.2f}%)")
            else:
                print("    No cluster stats (e.g., no anomalies found).")

            print(f"    Plot saved to: {plot_path}")
        else:
            print(f"  Threshold {threshold:.2f}: No results (skipped/failed).")


def main():
    """Main execution function."""
    args = _parse_arguments()
    image_size_for_pipeline = _get_image_size(args.image_size)

    # Optional: Check if data_dir exists here for early feedback
    if not os.path.isdir(args.data_dir):
        log.warning(f"Data directory '{args.data_dir}' not found. Pipeline will likely fail.")
        # Consider exiting here if you want stricter checks upfront:
        # log.critical(f"Data directory '{args.data_dir}' not found. Exiting.")
        # exit(1)

    log.info("--- Starting K-Means Anomaly Detection Pipeline ---")
    log.info(f"Arguments received: {vars(args)}")

    try:
        pipeline = SolarAnomalyPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            channels=args.channels,
            image_size=image_size_for_pipeline,
            contamination=args.contamination,
            n_clusters=args.n_clusters,
            cluster_method="KMeans",  # Specific to this script
            random_state=args.random_state,
        )

        results = pipeline.run(args.anomaly_thresholds)
        _print_results_summary(results, pipeline, args.anomaly_thresholds)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.critical(f"Pipeline execution failed: {e}")
        # Optionally print traceback for these expected errors if helpful
        # import traceback
        # traceback.print_exc()
    except Exception as e:
        log.critical(
            f"An unexpected error occurred during pipeline execution: {e}",
            exc_info=True,  # Log traceback for unexpected errors
        )

    log.info("--- K-Means Anomaly Detection Pipeline Finished ---")


if __name__ == "__main__":
    main()
