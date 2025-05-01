# scripts/run_kmeans_pipeline.py

import argparse
import logging
import os
from typing import Optional

# Configure logging for the script
# This sets up basic logging that will show messages from this script
# and also from the pipeline and utils/plotting modules which use the root
# logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# Import the SolarAnomalyPipeline class
# project_root/
# ├── scripts/
# │   └── run_kmeans_pipeline.py  <-- This file
# └── src/
#     └── Models/
#         └── pipeline.py         <-- The pipeline class
#     └── utils/
#         ├── utils.py
#         └── plotting.py
#     └── __init__.py
#
# If your 'pipeline.py' is directly in 'src/solar/', the import would be
# 'from src.solar.pipeline import SolarAnomalyPipeline'
# Using the structure you showed, the import is from src.Models.pipeline
try:
    # Assuming src is part of the Python path when running from project_root
    from src.Models.pipeline import SolarAnomalyPipeline
    log.info("Successfully imported SolarAnomalyPipeline.")
except ImportError:
    log.critical("Failed to import SolarAnomalyPipeline.")
    log.critical(
        "Please ensure you are running this script from the project root "
        "directory (where 'src' and 'scripts' are located)."
    )
    log.critical(
        "Example execution: python3 scripts/run_kmeans_pipeline.py ..."
    )
    exit(1)  # Exit if the necessary class cannot be imported


def main():
    """Parses command-line arguments and runs the Solar Anomaly Detection Pipeline
    specifically configured for K-Means clustering."""
    parser = argparse.ArgumentParser(
        description=("Run SDO/AIA Anomaly Detection Pipeline with K-Means "
                     "Clustering")
    )
    # Using default paths relative to where the script might be run or
    # assuming they are set appropriately.
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Data/sdo_data",
        # Default path adjusted
        help=("Path to the directory containing SDO/AIA channel "
              "subdirectories.")
    )
    # Default channels for the 9 EUV ones
    parser.add_argument(
        "--channels",
        type=str,
        nargs='+',
        default=['94', '131', '171', '193', '211', '233', '304', '335', '700'],
        help=("List of AIA channel wavelengths (e.g., '94' '131'). Default "
              "uses all 9 main EUV channels.")
    )
    parser.add_argument(
        "--anomaly_thresholds",
        type=float,
        nargs='+',
        default=[0.1],
        help=("Threshold(s) for anomaly detection scores. Lower values are "
              "more sensitive.")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_figures",
        help="Directory to save output figures and results."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help=("Size to resize images to (square). Set to -1 to use original "
              "image size.")
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Estimated proportion of anomalies for Isolation Forest."
    )
    # n_clusters default is 7 as per your original K-Means code
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=7,
        help="Number of clusters for K-Means clustering."
    )
    # Added max_k argument if you still want the Elbow method logic,
    # but it's not currently used in the main pipeline run method.
    # If needed, the pipeline class would need a method to run Elbow.
    # For now, keep it just in case.
    parser.add_argument(
        "--max_k",
        type=int,
        default=10,
        help=("Maximum number of clusters to test with Elbow method "
              "(if used).")
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help=("Random seed for reproducibility across Isolation Forest and "
              "K-Means.")
    )

    args = parser.parse_args()

    # --- Data Path Check (Optional but Recommended) ---
    # Check if the default data directory exists, otherwise prompt user or exit
    if args.data_dir == "Data/sdo_data" and not os.path.isdir(args.data_dir):
        log.warning(f"Default data directory '{args.data_dir}' not found.")
        # In a robust script, you might ask for input or try other paths
        # For this example, we'll let the pipeline class handle the
        # FileNotFoundError
        pass  # Let the pipeline class handle the missing directory error

    # Convert image_size argument to None if specified as -1 to indicate
    # original size. The pipeline class handles None for original size.
    image_size_for_pipeline: Optional[int]
    if args.image_size == -1:
        image_size_for_pipeline = None
        log.info("Image size set to None, processing images at original size.")
    elif args.image_size is not None and args.image_size <= 0:
        log.error(f"Invalid image size specified: {args.image_size}. "
                  "Must be positive or -1.")
        exit(1)
    else:
        image_size_for_pipeline = args.image_size
        log.info(
            f"Images will be resized to: {image_size_for_pipeline}x"
            f"{image_size_for_pipeline}"
        )

    log.info("--- Starting K-Means Anomaly Detection Pipeline ---")
    log.info(f"Arguments received: {vars(args)}")

    try:
        # Instantiate the Solar Anomaly Pipeline class
        # This is where the orchestration begins
        pipeline = SolarAnomalyPipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            channels=args.channels,
            image_size=image_size_for_pipeline,
            contamination=args.contamination,
            n_clusters=args.n_clusters,
            cluster_method='KMeans',
            random_state=args.random_state
            # Note: max_k is not passed to the pipeline class in this K-Means
            # specific version, as the Elbow method is not part of the main
            # run sequence.
        )

        # Run the pipeline with the specified anomaly thresholds
        # The run method handles the sequence of steps: load, preprocess,
        # detect, cluster, plot
        results = pipeline.run(args.anomaly_thresholds)

        # --- Process and Print the Results Summary ---
        print("\n--- Pipeline Run Summary ---")
        print(f"Status: {results.get('status', 'Unknown')}")
        if results.get("status") == "error":
            print(
                f"Message: {results.get('message', 'No error message provided.')}"
            )
        else:
            print(f"Data directory: {pipeline.data_dir}")
            print(f"Output directory: {pipeline.output_dir}")
            print(
                f"Channels successfully processed: "
                f"{pipeline._loaded_channel_names}"
            )
            print(f"Final image shape: {pipeline._image_shape}")
            print(
                f"Total valid pixels after masking: "
                f"{pipeline._total_valid_pixels}"
            )
            print(
                f"Clustering Method: {pipeline.cluster_method.capitalize()}"
            )
            print(f"N Clusters Attempted: {pipeline.n_clusters}")
            print(f"Random State: {pipeline.random_state}")

            print("\nResults per Anomaly Threshold:")
            for threshold in args.anomaly_thresholds:
                threshold_results = results.get(threshold)
                if threshold_results:
                    print(f"  Threshold {threshold:.2f}:")
                    print(
                        " Anomalous Pixels: "
                        f"{threshold_results.get('anomalous_pixels_count', 'N/A')}"
                        " / "
                        f"{threshold_results.get(
                            'total_pixels_in_image_grid', 'N/A')} "
                        f"({threshold_results.get(
                            'anomaly_percentage_of_total', 0.0):.2f}%)"
                    )
                    n_clusters_found = threshold_results.get(
                        'n_clusters_found', 'N/A'
                    )
                    print(f"    Clusters Found: {n_clusters_found}")
                    clustering_inertia = threshold_results.get('clustering_inertia')
                    if clustering_inertia is not None:
                        print(
                            f"    Clustering Inertia: {clustering_inertia:.2f}"
                        )
                    if threshold_results["cluster_stats"]:
                        print("    Cluster Stats:")
                        for stats in threshold_results["cluster_stats"]:
                            print(
                                f"      Cluster "
                                f"{stats.get('cluster_index', 'N/A')}: "
                                f"{stats.get('pixel_count', 'N/A')} pixels "
                                f"({stats.get('percentage_of_anomalies', 0.0):.2f}%)"
                            )
                    else:
                        print(
                            "    No cluster stats available "
                            "(e.g., no anomalies found for this threshold)."
                        )
                    plot_path = threshold_results.get('plot_path', 'N/A')
                    print(f"    Plot saved to: {plot_path}")
                else:
                    print(
                        f"  Threshold {threshold:.2f}: No results found "
                        "(e.g., skipped or failed)."
                    )

    except FileNotFoundError as e:
        log.critical(
            f"Pipeline failed because a required data directory was not found: "
            f"{e}"
        )
    except ValueError as e:
        log.critical(
            f"Pipeline failed due to a configuration or data error: {e}"
        )
    except RuntimeError as e:
        log.critical(
            f"Pipeline failed due to a runtime issue: {e}"
        )
    except Exception as e:
        log.critical(
            f"An unexpected error caused the pipeline to fail: {e}",
            exc_info=True
        )

    log.info("--- K-Means Anomaly Detection Pipeline Finished ---")


if __name__ == "__main__":
    main()
