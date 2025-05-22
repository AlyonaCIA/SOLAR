import os
import requests
import datetime
import logging
from flask import Flask, request, jsonify
from google.cloud import storage
from apscheduler.schedulers.background import BackgroundScheduler
import json
import time
import threading

# Locks to prevent concurrent processing of the same timestamp
processing_locks = {}
processing_lock = threading.Lock()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Constants
SOLAR_API_URL = "https://solar-api-865605005704.us-central1.run.app"
GCS_BUCKET_NAME = "mlopsdev2-solar-images"
PROCESSED_IMAGES_PREFIX = "results/fits"

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# Initialize scheduler without the job for now
scheduler = BackgroundScheduler()

def format_iso_time(dt=None):
    """Format datetime to ISO 8601 format with Z suffix with one-day lag"""
    if dt is None:
        dt = datetime.datetime.utcnow()
    # Apply one-day lag (process yesterday's data)
    dt = dt - datetime.timedelta(days=1)
    # Round to the nearest hour for consistency
    dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def process_solar_images(timestamp=None):
    """Process solar images for a given timestamp or current hour"""
    if timestamp is None:
        timestamp = format_iso_time()
    
    # Check if we're already processing this timestamp
    with processing_lock:
        if timestamp in processing_locks:
            logger.info(f"Already processing timestamp: {timestamp}. Skipping duplicate request.")
            return {"status": "skipped", "timestamp": timestamp, "reason": "already_processing"}
        else:
            processing_locks[timestamp] = True

    # Call the FITS analysis endpoint
    try:
        start_time = time.time()
        response = requests.post(
            f"{SOLAR_API_URL}/start-fits-analysis",
            json={"timestamp": timestamp}
        )
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("job_id")
        
        if not job_id:
            logger.error(f"No job_id in response: {job_data}")
            return {"status": "error", "timestamp": timestamp, "error": "No job ID received"}
        
        logger.info(f"Started job {job_id} for timestamp {timestamp}")
        
        # Poll the job status until complete or timeout
        timeout = 3600  # 60 minutes timeout
        polling_interval = 15  # seconds between status checks
        elapsed_time = 0
        
        while elapsed_time < timeout:
            time.sleep(polling_interval)
            elapsed_time = time.time() - start_time
            
            # Check job status
            status_response = requests.get(f"{SOLAR_API_URL}/job-status/{job_id}")
            status_response.raise_for_status()
            status_data = status_response.json()
            
            job_status = status_data.get("status")
            logger.info(f"Job {job_id} status: {job_status}, elapsed time: {elapsed_time:.1f}s")
            
            if job_status == "COMPLETED":
                result = status_data.get("result", {})
                
                # Create metadata with processed image info
                metadata = {
                    "timestamp": timestamp,
                    "processed_at": datetime.datetime.utcnow().isoformat(),
                    "job_id": job_id,
                    "channels": result.get("processed_channels", []),
                    "images": result.get("images", [])
                }
                
                # Save metadata to GCS
                metadata_blob = bucket.blob(f"{PROCESSED_IMAGES_PREFIX}/{timestamp}/metadata.json")
                metadata_blob.upload_from_string(
                    json.dumps(metadata, indent=2),
                    content_type="application/json"
                )
                
                logger.info(f"Successfully processed images for {timestamp}")
                return {
                    "status": "success", 
                    "timestamp": timestamp,
                    "channels": metadata["channels"],
                    "image_count": len(metadata["images"])
                }
            
            elif job_status == "FAILED":
                error = status_data.get("error", "Unknown error")
                logger.error(f"Job {job_id} failed: {error}")
                return {"status": "error", "timestamp": timestamp, "job_id": job_id, "error": error}
            
            # Otherwise continue polling
        
        # If we get here, we've timed out
        logger.error(f"Job {job_id} timed out after {timeout} seconds")
        return {"status": "error", "timestamp": timestamp, "job_id": job_id, "error": "timeout"}
        
    except Exception as e:
        logger.exception(f"Error processing images for {timestamp}: {e}")
        return {"status": "error", "timestamp": timestamp, "error": str(e)}

    finally:
        # Clean up the lock when done
        logger.info(f"Processing completed for {timestamp} after {elapsed_time:.1f}s")
        with processing_lock:
            if timestamp in processing_locks:
                del processing_locks[timestamp]

@app.route("/")
def index():
    """Root endpoint with links to other endpoints"""
    return jsonify({
        "status": "ok",
        "endpoints": {
            "list": "/list\n",
            "latest": "/latest\n", 
            "images": "/images/<timestamp>\n",
            "healthz": "/healthz\n",
            "process": "/process\n",
            "start_scheduler": "/start-scheduler\n",
            "start_processing": "/start-processing\n"
        }
    })

@app.route("/process", methods=["POST"])
def process_endpoint():
    """Endpoint to trigger image processing for a specific time or current hour"""
    data = request.get_json() or {}
    timestamp = data.get("timestamp", format_iso_time())
    result = process_solar_images(timestamp)
    return jsonify(result)

@app.route("/list", methods=["GET"])
def list_processed():
    """List all processed timestamps with debug info"""
    debug_info = {
        "bucket": GCS_BUCKET_NAME,
        "prefix": f"{PROCESSED_IMAGES_PREFIX}/",
    }
    
    # Try to list some blobs directly to confirm access
    try:
        all_blobs = list(storage_client.list_blobs(
            GCS_BUCKET_NAME, 
            prefix=f"{PROCESSED_IMAGES_PREFIX}/",
            max_results=10
        ))
        debug_info["raw_blobs"] = [b.name for b in all_blobs]
    except Exception as e:
        debug_info["error"] = str(e)
        
    # Try with prefixes as before
    blobs = storage_client.list_blobs(
        GCS_BUCKET_NAME, 
        prefix=f"{PROCESSED_IMAGES_PREFIX}/", 
        delimiter="/"
    )
    
    # Iterate to populate prefixes
    for _ in blobs:
        pass

    timestamps = []
    prefixes_list = []
    for prefix in blobs.prefixes:
        prefixes_list.append(prefix)
        # Extract timestamp from the prefix path
        timestamp = prefix.strip("/").split("/")[-1]
        timestamps.append(timestamp)
    
    debug_info["prefixes"] = prefixes_list
    timestamps.sort(reverse=True)
    
    return jsonify({
        "status": "success",
        "count": len(timestamps),
        "timestamps": timestamps,
        "debug_info": debug_info
    })

@app.route("/images/<timestamp>", methods=["GET"])
def get_processed_images(timestamp):
    """Get processed images for a specific timestamp"""
    # List all objects under the timestamp prefix to find available files
    blobs = list(storage_client.list_blobs(
        GCS_BUCKET_NAME,
        prefix=f"{PROCESSED_IMAGES_PREFIX}/{timestamp}/"
    ))
    
    if not blobs:
        return jsonify({
            "status": "error", 
            "error": f"No processed images found for timestamp {timestamp}"
        }), 404
    
    # Extract image paths and group by threshold
    image_paths = [b.name for b in blobs if b.name.endswith('.png')]
    
    # Group by wavelength channel and threshold
    images_by_channel = {}
    for path in image_paths:
        # Parse the filename to extract channel and threshold
        filename = path.split('/')[-1]
        if filename.startswith('aia_'):
            parts = filename.replace('.png', '').split('_')
            channel = parts[1]
            if len(parts) > 2:
                threshold = parts[3]
                if threshold not in images_by_channel:
                    images_by_channel[threshold] = {}
                images_by_channel[threshold][channel] = {
                    "url": f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{path}",
                    "path": path
                }
    
    return jsonify({
        "status": "success",
        "timestamp": timestamp,
        "images_by_threshold": images_by_channel
    })

@app.route("/latest", methods=["GET"])
def get_latest():
    """Get the latest processed images"""
    # List some raw blobs for debugging
    debug_info = {
        "raw_blobs": [b.name for b in list(bucket.list_blobs(
            prefix=f"{PROCESSED_IMAGES_PREFIX}/",
            max_results=5
        ))]
    }

    # Important: You need to explicitly iterate over the prefixes
    prefixes = []
    blob_iter = storage_client.list_blobs(
        GCS_BUCKET_NAME, 
        prefix=f"{PROCESSED_IMAGES_PREFIX}/", 
        delimiter="/"
    )
    # Iterate to populate prefixes
    for _ in blob_iter:
        pass
    
    prefixes = list(blob_iter.prefixes)
    debug_info["prefixes"] = list(prefixes)
    
    if not prefixes:
        return jsonify({
            "status": "error",
            "error": "No processed images found",
            "debug_info": debug_info
        }), 404
    
    # Sort prefixes to get the latest timestamp
    latest_prefix = sorted(prefixes)[-1]
    timestamp = latest_prefix.strip("/").split("/")[-1]
    
    # Forward to the specific timestamp endpoint
    return get_processed_images(timestamp)

@app.route('/healthz')
def healthz():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"})

@app.route('/start-scheduler', methods=['GET'])
def start_scheduler_route():
    if not scheduler.running:
        scheduler.start()
        return jsonify({"status": "scheduler started"})
    return jsonify({"status": "scheduler already running"})

@app.route('/start-processing', methods=['GET'])
def start_processing_route():
    """Endpoint to trigger processing in background"""
    # Get a timestamp to use for this job
    timestamp = format_iso_time()
    
    # Check if already processing before starting the thread
    with processing_lock:
        if timestamp in processing_locks:
            return jsonify({
                "status": "skipped", 
                "timestamp": timestamp,
                "reason": "already_processing"
            })
    
    # Start processing in a background thread
    thread = threading.Thread(target=process_solar_images, args=(timestamp,))
    thread.daemon = True
    thread.start()
    
    # Return immediately with the timestamp being processed
    return jsonify({
        "status": "processing_started", 
        "timestamp": timestamp,
        "message": "Processing started in background"
    })

def run_initial_tasks():
    """Run initial tasks in background after server starts"""
    logger.info("Starting initial processing in background thread")
    try:
        # Process current hour
        process_solar_images()
    except Exception as e:
        logger.error(f"Error in initial processing: {e}")

if __name__ == "__main__":
    # Start the Flask app immediately
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)