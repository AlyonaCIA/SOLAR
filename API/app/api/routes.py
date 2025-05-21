import os
import shutil
import uuid
from typing import Union
import re
import datetime
import traceback
import time
import threading

import asyncio
from pathlib import Path

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import necessary modules from sunpy and astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.net import Fido, vso
from sunpy.net import attrs as a

# Import our custom modules
from app.api.pipeline.executor import run_pipeline, run_single_channel_pipeline, save_and_list_raw_fits
from app.api.pipeline.job_manager import job_manager, JobStatus
from app.api.pipeline.background_job import run_in_background
from app.config.settings import config

from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the API router
router = APIRouter()

# Constants
HELIOVIEWER_BASE_URL = "https://api.helioviewer.org"
SOURCE_IDS = [8, 9, 10, 11, 12, 13, 14]
SOURCE_ID_TO_CHANNEL_MAP = {8: '94', 9: '131', 10: '171', 11: '193', 12: '211', 13: '304', 14: '335'}
GCS_BUCKET_NAME = "mlopsdev2-solar-images"
LOCAL_STORAGE_DIR = "local_storage"

# Define the center coordinates and the square size in arcseconds
center_x, center_y = 0, 0  # arcseconds
half_square_size = 1210  # arcseconds

# Define the date format
date_format = "%Y-%m-%dT%H:%M:%S.%f"

# Try to initialize GCS client, but handle the case when credentials aren't available
USE_GCS = False
bucket = None
try:
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    USE_GCS = True
    print("Google Cloud Storage initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize GCS client: {e}")
    print("Will save files locally instead")
    # Create local storage directory
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

# Helper function to save files to either GCS or local storage
def save_to_storage(file_path: str, object_name: str) -> dict:
    """
    Save a file to either GCS or local storage, depending on availability.
    Returns a dictionary with url, path, and storage_type.
    """
    if USE_GCS and bucket:
        try:
            # Upload to GCS
            blob = bucket.blob(object_name)
            blob.upload_from_filename(file_path)
            blob.make_public()
            return {
                "url": blob.public_url,
                "path": object_name,
                "storage_type": "gcs"
            }
        except Exception as e:
            print(f"Error uploading to GCS: {e}, falling back to local storage")
            # Fall back to local storage if GCS upload fails
            pass
    
    # Use local storage
    local_path = os.path.join(LOCAL_STORAGE_DIR, object_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy(file_path, local_path)
    return {
        "url": f"file://{os.path.abspath(local_path)}",
        "path": local_path,
        "storage_type": "local"
    }

# New function to save directly from memory buffer
def save_buffer_to_storage(buffer, object_name: str) -> dict:
    """
    Save a memory buffer directly to either GCS or local storage.
    Returns a dictionary with url, path, and storage_type.
    """
    if USE_GCS and bucket:
        try:
            # Upload to GCS directly from buffer
            blob = bucket.blob(object_name)
            blob.upload_from_file(buffer)
            blob.make_public()
            return {
                "url": blob.public_url,
                "path": object_name,
                "storage_type": "gcs"
            }
        except Exception as e:
            print(f"Error uploading buffer to GCS: {e}, falling back to local storage")
            # Fall back to local storage if GCS upload fails
            pass
    
    # Use local storage - have to write from buffer to file
    local_path = os.path.join(LOCAL_STORAGE_DIR, object_name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Write buffer to file
    buffer.seek(0)  # Reset the buffer position
    with open(local_path, 'wb') as f:
        f.write(buffer.getvalue())
        
    return {
        "url": f"file://{os.path.abspath(local_path)}",
        "path": local_path,
        "storage_type": "local"
    }

# Helper function for parallel uploads
def upload_buffer(buffer_object_tuple, gcs_path_prefix):
    """Upload a single buffer to storage with error handling and timing"""
    buffer, object_name = buffer_object_tuple
    start_time = time.time()
    
    if buffer is None:
        return object_name, {
            "url": f"error:{object_name}",
            "path": object_name, 
            "storage_type": "error",
            "error": "Empty buffer"
        }
    
    try:
        full_object_name = f"{gcs_path_prefix}/{object_name}"
        result = save_buffer_to_storage(buffer, full_object_name)
        elapsed = time.time() - start_time
        print(f"Uploaded {full_object_name} in {elapsed:.2f}s")
        return object_name, result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Error uploading {object_name} after {elapsed:.2f}s: {e}")
        return object_name, {
            "url": f"error:{object_name}",
            "path": object_name,
            "storage_type": "error",
            "error": str(e)
        }

# Helper functions for FITS data retrieval
def construct_query(
        item: Union[int, str],
        bottom_left: SkyCoord,
        top_right: SkyCoord,
        start_time: str,
        end_time: str,
        email: str,
        duration: u.Quantity,
        tracking: bool) -> Fido.search:
    """Constructs a Fido search query for SDO data."""
    
    # Define the attributes for the query
    time = a.Time(start_time, end_time)
    wave = a.Wavelength(item * u.angstrom, item * u.angstrom)
    
    # Define the email notification
    email_attr = a.jsoc.Notify(email)
    sample = a.Sample(duration)
    
    # Define the series based on whether tracking is enabled
    series = a.jsoc.Series.aia_lev1_euv_12s if not tracking else a.jsoc.Series.aia_lev1euvt_12h
    
    # Create query without spatial constraints
    query = Fido.search(
        time,
        series,
        wave,
        email_attr,
        sample
    )
    
    return query

def get_query_sdo(
        item: Union[int, str],
        bottom_left: SkyCoord,
        top_right: SkyCoord,
        start_time: str,
        end_time: str,
        email: str,
        duration: u.Quantity,
        tracking: bool = False) -> Union[Fido.search, None]:
    """Retrieves the query result for SDO data."""
    
    try:
        query = construct_query(item, bottom_left, top_right, start_time, end_time, email, duration, tracking)
        return query
    except Exception as e:
        print(f"Error constructing query: {e}")
        return None

def process_fits_analysis(request):
    """
    Processes FITS data for analysis.
    This function is meant to be run in a background thread.
    """
    # Track overall processing time
    start_time = time.time()
    last_heartbeat = start_time
    
    # Helper function for timing logs
    def log_with_timing(message):
        nonlocal last_heartbeat
        elapsed = time.time() - start_time
        last_heartbeat = time.time()
        print(f"[{elapsed:.1f}s] {message}")
    
    # Start a heartbeat thread to verify the process is still running
    def heartbeat():
        nonlocal last_heartbeat
        while True:
            time.sleep(60)  # Check every minute
            if time.time() - last_heartbeat > 60:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Still processing...")
                last_heartbeat = time.time()
    
    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    
    log_with_timing("Starting FITS analysis")
    
    timestamp = request.get("timestamp")
    channels = ["94", "131", "171", "193", "211", "304", "335"]
    
    if not timestamp:
        return {"error": "Timestamp is required", "status_code": 400}
        
    # Validate ISO format
    iso_format = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
    if not re.match(iso_format, timestamp):
        return {
            "error": "Invalid timestamp format",
            "details": "Timestamp must be in ISO format (YYYY-MM-DDThh:mm:ssZ)",
            "example": "2024-01-01T12:00:00Z",
            "received": timestamp,
            "status_code": 400
        }
    
    # Create temp directories with unique ID
    tmp_id = str(uuid.uuid4())
    input_dir = f"temp_data/{tmp_id}/fits_data"
    output_dir = f"outputs/{tmp_id}"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Format timestamp
    try:
        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        formatted_timestamp = dt.strftime("%Y%m%d_%H%M%S")
    except ValueError as e:
        return {"error": f"Invalid timestamp format: {str(e)}", "status_code": 400}

    log_with_timing(f"Creating directories for {len(channels)} channels")
    
    # Create channel directories for FITS files
    channel_dirs = {}
    for channel in channels:
        channel_dir = os.path.join(input_dir, f"aia_{channel}")
        os.makedirs(channel_dir, exist_ok=True)
        channel_dirs[channel] = channel_dir

    # Define the coordinates for the query
    bottom_left_coord = SkyCoord(center_x * u.arcsec - half_square_size * u.arcsec,
                                 center_y * u.arcsec - half_square_size * u.arcsec,
                                 frame='helioprojective')
    top_right_coord = SkyCoord(center_x * u.arcsec + half_square_size * u.arcsec,
                               center_y * u.arcsec + half_square_size * u.arcsec,
                               frame='helioprojective')

    log_with_timing("Starting FITS file downloads")
    
    # Fetch FITS files using VSO instead of JSOC
    downloaded_files = []
    download_logs = []
    
    # Download files with better logging
    download_start = time.time()
    for channel in channels:
        try:
            channel_dir = channel_dirs[channel]
            download_logs.append(f"Fetching AIA {channel} data via VSO...")
            
            # Convert timestamp for VSO query
            vso_time_start = (dt - datetime.timedelta(minutes=10)).strftime("%Y/%m/%d %H:%M:%S")
            vso_time_end = (dt + datetime.timedelta(minutes=10)).strftime("%Y/%m/%d %H:%M:%S")
            
            channel_start_time = time.time()
            log_with_timing(f"Searching for AIA {channel} data")
            
            # Initialize VSO client
            client = vso.VSOClient()
            
            # Search for files
            query_result = client.search(
                a.Time(vso_time_start, vso_time_end),
                a.Instrument('aia'),
                a.Wavelength(int(channel) * u.angstrom)
            )
            
            if len(query_result) == 0:
                log_with_timing(f"No AIA {channel} files found")
                download_logs.append(f"No AIA {channel} files found for time range")
                continue
                
            log_with_timing(f"Found {len(query_result)} files for channel {channel}, downloading latest")
            download_logs.append(f"Found {len(query_result)} files, downloading latest...")
            
            # Explicitly set download path to the channel directory
            download_path = os.path.join(channel_dir, "{file}")
            
            # Download the latest file (they should be sorted by time)
            try:
                files = client.fetch(query_result[-1:], path=download_path)
                
                if files:
                    downloaded_files.extend(files)
                    channel_elapsed = time.time() - channel_start_time
                    log_with_timing(f"Downloaded AIA {channel} file in {channel_elapsed:.1f}s: {files[0]}")
                    download_logs.append(f"Success: Downloaded AIA {channel} file: {files[0]}")
                else:
                    log_with_timing(f"Warning: No files downloaded for channel {channel}")
                    download_logs.append(f"Warning: No files downloaded for channel {channel}")
            except Exception as e:
                log_with_timing(f"Error downloading file for channel {channel}: {str(e)}")
                download_logs.append(f"Error downloading file for channel {channel}: {str(e)}")
                
        except Exception as e:
            error_msg = f"Error processing channel {channel}: {str(e)}"
            log_with_timing(error_msg)
            download_logs.append(error_msg)
            continue

    download_elapsed = time.time() - download_start
    log_with_timing(f"Download phase completed in {download_elapsed:.1f}s, got {len(downloaded_files)} files")
    
    if not downloaded_files:
        return {
            "error": "No FITS files were downloaded.",
            "logs": download_logs,
            "status_code": 500
        }
    
    # Update config for FITS pipeline
    analysis_config = config.copy()
    analysis_config.update({
        "data_dir": input_dir,
        "output_dir": output_dir,
        "file_type": "fits",
        "channels": channels,
        "image_size": 512,
    })

    try:
        # Run the pipeline - this now returns visualization buffers along with other results
        log_with_timing("Starting pipeline processing")
        pipeline_start = time.time()
        pipeline_result = run_pipeline(analysis_config)
        pipeline_elapsed = time.time() - pipeline_start
        log_with_timing(f"Pipeline processing completed in {pipeline_elapsed:.1f}s")
        
        # Get visualization results
        visualization_results = pipeline_result.get("visualization_results", [])
        log_with_timing(f"Got {len(visualization_results)} visualization results to upload")
        
        # Process and upload the visualization results in parallel
        storage_results = []
        gcs_path_prefix = f"results/fits/{formatted_timestamp}/{tmp_id}"
        
        # Upload visualization results directly from memory buffers using parallel processing
        log_with_timing(f"Starting parallel upload of {len(visualization_results)} files")
        upload_start = time.time()
        
        # Limit workers to avoid overwhelming GCS and/or network
        max_workers = min(10, len(visualization_results)) 
        
        # Use ThreadPoolExecutor for parallel uploads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all uploads
            futures = {executor.submit(upload_buffer, item, gcs_path_prefix): item for item in visualization_results}
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures)):
                object_name, result = future.result()
                storage_results.append((object_name, result))
                if (i+1) % 5 == 0 or (i+1) == len(visualization_results):
                    log_with_timing(f"Upload progress: {i+1}/{len(visualization_results)} files uploaded")
        
        upload_elapsed = time.time() - upload_start
        log_with_timing(f"All uploads completed in {upload_elapsed:.1f}s")
        
        # Process the results into images_data
        images_data = []
        for object_name, storage_result in storage_results:
            # Extract threshold and channel info from object path
            filename = os.path.basename(object_name)
            dir_name = os.path.basename(os.path.dirname(object_name))
            
            # Extract threshold from filename or directory if applicable
            threshold_value = None
            if "threshold_" in filename:
                threshold_parts = filename.split("threshold_")[1].split(".")[0]
                threshold_value = threshold_parts.replace("_", ".")
            elif "threshold_" in dir_name:
                threshold_value = dir_name.replace("threshold_", "").replace("_", ".")
            
            # Add image data
            images_data.append({
                "filename": object_name,
                "threshold": threshold_value,
                "url": storage_result.get("url", f"error:{object_name}"),
                "path": storage_result.get("path", object_name),
                "storage_type": storage_result.get("storage_type", "error"),
                "type": "image/jpeg",  # Assuming we've switched to JPEG for faster uploads
                "error": storage_result.get("error", None)
            })

        # Cleanup temporary directories
        log_with_timing("Cleaning up temporary files")
        shutil.rmtree(input_dir, ignore_errors=True)
        # Keep output_dir for debug if needed
        
        total_elapsed = time.time() - start_time
        log_with_timing(f"Total processing completed in {total_elapsed:.1f}s")
        
        # Return success with images
        return {
            "status": "success",
            "timestamp": timestamp,
            "processed_channels": channels,
            "images": images_data,
            "pipeline_result": {
                "num_anomalies": pipeline_result.get("num_anomalies", 0),
                "num_clusters": pipeline_result.get("num_clusters", 0),
                "thresholds": pipeline_result.get("thresholds", [])
            },
            "download_logs": download_logs,
            "timing": {
                "total_seconds": round(total_elapsed, 1),
                "download_seconds": round(download_elapsed, 1),
                "pipeline_seconds": round(pipeline_elapsed, 1),
                "upload_seconds": round(upload_elapsed, 1)
            }
        }
    
    except Exception as e:
        # Log the error and traceback
        error_msg = f"Pipeline failed: {str(e)}"
        log_with_timing(error_msg)
        tb = traceback.format_exc()
        
        # Cleanup on error
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return {
            "error": error_msg,
            "traceback": tb,
            "download_logs": download_logs,
            "status_code": 500
        }
    
# API Endpoints

@router.post("/start-fits-analysis")
async def start_fits_analysis(request: dict = Body(...)):
    """
    Start FITS analysis as a background job and return immediately with a job ID
    """
    # Validate required parameters
    if "timestamp" not in request:
        return JSONResponse(
            status_code=400,
            content={"error": "Timestamp is required"}
        )
        
    # Create a job with the request parameters
    job_id = job_manager.create_job(request)
    
    # Start the job in the background
    run_in_background(process_fits_analysis, job_id, request)
    
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Analysis started. Check status with /job-status/{job_id}"
    }

@router.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a background job"""
    job = job_manager.get_job(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"error": f"Job {job_id} not found"}
        )
    
    response = {
        "job_id": job["id"],
        "status": job["status"].value,
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    
    # Include result or error based on job status
    if job["status"] == JobStatus.COMPLETED:
        result = job["result"]
        if isinstance(result, dict) and "status_code" in result and result.get("error"):
            # If the job function returned an error
            return JSONResponse(
                status_code=result["status_code"],
                content={"error": result["error"], "details": result}
            )
        response["result"] = result
    elif job["status"] == JobStatus.FAILED:
        response["error"] = job["error"]
    
    return response

# Endpoint to process a single channel (for more focused analysis)
@router.post("/analyze-fits-channel")
async def analyze_fits_channel(request: dict = Body(...)):
    """Analyze a single FITS channel"""
    timestamp = request.get("timestamp")
    channel = request.get("channel")  # Just one channel
    
    if not timestamp or not channel:
        return JSONResponse(
            status_code=400,
            content={"error": "Both timestamp and channel are required"}
        )
    
    # Create a job with the request parameters
    # Use a subset of the original process function
    job_id = job_manager.create_job({
        "timestamp": timestamp,
        "channels": [channel],  # Single channel in a list
        "email": request.get("email", "j.c.g.gomez@astro.uio.no")
    })
    
    # Start the job in the background
    run_in_background(process_fits_analysis, job_id, {
        "timestamp": timestamp,
        "channels": [channel],
        "email": request.get("email", "j.c.g.gomez@astro.uio.no")
    })
    
    return {
        "status": "accepted",
        "job_id": job_id,
        "channel": channel,
        "message": "Analysis started. Check status with /job-status/{job_id}"
    }


########## TESTING ENDPOINTS ##########

@router.post("/fits-raw-files")
async def fits_raw_files(request: dict = Body(...)):
    """
    Returns a list of all raw FITS files for the requested timestamp/channels.
    """
    timestamp = request.get("timestamp")
    channels = request.get("channels", ["171", "211"])  # More focused default
    
    if not timestamp:
        return JSONResponse(
            status_code=400,
            content={"error": "Timestamp is required"}
        )
        
    # Validate ISO format
    iso_format = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
    if not re.match(iso_format, timestamp):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid timestamp format",
                "details": "Timestamp must be in ISO format (YYYY-MM-DDThh:mm:ssZ)",
                "example": "2024-01-01T12:00:00Z",
                "received": timestamp
            }
        )
        
    # Format timestamp
    try:
        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid timestamp format: {str(e)}"}
        )

    # Create temp directories
    tmp_id = str(uuid.uuid4())
    input_dir = f"temp_data/{tmp_id}/fits_data"
    
    # Start a background job to fetch the FITS files
    job_id = job_manager.create_job({
        "timestamp": timestamp, 
        "channels": channels,
        "input_dir": input_dir,
        "operation": "download_only"
    })
    
    async def download_fits_files():
        try:
            job_manager.start_job(job_id)
            
            os.makedirs(input_dir, exist_ok=True)
            
            # Create channel directories for FITS files
            channel_dirs = {}
            downloaded_files = []
            
            for channel in channels:
                channel_dir = os.path.join(input_dir, f"aia_{channel}")
                os.makedirs(channel_dir, exist_ok=True)
                channel_dirs[channel] = channel_dir
                
                try:
                    # Convert timestamp for VSO query
                    vso_time_start = (dt - datetime.timedelta(minutes=10)).strftime("%Y/%m/%d %H:%M:%S")
                    vso_time_end = (dt + datetime.timedelta(minutes=10)).strftime("%Y/%m/%d %H:%M:%S")
                    
                    # Initialize VSO client
                    client = vso.VSOClient()
                    
                    # Search for files
                    query_result = client.search(
                        a.Time(vso_time_start, vso_time_end),
                        a.Instrument('aia'),
                        a.Wavelength(int(channel) * u.angstrom)
                    )
                    
                    if len(query_result) == 0:
                        print(f"No AIA {channel} files found for time range")
                        continue
                        
                    print(f"Found {len(query_result)} files, downloading latest...")
                    
                    # Explicitly set download path to the channel directory
                    download_path = os.path.join(channel_dir, "{file}")
                    
                    # Download the latest file (they should be sorted by time)
                    files = client.fetch(query_result[-1:], path=download_path)
                    downloaded_files.extend(files)
                    
                except Exception as e:
                    print(f"Error downloading file for channel {channel}: {e}")
            
            if not downloaded_files:
                job_manager.fail_job(job_id, "No FITS files were downloaded.")
                return
            
            # Get the list of downloaded files
            fits_files = save_and_list_raw_fits({"data_dir": input_dir, "channels": channels})
            
            # Complete the job
            job_manager.complete_job(job_id, {
                "status": "success",
                "fits_files": fits_files
            })
            
        except Exception as e:
            tb = traceback.format_exc()
            job_manager.fail_job(job_id, f"Failed to download FITS files: {str(e)}\n{tb}")
    
    # Start the download in the background
    asyncio.create_task(download_fits_files())
    
    # Return immediately with the job ID
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Download started. Check status with /job-status/{job_id}"
    }