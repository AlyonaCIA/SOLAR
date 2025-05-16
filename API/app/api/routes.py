import os
import shutil
import uuid
from typing import List, Dict, Optional, Union
import re
import datetime
import traceback

import aiohttp
import asyncio
import urllib.parse
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import necessary modules from sunpy and astropy
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.net import Fido
from sunpy.net import attrs as a

# Import our custom modules
from app.api.pipeline.executor import run_pipeline, run_single_channel_pipeline, save_and_list_raw_fits
from app.api.pipeline.job_manager import job_manager, JobStatus
from app.api.pipeline.background_job import run_in_background
from app.config.settings import config

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

# Helper functions for fetching JP2 images
async def fetch_jp2_image(session: aiohttp.ClientSession, source_id: int, timestamp: str) -> bytes:
    """Fetches a JP2 image from Helioviewer API."""
    params = {
        "sourceId": source_id,
        "date": timestamp,
        "json": 0
    }
    
    url = f"{HELIOVIEWER_BASE_URL}/v2/getJP2Image/?"
    encoded_params = urllib.parse.urlencode(params)
    full_url = url + encoded_params

    async with session.get(full_url) as response:
        if response.status != 200:
            raise Exception(f"Failed to fetch image for sourceId {source_id}: {response.status}")
        return await response.read()

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
    timestamp = request.get("timestamp")
    channels = request.get("channels", ["94", "131", "171", "193", "211", "304", "335"])
    email = request.get("email", "j.c.g.gomez@astro.uio.no")
    
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

    # Fetch FITS files from JSOC
    downloaded_files = []
    download_logs = []
    
    for channel in channels:
        try:
            # Convert channel to integer
            channel_int = int(channel)
            
            # Define the time range and duration
            start_time = dt.strftime(date_format)
            end_time = (dt + datetime.timedelta(minutes=1)).strftime(date_format)
            sample_interval = 12 * u.s
            
            download_logs.append(f"Querying channel {channel}: {start_time} to {end_time}")
            
            # Get the query result
            query_result = get_query_sdo(
                channel_int, bottom_left_coord, top_right_coord,
                start_time, end_time, email, sample_interval,
                tracking=False
            )
            
            if query_result:
                download_logs.append(f"Query for channel {channel} returned results")
                # Download the files
                files = Fido.fetch(query_result, path=channel_dirs[channel])
                download_logs.append(f"Downloaded {len(files)} files for channel {channel}")
                downloaded_files.extend(files)
            else:
                download_logs.append(f"No query result for channel {channel}")
        except Exception as e:
            download_logs.append(f"Error processing channel {channel}: {str(e)}")
            continue

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
        # Run the pipeline
        pipeline_result = run_pipeline(analysis_config)
        
        # Get output files and save to storage (GCS or local)
        images_data = []
        gcs_path_prefix = f"results/fits/{formatted_timestamp}/{tmp_id}"
        
        # Walk through all subdirectories in the output directory
        for root, _, files in os.walk(output_dir):
            for filename in files:
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                    file_path = Path(root) / filename
                    
                    # Get relative path from output_dir
                    rel_path = os.path.relpath(file_path, output_dir)
                    
                    # Extract threshold from directory name
                    threshold_dir = os.path.basename(os.path.dirname(file_path))
                    
                    # Create storage object name
                    object_name = f"{gcs_path_prefix}/{rel_path}"
                    
                    # Save to storage (GCS or local)
                    storage_result = save_to_storage(str(file_path), object_name)
                    
                    # Add image data
                    images_data.append({
                        "filename": rel_path,
                        "threshold": threshold_dir.replace("_", ".") if "_" in threshold_dir else threshold_dir,
                        "url": storage_result["url"],
                        "path": storage_result["path"],
                        "storage_type": storage_result["storage_type"],
                        "type": "image/" + filename.split('.')[-1].lower(),
                    })

        # Cleanup temporary directories
        shutil.rmtree(input_dir, ignore_errors=True)
        
        # Return success with images
        return {
            "status": "success",
            "timestamp": timestamp,
            "processed_channels": channels,
            "images": images_data,
            "pipeline_result": pipeline_result,
            "download_logs": download_logs
        }
    
    except Exception as e:
        # Log the error and traceback
        error_msg = f"Pipeline failed: {str(e)}"
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

# Keep backward compatibility with existing endpoint but make it use the async pattern
@router.post("/get-fits-analyze")
async def get_and_analyze_fits(request: dict = Body(...)):
    """
    Legacy endpoint that now uses the async job pattern internally.
    It starts a job and polls for results with a reasonable timeout.
    """
    # Start the job
    start_response = await start_fits_analysis(request)
    job_id = start_response["job_id"]
    
    # For backward compatibility, wait for a short time to see if job completes quickly
    max_wait_seconds = 10
    poll_interval = 0.5
    elapsed = 0
    
    while elapsed < max_wait_seconds:
        # Wait a bit
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        
        # Check job status
        job_status = await get_job_status(job_id)
        
        # If job is complete or failed, return the result
        if job_status.get("status") == "completed":
            if "result" in job_status:
                return job_status["result"]
        elif job_status.get("status") == "failed":
            return JSONResponse(
                status_code=500,
                content={"error": f"Analysis failed: {job_status.get('error', 'Unknown error')}"}
            )
    
    # If we reach here, the job is still running
    return {
        "status": "processing",
        "job_id": job_id,
        "message": "Analysis is still running. Check status with /job-status/{job_id}"
    }

@router.post("/fits-raw-files")
async def fits_raw_files(request: dict = Body(...)):
    """
    Returns a list of all raw FITS files for the requested timestamp/channels.
    """
    timestamp = request.get("timestamp")
    channels = request.get("channels", ["171", "211"])  # More focused default
    email = request.get("email", "j.c.g.gomez@astro.uio.no")
    
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
        "email": email,
        "input_dir": input_dir,
        "operation": "download_only"
    })
    
    async def download_fits_files():
        try:
            job_manager.start_job(job_id)
            
            os.makedirs(input_dir, exist_ok=True)
            
            # Create channel directories for FITS files
            channel_dirs = {}
            for channel in channels:
                channel_dir = os.path.join(input_dir, f"aia_{channel}")
                os.makedirs(channel_dir, exist_ok=True)
                channel_dirs[channel] = channel_dir
            
            # Define coordinates
            bottom_left_coord = SkyCoord(center_x * u.arcsec - half_square_size * u.arcsec,
                                        center_y * u.arcsec - half_square_size * u.arcsec,
                                        frame='helioprojective')
            top_right_coord = SkyCoord(center_x * u.arcsec + half_square_size * u.arcsec,
                                      center_y * u.arcsec + half_square_size * u.arcsec,
                                      frame='helioprojective')

            # Fetch FITS files from JSOC
            downloaded_files = []
            for channel in channels:
                channel_int = int(channel)
                start_time = dt.strftime(date_format)
                end_time = (dt + datetime.timedelta(minutes=1)).strftime(date_format)
                sample_interval = 12 * u.s
                
                # Get the query result
                query_result = get_query_sdo(
                    channel_int, bottom_left_coord, top_right_coord,
                    start_time, end_time, email, sample_interval,
                    tracking=False
                )
                
                if query_result:
                    try:
                        # Download the files
                        files = Fido.fetch(query_result, path=channel_dirs[channel])
                        downloaded_files.extend(files)
                    except Exception as e:
                        print(f"Error downloading files for channel {channel}: {e}")
            
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

@router.post("/test-jsoc-connection")
async def test_jsoc_connection(request: dict = Body(...)):
    """Test the connection to JSOC API and file downloads."""
    timestamp = request.get("timestamp", "2024-05-01T12:00:00Z")
    channels = request.get("channels", ["171"])  # Just one channel for testing
    email = request.get("email", "j.c.g.gomez@astro.uio.no")
    
    try:
        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Create a test directory
        test_dir = f"test_jsoc_{uuid.uuid4()}"
        os.makedirs(test_dir, exist_ok=True)
        
        # Log every step
        logs = []
        logs.append(f"Test directory created: {test_dir}")
        
        # Define coordinates
        bottom_left_coord = SkyCoord(-1210 * u.arcsec, -1210 * u.arcsec, frame='helioprojective')
        top_right_coord = SkyCoord(1210 * u.arcsec, 1210 * u.arcsec, frame='helioprojective')
        logs.append("Coordinates defined")
        
        # Try to construct a query
        channel_int = int(channels[0])
        start_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        end_time = (dt + datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        sample_interval = 12 * u.s
        
        logs.append(f"Attempting JSOC query: channel={channel_int}, time={start_time} to {end_time}")
        
        try:
            # First, just create the query object
            time_attr = a.Time(start_time, end_time)
            wave_attr = a.Wavelength(channel_int * u.angstrom, channel_int * u.angstrom)
            email_attr = a.jsoc.Notify(email)
            sample = a.Sample(sample_interval)
            series = a.jsoc.Series.aia_lev1_euv_12s
            
            logs.append("Query attributes created successfully")
            
            # Then try the actual query
            query = Fido.search(time_attr, series, wave_attr, email_attr, sample)
            logs.append(f"Query successful. Results: {len(query)}")
            logs.append(f"Query details: {query}")
            
            # Try downloading a single file
            logs.append("Attempting download...")
            files = Fido.fetch(query, path=test_dir, max_conn=1)
            logs.append(f"Download successful! Files: {files}")
            
            # List the directory contents
            dir_contents = os.listdir(test_dir)
            logs.append(f"Directory contents: {dir_contents}")
            
        except Exception as e:
            logs.append(f"JSOC query/fetch failed: {str(e)}")
            logs.append(f"Exception type: {type(e)}")
            logs.append(f"Traceback: {traceback.format_exc()}")
        
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)
        logs.append("Test directory removed")
        
        return {
            "status": "test_complete",
            "logs": logs
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Test failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
        )

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

async def fetch_jp2_image(session: aiohttp.ClientSession, source_id: int, timestamp: str) -> bytes:
    """Fetches a JP2 image from Helioviewer API."""
    params = {
        "sourceId": source_id,
        "date": timestamp,
        "json": 0
    }
    
    url = f"{HELIOVIEWER_BASE_URL}/v2/getJP2Image/?"
    encoded_params = urllib.parse.urlencode(params)
    full_url = url + encoded_params

    async with session.get(full_url) as response:
        if response.status != 200:
            raise Exception(f"Failed to fetch image for sourceId {source_id}: {response.status}")
        return await response.read()
    
@router.post("/get-image-analyze")
async def get_and_analyze_images(request: dict = Body(...)):
    """
    Fetches JP2 images from Helioviewer based on timestamp and runs analysis, 
    then saves it to storage (GCS in production, local in development).
    """
    timestamp = request.get("timestamp")
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
    # Format timestamp for filenames (assuming timestamp is in format like "2024-01-01T12:00:00Z")
    # Convert from ISO format to our filename format
    try:
        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        formatted_timestamp = dt.strftime("%Y%m%d_%H%M%S")
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid timestamp format: {str(e)}"}
        )

    # Create temp directories
    tmp_id = str(uuid.uuid4())
    print(f"tmp_id: {tmp_id}")
    input_dir = f"temp_data/{tmp_id}/input_jp2_images"
    output_dir = f"outputs/{tmp_id}"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Fetch all images concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_jp2_image(session, source_id, timestamp) 
            for source_id in SOURCE_IDS
        ]
        
        try:
            jp2_contents = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to fetch images: {str(e)}"}
            )

        # Save fetched images
        for source_id, content in zip(SOURCE_IDS, jp2_contents):
            if isinstance(content, Exception):
                print(f"Error fetching source {source_id}: {content}")
                continue
                
            file_path = os.path.join(input_dir, f"AIA_{SOURCE_ID_TO_CHANNEL_MAP[source_id]}.jp2")
            with open(file_path, "wb") as f:
                f.write(content)

    # Update config for pipeline
    config["data_dir"] = input_dir
    config["output_dir"] = output_dir

    try:
        run_single_channel_pipeline(config, formatted_timestamp)
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Pipeline failed: {str(e)}"}
        )

    # Get output files - Modified to search in threshold subdirectories
    images_data = []
    gcs_path_prefix = f"results/{formatted_timestamp}/{tmp_id}"

    # Walk through all subdirectories in the output directory
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                file_path = Path(root) / filename
                
                # Get file size
                file_size = file_path.stat().st_size
                
                # Get relative path from output_dir
                rel_path = os.path.relpath(file_path, output_dir)
                
                # Extract threshold from directory name
                threshold_dir = os.path.basename(os.path.dirname(file_path))
                
                # Create storage object name
                object_name = f"{gcs_path_prefix}/{rel_path}"
                
                # Save to storage (GCS or local)
                storage_result = save_to_storage(str(file_path), object_name)

                # Add image data
                images_data.append({
                    "filename": rel_path,
                    "threshold": threshold_dir.replace("_", "."),
                    "size": file_size,
                    "url": storage_result["url"],
                    "path": storage_result["path"],
                    "storage_type": storage_result["storage_type"],
                    "type": "image/" + filename.split('.')[-1].lower(),
                })

    if not images_data:
        return JSONResponse(
            status_code=404,
            content={"error": "No output images found"}
        )

    try:
        # Cleanup temporary directories
        shutil.rmtree(input_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "images": images_data
        }
    
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return JSONResponse(
            status_code=500,
            content={"error": f"Files encoding and transfer failed: {str(e)}"}
        )

@router.post("/analyze")
async def analyze_jp2(files: List[UploadFile] = File(...)):
    """Accepts multiple .jp2 files, runs the anomaly detection pipeline, and saves the results
    to storage (GCS in production, local in development)."""

    tmp_id = str(uuid.uuid4())
    input_dir = f"temp_data/{tmp_id}"
    output_dir = f"outputs/{tmp_id}"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Save all uploaded files
    for file in files:
        file_path = os.path.join(input_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

    # Update config
    config["data_dir"] = input_dir
    config["output_dir"] = output_dir

    try:
        run_pipeline(config)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Pipeline failed: {e}"})

    # Upload results to storage and return URLs
    gcs_path_prefix = f"results/uploads/{tmp_id}"
    output_files = []
    
    for root, dirs, files in os.walk(output_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
                file_path = os.path.join(root, filename)
                
                # Extract threshold from directory name if available
                rel_path = os.path.relpath(file_path, output_dir)
                parent_dir = os.path.basename(os.path.dirname(file_path))
                threshold_str = parent_dir if parent_dir != output_dir else "default"
                
                # Create storage object name
                object_name = f"{gcs_path_prefix}/{rel_path}"
                
                # Save to storage (GCS or local)
                storage_result = save_to_storage(file_path, object_name)
                
                output_files.append({
                    "filename": filename,
                    "threshold": threshold_str.replace("_", ".") if "_" in threshold_str else threshold_str,
                    "url": storage_result["url"],
                    "path": storage_result["path"],
                    "storage_type": storage_result["storage_type"],
                    "type": "image/" + filename.split('.')[-1].lower()
                })

    if not output_files:
        return JSONResponse(status_code=404, content={
                            "error": "No output images found."})
    
    # Clean up temporary files
    shutil.rmtree(input_dir, ignore_errors=True)

    return {
        "status": "success",
        "images": output_files
    }

########## TESTING ENDPOINTS ##########

@router.post("/fits-raw-files")
async def fits_raw_files(request: dict = Body(...)):
    """
    Returns a list of all raw FITS files for the requested timestamp/channels.
    """
    timestamp = request.get("timestamp")
    channels = request.get("channels", ["94", "131", "171", "193", "211", "304", "335"])
    email = request.get("email", "j.c.g.gomez@astro.uio.no")
    
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
        formatted_timestamp = dt.strftime("%Y%m%d_%H%M%S")
        jsoc_timestamp = dt.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid timestamp format: {str(e)}"}
        )

    # Create temp directories
    tmp_id = str(uuid.uuid4())
    input_dir = f"temp_data/{tmp_id}/fits_data"
    output_dir = f"outputs/{tmp_id}"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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

    # Fetch FITS files from JSOC
    downloaded_files = []
    for channel in channels:
        # Convert channel to integer
        channel_int = int(channel)
        
        # Define the time range and duration
        start_time = dt.strftime(date_format)
        end_time = (dt + datetime.timedelta(minutes=1)).strftime(date_format)  # Add 1 minute
        sample_interval = 12 * u.s  # 12 seconds
        
        # Get the query result
        query_result = get_query_sdo(
            channel_int, bottom_left_coord, top_right_coord,
            start_time, end_time, email, sample_interval,
            tracking=False
        )
        
        if query_result:
            try:
                # Download the files
                files = Fido.fetch(query_result, path=channel_dirs[channel])
                downloaded_files.extend(files)
            except Exception as e:
                print(f"Error downloading files for channel {channel}: {e}")
        else:
            print(f"No query result for channel {channel}")

    if not downloaded_files:
        return JSONResponse(
            status_code=500,
            content={"error": "No FITS files were downloaded."}
        )

    config.update({
        "data_dir": input_dir,
        "channels": channels,
    })

    fits_files = save_and_list_raw_fits(config)
    if not fits_files:
        return JSONResponse(status_code=404, content={"error": "No FITS files found."})
    return {"status": "success", "fits_files": fits_files}

@router.post("/test-jsoc-connection")
async def test_jsoc_connection(request: dict = Body(...)):
    """Test the connection to JSOC API and file downloads."""
    timestamp = request.get("timestamp", "2024-05-01T12:00:00Z")
    channels = request.get("channels", ["171"])  # Just one channel for testing
    email = request.get("email", "j.c.g.gomez@astro.uio.no")
    
    try:
        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Create a test directory
        test_dir = f"test_jsoc_{uuid.uuid4()}"
        os.makedirs(test_dir, exist_ok=True)
        
        # Log every step
        logs = []
        logs.append(f"Test directory created: {test_dir}")
        
        # Define coordinates
        bottom_left_coord = SkyCoord(-1210 * u.arcsec, -1210 * u.arcsec, frame='helioprojective')
        top_right_coord = SkyCoord(1210 * u.arcsec, 1210 * u.arcsec, frame='helioprojective')
        logs.append("Coordinates defined")
        
        # Try to construct a query
        channel_int = int(channels[0])
        start_time = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        end_time = (dt + datetime.timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")
        sample_interval = 12 * u.s
        
        logs.append(f"Attempting JSOC query: channel={channel_int}, time={start_time} to {end_time}")
        
        try:
            # First, just create the query object
            time_attr = a.Time(start_time, end_time)
            wave_attr = a.Wavelength(channel_int * u.angstrom, channel_int * u.angstrom)
            email_attr = a.jsoc.Notify(email)
            sample = a.Sample(sample_interval)
            series = a.jsoc.Series.aia_lev1_euv_12s
            
            logs.append("Query attributes created successfully")
            
            # Then try the actual query
            query = Fido.search(time_attr, series, wave_attr, email_attr, sample)
            logs.append(f"Query successful. Results: {len(query)}")
            logs.append(f"Query details: {query}")
            
            # Try downloading a single file
            logs.append("Attempting download...")
            files = Fido.fetch(query, path=test_dir, max_conn=1)
            logs.append(f"Download successful! Files: {files}")
            
            # List the directory contents
            dir_contents = os.listdir(test_dir)
            logs.append(f"Directory contents: {dir_contents}")
            
        except Exception as e:
            logs.append(f"JSOC query/fetch failed: {str(e)}")
            logs.append(f"Exception type: {type(e)}")
            import traceback
            logs.append(f"Traceback: {traceback.format_exc()}")
        
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)
        logs.append("Test directory removed")
        
        return {
            "status": "test_complete",
            "logs": logs
        }
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Test failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
        )