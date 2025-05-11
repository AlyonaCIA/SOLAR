import os
import shutil
import uuid
from typing import List

from app.api.pipeline.executor import run_pipeline, run_single_channel_pipeline
from app.config.settings import config
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

import aiohttp
import asyncio
from typing import List, Dict
import urllib.parse
import base64
from pathlib import Path
import datetime
import re

# Add these constants near the top
HELIOVIEWER_BASE_URL = "https://api.helioviewer.org"
SOURCE_IDS = [8, 9, 10, 11, 12, 13, 14]
SOURCE_ID_TO_CHANNEL_MAP = {8: '94', 9: '131', 10: '171', 11: '193', 12: '211', 13: '304', 14: '335'}

router = APIRouter()

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
async def get_and_analyze_images(request: Dict[str, str]):
    """
    Fetches JP2 images from Helioviewer based on timestamp and runs analysis.
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
                
                images_data.append({
                    "filename": rel_path,
                    "threshold": threshold_dir.replace("_", "."),
                    "size": file_size,
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
    """Accepts multiple .jp2 files, runs the anomaly detection pipeline, and returns a
    list of generated output image filenames."""

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

    # List all image files in the output directory
    output_files = [
        f for f in os.listdir(output_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff"))
    ]

    if not output_files:
        return JSONResponse(status_code=404, content={
                            "error": "No output images found."})

    return {
        "output_dir": output_dir,
        "files": output_files
    }
