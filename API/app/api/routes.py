from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
import shutil
import uuid
from typing import List
from app.api.pipeline.executor import run_pipeline
from app.config.settings import config

router = APIRouter()

@router.post("/analyze")
async def analyze_fits(files: List[UploadFile] = File(...)):
    """
    Accepts multiple .fits files, runs the anomaly detection pipeline,
    and returns a list of generated output image filenames.
    """

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
        return JSONResponse(status_code=404, content={"error": "No output images found."})

    return {
        "output_dir": output_dir,
        "files": output_files
    }
