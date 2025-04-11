from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from app.services.image_processor import increase_contrast
from io import BytesIO

router = APIRouter()

@router.post("/contraste")
async def aumentar_contraste(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = increase_contrast(image_bytes)

    return StreamingResponse(BytesIO(processed_image), media_type="image/jpeg")
