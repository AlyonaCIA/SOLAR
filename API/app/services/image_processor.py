from PIL import Image, ImageEnhance
from io import BytesIO
from app.config.settings import CONTRAST_FACTOR

def increase_contrast(image_bytes: bytes) -> bytes:
    image = Image.open(BytesIO(image_bytes))
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(CONTRAST_FACTOR)

    output_buffer = BytesIO()
    enhanced_image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()
