from pydantic import BaseModel

class RequestModel(BaseModel):
    mensaje: str

class ResponseModel(BaseModel):
    resultado: str
