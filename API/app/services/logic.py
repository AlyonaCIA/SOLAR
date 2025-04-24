from app.models.schemas import RequestModel, ResponseModel

def process_data(data: RequestModel) -> ResponseModel:
    resultado = data.mensaje.upper()
    return ResponseModel(resultado=resultado)
