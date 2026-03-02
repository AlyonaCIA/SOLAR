from app.api.routes import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Solar API",
    description="API for analyzing solar images",
    version="0.1.3",  # Fixed version string (was "0I.0.3" with a letter I)
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include router only once (was included twice before)
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Solar Analysis API is running. Go to /docs for the API documentation."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, timeout_keep_alive=600)
