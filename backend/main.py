import os
import logging
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Setup standard logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("multimodal_backend")

from backend.api import audio, text, video, multimodal, webcam
from backend.schemas.response_schemas import HealthResponse

app = FastAPI(
    title="Multimodal Emotion AI Recognition System",
    description="Unified Audio, Text, and Video Emotion Classification on MELD with Attention-Gated Multimodal Fusion",
    version="2.1.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(audio.router)
app.include_router(text.router)
app.include_router(video.router)
app.include_router(multimodal.router)
app.include_router(webcam.router)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_loaded = {
        "audio": os.path.exists("models/audio_classifier/best_audio_model.pth"),
        "text": os.path.exists("models/text_classifier/best_text_model.joblib"),
        "video": os.path.exists("models/video_classifier/best_video_model.pth"),
        "fusion": os.path.exists("models/fusion_classifier/best_fusion_model.pth"),
        "webcam_yunet": os.path.exists("models/face_detection_yunet_2023mar.onnx")
    }
    return HealthResponse(
        status="healthy",
        device=device,
        models_loaded=models_loaded,
        supported_emotions=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
        version="2.1.0"
    )

# Mount frontend static files if directory exists
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    @app.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(
                index_file,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
        return {"message": "Frontend index.html not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
