import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services import video_service
from backend.schemas.response_schemas import PredictionResponse
from backend.utils.file_utils import temporary_upload_file, ALLOWED_VIDEO_EXTENSIONS

router = APIRouter(prefix="/predict", tags=["Video"])

@router.post("/video", response_model=PredictionResponse)
async def predict_video(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
        
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
        
    try:
        with temporary_upload_file(content, filename) as temp_path:
            result = video_service.predict_from_video(temp_path)
            return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video inference error: {str(e)}")
