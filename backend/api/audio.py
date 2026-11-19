import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services import audio_service
from backend.schemas.response_schemas import PredictionResponse
from backend.utils.file_utils import temporary_upload_file, ALLOWED_AUDIO_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, extract_audio_from_video

router = APIRouter(prefix="/predict", tags=["Audio"])

@router.post("/audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_AUDIO_EXTENSIONS and ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS)}"
        )
        
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    try:
        with temporary_upload_file(content, filename) as temp_path:
            # If video file, extract audio first
            if ext in ALLOWED_VIDEO_EXTENSIONS:
                extracted_wav = extract_audio_from_video(temp_path)
                try:
                    result = audio_service.predict_from_audio(extracted_wav)
                finally:
                    if os.path.exists(extracted_wav):
                        os.remove(extracted_wav)
            else:
                result = audio_service.predict_from_audio(temp_path)
                
            return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio inference error: {str(e)}")
