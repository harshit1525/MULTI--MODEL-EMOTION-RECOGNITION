import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Body
from typing import Optional
from pydantic import BaseModel
from backend.services import text_service
from backend.schemas.response_schemas import PredictionResponse
from backend.utils.file_utils import ALLOWED_TEXT_EXTENSIONS

router = APIRouter(prefix="/predict", tags=["Text"])

class TextRequest(BaseModel):
    text: str

@router.post("/text", response_model=PredictionResponse)
async def predict_text_json(request: TextRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty or solely whitespace.")
    try:
        result = text_service.predict_from_text(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text inference error: {str(e)}")

@router.post("/text/file", response_model=PredictionResponse)
async def predict_text_file(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_text.txt"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(ALLOWED_TEXT_EXTENSIONS)}"
        )
    content_bytes = await file.read()
    if len(content_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        text = content_bytes.decode('utf-8', errors='ignore')
        result = text_service.predict_from_text(text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text file inference error: {str(e)}")
