import os
import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Body, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.services.webcam_service import webcam_service
from backend.services.fusion_service import EMOTIONS

logger = logging.getLogger("multimodal_backend.api.webcam")

router = APIRouter(prefix="/predict/webcam", tags=["Real-Time Webcam"])

class WebcamFramePayload(BaseModel):
    frame: str = Field(..., description="Base64 encoded JPEG/PNG image data URL")
    client_id: str = Field("default", description="Client identifier for temporal EMA smoothing")
    apply_smoothing: bool = Field(True, description="Whether to apply EMA smoothing across consecutive frames")
    calibrate: bool = Field(False, description="Whether to accumulate this frame into neutral baseline calibration buffer")

class SessionResetPayload(BaseModel):
    client_id: str = Field("default", description="Client identifier whose session state should be reset")

@router.get("/status")
def get_webcam_status():
    """
    Returns the real-time webcam inference service health, detector status, and supported classes.
    """
    return {
        "status": "online",
        "service": "Real-Time Webcam Facial Expression & Multimodal AI",
        "models_loaded": webcam_service.is_loaded,
        "detector_available": webcam_service.face_detector is not None,
        "device": str(webcam_service.device),
        "supported_emotions": EMOTIONS,
        "features": [
            "YuNet Real-Time Face Detection (~30ms)",
            "CLAHE Adaptive Lighting & Contrast Equalization",
            "ResNet-50 Feature Extraction",
            "Temporal Exponential Moving Average (EMA) Smoothing",
            "Centered Bayesian Logit Consensus Calibration (T*=0.90)",
            "Synchronized Microphone Audio & Speech-to-Text Fusion"
        ]
    }

@router.post("/frame")
async def predict_webcam_frame_form(
    file: Optional[UploadFile] = File(None),
    client_id: str = Form("default"),
    apply_smoothing: bool = Form(True)
):
    """
    Processes a single live webcam frame transmitted as binary multipart file.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="No frame file uploaded.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty frame uploaded.")
        
    try:
        result = webcam_service.predict_frame(content, client_id=client_id, apply_smoothing=apply_smoothing)
        return result
    except Exception as e:
        logger.error(f"Error processing webcam frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Webcam inference error: {str(e)}")

@router.post("/frame-json")
async def predict_webcam_frame_json(payload: WebcamFramePayload):
    """
    Processes a single live webcam frame transmitted as JSON Base64 data URL.
    Optimized for high-speed HTML5 Canvas frame streaming.
    """
    if not payload.frame:
        raise HTTPException(status_code=400, detail="Empty frame payload provided.")
        
    try:
        result = webcam_service.predict_frame(
            payload.frame,
            client_id=payload.client_id,
            apply_smoothing=payload.apply_smoothing,
            calibrate=payload.calibrate
        )
        return result
    except Exception as e:
        logger.error(f"Error processing webcam JSON frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Webcam JSON inference error: {str(e)}")

@router.post("/calibrate")
def calibrate_webcam(payload: SessionResetPayload):
    """
    Initiates personal resting-face baseline calibration for the specified client.
    """
    res = webcam_service.start_calibration(payload.client_id)
    return res

@router.websocket("/ws")
@router.websocket("/ws/webcam")
async def websocket_webcam_stream(websocket: WebSocket):
    """
    High-Throughput Full-Duplex WebSocket Streaming Endpoint for Real-Time Webcam Frames.
    Reduces roundtrip latency to <8ms by eliminating per-frame HTTP handshakes.
    """
    await websocket.accept()
    client_id = f"ws_{id(websocket)}"
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                result = webcam_service.predict_frame(message["bytes"], client_id=client_id)
                await websocket.send_json(result)
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    frame_data = data.get("frame")
                    cid = data.get("client_id", client_id)
                    apply_sm = data.get("apply_smoothing", True)
                    calib = data.get("calibrate", False)
                    result = webcam_service.predict_frame(
                        frame_data,
                        client_id=cid,
                        apply_smoothing=apply_sm,
                        calibrate=calib
                    )
                    await websocket.send_json(result)
                except Exception as inner_e:
                    await websocket.send_json({"status": "error", "message": str(inner_e)})
    except WebSocketDisconnect:
        webcam_service.reset_session(client_id)
        logger.info(f"WebSocket client {client_id} disconnected.")

@router.post("/multimodal")
async def predict_webcam_multimodal(
    frame_file: UploadFile = File(...),
    audio_file: Optional[UploadFile] = File(None),
    transcript_hint: Optional[str] = Form(None),
    client_id: str = Form("default")
):
    """
    Unified Real-Time Multimodal Emotion Analysis:
    Combines live camera frame with synchronized microphone audio chunk and optional transcript.
    """
    frame_bytes = await frame_file.read()
    if len(frame_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty frame file provided.")
        
    audio_bytes = None
    if audio_file is not None:
        audio_bytes = await audio_file.read()
        if len(audio_bytes) == 0:
            audio_bytes = None
            
    try:
        result = webcam_service.predict_multimodal_stream(
            frame_data=frame_bytes,
            audio_bytes=audio_bytes,
            transcript_hint=transcript_hint,
            client_id=client_id
        )
        return result
    except Exception as e:
        logger.error(f"Error in webcam multimodal stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Real-time multimodal inference error: {str(e)}")

@router.post("/reset-session")
def reset_webcam_session(payload: SessionResetPayload):
    """
    Resets temporal smoothing history and rolling buffers for a client session.
    """
    webcam_service.reset_session(payload.client_id)
    return {"status": "success", "message": f"Session for client '{payload.client_id}' reset successfully."}

