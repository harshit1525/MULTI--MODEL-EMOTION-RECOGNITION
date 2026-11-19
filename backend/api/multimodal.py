import os
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from backend.services import audio_service, text_service, video_service, fusion_service, transcription_service
from backend.schemas.response_schemas import MultimodalPredictionResponse
from backend.utils.file_utils import (
    temporary_upload_file,
    extract_audio_from_video,
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS
)

logger = logging.getLogger("multimodal_backend.api")
router = APIRouter(prefix="/predict", tags=["Multimodal"])

@router.post("/multimodal", response_model=MultimodalPredictionResponse)
async def predict_multimodal(
    audio_file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None),
    video_file: Optional[UploadFile] = File(None)
):
    """
    Unified Multimodal Emotion Recognition Endpoint:
    Combines Audio (Prosody / Wav2Vec2), Text (Linguistic / TF-IDF), and Video (Visual / ResNet-50)
    using Learned Cross-Modal Attention and Reliability Gating into ONE unified final emotion prediction.
    
    If only a video recording is supplied:
    - Automatically extracts video frames for visual emotion classification.
    - Automatically extracts audio track for acoustic vocal classification.
    - Automatically transcribes speech to text for dialogue linguistic classification.
    All 3 modalities are synchronized and evaluated together.
    """
    logger.info("Received request for unified multimodal emotion recognition.")
    audio_result = None
    text_result = None
    video_result = None
    transcript_used = text_content.strip() if (text_content and text_content.strip()) else ""

    # 1. Process Explicit Text if supplied
    if transcript_used:
        try:
            logger.info(f"Processing linguistic modality with provided transcript: '{transcript_used[:50]}...'")
            text_result = text_service.predict_from_text(transcript_used)
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            text_result = {"status": "error", "message": f"Text error: {str(e)}", "probabilities": None}

    # 2. Process Explicit Audio if supplied
    if audio_file and audio_file.filename:
        audio_bytes = await audio_file.read()
        if len(audio_bytes) > 0:
            logger.info(f"Processing explicit audio file: {audio_file.filename} ({len(audio_bytes)} bytes)")
            try:
                with temporary_upload_file(audio_bytes, audio_file.filename) as temp_audio:
                    audio_result = audio_service.predict_from_audio(temp_audio)
                    audio_result["details"]["source_filename"] = audio_file.filename

                    # If text was NOT provided, transcribe audio to populate linguistic stream!
                    if not text_result:
                        trans_info = transcription_service.transcribe(temp_audio, filename_hint=audio_file.filename)
                        if trans_info.get("text"):
                            transcript_used = trans_info["text"]
                            logger.info(f"Transcribed audio speech: '{transcript_used}' via {trans_info.get('source')}")
                            text_result = text_service.predict_from_text(transcript_used)
                            text_result["details"]["transcription_source"] = trans_info.get("source")
            except Exception as e:
                logger.error(f"Error processing audio input: {e}")
                audio_result = {"status": "error", "message": f"Audio error: {str(e)}", "probabilities": None}

    # 3. Process Video if supplied
    if video_file and video_file.filename:
        video_bytes = await video_file.read()
        if len(video_bytes) > 0:
            logger.info(f"Processing video recording: {video_file.filename} ({len(video_bytes)} bytes)")
            try:
                with temporary_upload_file(video_bytes, video_file.filename) as temp_video:
                    # A. Visual Stream Prediction
                    video_result = video_service.predict_from_video(temp_video)
                    video_result["details"]["source_filename"] = video_file.filename

                    # B. Acoustic Stream (if not already provided explicitly)
                    extracted_wav = None
                    try:
                        extracted_wav = extract_audio_from_video(temp_video)
                        if audio_result is None and os.path.exists(extracted_wav):
                            logger.info("Automatically extracting and processing audio track from video recording...")
                            audio_result = audio_service.predict_from_audio(extracted_wav)
                            audio_result["details"]["source"] = "Extracted synchronized audio from uploaded video"

                        # C. Linguistic Stream: Automatic Speech-to-Text Transcription (if text was not supplied)
                        if text_result is None and os.path.exists(extracted_wav):
                            logger.info("Automatically transcribing spoken words from video audio track...")
                            trans_info = transcription_service.transcribe(extracted_wav, filename_hint=video_file.filename)
                            if trans_info.get("text"):
                                transcript_used = trans_info["text"]
                                logger.info(f"Transcribed video utterance: '{transcript_used}' (Source: {trans_info.get('source')})")
                                text_result = text_service.predict_from_text(transcript_used)
                                text_result["details"]["transcription_source"] = trans_info.get("source")
                    finally:
                        if extracted_wav and os.path.exists(extracted_wav):
                            try:
                                os.remove(extracted_wav)
                            except Exception:
                                pass
            except Exception as e:
                logger.error(f"Error processing video input: {e}")
                video_result = {"status": "error", "message": f"Video error: {str(e)}", "probabilities": None}

    # Check if at least one modality succeeded
    has_valid_audio = audio_result and audio_result.get("probabilities") is not None
    has_valid_text = text_result and text_result.get("probabilities") is not None
    has_valid_video = video_result and video_result.get("probabilities") is not None

    if not has_valid_audio and not has_valid_text and not has_valid_video:
        raise HTTPException(
            status_code=400,
            detail="At least one valid modality (Video, Audio, or Text) must be successfully processed."
        )

    # 4. Perform Multimodal Fusion
    logger.info(
        f"Fusing active modalities -> Audio: {has_valid_audio}, Text: {has_valid_text}, Video: {has_valid_video}"
    )
    fused_result = fusion_service.fuse_predictions(
        audio_result=audio_result,
        text_result=text_result,
        video_result=video_result,
        transcript_text=transcript_used
    )

    logger.info(
        f"Multimodal Fusion Complete -> Unified Emotion: {fused_result.get('fused_predicted_class').upper()} "
        f"({fused_result.get('fused_confidence'):.1%}), Agreement: {fused_result.get('modality_agreement')}"
    )

    return MultimodalPredictionResponse(**fused_result)
