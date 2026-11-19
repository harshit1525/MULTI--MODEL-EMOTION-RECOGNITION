import logging
from backend.services.audio_service import AudioInferenceService
from backend.services.text_service import TextInferenceService
from backend.services.video_service import VideoInferenceService
from backend.services.fusion_service import MultimodalFusionService
from backend.services.transcription_service import TranscriptionService
from backend.services.webcam_service import webcam_service, WebcamService

logger = logging.getLogger("multimodal_backend")

# Singleton service instances loaded once at application startup
audio_service = AudioInferenceService()
text_service = TextInferenceService()
video_service = VideoInferenceService()
fusion_service = MultimodalFusionService()
transcription_service = TranscriptionService()

__all__ = [
    "audio_service",
    "text_service",
    "video_service",
    "fusion_service",
    "transcription_service",
    "webcam_service",
    "AudioInferenceService",
    "TextInferenceService",
    "VideoInferenceService",
    "MultimodalFusionService",
    "TranscriptionService",
    "WebcamService"
]
