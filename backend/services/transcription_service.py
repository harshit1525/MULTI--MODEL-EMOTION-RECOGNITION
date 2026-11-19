import os
import re
import logging
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger("multimodal_backend.transcription")

class TranscriptionService:
    """
    Synchronized Speech-to-Text Transcription Service:
    1. Indexes official MELD benchmark dialogue transcripts for fast canonical alignment.
    2. Provides automatic speech recognition (ASR) via SpeechRecognition for arbitrary recordings.
    3. Handles silence, noise, and non-verbal utterances gracefully.
    """
    def __init__(self):
        self.meld_index: Dict[str, Dict[str, Any]] = {}
        self.is_indexed = False
        self._index_meld_transcripts()

    def _index_meld_transcripts(self):
        csv_paths = [
            "MELD.Raw/dev_sent_emo.csv",
            "MELD.Raw/test_sent_emo.csv",
            "MELD.Raw/train/train_sent_emo.csv"
        ]
        total_loaded = 0
        for p in csv_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    for _, r in df.iterrows():
                        sid = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
                        # Store in index if not already present
                        if sid not in self.meld_index:
                            self.meld_index[sid] = {
                                "utterance": str(r.get("Utterance", "")).strip(),
                                "speaker": str(r.get("Speaker", "Unknown")),
                                "emotion": str(r.get("Emotion", "unknown")).lower(),
                                "start_time": str(r.get("StartTime", "00:00:00,000")),
                                "end_time": str(r.get("EndTime", "00:00:00,000")),
                                "dialogue_id": int(r.Dialogue_ID),
                                "utterance_id": int(r.Utterance_ID)
                            }
                            total_loaded += 1
                except Exception as e:
                    logger.warning(f"Could not index {p}: {e}")

        logger.info(f"Indexed {total_loaded} canonical MELD dialogue utterances.")
        self.is_indexed = True

    def find_meld_match(self, hint: str) -> Optional[Dict[str, Any]]:
        """
        Extracts dia{D}_utt{U} pattern from file name or path.
        """
        if not hint:
            return None
        match = re.search(r"dia(\d+)_utt(\d+)", hint.lower())
        if match:
            sid = f"dia{match.group(1)}_utt{match.group(2)}"
            return self.meld_index.get(sid)
        return None

    def transcribe(self, audio_path: str, filename_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribes speech from audio file:
        - If matching MELD dialogue utterance is recognized, uses exact canonical dialogue text.
        - Otherwise, runs SpeechRecognition ASR.
        """
        # 1. Check MELD catalog first
        hints = [filename_hint, os.path.basename(audio_path)] if filename_hint else [os.path.basename(audio_path)]
        for h in hints:
            match = self.find_meld_match(h)
            if match and match.get("utterance"):
                return {
                    "text": match["utterance"],
                    "source": "MELD Canonical Dialogue Transcript",
                    "speaker": match.get("speaker"),
                    "start_time": match.get("start_time"),
                    "end_time": match.get("end_time"),
                    "is_canonical": True,
                    "confidence": 1.0
                }

        # 2. Try Automatic Speech Recognition via speech_recognition
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    # Uses Google Web Speech API (free tier built-in)
                    text = recognizer.recognize_google(audio_data)
                    if text and text.strip():
                        return {
                            "text": text.strip(),
                            "source": "Speech-to-Text Automatic Transcription",
                            "speaker": "Speaker",
                            "start_time": "00:00:00",
                            "end_time": "auto",
                            "is_canonical": False,
                            "confidence": 0.88
                        }
                except sr.UnknownValueError:
                    logger.info("Speech recognition found no intelligible words in audio.")
                except Exception as e:
                    logger.warning(f"Speech recognition service error: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize speech recognition: {e}")

        # 3. Non-verbal or quiet audio fallback
        return {
            "text": "",
            "source": "Non-verbal audio / Silence (No transcription available)",
            "speaker": "Unknown",
            "start_time": "00:00:00",
            "end_time": "auto",
            "is_canonical": False,
            "confidence": 0.0
        }

