import os
import shutil
import tempfile
import uuid
import subprocess
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_TEXT_EXTENSIONS = {'.txt', '.csv', '.json', '.md'}

TEMP_BASE_DIR = os.path.join(tempfile.gettempdir(), "multimodal_ai_tmp")
os.makedirs(TEMP_BASE_DIR, exist_ok=True)

@contextmanager
def temporary_upload_file(file_bytes: bytes, filename: str) -> Generator[str, None, None]:
    """
    Saves uploaded file bytes to a temporary file and guarantees cleanup on context exit.
    """
    ext = Path(filename).suffix.lower()
    temp_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(TEMP_BASE_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def extract_audio_from_video(video_path: str) -> str:
    """
    Uses ffmpeg to extract 16kHz mono 16-bit PCM WAV from a video file.
    Returns path to the generated WAV in the temporary directory.
    """
    out_wav = os.path.join(TEMP_BASE_DIR, f"{uuid.uuid4().hex}_extracted.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        out_wav
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(out_wav):
        raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr}")
    return out_wav
