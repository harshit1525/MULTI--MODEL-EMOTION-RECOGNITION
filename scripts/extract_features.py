import os
import sys
import argparse
import time
import subprocess
import glob
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# 1. AUDIO WAV EXTRACTION FROM VIDEO
# ---------------------------------------------------------------------------
def extract_audio_from_videos(split: str = "dev", max_files: int = None):
    print(f"\n[AUDIO EXTRACTION] Extracting 16kHz mono PCM WAV from {split.upper()} video split...")
    video_dirs = {
        "train": "MELD.Raw/train/train_splits",
        "dev": "MELD.Raw/dev/dev_splits_complete",
        "test": "MELD.Raw/test/output_repeated_splits_test"
    }
    out_dir = os.path.join("audio_data", split)
    os.makedirs(out_dir, exist_ok=True)

    vdir = video_dirs.get(split)
    if not vdir or not os.path.exists(vdir):
        print(f"[ERROR] Video directory not found for split '{split}': {vdir}")
        return

    mp4_files = glob.glob(os.path.join(vdir, "*.mp4"))
    if max_files:
        mp4_files = mp4_files[:max_files]
    print(f"Found {len(mp4_files)} MP4 files to process -> Output: {out_dir}")

    success, failed = 0, 0
    for mp4 in tqdm(mp4_files, desc=f"Extracting {split}"):
        sid = os.path.splitext(os.path.basename(mp4))[0]
        out_wav = os.path.join(out_dir, f"{sid}.wav")
        if os.path.exists(out_wav):
            success += 1
            continue
        # Run ffmpeg to extract 16kHz mono audio
        cmd = ["ffmpeg", "-y", "-i", mp4, "-ar", "16000", "-ac", "1", "-vn", out_wav]
        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"[DONE] Extraction completed for {split}: {success} succeeded, {failed} failed.")

# ---------------------------------------------------------------------------
# 2. WAV2VEC2 EMBEDDING EXTRACTION
# ---------------------------------------------------------------------------
def extract_wav2vec(split: str = "dev", max_files: int = None):
    print(f"\n[EMBEDDING EXTRACTION] Extracting 768-D Wav2Vec2 representations for {split.upper()}...")
    csv_paths = {
        "train": "MELD.Raw/train/train_sent_emo.csv",
        "dev": "MELD.Raw/dev_sent_emo.csv",
        "test": "MELD.Raw/test_sent_emo.csv"
    }
    csv_path = csv_paths.get(split)
    if not csv_path or not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found for split '{split}': {csv_path}")
        return

    df = pd.read_csv(csv_path)
    audio_dir = os.path.join("audio_data", split)
    out_dir = "audio_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Wav2Vec2 model (facebook/wav2vec2-base) on {device}...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", local_files_only=True)
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", local_files_only=True).to(device)
    wav2vec.eval()

    records = []
    if max_files:
        df = df.head(max_files)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Wav2Vec2 {split}"):
        sid = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
        wav_path = os.path.join(audio_dir, f"{sid}.wav")
        if not os.path.exists(wav_path):
            continue

        try:
            waveform, sr = sf.read(wav_path)
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)

            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = wav2vec(**inputs)
                emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()

            records.append({
                "sample_id": sid,
                "dialogue_id": int(row["Dialogue_ID"]),
                "utterance_id": int(row["Utterance_ID"]),
                "speaker": str(row.get("Speaker", "Unknown")),
                "Emotion": str(row.get("Emotion", "neutral")).lower(),
                "audio_embedding": emb
            })
        except Exception as e:
            continue

    out_file = os.path.join(out_dir, f"{split}_embeddings.pt")
    torch.save(records, out_file)
    print(f"[DONE] Saved {len(records)} embeddings for {split} -> {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)")

# ---------------------------------------------------------------------------
# 3. AUDIO DATASET DIAGNOSTIC INSPECTION
# ---------------------------------------------------------------------------
def inspect_audio(split: str = "dev"):
    audio_dir = os.path.join("audio_data", split)
    print(f"\n[INSPECT AUDIO] Inspecting audio quality metrics in {audio_dir}...")
    if not os.path.exists(audio_dir):
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        return

    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print(f"[WARN] No WAV files found in {audio_dir}.")
        return

    sample_check = wav_files[:min(len(wav_files), 100)]
    durations, srs = [], []
    for w in sample_check:
        try:
            info = sf.info(w)
            durations.append(info.duration)
            srs.append(info.samplerate)
        except Exception:
            continue

    print(f"Total files sampled : {len(sample_check)}")
    print(f"Sample rates seen   : {set(srs)}")
    print(f"Mean duration       : {np.mean(durations):.2f}s")
    print(f"Min / Max duration  : {np.min(durations):.2f}s / {np.max(durations):.2f}s")

# ---------------------------------------------------------------------------
# CLI DISPATCHER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Multimodal Feature Extraction & Audio Pipeline CLI")
    parser.add_argument("--modality", type=str, default="all", choices=["audio", "wav2vec", "all"], help="Extraction target")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test", "all"], help="Dataset split")
    parser.add_argument("--inspect", action="store_true", help="Run audio inspection diagnostic")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files to process")

    args = parser.parse_args()

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]

    if args.inspect:
        for s in splits:
            inspect_audio(s)
        return

    if args.modality in ["audio", "all"]:
        for s in splits:
            extract_audio_from_videos(split=s, max_files=args.max_files)

    if args.modality in ["wav2vec", "all"]:
        for s in splits:
            extract_wav2vec(split=s, max_files=args.max_files)

if __name__ == "__main__":
    main()

