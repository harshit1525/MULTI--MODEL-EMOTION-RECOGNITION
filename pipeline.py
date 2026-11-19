"""
Multimodal Emotion Recognition System (MELD Benchmark)
======================================================
Unified Data Pipeline, Feature Extraction & Verification

Usage:
  python pipeline.py --validate             # Validate environment, MELD dataset & model checkpoints
  python pipeline.py --verify               # Run statistical verification on audio embeddings
  python pipeline.py --extract audio        # Extract 16kHz mono WAV from MELD video splits
  python pipeline.py --extract wav2vec      # Extract 768-D Wav2Vec2 embeddings
  python pipeline.py --inspect              # Inspect audio dataset duration and sample rate stats
  python pipeline.py --all                  # Run validation and embedding verification
"""

import os
import sys
import argparse
import time
import subprocess
import shutil
import glob
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}


# ---------------------------------------------------------------------------
# 1. ENVIRONMENT & DATASET VALIDATION
# ---------------------------------------------------------------------------
def validate_environment() -> bool:
    print("\n" + "=" * 70)
    print(" 1. ENVIRONMENT & DEPENDENCY VALIDATION")
    print("=" * 70)
    all_ok = True

    print(f" Python Version   : {sys.version.split()[0]}")
    print(f" PyTorch Version  : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f" CUDA Available   : {cuda_avail} ({torch.cuda.get_device_name(0) if cuda_avail else 'CPU Execution'})")

    packages = ["transformers", "cv2", "torchaudio", "torchvision", "fastapi", "uvicorn", "sklearn", "soundfile", "librosa"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"   [OK] {pkg:<14} : {ver}")
        except ImportError:
            print(f"   [FAIL] {pkg:<12} : NOT INSTALLED")
            all_ok = False

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        print(f"   [OK] ffmpeg         : {ffmpeg_bin}")
    else:
        print("   [INFO] ffmpeg       : Not in PATH (only required when extracting raw video audio).")

    return all_ok


def validate_meld_dataset() -> bool:
    print("\n" + "=" * 70)
    print(" 2. MELD BENCHMARK DATASET INTEGRITY AUDIT")
    print("=" * 70)
    splits = {
        "train": ("MELD.Raw/train/train_sent_emo.csv", "MELD.Raw/train/train_splits"),
        "dev": ("MELD.Raw/dev_sent_emo.csv", "MELD.Raw/dev/dev_splits_complete"),
        "test": ("MELD.Raw/test_sent_emo.csv", "MELD.Raw/test/output_repeated_splits_test")
    }
    all_ok = True

    for name, (csv_p, vid_p) in splits.items():
        if not os.path.exists(csv_p):
            print(f"   [FAIL] Missing CSV for {name.upper()}: {csv_p}")
            all_ok = False
            continue

        df = pd.read_csv(csv_p)
        required_cols = {"Dialogue_ID", "Utterance_ID", "Emotion", "Utterance"}
        diff = required_cols - set(df.columns)
        if diff:
            print(f"   [FAIL] CSV {csv_p} missing columns: {diff}")
            all_ok = False
            continue

        emotions_found = df["Emotion"].str.lower().unique()
        print(f"   [OK] {name.upper():<5} CSV: {len(df):>5} utterances, {len(emotions_found)} distinct emotion classes.")

        if os.path.exists(vid_p):
            vid_count = len(os.listdir(vid_p))
            print(f"        {name.upper():<5} Video Dir: {vid_count:>5} files in {vid_p}")
        else:
            print(f"   [WARN] Video directory not found: {vid_p}")

    return all_ok


def validate_checkpoints() -> bool:
    print("\n" + "=" * 70)
    print(" 3. TRAINED MODEL CHECKPOINTS AUDIT")
    print("=" * 70)
    checkpoints = [
        ("Audio Classifier", "models/audio_classifier/best_audio_model_balanced.pth"),
        ("Video Classifier", "models/video_classifier/best_video_model.pth"),
        ("Text Classifier", "models/text_classifier/best_text_model_contextual.pt"),
        ("Fusion Classifier", "models/fusion_classifier/optimized_fusion_model.pth"),
        ("Face Detector (YuNet)", "models/face_detection_yunet_2023mar.onnx")
    ]
    all_ok = True

    for name, path in checkpoints:
        if os.path.exists(path):
            sz_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   [OK] {name:<22} : {path} ({sz_mb:.2f} MB)")
        else:
            print(f"   [FAIL] {name:<22} : NOT FOUND at {path}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 2. AUDIO EXTRACTION & WAV2VEC2 EXTRACTION
# ---------------------------------------------------------------------------
def extract_audio(split: str = "dev", max_files: int = None):
    print(f"\n[AUDIO EXTRACTION] Extracting 16kHz mono WAV from {split.upper()} video split...")
    video_dirs = {
        "train": "MELD.Raw/train/train_splits",
        "dev": "MELD.Raw/dev/dev_splits_complete",
        "test": "MELD.Raw/test/output_repeated_splits_test"
    }
    out_dir = os.path.join(PROJECT_ROOT, "audio_data", split)
    os.makedirs(out_dir, exist_ok=True)

    vdir = video_dirs.get(split)
    if not vdir or not os.path.exists(vdir):
        print(f"[ERROR] Video directory not found: {vdir}")
        return

    mp4_files = [f for f in glob.glob(os.path.join(vdir, "*.mp4")) if not os.path.basename(f).startswith("._")]
    if max_files:
        mp4_files = mp4_files[:max_files]
    print(f"Found {len(mp4_files)} MP4 files to process -> Output: {out_dir}")

    success, failed = 0, 0
    for mp4 in mp4_files:
        sid = os.path.splitext(os.path.basename(mp4))[0]
        out_wav = os.path.join(out_dir, f"{sid}.wav")
        if os.path.exists(out_wav):
            success += 1
            continue
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


def extract_wav2vec(split: str = "dev", max_files: int = None):
    print(f"\n[EMBEDDING EXTRACTION] Extracting 768-D Wav2Vec2 representations for {split.upper()}...")
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import soundfile as sf
    import librosa

    csv_paths = {
        "train": "MELD.Raw/train/train_sent_emo.csv",
        "dev": "MELD.Raw/dev_sent_emo.csv",
        "test": "MELD.Raw/test_sent_emo.csv"
    }
    csv_path = csv_paths.get(split)
    if not csv_path or not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    audio_dir = os.path.join(PROJECT_ROOT, "audio_data", split)
    out_dir = os.path.join(PROJECT_ROOT, "audio_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Wav2Vec2 model (facebook/wav2vec2-base) on {device}...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base", local_files_only=True)
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", local_files_only=True).to(device)
    wav2vec.eval()

    records = []
    if max_files:
        df = df.head(max_files)

    for _, row in df.iterrows():
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
        except Exception:
            continue

    out_file = os.path.join(out_dir, f"{split}_embeddings.pt")
    torch.save(records, out_file)
    print(f"[DONE] Saved {len(records)} embeddings -> {out_file} ({os.path.getsize(out_file)/(1024*1024):.1f} MB)")


# ---------------------------------------------------------------------------
# 3. STATISTICAL VERIFICATION ON AUDIO EMBEDDINGS
# ---------------------------------------------------------------------------
def verify_embeddings():
    print("\n" + "=" * 70)
    print(" AUDIO EMBEDDING DATASET STATISTICAL VERIFICATION")
    print("=" * 70)

    emb_dir = os.path.join(PROJECT_ROOT, "audio_embeddings")
    splits = ["train", "dev", "test"]
    report = {}

    for split in splits:
        path = os.path.join(emb_dir, f"{split}_embeddings.pt")
        if not os.path.exists(path):
            print(f"   [FAIL] Missing {path}")
            continue

        sz_mb = os.path.getsize(path) / (1024 * 1024)
        data = torch.load(path, map_location="cpu", weights_only=False)
        num_samples = len(data)

        dim_768_count = 0
        all_embeddings = []
        emotion_counts = {emo: 0 for emo in EMOTIONS}

        for item in data:
            emb = item["audio_embedding"]
            all_embeddings.append(emb)
            emo = item.get("Emotion", "").lower()
            if emo in emotion_counts:
                emotion_counts[emo] += 1
            if list(emb.shape) == [768]:
                dim_768_count += 1

        all_tensor = torch.stack(all_embeddings)
        nan_count = torch.isnan(all_tensor).sum().item()
        inf_count = torch.isinf(all_tensor).sum().item()

        print(f"   [OK] {split.upper():<5}: {num_samples:>5} samples | {sz_mb:>6.2f} MB | Dim=768: {dim_768_count==num_samples} | NaNs: {nan_count}")

        report[split] = {
            "samples": num_samples,
            "size_mb": round(sz_mb, 2),
            "dim_correct": dim_768_count == num_samples,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "emotions": emotion_counts
        }

    out_json = os.path.join(PROJECT_ROOT, "results", "embedding_dataset_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[DONE] Statistical report generated and saved to -> {out_json}")
    return report


def inspect_audio(split: str = "dev"):
    import soundfile as sf
    audio_dir = os.path.join(PROJECT_ROOT, "audio_data", split)
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
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition Pipeline & Data CLI")
    parser.add_argument("--validate", action="store_true", help="Validate environment, dataset & checkpoints")
    parser.add_argument("--verify", action="store_true", help="Run statistical verification on embeddings")
    parser.add_argument("--extract", type=str, choices=["audio", "wav2vec"], help="Run feature extraction")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"], help="Dataset split")
    parser.add_argument("--inspect", action="store_true", help="Inspect audio duration & sample rate stats")
    parser.add_argument("--all", action="store_true", help="Run complete validation & verification suite")

    args = parser.parse_args()

    if args.validate or args.all:
        validate_environment()
        validate_meld_dataset()
        validate_checkpoints()

    if args.verify or args.all:
        verify_embeddings()

    if args.inspect:
        inspect_audio(split=args.split)

    if args.extract == "audio":
        extract_audio(split=args.split)
    elif args.extract == "wav2vec":
        extract_wav2vec(split=args.split)

    if not any([args.validate, args.verify, args.extract, args.inspect, args.all]):
        # Default action
        validate_environment()
        validate_checkpoints()
        verify_embeddings()


if __name__ == "__main__":
    main()
