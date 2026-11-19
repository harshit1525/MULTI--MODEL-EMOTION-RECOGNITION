import os
import sys
import argparse
import subprocess
import shutil
import pandas as pd
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# 1. ENVIRONMENT VALIDATION
# ---------------------------------------------------------------------------
def check_environment() -> bool:
    print("\n" + "=" * 60)
    print(" 1. ENVIRONMENT & HARDWARE DEPENDENCY AUDIT")
    print("=" * 60)
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
        print("   [WARN] ffmpeg       : Not found in system PATH (needed only for raw video audio extraction).")

    return all_ok

# ---------------------------------------------------------------------------
# 2. MELD BENCHMARK DATASET INTEGRITY
# ---------------------------------------------------------------------------
def check_meld() -> bool:
    print("\n" + "=" * 60)
    print(" 2. MELD BENCHMARK DATASET VALIDATION")
    print("=" * 60)
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

# ---------------------------------------------------------------------------
# 3. EMBEDDING INTEGRITY & SHAPE VERIFICATION
# ---------------------------------------------------------------------------
def check_embeddings() -> bool:
    print("\n" + "=" * 60)
    print(" 3. EXTRACTED EMBEDDING TENSOR INTEGRITY AUDIT")
    print("=" * 60)
    files = {
        "train": "audio_embeddings/train_embeddings.pt",
        "dev": "audio_embeddings/dev_embeddings.pt",
        "test": "audio_embeddings/test_embeddings.pt"
    }
    all_ok = True

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"   [FAIL] Missing embedding file: {path}")
            all_ok = False
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        data = torch.load(path, map_location="cpu", weights_only=False)
        print(f"   [OK] {name.upper():<5} Embeddings: {len(data):>5} samples ({size_mb:.1f} MB) at {path}")

        if len(data) > 0:
            first_emb = data[0]["audio_embedding"]
            shape = list(first_emb.shape)
            has_nan = torch.isnan(first_emb).any().item()
            print(f"        Sample ID: {data[0]['sample_id']}, Tensor Shape: {shape}, NaN: {has_nan}")
            if shape != [768] or has_nan:
                print(f"   [FAIL] Embedding shape mismatch or NaNs detected in {name}!")
                all_ok = False

    return all_ok

# ---------------------------------------------------------------------------
# 4. SYSTEM END-TO-END AUDIT
# ---------------------------------------------------------------------------
def check_system() -> bool:
    print("\n" + "=" * 60)
    print(" 4. SYSTEM API & IN-PROCESS AUDIT")
    print("=" * 60)
    try:
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)

        r = client.get("/health")
        if r.status_code != 200:
            print(f"   [FAIL] GET /health returned {r.status_code}")
            return False

        data = r.json()
        print(f"   [OK] Backend health: {data['status']}, device: {data['device']}")
        for m, loaded in data.get("models_loaded", {}).items():
            status = "LOADED" if loaded else "MISSING"
            print(f"        Model '{m}': {status}")

        r_webcam = client.get("/predict/webcam/status")
        if r_webcam.status_code == 200:
            wdata = r_webcam.json()
            print(f"   [OK] Webcam Service: {wdata['status']} (Detector: {wdata['detector_available']})")

        return True
    except Exception as e:
        print(f"   [FAIL] System audit exception: {e}")
        return False

# ---------------------------------------------------------------------------
# CLI DISPATCHER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Multimodal System Validation & Integrity Verification CLI")
    parser.add_argument("--check", type=str, default="all", choices=["environment", "meld", "embeddings", "system", "all"], help="Check target")

    args = parser.parse_args()

    results = {}
    if args.check in ["environment", "all"]:
        results["Environment"] = check_environment()

    if args.check in ["meld", "all"]:
        results["MELD Dataset"] = check_meld()

    if args.check in ["embeddings", "all"]:
        results["Embeddings"] = check_embeddings()

    if args.check in ["system", "all"]:
        results["System"] = check_system()

    print("\n" + "=" * 60)
    print(" VALIDATION SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f" {name:<25} : {status}")
    print("=" * 60)

if __name__ == "__main__":
    main()

