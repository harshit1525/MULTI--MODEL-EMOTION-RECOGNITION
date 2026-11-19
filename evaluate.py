"""
Multimodal Emotion Recognition System (MELD Benchmark)
======================================================
Unified Model Evaluation, Benchmarking & Ablation Study

Usage:
  python evaluate.py --mode ablation      # 7-Configuration Multimodal Ablation Study
  python evaluate.py --mode benchmark     # End-to-end multimodal benchmark
  python evaluate.py --mode audio         # Audio classifier evaluation on test split
  python evaluate.py --mode all           # Complete evaluation suite
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models import (
    AudioClassifier,
    StrongerAudioClassifier,
    VideoClassifier,
    AdvancedCrossModalAttentionFusionNet,
    LearnedAttentionFusionNet,
    TextPreprocessingUtils,
    EMOTIONS
)
from backend.utils.metrics import compute_calibration_metrics, plot_confusion_matrix, plot_calibration_curves

EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}


# ---------------------------------------------------------------------------
# DATASET HELPER
# ---------------------------------------------------------------------------
class MELDAudioDataset(Dataset):
    def __init__(self, embeddings_path: str):
        raw = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        self.data = [d for d in raw if d.get("Emotion", "").lower() in EMOTION_TO_IDX]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["audio_embedding"], EMOTION_TO_IDX[item["Emotion"].lower()]


# ---------------------------------------------------------------------------
# 1. AUDIO CLASSIFIER EVALUATION
# ---------------------------------------------------------------------------
def evaluate_audio(model_path: str = "models/audio_classifier/best_audio_model_balanced.pth", output_dir: str = "results/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f" EVALUATING AUDIO EMOTION RECOGNITION (Held-Out Test Set)")
    print(f" Checkpoint: {model_path} | Device: {device}")
    print("=" * 70)

    test_path = os.path.join(PROJECT_ROOT, "audio_embeddings", "test_embeddings.pt")
    if not os.path.exists(test_path):
        print(f"[ERROR] Test embeddings missing at {test_path}")
        return None

    test_ds = MELDAudioDataset(test_path)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", "")
    if arch in ["StrongerAudioClassifier", "stronger_mlp"] or ckpt["model_state_dict"]["net.0.weight"].shape[0] == 512:
        model = StrongerAudioClassifier(768, 7).to(device)
    else:
        model = AudioClassifier(768, 256, 7).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for embs, lbls in test_loader:
            embs = embs.to(device)
            logits = model(embs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    print(f"\n[RESULTS] Accuracy    : {acc*100:.2f}%")
    print(f"          Macro F1    : {macro_f1*100:.2f}%")
    print(f"          Weighted F1 : {weighted_f1*100:.2f}%\n")

    cm_path = os.path.join(PROJECT_ROOT, "results", "plots", "audio_confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plot_confusion_matrix(all_labels, all_preds, EMOTIONS, cm_path, title=f"Audio Classifier ({os.path.basename(model_path)})")
    print(f"[SAVED] Confusion matrix: {cm_path}")

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# ---------------------------------------------------------------------------
# 2. 7-CONFIGURATION MULTIMODAL ABLATION STUDY
# ---------------------------------------------------------------------------
def run_ablation_study(
    ckpt_path: str = "models/fusion_classifier/optimized_fusion_model.pth",
    output_dir: str = "results"
):
    print("\n" + "=" * 70)
    print(" RUNNING 7-CONFIGURATION MULTIMODAL ABLATION BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        ckpt_path = "models/fusion_classifier/best_fusion_model.pth"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AdvancedCrossModalAttentionFusionNet(
        in_features=16,
        embed_dim=ckpt.get("embed_dim", 96),
        num_heads=ckpt.get("num_heads", 4),
        num_layers=ckpt.get("num_layers", 2),
        num_classes=7
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    T = float(ckpt.get("temperature", 0.42))
    print(f"[MODEL] Loaded {ckpt_path} (Calibration Temperature T*={T:.4f})")

    test_data_path = os.path.join(PROJECT_ROOT, "data", "fusion_test_dataset.pt")
    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test dataset missing at {test_data_path}")
        return None

    test_data = torch.load(test_data_path, map_location="cpu", weights_only=False)
    X_a = test_data["X_a"].to(device)
    X_t = test_data["X_t"].to(device)
    X_v = test_data["X_v"].to(device)
    y_test = test_data["y"].numpy()
    n_samples = len(y_test)

    # 7 Modality configurations: (Name, mask: [audio, text, video])
    configs = [
        ("Audio Only", (True, False, False)),
        ("Video Only", (False, False, True)),
        ("Text Only", (False, True, False)),
        ("Audio + Video", (True, False, True)),
        ("Audio + Text", (True, True, False)),
        ("Video + Text", (False, True, True)),
        ("Audio + Video + Text (Trimodal)", (True, True, True))
    ]

    print(f"\nEvaluating {n_samples} held-out test samples across 7 modality subsets:\n")
    print(f"{'Configuration':<32} | {'Accuracy':<9} | {'Macro F1':<9} | {'Weighted F1':<11} | {'Confidence':<10} | {'ECE':<8}")
    print("-" * 88)

    ablation_results = []
    for name, mask_vals in configs:
        mask = torch.tensor([list(mask_vals)] * n_samples, dtype=torch.bool, device=device)
        t0 = time.time()
        with torch.no_grad():
            logits, gates = model(X_a, X_t, X_v, mask=mask)
            logits_np = logits.cpu().numpy()
        latency_ms = ((time.time() - t0) / n_samples) * 1000

        scaled_logits = logits_np / T
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        acc = float(accuracy_score(y_test, preds))
        mf1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        wf1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        mean_conf = float(np.mean(confs))
        cal_m = compute_calibration_metrics(probs, y_test)

        print(f"{name:<32} | {acc*100:>7.2f}% | {mf1:>9.4f} | {wf1:>11.4f} | {mean_conf*100:>8.1f}% | {cal_m['ece']:>8.4f}")

        ablation_results.append({
            "configuration": name,
            "mask": mask_vals,
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "weighted_f1": round(wf1, 4),
            "mean_confidence": round(mean_conf, 4),
            "ece": cal_m["ece"],
            "brier_score": cal_m["brier_score"],
            "nll": cal_m["nll"],
            "latency_ms": round(latency_ms, 2)
        })

    print("-" * 88)

    # Save full report
    out_json = os.path.join(output_dir, "full_ablation_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n[DONE] Saved ablation benchmark results -> {out_json}")

    return ablation_results


# ---------------------------------------------------------------------------
# 3. END-TO-END MULTIMODAL BENCHMARK
# ---------------------------------------------------------------------------
def run_benchmark():
    print("\n" + "=" * 70)
    print(" RUNNING COMPLETE MULTIMODAL SYSTEM BENCHMARK")
    print("=" * 70)

    # Run audio evaluation
    audio_metrics = evaluate_audio()

    # Run full ablation study
    ablation_res = run_ablation_study()

    print("\n" + "=" * 70)
    print(" [SUMMARY] MULTIMODAL PERFORMANCE BENCHMARK COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI DISPATCHER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition Evaluation CLI")
    parser.add_argument("--mode", type=str, default="ablation", choices=["ablation", "benchmark", "audio", "all"], help="Evaluation mode")
    parser.add_argument("--checkpoint", type=str, default="models/fusion_classifier/optimized_fusion_model.pth", help="Specific checkpoint path")

    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation_study(ckpt_path=args.checkpoint)
    elif args.mode == "audio":
        evaluate_audio(model_path=args.checkpoint)
    elif args.mode in ["benchmark", "all"]:
        run_benchmark()


if __name__ == "__main__":
    main()
