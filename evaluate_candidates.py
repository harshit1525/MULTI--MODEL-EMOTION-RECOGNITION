"""
evaluate_candidates.py
======================
Comprehensive Calibration & Multi-Dimensional Benchmark for Webcam Emotion Models.

Evaluates:
- Step 4: Temperature Calibration on Validation Set (ECE before and after T* fitting).
- Step 5A: RAF-DB Held-Out Test Set (3,068 images).
- Step 5B: MELD Held-Out Face Crops (699 images).
- Step 5C: All 9 User Webcam Clips in `New folder/` through the exact live pipeline in memory.
- Step 5D: End-to-end Latency & Throughput (p50, p95, mean latency, FPS).

Saves to:
  results/webcam_rafdb_expA.json
  results/webcam_rafdb_expB.json
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.webcam_service import WebcamService, EMOTIONS
from benchmark_webcam import compute_ece, map_filename_to_emotion, normalize_filename_stem
from train_rafdb import (
    EmotionDataset,
    create_splits,
    RAF_FOLDER_TO_IDX,
    LABEL_MAPPING,
    IDX_TO_LABEL
)


# ─── TEMPERATURE FITTING ──────────────────────────────────────────────────────
def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits temperature T* by minimizing negative log-likelihood on validation logits.
    Returns: (T_opt, ece_before, ece_after)
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    nll_fn = nn.CrossEntropyLoss()

    def loss_fun(T):
        scaled = logits_t / T
        return nll_fn(scaled, labels_t).item()

    res = minimize_scalar(loss_fun, bounds=(0.1, 5.0), method="bounded")
    T_opt = float(res.x)

    probs_uncal = torch.softmax(logits_t, dim=-1).numpy()
    probs_cal = torch.softmax(logits_t / T_opt, dim=-1).numpy()

    ece_before = compute_ece(probs_uncal, labels)
    ece_after = compute_ece(probs_cal, labels)
    return round(T_opt, 4), round(ece_before, 4), round(ece_after, 4)


# ─── EVALUATE DATASET SPLIT ───────────────────────────────────────────────────
def evaluate_dataset_split(service: WebcamService, samples: List[Tuple[Path, int]], img_size: int = 224) -> Dict[str, Any]:
    """Evaluates an image list using the service's backbone and preprocessing."""
    all_true = []
    all_pred = []
    all_logits = []
    all_probs = []

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    service.backbone.eval()
    with torch.inference_mode():
        for p, label in samples:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            resized = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = norm(t).unsqueeze(0).to(service.device)

            logits = service.backbone(t)[0].cpu().numpy()
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            pred = int(np.argmax(probs))

            all_true.append(label)
            all_pred.append(pred)
            all_logits.append(logits)
            all_probs.append(probs)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)

    acc = float(np.mean(all_true == all_pred) * 100.0)
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    per_class_f1 = {}
    per_class_prec = {}
    per_class_rec = {}
    for idx, emo in enumerate(EMOTIONS):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_prec[emo] = round(prec, 4)
        per_class_rec[emo] = round(rec, 4)
        per_class_f1[emo] = round(f1, 4)

    macro_f1 = float(np.mean(list(per_class_f1.values())))

    return {
        "accuracy_pct": round(acc, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_prec,
        "per_class_recall": per_class_rec,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(all_true),
        "logits": all_logits,
        "labels": all_true,
    }


# ─── EVALUATE USER WEBCAM CLIPS ────────────────────────────────────────────────
def evaluate_webcam_clips(service: WebcamService, clips_dir: str = "New folder", pipeline_name: str = "eval") -> Dict[str, Any]:
    """Runs all MP4 clips in `clips_dir` in memory through the live pipeline."""
    cdir = Path(clips_dir)
    clips = sorted(list(cdir.glob("*.mp4")))
    clip_results = {}

    all_true_frames = []
    all_pred_frames = []
    class_frame_counts = {emo: 0 for emo in EMOTIONS}
    class_frame_correct = {emo: 0 for emo in EMOTIONS}
    full_accs = []
    mid60_accs = []
    majority_correct_cnt = 0

    for cpath in clips:
        gt_emo = map_filename_to_emotion(cpath.name)
        if gt_emo is None:
            continue
        gt_idx = LABEL_MAPPING[gt_emo]

        cap = cv2.VideoCapture(str(cpath))
        if not cap.isOpened():
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        client_id = f"{pipeline_name}_{cpath.stem.lower()}"
        service.reset_session(client_id)

        preds = []
        confs = []
        gt_confs = []
        lats = []
        frame_idx = 0
        first_correct_ms = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            _, buf = cv2.imencode(".jpg", frame)
            raw_bytes = buf.tobytes()

            t0 = time.perf_counter()
            res = service.predict_frame(raw_bytes, client_id=client_id, apply_smoothing=True)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            lats.append(lat_ms)

            pred_exp = res["apparent_expression"].lower() if res.get("apparent_expression") else "no_face"
            conf = float(res.get("calibrated_confidence", 0.0))
            probs = res.get("probabilities", {})
            gt_prob = float(probs.get(gt_emo, 0.0))

            preds.append(pred_exp)
            confs.append(conf)
            gt_confs.append(gt_prob)

            if pred_exp == gt_emo and first_correct_ms is None:
                first_correct_ms = (frame_idx / native_fps) * 1000.0

            if pred_exp in LABEL_MAPPING:
                p_idx = LABEL_MAPPING[pred_exp]
                all_true_frames.append(gt_idx)
                all_pred_frames.append(p_idx)
                class_frame_counts[gt_emo] += 1
                if pred_exp == gt_emo:
                    class_frame_correct[gt_emo] += 1

            frame_idx += 1

        cap.release()
        n_frames = len(preds)
        duration_sec = n_frames / native_fps if native_fps > 0 else 0.0

        correct_frames = sum(1 for p in preds if p == gt_emo)
        full_acc = (correct_frames / n_frames) * 100.0 if n_frames > 0 else 0.0

        m_start = int(n_frames * 0.20)
        m_end = int(n_frames * 0.80)
        mid_preds = preds[m_start:m_end]
        mid_correct = sum(1 for p in mid_preds if p == gt_emo)
        mid60_acc = (mid_correct / max(len(mid_preds), 1)) * 100.0

        maj_label = Counter(preds).most_common(1)[0][0] if preds else "NONE"
        maj_correct = (maj_label == gt_emo)
        if maj_correct:
            majority_correct_cnt += 1

        label_changes = sum(1 for i in range(1, n_frames) if preds[i] != preds[i - 1])
        flicker_rate = label_changes / duration_sec if duration_sec > 0 else 0.0
        avg_gt_conf = float(np.mean(gt_confs)) if gt_confs else 0.0

        full_accs.append(full_acc)
        mid60_accs.append(mid60_acc)

        clip_results[cpath.name] = {
            "filename": cpath.name,
            "ground_truth": gt_emo,
            "total_frames": n_frames,
            "duration_sec": round(duration_sec, 2),
            "native_fps": round(native_fps, 2),
            "frame_accuracy_pct": round(full_acc, 2),
            "mid60_accuracy_pct": round(mid60_acc, 2),
            "majority_vote_label": maj_label,
            "majority_vote_correct": maj_correct,
            "time_to_first_correct_ms": round(first_correct_ms, 1) if first_correct_ms is not None else None,
            "flicker_rate": round(flicker_rate, 3),
            "avg_correct_class_confidence": round(avg_gt_conf, 4),
            "mean_latency_ms": round(float(np.mean(lats)), 2) if lats else 0.0,
            "distribution": dict(Counter(preds))
        }

    # 7x7 confusion matrix for clips
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(all_true_frames, all_pred_frames):
        cm[t, p] += 1

    per_class_f1 = {}
    per_class_prec = {}
    per_class_rec = {}
    per_class_pooled_acc = {}
    for idx, emo in enumerate(EMOTIONS):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_prec[emo] = round(prec, 4)
        per_class_rec[emo] = round(rec, 4)
        per_class_f1[emo] = round(f1, 4)
        total_gt = class_frame_counts[emo]
        per_class_pooled_acc[emo] = round(class_frame_correct[emo] / max(total_gt, 1) * 100.0, 2)

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    overall_frame_acc = float(np.mean(np.array(all_true_frames) == np.array(all_pred_frames)) * 100.0) if all_true_frames else 0.0

    # Specific Confusion Flags
    sad_idx, neut_idx = LABEL_MAPPING["sadness"], LABEL_MAPPING["neutral"]
    fear_idx, surp_idx = LABEL_MAPPING["fear"], LABEL_MAPPING["surprise"]
    disg_idx, ang_idx = LABEL_MAPPING["disgust"], LABEL_MAPPING["anger"]

    neut_as_sad = int(cm[neut_idx, sad_idx])
    neut_total = int(cm[neut_idx].sum())
    neut_sad_fp_rate = round(neut_as_sad / max(neut_total, 1) * 100.0, 2)

    return {
        "clips": clip_results,
        "pooled_summary": {
            "num_clips": len(clip_results),
            "majority_correct_count": majority_correct_cnt,
            "majority_accuracy_pct": round(majority_correct_cnt / max(len(clip_results), 1) * 100.0, 2),
            "mean_full_accuracy_pct": round(float(np.mean(full_accs)), 2) if full_accs else 0.0,
            "mean_mid60_accuracy_pct": round(float(np.mean(mid60_accs)), 2) if mid60_accs else 0.0,
            "overall_frame_accuracy_pct": round(overall_frame_acc, 2),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1,
            "per_class_precision": per_class_prec,
            "per_class_recall": per_class_rec,
            "per_class_pooled_acc": per_class_pooled_acc,
            "confusion_matrix": cm.tolist(),
            "confusion_flags": {
                "sadness_vs_neutral": {
                    "sadness_predicted_as_neutral_frames": int(cm[sad_idx, neut_idx]),
                    "neutral_predicted_as_sadness_frames": neut_as_sad,
                    "neutral_clip_sadness_fp_rate": neut_sad_fp_rate
                },
                "fear_vs_surprise": {
                    "fear_predicted_as_surprise_frames": int(cm[fear_idx, surp_idx]),
                    "surprise_predicted_as_fear_frames": int(cm[surp_idx, fear_idx])
                },
                "disgust_vs_anger": {
                    "disgust_predicted_as_anger_frames": int(cm[disg_idx, ang_idx]),
                    "anger_predicted_as_disgust_frames": int(cm[ang_idx, disg_idx])
                }
            }
        }
    }


# ─── MEASURE PIPELINE LATENCY ──────────────────────────────────────────────────
def measure_pipeline_latency(service: WebcamService, n_frames: int = 100) -> Dict[str, float]:
    """Measures end-to-end frame prediction latency on synthetic webcam frames."""
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Draw simple face-like ellipse so YuNet detects a face
    cv2.ellipse(dummy_frame, (320, 240), (100, 130), 0, 0, 360, (200, 200, 200), -1)
    cv2.circle(dummy_frame, (285, 200), 15, (50, 50, 50), -1)
    cv2.circle(dummy_frame, (355, 200), 15, (50, 50, 50), -1)
    cv2.ellipse(dummy_frame, (320, 290), (40, 20), 0, 0, 180, (50, 50, 50), -1)

    _, buf = cv2.imencode(".jpg", dummy_frame)
    raw_bytes = buf.tobytes()

    # Warmup
    for _ in range(5):
        service.predict_frame(raw_bytes, client_id="bench_warmup", apply_smoothing=True)

    latencies = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        service.predict_frame(raw_bytes, client_id="bench_lat", apply_smoothing=True)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "effective_fps": round(fps, 1),
    }


# ─── MAIN CANDIDATE EVALUATION RUNNER ──────────────────────────────────────────
def run_candidate_evaluation(ckpt_path: str, exp_name: str, out_json: str):
    print(f"\n{'='*70}\nEVALUATING CANDIDATE {exp_name}: {ckpt_path}\n{'='*70}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load service
    service = WebcamService(video_ckpt=ckpt_path)
    img_size = getattr(service, "_img_size", 224)

    # 1. Prepare Splits
    train_samples, val_samples, raf_test, meld_test = create_splits(
        rafdb_dir=Path("DATASET"),
        meld_cache_dir=Path("models/video_classifier/face_crops_cache"),
        seed=42
    )

    # 2. Step 4: Temperature Calibration on Validation Set
    print("\n--- STEP 4: Temperature Calibration on Validation Set ---")
    val_res = evaluate_dataset_split(service, val_samples, img_size=img_size)
    T_opt, ece_before, ece_after = fit_temperature(val_res["logits"], val_res["labels"])
    print(f"Optimal Temperature T*: {T_opt:.4f}")
    print(f"Validation ECE: {ece_before:.4f} -> {ece_after:.4f} (fitted on validation set only)")

    # Save calibrated temperature into checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["calibrated_temperature"] = T_opt
    torch.save(ckpt, ckpt_path)
    service.calibrated_temperature = T_opt

    # 3. Step 5A: RAF-DB Held-Out Test Set
    print("\n--- STEP 5A: RAF-DB Held-Out Test Set ---")
    raf_res = evaluate_dataset_split(service, raf_test, img_size=img_size)
    print(f"RAF-DB Test Accuracy: {raf_res['accuracy_pct']:.2f}%")
    print(f"RAF-DB Test Macro-F1: {raf_res['macro_f1']:.4f}")
    print(f"RAF-DB Per-class F1: {raf_res['per_class_f1']}")

    # 4. Step 5B: MELD Held-Out Face Crops
    print("\n--- STEP 5B: MELD Held-Out Crops ---")
    meld_res = evaluate_dataset_split(service, meld_test, img_size=img_size)
    print(f"MELD Held-Out Macro-F1: {meld_res['macro_f1']:.4f}")
    print(f"MELD Held-Out Accuracy: {meld_res['accuracy_pct']:.2f}%")

    # 5. Step 5C: User Webcam Clips (Live Pipeline In Memory)
    print("\n--- STEP 5C: All 9 User Webcam Clips (Exact Live Pipeline) ---")
    clips_res = evaluate_webcam_clips(service, clips_dir="New folder", pipeline_name=f"eval_{exp_name}")
    pooled = clips_res["pooled_summary"]
    print(f"Clips Majority Correct: {pooled['majority_correct_count']} / {pooled['num_clips']} ({pooled['majority_accuracy_pct']}%)")
    print(f"Overall Frame Accuracy: {pooled['overall_frame_accuracy_pct']:.2f}%")
    print(f"Clips Macro-F1: {pooled['macro_f1']:.4f}")
    print(f"Pooled Per-Class Acc: {pooled['per_class_pooled_acc']}")
    print(f"Neutral clip sadness FP rate: {pooled['confusion_flags']['sadness_vs_neutral']['neutral_clip_sadness_fp_rate']}%")

    # 6. Step 5D: End-to-End Latency
    print("\n--- STEP 5D: End-to-End Latency & Throughput ---")
    lat_res = measure_pipeline_latency(service, n_frames=80)
    print(f"Latency: p50={lat_res['p50_latency_ms']} ms, p95={lat_res['p95_latency_ms']} ms, Mean={lat_res['mean_latency_ms']} ms, FPS={lat_res['effective_fps']}")

    # Compile Final JSON Report
    output_data = {
        "candidate": exp_name,
        "checkpoint": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration": {
            "optimal_temperature": T_opt,
            "validation_ece_before": ece_before,
            "validation_ece_after": ece_after,
        },
        "rafdb_test": {
            "total_samples": raf_res["total_samples"],
            "accuracy_pct": raf_res["accuracy_pct"],
            "macro_f1": raf_res["macro_f1"],
            "per_class_f1": raf_res["per_class_f1"],
            "per_class_precision": raf_res["per_class_precision"],
            "per_class_recall": raf_res["per_class_recall"],
            "confusion_matrix": raf_res["confusion_matrix"],
        },
        "meld_heldout": {
            "total_samples": meld_res["total_samples"],
            "accuracy_pct": meld_res["accuracy_pct"],
            "macro_f1": meld_res["macro_f1"],
            "per_class_f1": meld_res["per_class_f1"],
            "per_class_precision": meld_res["per_class_precision"],
            "per_class_recall": meld_res["per_class_recall"],
            "confusion_matrix": meld_res["confusion_matrix"],
        },
        "user_webcam_clips": clips_res,
        "latency": lat_res,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Successfully saved evaluation report to: {out_json}")
    return output_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=str, required=True, choices=["A", "B", "a", "b"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    c = args.candidate.upper()
    ckpt = args.ckpt or f"models/video_classifier/best_webcam_model_rafdb_{c}.pth"
    out = args.output or f"results/webcam_rafdb_exp{c}.json"

    run_candidate_evaluation(ckpt_path=ckpt, exp_name=f"Exp_{c}", out_json=out)
