"""
benchmark_webcam.py
===================
Reproducible Benchmark Suite for Real-Time Webcam Emotion Recognition Pipeline.

Evaluates:
  a) Held-out face-crop metrics: Accuracy, Macro-F1, per-class F1, Confusion Matrix, ECE.
  b) Neutral-frame / grey-frame bias using check_model_bias.py logic.
  c) End-to-end latency (p50/p95, mean) and FPS.
  d) Temporal stability: flicker rate (label switches/sec) on static sequences.
  e) ECE after temperature calibration.
  f) Real webcam user clips in `webcam_test_dir` (Anger, Joy, Neutal, surprise):
     - Frame-level accuracy (full clip & mid-60%)
     - Majority-vote label & correctness
     - Time-to-first-correct prediction (ms)
     - Flicker rate (label switches/sec)
     - Average confidence of ground truth class
     - False positive tracking for absent classes (sadness, disgust, fear)

Usage:
  python benchmark_webcam.py [--output results/webcam_baseline.json]
"""

import os
import sys
import re
import time
import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np
import yaml
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.webcam_service import webcam_service, EMOTIONS

LABEL_MAPPING = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam Emotion Recognition Benchmark Suite")
    parser.add_argument("--output", type=str, default="results/webcam_baseline.json",
                        help="Path to save benchmark JSON output")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--cache_dir", type=str, default="models/video_classifier/face_crops_cache",
                        help="Path to face crops cache directory")
    parser.add_argument("--seed", type=int, default=101,
                        help="Random seed for held-out split")
    parser.add_argument("--split_ratio", type=float, default=0.90,
                        help="Train/val split ratio (val = 1.0 - split_ratio)")
    parser.add_argument("--max_per_class", type=int, default=1200,
                        help="Max samples per class in crop cache")
    parser.add_argument("--trace", action="store_true",
                        help="Record per-frame probabilities and top class after each pipeline stage")
    return parser.parse_args()


# ─── HELPER: ECE CALCULATION ──────────────────────────────────────────────────
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width confidence bins.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


# ─── SECTION A & E: HELD-OUT FACE CROPS BENCHMARK ─────────────────────────────
def evaluate_held_out_crops(cache_dir: str, seed: int = 101, split_ratio: float = 0.90,
                            max_per_class: int = 1200) -> Dict[str, Any]:
    """
    Evaluates model on strictly held-out split of face crops.
    """
    print("\n" + "=" * 70)
    print("SECTION A: HELD-OUT FACE CROPS EVALUATION")
    print("=" * 70)

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"[WARN] Cache dir {cache_dir} not found. Skipping held-out crop eval.")
        return {"status": "skipped", "error": f"Cache dir {cache_dir} not found"}

    samples: List[Tuple[Path, int]] = []
    rng = random.Random(seed)

    for cls, idx in LABEL_MAPPING.items():
        cls_dir = cache_path / cls
        if not cls_dir.exists():
            continue
        paths = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(paths)
        if max_per_class > 0 and len(paths) > max_per_class:
            paths = paths[:max_per_class]
        split_at = int(len(paths) * split_ratio)
        val_paths = paths[split_at:]  # Held-out 10%
        for p in val_paths:
            samples.append((p, idx))

    print(f"Total held-out validation crops: {len(samples)} (Seed={seed}, Split={split_ratio})")
    counts = Counter(s[1] for s in samples)
    for cls, idx in LABEL_MAPPING.items():
        print(f"  {cls:10s}: {counts.get(idx, 0)} crops")

    if not samples:
        return {"status": "empty"}

    # Prepare model & normalization
    device = webcam_service.device
    model = webcam_service.backbone
    img_size = getattr(webcam_service, "_img_size", 96)
    norm = webcam_service.norm or transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    all_preds = []
    all_targets = []
    all_raw_probs = []
    all_calib_probs = []

    model.eval()
    with torch.no_grad():
        for p, label in samples:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            resized = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = norm(t).unsqueeze(0).to(device)

            logits = model(t)
            raw_p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Post-hoc temperature scaling (as configured in service, default T* = 0.85)
            temp = 0.85
            log_p = np.log(np.clip(raw_p, 1e-6, 1.0))
            centered = log_p - np.mean(log_p)
            scaled = centered / temp
            exp_s = np.exp(scaled - np.max(scaled))
            calib_p = exp_s / np.sum(exp_s)

            pred_idx = int(np.argmax(calib_p))
            all_preds.append(pred_idx)
            all_targets.append(label)
            all_raw_probs.append(raw_p)
            all_calib_probs.append(calib_p)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_calib_probs = np.array(all_calib_probs)

    # Accuracy
    total_acc = float(np.mean(all_preds == all_targets) * 100.0)

    # Confusion Matrix (7x7)
    n_classes = len(LABEL_MAPPING)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    # Per-class metrics: Precision, Recall, F1
    per_class_f1: Dict[str, float] = {}
    per_class_precision: Dict[str, float] = {}
    per_class_recall: Dict[str, float] = {}

    for cls, idx in LABEL_MAPPING.items():
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class_precision[cls] = round(prec, 4)
        per_class_recall[cls] = round(rec, 4)
        per_class_f1[cls] = round(f1, 4)

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    ece = compute_ece(all_calib_probs, all_targets, n_bins=10)

    print(f"\nHeld-Out Accuracy : {total_acc:.2f}%")
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"ECE (Calibrated)  : {ece:.4f}")
    print("\nPer-Class Breakdown:")
    for cls in LABEL_MAPPING:
        print(f"  {cls:10s} -> Precision: {per_class_precision[cls]:.3f} | Recall: {per_class_recall[cls]:.3f} | F1: {per_class_f1[cls]:.4f}")

    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    header = "       " + " ".join(f"{cls[:4]:>5}" for cls in LABEL_MAPPING)
    print(header)
    for idx, cls in enumerate(LABEL_MAPPING):
        row_str = f"{cls[:6]:<6} " + " ".join(f"{cm[idx, j]:5d}" for j in range(n_classes))
        print(row_str)

    return {
        "num_samples": len(samples),
        "accuracy_pct": round(total_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "ece": round(ece, 4),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "class_labels": list(LABEL_MAPPING.keys()),
    }


# ─── SECTION B: NEUTRAL / GREY FRAME BIAS ─────────────────────────────────────
def evaluate_neutral_bias() -> Dict[str, Any]:
    """
    Evaluates baseline bias on a blank neutral grey frame (no face features).
    Uses check_model_bias.py logic.
    """
    print("\n" + "=" * 70)
    print("SECTION B: BLANK GREY FRAME BIAS TEST")
    print("=" * 70)

    device = webcam_service.device
    model = webcam_service.backbone
    img_size = getattr(webcam_service, "_img_size", 96)
    norm = webcam_service.norm or transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    grey_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 128
    t = torch.from_numpy(grey_img).permute(2, 0, 1).float() / 255.0
    t = norm(t).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(t)[0].cpu().numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()

    winner_idx = int(np.argmax(logits))
    winner_label = IDX_TO_LABEL[winner_idx]
    max_spread = float(np.max(logits) - np.min(logits))

    raw_logits_dict = {IDX_TO_LABEL[i]: round(float(logits[i]), 4) for i in range(len(logits))}
    probs_dict = {IDX_TO_LABEL[i]: round(float(probs[i]), 4) for i in range(len(probs))}

    print(f"Blank Frame Winner  : {winner_label.upper()} ({probs[winner_idx]*100:.1f}%)")
    print(f"Max Logit Spread    : {max_spread:.4f} (lower is better, <1.0 = balanced)")
    print("Raw Logits & Probabilities:")
    for cls in LABEL_MAPPING:
        bar = "#" * int(probs_dict[cls] * 50)
        print(f"  {cls:10s}: logit={raw_logits_dict[cls]:+6.3f} | prob={probs_dict[cls]*100:5.1f}% | {bar}")

    return {
        "predicted_class": winner_label,
        "predicted_confidence": round(float(probs[winner_idx]), 4),
        "max_logit_spread": round(max_spread, 4),
        "raw_logits": raw_logits_dict,
        "probabilities": probs_dict,
    }


# ─── SECTION C: END-TO-END LATENCY & FPS PROFILE ──────────────────────────────
def evaluate_latency_and_fps(n_frames: int = 50) -> Dict[str, Any]:
    """
    Measures end-to-end inference latency: capture/decode -> YuNet detect -> crop/align -> EfficientNet -> smoothing -> response.
    """
    print("\n" + "=" * 70)
    print(f"SECTION C: END-TO-END LATENCY & FPS PROFILE ({n_frames} frames)")
    print("=" * 70)

    # Synthesize realistic 1280x720 webcam frame with a face region
    dummy_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 180
    # Add a mock oval face shape to trigger YuNet detection path gracefully
    cv2.ellipse(dummy_frame, (640, 360), (120, 160), 0, 0, 360, (140, 170, 210), -1)
    _, buf = cv2.imencode(".jpg", dummy_frame)
    frame_bytes = buf.tobytes()

    client_id = "bench_latency"
    webcam_service.reset_session(client_id)

    # Warmup
    for _ in range(5):
        webcam_service.predict_frame(frame_bytes, client_id=client_id, apply_smoothing=False)

    latencies_ms = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        webcam_service.predict_frame(frame_bytes, client_id=client_id, apply_smoothing=True)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    latencies_ms = np.array(latencies_ms)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    effective_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    print(f"Mean Latency : {mean_lat:.2f} ms")
    print(f"p50 Latency  : {p50:.2f} ms")
    print(f"p95 Latency  : {p95:.2f} ms")
    print(f"p99 Latency  : {p99:.2f} ms")
    print(f"Effective FPS: {effective_fps:.1f} FPS")

    return {
        "num_frames": n_frames,
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "effective_fps": round(effective_fps, 1),
    }


# ─── SECTION D: TEMPORAL STABILITY & FLICKER RATE ─────────────────────────────
def evaluate_temporal_stability(n_frames: int = 60, target_fps: float = 15.0) -> Dict[str, Any]:
    """
    Measures temporal flicker rate on a static expression frame with simulated sensor noise.
    Flicker rate = label changes per second.
    """
    print("\n" + "=" * 70)
    print(f"SECTION D: TEMPORAL STABILITY & FLICKER RATE ({n_frames} frames @ {target_fps} FPS)")
    print("=" * 70)

    # Base frame with synthetic face
    base_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 160
    cv2.ellipse(base_frame, (640, 360), (120, 160), 0, 0, 360, (140, 170, 210), -1)

    client_id = "bench_stability"
    webcam_service.reset_session(client_id)

    predicted_labels = []
    confidences = []

    for i in range(n_frames):
        # Add slight sensor noise (+/- 2 intensity levels)
        noise = np.random.randint(-2, 3, base_frame.shape, dtype=np.int16)
        noisy_frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", noisy_frame)

        res = webcam_service.predict_frame(buf.tobytes(), client_id=client_id, apply_smoothing=True)
        lbl = res["apparent_expression"].lower() if res["apparent_expression"] else "no_face"
        predicted_labels.append(lbl)
        confidences.append(res["calibrated_confidence"])

    # Count label changes
    label_switches = 0
    for i in range(1, len(predicted_labels)):
        if predicted_labels[i] != predicted_labels[i - 1]:
            label_switches += 1

    duration_sec = n_frames / target_fps
    flicker_rate = float(label_switches / duration_sec)
    dominant_label = Counter(predicted_labels).most_common(1)[0][0]

    print(f"Total Frames        : {n_frames} ({duration_sec:.1f}s)")
    print(f"Dominant Expression : {dominant_label.upper()}")
    print(f"Label Switches      : {label_switches}")
    print(f"Flicker Rate        : {flicker_rate:.2f} switches/second (lower is better, target < 1.0)")

    return {
        "num_frames": n_frames,
        "duration_sec": round(duration_sec, 2),
        "dominant_expression": dominant_label,
        "label_switches": label_switches,
        "flicker_rate": round(flicker_rate, 3),
        "avg_confidence": round(float(np.mean(confidences)), 4),
    }


# ─── SECTION F: USER WEBCAM VIDEO CLIPS BENCHMARK ─────────────────────────────
def normalize_filename_stem(filename: str) -> str:
    """
    Normalizes a video filename stem: lowercases and strips trailing duplicate suffixes
    such as ' (2)', ' (3)', '_2', '-2', ' - Copy'.
    """
    stem = Path(filename).stem.strip().lower()
    stem = re.sub(r'[\s_\-]+copy$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'[\s_\-]*\(\d+\)$', '', stem)
    stem = re.sub(r'[\s_\-]+\d+$', '', stem)
    return stem.strip()


def map_filename_to_emotion(filename: str) -> Optional[str]:
    """
    Derives ground truth emotion from filename (case-insensitive, tolerating typos and suffixes).
    Anger.mp4        -> anger
    Disgust (2).mp4  -> disgust
    disgust.mp4      -> disgust
    Fear (2).mp4     -> fear
    Fear.mp4         -> fear
    Joy.mp4          -> joy
    Neutal.mp4       -> neutral  (handles typo)
    sadness.mp4      -> sadness
    surprise.mp4     -> surprise
    """
    clean_stem = normalize_filename_stem(filename)
    if "anger" in clean_stem:
        return "anger"
    elif "disgust" in clean_stem:
        return "disgust"
    elif "fear" in clean_stem:
        return "fear"
    elif "joy" in clean_stem:
        return "joy"
    elif "neut" in clean_stem:  # Matches 'neutral', 'neutal', etc.
        return "neutral"
    elif "sad" in clean_stem:
        return "sadness"
    elif "surp" in clean_stem:
        return "surprise"
    print(f"[WARN] Unknown filename '{filename}' cannot be mapped to any emotion. Skipping.")
    return None


def evaluate_user_clips(test_dir: str, trace: bool = False) -> Dict[str, Any]:
    """
    Evaluates real webcam MP4 clips in `test_dir`.
    Rules:
      - Simulates live stream frame by frame in memory at native FPS.
      - Never writes frames to disk.
      - Reports full clip and middle 60% accuracy.
      - Tracks majority vote, time-to-first-correct (ms), flicker rate, correct class confidence,
        and false positives for absent classes (sadness, fear, disgust).
    """
    print("\n" + "=" * 70)
    print(f"SECTION F: REAL-WEBCAM USER CLIPS EVALUATION ({test_dir})")
    print("=" * 70)

    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"[ERROR] Test dir {test_dir} not found.")
        return {"status": "error", "message": f"Test dir {test_dir} not found"}

    TARGET_CLIPS = ["anger.mp4", "joy.mp4", "neutal.mp4", "surprise.mp4"]
    video_files = [f for f in sorted(list(test_path.glob("*.mp4"))) if f.name.lower() in TARGET_CLIPS]
    if not video_files:
        video_files = sorted(list(test_path.glob("*.mp4")))
    if not video_files:
        print(f"[ERROR] No MP4 files found in {test_dir}.")
        return {"status": "error", "message": f"No MP4 files in {test_dir}"}

    clips_results: Dict[str, Any] = {}
    total_correct_majority = 0
    total_clips_evaluated = 0
    full_accuracies = []
    mid60_accuracies = []
    absent_fps_counts: Counter = Counter()
    sadness_on_neutral_count = 0
    neutral_total_frames = 0

    for vpath in video_files:
        gt_emotion = map_filename_to_emotion(vpath.name)
        if gt_emotion is None:
            print(f"[SKIP] Cannot map {vpath.name} to known emotion.")
            continue

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[ERROR] Could not open {vpath.name}")
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        clip_id = f"eval_clip_{vpath.stem.lower()}"
        webcam_service.reset_session(clip_id)

        frame_predictions: List[str] = []
        frame_confidences: List[float] = []
        gt_class_confidences: List[float] = []
        frame_latencies: List[float] = []
        clip_traces: List[Dict[str, Any]] = []

        frame_idx = 0
        first_correct_ms: Optional[float] = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Simulate live incoming frame through live pipeline in memory
            _, buf = cv2.imencode(".jpg", frame)
            raw_bytes = buf.tobytes()

            t_start = time.perf_counter()
            res = webcam_service.predict_frame(raw_bytes, client_id=clip_id, apply_smoothing=True, return_trace=trace)
            lat_ms = (time.perf_counter() - t_start) * 1000.0
            frame_latencies.append(lat_ms)
            if trace and res.get("trace"):
                clip_traces.append(res["trace"])

            pred_exp = res["apparent_expression"].lower() if res["apparent_expression"] else "no_face"
            conf = float(res["calibrated_confidence"])
            probs = res.get("probabilities", {})
            gt_prob = float(probs.get(gt_emotion, 0.0))

            frame_predictions.append(pred_exp)
            frame_confidences.append(conf)
            gt_class_confidences.append(gt_prob)

            if pred_exp == gt_emotion and first_correct_ms is None:
                first_correct_ms = (frame_idx / native_fps) * 1000.0

            # Track absent class predictions (fear, disgust, sadness)
            if pred_exp in ["fear", "disgust", "sadness"]:
                absent_fps_counts[pred_exp] += 1
            if gt_emotion == "neutral":
                neutral_total_frames += 1
                if pred_exp == "sadness":
                    sadness_on_neutral_count += 1

            frame_idx += 1

        cap.release()
        n_frames = len(frame_predictions)
        if n_frames == 0:
            continue

        if trace and clip_traces:
            print(f"\n--- STAGE TRACE FOR {vpath.name} ({len(clip_traces)} frames) ---")
            stages = ["stage1_raw_cnn", "stage2_prior", "stage3_user_baseline", "stage4_geometry", "stage5_smoothed", "stage6_calibrated"]
            first_overtake_stage = None
            for st in stages:
                valid_probs = [tr[st]["probs"] for tr in clip_traces if tr and st in tr]
                if not valid_probs:
                    continue
                mean_p = np.mean(valid_probs, axis=0)
                neu_p = mean_p[4] * 100
                surp_p = mean_p[6] * 100
                top_emo = EMOTIONS[int(np.argmax(mean_p))]
                if surp_p > neu_p and first_overtake_stage is None:
                    first_overtake_stage = st
                marker = " <-- SURPRISE > NEUTRAL" if surp_p > neu_p else ""
                print(f"  {st:22s} | Top: {top_emo:8s} | Neutral: {neu_p:5.1f}% | Surprise: {surp_p:5.1f}%{marker}")
            if first_overtake_stage:
                print(f"  => First stage where Surprise overtakes Neutral: {first_overtake_stage}")
            else:
                print(f"  => Surprise never overtakes Neutral across all stages.")

        duration_sec = n_frames / native_fps

        # Frame-level accuracy (Full clip)
        correct_frames = sum(1 for p in frame_predictions if p == gt_emotion)
        full_acc = (correct_frames / n_frames) * 100.0
        full_accuracies.append(full_acc)

        # Middle 60% accuracy (20% to 80%)
        m_start = int(n_frames * 0.20)
        m_end = int(n_frames * 0.80)
        mid_preds = frame_predictions[m_start:m_end]
        mid_correct = sum(1 for p in mid_preds if p == gt_emotion)
        mid60_acc = (mid_correct / max(len(mid_preds), 1)) * 100.0
        mid60_accuracies.append(mid60_acc)

        # Majority-vote label
        majority_vote = Counter(frame_predictions).most_common(1)[0][0]
        majority_correct = (majority_vote == gt_emotion)
        if majority_correct:
            total_correct_majority += 1
        total_clips_evaluated += 1

        # Flicker rate (label changes per second)
        label_changes = sum(1 for i in range(1, n_frames) if frame_predictions[i] != frame_predictions[i - 1])
        flicker_rate = label_changes / duration_sec if duration_sec > 0 else 0.0

        # Average confidence of correct class
        avg_gt_conf = float(np.mean(gt_class_confidences)) if gt_class_confidences else 0.0

        clip_data = {
            "filename": vpath.name,
            "ground_truth": gt_emotion,
            "total_frames": n_frames,
            "duration_sec": round(duration_sec, 2),
            "native_fps": round(native_fps, 2),
            "frame_accuracy_pct": round(full_acc, 2),
            "mid60_accuracy_pct": round(mid60_acc, 2),
            "majority_vote_label": majority_vote,
            "majority_vote_correct": majority_correct,
            "time_to_first_correct_ms": round(first_correct_ms, 1) if first_correct_ms is not None else None,
            "flicker_rate": round(flicker_rate, 3),
            "avg_correct_class_confidence": round(avg_gt_conf, 4),
            "mean_latency_ms": round(float(np.mean(frame_latencies)), 2),
            "class_distribution": dict(Counter(frame_predictions)),
        }
        clips_results[vpath.name] = clip_data

        status_sym = "[PASS]" if majority_correct else "[FAIL]"
        print(f"\n{status_sym} Clip: {vpath.name} (GT: {gt_emotion.upper()})")
        print(f"  Frames: {n_frames} | Duration: {duration_sec:.1f}s | FPS: {native_fps:.1f}")
        print(f"  Accuracy (Full)       : {full_acc:.1f}% ({correct_frames}/{n_frames})")
        print(f"  Accuracy (Mid-60%)    : {mid60_acc:.1f}% ({mid_correct}/{len(mid_preds)})")
        print(f"  Majority Vote         : {majority_vote.upper()} (Correct: {majority_correct})")
        ttfc_str = f"{first_correct_ms:.1f} ms" if first_correct_ms is not None else "NEVER"
        print(f"  Time-To-First-Correct : {ttfc_str}")
        print(f"  Flicker Rate          : {flicker_rate:.2f} switches/sec ({label_changes} switches)")
        print(f"  Avg GT Confidence     : {avg_gt_conf*100:.1f}%")
        print(f"  Predictions Breakdown : {dict(Counter(frame_predictions))}")

    # Summary across user clips
    mean_full_acc = float(np.mean(full_accuracies)) if full_accuracies else 0.0
    mean_mid60_acc = float(np.mean(mid60_accuracies)) if mid60_accuracies else 0.0
    all_majority_correct = (total_correct_majority == total_clips_evaluated and total_clips_evaluated > 0)
    sadness_on_neutral_rate = (sadness_on_neutral_count / max(neutral_total_frames, 1)) * 100.0

    print("\n" + "-" * 70)
    print("USER CLIPS OVERALL SUMMARY:")
    print(f"  Clips Evaluated          : {total_clips_evaluated}")
    print(f"  Majority Vote Accuracy   : {total_correct_majority}/{total_clips_evaluated} ({total_correct_majority/max(total_clips_evaluated,1)*100:.1f}%)")
    print(f"  Mean Frame Accuracy (Full): {mean_full_acc:.1f}%")
    print(f"  Mean Frame Accuracy (Mid) : {mean_mid60_acc:.1f}%")
    print(f"  Absent Class FPs (Total)  : {dict(absent_fps_counts)}")
    print(f"  Sadness on Neutral Clip   : {sadness_on_neutral_count}/{neutral_total_frames} frames ({sadness_on_neutral_rate:.1f}%)")

    return {
        "clips": clips_results,
        "summary": {
            "total_clips": total_clips_evaluated,
            "majority_correct_count": total_correct_majority,
            "all_majority_correct": all_majority_correct,
            "mean_full_accuracy_pct": round(mean_full_acc, 2),
            "mean_mid60_accuracy_pct": round(mean_mid60_acc, 2),
            "absent_class_fps": dict(absent_fps_counts),
            "sadness_on_neutral_frames": sadness_on_neutral_count,
            "sadness_on_neutral_rate_pct": round(sadness_on_neutral_rate, 2),
        }
    }


# ─── MAIN RUNNER ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print("=" * 70)
    print("  MULTIMODAL EMOTION RECOGNITION - WEBCAM BENCHMARK SUITE")
    print("=" * 70)

    # 1. Load config
    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    test_dir = cfg.get("webcam_test_dir", "New folder")

    # 2. Section A & E: Held-Out Crops Evaluation & ECE
    held_out_metrics = evaluate_held_out_crops(
        cache_dir=args.cache_dir,
        seed=args.seed,
        split_ratio=args.split_ratio,
        max_per_class=args.max_per_class,
    )

    # 3. Section B: Neutral / Grey Frame Bias
    bias_metrics = evaluate_neutral_bias()

    # 4. Section C: End-to-End Latency & FPS
    latency_metrics = evaluate_latency_and_fps(n_frames=50)

    # 5. Section D: Temporal Stability & Flicker Rate
    stability_metrics = evaluate_temporal_stability(n_frames=60, target_fps=15.0)

    # 6. Section F: User Webcam Video Clips
    user_clips_metrics = evaluate_user_clips(test_dir=test_dir, trace=args.trace)

    # 7. Assemble Full Benchmark Record
    full_benchmark = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": getattr(webcam_service, "_architecture", "unknown"),
        "img_size": getattr(webcam_service, "_img_size", 96),
        "held_out_crops": held_out_metrics,
        "neutral_bias": bias_metrics,
        "latency_and_fps": latency_metrics,
        "temporal_stability": stability_metrics,
        "user_clips": user_clips_metrics,
    }

    # 8. Save JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_benchmark, f, indent=2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE! Full results saved to: {out_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
