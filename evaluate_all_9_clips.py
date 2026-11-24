"""
evaluate_all_9_clips.py
=======================
Evaluation of all 9 webcam emotion video clips in `webcam_test_dir`:
Anger.mp4, disgust.mp4, Disgust (2).mp4, Fear.mp4, Fear (2).mp4,
Joy.mp4, Neutal.mp4, sadness.mp4, surprise.mp4.

Executes:
- STEP 1: Dynamic loading with stem normalization (stripping ' (2)', etc.).
- STEP 2: Evaluates the 2 new clips on the ORIGINAL baseline pipeline, copies 7 clips from baseline_v2 -> results/webcam_baseline_v3.json.
- STEP 3: Evaluates all 9 clips on the CURRENT optimized pipeline -> results/webcam_current_v3.json.
- Computes both views:
  (a) Original 7 clips only (comparable to v2).
  (b) Full 9 clips view with pooled class-level metrics.
- Side-by-side comparison of old vs. new recordings (Disgust vs Disgust (2), Fear vs Fear (2)).
- Specific confusion pair flags (Sadness vs Neutral, Fear vs Surprise, Disgust vs Anger).
"""

import os
import sys
import time
import json
import yaml
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add artifact scratch directory to sys.path to access original baseline service
SCRATCH_DIR = Path("C:/Users/harsh/.gemini/antigravity/brain/5fbc627a-d167-41f4-a4f2-a714adaefeec/scratch")
if str(SCRATCH_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_DIR))

from benchmark_webcam import normalize_filename_stem, map_filename_to_emotion, EMOTIONS, LABEL_MAPPING, IDX_TO_LABEL


def load_config_test_dir(config_path: str = "config.yaml") -> str:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("webcam_test_dir", "New folder")
    return "New folder"


def evaluate_single_clip(service_instance: Any, vpath: Path, pipeline_name: str) -> Dict[str, Any]:
    """Runs a single MP4 clip through the service in memory at native FPS."""
    gt_emotion = map_filename_to_emotion(vpath.name)
    if gt_emotion is None:
        raise ValueError(f"Unknown emotion mapping for {vpath.name}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise IOError(f"Could not open {vpath.name}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    client_id = f"eval_{pipeline_name}_{vpath.stem.lower()}"
    if hasattr(service_instance, "reset_session"):
        service_instance.reset_session(client_id)

    frame_predictions: List[str] = []
    frame_confidences: List[float] = []
    gt_class_confidences: List[float] = []
    frame_latencies: List[float] = []

    frame_idx = 0
    first_correct_ms: Optional[float] = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        _, buf = cv2.imencode(".jpg", frame)
        raw_bytes = buf.tobytes()

        t0 = time.perf_counter()
        res = service_instance.predict_frame(raw_bytes, client_id=client_id, apply_smoothing=True)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        frame_latencies.append(lat_ms)

        pred_exp = res["apparent_expression"].lower() if res.get("apparent_expression") else "no_face"
        conf = float(res.get("calibrated_confidence", 0.0))
        probs = res.get("probabilities", {})
        gt_prob = float(probs.get(gt_emotion, 0.0))

        frame_predictions.append(pred_exp)
        frame_confidences.append(conf)
        gt_class_confidences.append(gt_prob)

        if pred_exp == gt_emotion and first_correct_ms is None:
            first_correct_ms = (frame_idx / native_fps) * 1000.0

        frame_idx += 1

    cap.release()
    n_frames = len(frame_predictions)
    duration_sec = n_frames / native_fps if native_fps > 0 else 0.0

    # Full Accuracy
    correct_frames = sum(1 for p in frame_predictions if p == gt_emotion)
    full_acc = (correct_frames / n_frames) * 100.0 if n_frames > 0 else 0.0

    # Mid-60% Accuracy
    m_start = int(n_frames * 0.20)
    m_end = int(n_frames * 0.80)
    mid_preds = frame_predictions[m_start:m_end]
    mid_correct = sum(1 for p in mid_preds if p == gt_emotion)
    mid60_acc = (mid_correct / max(len(mid_preds), 1)) * 100.0

    # Majority Vote
    majority_vote = Counter(frame_predictions).most_common(1)[0][0] if frame_predictions else "NONE"
    majority_correct = (majority_vote == gt_emotion)

    # Flicker Rate
    label_changes = sum(1 for i in range(1, n_frames) if frame_predictions[i] != frame_predictions[i - 1])
    flicker_rate = label_changes / duration_sec if duration_sec > 0 else 0.0

    avg_gt_conf = float(np.mean(gt_class_confidences)) if gt_class_confidences else 0.0

    return {
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
        "mean_latency_ms": round(float(np.mean(frame_latencies)), 2) if frame_latencies else 0.0,
        "distribution": dict(Counter(frame_predictions)),
        "frame_predictions": frame_predictions  # saved for pooled confusion matrix
    }


def compute_pooled_metrics(clips_data: Dict[str, Any], filter_filenames: Optional[List[str]] = None) -> Dict[str, Any]:
    """Computes summary metrics, pooled class metrics, and 7x7 confusion matrix."""
    selected = {k: v for k, v in clips_data.items() if (filter_filenames is None or k in filter_filenames)}

    all_true = []
    all_pred = []
    full_accs = []
    mid_accs = []
    majority_correct_cnt = 0

    # Class-level frame pooling: true_class -> list of (is_correct)
    class_frame_counts: Dict[str, int] = {emo: 0 for emo in EMOTIONS}
    class_frame_correct: Dict[str, int] = {emo: 0 for emo in EMOTIONS}

    for fname, cdata in selected.items():
        gt = cdata["ground_truth"]
        gt_idx = LABEL_MAPPING[gt]
        
        full_accs.append(cdata["frame_accuracy_pct"])
        mid_accs.append(cdata["mid60_accuracy_pct"])
        if cdata["majority_vote_correct"]:
            majority_correct_cnt += 1

        if "frame_predictions" in cdata:
            preds = cdata["frame_predictions"]
            for p in preds:
                if p in LABEL_MAPPING:
                    all_true.append(gt_idx)
                    all_pred.append(LABEL_MAPPING[p])
                    class_frame_counts[gt] += 1
                    if p == gt:
                        class_frame_correct[gt] += 1
        elif "distribution" in cdata:
            for p, cnt in cdata["distribution"].items():
                if p in LABEL_MAPPING:
                    p_idx = LABEL_MAPPING[p]
                    for _ in range(cnt):
                        all_true.append(gt_idx)
                        all_pred.append(p_idx)
                        class_frame_counts[gt] += 1
                        if p == gt:
                            class_frame_correct[gt] += 1

    n_classes = len(EMOTIONS)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_pooled_acc = {}

    for idx, emo in enumerate(EMOTIONS):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_precision[emo] = round(prec, 4)
        per_class_recall[emo] = round(rec, 4)
        per_class_f1[emo] = round(f1, 4)
        total_gt_frames = class_frame_counts[emo]
        per_class_pooled_acc[emo] = round(class_frame_correct[emo] / max(total_gt_frames, 1) * 100.0, 2)

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    overall_frame_acc = float(np.mean(np.array(all_true) == np.array(all_pred)) * 100.0) if all_true else 0.0

    # Specific Confusion Pair Analyses:
    sad_idx, neut_idx = LABEL_MAPPING["sadness"], LABEL_MAPPING["neutral"]
    sad_as_neut = int(cm[sad_idx, neut_idx])
    neut_as_sad = int(cm[neut_idx, sad_idx])

    fear_idx, surp_idx = LABEL_MAPPING["fear"], LABEL_MAPPING["surprise"]
    fear_as_surp = int(cm[fear_idx, surp_idx])
    surp_as_fear = int(cm[surp_idx, fear_idx])

    disg_idx, ang_idx = LABEL_MAPPING["disgust"], LABEL_MAPPING["anger"]
    disg_as_ang = int(cm[disg_idx, ang_idx])
    ang_as_disg = int(cm[ang_idx, disg_idx])

    return {
        "num_clips": len(selected),
        "majority_correct_count": majority_correct_cnt,
        "majority_accuracy_pct": round(majority_correct_cnt / max(len(selected), 1) * 100.0, 2),
        "mean_full_accuracy_pct": round(float(np.mean(full_accs)), 2) if full_accs else 0.0,
        "mean_mid60_accuracy_pct": round(float(np.mean(mid_accs)), 2) if mid_accs else 0.0,
        "overall_frame_accuracy_pct": round(overall_frame_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_pooled_acc": per_class_pooled_acc,
        "confusion_matrix": cm.tolist(),
        "confusion_flags": {
            "sadness_vs_neutral": {
                "sadness_predicted_as_neutral_frames": sad_as_neut,
                "neutral_predicted_as_sadness_frames": neut_as_sad,
                "neutral_clip_sadness_fp_rate": round(neut_as_sad / max(cm[neut_idx].sum(), 1) * 100.0, 2)
            },
            "fear_vs_surprise": {
                "fear_predicted_as_surprise_frames": fear_as_surp,
                "surprise_predicted_as_fear_frames": surp_as_fear
            },
            "disgust_vs_anger": {
                "disgust_predicted_as_anger_frames": disg_as_ang,
                "anger_predicted_as_disgust_frames": ang_as_disg
            }
        }
    }


def main():
    test_dir = load_config_test_dir()
    test_path = Path(test_dir)
    print(f"Loaded webcam_test_dir from config.yaml: {test_dir}")

    # Discover all .mp4 files dynamically
    all_mp4_files = sorted(list(test_path.glob("*.mp4")))
    print(f"Discovered {len(all_mp4_files)} MP4 video files:")
    for f in all_mp4_files:
        emo = map_filename_to_emotion(f.name)
        print(f"  - {f.name:18s} -> {emo}")

    ORIGINAL_7_FILENAMES = [
        "Anger.mp4", "disgust.mp4", "Fear.mp4", "Joy.mp4", "Neutal.mp4", "sadness.mp4", "surprise.mp4"
    ]
    NEW_2_FILENAMES = ["Disgust (2).mp4", "Fear (2).mp4"]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Fair Baseline for the 2 New Clips & Combine into baseline_v3.json
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 76)
    print("STEP 2: RUNNING ORIGINAL BASELINE ON THE 2 NEW CLIPS")
    print("#" * 76)

    # 1. Load baseline_v2
    with open("results/webcam_baseline_v2.json", "r", encoding="utf-8") as f:
        baseline_v2_json = json.load(f)

    baseline_v3_clips = {}
    # Copy existing 7 clips from baseline_v2 unchanged
    for k, v in baseline_v2_json["clips"].items():
        baseline_v3_clips[k] = v

    # 2. Run the 2 new clips through scratch_orig_baseline
    from scratch_orig_baseline import WebcamService as OrigWebcamService
    orig_service = OrigWebcamService()

    for fname in NEW_2_FILENAMES:
        vpath = test_path / fname
        if vpath.exists():
            print(f"Evaluating {fname} on ORIGINAL BASELINE...")
            res_clip = evaluate_single_clip(orig_service, vpath, "orig_baseline_v3")
            baseline_v3_clips[fname] = res_clip
            print(f"  Result: Majority={res_clip['majority_vote_label'].upper()} (Correct={res_clip['majority_vote_correct']}) | Acc={res_clip['frame_accuracy_pct']}% | Dist={res_clip['distribution']}")
        else:
            print(f"[WARN] {fname} not found in {test_dir}!")

    # Compute pooled metrics for all 9 clips under baseline
    baseline_v3_summary = compute_pooled_metrics(baseline_v3_clips)

    baseline_v3_json = {
        "pipeline": "original_baseline_v3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": {k: {col: val for col, val in v.items() if col != "frame_predictions"} for k, v in baseline_v3_clips.items()},
        "summary": baseline_v3_summary
    }

    with open("results/webcam_baseline_v3.json", "w", encoding="utf-8") as f:
        json.dump(baseline_v3_json, f, indent=2)
    print("\nSuccessfully created results/webcam_baseline_v3.json (9 clips)")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Evaluate All 9 Clips on Current Pipeline
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 76)
    print("STEP 3: RUNNING CURRENT OPTIMIZED PIPELINE ON ALL 9 CLIPS")
    print("#" * 76)

    from backend.services.webcam_service import webcam_service as current_service
    current_v3_clips = {}

    for f in all_mp4_files:
        emo = map_filename_to_emotion(f.name)
        if emo is None:
            continue
        print(f"Evaluating {f.name} on CURRENT PIPELINE...")
        res_clip = evaluate_single_clip(current_service, f, "current_pipeline_v3")
        current_v3_clips[f.name] = res_clip
        status = "[PASS]" if res_clip["majority_vote_correct"] else "[FAIL]"
        print(f"  {status} Majority={res_clip['majority_vote_label'].upper():8s} | Acc={res_clip['frame_accuracy_pct']:5.1f}% | Mid={res_clip['mid60_accuracy_pct']:5.1f}% | Flicker={res_clip['flicker_rate']:4.2f} | Dist={res_clip['distribution']}")

    current_v3_summary_9 = compute_pooled_metrics(current_v3_clips)
    current_v3_summary_7 = compute_pooled_metrics(current_v3_clips, filter_filenames=ORIGINAL_7_FILENAMES)

    current_v3_json = {
        "pipeline": "current_pipeline_v3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": {k: {col: val for col, val in v.items() if col != "frame_predictions"} for k, v in current_v3_clips.items()},
        "summary_all_9_clips": current_v3_summary_9,
        "summary_original_7_clips": current_v3_summary_7
    }

    with open("results/webcam_current_v3.json", "w", encoding="utf-8") as f:
        json.dump(current_v3_json, f, indent=2)
    print("\nSuccessfully created results/webcam_current_v3.json (9 clips)")

    # ──────────────────────────────────────────────────────────────────────────
    # DETAILED REPORTING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("REPORT VIEW (A): ORIGINAL 7 CLIPS ONLY (Direct Comparison to v2)")
    print("=" * 76)
    s7 = current_v3_summary_7
    print(f"Majority Correct : {s7['majority_correct_count']}/7 ({s7['majority_accuracy_pct']}%)")
    print(f"Mean Full Acc    : {s7['mean_full_accuracy_pct']}%")
    print(f"Mean Mid-60% Acc : {s7['mean_mid60_accuracy_pct']}%")
    print(f"Macro-F1 (7)     : {s7['macro_f1']:.4f}")

    print("\n" + "=" * 76)
    print("REPORT VIEW (B): ALL 9 CLIPS (Full Evaluation Set)")
    print("=" * 76)
    s9 = current_v3_summary_9
    b9 = baseline_v3_summary
    print(f"Majority Correct Clips : {b9['majority_correct_count']}/9 ({b9['majority_accuracy_pct']}%) -> {s9['majority_correct_count']}/9 ({s9['majority_accuracy_pct']}%)")
    print(f"Mean Full Acc          : {b9['mean_full_accuracy_pct']}% -> {s9['mean_full_accuracy_pct']}% (Delta: {s9['mean_full_accuracy_pct'] - b9['mean_full_accuracy_pct']:+.2f}%)")
    print(f"Mean Mid-60% Acc       : {b9['mean_mid60_accuracy_pct']}% -> {s9['mean_mid60_accuracy_pct']}% (Delta: {s9['mean_mid60_accuracy_pct'] - b9['mean_mid60_accuracy_pct']:+.2f}%)")
    print(f"Overall Frame Acc      : {b9['overall_frame_accuracy_pct']}% -> {s9['overall_frame_accuracy_pct']}% (Delta: {s9['overall_frame_accuracy_pct'] - b9['overall_frame_accuracy_pct']:+.2f}%)")
    print(f"Macro-F1               : {b9['macro_f1']:.4f} -> {s9['macro_f1']:.4f} (Delta: {s9['macro_f1'] - b9['macro_f1']:+.4f})")

    print("\nPer-Clip Full Comparison (Baseline_v3 vs Current_v3):")
    for fname in sorted(list(current_v3_clips.keys())):
        b_c = baseline_v3_clips.get(fname, {})
        c_c = current_v3_clips[fname]
        gt = c_c["ground_truth"]
        b_maj = b_c.get("majority_vote_label", "N/A").upper()
        b_pass = "PASS" if b_c.get("majority_vote_correct") else "FAIL"
        c_maj = c_c["majority_vote_label"].upper()
        c_pass = "PASS" if c_c["majority_vote_correct"] else "FAIL"
        print(f"  {fname:18s} (GT: {gt.upper():8s}):")
        print(f"     Majority Vote: {b_maj:8s} ({b_pass}) -> {c_maj:8s} ({c_pass})")
        print(f"     Full Accuracy: {b_c.get('frame_accuracy_pct', 0.0):5.1f}% -> {c_c['frame_accuracy_pct']:5.1f}% (Delta: {c_c['frame_accuracy_pct'] - b_c.get('frame_accuracy_pct', 0.0):+5.1f}%)")
        print(f"     Mid Accuracy : {b_c.get('mid60_accuracy_pct', 0.0):5.1f}% -> {c_c['mid60_accuracy_pct']:5.1f}%")
        print(f"     Flicker Rate : {b_c.get('flicker_rate', 0.0):5.2f} -> {c_c['flicker_rate']:5.2f} switches/sec")
        print(f"     TTFC         : {str(b_c.get('time_to_first_correct_ms')) + ' ms' if b_c.get('time_to_first_correct_ms') is not None else 'NEVER'} -> {str(c_c['time_to_first_correct_ms']) + ' ms' if c_c['time_to_first_correct_ms'] is not None else 'NEVER'}")
        print(f"     Current Dist : {c_c['distribution']}")

    print("\n" + "=" * 76)
    print("SIDE-BY-SIDE RECORDING COMPARISON (OLD vs RE-RECORDED)")
    print("=" * 76)
    print("1. DISGUST COMPARISON:")
    print(f"   Old: disgust.mp4      -> Majority: {current_v3_clips['disgust.mp4']['majority_vote_label'].upper()} | Acc: {current_v3_clips['disgust.mp4']['frame_accuracy_pct']}% | Breakdown: {current_v3_clips['disgust.mp4']['distribution']}")
    print(f"   New: Disgust (2).mp4  -> Majority: {current_v3_clips['Disgust (2).mp4']['majority_vote_label'].upper()} | Acc: {current_v3_clips['Disgust (2).mp4']['frame_accuracy_pct']}% | Breakdown: {current_v3_clips['Disgust (2).mp4']['distribution']}")

    print("\n2. FEAR COMPARISON:")
    print(f"   Old: Fear.mp4         -> Majority: {current_v3_clips['Fear.mp4']['majority_vote_label'].upper()} | Acc: {current_v3_clips['Fear.mp4']['frame_accuracy_pct']}% | Breakdown: {current_v3_clips['Fear.mp4']['distribution']}")
    print(f"   New: Fear (2).mp4     -> Majority: {current_v3_clips['Fear (2).mp4']['majority_vote_label'].upper()} | Acc: {current_v3_clips['Fear (2).mp4']['frame_accuracy_pct']}% | Breakdown: {current_v3_clips['Fear (2).mp4']['distribution']}")

    print("\n" + "=" * 76)
    print("POOLED CLASS-LEVEL METRICS (All frames of same ground truth class):")
    print("=" * 76)
    for emo in EMOTIONS:
        print(f"  {emo:10s} -> Pooled Acc: {s9['per_class_pooled_acc'][emo]:5.1f}% | Precision: {s9['per_class_precision'][emo]:.3f} | Recall: {s9['per_class_recall'][emo]:.3f} | F1: {s9['per_class_f1'][emo]:.4f}")

    print("\n7x7 Confusion Matrix (Rows=True, Cols=Pred) Across All 9 Clips:")
    header = "       " + " ".join(f"{emo[:4]:>5}" for emo in EMOTIONS)
    print(header)
    cm = np.array(s9["confusion_matrix"])
    for idx, emo in enumerate(EMOTIONS):
        row_str = f"{emo[:6]:<6} " + " ".join(f"{cm[idx, j]:5d}" for j in range(len(EMOTIONS)))
        print(row_str)

    print("\nConfusion Pairs:")
    cf = s9["confusion_flags"]
    print(f"  Sadness vs Neutral: Sadness->Neutral = {cf['sadness_vs_neutral']['sadness_predicted_as_neutral_frames']} frames | Neutral->Sadness = {cf['sadness_vs_neutral']['neutral_predicted_as_sadness_frames']} frames (Neutral Sadness FP Rate = {cf['sadness_vs_neutral']['neutral_clip_sadness_fp_rate']}%)")
    print(f"  Fear vs Surprise  : Fear->Surprise = {cf['fear_vs_surprise']['fear_predicted_as_surprise_frames']} frames | Surprise->Fear = {cf['fear_vs_surprise']['surprise_predicted_as_fear_frames']} frames")
    print(f"  Disgust vs Anger  : Disgust->Anger = {cf['disgust_vs_anger']['disgust_predicted_as_anger_frames']} frames | Anger->Disgust = {cf['disgust_vs_anger']['anger_predicted_as_disgust_frames']} frames")


if __name__ == "__main__":
    main()
