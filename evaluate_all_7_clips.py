"""
evaluate_all_7_clips.py
=======================
Evaluation of all 7 webcam emotion video clips in `webcam_test_dir` (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise).

Executes:
- STEP 1: Dynamic loading & case-insensitive filename mapping.
- STEP 2: Fair baseline evaluation using the original baseline pipeline -> results/webcam_baseline_v2.json.
- STEP 3: Current pipeline evaluation -> results/webcam_current_v2.json.
- Full comparative analysis: Frame Acc (Full/Mid), Majority Vote, TTFC, Flicker, 7x7 Confusion Matrix, Macro-F1.
- Specific confusion flagging (Sadness vs Neutral, Fear vs Surprise, Disgust vs Anger).
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
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_MAPPING = {emo: idx for idx, emo in enumerate(EMOTIONS)}
IDX_TO_LABEL = {idx: emo for idx, emo in enumerate(EMOTIONS)}


def load_config_test_dir(config_path: str = "config.yaml") -> str:
    """Reads webcam_test_dir dynamically from config.yaml."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("webcam_test_dir", "New folder")
    return "New folder"


def map_filename_to_emotion(filename: str) -> Optional[str]:
    """
    Derives ground truth emotion from filename (case-insensitive, tolerating typos).
    - anger -> anger
    - disgust -> disgust
    - fear -> fear
    - joy -> joy
    - neutal / neutral -> neutral
    - sadness / sad -> sadness
    - surprise / surp -> surprise
    """
    stem = Path(filename).stem.strip().lower()
    if "anger" in stem:
        return "anger"
    elif "disgust" in stem:
        return "disgust"
    elif "fear" in stem:
        return "fear"
    elif "joy" in stem:
        return "joy"
    elif "neut" in stem:
        return "neutral"
    elif "sad" in stem:
        return "sadness"
    elif "surp" in stem:
        return "surprise"
    print(f"[WARN] Unknown filename '{filename}' cannot be mapped to any of {EMOTIONS}. Skipping.")
    return None


def run_pipeline_on_clips(service_instance: Any, test_dir: str, pipeline_name: str) -> Dict[str, Any]:
    """
    Runs video clips through the provided webcam service instance entirely in-memory at native FPS.
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    mp4_files = sorted(list(test_path.glob("*.mp4")))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 video files found in {test_dir}")

    clips_data: Dict[str, Any] = {}
    all_true_labels: List[int] = []
    all_pred_labels: List[int] = []

    total_correct_majority = 0
    total_clips_evaluated = 0
    full_accuracies: List[float] = []
    mid60_accuracies: List[float] = []

    print("\n" + "=" * 76)
    print(f"EVALUATING: {pipeline_name.upper()} on {len(mp4_files)} clips in '{test_dir}'")
    print("=" * 76)

    for vpath in mp4_files:
        gt_emotion = map_filename_to_emotion(vpath.name)
        if gt_emotion is None:
            continue

        gt_idx = LABEL_MAPPING[gt_emotion]
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"[ERROR] Could not open {vpath.name}")
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

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

            # In-memory JPEG encoding simulating stream
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

            if pred_exp in LABEL_MAPPING:
                all_true_labels.append(gt_idx)
                all_pred_labels.append(LABEL_MAPPING[pred_exp])

            if pred_exp == gt_emotion and first_correct_ms is None:
                first_correct_ms = (frame_idx / native_fps) * 1000.0

            frame_idx += 1

        cap.release()
        n_frames = len(frame_predictions)
        if n_frames == 0:
            continue

        duration_sec = n_frames / native_fps

        # Full Accuracy
        correct_frames = sum(1 for p in frame_predictions if p == gt_emotion)
        full_acc = (correct_frames / n_frames) * 100.0
        full_accuracies.append(full_acc)

        # Mid-60% Accuracy
        m_start = int(n_frames * 0.20)
        m_end = int(n_frames * 0.80)
        mid_preds = frame_predictions[m_start:m_end]
        mid_correct = sum(1 for p in mid_preds if p == gt_emotion)
        mid60_acc = (mid_correct / max(len(mid_preds), 1)) * 100.0
        mid60_accuracies.append(mid60_acc)

        # Majority Vote
        majority_vote = Counter(frame_predictions).most_common(1)[0][0]
        majority_correct = (majority_vote == gt_emotion)
        if majority_correct:
            total_correct_majority += 1
        total_clips_evaluated += 1

        # Flicker Rate
        label_changes = sum(1 for i in range(1, n_frames) if frame_predictions[i] != frame_predictions[i - 1])
        flicker_rate = label_changes / duration_sec if duration_sec > 0 else 0.0

        # Mean GT class confidence
        avg_gt_conf = float(np.mean(gt_class_confidences)) if gt_class_confidences else 0.0

        clip_result = {
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
            "distribution": dict(Counter(frame_predictions))
        }
        clips_data[vpath.name] = clip_result

        status_str = "[PASS]" if majority_correct else "[FAIL]"
        print(f"{status_str} {vpath.name:14s} (GT: {gt_emotion.upper():8s}) -> Maj: {majority_vote.upper():8s} | Full: {full_acc:5.1f}% | Mid: {mid60_acc:5.1f}% | Flicker: {flicker_rate:4.2f} | TTFC: {str(clip_result['time_to_first_correct_ms']) + ' ms' if clip_result['time_to_first_correct_ms'] is not None else 'NEVER'}")
        print(f"      Distribution: {dict(Counter(frame_predictions))}")

    # 7x7 Confusion Matrix across all frames
    n_classes = len(EMOTIONS)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_true_labels, all_pred_labels):
        cm[t, p] += 1

    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}
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

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    overall_frame_acc = float(np.mean(np.array(all_true_labels) == np.array(all_pred_labels)) * 100.0) if all_true_labels else 0.0

    # Specific Confusion Pair Analyses:
    # 1. Sadness vs Neutral
    sad_idx, neut_idx = LABEL_MAPPING["sadness"], LABEL_MAPPING["neutral"]
    sad_as_neut = int(cm[sad_idx, neut_idx])
    neut_as_sad = int(cm[neut_idx, sad_idx])
    # 2. Fear vs Surprise
    fear_idx, surp_idx = LABEL_MAPPING["fear"], LABEL_MAPPING["surprise"]
    fear_as_surp = int(cm[fear_idx, surp_idx])
    surp_as_fear = int(cm[surp_idx, fear_idx])
    # 3. Disgust vs Anger
    disg_idx, ang_idx = LABEL_MAPPING["disgust"], LABEL_MAPPING["anger"]
    disg_as_ang = int(cm[disg_idx, ang_idx])
    ang_as_disg = int(cm[ang_idx, disg_idx])

    summary = {
        "pipeline_name": pipeline_name,
        "total_clips": total_clips_evaluated,
        "majority_correct_count": total_correct_majority,
        "majority_accuracy_pct": round(total_correct_majority / max(total_clips_evaluated, 1) * 100.0, 2),
        "mean_full_accuracy_pct": round(float(np.mean(full_accuracies)), 2),
        "mean_mid60_accuracy_pct": round(float(np.mean(mid60_accuracies)), 2),
        "overall_frame_accuracy_pct": round(overall_frame_acc, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
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

    print("\n" + "-" * 76)
    print(f"SUMMARY ({pipeline_name.upper()}):")
    print(f"  Clips Evaluated       : {total_clips_evaluated}")
    print(f"  Majority Vote Accuracy: {total_correct_majority}/{total_clips_evaluated} ({summary['majority_accuracy_pct']}%)")
    print(f"  Mean Frame Acc (Full) : {summary['mean_full_accuracy_pct']}%")
    print(f"  Mean Frame Acc (Mid)  : {summary['mean_mid60_accuracy_pct']}%")
    print(f"  Overall Frame Acc     : {summary['overall_frame_accuracy_pct']}%")
    print(f"  Macro-F1 (7 classes)  : {summary['macro_f1']:.4f}")
    print("\nConfusion Pairs:")
    print(f"  Sadness vs Neutral: Sadness->Neutral={sad_as_neut} frames | Neutral->Sadness={neut_as_sad} frames (FP rate={summary['confusion_flags']['sadness_vs_neutral']['neutral_clip_sadness_fp_rate']}%)")
    print(f"  Fear vs Surprise  : Fear->Surprise={fear_as_surp} frames | Surprise->Fear={surp_as_fear} frames")
    print(f"  Disgust vs Anger  : Disgust->Anger={disg_as_ang} frames | Anger->Disgust={ang_as_disg} frames")

    print("\n7x7 Confusion Matrix (Rows=True, Cols=Pred):")
    header = "       " + " ".join(f"{emo[:4]:>5}" for emo in EMOTIONS)
    print(header)
    for idx, emo in enumerate(EMOTIONS):
        row_str = f"{emo[:6]:<6} " + " ".join(f"{cm[idx, j]:5d}" for j in range(n_classes))
        print(row_str)

    return {
        "pipeline": pipeline_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": clips_data,
        "summary": summary
    }


def main():
    test_dir = load_config_test_dir()
    print(f"Loaded webcam_test_dir from config.yaml: {test_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Fair Baseline for All 7 Clips using Original Baseline Service
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 76)
    print("STEP 2: RUNNING ORIGINAL BASELINE ON ALL 7 CLIPS")
    print("#" * 76)
    try:
        from scratch_orig_baseline import WebcamService as OrigWebcamService
        orig_service = OrigWebcamService()
        baseline_v2_results = run_pipeline_on_clips(orig_service, test_dir, "original_baseline_v2")

        os.makedirs("results", exist_ok=True)
        with open("results/webcam_baseline_v2.json", "w", encoding="utf-8") as f:
            json.dump(baseline_v2_results, f, indent=2)
        print("\nSuccessfully saved original baseline results to results/webcam_baseline_v2.json")
    except Exception as e:
        print(f"[ERROR] Failed to run original baseline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Evaluate Current Pipeline on All 7 Clips
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "#" * 76)
    print("STEP 3: RUNNING CURRENT OPTIMIZED PIPELINE ON ALL 7 CLIPS")
    print("#" * 76)
    try:
        from backend.services.webcam_service import webcam_service as current_service
        current_v2_results = run_pipeline_on_clips(current_service, test_dir, "current_pipeline_v2")

        with open("results/webcam_current_v2.json", "w", encoding="utf-8") as f:
            json.dump(current_v2_results, f, indent=2)
        print("\nSuccessfully saved current pipeline results to results/webcam_current_v2.json")
    except Exception as e:
        print(f"[ERROR] Failed to run current pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: Print Comparative Summary Table
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("COMPARATIVE EVALUATION: BASELINE_V2 vs CURRENT (7 CLIPS)")
    print("=" * 76)

    b_sum = baseline_v2_results["summary"]
    c_sum = current_v2_results["summary"]

    print(f"Majority Correct Clips : {b_sum['majority_correct_count']}/7 ({b_sum['majority_accuracy_pct']}%) -> {c_sum['majority_correct_count']}/7 ({c_sum['majority_accuracy_pct']}%)")
    print(f"Mean Frame Acc (Full)  : {b_sum['mean_full_accuracy_pct']}% -> {c_sum['mean_full_accuracy_pct']}% (Delta: {c_sum['mean_full_accuracy_pct'] - b_sum['mean_full_accuracy_pct']:+.2f}%)")
    print(f"Mean Frame Acc (Mid-60%): {b_sum['mean_mid60_accuracy_pct']}% -> {c_sum['mean_mid60_accuracy_pct']}% (Delta: {c_sum['mean_mid60_accuracy_pct'] - b_sum['mean_mid60_accuracy_pct']:+.2f}%)")
    print(f"Macro-F1 (7 classes)   : {b_sum['macro_f1']:.4f} -> {c_sum['macro_f1']:.4f} (Delta: {c_sum['macro_f1'] - b_sum['macro_f1']:+.4f})")

    print("\nPer-Clip Breakdown:")
    for clip_name in baseline_v2_results["clips"]:
        b_c = baseline_v2_results["clips"][clip_name]
        c_c = current_v2_results["clips"][clip_name]
        gt = b_c["ground_truth"]
        b_pass = "PASS" if b_c["majority_vote_correct"] else "FAIL"
        c_pass = "PASS" if c_c["majority_vote_correct"] else "FAIL"
        print(f"  {clip_name:14s} (GT: {gt.upper():8s}):")
        print(f"     Majority Vote: {b_c['majority_vote_label'].upper():8s} ({b_pass}) -> {c_c['majority_vote_label'].upper():8s} ({c_pass})")
        print(f"     Full Accuracy: {b_c['frame_accuracy_pct']:5.1f}% -> {c_c['frame_accuracy_pct']:5.1f}% (Delta: {c_c['frame_accuracy_pct'] - b_c['frame_accuracy_pct']:+5.1f}%)")
        print(f"     Mid Accuracy : {b_c['mid60_accuracy_pct']:5.1f}% -> {c_c['mid60_accuracy_pct']:5.1f}%")
        print(f"     Flicker Rate : {b_c['flicker_rate']:5.2f} -> {c_c['flicker_rate']:5.2f} switches/sec")
        print(f"     TTFC         : {str(b_c['time_to_first_correct_ms']) + ' ms' if b_c['time_to_first_correct_ms'] is not None else 'NEVER'} -> {str(c_c['time_to_first_correct_ms']) + ' ms' if c_c['time_to_first_correct_ms'] is not None else 'NEVER'}")


if __name__ == "__main__":
    main()
