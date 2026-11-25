#!/usr/bin/env python3
"""
eval_clips.py – Evaluate webcam emotion detection on personal video clips.

Usage:
    python eval_clips.py --backbone efficientnet_current
    python eval_clips.py --backbone resnet18_raf
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Filename stem → ground truth label (case-insensitive, strip trailing " (N)")
STEM_ALIASES = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "neutal": "neutral",   # known typo
    "sadness": "sadness",
    "sad": "sadness",
    "surprise": "surprise",
}


def parse_stem(filename: str):
    """Return (ground_truth_label, is_suffix_3) from clip filename, or (None, False) if unknown."""
    stem = Path(filename).stem
    # Strip trailing " (N)" suffix
    suffix_match = re.search(r"\s*\((\d+)\)\s*$", stem)
    suffix_num = int(suffix_match.group(1)) if suffix_match else None
    clean = re.sub(r"\s*\(\d+\)\s*$", "", stem).strip().lower()

    label = STEM_ALIASES.get(clean)
    is_suffix_3 = suffix_num == 3
    return label, is_suffix_3


def evaluate_clip(service, clip_path: str, ground_truth: str, apply_smoothing: bool = True):
    """Run a single clip through the engine frame-by-frame in memory, return per-frame predictions."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {clip_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    client_id = f"eval_{Path(clip_path).stem}"
    service.reset_session(client_id)

    frame_results = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = service.process_frame(
            bgr_frame=frame,
            session_state_or_client_id=client_id,
            apply_smoothing=apply_smoothing,
            calibrate=False,
            return_trace=False,
        )
        pred = result.get("apparent_expression", "NO_FACE")
        if pred and pred != "NO_FACE":
            pred = pred.lower()
        frame_results.append({
            "frame_idx": idx,
            "pred": pred,
            "conf": result.get("calibrated_confidence", 0.0),
            "face": result.get("face_detected", False),
        })
        idx += 1

    cap.release()
    return frame_results, fps


def summarize(frame_results, ground_truth):
    """Compute summary stats for a single clip."""
    preds = [r["pred"] for r in frame_results if r["face"] and r["pred"] != "NO_FACE"]
    total = len(frame_results)
    if not preds:
        return {
            "total_frames": total,
            "face_frames": 0,
            "majority_vote": "NO_FACE",
            "frame_accuracy": 0.0,
            "top_wrong_class": "-",
            "top_wrong_share": 0.0,
            "class_counts": {},
        }

    counts = Counter(preds)
    majority = counts.most_common(1)[0][0]
    correct = sum(1 for p in preds if p == ground_truth)
    frame_acc = correct / len(preds) if preds else 0.0

    wrong_counts = {k: v for k, v in counts.items() if k != ground_truth}
    if wrong_counts:
        top_wrong = max(wrong_counts, key=wrong_counts.get)
        top_wrong_share = wrong_counts[top_wrong] / len(preds)
    else:
        top_wrong = "-"
        top_wrong_share = 0.0

    return {
        "total_frames": total,
        "face_frames": len(preds),
        "majority_vote": majority,
        "frame_accuracy": round(frame_acc, 4),
        "top_wrong_class": top_wrong,
        "top_wrong_share": round(top_wrong_share, 4),
        "class_counts": dict(counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate webcam emotion clips")
    parser.add_argument("--backbone", default="efficientnet_current",
                        choices=["efficientnet_current", "resnet18_raf"])
    parser.add_argument("--clip-dir", default="New folder")
    parser.add_argument("--final", action="store_true",
                        help="Include (3) suffix clips (final evaluation only)")
    parser.add_argument("--no-smoothing", action="store_true",
                        help="Disable EMA smoothing (default uses live engine smoothing)")
    args = parser.parse_args()

    from backend.services.webcam_service import webcam_service, WebcamService
    if args.backbone == "resnet18_raf":
        raf_ckpt = webcam_service._cfg.get(
            "webcam_resnet18_raf_path", "external_models/resnet18_raf/best_resnet18.pt"
        )
        if not os.path.exists(raf_ckpt):
            print(f"[SKIP] resnet18_raf checkpoint missing at '{raf_ckpt}'; skipping evaluation.")
            return
        service = WebcamService(backbone="resnet18_raf")
    elif webcam_service._backbone_name == args.backbone:
        service = webcam_service
    else:
        service = WebcamService(backbone=args.backbone)

    print(f"\n=== Evaluating backbone: {args.backbone} ({service._architecture}, ckpt={service.video_ckpt}) ===")

    clip_dir = Path(args.clip_dir)
    if not clip_dir.exists():
        print(f"[ERROR] Clip directory not found: {clip_dir}")
        sys.exit(1)

    clips = sorted(clip_dir.glob("*.mp4"))
    results_all = []

    for clip_path in clips:
        label, is_suffix_3 = parse_stem(clip_path.name)
        if label is None:
            print(f"  [WARNING] Skipping unknown clip label: {clip_path.name}")
            continue
        if is_suffix_3 and not args.final:
            print(f"  [WARNING] Skipping reserved (3) clip (pass --final to include): {clip_path.name}")
            continue

        out = evaluate_clip(service, str(clip_path), label, apply_smoothing=not args.no_smoothing)
        if out is None:
            continue
        frame_results, native_fps = out

        stats = summarize(frame_results, label)
        entry = {
            "clip": clip_path.name,
            "ground_truth": label,
            "native_fps": round(native_fps, 2),
            **stats,
            "frames": frame_results,
        }
        results_all.append(entry)

    # Print ONE compact table
    print(f"\n{'Clip':<20} {'Truth':<10} {'Majority':<10} {'FrameAcc':>9} {'TopWrong':<10} {'WrongPct':>9}")
    print("-" * 72)
    for r in results_all:
        print(f"{r['clip']:<20} {r['ground_truth']:<10} {r['majority_vote']:<10} "
              f"{r['frame_accuracy']:>8.1%} {r['top_wrong_class']:<10} {r['top_wrong_share']:>8.1%}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/eval_{args.backbone}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "backbone": args.backbone,
            "architecture": service._architecture,
            "checkpoint": service.video_ckpt,
            "apply_smoothing": not args.no_smoothing,
            "clips": results_all,
        }, f, indent=2)
    print(f"\nSaved full details to: {out_path}")


if __name__ == "__main__":
    main()
