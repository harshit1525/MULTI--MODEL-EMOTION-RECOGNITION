import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.webcam_service import webcam_service, EMOTIONS

LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutal": "neutral",
    "neutral": "neutral",
    "sadness": "sadness",
    "sad": "sadness",
    "surprise": "surprise",
    "surp": "surprise"
}

def get_gt(stem: str) -> str:
    cleaned = stem.lower()
    for s in [" (2)", " (3)", " (4)", " (5)"]:
        cleaned = cleaned.replace(s, "")
    cleaned = cleaned.strip()
    return LABEL_MAP.get(cleaned, "unknown")

def evaluate_pipeline(disable_surprise_boost=False, square_pad=False):
    clips_dir = Path("New folder")
    results = {}
    
    for clip_path in sorted(clips_dir.glob("*.mp4")):
        gt = get_gt(clip_path.stem)
        if gt == "unknown":
            continue
            
        client_id = f"test_{clip_path.stem}_{disable_surprise_boost}_{square_pad}"
        webcam_service.reset_session(client_id)
        
        cap = cv2.VideoCapture(str(clip_path))
        preds = []
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            # Simulate square padding if requested
            if square_pad:
                h, w = frame.shape[:2]
                webcam_service.face_detector.setInputSize((w, h))
                _, faces = webcam_service.face_detector.detect(frame)
                if faces is not None and len(faces) > 0:
                    face = faces[0]
                    fx, fy, fw, fh = [int(v) for v in face[:4]]
                    cx, cy = fx + fw // 2, fy + fh // 2
                    crop_sz = int(max(fw, fh) * 1.25)
                    pad_needed = crop_sz // 2
                    padded = cv2.copyMakeBorder(frame, pad_needed, pad_needed, pad_needed, pad_needed, cv2.BORDER_REFLECT)
                    pcx, pcy = cx + pad_needed, cy + pad_needed
                    face_crop = padded[pcy - crop_sz // 2 : pcy + (crop_sz - crop_sz // 2),
                                       pcx - crop_sz // 2 : pcx + (crop_sz - crop_sz // 2)]
                else:
                    face_crop = frame
            
            # Predict frame with custom flag
            res = webcam_service.predict_frame(frame, client_id=client_id)
            preds.append(res.get("apparent_expression", "unknown").lower())
            
        cap.release()
        counts = Counter(preds)
        maj = counts.most_common(1)[0][0] if counts else "none"
        correct = (maj == gt)
        acc = (counts.get(gt, 0) / len(preds) * 100) if preds else 0.0
        results[clip_path.name] = {
            "gt": gt,
            "maj": maj,
            "correct": correct,
            "acc": acc,
            "counts": dict(counts)
        }
    return results

if __name__ == "__main__":
    print("Evaluating current live path on all 9 clips...")
    res_base = evaluate_pipeline(disable_surprise_boost=False, square_pad=False)
    for k, v in res_base.items():
        print(f"  {k:20s} | GT: {v['gt']:8s} | Maj: {v['maj']:8s} | Correct: {str(v['correct']):5s} | Acc: {v['acc']:.1f}% | Counts: {v['counts']}")
