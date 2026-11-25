import os
import sys
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any

# Ensure utf-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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

def load_meld_val_crops(cache_dir="models/video_classifier/face_crops_cache", seed=101, split_ratio=0.90, max_per_class=1200):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []
    samples = []
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
        val_paths = paths[split_at:]
        for p in val_paths:
            samples.append((p, idx, "meld"))
    return samples

def load_rafdb_val_crops(rafdb_dir="DATASET/test", max_per_class=100):
    rafdb_path = Path(rafdb_dir)
    if not rafdb_path.exists():
        return []
    # RAF-DB mapping: 1:surprise, 2:fear, 3:disgust, 4:joy, 5:sadness, 6:anger, 7:neutral
    raf_to_meld = {
        "1": 6,  # surprise
        "2": 2,  # fear
        "3": 1,  # disgust
        "4": 3,  # joy
        "5": 5,  # sadness
        "6": 0,  # anger
        "7": 4,  # neutral
    }
    samples = []
    rng = random.Random(101)
    for folder_name, meld_idx in raf_to_meld.items():
        fdir = rafdb_path / folder_name
        if not fdir.exists():
            continue
        paths = sorted(list(fdir.glob("*.jpg")) + list(fdir.glob("*.png")))
        rng.shuffle(paths)
        if max_per_class > 0 and len(paths) > max_per_class:
            paths = paths[:max_per_class]
        for p in paths:
            samples.append((p, meld_idx, "rafdb"))
    return samples

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred) * 100.0)
    
    n_classes = len(LABEL_MAPPING)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    per_class_f1 = {}
    for cls, idx in LABEL_MAPPING.items():
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_f1[cls] = round(f1, 4)
        
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    return acc, macro_f1, per_class_f1

def precompute_logits(samples):
    device = webcam_service.device
    model = webcam_service.backbone
    img_size = getattr(webcam_service, "_img_size", 96)
    norm = webcam_service.norm
    
    model.eval()
    precomputed = []
    batch_size = 64
    
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        tensors = []
        valid_samples = []
        for p, target, src in batch_samples:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            resized = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = norm(t)
            tensors.append(t)
            valid_samples.append((target, src))
            
        if not tensors:
            continue
            
        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            batch_logits = model(batch_t).cpu().numpy()
            
        for (target, src), logits in zip(valid_samples, batch_logits):
            precomputed.append((target, src, logits))
            
    return precomputed

def evaluate_ablation_on_logits(precomputed, ablate_geom=False, ablate_neutral_boost=False, ablate_prior=False):
    y_true = []
    y_pred = []
    
    for target, src, raw_logits in precomputed:
        logits = raw_logits.copy()
        
        # Prior debiasing
        if not ablate_prior:
            prior_bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            logits = logits - prior_bias
            
        # Geometry adjustments
        if not ablate_geom:
            if not ablate_neutral_boost:
                # Resting neutral boost rule
                logits[4] += 1.2
                logits[1] -= 1.4
                logits[5] -= 1.0
                logits[6] -= 1.0
                
        # Temperature scaling (T*=0.85)
        log_p = np.log(np.clip(torch.softmax(torch.from_numpy(logits), dim=-1).numpy(), 1e-6, 1.0))
        centered = log_p - np.mean(log_p)
        scaled = centered / 0.85
        exp_s = np.exp(scaled - np.max(scaled))
        calib_p = exp_s / np.sum(exp_s)
        
        pred_idx = int(np.argmax(calib_p))
        y_true.append(target)
        y_pred.append(pred_idx)
        
    acc, macro_f1, per_class_f1 = compute_metrics(y_true, y_pred)
    return acc, macro_f1, per_class_f1

def main():
    meld_crops = load_meld_val_crops()
    rafdb_crops = load_rafdb_val_crops()
    all_val = meld_crops + rafdb_crops
    
    print("=" * 80, flush=True)
    print(f"PHASE 2: VALIDATION ABLATIONS ON HELD-OUT DATA ({len(all_val)} crops: {len(meld_crops)} MELD + {len(rafdb_crops)} RAF-DB)", flush=True)
    print("=" * 80, flush=True)
    print("Precomputing batched vision backbone logits...", flush=True)
    
    precomputed = precompute_logits(all_val)
    print(f"Precomputed {len(precomputed)} logits.", flush=True)
    
    ablations = [
        ("Baseline (All rules active)", False, False, False),
        ("Ablate (a): Geometry rules entirely", True, True, False),
        ("Ablate (b): Resting neutral boost only", False, True, False),
        ("Ablate (c): Prior debiasing", False, False, True),
        ("Combined: Pure Vision Model (No Geometry Hacks)", True, True, True),
    ]
    
    results = {}
    print("\n" + "=" * 80, flush=True)
    print("EVALUATION ON COMBINED VALIDATION CROPS (MELD + RAF-DB):", flush=True)
    print("=" * 80, flush=True)
    
    for name, ab_geom, ab_neut, ab_prior in ablations:
        acc, f1, pc_f1 = evaluate_ablation_on_logits(precomputed, ablate_geom=ab_geom, ablate_neutral_boost=ab_neut, ablate_prior=ab_prior)
        results[name] = {"acc": round(acc, 2), "macro_f1": round(f1, 4), "per_class_f1": pc_f1}
        print(f"Mode: {name:48s} | Acc: {acc:5.2f}% | Macro-F1: {f1:.4f}", flush=True)
        for cls in EMOTIONS:
            print(f"    {cls:10s}: F1={pc_f1.get(cls, 0.0):.4f}", flush=True)
            
    # MELD only
    meld_precomputed = [item for item in precomputed if item[1] == "meld"]
    print("\n" + "=" * 80, flush=True)
    print(f"EVALUATION ON MELD HELD-OUT CROPS ONLY ({len(meld_precomputed)} crops):", flush=True)
    print("=" * 80, flush=True)
    meld_results = {}
    for name, ab_geom, ab_neut, ab_prior in ablations:
        acc, f1, pc_f1 = evaluate_ablation_on_logits(meld_precomputed, ablate_geom=ab_geom, ablate_neutral_boost=ab_neut, ablate_prior=ab_prior)
        meld_results[name] = {"acc": round(acc, 2), "macro_f1": round(f1, 4), "per_class_f1": pc_f1}
        print(f"Mode: {name:48s} | Acc: {acc:5.2f}% | Macro-F1: {f1:.4f}", flush=True)
        for cls in EMOTIONS:
            print(f"    {cls:10s}: F1={pc_f1.get(cls, 0.0):.4f}", flush=True)

    # Save to JSON
    import json
    with open("results/phase2_ablations.json", "w", encoding="utf-8") as f:
        json.dump({"combined_val": results, "meld_val": meld_results}, f, indent=2)
    print("\nSaved ablation results to results/phase2_ablations.json", flush=True)

if __name__ == "__main__":
    main()
