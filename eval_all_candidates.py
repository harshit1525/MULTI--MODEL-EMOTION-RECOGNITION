import os
import sys
import time
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backend.services.webcam_service import EMOTIONS

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

def load_meld_val(cache_dir="models/video_classifier/face_crops_cache", seed=101, split_ratio=0.90, max_per_class=1200):
    cache_path = Path(cache_dir)
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
            samples.append((p, idx))
    return samples

def load_rafdb_test(rafdb_dir="DATASET/test", max_per_class=200):
    rafdb_path = Path(rafdb_dir)
    if not rafdb_path.exists():
        return []
    # 1:surprise, 2:fear, 3:disgust, 4:joy, 5:sadness, 6:anger, 7:neutral
    raf_to_meld = {"1": 6, "2": 2, "3": 1, "4": 3, "5": 5, "6": 0, "7": 4}
    samples = []
    rng = random.Random(101)
    for f_name, meld_idx in raf_to_meld.items():
        fdir = rafdb_path / f_name
        if not fdir.exists():
            continue
        paths = sorted(list(fdir.glob("*.jpg")) + list(fdir.glob("*.png")))
        rng.shuffle(paths)
        if max_per_class > 0 and len(paths) > max_per_class:
            paths = paths[:max_per_class]
        for p in paths:
            samples.append((p, meld_idx))
    return samples

def load_candidate_model(ckpt_path):
    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", "efficientnet_b0")
    img_size = ckpt.get("img_size", 224 if arch == "timm_tf_efficientnet_b0" else 96)
    
    if arch == "timm_tf_efficientnet_b0":
        import timm
        model = timm.create_model("tf_efficientnet_b0", num_classes=7, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
    elif arch == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        import torch.nn as nn
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.35, inplace=True),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, 7),
        )
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Unknown architecture: {arch}")
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, img_size

def eval_dataset(model, img_size, samples, temp=0.85):
    from torchvision import transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    y_true = []
    y_pred = []
    latencies = []
    
    for p, target in samples:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        resized = cv2.resize(bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        t = norm(t).unsqueeze(0)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(t)[0].cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
        # Temperature scaling
        log_p = np.log(np.clip(torch.softmax(torch.from_numpy(logits), dim=-1).numpy(), 1e-6, 1.0))
        centered = log_p - np.mean(log_p)
        calib_logits = centered / temp
        calib_probs = np.exp(calib_logits - np.max(calib_logits))
        calib_probs = calib_probs / np.sum(calib_probs)
        
        y_true.append(target)
        y_pred.append(int(np.argmax(calib_probs)))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred) * 100.0)
    
    n_classes = len(LABEL_MAPPING)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t_val, p_val in zip(y_true, y_pred):
        cm[t_val, p_val] += 1
        
    f1s = []
    for idx in range(n_classes):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        
    macro_f1 = float(np.mean(f1s))
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    return acc, macro_f1, f1s, p95_lat

def test_neutral_bias(model, img_size):
    # Test on uniform grey frame (128, 128, 128)
    from torchvision import transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    grey = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    t = torch.from_numpy(grey).permute(2, 0, 1).float() / 255.0
    t = norm(t).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)[0].cpu().numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    top2_idx = np.argsort(probs)[::-1][:2]
    neutral_in_top2 = (4 in top2_idx)
    surprise_prob = float(probs[6])
    neutral_prob = float(probs[4])
    return neutral_in_top2, surprise_prob, neutral_prob, probs

def main():
    meld_val = load_meld_val()
    rafdb_test = load_rafdb_test()
    print("=" * 80)
    print("PHASE 3: CANDIDATE MODEL EVALUATION & SELECTION")
    print(f"Data: {len(meld_val)} MELD val crops | {len(rafdb_test)} RAF-DB test crops")
    print("=" * 80)
    
    candidates = [
        ("Candidate 0: best_webcam_model.pth (Current)", "models/video_classifier/best_webcam_model.pth"),
        ("Candidate 1a: best_webcam_model_rafdb_A.pth", "models/video_classifier/best_webcam_model_rafdb_A.pth"),
        ("Candidate 1b: best_webcam_model_rafdb_B.pth", "models/video_classifier/best_webcam_model_rafdb_B.pth"),
    ]
    
    cand0_meld_f1 = None
    results = {}
    
    for name, path in candidates:
        if not os.path.exists(path):
            print(f"[SKIP] {name}: file not found at {path}")
            continue
        print(f"\nEvaluating {name}...")
        try:
            model, img_size = load_candidate_model(path)
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue
            
        m_acc, m_f1, m_f1s, p95_lat = eval_dataset(model, img_size, meld_val)
        r_acc, r_f1, r_f1s, _ = eval_dataset(model, img_size, rafdb_test)
        neut_top2, surp_p, neut_p, raw_p = test_neutral_bias(model, img_size)
        
        if "Candidate 0" in name:
            cand0_meld_f1 = m_f1
            cand0_raf_f1 = r_f1
            
        # Check 4 selection rules
        rule_a = (r_f1 > cand0_raf_f1) if ("Candidate 0" not in name) else True
        rule_b = ((cand0_meld_f1 - m_f1) <= 0.02) if ("Candidate 0" not in name) else True
        rule_c = (neut_top2 and surp_p < 0.40)
        rule_d = (p95_lat <= 100.0)
        
        all_passed = rule_a and rule_b and rule_c and rule_d
        
        results[name] = {
            "meld_acc": round(m_acc, 2), "meld_f1": round(m_f1, 4),
            "rafdb_acc": round(r_acc, 2), "rafdb_f1": round(r_f1, 4),
            "p95_latency_ms": round(p95_lat, 1),
            "neutral_prob_grey": round(neut_p, 4),
            "surprise_prob_grey": round(surp_p, 4),
            "neutral_in_top2": neut_top2,
            "rule_a_raf_f1_higher": rule_a,
            "rule_b_meld_drop_le_002": rule_b,
            "rule_c_neutral_top2_surp_lt_40": rule_c,
            "rule_d_p95_le_100ms": rule_d,
            "ALL_RULES_PASSED": all_passed
        }
        
        print(f"  MELD Val Held-Out : Acc={m_acc:5.2f}% | Macro-F1={m_f1:.4f}")
        print(f"  RAF-DB Test Set   : Acc={r_acc:5.2f}% | Macro-F1={r_f1:.4f}")
        print(f"  CPU p95 Latency   : {p95_lat:.1f} ms")
        print(f"  Grey Frame Probs  : Neutral={neut_p:.1%}, Surprise={surp_p:.1%} (Neutral in top 2: {neut_top2})")
        print(f"  Rules: RuleA={rule_a}, RuleB={rule_b}, RuleC={rule_c}, RuleD={rule_d} -> ALL PASSED: {all_passed}")

    import json
    with open("results/candidate_model_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved candidate model evaluation to results/candidate_model_eval.json")

if __name__ == "__main__":
    main()
