import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import torchvision.models as models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import (
    AudioClassifier,
    StrongerAudioClassifier,
    VideoClassifier,
    LearnedAttentionFusionNet,
    AdvancedCrossModalAttentionFusionNet,
    TextPreprocessingUtils,
    EMOTIONS
)
from backend.services import audio_service, video_service, text_service
from backend.utils.metrics import compute_calibration_metrics, plot_confusion_matrix, plot_calibration_curves

EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}

# ---------------------------------------------------------------------------
# 1. FEATURE EXTRACTION HELPERS (16-D PER MODALITY)
# ---------------------------------------------------------------------------
def compute_shannon_certainty(prob_vec: np.ndarray) -> float:
    p = np.clip(prob_vec, 1e-6, 1.0)
    p = p / p.sum()
    ent = float(-np.sum(p * np.log(p)) / np.log(7.0))
    return max(0.02, 1.0 - ent)

def to_centered_logit(prob_vec: np.ndarray) -> np.ndarray:
    log_p = np.log(np.clip(prob_vec, 1e-6, 1.0))
    return log_p - np.mean(log_p)

def build_modality_16d_feature(prob_dict: dict, quality_discount: float = 1.0) -> np.ndarray:
    p_vec = np.array([prob_dict.get(emo, 1e-4) for emo in EMOTIONS], dtype=np.float32)
    p_vec = p_vec / (p_vec.sum() + 1e-8)
    logits = to_centered_logit(p_vec)
    cert = compute_shannon_certainty(p_vec)
    qual = float(quality_discount)
    # 16-D vector: [probs (7), logits (7), cert (1), qual (1)]
    return np.concatenate([p_vec, logits, [cert], [qual]]).astype(np.float32)

# ---------------------------------------------------------------------------
# 2. DATASET BUILDING & CACHING
# ---------------------------------------------------------------------------
def extract_or_load_dataset(cache_path: str, records: list, name: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print(f"[CACHE] Loading cached {name} dataset from: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return data["X_a"], data["X_t"], data["X_v"], data["y"]

    print(f"\n[EXTRACT] Extracting 16-D multimodal features for {name} ({len(records)} samples)...")
    t0 = time.time()
    X_a_list, X_t_list, X_v_list, y_list = [], [], [], []

    for idx, rec in enumerate(records):
        emo = rec["emotion"]
        y_val = EMOTION_TO_IDX[emo]

        # 1. Text feature
        try:
            t_res = text_service.predict_from_text(rec["utterance"])
            t_feat = build_modality_16d_feature(
                t_res["probabilities"],
                t_res.get("details", {}).get("quality_metrics", {}).get("quality_discount", 1.0)
            )
        except Exception:
            t_feat = build_modality_16d_feature({e: 1/7.0 for e in EMOTIONS}, 0.5)

        # 2. Audio feature
        try:
            a_emb = rec["audio_emb"].unsqueeze(0).to(audio_service.device)
            a_logits = audio_service.classifier(a_emb)[0]
            a_probs = torch.softmax(a_logits, dim=-1).cpu().detach().numpy()
            a_dict = {EMOTIONS[i]: float(a_probs[i]) for i in range(7)}
            a_feat = build_modality_16d_feature(a_dict, 1.0)
        except Exception:
            a_feat = build_modality_16d_feature({e: 1/7.0 for e in EMOTIONS}, 0.5)

        # 3. Video feature
        try:
            v_res = video_service.predict_from_video(rec["video_path"], num_frames=4, generate_preview=False)
            v_feat = build_modality_16d_feature(
                v_res["probabilities"],
                v_res.get("details", {}).get("quality_metrics", {}).get("quality_discount", 1.0)
            )
        except Exception:
            v_feat = build_modality_16d_feature({e: 1/7.0 for e in EMOTIONS}, 0.5)

        X_a_list.append(a_feat)
        X_t_list.append(t_feat)
        X_v_list.append(v_feat)
        y_list.append(y_val)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - t0
            print(f"  [{name}] {idx + 1}/{len(records)} samples extracted ({elapsed:.1f}s, {elapsed/(idx+1):.2f}s/sample)")

    X_a = torch.tensor(np.array(X_a_list), dtype=torch.float32)
    X_t = torch.tensor(np.array(X_t_list), dtype=torch.float32)
    X_v = torch.tensor(np.array(X_v_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)

    torch.save({"X_a": X_a, "X_t": X_t, "X_v": X_v, "y": y}, cache_path)
    print(f"[CACHE] Saved {len(y)} samples to {cache_path}")
    return X_a, X_t, X_v, y

# ---------------------------------------------------------------------------
# 3. SAMPLE COLLECTION
# ---------------------------------------------------------------------------
def collect_samples(csv_path: str, audio_path: str, video_dir: str, n_per_emo: int):
    df = pd.read_csv(csv_path)
    audio_data = torch.load(audio_path, map_location="cpu", weights_only=False)
    audio_lookup = {d["sample_id"]: d["audio_embedding"] for d in audio_data}

    records = []
    for emo in EMOTIONS:
        sub = df[df["Emotion"].str.lower() == emo]
        collected = 0
        for _, row in sub.iterrows():
            d_id = row["Dialogue_ID"]
            u_id = row["Utterance_ID"]
            sid = f"dia{d_id}_utt{u_id}"
            vpath = os.path.join(video_dir, f"{sid}.mp4")
            utt = str(row["Utterance"])
            if sid in audio_lookup and os.path.exists(vpath) and len(utt.strip()) > 0:
                records.append({
                    "sample_id": sid,
                    "emotion": emo,
                    "utterance": utt,
                    "audio_emb": audio_lookup[sid],
                    "video_path": vpath
                })
                collected += 1
                if collected >= n_per_emo:
                    break
    return records

# ---------------------------------------------------------------------------
# 4. TRAINING & CALIBRATION PIPELINE
# ---------------------------------------------------------------------------
def train_and_optimize():
    print("=" * 80)
    print(" ADVANCED MULTIMODAL FUSION OPTIMIZATION & CONFIDENCE CALIBRATION PIPELINE")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device}")

    # Step 1: Collect balanced records
    print("\n[STEP 1] Assembling balanced multimodal cohorts from MELD partitions...")
    train_records = collect_samples(
        "MELD.Raw/train/train_sent_emo.csv",
        "audio_embeddings/train_embeddings.pt",
        "MELD.Raw/train/train_splits",
        n_per_emo=30 # 30 * 7 = 210 train samples
    )
    val_records = collect_samples(
        "MELD.Raw/dev_sent_emo.csv",
        "audio_embeddings/dev_embeddings.pt",
        "MELD.Raw/dev/dev_splits_complete",
        n_per_emo=15 # 15 * 7 = 105 val samples
    )
    test_records = collect_samples(
        "MELD.Raw/test_sent_emo.csv",
        "audio_embeddings/test_embeddings.pt",
        "MELD.Raw/test/output_repeated_splits_test",
        n_per_emo=15 # 15 * 7 = 105 test samples
    )

    print(f"  Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    # Step 2: Extract or load cached feature tensors
    os.makedirs("data", exist_ok=True)
    X_a_train, X_t_train, X_v_train, y_train = extract_or_load_dataset(
        "data/fusion_train_dataset.pt", train_records, "TRAIN"
    )
    X_a_val, X_t_val, X_v_val, y_val = extract_or_load_dataset(
        "data/fusion_val_dataset.pt", val_records, "VAL"
    )
    X_a_test, X_t_test, X_v_test, y_test = extract_or_load_dataset(
        "data/fusion_test_dataset.pt", test_records, "TEST"
    )

    # Step 3: Train AdvancedCrossModalAttentionFusionNet
    print("\n[STEP 2] Compiling and training AdvancedCrossModalAttentionFusionNet...")
    torch.manual_seed(42)
    np.random.seed(42)

    model = AdvancedCrossModalAttentionFusionNet(
        in_features=16,
        embed_dim=96,
        num_heads=4,
        num_layers=2,
        num_classes=7,
        dropout=0.2
    ).to(device)

    # Label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45, eta_min=1e-5)

    batch_size = 32
    n_train = len(y_train)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None

    epochs = 45
    t_train_start = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start:b_start+batch_size]
            b_a = X_a_train[b_idx].to(device)
            b_t = X_t_train[b_idx].to(device)
            b_v = X_v_train[b_idx].to(device)
            b_y = y_train[b_idx].to(device)

            # Modality Dropout: 15% chance to mask out audio, text, or video for robustness
            mask = torch.ones((len(b_idx), 3), dtype=torch.bool, device=device)
            if np.random.rand() < 0.15:
                drop_mod = np.random.randint(0, 3)
                mask[:, drop_mod] = False

            optimizer.zero_grad()
            logits, gates = model(b_a, b_t, b_v, mask=mask)
            loss = criterion(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            v_a = X_a_val.to(device)
            v_t = X_t_val.to(device)
            v_v = X_v_val.to(device)
            v_y = y_val.to(device)

            v_logits, _ = model(v_a, v_t, v_v)
            val_loss = criterion(v_logits, v_y).item()
            val_preds = torch.argmax(v_logits, dim=-1).cpu().numpy()
            val_acc = float(accuracy_score(y_val.numpy(), val_preds))
            val_mf1 = float(f1_score(y_val.numpy(), val_preds, average="macro", zero_division=0))

        if val_loss < best_val_loss or (val_loss == best_val_loss and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss/n_batches:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | Val Macro F1: {val_mf1:.2%}")

    print(f"[TRAINED] Best Validation Loss: {best_val_loss:.4f} | Best Val Accuracy: {best_val_acc:.2%} ({time.time() - t_train_start:.1f}s)")

    # Load best checkpoint
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    # Step 4: Advanced Confidence Optimization & Calibration (Phase 4)
    print("\n[STEP 3] Performing Advanced Post-Hoc Confidence Calibration...")
    with torch.no_grad():
        val_logits_raw, _ = model(X_a_val.to(device), X_t_val.to(device), X_v_val.to(device))
        val_logits_np = val_logits_raw.cpu().numpy()
        y_val_np = y_val.numpy()

    # A. Uncalibrated (T = 1.0)
    p_uncal = np.exp(val_logits_np - np.max(val_logits_np, axis=1, keepdims=True))
    p_uncal = p_uncal / np.sum(p_uncal, axis=1, keepdims=True)
    uncal_cal_metrics = compute_calibration_metrics(p_uncal, y_val_np)
    print(f"  Uncalibrated Softmax (T=1.00) -> ECE: {uncal_cal_metrics['ece']:.4f} | Brier: {uncal_cal_metrics['brier_score']:.4f} | NLL: {uncal_cal_metrics['nll']:.4f}")

    # B. Temperature Scaling (NLL optimization)
    def nll_obj(T):
        scaled = val_logits_np / T
        exps = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        eps = 1e-12
        return -np.mean(np.log(p[np.arange(len(y_val_np)), y_val_np] + eps))

    opt_res = minimize_scalar(nll_obj, bounds=(0.5, 3.0), method="bounded")
    optimal_T = float(opt_res.x)

    p_temp_cal = np.exp(val_logits_np / optimal_T - np.max(val_logits_np / optimal_T, axis=1, keepdims=True))
    p_temp_cal = p_temp_cal / np.sum(p_temp_cal, axis=1, keepdims=True)
    temp_cal_metrics = compute_calibration_metrics(p_temp_cal, y_val_np)
    print(f"  Optimal Temperature Scaling (T*={optimal_T:.4f}) -> ECE: {temp_cal_metrics['ece']:.4f} | Brier: {temp_cal_metrics['brier_score']:.4f} | NLL: {temp_cal_metrics['nll']:.4f}")

    # C. Vector Scaling (Class-specific diagonal weights)
    def vector_scaling_obj(params):
        W = params[:7]
        b = params[7:]
        scaled = val_logits_np * W + b
        exps = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exps / np.sum(exps, axis=1, keepdims=True)
        eps = 1e-12
        return -np.mean(np.log(p[np.arange(len(y_val_np)), y_val_np] + eps))

    init_params = np.concatenate([np.ones(7), np.zeros(7)])
    res_vec = minimize(vector_scaling_obj, init_params, method="L-BFGS-B")
    W_vec = res_vec.x[:7]
    b_vec = res_vec.x[7:]
    scaled_vec = val_logits_np * W_vec + b_vec
    p_vec_cal = np.exp(scaled_vec - np.max(scaled_vec, axis=1, keepdims=True))
    p_vec_cal = p_vec_cal / np.sum(p_vec_cal, axis=1, keepdims=True)
    vec_cal_metrics = compute_calibration_metrics(p_vec_cal, y_val_np)
    print(f"  Vector Scaling (7 classes) -> ECE: {vec_cal_metrics['ece']:.4f} | Brier: {vec_cal_metrics['brier_score']:.4f} | NLL: {vec_cal_metrics['nll']:.4f}")

    # Choose best calibration technique
    chosen_technique = "Temperature Scaling"
    chosen_T = optimal_T
    calibration_summary = {
        "chosen_method": chosen_technique,
        "temperature": chosen_T,
        "uncalibrated_ece": uncal_cal_metrics['ece'],
        "calibrated_ece": temp_cal_metrics['ece'],
        "uncalibrated_brier": uncal_cal_metrics['brier_score'],
        "calibrated_brier": temp_cal_metrics['brier_score'],
        "uncalibrated_nll": uncal_cal_metrics['nll'],
        "calibrated_nll": temp_cal_metrics['nll']
    }

    # Step 5: Held-Out Test Evaluation (Phase 7 Controlled Comparison)
    print("\n[STEP 4] Evaluating Optimized Architecture on Held-Out Test Partition...")
    t_test_0 = time.time()
    with torch.no_grad():
        test_logits_raw, test_gates = model(X_a_test.to(device), X_t_test.to(device), X_v_test.to(device))
        test_logits_np = test_logits_raw.cpu().numpy()
        test_gates_np = test_gates.cpu().numpy()
        y_test_np = y_test.numpy()

    test_latency_ms = ((time.time() - t_test_0) / len(y_test_np)) * 1000

    # Calibrate test logits
    test_logits_cal = test_logits_np / chosen_T
    p_test = np.exp(test_logits_cal - np.max(test_logits_cal, axis=1, keepdims=True))
    p_test = p_test / np.sum(p_test, axis=1, keepdims=True)

    test_preds = np.argmax(p_test, axis=1)
    test_confs = np.max(p_test, axis=1)

    test_acc = float(accuracy_score(y_test_np, test_preds))
    test_mf1 = float(f1_score(y_test_np, test_preds, average="macro", zero_division=0))
    test_wf1 = float(f1_score(y_test_np, test_preds, average="weighted", zero_division=0))
    test_mean_conf = float(np.mean(test_confs))
    test_cal_metrics = compute_calibration_metrics(p_test, y_test_np)

    mean_gates = test_gates_np.mean(axis=0)
    print("\n" + "=" * 80)
    print("            OPTIMIZED MULTIMODAL SYSTEM TEST RESULTS            ")
    print("=" * 80)
    print(f" Test Accuracy          : {test_acc:.2%}")
    print(f" Macro F1-Score         : {test_mf1:.2%}")
    print(f" Weighted F1-Score      : {test_wf1:.2%}")
    print(f" Mean Confident Score   : {test_mean_conf:.1%}")
    print(f" Calibrated ECE         : {test_cal_metrics['ece']:.4f}")
    print(f" Brier Score            : {test_cal_metrics['brier_score']:.4f}")
    print(f" Negative Log-Likelihood: {test_cal_metrics['nll']:.4f}")
    print(f" Mean Attention Gates   : Audio: {mean_gates[0]:.1%} | Text: {mean_gates[1]:.1%} | Video: {mean_gates[2]:.1%}")
    print(f" Mean Inference Latency : {test_latency_ms:.2f} ms")
    print("=" * 80)

    # Step 6: Save Checkpoint (Phase 8: Non-destructive preservation)
    os.makedirs("models/fusion_classifier", exist_ok=True)
    optimized_ckpt_path = "models/fusion_classifier/optimized_fusion_model.pth"
    torch.save({
        "model_state_dict": best_model_state,
        "embed_dim": 96,
        "num_heads": 4,
        "num_layers": 2,
        "num_classes": 7,
        "temperature": chosen_T,
        "architecture": "AdvancedCrossModalAttentionFusionNet",
        "calibration": calibration_summary,
        "test_metrics": {
            "accuracy": test_acc,
            "macro_f1": test_mf1,
            "weighted_f1": test_wf1,
            "mean_confidence": test_mean_conf,
            "ece": test_cal_metrics['ece'],
            "brier_score": test_cal_metrics['brier_score'],
            "nll": test_cal_metrics['nll'],
            "latency_ms": test_latency_ms
        }
    }, optimized_ckpt_path)
    print(f"\n[SAVED] Optimized model checkpoint written to: {optimized_ckpt_path}")

    # Generate plots
    os.makedirs("results/plots", exist_ok=True)
    cm_path = "results/plots/optimized_fusion_confusion_matrix.png"
    plot_confusion_matrix(y_test_np, test_preds, EMOTIONS, cm_path, title="Optimized Cross-Modal Attention Fusion (Held-Out Test)")
    print(f"[PLOT] Confusion matrix saved to: {cm_path}")

    cal_curve_path = "results/plots/optimized_calibration_curves.png"
    plot_calibration_curves(
        np.exp(test_logits_np - np.max(test_logits_np, axis=1, keepdims=True)) / np.sum(np.exp(test_logits_np - np.max(test_logits_np, axis=1, keepdims=True)), axis=1, keepdims=True),
        p_test,
        y_test_np,
        cal_curve_path
    )
    print(f"[PLOT] Calibration curves saved to: {cal_curve_path}")

    return {
        "accuracy": test_acc,
        "macro_f1": test_mf1,
        "weighted_f1": test_wf1,
        "mean_confidence": test_mean_conf,
        "ece": test_cal_metrics['ece'],
        "brier_score": test_cal_metrics['brier_score'],
        "nll": test_cal_metrics['nll'],
        "latency_ms": test_latency_ms,
        "temperature": chosen_T,
        "gates": {"audio": float(mean_gates[0]), "text": float(mean_gates[1]), "video": float(mean_gates[2])}
    }

if __name__ == "__main__":
    train_and_optimize()
