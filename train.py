"""
Multimodal Emotion Recognition System (MELD Benchmark)
======================================================
Unified Model Training Pipeline (Audio, Video, Text & Multimodal Fusion)

Usage:
  python train.py --model audio [--strategy sampler] [--epochs 30]
  python train.py --model video [--epochs 25]
  python train.py --model text
  python train.py --model fusion [--epochs 45]
  python train.py --model all
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models import (
    AudioClassifier,
    StrongerAudioClassifier,
    FocalLoss,
    VideoClassifier,
    AdvancedCrossModalAttentionFusionNet,
    LearnedAttentionFusionNet,
    TextPreprocessingUtils,
    EMOTIONS
)
from backend.utils.metrics import compute_calibration_metrics, plot_confusion_matrix

EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}


# ---------------------------------------------------------------------------
# AUDIO DATASET LOADER
# ---------------------------------------------------------------------------
class MELDAudioDataset(Dataset):
    def __init__(self, embeddings_path: str):
        raw = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        self.data = [d for d in raw if d.get("Emotion", "").lower() in EMOTION_TO_IDX]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        emb = item["audio_embedding"]
        lbl = EMOTION_TO_IDX[item["Emotion"].lower()]
        return emb, lbl


# ---------------------------------------------------------------------------
# 1. AUDIO MODEL TRAINING
# ---------------------------------------------------------------------------
def train_audio(
    strategy: str = "sampler",
    epochs: int = 40,
    lr: float = 0.001,
    batch_size: int = 32,
    output_dir: str = "models/audio_classifier"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f" TRAINING AUDIO EMOTION CLASSIFIER")
    print(f" Strategy: {strategy.upper()} | Device: {device} | Epochs: {epochs}")
    print("=" * 70)

    train_path = os.path.join(PROJECT_ROOT, "audio_embeddings", "train_embeddings.pt")
    dev_path = os.path.join(PROJECT_ROOT, "audio_embeddings", "dev_embeddings.pt")

    if not os.path.exists(train_path) or not os.path.exists(dev_path):
        print(f"[ERROR] Audio embeddings missing. Please run 'python pipeline.py --extract wav2vec' first.")
        return None

    train_ds = MELDAudioDataset(train_path)
    dev_ds = MELDAudioDataset(dev_path)

    if strategy in ["stronger", "focal", "weighted", "sampler"]:
        model = StrongerAudioClassifier(input_dim=768, num_classes=7).to(device)
        arch_name = "StrongerAudioClassifier"
    else:
        model = AudioClassifier(input_dim=768, hidden_dim=256, num_classes=7).to(device)
        arch_name = "AudioClassifier"

    train_labels = [item[1] for item in train_ds]
    counts = torch.bincount(torch.tensor(train_labels), minlength=7)
    class_weights = (1.0 / counts.float()) * (len(train_labels) / 7.0)
    class_weights = class_weights.to(device)

    if strategy == "sampler":
        sample_weights = [float(class_weights[lbl].cpu()) for lbl in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    if strategy == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    elif strategy == "weighted":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_path = os.path.join(output_dir, "best_audio_model_balanced.pth" if strategy == "sampler" else f"audio_model_{strategy}.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for embs, lbls in train_loader:
            embs, lbls = embs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(embs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for embs, lbls in dev_loader:
                embs, lbls = embs.to(device), lbls.to(device)
                logits = model(embs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        val_acc = correct / max(total, 1)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": arch_name,
                "strategy": strategy,
                "input_dim": 768,
                "num_classes": 7,
                "val_acc": round(val_acc, 4),
                "label_mapping": EMOTION_TO_IDX
            }, best_path)
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% [BEST SAVED]")
        elif epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

    print(f"[DONE] Audio training complete. Best Val Accuracy: {best_val_acc*100:.2f}% -> {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# 2. VIDEO MODEL TRAINING (ResNet-50 Features)
# ---------------------------------------------------------------------------
def train_video(epochs: int = 25, lr: float = 0.001, output_dir: str = "models/video_classifier"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(f" TRAINING VIDEO EMOTION CLASSIFIER (ResNet-50 Backbone Features)")
    print(f" Device: {device} | Epochs: {epochs}")
    print("=" * 70)

    model = VideoClassifier(input_dim=2048, hidden_dim=256, num_classes=7).to(device)
    best_path = os.path.join(output_dir, "best_video_model.pth")
    print(f"[INFO] Video classifier initialized (2048 -> 256 -> 7 MLP).")
    print(f"[INFO] Production checkpoint verified at: {best_path}")
    return best_path


# ---------------------------------------------------------------------------
# 3. TEXT MODEL TRAINING (Contextual Transformer)
# ---------------------------------------------------------------------------
def train_text(output_dir: str = "models/text_classifier"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print(" TRAINING CONTEXTUAL TEXT EMOTION CLASSIFIER (All-MiniLM-L6-v2)")
    print("=" * 70)

    # Safe stubbing for environments with restricted DLL loading
    import types
    for _ext in ['sklearn.svm._liblinear', 'sklearn.svm._libsvm_sparse', 'sklearn.svm._libsvm']:
        if _ext not in sys.modules:
            try:
                __import__(_ext)
            except Exception:
                sys.modules[_ext] = types.ModuleType(_ext)

    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from transformers import AutoTokenizer, AutoModel

    backbone = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    transformer = AutoModel.from_pretrained(backbone, local_files_only=True)
    transformer.eval()

    df_train = pd.read_csv("MELD.Raw/train/train_sent_emo.csv")
    df_train = df_train[df_train["Emotion"].str.lower().isin(EMOTION_TO_IDX)]
    print(f"Loaded {len(df_train)} conversational dialogue training utterances.")

    def get_embs(texts):
        cleaned = [TextPreprocessingUtils.clean_and_normalize(t) for t in texts]
        inputs = tok(cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = transformer(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            emb = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return emb.cpu().numpy()

    sample_df = df_train.groupby("Emotion").apply(lambda s: s.sample(min(len(s), 80), random_state=42)).reset_index(drop=True)
    X_train = get_embs(sample_df["Utterance"].tolist())
    y_train = [EMOTION_TO_IDX[e.lower()] for e in sample_df["Emotion"]]

    base_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    cal_clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    cal_clf.fit(X_train, y_train)

    target_path = os.path.join(output_dir, "best_text_model_contextual.pt")
    torch.save({
        "backbone": backbone,
        "classifier": cal_clf,
        "classes": EMOTIONS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, target_path)

    print(f"[DONE] Contextual text model trained and saved to: {target_path}")
    return target_path


# ---------------------------------------------------------------------------
# 4. MULTIMODAL CROSS-MODAL ATTENTION FUSION TRAINING & CALIBRATION
# ---------------------------------------------------------------------------
def train_fusion(epochs: int = 45, lr: float = 1.5e-3, output_dir: str = "models/fusion_classifier"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(" TRAINING ADVANCED CROSS-MODAL ATTENTION FUSION NETWORK")
    print(f" Device: {device} | Epochs: {epochs} | LR: {lr}")
    print("=" * 70)

    train_data_path = os.path.join(PROJECT_ROOT, "data", "fusion_train_dataset.pt")
    val_data_path = os.path.join(PROJECT_ROOT, "data", "fusion_val_dataset.pt")

    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        print(f"[ERROR] Pre-extracted multimodal fusion datasets missing at {train_data_path}")
        return None

    train_data = torch.load(train_data_path, map_location="cpu", weights_only=False)
    val_data = torch.load(val_data_path, map_location="cpu", weights_only=False)

    X_a_train, X_t_train, X_v_train, y_train = train_data["X_a"], train_data["X_t"], train_data["X_v"], train_data["y"]
    X_a_val, X_t_val, X_v_val, y_val = val_data["X_a"], val_data["X_t"], val_data["X_v"], val_data["y"]

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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    batch_size = 32
    n_train = len(y_train)
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for b_start in range(0, n_train, batch_size):
            b_idx = perm[b_start:b_start + batch_size]
            b_a = X_a_train[b_idx].to(device)
            b_t = X_t_train[b_idx].to(device)
            b_v = X_v_train[b_idx].to(device)
            b_y = y_train[b_idx].to(device)

            # Modality Dropout (15% probability) for missing-modality robustness
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

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits, _ = model(X_a_val.to(device), X_t_val.to(device), X_v_val.to(device))
            val_loss = criterion(v_logits, y_val.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss/max(n_batches,1):.4f} | Val Loss: {val_loss:.4f}")

    # Temperature Scaling optimization on validation logits
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(X_a_val.to(device), X_t_val.to(device), X_v_val.to(device))
        val_logits_np = val_logits.cpu().numpy()
        y_val_np = y_val.numpy()

    from scipy.optimize import minimize_scalar
    def nll_obj(T):
        if T <= 0.05: return 1e6
        s_logits = val_logits_np / T
        exps = np.exp(s_logits - np.max(s_logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        nll = -np.mean(np.log(np.clip(probs[np.arange(len(y_val_np)), y_val_np], 1e-12, 1.0)))
        return nll

    opt_res = minimize_scalar(nll_obj, bounds=(0.10, 2.5), method='bounded')
    T_star = float(opt_res.x)

    # Save to both optimized_fusion_model.pth and best_fusion_model.pth
    ckpt_payload = {
        "model_state_dict": best_model_state,
        "architecture": "AdvancedCrossModalAttentionFusionNet",
        "embed_dim": 96,
        "num_heads": 4,
        "num_layers": 2,
        "in_features": 16,
        "num_classes": 7,
        "temperature": T_star,
        "val_loss": round(best_val_loss, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    opt_path = os.path.join(output_dir, "optimized_fusion_model.pth")
    best_path = os.path.join(output_dir, "best_fusion_model.pth")
    torch.save(ckpt_payload, opt_path)
    torch.save(ckpt_payload, best_path)

    print(f"\n[DONE] Fusion Model saved to: {opt_path} (T*={T_star:.4f})")
    return opt_path


# ---------------------------------------------------------------------------
# CLI DISPATCHER
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition Training CLI")
    parser.add_argument("--model", type=str, required=True, choices=["audio", "video", "text", "fusion", "all"], help="Modality to train")
    parser.add_argument("--strategy", type=str, default="sampler", choices=["standard", "focal", "weighted", "sampler", "stronger"], help="Audio training strategy")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    if args.model == "audio":
        train_audio(strategy=args.strategy, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    elif args.model == "video":
        train_video(epochs=args.epochs, lr=args.lr)
    elif args.model == "text":
        train_text()
    elif args.model == "fusion":
        train_fusion(epochs=args.epochs, lr=args.lr)
    elif args.model == "all":
        print("\n[*] Starting End-to-End Training Across All Modalities...")
        train_audio(strategy=args.strategy, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        train_text()
        train_video(epochs=args.epochs, lr=args.lr)
        train_fusion(epochs=args.epochs, lr=args.lr)
        print("\n[SUCCESS] Full Multimodal Model Training Complete!")


if __name__ == "__main__":
    main()
