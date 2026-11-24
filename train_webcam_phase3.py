"""
train_webcam_phase3.py
======================
Phase 3 Fine-Tuning for Real-Time Webcam Facial Expression Recognition.

Resumes from the best EfficientNet-B0 checkpoint (models/video_classifier/best_webcam_model.pth, 44.39% val_acc).
Expands class limits up to 1200 samples/class (~7,000 total face crops)
Uses class-weighted CrossEntropyLoss + CosineAnnealingLR for stable convergence.
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3_trainer")

LABEL_MAPPING = {
    'anger':    0,
    'disgust':  1,
    'fear':     2,
    'joy':      3,
    'neutral':  4,
    'sadness':  5,
    'surprise': 6,
}
CACHE_DIR = "models/video_classifier/face_crops_cache"
OUTPUT_MODEL = "models/video_classifier/best_webcam_model.pth"
IMG_SIZE = 96
MAX_PER_CLASS = 1200
BATCH_SIZE = 64
NUM_EPOCHS = 20
INITIAL_LR = 3.5e-5
WEIGHT_DECAY = 1e-4

class FaceCropDataset(Dataset):
    def __init__(self, cache_dir, transform=None, split_ratio=0.90, seed=101, is_train=True):
        self.transform = transform
        self.samples = []

        rng = random.Random(seed)
        for cls, idx in LABEL_MAPPING.items():
            cls_dir = Path(cache_dir) / cls
            if not cls_dir.exists():
                continue
            paths = sorted(cls_dir.glob("*.jpg"))
            rng.shuffle(paths)
            if MAX_PER_CLASS > 0 and len(paths) > MAX_PER_CLASS:
                paths = paths[:MAX_PER_CLASS]
            split_at = int(len(paths) * split_ratio)
            selected = paths[:split_at] if is_train else paths[split_at:]
            for p in selected:
                self.samples.append((p, idx))

        rng.shuffle(self.samples)
        counts = Counter(s[1] for s in self.samples)
        split_name = "train" if is_train else "val"
        logger.info(
            f"Dataset ({split_name}): {len(self.samples)} samples | "
            + " ".join(f"{k}:{counts[v]}" for k, v in LABEL_MAPPING.items())
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            img_rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        if self.transform:
            return self.transform(pil_img), label
        else:
            t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            return t, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Transforms with facial expression preservation & webcam robustness
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 12, IMG_SIZE + 12)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.06),
        transforms.RandomRotation(degrees=12),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FaceCropDataset(CACHE_DIR, transform=train_tf, split_ratio=0.90, is_train=True)
    val_ds = FaceCropDataset(CACHE_DIR, transform=val_tf, split_ratio=0.90, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weighting for balanced gradient backprop
    counts = Counter(s[1] for s in train_ds.samples)
    total_samples = len(train_ds)
    num_classes = len(LABEL_MAPPING)
    weights = [total_samples / (num_classes * max(counts.get(i, 1), 1)) for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes
    logger.info(f"Class weights: {[round(w.item(), 3) for w in class_weights]}")

    # 2. Load model from existing checkpoint
    logger.info(f"Loading checkpoint from {OUTPUT_MODEL}...")
    ckpt = torch.load(OUTPUT_MODEL, map_location=device, weights_only=False)
    in_features = 1280
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    best_val_acc = float(ckpt.get("val_acc", 0.0))
    start_epoch = int(ckpt.get("epoch", 0))
    logger.info(f"Resumed from epoch {start_epoch} | Current best val_acc = {best_val_acc:.2f}%")

    # 3. Optimizer & Scheduler
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    logger.info(f"Starting Phase 3 training for {NUM_EPOCHS} epochs (LR={INITIAL_LR} -> 1e-6)...")

    for ep in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += len(y)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_acc = train_correct / max(train_total, 1) * 100.0
        avg_loss = train_loss / max(train_total, 1)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)

        val_acc = val_correct / max(val_total, 1) * 100.0
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {ep:02d}/{NUM_EPOCHS} | "
            f"TrainLoss={avg_loss:.4f} | TrainAcc={train_acc:.1f}% | "
            f"ValAcc={val_acc:.2f}% | LR={current_lr:.2e} | {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = start_epoch + ep
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_mapping":    LABEL_MAPPING,
                "input_dim":        in_features,
                "architecture":     "efficientnet_b0",
                "img_size":         IMG_SIZE,
                "val_acc":          best_val_acc,
                "epoch":            best_epoch,
            }, OUTPUT_MODEL)
            logger.info(f"  ** New best val_acc: {best_val_acc:.2f}% at epoch {best_epoch} -> saved to {OUTPUT_MODEL}")

    logger.info(f"\nPhase 3 training complete! Overall best val_acc = {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
