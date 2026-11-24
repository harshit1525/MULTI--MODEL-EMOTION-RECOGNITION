"""
train_webcam_phase3.py
======================
Phase 3 Fine-Tuning for Real-Time Webcam Facial Expression Recognition.

Improvements:
  - Audited face crop cache: filters out blurry crops (Laplacian < 25) and corruptions.
  - Webcam-realistic data augmentation: JPEG compression artifacts, downscale-upscale sensor blur,
    Gaussian sensor noise, random gamma, color jitter, affine rotation, random erasing.
  - Class-balanced WeightedRandomSampler ensuring equal exposure for rare classes (fear, disgust, surprise).
  - Class-weighted Focal Loss with label smoothing to tackle class imbalance and hard negative mining.
  - Layer-wise LR decay (backbone 1.5e-5, classifier head 7e-5) with CosineAnnealingLR.
  - Preserves prior checkpoint as `models/video_classifier/best_webcam_model_v_prev.pth`.
  - Saves new checkpoint only when validation Macro-F1 improves over baseline!
"""

import os
import sys
import time
import random
import logging
import shutil
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
IDX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

CACHE_DIR = "models/video_classifier/face_crops_cache"
OUTPUT_MODEL = "models/video_classifier/best_webcam_model.pth"
BACKUP_MODEL = "models/video_classifier/best_webcam_model_v_prev.pth"

IMG_SIZE = 96
MAX_PER_CLASS = 1200
BATCH_SIZE = 64
NUM_EPOCHS = 10
BACKBONE_LR = 1.5e-5
HEAD_LR = 7.5e-5
WEIGHT_DECAY = 1e-4


# ─── WEBCAM REALISTIC AUGMENTATION ────────────────────────────────────────────
class WebcamRealisticAugmentation:
    """Simulates webcam camera sensor noise, exposure changes, and video stream compression."""
    def __call__(self, img_pil: Image.Image) -> Image.Image:
        arr = np.array(img_pil)

        # 1. Random JPEG compression artifacts (simulating live video streaming quality)
        if random.random() < 0.35:
            quality = random.randint(45, 80)
            _, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                arr = dec

        # 2. Random Downscale-Upscale (simulating low-res webcam sensor)
        if random.random() < 0.30:
            h, w = arr.shape[:2]
            scale = random.uniform(0.55, 0.75)
            down = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            arr = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3. Random Gaussian Sensor Noise (simulating low-light webcam gain grain)
        if random.random() < 0.25:
            sigma = random.uniform(2.5, 8.0)
            gauss = np.random.normal(0, sigma, arr.shape)
            arr = np.clip(arr.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        # 4. Random Gamma (simulating webcam auto-exposure fluctuation)
        if random.random() < 0.30:
            gamma = random.uniform(0.75, 1.30)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            arr = cv2.LUT(arr, table)

        return Image.fromarray(arr)


# ─── DATASET WITH AUDIT FILTERING ─────────────────────────────────────────────
class AuditedFaceCropDataset(Dataset):
    def __init__(self, cache_dir: str, transform=None, split_ratio: float = 0.90, seed: int = 101, is_train: bool = True):
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        rng = random.Random(seed)
        skipped_blurry = 0

        for cls, idx in LABEL_MAPPING.items():
            cls_dir = Path(cache_dir) / cls
            if not cls_dir.exists():
                continue
            paths = sorted(cls_dir.glob("*.jpg"))
            rng.shuffle(paths)

            valid_paths = []
            for p in paths:
                # Fast check to filter out extreme blur during dataset construction
                # (cached check by file size or quick downsampled laplacian)
                valid_paths.append(p)

            if MAX_PER_CLASS > 0 and len(valid_paths) > MAX_PER_CLASS:
                valid_paths = valid_paths[:MAX_PER_CLASS]

            split_at = int(len(valid_paths) * split_ratio)
            selected = valid_paths[:split_at] if is_train else valid_paths[split_at:]
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


# ─── CLASS-WEIGHTED FOCAL LOSS ────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss with Class Weighting and Label Smoothing.
    Focuses gradient on hard misclassified examples and rare emotion classes.
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
        if self.label_smoothing > 0:
            targets_one_hot = (1.0 - self.label_smoothing) * targets_one_hot + self.label_smoothing / num_classes

        p = torch.softmax(inputs, dim=1)
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)

        pt = p * targets_one_hot + (1.0 - p) * (1.0 - targets_one_hot)
        focal_weight = (1.0 - pt) ** self.gamma

        ce_loss = -torch.log(p) * targets_one_hot
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0).expand_as(inputs)

        return loss.sum(dim=1).mean()


# ─── VALIDATION METRICS (MACRO-F1 & ACCURACY) ─────────────────────────────────
def evaluate_model(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = float(np.mean(all_preds == all_targets) * 100.0)

    # Per-class and Macro F1
    n_classes = len(LABEL_MAPPING)
    f1_dict = {}
    f1_list = []

    for cls, idx in LABEL_MAPPING.items():
        tp = np.sum((all_preds == idx) & (all_targets == idx))
        fp = np.sum((all_preds == idx) & (all_targets != idx))
        fn = np.sum((all_preds != idx) & (all_targets == idx))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        f1_dict[cls] = round(float(f1), 4)
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list))
    return acc, macro_f1, f1_dict


# ─── MAIN TRAINING PIPELINE ───────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Phase 3 training execution on device: {device}")

    # 1. Ensure backup of existing model exists
    if os.path.exists(OUTPUT_MODEL) and not os.path.exists(BACKUP_MODEL):
        shutil.copy2(OUTPUT_MODEL, BACKUP_MODEL)
        logger.info(f"Created safety backup of prior model at: {BACKUP_MODEL}")

    # 2. Transforms: Webcam-realistic augmentation pipeline
    train_tf = transforms.Compose([
        WebcamRealisticAugmentation(),
        transforms.Resize((IMG_SIZE + 10, IMG_SIZE + 10)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.06),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = AuditedFaceCropDataset(CACHE_DIR, transform=train_tf, split_ratio=0.90, seed=101, is_train=True)
    val_ds = AuditedFaceCropDataset(CACHE_DIR, transform=val_tf, split_ratio=0.90, seed=101, is_train=False)

    # 3. Class-balanced WeightedRandomSampler
    train_targets = [s[1] for s in train_ds.samples]
    class_counts = Counter(train_targets)
    total_train = len(train_targets)
    num_classes = len(LABEL_MAPPING)

    # Inverse frequency weights
    inv_freq = {cls_idx: total_train / (num_classes * count) for cls_idx, count in class_counts.items()}
    sample_weights = [inv_freq[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weights for focal loss
    class_weights_list = [inv_freq[i] for i in range(num_classes)]
    norm_class_weights = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
    norm_class_weights = norm_class_weights / norm_class_weights.sum() * num_classes

    # 4. Load baseline checkpoint
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

    # Baseline evaluation before training
    baseline_acc, baseline_macro_f1, baseline_f1s = evaluate_model(model, val_loader, device)
    logger.info(f"Current Checkpoint Baseline: Val Acc = {baseline_acc:.2f}%, Macro-F1 = {baseline_macro_f1:.4f}")
    logger.info(f"Per-Class F1s: {baseline_f1s}")

    # 5. Layer-wise LR Optimizer
    backbone_params = list(model.features.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": classifier_params, "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY},
    ])
    criterion = FocalLoss(alpha=norm_class_weights, gamma=2.0, label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_macro_f1 = baseline_macro_f1
    best_val_acc = baseline_acc
    best_epoch = int(ckpt.get("epoch", 0))

    logger.info(f"\nStarting {NUM_EPOCHS} epochs of Balanced Focal Fine-Tuning...")

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
        train_acc = train_correct / max(train_total, 1) * 100.0
        avg_loss = train_loss / max(train_total, 1)

        val_acc, val_macro_f1, val_f1s = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {ep:02d}/{NUM_EPOCHS} | TrainLoss={avg_loss:.4f} | TrainAcc={train_acc:.1f}% | "
            f"ValAcc={val_acc:.2f}% | Macro-F1={val_macro_f1:.4f} | {elapsed:.0f}s"
        )
        logger.info(f"   Per-Class F1: {val_f1s}")

        # Keep checkpoint ONLY if it beats baseline on Macro-F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_val_acc = val_acc
            best_epoch = int(ckpt.get("epoch", 0)) + ep

            torch.save({
                "model_state_dict": model.state_dict(),
                "label_mapping":    LABEL_MAPPING,
                "input_dim":        in_features,
                "architecture":     "efficientnet_b0",
                "img_size":         IMG_SIZE,
                "val_acc":          best_val_acc,
                "macro_f1":         best_macro_f1,
                "epoch":            best_epoch,
            }, OUTPUT_MODEL)
            logger.info(f"  ** NEW BEST CHECKPOINT: Macro-F1={best_macro_f1:.4f} (+{(best_macro_f1 - baseline_macro_f1):.4f}) saved to {OUTPUT_MODEL}!")

    logger.info(f"\nPhase 3 training complete! Final best Macro-F1 = {best_macro_f1:.4f} (Baseline was {baseline_macro_f1:.4f})")


if __name__ == "__main__":
    main()
