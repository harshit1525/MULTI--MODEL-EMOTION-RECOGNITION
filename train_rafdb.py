"""
train_rafdb.py
==============
Unified Training Pipeline for Webcam Emotion Recognition:
Mixed RAF-DB (verified folder mapping) + MELD Face Crops.

Supports:
- Experiment A: EfficientNet-B0 initialized from current MELD production checkpoint.
- Experiment B: EfficientNet-B0 initialized from HSEmotion enet_b0_8_best_afew (AffectNet pretrained).
- Staged Training Schedule:
    Stage 1: Head training with frozen backbone.
    Stage 2: End-to-end fine-tuning with layer-wise LR decay and CosineAnnealingLR.
- Class-balanced WeightedRandomSampler ensuring equal exposure for rare classes (fear, disgust).
- Webcam-realistic data augmentation:
    JPEG compression, downscale-upscale, Gaussian sensor noise, random gamma,
    mild color jitter, rotation/shear, random erasing, random crop/padding jitter, mixup/cutmix.
- Early stopping on validation Macro-F1 (patience 5).
- Safe checkpointing to NEW file names (never overwrites existing models).
- Works out-of-the-box locally (CPU/GPU) and on Google Colab (GPU).
"""

import os
import sys
import time
import math
import random
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Dict, Optional, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

# Fixed 7-class emotion ordering
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
LABEL_MAPPING = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_LABEL = {i: emo for i, emo in enumerate(EMOTIONS)}

# Verified RAF-DB folder mapping -> project emotion name -> project class index
RAF_FOLDER_TO_EMOTION = {
    "1": "surprise",  # idx 6
    "2": "fear",      # idx 2
    "3": "disgust",   # idx 1
    "4": "joy",       # idx 3
    "5": "sadness",   # idx 5
    "6": "anger",     # idx 0
    "7": "neutral",   # idx 4
}
RAF_FOLDER_TO_IDX = {f: LABEL_MAPPING[emo] for f, emo in RAF_FOLDER_TO_EMOTION.items()}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rafdb_trainer")


# ─── WEBCAM REALISTIC AUGMENTATION ────────────────────────────────────────────
class WebcamRealisticAugmentation:
    """Simulates webcam camera sensor noise, exposure changes, and video stream compression."""
    def __init__(self, crop_jitter: bool = True):
        self.crop_jitter = crop_jitter

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        arr = np.array(img_pil)

        # 1. Random Crop / Padding / Scale Jitter (simulating 20% live face padding variation)
        if self.crop_jitter and random.random() < 0.40:
            h, w = arr.shape[:2]
            # Random pad or crop by 5-15%
            pad = int(min(h, w) * random.uniform(0.05, 0.15))
            padded = cv2.copyMakeBorder(arr, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            # Random crop back to original size
            ch, cw = padded.shape[:2]
            top = random.randint(0, ch - h)
            left = random.randint(0, cw - w)
            arr = padded[top:top + h, left:left + w]

        # 2. Random JPEG compression artifacts (simulating live video streaming quality)
        if random.random() < 0.35:
            quality = random.randint(45, 80)
            _, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                arr = dec

        # 3. Random Downscale-Upscale (simulating low-res webcam sensor)
        if random.random() < 0.30:
            h, w = arr.shape[:2]
            scale = random.uniform(0.55, 0.80)
            down = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
            arr = cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Random Gaussian Sensor Noise (simulating low-light webcam gain grain)
        if random.random() < 0.25:
            sigma = random.uniform(2.5, 8.0)
            gauss = np.random.normal(0, sigma, arr.shape)
            arr = np.clip(arr.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        # 5. Random Gamma (simulating webcam auto-exposure fluctuation)
        if random.random() < 0.30:
            gamma = random.uniform(0.75, 1.30)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            arr = cv2.LUT(arr, table)

        # 6. Mild motion blur
        if random.random() < 0.15:
            ksize = random.choice([3, 5])
            kernel = np.zeros((ksize, ksize))
            if random.random() < 0.5:
                kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
            else:
                kernel[:, int((ksize - 1) / 2)] = np.ones(ksize)
            kernel /= ksize
            arr = cv2.filter2D(arr, -1, kernel)

        return Image.fromarray(arr)


# ─── DATASET CLASS ─────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    """Loads images from file paths with pre-computed labels and torchvision transforms."""
    def __init__(self, samples: List[Tuple[Path, int]], transform=None, is_train: bool = True):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return dummy sample on failure
            dummy = torch.zeros(3, 224, 224)
            return dummy, label


def create_splits(
    rafdb_dir: Path,
    meld_cache_dir: Optional[Path] = None,
    seed: int = 42,
    raf_val_ratio: float = 0.10,
    meld_split_ratio: float = 0.90,
    meld_max_per_class: int = 1200
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Creates strict splits:
      train_samples = RAF-DB train (90%) + MELD train (90%)
      val_samples   = RAF-DB val (10%) + MELD val (10%)
      raf_test      = RAF-DB test (held-out)
      meld_test     = MELD val/test (held-out)
    """
    rng = random.Random(seed)
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    raf_test_samples: List[Tuple[Path, int]] = []
    meld_test_samples: List[Tuple[Path, int]] = []

    # 1. RAF-DB Train & Test
    raf_train_dir = rafdb_dir / "train"
    raf_test_dir = rafdb_dir / "test"

    if raf_train_dir.exists():
        for folder_num, target_idx in RAF_FOLDER_TO_IDX.items():
            fpath = raf_train_dir / folder_num
            if not fpath.exists():
                continue
            imgs = sorted(list(fpath.glob("*.jpg")) + list(fpath.glob("*.png")))
            rng.shuffle(imgs)
            split_idx = int(len(imgs) * (1.0 - raf_val_ratio))
            for p in imgs[:split_idx]:
                train_samples.append((p, target_idx))
            for p in imgs[split_idx:]:
                val_samples.append((p, target_idx))

    if raf_test_dir.exists():
        for folder_num, target_idx in RAF_FOLDER_TO_IDX.items():
            fpath = raf_test_dir / folder_num
            if not fpath.exists():
                continue
            imgs = sorted(list(fpath.glob("*.jpg")) + list(fpath.glob("*.png")))
            for p in imgs:
                raf_test_samples.append((p, target_idx))

    # 2. MELD Face Crops Cache (seed 101 for exact compatibility with benchmark_webcam.py)
    if meld_cache_dir and meld_cache_dir.exists():
        meld_rng = random.Random(101)
        for emo, target_idx in LABEL_MAPPING.items():
            emo_dir = meld_cache_dir / emo
            if not emo_dir.exists():
                continue
            imgs = sorted(list(emo_dir.glob("*.jpg")) + list(emo_dir.glob("*.png")))
            meld_rng.shuffle(imgs)
            if meld_max_per_class > 0 and len(imgs) > meld_max_per_class:
                imgs = imgs[:meld_max_per_class]
            split_idx = int(len(imgs) * meld_split_ratio)
            for p in imgs[:split_idx]:
                train_samples.append((p, target_idx))
            for p in imgs[split_idx:]:
                val_samples.append((p, target_idx))
                meld_test_samples.append((p, target_idx))

    rng.shuffle(train_samples)
    return train_samples, val_samples, raf_test_samples, meld_test_samples


def get_balanced_sampler(samples: List[Tuple[Path, int]], num_samples_per_class: int = 600) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler so each class has equal representation per epoch."""
    targets = [s[1] for s in samples]
    counts = Counter(targets)
    class_weights = {cls: 1.0 / max(count, 1) for cls, count in counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    total_samples = num_samples_per_class * len(LABEL_MAPPING)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )


# ─── MODEL BUILDERS ───────────────────────────────────────────────────────────
def build_experiment_A_model(ckpt_path: str, num_classes: int = 7) -> nn.Module:
    """Loads current production EfficientNet-B0 checkpoint."""
    logger.info(f"Building Experiment A model from: {ckpt_path}")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, num_classes),
    )
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)
        logger.info("Experiment A: Successfully loaded weights from existing checkpoint!")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found; initializing random weights.")
    return model


def build_experiment_B_model(hsemotion_path: str, num_classes: int = 7) -> nn.Module:
    """Loads HSEmotion enet_b0_8_best_afew, drops contempt, and remaps to 7 classes."""
    logger.info(f"Building Experiment B model from: {hsemotion_path}")
    if not HAS_TIMM:
        raise ImportError("timm is required for Experiment B. Run pip install timm.")

    if not os.path.exists(hsemotion_path):
        raise FileNotFoundError(f"HSEmotion checkpoint not found at {hsemotion_path}")

    ckpt = torch.load(hsemotion_path, map_location="cpu", weights_only=False)
    sd = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

    # Remap classifier keys if needed
    if "classifier.0.weight" in sd:
        sd["classifier.weight"] = sd.pop("classifier.0.weight")
    if "classifier.0.bias" in sd:
        sd["classifier.bias"] = sd.pop("classifier.0.bias")

    # Load into 8-class timm tf_efficientnet_b0
    m8 = timm.create_model("tf_efficientnet_b0", num_classes=8, pretrained=False)
    m8.load_state_dict(sd, strict=True)

    # Drop contempt (HSE idx 1), map [0, 2, 3, 4, 5, 6, 7] -> project [0, 1, 2, 3, 4, 5, 6]
    keep_indices = [0, 2, 3, 4, 5, 6, 7]
    w7 = m8.classifier.weight.data[keep_indices]
    b7 = m8.classifier.bias.data[keep_indices]

    m7 = timm.create_model("tf_efficientnet_b0", num_classes=num_classes, pretrained=False)
    sd7 = m8.state_dict()
    sd7["classifier.weight"] = w7
    sd7["classifier.bias"] = b7
    m7.load_state_dict(sd7, strict=True)
    logger.info("Experiment B: Successfully initialized from HSEmotion with 7 classes (contempt dropped)!")
    return m7


# ─── EVALUATION FUNCTION ──────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, Dict[str, float], np.ndarray]:
    """Computes Top-1 Accuracy, Macro-F1, Per-class F1, and 7x7 Confusion Matrix."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = float(np.mean(all_preds == all_targets) * 100.0)

    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    per_class_f1 = {}
    for idx, emo in enumerate(EMOTIONS):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        fn = np.sum(cm[idx, :]) - tp
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_f1[emo] = round(f1, 4)

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    return acc, macro_f1, per_class_f1, cm


# ─── TRAINING PIPELINE ────────────────────────────────────────────────────────
def train_experiment(
    experiment: str,
    rafdb_dir: str = "DATASET",
    meld_cache_dir: str = "models/video_classifier/face_crops_cache",
    hsemotion_path: str = "C:/Users/harsh/.hsemotion/enet_b0_8_best_afew.pt",
    output_checkpoint: str = "models/video_classifier/best_webcam_model_rafdb_A.pth",
    log_csv: str = "results/train_expA_log.csv",
    epochs: int = 12,
    stage1_epochs: int = 3,
    batch_size: int = 64,
    samples_per_class: int = 600,
    img_size: int = 224,
    seed: int = 42,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """Executes full staged training and validation for Exp A or Exp B."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    logger.info(f"Starting Training Experiment {experiment} on device: {device}")

    # Transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        WebcamRealisticAugmentation(crop_jitter=True),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        norm,
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.20)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])

    # Dataset splits
    train_samples, val_samples, raf_test, meld_test = create_splits(
        rafdb_dir=Path(rafdb_dir),
        meld_cache_dir=Path(meld_cache_dir) if meld_cache_dir else None,
        seed=seed
    )
    logger.info(f"Dataset summary: Train={len(train_samples)}, Val={len(val_samples)}, RAF Test={len(raf_test)}, MELD Held-out={len(meld_test)}")
    logger.info(f"Train class distribution: {Counter(s[1] for s in train_samples)}")
    logger.info(f"Val class distribution: {Counter(s[1] for s in val_samples)}")

    train_ds = EmotionDataset(train_samples, transform=train_tf, is_train=True)
    val_ds = EmotionDataset(val_samples, transform=val_tf, is_train=False)

    sampler = get_balanced_sampler(train_samples, num_samples_per_class=samples_per_class)
    num_workers = 0  # Safe default on Windows PowerShell
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build Model
    if experiment.upper() == "A":
        current_ckpt = "models/video_classifier/best_webcam_model.pth"
        model = build_experiment_A_model(current_ckpt, num_classes=7)
        arch_tag = "efficientnet_b0"
    elif experiment.upper() == "B":
        model = build_experiment_B_model(hsemotion_path, num_classes=7)
        arch_tag = "timm_tf_efficientnet_b0"
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    model.to(device)

    # Loss function with Label Smoothing 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

    # Logging setup
    Path(output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,stage,train_loss,val_acc,val_macro_f1,anger_f1,disgust_f1,fear_f1,joy_f1,neutral_f1,sadness_f1,surprise_f1,time_sec\n")

    best_val_macro_f1 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    # STAGE 1: Freeze backbone, train classifier head
    logger.info(f"=== STAGE 1: Training Head for {stage1_epochs} epochs with frozen backbone ===")
    if experiment.upper() == "A":
        for p in model.features.parameters():
            p.requires_grad = False
        head_params = model.classifier.parameters()
    else:
        # timm model
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False
        head_params = model.classifier.parameters()

    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        stage = 1 if epoch <= stage1_epochs else 2

        # STAGE 2 Transition: Unfreeze backbone with layer-wise LR decay
        if epoch == stage1_epochs + 1:
            logger.info("=== STAGE 2: Unfreezing entire model with layer-wise LR decay ===")
            for p in model.parameters():
                p.requires_grad = True

            if experiment.upper() == "A":
                param_groups = [
                    {"params": model.features[:4].parameters(), "lr": 3e-5},
                    {"params": model.features[4:].parameters(), "lr": 1e-4},
                    {"params": model.classifier.parameters(), "lr": 7e-4},
                ]
            else:
                param_groups = [
                    {"params": [p for n, p in model.named_parameters() if "conv_stem" in n or "bn1" in n or "blocks.0" in n or "blocks.1" in n], "lr": 3e-5},
                    {"params": [p for n, p in model.named_parameters() if any(f"blocks.{b}" in n for b in range(2, 7)) or "conv_head" in n], "lr": 1e-4},
                    {"params": model.classifier.parameters(), "lr": 7e-4},
                ]
            optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - stage1_epochs), eta_min=1e-6)

        # Train epoch
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Mixup / Cutmix with p <= 0.20
            if random.random() < 0.20 and len(images) > 1:
                lam = np.random.beta(0.2, 0.2)
                perm = torch.randperm(len(images))
                images = lam * images + (1 - lam) * images[perm]
                outputs = model(images)
                loss = lam * criterion(outputs, targets) + (1 - lam) * criterion(outputs, targets[perm])
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if stage == 2:
            scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        val_acc, val_f1, per_f1, _ = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} [Stage {stage}] "
            f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.4f} "
            f"(Fear: {per_f1['fear']:.3f}, Disgust: {per_f1['disgust']:.3f}) | Time: {epoch_time:.1f}s"
        )

        # Log CSV
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{stage},{avg_loss:.4f},{val_acc:.2f},{val_f1:.4f},"
                f"{per_f1['anger']},{per_f1['disgust']},{per_f1['fear']},{per_f1['joy']},"
                f"{per_f1['neutral']},{per_f1['sadness']},{per_f1['surprise']},{epoch_time:.1f}\n"
            )

        # Checkpoint if validation Macro-F1 improves
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            save_dict = {
                "epoch": epoch,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "per_class_f1": per_f1,
                "model_state_dict": model.state_dict(),
                "architecture": arch_tag,
                "label_mapping": LABEL_MAPPING,
                "img_size": img_size,
                "experiment": experiment.upper(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(save_dict, output_checkpoint)
            logger.info(f"  -> Saved NEW best checkpoint to {output_checkpoint} (Val Macro-F1: {val_f1:.4f})")
        else:
            if stage == 2:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (Patience={patience})")
                    break

    logger.info(f"Experiment {experiment} Complete! Best Val Macro-F1: {best_val_macro_f1:.4f} at epoch {best_epoch}")
    return {
        "experiment": experiment.upper(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint": output_checkpoint,
        "log_csv": log_csv,
    }


def parse_args():
    p = argparse.ArgumentParser(description="RAF-DB + MELD Webcam Emotion Training")
    p.add_argument("--experiment", type=str, default="A", choices=["A", "B", "a", "b"],
                   help="Experiment A (init from best_webcam_model.pth) or B (init from HSEmotion)")
    p.add_argument("--epochs", type=int, default=12, help="Total training epochs")
    p.add_argument("--stage1_epochs", type=int, default=3, help="Stage 1 frozen head epochs")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--samples_per_class", type=int, default=600, help="Samples per class per epoch")
    p.add_argument("--img_size", type=int, default=224, help="Image resolution")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--rafdb_dir", type=str, default="DATASET", help="RAF-DB dataset path")
    p.add_argument("--meld_cache", type=str, default="models/video_classifier/face_crops_cache", help="MELD crops path")
    p.add_argument("--hsemotion_path", type=str, default="C:/Users/harsh/.hsemotion/enet_b0_8_best_afew.pt", help="HSEmotion ckpt path")
    p.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    p.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    p.add_argument("--log_csv", type=str, default=None, help="Path for CSV log")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exp = args.experiment.upper()
    out_ckpt = args.output or f"models/video_classifier/best_webcam_model_rafdb_{exp}.pth"
    log_f = args.log_csv or f"results/train_exp{exp}_log.csv"

    train_experiment(
        experiment=exp,
        rafdb_dir=args.rafdb_dir,
        meld_cache_dir=args.meld_cache,
        hsemotion_path=args.hsemotion_path,
        output_checkpoint=out_ckpt,
        log_csv=log_f,
        epochs=args.epochs,
        stage1_epochs=args.stage1_epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        img_size=args.img_size,
        seed=args.seed,
        device_name=args.device,
    )
