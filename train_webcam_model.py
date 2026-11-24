"""
train_webcam_model.py
=====================
Train a fresh EfficientNet-B0 facial-expression classifier on face crops
extracted directly from MELD training videos.

Why this is better than the existing best_video_model.pth:
  - best_video_model.pth was trained on full video frames (TV-show clips),
    not tight face crops => very biased toward JOY/SADNESS on any face.
  - This script extracts ~3-5 face crops per video at key moments,
    so the model learns from actual close-up faces (same domain as webcam).
  - EfficientNet-B0 is a vastly better backbone for image classification
    than ResNet-50 + MLP head for 7-class emotion from still images.

Usage:
    python train_webcam_model.py [--max_videos N] [--epochs E] [--fast]

Output:
    models/video_classifier/best_webcam_model.pth
    (drop-in replacement, same label_mapping format)
"""
import os
import sys
import csv
import time
import random
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[ERROR] cv2 not available. Install opencv-python.")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webcam_trainer")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
LABEL_MAPPING = {
    "anger":    0,
    "disgust":  1,
    "fear":     2,
    "joy":      3,
    "neutral":  4,
    "sadness":  5,
    "surprise": 6,
}
# MELD emotion name -> our label key
MELD_EMO_MAP = {
    "anger":    "anger",
    "disgust":  "disgust",
    "fear":     "fear",
    "joy":      "joy",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": "surprise",
}

TRAIN_VIDEO_DIR  = "MELD.Raw/train/train_splits"
TRAIN_CSV        = "MELD.Raw/train/train_sent_emo.csv"
DEV_VIDEO_DIR    = "MELD.Raw/dev/dev_splits_complete"
DEV_CSV          = "MELD.Raw/dev/dev_sent_emo.csv"
YUNET_PATH       = "models/face_detection_yunet_2023mar.onnx"
CROP_CACHE_DIR   = "models/video_classifier/face_crops_cache"
OUTPUT_MODEL     = "models/video_classifier/best_webcam_model.pth"
BACKUP_OLD_MODEL = "models/video_classifier/best_video_model_backup.pth"

IMG_SIZE         = 96           # Smaller input = much faster CPU training; EfficientNet-B0 handles 96px well
BATCH_SIZE       = 64           # Larger batch = better CPU utilization
NUM_EPOCHS       = 20
LR               = 1e-3
WEIGHT_DECAY     = 1e-4
FRAMES_PER_VIDEO = 4            # sample N frames uniformly per clip
MIN_FACE_PIX     = 32           # skip faces smaller than this in px
FACE_SCORE_THRESH = 0.50        # YuNet confidence threshold
MAX_SAMPLES_PER_CLASS = 600     # Cap per class to keep total dataset ~4200 → ~1 min/epoch on CPU
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_videos", type=int, default=0,
                   help="Cap number of training videos (0 = all)")
    p.add_argument("--epochs",     type=int, default=NUM_EPOCHS)
    p.add_argument("--fast",       action="store_true",
                   help="Fast mode: 500 videos, 5 epochs (for quick testing)")
    p.add_argument("--use_cache",  action="store_true",
                   help="Skip extraction and use existing crop cache")
    p.add_argument("--no_cache",   action="store_true",
                   help="Always re-extract crops (ignore cache)")
    p.add_argument("--resume",     action="store_true",
                   help="Resume training from existing best_webcam_model.pth at lower LR")
    return p.parse_args()


# ─── YUNET FACE DETECTOR ──────────────────────────────────────────────────────
class YuNetDetector:
    def __init__(self, model_path, score_threshold=FACE_SCORE_THRESH):
        if not os.path.exists(model_path):
            logger.warning(f"YuNet not found at {model_path}. Using center-crop fallback.")
            self.detector = None
        else:
            self.detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(320, 240),
                score_threshold=score_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )

    def detect_largest(self, bgr_frame):
        """Returns (x, y, w, h) of the largest face, or None."""
        if self.detector is None:
            # center crop fallback
            h, w = bgr_frame.shape[:2]
            s = min(h, w)
            return (w // 2 - s // 2, h // 2 - s // 2, s, s)

        h, w = bgr_frame.shape[:2]
        self.detector.setInputSize((w, h))
        try:
            _, faces = self.detector.detect(bgr_frame)
        except Exception:
            return None
        if faces is None or len(faces) == 0:
            return None
        # Pick the largest face by area
        best = max(faces, key=lambda f: float(f[2] * f[3]))
        return tuple(int(v) for v in best[:4])  # x, y, w, h


# ─── CROP EXTRACTION ──────────────────────────────────────────────────────────
def load_csv_labels(csv_path):
    """Returns dict: filename_no_ext -> emotion_label_key"""
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dia = int(row.get("Dialogue_ID", row.get("dialogue_ID", -1)))
                utt = int(row.get("Utterance_ID", row.get("utterance_ID", -1)))
                emo = row.get("Emotion", row.get("emotion", "")).strip().lower()
            except (ValueError, KeyError):
                continue
            emo_key = MELD_EMO_MAP.get(emo)
            if emo_key is None:
                continue
            fname = f"dia{dia}_utt{utt}"
            labels[fname] = emo_key
    return labels


def extract_crops_from_video(video_path, detector, n_frames=FRAMES_PER_VIDEO, img_size=IMG_SIZE):
    """
    Opens a video, samples n_frames uniformly, detects the largest face,
    and returns a list of (img_size x img_size x 3) uint8 BGR arrays.
    """
    crops = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return crops

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return crops

    frame_indices = np.linspace(0, max(0, total - 1), n_frames, dtype=int)
    seen = set()
    for fi in frame_indices:
        if fi in seen:
            continue
        seen.add(fi)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        bbox = detector.detect_largest(frame)
        if bbox is None:
            continue

        x, y, w, h = bbox
        # expand bbox by 25% to include jaw, brow, forehead
        pad_x, pad_y = int(w * 0.25), int(h * 0.25)
        fh, fw = frame.shape[:2]
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)

        face = frame[y1:y2, x1:x2]
        if face.shape[0] < MIN_FACE_PIX or face.shape[1] < MIN_FACE_PIX:
            continue

        face_resized = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)
        crops.append(face_resized)

    cap.release()
    return crops


def build_crop_cache(video_dir, csv_path, cache_dir, detector, max_videos=0):
    """
    Extract face crops from all videos and save to cache_dir/<label>/<stem>_<frame>.jpg
    Returns: total crops written
    """
    labels = load_csv_labels(csv_path)
    if not labels:
        logger.error(f"No labels loaded from {csv_path}")
        return 0

    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if max_videos > 0:
        video_files = video_files[:max_videos]

    logger.info(f"Found {len(video_files)} videos in {video_dir}")

    # Create per-class subdirs
    for cls in LABEL_MAPPING:
        (Path(cache_dir) / cls).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_no_label = 0
    skipped_no_face = 0

    for idx, vpath in enumerate(video_files):
        stem = vpath.stem
        label_key = labels.get(stem)
        if label_key is None:
            skipped_no_label += 1
            continue

        crops = extract_crops_from_video(vpath, detector)
        if not crops:
            skipped_no_face += 1
            continue

        for fi, crop in enumerate(crops):
            out_path = Path(cache_dir) / label_key / f"{stem}_{fi}.jpg"
            cv2.imwrite(str(out_path), crop)
            written += 1

        if (idx + 1) % 500 == 0:
            logger.info(f"  Extracted {idx+1}/{len(video_files)} videos | {written} crops so far")

    logger.info(
        f"Extraction done: {written} crops | "
        f"{skipped_no_label} no-label | {skipped_no_face} no-face"
    )
    return written


# ─── PYTORCH DATASET ──────────────────────────────────────────────────────────
class FaceCropDataset(Dataset):
    def __init__(self, cache_dir, transform=None, split_ratio=1.0, seed=42, is_train=True,
                 max_per_class=MAX_SAMPLES_PER_CLASS):
        self.transform = transform
        self.samples = []  # (path, label_idx)

        rng = random.Random(seed)
        for cls, idx in LABEL_MAPPING.items():
            cls_dir = Path(cache_dir) / cls
            if not cls_dir.exists():
                continue
            paths = sorted(cls_dir.glob("*.jpg"))
            rng.shuffle(paths)
            # Apply per-class cap BEFORE splitting (keeps class balance)
            if max_per_class > 0 and len(paths) > max_per_class:
                paths = paths[:max_per_class]
            split_at = int(len(paths) * split_ratio)
            if is_train:
                selected = paths[:split_at]
            else:
                selected = paths[split_at:]
            for p in selected:
                self.samples.append((p, idx))

        rng.shuffle(self.samples)
        counts = Counter(s[1] for s in self.samples)
        logger.info(
            f"Dataset ({'train' if is_train else 'val'}): "
            f"{len(self.samples)} samples | "
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

        if self.transform:
            # transforms expect PIL Image or tensor; convert
            from PIL import Image
            pil = Image.fromarray(img_rgb)
            return self.transform(pil), label
        else:
            t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            return t, label


# ─── TRAINING ─────────────────────────────────────────────────────────────────
def build_transforms(img_size=IMG_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 12, img_size + 12)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
        transforms.RandomRotation(degrees=12),
        transforms.RandomGrayscale(p=0.08),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def compute_class_weights(dataset):
    """Inverse-frequency class weights to handle MELD class imbalance."""
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    n_cls = len(LABEL_MAPPING)
    weights = []
    for i in range(n_cls):
        cnt = counts.get(i, 1)
        weights.append(total / (n_cls * cnt))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * n_cls  # normalize so mean weight = 1.0


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.fast:
        args.max_videos = 500
        args.epochs = 5
        logger.info("FAST mode: 500 videos, 5 epochs")

    if args.resume:
        logger.info(f"RESUME mode: continuing from {OUTPUT_MODEL} for {args.epochs} more epochs at LR={LR*0.05:.6f}")

    # ── 1. Extract face crops ──────────────────────────────────────────────
    cache_exists = (
        os.path.exists(CROP_CACHE_DIR)
        and any(Path(CROP_CACHE_DIR).rglob("*.jpg"))
    )

    if args.no_cache or not cache_exists:
        logger.info("Extracting face crops from MELD training videos...")
        detector = YuNetDetector(YUNET_PATH)
        n = build_crop_cache(TRAIN_VIDEO_DIR, TRAIN_CSV, CROP_CACHE_DIR, detector, args.max_videos)
        if n == 0:
            logger.error("No crops extracted! Check MELD.Raw/train/train_splits path.")
            sys.exit(1)
    else:
        n_cached = len(list(Path(CROP_CACHE_DIR).rglob("*.jpg")))
        logger.info(f"Using existing crop cache: {n_cached} images in {CROP_CACHE_DIR}")

    # Also build a small val set from dev split
    dev_cache = CROP_CACHE_DIR + "_dev"
    dev_cache_exists = (
        os.path.exists(dev_cache) and any(Path(dev_cache).rglob("*.jpg"))
    )
    if not dev_cache_exists and os.path.exists(DEV_VIDEO_DIR):
        logger.info("Extracting dev face crops for validation...")
        detector = YuNetDetector(YUNET_PATH)
        build_crop_cache(DEV_VIDEO_DIR, DEV_CSV, dev_cache, detector, max_videos=200)

    # ── 2. Build datasets ──────────────────────────────────────────────────
    train_tf, val_tf = build_transforms(IMG_SIZE)

    # 90% train / 10% val from cache; also try dev cache if available
    train_ds = FaceCropDataset(CROP_CACHE_DIR, transform=train_tf, split_ratio=0.90, is_train=True)
    val_ds   = FaceCropDataset(CROP_CACHE_DIR, transform=val_tf,   split_ratio=0.90, is_train=False)

    if dev_cache_exists or os.path.exists(dev_cache):
        dev_ds = FaceCropDataset(dev_cache, transform=val_tf, split_ratio=1.0, is_train=True)
        # Combine val_ds + dev_ds
        from torch.utils.data import ConcatDataset
        val_ds = ConcatDataset([val_ds, dev_ds])
        logger.info(f"Val set includes dev-split crops. Total val: {len(val_ds)}")

    if len(train_ds) < 10:
        logger.error("Training dataset is empty! Aborting.")
        sys.exit(1)

    class_weights = compute_class_weights(train_ds).to(device)
    logger.info(f"Class weights: {class_weights.cpu().tolist()}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── 3. Build model ────────────────────────────────────────────────────
    logger.info("Loading EfficientNet-B0 (ImageNet pretrained)...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # Replace the final classifier: 1280 -> 7
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.25),
        nn.Linear(256, len(LABEL_MAPPING)),
    )
    model = model.to(device)
    logger.info(f"Model: EfficientNet-B0, output dim: {in_features} -> {len(LABEL_MAPPING)} classes")

    # ── 4. Optimizer: two-phase LR (head warm-up → full fine-tune) ───────
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Phase 1: train only the new head (first 5 epochs)
    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    best_epoch   = 0
    warmup_epochs = min(3, args.epochs // 5)

    logger.info(f"Training for {args.epochs} epochs (warmup: {warmup_epochs} epochs head-only)")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Switch to full fine-tune after warmup
        if epoch == warmup_epochs + 1:
            logger.info("Unfreezing backbone for full fine-tune...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)

        # ── Train ──
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
            train_loss   += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total   += len(y)

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc      = train_correct / max(train_total, 1) * 100.0

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds  = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total   += len(y)

        val_acc = val_correct / max(val_total, 1) * 100.0
        elapsed = time.time() - t_start

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"TrainLoss={avg_train_loss:.4f} | TrainAcc={train_acc:.1f}% | "
            f"ValAcc={val_acc:.1f}% | {elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            # Save checkpoint
            os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_mapping":    LABEL_MAPPING,
                "input_dim":        in_features,
                "architecture":     "efficientnet_b0",
                "img_size":         IMG_SIZE,
                "val_acc":          best_val_acc,
                "epoch":            epoch,
            }, OUTPUT_MODEL)
            logger.info(f"  ** New best: {best_val_acc:.1f}% at epoch {epoch} → saved to {OUTPUT_MODEL}")

    logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}% at epoch {best_epoch}")
    logger.info(f"Model saved to: {OUTPUT_MODEL}")

    # Backup old model if it exists
    old_model = "models/video_classifier/best_video_model.pth"
    if os.path.exists(old_model) and not os.path.exists(BACKUP_OLD_MODEL):
        import shutil
        shutil.copy2(old_model, BACKUP_OLD_MODEL)
        logger.info(f"Backed up old model to {BACKUP_OLD_MODEL}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
