"""Fine-tune the RAF-DB ResNet-18 checkpoint on mixed RAF-DB + RAVDESS face crops."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode

CANONICAL_CLASSES = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)
CANONICAL_TO_IDX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

NAME_ALIASES = {
    "anger": "anger",
    "angry": "anger",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "sad": "sadness",
    "surprise": "surprise",
    "surprised": "surprise",
}

RAF_FOLDER_TO_NAME = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happiness",
    "5": "sadness",
    "6": "anger",
    "7": "neutral",
}

# Exact split seed/fraction from train_raf_resnet18.py so validation matches original training.
RAF_SPLIT_SEED = 42
RAF_VAL_FRACTION = 0.15
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

FREEZE_STAGE_PREFIXES = {
    "none": (),
    "layer1": ("conv1", "bn1", "layer1"),
    "layer2": ("conv1", "bn1", "layer1", "layer2"),
    "layer3": ("conv1", "bn1", "layer1", "layer2", "layer3"),
}


def canonicalize_class_name(name: str) -> str:
    """Normalize emotion aliases to the fixed canonical class vocabulary."""
    key = str(name).strip().lower()
    if key not in NAME_ALIASES:
        raise ValueError(f"Unknown emotion class name: '{name}'")
    return NAME_ALIASES[key]


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker RNGs from PyTorch's per-worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_init_checkpoint(init_path: Path) -> tuple[nn.Module, dict[str, Any], list[str]]:
    """Load notebook-format ResNet-18 checkpoint and align head rows by class name."""
    ck = torch.load(init_path, map_location="cpu", weights_only=True)
    for required in ("architecture", "class_names", "preprocessing", "model_state_dict"):
        if required not in ck:
            raise ValueError(f"Checkpoint missing required key '{required}': {init_path}")
    if ck["architecture"] != "resnet18":
        raise ValueError(f"Unsupported checkpoint architecture: {ck['architecture']}")

    raw_names = list(ck["class_names"])
    if len(raw_names) != len(CANONICAL_CLASSES):
        raise ValueError(f"Expected {len(CANONICAL_CLASSES)} classes, got {len(raw_names)}")

    mapped_names = [canonicalize_class_name(n) for n in raw_names]
    if len(set(mapped_names)) != len(CANONICAL_CLASSES):
        raise ValueError(f"Checkpoint class_names do not map cleanly to 7 classes: {raw_names}")

    # Reorder head rows from checkpoint class order into fixed canonical class order.
    name_to_src_idx = {canon: idx for idx, canon in enumerate(mapped_names)}
    perm = torch.tensor([name_to_src_idx[c] for c in CANONICAL_CLASSES], dtype=torch.long)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CANONICAL_CLASSES))
    model.load_state_dict(ck["model_state_dict"])
    with torch.no_grad():
        model.fc.weight.copy_(model.fc.weight[perm])
        model.fc.bias.copy_(model.fc.bias[perm])

    prep = dict(ck["preprocessing"])
    for key in ("image_size", "eval_resize", "mean", "std"):
        if key not in prep:
            raise ValueError(f"Checkpoint preprocessing missing '{key}'")

    return model.eval(), prep, mapped_names


def _verify_raf_folder_mapping(checkpoint_names: set[str]) -> dict[int, int]:
    """Map RAF-DB folder indices 1..7 by name and verify against checkpoint classes."""
    folder_to_target: dict[int, int] = {}
    for folder_str, raf_name in RAF_FOLDER_TO_NAME.items():
        canon = canonicalize_class_name(raf_name)
        if canon not in checkpoint_names:
            raise ValueError(f"RAF class '{raf_name}' missing from checkpoint class_names")
        folder_to_target[int(folder_str)] = CANONICAL_TO_IDX[canon]
    return folder_to_target


def _read_raf_split_frame(
    raf_dir: Path, split: str, folder_to_target: dict[int, int]
) -> tuple[pd.DataFrame, bool]:
    """Read RAF-DB CSV annotations when available, or scan numbered class folders."""
    csv_path = raf_dir / f"{split}_labels.csv"
    split_root = (raf_dir / split).resolve()
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)[["image", "label"]].copy()
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
        frame["path"] = [
            str((split_root / str(row.label) / str(row.image)).resolve())
            for row in frame.itertuples(index=False)
        ]
        frame["target"] = frame["label"].map(folder_to_target)
        return frame, True

    rows: list[dict[str, Any]] = []
    for folder_idx in sorted(folder_to_target):
        cls_dir = split_root / str(folder_idx)
        if not cls_dir.is_dir():
            continue
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "image": img_path.name,
                    "label": folder_idx,
                    "path": str(img_path.resolve()),
                    "target": folder_to_target[folder_idx],
                })
    return pd.DataFrame(rows), False


def _dedup_raf_train_pixels(train_df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce train_raf_resnet18.py pixel-hash deduplication before train/val split."""
    hashes: list[str] = []
    for path in train_df["path"]:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode())
            digest.update(rgb.tobytes())
            hashes.append(digest.hexdigest())
    deduped = train_df.copy()
    deduped["pixel_hash"] = hashes
    return deduped.drop_duplicates("pixel_hash").reset_index(drop=True)


def load_raf_splits(
    raf_dir: Path, checkpoint_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load RAF-DB train/val/test reproducing the original train/val split."""
    folder_to_target = _verify_raf_folder_mapping(set(checkpoint_names))
    full_train_df, has_train_csv = _read_raf_split_frame(raf_dir, "train", folder_to_target)
    test_df, _ = _read_raf_split_frame(raf_dir, "test", folder_to_target)
    if full_train_df.empty or test_df.empty:
        raise ValueError(f"Empty RAF-DB train or test data under {raf_dir}")

    if has_train_csv:
        full_train_df = _dedup_raf_train_pixels(full_train_df)

    n_classes = full_train_df["target"].nunique()
    n_val = int(round(len(full_train_df) * RAF_VAL_FRACTION))
    if n_val < n_classes or (len(full_train_df) - n_val) < n_classes:
        # Tiny synthetic datasets cannot stratify <1 sample per class.
        test_size: Any = max(n_classes, n_val)
        if len(full_train_df) - test_size < n_classes:
            raise ValueError("RAF-DB train split too small to stratify across present classes.")
    else:
        test_size = RAF_VAL_FRACTION

    raf_train, raf_val = train_test_split(
        full_train_df,
        test_size=test_size,
        random_state=RAF_SPLIT_SEED,
        stratify=full_train_df["target"],
    )
    return (
        raf_train.reset_index(drop=True),
        raf_val.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _scan_named_split(split_dir: Path) -> pd.DataFrame:
    """Scan <split_dir>/<class_name>/* folders and map labels by canonical name."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {split_dir}")
    rows: list[dict[str, Any]] = []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        try:
            canon = canonicalize_class_name(cls_dir.name)
        except ValueError:
            warnings.warn(f"Skipping unknown emotion folder '{cls_dir.name}' in {split_dir}")
            continue
        target = CANONICAL_TO_IDX[canon]
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": str(img_path.resolve()), "target": target, "name": img_path.name})
    return pd.DataFrame(rows)


def load_extra_splits(extra_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RAVDESS train and val crop folders and verify actor-disjoint splits."""
    train_df = _scan_named_split(extra_dir / "train")
    val_df = _scan_named_split(extra_dir / "val")
    if train_df.empty or val_df.empty:
        raise ValueError(f"Empty RAVDESS train or val crops under {extra_dir}")

    actor_re = re.compile(r"^(actor\d+)_", re.IGNORECASE)
    train_actors = {m.group(1).lower() for n in train_df["name"] if (m := actor_re.match(n))}
    val_actors = {m.group(1).lower() for n in val_df["name"] if (m := actor_re.match(n))}
    overlap = train_actors & val_actors
    if overlap:
        raise ValueError(f"Actor leakage between train and val crops: {sorted(overlap)}")
    return train_df, val_df


class MixedAugmenter:
    """Shared webcam-domain augmentation pipeline for RAF-DB and RAVDESS crops."""

    def __init__(self, eval_resize: int, image_size: int, interpolation: InterpolationMode, antialias: bool) -> None:
        self.image_size = image_size
        self.geo = transforms.Compose([
            transforms.Resize(eval_resize, interpolation=interpolation, antialias=antialias),
            transforms.RandomResizedCrop(
                image_size, scale=(0.75, 1.0), ratio=(0.95, 1.05),
                interpolation=interpolation, antialias=antialias,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12.0, interpolation=interpolation),
        ])

    @staticmethod
    def _apply_glasses_occlusion(rgb: np.ndarray) -> np.ndarray:
        """Draw thin dark eyeglass rims + bridge + soft specular glare near the eye line."""
        h, w = rgb.shape[:2]
        out = rgb.copy()
        eye_y = int(round(h * random.uniform(0.34, 0.43)))
        lx = int(round(w * random.uniform(0.28, 0.34)))
        rx = int(round(w * random.uniform(0.66, 0.72)))
        rad_x = max(4, int(round(w * random.uniform(0.13, 0.17))))
        rad_y = max(3, int(round(h * random.uniform(0.08, 0.12))))
        rim_shade = random.randint(15, 55)
        rim_color = (rim_shade, rim_shade, rim_shade)
        thickness = random.choice((1, 2))

        for cx in (lx, rx):
            cv2.ellipse(out, (cx, eye_y), (rad_x, rad_y), 0, 0, 360, rim_color, thickness, cv2.LINE_AA)
        cv2.line(out, (lx + rad_x, eye_y), (rx - rad_x, eye_y), rim_color, thickness, cv2.LINE_AA)
        cv2.line(out, (0, eye_y), (max(0, lx - rad_x), eye_y), rim_color, 1, cv2.LINE_AA)
        cv2.line(out, (min(w - 1, rx + rad_x), eye_y), (w - 1, eye_y), rim_color, 1, cv2.LINE_AA)

        glare_overlay = out.copy()
        alpha = random.uniform(0.18, 0.38)
        for cx in (lx, rx):
            gx0 = cx - rad_x // 2
            gy0 = eye_y - rad_y // 2
            gx1 = cx + rad_x // 3
            gy1 = eye_y + rad_y // 3
            cv2.line(glare_overlay, (gx0, gy1), (gx1, gy0), (245, 250, 255), max(2, rad_y // 3), cv2.LINE_AA)
        return cv2.addWeighted(glare_overlay, alpha, out, 1.0 - alpha, 0.0)

    def __call__(self, image: Image.Image) -> Image.Image:
        img = self.geo(image)
        arr = np.asarray(img, dtype=np.uint8)

        # Brightness, contrast, and gamma jitter
        if random.random() < 0.7:
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-24.0, 24.0)
            arr = np.clip(arr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            gamma = random.uniform(0.75, 1.35)
            lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            arr = cv2.LUT(arr, lut)

        # Color-temperature shift (warm/cool)
        if random.random() < 0.5:
            dt = random.uniform(-0.12, 0.12)
            scales = np.array([1.0 + dt, 1.0 + 0.15 * dt, 1.0 - dt], dtype=np.float32)
            arr = np.clip(arr.astype(np.float32) * scales, 0, 255).astype(np.uint8)

        # Downscale-upscale (webcam softness)
        if random.random() < 0.4:
            scale = random.uniform(0.45, 0.80)
            small = max(16, int(round(self.image_size * scale)))
            down = cv2.resize(arr, (small, small), interpolation=cv2.INTER_AREA)
            arr = cv2.resize(down, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Blur
        if random.random() < 0.35:
            sigma = random.uniform(0.4, 1.4)
            arr = cv2.GaussianBlur(arr, (3, 3), sigmaX=sigma)

        # JPEG compression artifacts
        if random.random() < 0.4:
            quality = random.randint(35, 85)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    arr = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

        # Sensor noise
        if random.random() < 0.4:
            sigma = random.uniform(2.0, 10.0)
            noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
            arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random glasses-like occlusion (~30% of images)
        if random.random() < 0.30:
            arr = self._apply_glasses_occlusion(arr)

        return Image.fromarray(arr)


def build_transforms(prep: dict[str, Any]) -> tuple[Callable, Callable]:
    """Construct train and eval transforms strictly from checkpoint preprocessing."""
    image_size = int(prep["image_size"])
    eval_resize = int(prep["eval_resize"])
    mean = tuple(float(v) for v in prep["mean"])
    std = tuple(float(v) for v in prep["std"])
    interp_name = str(prep.get("interpolation", "bilinear")).upper()
    interp = getattr(InterpolationMode, interp_name, InterpolationMode.BILINEAR)
    antialias = bool(prep.get("antialias", True))

    normalize = transforms.Normalize(mean, std)
    eval_tf = transforms.Compose([
        transforms.Resize(eval_resize, interpolation=interp, antialias=antialias),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_tf = transforms.Compose([
        MixedAugmenter(eval_resize, image_size, interp, antialias),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


class PathTargetDataset(Dataset):
    """Load RGB crops by path and return canonical integer targets."""

    def __init__(self, frame: pd.DataFrame, transform: Callable) -> None:
        self.paths = frame["path"].to_numpy(dtype=str, copy=True)
        self.targets = frame["target"].to_numpy(dtype=np.int64, copy=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, int(self.targets[idx])


def make_balanced_train_loader(
    frame: pd.DataFrame,
    transform: Callable,
    batch_size: int,
    num_samples: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a class-balanced DataLoader via WeightedRandomSampler."""
    targets = frame["target"].to_numpy(dtype=np.int64)
    counts = np.bincount(targets, minlength=len(CANONICAL_CLASSES)).astype(np.float64)
    inv_freq = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
    sample_weights = torch.as_tensor(inv_freq[targets], dtype=torch.double)
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=num_samples, replacement=True, generator=gen
    )
    return DataLoader(
        PathTargetDataset(frame, transform),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )


def make_eval_loader(
    frame: pd.DataFrame, transform: Callable, batch_size: int, pin_memory: bool
) -> DataLoader:
    """Create a deterministic evaluation DataLoader."""
    return DataLoader(
        PathTargetDataset(frame, transform),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin_memory,
    )


def apply_freeze(model: nn.Module, freeze_until: str) -> None:
    """Freeze backbone prefixes up to the chosen stage while leaving head trainable."""
    prefixes = FREEZE_STAGE_PREFIXES[freeze_until]
    for name, param in model.named_parameters():
        is_frozen = any(name == p or name.startswith(p + ".") for p in prefixes)
        param.requires_grad = not is_frozen


def set_train_eval_modes(model: nn.Module, freeze_until: str, training: bool) -> None:
    """Keep frozen BatchNorm layers in eval mode so running stats do not drift."""
    if not training:
        model.eval()
        return
    model.train()
    for prefix in FREEZE_STAGE_PREFIXES[freeze_until]:
        mod = getattr(model, prefix, None)
        if mod is not None:
            mod.eval()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """AdamW with backbone LR 2e-5 and head LR 2e-4 over trainable parameters."""
    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("fc.")
    ]
    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": 2e-5})
    if head_params:
        groups.append({"params": head_params, "lr": 2e-4})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, macro-F1, and per-class F1 from a NxN confusion matrix."""
    tp = np.diag(matrix).astype(np.float64)
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    per_class = {
        CANONICAL_CLASSES[i]: round(float(f1[i]), 4)
        for i in range(len(CANONICAL_CLASSES))
    }
    return {
        "accuracy": float(tp.sum() / max(int(matrix.sum()), 1)),
        "macro_f1": float(f1.mean()),
        "per_class_f1": per_class,
    }


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    """Evaluate a model on one DataLoader without modifying gradients or BN stats."""
    model.eval()
    n_cls = len(CANONICAL_CLASSES)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        t_np = targets.numpy()
        np.add.at(cm, (t_np, preds), 1)
    return metrics_from_confusion(cm)


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write checkpoint atomically via a temporary file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def save_inference_checkpoint(
    model: nn.Module,
    prep: dict[str, Any],
    epoch: int,
    avg_val_macro_f1: float,
    raf_val_macro_f1: float,
    ravdess_val_macro_f1: float,
    destination: Path,
) -> None:
    """Save checkpoint in the notebook format compatible with live_resnet18_demo."""
    payload = {
        "schema_version": 2,
        "architecture": "resnet18",
        "model_state_dict": model.state_dict(),
        "class_names": list(CANONICAL_CLASSES),
        "epoch": int(epoch),
        "val_macro_f1": float(avg_val_macro_f1),
        "raf_val_macro_f1": float(raf_val_macro_f1),
        "ravdess_val_macro_f1": float(ravdess_val_macro_f1),
        "preprocessing": dict(prep),
    }
    atomic_torch_save(payload, destination)


def _capture_rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "kind": np_state[0],
            "keys": np_state[1].tolist(),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((
        ns["kind"],
        np.asarray(ns["keys"], dtype=np.uint32),
        ns["position"],
        ns["has_gauss"],
        ns["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def write_summary_json(summary: dict[str, Any], path: Path) -> None:
    """Write compact train_summary.json strictly under 30 lines."""
    lines = ["{"]
    items = list(summary.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f'  {json.dumps(k)}: {json.dumps(v)}{comma}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    if len(text.splitlines()) > 30:
        raise ValueError(f"train_summary.json exceeded 30 lines ({len(text.splitlines())})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_mixed_loaders(
    raf_train: pd.DataFrame,
    extra_train: pd.DataFrame,
    train_tf: Callable,
    batch_size: int,
    extra_share: float,
    epoch_seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, Optional[DataLoader], int, int, int]:
    """Compute per-source batch slice sizes and build class-balanced loaders."""
    if not 0.0 <= extra_share < 1.0:
        raise ValueError(f"--extra-share must be in [0, 1), got {extra_share}")
    n_extra = int(round(batch_size * extra_share))
    if extra_share > 0.0 and batch_size >= 2:
        n_extra = max(1, min(batch_size - 1, n_extra))
    n_raf = batch_size - n_extra
    steps = max(1, math.ceil(len(raf_train) / max(1, n_raf)))

    raf_loader = make_balanced_train_loader(
        raf_train, train_tf, n_raf, steps * n_raf, epoch_seed, pin_memory
    )
    extra_loader = (
        make_balanced_train_loader(
            extra_train, train_tf, n_extra, steps * n_extra, epoch_seed + 10000, pin_memory
        )
        if n_extra > 0
        else None
    )
    return raf_loader, extra_loader, steps, n_raf, n_extra


def _train_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    n_raf: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    amp_enabled: bool,
    device: torch.device,
    cm_raf: np.ndarray,
    cm_extra: np.ndarray,
) -> float:
    """Run one mixed forward/backward step and update per-source train confusion matrices."""
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        logits = model(images)
        loss = criterion(logits, targets)
    if amp_enabled:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    preds = logits.detach().argmax(dim=1).cpu().numpy()
    t_np = targets.detach().cpu().numpy()
    if n_raf > 0:
        np.add.at(cm_raf, (t_np[:n_raf], preds[:n_raf]), 1)
    if len(t_np) > n_raf:
        np.add.at(cm_extra, (t_np[n_raf:], preds[n_raf:]), 1)
    return float(loss.item())


def run_dry_run(
    model: nn.Module,
    raf_train: pd.DataFrame,
    extra_train: pd.DataFrame,
    train_tf: Callable,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> None:
    """Time 20 training batches and print estimated epoch minutes and CPU guidance."""
    apply_freeze(model, args.freeze_until)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    pin_memory = device.type == "cuda"

    raf_loader, extra_loader, steps, n_raf, _ = _build_mixed_loaders(
        raf_train, extra_train, train_tf, args.batch_size, args.extra_share, args.seed, pin_memory
    )
    set_train_eval_modes(model, args.freeze_until, training=True)

    n_cls = len(CANONICAL_CLASSES)
    cm_raf = np.zeros((n_cls, n_cls), dtype=np.int64)
    cm_extra = np.zeros((n_cls, n_cls), dtype=np.int64)

    batch_iter = (
        zip(itertools.cycle(raf_loader), itertools.cycle(extra_loader))
        if extra_loader is not None
        else ((b, None) for b in itertools.cycle(raf_loader))
    )

    t0 = time.perf_counter()
    for (x_r, y_r), extra_batch in itertools.islice(batch_iter, 20):
        if extra_batch is not None:
            x_e, y_e = extra_batch
            images = torch.cat([x_r, x_e], dim=0)
            targets = torch.cat([y_r, y_e], dim=0)
        else:
            images, targets = x_r, y_r
        _train_step(
            model, images, targets, len(y_r), criterion, optimizer,
            scaler, amp_enabled, device, cm_raf, cm_extra,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    est_min = (elapsed / 20.0) * steps / 60.0
    print(
        f"Dry-run: 20 batches took {elapsed:.2f}s | steps/epoch={steps} | "
        f"estimated={est_min:.2f} min/epoch on {device.type}"
    )
    if device.type != "cuda":
        print(
            "CPU detected (no GPU): recommended --freeze-until layer3 "
            "(fastest CPU backward pass while adapting layer4 + fc; use layer2 if time permits)."
        )


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run full mixed RAF-DB + RAVDESS training or --dry-run."""
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    pin_memory = device.type == "cuda"

    init_path = args.init.resolve()
    out_dir = args.out
    best_ckpt_path = out_dir / "best_resnet18_ravdess.pt"
    last_ckpt_path = out_dir / "last_training.pt"
    summary_path = out_dir / "train_summary.json"
    if best_ckpt_path.resolve() == init_path or out_dir.resolve() == init_path.parent:
        raise ValueError("Refusing to write inside --init checkpoint directory.")
    if not args.dry_run and args.resume is None and (best_ckpt_path.exists() or last_ckpt_path.exists()):
        raise FileExistsError(
            f"Checkpoints already exist in {out_dir}; choose a new --out or pass --resume."
        )

    model, prep, ck_names = load_init_checkpoint(init_path)
    model = model.to(device)
    train_tf, eval_tf = build_transforms(prep)

    raf_train, raf_val, raf_test = load_raf_splits(args.raf_dir, ck_names)
    extra_train, extra_val = load_extra_splits(args.extra_dir)

    if args.dry_run:
        run_dry_run(model, raf_train, extra_train, train_tf, args, device, amp_enabled)
        return {"dry_run": True}

    raf_val_loader = make_eval_loader(raf_val, eval_tf, args.batch_size, pin_memory)
    raf_test_loader = make_eval_loader(raf_test, eval_tf, args.batch_size, pin_memory)
    extra_val_loader = make_eval_loader(extra_val, eval_tf, args.batch_size, pin_memory)

    apply_freeze(model, args.freeze_until)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_details: dict[str, Any] = {}

    if args.resume is not None:
        resume_path = last_ckpt_path if args.resume == "__AUTO__" else Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if amp_enabled and state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        best_details = dict(state["best_details"])
        raf_test_before = dict(state["raf_test_before"])
        _restore_rng_state(state["rng"])
    else:
        before_metrics = evaluate_loader(model, raf_test_loader, device)
        raf_test_before = {
            "accuracy": round(before_metrics["accuracy"], 4),
            "macro_f1": round(before_metrics["macro_f1"], 4),
        }

    patience = 3
    n_cls = len(CANONICAL_CLASSES)
    for epoch in range(start_epoch, args.epochs + 1):
        raf_loader, extra_loader, steps, _, _ = _build_mixed_loaders(
            raf_train, extra_train, train_tf, args.batch_size,
            args.extra_share, args.seed + epoch * 101, pin_memory,
        )
        set_train_eval_modes(model, args.freeze_until, training=True)
        cm_raf_tr = np.zeros((n_cls, n_cls), dtype=np.int64)
        cm_ext_tr = np.zeros((n_cls, n_cls), dtype=np.int64)
        loss_sum = 0.0

        batch_iter = (
            zip(raf_loader, extra_loader)
            if extra_loader is not None
            else ((b, None) for b in raf_loader)
        )
        for (x_r, y_r), extra_batch in batch_iter:
            if extra_batch is not None:
                x_e, y_e = extra_batch
                images = torch.cat([x_r, x_e], dim=0)
                targets = torch.cat([y_r, y_e], dim=0)
            else:
                images, targets = x_r, y_r
            loss_sum += _train_step(
                model, images, targets, len(y_r), criterion, optimizer,
                scaler, amp_enabled, device, cm_raf_tr, cm_ext_tr,
            )

        raf_tr_m = metrics_from_confusion(cm_raf_tr)
        ext_tr_m = metrics_from_confusion(cm_ext_tr)
        raf_va_m = evaluate_loader(model, raf_val_loader, device)
        ext_va_m = evaluate_loader(model, extra_val_loader, device)
        avg_val = 0.5 * (raf_va_m["macro_f1"] + ext_va_m["macro_f1"])

        print(
            f"Epoch {epoch}/{args.epochs} | loss={loss_sum / max(1, steps):.4f} | "
            f"raf_val_f1={raf_va_m['macro_f1']:.4f} | ravdess_val_f1={ext_va_m['macro_f1']:.4f} | "
            f"avg_val_f1={avg_val:.4f}"
        )

        if avg_val > best_score:
            best_score = avg_val
            best_epoch = epoch
            stale_epochs = 0
            best_details = {
                "raf_val_per_class_f1": raf_va_m["per_class_f1"],
                "ravdess_val_per_class_f1": ext_va_m["per_class_f1"],
                "train_val_gap": {
                    "raf": round(raf_tr_m["macro_f1"] - raf_va_m["macro_f1"], 4),
                    "ravdess": round(ext_tr_m["macro_f1"] - ext_va_m["macro_f1"], 4),
                },
            }
            save_inference_checkpoint(
                model, prep, epoch, avg_val,
                raf_va_m["macro_f1"], ext_va_m["macro_f1"], best_ckpt_path,
            )
        else:
            stale_epochs += 1

        atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "best_details": best_details,
            "raf_test_before": raf_test_before,
            "rng": _capture_rng_state(),
        }, last_ckpt_path)

        if stale_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    best_model, _, _ = load_init_checkpoint(best_ckpt_path)
    best_model = best_model.to(device)
    after_metrics = evaluate_loader(best_model, raf_test_loader, device)
    summary = {
        "best_epoch": best_epoch,
        "avg_val_macro_f1": round(best_score, 4),
        "raf_test_before": raf_test_before,
        "raf_test_after": {
            "accuracy": round(after_metrics["accuracy"], 4),
            "macro_f1": round(after_metrics["macro_f1"], 4),
        },
        "raf_val_per_class_f1": best_details.get("raf_val_per_class_f1", {}),
        "ravdess_val_per_class_f1": best_details.get("ravdess_val_per_class_f1", {}),
        "train_val_gap": best_details.get("train_val_gap", {}),
    }
    write_summary_json(summary, summary_path)
    print(f"Saved checkpoint: {best_ckpt_path} | Summary: {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mixed RAF-DB + RAVDESS ResNet-18 fine-tuning.")
    ap.add_argument("--init", type=Path, required=True, help="Init notebook-format checkpoint.")
    ap.add_argument("--raf-dir", type=Path, default=Path("DATASET"))
    ap.add_argument("--extra-dir", type=Path, default=Path("ravdess_crops"))
    ap.add_argument("--extra-share", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument(
        "--freeze-until",
        choices=("none", "layer1", "layer2", "layer3"),
        default="none",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("external_models") / "ravdess")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", nargs="?", const="__AUTO__", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
