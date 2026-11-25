"""Extract aligned facial expression crops from RAVDESS speech videos.

Reuses YuNet + ``aligned_square_crop`` from ``live_resnet18_demo.py`` so training
crops follow the exact same alignment and quality gates as live webcam inference.
"""

from __future__ import annotations

import argparse
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2

from live_resnet18_demo import aligned_square_crop

CANONICAL_CLASSES = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

# RAVDESS filename emotion field (02 = calm is intentionally dropped).
EMOTION_MAP = {
    "01": "neutral",
    "03": "joy",
    "04": "sadness",
    "05": "anger",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

TRAIN_ACTORS = set(range(1, 9))
VAL_ACTORS = {9, 10}


def parse_ravdess_filename(path_or_name: str | Path) -> tuple[str, Optional[dict[str, Any]]]:
    """Parse a 7-field RAVDESS filename and apply modality/emotion/actor rules.

    Returns ``(status, metadata)`` where ``status == 'keep'`` when the clip is
    valid audio-video speech for actors 01-10 and a mapped emotion.
    """
    stem = Path(path_or_name).stem
    parts = stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() for p in parts):
        warnings.warn(f"Skipping unrecognized RAVDESS filename: {Path(path_or_name).name}")
        return "invalid_name", None

    modality, vocal_channel, emotion, _intensity, _stmt, _rep, actor = parts
    # Keep ONLY modality 01 (audio-video) and vocal_channel 01 (speech); drop modality 02 duplicates.
    if modality != "01" or vocal_channel != "01":
        return "ignored_modality_or_channel", None

    if emotion == "02":
        return "dropped_calm", None

    cls_name = EMOTION_MAP.get(emotion)
    if cls_name is None:
        warnings.warn(f"Skipping unknown emotion code '{emotion}' in {Path(path_or_name).name}")
        return "unknown_emotion", None

    actor_id = int(actor)
    if actor_id in TRAIN_ACTORS:
        split = "train"
    elif actor_id in VAL_ACTORS:
        split = "val"
    else:
        warnings.warn(f"Skipping actor '{actor}' outside 01-10 in {Path(path_or_name).name}")
        return "unknown_actor", None

    return "keep", {
        "emotion": cls_name,
        "split": split,
        "actor": actor_id,
        "stem": stem,
    }


def create_face_detector(yunet_path: Path, score_threshold: float = 0.55) -> Any:
    """Create the OpenCV YuNet face detector shared with webcam_service / live demo."""
    if not yunet_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {yunet_path}")
    return cv2.FaceDetectorYN.create(
        model=str(yunet_path),
        config="",
        input_size=(640, 480),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )


def compute_sample_indices(frame_count: int, fps: float, target_fps: float) -> list[int]:
    """Return frame indices at ~target_fps inside the middle 30%-70% window."""
    if frame_count <= 0:
        return []
    safe_fps = fps if fps and not math.isnan(fps) and fps > 1.0 else 30.0
    start_idx = max(0, int(math.floor(0.30 * frame_count)))
    end_idx = min(frame_count, int(math.ceil(0.70 * frame_count)))
    if end_idx <= start_idx:
        end_idx = min(frame_count, start_idx + 1)
    stride = max(1, int(round(safe_fps / max(0.5, target_fps))))
    return list(range(start_idx, end_idx, stride))


def _passes_quality_gate(
    frame: Any,
    face: Any,
    pad: float,
    min_side: int,
    blur_threshold: float,
    min_conf: float,
) -> Optional[Any]:
    """Run aligned_square_crop and reject small, low-confidence, or blurry faces."""
    fw, fh = float(face[2]), float(face[3])
    conf = float(face[-1])
    if conf < min_conf or fw < min_side or fh < min_side:
        return None
    crop = aligned_square_crop(frame, face, pad=pad, min_side=min_side)
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < blur_threshold:
        return None
    return crop


def extract_clip_crops(
    video_path: Path,
    meta: dict[str, Any],
    out_dir: Path,
    detector: Any,
    target_fps: float,
    pad: float,
    min_side: int,
    blur_threshold: float,
    min_conf: float,
) -> tuple[int, dict[str, int]]:
    """Sample middle frames of one clip, detect/align faces, and write JPG crops."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        warnings.warn(f"Could not open video: {video_path}")
        return 0, {"open_failed": 1}

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = compute_sample_indices(frame_count, fps, target_fps)
        if not sample_indices:
            return 0, {"empty_video": 1}

        target_set = set(sample_indices)
        max_idx = sample_indices[-1]
        dest_dir = out_dir / meta["split"] / meta["emotion"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        rejects: Counter[str] = Counter()
        for idx in range(max_idx + 1):
            if not cap.grab():
                break
            if idx not in target_set:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                rejects["decode_failed"] += 1
                continue
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            if faces is None or len(faces) == 0:
                rejects["no_face"] += 1
                continue
            best_face = max(faces, key=lambda f: float(f[2] * f[3]))
            crop = _passes_quality_gate(
                frame, best_face, pad=pad, min_side=min_side,
                blur_threshold=blur_threshold, min_conf=min_conf,
            )
            if crop is None:
                rejects["quality_rejected"] += 1
                continue

            filename = f"actor{meta['actor']:02d}_{meta['stem']}_f{idx:04d}.jpg"
            cv2.imwrite(str(dest_dir / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        return saved, dict(rejects)
    finally:
        cap.release()


def print_summary_table(
    crop_counts: dict[str, dict[str, int]],
    clips_kept: int,
    clip_skip_reasons: Counter[str],
) -> None:
    """Print one crops-per-class-x-split table and kept/skipped clip totals."""
    print(f"\n{'Class':<12} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 40)
    train_total = 0
    val_total = 0
    for cls_name in CANONICAL_CLASSES:
        tr = crop_counts[cls_name]["train"]
        va = crop_counts[cls_name]["val"]
        train_total += tr
        val_total += va
        print(f"{cls_name:<12} {tr:>8} {va:>8} {tr + va:>8}")
    print("-" * 40)
    print(f"{'TOTAL':<12} {train_total:>8} {val_total:>8} {train_total + val_total:>8}")

    clips_skipped = sum(clip_skip_reasons.values())
    reasons_str = ", ".join(f"{k}={v}" for k, v in sorted(clip_skip_reasons.items()))
    print(f"\nClips kept: {clips_kept} | Clips skipped: {clips_skipped} ({reasons_str})")


def prepare_dataset(
    input_dir: Path,
    out_dir: Path,
    yunet_path: Path,
    target_fps: float = 5.0,
    pad: float = 1.10,
    min_side: int = 64,
    blur_threshold: float = 30.0,
    min_conf: float = 0.50,
) -> dict[str, Any]:
    """Scan RAVDESS videos, extract quality-gated middle crops, and print stats."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    mp4_files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"
    )
    detector = create_face_detector(yunet_path, score_threshold=max(0.55, min_conf))

    crop_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    clip_skip_reasons: Counter[str] = Counter()
    clips_kept = 0

    for video_path in mp4_files:
        status, meta = parse_ravdess_filename(video_path)
        if status != "keep" or meta is None:
            clip_skip_reasons[status] += 1
            continue

        saved, _ = extract_clip_crops(
            video_path=video_path,
            meta=meta,
            out_dir=out_dir,
            detector=detector,
            target_fps=target_fps,
            pad=pad,
            min_side=min_side,
            blur_threshold=blur_threshold,
            min_conf=min_conf,
        )
        if saved > 0:
            clips_kept += 1
            crop_counts[meta["emotion"]][meta["split"]] += saved
        else:
            clip_skip_reasons["no_valid_crops"] += 1

    print_summary_table(crop_counts, clips_kept, clip_skip_reasons)
    return {
        "crop_counts": {k: dict(v) for k, v in crop_counts.items()},
        "clips_kept": clips_kept,
        "clip_skip_reasons": dict(clip_skip_reasons),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare RAVDESS speech face crops.")
    ap.add_argument("--input-dir", type=Path, default=Path("Video speech dataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("ravdess_crops"))
    ap.add_argument(
        "--yunet",
        type=Path,
        default=Path("models") / "face_detection_yunet_2023mar.onnx",
    )
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--pad", type=float, default=1.10)
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--blur-threshold", type=float, default=30.0)
    ap.add_argument("--min-conf", type=float, default=0.50)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        yunet_path=args.yunet,
        target_fps=args.fps,
        pad=args.pad,
        min_side=args.min_side,
        blur_threshold=args.blur_threshold,
        min_conf=args.min_conf,
    )


if __name__ == "__main__":
    main()
