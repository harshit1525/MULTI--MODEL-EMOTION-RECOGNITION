"""Standalone live-webcam test for the RAF-DB ResNet-18 model.

Pipeline: webcam frame -> YuNet face detection -> eye-line alignment ->
square crop -> the checkpoint's own preprocessing -> ResNet-18 -> EMA smoothing.

Run from the "Major project" folder (PowerShell):
    python live_resnet18_demo.py --checkpoint external_models\\resnet18_raf\\best_resnet18.pt

Needs: pip install torch torchvision opencv-python pillow numpy
Keys: q = quit.  Nothing is written to disk.
"""
import argparse
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


def load_model(path):
    """Load the notebook's checkpoint exactly as its own predictor.py does."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["architecture"] != "resnet18":
        raise ValueError("Unsupported checkpoint architecture.")
    names = ck["class_names"]
    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(names))
    net.load_state_dict(ck["model_state_dict"])
    p = ck["preprocessing"]
    interp = getattr(InterpolationMode, str(p.get("interpolation", "bilinear")).upper(), InterpolationMode.BILINEAR)
    prep = transforms.Compose([
        transforms.Resize(
            p["eval_resize"], interpolation=interp,
            antialias=p["antialias"],
        ),
        transforms.CenterCrop(p["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(p["mean"], p["std"]),
    ])
    return net.eval(), prep, names


def aligned_square_crop(frame, face, pad, min_side=64):
    """Level the eye line, then cut a square (aspect preserved) around the face.

    YuNet row: x, y, w, h, right-eye(x,y), left-eye(x,y), nose, mouth corners, score.
    Returns a BGR crop, or None if the face is smaller than min_side.
    """
    x, y, w, h = face[:4]
    rex, rey, lex, ley = face[4:8]
    angle = math.degrees(math.atan2(ley - rey, lex - rex))
    if abs(angle) > 45:  # implausible tilt: skip alignment instead of guessing
        angle = 0.0
    cx, cy = (rex + lex) / 2.0, (rey + ley) / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    height, width = frame.shape[:2]
    rotated = cv2.warpAffine(
        frame, m, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    centre = m @ np.array([x + w / 2.0, y + h / 2.0, 1.0])
    side = int(max(w, h) * pad)
    if side < min_side:
        return None
    x0 = int(round(centre[0] - side / 2.0))
    y0 = int(round(centre[1] - side / 2.0))
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - width), max(0, y0 + side - height)
    if pad_l or pad_t or pad_r or pad_b:
        rotated = cv2.copyMakeBorder(
            rotated, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )
        x0 += pad_l
        y0 += pad_t
    return rotated[y0:y0 + side, x0:x0 + side]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--yunet", default=r"models\face_detection_yunet_2023mar.onnx")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pad", type=float, default=1.10,
                    help="crop side = pad x face box (checkpoint then centre-crops 224/256)")
    ap.add_argument("--alpha", type=float, default=0.4, help="EMA weight of the new frame")
    args = ap.parse_args()

    net, prep, names = load_model(args.checkpoint)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera not available")
    detector = cv2.FaceDetectorYN.create(
        args.yunet, "", (frame.shape[1], frame.shape[0]), 0.7, 0.3, 5000
    )
    smooth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)  # inference uses the RAW (unmirrored) frame
        display = cv2.flip(frame, 1)       # mirror only for display

        label = "no face"
        if faces is not None and len(faces):
            face = max(faces, key=lambda f: f[2] * f[3])
            crop = aligned_square_crop(frame, face, args.pad)
            if crop is not None and crop.size:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                with torch.inference_mode():
                    tensor = prep(Image.fromarray(rgb)).unsqueeze(0)
                    probs = net(tensor).softmax(dim=1)[0].numpy()
                smooth = probs if smooth is None else (
                    args.alpha * probs + (1 - args.alpha) * smooth
                )
                order = np.argsort(smooth)[::-1]
                label = f"{names[order[0]]} {smooth[order[0]] * 100:.0f}%"
                bx, by, bw, bh = (int(v) for v in face[:4])
                bx = width - (bx + bw)  # box in mirrored display coordinates
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
                cv2.putText(display, label, (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                for i, idx in enumerate(order[:3]):
                    cv2.putText(
                        display, f"{names[idx]}: {smooth[idx] * 100:.1f}%",
                        (10, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2,
                    )
            else:
                smooth = None
                label = "face too small"
        else:
            smooth = None

        ms = (time.perf_counter() - t0) * 1000
        cv2.putText(display, f"{ms:.0f} ms", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if label in ("no face", "face too small"):
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        cv2.imshow("RAF-DB ResNet-18 live test (q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
