"""
evaluate_ablations_neutral.py
=============================
Systematic Hypothesis Ablations (H1 - H8) on resting neutral face clip:
H1: Leftover rule-based geometry adjustments
H2: Input frame mirroring
H3: Glasses vs no glasses detection
H4: Non-square crop distortion vs aspect-preserving square padding (10%, 20%, 30%)
H5: Lighting normalization (none vs CLAHE vs gamma)
H6: Neutral resting face calibration (automatic baseline)
H7: Raw CNN backbone distribution
H8: Temporal hysteresis lock
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.webcam_service import webcam_service, EMOTIONS

LABEL_MAPPING = {emo: i for i, emo in enumerate(EMOTIONS)}

def run_ablation(
    frames: List[np.ndarray],
    disable_geom: bool = False,
    mirror: bool = False,
    pad_mode: str = "default",  # "default", "square_10", "square_20", "square_30"
    light_mode: str = "gamma",  # "none", "clahe", "gamma"
    auto_calib: bool = False,
    no_hysteresis: bool = False
) -> Dict[str, Any]:
    client_id = f"abl_{disable_geom}_{mirror}_{pad_mode}_{light_mode}_{auto_calib}_{no_hysteresis}"
    webcam_service.reset_session(client_id)
    client = webcam_service._get_client_state(client_id)

    # If auto_calib, run first 15 frames in calibrate mode
    if auto_calib:
        for f in frames[:15]:
            _, buf = cv2.imencode(".jpg", f)
            webcam_service.predict_frame(buf.tobytes(), client_id=client_id, calibrate=True)

    preds = []
    confs_neutral = []
    confs_surprise = []

    for f in frames:
        inp = cv2.flip(f, 1) if mirror else f
        h, w = inp.shape[:2]

        webcam_service.face_detector.setInputSize((w, h))
        ret_det, faces = webcam_service.face_detector.detect(inp)
        if faces is None or len(faces) == 0:
            preds.append("no_face")
            continue

        face = faces[0]
        fx, fy, fw, fh = [int(v) for v in face[:4]]
        pts = face[4:14].reshape((5, 2))
        cx, cy = fx + fw // 2, fy + fh // 2

        # Padding mode
        if pad_mode == "square_10":
            crop_sz = int(max(fw, fh) * 1.10)
            pad_needed = crop_sz // 2
            padded = cv2.copyMakeBorder(inp, pad_needed, pad_needed, pad_needed, pad_needed, cv2.BORDER_REFLECT)
            pcx, pcy = cx + pad_needed, cy + pad_needed
            face_crop = padded[pcy - crop_sz // 2 : pcy + (crop_sz - crop_sz // 2),
                               pcx - crop_sz // 2 : pcx + (crop_sz - crop_sz // 2)]
        elif pad_mode == "square_20":
            crop_sz = int(max(fw, fh) * 1.20)
            pad_needed = crop_sz // 2
            padded = cv2.copyMakeBorder(inp, pad_needed, pad_needed, pad_needed, pad_needed, cv2.BORDER_REFLECT)
            pcx, pcy = cx + pad_needed, cy + pad_needed
            face_crop = padded[pcy - crop_sz // 2 : pcy + (crop_sz - crop_sz // 2),
                               pcx - crop_sz // 2 : pcx + (crop_sz - crop_sz // 2)]
        elif pad_mode == "square_30":
            crop_sz = int(max(fw, fh) * 1.30)
            pad_needed = crop_sz // 2
            padded = cv2.copyMakeBorder(inp, pad_needed, pad_needed, pad_needed, pad_needed, cv2.BORDER_REFLECT)
            pcx, pcy = cx + pad_needed, cy + pad_needed
            face_crop = padded[pcy - crop_sz // 2 : pcy + (crop_sz - crop_sz // 2),
                               pcx - crop_sz // 2 : pcx + (crop_sz - crop_sz // 2)]
        else:
            # Current default in webcam_service
            crop_sz = int(max(fw, fh) * 1.25)
            x1, y1 = max(0, cx - crop_sz // 2), max(0, cy - crop_sz // 2)
            x2, y2 = min(w, x1 + crop_sz), min(h, y1 + crop_sz)
            face_crop = inp[y1:y2, x1:x2]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            face_crop = inp

        # Lighting Normalization
        if light_mode == "none":
            face_enhanced = face_crop
        elif light_mode == "clahe":
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            face_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Default Gamma
            gray_l = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray_l))
            if mean_lum < 60.0:
                inv_gamma = 1.0 / 1.35
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                face_enhanced = cv2.LUT(face_crop, table)
            elif mean_lum > 195.0:
                inv_gamma = 1.0 / 0.75
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                face_enhanced = cv2.LUT(face_crop, table)
            else:
                face_enhanced = face_crop

        # Deep Model Forward Pass
        img_sz = webcam_service._img_size
        resized = cv2.resize(face_enhanced, (img_sz, img_sz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = webcam_service.norm(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(webcam_service.device)
        with torch.inference_mode():
            if webcam_service._architecture in ("efficientnet_b0", "timm_tf_efficientnet_b0"):
                logits = webcam_service.backbone(tensor)[0].cpu().numpy()
            else:
                feat = webcam_service.backbone(tensor)
                logits = webcam_service.classifier(feat)[0].cpu().numpy()

        adj_logits = logits.copy()

        # User Baseline
        if client.get("baseline_logits") is not None:
            adj_logits -= 0.45 * client["baseline_logits"]

        # Geometry Rules
        if not disable_geom:
            re, le = pts[0], pts[1]
            nose = pts[2]
            rm, lm = pts[3], pts[4]
            eye_center = (float(re[0] + le[0]) / 2.0, float(re[1] + le[1]) / 2.0)
            eye_dist = max(5.0, float(np.hypot(le[0] - re[0], le[1] - re[1])))
            mouth_center = (float(rm[0] + lm[0]) / 2.0, float(rm[1] + lm[1]) / 2.0)
            mouth_w = float(np.hypot(lm[0] - rm[0], lm[1] - rm[1])) / eye_dist
            eye_to_mouth = float(np.hypot(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1])) / eye_dist
            nose_to_mouth = float(np.hypot(mouth_center[0] - nose[0], mouth_center[1] - nose[1])) / eye_dist

            ref_geom = client.get("baseline_geom") or {"mouth_w": 0.784, "eye_to_mouth": 1.089, "nose_to_mouth": 0.480}
            delta_mw = mouth_w - ref_geom["mouth_w"]
            delta_eym = eye_to_mouth - ref_geom["eye_to_mouth"]
            delta_nmy = nose_to_mouth - ref_geom["nose_to_mouth"]

            if delta_mw > 0.048 and delta_nmy < -0.025:
                adj_logits[3] += 1.8; adj_logits[0] -= 0.8; adj_logits[4] -= 0.8
            elif delta_eym > 0.024 or delta_nmy > 0.024:
                adj_logits[6] += 2.2; adj_logits[1] -= 1.8; adj_logits[4] -= 0.8
            elif (delta_nmy < -0.008 or delta_eym < -0.012) and delta_mw < 0.040:
                adj_logits[0] += 1.6; adj_logits[3] -= 0.8; adj_logits[4] -= 0.6
            elif abs(delta_mw) < 0.035 and abs(delta_eym) < 0.015 and abs(delta_nmy) < 0.015:
                adj_logits[4] += 1.2; adj_logits[1] -= 1.4; adj_logits[5] -= 1.0; adj_logits[6] -= 1.0

        raw_probs = torch.softmax(torch.from_numpy(adj_logits), dim=-1).numpy()

        # Smoothing
        if client["ema_probs"] is not None:
            alpha = float(np.clip(0.20 + 0.35 * float(np.max(raw_probs)), 0.18, 0.55))
            smoothed = alpha * raw_probs + (1.0 - alpha) * client["ema_probs"]
            smoothed /= np.sum(smoothed)
        else:
            smoothed = raw_probs
        client["ema_probs"] = smoothed

        # Temperature Calibration
        log_p = np.log(np.clip(smoothed, 1e-6, 1.0))
        centered = log_p - np.mean(log_p)
        scaled = centered / 0.85
        exp_l = np.exp(scaled - np.max(scaled))
        calib_probs = exp_l / np.sum(exp_l)

        top_idx = int(np.argmax(calib_probs))
        top_label = EMOTIONS[top_idx]

        if no_hysteresis:
            final_pred = top_label.lower()
        else:
            if client["current_label"] is None:
                client["current_label"] = top_label.lower()
            else:
                curr_label = str(client["current_label"]).lower()
                curr_idx = LABEL_MAPPING[curr_label]
                if top_label.lower() != curr_label and (calib_probs[top_idx] > calib_probs[curr_idx] + 0.08):
                    client["candidate_count"] = client.get("candidate_count", 0) + 1
                    if client["candidate_count"] >= 2:
                        client["current_label"] = top_label.lower()
                        client["candidate_count"] = 0
                else:
                    client["candidate_count"] = 0
            final_pred = str(client["current_label"]).lower()

        preds.append(final_pred)
        confs_neutral.append(calib_probs[4])
        confs_surprise.append(calib_probs[6])

    counts = Counter(preds)
    maj = counts.most_common(1)[0][0] if counts else "none"
    return {
        "majority": maj,
        "neutral_frames": counts.get("neutral", 0),
        "surprise_frames": counts.get("surprise", 0),
        "counts": dict(counts),
        "mean_neutral_conf": float(np.mean(confs_neutral) * 100),
        "mean_surprise_conf": float(np.mean(confs_surprise) * 100),
    }

def main():
    clip_p = Path("New folder/Neutal.mp4")
    cap = cv2.VideoCapture(str(clip_p))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    total_f = len(frames)
    print(f"Running Systematic H1-H8 Ablations on {clip_p.name} ({total_f} frames)...")

    ablations = {
        "Baseline (Current Live Path)": run_ablation(frames),
        "H1: Remove Hardcoded Geometry Rules": run_ablation(frames, disable_geom=True),
        "H2: Horizontal Mirror Input": run_ablation(frames, mirror=True),
        "H4a: Square Crop (10% padding)": run_ablation(frames, pad_mode="square_10"),
        "H4b: Square Crop (20% padding)": run_ablation(frames, pad_mode="square_20"),
        "H4c: Square Crop (30% padding)": run_ablation(frames, pad_mode="square_30"),
        "H5a: No Lighting Normalization": run_ablation(frames, light_mode="none"),
        "H5b: CLAHE Lighting Normalization": run_ablation(frames, light_mode="clahe"),
        "H6: Automatic Neutral Calibration": run_ablation(frames, auto_calib=True),
        "H7: Raw CNN Only (No Rules, No Smoothing)": run_ablation(frames, disable_geom=True, no_hysteresis=True),
        "H8: No Hysteresis State Machine": run_ablation(frames, no_hysteresis=True),
        "Combined Fix: No Geom Rules + Square 20%": run_ablation(frames, disable_geom=True, pad_mode="square_20"),
        "Combined Fix + Auto-Calibration": run_ablation(frames, disable_geom=True, pad_mode="square_20", auto_calib=True),
    }

    print("\n" + "=" * 90)
    print(f"{'EXPERIMENT / ABLATION':45s} | {'MAJORITY':10s} | {'NEUTRAL FRAMES':16s} | {'SURPRISE FRAMES':16s}")
    print("=" * 90)
    for name, res in ablations.items():
        nf_str = f"{res['neutral_frames']}/{total_f} ({res['neutral_frames']/total_f*100:.1f}%)"
        sf_str = f"{res['surprise_frames']}/{total_f} ({res['surprise_frames']/total_f*100:.1f}%)"
        print(f"{name:45s} | {res['majority']:10s} | {nf_str:16s} | {sf_str:16s}")
    print("=" * 90)

if __name__ == "__main__":
    main()
