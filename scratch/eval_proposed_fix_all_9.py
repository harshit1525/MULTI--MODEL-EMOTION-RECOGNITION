import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.webcam_service import webcam_service, EMOTIONS

LABEL_MAP = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutal": "neutral",
    "neutral": "neutral",
    "sadness": "sadness",
    "sad": "sadness",
    "surprise": "surprise",
    "surp": "surprise"
}

def get_gt(stem: str) -> str:
    cleaned = stem.lower()
    for s in [" (2)", " (3)", " (4)", " (5)"]:
        cleaned = cleaned.replace(s, "")
    return LABEL_MAP.get(cleaned.strip(), "unknown")

def test_proposed_fix():
    clips_dir = Path("New folder")
    results = {}
    
    for clip_path in sorted(clips_dir.glob("*.mp4")):
        gt = get_gt(clip_path.stem)
        if gt == "unknown": continue
        
        cap = cv2.VideoCapture(str(clip_path))
        preds = []
        client_id = f"fix_{clip_path.stem}"
        webcam_service.reset_session(client_id)
        client = webcam_service._get_client_state(client_id)
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            h, w = frame.shape[:2]
            webcam_service.face_detector.setInputSize((w, h))
            _, faces = webcam_service.face_detector.detect(frame)
            if faces is None or len(faces) == 0: continue
            
            face = faces[0]
            fx, fy, fw, fh = [int(v) for v in face[:4]]
            pts = face[4:14].reshape((5, 2))
            cx, cy = fx + fw // 2, fy + fh // 2
            
            # Aspect-preserving square crop with 25% padding centered on face
            crop_sz = int(max(fw, fh) * 1.25)
            pad_needed = crop_sz // 2
            padded = cv2.copyMakeBorder(frame, pad_needed, pad_needed, pad_needed, pad_needed, cv2.BORDER_REFLECT)
            pcx, pcy = cx + pad_needed, cy + pad_needed
            face_crop = padded[pcy - crop_sz // 2 : pcy + (crop_sz - crop_sz // 2),
                               pcx - crop_sz // 2 : pcx + (crop_sz - crop_sz // 2)]
            
            face_enhanced = face_crop
            img_sz = webcam_service._img_size
            resized = cv2.resize(face_enhanced, (img_sz, img_sz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = webcam_service.norm(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(webcam_service.device)
            with torch.inference_mode():
                logits = webcam_service.backbone(tensor)[0].cpu().numpy()
            
            adj_logits = logits.copy()
            re, le, nose, rm, lm = pts[0], pts[1], pts[2], pts[3], pts[4]
            eye_dist = max(5.0, float(np.hypot(le[0] - re[0], le[1] - re[1])))
            mouth_center = ((rm[0] + lm[0]) / 2.0, (rm[1] + lm[1]) / 2.0)
            eye_center = ((re[0] + le[0]) / 2.0, (re[1] + le[1]) / 2.0)
            mouth_w = float(np.hypot(lm[0] - rm[0], lm[1] - rm[1])) / eye_dist
            eye_to_mouth = float(np.hypot(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1])) / eye_dist
            nose_to_mouth = float(np.hypot(mouth_center[0] - nose[0], mouth_center[1] - nose[1])) / eye_dist
            
            ref_geom = {'mouth_w': 0.784, 'eye_to_mouth': 1.089, 'nose_to_mouth': 0.480}
            delta_mw = mouth_w - ref_geom['mouth_w']
            delta_eym = eye_to_mouth - ref_geom['eye_to_mouth']
            delta_nmy = nose_to_mouth - ref_geom['nose_to_mouth']
            
            # Principled Geometry Adjustments
            if delta_mw > 0.048 and delta_nmy < -0.025:
                adj_logits[3] += 1.8; adj_logits[0] -= 0.8; adj_logits[4] -= 0.8
            elif delta_eym > 0.06 or delta_nmy > 0.05:
                adj_logits[6] += 2.2; adj_logits[1] -= 1.8; adj_logits[4] -= 0.8
            elif (delta_nmy < -0.008 or delta_eym < -0.012) and delta_mw < 0.040:
                adj_logits[0] += 1.6; adj_logits[3] -= 0.8; adj_logits[4] -= 0.6
            elif abs(delta_mw) < 0.040 and abs(delta_eym) < 0.035 and abs(delta_nmy) < 0.035:
                adj_logits[4] += 1.2; adj_logits[1] -= 1.4; adj_logits[5] -= 1.0; adj_logits[6] -= 1.0
                
            raw_probs = torch.softmax(torch.from_numpy(adj_logits), dim=-1).numpy()
            if client['ema_probs'] is not None:
                alpha = float(np.clip(0.20 + 0.35 * float(np.max(raw_probs)), 0.18, 0.55))
                smoothed = alpha * raw_probs + (1.0 - alpha) * client['ema_probs']
                smoothed /= np.sum(smoothed)
            else:
                smoothed = raw_probs
            client['ema_probs'] = smoothed
            
            log_p = np.log(np.clip(smoothed, 1e-6, 1.0))
            centered = log_p - np.mean(log_p)
            scaled = centered / 0.85
            exp_l = np.exp(scaled - np.max(scaled))
            calib_probs = exp_l / np.sum(exp_l)
            
            top_idx = int(np.argmax(calib_probs))
            top_label = EMOTIONS[top_idx]
            
            if client['current_label'] is None:
                client['current_label'] = top_label
            else:
                curr_label = client['current_label']
                curr_idx = EMOTIONS.index(curr_label)
                if top_label != curr_label and (calib_probs[top_idx] > calib_probs[curr_idx] + 0.08):
                    client['candidate_count'] = client.get('candidate_count', 0) + 1
                    if client['candidate_count'] >= 2:
                        client['current_label'] = top_label
                        client['candidate_count'] = 0
                else:
                    client['candidate_count'] = 0
            preds.append(client['current_label'])
            
        cap.release()
        counts = Counter(preds)
        maj = counts.most_common(1)[0][0] if counts else 'none'
        acc = (counts.get(gt, 0) / len(preds) * 100) if preds else 0.0
        results[clip_path.name] = {
            'gt': gt,
            'maj': maj,
            'correct': (maj == gt),
            'acc': acc,
            'total': len(preds),
            'counts': dict(counts)
        }
    return results

if __name__ == "__main__":
    print("Evaluating proposed fix on all 9 clips...")
    res = test_proposed_fix()
    print("=" * 90)
    for k, v in res.items():
        print(f"  {k:20s} | GT: {v['gt']:8s} | Maj: {v['maj']:8s} | Correct: {str(v['correct']):5s} | Acc: {v['acc']:.1f}% | Counts: {v['counts']}")
    print("=" * 90)
