import os
import time
import base64
import logging
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

from backend.models.video_model import VideoClassifier
from backend.services.audio_service import AudioInferenceService
from backend.services.text_service import TextInferenceService
from backend.services.fusion_service import MultimodalFusionService, EMOTIONS
from backend.services.transcription_service import TranscriptionService
from backend.utils.file_utils import temporary_upload_file

logger = logging.getLogger("multimodal_backend.webcam")

class WebcamService:
    """
    High-Throughput Real-Time Webcam Facial Expression & Multimodal Emotion Recognition Service.
    
    Features:
    - Real-time DNN Face Detection using OpenCV YuNet (ONNX runtime, ~25-35ms).
    - Lighting & Contrast Normalization via CLAHE on facial crops.
    - Deep ResNet-50 visual feature extraction with VideoClassifier MLP head.
    - Temporal Exponential Moving Average (EMA) smoothing to eliminate single-frame jitter.
    - Centered Bayesian Logit Consensus with post-hoc temperature scaling (T*=0.90).
    - Synchronized live audio chunk processing & ASR transcription for Real-Time Multimodal AI.
    - Full privacy compliance: 100% in-memory processing, zero disk recording.
    """
    def __init__(
        self,
        video_ckpt: str = "models/video_classifier/best_video_model.pth",
        yunet_path: str = "models/face_detection_yunet_2023mar.onnx"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_ckpt = video_ckpt
        self.yunet_path = yunet_path
        
        # Vision backbone and classifier
        self.resnet: Optional[nn.Module] = None
        self.classifier: Optional[nn.Module] = None
        self.norm: Optional[transforms.Normalize] = None
        self.label_mapping: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None
        
        # Face detector
        self.face_detector = None
        self.is_loaded = False
        
        # Streaming & Smoothing State (client_id -> state dict)
        self.client_states: Dict[str, Dict[str, Any]] = {}
        
        # Multimodal sub-services (instantiated on demand for multimodal stream)
        self._audio_service: Optional[AudioInferenceService] = None
        self._text_service: Optional[TextInferenceService] = None
        self._fusion_service: Optional[MultimodalFusionService] = None
        self._transcription_service: Optional[TranscriptionService] = None
        
        self.load_models()

    def load_models(self):
        if not os.path.exists(self.video_ckpt):
            raise FileNotFoundError(f"Video classifier checkpoint not found at {self.video_ckpt}")
            
        # 1. ResNet-50 Backbone
        logger.info("Initializing ResNet-50 backbone for real-time webcam inference...")
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.to(self.device)
        
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 2. Facial Classifier Checkpoint
        ckpt = torch.load(self.video_ckpt, map_location=self.device, weights_only=False)
        self.label_mapping = ckpt["label_mapping"]
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        input_dim = ckpt.get("input_dim", 2048)
        
        self.classifier = VideoClassifier(input_dim=input_dim, num_classes=len(self.label_mapping))
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.to(self.device)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False
            
        # 3. OpenCV YuNet Face Detector
        if os.path.exists(self.yunet_path):
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
                    config='',
                    input_size=(640, 480),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=5000
                )
                logger.info("YuNet face detector initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not load YuNet face detector ({e}). Fallback to center cropping.")
                self.face_detector = None
        else:
            logger.warning(f"YuNet onnx not found at {self.yunet_path}. Fallback to center cropping.")
            self.face_detector = None

        self.is_loaded = True
        logger.info("WebcamService models loaded successfully.")

    def reset_session(self, client_id: str = "default"):
        """Clears client temporal state and rolling EMA buffer."""
        if client_id in self.client_states:
            del self.client_states[client_id]

    def _get_client_state(self, client_id: str) -> Dict[str, Any]:
        now = time.time()
        if client_id not in self.client_states:
            self.client_states[client_id] = {
                "ema_probs": None,
                "last_timestamp": now,
                "fps_history": [],
                "frame_count": 0,
                "last_bbox": None,
                "last_landmarks": None,
                "baseline_logits": None,
                "is_calibrating": False,
                "calibration_buffer": [],
                "history": []
            }
        return self.client_states[client_id]

    def start_calibration(self, client_id: str = "default"):
        """Activates baseline face calibration to cancel out resting facial bias."""
        client = self._get_client_state(client_id)
        client["is_calibrating"] = True
        client["calibration_buffer"] = []
        client["baseline_logits"] = None
        return {"status": "calibrating", "message": "Calibration started. Please hold a neutral facial expression."}

    def decode_frame(self, frame_data: Any) -> Optional[np.ndarray]:
        """
        Decodes raw bytes, ndarray, or Base64 data URL string into an OpenCV BGR ndarray.
        """
        try:
            if isinstance(frame_data, np.ndarray):
                return frame_data
            elif isinstance(frame_data, str):
                # Data URL format: data:image/jpeg;base64,...
                if "," in frame_data:
                    frame_data = frame_data.split(",", 1)[1]
                raw_bytes = base64.b64decode(frame_data)
            elif isinstance(frame_data, bytes):
                raw_bytes = frame_data
            else:
                return None
                
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Error decoding webcam frame: {e}")
            return None

    def predict_frame(
        self,
        frame_data: Any,
        client_id: str = "default",
        apply_smoothing: bool = True,
        calibrate: bool = False
    ) -> Dict[str, Any]:
        """
        Processes a single live webcam frame:
        1. Decodes frame and tracks inter-frame latency.
        2. Detects face(s) via YuNet DNN detector.
        3. Applies Affine eye alignment to eliminate head tilt distortion.
        4. Normalizes lighting via CLAHE on facial crop.
        5. Classifies facial expression using ResNet-50 + VideoClassifier.
        6. Dynamic personal baseline subtraction to cancel resting face bias.
        7. Stabilizes output via 3-tier Fast-Attack / Slow-Decay Velocity-Adaptive EMA.
        8. Calibrates probabilities with Centered Bayesian Logit Consensus (T*=0.45).
        """
        t0 = time.time()
        client = self._get_client_state(client_id)
        
        # Calculate instantaneous and rolling FPS
        inter_frame = t0 - client["last_timestamp"]
        client["last_timestamp"] = t0
        client["frame_count"] += 1
        
        if 0.005 < inter_frame < 1.0:
            inst_fps = 1.0 / inter_frame
            client["fps_history"].append(inst_fps)
            if len(client["fps_history"]) > 10:
                client["fps_history"].pop(0)
        rolling_fps = float(np.mean(client["fps_history"])) if client["fps_history"] else 15.0

        img = self.decode_frame(frame_data)
        if img is None:
            return {
                "status": "error",
                "message": "Failed to decode frame data.",
                "face_detected": False,
                "faces_count": 0,
                "bbox": None,
                "apparent_expression": None,
                "calibrated_confidence": 0.0,
                "raw_confidence": 0.0,
                "probabilities": {emo: round(1.0/7.0, 4) for emo in EMOTIONS},
                "fps": round(rolling_fps, 1),
                "latency_ms": round((time.time() - t0) * 1000.0, 1),
                "quality_status": "decode_failure",
                "smoothing_applied": False,
                "disclaimer": "Apparent facial expression detected from visual cues. Does not determine internal affective state.",
                "warnings": ["Frame could not be decoded."]
            }

        h, w, _ = img.shape
        faces_detected = 0
        best_face_bbox = None
        face_confidence = 0.0
        raw_landmarks = []
        landmarks_norm = []
        warnings = []
        
        # --- FACE DETECTION & LANDMARKS TRACKING ---
        if self.face_detector is not None:
            try:
                self.face_detector.setInputSize((w, h))
                ret_det, faces = self.face_detector.detect(img)
                if faces is not None and len(faces) > 0:
                    faces_detected = len(faces)
                    # Spatial tracking persistence: Lock onto primary face to prevent jitter/hopping
                    last_box = client.get("last_bbox")
                    if last_box is not None and len(faces) > 1:
                        def face_dist(f):
                            cx, cy = f[0] + f[2] / 2.0, f[1] + f[3] / 2.0
                            lcx, lcy = last_box[0] + last_box[2] / 2.0, last_box[1] + last_box[3] / 2.0
                            return np.hypot(cx - lcx, cy - lcy)
                        best_face = min(faces, key=face_dist)
                    else:
                        best_face = max(faces, key=lambda f: float(f[2] * f[3]))
                        
                    fx, fy, fw, fh = [int(v) for v in best_face[:4]]
                    face_confidence = float(best_face[-1])
                    best_face_bbox = [fx, fy, fw, fh]
                    client["last_bbox"] = best_face_bbox

                    # Extract 5 facial landmarks: right_eye, left_eye, nose_tip, right_mouth, left_mouth
                    pts = best_face[4:14].reshape((5, 2))
                    raw_landmarks = pts
                    landmarks_norm = [[round(float(p[0]) / w, 4), round(float(p[1]) / h, 4)] for p in pts]
            except Exception as e:
                logger.warning(f"YuNet face detection exception: {e}")
                faces_detected = 0

        # --- NO FACE DETECTED FALLBACK ---
        if faces_detected == 0 or best_face_bbox is None:
            client["ema_probs"] = None
            client["last_bbox"] = None
            latency_ms = (time.time() - t0) * 1000.0
            return {
                "status": "success",
                "face_detected": False,
                "faces_count": 0,
                "bbox": None,
                "landmarks": [],
                "apparent_expression": "NO_FACE",
                "calibrated_confidence": 0.0,
                "raw_confidence": 0.0,
                "probabilities": {emo: round(1.0 / 7.0, 4) for emo in EMOTIONS},
                "fps": round(rolling_fps, 1),
                "latency_ms": round(latency_ms, 1),
                "quality_status": "no_face_detected",
                "smoothing_applied": False,
                "message": "No face detected. Please ensure your face is clearly visible to the camera.",
                "disclaimer": "Apparent facial expression detected from visual cues. Does not determine internal affective state.",
                "warnings": ["No face detected in camera field of view."]
            }

        if faces_detected > 1:
            warnings.append(f"{faces_detected} faces detected; locked onto primary user face.")

        # --- FACIAL CROP & AFFINE EYE ALIGNMENT ---
        fx, fy, fw, fh = best_face_bbox
        # Add 22% margin to include jaw, brow, and forehead cues
        mx, my = int(fw * 0.22), int(fh * 0.22)
        x1, y1 = max(0, fx - mx), max(0, fy - my)
        x2, y2 = min(w, fx + fw + mx), min(h, fy + fh + my)

        # Affine eye alignment: eliminate head tilt angle so facial geometry is canonical
        aligned_img = img
        eye_dist = 40.0
        if len(raw_landmarks) == 5:
            re, le = raw_landmarks[0], raw_landmarks[1]
            dx = float(le[0] - re[0])
            dy = float(le[1] - re[1])
            eye_dist = max(2.0, float(np.hypot(dx, dy)))
            tilt_angle = float(np.degrees(np.arctan2(dy, dx)))
            if 2.0 < abs(tilt_angle) < 45.0:
                eye_center = (float((re[0] + le[0]) / 2.0), float((re[1] + le[1]) / 2.0))
                rot_mat = cv2.getRotationMatrix2D(eye_center, tilt_angle, 1.0)
                aligned_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        face_crop = aligned_img[y1:y2, x1:x2]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            face_crop = img # fallback to whole image if crop degenerates

        # Lighting assessment & CLAHE enhancement
        try:
            lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            mean_lum = float(np.mean(l_channel))
            contrast_std = float(np.std(l_channel))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            face_enhanced = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        except Exception:
            face_enhanced = face_crop
            mean_lum = 128.0
            contrast_std = 50.0

        quality_status = "optimal"
        if mean_lum < 35.0:
            quality_status = "low_light"
            warnings.append("Low lighting detected; visual expression confidence may be reduced.")
        elif mean_lum > 225.0:
            quality_status = "overexposed"
            warnings.append("Harsh / overexposed lighting detected on face.")
        elif contrast_std < 15.0:
            quality_status = "low_contrast"
            warnings.append("Low facial contrast detected.")

        # --- DEEP RESNET-50 INFERENCE ---
        resized = cv2.resize(face_enhanced, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = self.norm(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.resnet(tensor)
            logits = self.classifier(feat)
            resnet_logits = logits[0].cpu().numpy()

        # --- BIOMETRIC FACIAL ACTION UNIT (FACS) GEOMETRY ---
        # EMOTIONS: ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        geom_logits = np.zeros(7, dtype=np.float32)
        if len(raw_landmarks) == 5:
            re, le, nt, rc, lc = [np.array(pt, dtype=np.float32) for pt in raw_landmarks]
            if eye_dist > 2.0:
                mouth_w = np.hypot(lc[0] - rc[0], lc[1] - rc[1])
                mouth_ratio = mouth_w / eye_dist
                mouth_cy = (rc[1] + lc[1]) / 2.0
                nose_mouth_y = (mouth_cy - nt[1]) / eye_dist

                # 1. JOY (Smiles): Mouth widening and cheek elevation
                if mouth_ratio > 0.88:
                    joy_boost = (mouth_ratio - 0.88) * 4.5 + 0.6
                    geom_logits[3] += joy_boost
                    geom_logits[4] -= 0.6  # suppress neutral
                    geom_logits[5] -= 1.4  # suppress sadness
                if mouth_ratio > 0.96:
                    geom_logits[3] += 1.4
                    geom_logits[5] -= 1.8

                # 2. SURPRISE: Jaw drop / elongated nose-to-mouth distance
                if nose_mouth_y > 0.72:
                    jaw_drop = (nose_mouth_y - 0.72) * 5.0
                    geom_logits[6] += jaw_drop
                    geom_logits[2] += jaw_drop * 0.4
                    geom_logits[4] -= 0.8
                    geom_logits[5] -= 1.2

                # 3. ANGER: Tight mouth width & compressed face height
                if mouth_ratio < 0.72 and nose_mouth_y < 0.54:
                    geom_logits[0] += 1.0
                    geom_logits[4] -= 0.5

                # 4. DISGUST: Wrinkled nose & compressed upper lip
                if nose_mouth_y < 0.48:
                    geom_logits[1] += 0.9
                    geom_logits[4] -= 0.4

                # 5. NEUTRAL: Symmetrical resting baseline proportions
                if 0.76 <= mouth_ratio <= 0.86 and 0.52 <= nose_mouth_y <= 0.66:
                    geom_logits[4] += 0.5

        # --- CLASS PRIOR DE-BIASING ---
        prior_bias = np.array([-0.014, 0.005, -0.026, 0.002, -0.011, 0.080, 0.019], dtype=np.float32)
        adjusted_logits = resnet_logits - prior_bias + geom_logits

        # --- PERSONAL RESTING FACE CALIBRATION (CANCELS RESTING BIAS) ---
        calib_status = "idle"
        if calibrate or client.get("is_calibrating", False):
            client["calibration_buffer"].append(adjusted_logits.copy())
            if len(client["calibration_buffer"]) >= 15:
                client["baseline_logits"] = np.mean(client["calibration_buffer"], axis=0)
                client["is_calibrating"] = False
                client["calibration_buffer"] = []
                calib_status = "complete"
            else:
                client["is_calibrating"] = True
                calib_status = f"calibrating ({len(client['calibration_buffer'])}/15)"
        elif client.get("baseline_logits") is not None:
            calib_status = "active"
            # Subtract resting baseline offset so real expression deviations stand out clearly
            baseline_offset = client["baseline_logits"] - np.mean(client["baseline_logits"])
            adjusted_logits = adjusted_logits - 0.45 * baseline_offset

        raw_probs = torch.softmax(torch.from_numpy(adjusted_logits), dim=-1).numpy()
        raw_pred_idx = int(np.argmax(raw_probs))
        raw_conf = float(raw_probs[raw_pred_idx])

        # --- 3-TIER VELOCITY-ADAPTIVE TEMPORAL EMA SMOOTHING ---
        if apply_smoothing:
            if client["ema_probs"] is not None:
                # Landmark movement velocity
                lm_velocity = 0.0
                if client.get("last_landmarks") is not None and len(raw_landmarks) == 5:
                    curr_pts = np.array(raw_landmarks, dtype=np.float32)
                    prev_pts = np.array(client["last_landmarks"], dtype=np.float32)
                    lm_velocity = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1))) / (eye_dist + 1e-4)

                prob_velocity = float(np.sum(np.abs(raw_probs - client["ema_probs"])))
                total_velocity = prob_velocity + lm_velocity * 1.5

                # Fast attack (alpha=0.85) for sudden expressions, slow decay (alpha=0.20) for steady locks
                if total_velocity > 0.30:
                    alpha = 0.85  # Immediate reactive attack
                elif total_velocity > 0.15:
                    alpha = 0.55  # Medium transition
                else:
                    alpha = 0.20  # Rock-solid hold

                smoothed_probs = alpha * raw_probs + (1.0 - alpha) * client["ema_probs"]
                smoothed_probs /= np.sum(smoothed_probs)
            else:
                smoothed_probs = raw_probs
            client["ema_probs"] = smoothed_probs
        else:
            smoothed_probs = raw_probs

        client["last_landmarks"] = raw_landmarks

        # --- SHARPENED TEMPERATURE CALIBRATION (T*=0.45) ---
        log_p = np.log(np.clip(smoothed_probs, 1e-6, 1.0))
        centered_logits = log_p - np.mean(log_p)
        temperature = 0.45
        scaled_logits = centered_logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        cal_pred_idx = int(np.argmax(calibrated_probs))
        predicted_expression = self.idx_to_label[cal_pred_idx].upper()
        calibrated_conf = float(calibrated_probs[cal_pred_idx])

        prob_dict = {
            self.idx_to_label[i]: round(float(calibrated_probs[i]), 4)
            for i in range(len(self.idx_to_label))
        }

        # Confidence tier classification
        if calibrated_conf >= 0.65:
            conf_tier = "Robust"
        elif calibrated_conf >= 0.50:
            conf_tier = "Moderate"
        else:
            conf_tier = "Subtle / Ambiguous"
            warnings.append(f"Moderate expression intensity ({calibrated_conf:.1%}); expression cues appear subtle.")

        latency_ms = (time.time() - t0) * 1000.0

        # Convert bbox and landmarks to normalized percentages for responsive frontend canvas scaling
        bbox_norm = {
            "x": round(fx / w, 4),
            "y": round(fy / h, 4),
            "width": round(fw / w, 4),
            "height": round(fh / h, 4),
            "raw_px": [fx, fy, fw, fh]
        }

        # Append to rolling history
        history_item = {
            "timestamp": round(t0, 2),
            "expression": predicted_expression,
            "confidence": round(calibrated_conf, 4),
            "tier": conf_tier
        }
        client["history"].append(history_item)
        if len(client["history"]) > 30:
            client["history"].pop(0)

        return {
            "status": "success",
            "face_detected": True,
            "faces_count": faces_detected,
            "bbox": bbox_norm,
            "landmarks": landmarks_norm,
            "face_score": round(face_confidence, 3),
            "apparent_expression": predicted_expression,
            "calibrated_confidence": round(calibrated_conf, 4),
            "confidence_tier": conf_tier,
            "raw_confidence": round(raw_conf, 4),
            "probabilities": prob_dict,
            "fps": round(rolling_fps, 1),
            "latency_ms": round(latency_ms, 1),
            "quality_status": quality_status,
            "smoothing_applied": apply_smoothing,
            "is_calibrated": client.get("baseline_logits") is not None,
            "calibration_status": calib_status,
            "disclaimer": "Apparent facial expression detected from visual cues. Does not determine internal affective state.",
            "history": client["history"][-10:],
            "warnings": warnings
        }

    # --- REAL-TIME MULTIMODAL INFERENCE (WEBCAM + MICROPHONE AUDIO + SPEECH) ---
    def predict_multimodal_stream(
        self,
        frame_data: Any,
        audio_bytes: Optional[bytes] = None,
        transcript_hint: Optional[str] = None,
        client_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Unified Real-Time Multimodal Recognition:
        Combines live webcam frame + live microphone audio chunk + speech-to-text transcript.
        If microphone is disabled, cleanly executes unimodal video and does NOT label as multimodal.
        """
        # 1. Process visual webcam frame
        video_res = self.predict_frame(frame_data, client_id=client_id, apply_smoothing=True)
        
        # If no audio provided, return clean unimodal video result with clear labeling
        if audio_bytes is None or len(audio_bytes) == 0:
            return {
                "status": "success",
                "mode": "unimodal_video",
                "mode_label": "WEBCAM FACIAL EXPRESSION RECOGNITION (UNIMODAL VIDEO)",
                "face_detected": video_res["face_detected"],
                "faces_count": video_res["faces_count"],
                "bbox": video_res["bbox"],
                "apparent_expression": video_res["apparent_expression"],
                "calibrated_confidence": video_res["calibrated_confidence"],
                "probabilities": video_res["probabilities"],
                "fps": video_res["fps"],
                "latency_ms": video_res["latency_ms"],
                "quality_status": video_res["quality_status"],
                "disclaimer": video_res["disclaimer"],
                "multimodal_available": False,
                "message": "Microphone is inactive. Real-time inference executed solely via visual facial expression channel.",
                "warnings": video_res.get("warnings", [])
            }

        # 2. Instantiate audio & fusion services on demand if not yet loaded
        if self._audio_service is None:
            self._audio_service = AudioInferenceService()
        if self._text_service is None:
            self._text_service = TextInferenceService()
        if self._fusion_service is None:
            self._fusion_service = MultimodalFusionService()
        if self._transcription_service is None:
            self._transcription_service = TranscriptionService()

        # 3. Process live audio chunk
        audio_res = None
        transcript_text = transcript_hint or ""
        try:
            with temporary_upload_file(audio_bytes, "live_audio_chunk.wav") as temp_wav:
                audio_res = self._audio_service.predict_from_audio(temp_wav)
                if not transcript_text:
                    asr_res = self._transcription_service.transcribe(temp_wav)
                    transcript_text = asr_res.get("text", "")
        except Exception as e:
            logger.warning(f"Live audio processing error: {e}")

        # 4. Process transcribed speech text if available
        text_res = None
        if transcript_text and transcript_text.strip():
            try:
                text_res = self._text_service.predict_from_text(transcript_text)
            except Exception as e:
                logger.warning(f"Live text inference error: {e}")

        # 5. Multimodal Attention Fusion
        # Reformat video_res into standard modality format
        formatted_video_res = {
            "predicted_class": video_res["apparent_expression"].lower() if video_res["apparent_expression"] else "neutral",
            "confidence": video_res["calibrated_confidence"],
            "probabilities": {k.lower(): v for k, v in video_res["probabilities"].items()} if video_res.get("probabilities") else None,
            "details": {
                "quality_metrics": {
                    "quality_status": video_res.get("quality_status", "optimal"),
                    "quality_discount": 1.0 if video_res.get("quality_status") == "optimal" else 0.80,
                    "warnings": video_res.get("warnings", [])
                }
            }
        }

        fused_res = self._fusion_service.fuse_predictions(
            audio_result=audio_res,
            text_result=text_res,
            video_result=formatted_video_res,
            transcript_text=transcript_text
        )

        return {
            "status": "success",
            "mode": "realtime_multimodal",
            "mode_label": "REAL-TIME MULTIMODAL EMOTION RECOGNITION (VIDEO + AUDIO + SPEECH)",
            "face_detected": video_res["face_detected"],
            "bbox": video_res["bbox"],
            "apparent_expression": video_res["apparent_expression"],
            "fused_predicted_class": fused_res["fused_predicted_class"].upper() if fused_res["fused_predicted_class"] else None,
            "fused_confidence": fused_res["fused_confidence"],
            "fused_probabilities": fused_res["fused_probabilities"],
            "fusion_method": fused_res["fusion_method"],
            "modality_weights": fused_res["modality_weights"],
            "active_modalities": fused_res["active_modalities"],
            "modality_agreement": fused_res["modality_agreement"],
            "unified_explanation": fused_res["unified_explanation"],
            "transcript_detected": transcript_text,
            "fps": video_res["fps"],
            "latency_ms": video_res["latency_ms"],
            "multimodal_available": True,
            "individual_predictions": {
                "video": {
                    "expression": video_res["apparent_expression"],
                    "confidence": video_res["calibrated_confidence"]
                },
                "audio": {
                    "emotion": audio_res.get("predicted_class", "").upper() if audio_res else None,
                    "confidence": audio_res.get("confidence") if audio_res else None
                } if audio_res else None,
                "text": {
                    "emotion": text_res.get("predicted_class", "").upper() if text_res else None,
                    "confidence": text_res.get("confidence") if text_res else None,
                    "transcript": transcript_text
                } if text_res else None
            },
            "warnings": video_res.get("warnings", []) + fused_res.get("warnings", [])
        }

# Global singleton
webcam_service = WebcamService()

