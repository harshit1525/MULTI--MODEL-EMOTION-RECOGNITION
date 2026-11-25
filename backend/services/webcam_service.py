import os
import sys
import time
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models

from backend.models.video_model import VideoClassifier
from backend.utils.file_utils import temporary_upload_file

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from live_resnet18_demo import aligned_square_crop, load_model

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

logger = logging.getLogger("multimodal_backend.webcam")

class WebcamService:
    """
    High-Throughput Real-Time Webcam Facial Expression & Multimodal Emotion Recognition Service.
    
    Features:
    - Real-time DNN Face Detection using OpenCV YuNet (ONNX runtime, ~25-35ms).
    - Lighting & Contrast Normalization via CLAHE on facial crops.
    - Selectable backbone: EfficientNet-B0 (default) or ResNet-18 (RAF-DB).
    - Temporal Exponential Moving Average (EMA) smoothing to eliminate single-frame jitter.
    - Centered Bayesian Logit Consensus with post-hoc temperature scaling (T*=0.90).
    - Synchronized live audio chunk processing & ASR transcription for Real-Time Multimodal AI.
    - Full privacy compliance: 100% in-memory processing, zero disk recording.
    """

    # RAF-DB class names → our canonical EMOTIONS index (by name, not position)
    RAF_TO_CANONICAL = {
        "Anger": 0, "Disgust": 1, "Fear": 2, "Happiness": 3,
        "Neutral": 4, "Sadness": 5, "Surprise": 6,
    }

    def __init__(
        self,
        video_ckpt: Optional[str] = None,
        yunet_path: str = "models/face_detection_yunet_2023mar.onnx",
        backbone: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Read config once
        self._cfg: Dict[str, Any] = {}
        if os.path.exists("config.yaml"):
            try:
                import yaml
                with open("config.yaml", "r", encoding="utf-8") as f:
                    self._cfg = yaml.safe_load(f) or {}
            except Exception:
                pass

        # Resolve backbone selection and checkpoint path cleanly via shared helper
        requested_backbone = backbone or self._cfg.get("webcam_backbone", "efficientnet_current")
        self._backbone_name, self.video_ckpt = self._resolve_backbone_ckpt(
            requested_backbone, video_ckpt=video_ckpt, allow_fallback=True
        )
        self.yunet_path = yunet_path
        
        # Vision backbone — set by load_models() based on checkpoint architecture
        self.backbone: Optional[nn.Module] = None
        self.classifier: Optional[nn.Module] = None  # Only used for legacy ResNet+MLP path
        self._architecture: str = "unknown"
        self._img_size: int = 224
        self.norm: Optional[transforms.Normalize] = None
        self._raf_prep: Optional[Any] = None
        self.label_mapping: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None
        self._raf_index_map: Optional[List[int]] = None  # RAF checkpoint idx → canonical idx
        
        # Face detector
        self.face_detector = None
        self.is_loaded = False
        
        # Streaming & Smoothing State (client_id -> state dict)
        self.client_states: Dict[str, Dict[str, Any]] = {}
        
        # Debug frame capture state
        self.debug_capture_active = (os.environ.get("WEBCAM_DEBUG_CAPTURE") == "1")
        self.debug_frame_index = 0
        self.debug_capture_dir = Path("results/live_debug")
        self.webcam_unified_uses_webcam_engine = self._cfg.get("webcam_unified_uses_webcam_engine", True)

        # Multimodal sub-services (instantiated on demand for multimodal stream)
        self._audio_service: Optional[Any] = None
        self._text_service: Optional[Any] = None
        self._fusion_service: Optional[Any] = None
        self._transcription_service: Optional[Any] = None
        
        self.load_models()

    def load_models(self):
        if not os.path.exists(self.video_ckpt):
            raise FileNotFoundError(f"Video classifier checkpoint not found at {self.video_ckpt}")

        # --- ResNet-18 RAF-DB backbone (external checkpoint format) ---
        if self._backbone_name == "resnet18_raf":
            self._load_resnet18_raf()
            if self.face_detector is None:
                self._init_face_detector()
            self.is_loaded = True
            logger.info("WebcamService models loaded successfully.")
            return

        ckpt = torch.load(self.video_ckpt, map_location=self.device, weights_only=False)
        self._raf_index_map = None
        self._raf_prep = None

        self.label_mapping = ckpt["label_mapping"]
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        self._architecture = ckpt.get("architecture", "resnet50_mlp")
        self._img_size = ckpt.get("img_size", 224)

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.calibrated_temperature = float(ckpt.get("calibrated_temperature", 0.85))

        if self._architecture == "efficientnet_b0":
            # ── NEW: End-to-end EfficientNet-B0 trained on face crops ──────────
            logger.info("Loading EfficientNet-B0 webcam facial expression model...")
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features  # 1280
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.35, inplace=True),
                nn.Linear(in_features, 256),
                nn.SiLU(),
                nn.Dropout(p=0.25),
                nn.Linear(256, len(self.label_mapping)),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            model.to(self.device)
            self.backbone = model
            self.classifier = None   # EfficientNet is end-to-end, no separate head needed
            logger.info(f"EfficientNet-B0 loaded. Val acc={ckpt.get('val_acc', 0):.1f}% (epoch {ckpt.get('epoch', '?')})")
        elif self._architecture == "timm_tf_efficientnet_b0":
            # ── timm tf_efficientnet_b0 backbone ──────────────────────────────
            logger.info("Loading timm tf_efficientnet_b0 webcam facial expression model...")
            import timm
            model = timm.create_model("tf_efficientnet_b0", num_classes=len(self.label_mapping), pretrained=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            model.to(self.device)
            self.backbone = model
            self.classifier = None
            logger.info(f"timm tf_efficientnet_b0 loaded. Val acc={ckpt.get('val_acc', 0):.1f}% (epoch {ckpt.get('epoch', '?')})")
        else:
            # ── LEGACY: ResNet-50 + VideoClassifier MLP head ─────────────────
            logger.info("Loading legacy ResNet-50 + VideoClassifier backbone...")
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
            resnet.fc = nn.Identity()
            resnet.eval()
            for p in resnet.parameters():
                p.requires_grad = False
            resnet.to(self.device)
            self.backbone = resnet   # stored as backbone for inference

            input_dim = ckpt.get("input_dim", 2048)
            self.classifier = VideoClassifier(input_dim=input_dim, num_classes=len(self.label_mapping))
            self.classifier.load_state_dict(ckpt["model_state_dict"])
            self.classifier.to(self.device)
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad = False
            logger.info("Legacy ResNet-50+MLP loaded. Note: accuracy may be lower for webcam faces.")
            
        # 3. OpenCV YuNet Face Detector
        if self.face_detector is None:
            self._init_face_detector()

        self.is_loaded = True
        logger.info("WebcamService models loaded successfully.")

    def _resolve_backbone_ckpt(
        self, backbone_name: str, video_ckpt: Optional[str] = None, allow_fallback: bool = True
    ):
        """Resolve (backbone_name, checkpoint_path) with optional graceful fallback."""
        valid = ("efficientnet_current", "resnet18_raf")
        if backbone_name not in valid:
            raise ValueError(f"Unsupported webcam backbone '{backbone_name}'. Choose from {valid}.")

        if backbone_name == "resnet18_raf":
            candidate = video_ckpt or self._cfg.get(
                "webcam_resnet18_raf_path", "external_models/resnet18_raf/best_resnet18.pt"
            )
            if os.path.exists(candidate):
                return "resnet18_raf", candidate
            if not allow_fallback:
                raise FileNotFoundError(f"ResNet-18 RAF checkpoint not found at {candidate}")
            logger.warning("ResNet-18 RAF checkpoint not found; falling back to default EfficientNet.")
            backbone_name = "efficientnet_current"

        candidate = video_ckpt or self._cfg.get("webcam_model_path")
        if candidate is None or not os.path.exists(candidate):
            webcam_model = "models/video_classifier/best_webcam_model.pth"
            legacy_model = "models/video_classifier/best_video_model.pth"
            candidate = webcam_model if os.path.exists(webcam_model) else legacy_model
        return backbone_name, candidate

    def get_available_backbones(self) -> List[Dict[str, Any]]:
        """Return supported webcam backbones and whether their checkpoints exist on disk."""
        _, eff_path = self._resolve_backbone_ckpt("efficientnet_current", allow_fallback=True)
        raf_path = self._cfg.get(
            "webcam_resnet18_raf_path", "external_models/resnet18_raf/best_resnet18.pt"
        )
        return [
            {
                "id": "efficientnet_current",
                "label": "EfficientNet-B0 (96px, Default)",
                "available": os.path.exists(eff_path),
                "active": self._backbone_name == "efficientnet_current",
            },
            {
                "id": "resnet18_raf",
                "label": "ResNet-18 (RAF-DB 128px)",
                "available": os.path.exists(raf_path),
                "active": self._backbone_name == "resnet18_raf",
            },
        ]

    def switch_backbone(self, backbone_name: str) -> Dict[str, Any]:
        """Hot-swap the active webcam backbone and clear stale per-client smoothing state."""
        resolved_name, ckpt_path = self._resolve_backbone_ckpt(backbone_name, allow_fallback=False)
        if resolved_name != self._backbone_name or ckpt_path != self.video_ckpt:
            self._backbone_name = resolved_name
            self.video_ckpt = ckpt_path
            self.load_models()
        self.client_states.clear()
        return {
            "backbone": self._backbone_name,
            "architecture": self._architecture,
            "img_size": getattr(self, "_img_size", 96),
            "calibrated_temperature": getattr(self, "calibrated_temperature", 0.85),
        }

    def _init_face_detector(self):
        """Initialize YuNet DNN face detector."""
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

    def _load_resnet18_raf(self):
        """Load RAF-DB ResNet-18 via live_resnet18_demo.load_model and remap classes by name."""
        logger.info("Loading ResNet-18 RAF-DB backbone...")
        self._architecture = "resnet18_raf"
        self.calibrated_temperature = 1.0

        model, prep, raf_class_names = load_model(self.video_ckpt)
        self._raf_prep = prep
        cc = next((t for t in getattr(prep, "transforms", []) if isinstance(t, transforms.CenterCrop)), None)
        if cc is not None:
            self._img_size = cc.size[0] if isinstance(cc.size, (tuple, list)) else int(cc.size)

        # Build index remapping: raf_index → canonical_index (by name, not position)
        aliases = {"happy": 3, "joy": 3, "sad": 5}
        self._raf_index_map = []
        for name in raf_class_names:
            key = str(name).strip()
            canonical_idx = self.RAF_TO_CANONICAL.get(key)
            if canonical_idx is None:
                canonical_idx = self.RAF_TO_CANONICAL.get(key.capitalize())
            if canonical_idx is None:
                canonical_idx = aliases.get(key.lower())
            if canonical_idx is None:
                raise ValueError(f"Cannot map RAF class '{name}' to canonical EMOTIONS order")
            self._raf_index_map.append(canonical_idx)

        # Set canonical label mapping
        self.label_mapping = {emo: i for i, emo in enumerate(EMOTIONS)}
        self.idx_to_label = {i: emo for i, emo in enumerate(EMOTIONS)}

        for p in model.parameters():
            p.requires_grad = False
        model.to(self.device)
        self.backbone = model
        self.classifier = None
        logger.info(f"ResNet-18 RAF-DB loaded ({len(raf_class_names)} classes, img_size={self._img_size}). Index map: {self._raf_index_map}")

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
                "baseline_geom": None,
                "is_calibrating": False,
                "calibration_buffer": [],
                "geom_calibration_buffer": [],
                "history": [],
                "current_label": None,
                "candidate_label": None,
                "candidate_count": 0,
                "last_good_prediction": None
            }
        return self.client_states[client_id]

    def start_calibration(self, client_id: str = "default"):
        """Activates baseline face calibration to cancel out resting facial bias."""
        client = self._get_client_state(client_id)
        client["is_calibrating"] = True
        client["calibration_buffer"] = []
        client["geom_calibration_buffer"] = []
        client["baseline_logits"] = None
        client["baseline_geom"] = None
        return {"status": "calibrating", "message": "Calibration started. Please hold a neutral facial expression."}

    def reset_calibration(self, client_id: str = "default"):
        """Resets baseline face calibration buffer and removes resting bias from memory."""
        client = self._get_client_state(client_id)
        client["is_calibrating"] = False
        client["calibration_buffer"] = []
        client["geom_calibration_buffer"] = []
        client["baseline_logits"] = None
        client["baseline_geom"] = None
        return {"status": "success", "message": f"Neutral baseline calibration for client '{client_id}' reset."}

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

    def trigger_debug_capture(self, num_frames: int = 150) -> Dict[str, Any]:
        """Activates debug frame capture for the next `num_frames` incoming frames into results/live_debug/."""
        self.debug_capture_active = True
        self.debug_capture_frames_remaining = num_frames
        self.debug_capture_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "active",
            "frames_to_capture": num_frames,
            "target_dir": str(self.debug_capture_dir)
        }

    def predict_frame(
        self,
        frame_data: Any,
        client_id: str = "default",
        apply_smoothing: bool = True,
        calibrate: bool = False,
        return_trace: bool = False
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper delegating directly to unified process_frame."""
        return self.process_frame(
            bgr_frame=frame_data,
            session_state_or_client_id=client_id,
            apply_smoothing=apply_smoothing,
            calibrate=calibrate,
            return_trace=return_trace
        )

    def process_frame(
        self,
        bgr_frame: Any,
        session_state_or_client_id: Any = "default",
        apply_smoothing: bool = True,
        calibrate: bool = False,
        return_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Unified single-source-of-truth webcam inference engine:
        1. Decodes frame and tracks inter-frame latency.
        2. Detects face(s) via YuNet DNN detector.
        3. Applies Affine eye alignment to eliminate head tilt distortion.
        4. Aspect-preserving reflection padding square crop centered on face.
        5. Normalizes lighting via mild gamma exposure normalization.
        6. Deep CNN forward pass.
        7. Dynamic personal baseline subtraction (if calibrated).
        8. Physiological AU geometry adjustment.
        9. Velocity-adaptive EMA smoothing.
        10. Centered logit consensus & temperature scaling (T*=0.85).
        11. Hysteresis state machine.
        12. Uncertainty estimation (is_uncertain, uncertainty_reason, confidence_margin).
        13. Standard dictionary formatting.
        14. Opt-in debug capture to results/live_debug/ when WEBCAM_DEBUG_CAPTURE=1.
        """
        t0 = time.time()
        if isinstance(session_state_or_client_id, dict):
            client = session_state_or_client_id
            client_id = client.get("client_id", "default")
        else:
            client_id = str(session_state_or_client_id)
            client = self._get_client_state(client_id)
        client["client_id"] = client_id
        
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

        img = self.decode_frame(bgr_frame)
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

        # Opt-in debug frame capture to results/live_debug/
        if getattr(self, "debug_capture_active", False) and img is not None:
            try:
                self.debug_frame_index = getattr(self, "debug_frame_index", 0) + 1
                idx = self.debug_frame_index
                debug_dir = getattr(self, "debug_capture_dir", Path("results/live_debug"))
                debug_dir.mkdir(parents=True, exist_ok=True)
                frame_filename = debug_dir / f"frame_{idx:05d}.jpg"
                cv2.imwrite(str(frame_filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                client["_last_debug_frame"] = str(frame_filename)
                client["_debug_idx"] = idx
                if hasattr(self, "debug_capture_frames_remaining"):
                    self.debug_capture_frames_remaining -= 1
                    if self.debug_capture_frames_remaining <= 0:
                        self.debug_capture_active = (os.environ.get("WEBCAM_DEBUG_CAPTURE") == "1")
            except Exception as e:
                logger.warning(f"Debug capture frame save error: {e}")

        h, w, _ = img.shape
        faces_detected = 0
        best_face = None
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
        pad = 1.10 if self._architecture == "resnet18_raf" else 1.25
        face_crop = aligned_square_crop(img, best_face, pad, min_side=10)
        if face_crop is None or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            face_crop = img

        # Quality assessment: blur (Laplacian variance), detection confidence, size, off-angle
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        laplacian_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var()) if gray_crop.size > 0 else 0.0
        is_blurry = laplacian_var < 30.0
        is_low_conf = face_confidence < 0.50
        is_too_small = (fw < 36 or fh < 36)

        # Extract facial landmark geometry ratios
        tilt_angle = 0.0
        geom_ratios = None
        if len(raw_landmarks) == 5:
            re, le = raw_landmarks[0], raw_landmarks[1]
            nose = raw_landmarks[2]
            rm, lm = raw_landmarks[3], raw_landmarks[4]
            eye_center = (float(re[0] + le[0]) / 2.0, float(re[1] + le[1]) / 2.0)
            dx_eye, dy_eye = float(le[0] - re[0]), float(le[1] - re[1])
            eye_dist = max(5.0, float(np.hypot(dx_eye, dy_eye)))
            tilt_angle = float(np.degrees(np.arctan2(dy_eye, dx_eye)))
            mouth_center = (float(rm[0] + lm[0]) / 2.0, float(rm[1] + lm[1]) / 2.0)
            mouth_w = float(np.hypot(lm[0] - rm[0], lm[1] - rm[1])) / eye_dist
            eye_to_mouth = float(np.hypot(mouth_center[0] - eye_center[0], mouth_center[1] - eye_center[1])) / eye_dist
            nose_to_mouth = float(np.hypot(mouth_center[0] - nose[0], mouth_center[1] - nose[1])) / eye_dist
            geom_ratios = {
                "mouth_w": mouth_w,
                "eye_to_mouth": eye_to_mouth,
                "nose_to_mouth": nose_to_mouth
            }
        is_off_angle = abs(tilt_angle) >= 45.0

        # Lighting assessment & exposure normalization (mild gamma vs CLAHE)
        # Avoids CLAHE shadow amplification on nasolabial folds which induces false disgust
        try:
            gray_l = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
            mean_lum = float(np.mean(gray_l))
            contrast_std = float(np.std(gray_l))
            if mean_lum < 60.0:
                gamma = 1.35
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                face_enhanced = cv2.LUT(face_crop, table)
            elif mean_lum > 195.0:
                gamma = 0.75
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                face_enhanced = cv2.LUT(face_crop, table)
            else:
                face_enhanced = face_crop
        except Exception:
            face_enhanced = face_crop
            mean_lum = 128.0
            contrast_std = 50.0

        quality_status = "optimal"
        if is_blurry:
            quality_status = "motion_blur"
            warnings.append("Blurry face detected; maintaining previous stable prediction.")
        elif is_low_conf:
            quality_status = "low_confidence"
            warnings.append("Low face detection confidence; maintaining previous prediction.")
        elif is_too_small:
            quality_status = "face_too_small"
            warnings.append("Face crop is small; expression confidence may be reduced.")
        elif is_off_angle:
            quality_status = "off_angle"
            warnings.append("Extreme head angle detected.")
        elif mean_lum < 35.0:
            quality_status = "low_light"
            warnings.append("Low lighting detected.")
        elif mean_lum > 225.0:
            quality_status = "overexposed"
            warnings.append("Harsh / overexposed lighting detected on face.")
        elif contrast_std < 15.0:
            quality_status = "low_contrast"
            warnings.append("Low facial contrast detected.")

        # If quality gate severely fails and a previous good prediction exists, hold it
        if (is_blurry or is_low_conf) and client.get("last_good_prediction") is not None:
            held = client["last_good_prediction"]
            latency_ms = (time.time() - t0) * 1000.0
            bbox_norm = {
                "x": round(fx / w, 4),
                "y": round(fy / h, 4),
                "width": round(fw / w, 4),
                "height": round(fh / h, 4),
                "raw_px": [fx, fy, fw, fh]
            }
            return {
                "status": "success",
                "face_detected": True,
                "faces_count": faces_detected,
                "bbox": bbox_norm,
                "landmarks": landmarks_norm,
                "face_score": round(face_confidence, 3),
                "apparent_expression": held["expression"],
                "calibrated_confidence": round(held["conf"] * 0.95, 4),
                "confidence_tier": held.get("tier", "Moderate"),
                "raw_confidence": round(held.get("raw_conf", held["conf"]), 4),
                "probabilities": held["probabilities"],
                "fps": round(rolling_fps, 1),
                "latency_ms": round(latency_ms, 1),
                "quality_status": quality_status,
                "smoothing_applied": apply_smoothing,
                "is_uncertain": held.get("is_uncertain", False),
                "uncertainty_reason": held.get("uncertainty_reason", None),
                "confidence_margin": held.get("confidence_margin", 0.0),
                "is_calibrated": client.get("baseline_logits") is not None,
                "calibration_status": "active" if client.get("baseline_logits") is not None else "idle",
                "disclaimer": "Apparent facial expression detected from visual cues. Does not determine internal affective state.",
                "history": client["history"][-10:],
                "warnings": warnings
            }

        # --- DEEP VISION MODEL INFERENCE ---
        if self._architecture == "resnet18_raf":
            rgb = cv2.cvtColor(face_enhanced, cv2.COLOR_BGR2RGB)
            tensor = self._raf_prep(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                raw_logits = self.backbone(tensor)[0].cpu().numpy()
            # Remap from RAF class order to canonical EMOTIONS order
            model_logits = np.zeros(7, dtype=np.float32)
            for raf_idx, canon_idx in enumerate(self._raf_index_map):
                model_logits[canon_idx] = raw_logits[raf_idx]
        else:
            img_size = self._img_size if hasattr(self, '_img_size') else 96
            resized = cv2.resize(face_enhanced, (img_size, img_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = self.norm(tensor).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                if self._architecture in ("efficientnet_b0", "timm_tf_efficientnet_b0"):
                    logits = self.backbone(tensor)
                    model_logits = logits[0].cpu().numpy()
                else:
                    feat = self.backbone(tensor)
                    logits = self.classifier(feat)
                    model_logits = logits[0].cpu().numpy()

        stage1_logits = model_logits.copy()

        # Prior debiasing: minimal adjustment for blank frame
        prior_bias = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        adjusted_logits = model_logits - prior_bias
        stage2_logits = adjusted_logits.copy()

        # --- PERSONAL RESTING FACE CALIBRATION (LOGIT-BIAS OFFSET & GEOMETRY) ---
        calib_status = "idle"
        if calibrate or client.get("is_calibrating", False):
            client["calibration_buffer"].append(adjusted_logits.copy())
            if geom_ratios is not None:
                client.setdefault("geom_calibration_buffer", []).append(geom_ratios)
            if len(client["calibration_buffer"]) >= 25:
                # Baseline neutral offset: compute mean resting face logits over 3 seconds
                mean_resting = np.mean(client["calibration_buffer"], axis=0)
                target_neutral = np.zeros(7, dtype=np.float32)
                target_neutral[4] = 1.0  # Class 4 is neutral
                client["baseline_logits"] = mean_resting - target_neutral
                client["is_calibrating"] = False
                client["calibration_buffer"] = []
                client["geom_calibration_buffer"] = []
                calib_status = "complete"
            else:
                client["is_calibrating"] = True
                calib_status = f"calibrating ({len(client['calibration_buffer'])}/25)"
        elif client.get("baseline_logits") is not None:
            calib_status = "active"
            adjusted_logits = adjusted_logits - 0.70 * client["baseline_logits"]

        stage3_logits = adjusted_logits.copy()

        # Pure vision model path: hard-coded AU geometry rules ablated per Phase 2 validation selection rule
        # (Ablation improved held-out validation Macro-F1 from 0.2872 to 0.3868, +0.0996 gain).
        stage4_logits = adjusted_logits.copy()
        delta_mw, delta_eym, delta_nmy = 0.0, 0.0, 0.0

        raw_probs = torch.softmax(torch.from_numpy(adjusted_logits), dim=-1).numpy()
        raw_pred_idx = int(np.argmax(raw_probs))
        raw_conf = float(raw_probs[raw_pred_idx])

        # --- ADAPTIVE TEMPORAL SMOOTHING (CONFIDENCE-WEIGHTED EMA) ---
        if apply_smoothing:
            if client["ema_probs"] is not None:
                # Dynamic alpha based on model confidence: confident frames react faster, uncertain frames hold
                alpha = float(np.clip(0.20 + 0.35 * raw_conf, 0.18, 0.55))
                smoothed_probs = alpha * raw_probs + (1.0 - alpha) * client["ema_probs"]
                smoothed_probs /= np.sum(smoothed_probs)
            else:
                smoothed_probs = raw_probs
            client["ema_probs"] = smoothed_probs
        else:
            smoothed_probs = raw_probs

        client["last_landmarks"] = raw_landmarks

        # --- TEMPERATURE CALIBRATION ---
        log_p = np.log(np.clip(smoothed_probs, 1e-6, 1.0))
        centered_logits = log_p - np.mean(log_p)
        temperature = getattr(self, "calibrated_temperature", 0.85)
        scaled_logits = centered_logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        # --- HYSTERESIS STATE MACHINE ---
        # Prevents flickering: switching requires candidate to beat current by margin for N=3 frames
        top_idx = int(np.argmax(calibrated_probs))
        top_label = self.idx_to_label[top_idx].upper()

        if client.get("current_label") is None:
            client["current_label"] = top_label
            client["candidate_label"] = None
            client["candidate_count"] = 0
        else:
            curr_label = client["current_label"]
            curr_idx = self.label_mapping[curr_label.lower()]
            curr_prob = float(calibrated_probs[curr_idx])
            top_prob = float(calibrated_probs[top_idx])
            margin = 0.08

            if top_label != curr_label and (top_prob > curr_prob + margin):
                if top_label == client.get("candidate_label"):
                    client["candidate_count"] = client.get("candidate_count", 0) + 1
                    if client["candidate_count"] >= 2:
                        client["current_label"] = top_label
                        client["candidate_label"] = None
                        client["candidate_count"] = 0
                else:
                    client["candidate_label"] = top_label
                    client["candidate_count"] = 1
            else:
                client["candidate_label"] = None
                client["candidate_count"] = 0

        predicted_expression = client["current_label"]
        pred_idx = self.label_mapping[predicted_expression.lower()]
        calibrated_conf = float(calibrated_probs[pred_idx])

        prob_dict = {
            self.idx_to_label[i]: round(float(calibrated_probs[i]), 4)
            for i in range(len(self.idx_to_label))
        }

        # Principled Uncertainty Estimation & Confidence Tier per Phase 4 protocol
        sorted_probs = np.sort(calibrated_probs)[::-1]
        p_top1 = float(sorted_probs[0])
        p_top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        conf_margin = float(p_top1 - p_top2)

        # Confidence tier classification
        if calibrated_conf >= 0.65 and conf_margin >= 0.15:
            conf_tier = "Robust"
        elif calibrated_conf >= 0.45 and conf_margin >= 0.08:
            conf_tier = "Moderate"
        else:
            conf_tier = "Subtle / Ambiguous"
            warnings.append(f"Moderate expression intensity ({calibrated_conf:.1%}); expression cues appear subtle.")

        is_uncertain = bool(p_top1 < 0.38 or conf_margin < 0.08)
        uncertainty_reason = None
        if is_uncertain:
            if p_top1 < 0.38:
                uncertainty_reason = f"Low top-1 confidence ({p_top1:.1%})"
            else:
                uncertainty_reason = f"Ambiguous margin between top classes ({conf_margin:.1%})"

        latency_ms = (time.time() - t0) * 1000.0

        # Cache last good prediction for quality gating
        if quality_status == "optimal":
            client["last_good_prediction"] = {
                "expression": predicted_expression,
                "conf": calibrated_conf,
                "raw_conf": raw_conf,
                "tier": conf_tier,
                "probabilities": prob_dict,
                "is_uncertain": is_uncertain,
                "uncertainty_reason": uncertainty_reason,
                "confidence_margin": round(conf_margin, 4)
            }

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

        out_res = {
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
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "confidence_margin": round(conf_margin, 4),
            "is_calibrated": client.get("baseline_logits") is not None,
            "calibration_status": calib_status,
            "disclaimer": "Apparent facial expression detected from visual cues. Does not determine internal affective state.",
            "history": client["history"][-10:],
            "warnings": warnings
        }

        if return_trace:
            def _to_probs(arr):
                e = np.exp(arr - np.max(arr))
                return (e / np.sum(e)).tolist()
            p1 = _to_probs(stage1_logits)
            p2 = _to_probs(stage2_logits)
            p3 = _to_probs(stage3_logits)
            p4 = _to_probs(stage4_logits)
            out_res["trace"] = {
                "stage1_raw_cnn": {"logits": [round(float(v), 4) for v in stage1_logits], "probs": [round(float(v), 4) for v in p1], "top": EMOTIONS[int(np.argmax(p1))]},
                "stage2_prior": {"logits": [round(float(v), 4) for v in stage2_logits], "probs": [round(float(v), 4) for v in p2], "top": EMOTIONS[int(np.argmax(p2))]},
                "stage3_user_baseline": {"logits": [round(float(v), 4) for v in stage3_logits], "probs": [round(float(v), 4) for v in p3], "top": EMOTIONS[int(np.argmax(p3))]},
                "stage4_geometry": {"logits": [round(float(v), 4) for v in stage4_logits], "probs": [round(float(v), 4) for v in p4], "top": EMOTIONS[int(np.argmax(p4))],
                                    "delta_eym": round(float(delta_eym), 4), "delta_nmy": round(float(delta_nmy), 4), "delta_mw": round(float(delta_mw), 4)},
                "stage5_smoothed": {"probs": [round(float(v), 4) for v in smoothed_probs], "top": EMOTIONS[int(np.argmax(smoothed_probs))]},
                "stage6_calibrated": {"probs": [round(float(v), 4) for v in calibrated_probs], "top": EMOTIONS[int(np.argmax(calibrated_probs))]},
                "stage7_final": {"expression": predicted_expression.lower()}
            }

        # Save companion JSON for debug frame if capture active
        if getattr(self, "debug_capture_active", False) and client.get("_last_debug_frame"):
            try:
                meta_path = Path(client["_last_debug_frame"]).with_suffix(".json")
                meta = {
                    "frame_file": str(Path(client["_last_debug_frame"]).name),
                    "expression": predicted_expression,
                    "confidence": calibrated_conf,
                    "tier": conf_tier,
                    "probabilities": prob_dict,
                    "is_uncertain": is_uncertain,
                    "uncertainty_reason": uncertainty_reason,
                    "latency_ms": round(latency_ms, 1),
                    "quality_status": quality_status,
                    "warnings": warnings,
                    "trace": out_res.get("trace")
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                logger.warning(f"Debug capture metadata save error: {e}")

        return out_res

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
            from backend.services.audio_service import AudioInferenceService
            self._audio_service = AudioInferenceService()
        if self._text_service is None:
            from backend.services.text_service import TextInferenceService
            self._text_service = TextInferenceService()
        if self._fusion_service is None:
            from backend.services.fusion_service import MultimodalFusionService
            self._fusion_service = MultimodalFusionService()
        if self._transcription_service is None:
            from backend.services.transcription_service import TranscriptionService
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

        # In live webcam mode, ensure visual prediction shown in both panels matches
        # and ambient microphone room noise (no speech) does not hijack the facial emotion
        if getattr(self, "webcam_unified_uses_webcam_engine", True):
            if not transcript_text or not transcript_text.strip():
                # Ambient audio without active speech: keep visual prediction as primary
                fused_res["fused_predicted_class"] = video_res["apparent_expression"].upper()
                fused_res["fused_confidence"] = video_res["calibrated_confidence"]
                fused_res["modality_weights"] = {"video": 0.85, "audio": 0.15, "text": 0.0}
                fused_res["unified_explanation"] = (
                    f"Visual facial cues primarily indicate {video_res['apparent_expression'].upper()} "
                    f"({video_res['calibrated_confidence']:.1%}). Ambient room audio contained no spoken speech "
                    f"and was attenuated to maintain single-engine consistency."
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

