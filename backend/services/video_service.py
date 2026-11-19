import os
import time
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from typing import Dict, Any, List

from backend.models.video_model import VideoClassifier

class VideoInferenceService:
    def __init__(self, checkpoint_path: str = "models/video_classifier/best_video_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.resnet = None
        self.classifier = None
        self.label_mapping = None
        self.idx_to_label = None
        self.norm = None
        self.is_loaded = False
        
        self.load_models()
        
    def load_models(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Video checkpoint not found at {self.checkpoint_path}")
            
        # 1. ResNet-50 backbone
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        self.resnet.fc = nn.Identity()
        self.resnet.eval()
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.to(self.device)
        
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 2. Classifier
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.label_mapping = ckpt["label_mapping"]
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        input_dim = ckpt.get("input_dim", 2048)
        
        self.classifier = VideoClassifier(input_dim=input_dim, num_classes=len(self.label_mapping))
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.to(self.device)
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad = False
            
        self.is_loaded = True

    def predict_from_video(self, video_path: str, num_frames: int = 8, generate_preview: bool = True) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()
            
        t0 = time.time()
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        if frame_count <= 0:
            cap.release()
            raise ValueError("Could not decode any frames from the provided video file.")
            
        indices = [int(i) for i in torch.linspace(0, max(0, frame_count - 1), num_frames)]
        tensors = []
        preview_thumbnails = []
        luminance_vals = []
        contrast_vals = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 1. Aspect-Ratio Preserving Facial/Head Region Crop
                # Avoids 16:9 -> 1:1 face distortion and eliminates peripheral background set clutter
                h, w, _ = frame.shape
                crop_size = min(h, w)
                y_start = int(max(0, (h - crop_size) * 0.15))
                x_start = int(max(0, (w - crop_size) // 2))
                cropped_face = frame[y_start:y_start+crop_size, x_start:x_start+crop_size]

                # 2. Lighting Normalization via CLAHE (Contrast-Limited Adaptive Histogram Equalization)
                try:
                    lab = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    luminance_vals.append(float(np.mean(l)))
                    contrast_vals.append(float(np.std(l)))
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cl = clahe.apply(l)
                    enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                except Exception:
                    enhanced_frame = cropped_face

                # 3. Resize to ResNet-50 224x224 input
                resized = cv2.resize(enhanced_frame, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensors.append(self.norm(tensor))
                
                # Optional thumbnail preview
                if generate_preview and len(preview_thumbnails) < 4:
                    thumb = cv2.resize(enhanced_frame, (160, 160))
                    _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    preview_thumbnails.append(f"data:image/jpeg;base64,{b64}")
                    
        cap.release()
        
        if len(tensors) == 0:
            raise ValueError("Failed to extract valid frames from video.")

        duration = round(frame_count / fps, 2)
        mean_lum = float(np.mean(luminance_vals)) if luminance_vals else 128.0
        mean_contrast = float(np.mean(contrast_vals)) if contrast_vals else 50.0

        quality_warnings = []
        if duration < 0.6:
            quality_warnings.append("Video duration very brief (<0.6s); temporal expressions may be clipped.")
        if mean_lum < 35.0:
            quality_warnings.append("Severe underexposure / dark lighting detected in facial frames.")
        elif mean_lum > 220.0:
            quality_warnings.append("Overexposure / harsh lighting detected in facial frames.")
        if mean_contrast < 15.0:
            quality_warnings.append("Low facial contrast detected.")

        quality_status = "degraded" if len(quality_warnings) > 0 else "optimal"
        quality_discount = 0.80 if len(quality_warnings) > 0 else 1.0
            
        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            frame_embs = self.resnet(batch_tensor)
            video_emb = frame_embs.mean(dim=0, keepdim=True)
            logits = self.classifier(video_emb)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        pred_idx = int(probs.argmax())
        predicted_class = self.idx_to_label[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            self.idx_to_label[i]: float(probs[i])
            for i in range(len(self.idx_to_label))
        }
        
        latency_ms = (time.time() - t0) * 1000.0
        duration = round(frame_count / fps, 2)
        
        return {
            "status": "success",
            "modality": "video",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "model_used": "ResNet-50 + Temporal Pooling + VideoClassifier (models/video_classifier/best_video_model.pth)",
            "inference_time_ms": round(latency_ms, 2),
            "details": {
                "total_frames": frame_count,
                "frames_sampled": len(tensors),
                "fps": round(fps, 2),
                "duration_seconds": duration,
                "preview_thumbnails": preview_thumbnails,
                "quality_metrics": {
                    "mean_luminance": round(mean_lum, 2),
                    "contrast_std": round(mean_contrast, 2),
                    "quality_status": quality_status,
                    "quality_discount": quality_discount,
                    "warnings": quality_warnings
                }
            }
        }
