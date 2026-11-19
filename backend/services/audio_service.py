import os
import time
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import Dict, Any

from backend.models.audio_model import AudioClassifier, StrongerAudioClassifier

class AudioInferenceService:
    def __init__(self, checkpoint_path: str = "models/audio_classifier/best_audio_model_balanced.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.legacy_path = "models/audio_classifier/best_audio_model.pth"
        self.processor = None
        self.wav2vec = None
        self.classifier = None
        self.label_mapping = None
        self.idx_to_label = None
        self.is_loaded = False
        self.arch_name = "AudioClassifier"
        
        self.load_models()
        
    def load_models(self):
        target_path = self.checkpoint_path if os.path.exists(self.checkpoint_path) else self.legacy_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Audio checkpoint not found at {target_path}")
            
        # 1. Load Wav2Vec2 backbone
        model_name = "facebook/wav2vec2-base"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, local_files_only=True)
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True)
        self.wav2vec.eval()
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.wav2vec.to(self.device)
        
        # 2. Load Classifier Checkpoint
        ckpt = torch.load(target_path, map_location=self.device, weights_only=False)
        input_dim = ckpt.get("input_dim", 768)
        self.label_mapping = ckpt["label_mapping"]
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        self.arch_name = ckpt.get("architecture", "AudioClassifier")
        if self.arch_name == "StrongerAudioClassifier":
            self.classifier = StrongerAudioClassifier(input_dim=input_dim, num_classes=len(self.label_mapping))
        else:
            self.classifier = AudioClassifier(input_dim=input_dim, num_classes=len(self.label_mapping))
            
        self.classifier.load_state_dict(ckpt["model_state_dict"])
        self.classifier.to(self.device)
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
            
        self.is_loaded = True

    def predict_from_audio(self, audio_path: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()
            
        t0 = time.time()
        
        # Read waveform
        try:
            waveform, sr = sf.read(audio_path)
        except Exception:
            # Fallback to librosa
            waveform, sr = librosa.load(audio_path, sr=None)
            
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)
            
        # Resample to 16kHz if necessary
        target_sr = 16000
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        # Signal Quality Assessment
        raw_rms = float(np.sqrt(np.mean(waveform ** 2)))
        clipping_ratio = float(np.mean(np.abs(waveform) > 0.98))
        duration_sec = round(len(waveform) / sr, 2)

        # Audio Quality Enhancement: DC offset removal & Peak Normalization
        waveform = waveform - np.mean(waveform)
        peak = np.max(np.abs(waveform))
        if peak > 1e-6:
            waveform = waveform / peak
            
        # Trim leading and trailing silence (top_db=30)
        try:
            trimmed, _ = librosa.effects.trim(waveform, top_db=30)
            if len(trimmed) >= 1600: # at least 100ms
                waveform = trimmed
        except Exception:
            pass

        # Estimated SNR
        trimmed_rms = float(np.sqrt(np.mean(waveform ** 2)))
        snr_est_db = round(float(20 * np.log10(max(1e-4, trimmed_rms) / max(1e-5, raw_rms * 0.1))), 1)

        quality_warnings = []
        if duration_sec < 0.6:
            quality_warnings.append("Audio duration is very brief (<0.6s); vocal prosody cues may be partial.")
        if raw_rms < 0.008:
            quality_warnings.append("Low acoustic energy/volume detected; signal may be close to silence.")
        if clipping_ratio > 0.05:
            quality_warnings.append("Significant acoustic clipping detected (>5% samples clipped).")

        quality_status = "degraded" if len(quality_warnings) > 0 else "optimal"
        quality_discount = 0.85 if len(quality_warnings) > 0 else 1.0

        # Truncate if exceeds 30s
        max_duration = 30
        max_samples = max_duration * target_sr
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
            
        # Feature extraction
        inputs = self.processor(waveform, sampling_rate=target_sr, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else None
        
        with torch.no_grad():
            outputs = self.wav2vec(input_values, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            if attention_mask is not None:
                expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
                sum_embeddings = torch.sum(last_hidden_state * expanded_mask, dim=1)
                sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            else:
                embedding = torch.mean(last_hidden_state, dim=1)
                
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        pred_idx = int(probs.argmax())
        predicted_class = self.idx_to_label[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            self.idx_to_label[i]: float(probs[i])
            for i in range(len(self.idx_to_label))
        }
        
        latency_ms = (time.time() - t0) * 1000.0
        
        return {
            "status": "success",
            "modality": "audio",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "model_used": f"Wav2Vec2-Base + {self.arch_name} ({os.path.basename(self.checkpoint_path if os.path.exists(self.checkpoint_path) else self.legacy_path)})",
            "inference_time_ms": round(latency_ms, 2),
            "details": {
                "sample_rate": sr,
                "duration_seconds": duration_sec,
                "embedding_dim": 768,
                "device": str(self.device),
                "quality_metrics": {
                    "rms_energy": round(raw_rms, 4),
                    "estimated_snr_db": snr_est_db,
                    "clipping_ratio": round(clipping_ratio, 4),
                    "quality_status": quality_status,
                    "quality_discount": quality_discount,
                    "warnings": quality_warnings
                }
            }
        }

audio_service = AudioInferenceService()
