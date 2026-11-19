import os
import sys
import time
import types

# Ensure safe unpickling on environments with strict DLL/C-extension policies
for _ext in ['sklearn.svm._liblinear', 'sklearn.svm._libsvm_sparse', 'sklearn.svm._libsvm']:
    if _ext not in sys.modules:
        try:
            __import__(_ext)
        except Exception:
            sys.modules[_ext] = types.ModuleType(_ext)

import joblib
import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModel
from backend.models.text_model import TextPreprocessingUtils, EMOTIONS

class TextInferenceService:
    def __init__(
        self, 
        contextual_path: str = "models/text_classifier/best_text_model_contextual.pt",
        legacy_path: str = "models/text_classifier/best_text_model.joblib"
    ):
        self.contextual_path = contextual_path
        self.legacy_path = legacy_path
        self.mode = None
        self.classifier = None
        self.tokenizer = None
        self.transformer = None
        self.pipeline = None
        self.classes = None
        self.metrics = None
        self.is_loaded = False
        
        self.load_model()
        
    def load_model(self):
        # 1. Try to load contextual transformer model if available
        if os.path.exists(self.contextual_path):
            try:
                bundle = torch.load(self.contextual_path, map_location="cpu", weights_only=False)
                model_name = bundle.get("backbone", "sentence-transformers/all-MiniLM-L6-v2")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
                self.transformer.eval()
                for p in self.transformer.parameters():
                    p.requires_grad = False
                self.classifier = bundle["classifier"]
                self.classes = list(bundle.get("classes", ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']))
                self.metrics = bundle.get("metrics", {})
                self.mode = "contextual"
                self.is_loaded = True
                return
            except Exception as e:
                print(f"[TextService] Note: Failed to load contextual model ({e}), falling back to legacy TF-IDF.")
                
        # 2. Fallback to legacy TF-IDF pipeline
        if os.path.exists(self.legacy_path):
            data = joblib.load(self.legacy_path)
            self.pipeline = data["pipeline"]
            self.classes = list(self.pipeline.classes_)
            self.metrics = data.get("metrics", {})
            self.mode = "legacy"
            self.is_loaded = True
        else:
            raise FileNotFoundError(f"Neither contextual nor legacy text checkpoint found at {self.contextual_path} or {self.legacy_path}")
        
    def predict_from_text(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_model()
            
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Input text cannot be empty or solely whitespace.")
            
        # Text Preprocessing & Contraction Normalization
        normalized = TextPreprocessingUtils.clean_and_normalize(cleaned_text)

        t0 = time.time()
        
        if self.mode == "contextual":
            inputs = self.tokenizer([normalized], padding=True, truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                out = self.transformer(**inputs)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                emb = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                emb_np = emb.cpu().numpy()
            probs = self.classifier.predict_proba(emb_np)[0]
            model_used = "Contextual Transformer (all-MiniLM-L6-v2) + Calibrated Classifier"
        else:
            probs = self.pipeline.predict_proba([normalized])[0]
            model_used = "TF-IDF N-gram + Calibrated Classifier"
            
        pred_idx = int(probs.argmax())
        predicted_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        prob_dict = {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }
        
        latency_ms = (time.time() - t0) * 1000.0
        
        words = cleaned_text.split()
        quality_warnings = []
        if len(words) <= 1 and len(cleaned_text) < 5:
            quality_warnings.append("Utterance is exceptionally short (<5 characters); linguistic context is limited.")
        quality_status = "degraded" if len(quality_warnings) > 0 else "optimal"
        quality_discount = 0.85 if len(quality_warnings) > 0 else 1.0

        return {
            "status": "success",
            "modality": "text",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "model_used": model_used,
            "inference_time_ms": round(latency_ms, 2),
            "details": {
                "text": cleaned_text,
                "character_length": len(cleaned_text),
                "word_count": len(words),
                "benchmark_accuracy": self.metrics.get("accuracy"),
                "benchmark_macro_f1": self.metrics.get("macro_f1"),
                "quality_metrics": {
                    "word_count": len(words),
                    "character_length": len(cleaned_text),
                    "quality_status": quality_status,
                    "quality_discount": quality_discount,
                    "warnings": quality_warnings
                }
            }
        }
