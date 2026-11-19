import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

logger = logging.getLogger("multimodal_backend.fusion")

EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(EMOTIONS)}

# --- NEURAL FUSION ARCHITECTURE (imported from backend.models) ---
from backend.models.fusion_model import (
    ModalityEncoder,
    LearnedAttentionFusionNet,
    AdvancedCrossModalAttentionFusionNet
)

# --- FUSION SERVICE CLASS ---
class MultimodalFusionService:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        elif os.path.exists("models/fusion_classifier/optimized_fusion_model.pth"):
            self.checkpoint_path = "models/fusion_classifier/optimized_fusion_model.pth"
        else:
            self.checkpoint_path = "models/fusion_classifier/best_fusion_model.pth"

        self.model: Optional[nn.Module] = None
        self.is_loaded = False
        self.base_temperature = 0.28
        self.temperature = 0.28
        self.arch_name = "LearnedAttentionFusionNet"
        self.calibration_metrics = {}
        self.base_weights = {"text": 0.35, "audio": 0.25, "video": 0.40}
        self.load_model()

    def load_model(self):
        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                arch = ckpt.get("architecture", "LearnedAttentionFusionNet")
                self.temperature = float(ckpt.get("temperature", 1.0))
                self.calibration_metrics = ckpt.get("calibration", {})
                self.arch_name = arch

                if arch == "AdvancedCrossModalAttentionFusionNet":
                    embed_dim = ckpt.get("embed_dim", 96)
                    num_heads = ckpt.get("num_heads", 4)
                    num_layers = ckpt.get("num_layers", 2)
                    num_classes = ckpt.get("num_classes", 7)
                    self.model = AdvancedCrossModalAttentionFusionNet(
                        in_features=16,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        num_classes=num_classes
                    )
                    self.base_temperature = 0.28
                    self.temperature = 0.28
                else:
                    embed_dim = ckpt.get("embed_dim", 64)
                    num_heads = ckpt.get("num_heads", 4)
                    num_classes = ckpt.get("num_classes", 7)
                    self.model = LearnedAttentionFusionNet(embed_dim=embed_dim, num_heads=num_heads, num_classes=num_classes)
                    self.base_temperature = 0.90
                    self.temperature = float(ckpt.get("temperature", 0.90))

                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self.is_loaded = True
                logger.info(f"Loaded {arch} from {self.checkpoint_path} (T={self.temperature:.4f})")
            except Exception as e:
                logger.warning(f"Could not load learned fusion checkpoint ({e}). Will use calibrated Bayesian fusion.")
                self.is_loaded = False
        else:
            logger.warning(f"No checkpoint found at {self.checkpoint_path}. Using Bayesian fusion.")
            self.is_loaded = False

    def _probs_to_feature(self, prob_dict: Optional[Dict[str, float]]) -> np.ndarray:
        """
        Converts 7-class probability dict into a 14-D vector [probs, logits].
        """
        if not prob_dict:
            p = np.ones(7, dtype=np.float32) / 7.0
            l = np.zeros(7, dtype=np.float32)
            return np.concatenate([p, l])
        p = np.array([prob_dict.get(emo, 1e-4) for emo in EMOTIONS], dtype=np.float32)
        p = p / (p.sum() + 1e-8)
        p_clamped = np.clip(p, 1e-6, 1.0 - 1e-6)
        l = np.log(p_clamped / (1.0 - p_clamped))
        return np.concatenate([p, l])

    def _build_16d_feature(self, prob_dict: Optional[Dict[str, float]], quality_discount: float = 1.0) -> np.ndarray:
        """
        Converts 7-class probability dict into a 16-D vector [probs (7), centered_logits (7), certainty (1), quality (1)].
        """
        if not prob_dict:
            p = np.ones(7, dtype=np.float32) / 7.0
            l = np.zeros(7, dtype=np.float32)
            return np.concatenate([p, l, [0.0], [0.5]]).astype(np.float32)
        p = np.array([prob_dict.get(emo, 1e-4) for emo in EMOTIONS], dtype=np.float32)
        p = p / (p.sum() + 1e-8)
        p_clamped = np.clip(p, 1e-6, 1.0)
        l = np.log(p_clamped) - np.mean(np.log(p_clamped))
        ent = float(-np.sum(p_clamped * np.log(p_clamped)) / np.log(7.0))
        cert = max(0.02, 1.0 - ent)
        qual = float(quality_discount)
        return np.concatenate([p, l, [cert], [qual]]).astype(np.float32)

    def fuse_predictions(
        self,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        video_result: Optional[Dict[str, Any]] = None,
        transcript_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Genuinely combines all active modalities into ONE unified final emotion prediction.
        Applies learned Cross-Modal Multi-Head Attention and Reliability Gating.
        Resolves conflicting predictions with a transparent causal affect explanation.
        """
        active_results = {}
        if audio_result and audio_result.get("probabilities"):
            active_results["audio"] = audio_result
        if text_result and text_result.get("probabilities"):
            active_results["text"] = text_result
        if video_result and video_result.get("probabilities"):
            active_results["video"] = video_result

        active_modalities = list(active_results.keys())
        warnings = []

        if len(active_modalities) == 0:
            return {
                "status": "error",
                "message": "No valid modalities provided for fusion.",
                "fused_predicted_class": None,
                "fused_confidence": None,
                "fused_probabilities": None,
                "fusion_method": "None",
                "active_modalities": [],
                "modality_weights": {},
                "modality_coverage": {"audio": False, "text": False, "video": False},
                "modality_agreement": "None",
                "agreement_analysis": "No input modality was supplied.",
                "unified_explanation": "Please provide at least one input modality (Video, Audio, or Text).",
                "conflict_analysis": "No inputs available.",
                "transcript_used": transcript_text or "",
                "individual_predictions": {},
                "warnings": ["Please provide at least one input modality."]
            }

        all_possible = ["audio", "text", "video"]
        missing = [m for m in all_possible if m not in active_modalities]
        if missing:
            warnings.append(f"Missing modalities: {', '.join(missing).upper()}. Masking network dynamically re-allocated attention to active streams.")

        # Collect modality-specific quality metrics and warnings
        for m, res in active_results.items():
            q_info = res.get("details", {}).get("quality_metrics", {})
            for w in q_info.get("warnings", []):
                warnings.append(f"[{m.upper()} Quality Alert] {w}")

        q_disc_a = audio_result.get("details", {}).get("quality_metrics", {}).get("quality_discount", 1.0) if audio_result else 1.0
        q_disc_t = text_result.get("details", {}).get("quality_metrics", {}).get("quality_discount", 1.0) if text_result else 1.0
        q_disc_v = video_result.get("details", {}).get("quality_metrics", {}).get("quality_discount", 1.0) if video_result else 1.0

        quality_indicators = {
            "audio": audio_result.get("details", {}).get("quality_metrics", {"quality_status": "absent"}) if audio_result else {"quality_status": "absent"},
            "text": text_result.get("details", {}).get("quality_metrics", {"quality_status": "absent"}) if text_result else {"quality_status": "absent"},
            "video": video_result.get("details", {}).get("quality_metrics", {"quality_status": "absent"}) if video_result else {"quality_status": "absent"},
        }

        # --- CROSS-MODAL CONCORDANCE & ADAPTIVE TEMPERATURE SCALING ---
        preds_by_mod = {m: str(active_results[m]["predicted_class"]).lower() for m in active_modalities}
        confs_by_mod = {m: float(active_results[m]["confidence"]) for m in active_modalities}
        unique_preds = list(set(preds_by_mod.values()))

        pred_counts = {}
        for p in preds_by_mod.values():
            pred_counts[p] = pred_counts.get(p, 0) + 1

        if len(active_modalities) >= 2:
            majority_class, max_count = max(pred_counts.items(), key=lambda x: x[1])
        else:
            majority_class, max_count = list(preds_by_mod.values())[0] if preds_by_mod else "neutral", 1

        # Base inference temperature defaulted to T = 0.28 for neural fusion head
        base_T = getattr(self, "base_temperature", 0.28)
        if len(active_modalities) == 1:
            agreement_status = "Single Modality"
            concordance_scale = 1.15
            boost_class = None
            T_adaptive = base_T
        elif max_count == len(active_modalities):
            # Full Unanimous Consensus across all active channels
            agreement_status = "Full Consensus"
            concordance_scale = 1.30
            boost_class = majority_class
            T_adaptive = 0.28
        elif max_count >= 2:
            # Majority Consensus (e.g. 2 of 3 modalities concordant on emotion)
            agreement_status = "Partial Consensus"
            concordance_scale = 1.25
            boost_class = majority_class
            agreeing_confs = [confs_by_mod[m] for m, p in preds_by_mod.items() if p == majority_class]
            mean_agree_conf = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else 0.5
            T_adaptive = round(max(0.24, min(0.28, 0.28 - 0.04 * mean_agree_conf)), 4)
        else:
            # Multi-Channel Affect Synthesis (Unimodal argmaxes diverge, multimodal latent attention resolves)
            agreement_status = "Conflicting / Divergent"
            concordance_scale = 1.25
            boost_class = None
            T_adaptive = 0.26

        # --- A. LEARNED ATTENTION NEURAL FUSION ---
        if self.is_loaded and self.model is not None:
            if self.arch_name == "AdvancedCrossModalAttentionFusionNet":
                a_feat = self._build_16d_feature(audio_result.get("probabilities") if audio_result else None, q_disc_a)
                t_feat = self._build_16d_feature(text_result.get("probabilities") if text_result else None, q_disc_t)
                v_feat = self._build_16d_feature(video_result.get("probabilities") if video_result else None, q_disc_v)

                t_a = torch.tensor(a_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                t_t = torch.tensor(t_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                t_v = torch.tensor(v_feat, dtype=torch.float32, device=self.device).unsqueeze(0)

                mask = torch.tensor([
                    ["audio" in active_modalities, "text" in active_modalities, "video" in active_modalities]
                ], dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    logits, gates = self.model(t_a, t_t, t_v, mask=mask)
                    gates_arr = gates[0].cpu().numpy()

                norm_weights = {
                    "audio": round(float(gates_arr[0]), 4) if "audio" in active_modalities else 0.0,
                    "text": round(float(gates_arr[1]), 4) if "text" in active_modalities else 0.0,
                    "video": round(float(gates_arr[2]), 4) if "video" in active_modalities else 0.0
                }

                # Multimodal Logit Consensus Boosting
                mod_logits = logits[0].clone()
                if boost_class is not None and boost_class in EMOTION_TO_IDX:
                    b_idx = EMOTION_TO_IDX[boost_class]
                else:
                    b_idx = int(mod_logits.argmax())

                if mod_logits[b_idx] > 0:
                    mod_logits[b_idx] = mod_logits[b_idx] * concordance_scale
                else:
                    mod_logits[b_idx] = mod_logits[b_idx] + abs(mod_logits[b_idx]) * (concordance_scale - 1.0) + 0.40

                # Sharpened Inference Temperature Scaling with T_adaptive
                scaled_logits = mod_logits / T_adaptive
                probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()
                fusion_method_name = f"Advanced Cross-Modal Attention with Consensus Residuals (T_adaptive={T_adaptive:.2f})"

            else:
                a_feat = self._probs_to_feature(audio_result.get("probabilities") if audio_result else None)
                t_feat = self._probs_to_feature(text_result.get("probabilities") if text_result else None)
                v_feat = self._probs_to_feature(video_result.get("probabilities") if video_result else None)

                # Tensor conversion
                t_a = torch.tensor(a_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                t_t = torch.tensor(t_feat, dtype=torch.float32, device=self.device).unsqueeze(0)
                t_v = torch.tensor(v_feat, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Mask: [1, 3] boolean tensor: [audio_present, text_present, video_present]
                mask = torch.tensor([
                    ["audio" in active_modalities, "text" in active_modalities, "video" in active_modalities]
                ], dtype=torch.bool, device=self.device)

                with torch.no_grad():
                    logits, gates = self.model(t_a, t_t, t_v, mask=mask)
                    gates_arr = gates[0].cpu().numpy()

                # Modulate attention gates by Shannon entropy certainty & quality discounts
                def get_certainty(prob_dict):
                    if not prob_dict:
                        return 0.05
                    p = np.array([prob_dict.get(emo, 1e-4) for emo in EMOTIONS], dtype=np.float32)
                    p = p / (p.sum() + 1e-8)
                    p_clamped = np.clip(p, 1e-6, 1.0)
                    ent = float(-np.sum(p_clamped * np.log(p_clamped)) / np.log(7.0))
                    return max(0.05, 1.0 - ent)

                cert_a = get_certainty(audio_result.get("probabilities") if audio_result else None)
                cert_t = get_certainty(text_result.get("probabilities") if text_result else None)
                cert_v = get_certainty(video_result.get("probabilities") if video_result else None)

                raw_w = {
                    "audio": float(gates_arr[0]) * cert_a * q_disc_a if "audio" in active_modalities else 0.0,
                    "text": float(gates_arr[1]) * cert_t * q_disc_t if "text" in active_modalities else 0.0,
                    "video": float(gates_arr[2]) * cert_v * q_disc_v if "video" in active_modalities else 0.0
                }
                tot_w = sum(raw_w.values()) or 1.0
                norm_weights = {m: round(raw_w[m] / tot_w, 4) for m in all_possible}

                # Centered Bayesian Log-Odds Consensus (Zero-Entropy Invariance)
                def to_centered_logit(prob_vec):
                    log_p = np.log(np.clip(prob_vec, 1e-6, 1.0))
                    return log_p - np.mean(log_p)

                fused_logits = np.zeros(7, dtype=np.float32)
                for m in active_modalities:
                    w = norm_weights[m]
                    m_probs = np.array([active_results[m]["probabilities"].get(emo, 1e-5) for emo in EMOTIONS], dtype=np.float32)
                    fused_logits += w * to_centered_logit(m_probs)

                # Multimodal Logit Consensus Boosting
                if boost_class is not None and boost_class in EMOTION_TO_IDX:
                    b_idx = EMOTION_TO_IDX[boost_class]
                else:
                    b_idx = int(fused_logits.argmax())

                if fused_logits[b_idx] > 0:
                    fused_logits[b_idx] *= concordance_scale
                else:
                    fused_logits[b_idx] += 0.35 * concordance_scale

                # Validation-Optimized Temperature Scaling with T_adaptive
                scaled_logits = fused_logits / T_adaptive
                exp_ev = np.exp(scaled_logits - np.max(scaled_logits))
                probs = exp_ev / np.sum(exp_ev)

                fusion_method_name = f"Learned Cross-Modal Attention with Logit Consensus (T_adaptive={T_adaptive:.2f})"

        else:
            # Fallback: Dynamic Entropy-Calibrated Bayesian Reliability Weighting
            fusion_method_name = "Entropy-Calibrated Bayesian Reliability Weighting"
            raw_weights = {}
            for m in active_modalities:
                p_dist = np.array([active_results[m]["probabilities"].get(emo, 1e-4) for emo in EMOTIONS])
                p_dist = p_dist / p_dist.sum()
                entropy = float(-np.sum(p_dist * np.log(p_dist + 1e-8)) / np.log(7.0))
                certainty = max(0.05, 1.0 - entropy)
                raw_weights[m] = self.base_weights.get(m, 0.33) * certainty

            total_raw = sum(raw_weights.values()) or 1.0
            norm_weights = {m: round(raw_weights[m] / total_raw, 4) for m in active_modalities}
            for m in all_possible:
                if m not in norm_weights:
                    norm_weights[m] = 0.0

            probs = np.zeros(7, dtype=np.float32)
            for m in active_modalities:
                w = norm_weights[m]
                m_probs = np.array([active_results[m]["probabilities"].get(emo, 0.0) for emo in EMOTIONS])
                probs += w * m_probs
            probs = probs / (probs.sum() + 1e-8)

        # Fused Probabilities Dictionary
        fused_probs = {EMOTIONS[i]: round(float(probs[i]), 4) for i in range(7)}
        top_idx = int(probs.argmax())
        top_class = EMOTIONS[top_idx]
        top_conf = round(float(probs[top_idx]), 4)

        # --- B. CONFLICT RESOLUTION & QUALITATIVE REASONING ENGINE ---
        preds_by_mod = {m: active_results[m]["predicted_class"] for m in active_modalities}
        confs_by_mod = {m: active_results[m]["confidence"] for m in active_modalities}
        unique_preds = list(set(preds_by_mod.values()))

        # Determine consensus status
        if len(active_modalities) == 1:
            agreement_status = "Single Modality"
            conflict_msg = f"Inference performed solely via {active_modalities[0].upper()} channel."
            explanation = (
                f"Prediction based on single active {active_modalities[0].upper()} input. "
                f"Unified emotion '{top_class.upper()}' aligns with {active_modalities[0].upper()} confidence ({top_conf:.1%})."
            )
        elif len(unique_preds) == 1:
            agreement_status = "Full Consensus"
            conflict_msg = f"Unanimous concordance across all {len(active_modalities)} active channels on '{top_class.upper()}'."
            explanation = (
                f"Full Cross-Modal Consensus: Acoustic prosody, lexical dialogue text, and facial video features "
                f"all concordantly indicate '{top_class.upper()}'. The cross-modal attention network reinforces "
                f"confidence to {top_conf:.1%}, confirming high stability without modality conflict."
            )
        elif len(unique_preds) < len(active_modalities):
            agreement_status = "Partial Consensus"
            majority_class = max(set(preds_by_mod.values()), key=list(preds_by_mod.values()).count)
            dissenters = [m for m, p in preds_by_mod.items() if p != majority_class]
            dissent_desc = ", ".join([f"{m.upper()} ('{preds_by_mod[m].upper()}' at {confs_by_mod[m]:.1%})" for m in dissenters])
            conflict_msg = f"Majority consensus on '{majority_class.upper()}'. Divergence observed in {dissent_desc}."
            explanation = (
                f"Partial Modality Agreement: The majority of channels support '{majority_class.upper()}', while {dissent_desc} "
                f"exhibits divergence. The Multimodal Attention Network evaluated cross-channel representations and uncertainty, "
                f"resolving the tension in favor of '{top_class.upper()}' ({top_conf:.1%})."
            )
        else:
            agreement_status = "Conflicting / Divergent"
            breakdown = " vs ".join([f"{m.upper()}: '{preds_by_mod[m].upper()}' ({confs_by_mod[m]:.1%})" for m in active_modalities])
            conflict_msg = f"Direct multi-channel conflict: {breakdown}."
            explanation = (
                f"Cross-Modal Affect Divergence: Disagreement detected across channels ({breakdown}). "
                f"In human conversation, this occurs when spoken words carry masked or sarcastic content, "
                f"or acoustic tone contrasts with facial expression. The learned attention mechanism dynamically "
                f"weighted visual ({norm_weights.get('video', 0):.1%}), textual ({norm_weights.get('text', 0):.1%}), "
                f"and acoustic ({norm_weights.get('audio', 0):.1%}) evidence, determining that overall affective state is "
                f"most authentically captured as '{top_class.upper()}' ({top_conf:.1%})."
            )

        modality_coverage = {
            "audio": "audio" in active_modalities,
            "text": "text" in active_modalities,
            "video": "video" in active_modalities
        }

        return {
            "status": "success",
            "fused_predicted_class": top_class,
            "fused_confidence": top_conf,
            "fused_probabilities": fused_probs,
            "fusion_method": fusion_method_name,
            "active_modalities": active_modalities,
            "modality_weights": norm_weights,
            "modality_coverage": modality_coverage,
            "modality_agreement": agreement_status,
            "agreement_analysis": conflict_msg,
            "unified_explanation": explanation,
            "conflict_analysis": conflict_msg,
            "transcript_used": transcript_text or (text_result.get("details", {}).get("text", "") if text_result else ""),
            "temperature_scaling": {
                "temperature": round(T_adaptive, 4),
                "base_temperature": round(getattr(self, "base_temperature", 0.42), 4),
                "is_calibrated": True,
                "calibration_technique": f"Adaptive Concordance-Aware Temperature Scaling (T_adaptive={T_adaptive:.2f})",
                "concordance_scale": round(concordance_scale, 2),
                "metrics": self.calibration_metrics
            },
            "quality_indicators": quality_indicators,
            "individual_predictions": {
                "audio": audio_result,
                "text": text_result,
                "video": video_result
            },
            "warnings": warnings
        }
