import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import ModalityEncoder, LearnedAttentionFusionNet
from backend.services import fusion_service

def test_fusion_model_architecture():
    enc = ModalityEncoder(in_features=14, embed_dim=64)
    dummy_feat = torch.randn(2, 14)
    h = enc(dummy_feat)
    assert h.shape == (2, 64)

    net = LearnedAttentionFusionNet(embed_dim=64, num_heads=4, num_classes=7)
    net.eval()
    feat_a = torch.randn(2, 14)
    feat_t = torch.randn(2, 14)
    feat_v = torch.randn(2, 14)
    mask = torch.tensor([[True, True, True], [True, False, True]])

    with torch.no_grad():
        logits, gates = net(feat_a, feat_t, feat_v, mask=mask)
    assert logits.shape == (2, 7)
    assert gates.shape == (2, 3)

def test_multimodal_service_fusion():
    # Mock audio, text, video outputs
    a_res = {"predicted_class": "sadness", "confidence": 0.85, "probabilities": {"sadness": 0.85, "neutral": 0.15}}
    t_res = {"predicted_class": "sadness", "confidence": 0.70, "probabilities": {"sadness": 0.70, "joy": 0.10, "neutral": 0.20}}
    v_res = {"predicted_class": "sadness", "confidence": 0.90, "probabilities": {"sadness": 0.90, "fear": 0.10}}

    fused = fusion_service.fuse_predictions(audio_result=a_res, text_result=t_res, video_result=v_res)
    assert fused["status"] == "success"
    assert fused["fused_predicted_class"] == "sadness"
    assert fused["fused_confidence"] > 0.60
    assert fused["modality_agreement"] in ["Full Consensus", "Partial Consensus"]
    assert "temperature_scaling" in fused
    assert fused["temperature_scaling"]["is_calibrated"] is True

def test_fusion_conflicting_modalities():
    a_res = {"predicted_class": "anger", "confidence": 0.80, "probabilities": {"anger": 0.80, "neutral": 0.20}}
    t_res = {"predicted_class": "joy", "confidence": 0.85, "probabilities": {"joy": 0.85, "neutral": 0.15}}
    v_res = {"predicted_class": "sadness", "confidence": 0.82, "probabilities": {"sadness": 0.82, "neutral": 0.18}}

    fused = fusion_service.fuse_predictions(audio_result=a_res, text_result=t_res, video_result=v_res)
    assert fused["status"] == "success"
    assert fused["modality_agreement"] == "Conflicting / Divergent"
    assert len(fused["conflict_analysis"]) > 0

if __name__ == "__main__":
    test_fusion_model_architecture()
    test_multimodal_service_fusion()
    test_fusion_conflicting_modalities()
    print("All fusion unit tests passed!")
