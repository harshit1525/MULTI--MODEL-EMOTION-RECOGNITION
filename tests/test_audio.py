import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import AudioClassifier, StrongerAudioClassifier
from backend.services import audio_service

SAMPLE_WAV = "frontend/samples/dia0_utt0.wav"

def test_audio_model_architectures():
    # Test standard AudioClassifier
    model_std = AudioClassifier(input_dim=768, hidden_dim=256, num_classes=7)
    dummy_input = torch.randn(2, 768)
    out_std = model_std(dummy_input)
    assert out_std.shape == (2, 7)

    # Test StrongerAudioClassifier
    model_strong = StrongerAudioClassifier(input_dim=768, num_classes=7)
    model_strong.eval()
    out_strong = model_strong(dummy_input)
    assert out_strong.shape == (2, 7)

def test_audio_service_inference():
    if not os.path.exists(SAMPLE_WAV):
        print(f"[SKIP] Audio sample not found at {SAMPLE_WAV}")
        return

    res = audio_service.predict_from_audio(SAMPLE_WAV)
    assert res["status"] == "success"
    assert res["modality"] == "audio"
    assert res["predicted_class"] in audio_service.label_mapping
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 7
    total_prob = sum(res["probabilities"].values())
    assert abs(total_prob - 1.0) < 0.01

if __name__ == "__main__":
    test_audio_model_architectures()
    test_audio_service_inference()
    print("All audio unit tests passed!")
