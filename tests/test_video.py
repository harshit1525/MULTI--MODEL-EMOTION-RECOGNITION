import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import VideoClassifier
from backend.services import video_service

SAMPLE_MP4 = "frontend/samples/dia0_utt0.mp4"

def test_video_model_architecture():
    model = VideoClassifier(input_dim=2048, hidden_dim=256, num_classes=7)
    dummy_input = torch.randn(2, 2048)
    out = model(dummy_input)
    assert out.shape == (2, 7)

def test_video_service_inference():
    if not os.path.exists(SAMPLE_MP4):
        print(f"[SKIP] Video sample not found at {SAMPLE_MP4}")
        return

    res = video_service.predict_from_video(SAMPLE_MP4, num_frames=4)
    assert res["status"] == "success"
    assert res["modality"] == "video"
    assert res["predicted_class"] in video_service.label_mapping
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 7
    assert "quality_metrics" in res["details"]

if __name__ == "__main__":
    test_video_model_architecture()
    test_video_service_inference()
    print("All video unit tests passed!")
