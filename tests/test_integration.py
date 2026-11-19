import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

SAMPLE_WAV = "frontend/samples/dia0_utt0.wav"
if not os.path.exists(SAMPLE_WAV):
    SAMPLE_WAV = "audio_data/test/dia0_utt0.wav"

SAMPLE_MP4 = "frontend/samples/dia0_utt0.mp4"
if not os.path.exists(SAMPLE_MP4):
    SAMPLE_MP4 = "MELD.Raw/dev/dev_splits_complete/dia0_utt0.mp4"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["models_loaded"]["audio"] is True
    assert data["models_loaded"]["text"] is True
    assert data["models_loaded"]["video"] is True
    assert data["models_loaded"]["fusion"] is True

def test_predict_text_valid():
    payload = {"text": "I am so happy and thrilled that our project is complete!"}
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["modality"] == "text"
    assert data["predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    assert data["confidence"] > 0
    assert len(data["probabilities"]) == 7

def test_predict_text_empty():
    response = client.post("/predict/text", json={"text": "   "})
    assert response.status_code == 400

def test_predict_audio_valid():
    if not os.path.exists(SAMPLE_WAV):
        pytest.skip(f"Test sample {SAMPLE_WAV} not found")
        
    with open(SAMPLE_WAV, "rb") as f:
        response = client.post(
            "/predict/audio",
            files={"file": ("test_audio.wav", f, "audio/wav")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["modality"] == "audio"
    assert data["predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    assert len(data["probabilities"]) == 7
    assert data["confidence"] > 0

def test_predict_video_valid():
    if not os.path.exists(SAMPLE_MP4):
        pytest.skip(f"Test sample {SAMPLE_MP4} not found")
        
    with open(SAMPLE_MP4, "rb") as f:
        response = client.post(
            "/predict/video",
            files={"file": ("test_video.mp4", f, "video/mp4")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["modality"] == "video"
    assert data["predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    assert len(data["probabilities"]) == 7

def test_predict_multimodal_audio_text():
    if not os.path.exists(SAMPLE_WAV):
        pytest.skip("Test WAV missing")
        
    with open(SAMPLE_WAV, "rb") as f:
        response = client.post(
            "/predict/multimodal",
            data={"text_content": "Why do all your coffee mugs have numbers on the bottom?"},
            files={"audio_file": ("test.wav", f, "audio/wav")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["fused_predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    assert "audio" in data["active_modalities"]
    assert "text" in data["active_modalities"]
    assert data["modality_agreement"] in ["Full Consensus", "Partial Consensus", "Divergent", "Conflicting / Divergent"]
    assert len(data["fused_probabilities"]) == 7

def test_predict_multimodal_trimodal():
    if not os.path.exists(SAMPLE_WAV) or not os.path.exists(SAMPLE_MP4):
        pytest.skip("Sample files missing")
        
    with open(SAMPLE_WAV, "rb") as fa, open(SAMPLE_MP4, "rb") as fv:
        response = client.post(
            "/predict/multimodal",
            data={"text_content": "Why do all your coffee mugs have numbers on the bottom?"},
            files={
                "audio_file": ("test.wav", fa, "audio/wav"),
                "video_file": ("test.mp4", fv, "video/mp4")
            }
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["active_modalities"]) == 3
    assert data["fused_confidence"] > 0
    assert "text" in data["individual_predictions"]
    assert "audio" in data["individual_predictions"]
    assert "video" in data["individual_predictions"]

def test_predict_multimodal_single_video_auto_sync():
    if not os.path.exists(SAMPLE_MP4):
        pytest.skip("Sample MP4 missing")

    with open(SAMPLE_MP4, "rb") as fv:
        response = client.post(
            "/predict/multimodal",
            files={"video_file": ("dia0_utt0.mp4", fv, "video/mp4")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["modality_coverage"]["video"] is True
    assert data["modality_coverage"]["audio"] is True
    assert data["modality_coverage"]["text"] is True
    assert data["fused_predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    assert len(data["unified_explanation"]) > 0
