import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torchvision
from fastapi.testclient import TestClient
from backend.main import app

def run_all_tests():
    print("=" * 80)
    print("       TRUE MULTIMODAL EMOTION RECOGNITION: COMPREHENSIVE INTEGRATION SUITE       ")
    print("=" * 80)
    
    client = TestClient(app)
    sample_wav = "frontend/samples/dia0_utt0.wav"
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if not os.path.exists(sample_wav):
        sample_wav = "audio_data/test/dia0_utt0.wav"
    if not os.path.exists(sample_mp4):
        sample_mp4 = "MELD.Raw/dev/dev_splits_complete/dia0_utt0.mp4"
    
    passed = 0
    total = 0
    
    def test(name, func):
        nonlocal passed, total
        total += 1
        t0 = time.time()
        print(f"\n[TEST {total:02d}] {name}...", end=" ", flush=True)
        try:
            func()
            elapsed = (time.time() - t0) * 1000
            print(f"PASSED ({elapsed:.1f}ms)")
            passed += 1
        except Exception as e:
            print(f"FAILED!\n   --> Error: {str(e)}")

    # 1. Health check
    def test_health():
        resp = client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"]["audio"] is True
        assert data["models_loaded"]["text"] is True
        assert data["models_loaded"]["video"] is True
        assert data["models_loaded"]["fusion"] is True
        print(f"(Device: {data['device']}, Models: Audio, Text, Video, Fusion OK)", end=" ")
    test("GET /health Endpoint & All 4 Model Checkpoints", test_health)

    # 2. Text prediction joy
    def test_text_pos():
        resp = client.post("/predict/text", json={"text": "I am so glad we accomplished this together!"})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["predicted_class"] == "joy"
        assert data["confidence"] > 0.4
        print(f"(Predicted: {data['predicted_class']} {data['confidence']:.1%})", end=" ")
    test("POST /predict/text (Joy Dialogue Sentence)", test_text_pos)

    # 3. Text prediction anger
    def test_text_anger():
        resp = client.post("/predict/text", json={"text": "Stop doing that right now, I am furious!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_class"] == "anger"
        print(f"(Predicted: {data['predicted_class']} {data['confidence']:.1%})", end=" ")
    test("POST /predict/text (Anger Expression)", test_text_anger)

    # 4. Text validation empty
    def test_text_validation():
        resp = client.post("/predict/text", json={"text": "   "})
        assert resp.status_code == 400, f"Expected 400 for empty text, got {resp.status_code}"
    test("POST /predict/text (Empty Input Error Handling)", test_text_validation)

    # 5. Audio prediction
    def test_audio():
        assert os.path.exists(sample_wav), f"WAV file not found at {sample_wav}"
        with open(sample_wav, "rb") as f:
            resp = client.post("/predict/audio", files={"file": ("sample.wav", f, "audio/wav")})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["modality"] == "audio"
        assert len(data["probabilities"]) == 7
        print(f"(Predicted: {data['predicted_class']} {data['confidence']:.1%})", end=" ")
    test("POST /predict/audio (Actual MELD Audio Sample)", test_audio)

    # 6. Video prediction
    def test_video():
        assert os.path.exists(sample_mp4), f"MP4 file not found at {sample_mp4}"
        with open(sample_mp4, "rb") as f:
            resp = client.post("/predict/video", files={"file": ("sample.mp4", f, "video/mp4")})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert data["modality"] == "video"
        assert len(data["probabilities"]) == 7
        print(f"(Predicted: {data['predicted_class']} {data['confidence']:.1%})", end=" ")
    test("POST /predict/video (Actual MELD Video Sample)", test_video)

    # 7. Multimodal: Audio + Text Bimodal
    def test_multimodal_audio_text():
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/predict/multimodal",
                data={"text_content": "Why do all your coffee mugs have numbers on the bottom?"},
                files={"audio_file": ("sample.wav", f, "audio/wav")}
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert set(data["active_modalities"]) == {"audio", "text"}
        assert data["fused_predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        print(f"(ONE Final Emotion: {data['fused_predicted_class']} {data['fused_confidence']:.1%}, Agreement: {data['modality_agreement']})", end=" ")
    test("POST /predict/multimodal (Audio + Text Bimodal Fusion)", test_multimodal_audio_text)

    # 8. Multimodal: Full Trimodal (Audio + Text + Video)
    def test_multimodal_trimodal():
        with open(sample_wav, "rb") as fa, open(sample_mp4, "rb") as fv:
            resp = client.post(
                "/predict/multimodal",
                data={"text_content": "Why do all your coffee mugs have numbers on the bottom?"},
                files={
                    "audio_file": ("dia0_utt0.wav", fa, "audio/wav"),
                    "video_file": ("dia0_utt0.mp4", fv, "video/mp4")
                }
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["active_modalities"]) == 3
        assert data["fused_predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        assert data["modality_coverage"]["audio"] is True
        assert data["modality_coverage"]["text"] is True
        assert data["modality_coverage"]["video"] is True
        assert len(data["unified_explanation"]) > 20
        print(f"(ONE Final Emotion: {data['fused_predicted_class']} {data['fused_confidence']:.1%}, Attention: {data['modality_weights']})", end=" ")
    test("POST /predict/multimodal (Full Trimodal Attention Fusion)", test_multimodal_trimodal)

    # 9. Multimodal: Single Video Input (Auto Frame Extraction + Audio Track Extraction + Speech Transcription)
    def test_multimodal_single_video():
        with open(sample_mp4, "rb") as fv:
            resp = client.post(
                "/predict/multimodal",
                files={"video_file": ("dia0_utt0.mp4", fv, "video/mp4")}
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        # All 3 modalities should be active via auto-extraction from single video!
        assert len(data["active_modalities"]) == 3
        assert data["modality_coverage"]["video"] is True
        assert data["modality_coverage"]["audio"] is True
        assert data["modality_coverage"]["text"] is True
        assert len(data["transcript_used"]) > 0
        assert data["fused_predicted_class"] is not None
        print(f"(Synced: Video+Audio+Text from 1 file -> Final: {data['fused_predicted_class']} {data['fused_confidence']:.1%})", end=" ")
    test("POST /predict/multimodal (Single Video Recording -> Auto Trimodal Extraction)", test_multimodal_single_video)

    # 10. Multimodal: Conflicting Modalities Handling & Resolution
    def test_multimodal_conflict_handling():
        # Conflicting scenario: dia0_utt0 video (Sadness 99%) + Joy text
        with open(sample_mp4, "rb") as fv:
            resp = client.post(
                "/predict/multimodal",
                data={"text_content": "I am so overjoyed and ecstatic about this wonderful news!"},
                files={"video_file": ("dia0_utt0.mp4", fv, "video/mp4")}
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["fused_predicted_class"] in ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        assert data["modality_agreement"] in ['Partial Consensus', 'Conflicting / Divergent']
        assert len(data["unified_explanation"]) > 20
        print(f"(Conflict Resolved -> Final Emotion: {data['fused_predicted_class']} {data['fused_confidence']:.1%}, Status: {data['modality_agreement']})", end=" ")
    test("POST /predict/multimodal (Conflicting Modalities Affect Resolution)", test_multimodal_conflict_handling)

    # 11. Multimodal: Missing All Inputs Error Handling
    def test_multimodal_missing():
        resp = client.post("/predict/multimodal", data={})
        assert resp.status_code == 400, f"Expected 400 for empty multimodal request, got {resp.status_code}"
    test("POST /predict/multimodal (Empty Request 400 Error Rejection)", test_multimodal_missing)

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("[SUCCESS] All true multimodal system integration tests passed with 100% compliance!")
        return 0
    else:
        print(f"[ERROR] {total - passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
