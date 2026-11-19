import os
import sys
import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_unit_tests():
    print("=" * 80)
    print("                PHASE 1: MULTIMODAL SUBSYSTEM UNIT TESTS                ")
    print("=" * 80)
    
    passed = 0
    total = 0
    
    def execute_suite(name, func):
        nonlocal passed, total
        total += 1
        t0 = time.time()
        print(f"\n[UNIT {total:02d}] {name}...", end=" ", flush=True)
        try:
            func()
            elapsed = (time.time() - t0) * 1000
            print(f"PASSED ({elapsed:.1f}ms)")
            passed += 1
        except Exception as e:
            print(f"FAILED!\n   --> Error: {str(e)}")

    # 1. Audio unit tests
    def test_audio():
        from tests import test_audio as ta
        ta.test_audio_model_architectures()
        ta.test_audio_service_inference()
    execute_suite("Audio Architecture & Feature Inference (Wav2Vec 2.0)", test_audio)

    # 2. Text unit tests
    def test_text():
        from tests import test_text as tt
        tt.test_text_preprocessing()
        tt.test_text_inference_valid()
        tt.test_text_inference_empty_error()
    execute_suite("Text Preprocessing & Contextual MiniLM Classification", test_text)

    # 3. Video unit tests
    def test_video():
        from tests import test_video as tv
        tv.test_video_model_architecture()
        tv.test_video_service_inference()
    execute_suite("Video Architecture & Spatial-Temporal ResNet-50 Features", test_video)

    # 4. Fusion unit tests
    def test_fusion():
        from tests import test_fusion as tf
        tf.test_fusion_model_architecture()
        tf.test_multimodal_service_fusion()
        tf.test_fusion_conflicting_modalities()
    execute_suite("Cross-Modal Attention Fusion & Conflict Arbitration", test_fusion)

    # 5. Webcam unit tests
    def test_webcam():
        from tests import test_webcam as tw
        tw.test_webcam_blank_frame_no_face()
        tw.test_webcam_face_frame_detection()
        tw.test_webcam_session_reset()
    execute_suite("Real-Time Webcam Face Tracking (YuNet) & Landmark Engine", test_webcam)

    print("\n" + "-" * 80)
    print(f"Unit Tests Summary: {passed}/{total} suites passed ({passed/total*100:.1f}%)")
    print("-" * 80)
    return passed == total


def run_integration_tests():
    print("\n" + "=" * 80)
    print("             PHASE 2: FASTAPI TRUE MULTIMODAL INTEGRATION SUITE         ")
    print("=" * 80)

    from fastapi.testclient import TestClient
    from backend.main import app
    
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
        print(f"\n[INTEGRATION {total:02d}] {name}...", end=" ", flush=True)
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
        print(f"(Device: {data['device']}, 4 Models Loaded)", end=" ")
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
        print(f"(Final Emotion: {data['fused_predicted_class']} {data['fused_confidence']:.1%}, Agreement: {data['modality_agreement']})", end=" ")
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
        print(f"(Final Emotion: {data['fused_predicted_class']} {data['fused_confidence']:.1%}, Attention: {data['modality_weights']})", end=" ")
    test("POST /predict/multimodal (Full Trimodal Attention Fusion)", test_multimodal_trimodal)

    # 9. Multimodal: Single Video Input (Auto Frame + Audio + Text Extraction)
    def test_multimodal_single_video():
        with open(sample_mp4, "rb") as fv:
            resp = client.post(
                "/predict/multimodal",
                files={"video_file": ("dia0_utt0.mp4", fv, "video/mp4")}
            )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["active_modalities"]) == 3
        assert data["modality_coverage"]["video"] is True
        assert data["modality_coverage"]["audio"] is True
        assert data["modality_coverage"]["text"] is True
        assert len(data["transcript_used"]) > 0
        assert data["fused_predicted_class"] is not None
        print(f"(Synced Trimodal -> Final: {data['fused_predicted_class']} {data['fused_confidence']:.1%})", end=" ")
    test("POST /predict/multimodal (Single Video Recording -> Auto Trimodal Extraction)", test_multimodal_single_video)

    # 10. Multimodal: Conflicting Modalities Handling & Resolution
    def test_multimodal_conflict_handling():
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

    print("\n" + "-" * 80)
    print(f"Integration Tests Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("-" * 80)
    return passed == total


def run_live_system_tests(base_url="http://127.0.0.1:8000"):
    print("\n" + "=" * 80)
    print(f"         PHASE 3: LIVE HTTP & WEBCAM SYSTEM VERIFICATION ({base_url})")
    print("=" * 80)
    import requests
    import cv2
    import base64

    passed = 0
    total = 0

    def test(name, func):
        nonlocal passed, total
        total += 1
        t0 = time.time()
        print(f"\n[LIVE {total:02d}] {name}...", end=" ", flush=True)
        try:
            func()
            elapsed = (time.time() - t0) * 1000
            print(f"PASSED ({elapsed:.1f}ms)")
            passed += 1
        except Exception as e:
            print(f"FAILED!\n   --> Error: {str(e)}")

    def test_live_health():
        r = requests.get(f"{base_url}/health", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "healthy"
    test("Live /health Check", test_live_health)

    def test_live_webcam_status():
        r = requests.get(f"{base_url}/predict/webcam/status", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert d["models_loaded"] is True
    test("Live Webcam Endpoint Status", test_live_webcam_status)

    def test_live_webcam_frame():
        sample_mp4 = "frontend/samples/dia0_utt0.mp4"
        cap = cv2.VideoCapture(sample_mp4)
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read sample frame from video."
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        payload = {"frame": "data:image/jpeg;base64," + b64, "client_id": "live_test", "apply_smoothing": True}
        r = requests.post(f"{base_url}/predict/webcam/frame-json", json=payload, timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "success"
        assert d["face_detected"] is True
    test("Live Webcam Frame Tracking (YuNet)", test_live_webcam_frame)

    print("\n" + "-" * 80)
    print(f"Live System Tests Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("-" * 80)
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition Test Suite Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--live", action="store_true", help="Run live HTTP/Webcam system tests against 127.0.0.1:8000")
    parser.add_argument("--all", action="store_true", help="Run all unit and integration tests (default)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Base URL for live tests")
    args = parser.parse_args()

    # Default to running both unit and integration tests
    run_all = args.all or (not args.unit and not args.integration and not args.live)
    
    overall_success = True
    
    if args.unit or run_all:
        u_ok = run_unit_tests()
        overall_success = overall_success and u_ok

    if args.integration or run_all:
        i_ok = run_integration_tests()
        overall_success = overall_success and i_ok

    if args.live:
        l_ok = run_live_system_tests(base_url=args.url)
        overall_success = overall_success and l_ok

    print("\n" + "=" * 80)
    if overall_success:
        print("          ALL TEST SUITES EXECUTED SUCCESSFULLY WITH 100% PASS RATE!     ")
    else:
        print("          ONE OR MORE TEST SUITES REPORTED FAILURES.                    ")
    print("=" * 80)
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
