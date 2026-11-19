import os
import sys
import time
import requests
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.services import audio_service, text_service, video_service, fusion_service

BASE_URL = "http://127.0.0.1:8000"

def run_all_tests():
    print("=" * 80)
    print("           AUTOMATED SYSTEM TEST SUITE (PHASE 9 VERIFICATION)           ")
    print("=" * 80)
    
    test_results = []
    
    # -------------------------------------------------------------
    # TEST 1: Load All Trained Models
    # -------------------------------------------------------------
    print("\n[TEST 1] Verifying all trained model checkpoints and architectures...")
    try:
        assert audio_service.is_loaded, "Audio model not loaded!"
        assert text_service.is_loaded, "Text model not loaded!"
        assert video_service.is_loaded, "Video model not loaded!"
        assert fusion_service.is_loaded, "Fusion model not loaded!"
        assert text_service.mode == "contextual", f"Text service mode is {text_service.mode}, expected 'contextual'"
        assert len(text_service.classes) == 7
        assert len(audio_service.label_mapping) == 7
        assert len(video_service.label_mapping) == 7
        print("  --> [PASS] All 4 models loaded into memory with 7 matching emotion classes.")
        test_results.append(("1. Load Trained Models", "PASS", "All 4 models active in memory"))
    except Exception as e:
        print(f"  --> [FAIL] Test 1 failed: {e}")
        test_results.append(("1. Load Trained Models", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 2: Audio Inference
    # -------------------------------------------------------------
    print("\n[TEST 2] Testing Audio Inference Service...")
    try:
        a_res = audio_service.predict_from_audio("frontend/samples/dia0_utt0.wav")
        assert a_res["status"] == "success"
        assert a_res["predicted_class"] in audio_service.label_mapping
        assert 0.0 <= a_res["confidence"] <= 1.0
        assert len(a_res["probabilities"]) == 7
        assert "quality_metrics" in a_res["details"]
        print(f"  --> [PASS] Audio predicted: {a_res['predicted_class']} ({a_res['confidence']*100:.1f}%) in {a_res['inference_time_ms']}ms")
        test_results.append(("2. Audio Inference", "PASS", f"{a_res['predicted_class']} ({a_res['confidence']*100:.1f}%)"))
    except Exception as e:
        print(f"  --> [FAIL] Test 2 failed: {e}")
        test_results.append(("2. Audio Inference", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 3: Video Inference
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing Video Inference Service...")
    try:
        v_res = video_service.predict_from_video("frontend/samples/dia0_utt0.mp4")
        assert v_res["status"] == "success"
        assert v_res["predicted_class"] == "sadness", f"Expected sadness, got {v_res['predicted_class']}"
        assert v_res["confidence"] > 0.80
        assert "quality_metrics" in v_res["details"]
        print(f"  --> [PASS] Video predicted: {v_res['predicted_class']} ({v_res['confidence']*100:.1f}%) in {v_res['inference_time_ms']}ms")
        test_results.append(("3. Video Inference", "PASS", f"{v_res['predicted_class']} ({v_res['confidence']*100:.1f}%)"))
    except Exception as e:
        print(f"  --> [FAIL] Test 3 failed: {e}")
        test_results.append(("3. Video Inference", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 4: Text Contextual Inference
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Contextual Text Inference Service...")
    try:
        t_res = text_service.predict_from_text("Oh my God, he's lost it. He's totally lost it.")
        assert t_res["status"] == "success"
        assert t_res["predicted_class"] == "sadness", f"Expected sadness, got {t_res['predicted_class']}"
        assert t_res["confidence"] > 0.45
        assert "quality_metrics" in t_res["details"]
        print(f"  --> [PASS] Text predicted: {t_res['predicted_class']} ({t_res['confidence']*100:.1f}%) in {t_res['inference_time_ms']}ms")
        test_results.append(("4. Text Inference", "PASS", f"{t_res['predicted_class']} ({t_res['confidence']*100:.1f}%)"))
    except Exception as e:
        print(f"  --> [FAIL] Test 4 failed: {e}")
        test_results.append(("4. Text Inference", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 5: Multimodal Attention Fusion
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing Multimodal Attention Fusion...")
    try:
        f_res = fusion_service.fuse_predictions(
            audio_result=a_res,
            text_result=t_res,
            video_result=v_res,
            transcript_text="Oh my God, he's lost it. He's totally lost it."
        )
        assert f_res["status"] == "success"
        assert f_res["fused_predicted_class"] == "sadness"
        assert f_res["fused_confidence"] > 0.60, f"Expected confidence > 0.60, got {f_res['fused_confidence']}"
        prob_sum = sum(f_res["fused_probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1: {prob_sum}"
        weight_sum = sum(f_res["modality_weights"].values())
        assert abs(weight_sum - 1.0) < 0.01, f"Weights do not sum to 1: {weight_sum}"
        print(f"  --> [PASS] Fused Prediction: {f_res['fused_predicted_class'].upper()} ({f_res['fused_confidence']*100:.1f}%)")
        print(f"  --> Modality Weights: {f_res['modality_weights']}")
        test_results.append(("5. Multimodal Fusion", "PASS", f"{f_res['fused_predicted_class'].upper()} ({f_res['fused_confidence']*100:.1f}%)"))
    except Exception as e:
        print(f"  --> [FAIL] Test 5 failed: {e}")
        test_results.append(("5. Multimodal Fusion", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 6: Conflicting Modalities
    # -------------------------------------------------------------
    print("\n[TEST 6] Testing Conflicting Modalities Handling...")
    try:
        joy_text = text_service.predict_from_text("This is the most joyous, happiest, most wonderful celebration!")
        f_conflict = fusion_service.fuse_predictions(
            audio_result=a_res,
            text_result=joy_text,
            video_result=v_res,
            transcript_text="This is the most joyous, happiest, most wonderful celebration!"
        )
        assert f_conflict["modality_agreement"] == "Conflicting / Divergent"
        assert f_conflict["conflict_analysis"] is not None
        assert "Cross-Modal Affect Divergence" in f_conflict["unified_explanation"]
        print(f"  --> [PASS] Conflict successfully detected: '{f_conflict['modality_agreement']}'")
        print(f"  --> Explanation: {f_conflict['conflict_analysis']}")
        test_results.append(("6. Conflicting Modalities", "PASS", "Conflict detected & explained"))
    except Exception as e:
        print(f"  --> [FAIL] Test 6 failed: {e}")
        test_results.append(("6. Conflicting Modalities", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 7: Noisy & Low-Quality Inputs
    # -------------------------------------------------------------
    print("\n[TEST 7] Testing Noisy / Low-Quality Input Handling...")
    try:
        tiny_text = text_service.predict_from_text("Hi.")
        assert "warnings" in tiny_text["details"]["quality_metrics"]
        assert tiny_text["details"]["quality_metrics"]["quality_status"] == "degraded"
        f_noisy = fusion_service.fuse_predictions(
            audio_result=None,
            text_result=tiny_text,
            video_result=v_res
        )
        assert len(f_noisy["warnings"]) > 0
        assert f_noisy["quality_indicators"]["text"]["quality_status"] == "degraded"
        print(f"  --> [PASS] Quality discount applied: {tiny_text['details']['quality_metrics']['quality_discount']}")
        print(f"  --> Warnings captured: {f_noisy['warnings']}")
        test_results.append(("7. Low-Quality Inputs", "PASS", "Quality warnings & discount applied"))
    except Exception as e:
        print(f"  --> [FAIL] Test 7 failed: {e}")
        test_results.append(("7. Low-Quality Inputs", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 8: Missing Modalities
    # -------------------------------------------------------------
    print("\n[TEST 8] Testing Missing Modalities (Unimodal & Bimodal Fallbacks)...")
    try:
        # Text Only
        f_t = fusion_service.fuse_predictions(text_result=t_res)
        assert f_t["modality_agreement"] == "Single Modality"
        assert f_t["modality_weights"]["text"] == 1.0
        assert f_t["modality_weights"]["video"] == 0.0

        # Video + Audio (No Text)
        f_va = fusion_service.fuse_predictions(video_result=v_res, audio_result=a_res)
        assert f_va["modality_weights"]["text"] == 0.0
        assert f_va["modality_weights"]["video"] > 0.0
        assert f_va["modality_weights"]["audio"] > 0.0
        print(f"  --> [PASS] Unimodal & Bimodal sub-networks function with dynamic masking.")
        test_results.append(("8. Missing Modalities", "PASS", "Dynamic masking handles all subsets"))
    except Exception as e:
        print(f"  --> [FAIL] Test 8 failed: {e}")
        test_results.append(("8. Missing Modalities", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 9: Confidence Calibration
    # -------------------------------------------------------------
    print("\n[TEST 9] Testing Post-Hoc Confidence Calibration...")
    try:
        cal_meta = f_res["temperature_scaling"]
        assert cal_meta["is_calibrated"] is True
        assert cal_meta["temperature"] > 0.20
        assert "calibration_technique" in cal_meta
        print(f"  --> [PASS] Temperature Parameter T* = {cal_meta['temperature']}")
        print(f"  --> Method: {cal_meta['calibration_technique']}")
        test_results.append(("9. Confidence Calibration", "PASS", f"T*={cal_meta['temperature']}"))
    except Exception as e:
        print(f"  --> [FAIL] Test 9 failed: {e}")
        test_results.append(("9. Confidence Calibration", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 10: Complete Frontend-to-Backend Integration
    # -------------------------------------------------------------
    print("\n[TEST 10] Testing Complete Frontend-to-Backend API Integration...")
    try:
        r_health = requests.get(f"{BASE_URL}/health", timeout=5)
        assert r_health.status_code == 200
        
        r_frontend = requests.get(f"{BASE_URL}/static/index.html", timeout=5)
        assert r_frontend.status_code == 200
        assert "Multimodal Emotion AI" in r_frontend.text
        
        with open("frontend/samples/dia0_utt0.mp4", "rb") as f_vid:
            r_post = requests.post(
                f"{BASE_URL}/predict/multimodal",
                files={"video_file": ("dia0_utt0.mp4", f_vid, "video/mp4")}
            )
        assert r_post.status_code == 200
        d_post = r_post.json()
        assert d_post["fused_predicted_class"].lower() == "sadness"
        assert d_post["fused_confidence"] > 0.60, f"Expected confidence > 0.60, got {d_post['fused_confidence']}"
        print(f"  --> [PASS] Live API successfully served frontend and resolved multimodal inference: {d_post['fused_predicted_class'].upper()} ({d_post['fused_confidence']*100:.1f}%)")
        test_results.append(("10. End-to-End Integration", "PASS", "HTTP 200 & Correct Prediction"))
    except Exception as e:
        print(f"  --> [FAIL] Test 10 failed: {e}")
        test_results.append(("10. End-to-End Integration", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 11: Real-Time Webcam Face Detection & Frame Processing
    # -------------------------------------------------------------
    print("\n[TEST 11] Testing Real-Time Webcam Face Detection & Frame Endpoint...")
    try:
        import cv2, base64
        # Test webcam status endpoint
        r_wstatus = requests.get(f"{BASE_URL}/predict/webcam/status", timeout=5)
        assert r_wstatus.status_code == 200
        wstatus_data = r_wstatus.json()
        assert wstatus_data["models_loaded"] is True
        assert wstatus_data["detector_available"] is True

        # Extract test frame from video
        cap = cv2.VideoCapture("frontend/samples/dia0_utt0.mp4")
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read sample frame from video."

        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        payload = {
            "frame": "data:image/jpeg;base64," + b64,
            "client_id": "pytest_client",
            "apply_smoothing": True
        }

        r_frame = requests.post(f"{BASE_URL}/predict/webcam/frame-json", json=payload, timeout=5)
        assert r_frame.status_code == 200
        d_frame = r_frame.json()
        assert d_frame["status"] == "success"
        assert d_frame["face_detected"] is True
        assert d_frame["apparent_expression"] is not None
        assert d_frame["bbox"] is not None
        assert 0.0 <= d_frame["calibrated_confidence"] <= 1.0
        assert d_frame["fps"] > 0
        print(f"  --> [PASS] Webcam frame processed in {d_frame['latency_ms']}ms: Face Detected={d_frame['face_detected']}, Expression={d_frame['apparent_expression']} ({d_frame['calibrated_confidence']*100:.1f}%), BBox={d_frame['bbox']['raw_px']}")
        test_results.append(("11. Real-Time Webcam Engine", "PASS", f"Face Locked, {d_frame['apparent_expression']} ({d_frame['calibrated_confidence']*100:.1f}%)"))
    except Exception as e:
        print(f"  --> [FAIL] Test 11 failed: {e}")
        test_results.append(("11. Real-Time Webcam Engine", "FAIL", str(e)))

    # -------------------------------------------------------------
    # TEST 12: Real-Time Multimodal Stream Endpoint
    # -------------------------------------------------------------
    print("\n[TEST 12] Testing Real-Time Multimodal Stream Endpoint...")
    try:
        with open("frontend/samples/dia0_utt0.wav", "rb") as f_aud:
            aud_bytes = f_aud.read()

        # Ensure frame buffer exists
        cap = cv2.VideoCapture("frontend/samples/dia0_utt0.mp4")
        ret, frame = cap.read()
        cap.release()
        _, buf = cv2.imencode(".jpg", frame)

        files = {
            "frame_file": ("frame.jpg", buf.tobytes(), "image/jpeg"),
            "audio_file": ("chunk.wav", aud_bytes, "audio/wav")
        }
        data = {
            "client_id": "pytest_client",
            "transcript_hint": "Oh my God, he's lost it. He's totally lost it."
        }
        r_mm = requests.post(f"{BASE_URL}/predict/webcam/multimodal", files=files, data=data, timeout=10)
        assert r_mm.status_code == 200
        d_mm = r_mm.json()
        assert d_mm["status"] == "success"
        assert d_mm["mode"] == "realtime_multimodal"
        assert d_mm["fused_predicted_class"] is not None
        assert d_mm["fused_confidence"] > 0.25, f"Expected confidence > 0.25, got {d_mm['fused_confidence']}"
        assert "modality_weights" in d_mm
        print(f"  --> [PASS] Real-Time Multimodal Stream: Fused={d_mm['fused_predicted_class']} ({d_mm['fused_confidence']*100:.1f}%), Weights={d_mm['modality_weights']}")
        test_results.append(("12. Real-Time Multimodal Stream", "PASS", f"Unified: {d_mm['fused_predicted_class']} ({d_mm['fused_confidence']*100:.1f}%)"))
    except Exception as e:
        print(f"  --> [FAIL] Test 12 failed: {e}")
        test_results.append(("12. Real-Time Multimodal Stream", "FAIL", str(e)))

    # -------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    AUTOMATED TEST SUITE SUMMARY                          ")
    print("=" * 80)
    all_passed = True
    for name, status, detail in test_results:
        print(f" {name:<32} | {status:<8} | {detail}")
        if status != "PASS":
            all_passed = False
    print("=" * 80)
    if all_passed:
        print(f" [SUCCESS] ALL {len(test_results)} SYSTEM TESTS PASSED WITH 100% SUCCESS RATE!")
    else:
        print(" [WARNING] Some tests encountered issues.")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()

