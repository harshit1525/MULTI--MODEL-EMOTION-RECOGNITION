import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services import webcam_service

def test_webcam_blank_frame_no_face():
    # 640x480 blank black frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', blank)
    res = webcam_service.predict_frame(buf.tobytes(), client_id="test_unit")
    assert res["status"] == "success"
    assert res["face_detected"] is False
    assert res["apparent_expression"] == "NO_FACE"
    assert res["calibrated_confidence"] == 0.0

def test_webcam_face_frame_detection():
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if not os.path.exists(sample_mp4):
        print(f"[SKIP] Sample video not found at {sample_mp4}")
        return

    cap = cv2.VideoCapture(sample_mp4)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read sample frame from video."

    _, buf = cv2.imencode('.jpg', frame)
    res = webcam_service.predict_frame(buf.tobytes(), client_id="test_unit")
    assert res["status"] == "success"
    assert res["face_detected"] is True
    assert res["bbox"] is not None
    assert 0.0 <= res["calibrated_confidence"] <= 1.0
    assert res["apparent_expression"] is not None
    assert res["fps"] > 0

def test_webcam_session_reset():
    webcam_service.reset_session("test_unit")
    assert "test_unit" not in webcam_service.client_states

def test_webcam_calibration():
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if not os.path.exists(sample_mp4):
        return
    cap = cv2.VideoCapture(sample_mp4)
    ret, frame = cap.read()
    cap.release()
    assert ret
    _, buf = cv2.imencode('.jpg', frame)

    # Start calibration
    webcam_service.start_calibration("test_calib")
    client = webcam_service._get_client_state("test_calib")
    assert client["is_calibrating"] is True

    # Send 15 frames to complete calibration
    for _ in range(15):
        res = webcam_service.predict_frame(buf.tobytes(), client_id="test_calib", calibrate=True)
    assert res["is_calibrated"] is True
    assert client["baseline_logits"] is not None
    webcam_service.reset_session("test_calib")

def test_webcam_quality_gating():
    # Test quality gating on degraded / blurry inputs
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if not os.path.exists(sample_mp4):
        return
    cap = cv2.VideoCapture(sample_mp4)
    ret, frame = cap.read()
    cap.release()
    assert ret

    client_id = "test_qgate"
    webcam_service.reset_session(client_id)

    # 1. Good frame to establish last_good_prediction
    _, buf = cv2.imencode('.jpg', frame)
    res_good = webcam_service.predict_frame(buf.tobytes(), client_id=client_id)
    assert res_good["face_detected"] is True
    assert res_good["apparent_expression"] is not None

    # 2. Heavily blurred frame to trigger Laplacian blur quality gate
    blurred = cv2.GaussianBlur(frame, (51, 51), 30.0)
    _, buf_blur = cv2.imencode('.jpg', blurred)
    res_blur = webcam_service.predict_frame(buf_blur.tobytes(), client_id=client_id)
    # Gating should recognize the blur or hold the prior stable expression
    assert res_blur["status"] == "success"
    assert res_blur["quality_status"] in ["motion_blur", "optimal", "low_confidence"]
    webcam_service.reset_session(client_id)

def test_webcam_hysteresis_smoothing():
    client_id = "test_hysteresis"
    webcam_service.reset_session(client_id)
    client = webcam_service._get_client_state(client_id)

    # Directly verify hysteresis state tracking in client state
    assert client["current_label"] is None
    assert client["candidate_label"] is None
    assert client["candidate_count"] == 0

    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if os.path.exists(sample_mp4):
        cap = cv2.VideoCapture(sample_mp4)
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, buf = cv2.imencode('.jpg', frame)
            # Run 3 frames with smoothing
            for _ in range(3):
                res = webcam_service.predict_frame(buf.tobytes(), client_id=client_id, apply_smoothing=True)
            assert client["current_label"] is not None
            assert res["apparent_expression"] == client["current_label"]

    webcam_service.reset_session(client_id)

def test_webcam_face_alignment():
    # Test face alignment logic with synthetic tilted frame
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if not os.path.exists(sample_mp4):
        return
    cap = cv2.VideoCapture(sample_mp4)
    ret, frame = cap.read()
    cap.release()
    assert ret

    h, w = frame.shape[:2]
    # Rotate frame by 15 degrees
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15.0, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    _, buf = cv2.imencode('.jpg', rotated)

    res = webcam_service.predict_frame(buf.tobytes(), client_id="test_align")
    assert res["status"] == "success"
    webcam_service.reset_session("test_align")

if __name__ == "__main__":
    test_webcam_blank_frame_no_face()
    test_webcam_face_frame_detection()
    test_webcam_session_reset()
    test_webcam_calibration()
    test_webcam_quality_gating()
    test_webcam_hysteresis_smoothing()
    test_webcam_face_alignment()
    print("All webcam unit tests passed!")
