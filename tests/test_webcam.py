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

if __name__ == "__main__":
    test_webcam_blank_frame_no_face()
    test_webcam_face_frame_detection()
    test_webcam_session_reset()
    print("All webcam unit tests passed!")
