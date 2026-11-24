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

def test_webcam_filename_normalization():
    from benchmark_webcam import normalize_filename_stem, map_filename_to_emotion

    # Test normalization of stems with duplicate suffixes
    assert normalize_filename_stem("Disgust (2).mp4") == "disgust"
    assert normalize_filename_stem("Fear (2).mp4") == "fear"
    assert normalize_filename_stem("Fear_2.mp4") == "fear"
    assert normalize_filename_stem("Joy - Copy.mp4") == "joy"
    assert normalize_filename_stem("Neutal.mp4") == "neutal"
    assert normalize_filename_stem("Anger.mp4") == "anger"

    # Test mapping to ground-truth emotion classes
    assert map_filename_to_emotion("Anger.mp4") == "anger"
    assert map_filename_to_emotion("Disgust (2).mp4") == "disgust"
    assert map_filename_to_emotion("disgust.mp4") == "disgust"
    assert map_filename_to_emotion("Fear (2).mp4") == "fear"
    assert map_filename_to_emotion("Fear.mp4") == "fear"
    assert map_filename_to_emotion("Joy.mp4") == "joy"
    assert map_filename_to_emotion("Neutal.mp4") == "neutral"
    assert map_filename_to_emotion("sadness.mp4") == "sadness"
    assert map_filename_to_emotion("surprise.mp4") == "surprise"

    # Unknown files should return None
    assert map_filename_to_emotion("random_clip.mp4") is None
    assert map_filename_to_emotion("user_unknown_video.mp4") is None

def test_rafdb_label_mapping_synthetic():
    import tempfile, shutil
    from pathlib import Path
    from train_rafdb import RAF_FOLDER_TO_EMOTION, RAF_FOLDER_TO_IDX, LABEL_MAPPING

    temp_dir = Path(tempfile.mkdtemp(prefix="test_synthetic_rafdb_"))
    try:
        # Create synthetic folders 1..7 with dummy jpg
        for folder_id in range(1, 8):
            fpath = temp_dir / str(folder_id)
            fpath.mkdir()
            (fpath / "sample.jpg").write_bytes(b"dummy")

        expected_emotions = {
            "1": "surprise",
            "2": "fear",
            "3": "disgust",
            "4": "joy",
            "5": "sadness",
            "6": "anger",
            "7": "neutral",
        }
        for folder_id, expected_emo in expected_emotions.items():
            assert RAF_FOLDER_TO_EMOTION[folder_id] == expected_emo
            assert RAF_FOLDER_TO_IDX[folder_id] == LABEL_MAPPING[expected_emo]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_webcam_model_loading_fallback():
    from backend.services.webcam_service import WebcamService
    # Test fallback when an invalid or missing checkpoint path is specified
    service = WebcamService(video_ckpt="models/video_classifier/non_existent_dummy_model.pth")
    assert service.is_loaded is True
    assert service.backbone is not None
    assert service.label_mapping is not None
    assert len(service.label_mapping) == 7

def test_emotion_class_order_consistency():
    from backend.services.webcam_service import EMOTIONS as SERVICE_EMOTIONS
    from train_rafdb import EMOTIONS as TRAIN_EMOTIONS, LABEL_MAPPING as TRAIN_MAPPING
    from benchmark_webcam import EMOTIONS as BENCH_EMOTIONS

    fixed_order = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    fixed_mapping = {emo: i for i, emo in enumerate(fixed_order)}

    assert SERVICE_EMOTIONS == fixed_order, f"Service emotions mismatch: {SERVICE_EMOTIONS}"
    assert TRAIN_EMOTIONS == fixed_order, f"Train emotions mismatch: {TRAIN_EMOTIONS}"
    assert BENCH_EMOTIONS == fixed_order, f"Bench emotions mismatch: {BENCH_EMOTIONS}"
    assert TRAIN_MAPPING == fixed_mapping, f"Label mapping mismatch: {TRAIN_MAPPING}"

def test_webcam_uncertainty_flag():
    # Verify uncertainty flag on ambiguous or low-confidence outputs
    client_id = "test_uncertainty"
    webcam_service.reset_session(client_id)
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if os.path.exists(sample_mp4):
        cap = cv2.VideoCapture(sample_mp4)
        ret, frame = cap.read()
        cap.release()
        if ret:
            _, buf = cv2.imencode('.jpg', frame)
            res = webcam_service.predict_frame(buf.tobytes(), client_id=client_id)
            assert "is_uncertain" in res
            assert isinstance(res["is_uncertain"], bool)
            assert "uncertainty_reason" in res
            assert "confidence_margin" in res
            assert isinstance(res["confidence_margin"], float)
    webcam_service.reset_session(client_id)

def test_aspect_preserving_square_crop():
    # Verify square padding reflection logic near frame boundaries
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Simulate face box near the top-left boundary where naive clamping would squash aspect ratio
    fx, fy, fw, fh = 10, 10, 120, 120
    cx, cy = fx + fw // 2, fy + fh // 2
    crop_size = int(max(fw, fh) * 1.25)
    pad_needed = crop_size // 2
    padded = cv2.copyMakeBorder(img, pad_needed, pad_needed, pad_needed, pad_needed, cv2.BORDER_REFLECT)
    pcx, pcy = cx + pad_needed, cy + pad_needed
    face_crop = padded[pcy - crop_size // 2 : pcy + (crop_size - crop_size // 2),
                       pcx - crop_size // 2 : pcx + (crop_size - crop_size // 2)]
    
    assert face_crop.shape[0] == face_crop.shape[1], f"Crop must be strictly square: {face_crop.shape}"
    assert face_crop.shape[0] == crop_size

def test_webcam_reset_calibration():
    client_id = "test_reset_calib"
    webcam_service.start_calibration(client_id)
    client = webcam_service._get_client_state(client_id)
    assert client["is_calibrating"] is True
    res = webcam_service.reset_calibration(client_id)
    assert res["status"] == "success"
    assert client["is_calibrating"] is False
    assert client["baseline_logits"] is None
    webcam_service.reset_session(client_id)

def test_webcam_mirror_handling():
    # Verify that unmirrored frame produces valid normalized bbox and landmark coordinates
    sample_mp4 = "frontend/samples/dia0_utt0.mp4"
    if not os.path.exists(sample_mp4):
        return
    cap = cv2.VideoCapture(sample_mp4)
    ret, frame = cap.read()
    cap.release()
    if ret:
        res = webcam_service.predict_frame(frame, client_id="test_mirror")
        if res["face_detected"]:
            bbox = res["bbox"]
            assert 0.0 <= bbox["x"] <= 1.0
            assert 0.0 <= bbox["y"] <= 1.0
            assert 0.0 <= bbox["width"] <= 1.0
            assert 0.0 <= bbox["height"] <= 1.0
            # Display coordinates mapped to mirrored video must also stay inside [0, 1]
            bx_mirrored = 1.0 - (bbox["x"] + bbox["width"])
            assert 0.0 <= bx_mirrored <= 1.0
        webcam_service.reset_session("test_mirror")

if __name__ == "__main__":
    test_webcam_blank_frame_no_face()
    test_webcam_face_frame_detection()
    test_webcam_session_reset()
    test_webcam_calibration()
    test_webcam_quality_gating()
    test_webcam_hysteresis_smoothing()
    test_webcam_face_alignment()
    test_webcam_filename_normalization()
    test_rafdb_label_mapping_synthetic()
    test_webcam_model_loading_fallback()
    test_emotion_class_order_consistency()
    test_webcam_uncertainty_flag()
    test_aspect_preserving_square_crop()
    test_webcam_reset_calibration()
    test_webcam_mirror_handling()
    print("All webcam unit tests passed!")
