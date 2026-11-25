import requests
import cv2

cap = cv2.VideoCapture('New folder/Joy.mp4')
ret, frame = cap.read()
cap.release()

if ret:
    _, buf = cv2.imencode('.jpg', frame)
    files = {'file': ('frame.jpg', buf.tobytes(), 'image/jpeg')}
    resp = requests.post('http://127.0.0.1:8000/predict/webcam/frame', files=files, data={'client_id': 'test_verify_live'})
    print('Status:', resp.status_code)
    r = resp.json()
    print('Face detected:', r['face_detected'])
    print('Apparent Expression:', r['apparent_expression'], f"{r['calibrated_confidence']*100:.1f}%")
    print('Latency:', r['latency_ms'], 'ms')
    print('Probabilities:', r['probabilities'])
else:
    print('Failed to read frame from Joy.mp4')
