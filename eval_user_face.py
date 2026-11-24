import cv2, json
from backend.services.webcam_service import WebcamService

ws = WebcamService()
img_path = r'C:\Users\harsh\.gemini\antigravity\brain\9d238db6-c557-4b64-afff-83f3e5b3f4cb\.user_uploaded\media_1795425430205_4ef087cd.jpg'
img = cv2.imread(img_path)
if img is not None:
    res = ws.predict_frame(img, client_id='final_eval')
    print('USER RESTING FACE RESULT:')
    print(f"  Apparent Expression: {res.get('apparent_expression')}")
    print(f"  Calibrated Confidence: {res.get('calibrated_confidence')*100:.2f}%")
    print('  Probabilities:')
    for k, v in res.get('probabilities', {}).items():
        print(f"    {k:10s}: {v*100:5.2f}%")
else:
    print('Error: User image not found')
