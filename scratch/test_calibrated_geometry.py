import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from backend.services.webcam_service import webcam_service, EMOTIONS

clips_dir = Path("New folder")
video_files = sorted(list(clips_dir.glob("*.mp4")))

print("=" * 80)
print("EVALUATING PRINCIPLED PHYSIOLOGICAL AU GEOMETRY ON ALL CLIPS")
print("=" * 80)

for vpath in video_files:
    cap = cv2.VideoCapture(str(vpath))
    preds = []
    confs = []
    cid = f"test_{vpath.stem}"
    webcam_service.reset_session(cid)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        _, buf = cv2.imencode(".jpg", frame)
        res = webcam_service.process_frame(buf.tobytes(), session_state_or_client_id=cid, apply_smoothing=True)
        preds.append(res["apparent_expression"])
        confs.append(res["calibrated_confidence"])
    cap.release()
    
    counts = Counter(preds)
    maj, maj_n = counts.most_common(1)[0]
    total = len(preds)
    maj_pct = (maj_n / total) * 100.0
    print(f"  {vpath.name:20s} | Total: {total:3d} | Majority: {maj:8s} ({maj_pct:5.1f}%) | Counts: {dict(counts)}")
