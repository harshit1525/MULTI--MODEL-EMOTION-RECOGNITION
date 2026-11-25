import os
import sys
import gc
import base64
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from fastapi.testclient import TestClient

# Ensure utf-8 stdout
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backend.main import app
from backend.services.webcam_service import webcam_service, EMOTIONS

def run_parity_comparison():
    import json
    clips_dir = Path("New folder")
    video_files = sorted(list(clips_dir.glob("*.mp4")))

    native_eval = {}
    if os.path.exists("results/eval_efficientnet_current.json"):
        with open("results/eval_efficientnet_current.json", "r", encoding="utf-8") as f:
            for c in json.load(f).get("clips", []):
                native_eval[c["clip"]] = [fr["pred"] for fr in c.get("frames", [])]

    print("=" * 95)
    print("END-TO-END PARITY: Browser Encoding (320x180 JPEG q=85) vs Direct Engine & Native Benchmark")
    print("=" * 95)

    results = {}

    for clip_idx, vpath in enumerate(video_files):
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue

        clip_name = vpath.name
        bench_client_id = f"bench_{vpath.stem}"
        api_client_id = f"api_sim_{vpath.stem}"
        form_client_id = f"form_sim_{vpath.stem}"
        ws_client_id = f"ws_sim_{vpath.stem}"
        for cid in (bench_client_id, api_client_id, form_client_id, ws_client_id):
            webcam_service.reset_session(cid)

        bench_preds = []
        api_preds = []
        max_prob_diffs = []
        matches = 0
        ws_matches = 0
        form_matches = 0
        n_frames = 0
        check_all_transports = (clip_idx == 0)  # check /frame + /ws/webcam on 1st clip too

        with TestClient(app) as client:
            ws_ctx = client.websocket_connect("/predict/webcam/ws/webcam") if check_all_transports else None
            ws = ws_ctx.__enter__() if ws_ctx is not None else None
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    n_frames += 1

                    # Exact browser encoding from frontend/js/app.js: 320x180, JPEG q=85
                    browser_frame = cv2.resize(frame, (320, 180))
                    _, buf = cv2.imencode(".jpg", browser_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    jpeg_bytes = buf.tobytes()
                    del buf

                    # 1. Direct Engine Call: webcam_service.process_frame
                    bench_res = webcam_service.process_frame(
                        bgr_frame=jpeg_bytes,
                        session_state_or_client_id=bench_client_id,
                        apply_smoothing=True,
                    )
                    b_expr = str(bench_res.get("apparent_expression", "UNKNOWN")).lower()
                    b_probs = bench_res.get("probabilities", {})
                    bench_preds.append(b_expr)

                    # 2. POST /predict/webcam/frame-json
                    b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
                    data_url = f"data:image/jpeg;base64,{b64_str}"
                    payload = {
                        "frame": data_url,
                        "client_id": api_client_id,
                        "apply_smoothing": True,
                        "calibrate": False,
                    }
                    resp = client.post("/predict/webcam/frame-json", json=payload)
                    assert resp.status_code == 200, f"API error: {resp.text}"
                    api_res = resp.json()
                    a_expr = str(api_res.get("apparent_expression", "UNKNOWN")).lower()
                    a_probs = api_res.get("probabilities", {})
                    api_preds.append(a_expr)

                    prob_diff = max(abs(float(b_probs.get(e, 0.0)) - float(a_probs.get(e, 0.0))) for e in EMOTIONS)
                    max_prob_diffs.append(prob_diff)
                    if b_expr == a_expr and prob_diff < 1e-4:
                        matches += 1

                    if check_all_transports:
                        r_form = client.post(
                            "/predict/webcam/frame",
                            files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
                            data={"client_id": form_client_id, "apply_smoothing": "true"},
                        )
                        assert r_form.status_code == 200
                        if str(r_form.json().get("apparent_expression", "")).lower() == b_expr:
                            form_matches += 1
                        ws.send_json({
                            "frame": data_url,
                            "client_id": ws_client_id,
                            "apply_smoothing": True,
                            "calibrate": False,
                        })
                        r_ws = ws.receive_json()
                        if str(r_ws.get("apparent_expression", "")).lower() == b_expr:
                            ws_matches += 1

                    del jpeg_bytes, b64_str
            finally:
                if ws_ctx is not None:
                    ws_ctx.__exit__(None, None, None)

        cap.release()
        gc.collect()

        if n_frames == 0:
            continue

        parity_pct = (matches / n_frames) * 100.0
        max_diff = max(max_prob_diffs) if max_prob_diffs else 0.0
        bench_maj = Counter(bench_preds).most_common(1)[0][0] if bench_preds else "none"
        api_maj = Counter(api_preds).most_common(1)[0][0] if api_preds else "none"

        native_preds = native_eval.get(clip_name, [])
        native_agree_pct = None
        if len(native_preds) == n_frames:
            native_agree = sum(1 for a, n in zip(api_preds, native_preds) if a == n)
            native_agree_pct = round(native_agree / n_frames * 100.0, 2)

        results[clip_name] = {
            "n_frames": n_frames,
            "matches": matches,
            "parity_pct": parity_pct,
            "max_prob_diff": max_diff,
            "bench_maj": bench_maj,
            "api_maj": api_maj,
            "native_1280x720_agree_pct": native_agree_pct,
            "bench_counts": dict(Counter(bench_preds)),
            "api_counts": dict(Counter(api_preds)),
        }
        if check_all_transports:
            results[clip_name]["form_parity_pct"] = round(form_matches / n_frames * 100.0, 2)
            results[clip_name]["ws_parity_pct"] = round(ws_matches / n_frames * 100.0, 2)

        nat_str = f"{native_agree_pct:5.1f}%" if native_agree_pct is not None else "  n/a "
        print(
            f"  {clip_name:18s} | {n_frames:3d}f | Endpoint Parity: {matches:3d}/{n_frames:3d} ({parity_pct:5.1f}%) "
            f"| Native Agree: {nat_str} | Bench:{bench_maj:8s} API:{api_maj:8s}"
        )

    print("=" * 95)
    overall_matches = sum(r["matches"] for r in results.values())
    overall_frames = sum(r["n_frames"] for r in results.values())
    overall_parity = (overall_matches / overall_frames) * 100.0 if overall_frames > 0 else 0.0
    print(f"OVERALL ENDPOINT PARITY: {overall_matches}/{overall_frames} frames ({overall_parity:.2f}%)")
    print("=" * 95)

    results_summary = {
        "overall_parity_pct": round(overall_parity, 4),
        "total_frames": overall_frames,
        "matched_frames": overall_matches,
        "clips": results,
    }
    with open("results/parity_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print("Saved parity report to results/parity_test_results.json")

    return overall_parity

if __name__ == "__main__":
    parity = run_parity_comparison()
    sys.exit(0 if parity >= 98.0 else 1)
