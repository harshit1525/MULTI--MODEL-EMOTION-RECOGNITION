"""
Multimodal Emotion Recognition System (MELD Benchmark)
======================================================
Canonical Application Entry Point & Demonstration CLI

Usage:
  python main.py                 # Launch the Web Studio (FastAPI server at http://127.0.0.1:8000)
  python main.py --serve         # Launch the Web Studio server
  python main.py --demo          # Run quick terminal demo of all 3 modalities + unified fusion
  python main.py --test          # Run full automated test suite
  python main.py --audio <file>  # Run audio emotion classifier on a WAV file
  python main.py --text <str>    # Run contextual text emotion classifier on dialogue string
  python main.py --video <file>  # Run visual expression classifier on MP4 file
"""

import os
import sys
import argparse
import time

# Ensure project root is in python module search path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def print_banner():
    print("""
================================================================================
     MULTIMODAL EMOTION RECOGNITION SYSTEM (AUDIO + VIDEO + TEXT)
================================================================================
  Benchmark : MELD (Multimodal EmotionLines Dataset - Friends TV Series)
  Classes   : 7 Emotions (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise)
  Acoustic  : Wav2Vec2-Base (768-D) + Stronger MLP Classifier
  Visual    : ResNet-50 Feature Backbone + YuNet Facial Action Unit Tracker
  Text      : All-MiniLM-L6-v2 Contextual Transformer + Calibrated Classifier
  Fusion    : Advanced Cross-Modal Attention Net + Temperature Calibration
================================================================================
""")


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Launches the FastAPI Web Studio backend."""
    import uvicorn
    print_banner()
    print(f"[*] Starting Multimodal Web Studio at http://{host}:{port}")
    print(f"[*] Interactive Web UI available in your browser at http://{host}:{port}")
    print("[*] Press Ctrl+C to shut down the server.\n")
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)


def run_demo():
    """Runs a complete end-to-end multimodal demonstration in the terminal."""
    print_banner()
    print("[*] Running Live End-to-End Multimodal Emotion Inference Demonstration...\n")

    from backend.services import audio_service, video_service, text_service, fusion_service

    sample_audio = os.path.join(PROJECT_ROOT, "frontend", "samples", "dia0_utt0.wav")
    sample_video = os.path.join(PROJECT_ROOT, "frontend", "samples", "dia0_utt0.mp4")
    sample_text = "Why do all your coffee mugs have numbers on the bottom?"

    print("-" * 70)
    print(f"1. LINGUISTIC MODALITY (Contextual NLP)")
    print(f"   Utterance : \"{sample_text}\"")
    t_res = text_service.predict_from_text(sample_text)
    print(f"   --> Predicted Emotion : {t_res['predicted_class'].upper()} ({t_res['confidence']*100:.1f}%)")
    print(f"   --> Inference Latency : {t_res['inference_time_ms']:.2f} ms")

    print("-" * 70)
    print(f"2. ACOUSTIC MODALITY (Wav2Vec2 Prosody)")
    print(f"   Audio File: {sample_audio}")
    a_res = audio_service.predict_from_audio(sample_audio)
    print(f"   --> Predicted Emotion : {a_res['predicted_class'].upper()} ({a_res['confidence']*100:.1f}%)")
    print(f"   --> Inference Latency : {a_res['inference_time_ms']:.2f} ms")

    print("-" * 70)
    print(f"3. VISUAL MODALITY (ResNet-50 Facial Expressions)")
    print(f"   Video File: {sample_video}")
    v_res = video_service.predict_from_video(sample_video, num_frames=8)
    print(f"   --> Predicted Emotion : {v_res['predicted_class'].upper()} ({v_res['confidence']*100:.1f}%)")
    print(f"   --> Inference Latency : {v_res['inference_time_ms']:.2f} ms")

    print("-" * 70)
    print(f"4. MULTIMODAL CROSS-ATTENTION FUSION (Unified Final Decision)")
    f_res = fusion_service.fuse_predictions(
        audio_result=a_res,
        text_result=t_res,
        video_result=v_res
    )
    print(f"   ============================================================")
    print(f"   >>> UNIFIED FINAL EMOTION : {f_res['fused_predicted_class'].upper()}")
    print(f"   >>> CALIBRATED CONFIDENCE : {f_res['fused_confidence']*100:.1f}%")
    print(f"   >>> CROSS-MODAL AGREEMENT : {f_res['modality_agreement']}")
    print(f"   >>> ATTENTION WEIGHTS     : {f_res['modality_weights']}")
    print(f"   >>> AGREEMENT ANALYSIS    : {f_res['agreement_analysis']}")
    print(f"   ============================================================")
    print("\n[SUCCESS] Multimodal demonstration completed successfully!\n")


def run_tests():
    """Runs the complete verification test suite."""
    from tests.run_tests import run_all_tests
    return run_all_tests()


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Emotion Recognition System — Master CLI & Application Launcher"
    )
    parser.add_argument("--serve", action="store_true", help="Launch the Web Studio FastAPI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--demo", action="store_true", help="Run full terminal multimodal demonstration")
    parser.add_argument("--test", action="store_true", help="Run the automated test suite")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav) for inference")
    parser.add_argument("--text", type=str, default=None, help="Dialogue text for linguistic inference")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (.mp4) for inference")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.test:
        run_tests()
    elif args.audio:
        from backend.services import audio_service
        res = audio_service.predict_from_audio(args.audio)
        print(f"Audio Emotion: {res['predicted_class'].upper()} ({res['confidence']*100:.1f}%)")
    elif args.text:
        from backend.services import text_service
        res = text_service.predict_from_text(args.text)
        print(f"Text Emotion: {res['predicted_class'].upper()} ({res['confidence']*100:.1f}%)")
    elif args.video:
        from backend.services import video_service
        res = video_service.predict_from_video(args.video)
        print(f"Video Emotion: {res['predicted_class'].upper()} ({res['confidence']*100:.1f}%)")
    else:
        # Default behavior: serve the Web Studio
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
