import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import cv2
from torchvision import transforms
import torchvision.models as models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import (
    AudioClassifier,
    StrongerAudioClassifier,
    VideoClassifier,
    LearnedAttentionFusionNet,
    TextPreprocessingUtils,
    EMOTIONS
)
from backend.services.fusion_service import MultimodalFusionService
from backend.services import audio_service, video_service, text_service, fusion_service
from backend.utils.metrics import compute_calibration_metrics, plot_confusion_matrix

EMOTION_TO_IDX = {emo: i for i, emo in enumerate(EMOTIONS)}

def run_baseline_benchmark(n_per_emotion: int = 15):
    print("=" * 80)
    print("        MEASURING CURRENT BASELINE SYSTEM PERFORMANCE        ")
    print(f" Dataset: Held-Out MELD Test Set ({n_per_emotion} samples per emotion = {n_per_emotion * 7} total)")
    print("=" * 80)

    df_test = pd.read_csv("MELD.Raw/test_sent_emo.csv")
    test_audio = torch.load("audio_embeddings/test_embeddings.pt", map_location="cpu", weights_only=False)
    audio_lookup = {d["sample_id"]: d["audio_embedding"] for d in test_audio}
    video_dir = "MELD.Raw/test/output_repeated_splits_test"

    # Select balanced samples per emotion
    sampled_records = []
    for emo in EMOTIONS:
        sub = df_test[df_test["Emotion"].str.lower() == emo]
        collected = 0
        for _, row in sub.iterrows():
            d_id = row["Dialogue_ID"]
            u_id = row["Utterance_ID"]
            sid = f"dia{d_id}_utt{u_id}"
            vpath = os.path.join(video_dir, f"{sid}.mp4")
            utt = str(row["Utterance"])
            if sid in audio_lookup and os.path.exists(vpath) and len(utt.strip()) > 0:
                sampled_records.append({
                    "sample_id": sid,
                    "emotion": emo,
                    "utterance": utt,
                    "audio_emb": audio_lookup[sid],
                    "video_path": vpath
                })
                collected += 1
                if collected >= n_per_emotion:
                    break

    print(f"Successfully assembled {len(sampled_records)} balanced multimodal test samples.")

    # Run predictions with current baseline fusion_service
    y_true = []
    y_pred = []
    confidences = []
    all_probs = []
    latencies = []

    print("\nRunning inference through baseline MultimodalFusionService...")
    t_start = time.time()
    for idx, rec in enumerate(sampled_records):
        t0 = time.time()
        emo_label = rec["emotion"]
        y_true.append(EMOTION_TO_IDX[emo_label])

        # 1. Text prediction
        t_res = text_service.predict_from_text(rec["utterance"])

        # 2. Audio prediction via direct classifier on embedding
        a_logits = audio_service.classifier(rec["audio_emb"].unsqueeze(0).to(audio_service.device))
        a_probs = torch.softmax(a_logits, dim=-1)[0].cpu().detach().numpy()
        a_pred_idx = int(a_probs.argmax())
        a_res = {
            "status": "success",
            "modality": "audio",
            "predicted_class": EMOTIONS[a_pred_idx],
            "confidence": float(a_probs[a_pred_idx]),
            "probabilities": {EMOTIONS[i]: float(a_probs[i]) for i in range(7)},
            "details": {"quality_metrics": {"quality_discount": 1.0, "warnings": []}}
        }

        # 3. Video prediction via VideoService
        v_res = video_service.predict_from_video(rec["video_path"], num_frames=8, generate_preview=False)

        # 4. Multimodal Fusion
        f_res = fusion_service.fuse_predictions(
            audio_result=a_res,
            text_result=t_res,
            video_result=v_res,
            transcript_text=rec["utterance"]
        )

        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        f_pred_class = f_res["fused_predicted_class"]
        f_conf = f_res["fused_confidence"]
        f_probs_dict = f_res["fused_probabilities"]
        p_vec = np.array([f_probs_dict[e] for e in EMOTIONS], dtype=np.float32)

        y_pred.append(EMOTION_TO_IDX[f_pred_class])
        confidences.append(f_conf)
        all_probs.append(p_vec)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(sampled_records):
            print(f"  Processed {idx + 1}/{len(sampled_records)} samples... Current mean conf: {np.mean(confidences):.1%}")

    total_time = time.time() - t_start
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_probs = np.array(all_probs)

    # Calculate metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mean_conf = float(np.mean(confidences))
    median_conf = float(np.median(confidences))
    mean_lat = float(np.mean(latencies))

    cal_metrics = compute_calibration_metrics(all_probs, y_true)

    print("\n" + "=" * 80)
    print("               CURRENT BASELINE BENCHMARK RESULTS               ")
    print("=" * 80)
    print(f" Test Samples Evaluated : {len(y_true)}")
    print(f" Accuracy               : {acc:.2%}")
    print(f" Macro F1-Score         : {macro_f1:.2%}")
    print(f" Weighted F1-Score      : {weighted_f1:.2%}")
    print(f" Mean Confidence        : {mean_conf:.1%}")
    print(f" Median Confidence      : {median_conf:.1%}")
    print(f" Expected Cal. Error    : {cal_metrics['ece']:.4f}")
    print(f" Brier Score            : {cal_metrics['brier_score']:.4f}")
    print(f" Negative Log-Likelihood: {cal_metrics['nll']:.4f}")
    print(f" Mean Latency           : {mean_lat:.2f} ms")
    print("=" * 80)

    # Save baseline report
    os.makedirs("results/baseline_audit", exist_ok=True)
    baseline_summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "ece": cal_metrics["ece"],
        "brier_score": cal_metrics["brier_score"],
        "nll": cal_metrics["nll"],
        "mean_latency_ms": mean_lat,
        "classification_report": classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True)
    }

    with open("results/baseline_audit/baseline_performance.json", "w") as f:
        json.dump(baseline_summary, f, indent=2)

    plot_confusion_matrix(
        y_true, y_pred, EMOTIONS,
        "results/baseline_audit/baseline_confusion_matrix.png",
        title="Baseline Multimodal Fusion (Pre-Optimization)"
    )

    print(f"\n[SAVED] Baseline audit written to results/baseline_audit/baseline_performance.json")
    return baseline_summary

if __name__ == "__main__":
    run_baseline_benchmark(n_per_emotion=15)
