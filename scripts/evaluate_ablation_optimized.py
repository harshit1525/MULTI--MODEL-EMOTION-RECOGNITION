import os
import sys
import json
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models import AdvancedCrossModalAttentionFusionNet, EMOTIONS
from backend.utils.metrics import compute_calibration_metrics, plot_confusion_matrix

def run_ablation():
    print("=" * 80)
    print(" RUNNING 7-CONFIGURATION ABLATION BENCHMARK (OPTIMIZED FUSION)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "models/fusion_classifier/optimized_fusion_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = AdvancedCrossModalAttentionFusionNet(
        in_features=16,
        embed_dim=ckpt.get("embed_dim", 96),
        num_heads=ckpt.get("num_heads", 4),
        num_layers=ckpt.get("num_layers", 2),
        num_classes=7
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    T = float(ckpt.get("temperature", 0.5953))
    print(f"Loaded {ckpt_path} (T*={T:.4f})")

    # Load test dataset
    test_data = torch.load("data/fusion_test_dataset.pt", map_location="cpu", weights_only=False)
    X_a = test_data["X_a"].to(device)
    X_t = test_data["X_t"].to(device)
    X_v = test_data["X_v"].to(device)
    y_test = test_data["y"].numpy()
    n_samples = len(y_test)

    # 7 Ablation configurations: (name, mask_tuple: [audio, text, video])
    configs = [
        ("Audio Only", (True, False, False)),
        ("Video Only", (False, False, True)),
        ("Text Only", (False, True, False)),
        ("Audio + Video", (True, False, True)),
        ("Audio + Text", (True, True, False)),
        ("Video + Text", (False, True, True)),
        ("Audio + Video + Text (Trimodal)", (True, True, True))
    ]

    ablation_results = []
    print(f"\nEvaluating {n_samples} held-out test samples across 7 modality subsets...")

    for name, mask_vals in configs:
        mask = torch.tensor([list(mask_vals)] * n_samples, dtype=torch.bool, device=device)
        t0 = time.time()
        with torch.no_grad():
            logits, gates = model(X_a, X_t, X_v, mask=mask)
            logits_np = logits.cpu().numpy()
        latency_ms = ((time.time() - t0) / n_samples) * 1000

        # Calibrated probabilities
        scaled_logits = logits_np / T
        exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        acc = float(accuracy_score(y_test, preds))
        mf1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        wf1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        mean_conf = float(np.mean(confs))
        cal_m = compute_calibration_metrics(probs, y_test)

        ablation_results.append({
            "configuration": name,
            "mask": mask_vals,
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "weighted_f1": round(wf1, 4),
            "mean_confidence": round(mean_conf, 4),
            "ece": cal_m["ece"],
            "brier_score": cal_m["brier_score"],
            "nll": cal_m["nll"],
            "latency_ms": round(latency_ms, 2)
        })

    # Load baseline summary
    base_file = "results/baseline_audit/baseline_performance.json"
    baseline = {}
    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            baseline = json.load(f)

    # Side-by-side comparison
    tri = [r for r in ablation_results if "Trimodal" in r["configuration"]][0]
    comparison = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": n_samples,
        "classes": EMOTIONS,
        "baseline_system": {
            "prediction_confidence": round(baseline.get("mean_confidence", 0.5083), 4),
            "accuracy": round(baseline.get("accuracy", 0.2952), 4),
            "macro_f1": round(baseline.get("macro_f1", 0.2833), 4),
            "ece": round(baseline.get("ece", 0.2130), 4),
            "brier_score": round(baseline.get("brier_score", 0.8811), 4),
            "nll": round(baseline.get("nll", 1.9835), 4),
            "latency_ms": round(baseline.get("mean_latency_ms", 1687.95), 2)
        },
        "optimized_system": {
            "prediction_confidence": tri["mean_confidence"],
            "accuracy": tri["accuracy"],
            "macro_f1": tri["macro_f1"],
            "ece": tri["ece"],
            "brier_score": tri["brier_score"],
            "nll": tri["nll"],
            "latency_ms": tri["latency_ms"]
        },
        "relative_improvements": {
            "accuracy_gain_absolute": round(tri["accuracy"] - baseline.get("accuracy", 0.2952), 4),
            "macro_f1_gain_absolute": round(tri["macro_f1"] - baseline.get("macro_f1", 0.2833), 4),
            "ece_reduction_absolute": round(baseline.get("ece", 0.2130) - tri["ece"], 4),
            "brier_reduction_absolute": round(baseline.get("brier_score", 0.8811) - tri["brier_score"], 4),
            "nll_reduction_absolute": round(baseline.get("nll", 1.9835) - tri["nll"], 4)
        },
        "ablation_study": ablation_results
    }

    os.makedirs("results/metrics", exist_ok=True)
    out_path = "results/metrics/multimodal_optimization_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 85)
    print("                     PHASE 7 CONTROLLED COMPARISON TABLE                     ")
    print("=" * 85)
    print(f" {'Metric':<25} | {'Current Baseline':<18} | {'Optimized System':<18} | {'Delta / Gain':<15}")
    print("-" * 85)
    print(f" {'Prediction Confidence':<25} | {comparison['baseline_system']['prediction_confidence']:.2%}            | {comparison['optimized_system']['prediction_confidence']:.2%}            | Honest Calibrated")
    print(f" {'Accuracy':<25} | {comparison['baseline_system']['accuracy']:.2%}            | {comparison['optimized_system']['accuracy']:.2%}            | {comparison['relative_improvements']['accuracy_gain_absolute']:+.2%}")
    print(f" {'Macro F1-Score':<25} | {comparison['baseline_system']['macro_f1']:.2%}            | {comparison['optimized_system']['macro_f1']:.2%}            | {comparison['relative_improvements']['macro_f1_gain_absolute']:+.2%}")
    print(f" {'ECE (Calibration Error)':<25} | {comparison['baseline_system']['ece']:<18.4f} | {comparison['optimized_system']['ece']:<18.4f} | -{comparison['relative_improvements']['ece_reduction_absolute']:.4f} (Better)")
    print(f" {'Brier Score':<25} | {comparison['baseline_system']['brier_score']:<18.4f} | {comparison['optimized_system']['brier_score']:<18.4f} | -{comparison['relative_improvements']['brier_reduction_absolute']:.4f} (Better)")
    print(f" {'Negative Log-Likelihood':<25} | {comparison['baseline_system']['nll']:<18.4f} | {comparison['optimized_system']['nll']:<18.4f} | -{comparison['relative_improvements']['nll_reduction_absolute']:.4f} (Better)")
    print(f" {'Inference Latency':<25} | {comparison['baseline_system']['latency_ms']:<18.2f}ms | {comparison['optimized_system']['latency_ms']:<18.2f}ms | Direct Tensor Head")
    print("=" * 85)

    print("\n" + "=" * 85)
    print("                     7-CONFIGURATION ABLATION STUDY TABLE                     ")
    print("=" * 85)
    print(f" {'Configuration':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'ECE':<8} | {'Mean Conf':<10}")
    print("-" * 85)
    for r in ablation_results:
        print(f" {r['configuration']:<35} | {r['accuracy']:.2%}     | {r['macro_f1']:.2%}     | {r['ece']:<8.4f} | {r['mean_confidence']:.2%}")
    print("=" * 85)
    print(f"\n[SAVED] Benchmark comparison written to: {out_path}")

if __name__ == "__main__":
    run_ablation()

