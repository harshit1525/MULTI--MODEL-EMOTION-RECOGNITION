import os
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, brier_score_loss

def compute_calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    Brier Score, and Negative Log-Likelihood (NLL).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            gap = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * gap
            mce = max(mce, gap)
            
    # Brier Score across all classes (one-hot)
    num_classes = probs.shape[1]
    one_hot_labels = np.eye(num_classes)[labels]
    brier_score = float(np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1)))
    
    # Negative Log-Likelihood
    eps = 1e-12
    nll = float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))))
    
    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier_score, 4),
        "nll": round(nll, 4)
    }

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], classes: List[str], output_path: str, title: str = "Confusion Matrix"):
    """
    Generates and saves a publication-quality confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=11)
    plt.xlabel('Predicted Emotion', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_calibration_curves(uncal_probs: np.ndarray, cal_probs: np.ndarray, labels: np.ndarray, output_path: str, n_bins: int = 10):
    """
    Plots side-by-side reliability diagrams for uncalibrated vs. calibrated probabilities.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for ax, p, title in zip(axes, [uncal_probs, cal_probs], ["Before Calibration", "After Temperature Calibration (T*=0.90)"]):
        confs = np.max(p, axis=1)
        preds = np.argmax(p, axis=1)
        accs = (preds == labels).astype(float)
        
        bin_accs, bin_confs, bin_counts = [], [], []
        for i in range(n_bins):
            in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                bin_accs.append(np.mean(accs[in_bin]))
                bin_confs.append(np.mean(confs[in_bin]))
                bin_counts.append(np.sum(in_bin))
                
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(bin_confs, bin_accs, "s-", color="#4f46e5", label="Empirical Calibration")
        ax.set_xlabel("Mean Predicted Confidence", fontsize=11)
        ax.set_ylabel("Empirical Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

