"""
GhostShot Phase 5 — Evaluation metrics.
AUC, EER, per-class metrics, confusion matrix, ROC curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast


CLASS_NAMES = {0: "Real", 1: "Fake"}


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr     = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return float(eer)


@torch.no_grad()
def run_inference(model, loader, device, cfg):
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        with autocast(enabled=cfg["training"]["amp"]):
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


def full_evaluation(probs, labels, split_name="test",
                    out_dir=Path("results"), num_classes=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(axis=1)

    # Overall metrics
    auc = roc_auc_score(labels, probs[:, 1])
    acc = accuracy_score(labels, preds)
    eer = compute_eer(labels, probs[:, 1])

    # Per-class
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )

    results = {
        "split": split_name,
        "auc":   round(auc, 4),
        "acc":   round(acc, 4),
        "eer":   round(eer, 4),
    }

    print(f"\n── Evaluation: {split_name.upper()} ──────────────────────────")
    print(f"  AUC : {auc:.4f}")
    print(f"  Acc : {acc:.4f}")
    print(f"  EER : {eer:.4f}  (lower is better)")
    print(f"\n  {'Class':<10} {'P':>6} {'R':>6} {'F1':>6} {'N':>6}")
    print(f"  {'─'*36}")
    for i in range(num_classes):
        print(f"  {CLASS_NAMES[i]:<10} "
              f"{precision[i]:>6.3f} {recall[i]:>6.3f} "
              f"{f1[i]:>6.3f} {support[i]:>6}")

    # Confusion matrix
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[CLASS_NAMES[i] for i in range(num_classes)],
                yticklabels=[CLASS_NAMES[i] for i in range(num_classes)], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {split_name}")
    plt.tight_layout()
    cm_path = out_dir / f"figures/confusion_matrix_{split_name}.png"
    cm_path.parent.mkdir(exist_ok=True)
    plt.savefig(cm_path, dpi=150)
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {split_name}")
    ax.legend()
    plt.tight_layout()
    roc_path = out_dir / f"figures/roc_curves_{split_name}.png"
    plt.savefig(roc_path, dpi=150)
    plt.show()

    return results
