"""
GhostShot Phase 4 — Loss functions.
Focal loss + class weights for imbalanced dataset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path


def compute_class_weights(dataset_csv: Path, num_classes: int = 2) -> torch.Tensor:
    df      = pd.read_csv(dataset_csv)
    col     = "label" if "label" in df.columns else "ghostshot_label"
    counts  = df[col].value_counts().sort_index()
    total   = len(df)

    weights = []
    for cls in range(num_classes):
        n = counts.get(cls, 1)
        weights.append(total / (num_classes * n))

    weights = torch.tensor(weights, dtype=torch.float32)
    print(f"[loss] Class weights: {weights.numpy().round(3)}")
    return weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(
            logits, labels,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            reduction="none",
        )
        pt    = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(dataset_csv: Path, cfg: dict) -> nn.Module:
    weights = compute_class_weights(dataset_csv, cfg["data"]["num_classes"])
    return FocalLoss(gamma=2.0, weight=weights)
