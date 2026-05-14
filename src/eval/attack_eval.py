"""
GhostShot Phase 5 — Attack Success Rate Reduction (ASRR).
Novel metric: % of successful attacks caught by GhostShot.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast


TRANSFORM = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def compute_asrr(model, attack_log_path, device, cfg):
    log = pd.read_csv(attack_log_path)
    successful = log[log["passed"] == True].copy()
    total      = len(log)
    n_success  = len(successful)

    if n_success == 0:
        print("[ASRR] No successful attacks — nothing to evaluate.")
        return {}

    original_asr = n_success / total * 100
    model.eval()
    caught = 0

    for _, row in successful.iterrows():
        probe_path = row["probe_path"]
        img = cv2.imread(probe_path)
        if img is None:
            continue
        img    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = TRANSFORM(image=img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(enabled=cfg["training"]["amp"]):
                logits = model(tensor)
                pred   = torch.softmax(logits, dim=1).argmax().item()

        if pred != 0:
            caught += 1

    remaining_asr = (n_success - caught) / total * 100
    asrr          = (original_asr - remaining_asr) / original_asr * 100 if original_asr > 0 else 0

    results = {
        "total_probes":       total,
        "successful_before":  n_success,
        "caught_by_detector": caught,
        "original_asr":       round(original_asr, 2),
        "remaining_asr":      round(remaining_asr, 2),
        "asrr":               round(asrr, 2),
    }

    print("\n── Attack Success Rate Reduction (ASRR) ──────────────")
    print(f"  Total probes         : {total}")
    print(f"  Fooled API (before)  : {n_success} ({original_asr:.1f}%)")
    print(f"  Caught by GhostShot  : {caught}")
    print(f"  Still fooling (after): {n_success-caught} ({remaining_asr:.1f}%)")
    print(f"\n  ★ ASRR = {asrr:.1f}%  (higher is better)")
    print("──────────────────────────────────────────────────────")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Before GhostShot\n(API only)", "After GhostShot\n(API + Detector)"],
        [original_asr, remaining_asr],
        color=["crimson", "steelblue"], width=0.45,
    )
    for bar, val in zip(bars, [original_asr, remaining_asr]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title(f"GhostShot reduces ASR by {asrr:.1f}%")
    ax.set_ylim(0, max(original_asr, 10) * 1.4)
    plt.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("results/figures/asrr.png", dpi=150)
    plt.show()

    return results
