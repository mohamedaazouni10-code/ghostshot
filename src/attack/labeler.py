"""
GhostShot Phase 3 — Novel label creator.

Produces ghostshot_dataset.csv with 3 labels:
    0 = real face
    1 = deepfake (did NOT fool the API)
    2 = deepfake (DID fool the API — attack_success)
"""
import pandas as pd
from pathlib import Path


def build_ghostshot_dataset(
    manifest_path:   Path,
    attack_log_path: Path,
    out_path:        Path,
) -> pd.DataFrame:
    manifest   = pd.read_csv(manifest_path)
    attack_log = pd.read_csv(attack_log_path)

    # Real faces: label stays 0
    real_df = manifest[manifest["label"] == 0].copy()
    real_df["ghostshot_label"] = 0
    real_df["attack_success"]  = False
    real_df["similarity"]      = None
    real_df["generator"]       = None

    # Attack probes: label 1 (failed) or 2 (succeeded)
    attack_df = attack_log.copy()
    attack_df["ghostshot_label"] = attack_df["passed"].apply(
        lambda x: 2 if x else 1
    )
    attack_df["attack_success"] = attack_df["passed"]
    attack_df = attack_df.rename(columns={"probe_path": "crop_dir"})

    # Combine
    combined = pd.concat([
        real_df[["crop_dir", "ghostshot_label",
                 "attack_success", "similarity", "generator"]],
        attack_df[["crop_dir", "ghostshot_label",
                   "attack_success", "similarity", "generator"]],
    ], ignore_index=True).sample(frac=1, random_state=42)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print("\n── GhostShot Dataset Summary ────────────────────────")
    counts = combined["ghostshot_label"].value_counts().sort_index()
    labels = {0: "real", 1: "deepfake (blocked)", 2: "deepfake (succeeded)"}
    for lbl, count in counts.items():
        pct = count / len(combined) * 100
        print(f"  Label {lbl} — {labels[lbl]:<25}: {count:>5} ({pct:.1f}%)")
    print(f"  Total                              : {len(combined)}")
    print(f"  Saved to: {out_path}")
    print("──────────────────────────────────────────────────────")

    return combined
