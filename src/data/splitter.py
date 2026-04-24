"""
GhostShot Phase 2 — Per-identity train/val/test splitter.

Splits happen AT THE IDENTITY LEVEL — never at the frame level.
This prevents data leakage (same person in train and test).
"""
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_manifest(
    manifest_path: Path,
    out_dir: Path,
    train: float = 0.70,
    val:   float = 0.15,
    test:  float = 0.15,
    seed:  int   = 42,
) -> dict:
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    df = pd.read_csv(manifest_path)
    print(f"Manifest loaded: {len(df)} entries, "
          f"{df['identity_id'].nunique()} unique identities")

    identities = df["identity_id"].unique().tolist()
    random.seed(seed)
    random.shuffle(identities)

    train_ids, valtest_ids = train_test_split(
        identities,
        test_size=(val + test),
        random_state=seed,
    )

    val_ids, test_ids = train_test_split(
        valtest_ids,
        test_size=test / (val + test),
        random_state=seed,
    )

    splits = {
        "train": set(train_ids),
        "val":   set(val_ids),
        "test":  set(test_ids),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for split_name, id_set in splits.items():
        subset = df[df["identity_id"].isin(id_set)].reset_index(drop=True)
        out_path = out_dir / f"{split_name}.csv"
        subset.to_csv(out_path, index=False)
        paths[split_name] = out_path

        n_real = (subset["label"] == 0).sum()
        n_fake = (subset["label"] == 1).sum()
        print(f"  {split_name:<6}: {len(subset):>5} entries | "
              f"real={n_real} fake={n_fake} | "
              f"{len(id_set)} identities")

    return paths


def print_split_stats(paths: dict) -> None:
    print("\n── Split statistics ─────────────────────────────────────")
    for name, path in paths.items():
        df = pd.read_csv(path)
        total_crops = df["n_crops"].sum()
        print(f"  {name:<6} → {len(df):>4} videos, "
              f"{total_crops:>6} crops, "
              f"labels: {df['label'].value_counts().to_dict()}")
