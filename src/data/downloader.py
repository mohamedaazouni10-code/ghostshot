"""
GhostShot Phase 2 — Dataset download helpers.
All downloads go to data/raw/{dataset_name}/
"""
import os
import subprocess
from pathlib import Path


RAW_DIR = Path("data/raw")


def _makedirs(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_celebdf(raw_dir: Path = RAW_DIR) -> Path:
    out = _makedirs(raw_dir / "celebdf_v2")
    print("Celeb-DF v2: request access at https://github.com/yuezunli/celeb-deepfakeforensics")
    print(f"Once you have the link, run:\n  gdown <YOUR_LINK> -O {out}/")
    return out


def download_dfdc(raw_dir: Path = RAW_DIR) -> Path:
    out = _makedirs(raw_dir / "dfdc")
    print("Downloading DFDC preview set from Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "c2-d2/deepfake-detection-challenge-data",
        "-p", str(out), "--unzip"
    ], check=True)
    print(f"DFDC saved to {out}")
    return out


def download_ff_plus_plus(raw_dir: Path = RAW_DIR) -> Path:
    out = _makedirs(raw_dir / "faceforensics")
    print("FF++ requires access approval.")
    print("Apply at: https://github.com/ondyari/FaceForensics")
    print(f"Place downloaded folders into: {out}")
    return out


def download_ffhq_sample(n: int = 500, raw_dir: Path = RAW_DIR) -> Path:
    out = _makedirs(raw_dir / "ffhq_sample")
    print(f"Downloading {n}-image FFHQ sample...")
    subprocess.run([
        "gdown",
        "https://drive.google.com/uc?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL",
        "-O", str(out / "ffhq_thumbnails.zip")
    ], check=True)
    subprocess.run(["unzip", "-q", str(out / "ffhq_thumbnails.zip"),
                    "-d", str(out)], check=True)
    print(f"FFHQ sample saved to {out}")
    return out
