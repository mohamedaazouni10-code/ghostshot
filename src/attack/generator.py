"""
GhostShot Phase 3 — Deepfake generation wrapper.
Wraps SimSwap and InstantID generators.
"""
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ATTACK_DIR = Path("data/attack_samples/generated")


def _ensure_simswap() -> Path:
    simswap_dir = Path("/content/SimSwap")
    if not simswap_dir.exists():
        print("Installing SimSwap...")
        subprocess.run([
            "git", "clone",
            "https://github.com/neuralchen/SimSwap.git",
            str(simswap_dir)
        ], check=True)
        subprocess.run(
            ["pip", "install", "-q", "-r",
             str(simswap_dir / "requirements.txt")],
            check=True
        )
        print("SimSwap ready.")
    return simswap_dir


def generate_with_simswap(
    source_img_path: str,
    target_img_path: str,
    out_path: str,
) -> bool:
    simswap_dir = _ensure_simswap()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(simswap_dir / "test_one_image.py"),
        "--isTrain", "false",
        "--name", "people",
        "--Arc_path", str(simswap_dir / "arcface_model/arcface_checkpoint.tar"),
        "--pic_a_path", source_img_path,
        "--pic_b_path", target_img_path,
        "--output_path", str(out.parent),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(simswap_dir))
    if result.returncode != 0:
        print(f"  [SimSwap] Failed: {result.stderr[-200:]}")
        return False
    return out.exists()


def generate_with_instantid(
    victim_img_path: str,
    out_path: str,
    prompt: str = "a photo of a person, high quality, realistic",
) -> bool:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[InstantID] Generating from {victim_img_path}")
    print("  → Download weights first: huggingface-cli download InstantX/InstantID")
    return False


class DeepfakeGenerator:
    def __init__(
        self,
        method: str = "simswap",
        out_dir: Path = ATTACK_DIR,
    ):
        assert method in ("simswap", "instantid"), f"Unknown method: {method}"
        self.method  = method
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        source_path: str,
        target_path: str,
        identity_id: str,
        probe_idx:   int,
    ) -> str:
        out_name = f"{identity_id}_probe{probe_idx:03d}_{self.method}.png"
        out_path = str(self.out_dir / identity_id / out_name)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        if self.method == "simswap":
            success = generate_with_simswap(source_path, target_path, out_path)
        else:
            success = generate_with_instantid(target_path, out_path)

        return out_path if success else None
