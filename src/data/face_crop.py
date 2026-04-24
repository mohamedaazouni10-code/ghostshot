"""
GhostShot Phase 2 — Face extraction pipeline.

For every video (or image) in data/raw/, this module:
  1. Detects faces using MTCNN
  2. Aligns and crops to 224x224
  3. Saves crops to data/processed/{dataset}/{label}/{identity_id}/
  4. Writes a manifest CSV for the splitter

Label convention:
  0 = real
  1 = deepfake
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


IMG_SIZE  = 224
FRAMES_PER_VIDEO = 32
MIN_FACE_CONFIDENCE = 0.90


def get_mtcnn(device: torch.device) -> MTCNN:
    return MTCNN(
        image_size=IMG_SIZE,
        margin=20,
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.80],
        factor=0.709,
        keep_all=False,
        device=device,
    )


def sample_frames(video_path: str, n: int = FRAMES_PER_VIDEO) -> list:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def crop_and_save(
    mtcnn: MTCNN,
    frames: list,
    out_dir: Path,
    video_stem: str,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i, frame in enumerate(frames):
        pil_img = Image.fromarray(frame)
        try:
            face_tensor, prob = mtcnn(pil_img, return_prob=True)
        except Exception:
            continue

        if face_tensor is None or prob < MIN_FACE_CONFIDENCE:
            continue

        face_np = (
            face_tensor.permute(1, 2, 0).numpy() * 128 + 127.5
        ).clip(0, 255).astype(np.uint8)

        out_path = out_dir / f"{video_stem}_f{i:04d}.png"
        Image.fromarray(face_np).save(out_path)
        saved += 1

    return saved


def process_dataset(
    raw_dir,
    processed_dir: Path,
    dataset_name: str,
    label: int,
    device: torch.device,
    manifest_rows: list,
    video_extensions: tuple = (".mp4", ".avi", ".mov"),
) -> None:
    mtcnn = get_mtcnn(device)
    video_paths = [
        p for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in video_extensions
    ]

    print(f"\nProcessing '{dataset_name}' — {len(video_paths)} videos, label={label}")

    for vp in tqdm(video_paths, desc=dataset_name):
        identity_id = vp.stem
        out_dir = processed_dir / dataset_name / str(label) / identity_id
        frames  = sample_frames(str(vp))
        n_saved = crop_and_save(mtcnn, frames, out_dir, vp.stem)

        if n_saved > 0:
            manifest_rows.append({
                "identity_id":  identity_id,
                "dataset":      dataset_name,
                "label":        label,
                "crop_dir":     str(out_dir),
                "n_crops":      n_saved,
            })


def write_manifest(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["identity_id", "dataset", "label", "crop_dir", "n_crops"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nManifest written: {out_path} ({len(rows)} entries)")
