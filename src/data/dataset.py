"""
GhostShot Phase 2 — PyTorch Dataset class.

Reads from the split CSVs produced by splitter.py.
Each sample is one face-crop image with label {0: real, 1: deepfake}.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=16,
                        max_width=16, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class GhostShotDataset(Dataset):
    """
    Loads face crops from split CSVs.

    Each row in the CSV has:
        crop_dir  : directory containing PNG face crops for one video
        label     : 0 (real) or 1 (deepfake)
        n_crops   : how many crops are in crop_dir

    One __getitem__ call returns one randomly sampled crop from crop_dir.
    """

    def __init__(
        self,
        csv_path,
        transform=None,
        max_crops_per_video: int = 8,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.max_crops = max_crops_per_video
        self.samples = self._expand_to_crops()

    def _expand_to_crops(self) -> list:
        samples = []
        for _, row in self.df.iterrows():
            crop_dir = Path(row["crop_dir"])
            label    = int(row["label"])
            crops    = sorted(crop_dir.glob("*.png"))

            if not crops:
                continue

            selected = random.sample(crops, min(len(crops), self.max_crops))
            for crop_path in selected:
                samples.append((crop_path, label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        crop_path, label = self.samples[idx]

        img = cv2.imread(str(crop_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


def build_dataloaders(
    split_paths: dict,
    cfg: dict,
) -> dict:
    img_size   = cfg["data"]["img_size"]
    batch_size = cfg["training"]["batch_size"]

    transforms = {
        "train": get_train_transforms(img_size),
        "val":   get_val_transforms(img_size),
        "test":  get_val_transforms(img_size),
    }

    loaders = {}
    for split, path in split_paths.items():
        ds = GhostShotDataset(
            csv_path=path,
            transform=transforms[split],
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"  {split:<6} DataLoader: {len(ds):>6} samples, "
              f"{len(loaders[split]):>4} batches")

    return loaders
