"""
Reproducibility utilities for GhostShot.
Call set_seed(42) at the top of every script and notebook.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[seed] All random seeds set to {seed}")


def get_device(cfg: dict | None = None) -> torch.device:
    if cfg is not None:
        requested = cfg.get("project", {}).get("device", "auto")
        if requested != "auto":
            device = torch.device(requested)
            print(f"[device] Using configured device: {device}")
            return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[device] Auto-selected: {device}")
    return device
