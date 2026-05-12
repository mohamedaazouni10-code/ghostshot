"""
GhostShot Phase 4 — Training loop.
AMP, cosine LR, early stopping, WandB, checkpointing.
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import wandb


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, val_auc: float) -> bool:
        if self.best is None or val_auc > self.best + self.min_delta:
            self.best    = val_auc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=cfg["training"]["amp"]):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds[:, 1])
    except ValueError:
        auc = 0.0

    acc = accuracy_score(all_labels, all_preds.argmax(axis=1))
    return {"loss": total_loss / len(loader), "auc": auc, "acc": acc}


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=cfg["training"]["amp"]):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds       = torch.softmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    try:
        auc = roc_auc_score(all_labels, all_preds[:, 1])
    except ValueError:
        auc = 0.0

    acc = accuracy_score(all_labels, all_preds.argmax(axis=1))
    return {"loss": total_loss / len(loader), "auc": auc, "acc": acc}


def save_checkpoint(model, optimizer, epoch, val_auc, cfg, tag="best"):
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ghostshot_{tag}_epoch{epoch:02d}_auc{val_auc:.4f}.pt"
    torch.save({
        "epoch":       epoch,
        "val_auc":     val_auc,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "cfg":         cfg,
    }, path)
    print(f"  [ckpt] Saved: {path}")
    return path


def train(model, loaders, criterion, cfg, device, run_name="ghostshot-run"):
    epochs     = cfg["training"]["epochs"]
    freeze_eps = cfg["model"]["freeze_backbone_epochs"]
    use_wandb  = cfg["wandb"]["enabled"]

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["training"]["lr_min"])
    scaler     = GradScaler(enabled=cfg["training"]["amp"])
    early_stop = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])
    best_auc   = 0.0
    best_ckpt  = None

    print(f"\n{'='*56}")
    print(f"  GhostShot Training — {run_name}")
    print(f"  Epochs: {epochs} | Batch: {cfg['training']['batch_size']}")
    print(f"  Freeze backbone for first {freeze_eps} epochs")
    print(f"{'='*56}\n")

    model.freeze_backbone()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if epoch == freeze_eps + 1:
            model.unfreeze_backbone()
            optimizer = AdamW(
                model.parameters(),
                lr           = cfg["training"]["lr"] * 0.1,
                weight_decay = cfg["training"]["weight_decay"],
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_eps,
                eta_min=cfg["training"]["lr_min"]
            )

        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, cfg)
        val_metrics   = evaluate(model, loaders["val"], criterion, device, cfg)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} auc {train_metrics['auc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} auc {val_metrics['auc']:.4f} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch":      epoch,
                "train/loss": train_metrics["loss"],
                "train/auc":  train_metrics["auc"],
                "train/acc":  train_metrics["acc"],
                "val/loss":   val_metrics["loss"],
                "val/auc":    val_metrics["auc"],
                "val/acc":    val_metrics["acc"],
                "lr":         lr_now,
            })

        if val_metrics["auc"] > best_auc:
            best_auc  = val_metrics["auc"]
            best_ckpt = save_checkpoint(model, optimizer, epoch, best_auc, cfg, tag="best")
            if use_wandb:
                wandb.run.summary["best_val_auc"] = best_auc

        if early_stop(val_metrics["auc"]):
            print(f"\n[early stop] No improvement for {early_stop.patience} epochs. Stopping.")
            break

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    return model
