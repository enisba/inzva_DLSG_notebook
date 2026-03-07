"""
utils/utilities.py  –  Training utilities.

Trainer, EarlyStopping, metrics, reproducibility, and I/O helpers.

TODOs in this file
------------------
  TODO 8  : Trainer.train_epoch  – one training epoch ------------- DONE
  TODO 9  : Trainer.eval_epoch   – one evaluation epoch ----------- DONE
  TODO 10 : compute_metric       – ROC-AUC and accuracy ----------- DONE
  TODO 11 : EarlyStopping        – patience-based early stopping -- DONE
 """

import csv
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# Reproducibility (do not modify)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def compute_metric(logits: np.ndarray, labels: np.ndarray, metric: str) -> float:
    """
    Compute ROC-AUC or accuracy from raw logits and labels.

    Parameters
    ----------
    logits : (N, T) or (N,) – raw model outputs (before sigmoid)
    labels : (N, T) or (N,) – ground-truth labels (may contain NaN)
    metric : 'rocauc' | 'accuracy'

    For 'rocauc'
    - Apply sigmoid to get probabilities.
    - For each task t, filter rows where labels[:, t] is not NaN.
    - Compute roc_auc_score; skip tasks with only one class present.
    - Return the mean AUC across tasks (return 0.5 if no valid task).

    For 'accuracy'
    - Apply sigmoid, threshold at 0.5 to get binary predictions.
    - Ignore NaN label positions.
    - Return fraction of correct predictions.

    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    probs = 1.0 / (1.0 + np.exp(-logits))

    if metric == "rocauc":
        aucs = []
        for t in range(labels.shape[1]):
            mask = ~np.isnan(labels[:, t])
            if mask.sum() == 0:
                continue

            y_true = labels[mask, t]
            y_prob = probs[mask, t]

            if len(np.unique(y_true)) < 2:
                continue

            aucs.append(roc_auc_score(y_true, y_prob))

        return float(np.mean(aucs)) if len(aucs) > 0 else 0.5

    if metric == "accuracy":
        preds = (probs >= 0.5).astype(np.float32)
        mask = ~np.isnan(labels)
        if mask.sum() == 0:
            return 0.0

        correct = (preds[mask] == labels[mask]).sum()
        total = mask.sum()
        return float(correct / total)

    raise ValueError(f"Unknown metric: {metric}")


class EarlyStopping:
    """
    Stop training when validation loss does not improve for `patience` epochs.
    Saves the best model weights so they can be restored after training.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model) -> bool:
        """
        Update state. Returns True when training should stop.
        Saves a deepcopy of the model whenever a new best loss is found.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model) -> None:
        """Load the best saved weights back into `model`."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)



class Trainer:
    """Wraps a model + optimizer + scheduler and exposes train/eval epoch methods."""

    def __init__(self, model, optimizer, scheduler, device: torch.device, meta: dict):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.meta      = meta

    def train_epoch(self, loader) -> float:
        """
        Run one full training epoch and return the mean binary-cross-entropy loss.

        Steps
        -----
        1. Set model to training mode.
        2. For each batch:
           a. Move batch to device.
           b. Zero gradients.
           c. Forward pass: out = model(batch.x, batch.edge_index, batch.batch,
                                        getattr(batch, 'edge_attr', None))
              out shape: (B, num_tasks)
           d. Reshape labels: y = batch.y.view(out.shape).float()
           e. Build a boolean mask for non-NaN labels.
              If mask.sum() == 0, skip this batch.
           f. Compute loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
           g. Backprop, clip gradients (max_norm=1.0), optimizer step.
           h. Accumulate loss weighted by the number of valid labels.
        3. Return total_loss / total_valid_labels.
        """
        self.model.train()
        total_loss = 0.0
        total_valid = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "edge_attr", None)
            )

            y = batch.y.view(out.shape).float()
            mask = ~torch.isnan(y)

            if mask.sum() == 0:
                continue

            loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            num_valid = int(mask.sum().item())
            total_loss += loss.item() * num_valid
            total_valid += num_valid

        if total_valid == 0:
            return 0.0

        return total_loss / total_valid

    @torch.no_grad()
    def eval_epoch(self, loader) -> tuple[float, float]:
        """
        Evaluate the model on a DataLoader.

        Returns
        -------
        mean_loss  : float  – mean binary-cross-entropy over all valid labels
        score      : float  – metric score (ROC-AUC or accuracy)

        Steps
        -----
        1. Set model to eval mode (use @torch.no_grad() decorator above).
        2. Collect all logits and labels over batches (accumulate in lists,
           concatenate after the loop).
        3. Compute mean_loss the same way as in train_epoch (use valid mask).
        4. Call compute_metric(logits, labels, metric) for the score.
        5. Return (mean_loss, score).
        """
        self.model.eval()
        total_loss = 0.0
        total_valid = 0

        all_logits = []
        all_labels = []

        for batch in loader:
            batch = batch.to(self.device)

            out = self.model(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "edge_attr", None)
            )

            y = batch.y.view(out.shape).float()
            mask = ~torch.isnan(y)

            if mask.sum() > 0:
                loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
                num_valid = int(mask.sum().item())
                total_loss += loss.item() * num_valid
                total_valid += num_valid

            all_logits.append(out.detach().cpu())
            all_labels.append(y.detach().cpu())

        if len(all_logits) == 0:
            return 0.0, 0.0

        logits = torch.cat(all_logits, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        mean_loss = total_loss / total_valid if total_valid > 0 else 0.0
        score = compute_metric(logits, labels, self.meta["metric"])

        return mean_loss, score


# I/O helpers (do not modify)

def save_curves(history: dict, metric: str, path, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=10)
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train loss")
    ax1.plot(epochs, history["val_loss"],   label="val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history["val_score"], label=f"val {metric}", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(metric.upper())
    ax2.set_title(f"Validation {metric.upper()}"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_json(obj, path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def print_table(records: list[dict]) -> None:
    metric = records[0]["metric"] if records else "score"
    header = f"{'Dataset':<20} {'Model':<10} {'Pool':<10} {'Test ' + metric.upper():>14}  {'#Params':>10}"
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in sorted(records, key=lambda x: -x.get("test_mean", x["test_score"])):
        mean = r.get("test_mean", r["test_score"])
        std  = r.get("test_std",  0.0)
        print(f"{r['dataset']:<20} {r['model']:<10} {r['pool']:<10} "
              f"{mean:.4f} ± {std:.4f}  {r['n_params']:>10,}")
    print(sep)


def save_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f"\n  Results saved to {path}")
