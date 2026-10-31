"""
Training Loop and Validation with Checkpointing (Ivanovs & Ivanova, Baltic DB&IS 2026)
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- Batch size: 256
- Epochs: 100
- Checkpointing: Saves the model with the highest Validation Macro F1 score
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from config import (
    CHECKPOINTS_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    RESULTS_DIR,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())

    total_samples = len(all_targets)
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = (
        f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    )

    return epoch_loss, acc, macro_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
):
    """Evaluate on validation or test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())

    total_samples = len(all_targets)
    eval_loss = running_loss / total_samples if total_samples > 0 else 0.0
    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = (
        f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    )

    return eval_loss, acc, macro_f1, np.array(all_preds), np.array(all_targets)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "temporal_cnn",
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LEARNING_RATE,
    device: str | None = None,
    save_prefix: str = "best_model",
):
    """
    Complete training pipeline matching paper parameters.
    Saves checkpoint with highest Validation Macro F1.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Training {model_name} on device: {device}")
    model = model.to(device)

    # Compute balanced class weights to prevent majority class (Class 0) dominance
    from config import NUM_CLASSES

    labels_tensor = train_loader.dataset.labels
    class_counts = torch.bincount(labels_tensor, minlength=NUM_CLASSES).float()
    weights = len(labels_tensor) / (NUM_CLASSES * class_counts + 1e-5)
    weights = weights / weights.sum() * NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_f1 = -1.0
    best_epoch = -1
    checkpoint_path = CHECKPOINTS_DIR / f"{save_prefix}_{model_name}.pt"

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "best_epoch": -1,
        "best_val_f1": 0.0,
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Checkpoint selection by highest validation macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "model_name": model_name,
                },
                checkpoint_path,
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2f}%, F1: {tr_f1:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}% | "
                f"Best Val F1: {best_val_f1:.2f}% (Epoch {best_epoch})"
            )

    elapsed = time.time() - start_time
    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_val_f1
    history["training_time_sec"] = elapsed

    # Save training history
    history_path = RESULTS_DIR / f"history_{save_prefix}_{model_name}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(
        f"\nTraining completed in {elapsed:.1f}s. Best checkpoint saved to {checkpoint_path}"
    )
    return checkpoint_path, history
