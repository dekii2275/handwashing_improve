"""
Visualization Utilities:
1. Training Loss & Macro F1 curves over epochs
2. Confusion Matrix heatmap for the 7 WHO hand-washing classes
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from config import CLASS_NAMES, RESULTS_DIR


def plot_training_history(history_file: Path, output_file: Path | None = None):
    """Plot training and validation loss, accuracy, and macro F1 curves."""
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    _fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="tab:blue")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="tab:orange")
    axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    # Macro F1
    axes[2].plot(epochs, history["train_f1"], label="Train Macro F1", color="tab:blue")
    axes[2].plot(epochs, history["val_f1"], label="Val Macro F1", color="tab:orange")
    best_f1 = history.get("best_val_f1", 0)
    best_ep = history.get("best_epoch", 0)
    axes[2].axhline(
        y=best_f1,
        color="tab:red",
        linestyle=":",
        label=f"Best Val F1: {best_f1:.1f}% (Ep {best_ep})",
    )
    axes[2].set_title("Macro F1-score (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1 (%)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    if output_file is None:
        output_file = RESULTS_DIR / f"{Path(history_file).stem}_curves.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Saved training curve plot: {output_file}")


def plot_confusion_matrix(
    cm: list,
    output_file: Path | None = None,
    title: str = "Confusion Matrix (7 Classes)",
):
    """Plot and save confusion matrix heatmap."""
    cm_arr = np.array(cm)
    _fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = [f"C{i}" for i in range(len(CLASS_NAMES))]
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm_arr.max() / 2.0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                format(cm_arr[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )

    plt.tight_layout()
    if output_file is None:
        output_file = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot: {output_file}")
