"""
Evaluation Module on Held-out Test Set (Ivanovs & Ivanova, Baltic DB&IS 2026)
Computes Macro F1, Accuracy, per-class metrics, and generates classification reports.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from torch import nn

from config import BEST_CONFIGS, CLASS_NAMES, RESULTS_DIR
from src.train import evaluate_epoch


def evaluate_test_set(
    model: nn.Module,
    checkpoint_path: Path,
    test_loader,
    model_name: str = "temporal_cnn",
    device: str | None = None,
):
    """
    Load checkpoint and evaluate on held-out test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded checkpoint from {checkpoint_path} (saved at epoch {ckpt.get('epoch')})"
        )

    criterion = nn.CrossEntropyLoss()
    _test_loss, test_acc, test_macro_f1, preds, targets = evaluate_epoch(
        model, test_loader, criterion, device
    )

    # Per-class metrics
    target_names = [f"Class {i}: {CLASS_NAMES[i]}" for i in range(len(CLASS_NAMES))]
    report_dict = classification_report(
        targets, preds, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        targets, preds, target_names=target_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(targets, preds).tolist()

    # Comparison with paper
    reported_metrics = BEST_CONFIGS.get(model_name, {})
    rep_f1 = reported_metrics.get("reported_f1", "N/A")
    rep_acc = reported_metrics.get("reported_acc", "N/A")

    print("\n" + "=" * 65)
    print(f"   EVALUATION RESULTS ON TEST SET: {model_name.upper()}")
    print("=" * 65)
    print(f"Test Accuracy:       {test_acc:.2f}%  (Paper reported: {rep_acc}%)")
    print(f"Test Macro F1-score: {test_macro_f1:.2f}%  (Paper reported: {rep_f1}%)")
    print("-" * 65)
    print("Per-class Classification Report:")
    print(report_text)
    print("=" * 65)

    results_data = {
        "model_name": model_name,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "reported_f1": rep_f1,
        "reported_acc": rep_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    out_json = RESULTS_DIR / f"test_results_{model_name}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Detailed results saved to {out_json}")

    return results_data
