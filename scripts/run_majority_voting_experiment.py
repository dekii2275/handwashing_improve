"""
Experiment: Majority Voting Labeling vs Final-Frame Labeling
Implements:
1. Sliding window labeling with Mode (Majority Vote) instead of final-frame label.
2. Label purity filtering (min_purity >= 0.6) for training set.
3. Automatic GPU/CPU selection (uses CUDA if available, CPU fallback).
4. Saves full evaluation metrics and compares with baseline.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import Dataset

from src.features import FeatureStandardizer, extract_features


class PaperCNN(nn.Module):
    def __init__(self, pooling="avg"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(126, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pool = (
            nn.AdaptiveAvgPool1d(1) if pooling == "avg" else nn.AdaptiveMaxPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 7)
        )

    def forward(self, x):
        return self.head(self.pool(self.conv(x.transpose(1, 2))).squeeze(-1))


class MajorityVotingWindows(Dataset):
    """
    Sliding window dataset supporting:
    - method='majority': window label is the most frequent label in window.
    - method='final_frame': legacy paper method (label of frame t=48).
    - min_purity: minimum fraction of window frames belonging to majority label (for train).
    """
    def __init__(self, features, presence, labels, window=48, method="majority", min_purity=0.6, is_train=False):
        self.features = [torch.from_numpy(x) for x in features]
        self.window = window
        self.method = method
        
        refs = []
        targets = []
        purities = []
        
        for i, (x, p, y) in enumerate(zip(features, presence, labels)):
            if len(x) < window:
                continue
            valid = np.any(p > 0, axis=1).astype(np.int32)
            counts = np.convolve(valid, np.ones(window, dtype=np.int32), mode="valid")
            
            # All valid starts where all window frames have hand detection
            valid_starts = np.flatnonzero(counts == window)
            
            for start in valid_starts:
                window_y = y[start : start + window]
                valid_mask = (window_y >= 0) & (window_y < 7)
                if not np.any(valid_mask):
                    continue
                
                valid_labels = window_y[valid_mask]
                bincount = np.bincount(valid_labels, minlength=7)
                maj_label = int(np.argmax(bincount))
                purity = bincount[maj_label] / len(valid_labels)
                
                if method == "majority":
                    # Filter out transition boundary noise in training
                    if is_train and purity < min_purity:
                        continue
                    label = maj_label
                else:
                    # Final frame method
                    final_label = int(y[start + window - 1])
                    if final_label < 0 or final_label >= 7:
                        continue
                    label = final_label
                    
                refs.append((i, int(start)))
                targets.append(label)
                purities.append(purity)
                
        self.refs = np.asarray(refs, dtype=np.int32).reshape(-1, 2)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.purities = np.asarray(purities, dtype=np.float32)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        i, start = self.refs[index]
        return self.features[i][start : start + self.window], int(self.targets[index])


class FastDeviceWindows:
    """Pre-places tensors on device (GPU/CPU) for rapid batch slicing."""
    def __init__(self, dataset, device):
        lengths = [len(x) for x in dataset.features]
        offsets = np.cumsum([0] + lengths[:-1])
        refs = dataset.refs
        starts = offsets[refs[:, 0]] + refs[:, 1]
        
        self.frames = torch.cat(dataset.features).to(device)
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=device)
        self.targets = torch.as_tensor(dataset.targets, dtype=torch.long, device=device)
        self.steps = torch.arange(dataset.window, device=device)

    def batches(self, shuffle=False, batch_size=256):
        order = (
            torch.randperm(len(self.starts), device=self.starts.device)
            if shuffle
            else torch.arange(len(self.starts), device=self.starts.device)
        )
        for batch in order.split(batch_size):
            yield (
                self.frames[self.starts[batch, None] + self.steps],
                self.targets[batch],
            )


def load_dataset_split(cache, split, manifest, presence_filter="detected", method="majority", min_purity=0.6):
    features, flags, labels = {}, {}, {}
    for part in ("train", "val", "test"):
        features[part], flags[part], labels[part] = [], [], []
        for r in split[part]:
            stem = r["video_stem"]
            path = cache / (stem + ".npz")
            if stem not in manifest or not path.exists():
                raise RuntimeError(f"Missing baseline cache: {stem}")
            with np.load(path) as d:
                raw, presence, y = d["coords"], d["presence"], d["labels"]
                x, imputed_presence = extract_features(
                    raw, presence, "raw_landmarks", "norm_loc", True
                )
                features[part].append(x)
                flags[part].append(
                    (presence if presence_filter == "detected" else imputed_presence).copy()
                )
                labels[part].append(y.copy())
                
    scaler = FeatureStandardizer().fit(features["train"])
    scaled_features = {
        part: [scaler.transform(x) for x in features[part]]
        for part in features
    }
    
    datasets = {
        "train": MajorityVotingWindows(
            scaled_features["train"], flags["train"], labels["train"],
            window=48, method=method, min_purity=min_purity, is_train=True
        ),
        "val": MajorityVotingWindows(
            scaled_features["val"], flags["val"], labels["val"],
            window=48, method=method, min_purity=min_purity, is_train=False
        ),
        "test": MajorityVotingWindows(
            scaled_features["test"], flags["test"], labels["test"],
            window=48, method=method, min_purity=min_purity, is_train=False
        ),
    }
    return datasets, scaler


def evaluate_model(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches():
            outputs.append(model(x).argmax(1))
    ys = data.targets.cpu().numpy()
    ps = torch.cat(outputs).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, labels=list(range(7)), average="macro", zero_division=0)),
        "classification_report": classification_report(
            ys, ps, labels=list(range(7)), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(ys, ps, labels=list(range(7))).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["majority", "final_frame"], default="majority")
    parser.add_argument("--min-purity", type=float, default=0.6)
    parser.add_argument("--presence-filter", choices=["detected", "imputed"], default="detected")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "majority_voting_experiment")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device} | Method: {args.method} (purity >= {args.min_purity})")

    cache = ROOT / "data" / "landmarks_baseline"
    manifest = json.loads((cache / "manifest.json").read_bytes())
    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())

    datasets, scaler = load_dataset_split(
        cache, split, manifest,
        presence_filter=args.presence_filter,
        method=args.method,
        min_purity=args.min_purity
    )

    print(f"[*] Window counts -> Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

    device_data = {k: FastDeviceWindows(v, device) for k, v in datasets.items()}
    model = PaperCNN(pooling="avg").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    history = []

    print("[*] Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in device_data["train"].batches(shuffle=True, batch_size=args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(datasets["train"])
        val_res = evaluate_model(model, device_data["val"])
        val_acc, val_f1 = val_res["accuracy"], val_res["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_macro_f1": best_val_f1
            }, args.output / "best_model.pt")
            (args.output / "val_metrics.json").write_text(json.dumps(val_res, indent=2))

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Best Val F1: {best_val_f1*100:.2f}%")

    # Final evaluation on Test Set using best model checkpoint
    best_ckpt = torch.load(args.output / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_res = evaluate_model(model, device_data["test"])
    test_res["best_epoch"] = best_ckpt["epoch"]
    test_res["best_val_macro_f1"] = best_ckpt["val_macro_f1"]

    (args.output / "test_metrics.json").write_text(json.dumps(test_res, indent=2))
    (args.output / "history.json").write_text(json.dumps(history, indent=2))

    print("\n" + "=" * 60)
    print(f"FINAL TEST RESULTS (Method: {args.method})")
    print(f"Best Epoch: {best_ckpt['epoch']}")
    print(f"Test Accuracy: {test_res['accuracy']*100:.2f}%")
    print(f"Test Macro-F1: {test_res['macro_f1']*100:.2f}%")
    print("=" * 60)
    for c in range(7):
        cr = test_res["classification_report"][str(c)]
        print(f"Class {c}: Precision={cr['precision']*100:.2f}%, Recall={cr['recall']*100:.2f}%, F1={cr['f1-score']*100:.2f}%, Support={int(cr['support'])}")


if __name__ == "__main__":
    main()
