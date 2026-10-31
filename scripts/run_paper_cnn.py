"""Run the paper-described temporal CNN in an isolated output directory.

The paper does not specify every implementation detail. This implementation uses
unweighted cross entropy, Adam without weight decay/scheduler, average pooling,
same convolution padding, and original detection flags for the window filter.
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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


class Windows(Dataset):
    def __init__(self, features, presence, labels, window=48):
        self.features = [torch.from_numpy(x) for x in features]
        self.labels = labels
        refs = []
        for i, (x, p, y) in enumerate(zip(features, presence, labels)):
            if len(x) < window:
                continue
            valid = np.any(p > 0, axis=1).astype(np.int32)
            counts = np.convolve(valid, np.ones(window, dtype=np.int32), mode="valid")
            starts = np.flatnonzero(
                (counts == window) & (y[window - 1 :] >= 0) & (y[window - 1 :] < 7)
            )
            refs.extend((i, int(start)) for start in starts)
        self.refs = np.asarray(refs, dtype=np.int32).reshape(-1, 2)
        self.window = window

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        i, start = self.refs[index]
        return self.features[i][start : start + self.window], int(
            self.labels[i][start + self.window - 1]
        )


def load_data(cache, split, manifest, presence_filter="detected"):
    features, flags, labels = {}, {}, {}
    for part in ("train", "val", "test"):
        features[part], flags[part], labels[part] = [], [], []
        for r in split[part]:
            stem = r["video_stem"]
            path = cache / (stem + ".npz")
            if stem not in manifest or not path.exists():
                raise RuntimeError(f"Missing baseline cache or provenance: {stem}")
            entry = manifest[stem]
            if (
                entry["source"] == "extracted_from_video"
                and len(
                    {
                        entry["video_frames_reported"],
                        entry["annotation_frames"],
                        entry["saved_frames"],
                    }
                )
                != 1
            ):
                raise RuntimeError(f"Video and annotation frame counts differ: {stem}")
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != manifest[stem]["landmarks_sha256"]
            ):
                raise RuntimeError(f"Baseline cache differs from manifest: {stem}")
            with np.load(path) as d:
                if np.any(d["confidence"] == np.float32(0.9)) and np.all(
                    (d["confidence"] == 0) | (d["confidence"] == np.float32(0.9))
                ):
                    raise RuntimeError(
                        f"Synthetic fingerprint in baseline cache: {stem}"
                    )
                raw, presence, y = d["coords"], d["presence"], d["labels"]
                if (
                    len(raw) != len(presence)
                    or len(raw) != len(y)
                    or raw.shape[1] != 126
                ):
                    raise ValueError(f"Invalid landmark shapes: {stem}")
                x, imputed_presence = extract_features(
                    raw, presence, "raw_landmarks", "norm_loc", True
                )
                features[part].append(x)
                flags[part].append(
                    (
                        presence if presence_filter == "detected" else imputed_presence
                    ).copy()
                )
                labels[part].append(y.copy())
    scaler = FeatureStandardizer().fit(features["train"])
    return {
        part: Windows(
            [scaler.transform(x) for x in features[part]], flags[part], labels[part]
        )
        for part in features
    }, scaler


class DeviceWindows:
    """Keep frame arrays on the GPU and materialize only the current batch."""

    def __init__(self, dataset, device):
        lengths = [len(x) for x in dataset.features]
        offsets = np.cumsum([0] + lengths[:-1])
        refs = dataset.refs
        starts = offsets[refs[:, 0]] + refs[:, 1]
        targets = np.array(
            [dataset.labels[i][start + dataset.window - 1] for i, start in refs],
            dtype=np.int64,
        )
        self.frames = torch.cat(dataset.features).to(device)
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=device)
        self.targets = torch.as_tensor(targets, dtype=torch.long, device=device)
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


def evaluate_device(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches():
            outputs.append(model(x).argmax(1))
    ys = data.targets.cpu().numpy()
    ps = torch.cat(outputs).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "macro_f1": float(
            f1_score(ys, ps, labels=list(range(7)), average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            ys, ps, labels=list(range(7)), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(ys, ps, labels=list(range(7))).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", choices=["avg", "max"], default="avg")
    parser.add_argument(
        "--presence-filter", choices=["detected", "imputed"], default="detected"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "paper_cnn_clean"
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Training must never silently fall back to CPU on the user's machine.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for training; PyTorch cannot access CUDA"
        )
    torch.set_num_threads(2)
    device = torch.device("cuda")
    cache = ROOT / "data" / "landmarks_baseline"
    manifest_path = cache / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    split_path = ROOT / "data" / "split_info.json"
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    stems = [r["video_stem"] for part in ("train", "val", "test") for r in split[part]]
    if len(stems) != len(set(stems)):
        raise ValueError("Video stems overlap across splits")
    datasets, scaler = load_data(cache, split, manifest, args.presence_filter)
    for part, data in datasets.items():
        if not len(data):
            raise RuntimeError(f"No {part} windows")
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "window": 48,
        "stride": 1,
        "presence_filter": args.presence_filter,
        "pooling": args.pooling,
        "batch_size": 256,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "unweighted_cross_entropy",
        "weight_decay": 0,
        "scheduler": None,
        "device": str(device),
        "torch": torch.__version__,
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance_counts": {
            s: sum(v["source"] == s for v in manifest.values())
            for s in ("extracted_from_video", "legacy_unverified")
        },
        "window_counts": {k: len(v) for k, v in datasets.items()},
    }
    (args.output / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (args.output / "split_snapshot.json").write_bytes(split_bytes)
    (args.output / "manifest_snapshot.json").write_bytes(manifest_bytes)
    np.savez_compressed(
        args.output / "scaler.npz", mean=scaler.scaler.mean_, scale=scaler.scaler.scale_
    )
    print(json.dumps(config, indent=2), flush=True)
    device_data = {k: DeviceWindows(v, device) for k, v in datasets.items()}
    model = PaperCNN(args.pooling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for x, y in device_data["train"].batches(shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(y)
        val = evaluate_device(model, device_data["val"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss.item() / len(datasets["train"]),
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
            }
        )
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": best,
                    "config": config,
                },
                args.output / "best_model.pt",
            )
            (args.output / "validation_metrics.json").write_text(
                json.dumps(val, indent=2), encoding="utf-8"
            )
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch {epoch}/{args.epochs} train_loss={history[-1]['train_loss']:.4f} val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} best={best:.4f}",
            flush=True,
        )
    checkpoint = torch.load(
        args.output / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate_device(model, device_data["test"])
    test["selected_epoch"] = checkpoint["epoch"]
    (args.output / "test_metrics.json").write_text(
        json.dumps(test, indent=2), encoding="utf-8"
    )
    print(
        f"TEST epoch={checkpoint['epoch']} accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
