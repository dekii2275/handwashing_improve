"""
Script to track Val and Test metrics at every epoch for:
1. Pure Baseline (126 dims)
2. Geometric Invariants (144 dims)
With dropout=0.3, weight_decay=1e-3, early stopping tracking.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.features import FeatureStandardizer, impute_missing_landmarks, normalize_spatial

WHO_STEP_NAMES = [
    "Step 1: Palm to palm",
    "Step 2: Palm over dorsum",
    "Step 3: Palm to palm interlaced",
    "Step 4: Backs of fingers",
    "Step 5: Thumb rubbing",
    "Step 6: Fingertips rubbing",
]


def extract_18_geom(raw_coords, norm_loc_coords):
    N = len(raw_coords)
    c_raw = raw_coords.reshape(N, 2, 21, 3)
    c_loc = norm_loc_coords.reshape(N, 2, 21, 3)

    v05_0 = c_loc[:, 0, 5, :] - c_loc[:, 0, 0, :]
    v017_0 = c_loc[:, 0, 17, :] - c_loc[:, 0, 0, :]
    n0 = np.cross(v05_0, v017_0)
    norm0 = np.linalg.norm(n0, axis=-1, keepdims=True) + 1e-6
    n0 = n0 / norm0

    v05_1 = c_loc[:, 1, 5, :] - c_loc[:, 1, 0, :]
    v017_1 = c_loc[:, 1, 17, :] - c_loc[:, 1, 0, :]
    n1 = np.cross(v05_1, v017_1)
    norm1 = np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-6
    n1 = n1 / norm1

    cos_theta = np.sum(n0 * n1, axis=-1, keepdims=True)

    p0 = c_raw[:, 0, [0, 5, 9, 17], :].mean(axis=1)
    p1 = c_raw[:, 1, [0, 5, 9, 17], :].mean(axis=1)
    palm_dist = np.linalg.norm(p0 - p1, axis=-1, keepdims=True)

    tips = [4, 8, 12, 16, 20]
    tips0 = c_raw[:, 0, tips, :]
    tips1 = c_raw[:, 1, tips, :]

    d01 = np.linalg.norm(tips0 - p1[:, None, :], axis=-1)
    d10 = np.linalg.norm(tips1 - p0[:, None, :], axis=-1)

    geom = np.concatenate([n0, n1, cos_theta, palm_dist, d01, d10], axis=-1)
    return geom.astype(np.float32)


class BiLSTM6ClassAttention(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x_proj = self.proj(x)
        lstm_out, _ = self.bilstm(x_proj)
        scores = self.attn_net(lstm_out)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        logits = self.head(context)
        return logits


class WHO6StepWindows:
    def __init__(self, features, presence, labels, window=48, device="cuda"):
        lengths = [len(x) for x in features]
        offsets = np.cumsum([0] + lengths[:-1])
        refs = []
        targets = []

        for i, (x, p, y) in enumerate(zip(features, presence, labels)):
            if len(x) < window:
                continue
            valid = np.any(p > 0, axis=1).astype(np.int32)
            counts = np.convolve(valid, np.ones(window, dtype=np.int32), mode="valid")
            starts = np.flatnonzero(counts == window)

            for s in starts:
                raw_lbl = int(y[s + window - 1])
                if 1 <= raw_lbl <= 6:
                    refs.append((i, int(s)))
                    targets.append(raw_lbl - 1)

        refs = np.asarray(refs, dtype=np.int32).reshape(-1, 2)
        starts = offsets[refs[:, 0]] + refs[:, 1]

        self.frames = torch.cat([torch.from_numpy(x) for x in features]).to(device)
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=device)
        self.targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        self.steps = torch.arange(window, device=device)
        self.window = window

    def __len__(self):
        return len(self.starts)

    def batches(self, shuffle=False, batch_size=256):
        order = (
            torch.randperm(len(self.starts), device=self.starts.device)
            if shuffle
            else torch.arange(len(self.starts), device=self.starts.device)
        )
        for batch in order.split(batch_size):
            yield self.frames[self.starts[batch, None] + self.steps], self.targets[batch]


def evaluate_metrics(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches(shuffle=False):
            outputs.append(model(x).argmax(1))
    ys = data.targets.cpu().numpy()
    ps = torch.cat(outputs).cpu().numpy()
    acc = float(accuracy_score(ys, ps))
    f1 = float(f1_score(ys, ps, labels=list(range(6)), average="macro", zero_division=0))
    return acc, f1, ps, ys


def run_experiment(use_geometric=True, epochs=15, weight_decay=1e-3, lr=0.001, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode_name = "geometric_144d" if use_geometric else "baseline_126d"
    out_dir = ROOT / "results" / f"who_6class_{mode_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RUNNING: {mode_name.upper()} (epochs={epochs}, wd={weight_decay})")
    print(f"{'='*60}", flush=True)

    cache = ROOT / "data" / "landmarks_baseline"
    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())

    features, flags, labels = {}, {}, {}
    for part in ("train", "val", "test"):
        features[part], flags[part], labels[part] = [], [], []
        for r in split[part]:
            stem = r["video_stem"]
            path = cache / (stem + ".npz")
            with np.load(path) as d:
                raw, presence, y = d["coords"], d["presence"], d["labels"]
                imputed_raw, imputed_presence = impute_missing_landmarks(raw, presence)
                norm_loc = normalize_spatial(imputed_raw, norm_type="norm_loc")

                if use_geometric:
                    geom = extract_18_geom(imputed_raw, norm_loc)
                    full_feat = np.concatenate([norm_loc, geom], axis=-1).astype(np.float32)
                else:
                    full_feat = norm_loc.astype(np.float32)

                features[part].append(full_feat)
                flags[part].append(presence.copy())
                labels[part].append(y.copy())

    scaler = FeatureStandardizer().fit(features["train"])
    scaled_features = {part: [scaler.transform(x) for x in features[part]] for part in features}

    datasets = {
        k: WHO6StepWindows(scaled_features[k], flags[k], labels[k], window=48, device=device)
        for k in ("train", "val", "test")
    }

    input_dim = 144 if use_geometric else 126
    model = BiLSTM6ClassAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = -1
    best_test_acc = 0.0
    best_test_f1 = 0.0
    trajectory = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in datasets["train"].batches(shuffle=True, batch_size=256):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        scheduler.step()
        train_loss = total_loss / len(datasets["train"])

        val_acc, val_f1, _, _ = evaluate_metrics(model, datasets["val"])
        test_acc, test_f1, test_ps, test_ys = evaluate_metrics(model, datasets["test"])

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_test_acc = test_acc
            best_test_f1 = test_f1
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "is_best": is_best,
        }
        trajectory.append(record)

        marker = " <== BEST VAL" if is_best else ""
        print(
            f"Ep {epoch:02d} | TrainLoss: {train_loss:.4f} | ValAcc: {val_acc*100:.2f}% (F1: {val_f1*100:.2f}%) | "
            f"TestAcc: {test_acc*100:.2f}% (F1: {test_f1*100:.2f}%){marker}",
            flush=True,
        )

    print(f"\n---> {mode_name.upper()} SUMMARY <---")
    print(f"Selected by Best Val F1 at Epoch {best_epoch}:")
    print(f"Val F1: {best_val_f1*100:.2f}%")
    print(f"Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"Test Macro-F1: {best_test_f1*100:.2f}%")

    (out_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    return best_test_acc, best_test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--wd", type=float, default=1e-3)
    args = parser.parse_args()

    # Run geometric model
    run_experiment(use_geometric=True, epochs=args.epochs, weight_decay=args.wd)
