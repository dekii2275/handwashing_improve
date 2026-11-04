"""
Script: Pure 6-Step WHO Action Recognition with Temporal Masking
Focuses strictly on the active WHO handwashing phases (Classes 1 to 6).
Filters out the non-WHO ambient noise (Class 0: water running, soap dispensing).
Runs on NVIDIA GPU with BiLSTM + Temporal Attention.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset

from src.features import FeatureStandardizer, extract_features

WHO_STEP_NAMES = [
    "Step 1: Palm to palm",
    "Step 2: Palm over dorsum",
    "Step 3: Palm to palm interlaced",
    "Step 4: Backs of fingers",
    "Step 5: Thumb rubbing",
    "Step 6: Fingertips rubbing",
]


class BiLSTM6ClassAttention(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
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
    """Extracts only valid windows belonging to WHO steps 1-6 (Mapped to 0-5)."""
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
                # Filter strictly for WHO steps 1 to 6 (map to 0-5)
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
            yield (
                self.frames[self.starts[batch, None] + self.steps],
                self.targets[batch],
            )


def evaluate_6class(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches():
            outputs.append(model(x).argmax(1))
    ys = data.targets.cpu().numpy()
    ps = torch.cat(outputs).cpu().numpy()
    return {
        "accuracy": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, labels=list(range(6)), average="macro", zero_division=0)),
        "classification_report": classification_report(
            ys, ps, labels=list(range(6)), target_names=WHO_STEP_NAMES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(ys, ps, labels=list(range(6))).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "who_6class_temporal_masking")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Pure WHO 6-Class Action Recognition on Device: {device}", flush=True)

    cache = ROOT / "data" / "landmarks_baseline"
    manifest = json.loads((cache / "manifest.json").read_bytes())
    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())

    features, flags, labels = {}, {}, {}
    for part in ("train", "val", "test"):
        features[part], flags[part], labels[part] = [], [], []
        for r in split[part]:
            stem = r["video_stem"]
            path = cache / (stem + ".npz")
            with np.load(path) as d:
                raw, presence, y = d["coords"], d["presence"], d["labels"]
                x, imputed_presence = extract_features(raw, presence, "raw_landmarks", "norm_loc", True)
                features[part].append(x)
                flags[part].append(presence.copy())
                labels[part].append(y.copy())

    scaler = FeatureStandardizer().fit(features["train"])
    scaled_features = {part: [scaler.transform(x) for x in features[part]] for part in features}

    datasets = {k: WHO6StepWindows(scaled_features[k], flags[k], labels[k], window=48, device=device) for k in ("train", "val", "test")}
    print(f"[*] Filtered WHO Windows -> Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}", flush=True)

    model = BiLSTM6ClassAttention(input_dim=126, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    history = []
    t0 = time.time()

    print("[*] Starting GPU Training for 6-Step WHO Action Recognition...", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in datasets["train"].batches(shuffle=True, batch_size=args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(datasets["train"])
        val_res = evaluate_6class(model, datasets["val"])
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
            (args.output / "val_metrics.json").write_text(json.dumps(val_res, indent=2), encoding="utf-8")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Best Val F1: {best_val_f1*100:.2f}%", flush=True)

    elapsed = time.time() - t0
    print(f"[*] Training finished in {elapsed:.1f}s. Evaluating on Test Set...", flush=True)

    best_ckpt = torch.load(args.output / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_res = evaluate_6class(model, datasets["test"])
    test_res["best_epoch"] = best_ckpt["epoch"]
    test_res["best_val_macro_f1"] = best_ckpt["val_macro_f1"]
    test_res["elapsed_s"] = elapsed

    (args.output / "test_metrics.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n" + "="*70, flush=True)
    print("FINAL TEST RESULTS: 6-STEP WHO RECOGNITION (TEMPORAL MASKING)", flush=True)
    print(f"Selected Best Epoch: {best_ckpt['epoch']}", flush=True)
    print(f">> TEST ACCURACY: {test_res['accuracy']*100:.2f}% <<", flush=True)
    print(f">> TEST MACRO-F1: {test_res['macro_f1']*100:.2f}% <<", flush=True)
    print("="*70, flush=True)
    for name in WHO_STEP_NAMES:
        cr = test_res["classification_report"][name]
        print(f"{name:<35}: Precision={cr['precision']*100:.1f}%, Recall={cr['recall']*100:.1f}%, F1={cr['f1-score']*100:.1f}% (Support={int(cr['support'])})", flush=True)


if __name__ == "__main__":
    main()
