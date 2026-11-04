"""
Script: BiLSTM + Temporal Attention Comparison Experiment
Evaluates:
  Case 1: BiLSTM + Attention with Final-Frame Labeling (Legacy paper labeling)
  Case 2: BiLSTM + Attention with Majority Voting Labeling (purity >= 0.6)
Both run on the same split, baseline cache, and standard evaluation metrics.
"""

import argparse
import hashlib
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


class BiLSTMTemporalAttention(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=64, num_layers=2, num_classes=7, dropout=0.2):
        super().__init__()
        # 1. Spatial linear projection with LayerNorm for numerical stability
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 2-layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 3. Temporal Self-Attention Layer
        # bilstm output dim is hidden_dim * 2
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Classification Head with Dropout & LayerNorm
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 48, 126)
        x_proj = self.proj(x)                         # (B, 48, 128)
        lstm_out, _ = self.bilstm(x_proj)             # (B, 48, hidden_dim * 2)
        
        # Attention scores across 48 timesteps
        scores = self.attn_net(lstm_out)              # (B, 48, 1)
        weights = F.softmax(scores, dim=1)            # (B, 48, 1)
        
        # Weighted context vector
        context = torch.sum(lstm_out * weights, dim=1) # (B, hidden_dim * 2)
        logits = self.head(context)                   # (B, num_classes)
        return logits


class ExperimentWindows(Dataset):
    def __init__(self, features, presence, labels, window=48, method="majority", min_purity=0.6, is_train=False):
        self.features = [torch.from_numpy(x) for x in features]
        self.window = window
        self.method = method
        
        refs = []
        targets = []
        
        for i, (x, p, y) in enumerate(zip(features, presence, labels)):
            if len(x) < window:
                continue
            valid = np.any(p > 0, axis=1).astype(np.int32)
            counts = np.convolve(valid, np.ones(window, dtype=np.int32), mode="valid")
            valid_starts = np.flatnonzero(counts == window)
            
            for start in valid_starts:
                window_y = y[start : start + window]
                valid_mask = (window_y >= 0) & (window_y < 7)
                if not np.any(valid_mask):
                    continue
                
                if method == "majority":
                    valid_labels = window_y[valid_mask]
                    bincount = np.bincount(valid_labels, minlength=7)
                    maj_label = int(np.argmax(bincount))
                    purity = bincount[maj_label] / len(valid_labels)
                    if is_train and purity < min_purity:
                        continue
                    label = maj_label
                else:
                    final_label = int(y[start + window - 1])
                    if final_label < 0 or final_label >= 7:
                        continue
                    label = final_label
                    
                refs.append((i, int(start)))
                targets.append(label)
                
        self.refs = np.asarray(refs, dtype=np.int32).reshape(-1, 2)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        i, start = self.refs[index]
        return self.features[i][start : start + self.window], int(self.targets[index])


class FastDeviceWindows:
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
        "train": ExperimentWindows(
            scaled_features["train"], flags["train"], labels["train"],
            window=48, method=method, min_purity=min_purity, is_train=True
        ),
        "val": ExperimentWindows(
            scaled_features["val"], flags["val"], labels["val"],
            window=48, method=method, min_purity=min_purity, is_train=False
        ),
        "test": ExperimentWindows(
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


def train_and_eval_case(case_name, method, min_purity, output_dir, epochs=40, seed=42):
    print(f"\n{'='*70}", flush=True)
    print(f"RUNNING EXPERIMENT: {case_name.upper()}", flush=True)
    print(f"Method: {method} | Min Purity: {min_purity} | Output: {output_dir}", flush=True)
    print(f"{'='*70}", flush=True)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    cache = ROOT / "data" / "landmarks_baseline"
    manifest = json.loads((cache / "manifest.json").read_bytes())
    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())
    
    datasets, scaler = load_dataset_split(
        cache, split, manifest,
        presence_filter="detected",
        method=method,
        min_purity=min_purity
    )
    print(f"Windows -> Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}", flush=True)
    
    device_data = {k: FastDeviceWindows(v, device) for k, v in datasets.items()}
    model = BiLSTMTemporalAttention(input_dim=126, hidden_dim=64, num_layers=2, num_classes=7, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    best_val_f1 = -1.0
    history = []
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in device_data["train"].batches(shuffle=True, batch_size=256):
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
            }, output_path / "best_model.pt")
            (output_path / "val_metrics.json").write_text(json.dumps(val_res, indent=2), encoding="utf-8")
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Best Val F1: {best_val_f1*100:.2f}%", flush=True)
            
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}m). Evaluating on Test...", flush=True)
    
    # Load best checkpoint
    best_ckpt = torch.load(output_path / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_res = evaluate_model(model, device_data["test"])
    test_res["best_epoch"] = best_ckpt["epoch"]
    test_res["best_val_macro_f1"] = best_ckpt["val_macro_f1"]
    test_res["elapsed_s"] = elapsed
    
    (output_path / "test_metrics.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")
    (output_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    
    print(f"\n{case_name} RESULTS:")
    print(f"Selected Best Epoch: {best_ckpt['epoch']}")
    print(f"Test Accuracy: {test_res['accuracy']*100:.2f}%")
    print(f"Test Macro-F1: {test_res['macro_f1']*100:.2f}%")
    for c in range(7):
        cr = test_res["classification_report"][str(c)]
        print(f"  Class {c}: P={cr['precision']*100:.1f}%, R={cr['recall']*100:.1f}%, F1={cr['f1-score']*100:.1f}%")
        
    return test_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--case", choices=["all", "case1", "case2"], default="all")
    args = parser.parse_args()
    
    res1, res2 = None, None
    if args.case in ("all", "case1"):
        res1 = train_and_eval_case(
            case_name="Case 1: BiLSTM + Attention (Final-Frame Labeling)",
            method="final_frame",
            min_purity=0.0,
            output_dir=ROOT / "results" / "bilstm_attention_final_frame",
            epochs=args.epochs
        )
        
    if args.case in ("all", "case2"):
        res2 = train_and_eval_case(
            case_name="Case 2: BiLSTM + Attention (Majority Voting Labeling)",
            method="majority",
            min_purity=0.6,
            output_dir=ROOT / "results" / "bilstm_attention_majority_voting",
            epochs=args.epochs
        )
        
    if res1 and res2:
        print("\n" + "#"*70, flush=True)
        print("FINAL COMPARISON: BiLSTM + TEMPORAL ATTENTION", flush=True)
        print("#"*70, flush=True)
        print(f"{'Metric':<25} | {'Case 1 (Final-Frame)':<22} | {'Case 2 (Majority-Vote)':<22}", flush=True)
        print("-" * 75, flush=True)
        print(f"{'Val Macro-F1':<25} | {res1['best_val_macro_f1']*100:.2f}%{'':<16} | {res2['best_val_macro_f1']*100:.2f}%", flush=True)
        print(f"{'Test Accuracy':<25} | {res1['accuracy']*100:.2f}%{'':<16} | {res2['accuracy']*100:.2f}%", flush=True)
        print(f"{'Test Macro-F1':<25} | {res1['macro_f1']*100:.2f}%{'':<16} | {res2['macro_f1']*100:.2f}%", flush=True)
        print(f"{'Best Selected Epoch':<25} | {res1['best_epoch']:<22} | {res2['best_epoch']:<22}", flush=True)
        print("-" * 75, flush=True)
        for c in range(7):
            f1_1 = res1['classification_report'][str(c)]['f1-score'] * 100
            f1_2 = res2['classification_report'][str(c)]['f1-score'] * 100
            diff = f1_2 - f1_1
            sign = "+" if diff >= 0 else ""
            print(f"Class {c} F1{'':<17} | {f1_1:.2f}%{'':<16} | {f1_2:.2f}% ({sign}{diff:.2f}%)", flush=True)
        print("#"*75, flush=True)


if __name__ == "__main__":
    main()
