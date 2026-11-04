"""
Script: Advanced Geometric Features & Class-Balanced Focal Loss Experiments
Tests:
  Exp 1: Geometric Invariants Only (Palm Normal Cosine + Tip-to-Palm Distances)
  Exp 2: Class-Balanced Focal Loss Only (gamma=2.0)
  Exp 3: Combined (Geometric Invariants + Focal Loss + BiLSTM Attention)
All run on GPU (CUDA) with 30 epochs and standard evaluation.
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


def compute_geometric_features(x):
    """
    Computes Palm Normal Cosine and Tip-to-Palm distances on PyTorch GPU tensor.
    Args:
        x: Tensor of shape (B, T, 126)
    Returns:
        feats: Tensor of shape (B, T, 144) (126 raw coords + 18 geometric invariants)
    """
    B, T, _ = x.shape
    coords = x.view(B, T, 2, 21, 3) # (B, T, Hand:2, Joint:21, Coord:3)
    
    # 1. Palm Normal Vectors: Wrist (0), Index MCP (5), Pinky MCP (17)
    v05 = coords[:, :, :, 5, :] - coords[:, :, :, 0, :]
    v017 = coords[:, :, :, 17, :] - coords[:, :, :, 0, :]
    normals = torch.cross(v05, v017, dim=-1)
    norm_normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    
    n_left = norm_normals[:, :, 0, :]   # (B, T, 3)
    n_right = norm_normals[:, :, 1, :]  # (B, T, 3)
    
    # Palm Orientation Cosine: +1 (Same direction - Palm on Dorsum), -1 (Opposite - Palm on Palm)
    cos_theta = torch.sum(n_left * n_right, dim=-1, keepdim=True) # (B, T, 1)
    
    # 2. Palm Centers: Average of Wrist(0), Index(5), Middle(9), Pinky(17)
    palm_left = (coords[:, :, 0, 0] + coords[:, :, 0, 5] + coords[:, :, 0, 9] + coords[:, :, 0, 17]) * 0.25
    palm_right = (coords[:, :, 1, 0] + coords[:, :, 1, 5] + coords[:, :, 1, 9] + coords[:, :, 1, 17]) * 0.25
    
    # Palm-to-palm distance
    palm_dist = torch.norm(palm_left - palm_right, p=2, dim=-1, keepdim=True) # (B, T, 1)
    
    # 3. Fingertip-to-opposite-palm distances: Tips [4, 8, 12, 16, 20]
    tip_indices = [4, 8, 12, 16, 20]
    tips_left = coords[:, :, 0, tip_indices, :]  # (B, T, 5, 3)
    tips_right = coords[:, :, 1, tip_indices, :] # (B, T, 5, 3)
    
    dist_l_tips_to_r_palm = torch.norm(tips_left - palm_right.unsqueeze(2), p=2, dim=-1) # (B, T, 5)
    dist_r_tips_to_l_palm = torch.norm(tips_right - palm_left.unsqueeze(2), p=2, dim=-1) # (B, T, 5)
    
    # Concatenate raw coords + 18 geometric features
    geom_features = torch.cat([
        x,
        n_left,
        n_right,
        cos_theta,
        palm_dist,
        dist_l_tips_to_r_palm,
        dist_r_tips_to_l_palm
    ], dim=-1) # (B, T, 126 + 3 + 3 + 1 + 1 + 5 + 5 = 144)
    return geom_features


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class BiLSTMTemporalAttention(nn.Module):
    def __init__(self, input_dim=126, hidden_dim=64, num_layers=2, num_classes=7, dropout=0.2):
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


class FastWindows:
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
                lbl = int(y[s + window - 1])
                if 0 <= lbl < 7:
                    refs.append((i, int(s)))
                    targets.append(lbl)
                    
        refs = np.asarray(refs, dtype=np.int32).reshape(-1, 2)
        starts = offsets[refs[:, 0]] + refs[:, 1]
        
        self.frames = torch.cat([torch.from_numpy(x) for x in features]).to(device)
        self.starts = torch.as_tensor(starts, dtype=torch.long, device=device)
        self.targets = torch.as_tensor(targets, dtype=torch.long, device=device)
        self.steps = torch.arange(window, device=device)
        self.window = window

    def __len__(self):
        return len(self.starts)

    def batches(self, shuffle=False, batch_size=256, use_geometric=False):
        order = (
            torch.randperm(len(self.starts), device=self.starts.device)
            if shuffle
            else torch.arange(len(self.starts), device=self.starts.device)
        )
        for batch in order.split(batch_size):
            raw_batch = self.frames[self.starts[batch, None] + self.steps]
            if use_geometric:
                batch_x = compute_geometric_features(raw_batch)
            else:
                batch_x = raw_batch
            yield batch_x, self.targets[batch]


def evaluate_model(model, data, use_geometric=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches(use_geometric=use_geometric):
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


def run_experiment(exp_name, use_geometric, use_focal_loss, output_dir, epochs=30):
    print("\n" + "="*70, flush=True)
    print(f"RUNNING: {exp_name}", flush=True)
    print(f"Geometric Features: {use_geometric} | Focal Loss: {use_focal_loss}", flush=True)
    print("="*70, flush=True)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    datasets = {k: FastWindows(scaled_features[k], flags[k], labels[k], window=48, device=device) for k in ("train", "val", "test")}
    print(f"Windows -> Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}", flush=True)
    
    # Compute inverse class frequencies for Focal Loss weighting
    train_targets = datasets["train"].targets.cpu().numpy()
    class_counts = np.bincount(train_targets, minlength=7)
    weights = len(train_targets) / (7.0 * np.maximum(class_counts, 1))
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    
    input_dim = 144 if use_geometric else 126
    model = BiLSTMTemporalAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=7, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    if use_focal_loss:
        loss_fn = FocalLoss(gamma=2.0, weight=class_weights_t)
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    best_val_f1 = -1.0
    history = []
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in datasets["train"].batches(shuffle=True, batch_size=256, use_geometric=use_geometric):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            
        train_loss = total_loss / len(datasets["train"])
        val_res = evaluate_model(model, datasets["val"], use_geometric=use_geometric)
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
            }, out_path / "best_model.pt")
            (out_path / "val_metrics.json").write_text(json.dumps(val_res, indent=2), encoding="utf-8")
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Best Val F1: {best_val_f1*100:.2f}%", flush=True)
            
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s. Evaluating on Test...", flush=True)
    
    best_ckpt = torch.load(out_path / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_res = evaluate_model(model, datasets["test"], use_geometric=use_geometric)
    test_res["best_epoch"] = best_ckpt["epoch"]
    test_res["best_val_macro_f1"] = best_ckpt["val_macro_f1"]
    test_res["elapsed_s"] = elapsed
    
    (out_path / "test_metrics.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")
    (out_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    
    print(f"\n{exp_name} RESULTS:")
    print(f"Test Accuracy: {test_res['accuracy']*100:.2f}% | Test Macro-F1: {test_res['macro_f1']*100:.2f}%")
    for c in range(7):
        cr = test_res["classification_report"][str(c)]
        print(f"  Class {c}: P={cr['precision']*100:.1f}%, R={cr['recall']*100:.1f}%, F1={cr['f1-score']*100:.1f}%")
        
    return test_res


def main():
    experiments = [
        ("Exp 1: Geometric Invariants (Normals + Tip Distances)", True, False, ROOT / "results" / "exp1_geometric_only"),
        ("Exp 2: Class-Balanced Focal Loss (gamma=2.0)", False, True, ROOT / "results" / "exp2_focal_loss_only"),
        ("Exp 3: Combined (Geometric + Focal Loss + BiLSTM)", True, True, ROOT / "results" / "exp3_combined_advanced"),
    ]
    
    all_res = {}
    for name, geom, focal, out in experiments:
        res = run_experiment(name, geom, focal, out, epochs=30)
        all_res[name] = res
        
    print("\n" + "#"*80, flush=True)
    print("SUMMARY OF ADVANCED IMPROVEMENTS", flush=True)
    print("#"*80, flush=True)
    print(f"{'Experiment':<50} | {'Test Accuracy':<14} | {'Test Macro-F1':<14}", flush=True)
    print("-" * 84, flush=True)
    print(f"{'Baseline (BiLSTM Attention Raw Coords)':<50} | {'65.95%':<14} | {'60.81%':<14}", flush=True)
    for name, r in all_res.items():
        print(f"{name:<50} | {r['accuracy']*100:.2f}%{'':<8} | {r['macro_f1']*100:.2f}%", flush=True)
    print("#"*84, flush=True)


if __name__ == "__main__":
    main()
