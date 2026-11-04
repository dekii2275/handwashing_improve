"""
Script: High-Performance 6-Step WHO Recognition with Geometric Invariants & Balanced Loss
Key Highlights:
1. Pure WHO 6-Class dataset (Temporal Masking, Classes 1 to 6 mapped to 0 to 5)
2. 18-dim Geometric Invariants:
   - Left/Right Palm Normal Vectors (6 dims)
   - Palm Orientation Cosine (1 dim)
   - Inter-palm 3D Euclidean Distance (1 dim)
   - 10 Inter-hand Fingertip-to-Palm Distances (10 dims)
   Total features: 126 (norm_loc coords) + 18 (geometry) = 144 features
3. Standardized with FeatureStandardizer on Training split only
4. Class-Balanced CrossEntropyLoss with smoothed inverse-frequency weights
5. Cosine Annealing Learning Rate Scheduler
6. Balanced Checkpoint Selection: Score = 0.5 * Val_Acc + 0.5 * Val_Macro_F1
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
from scipy.stats import mode
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

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
    """
    Computes 18 geometric invariants from 3D coordinates.
    Args:
        raw_coords: (N, 126) in camera space (for inter-hand distances)
        norm_loc_coords: (N, 126) wrist-centered (for palm normals)
    Returns:
        geom: (N, 18)
    """
    N = len(raw_coords)
    c_raw = raw_coords.reshape(N, 2, 21, 3)
    c_loc = norm_loc_coords.reshape(N, 2, 21, 3)

    # 1. Palm Normal Vectors: Wrist (0), Index MCP (5), Pinky MCP (17)
    v05_0 = c_loc[:, 0, 5, :] - c_loc[:, 0, 0, :]
    v017_0 = c_loc[:, 0, 17, :] - c_loc[:, 0, 0, :]
    n0 = np.cross(v05_0, v017_0)
    norm0 = np.linalg.norm(n0, axis=-1, keepdims=True) + 1e-6
    n0 = n0 / norm0  # (N, 3)

    v05_1 = c_loc[:, 1, 5, :] - c_loc[:, 1, 0, :]
    v017_1 = c_loc[:, 1, 17, :] - c_loc[:, 1, 0, :]
    n1 = np.cross(v05_1, v017_1)
    norm1 = np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-6
    n1 = n1 / norm1  # (N, 3)

    # 2. Palm Orientation Cosine
    cos_theta = np.sum(n0 * n1, axis=-1, keepdims=True)  # (N, 1)

    # 3. Inter-palm 3D Euclidean Distance (using camera coords)
    p0 = c_raw[:, 0, [0, 5, 9, 17], :].mean(axis=1)  # (N, 3)
    p1 = c_raw[:, 1, [0, 5, 9, 17], :].mean(axis=1)  # (N, 3)
    palm_dist = np.linalg.norm(p0 - p1, axis=-1, keepdims=True)  # (N, 1)

    # 4. Fingertip-to-opposite-palm distances (indices: 4, 8, 12, 16, 20)
    tips = [4, 8, 12, 16, 20]
    tips0 = c_raw[:, 0, tips, :]  # (N, 5, 3)
    tips1 = c_raw[:, 1, tips, :]  # (N, 5, 3)

    d01 = np.linalg.norm(tips0 - p1[:, None, :], axis=-1)  # (N, 5)
    d10 = np.linalg.norm(tips1 - p0[:, None, :], axis=-1)  # (N, 5)

    geom = np.concatenate([n0, n1, cos_theta, palm_dist, d01, d10], axis=-1)
    return geom.astype(np.float32)


class BiLSTM6ClassAttention(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.25):
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
    def __init__(self, features, presence, labels, window=48, label_strategy="majority_vote", min_purity=0.5, device="cuda"):
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
                win_y = y[s : s + window]
                if label_strategy == "majority_vote":
                    m_val, m_cnt = mode(win_y, keepdims=False)
                    raw_lbl = int(m_val)
                    purity = float(m_cnt) / float(window)
                    if 1 <= raw_lbl <= 6 and purity >= min_purity:
                        refs.append((i, int(s)))
                        targets.append(raw_lbl - 1)
                else:  # final_frame
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


def evaluate_6class(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches(shuffle=False):
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
    parser.add_argument("--label-strategy", type=str, default="majority_vote", choices=["majority_vote", "final_frame"])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "who_6class_balanced_geometric")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Balanced Geometric 6-Step WHO Model on: {device}", flush=True)
    print(f"[*] Label Strategy: {args.label_strategy}", flush=True)

    cache = ROOT / "data" / "landmarks_baseline"
    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())

    print("[*] Extracting 144-dim features (126 norm_loc + 18 geometric invariants)...", flush=True)
    features, flags, labels = {}, {}, {}
    for part in ("train", "val", "test"):
        features[part], flags[part], labels[part] = [], [], []
        for r in split[part]:
            stem = r["video_stem"]
            path = cache / (stem + ".npz")
            with np.load(path) as d:
                raw, presence, y = d["coords"], d["presence"], d["labels"]
                # Impute missing landmarks
                imputed_raw, imputed_presence = impute_missing_landmarks(raw, presence)
                # Norm loc coords (wrist centered)
                norm_loc = normalize_spatial(imputed_raw, norm_type="norm_loc")
                # 18 geometric invariants
                geom = extract_18_geom(imputed_raw, norm_loc)
                # Concatenate 126 + 18 = 144
                full_feat = np.concatenate([norm_loc, geom], axis=-1).astype(np.float32)

                features[part].append(full_feat)
                flags[part].append(presence.copy())
                labels[part].append(y.copy())

    print("[*] Standardizing features using training statistics...", flush=True)
    scaler = FeatureStandardizer().fit(features["train"])
    scaled_features = {part: [scaler.transform(x) for x in features[part]] for part in features}

    datasets = {
        k: WHO6StepWindows(
            scaled_features[k],
            flags[k],
            labels[k],
            window=48,
            label_strategy=args.label_strategy,
            min_purity=0.5 if k == "train" else 0.0,
            device=device,
        )
        for k in ("train", "val", "test")
    }
    print(f"[*] Filtered WHO Windows -> Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}", flush=True)

    # Compute inverse class weights with soft smoothing (sqrt)
    train_targets = datasets["train"].targets.cpu().numpy()
    counts = np.bincount(train_targets, minlength=6)
    raw_weights = len(train_targets) / (6.0 * np.maximum(counts, 1))
    smoothed_weights = np.sqrt(raw_weights)
    smoothed_weights = smoothed_weights / np.mean(smoothed_weights)
    weights_t = torch.tensor(smoothed_weights, dtype=torch.float32, device=device)
    print(f"[*] Class Counts: {counts.tolist()}", flush=True)
    print(f"[*] Class Weights: {np.round(smoothed_weights, 2).tolist()}", flush=True)

    # Model accepts 144 dimensions
    model = BiLSTM6ClassAttention(input_dim=144, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)

    best_score = -1.0
    history = []
    t0 = time.time()

    print("[*] Training with Cosine Scheduler & Geometric Invariants...", flush=True)
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

        scheduler.step()
        train_loss = total_loss / len(datasets["train"])
        val_res = evaluate_6class(model, datasets["val"])
        val_acc, val_f1 = val_res["accuracy"], val_res["macro_f1"]

        # Balanced score combines both accuracy and macro F1
        score = 0.5 * val_acc + 0.5 * val_f1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "score": score,
        })

        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "score": best_score,
            }, args.output / "best_model.pt")
            (args.output / "val_metrics.json").write_text(json.dumps(val_res, indent=2), encoding="utf-8")

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}% | Best Score: {best_score*100:.2f}%", flush=True)

    elapsed = time.time() - t0
    print(f"[*] Training finished in {elapsed:.1f}s. Evaluating on Test Set...", flush=True)

    best_ckpt = torch.load(args.output / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_res = evaluate_6class(model, datasets["test"])
    test_res["best_epoch"] = best_ckpt["epoch"]
    test_res["best_val_acc"] = best_ckpt["val_acc"]
    test_res["best_val_macro_f1"] = best_ckpt["val_macro_f1"]
    test_res["elapsed_s"] = elapsed

    (args.output / "test_metrics.json").write_text(json.dumps(test_res, indent=2), encoding="utf-8")
    (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n" + "=" * 70, flush=True)
    print("FINAL TEST RESULTS: BALANCED GEOMETRIC 6-STEP WHO MODEL", flush=True)
    print(f"Selected Best Epoch: {best_ckpt['epoch']}", flush=True)
    print(f">> TEST ACCURACY: {test_res['accuracy']*100:.2f}% <<", flush=True)
    print(f">> TEST MACRO-F1: {test_res['macro_f1']*100:.2f}% <<", flush=True)
    print("=" * 70, flush=True)
    for name in WHO_STEP_NAMES:
        cr = test_res["classification_report"][name]
        print(f"{name:<35}: Precision={cr['precision']*100:.1f}%, Recall={cr['recall']*100:.1f}%, F1={cr['f1-score']*100:.1f}% (Support={int(cr['support'])})", flush=True)


if __name__ == "__main__":
    main()
