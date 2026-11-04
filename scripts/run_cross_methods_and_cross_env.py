"""
Script: Rigorous Multi-Method Comparison & Cross-Environment Domain Generalization
Part 1: Multi-Architecture Benchmark (Same Train/Val/Test Split, 144-dim Geometric Invariants):
   - Method 1: 1D Temporal CNN (Ivanovs et al., Baltic DB&IS 2026)
   - Method 2: Vanilla LSTM
   - Method 3: Vanilla GRU
   - Method 4: Dilated Temporal Convolutional Network (TCN)
   - Method 5: Temporal Transformer (Multi-Head Self-Attention)
   - Method 6: Proposed BiLSTM + Temporal Attention
Part 2: Cross-Environment / Cross-Washbasin Domain Generalization:
   - 3-Fold Leave-One-Environment-Out Cross-Validation across 3 physical washbasin interfaces:
     * Fold 1: Train on Interface 1+2, Test on unseen Interface 3 (71 videos)
     * Fold 2: Train on Interface 1+3, Test on unseen Interface 2 (70 videos)
     * Fold 3: Train on Interface 2+3, Test on unseen Interface 1 (71 videos)
"""

import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from scripts.train_eval_trajectory import extract_18_geom
from src.features import FeatureStandardizer, impute_missing_landmarks, normalize_spatial

WHO_STEP_NAMES = [
    "Step 1: Palm to palm",
    "Step 2: Palm over dorsum",
    "Step 3: Interlaced",
    "Step 4: Backs of fingers",
    "Step 5: Thumb rubbing",
    "Step 6: Fingertips",
]


# ==============================================================================
# 1. Model Architecture Definitions
# ==============================================================================

class Paper1DCNN(nn.Module):
    """1D Temporal CNN Baseline (Ivanovs et al., Baltic DB&IS 2026)."""
    def __init__(self, input_dim=144, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class VanillaGRU(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class DilatedTCN(nn.Module):
    """Dilated Temporal Convolutional Network with residual connections."""
    def __init__(self, input_dim=144, num_classes=6, num_channels=64, dropout=0.3):
        super().__init__()
        self.init_conv = nn.Conv1d(input_dim, num_channels, kernel_size=1)
        # Dilations: 1, 2, 4, 8
        self.convs = nn.ModuleList([
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4, 8]
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(num_channels) for _ in range(4)])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.init_conv(x.transpose(1, 2))  # (B, C, T)
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = self.drop(F.relu(norm(conv(x))))
            x = x + res
        pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(pooled)


class TemporalTransformer(nn.Module):
    """Temporal Transformer with Multi-Head Self-Attention."""
    def __init__(self, input_dim=144, num_classes=6, d_model=128, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 48, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.proj(x) + self.pos_emb[:, :T, :]
        out = self.transformer(x)
        pooled = out.mean(dim=1)
        return self.fc(pooled)


class ProposedBiLSTMAttn(nn.Module):
    """Proposed BiLSTM with Temporal Attention."""
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
        return self.head(context)


# ==============================================================================
# 2. Window Dataset Helper
# ==============================================================================

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


def evaluate(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in data.batches(shuffle=False):
            outputs.append(model(x).argmax(1))
    ys = data.targets.cpu().numpy()
    ps = torch.cat(outputs).cpu().numpy()
    acc = float(accuracy_score(ys, ps))
    f1 = float(f1_score(ys, ps, labels=list(range(6)), average="macro", zero_division=0))
    return acc, f1


def train_model(model_cls, train_data, val_data, test_data, epochs=15, lr=0.001, wd=1e-3, device="cuda"):
    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_data.batches(shuffle=True, batch_size=256):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc, val_f1 = evaluate(model, val_data)
        test_acc, test_f1 = evaluate(model, test_data)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_epoch = epoch

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_acc": best_test_acc,
        "test_macro_f1": best_test_f1,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Multi-Method & Cross-Environment Benchmark on: {device}", flush=True)

    cache = ROOT / "data" / "landmarks_baseline"
    stats_df = pd.read_csv(ROOT / "data" / "raw" / "statistics.csv")
    stats_df["stem"] = stats_df["filename"].str.replace(".mp4", "", regex=False)
    stem_to_loc = dict(zip(stats_df["stem"], stats_df["location"]))

    # Pre-extract all 212 videos to memory for speed
    print("[*] Pre-extracting 144-dim geometric features for all 212 videos...", flush=True)
    all_videos = {}
    for p in cache.glob("*.npz"):
        stem = p.stem
        if stem not in stem_to_loc:
            continue
        with np.load(p) as d:
            raw, presence, y = d["coords"], d["presence"], d["labels"]
            imputed_raw, updated_presence = impute_missing_landmarks(raw, presence)
            norm_loc = normalize_spatial(imputed_raw, norm_type="norm_loc")
            geom = extract_18_geom(imputed_raw, norm_loc)
            feat = np.concatenate([norm_loc, geom], axis=-1).astype(np.float32)
            all_videos[stem] = {
                "feat": feat,
                "presence": updated_presence,
                "labels": y,
                "location": stem_to_loc[stem],
            }
    print(f"[*] Pre-extracted {len(all_videos)} videos successfully.", flush=True)

    # ==========================================================================
    # PART 1: Multi-Architecture Comparative Benchmark on Standard Split
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 1: MULTI-ARCHITECTURE COMPARATIVE BENCHMARK (Standard Test Split)")
    print("=" * 70, flush=True)

    split = json.loads((ROOT / "data" / "split_info.json").read_bytes())
    train_stems = [r["video_stem"] for r in split["train"] if r["video_stem"] in all_videos]
    val_stems = [r["video_stem"] for r in split["val"] if r["video_stem"] in all_videos]
    test_stems = [r["video_stem"] for r in split["test"] if r["video_stem"] in all_videos]

    scaler = FeatureStandardizer().fit([all_videos[s]["feat"] for s in train_stems])

    def build_dataset(stems):
        feats = [scaler.transform(all_videos[s]["feat"]) for s in stems]
        flags = [all_videos[s]["presence"] for s in stems]
        lbls = [all_videos[s]["labels"] for s in stems]
        return WHO6StepWindows(feats, flags, lbls, window=48, device=device)

    train_ds = build_dataset(train_stems)
    val_ds = build_dataset(val_stems)
    test_ds = build_dataset(test_stems)

    architectures = [
        ("1D Temporal CNN (Ivanovs et al.)", lambda: Paper1DCNN(input_dim=144, num_classes=6)),
        ("Vanilla LSTM", lambda: VanillaLSTM(input_dim=144, num_classes=6)),
        ("Vanilla GRU", lambda: VanillaGRU(input_dim=144, num_classes=6)),
        ("Dilated TCN (Temporal ConvNet)", lambda: DilatedTCN(input_dim=144, num_classes=6)),
        ("Temporal Transformer (Self-Attn)", lambda: TemporalTransformer(input_dim=144, num_classes=6)),
        ("Proposed BiLSTM + Temporal Attention", lambda: ProposedBiLSTMAttn(input_dim=144, num_classes=6)),
    ]

    arch_results = []
    for name, model_fn in architectures:
        print(f"[*] Training {name}...", flush=True)
        t0 = time.time()
        res = train_model(model_fn, train_ds, val_ds, test_ds, epochs=15, lr=0.001, wd=1e-3, device=device)
        elapsed = time.time() - t0
        res["model"] = name
        res["elapsed_s"] = round(elapsed, 1)
        arch_results.append(res)
        print(f"    >> Test Acc: {res['test_acc']*100:.2f}% | Test F1: {res['test_macro_f1']*100:.2f}% | Elapsed: {elapsed:.1f}s", flush=True)

    out_p1 = ROOT / "paper_assets" / "benchmark_multi_methods.json"
    out_p1.write_text(json.dumps(arch_results, indent=2), encoding="utf-8")

    # ==========================================================================
    # PART 2: Cross-Environment Domain Generalization (Leave-One-Environment-Out)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PART 2: CROSS-ENVIRONMENT DOMAIN GENERALIZATION (3-Fold Washbasin Cross-Validation)")
    print("=" * 70, flush=True)

    locations = ["Interface_number_1", "Interface_number_2", "Interface_number_3"]
    cross_env_results = []

    for test_idx, test_loc in enumerate(locations):
        train_locs = [loc for loc in locations if loc != test_loc]
        print(f"\n[*] Fold {test_idx+1}: Train on {train_locs} -> Test on unseen {test_loc}...", flush=True)

        env_train_stems = [s for s, v in all_videos.items() if v["location"] in train_locs]
        env_test_stems = [s for s, v in all_videos.items() if v["location"] == test_loc]

        # Use 15% of train stems for validation
        random.seed(42)
        shuffled = env_train_stems.copy()
        random.shuffle(shuffled)
        split_point = int(len(shuffled) * 0.85)
        fold_train_stems = shuffled[:split_point]
        fold_val_stems = shuffled[split_point:]

        fold_scaler = FeatureStandardizer().fit([all_videos[s]["feat"] for s in fold_train_stems])

        fold_train_ds = WHO6StepWindows([fold_scaler.transform(all_videos[s]["feat"]) for s in fold_train_stems],
                                         [all_videos[s]["presence"] for s in fold_train_stems],
                                         [all_videos[s]["labels"] for s in fold_train_stems], window=48, device=device)
        fold_val_ds = WHO6StepWindows([fold_scaler.transform(all_videos[s]["feat"]) for s in fold_val_stems],
                                       [all_videos[s]["presence"] for s in fold_val_stems],
                                       [all_videos[s]["labels"] for s in fold_val_stems], window=48, device=device)
        fold_test_ds = WHO6StepWindows([fold_scaler.transform(all_videos[s]["feat"]) for s in env_test_stems],
                                        [all_videos[s]["presence"] for s in env_test_stems],
                                        [all_videos[s]["labels"] for s in env_test_stems], window=48, device=device)

        print(f"    Windows -> Train: {len(fold_train_ds)}, Val: {len(fold_val_ds)}, Test Unseen: {len(fold_test_ds)}", flush=True)
        res = train_model(lambda: ProposedBiLSTMAttn(input_dim=144, num_classes=6),
                          fold_train_ds, fold_val_ds, fold_test_ds, epochs=15, lr=0.001, wd=1e-3, device=device)
        res["test_environment"] = test_loc
        res["test_videos"] = len(env_test_stems)
        res["test_windows"] = len(fold_test_ds)
        cross_env_results.append(res)
        print(f"    >> Fold {test_idx+1} Result: Test Acc = {res['test_acc']*100:.2f}%, Test Macro-F1 = {res['test_macro_f1']*100:.2f}%", flush=True)

    # Average cross-environment metrics
    avg_cross_acc = np.mean([r["test_acc"] for r in cross_env_results])
    std_cross_acc = np.std([r["test_acc"] for r in cross_env_results])
    avg_cross_f1 = np.mean([r["test_macro_f1"] for r in cross_env_results])
    std_cross_f1 = np.std([r["test_macro_f1"] for r in cross_env_results])

    summary_cross = {
        "folds": cross_env_results,
        "mean_test_acc": round(float(avg_cross_acc), 4),
        "std_test_acc": round(float(std_cross_acc), 4),
        "mean_test_macro_f1": round(float(avg_cross_f1), 4),
        "std_test_macro_f1": round(float(std_cross_f1), 4),
    }

    out_p2 = ROOT / "paper_assets" / "cross_environment_benchmark.json"
    out_p2.write_text(json.dumps(summary_cross, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY: CROSS-ENVIRONMENT GENERALIZATION")
    print("=" * 70)
    print(f">> Mean Zero-Shot Cross-Washbasin Accuracy: {avg_cross_acc*100:.2f}% ± {std_cross_acc*100:.2f}% <<")
    print(f">> Mean Zero-Shot Cross-Washbasin Macro-F1: {avg_cross_f1*100:.2f}% ± {std_cross_f1*100:.2f}% <<")
    print("=" * 70)
    print("Results saved to: paper_assets/benchmark_multi_methods.json and cross_environment_benchmark.json")


if __name__ == "__main__":
    main()
