"""
Unit tests to verify all components of the pipeline:
1. Neural network architectures (CNN, GRU, LSTM)
2. Normalization, imputation, bone vectors, pairwise distances
3. Sequence construction and sliding window generator
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config import NUM_CLASSES, RAW_FEATURE_DIM
from src.dataset import build_sliding_windows
from src.features import (
    FeatureStandardizer,
    compute_bone_vectors,
    compute_pairwise_distances,
    impute_missing_landmarks,
    normalize_spatial,
)
from src.models import HandGRU, HandLSTM, HandTemporalCNN


def test_models():
    print("Testing model forward passes...")
    batch_size = 8
    seq_len = 48
    input_dim = 126
    x = torch.randn(batch_size, seq_len, input_dim)

    # 1D Temporal CNN
    cnn = HandTemporalCNN(
        input_dim=input_dim, num_filters=64, kernel_size=3, num_layers=2
    )
    cnn_out = cnn(x)
    assert cnn_out.shape == (batch_size, NUM_CLASSES), (
        f"CNN output shape mismatch: {cnn_out.shape}"
    )
    print("  [PASS] HandTemporalCNN output shape:", cnn_out.shape)

    # GRU
    gru = HandGRU(input_dim=input_dim, hidden_size=128, num_layers=2)
    gru_out = gru(x)
    assert gru_out.shape == (batch_size, NUM_CLASSES), (
        f"GRU output shape mismatch: {gru_out.shape}"
    )
    print("  [PASS] HandGRU output shape:", gru_out.shape)

    # LSTM
    lstm = HandLSTM(input_dim=input_dim, hidden_size=64, num_layers=2)
    lstm_out = lstm(x)
    assert lstm_out.shape == (batch_size, NUM_CLASSES), (
        f"LSTM output shape mismatch: {lstm_out.shape}"
    )
    print("  [PASS] HandLSTM output shape:", lstm_out.shape)


def test_features():
    print("\nTesting feature engineering...")
    n_frames = 100
    rng = np.random.RandomState(42)
    coords = rng.randn(n_frames, RAW_FEATURE_DIM).astype(np.float32)
    presence = np.ones((n_frames, 2), dtype=np.int32)
    # inject missing hand detections
    presence[10:15, 0] = 0
    coords[10:15, :63] = 0.0

    # Imputation
    imputed, pres_upd = impute_missing_landmarks(coords, presence)
    assert imputed.shape == coords.shape
    assert pres_upd.shape == presence.shape
    # Missing gap at 10:15 should now be linearly interpolated (non-zero)
    assert np.all(imputed[12, :63] != 0.0), "Imputation failed to fill interior gap!"
    assert pres_upd[12, 0] == 1, (
        "Updated presence flag should be set for interpolated frames!"
    )
    print(
        "  [PASS] Imputation correctly filled missing landmark frames and updated presence."
    )

    # Spatial Normalization
    norm_loc = normalize_spatial(coords, norm_type="norm_loc")
    assert norm_loc.shape == coords.shape
    norm_zoom = normalize_spatial(coords, norm_type="norm_zoom")
    assert norm_zoom.shape == coords.shape
    print(
        "  [PASS] Spatial normalizations (norm_loc, norm_zoom) executed successfully."
    )

    # Bone vectors
    bones = compute_bone_vectors(coords)
    assert bones.shape == (n_frames, 120), f"Bone shape mismatch: {bones.shape}"
    print("  [PASS] Bone vectors computed shape:", bones.shape)

    # Pairwise distances
    dists = compute_pairwise_distances(coords)
    assert dists.shape == (n_frames, 30), f"Pairwise dist shape mismatch: {dists.shape}"
    print("  [PASS] Pairwise distances computed shape:", dists.shape)

    # FeatureStandardizer
    scaler = FeatureStandardizer()
    scaler.fit([coords[:50], coords[50:]])
    transformed = scaler.transform(coords)
    assert transformed.shape == coords.shape
    print("  [PASS] FeatureStandardizer fit & transform successful.")


def test_sliding_windows():
    print("\nTesting sliding window sequence generator...")
    n_frames = 100
    feats = np.random.randn(n_frames, 126).astype(np.float32)
    presence = np.ones((n_frames, 2), dtype=np.int32)
    labels = np.random.randint(0, NUM_CLASSES, size=n_frames)

    windows, targets = build_sliding_windows(
        [feats], [presence], [labels], window_size=48, stride=1
    )
    expected_samples = n_frames - 48 + 1  # 53
    assert len(windows) == expected_samples, (
        f"Expected {expected_samples}, got {len(windows)}"
    )
    assert windows.shape == (expected_samples, 48, 126)
    assert targets.shape == (expected_samples,)
    # Verify label is target of final frame
    assert targets[0] == labels[47]
    assert targets[-1] == labels[-1]
    print(
        f"  [PASS] Sliding windows shape: {windows.shape}, targets shape: {targets.shape}"
    )


if __name__ == "__main__":
    test_models()
    test_features()
    test_sliding_windows()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
