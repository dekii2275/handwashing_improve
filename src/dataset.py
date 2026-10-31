"""
PyTorch Dataset and Sequence Construction Module (Ivanovs & Ivanova, Baltic DB&IS 2026)
Implements sliding window generation:
- Stride = 1
- Window lengths T in {15, 24, 30, 48}
- Target: label of the final frame in the window
- Filter rule: At least one hand detected in every frame of the window
- Continuous sequences per video (no cross-video spanning)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import NUM_CLASSES


class HandwashingSequenceDataset(Dataset):
    """
    Dataset of fixed-length sliding window sequences for hand movement recognition.
    """

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Args:
            sequences: (N_samples, T, D) float32 array
            labels: (N_samples,) int64 array
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def build_sliding_windows(
    features_list: list[np.ndarray],
    presence_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    window_size: int = 48,
    stride: int = 1,
    filter_missing_hands: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct sliding window samples from a list of video feature arrays.
    Args:
        features_list: List of (N_frames_i, D) standardized feature arrays
        presence_list: List of (N_frames_i, 2) binary hand presence arrays
        labels_list: List of (N_frames_i,) frame label arrays
        window_size: T frames in sliding window (e.g. 15, 24, 30, 48)
        stride: Window stride (default 1)
        filter_missing_hands: Retain only windows where at least one hand is detected in EVERY frame
    Returns:
        all_windows: (N_total, T, D) float32 array
        all_targets: (N_total,) int64 array
    """
    all_windows = []
    all_targets = []

    for feats, presence, labels in zip(features_list, presence_list, labels_list):
        n_frames = len(feats)
        if n_frames < window_size:
            continue

        # Check frame-level presence: at least one hand present (left OR right)
        at_least_one_hand = np.any(presence > 0, axis=-1)  # (N_frames,)

        # Slide window across frames within this video
        for start_idx in range(0, n_frames - window_size + 1, stride):
            end_idx = start_idx + window_size  # exclusive

            # Filter rule from paper: at least one hand must be detected in every frame of the window
            if filter_missing_hands and not np.all(
                at_least_one_hand[start_idx:end_idx]
            ):
                continue

            window = feats[start_idx:end_idx]  # (T, D)
            target = labels[end_idx - 1]  # Label of the final frame

            # Ignore invalid class labels (< 0 or >= NUM_CLASSES)
            if 0 <= target < NUM_CLASSES:
                all_windows.append(window)
                all_targets.append(target)

    if len(all_windows) == 0:
        d = features_list[0].shape[-1] if features_list else 126
        return np.zeros((0, window_size, d), dtype=np.float32), np.zeros(
            (0,), dtype=np.int64
        )

    return np.array(all_windows, dtype=np.float32), np.array(
        all_targets, dtype=np.int64
    )


def create_dataloader(
    sequences: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for sequences."""
    dataset = HandwashingSequenceDataset(sequences, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
