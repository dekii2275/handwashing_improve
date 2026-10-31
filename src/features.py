"""
Feature Engineering and Preprocessing Module (Ivanovs & Ivanova, Baltic DB&IS 2026)
Implements:
1. Missing detection handling: raw zeros vs temporal linear interpolation
2. Spatial normalization: none, norm_loc (wrist centering), norm_zoom (location + scale)
3. Geometric feature representations: raw landmarks (126), bone vectors (120), pairwise distances (30)
4. Statistical standardization: StandardScaler fitted exclusively on training set
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import COORDS_PER_LANDMARK, LANDMARKS_PER_HAND

# MediaPipe Hand Skeleton Bone Connectivity (20 bones per hand)
HAND_BONES = [
    # Palm base connections
    (0, 1),
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    # Thumb
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger
    (17, 18),
    (18, 19),
    (19, 20),
]

# Fingertip landmark indices
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


def impute_missing_landmarks(coords: np.ndarray, presence: np.ndarray):
    """
    Handle missing detections via temporal linear interpolation.
    Args:
        coords: (N, 126) landmark array
        presence: (N, 2) binary hand presence array [left, right]
    Returns:
        imputed_coords: (N, 126) interpolated coordinates
        updated_presence: (N, 2) binary hand presence array reflecting interpolated frames
    """
    n_frames = coords.shape[0]
    if n_frames <= 1:
        return coords.copy(), presence.copy()

    imputed = coords.copy()
    updated_presence = presence.copy()

    # Process left hand (indices 0:63) and right hand (indices 63:126) independently
    for hand_idx, start_col in enumerate([0, 63]):
        hand_pres = presence[:, hand_idx]
        hand_coords = imputed[:, start_col : start_col + 63].copy()

        # Mark missing frames as NaN
        missing_mask = hand_pres == 0
        if np.any(missing_mask):
            hand_coords[missing_mask] = np.nan
            # Linear interpolation along time axis; interior gaps only
            df = pd.DataFrame(hand_coords)
            df = df.interpolate(method="linear", limit_area="inside", axis=0)
            # Leading or trailing NaNs that cannot be interpolated are set to 0.0
            df = df.fillna(0.0)
            imputed[:, start_col : start_col + 63] = df.values

            # Update presence for successfully interpolated frames
            valid_interpolated = ~np.all(
                imputed[:, start_col : start_col + 63] == 0.0, axis=-1
            )
            updated_presence[:, hand_idx] = np.where(valid_interpolated, 1, 0)

    return imputed.astype(np.float32), updated_presence.astype(np.int32)


def normalize_spatial(coords: np.ndarray, norm_type: str = "none") -> np.ndarray:
    """
    Apply spatial normalization to landmark coordinates.
    Options:
        - 'none': Raw coordinates
        - 'norm_loc': Centered relative to wrist landmark
        - 'norm_zoom' / 'norm_loc_scale': Location normalized and divided by max absolute amplitude
    """
    if norm_type == "none":
        return coords.copy()

    norm_coords = coords.copy()
    n_frames = coords.shape[0]

    # Process each hand independently
    for start_col in [0, 63]:
        # Reshape to (N, 21, 3)
        hand = norm_coords[:, start_col : start_col + 63].reshape(
            n_frames, LANDMARKS_PER_HAND, COORDS_PER_LANDMARK
        )
        wrist = hand[:, 0:1, :].copy()  # (N, 1, 3)

        # Only subtract wrist if hand is present (wrist not all zero)
        hand_present = np.any(hand != 0.0, axis=(1, 2), keepdims=True)  # (N, 1, 1)
        hand_centered = np.where(hand_present, hand - wrist, hand)

        if norm_type in ["norm_zoom", "norm_loc_scale"]:
            # Max absolute amplitude across coordinates in each frame
            abs_max = np.max(
                np.abs(hand_centered), axis=(1, 2), keepdims=True
            )  # (N, 1, 1)
            scale = np.where(abs_max > 1e-6, abs_max, 1.0)
            hand_centered = hand_centered / scale

        norm_coords[:, start_col : start_col + 63] = hand_centered.reshape(n_frames, 63)

    return norm_coords.astype(np.float32)


def compute_bone_vectors(coords: np.ndarray) -> np.ndarray:
    """
    Compute 3D bone displacement vectors between connected joints.
    Returns:
        bone_feats: (N, 120) [60 dims for left hand, 60 dims for right hand]
    """
    n_frames = coords.shape[0]
    all_bones = []

    for start_col in [0, 63]:
        hand = coords[:, start_col : start_col + 63].reshape(
            n_frames, LANDMARKS_PER_HAND, COORDS_PER_LANDMARK
        )
        hand_bones = []
        for j1, j2 in HAND_BONES:
            vec = hand[:, j2, :] - hand[:, j1, :]  # (N, 3)
            hand_bones.append(vec)
        hand_bones = np.concatenate(hand_bones, axis=-1)  # (N, 60)
        all_bones.append(hand_bones)

    return np.concatenate(all_bones, axis=-1).astype(np.float32)  # (N, 120)


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between selected landmarks:
    - Wrist to 5 fingertips (5 distances)
    - Fingertip to fingertip pairs (10 distances)
    Total: 15 per hand x 2 hands = 30 distances.
    """
    n_frames = coords.shape[0]
    all_dists = []

    for start_col in [0, 63]:
        hand = coords[:, start_col : start_col + 63].reshape(
            n_frames, LANDMARKS_PER_HAND, COORDS_PER_LANDMARK
        )
        wrist = hand[:, 0, :]  # (N, 3)
        dists = []

        # Wrist to fingertips
        for tip_idx in FINGERTIP_INDICES:
            tip = hand[:, tip_idx, :]
            d = np.linalg.norm(tip - wrist, axis=-1, keepdims=True)  # (N, 1)
            dists.append(d)

        # Fingertip pairs (10 pairs)
        for i in range(len(FINGERTIP_INDICES)):
            for j in range(i + 1, len(FINGERTIP_INDICES)):
                tip_i = hand[:, FINGERTIP_INDICES[i], :]
                tip_j = hand[:, FINGERTIP_INDICES[j], :]
                d = np.linalg.norm(tip_j - tip_i, axis=-1, keepdims=True)  # (N, 1)
                dists.append(d)

        dists = np.concatenate(dists, axis=-1)  # (N, 15)
        all_dists.append(dists)

    return np.concatenate(all_dists, axis=-1).astype(np.float32)  # (N, 30)


def extract_features(
    coords: np.ndarray,
    presence: np.ndarray,
    feature_type: str = "raw_landmarks",
    norm_type: str = "none",
    impute: bool = False,
) -> np.ndarray:
    """
    Complete feature engineering pipeline for a single video sequence.
    Args:
        coords: (N, 126)
        presence: (N, 2)
        feature_type: 'raw_landmarks', 'bone_vectors', 'pairwise_distances', or 'all'
        norm_type: 'none', 'norm_loc', 'norm_zoom'
        impute: bool
    Returns:
        features: (N, D)
    """
    # 1. Missing detection handling
    if impute:
        processed_coords, processed_presence = impute_missing_landmarks(
            coords, presence
        )
    else:
        processed_coords = coords.copy()
        processed_presence = presence.copy()

    # 2. Spatial normalization
    processed_coords = normalize_spatial(processed_coords, norm_type=norm_type)

    # 3. Geometric representation
    if feature_type == "raw_landmarks":
        feats = processed_coords
    elif feature_type == "bone_vectors":
        feats = compute_bone_vectors(processed_coords)
    elif feature_type == "pairwise_distances":
        feats = compute_pairwise_distances(processed_coords)
    elif feature_type == "all":
        bones = compute_bone_vectors(processed_coords)
        dists = compute_pairwise_distances(processed_coords)
        feats = np.concatenate([processed_coords, bones, dists], axis=-1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return feats.astype(np.float32), processed_presence.astype(np.int32)


class FeatureStandardizer:
    """StandardScaler wrapper fitted strictly on the Training split to prevent data leakage."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, train_feature_list):
        """Fit scaler on a list of (N_i, D) feature arrays from training videos."""
        all_train_frames = np.concatenate(train_feature_list, axis=0)
        self.scaler.fit(all_train_frames)
        self.is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "StandardScaler must be fitted on training data before transforming."
            )
        return self.scaler.transform(features).astype(np.float32)
