"""
End-to-End Pipeline for Hand-Washing Recognition (Ivanovs & Ivanova, Baltic DB&IS 2026)
Coordinates:
1. Video/Annotation loading & splitting (70/10/20)
2. MediaPipe Landmark Extraction & caching
3. Feature Engineering (imputation, spatial normalization, geometric representations)
4. StandardScaler fitted strictly on Train split
5. Sliding Window Sequence Construction (stride=1, length T)
6. Model Training with Validation Macro F1 checkpointing
7. Test Set Evaluation & Comparison with published results
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

from config import (
    BEST_CONFIGS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    LANDMARKS_DIR,
    RAW_DIR,
    RESULTS_DIR,
)
from src.dataset import build_sliding_windows, create_dataloader
from src.evaluate import evaluate_test_set
from src.extract_landmarks import HandLandmarkExtractor
from src.features import FeatureStandardizer, extract_features
from src.models import get_model
from src.split_data import load_split
from src.train import train_model
from src.visualize import plot_confusion_matrix, plot_training_history


def prepare_video_landmarks(split_dict: dict, max_videos_per_split: int | None = None):
    """
    Ensure landmark .npz files exist for all videos across train, val, and test splits.
    If video file is downloaded, extracts using MediaPipe.
    Refuses synthetic-looking caches and missing videos rather than generating
    label-conditioned landmarks in a research experiment.
    """
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    extractor = None

    for split_name in ["train", "val", "test"]:
        video_records = split_dict[split_name]
        if max_videos_per_split:
            video_records = video_records[:max_videos_per_split]

        for record in tqdm(video_records, desc=f"Landmarks [{split_name}]"):
            stem = record["video_stem"]
            loc = record["location"]
            filename = record["filename"]
            npz_path = LANDMARKS_DIR / f"{stem}.npz"

            if npz_path.exists():
                with np.load(npz_path) as cached:
                    confidence = cached["confidence"]
                    if np.any(confidence == np.float32(0.9)) and np.all(
                        (confidence == 0) | (confidence == np.float32(0.9))
                    ):
                        raise RuntimeError(
                            f"Synthetic landmark fingerprint: {npz_path}. "
                            "Use scripts/prepare_baseline_landmarks.py and scripts/run_paper_cnn.py."
                        )
                continue

            # Look for video file and annotation file
            video_path = RAW_DIR / loc / "Videos" / filename
            annot_path = RAW_DIR / loc / "Annotations" / "Annotator_1" / f"{stem}.json"
            if not annot_path.exists():
                # Check alternative annotator path
                alt_annot = RAW_DIR / loc / "Annotations" / f"{stem}.json"
                if alt_annot.exists():
                    annot_path = alt_annot

            if video_path.exists() and annot_path.exists():
                if extractor is None:
                    extractor = HandLandmarkExtractor()
                extractor.extract_from_video(
                    video_path, annot_path, output_path=npz_path
                )
            else:
                raise FileNotFoundError(f"Video or annotation missing for {stem}")

    if extractor is not None:
        extractor.close()


def load_split_features(
    split_records: list,
    feature_type: str = "raw_landmarks",
    norm_type: str = "norm_loc",
    impute: bool = True,
):
    """
    Load raw landmarks, extract geometric features, and return lists per video.
    """
    features_list = []
    presence_list = []
    labels_list = []

    for record in split_records:
        stem = record["video_stem"]
        npz_path = LANDMARKS_DIR / f"{stem}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing landmark cache: {npz_path}")

        data = np.load(npz_path)
        coords = data["coords"]
        presence = data["presence"]
        labels = data["labels"]

        # Feature extraction
        feats, pres_upd = extract_features(
            coords=coords,
            presence=presence,
            feature_type=feature_type,
            norm_type=norm_type,
            impute=impute,
        )

        features_list.append(feats)
        presence_list.append(pres_upd)
        labels_list.append(labels)

    return features_list, presence_list, labels_list


def run_pipeline(
    model_type: str = "temporal_cnn",
    feature_type: str | None = None,
    norm_type: str | None = None,
    impute: bool | None = None,
    window_size: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LEARNING_RATE,
    max_videos_per_split: int | None = None,
):
    """
    Execute complete end-to-end pipeline for a specified architecture.
    """
    best_defaults = BEST_CONFIGS.get(model_type, BEST_CONFIGS["temporal_cnn"])
    feature_type = feature_type or best_defaults["feature_type"]
    norm_type = norm_type or best_defaults["norm_type"]
    impute = impute if impute is not None else best_defaults["impute"]
    window_size = window_size or best_defaults["window_size"]

    print("=" * 65)
    print(f"RUNNING PIPELINE: {model_type.upper()}")
    print("  Configuration:")
    print(f"    - Feature Representation: {feature_type}")
    print(f"    - Spatial Normalization:  {norm_type}")
    print(f"    - Imputation Enabled:     {impute}")
    print(f"    - Window Size (T):        {window_size} frames")
    print(f"    - Epochs:                 {epochs}")
    print(f"    - Batch Size:             {batch_size}")
    print(f"    - Learning Rate:          {lr}")
    print("=" * 65)

    # 1. Load 70/10/20 Split
    split_dict = load_split()

    # 2. Extract or cache landmarks
    print("\n[Step 1/5] Checking landmark cache...")
    prepare_video_landmarks(split_dict, max_videos_per_split=max_videos_per_split)

    # 3. Feature extraction per split
    print("\n[Step 2/5] Extracting geometric features...")
    train_recs = (
        split_dict["train"][:max_videos_per_split]
        if max_videos_per_split
        else split_dict["train"]
    )
    val_recs = (
        split_dict["val"][:max_videos_per_split]
        if max_videos_per_split
        else split_dict["val"]
    )
    test_recs = (
        split_dict["test"][:max_videos_per_split]
        if max_videos_per_split
        else split_dict["test"]
    )

    tr_feats, tr_pres, tr_labels = load_split_features(
        train_recs, feature_type, norm_type, impute
    )
    val_feats, val_pres, val_labels = load_split_features(
        val_recs, feature_type, norm_type, impute
    )
    te_feats, te_pres, te_labels = load_split_features(
        test_recs, feature_type, norm_type, impute
    )

    # 4. Fit StandardScaler exclusively on Train split
    print("\n[Step 3/5] Fitting StandardScaler strictly on Train split...")
    scaler = FeatureStandardizer()
    scaler.fit(tr_feats)

    tr_feats = [scaler.transform(f) for f in tr_feats]
    val_feats = [scaler.transform(f) for f in val_feats]
    te_feats = [scaler.transform(f) for f in te_feats]

    # 5. Build sliding windows (stride=1, window_size=T)
    print(
        f"\n[Step 4/5] Building sliding window sequences (T={window_size}, stride=1)..."
    )
    x_train, y_train = build_sliding_windows(
        tr_feats, tr_pres, tr_labels, window_size=window_size
    )
    x_val, y_val = build_sliding_windows(
        val_feats, val_pres, val_labels, window_size=window_size
    )
    x_test, y_test = build_sliding_windows(
        te_feats, te_pres, te_labels, window_size=window_size
    )

    print(f"  Train sequences: {len(y_train)} samples")
    print(f"  Val sequences:   {len(y_val)} samples")
    print(f"  Test sequences:  {len(y_test)} samples")

    train_loader = create_dataloader(
        x_train, y_train, batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(
        x_test, y_test, batch_size=batch_size, shuffle=False
    )

    input_dim = x_train.shape[-1]

    # 6. Instantiate model
    model = get_model(
        model_name=model_type,
        input_dim=input_dim,
        filters=best_defaults.get("filters", 64),
        kernel_size=best_defaults.get("kernel_size", 3),
        hidden_size=best_defaults.get("hidden_size", 128),
        num_layers=best_defaults.get("num_layers", 2),
    )

    # 7. Train model
    print(f"\n[Step 5/5] Training {model_type} for {epochs} epochs...")
    ckpt_path, _history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_type,
        epochs=epochs,
        lr=lr,
    )

    # 8. Evaluate on held-out Test set
    test_results = evaluate_test_set(
        model=model,
        checkpoint_path=ckpt_path,
        test_loader=test_loader,
        model_name=model_type,
    )

    # 9. Plot curves and confusion matrix
    history_json = RESULTS_DIR / f"history_best_model_{model_type}.json"
    if history_json.exists():
        plot_training_history(history_json)
    if "confusion_matrix" in test_results:
        plot_confusion_matrix(
            test_results["confusion_matrix"],
            title=f"Confusion Matrix: {model_type.upper()} (Test Set)",
        )

    return test_results
