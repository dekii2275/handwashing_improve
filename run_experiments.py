"""
CLI Runner for Hand-Washing Movement Recognition Experiments
Reproduces the experiments from Ivanovs & Ivanova (Baltic DB&IS 2026).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BEST_CONFIGS
from src.download_dataset import download_all_annotations
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Hand-Washing Recognition Experiments (Ivanovs & Ivanova 2026)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="temporal_cnn",
        choices=["temporal_cnn", "gru", "lstm", "all"],
        help="Model architecture to train and evaluate",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
    parser.add_argument(
        "--window-size", type=int, default=None, help="Sliding window size T in frames"
    )
    parser.add_argument(
        "--norm-type",
        type=str,
        default=None,
        choices=["none", "norm_loc", "norm_zoom"],
        help="Spatial normalization type",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default=None,
        choices=["raw_landmarks", "bone_vectors", "pairwise_distances", "all"],
        help="Feature representation",
    )
    parser.add_argument(
        "--impute",
        action="store_true",
        default=None,
        help="Enable linear interpolation for missing hands",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos per split for quick verification",
    )
    parser.add_argument(
        "--download-annotations",
        action="store_true",
        help="Download metadata and all annotations from Zenodo before running",
    )

    args = parser.parse_args()

    if args.download_annotations:
        print("Ensuring all dataset annotations are downloaded...")
        download_all_annotations()

    models_to_run = (
        ["temporal_cnn", "gru", "lstm"] if args.model == "all" else [args.model]
    )

    all_results = {}
    for m in models_to_run:
        print(f"\n{'=' * 70}\nSTARTING EXPERIMENT FOR MODEL: {m.upper()}\n{'=' * 70}")
        res = run_pipeline(
            model_type=m,
            feature_type=args.feature_type,
            norm_type=args.norm_type,
            impute=args.impute,
            window_size=args.window_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_videos_per_split=args.max_videos,
        )
        all_results[m] = res

    # Print final summary comparison table
    print("\n" + "=" * 75)
    print("                 EXPERIMENTAL RESULTS COMPARISON TABLE")
    print("=" * 75)
    print(
        f"{'Model Architecture':<18} | {'Paper Acc (%)':<14} | {'Our Acc (%)':<12} | {'Paper F1 (%)':<13} | {'Our F1 (%)':<10}"
    )
    print("-" * 75)
    for m in models_to_run:
        res = all_results.get(m, {})
        rep_acc = BEST_CONFIGS.get(m, {}).get("reported_acc", "N/A")
        rep_f1 = BEST_CONFIGS.get(m, {}).get("reported_f1", "N/A")
        our_acc = f"{res.get('test_accuracy', 0.0):.2f}%"
        our_f1 = f"{res.get('test_macro_f1', 0.0):.2f}%"
        print(
            f"{m.upper():<18} | {str(rep_acc) + '%':<14} | {our_acc:<12} | {str(rep_f1) + '%':<13} | {our_f1:<10}"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
