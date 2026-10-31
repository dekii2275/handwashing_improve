"""
Prepare METC Dataset Artifacts from extracted landmarks (NPZ).
Generates features.parquet, window_index.parquet, manifests, and validation report.
Matches the feature pipeline and participant split from the friend's repo (Ivanovs & Ivanova reproduction).
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scratch" / "friend_repo"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.dataset_metadata import create_split_manifests, participant_id_from_filename
from src.feature_engineering import feature_names, process_video
from src.sequence_builder import WINDOW_COLUMNS, build_window_index
from src.validation import validate_artifacts


def run_preparation(
    output_dir: str = "data/processed_w48_s4",
    window_size: int = 48,
    stride: int = 4,
    seed: int = 42,
    interpolation_max_seconds: float = 0.25,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Preparing dataset artifacts at: {out_path.resolve()}")

    # 1. Load statistics and match participants
    stat_path = PROJECT_ROOT / "data" / "raw" / "statistics.csv"
    if not stat_path.exists():
        raise FileNotFoundError(f"Missing {stat_path}")
    stat = pd.read_csv(stat_path)
    stat["interface_id"] = stat["location"].str.extract(r"(\d+)$").astype(int)
    stat["participant_id"] = [
        participant_id_from_filename(name, interface)
        for name, interface in zip(stat["filename"], stat["interface_id"])
    ]

    # 2. Scan all NPZ files to get frame counts per class
    records = []
    landmarks_dir = PROJECT_ROOT / "data" / "landmarks"
    for fname in stat["filename"]:
        stem = os.path.splitext(fname)[0]
        npz_file = landmarks_dir / f"{stem}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing landmarks NPZ for {fname}: {npz_file}")
        d = np.load(npz_file)
        labels = d["labels"]
        counts = np.bincount(labels, minlength=7)
        rec = {
            "filename": fname,
            "local_frame_count": len(labels),
            "local_status": "available",
            "old_split": "train",
        }
        for i in range(7):
            rec[f"local_frames_{i}"] = int(counts[i])
        records.append(rec)

    local = pd.DataFrame(records)
    manifest = stat.merge(local, on="filename", how="inner")
    manifest.to_csv(out_path / "dataset_manifest.csv", index=False, encoding="utf-8-sig")

    # 3. Create participant-level split manifests
    config = {
        "paths": {"output_dir": str(out_path)},
        "source": {"expected_participants": 72},
        "split": {
            "seed": seed,
            "train_participants": 50,
            "validation_participants": 11,
            "test_participants": 11,
            "folds": 5,
            "search_iterations": 5000,
        },
        "sequence": {
            "window_size": window_size,
            "stride": stride,
        },
    }
    split_df, fold_df = create_split_manifests(config, manifest)
    print("Participant Split Summary:")
    print(split_df["new_split"].value_counts())
    print("Unique Participants by Split:")
    print(split_df.groupby("new_split")["participant_id"].nunique().to_dict())

    # Metadata map by filename
    metadata_map = split_df.merge(
        fold_df[["filename", "fold"]], on="filename", how="inner", validate="one_to_one"
    ).set_index("filename")

    # 4. Feature configuration: all groups enabled (shape + relation + motion)
    enabled = {"shape": True, "relation": True, "motion": True}
    names = feature_names(enabled)
    feature_manifest = {
        "groups": enabled,
        "feature_count": len(names),
        "feature_names": names,
    }
    (out_path / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    feature_file = out_path / "features.parquet"
    window_file = out_path / "window_index.parquet"
    feature_writer = None
    window_writer = None
    row_offset = 0
    video_count = 0

    t0 = time.time()
    for fname in stat["filename"]:
        if fname not in metadata_map.index:
            continue
        info = metadata_map.loc[fname].to_dict()
        stem = os.path.splitext(fname)[0]
        npz_file = landmarks_dir / f"{stem}.npz"
        data = np.load(npz_file)

        coords = data["coords"].reshape(-1, 2, 21, 3)
        presence = data["presence"]
        confidence = data["confidence"]
        labels = data["labels"]
        fps = float(data["fps"]) if "fps" in data else 16.0
        T = len(labels)

        # Build raw DataFrame expected by process_video
        raw_dict = {
            "video_name": fname,
            "frame_index": np.arange(T),
            "timestamp": np.arange(T) / fps,
            "motion_label": labels,
            "left_hand_detected": presence[:, 0].astype(bool),
            "right_hand_detected": presence[:, 1].astype(bool),
            "left_hand_confidence": confidence[:, 0].astype(np.float32),
            "right_hand_confidence": confidence[:, 1].astype(np.float32),
        }
        for i in range(21):
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                raw_dict[f"left_p{i}_{axis}"] = coords[:, 0, i, axis_idx].astype(np.float32)
                raw_dict[f"right_p{i}_{axis}"] = coords[:, 1, i, axis_idx].astype(np.float32)

        raw_df = pd.DataFrame(raw_dict)

        processed = process_video(
            raw_df,
            info,
            enabled,
            interpolation_max_seconds=interpolation_max_seconds,
        )

        windows = build_window_index(
            processed,
            row_offset,
            window_size=window_size,
            stride=stride,
        )

        feature_table = pa.Table.from_pandas(processed, preserve_index=False)
        if feature_writer is None:
            feature_writer = pq.ParquetWriter(feature_file, feature_table.schema, compression="zstd")
        feature_writer.write_table(feature_table)

        if not windows.empty:
            window_table = pa.Table.from_pandas(windows[WINDOW_COLUMNS], preserve_index=False)
            if window_writer is None:
                window_writer = pq.ParquetWriter(window_file, window_table.schema, compression="zstd")
            window_writer.write_table(window_table)

        row_offset += len(processed)
        video_count += 1

    if feature_writer is not None:
        feature_writer.close()
    if window_writer is not None:
        window_writer.close()

    elapsed = time.time() - t0
    print(f"Processed {video_count} videos ({row_offset} rows) in {elapsed:.2f} seconds.")

    # 5. Run validation report
    val_report = validate_artifacts(config)
    print("Validation Report:")
    print(json.dumps(val_report, indent=2))
    if not val_report.get("valid", False):
        raise RuntimeError("Artifact validation failed!")
    print(f"Artifacts successfully validated and saved to {output_dir}")


if __name__ == "__main__":
    run_preparation()
