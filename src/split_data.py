"""
Video-level Data Split Module (70% Train / 10% Validation / 20% Test)
Ensures frames from the same video never cross splits, adhering strictly to the paper.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_SEED, RAW_DIR, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def get_video_list_from_statistics():
    """Retrieve the 212 video records from statistics.csv."""
    stat_file = RAW_DIR / "statistics.csv"
    if not stat_file.exists():
        # Look in root if not in raw
        alt = PROJECT_ROOT / "statistics.csv"
        if alt.exists():
            stat_file = alt
        else:
            raise FileNotFoundError(f"statistics.csv not found at {stat_file}")

    df = pd.read_csv(stat_file)
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "filename": row["filename"],
                "location": row["location"],
                "total_duration": float(row["total_duration"]),
                "washing_duration": float(row["washing_duration"]),
                "video_stem": Path(row["filename"]).stem,
            }
        )
    return records


def create_train_val_test_split(seed: int = RANDOM_SEED):
    """
    Split the 212 videos into Train (70%), Val (10%), and Test (20%).
    Saves split metadata to data/split_info.json.
    """
    records = get_video_list_from_statistics()
    n_total = len(records)
    print(f"Total videos to split: {n_total}")

    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_total)

    n_train = int(n_total * TRAIN_RATIO)  # 148
    n_val = int(n_total * VAL_RATIO)  # 21
    n_test = n_total - n_train - n_val  # 43
    print(f"Splits: Train={n_train}, Val={n_val}, Test={n_test}")

    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train : n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val :]

    split_dict = {
        "metadata": {
            "total_videos": n_total,
            "train_count": len(train_indices),
            "val_count": len(val_indices),
            "test_count": len(test_indices),
            "seed": seed,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
        },
        "train": [records[i] for i in train_indices],
        "val": [records[i] for i in val_indices],
        "test": [records[i] for i in test_indices],
    }

    out_path = DATA_DIR / "split_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    print("Split created successfully:")
    print(
        f"  Train: {len(train_indices)} videos ({len(train_indices) / n_total * 100:.1f}%)"
    )
    print(
        f"  Val:   {len(val_indices)} videos ({len(val_indices) / n_total * 100:.1f}%)"
    )
    print(
        f"  Test:  {len(test_indices)} videos ({len(test_indices) / n_total * 100:.1f}%)"
    )
    print(f"Saved to: {out_path}")
    return split_dict


def load_split():
    """Load the split configuration from split_info.json."""
    split_path = DATA_DIR / "split_info.json"
    if not split_path.exists():
        return create_train_val_test_split()
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    create_train_val_test_split()
