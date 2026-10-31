"""
Batch MediaPipe Landmark Extraction with Multiprocessing
Extracts 3D hand skeleton landmarks across all 212 METC videos.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from config import LANDMARKS_DIR, RAW_DIR
from src.extract_landmarks import HandLandmarkExtractor
from src.split_data import load_split


def _extract_video_task(record):
    """Worker task to extract one video."""
    stem = record["video_stem"]
    loc = record["location"]
    filename = record["filename"]
    npz_path = LANDMARKS_DIR / f"{stem}.npz"

    # Skip if already extracted
    if npz_path.exists() and npz_path.stat().st_size > 1000:
        return stem, 0, True

    video_path = RAW_DIR / loc / "Videos" / filename
    annot_path = RAW_DIR / loc / "Annotations" / "Annotator_1" / f"{stem}.json"
    if not annot_path.exists():
        annot_path = RAW_DIR / loc / "Annotations" / f"{stem}.json"

    if not video_path.exists():
        return stem, -1, False

    extractor = HandLandmarkExtractor()
    try:
        res = extractor.extract_from_video(video_path, annot_path, output_path=npz_path)
        n_frames = res["total_frames"]
        return stem, n_frames, True
    except Exception as e:  # noqa: BLE001
        print(f"Error extracting {stem}: {e}")
        return stem, -1, False
    finally:
        extractor.close()


def extract_all_dataset_landmarks(workers: int = 4):
    """Extract landmarks for all 212 videos in parallel."""
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    split_dict = load_split()

    all_records = []
    for split_name in ["train", "val", "test"]:
        all_records.extend(split_dict[split_name])

    print(
        f"Starting batch extraction for {len(all_records)} videos with {workers} worker processes..."
    )
    already_done = 0
    to_process = []
    for r in all_records:
        stem = r["video_stem"]
        npz_path = LANDMARKS_DIR / f"{stem}.npz"
        if npz_path.exists() and npz_path.stat().st_size > 1000:
            already_done += 1
        else:
            to_process.append(r)

    print(
        f"Already extracted: {already_done} videos. Remaining to extract: {len(to_process)} videos."
    )

    if not to_process:
        print("All video landmarks are already extracted!")
        return

    start_time = time.time()
    extracted_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_video_task, r): r for r in to_process}
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting MediaPipe Landmarks",
        ):
            stem, n_frames, success = future.result()
            if success and n_frames > 0:
                extracted_count += 1

    elapsed = time.time() - start_time
    print(f"\nBatch extraction complete in {elapsed:.1f}s ({elapsed / 60:.1f} min).")
    print(f"Total newly extracted: {extracted_count} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel worker processes"
    )
    args = parser.parse_args()
    extract_all_dataset_landmarks(workers=args.workers)
