"""Prepare a separate baseline cache, replacing legacy synthetic fingerprints.

Legacy files without the synthetic fingerprint are reused but explicitly marked as
unverified: this is a pragmatic first experiment, not a provenance guarantee.
"""

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.extract_landmarks import HandLandmarkExtractor

OUT = ROOT / "data" / "landmarks_baseline"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_synthetic(path):
    with np.load(path) as data:
        c = data["confidence"]
        return bool(
            np.any(c == np.float32(0.9)) and np.all((c == 0) | (c == np.float32(0.9)))
        )


def extract(record):
    stem, loc, filename = record["video_stem"], record["location"], record["filename"]
    video = ROOT / "data" / "raw" / loc / "Videos" / filename
    annotation = (
        ROOT / "data" / "raw" / loc / "Annotations" / "Annotator_1" / (stem + ".json")
    )
    if not video.is_file() or not annotation.is_file():
        raise FileNotFoundError(f"Missing video or annotation: {stem}")
    target = OUT / (stem + ".npz")
    detector = HandLandmarkExtractor()
    try:
        result = detector.extract_from_video(video, annotation, target)
    finally:
        detector.close()
    cap = cv2.VideoCapture(str(video))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    with annotation.open(encoding="utf-8") as f:
        label_count = len(json.load(f)["labels"])
    return {
        "stem": stem,
        "source": "extracted_from_video",
        "video_frames_reported": frame_count,
        "annotation_frames": label_count,
        "saved_frames": result["total_frames"],
        "video_sha256": digest(video),
        "annotation_sha256": digest(annotation),
        "landmarks_sha256": digest(target),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all videos instead of reusing unverified cache",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "data" / "split_info.json").read_text(encoding="utf-8"))
    records = [r for name in ("train", "val", "test") for r in split[name]]
    manifest_file = OUT / "manifest.json"
    existing = (
        json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest_file.exists()
        else {}
    )
    manifest = {}
    tasks = []
    for record in records:
        stem = record["video_stem"]
        legacy = ROOT / "data" / "landmarks" / (stem + ".npz")
        target = OUT / (stem + ".npz")
        prev = existing.get(stem)
        if (
            prev
            and target.exists()
            and digest(target) == prev.get("landmarks_sha256")
            and (not args.all or prev["source"] == "extracted_from_video")
        ):
            manifest[stem] = prev
        elif not args.all and legacy.exists() and not is_synthetic(legacy):
            shutil.copy2(legacy, target)
            manifest[stem] = {
                "stem": stem,
                "source": "legacy_unverified",
                "landmarks_sha256": digest(target),
            }
        else:
            tasks.append(record)
    print(f"Reused {len(manifest)} files; extracting {len(tasks)} videos", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract, r): r for r in tasks}
        for future in as_completed(futures):
            item = future.result()
            manifest[item["stem"]] = item
            manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                f"{len(manifest)}/{len(records)}: {item['stem']} video={item['video_frames_reported']} labels={item['annotation_frames']} saved={item['saved_frames']}",
                flush=True,
            )
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
