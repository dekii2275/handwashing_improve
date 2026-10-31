"""
Dataset Downloader for METC Handwashing Dataset (Zenodo DOI: 10.5281/zenodo.5808789)
Supports downloading metadata CSVs, extracting annotations directly via HTTP Range requests,
and downloading complete video archives with resume support.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import io
import zipfile

import requests
from tqdm import tqdm

from config import RAW_DIR, ZENODO_FILES


class RemoteZipFile(io.RawIOBase):
    """File-like wrapper that uses HTTP Range requests with retries to read remote zip archives."""

    def __init__(self, url, session=None):
        self.url = url
        self.session = session or requests.Session()
        resp = self.session.head(url, timeout=30)
        self.length = int(resp.headers.get("Content-Length", 0))
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1:
            size = self.length - self.pos
        if size <= 0:
            return b""
        end = min(self.pos + size - 1, self.length - 1)

        for attempt in range(5):
            try:
                resp = self.session.get(
                    self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=30
                )
                if resp.status_code == 429:
                    wait_time = (attempt + 1) * 3
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.content
                self.pos += len(data)
                return data
            except Exception:
                if attempt == 4:
                    raise
                time.sleep((attempt + 1) * 2)
        return b""


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):
    """Download a file with resume support and progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    resume_header = {}
    existing_bytes = 0
    if temp_path.exists():
        existing_bytes = temp_path.stat().st_size
        resume_header["Range"] = f"bytes={existing_bytes}-"
        print(f"Resuming {dest_path.name} from {existing_bytes / (1024 * 1024):.1f} MB")

    response = requests.get(url, headers=resume_header, stream=True)
    total_size = int(response.headers.get("content-length", 0)) + existing_bytes

    mode = "ab" if existing_bytes > 0 else "wb"
    with (
        open(temp_path, mode) as f,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            initial=existing_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    if temp_path.exists():
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
    print(f"Saved: {dest_path}")


def download_csvs():
    """Download summary.csv and statistics.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for key in ["summary", "statistics"]:
        filename = ZENODO_FILES[key]
        dest = RAW_DIR / filename
        if not dest.exists():
            url = f"https://zenodo.org/api/records/5808789/files/{filename}/content"
            print(f"Downloading {filename}...")
            download_file(url, dest)
        else:
            print(f"{filename} already exists at {dest}")


def extract_annotations_from_remote_zip(interface_idx: int):
    """Extract all JSON annotations directly from remote zip without downloading entire videos."""
    zip_name = f"Interface_number_{interface_idx}.zip"
    url = f"https://zenodo.org/api/records/5808789/files/{zip_name}/content"
    print(f"\nScanning remote zip: {zip_name}...")

    session = requests.Session()
    rz = RemoteZipFile(url, session=session)
    zf = zipfile.ZipFile(rz)
    json_entries = [f for f in zf.namelist() if f.endswith(".json")]
    print(f"Found {len(json_entries)} annotation JSON files in {zip_name}.")

    extracted_count = 0
    for entry in tqdm(json_entries, desc=f"Annotations Int-{interface_idx}"):
        out_file = RAW_DIR / entry
        if out_file.exists() and out_file.stat().st_size > 0:
            continue
        out_file.parent.mkdir(parents=True, exist_ok=True)
        content = zf.read(entry)
        with open(out_file, "wb") as f:
            f.write(content)
        extracted_count += 1
        time.sleep(0.05)  # slight pause to respect rate limits

    print(
        f"Interface {interface_idx}: {extracted_count} new annotations extracted (Total: {len(json_entries)})."
    )
    return len(json_entries)


def download_all_annotations():
    """Download CSVs and extract annotations from all 3 interface archives."""
    download_csvs()
    total = 0
    for idx in [1, 2, 3]:
        total += extract_annotations_from_remote_zip(idx)
    print(f"\nAll annotations successfully extracted: {total} files in {RAW_DIR}")


def download_full_archive(interface_idx: int, extract_after: bool = True):
    """Download full zip archive for an interface and optionally extract it."""
    zip_name = f"Interface_number_{interface_idx}.zip"
    dest_zip = RAW_DIR / zip_name
    extracted_dir = RAW_DIR / f"Interface_number_{interface_idx}" / "Videos"

    # Check if already extracted
    if extracted_dir.exists() and len(list(extracted_dir.glob("*.mp4"))) >= 65:
        print(
            f"\n{zip_name} is already downloaded and extracted ({len(list(extracted_dir.glob('*.mp4')))} videos). Skipping."
        )
        return

    url = f"https://zenodo.org/api/records/5808789/files/{zip_name}/content"
    print(f"\nDownloading complete archive: {zip_name}...")
    download_file(url, dest_zip)

    if extract_after and dest_zip.exists():
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(RAW_DIR)
        print(f"Extracted to {RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download METC Handwashing Dataset from Zenodo"
    )
    parser.add_argument(
        "--only-annotations",
        action="store_true",
        help="Download only metadata and JSON annotations",
    )
    parser.add_argument(
        "--download-all", action="store_true", help="Download all video zip archives"
    )
    parser.add_argument(
        "--download-interface",
        type=int,
        choices=[1, 2, 3],
        help="Download a specific interface zip archive",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Thread workers for concurrent downloads"
    )
    args = parser.parse_args()

    if args.download_all:
        download_csvs()
        for idx in [1, 2, 3]:
            download_full_archive(idx)
    elif args.download_interface:
        download_csvs()
        download_full_archive(args.download_interface)
    else:
        # Default behavior: download metadata and all annotations
        download_all_annotations()
