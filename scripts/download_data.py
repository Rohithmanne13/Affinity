"""
MIND Dataset Downloader.

Downloads the Microsoft News Dataset (MIND-small) for development.
Falls back to clear manual instructions if automatic download fails.

Usage:
    python scripts/download_data.py [--variant small|large]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommender.config import get_config, setup_logging

logger = logging.getLogger(__name__)

# MIND dataset URLs (Microsoft Research)
MIND_URLS = {
    "small": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    },
    "large": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
        "test": "https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip",
    },
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Destination file path.
    chunk_size : int
        Download chunk size in bytes.

    Returns
    -------
    bool
        True if download succeeded.
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f, tqdm(
            desc=dest.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info("Downloaded: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True

    except requests.RequestException as e:
        logger.error("Download failed for %s: %s", url, e)
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a ZIP file to the target directory."""
    logger.info("Extracting %s → %s", zip_path.name, extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    logger.info("Extraction complete: %d files", len(list(extract_to.rglob("*"))))


def download_mind(variant: str = "small", raw_dir: Path | None = None) -> bool:
    """
    Download and extract the MIND dataset.

    Parameters
    ----------
    variant : str
        "small" or "large".
    raw_dir : Path, optional
        Directory to store raw data. Defaults to config.raw_dir.

    Returns
    -------
    bool
        True if all downloads succeeded.
    """
    cfg = get_config()
    raw_dir = raw_dir or cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = MIND_URLS.get(variant)
    if urls is None:
        logger.error("Unknown variant: %s. Use 'small' or 'large'.", variant)
        return False

    success = True
    for split_name, url in urls.items():
        split_dir = raw_dir / split_name
        marker = split_dir / "behaviors.tsv"

        if marker.exists():
            logger.info("Split '%s' already exists at %s — skipping", split_name, split_dir)
            continue

        zip_name = f"MIND{variant}_{split_name}.zip"
        zip_path = raw_dir / zip_name

        # Download
        if not zip_path.exists():
            logger.info("Downloading MIND-%s %s split...", variant, split_name)
            if not download_file(url, zip_path):
                _print_manual_instructions(variant)
                success = False
                continue

        # Extract
        extract_zip(zip_path, split_dir)

        # Clean up ZIP
        if zip_path.exists():
            zip_path.unlink()
            logger.info("Removed ZIP: %s", zip_name)

    return success


def _print_manual_instructions(variant: str) -> None:
    """Print manual download instructions when auto-download fails."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print(f"\nAutomatic download of MIND-{variant} failed.")
    print("Please download the dataset manually:\n")
    print("1. Visit: https://msnews.github.io/")
    print("2. Download the MIND-small dataset (train + dev splits)")
    print("3. Extract the contents to:")
    print(f"     data/raw/train/   (behaviors.tsv, news.tsv, entity_embedding.vec, ...)")
    print(f"     data/raw/dev/     (behaviors.tsv, news.tsv, entity_embedding.vec, ...)")
    print("\nExpected files in each split directory:")
    print("  - behaviors.tsv")
    print("  - news.tsv")
    print("  - entity_embedding.vec (optional)")
    print("  - relation_embedding.vec (optional)")
    print("=" * 70 + "\n")


def verify_dataset(raw_dir: Path | None = None) -> bool:
    """
    Verify that the MIND dataset files exist.

    Returns
    -------
    bool
        True if all required files are present.
    """
    cfg = get_config()
    raw_dir = raw_dir or cfg.raw_dir

    required_files = [
        raw_dir / "train" / "behaviors.tsv",
        raw_dir / "train" / "news.tsv",
        raw_dir / "dev" / "behaviors.tsv",
        raw_dir / "dev" / "news.tsv",
    ]

    all_present = True
    for f in required_files:
        if f.exists():
            size_mb = f.stat().st_size / 1e6
            logger.info("✓ Found: %s (%.1f MB)", f.relative_to(raw_dir), size_mb)
        else:
            logger.warning("✗ Missing: %s", f.relative_to(raw_dir))
            all_present = False

    return all_present


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Download the MIND dataset")
    parser.add_argument(
        "--variant",
        choices=["small", "large"],
        default="small",
        help="MIND dataset variant (default: small)",
    )
    args = parser.parse_args()

    logger.info("Starting MIND-%s dataset download...", args.variant)
    success = download_mind(variant=args.variant)

    if success and verify_dataset():
        logger.info("Dataset ready!")
    else:
        logger.warning("Dataset setup incomplete. See instructions above.")
        sys.exit(1)
