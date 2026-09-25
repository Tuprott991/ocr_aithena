"""Utility to download pretrained OCR weights from Hugging Face repository:
https://huggingface.co/Vantuk/ocr_aithena
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ID = "Vantuk/ocr_aithena"
FILES = {
    "vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth": "https://huggingface.co/Vantuk/ocr_aithena/resolve/main/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth",
    "best-parseq.ckpt": "https://huggingface.co/Vantuk/ocr_aithena/resolve/main/best-parseq.ckpt",
}


def _download_with_urllib(url: str, dest_path: Path) -> None:
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    class DownloadProgressBar:
        def __init__(self, filename: str):
            self.pbar: Optional[tqdm] = None
            self.filename = filename

        def __call__(self, block_num: int, block_size: int, total_size: int):
            if not has_tqdm:
                return
            if self.pbar is None:
                self.pbar = tqdm(
                    desc=self.filename,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.n = downloaded
                self.pbar.refresh()
            else:
                self.pbar.n = total_size
                self.pbar.close()

    print(f"Downloading {dest_path.name} from Hugging Face...")
    urllib.request.urlretrieve(
        url,
        temp_path,
        reporthook=DownloadProgressBar(dest_path.name),
    )
    temp_path.replace(dest_path)
    print(f"Successfully downloaded {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")


def download_weights(dest_dir: Path = Path("."), force: bool = False) -> None:
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first if available
    use_hf_hub = False
    try:
        from huggingface_hub import hf_hub_download
        use_hf_hub = True
    except ImportError:
        pass

    for filename, url in FILES.items():
        dest_file = dest_dir / filename
        if dest_file.exists() and dest_file.stat().st_size > 1024 * 1024 and not force:
            print(f"[OK] {filename} already exists at {dest_file} ({dest_file.stat().st_size / (1024*1024):.1f} MB). Skipping.")
            continue

        if use_hf_hub:
            try:
                print(f"Downloading {filename} via huggingface_hub...")
                downloaded_file = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    local_dir=str(dest_dir),
                )
                print(f"[OK] Downloaded {filename} -> {downloaded_file}")
                continue
            except Exception as e:
                print(f"Warning: huggingface_hub download failed ({e}), falling back to direct URL...")

        _download_with_urllib(url, dest_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OCR model weights from Hugging Face (Vantuk/ocr_aithena)")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("."),
        help="Directory to save downloaded weights (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )
    args = parser.parse_args()
    download_weights(dest_dir=args.dest_dir, force=args.force)


if __name__ == "__main__":
    main()
