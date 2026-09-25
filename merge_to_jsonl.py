"""CLI tool to merge all video OCR JSON files into a single Elasticsearch-ready JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

from retrieval_exporter import convert_ocr_result_to_retrieval


def merge_json_files_to_jsonl(
    input_paths: list[Path],
    output_file: Path,
    skip_empty: bool = True,
) -> tuple[int, int]:
    """Read all video JSON files and write each frame record as one JSON line."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    total_videos = 0

    with output_file.open("w", encoding="utf-8") as out_handle:
        for json_path in tqdm(input_paths, desc="Merging videos to JSONL", unit="video"):
            try:
                with json_path.open("r", encoding="utf-8") as in_handle:
                    data = json.load(in_handle)
            except Exception as exc:
                print(f"Warning: Could not read {json_path}: {exc}")
                continue

            video_name = json_path.stem
            # If filename ends with _clean or similar, clean it
            video_name = re.sub(r"(_ocr|_clean|_output)$", "", video_name)

            # Determine format: already list of records, or raw dict
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = convert_ocr_result_to_retrieval(data, video_name, skip_empty=False)
            else:
                continue

            total_videos += 1
            for rec in records:
                if skip_empty and not rec.get("cleaned_text", "").strip():
                    continue
                out_handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_frames += 1

    return total_videos, total_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge video OCR JSON files into a single JSONL for Elasticsearch")
    parser.add_argument("input", help="Directory containing video *.json files or glob pattern")
    parser.add_argument("--output", default="all_videos_ocr.jsonl", help="Output .jsonl path")
    from s3_uploader import S3Uploader, str2bool

    parser.add_argument("--include-empty", action="store_true", help="Include frames with no detected text")
    parser.add_argument(
        "--s3_upload",
        "--s3-upload",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Upload merged JSONL to S3 bucket (true/false, default: false)",
    )
    parser.add_argument("--s3-bucket", default="aithena2026", help="S3 bucket name (default: aithena2026)")
    parser.add_argument("--s3-prefix", default="ocr_data", help="S3 folder prefix (default: ocr_data)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        # Match all json files except checkpoint files
        json_files = sorted(
            p for p in input_path.glob("*.json")
            if not p.name.endswith("_checkpoint.json") and not p.name.endswith(".jsonl")
        )
    elif input_path.is_file():
        json_files = [input_path]
    else:
        json_files = sorted(Path(".").glob(args.input))

    if not json_files:
        raise SystemExit(f"No JSON files found at: {args.input}")

    output_path = Path(args.output)
    videos, frames = merge_json_files_to_jsonl(json_files, output_path, skip_empty=not args.include_empty)
    print(f"\n Successfully merged {videos} video(s) into {output_path}")
    print(f" Total indexed frame records: {frames}")

    if args.s3_upload:
        try:
            uploader = S3Uploader(bucket=args.s3_bucket, prefix=args.s3_prefix)
            s3_uri = uploader.upload_file(output_path)
            print(f" Uploaded to AWS S3: {s3_uri}")
        except Exception as exc:
            print(f"Warning: S3 upload failed for {output_path}: {exc}")


if __name__ == "__main__":
    main()
