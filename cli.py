"""Command-line entry point for the Vietnamese video-frame OCR MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from overlay_filter import OverlayConfig
from pipeline import OCRPipeline, collect_images, save_json


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSolo localization + Vietnamese PARSeq OCR")
    parser.add_argument("input", help="One image or a folder of sequential frames")
    parser.add_argument("--output", default="ocr.json", help="UTF-8 JSON output path")
    parser.add_argument("--visualize-dir", help="Save polygon KEEP/DROP visualizations here")
    parser.add_argument("--detector-checkpoint", default=str(ROOT / "vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth"))
    parser.add_argument("--recognizer-checkpoint", default=str(ROOT / "best-parseq.ckpt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--det-threshold", type=float, default=0.25, help="Recall-oriented DeepSolo threshold")
    parser.add_argument("--det-batch-size", type=int, default=4, help="Number of frames to detect concurrently")
    parser.add_argument("--rec-batch-size", type=int, default=128, help="Batch size for PARSeq crop recognition")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--crop-padding", type=float, default=0.04)
    parser.add_argument("--tiny-height", type=int, default=24)
    parser.add_argument("--max-frames", type=int, help="Optional smoke-test limit")
    parser.add_argument("--persistent-ratio", type=float, default=0.60)
    parser.add_argument("--persistent-min-frames", type=int, default=3)
    parser.add_argument("--ticker-min-y", type=float, default=0.88)
    parser.add_argument("--ticker-change-ratio", type=float, default=0.30)
    parser.add_argument("--checkpoint", help="Path to checkpoint JSON file for saving/resuming raw frame OCR")
    parser.add_argument(
        "--video-type",
        choices=["auto", "broadcast", "traffic"],
        default="auto",
        help="Video genre: 'auto' (intelligent classification), 'broadcast' (TV news), or 'traffic' (CCTV/surveillance)",
    parser.add_argument(
        "--format",
        choices=["retrieval", "debug"],
        default="retrieval",
        help="Output JSON format: 'retrieval' (AIC Elasticsearch list schema) or 'debug' (polygons & raw/clean dict)",
    )
    from s3_uploader import str2bool

    parser.add_argument(
        "--s3_upload",
        "--s3-upload",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Upload output JSON to S3 bucket (true/false, default: false)",
    )
    parser.add_argument(
        "--s3-bucket",
        default="aithena2026",
        help="AWS S3 bucket name (default: aithena2026)",
    )
    parser.add_argument(
        "--s3-prefix",
        default="ocr_data",
        help="AWS S3 folder prefix (default: ocr_data)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images = collect_images(args.input)
    if args.max_frames is not None:
        images = images[: args.max_frames]
    if not images:
        raise SystemExit("No supported images found")
    overlay = OverlayConfig(
        video_type=args.video_type,
        persistent_ratio=args.persistent_ratio,
        persistent_min_frames=args.persistent_min_frames,
        ticker_min_y=args.ticker_min_y,
        ticker_change_ratio=args.ticker_change_ratio,
    )
    pipeline = OCRPipeline(
        detector_checkpoint=args.detector_checkpoint,
        recognizer_checkpoint=args.recognizer_checkpoint,
        device=args.device,
        detector_threshold=args.det_threshold,
        detection_batch_size=args.det_batch_size,
        recognition_batch_size=args.rec_batch_size,
        fp16=not args.no_fp16,
        crop_padding=args.crop_padding,
        tiny_height=args.tiny_height,
        overlay_config=overlay,
    )
    result = pipeline.run(images, args.visualize_dir, checkpoint_file=args.checkpoint)
    video_name = Path(args.input).name

    if args.format == "retrieval":
        import json
        from retrieval_exporter import convert_ocr_result_to_retrieval
        records = convert_ocr_result_to_retrieval(result, video_name, skip_empty=False)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        non_empty = sum(1 for r in records if r["cleaned_text"])
        print(f"Wrote {args.output}: {len(records)} frame(s) in retrieval schema ({non_empty} with searchable text)")
    else:
        save_json(result, args.output)
        kept = sum(len(frame["clean"]) for frame in result.values())
        raw = sum(len(frame["raw"]) for frame in result.values())
        print(f"Wrote {args.output}: {len(result)} frame(s), kept {kept}/{raw} regions")

    if args.s3_upload:
        from s3_uploader import S3Uploader

        output_path = Path(args.output)
        try:
            uploader = S3Uploader(bucket=args.s3_bucket, prefix=args.s3_prefix)
            s3_uri = uploader.upload_file(output_path)
            print(f"Uploaded to AWS S3: {s3_uri}")
        except Exception as exc:
            print(f"Warning: S3 upload failed for {output_path}: {exc}")


if __name__ == "__main__":
    main()

