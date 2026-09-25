"""High-performance Multi-GPU Video OCR Batch Runner for Dual RTX 5090 Server.

Processes large-scale video frame folders formatted as:
    new_images_webp/<video_name>/<idx>.webp

Features:
- Dynamic video-level load balancing across multiple GPUs (e.g. cuda:0, cuda:1).
- Batched DeepSolo detection + Batched PARSeq recognition.
- Multi-threaded WebP decoding to keep GPU Tensor Cores 100% saturated.
- Crash-proof auto-resume: skips already processed videos.
- Outputs Elasticsearch-ready JSON per video and optional auto-merge into JSONL.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from overlay_filter import OverlayConfig
from pipeline import IMAGE_SUFFIXES, OCRPipeline, collect_images, natural_key
from retrieval_exporter import convert_ocr_result_to_retrieval


ROOT = Path(__file__).resolve().parent


def parse_gpu_list(gpu_arg: str | None) -> list[int]:
    if not gpu_arg:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    return [int(x.strip()) for x in gpu_arg.split(",") if x.strip().isdigit()]


def worker_process(
    worker_id: int,
    gpu_id: int,
    task_queue: mp.Queue,
    progress_queue: mp.Queue,
    args: argparse.Namespace,
) -> None:
    """Worker process bound to a single GPU, pulling video folders from the queue."""
    # Set CUDA device for this process
    device_str = f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda"):
        torch.cuda.set_device(gpu_id)

    # Initialize overlay configuration
    overlay = OverlayConfig(
        video_type=args.video_type,
        persistent_ratio=args.persistent_ratio,
        persistent_min_frames=args.persistent_min_frames,
        ticker_min_y=args.ticker_min_y,
        ticker_change_ratio=args.ticker_change_ratio,
    )

    # Initialize model pipeline once per worker
    try:
        pipeline = OCRPipeline(
            detector_checkpoint=args.detector_checkpoint,
            recognizer_checkpoint=args.recognizer_checkpoint,
            device=device_str,
            detector_threshold=args.det_threshold,
            detection_batch_size=args.det_batch_size,
            recognition_batch_size=args.rec_batch_size,
            fp16=not args.no_fp16,
            crop_padding=args.crop_padding,
            tiny_height=args.tiny_height,
            overlay_config=overlay,
            fast_mode=True,
        )
    except Exception as exc:
        progress_queue.put(("ERROR", worker_id, f"Failed to initialize models on {device_str}: {exc}"))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            video_folder_str = task_queue.get(timeout=2)
        except queue.Empty:
            break

        if video_folder_str is None:  # Poison pill
            break

        video_path = Path(video_folder_str)
        video_name = video_path.name
        output_file = output_dir / f"{video_name}.json"

        # Check resume condition
        if not args.force and output_file.is_file() and output_file.stat().st_size > 50:
            try:
                with output_file.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list) and len(data) > 0:
                    progress_queue.put(("SKIPPED", worker_id, video_name, len(data)))
                    continue
            except Exception:
                pass  # Corrupted output, re-run

        # Collect and naturally sort frames (e.g. 1.webp, 2.webp, ..., 100.webp)
        try:
            image_paths = collect_images(video_path)
            if args.max_frames_per_video:
                image_paths = image_paths[: args.max_frames_per_video]
        except Exception as exc:
            progress_queue.put(("WARN", worker_id, f"Could not collect images in {video_path}: {exc}"))
            continue

        if not image_paths:
            progress_queue.put(("WARN", worker_id, f"No supported images found in {video_path}"))
            continue

        # Optional video checkpoint file for saving raw detections
        checkpoint_file = (
            output_dir / f"{video_name}_checkpoint.json" if args.save_checkpoints else None
        )

        t0 = time.time()
        try:
            ocr_result = pipeline.run(image_paths, checkpoint_file=checkpoint_file, show_progress=False)
            retrieval_records = convert_ocr_result_to_retrieval(ocr_result, video_name, skip_empty=False)

            # Atomic write via temporary file
            temp_output = output_dir / f".{video_name}.tmp.json"
            with temp_output.open("w", encoding="utf-8") as handle:
                json.dump(retrieval_records, handle, ensure_ascii=False, indent=2)
            temp_output.replace(output_file)

            # Optional S3 Upload per video JSON
            if args.s3_upload:
                try:
                    from s3_uploader import S3Uploader

                    uploader = S3Uploader(bucket=args.s3_bucket, prefix=args.s3_prefix)
                    s3_uri = uploader.upload_file(output_file)
                    progress_queue.put(("S3", worker_id, video_name, s3_uri))
                except Exception as s3_exc:
                    progress_queue.put(("WARN", worker_id, f"S3 upload failed for {video_name}: {s3_exc}"))

            elapsed = max(0.001, time.time() - t0)
            fps = len(image_paths) / elapsed
            searchable_count = sum(1 for r in retrieval_records if r.get("cleaned_text", "").strip())
            progress_queue.put(
                ("DONE", worker_id, video_name, len(image_paths), searchable_count, elapsed, fps)
            )
        except Exception as exc:
            progress_queue.put(("ERROR", worker_id, f"Error processing {video_name}: {exc}"))


def scan_video_folders(root_dir: Path) -> list[Path]:
    """Scan root_dir for video subfolders containing image frames."""
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root_dir}")

    # Check if root_dir itself directly contains images (single video mode)
    direct_images = [p for p in root_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if direct_images:
        return [root_dir]

    # Find all subdirectories containing at least one image
    video_folders = []
    for sub in sorted(root_dir.iterdir(), key=natural_key):
        if sub.is_dir():
            has_image = any(p.suffix.lower() in IMAGE_SUFFIXES for p in sub.iterdir() if p.is_file())
            if has_image:
                video_folders.append(sub)

    return video_folders


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Massive Multi-GPU Vietnamese Video OCR Pipeline (Optimized for 2x RTX 5090)"
    )
    parser.add_argument(
        "--input-dir",
        default="new_images_webp",
        help="Root folder containing video subfolders (e.g. new_images_webp/<video_name>/*.webp)",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_results",
        help="Folder to store per-video Elasticsearch JSON results",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Comma-separated GPU indices (e.g. '0,1' for dual RTX 5090, or '0' for single GPU)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent video workers per GPU (default: 1)",
    )
    parser.add_argument(
        "--det-batch-size",
        type=int,
        default=8,
        help="Batch size for DeepSolo frame detection (8-16 recommended for RTX 5090 with 32GB VRAM)",
    )
    parser.add_argument(
        "--rec-batch-size",
        type=int,
        default=256,
        help="Batch size for Vietnamese PARSeq recognition (256-512 recommended for RTX 5090)",
    )
    parser.add_argument("--det-threshold", type=float, default=0.25, help="Recall-oriented detection threshold")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 half precision")
    parser.add_argument("--crop-padding", type=float, default=0.04)
    parser.add_argument("--tiny-height", type=int, default=24)
    parser.add_argument("--persistent-ratio", type=float, default=0.60)
    parser.add_argument("--persistent-min-frames", type=int, default=3)
    parser.add_argument("--ticker-min-y", type=float, default=0.88)
    parser.add_argument("--ticker-change-ratio", type=float, default=0.30)
    parser.add_argument("--max-frames-per-video", type=int, help="Limit frames per video (for testing)")
    parser.add_argument("--force", action="store_true", help="Force re-processing even if output JSON exists")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save intermediate raw detection checkpoints")
    parser.add_argument(
        "--video-type",
        choices=["auto", "broadcast", "traffic"],
        default="auto",
        help="Video genre: 'auto' (intelligent classification), 'broadcast' (TV news), or 'traffic' (CCTV/surveillance)",
    )
    parser.add_argument(
        "--merge-jsonl",
        help="Automatically merge all video JSONs into a single JSONL path upon completion",
    )
    from s3_uploader import str2bool

    parser.add_argument(
        "--s3_upload",
        "--s3-upload",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Upload per-video JSON and merged JSONL to S3 bucket (true/false, default: false)",
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
    parser.add_argument(
        "--detector-checkpoint",
        default=str(ROOT / "vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth"),
    )
    parser.add_argument(
        "--recognizer-checkpoint",
        default=str(ROOT / "best-parseq.ckpt"),
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    video_folders = scan_video_folders(input_path)
    if not video_folders:
        print(f"No video folders with image frames found in: {input_path}")
        sys.exit(1)

    gpu_list = parse_gpu_list(args.gpus)
    if not gpu_list:
        print("Warning: No CUDA GPUs detected or requested, falling back to CPU.")
        gpu_list = [-1]

    print("=" * 70)
    print(" AIthena-OCR Multi-GPU High-Throughput Pipeline (Dual RTX 5090)")
    print("=" * 70)
    print(f" Input Directory       : {input_path.resolve()}")
    print(f" Output Directory      : {Path(args.output_dir).resolve()}")
    print(f" Found Videos          : {len(video_folders)}")
    print(f" Active GPUs           : {gpu_list}")
    print(f" Workers per GPU       : {args.workers_per_gpu}")
    print(f" Detection Batch Size  : {args.det_batch_size}")
    print(f" Recognition Batch Size: {args.rec_batch_size}")
    print(f" Precision             : {'FP32' if args.no_fp16 else 'FP16'}")
    print(f" Video Type Mode       : {args.video_type.upper()}")
    s3_status = f"ENABLED (s3://{args.s3_bucket}/{args.s3_prefix})" if args.s3_upload else "DISABLED"
    print(f" S3 Upload             : {s3_status}")
    print("=" * 70)

    # Initialize multiprocessing queues
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    progress_queue = ctx.Queue()

    for folder in video_folders:
        task_queue.put(str(folder))

    # Calculate total workers
    total_workers = len(gpu_list) * args.workers_per_gpu
    processes = []
    worker_id = 0
    for gpu_id in gpu_list:
        for _ in range(args.workers_per_gpu):
            p = ctx.Process(
                target=worker_process,
                args=(worker_id, gpu_id, task_queue, progress_queue, args),
            )
            p.daemon = True
            p.start()
            processes.append(p)
            worker_id += 1

    # Monitor progress
    completed_videos = 0
    total_videos = len(video_folders)
    total_frames_processed = 0
    start_time = time.time()

    pbar = tqdm(total=total_videos, desc="Overall Video Progress", unit="video")

    active_processes = len(processes)
    while completed_videos < total_videos and active_processes > 0:
        try:
            msg = progress_queue.get(timeout=0.5)
            msg_type = msg[0]

            if msg_type == "DONE":
                _, w_id, v_name, n_frames, n_searchable, elapsed, fps = msg
                completed_videos += 1
                total_frames_processed += n_frames
                pbar.update(1)
                tqdm.write(
                    f" [Worker {w_id}] Video {v_name}: {n_frames} frames ({n_searchable} with text) in {elapsed:.1f}s ({fps:.1f} fps)"
                )
            elif msg_type == "S3":
                _, w_id, v_name, s3_uri = msg
                tqdm.write(f"  [Worker {w_id}] Video {v_name} uploaded -> {s3_uri}")
            elif msg_type == "SKIPPED":
                _, w_id, v_name, n_frames = msg
                completed_videos += 1
                pbar.update(1)
                tqdm.write(f"  [Worker {w_id}] Video {v_name}: already completed ({n_frames} frames), skipped.")
            elif msg_type == "WARN":
                _, w_id, text = msg
                tqdm.write(f" [Worker {w_id}] {text}")
            elif msg_type == "ERROR":
                _, w_id, text = msg
                tqdm.write(f" [Worker {w_id}] ERROR: {text}")

        except queue.Empty:
            # Check if all processes are still alive
            alive = sum(1 for p in processes if p.is_alive())
            if alive == 0 and task_queue.qsize() > 0:
                print("Error: All worker processes exited unexpectedly.")
                break
            active_processes = alive

    pbar.close()

    # Join processes
    for p in processes:
        p.join(timeout=3)

    total_time = max(0.001, time.time() - start_time)
    print("\n" + "=" * 70)
    print(f" Processing Complete in {total_time:.1f}s!")
    print(f" Total Videos Processed : {completed_videos}/{total_videos}")
    if total_frames_processed > 0:
        print(f" Total Frames Processed : {total_frames_processed}")
        print(f" Effective Throughput   : {total_frames_processed / total_time:.1f} frames/sec")
    print("=" * 70)

    # Optional auto-merge into single JSONL
    if args.merge_jsonl:
        from merge_to_jsonl import merge_json_files_to_jsonl

        out_path = Path(args.output_dir)
        json_files = sorted(
            p for p in out_path.glob("*.json")
            if not p.name.endswith("_checkpoint.json") and not p.name.startswith(".")
        )
        merge_output = Path(args.merge_jsonl)
        print(f"\nMerging {len(json_files)} video JSON files into {merge_output}...")
        v_count, f_count = merge_json_files_to_jsonl(json_files, merge_output, skip_empty=True)
        print(f" Master JSONL generated: {merge_output} ({v_count} videos, {f_count} searchable frames)")

        # Upload merged JSONL to S3 if requested
        if args.s3_upload:
            try:
                from s3_uploader import S3Uploader

                uploader = S3Uploader(bucket=args.s3_bucket, prefix=args.s3_prefix)
                s3_uri = uploader.upload_file(merge_output)
                print(f" Master JSONL uploaded to AWS S3: {s3_uri}")
            except Exception as s3_exc:
                print(f"Warning: S3 upload of merged JSONL failed: {s3_exc}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
