"""End-to-end localization, rectification, recognition, and temporal cleanup."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from crop_utils import reading_order_indices, rectify_polygon
from detector import DeepSoloDetector
from overlay_filter import OverlayConfig, assign_semantic_types, filter_overlays, semantic_type
from recognizer import PARSeqRecognizer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def set_deterministic(seed: int = 42, fast_mode: bool = True) -> None:
    # Required by CUDA >= 10.2 before the first cuBLAS operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if fast_mode and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # PyTorch 1.9 has no warn_only parameter.
            torch.use_deterministic_algorithms(True)


def natural_key(path: Path):
    import re

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)]


def collect_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        return sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
    raise FileNotFoundError(f"No supported image or directory found at {path}")


class OCRPipeline:
    def __init__(
        self,
        detector_checkpoint: str | Path,
        recognizer_checkpoint: str | Path,
        device: str = "cuda",
        detector_threshold: float = 0.25,
        detection_batch_size: int = 4,
        recognition_batch_size: int = 128,
        fp16: bool = True,
        crop_padding: float = 0.04,
        tiny_height: int = 24,
        overlay_config: OverlayConfig | None = None,
        fast_mode: bool = True,
    ) -> None:
        set_deterministic(fast_mode=fast_mode)
        self.detector = DeepSoloDetector(detector_checkpoint, device, detector_threshold)
        self.recognizer = PARSeqRecognizer(recognizer_checkpoint, device, recognition_batch_size, fp16)
        self.detection_batch_size = max(1, int(detection_batch_size))
        self.crop_padding = crop_padding
        self.tiny_height = tiny_height
        self.overlay_config = overlay_config or OverlayConfig()

    def run(
        self,
        image_paths: Sequence[Path],
        visualize_dir: str | Path | None = None,
        checkpoint_file: str | Path | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        frames: list[list[dict[str, Any]]] = [None] * len(image_paths)  # type: ignore
        sizes: list[tuple[int, int]] = [None] * len(image_paths)  # type: ignore
        cached_data: dict[str, Any] = {}
        checkpoint_path = Path(checkpoint_file) if checkpoint_file else None
        if checkpoint_path and checkpoint_path.is_file():
            try:
                with checkpoint_path.open("r", encoding="utf-8") as handle:
                    cached_data = json.load(handle)
            except Exception:
                cached_data = {}

        uncached_indices: list[int] = []
        for idx, path in enumerate(image_paths):
            if path.name in cached_data:
                cached_frame = cached_data[path.name]
                frames[idx] = cached_frame["raw"]
                sizes[idx] = tuple(cached_frame["size"])
            else:
                uncached_indices.append(idx)

        if uncached_indices:
            from concurrent.futures import ThreadPoolExecutor

            def _load_image(p: Path) -> np.ndarray:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    raise ValueError(f"Could not decode image: {p}")
                return im

            batch_size = self.detection_batch_size
            num_workers = min(8, max(2, (os.cpu_count() or 4) // 2))

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for b_start in tqdm(
                    range(0, len(uncached_indices), batch_size),
                    desc="DeepSolo + PARSeq",
                    unit="batch",
                    disable=not show_progress,
                ):
                    batch_frame_indices = uncached_indices[b_start : b_start + batch_size]
                    batch_paths = [image_paths[i] for i in batch_frame_indices]

                    batch_images = list(executor.map(_load_image, batch_paths))

                    for i, img in zip(batch_frame_indices, batch_images):
                        h, w = img.shape[:2]
                        sizes[i] = (w, h)

                    batch_detections = self.detector.detect_batch(batch_images)

                    all_crops = []
                    crop_meta = []  # (rel_idx, det)
                    for rel_idx, (img, detections) in enumerate(zip(batch_images, batch_detections)):
                        for det in detections:
                            try:
                                crop = rectify_polygon(
                                    img, det.polygon, padding_ratio=self.crop_padding, tiny_height=self.tiny_height
                                )
                                all_crops.append(crop)
                                crop_meta.append((rel_idx, det))
                            except (ValueError, cv2.error):
                                continue

                    recognitions = self.recognizer.recognize(all_crops) if all_crops else []

                    batch_raw = [[] for _ in range(len(batch_images))]
                    for (rel_idx, det), rec in zip(crop_meta, recognitions):
                        batch_raw[rel_idx].append(
                            {
                                "text": rec.text,
                                "bbox": [round(v, 2) for v in det.bbox],
                                "polygon": [[round(x, 2), round(y, 2)] for x, y in det.polygon],
                                "det_conf": round(det.confidence, 6),
                                "rec_conf": rec.confidence,
                            }
                        )

                    for rel_idx, orig_idx in enumerate(batch_frame_indices):
                        raw_list = batch_raw[rel_idx]
                        order = reading_order_indices([item["bbox"] for item in raw_list])
                        ordered_raw = [raw_list[k] for k in order]
                        frames[orig_idx] = ordered_raw
                        if checkpoint_path is not None:
                            cached_data[image_paths[orig_idx].name] = {
                                "raw": ordered_raw,
                                "size": sizes[orig_idx],
                            }

                    if checkpoint_path is not None and (
                        (b_start // batch_size + 1) % 10 == 0 or b_start + batch_size >= len(uncached_indices)
                    ):
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        with checkpoint_path.open("w", encoding="utf-8") as handle:
                            json.dump(cached_data, handle, ensure_ascii=False)

        decisions = filter_overlays(frames, sizes, self.overlay_config)
        result: dict[str, Any] = {}
        for frame_index, (path, raw, frame_decisions, size) in enumerate(zip(image_paths, frames, decisions, sizes)):
            clean_candidates = []
            for item, decision in zip(raw, frame_decisions):
                item.update(decision)
                if decision["decision"] == "KEEP" and item["text"].strip():
                    clean_candidates.append(item)
            types = assign_semantic_types(clean_candidates, size)
            clean = [
                {
                    "text": item["text"],
                    "bbox": item["bbox"],
                    "type": item_type,
                }
                for item, item_type in zip(clean_candidates, types)
            ]
            result[path.name] = {"raw": raw, "clean": clean, "text": " ".join(item["text"] for item in clean)}

        if visualize_dir is not None:
            self._visualize(image_paths, frames, visualize_dir)
        return result

    @staticmethod
    def _visualize(paths: Sequence[Path], frames, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path, items in tqdm(zip(paths, frames), total=len(paths), desc="Visualize", unit="frame"):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            canvas = image.copy()
            for index, item in enumerate(items):
                keep = item["decision"] == "KEEP"
                color = (40, 200, 40) if keep else (40, 40, 230)
                polygon = np.rint(np.asarray(item["polygon"])).astype(np.int32)
                cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
                x, y = polygon[:, 0].min(), polygon[:, 1].min()
                label = f"{index} {'KEEP' if keep else 'DROP'}:{item['reason']}"
                cv2.putText(canvas, label, (int(x), max(14, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(output_dir / f"{path.stem}.jpg"), canvas)


def save_json(result: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
