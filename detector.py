"""DeepSolo text localization using the official implementation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEEPSOLO_ROOT = ROOT / "third_party" / "DeepSolo" / "DeepSolo"
DEFAULT_CONFIG = DEEPSOLO_ROOT / "configs" / "ViTAEv2_S" / "pretrain" / "150k_tt_mlt_13_15_textocr.yaml"


@dataclass(frozen=True)
class Detection:
    polygon: list[list[float]]
    bbox: list[float]
    confidence: float
    centerline: list[list[float]]


def _validate_checkpoint(path: Path) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 1.9, used by the official DeepSolo environment.
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    signatures = {
        "detection_transformer.backbone.0.backbone.layers.0.RC.PCM.0.weight": (64, 3, 3, 3),
        "detection_transformer.ctrl_point_text.5.weight": (38, 256),
        "detection_transformer.boundary_offset.5.layers.2.weight": (4, 256),
    }
    bad = [key for key, shape in signatures.items() if key not in state or tuple(state[key].shape) != shape]
    if bad:
        raise ValueError(f"Checkpoint is not the expected DeepSolo ViTAEv2-S 37-vocabulary model: {bad}")


class DeepSoloDetector:
    """Thin inference wrapper around DeepSolo's model, transforms, and post-processing."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        score_threshold: float = 0.25,
        config: str | Path = DEFAULT_CONFIG,
    ) -> None:
        checkpoint = Path(checkpoint).resolve()
        config = Path(config).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint}'. Download it with: python download_weights.py")
        if not config.is_file():
            raise FileNotFoundError(config)
        _validate_checkpoint(checkpoint)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        if str(DEEPSOLO_ROOT) not in sys.path:
            sys.path.insert(0, str(DEEPSOLO_ROOT))
        try:
            import detectron2.data.transforms as T
            from adet.config import get_cfg
            from adet.modeling import swin, vitae_v2  # noqa: F401 - registers backbones
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.modeling import build_model
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "DeepSolo native dependencies are unavailable. Follow README.md's Linux/WSL setup; "
                "the official CUDA extension and Detectron2 are required."
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(str(config))
        cfg.defrost()
        cfg.MODEL.WEIGHTS = str(checkpoint)
        cfg.MODEL.DEVICE = device
        cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = float(score_threshold)
        cfg.freeze()

        self.cfg = cfg
        self.device = torch.device(device)
        self._transforms = T
        self.model = build_model(cfg).eval()
        incompatible = DetectionCheckpointer(self.model).load(str(checkpoint))
        missing = list(getattr(incompatible, "missing_keys", ()))
        unexpected = list(getattr(incompatible, "unexpected_keys", ()))
        if missing or unexpected:
            raise RuntimeError(f"DeepSolo checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.resize = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST] * 2, cfg.INPUT.MAX_SIZE_TEST)
        self.pad_divisor = 32
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    def _preprocess_image(self, image_bgr: np.ndarray) -> dict[str, Any]:
        if image_bgr is None or image_bgr.ndim != 3:
            raise ValueError("Expected a BGR HxWx3 image")
        height, width = image_bgr.shape[:2]
        image = image_bgr[:, :, ::-1] if self.cfg.INPUT.FORMAT == "RGB" else image_bgr
        image = self.resize.get_transform(image).apply_image(image)
        resized_h, resized_w = image.shape[:2]
        pad_h = (-resized_h) % self.pad_divisor
        pad_w = (-resized_w) % self.pad_divisor
        if pad_h or pad_w:
            image = self._transforms.PadTransform(0, 0, pad_w, pad_h, pad_value=0).apply_image(image)
        tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        return {"image": tensor, "height": height, "width": width}

    @torch.inference_mode()
    def detect_batch(self, images_bgr: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not images_bgr:
            return []
        inputs = [self._preprocess_image(img) for img in images_bgr]
        raw_outputs = self.model(inputs)
        results: list[list[Detection]] = []
        for inp, out in zip(inputs, raw_outputs):
            width = inp["width"]
            height = inp["height"]
            instances = out["instances"].to("cpu")
            scores = instances.scores.numpy()
            centerlines = instances.ctrl_points.numpy().reshape(-1, self.cfg.MODEL.TRANSFORMER.NUM_POINTS, 2)
            boundaries = instances.bd.numpy() if not isinstance(instances.bd, list) else [None] * len(scores)
            frame_detections: list[Detection] = []
            for score, centerline, boundary in zip(scores, centerlines, boundaries):
                if boundary is None:
                    polygon = centerline
                else:
                    sides = np.hsplit(np.asarray(boundary), 2)
                    polygon = np.vstack((sides[0], sides[1][::-1]))
                polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                x0, y0 = polygon.min(axis=0)
                x1, y1 = polygon.max(axis=0)
                w, h = x1 - x0, y1 - y0
                if w < 4 or h < 4 or (w * h) < 40:
                    continue
                frame_detections.append(
                    Detection(
                        polygon=polygon.astype(float).tolist(),
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        confidence=float(score),
                        centerline=centerline.astype(float).tolist(),
                    )
                )
            results.append(frame_detections)
        return results

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return self.detect_batch([image_bgr])[0]

