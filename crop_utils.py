"""Polygon rectification, tiny-text enlargement, and reading-order helpers."""

from __future__ import annotations

from typing import Sequence, TypeVar

import cv2
import numpy as np


T = TypeVar("T")


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
    return ordered


def rectify_polygon(
    image_bgr: np.ndarray,
    polygon: Sequence[Sequence[float]],
    padding_ratio: float = 0.04,
    tiny_height: int = 24,
    tiny_scale: float = 2.0,
) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32)
    if len(points) < 4:
        raise ValueError("A crop polygon needs at least four points")
    rect = cv2.minAreaRect(points)
    quad = cv2.boxPoints(rect)
    center = quad.mean(axis=0, keepdims=True)
    quad = center + (quad - center) * (1.0 + 2.0 * padding_ratio)
    h, w = image_bgr.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    quad = _order_quad(quad)

    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    out_w, out_h = max(2, int(round(width))), max(2, int(round(height)))
    target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    crop = cv2.warpPerspective(
        image_bgr,
        cv2.getPerspectiveTransform(quad, target),
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    if crop.shape[0] < tiny_height and tiny_scale > 1:
        scale = max(tiny_scale, tiny_height / max(crop.shape[0], 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return crop


def reading_order_indices(bboxes: Sequence[Sequence[float]], line_tolerance: float = 0.6) -> list[int]:
    """Group by overlapping baselines, then sort left-to-right within each line."""
    if not bboxes:
        return []
    indexed = list(enumerate(bboxes))
    indexed.sort(key=lambda item: ((item[1][1] + item[1][3]) / 2, item[1][0]))
    lines: list[dict] = []
    for index, box in indexed:
        cy = (box[1] + box[3]) / 2
        height = max(box[3] - box[1], 1.0)
        best = None
        best_distance = float("inf")
        for line in lines:
            distance = abs(cy - line["cy"])
            if distance <= line_tolerance * max(height, line["height"]) and distance < best_distance:
                best, best_distance = line, distance
        if best is None:
            lines.append({"cy": cy, "height": height, "items": [(index, box)]})
        else:
            best["items"].append((index, box))
            n = len(best["items"])
            best["cy"] = (best["cy"] * (n - 1) + cy) / n
            best["height"] = max(best["height"], height)
    lines.sort(key=lambda line: line["cy"])
    return [index for line in lines for index, _ in sorted(line["items"], key=lambda item: item[1][0])]

