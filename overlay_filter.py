"""Configurable OCR-driven filtering of broadcast overlays and news tickers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median, pstdev
from typing import Any, Sequence


CHANNEL_RE = re.compile(
    r"^(?:VTV(?:\s*[1-9])?(?:\s*HD)?|HTV(?:\s*[0-9O])?(?:\s*HD)?|HIV(?:\s*[0-9])?(?:\s*HD)?|THVL(?:\s*[1-4])?(?:\s*HD)?|"
    r"VTC\s*\d{0,2}|ANTV|QPVN|VNEWS|VOV(?:TV)?|BTV\s*\d*|HANOI\s*\d*)$",
    re.IGNORECASE,
)
CLOCK_RE = re.compile(r"^(?:[01]?\d|2[0-3])[:h.\-][0-5]\d(?:[:.\-][0-5]\d)?$", re.IGNORECASE)
PROGRAM_UI_RE = re.compile(
    r"^(?:CHƯƠNG\s*TRÌNH(?:\s*60\s*GIÂY)?|60\s*GIÂY|GIÂY|TIẾP\s*THEO|CHUONG\s*TRINH(?:\s*60\s*GIAY)?|60\s*GIAY|GIAY|TIEP\s*THEO)$",
    re.IGNORECASE,
)
DATE_INDICATOR_RE = re.compile(
    r"(?:\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{1,2}\.(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}\b|\b(?:20\d{2}|19\d{2})\b)",
    re.IGNORECASE,
)


@dataclass
class OverlayConfig:
    video_type: str = "auto"  # "auto", "broadcast", "traffic"
    max_track_gap: int = 2
    position_iou: float = 0.45
    center_distance: float = 0.035
    text_similarity: float = 0.72
    persistent_ratio: float = 0.60
    persistent_min_frames: int = 3
    persistent_max_area: float = 0.045
    position_std: float = 0.018
    repeated_text_similarity: float = 0.80
    program_token_ratio: float = 0.60
    program_token_max_chars: int = 6
    program_token_max_area: float = 0.030
    edge_margin: float = 0.20
    channel_cluster_distance: float = 0.065
    channel_cluster_max_area: float = 0.012
    ticker_min_y: float = 0.88
    ticker_min_width: float = 0.18
    ticker_min_aspect: float = 4.0
    ticker_presence_ratio: float = 0.20
    ticker_y_std: float = 0.025
    ticker_change_ratio: float = 0.30
    ticker_text_similarity: float = 0.80
    top_ticker_max_y: float = 0.15


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9:]", "", text.upper())


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _plain(a), _plain(b)).ratio()


def _norm_box(item: dict[str, Any], size: tuple[int, int]) -> list[float]:
    width, height = size
    x0, y0, x1, y1 = item["bbox"]
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def _center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return (((a[0] + a[2] - b[0] - b[2]) / 2) ** 2 + ((a[1] + a[3] - b[1] - b[3]) / 2) ** 2) ** 0.5


def _is_meaningful_content(
    item: dict[str, Any],
    box: Sequence[float],
    frame_items: Sequence[dict[str, Any]] | None = None,
    size: tuple[int, int] | None = None,
) -> bool:
    """Check if item is meaningful content that must NEVER be dropped as a generic overlay.

    Protects:
    1. Multi-word phrases in lower-thirds, headlines, or scene text.
    2. Words that form part of a composite text line (e.g. road names 'AN DUONG VUONG - LE HONG PHONG').
    3. Date / Surveillance camera OSD timestamps (e.g. '10.Jun', '2026').
    4. Scene text located in physical space (0.15 <= cy <= 0.88).
    """
    text = item["text"].strip()
    words = text.split()
    cy = (box[1] + box[3]) / 2

    # Direct multi-word check
    if len(words) >= 2 or len(_plain(text)) >= 10:
        return True

    # Date / Month / Year indicator (e.g. '10.Jun', '2026')
    if DATE_INDICATOR_RE.search(text):
        return True

    # Check horizontal line context in the same frame
    if frame_items is not None and size is not None:
        h = max(box[3] - box[1], 0.005)
        line_words = []
        for other in frame_items:
            other_box = _norm_box(other, size)
            other_cy = (other_box[1] + other_box[3]) / 2
            if abs(cy - other_cy) <= 0.6 * max(h, max(other_box[3] - other_box[1], 0.005)):
                line_words.append((other_box[0], other["text"].strip()))
        if len(line_words) >= 2:
            line_words.sort(key=lambda pair: pair[0])
            full_line = " ".join(w for _, w in line_words)
            if len(full_line.split()) >= 2 or len(_plain(full_line)) >= 8:
                return True

    # Center-region scene text (e.g. store names like 'CHAGEE')
    if 0.15 <= cy <= 0.88:
        return True

    return False


def _assign_tracks(frames: Sequence[list[dict[str, Any]]], sizes: Sequence[tuple[int, int]], cfg: OverlayConfig):
    tracks: list[list[tuple[int, int, list[float]]]] = []
    last_seen: list[tuple[int, list[float], str]] = []
    for frame_index, (items, size) in enumerate(zip(frames, sizes)):
        used: set[int] = set()
        for item_index, item in enumerate(items):
            box = _norm_box(item, size)
            best_track, best_score = None, -1.0
            for track_index, (last_frame, last_box, last_text) in enumerate(last_seen):
                if track_index in used or frame_index - last_frame > cfg.max_track_gap:
                    continue
                spatial = max(_iou(box, last_box), 1.0 - _center_distance(box, last_box) / max(cfg.center_distance, 1e-6))
                text_sim = _similarity(item["text"], last_text)
                if (_iou(box, last_box) >= cfg.position_iou or _center_distance(box, last_box) <= cfg.center_distance) and text_sim >= cfg.text_similarity:
                    score = spatial + text_sim
                    if score > best_score:
                        best_track, best_score = track_index, score
            if best_track is None:
                best_track = len(tracks)
                tracks.append([])
                last_seen.append((-999, box, item["text"]))
            tracks[best_track].append((frame_index, item_index, box))
            last_seen[best_track] = (frame_index, box, item["text"])
            used.add(best_track)
    return tracks


def filter_overlays(
    frames: Sequence[list[dict[str, Any]]],
    sizes: Sequence[tuple[int, int]],
    config: OverlayConfig | None = None,
) -> list[list[dict[str, str]]]:
    cfg = config or OverlayConfig()
    sequence_length = max(len(frames), 1)
    decisions = [[{"decision": "KEEP", "reason": "semantic"} for _ in frame] for frame in frames]

    # 1. Determine video genre (broadcast vs non-broadcast/traffic/CCTV)
    if cfg.video_type == "traffic":
        is_broadcast = False
    elif cfg.video_type == "broadcast":
        is_broadcast = True
    else:  # "auto"
        # Check channel logo occurrence
        channel_logo_hits = 0
        for items, size in zip(frames, sizes):
            for item in items:
                box = _norm_box(item, size)
                edge = box[0] < cfg.edge_margin or box[2] > 1 - cfg.edge_margin or box[1] < cfg.edge_margin
                plain = _plain(item["text"])
                if edge and CHANNEL_RE.fullmatch(re.sub(r"HD$", " HD", plain)):
                    channel_logo_hits += 1
                    break
        has_channel_logo = channel_logo_hits >= max(2, int(round(0.01 * sequence_length)))

        # Pre-check bottom ticker candidate presence
        ticker_candidate_count = 0
        for items, size in zip(frames, sizes):
            bottom_candidates = [
                b for it in items
                for b in [_norm_box(it, size)]
                if (b[1] + b[3]) / 2 >= cfg.ticker_min_y
            ]
            if bottom_candidates:
                span = max(b[2] for b in bottom_candidates) - min(b[0] for b in bottom_candidates)
                if span >= 0.20 or len(bottom_candidates) >= 2:
                    ticker_candidate_count += 1
        has_ticker = ticker_candidate_count >= max(2, int(round(cfg.ticker_presence_ratio * sequence_length)))
        is_broadcast = has_channel_logo or has_ticker

    # 2. Channel logos, broadcast clocks, and explicit program UI
    for frame_index, (items, size) in enumerate(zip(frames, sizes)):
        channel_seeds: list[list[float]] = []

        # Check if frame contains date indicators at the top (OSD timestamp guard)
        has_top_date = any(
            _norm_box(it, size)[1] < 0.20 and DATE_INDICATOR_RE.search(it["text"])
            for it in items
        )

        for item_index, item in enumerate(items):
            box = _norm_box(item, size)
            cy = (box[1] + box[3]) / 2
            edge = box[0] < cfg.edge_margin or box[2] > 1 - cfg.edge_margin or cy < cfg.edge_margin
            plain = _plain(item["text"])

            if edge and CHANNEL_RE.fullmatch(re.sub(r"HD$", " HD", plain)):
                decisions[frame_index][item_index] = {"decision": "DROP", "reason": "channel_logo"}
                channel_seeds.append(box)
            elif is_broadcast and not has_top_date and cy < 0.28 and CLOCK_RE.fullmatch(item["text"].replace(" ", "")):
                decisions[frame_index][item_index] = {"decision": "DROP", "reason": "broadcast_clock"}
            elif is_broadcast and plain in {"GIAY", "60GIAY", "CHUONGTRINH", "CHUONGTRINH60GIAY", "TIEPTHEO"}:
                decisions[frame_index][item_index] = {"decision": "DROP", "reason": "persistent_program_ui"}

        # Channel logo composite components (only in broadcast mode)
        if is_broadcast and channel_seeds:
            for item_index, item in enumerate(items):
                if decisions[frame_index][item_index]["decision"] == "DROP":
                    continue
                box = _norm_box(item, size)
                area = (box[2] - box[0]) * (box[3] - box[1])
                cy = (box[1] + box[3]) / 2
                if (
                    cy < cfg.edge_margin
                    and area <= cfg.channel_cluster_max_area
                    and len(_plain(item["text"])) <= 4
                    and any(_center_distance(box, seed) <= cfg.channel_cluster_distance for seed in channel_seeds)
                ):
                    decisions[frame_index][item_index] = {"decision": "DROP", "reason": "channel_logo_component"}

    # 3. Persistent program tokens (only in broadcast mode, at extreme edges, not meaningful content)
    if is_broadcast:
        token_occurrences: dict[str, list[tuple[int, int, float]]] = {}
        for frame_index, (items, size) in enumerate(zip(frames, sizes)):
            for item_index, item in enumerate(items):
                box = _norm_box(item, size)
                cy = (box[1] + box[3]) / 2
                if cy > cfg.edge_margin and cy < 0.90:
                    continue
                if _is_meaningful_content(item, box, items, size):
                    continue
                token = _plain(item["text"])
                area = (box[2] - box[0]) * (box[3] - box[1])
                if token and len(item["text"].strip().split()) == 1 and len(token) <= cfg.program_token_max_chars:
                    token_occurrences.setdefault(token, []).append((frame_index, item_index, area))

        minimum_frames = max(cfg.persistent_min_frames, int(round(cfg.program_token_ratio * sequence_length)))
        for occurrences in token_occurrences.values():
            if len({frame_index for frame_index, _, _ in occurrences}) < minimum_frames:
                continue
            if median([area for _, _, area in occurrences]) > cfg.program_token_max_area:
                continue
            for frame_index, item_index, _ in occurrences:
                if decisions[frame_index][item_index]["decision"] == "KEEP":
                    decisions[frame_index][item_index] = {"decision": "DROP", "reason": "persistent_program_ui"}

        # Channel logo anchor tracking across frames
        unique_channel_centers = list({
            (round((b[0] + b[2]) / 2, 3), round((b[1] + b[3]) / 2, 3))
            for frame_index, (items, size) in enumerate(zip(frames, sizes))
            for item_index, item in enumerate(items)
            if decisions[frame_index][item_index]["reason"] == "channel_logo"
            for b in [_norm_box(item, size)]
        })
        if unique_channel_centers:
            for frame_index, (items, size) in enumerate(zip(frames, sizes)):
                for item_index, item in enumerate(items):
                    if decisions[frame_index][item_index]["decision"] == "DROP":
                        continue
                    box = _norm_box(item, size)
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    box_cx, box_cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if (
                        box_cy < cfg.edge_margin
                        and area <= cfg.channel_cluster_max_area
                        and len(_plain(item["text"])) <= 4
                        and any(((box_cx - ac[0]) ** 2 + (box_cy - ac[1]) ** 2) ** 0.5 <= cfg.channel_cluster_distance for ac in unique_channel_centers)
                    ):
                        decisions[frame_index][item_index] = {"decision": "DROP", "reason": "channel_logo_component"}

        # Persistent track overlay removal
        tracks = _assign_tracks(frames, sizes, cfg)
        for track in tracks:
            unique_frames = len({entry[0] for entry in track})
            if unique_frames < cfg.persistent_min_frames or unique_frames / sequence_length < cfg.persistent_ratio:
                continue
            boxes = [entry[2] for entry in track]
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            centers_x = [(b[0] + b[2]) / 2 for b in boxes]
            centers_y = [(b[1] + b[3]) / 2 for b in boxes]
            anchor_text = frames[track[0][0]][track[0][1]]["text"]
            repetitions = [
                _similarity(anchor_text, frames[frame_index][item_index]["text"])
                for frame_index, item_index, _ in track
            ]
            stable = max(pstdev(centers_x), pstdev(centers_y)) <= cfg.position_std
            edge = median(centers_x) < cfg.edge_margin or median(centers_x) > 1 - cfg.edge_margin or median(centers_y) < cfg.edge_margin or median(centers_y) > 0.92
            if median(areas) <= cfg.persistent_max_area and stable and edge and median(repetitions) >= cfg.repeated_text_similarity:
                for frame_index, item_index, box in track:
                    item = frames[frame_index][item_index]
                    size = sizes[frame_index]
                    if decisions[frame_index][item_index]["decision"] == "KEEP" and not _is_meaningful_content(item, box, frames[frame_index], size):
                        decisions[frame_index][item_index] = {"decision": "DROP", "reason": "persistent_overlay"}

        # Bottom news ticker filtering
        ticker_rows: list[tuple[int, list[tuple[int, list[float]]], str, float]] = []
        for frame_index, (items, size) in enumerate(zip(frames, sizes)):
            candidates = []
            for item_index, item in enumerate(items):
                box = _norm_box(item, size)
                cy = (box[1] + box[3]) / 2
                if cy >= cfg.ticker_min_y:
                    candidates.append((item_index, box))
            if candidates:
                span = max(b[2] for _, b in candidates) - min(b[0] for _, b in candidates)
                if span >= 0.20 or len(candidates) >= 2 or any((b[2] - b[0]) >= cfg.ticker_min_width for _, b in candidates):
                    text = " ".join(items[i]["text"] for i, _ in sorted(candidates, key=lambda pair: pair[1][0]))
                    ticker_rows.append((frame_index, candidates, text, median([(b[1] + b[3]) / 2 for _, b in candidates])))

        if len(ticker_rows) >= max(2, int(round(cfg.ticker_presence_ratio * sequence_length))):
            y_values = [row[3] for row in ticker_rows]
            similarities = [_similarity(a[2], b[2]) for a, b in zip(ticker_rows, ticker_rows[1:])]
            change_ratio = sum(value < cfg.ticker_text_similarity for value in similarities) / max(len(similarities), 1)
            if pstdev(y_values) <= cfg.ticker_y_std and change_ratio >= cfg.ticker_change_ratio:
                ticker_band_y = min(b[1] for row in ticker_rows for _, b in row[1])
                effective_ticker_y = max(cfg.ticker_min_y, ticker_band_y - 0.01)
                for frame_index, items in enumerate(frames):
                    size = sizes[frame_index]
                    for item_index, item in enumerate(items):
                        box = _norm_box(item, size)
                        cy = (box[1] + box[3]) / 2
                        if (
                            decisions[frame_index][item_index]["decision"] == "KEEP"
                            and (cy >= effective_ticker_y or box[1] >= effective_ticker_y)
                        ):
                            decisions[frame_index][item_index] = {"decision": "DROP", "reason": "news_ticker"}

        # Top running banner / ticker detection (if present)
        if cfg.top_ticker_max_y > 0:
            top_ticker_rows: list[tuple[int, list[tuple[int, list[float]]], str, float]] = []
            for frame_index, (items, size) in enumerate(zip(frames, sizes)):
                candidates = []
                for item_index, item in enumerate(items):
                    box = _norm_box(item, size)
                    cy = (box[1] + box[3]) / 2
                    if cy <= cfg.top_ticker_max_y:
                        candidates.append((item_index, box))
                if candidates:
                    span = max(b[2] for _, b in candidates) - min(b[0] for _, b in candidates)
                    if span >= 0.35 or len(candidates) >= 4:
                        text = " ".join(items[i]["text"] for i, _ in sorted(candidates, key=lambda pair: pair[1][0]))
                        top_ticker_rows.append((frame_index, candidates, text, median([(b[1] + b[3]) / 2 for _, b in candidates])))
            if len(top_ticker_rows) >= max(2, int(round(cfg.ticker_presence_ratio * sequence_length))):
                y_values = [row[3] for row in top_ticker_rows]
                similarities = [_similarity(a[2], b[2]) for a, b in zip(top_ticker_rows, top_ticker_rows[1:])]
                change_ratio = sum(value < cfg.ticker_text_similarity for value in similarities) / max(len(similarities), 1)
                if pstdev(y_values) <= cfg.ticker_y_std and change_ratio >= cfg.ticker_change_ratio:
                    max_top_band_y = max(b[3] for row in top_ticker_rows for _, b in row[1])
                    for frame_index, items in enumerate(frames):
                        size = sizes[frame_index]
                        for item_index, item in enumerate(items):
                            box = _norm_box(item, size)
                            cy = (box[1] + box[3]) / 2
                            if (
                                decisions[frame_index][item_index]["decision"] == "KEEP"
                                and cy <= max_top_band_y + 0.01
                            ):
                                decisions[frame_index][item_index] = {"decision": "DROP", "reason": "top_ticker"}

    # Filter out standalone noise artifacts: isolated 1-character tokens or pure punctuation
    # with borderline confidence (e.g. video transition tiles or edge compression artifacts).
    for frame_index, (items, size) in enumerate(zip(frames, sizes)):
        frame_keep_indices = [idx for idx, d in enumerate(decisions[frame_index]) if d["decision"] == "KEEP"]
        for idx in frame_keep_indices:
            item = items[idx]
            raw_text = item.get("text", "").strip()
            plain_text = re.sub(r"[^\w]", "", raw_text)
            if not plain_text:
                decisions[frame_index][idx] = {"decision": "DROP", "reason": "noise_punctuation"}
                continue
            if len(plain_text) == 1:
                det_conf = item.get("det_conf", 1.0)
                rec_conf = item.get("rec_conf", 1.0)
                is_alone = len(frame_keep_indices) <= 2
                if is_alone or det_conf < 0.35 or rec_conf < 0.75:
                    decisions[frame_index][idx] = {"decision": "DROP", "reason": "noise_artifact"}

    return decisions


def semantic_type(item: dict[str, Any], image_size: tuple[int, int]) -> str:
    box = _norm_box(item, image_size)
    cy = (box[1] + box[3]) / 2
    width = box[2] - box[0]
    plain = item["text"].upper()
    if "TIN CHÍNH" in plain or (0.70 <= cy <= 0.95 and (width >= 0.20 or len(item["text"]) >= 8)):
        return "headline"
    if 0.45 <= cy < 0.75 and (len(item["text"].split()) >= 2 or width >= 0.14):
        return "lower_third"
    return "scene_text"


def assign_semantic_types(items: Sequence[dict[str, Any]], image_size: tuple[int, int]) -> list[str]:
    """Classify clean OCR items into scene_text, headline, or lower_third using line context."""
    if not items:
        return []
    types = [semantic_type(item, image_size) for item in items]
    bboxes = [item["bbox"] for item in items]
    indexed = list(enumerate(bboxes))
    indexed.sort(key=lambda it: ((it[1][1] + it[1][3]) / 2, it[1][0]))
    lines: list[list[int]] = []
    line_bboxes: list[list[float]] = []
    width, height = image_size
    for idx, box in indexed:
        cy = (box[1] + box[3]) / 2
        h = max(box[3] - box[1], 1.0)
        matched = False
        for l_items, l_box in zip(lines, line_bboxes):
            l_cy = (l_box[1] + l_box[3]) / 2
            l_h = max(l_box[3] - l_box[1], 1.0)
            if abs(cy - l_cy) <= 0.6 * max(h, l_h):
                l_items.append(idx)
                l_box[0] = min(l_box[0], box[0])
                l_box[1] = min(l_box[1], box[1])
                l_box[2] = max(l_box[2], box[2])
                l_box[3] = max(l_box[3], box[3])
                matched = True
                break
        if not matched:
            lines.append([idx])
            line_bboxes.append(list(box))

    for l_indices, l_box in zip(lines, line_bboxes):
        line_cy = ((l_box[1] + l_box[3]) / 2) / height
        line_w = (l_box[2] - l_box[0]) / width
        line_text = " ".join(items[i]["text"] for i in sorted(l_indices, key=lambda i: items[i]["bbox"][0]))
        words = line_text.strip().split()

        if "TIN CHÍNH" in line_text.upper() or (0.70 <= line_cy <= 0.95 and (len(words) >= 3 or line_w >= 0.18)):
            for i in l_indices:
                types[i] = "headline"
        elif 0.45 <= line_cy < 0.75 and (len(words) >= 2 or len(line_text) >= 5):
            for i in l_indices:
                types[i] = "lower_third"
        else:
            for i in l_indices:
                types[i] = "scene_text"
    return types

