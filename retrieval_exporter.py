"""Export clean OCR results to AIC/Elasticsearch retrieval format and merge to JSONL."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence


VI_STOPWORDS = {
    "và", "hoặc", "của", "thì", "là", "mà", "ở", "các", "những", "một", "cho",
    "để", "với", "trong", "đã", "đang", "sẽ", "được", "bị", "do", "bởi", "vì",
    "nên", "tại", "theo", "ra", "vào", "lên", "xuống", "này", "đó", "kia",
    "cái", "con", "cô", "anh", "chị", "ông", "bà", "vụ", "việc", "về", "như",
    "từ", "đến", "khi", "lúc", "nơi", "nào", "gì", "ai", "sao", "rất", "lắm",
    "quá", "cũng", "đều", "lại", "chỉ", "mới", "còn", "nữa", "nhất"
}


def extract_keywords(cleaned_text: str) -> list[str]:
    """Extract unique content words, filtering out common Vietnamese stopwords and digits."""
    tokens = cleaned_text.lower().split()
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        word = re.sub(r"[^\w]", "", token).strip()
        if (
            len(word) >= 2
            and word not in VI_STOPWORDS
            and word not in seen
            and not word.isdigit()
        ):
            seen.add(word)
            keywords.append(word)
    return keywords


def group_items_to_lines(clean_items: Sequence[dict[str, Any]], line_tolerance: float = 0.6) -> list[str]:
    """Group word boxes into distinct visual text lines based on vertical baseline overlap."""
    if not clean_items:
        return []
    indexed = list(enumerate([item["bbox"] for item in clean_items]))
    indexed.sort(key=lambda item: ((item[1][1] + item[1][3]) / 2, item[1][0]))
    lines: list[dict[str, Any]] = []
    for index, box in indexed:
        cy = (box[1] + box[3]) / 2
        height = max(box[3] - box[1], 1.0)
        matched = False
        for line in lines:
            if abs(cy - line["cy"]) <= line_tolerance * max(height, line["height"]):
                line["items"].append(clean_items[index])
                n = len(line["items"])
                line["cy"] = (line["cy"] * (n - 1) + cy) / n
                line["height"] = max(line["height"], height)
                matched = True
                break
        if not matched:
            lines.append({"cy": cy, "height": height, "items": [clean_items[index]]})

    lines.sort(key=lambda line: line["cy"])
    result: list[str] = []
    for line in lines:
        sorted_items = sorted(line["items"], key=lambda it: it["bbox"][0])
        line_text = " ".join(it["text"].strip() for it in sorted_items if it.get("text", "").strip())
        if line_text:
            result.append(line_text)
    return result


def clean_text_for_search(raw_text: str) -> str:
    """Lowercase and strip punctuation for BM25 matching."""
    cleaned = re.sub(r"[^\w\s]", " ", raw_text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def format_frame_record(
    video_name: str,
    image_name: str,
    clean_items: Sequence[dict[str, Any]],
    img_idx: int = 0,
) -> dict[str, Any]:
    """Format one frame's clean OCR items into the user's AIC Elasticsearch schema."""
    text_array = group_items_to_lines(clean_items)
    raw_text = " ".join(text_array)
    cleaned_text = clean_text_for_search(raw_text)
    words = cleaned_text.split()
    keywords = extract_keywords(cleaned_text)

    # Extract integer ID from filename (e.g. '486.webp' -> 486)
    digits = re.sub(r"\D", "", Path(image_name).stem)
    imgid = int(digits) if digits else int(img_idx)

    return {
        "video_name": video_name,
        "image_name": image_name,
        "imgid": imgid,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "text_array": text_array,
        "text_length": len(raw_text),
        "word_count": len(words),
        "keywords": keywords,
        "language": "vi",
    }


def convert_ocr_result_to_retrieval(
    ocr_result: dict[str, Any],
    video_name: str,
    skip_empty: bool = False,
) -> list[dict[str, Any]]:
    """Convert pipeline output dict into a list of retrieval frame records."""
    records: list[dict[str, Any]] = []
    for idx, (img_name, frame_data) in enumerate(ocr_result.items()):
        clean_items = frame_data.get("clean", [])
        record = format_frame_record(video_name, img_name, clean_items, img_idx=idx)
        if skip_empty and not record["cleaned_text"]:
            continue
        records.append(record)
    return records
