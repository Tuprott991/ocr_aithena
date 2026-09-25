import numpy as np

from crop_utils import reading_order_indices, rectify_polygon
from overlay_filter import OverlayConfig, filter_overlays


def test_rectify_and_reading_order():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = rectify_polygon(image, [[20, 20], [170, 30], [168, 55], [18, 45]])
    assert crop.shape[1] > crop.shape[0]
    assert reading_order_indices([[50, 10, 90, 20], [5, 11, 40, 21], [5, 40, 30, 50]]) == [1, 0, 2]


def test_broadcast_filter_protects_lower_third():
    ticker = [
        "GIÁ VÀNG TĂNG MẠNH TRONG PHIÊN SÁNG",
        "THỊ TRƯỜNG CHỨNG KHOÁN GIẢM CUỐI PHIÊN",
        "DỰ BÁO THỜI TIẾT CÁC TỈNH PHÍA BẮC",
        "TIN QUỐC TẾ MỚI NHẤT TRONG NGÀY",
    ]
    frames = []
    for text in ticker:
        frames.append(
            [
                {"text": "VTV1 HD", "bbox": [10, 10, 65, 35]},
                {"text": "NGUYỄN VĂN AN", "bbox": [100, 600, 500, 650]},
                {"text": text, "bbox": [20, 900, 800, 930]},
            ]
        )
    decisions = filter_overlays(frames, [(1000, 1000)] * len(frames), OverlayConfig())
    assert all(row[0]["reason"] == "channel_logo" for row in decisions)
    assert all(row[1]["decision"] == "KEEP" for row in decisions)
    assert all(row[2]["reason"] == "news_ticker" for row in decisions)


def test_assign_semantic_types_and_noise():
    from overlay_filter import assign_semantic_types

    items = [
        {"text": "TIN", "bbox": [100, 800, 160, 840]},
        {"text": "CHÍNH", "bbox": [170, 800, 260, 840]},
        {"text": "VĨNH", "bbox": [100, 550, 180, 580]},
        {"text": "PHÚ", "bbox": [190, 550, 250, 580]},
        {"text": "QUÁN ĂN BÌNH DÂN", "bbox": [100, 150, 400, 200]},
    ]
    types = assign_semantic_types(items, (1000, 1000))
    assert types[0] == "headline"
    assert types[1] == "headline"
    assert types[2] == "lower_third"
    assert types[3] == "lower_third"
    assert types[4] == "scene_text"

    # Test noise filtering
    single_frame = [[
        {"text": "K", "bbox": [30, 250, 120, 330], "det_conf": 0.25, "rec_conf": 0.60},
        {"text": "---", "bbox": [10, 10, 50, 20], "det_conf": 0.9, "rec_conf": 0.9},
    ]]
    dec = filter_overlays(single_frame, [(1000, 1000)], OverlayConfig())
    assert dec[0][0]["decision"] == "DROP" and dec[0][0]["reason"] == "noise_artifact"
    assert dec[0][1]["decision"] == "DROP" and dec[0][1]["reason"] == "noise_punctuation"


def test_collect_images_multi_format(tmp_path):
    from pipeline import collect_images

    (tmp_path / "1.jpg").write_text("dummy")
    (tmp_path / "2.JPEG").write_text("dummy")
    (tmp_path / "3.png").write_text("dummy")
    (tmp_path / "4.PNG").write_text("dummy")
    (tmp_path / "5.webp").write_text("dummy")
    (tmp_path / "6.bmp").write_text("dummy")
    (tmp_path / "readme.txt").write_text("ignore")

    images = collect_images(tmp_path)
    assert len(images) == 6
    names = [p.name for p in images]
    assert names == ["1.jpg", "2.JPEG", "3.png", "4.PNG", "5.webp", "6.bmp"]


def test_traffic_camera_metadata_preserved():
    words = ["AN", "DUONG", "VUONG", "LE", "HONG", "PHONG", "10.Jun", "2026"]
    cxs = [0.024, 0.090, 0.179, 0.253, 0.306, 0.389, 0.633, 0.703]
    frames = []
    sizes = [(897, 375)] * 10
    for f in range(10):
        items = []
        for w, cx in zip(words, cxs):
            items.append({"text": w, "bbox": [cx * 897 - 20, 0.049 * 375 - 10, cx * 897 + 20, 0.049 * 375 + 10]})
        items.append({"text": f"11:00:{17+f:02d}", "bbox": [0.799 * 897 - 30, 0.049 * 375 - 10, 0.799 * 897 + 30, 0.049 * 375 + 10]})
        items.append({"text": "CHAGEE", "bbox": [0.093 * 897 - 30, 0.398 * 375 - 10, 0.093 * 897 + 30, 0.098 * 375 + 10]})
        frames.append(items)

    decisions = filter_overlays(frames, sizes, OverlayConfig())
    # In auto mode on traffic footage, all location, clock, and scene items must be KEPT!
    for f_idx in range(len(frames)):
        for item, dec in zip(frames[f_idx], decisions[f_idx]):
            assert dec["decision"] == "KEEP", f"Expected KEEP for {item['text']}, got {dec}"




