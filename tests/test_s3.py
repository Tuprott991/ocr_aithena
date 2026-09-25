"""Unit tests for s3_uploader utility."""

import pytest
from pathlib import Path
from s3_uploader import S3Uploader, load_env_file, str2bool


def test_str2bool():
    assert str2bool(True) is True
    assert str2bool(False) is False
    assert str2bool("true") is True
    assert str2bool("True") is True
    assert str2bool("TRUE") is True
    assert str2bool("1") is True
    assert str2bool("yes") is True
    assert str2bool("false") is False
    assert str2bool("False") is False
    assert str2bool("FALSE") is False
    assert str2bool("0") is False
    assert str2bool("no") is False
    with pytest.raises(Exception):
        str2bool("maybe")


def test_load_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY_OCR=test_value_123\nANOTHER_KEY='hello_world'\n# Comment line\n")
    loaded = load_env_file(env_file)
    assert loaded.get("TEST_KEY_OCR") == "test_value_123"
    assert loaded.get("ANOTHER_KEY") == "hello_world"


def test_s3_uploader_init():
    uploader = S3Uploader(bucket="aithena2026", prefix="ocr_data")
    assert uploader.bucket == "aithena2026"
    assert uploader.prefix == "ocr_data"
