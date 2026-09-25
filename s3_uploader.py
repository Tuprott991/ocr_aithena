"""AWS S3 Upload utility for AIthena OCR.

Reads AWS credentials from .env or environment variables:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION / AWS_REGION (optional, default: us-east-1)
    AWS_ENDPOINT_URL (optional, for S3-compatible storage)

Uploads video OCR results to:
    s3://aithena2026/ocr_data/<video_name>.json
    s3://aithena2026/ocr_data/<dataset>.jsonl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def load_env_file(env_path: Path | str | None = None) -> dict[str, str]:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    if env_path is None:
        # Search current working directory and parent directories
        candidates = [
            Path(".env"),
            Path(__file__).resolve().parent / ".env",
            Path.cwd() / ".env",
        ]
        target = next((p for p in candidates if p.is_file()), None)
    else:
        target = Path(env_path) if Path(env_path).is_file() else None

    loaded = {}
    if target and target.is_file():
        try:
            with target.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val
                    loaded[key] = val
        except Exception as exc:
            print(f"Warning: Error reading .env file {target}: {exc}")

    # Also try python-dotenv if installed
    try:
        import dotenv
        dotenv.load_dotenv(dotenv_path=target)
    except ImportError:
        pass

    return loaded


def str2bool(v: Any) -> bool:
    """Parse boolean command-line arguments accepting true/false/yes/no/1/0."""
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    val = str(v).strip().lower()
    if val in ("yes", "true", "t", "y", "1"):
        return True
    if val in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected ('true' or 'false'), got: {v}")


class S3Uploader:
    """Helper to upload files to AWS S3 using boto3."""

    def __init__(
        self,
        bucket: str = "aithena2026",
        prefix: str = "ocr_data",
        env_file: str | Path | None = None,
    ) -> None:
        load_env_file(env_file)
        self.bucket = bucket.strip()
        self.prefix = prefix.strip().strip("/")

        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"
        self.endpoint_url = os.getenv("AWS_ENDPOINT_URL")

        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "boto3 is required for S3 upload. Please install it using: pip install boto3"
                ) from exc

            kwargs: dict[str, Any] = {"region_name": self.region}
            if self.access_key and self.secret_key:
                kwargs["aws_access_key_id"] = self.access_key
                kwargs["aws_secret_access_key"] = self.secret_key
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url

            self._client = boto3.client("s3", **kwargs)
        return self._client

    def upload_file(
        self,
        local_path: str | Path,
        s3_key: str | None = None,
        bucket: str | None = None,
    ) -> str:
        """Upload a local file to S3 and return the s3:// URI."""
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"Local file does not exist: {path}")

        target_bucket = bucket or self.bucket
        if s3_key is None:
            if self.prefix:
                s3_key = f"{self.prefix}/{path.name}"
            else:
                s3_key = path.name

        client = self._get_client()

        # Guess content type
        content_type = "application/json" if path.suffix in (".json", ".jsonl") else "application/octet-stream"

        extra_args = {"ContentType": content_type}
        client.upload_file(str(path), target_bucket, s3_key, ExtraArgs=extra_args)
        s3_uri = f"s3://{target_bucket}/{s3_key}"
        return s3_uri
