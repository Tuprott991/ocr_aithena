"""Batched Vietnamese PARSeq recognition through the official STRHub path."""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parent
PARSEQ_ROOT = ROOT / "third_party" / "parseq"


@dataclass(frozen=True)
class Recognition:
    text: str
    confidence: float


class PARSeqRecognizer:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        batch_size: int = 128,
        fp16: bool = True,
    ) -> None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint}'. Download it with: python download_weights.py")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.fp16 = bool(fp16 and self.device.type == "cuda")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        if str(PARSEQ_ROOT) not in sys.path:
            sys.path.insert(0, str(PARSEQ_ROOT))
        # The checkpoint predates Lightning 2; this restores its removed annotation only.
        import pytorch_lightning.utilities.types as pl_types

        if not hasattr(pl_types, "EPOCH_OUTPUT"):
            pl_types.EPOCH_OUTPUT = list
        from strhub.data.module import SceneTextDataModule
        from strhub.models.utils import load_from_checkpoint

        # Official STRHub loader. weights_only=False is required by PyTorch 2.6 for trusted
        # Lightning 1.6 checkpoints containing scheduler metadata.
        self.model = load_from_checkpoint(str(checkpoint), weights_only=False).eval().to(self.device)
        self.transform = SceneTextDataModule.get_transform(tuple(self.model.hparams.img_size))
        if tuple(self.model.hparams.img_size) != (32, 128):
            raise ValueError(f"Unexpected PARSeq image size: {self.model.hparams.img_size}")
        if tuple(self.model.head.weight.shape) != (220, 384):
            raise ValueError(f"Unexpected Vietnamese PARSeq output head: {tuple(self.model.head.weight.shape)}")
        required = "TiếngViệtĐắkLắk"
        if not all(char in self.model.tokenizer._stoi for char in required):
            raise ValueError("PARSeq checkpoint tokenizer does not preserve Vietnamese precomposed characters")

    @torch.inference_mode()
    def recognize(self, crops: Sequence[np.ndarray]) -> list[Recognition]:
        output: list[Recognition] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start : start + self.batch_size]
            tensors = []
            for crop in chunk:
                if crop is None or crop.size == 0:
                    raise ValueError("Empty crop passed to PARSeq")
                rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
                tensors.append(self.transform(Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")))
            batch = torch.stack(tensors).to(self.device, non_blocking=True)
            if hasattr(torch, "autocast"):
                context = torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16)
            else:  # PyTorch 1.9 official DeepSolo environment.
                context = torch.cuda.amp.autocast(enabled=self.fp16)
            with context:
                probabilities = self.model(batch).softmax(-1)
            texts, token_probabilities = self.model.tokenizer.decode(probabilities.float())
            for text, probs in zip(texts, token_probabilities):
                # This is the sequence confidence used by official STRHub evaluation.
                confidence = float(probs.prod().item())
                output.append(Recognition(unicodedata.normalize("NFC", text), confidence))
        return output
