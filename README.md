# Vietnamese broadcast-frame OCR MVP

This repository implements:

`frame -> DeepSolo ViTAEv2-S localization -> polygon rectification -> Vietnamese PARSeq -> temporal overlay cleanup -> JSON`

The detector's built-in Latin recognizer is deliberately ignored. DeepSolo supplies only its confidence, centerline, and learned two-sided boundary; the Vietnamese PARSeq checkpoint recognizes every rectified crop in CUDA batches.

## Verified checkpoint contracts

- `vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth`: official DeepSolo ViTAEv2-S pretraining config, 25 points, 100 queries, 37-character auxiliary vocabulary, and boundary head.
- `best-parseq.ckpt`: original (pre-February-2024) STRHub PARSeq layout, 32x128 input, maximum 25 characters, 220 output symbols, and a 220-character Vietnamese/punctuation charset. The vendored STRHub source is pinned to commit `ed3d847`, matching the unwrapped checkpoint keys.
- DeepSolo is vendored from official commit `dbadae995035246bad3376c7a44c015c69e9b313`.

The code validates key tensor shapes before inference and fails clearly on an incompatible checkpoint.

## Pretrained Model Weights

The pretrained models are hosted on Hugging Face: [**Vantuk/ocr_aithena**](https://huggingface.co/Vantuk/ocr_aithena).

Download both checkpoints to the root directory with a single command:
```bash
python download_weights.py
```
Or download manually via `huggingface-cli`:
```bash
huggingface-cli download Vantuk/ocr_aithena vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth --local-dir .
huggingface-cli download Vantuk/ocr_aithena best-parseq.ckpt --local-dir .
```

## Environment

DeepSolo's official stack is Linux-only in practice because it requires Detectron2 0.6 and a compiled deformable-attention CUDA extension. Use Linux or WSL2, Python 3.8, CUDA 11.1, and the versions documented by DeepSolo:

```bash
conda create -n aithena-ocr python=3.8 -y
conda activate aithena-ocr
pip install pip==24.0 setuptools==59.5.0
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install -r requirements.txt
pip install -e third_party/parseq --no-deps
cd third_party/DeepSolo/DeepSolo
python setup.py build develop
cd ../../..
```

`nvcc` and a C++ compiler must be available while building DeepSolo. The build must include CUDA sources; check that `CUDA_HOME` is set if it does not.

## High-Throughput Multi-GPU Batch Processing (Dual RTX 5090 Server)

For large-scale video frame processing on multi-GPU servers (e.g., dual NVIDIA RTX 5090 with 32GB VRAM each), use `batch_runner.py`.

### Frame Folder Structure
Organize extracted video frames as:
```text
new_images_webp/
├── L21_V001/
│   ├── 1.webp
│   ├── 2.webp
│   └── ...
├── L21_V002/
│   ├── 1.webp
│   └── ...
└── ...
```
> Frames named `<idx>.webp` are automatically naturally sorted (`1.webp`, `2.webp`, ..., `10.webp`, `100.webp`) so temporal ordering is strictly preserved.

### Run on Dual RTX 5090 with AWS S3 Upload:
```bash
python batch_runner.py \
  --input-dir new_images_webp \
  --output-dir ocr_results \
  --gpus 0,1 \
  --workers-per-gpu 1 \
  --det-batch-size 8 \
  --rec-batch-size 256 \
  --merge-jsonl all_videos_ocr.jsonl \
  --s3_upload true \
  --s3-bucket aithena2026 \
  --s3-prefix ocr_data
```

### AWS S3 Configuration (`.env`)
Create a `.env` file in the root directory:
```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1  # optional
```
When `--s3_upload true` is passed:
- Each video's JSON is automatically uploaded to `s3://aithena2026/ocr_data/<video_name>.json`.
- The consolidated master JSONL is uploaded to `s3://aithena2026/ocr_data/all_videos_ocr.jsonl`.

### Key Optimizations for Dual RTX 5090:
1. **Dynamic Video-Level Parallelism**: Videos are dynamically distributed across GPUs (`cuda:0` and `cuda:1`). Zero cross-GPU communication latency, achieving 100% linear speedup.
2. **Batched DeepSolo Localization**: Detects 8–16 frames concurrently in a single forward pass, keeping RTX 5090 Tensor Cores fully saturated.
3. **Cross-Frame PARSeq Recognition**: Aggregates all detected crops across batched frames and recognizes them with a batch size of 256–512 in FP16.
4. **Asynchronous Prefetching**: Multi-threaded WebP decoding decodes upcoming frames in RAM in background worker threads so the GPUs never wait on disk I/O.
5. **Crash-Proof Auto-Resume**: Skips already completed videos (`ocr_results/<video_name>.json`) if interrupted. Use `--force` to re-process.
6. **One-Step Elasticsearch JSONL**: Directly outputs Elasticsearch-compatible JSON per video and automatically merges into a single `.jsonl` for bulk indexing.

## Merging Existing JSONs to JSONL

To merge an existing directory of video JSON outputs into a single Elasticsearch-ready JSONL (with optional S3 upload):
```bash
python merge_to_jsonl.py ocr_results --output bulk_elasticsearch.jsonl --s3_upload true
```


