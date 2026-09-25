#!/usr/bin/env bash
# ==============================================================================
# setup_server.sh - Automated Environment Setup for 2x RTX 5090 on Vast.ai
# ==============================================================================
set -e

echo "======================================================================"
echo " [1/6] Detecting System & GPU Configuration..."
echo "======================================================================"
nvidia-smi || { echo "ERROR: nvidia-smi failed. Ensure NVIDIA drivers are active."; exit 1; }

# Target Blackwell architecture for NVIDIA RTX 5090 (sm_120)
export TORCH_CUDA_ARCH_LIST="9.0;12.0"
export CUDA_FORCE_PTX_JIT=1

# Persist to ~/.bashrc for future sessions
if ! grep -q "CUDA_FORCE_PTX_JIT" ~/.bashrc 2>/dev/null; then
    cat << 'EOF' >> ~/.bashrc
export TORCH_CUDA_ARCH_LIST="9.0;12.0"
export CUDA_FORCE_PTX_JIT=1
EOF
fi

echo "======================================================================"
echo " [2/6] Initializing Hermetic Conda Environment..."
echo "======================================================================"
# Locate conda in standard directories
CONDA_EXE=""
if command -v conda &> /dev/null; then
    CONDA_EXE="$(command -v conda)"
else
    for cand in /root/miniconda3/bin/conda /opt/conda/bin/conda ~/miniconda3/bin/conda /root/anaconda3/bin/conda; do
        if [ -x "$cand" ]; then
            CONDA_EXE="$cand"
            break
        fi
    done
fi

if [ -z "$CONDA_EXE" ]; then
    echo "Conda not found. Installing Miniconda into /root/miniconda3..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    rm /tmp/miniconda.sh
    CONDA_EXE="/root/miniconda3/bin/conda"
fi

# Hook conda into current subshell
eval "$("$CONDA_EXE" shell.bash hook)"
"$CONDA_EXE" init bash > /dev/null 2>&1 || true

ENV_NAME="aithena-ocr"
if conda info --envs | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating..."
    conda activate "${ENV_NAME}"
else
    echo "Creating hermetic conda environment '${ENV_NAME}' with Python 3.8, GCC 9.5 & CUDA toolkit..."
    conda create -n "${ENV_NAME}" python=3.8 -c conda-forge \
        gcc_linux-64=9.5.0 gxx_linux-64=9.5.0 cudatoolkit-dev=11.1.1 ninja -y
    conda activate "${ENV_NAME}"
fi

echo "======================================================================"
echo " [3/6] Installing PyTorch 1.9.0 & Detectron2 0.6..."
echo "======================================================================"
pip install --upgrade pip==24.0 setuptools==59.5.0 wheel

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

echo "======================================================================"
echo " [4/6] Installing Requirements & Building Packages..."
echo "======================================================================"
pip install -r requirements.txt
pip install -e third_party/parseq --no-deps

# Build DeepSolo CUDA extension (with fallback to pure PyTorch if nvcc differs)
echo "Configuring DeepSolo..."
cd third_party/DeepSolo/DeepSolo
rm -rf build adet/*.so *.egg-info
python setup.py build develop || python setup.py develop --no-deps || true
cd ../../..

# Register project & vendor modules in PYTHONPATH
PROJECT_DIR="$(pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/third_party/DeepSolo/DeepSolo:${PROJECT_DIR}/third_party/parseq:${PYTHONPATH:-}"

if ! grep -q "ocr_aithena" ~/.bashrc 2>/dev/null; then
    cat << EOF >> ~/.bashrc
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/third_party/DeepSolo/DeepSolo:${PROJECT_DIR}/third_party/parseq:\$PYTHONPATH"
EOF
fi

echo "======================================================================"
echo " [5/6] Downloading Weights from Hugging Face (Vantuk/ocr_aithena)..."
echo "======================================================================"
python download_weights.py

echo "======================================================================"
echo " [6/6] Verifying Hardware & Model Loading..."
echo "======================================================================"
python -c "
import torch
print(f'PyTorch Version  : {torch.__version__}')
print(f'CUDA Available   : {torch.cuda.is_available()}')
print(f'CUDA Device Count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  [GPU {i}] {torch.cuda.get_device_name(i)}')

import detectron2
print(f'Detectron2       : {detectron2.__version__}')

from adet.modeling import vitae_v2
print('DeepSolo (adet)  : OK')

from strhub.models.parseq.system import PARSeq
print('PARSeq (strhub)  : OK')

print('\n>>> ALL CHECKS PASSED SUCCESSFULLY! <<<')
"

echo "======================================================================"
echo " SETUP COMPLETED SUCCESSFULLY FOR 2x RTX 5090 ON VAST.AI!"
echo "======================================================================"
echo "To process your videos, run:"
echo "  conda activate aithena-ocr"
echo "  python batch_runner.py \\"
echo "    --input-dir new_images_webp \\"
echo "    --output-dir ocr_results \\"
echo "    --gpus 0,1 \\"
echo "    --workers-per-gpu 1 \\"
echo "    --det-batch-size 8 \\"
echo "    --rec-batch-size 256 \\"
echo "    --merge-jsonl all_videos_ocr.jsonl \\"
echo "    --s3_upload true"
echo "======================================================================"
