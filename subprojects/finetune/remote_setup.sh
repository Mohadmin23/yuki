#!/usr/bin/env bash
# Runs on the Thunder Compute box after `tnr connect 0`.
# Idempotent: safe to re-run (but on a clean .venv preferably).

set -euo pipefail

WORKDIR="${HOME}/yuki-finetune"
cd "${WORKDIR}"

echo "[remote_setup] python: $(python3 --version)"
echo "[remote_setup] nvidia-smi:"
nvidia-smi || { echo "no GPU visible"; exit 1; }

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

pip install --upgrade pip wheel setuptools

# Install Unsloth first and let it pull the matching torch/xformers/torchao.
# Pinning these ourselves causes resolver wars (xformers wants 2.5.1,
# torchvision wants 2.11+, unsloth wants <2.11, etc).
echo "[remote_setup] installing unsloth (pulls compatible torch + friends)..."
pip install unsloth

# HF training stack that Unsloth doesn't pull on its own.
echo "[remote_setup] installing HF + bitsandbytes stack..."
pip install \
    'bitsandbytes>=0.44.0' \
    'transformers>=4.46.0,<5' \
    'trl>=0.12.0,<0.25' \
    'peft>=0.13.0' \
    'accelerate>=1.1.0' \
    'datasets>=3.0.0,<4' \
    sentencepiece \
    protobuf \
    hf_transfer

echo "[remote_setup] sanity check torch + cuda:"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "[remote_setup] sanity check unsloth:"
python -c "from unsloth import FastLanguageModel; print('unsloth ok')"

echo
echo "[remote_setup] ready. To start training:"
echo "  source ${WORKDIR}/.venv/bin/activate"
echo "  python train.py --dataset ${WORKDIR}/yuki_clean_v2.jsonl --output ${WORKDIR}/output --epochs 3"
