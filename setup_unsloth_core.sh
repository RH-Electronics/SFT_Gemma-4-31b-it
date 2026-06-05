#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Clean isolated Unsloth Core environment for Gemma-4-31b-it training
# Does NOT touch existing Unsloth Studio or existing Python envs.
# Creates: ~/gemma_unsloth_core/.venv
# ============================================================

WORKDIR="$HOME/gemma_unsloth_core"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  printf "\n\033[1;35m%s\033[0m\n" "$*"
}

die() {
  printf "\n\033[1;31mERROR:\033[0m %s\n" "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Command not found: $1"
}

log "Checking NVIDIA GPU..."
need_cmd nvidia-smi
nvidia-smi

log "Creating isolated folder: $WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

log "Checking Python..."
need_cmd "$PYTHON_BIN"
"$PYTHON_BIN" --version

log "Creating venv: $WORKDIR/.venv"
"$PYTHON_BIN" -m venv .venv || die "Could not create venv. On Ubuntu install venv package, for example:
sudo apt install python3-venv
or:
sudo apt install python3.11-venv"

source .venv/bin/activate

log "Upgrading pip and installing uv inside this venv..."
python -m pip install --upgrade pip wheel setuptools
python -m pip install --upgrade uv

log "Installing Unsloth Core with uv torch backend auto..."
uv pip install --upgrade unsloth --torch-backend=auto

log "Installing/refreshing common training packages..."
uv pip install --upgrade \
  trl \
  transformers \
  accelerate \
  datasets \
  peft \
  bitsandbytes \
  sentencepiece \
  protobuf \
  safetensors \
  huggingface_hub

log "Testing imports and CUDA..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram free/total: {free/1024**3:.2f} / {total/1024**3:.2f} GB")

import unsloth
print("unsloth import: OK")

import trl, transformers, datasets, peft, bitsandbytes
print("trl:", trl.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("bitsandbytes import: OK")
PY

log "Done ❤️"
echo
echo "To activate later:"
echo "  source $WORKDIR/.venv/bin/activate"
echo
echo "To run training:"
echo "  cd $WORKDIR"
echo "  python train_gemma4-31b-it_text_only_vision_off.py --no-confirm"
