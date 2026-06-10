#!/usr/bin/env bash
# Fetch the OpenAI ADM (guided-diffusion) code + the unconditional 256x256
# checkpoint so the `adm` backend can run. ~2.1GB download.
set -euo pipefail
cd "$(dirname "$0")/.."

# Runtime deps for ADM (blobfile). Run ADM with `uv run --extra adm ...`.
uv sync --extra adm

# guided_diffusion code is not pip-installable (broken setup.py), so vendor it
# via git clone; the adm backend adds third_party/guided-diffusion to sys.path.
mkdir -p third_party
if [ ! -d third_party/guided-diffusion ]; then
  git clone --depth 1 https://github.com/openai/guided-diffusion.git third_party/guided-diffusion
fi

# Unconditional 256x256 diffusion checkpoint.
mkdir -p models
CKPT="models/256x256_diffusion_uncond.pt"
CKPT_SHA256="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"
if [ ! -f "$CKPT" ]; then
  curl -L --fail -o "$CKPT" \
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
fi
echo "verifying checkpoint sha256 ..."
ACTUAL=$( (sha256sum "$CKPT" 2>/dev/null || shasum -a 256 "$CKPT") | awk '{print $1}')
if [ "$ACTUAL" != "$CKPT_SHA256" ]; then
  echo "ERROR: $CKPT failed checksum verification — delete it and re-run." >&2
  exit 1
fi
echo "ADM setup complete -> $CKPT"
