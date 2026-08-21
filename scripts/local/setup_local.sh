#!/usr/bin/env bash
# Rebuild the local (on-device) generation workspace for relief-atlas.
#
# Prereqs: macOS on Apple Silicon, uv, aria2c, ~40GB disk for weights.
# No cloud APIs and no HuggingFace auth are needed at generation time:
#   - FLUX.2 [klein] 4B is Apache-2.0 (public download)
#   - TRELLIS.2-4B + TRELLIS-image-large ss_dec checkpoint are MIT (public)
#   - DINOv3-vitl16 is downloaded once into the HF cache by TRELLIS setup
#   - RMBG-2.0 is bypassed entirely (inputs carry their own alpha)
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/../meshmaker/local" 2>/dev/null && pwd || true)"
if [ -z "${LOCAL_DIR}" ]; then
    REPO="$(cd "$(dirname "$0")/../.." && pwd)"
    LOCAL_DIR="$REPO/meshmaker/local"
fi
mkdir -p "$LOCAL_DIR/models"
cd "$LOCAL_DIR"

HF="https://huggingface.co"

aria_dl() { # aria_dl <url> <dir> [name]
    local url="$1" dir="$2" name="${3:-$(basename "$url")}"
    mkdir -p "$dir"
    [ -f "$dir/$name" ] && { echo "  exists: $dir/$name"; return; }
    aria2c -x 8 -s 16 -d "$dir" -o "$name" "$url"
}

echo "=== 1/3 FLUX.2 [klein] 4B (text-to-image) ==="
FLUX_DIR="$LOCAL_DIR/models/FLUX.2-klein-4B"
if [ ! -d "$FLUX_DIR/transformer" ]; then
    # Full repo snapshot via hf CLI (public repo, no login needed)
    hf download black-forest-labs/FLUX.2-klein-4B --local-dir "$FLUX_DIR"
fi

echo "=== 2/3 TRELLIS.2-4B (image-to-3D) ==="
TRELLIS_DIR="$LOCAL_DIR/models/TRELLIS.2-4B"
if [ ! -d "$TRELLIS_DIR/ckpts" ]; then
    hf download microsoft/TRELLIS.2-4B --local-dir "$TRELLIS_DIR"
fi
# ss_dec decoder ships in the TRELLIS-image-large repo
aria_dl "$HF/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.json" \
    "$TRELLIS_DIR/ckpts"
aria_dl "$HF/microsoft/TRELLIS-image-large/resolve/main/ckpts/ss_dec_conv3d_16l8_fp16.safetensors" \
    "$TRELLIS_DIR/ckpts"
# Keep the pipeline fully offline: point the decoder at the local checkpoint
python3 - "$TRELLIS_DIR/pipeline.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
m = p["args"]["models"]
if "TRELLIS-image-large" in m["sparse_structure_decoder"]:
    m["sparse_structure_decoder"] = "ckpts/ss_dec_conv3d_16l8_fp16"
    json.dump(p, open(sys.argv[1], "w"), indent=4)
    print("pipeline.json: sparse_structure_decoder -> local ckpts")
EOF

echo "=== 3/3 Environments ==="
# Image-gen env (FLUX.2 via diffusers on MPS)
if [ ! -x imgenv/bin/python ]; then
    uv venv imgenv --python python3.12
fi
UV_HTTP_TIMEOUT=1200 uv pip install --python imgenv/bin/python \
    torch torchvision numpy pillow accelerate safetensors \
    huggingface_hub transformers diffusers

# Mesh env (trellis-mac port: venv + Metal backends + patches)
if [ ! -d trellis-mac ]; then
    git clone https://github.com/shivampkumar/trellis-mac.git
fi
cd trellis-mac
UV_HTTP_TIMEOUT=1200 bash setup.sh

echo
echo "Workspace ready."
echo "Smoke test:  python3 scripts/local/run_all.py --limit 2"
