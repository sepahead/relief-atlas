#!/usr/bin/env bash
# Rebuild the local (on-device) generation workspace for relief-atlas.
#
# Prereqs: macOS on Apple Silicon, uv, aria2c, ~40GB disk for weights.
# No cloud APIs and no HuggingFace auth are needed at generation time:
#   - FLUX.2 [klein] 4B is Apache-2.0 (public download)
#   - TRELLIS.2-4B + TRELLIS-image-large ss_dec checkpoint are MIT (public)
#   - DINOv3-vitl16 is downloaded once into the HF cache by TRELLIS setup
#   - BiRefNet (MIT) is the primary matting model
#   - Falconsai/nsfw_image_detection (Apache-2.0) backs the QA safety screen
#   - RMBG-2.0 is bypassed entirely (inputs carry their own alpha)
#
# The last two are not optional extras. matting.py prefers BiRefNet and only
# falls back to chroma hysteresis, and qa.py's NSFW guard is deliberately
# fail-CLOSED: it raises if the detector is missing rather than passing images
# through unscreened. A workspace without these weights therefore aborts in
# phase 1.5 instead of quietly degrading, so setup has to fetch them.
set -euo pipefail

# scripts/local/setup_local.sh -> repo root is two levels up. The previous
# form probed "$(dirname $0)/../meshmaker/local", i.e. scripts/meshmaker/local,
# which never exists, so it always fell through to the branch below.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/meshmaker/local"
IMG_PYTHON="$LOCAL_DIR/imgenv/bin/python"
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

echo "=== 3/5 BiRefNet (matting) ==="
BIREFNET_DIR="$LOCAL_DIR/models/BiRefNet"
if [ ! -f "$BIREFNET_DIR/model.safetensors" ]; then
    # trust_remote_code model: the custom BiRefNet_config.py / birefnet.py in
    # the repo are required, so take the whole snapshot rather than one file.
    hf download ZhengPeng7/BiRefNet --local-dir "$BIREFNET_DIR"
fi

echo "=== 4/5 NSFW image detector (QA safety screen) ==="
NSFW_DIR="$LOCAL_DIR/models/nsfw_image_detection"
if [ ! -f "$NSFW_DIR/config.json" ]; then
    hf download Falconsai/nsfw_image_detection --local-dir "$NSFW_DIR"
fi

echo "=== 5/5 Environments ==="
# Image-gen env (FLUX.2 via diffusers on MPS).
# scipy: connected-component labelling in matting.py and qa.py.
# imagehash: near-duplicate subject detection in the image QA gate.
# Both are hard imports on the phase-1/1.5 path, not optional.
if [ ! -x imgenv/bin/python ]; then
    uv venv imgenv --python python3.12
fi
UV_HTTP_TIMEOUT=1200 uv pip install --python imgenv/bin/python \
    torch torchvision numpy pillow accelerate safetensors \
    huggingface_hub transformers diffusers \
    scipy imagehash

# Mesh env (trellis-mac port: venv + Metal backends + patches)
if [ ! -d trellis-mac ]; then
    git clone https://github.com/shivampkumar/trellis-mac.git
fi
cd trellis-mac
UV_HTTP_TIMEOUT=1200 bash setup.sh

# .spz container support (official Niantic binding, github.com/nianticlabs/spz).
# Optional: if the build fails or is skipped, the mesh phase still runs and
# emits the universal .splat (32 B/splat) alone. Never install `spz` from PyPI
# as a substitute: that is a third-party Rust port whose __init__ and API are
# both broken (the exporter treats it as missing and falls back to .splat).
TRELLIS_PY="$LOCAL_DIR/trellis-mac/.venv/bin/python"
if ! "$TRELLIS_PY" -c "import spz" >/dev/null 2>&1; then
    if UV_HTTP_TIMEOUT=1200 uv pip install --python "$TRELLIS_PY" \
        "spz @ git+https://github.com/nianticlabs/spz.git"; then
        echo "spz: Niantic binding installed (.spz + .splat per item)"
    else
        echo "spz: build failed -- exporting .splat only (optional feature)" >&2
    fi
fi

# MLX: Apple-silicon on-device GPU stack (no cloud, no API keys). Used by the
# local pipeline tooling for matrix-heavy steps; like spz it is optional and
# a failed install must not abort setup.
if ! "$TRELLIS_PY" -c "import mlx.core" >/dev/null 2>&1; then
    if UV_HTTP_TIMEOUT=1200 uv pip install --python "$TRELLIS_PY" mlx; then
        echo "mlx: installed (on-device Apple GPU acceleration)"
    else
        echo "mlx: install failed -- continuing CPU/MPS only (optional)" >&2
    fi
fi

echo
echo "=== Verifying the workspace can run the gates ==="
cd "$REPO_ROOT"
python3 scripts/local/content_policy.py --selftest
"$IMG_PYTHON" - <<'EOF'
import importlib.util
import sys
from pathlib import Path

missing = [m for m in ("numpy", "PIL", "scipy", "imagehash", "transformers")
           if importlib.util.find_spec(m) is None]
models = Path("meshmaker/local/models")
absent = [name for name, probe in (
    ("FLUX.2-klein-4B", "transformer"),
    ("TRELLIS.2-4B", "ckpts"),
    ("BiRefNet", "config.json"),
    ("nsfw_image_detection", "config.json"),
) if not (models / name / probe).exists()]

print("imgenv packages missing:", missing or "none")
print("model weights missing:  ", absent or "none")
# The NSFW guard is fail-closed, so a missing detector is a hard setup failure
# rather than a warning: the pipeline would abort in phase 1.5 regardless.
# spz is intentionally NOT required: the mesh phase falls back to .splat-only.
sys.exit(1 if missing or absent else 0)
EOF

echo
echo "Workspace ready."
echo "Smoke test:      python3 scripts/local/run_all.py --limit 2"
echo "Policy audit:    python3 scripts/local/content_policy.py --report"
echo "Progress:        python3 scripts/local/verify_local.py"
