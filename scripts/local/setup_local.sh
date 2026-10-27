#!/usr/bin/env bash
# Prepare the Apple Silicon image/mesh environments and pinned local models.
# DINOv3 needs approved Hugging Face access once; generation then stays offline.
set -euo pipefail
export UV_CONCURRENT_DOWNLOADS=1 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export HF_HUB_DISABLE_XET=1
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCAL_DIR="$REPO_ROOT/meshmaker/local"
IMG_PYTHON="$LOCAL_DIR/imgenv/bin/python"
MESH_PYTHON="$LOCAL_DIR/trellis-mac/.venv/bin/python"
cd "$REPO_ROOT"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "This local backend requires macOS on Apple Silicon." >&2
    exit 1
fi
for command in python3 uv git; do
    command -v "$command" >/dev/null || { echo "Missing prerequisite: $command" >&2; exit 1; }
done
mkdir -p "$LOCAL_DIR/models"

# Keep a healthy existing environment intact. Missing/broken environments are
# rebuilt from the versions exercised by the local pipeline tests and pilot.
environment_ok() {
    python3 - "$1" "$2" <<'PY'
import sys
sys.path.insert(0, 'scripts/local')
from preflight import probe_environment
report = probe_environment(sys.argv[1], sys.argv[2])
for error in report.get('errors', []):
    print(error, file=sys.stderr)
sys.exit(bool(report.get('errors')))
PY
}
if ! environment_ok "$IMG_PYTHON" images; then
    [ -x "$IMG_PYTHON" ] || uv venv "$LOCAL_DIR/imgenv" --python 3.12
    UV_HTTP_TIMEOUT=1200 uv pip install --python "$IMG_PYTHON" -r requirements-image.txt
fi

read_runtime() {
    python3 - "$1" "$2" <<'PY'
import json, sys
with open('config/local_runtime.json') as fp:
    print(json.load(fp)[sys.argv[1]][sys.argv[2]])
PY
}
if [ ! -d "$LOCAL_DIR/trellis-mac" ]; then
    git clone --no-checkout "$(read_runtime trellis_mac repository)" "$LOCAL_DIR/trellis-mac"
    git -C "$LOCAL_DIR/trellis-mac" checkout --detach "$(read_runtime trellis_mac revision)"
fi
if ! environment_ok "$MESH_PYTHON" meshes; then
    # Relative path avoids UV_CONSTRAINT's space-separated path parsing when
    # the checkout itself has spaces. The upstream bootstrap uses uv too.
    cp requirements-mesh.txt "$LOCAL_DIR/trellis-mac/.relief-runtime.constraints.txt"
    (
        cd "$LOCAL_DIR/trellis-mac"
        UV_CONSTRAINT=.relief-runtime.constraints.txt UV_HTTP_TIMEOUT=1200 bash setup.sh
    )
fi

# Validate the actual Niantic API, not merely whether some package named spz
# imports. A failed optional build is visible and uses the larger SPLAT format.
if ! "$MESH_PYTHON" -c 'import spz; assert all(hasattr(spz,n) for n in ("GaussianCloud","PackOptions","save_spz","load_spz"))' >/dev/null 2>&1; then
    SPZ_SOURCE="spz @ git+$(read_runtime spz repository)@$(read_runtime spz revision)"
    if ! UV_HTTP_TIMEOUT=1200 uv pip install --python "$MESH_PYTHON" "$SPZ_SOURCE"; then
        echo "Niantic SPZ build failed; plan disk capacity for SPLAT fallback." >&2
    fi
fi

# Downloads only missing/invalid weights, validates shard/header integrity and
# rewrites TRELLIS's decoder and DINO references to absolute/local paths.
MODEL_TRANSFER_ARGS=()
if command -v aria2c >/dev/null; then
    MODEL_TRANSFER_ARGS+=(--resumable-qwen --resumable-dino)
fi
"$IMG_PYTHON" scripts/local/prepare_models.py --fingerprint "${MODEL_TRANSFER_ARGS[@]}"
python3 scripts/local/content_policy.py --selftest
python3 scripts/local/run_all.py --preflight-only

echo "Workspace ready for local generation."
echo "Representative pilot: python3 scripts/local/run_all.py --ids gen_dis_0001 ger_uav_0001 gen_med_0001 eu_rc_0001"
echo "Progress:             python3 scripts/local/pending.py --certified"
