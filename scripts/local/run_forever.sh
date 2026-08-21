#!/usr/bin/env bash
# Headless full-run driver: keeps re-running both pipeline phases until every
# manifest item has outputs (or only hard failures remain). Designed to be
# launched detached, e.g.:
#   nohup scripts/local/run_forever.sh > state/full_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p state
while true; do
    python3 scripts/local/run_all.py
    BEFORE=$(python3 scripts/local/verify_local.py | tail -3 | head -1)
    echo "=== pass finished: $BEFORE"
    # Stop if a full pass produced no mesh work (all done or all failing)
    MESHES_LEFT=$(python3 - <<'EOF'
import json, sys
sys.path.insert(0, "scripts/local")
from prompt_utils import MANIFESTS, load_manifest
from pathlib import Path
out = Path("outputs_relief")
fails = set()
fp = Path("state/local_mesh_failures.json")
if fp.exists():
    fails = set(json.loads(fp.read_text()))
left = 0
for geo in MANIFESTS:
    for it in load_manifest(geo):
        d = out / geo / it["category"] / it["id"]
        png = d / f"{it['id']}.png"
        glb = d / f"{it['id']}.glb"
        ply = d / f"{it['id']}.ply"
        if png.exists() and not (glb.exists() and ply.exists()):
            if it["id"] not in fails:
                left += 1
print(left)
EOF
    )
    echo "=== meshes pending: $MESHES_LEFT"
    [ "$MESHES_LEFT" -eq 0 ] && { echo "=== all done"; break; }
    sleep 60
done
