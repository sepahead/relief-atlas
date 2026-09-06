#!/usr/bin/env bash
# Headless full-run driver: keeps re-running both pipeline phases until every
# manifest item has outputs (or only hard failures remain). Designed to be
# launched detached, e.g.:
#   nohup scripts/local/run_forever.sh > state/full_run.log 2>&1 &
#
# Any arguments are forwarded verbatim to run_all.py, so policy overrides and
# scoping work here too:
#   nohup scripts/local/run_forever.sh --allow-restricted > state/full_run.log 2>&1 &
#
# Termination is the whole point of this script, so the pending-work count has
# to agree with the pipeline about what "done" and "never going to finish" mean:
#   * a finished item is GLB + a 3DGS container + metadata.json, where the
#     container may be .spz/.splat/.ply (prompt_utils.GS_EXTENSIONS). This file
#     used to hardcode `.ply`, which the condensed-export change stopped
#     emitting -- so the count never reached zero and this loop ran forever,
#     re-launching a full no-op pass every 60 seconds indefinitely.
#   * items QA has quarantined, or that lack a passing image-QA verdict, cannot
#     progress no matter how many passes run, so they are not pending work.
set -uo pipefail
cd "$(dirname "$0")/../.."

mkdir -p state

PENDING_PY=$(cat <<'EOF'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/local")
from prompt_utils import MANIFESTS, load_manifest, mesh_complete

out = Path("outputs_relief")
state = Path("state")


def load(name):
    p = state / name
    return json.loads(p.read_text()) if p.exists() else {}


hard_fails = set(load("local_mesh_failures.json"))
quarantine = set(load("qa_quarantine.json"))
image_qa = load("qa_images.json")

pending = 0
for geo in MANIFESTS:
    for it in load_manifest(geo):
        item_id = it["id"]
        d = out / geo / it["category"] / item_id
        if not (d / f"{item_id}.png").exists():
            continue                      # image phase has not produced it yet
        if item_id in quarantine or item_id in hard_fails:
            continue                      # will never progress
        if not image_qa.get(item_id, {}).get("pass"):
            continue                      # mesh phase refuses it by design
        if not mesh_complete(d, item_id):
            pending += 1
print(pending)
EOF
)

while true; do
    python3 scripts/local/run_all.py "$@"
    echo "=== pass finished"
    python3 scripts/local/verify_local.py
    MESHES_LEFT=$(python3 -c "$PENDING_PY")
    echo "=== meshes pending: $MESHES_LEFT"
    if ! [ "$MESHES_LEFT" -eq "$MESHES_LEFT" ] 2>/dev/null; then
        echo "=== could not compute pending count; stopping rather than spinning"
        exit 1
    fi
    [ "$MESHES_LEFT" -eq 0 ] && { echo "=== all done"; break; }
    sleep 60
done
