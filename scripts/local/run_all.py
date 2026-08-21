"""Orchestrator for the fully-local relief-atlas pipeline.

Phase 1 — FLUX.2 [klein] 4B text-to-image (meshmaker/local/imgenv)
Phase 2 — TRELLIS.2-4B image-to-3D: GLB mesh + 3DGS PLY (trellis-mac/.venv)

Both phases are checkpointed: items with existing outputs are skipped, so the
runner can be relaunched any number of times to continue a multi-day 10K run.

Usage:
    python3 scripts/local/run_all.py                    # both phases, all manifests
    python3 scripts/local/run_all.py --phase images     # only phase 1
    python3 scripts/local/run_all.py --limit 10         # smoke test
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_DIR / "meshmaker" / "local"
IMG_PYTHON = LOCAL_DIR / "imgenv" / "bin" / "python"
MESH_PYTHON = LOCAL_DIR / "trellis-mac" / ".venv" / "bin" / "python"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=PROJECT_DIR)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["images", "meshes", "all"], default="all")
    ap.add_argument("--geographies", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pipeline-type", default="512")
    ap.add_argument("--texture-size", type=int, default=1024)
    args = ap.parse_args()

    common = []
    if args.geographies:
        common += ["--geographies", *args.geographies]
    if args.limit:
        common += ["--limit", str(args.limit)]

    if args.phase in ("images", "all"):
        run([IMG_PYTHON, PROJECT_DIR / "scripts" / "local" / "flux_imagegen.py", *common])

    if args.phase in ("meshes", "all"):
        run([MESH_PYTHON, PROJECT_DIR / "scripts" / "local" / "trellis_meshgen.py",
             "--pipeline-type", args.pipeline_type,
             "--texture-size", str(args.texture_size), *common])


if __name__ == "__main__":
    main()
