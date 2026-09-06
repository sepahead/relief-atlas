"""Orchestrator for the fully-local relief-atlas pipeline.

Phase 0   — content policy gate: every phase screens item text and refuses
            weapons/munitions outright; unarmed military platforms and
            personnel need an explicit --allow-restricted, weapons an explicit
            --allow-blocked. The policy flags are forwarded to every phase
            INCLUDING the QA gate, which re-screens independently.
Phase 1   — FLUX.2 [klein] 4B text-to-image (meshmaker/local/imgenv)
Phase 1.5 — image QA gate: alpha/sharpness/duplicate checks + NSFW screen;
            failures delete the PNG and bump the item's retry seed, and after
            a few attempts the item is quarantined instead of looping
Phase 2   — TRELLIS.2-4B image-to-3D: GLB mesh + 3DGS PLY (trellis-mac/.venv);
            only items with a PASSING image QA verdict are meshed
Phase 2.5 — mesh QA gate: bbox sanity, white-slab and backdrop-residue checks
            plus a turntable ray-cast pass (16 views, silhouette coverage /
            holes / view consistency, <id>.qa_mesh_views.png montage);
            failures delete the GLB/3DGS and bump the retry seed
Phase 2.6 — gaussians QA gate: the 3DGS containers (.spz/.splat) are
            rasterised from the same 16-view turntable (coverage, solidity,
            holes, view consistency) and native primitive cues are checked on
            the cloud; failures wipe the containers so the mesh phase
            regenerates them, montage at <id>.qa_views.png

All phases are checkpointed: items with existing outputs are skipped, so the
runner can be relaunched any number of times to continue a multi-day 10K run.

Usage:
    python3 scripts/local/run_all.py                    # all phases, all manifests
    python3 scripts/local/run_all.py --phase images     # only phase 1
    python3 scripts/local/run_all.py --limit 10         # smoke test

    # audit what the policy gate catches, without producing the dataset:
    python3 scripts/local/content_policy.py --list blocked
    python3 scripts/local/run_all.py --only-policy blocked --allow-blocked
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_policy  # noqa: E402

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
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pipeline-type", default="512")
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--skip-image-qa", action="store_true",
                    help="skip phase 1.5; the mesh phase will then find no "
                         "passing verdicts and mesh nothing")
    ap.add_argument("--skip-mesh-qa", action="store_true")
    ap.add_argument("--skip-gs-qa", action="store_true",
                    help="skip phase 2.6 (turntable QA on the .spz/.splat)")
    content_policy.add_policy_args(ap)
    args = ap.parse_args()

    # The QA gate re-runs the same policy screen as the generators, so it must
    # be given the same policy flags. When it was not, --allow-restricted meant
    # "generate 540 restricted items, then quarantine every one of them as a
    # policy failure on the very next step".
    policy = content_policy.policy_argv(args)

    common = [*policy]
    if args.geographies:
        common += ["--geographies", *args.geographies]
    if args.ids:
        common += ["--ids", *args.ids]
    if args.limit:
        common += ["--limit", str(args.limit)]

    # qa.py takes neither --geographies nor --limit, so it gets the policy
    # flags plus whatever item selection it does understand.
    qa_common = [*policy]
    if args.ids:
        qa_common += ["--ids", *args.ids]

    def qa(python, mode):
        run([python, PROJECT_DIR / "scripts" / "local" / "qa.py",
             "--mode", mode, *qa_common])

    if args.phase in ("images", "all"):
        run([IMG_PYTHON, PROJECT_DIR / "scripts" / "local" / "flux_imagegen.py", *common])
        if not args.skip_image_qa:
            qa(IMG_PYTHON, "images")
            # QA may have wiped rejected images; regenerate them with the
            # perturbed retry seeds before moving on.
            run([IMG_PYTHON, PROJECT_DIR / "scripts" / "local" / "flux_imagegen.py", *common])
            qa(IMG_PYTHON, "images")

    if args.phase in ("meshes", "all"):
        # Bounded regeneration loop: each QA round wipes failing outputs, the
        # next meshgen pass regenerates them with the perturbed retry seed,
        # and the following round re-judges them. meshgen early-returns when
        # nothing needs meshing, so later rounds are cheap no-ops.
        for attempt in range(3):
            run([MESH_PYTHON, PROJECT_DIR / "scripts" / "local" / "trellis_meshgen.py",
                 "--pipeline-type", args.pipeline_type,
                 "--texture-size", str(args.texture_size), *common])
            if args.skip_mesh_qa and args.skip_gs_qa:
                break
            if not args.skip_mesh_qa:
                qa(MESH_PYTHON, "meshes")
            if not args.skip_gs_qa:
                qa(MESH_PYTHON, "gaussians")


if __name__ == "__main__":
    main()
