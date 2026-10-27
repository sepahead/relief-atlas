"""Generate and certify local assets in resumable, bounded batches.

Each batch passes image generation/matting, image QA, TRELLIS decoding/export,
mesh QA and Gaussian QA before moving on. No phase can broaden the selection.
Three rejected samples quarantine an item; infrastructure failures stop the run.
"""

import argparse
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import content_policy
from pipeline_utils import (
    LOCAL_DIR, OUTPUT_DIR, PROJECT_DIR, STATE_DIR, image_model_matches, item_certified, load_json,
    pipeline_lock, source_image_matches, verdict_matches, write_json,
)
from prompt_utils import MANIFESTS, mesh_complete, select_manifest_items

IMG_PYTHON = LOCAL_DIR / "imgenv" / "bin" / "python"
MESH_PYTHON = LOCAL_DIR / "trellis-mac" / ".venv" / "bin" / "python"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if os.environ.get("RELIEF_PHASE_FILE"):
        write_json(os.environ["RELIEF_PHASE_FILE"], {
            "at": datetime.now(timezone.utc).isoformat(), "run_id": os.environ.get("RELIEF_RUN_ID"),
            "stage": str(cmd[1]).rsplit("/", 1)[-1],
            "mode": str(cmd[cmd.index("--mode") + 1]) if "--mode" in cmd else None,
        })
    subprocess.run([str(c) for c in cmd], cwd=PROJECT_DIR, check=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["images", "meshes", "all"], default="all")
    ap.add_argument("--geographies", nargs="+", choices=list(MANIFESTS))
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pipeline-type", choices=["512", "1024", "1024_cascade"], default="512")
    ap.add_argument("--texture-size", type=int, choices=[512, 1024, 2048], default=1024)
    ap.add_argument("--mesh-steps", type=int, default=None,
                    help="override sampling steps for structure, shape and texture")
    ap.add_argument("--target-faces", type=int, default=200000)
    ap.add_argument("--gs-format", choices=["spz", "splat", "both"], default="spz")
    ap.add_argument("--image-steps", type=int, default=40)
    ap.add_argument("--image-guidance", type=float, default=1.0)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--structure-reference", type=Path,
                    help="native Qwen guide for exactly one explicit item; cached verified references are reused")
    ap.add_argument("--skip-image-qa", action="store_true", help="generate uncertified meshes")
    ap.add_argument("--skip-mesh-qa", action="store_true")
    ap.add_argument("--skip-gs-qa", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true", help="use only after a successful preflight")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="show the selection and batches without generating")
    content_policy.add_policy_args(ap)
    args = ap.parse_args(argv)
    if args.batch_size < 1 or args.image_steps < 1 or (args.limit is not None and args.limit < 1):
        ap.error("batch size, image steps and limit must be positive")
    if args.target_faces < 1 or (args.mesh_steps is not None and args.mesh_steps < 1):
        ap.error("mesh steps and target faces must be positive")
    if args.image_size < 256 or args.image_size % 16:
        ap.error("image size must be at least 256 and divisible by 16")
    if not math.isfinite(args.image_guidance) or args.image_guidance < 0:
        ap.error("image guidance must be finite and nonnegative")
    if args.preflight_only and args.skip_preflight:
        ap.error("--preflight-only cannot be combined with --skip-preflight")
    if args.structure_reference and (not args.ids or len(args.ids) != 1):
        ap.error("--structure-reference requires exactly one explicit --ids item")
    return args


def ledgers():
    return {mode: load_json(STATE_DIR / f"qa_{mode}.json")
            for mode in ("images", "meshes", "gaussians")}


def directory(item):
    return OUTPUT_DIR / item["geography"] / item["category"] / item["id"]


def phase_complete(item, args, qa_ledgers):
    d, item_id = directory(item), item["id"]
    if not image_model_matches(d, item_id):
        return False
    if args.phase == "images":
        if args.skip_image_qa:
            return (d / f"{item_id}.png").is_file()
        return verdict_matches(qa_ledgers["images"].get(item_id), d, item_id, "images")
    if not mesh_complete(d, item_id) or not source_image_matches(d, item_id):
        return False
    for mode, skip in (("images", args.skip_image_qa), ("meshes", args.skip_mesh_qa),
                       ("gaussians", args.skip_gs_qa)):
        if not skip and not verdict_matches(qa_ledgers[mode].get(item_id), d, item_id, mode):
            return False
    return True


def process_batch(items, args):
    ids = [it["id"] for it in items]
    common = [*content_policy.policy_argv(args), "--ids", *ids]
    qa_common = list(common)

    def qa(python, mode):
        run([python, PROJECT_DIR / "scripts/local/qa.py", "--mode", mode, *qa_common])

    if args.phase in ("images", "all"):
        for _ in range(3):
            guide = getattr(args, "structure_reference", None)
            if not guide or any(not image_model_matches(directory(it), it["id"]) for it in items):
                run([IMG_PYTHON, PROJECT_DIR / "scripts/local/qwen_imagegen.py", *common,
                     "--steps", args.image_steps, "--guidance", args.image_guidance,
                     "--size", args.image_size, *(["--structure-reference", guide] if guide else [])])
            if args.skip_image_qa:
                break
            qa(IMG_PYTHON, "images")
            image_qa = ledgers()["images"]
            quarantine = load_json(STATE_DIR / "qa_quarantine.json")
            if all(it["id"] in quarantine or verdict_matches(
                    image_qa.get(it["id"]), directory(it), it["id"], "images") for it in items):
                break
    elif not args.skip_image_qa:
        # A meshes-only resume still revalidates existing reference images.
        qa(IMG_PYTHON, "images")

    if args.phase in ("meshes", "all"):
        mesh_args = ["--pipeline-type", args.pipeline_type, "--texture-size", args.texture_size,
                     "--gs-format", args.gs_format, "--target-faces", args.target_faces]
        if args.mesh_steps is not None:
            mesh_args += ["--mesh-steps", args.mesh_steps]
        if args.skip_image_qa:
            mesh_args.append("--ignore-image-qa")
        for _ in range(3):
            run([MESH_PYTHON, PROJECT_DIR / "scripts/local/trellis_meshgen.py", *mesh_args, *common])
            if not args.skip_mesh_qa:
                qa(MESH_PYTHON, "meshes")
            if not args.skip_gs_qa:
                qa(MESH_PYTHON, "gaussians")
            current = ledgers()
            quarantine = load_json(STATE_DIR / "qa_quarantine.json")
            if all(it["id"] in quarantine or phase_complete(it, args, current) for it in items):
                break


def main(argv=None):
    args = parse_args(argv)
    try:
        selected = select_manifest_items(args.geographies, args.ids, args.limit, args.only_policy)
        permitted, refused = content_policy.screen_items(selected, args.allow_restricted, args.allow_blocked)
        current = ledgers()
        quarantine = load_json(STATE_DIR / "qa_quarantine.json")
        todo = [it for it in permitted if it["id"] not in quarantine
                and not phase_complete(it, args, current)]
        batches = [todo[start:start + args.batch_size] for start in range(0, len(todo), args.batch_size)]
        print(f"selection: {len(selected)} items | permitted {len(permitted)} | refused {len(refused)} | "
              f"pending {len(todo)} | batches {len(batches)}", flush=True)
        if args.dry_run:
            for number, batch in enumerate(batches, 1):
                print(f"batch {number}: " + " ".join(it["id"] for it in batch))
            return 0
        with pipeline_lock():
            if refused:
                content_policy.record_skipped(refused, phase="orchestrator")
            if not args.skip_preflight and (todo or args.preflight_only):
                from preflight import check
                report = check(phase=args.phase, geographies=args.geographies, ids=args.ids,
                               limit=args.limit, only_policy=args.only_policy,
                               allow_restricted=args.allow_restricted, allow_blocked=args.allow_blocked,
                               gs_format=args.gs_format, pipeline_type=args.pipeline_type,
                               texture_size=args.texture_size, pending_count=len(todo))
                write_json(STATE_DIR / "preflight.json", report)
                for message in report["errors"] + report["notes"]:
                    print(message, flush=True)
                if not report["ready"]:
                    print("preflight failed; see state/preflight.json", flush=True)
                    return 1
            if args.preflight_only:
                print("preflight: READY", flush=True)
                return 0
            for number, batch in enumerate(batches, 1):
                print(f"batch {number}/{len(batches)}: {len(batch)} items", flush=True)
                process_batch(batch, args)
                write_json(STATE_DIR / "run_progress.json", {
                    "at": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
                    "batch": number, "batches": len(batches), "batch_ids": [it["id"] for it in batch],
                    "settings": {k: str(v) if isinstance(v, Path) else v
                                 for k, v in vars(args).items() if k != "ids"},
                })
            current, quarantine = ledgers(), load_json(STATE_DIR / "qa_quarantine.json")
            done = sum(phase_complete(it, args, current) for it in permitted)
            certified = sum(item_certified(directory(it), it["id"], current) for it in permitted)
            quarantined = sum(it["id"] in quarantine and not phase_complete(it, args, current)
                              for it in permitted)
            print(f"run: {done}/{len(permitted)} complete for phase {args.phase}; "
                  f"{certified} fully certified; {quarantined} quarantined", flush=True)
            return 0 if done == len(permitted) else 1
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"pipeline stopped: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
