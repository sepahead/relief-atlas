"""Validate local models, GPU execution, packages, manifests and disk capacity."""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import content_policy
from pipeline_utils import (
    LOCAL_DIR, OUTPUT_DIR, PROJECT_DIR, STATE_DIR, image_model_matches, item_certified, load_json,
    verdict_matches, write_json,
)
from prepare_models import catalog, model_errors
from prompt_utils import MANIFESTS, select_manifest_items

IMG_PYTHON = LOCAL_DIR / "imgenv" / "bin" / "python"
MESH_PYTHON = LOCAL_DIR / "trellis-mac" / ".venv" / "bin" / "python"


def probe_environment(python, mode):
    """Exercise imports and an actual MPS allocation in the owning environment."""
    if not Path(python).is_file():
        return {"errors": [f"Missing environment: {python}"]}
    modules = (["torch", "torchvision", "diffusers", "transformers", "accelerate",
                "numpy", "PIL", "scipy", "imagehash", "timm"] if mode == "images"
               else ["torch", "torchvision", "transformers", "numpy", "PIL", "scipy",
                     "trimesh", "fast_simplification", "xatlas", "easydict"])
    code = '''
import importlib, json, sys
from importlib.metadata import version, PackageNotFoundError
report = {"errors": [], "versions": {}, "mps": False}
for name in json.loads(sys.argv[1]):
    try:
        importlib.import_module(name)
    except Exception as exc:
        report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
if sys.argv[2] == "images":
    try:
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    except Exception as exc:
        report["errors"].append(f"Qwen-Image-2.1 backend import: {type(exc).__name__}: {exc}")
if sys.argv[2] == "meshes":
    try:
        sys.path.insert(0, sys.argv[3])
        import trellis_meshgen
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    except Exception as exc:
        report["errors"].append(f"TRELLIS backend import: {type(exc).__name__}: {exc}")
try:
    import torch
    report["mps"] = torch.backends.mps.is_available()
    if report["mps"]:
        assert torch.ones(1, device="mps").cpu().item() == 1
    else:
        report["errors"].append("Apple MPS is unavailable")
except Exception as exc:
    report["errors"].append(f"MPS allocation: {exc}")
for name in ("torch", "torchvision", "diffusers", "transformers", "numpy", "scipy", "trimesh", "spz"):
    try:
        report["versions"][name] = version(name)
    except PackageNotFoundError:
        pass
try:
    import spz
    report["spz"] = all(hasattr(spz, name) for name in ("GaussianCloud", "PackOptions", "save_spz", "load_spz"))
except Exception:
    report["spz"] = False
print(json.dumps(report))
'''
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    try:
        proc = subprocess.run([str(python), "-c", code, json.dumps(modules), mode,
                               str(PROJECT_DIR / "scripts" / "local")],
                              capture_output=True, text=True, env=env, timeout=90)
        if proc.returncode:
            return {"errors": [f"Environment probe failed: {proc.stderr[-2000:]}"]}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, subprocess.TimeoutExpired, OSError) as exc:
        return {"errors": [f"Environment probe failed: {exc}"]}


def storage_estimate(count, gs_format="spz", pipeline_type="512", texture_size=1024):
    # Conservative budgets above the observed sample means, plus scratch space.
    per_item = {"spz": 24_000_000, "splat": 80_000_000, "both": 85_000_000}[gs_format]
    factor = (4 if pipeline_type != "512" else 1) * max(1, (texture_size / 1024) ** 2)
    return int(count * per_item * factor * 1.2 + 10_000_000_000)


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def check(phase="all", geographies=None, ids=None, limit=None, only_policy=None,
          allow_restricted=False, allow_blocked=False, gs_format="spz",
          pipeline_type="512", texture_size=1024, pending_count=None):
    errors, notes = [], []
    items = select_manifest_items(geographies, ids, limit, only_policy)
    if len({it["id"] for it in items}) != len(items):
        errors.append("Manifest selection contains duplicate IDs")
    permitted, refused = content_policy.screen_items(items, allow_restricted, allow_blocked)
    if pending_count is None:
        quarantine = load_json(STATE_DIR / "qa_quarantine.json")
        qa_ledgers = {mode: load_json(STATE_DIR / f"qa_{mode}.json")
                      for mode in ("images", "meshes", "gaussians")}
        pending_count = 0
        for it in permitted:
            if it["id"] in quarantine:
                continue
            d = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
            complete = (image_model_matches(d, it["id"]) and verdict_matches(
                        qa_ledgers["images"].get(it["id"]), d, it["id"], "images")
                        if phase == "images" else item_certified(d, it["id"], qa_ledgers))
            pending_count += not complete
    names = ("qwen", "matting", "nsfw") if phase == "images" else (
        ("trellis", "decoder", "dino", "nsfw") if phase == "meshes" else tuple(catalog()))
    specs = catalog()
    models = {}
    for name in names:
        failures = model_errors(specs[name])
        models[name] = {"ready": not failures, "errors": failures}
        errors += failures
        if failures and specs[name].get("gated"):
            notes.append(f"Approve https://huggingface.co/{specs[name]['repo_id']}, run hf auth login, "
                         "then meshmaker/local/imgenv/bin/python scripts/local/prepare_models.py --only dino")
    environments = {}
    environments["images"] = probe_environment(IMG_PYTHON, "images")
    if phase in ("all", "meshes"):
        environments["meshes"] = probe_environment(MESH_PYTHON, "meshes")
        if not (LOCAL_DIR / "trellis-mac" / "TRELLIS.2" / "trellis2").is_dir():
            errors.append("Missing TRELLIS.2 source under trellis-mac")
        if not environments["meshes"].get("spz") and gs_format in ("spz", "both"):
            notes.append("Niantic SPZ binding unavailable; generation will retain larger SPLAT files")
            gs_format = "splat"
    for report in environments.values():
        errors += report.get("errors", [])
    free = shutil.disk_usage(nearest_existing(OUTPUT_DIR)).free
    estimated = (int(pending_count * 2_000_000 * 1.2 + 1_000_000_000) if phase == "images"
                 else storage_estimate(pending_count, gs_format, pipeline_type, texture_size)) if pending_count else 0
    if free < estimated:
        errors.append(f"Insufficient disk budget: {free / 1e9:.1f} GB free, "
                      f"approximately {estimated / 1e9:.1f} GB required")
    return {"at": datetime.now(timezone.utc).isoformat(), "ready": not errors,
            "phase": phase, "selection": {"items": len(items), "permitted": len(permitted),
                                          "refused": len(refused), "pending": pending_count},
            "models": models, "environments": environments,
            "disk": {"free_bytes": free, "budget_bytes": estimated, "gs_format": gs_format},
            "output_dir": str(OUTPUT_DIR), "state_dir": str(STATE_DIR),
            "errors": errors, "notes": notes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["all", "images", "meshes"], default="all")
    ap.add_argument("--geographies", nargs="+", choices=list(MANIFESTS))
    ap.add_argument("--ids", nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gs-format", choices=["spz", "splat", "both"], default="spz")
    ap.add_argument("--pipeline-type", choices=["512", "1024", "1024_cascade"], default="512")
    ap.add_argument("--texture-size", choices=[512, 1024, 2048], type=int, default=1024)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true", help="save state/preflight.json")
    content_policy.add_policy_args(ap)
    args = ap.parse_args(argv)
    try:
        report = check(**{k: v for k, v in vars(args).items() if k not in ("json", "write")})
    except ValueError as exc:
        ap.error(str(exc))
    if args.write:
        write_json(STATE_DIR / "preflight.json", report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"preflight: {'READY' if report['ready'] else 'NOT READY'} | "
              f"{report['selection']['permitted']} permitted items | "
              f"{report['disk']['free_bytes'] / 1e9:.1f} GB free | "
              f"{report['disk']['budget_bytes'] / 1e9:.1f} GB budget")
        for message in report["errors"] + report["notes"]:
            print(f"  {message}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
