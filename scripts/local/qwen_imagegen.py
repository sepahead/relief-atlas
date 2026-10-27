"""Phase 1: generate transparent references with official Qwen-Image-2.1.

Runs on Apple Silicon MPS via diffusers' QwenImage21Pipeline, fully local.
Output per item: outputs_relief/<geo>/<category>/<item_id>/<item_id>.png
(RGBA: native alpha when present, matting fallback for opaque images) plus
image_meta.json with the prompt, seed, provenance and alpha statistics.

Run with: meshmaker/local/imgenv/bin/python scripts/local/qwen_imagegen.py
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_policy  # noqa: E402
from generation_prompts import RGBA_PREFIX, digest, is_optimized_prompt  # noqa: E402
from matting import composite_checker, has_object, reference_rgba  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR, PROJECT_DIR, STATE_DIR, clean_prompt, item_seed,
    select_manifest_items,
)
from pipeline_utils import (  # noqa: E402
    IMAGE_MODEL_ID, durable_replace, file_sha256, generation_provenance, image_model_matches, write_json,
    is_infrastructure_error, load_json, source_image_metadata, sync_directory, gpu_lock,
)

MODEL_DIR = PROJECT_DIR / "meshmaker" / "local" / "models" / "Qwen-Image-2.1"


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geographies", nargs="*", default=list(MANIFESTS),
                    choices=list(MANIFESTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit item ids (overrides geographies/limit)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--width", type=int, help="reference width; defaults to --size")
    ap.add_argument("--height", type=int, help="reference height; defaults to --size")
    reference = ap.add_mutually_exclusive_group()
    reference.add_argument("--reframe-reference", type=Path,
                    help="use native Qwen editing to reframe one verified reference into a separate output")
    reference.add_argument("--structure-reference", type=Path,
                    help="use one hash-bound eight-boom cargo guide and adjacent JSON in a separate output")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    ap.add_argument("--debug-masks", action="store_true",
                    help="also save a checkerboard composite next to each PNG")
    content_policy.add_policy_args(ap)
    args = ap.parse_args(argv)
    args.width = args.size if args.width is None else args.width
    args.height = args.size if args.height is None else args.height
    if args.steps < 1 or (args.limit is not None and args.limit < 1):
        ap.error("steps and limit must be positive")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        ap.error("guidance must be finite and nonnegative")
    if any(value < 256 or value % 16 for value in (args.size, args.width, args.height)):
        ap.error("image dimensions must be at least 256 and divisible by 16")
    if (args.reframe_reference or args.structure_reference) and (not args.ids or len(args.ids) != 1):
        ap.error("reference editing requires exactly one explicit --ids item")
    return args


def reframe_provenance(source, item_id, destination, *, override=False):
    """Bind an edit to this item's actual reference without overwriting it."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Reframing needs a separate output directory; preserve the source reference")
    metadata = source_image_metadata(source.parent, item_id)
    if (source.name != f"{item_id}.png" or metadata.get("id") != item_id
            or metadata.get("image_model") != IMAGE_MODEL_ID):
        raise ValueError("Reframing requires this item's hash-verified configured Qwen reference")
    policy = metadata.get("policy")
    if not isinstance(policy, dict) or "override" not in policy:
        raise ValueError("Source reference has no recorded policy provenance")
    if policy["override"] and not override:
        raise ValueError("An override reference cannot become a default-policy edit")
    result = {"mode": "native_reference_reframe", "edit_source": str(source.resolve()),
              "edit_source_image_sha256": file_sha256(source),
              "edit_source_metadata_sha256": file_sha256(source.parent / "image_meta.json")}
    return validate_edit_destination(source, item_id, destination, result)


def validate_edit_destination(source, item_id, destination, result):
    """Share source-bound interrupted-commit recovery between native edit modes."""
    intent = load_json(destination.parent / ".reference_edit.json")
    if intent and intent != result:
        raise ValueError("Output contains another pending edit; choose a separate output directory")
    if destination.exists():
        existing = source_image_metadata(destination.parent, item_id)
        committed_edit = (existing.get("image_model") == IMAGE_MODEL_ID and existing.get("mode") == result["mode"]
                and existing.get("edit_source_image_sha256") == result["edit_source_image_sha256"]
                and existing.get("edit_source_metadata_sha256") == result["edit_source_metadata_sha256"])
        # The intent is committed before inference and already binds the exact
        # source pixels and sidecar. A previous, interrupted QA cleanup can
        # leave stale metadata beside the new PNG; it must not strand an edit
        # that also stopped before its new metadata commit.
        interrupted_edit = intent == result
        if not (committed_edit or interrupted_edit):
            raise ValueError("Output already contains another reference; choose a separate output directory")
    return result


def structure_provenance(source, item, destination, *, override=False):
    """An authored guide is recorded as a guide, never as an earlier Qwen image."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("A structure guide requires a separate output; preserve the source")
    sidecar = source.with_suffix(".json")
    record = load_json(sidecar)
    if (record.get("kind") != "authored_geometry_guide" or record.get("schema_version") != 1
            or record.get("id") != item["id"] or record.get("image_sha256") != file_sha256(source)
            or record.get("manifest_prompt_sha256") != digest(item["prompt"])):
        raise ValueError("Structure guide must be hash-bound to this item and its current manifest prompt")
    expected = item.get("generation_prompt", {}).get("specifications", {}).get("prompt", {})
    if (expected.get("profile") != "cargo_flat_octocopter" or record.get("specification") != expected
            or len(record.get("motors", [])) != 8):
        raise ValueError("Native structure editing currently requires this item's eight-boom cargo guide")
    policy = record.get("policy")
    if not isinstance(policy, dict) or "override" not in policy:
        raise ValueError("Structure guide has no recorded policy provenance")
    if policy["override"] and not override:
        raise ValueError("An override guide cannot become a default-policy edit")
    result = {"mode": "native_structure_reference", "edit_source_kind": "authored_geometry_guide",
              "edit_source": str(source.resolve()), "edit_source_image_sha256": file_sha256(source),
              "edit_source_metadata_sha256": file_sha256(sidecar)}
    return validate_edit_destination(source, item["id"], destination, result)


def edit_source_provenance(args, item, destination):
    if args.reframe_reference:
        return reframe_provenance(args.reframe_reference, item["id"], destination, override=item["policy"]["override"])
    return structure_provenance(args.structure_reference, item, destination, override=item["policy"]["override"])


def inference_inputs(args, prompt):
    """Use Qwen's native image input for edits, retaining source RGBA pixels."""
    inputs = {"prompt": prompt, "num_inference_steps": args.steps,
              "true_cfg_scale": args.guidance,
              "negative_prompt": ("cropped object, extra objects, blurry image, text overlays, watermark"
                                  if args.guidance > 1 else None),
              "width": args.width, "height": args.height}
    if args.reframe_reference:
        from PIL import Image
        with Image.open(args.reframe_reference) as reference:
            inputs["image"] = reference.convert("RGBA")
        subject = prompt.removeprefix(RGBA_PREFIX).splitlines()[0].removeprefix("Subject: ")
        inputs["prompt"] = (
            "This is an RGBA image with transparency. Reframe the provided reference photograph. "
            "This is framing only: preserve the source subject, component positions, pose, "
            "viewing direction, payload, materials, proportions and markings. Do not add or remove components. "
            "Pull the camera back so the complete subject and every outer tip fit inside the central "
            "70 percent of the canvas, with generous empty space on every side. Complete cropped "
            "outer tips naturally. Keep thin parts sharp and fully visible. "
            "The image has alpha channel and the background is transparent. "
            "Even diffuse studio lighting. No floor, cast shadow, additional objects or text overlays. "
            "Reference subject: " + subject)
    elif args.structure_reference:
        from PIL import Image
        with Image.open(args.structure_reference) as reference:
            inputs["image"] = reference.convert("RGBA")
        subject = prompt.removeprefix(RGBA_PREFIX).splitlines()[0].removeprefix("Subject: ")
        surface = next((line for line in prompt.splitlines() if line.startswith("Surface: ")), "")
        inputs["prompt"] = (
            RGBA_PREFIX + "Turn the provided authored geometry guide into a realistic product reference photograph. "
            "Preserve its complete structure, component count, component positions and camera view. "
            "Keep every separate boom, motor housing, propeller hub and rigid stationary blade in place. "
            "Render coherent carbon-composite arms, metal motor housings and the closed cargo pod, with clean "
            "material boundaries and realistic local surface detail. Keep the landing supports separate from the booms. "
            "The whole assembly remains uncropped with generous empty margin on every side. "
            "Even diffuse neutral lighting, deep focus, crisp edges. "
            "The image has alpha channel and the background is transparent. "
            "Reference subject: " + subject + " " + surface)
    return inputs


def load_qa_retries():
    """QA retry counters: each QA rejection bumps a per-item retry count that
    perturbs the deterministic seed so regeneration explores a new sample."""
    path = STATE_DIR / "qa_retries.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_quarantine():
    """Items QA gave up on. Without this the QA/regenerate loop is infinite:
    QA deletes the rejected PNG, this phase regenerates it, QA rejects it
    again."""
    path = STATE_DIR / "qa_quarantine.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_json_atomic(path: Path, data):
    write_json(path, data)


def main():
    args = parse_args()

    items = select_manifest_items(args.geographies, args.ids, args.limit, args.only_policy)

    retries = load_qa_retries()
    quarantine = load_quarantine()

    # Content policy first: refuse blocked subjects before spending any
    # compute, and before a weapon prompt ever reaches the image model.
    items, rejected, overrides = content_policy.screen_from_args(
        items, args, phase="image_gen")
    print(content_policy.describe_screen(items, rejected, overrides))

    todo = []
    edit_provenance = {}
    n_quarantined = 0
    for it in items:
        if "prompt_clean" not in it:
            it["prompt_clean"] = clean_prompt(it["prompt"], it["id"], transparent=True)
        if it["id"] in quarantine:
            n_quarantined += 1
            continue
        png = OUTPUT_DIR / it["geography"] / it["category"] / it["id"] / f"{it['id']}.png"
        if args.reframe_reference or args.structure_reference:
            edit_provenance = edit_source_provenance(args, it, png)
        if not image_model_matches(png.parent, it["id"]):
            todo.append(it)
    print(f"{len(items)} items, {len(todo)} need images"
          + (f", {n_quarantined} quarantined by QA" if n_quarantined else ""))
    if not todo:
        return

    import torch
    from diffusers import QwenImage21Pipeline

    print(f"Loading {MODEL_DIR.name} ...")
    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, local_files_only=True)
    pipe.to("mps")
    print(f"Loaded in {time.time() - t0:.0f}s, device=mps")
    provenance = generation_provenance("qwen", "matting")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    failures_path = STATE_DIR / "local_image_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}

    done = 0
    # Items whose PNG we actually wrote under a policy override. Recorded after
    # the loop so the ledger names produced assets, not merely permitted ones.
    generated_overrides = []
    for it in todo:
        item_id = it["id"]
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / item_id
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{item_id}.png"

        t0 = time.time()
        seed = args.seed if args.seed is not None else (
            item_seed(item_id) + 7919 * retries.get(item_id, 0))
        try:
            if args.reframe_reference or args.structure_reference:
                # Loading the model can take minutes; recheck the source before
                # binding its current pixels to the edit's provenance.
                edit_provenance = edit_source_provenance(args, it, png)
                write_json(out_dir / ".reference_edit.json", edit_provenance)
            inputs = inference_inputs(args, it["prompt_clean"])
            with torch.inference_mode():
                gen = pipe(**inputs, generator=torch.Generator("cpu").manual_seed(seed))
            rgb = gen.images[0]
            # Keep the pre-matte frame: matting can be re-tuned and re-run
            # from this file without paying generation cost again.
            rgb.convert("RGB").save(out_dir / f"{item_id}.raw.jpg", quality=95)
            rgba, stats = reference_rgba(rgb)
            if not has_object(rgba):
                raise RuntimeError("matting found no object (empty backdrop mask)")
            png_tmp = png.with_name(f".{item_id}.tmp.png")
            rgba.save(png_tmp, format="PNG")
            durable_replace(png_tmp, png)
            image_meta = {
                "id": item_id,
                "image_model": IMAGE_MODEL_ID,
                "prompt": inputs["prompt"],
                "manifest_prompt": it["prompt_clean"],
                "manifest_prompt_sha256": digest(it["prompt"]),
                "prompt_recipe_version": it.get("generation_prompt", {}).get("version") if is_optimized_prompt(it["prompt"]) else None,
                "mode": "text_to_image",
                **edit_provenance,
                "provenance": provenance,
                "seed": seed,
                "steps": args.steps,
                "guidance": args.guidance,
                "size": max(rgba.size),
                "width": rgba.width,
                "height": rgba.height,
                "matting": stats,
                "policy": it["policy"],
                "image_sha256": file_sha256(png),
                "generation_seconds": round(time.time() - t0, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_json_atomic(out_dir / "image_meta.json", image_meta)
            if args.reframe_reference or args.structure_reference:
                (out_dir / ".reference_edit.json").unlink(missing_ok=True)
                sync_directory(out_dir)
            if args.debug_masks:
                composite_checker(rgba).save(out_dir / f"{item_id}.debug.png")
            failures.pop(item_id, None)
            done += 1
            if it["policy"]["override"]:
                generated_overrides.append(it)
            print(f"[{done}/{len(todo)}] {item_id} ok seed={seed} "
                  f"fg={stats['fg_fraction']:.2f} ({time.time() - t0:.1f}s)")
        except Exception as e:
            for stale in out_dir.glob(f".{item_id}.tmp.*"):
                stale.unlink(missing_ok=True)
            failures[item_id] = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            print(f"[{done}/{len(todo)}] {item_id} FAILED: {e}")
            if is_infrastructure_error(e):
                if generated_overrides:
                    content_policy.record_overrides(generated_overrides, phase="image_gen")
                raise RuntimeError(f"Image generation infrastructure failed for {item_id}; assets retained") from e
        finally:
            save_json_atomic(failures_path, failures)

    if generated_overrides:
        path = content_policy.record_overrides(generated_overrides,
                                              phase="image_gen")
        print(f"** {len(generated_overrides)} image(s) generated under a policy "
              f"override; logged to {path}")

    print(f"finished: {done} generated, {len(failures)} failures")


if __name__ == "__main__":
    with gpu_lock():
        main()
