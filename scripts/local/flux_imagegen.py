"""Phase 1: generate solid-backdrop reference images with FLUX.2 [klein] 4B.

Runs on Apple Silicon MPS via diffusers' Flux2KleinPipeline, fully local.
Output per item: outputs_relief/<geo>/<category>/<item_id>/<item_id>.png
(RGBA: uniform magenta studio backdrop matted away, ready to feed TRELLIS.2
without its gated background remover) plus image_meta.json with the seed and
matting stats for dataset QA.

Run with: meshmaker/local/imgenv/bin/python scripts/local/flux_imagegen.py
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_policy  # noqa: E402
from matting import composite_checker, has_object, matte_image  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, PROJECT_DIR, clean_prompt, item_seed, load_manifest,
)

MODEL_DIR = PROJECT_DIR / "meshmaker" / "local" / "models" / "FLUX.2-klein-4B"
OUTPUT_DIR = PROJECT_DIR / "outputs_relief"
STATE_DIR = PROJECT_DIR / "state"

IMAGE_MODEL_ID = "local/flux.2-klein-4b"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geographies", nargs="*", default=list(MANIFESTS),
                    choices=list(MANIFESTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit item ids (overrides geographies/limit)")
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    ap.add_argument("--debug-masks", action="store_true",
                    help="also save a checkerboard composite next to each PNG")
    content_policy.add_policy_args(ap)
    return ap.parse_args()


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    args = parse_args()

    import torch
    from diffusers import Flux2KleinPipeline

    items = []
    if args.ids:
        by_id = {it["id"]: it for geo in MANIFESTS for it in load_manifest(geo)}
        missing = [i for i in args.ids if i not in by_id]
        if missing:
            sys.exit(f"unknown ids: {missing}")
        items = [dict(by_id[i]) for i in args.ids]
    else:
        for geo in args.geographies:
            for it in load_manifest(geo):
                it = dict(it)
                it["prompt_clean"] = clean_prompt(it["prompt"], it["id"])
                items.append(it)
        # Tier selection precedes --limit: "--only-policy blocked --limit 10"
        # must mean ten blocked items, not "the blocked ones among the first
        # ten manifest rows", which is almost always none of them.
        items = content_policy.select_tier(items, args.only_policy)
        if args.limit:
            items = items[: args.limit]

    retries = load_qa_retries()
    quarantine = load_quarantine()

    # Content policy first: refuse blocked subjects before spending any
    # compute, and before a weapon prompt ever reaches the image model.
    items, rejected, overrides = content_policy.screen_from_args(
        items, args, phase="image_gen")
    print(content_policy.describe_screen(items, rejected, overrides))

    todo = []
    n_quarantined = 0
    for it in items:
        if "prompt_clean" not in it:
            it["prompt_clean"] = clean_prompt(it["prompt"], it["id"])
        if it["id"] in quarantine:
            n_quarantined += 1
            continue
        png = OUTPUT_DIR / it["geography"] / it["category"] / it["id"] / f"{it['id']}.png"
        if not png.exists():
            todo.append(it)
    print(f"{len(items)} items, {len(todo)} need images"
          + (f", {n_quarantined} quarantined by QA" if n_quarantined else ""))
    if not todo:
        return

    print(f"Loading {MODEL_DIR.name} ...")
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(str(MODEL_DIR), torch_dtype=torch.bfloat16)
    pipe.to("mps")
    print(f"Loaded in {time.time() - t0:.0f}s, device=mps")

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
            gen = pipe(
                prompt=it["prompt_clean"],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.size,
                height=args.size,
                generator=torch.Generator("cpu").manual_seed(seed),
            )
            rgb = gen.images[0]
            # Keep the pre-matte frame: matting can be re-tuned and re-run
            # from this file without paying generation cost again.
            rgb.convert("RGB").save(out_dir / f"{item_id}.raw.jpg", quality=95)
            rgba, stats = matte_image(rgb)
            if not has_object(rgba):
                raise RuntimeError("matting found no object (empty backdrop mask)")
            png_tmp = png.with_name(png.stem + ".tmp.png")
            rgba.save(png_tmp, format="PNG")
            png_tmp.replace(png)
            image_meta = {
                "id": item_id,
                "image_model": IMAGE_MODEL_ID,
                "seed": seed,
                "steps": args.steps,
                "guidance": args.guidance,
                "size": args.size,
                "matting": stats,
                "policy": it["policy"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_json_atomic(out_dir / "image_meta.json", image_meta)
            if args.debug_masks:
                composite_checker(rgba).save(out_dir / f"{item_id}.debug.png")
            failures.pop(item_id, None)
            done += 1
            if it["policy"]["override"]:
                generated_overrides.append(it)
            print(f"[{done}/{len(todo)}] {item_id} ok seed={seed} "
                  f"fg={stats['fg_fraction']:.2f} ({time.time() - t0:.1f}s)")
        except Exception as e:
            failures[item_id] = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            print(f"[{done}/{len(todo)}] {item_id} FAILED: {e}")
        finally:
            save_json_atomic(failures_path, failures)

    if generated_overrides:
        path = content_policy.record_overrides(generated_overrides,
                                              phase="image_gen")
        print(f"** {len(generated_overrides)} image(s) generated under a policy "
              f"override; logged to {path}")

    print(f"finished: {done} generated, {len(failures)} failures")


if __name__ == "__main__":
    main()
