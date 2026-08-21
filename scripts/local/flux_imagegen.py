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
from matting import composite_checker, has_object, solid_bg_to_rgba  # noqa: E402
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
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    ap.add_argument("--debug-masks", action="store_true",
                    help="also save a checkerboard composite next to each PNG")
    return ap.parse_args()


def main():
    args = parse_args()

    import torch
    from diffusers import Flux2KleinPipeline

    items = []
    for geo in args.geographies:
        for it in load_manifest(geo):
            it = dict(it)
            it["prompt_clean"] = clean_prompt(it["prompt"], it["id"])
            items.append(it)
    if args.limit:
        items = items[: args.limit]

    todo = []
    for it in items:
        png = OUTPUT_DIR / it["geography"] / it["category"] / it["id"] / f"{it['id']}.png"
        if not png.exists():
            todo.append(it)
    print(f"{len(items)} items, {len(todo)} need images")
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
    for it in todo:
        item_id = it["id"]
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / item_id
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{item_id}.png"

        t0 = time.time()
        seed = args.seed if args.seed is not None else item_seed(item_id)
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
            rgba, stats = solid_bg_to_rgba(rgb)
            if not has_object(rgba):
                raise RuntimeError("matting found no object (empty backdrop mask)")
            rgba.save(png)
            image_meta = {
                "id": item_id,
                "image_model": IMAGE_MODEL_ID,
                "seed": seed,
                "steps": args.steps,
                "guidance": args.guidance,
                "size": args.size,
                "matting": stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            (out_dir / "image_meta.json").write_text(json.dumps(image_meta, indent=2))
            if args.debug_masks:
                composite_checker(rgba).save(out_dir / f"{item_id}.debug.png")
            failures.pop(item_id, None)
            done += 1
            print(f"[{done}/{len(todo)}] {item_id} ok seed={seed} "
                  f"fg={stats['fg_fraction']:.2f} ({time.time() - t0:.1f}s)")
        except Exception as e:
            failures[item_id] = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            print(f"[{done}/{len(todo)}] {item_id} FAILED: {e}")
        finally:
            failures_path.write_text(json.dumps(failures, indent=2))

    print(f"finished: {done} generated, {len(failures)} failures")


if __name__ == "__main__":
    main()
