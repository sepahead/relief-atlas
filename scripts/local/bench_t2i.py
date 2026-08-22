"""Head-to-head T2I benchmark: FLUX.2-klein-4B vs Z-Image-Turbo on MPS.

Generates the same cleaned manifest prompts with both models, times each
image (after one warmup run per model), saves raw outputs side by side under
state/bench_t2i/<model>/<item_id>.raw.png for visual QA, and optionally runs
the offline matting pass to compare matte-ability (fg fraction / emptiness).

Run with: meshmaker/local/imgenv/bin/python scripts/local/bench_t2i.py
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matting import composite_checker, has_object, matte_image  # noqa: E402
from prompt_utils import PROJECT_DIR, clean_prompt  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "state" / "bench_t2i"

# One representative prompt per asset class that matters for the dataset:
# rubble (organic debris), rotorcraft (thin parts), tent (large fabric),
# vehicle (hard surfaces + livery).
BENCH_IDS = [
    "gen_dis_0001",  # earthquake rubble pile
    "ger_uav_0001",  # quadcopter drone
    "gen_med_0001",  # field hospital / triage tent
    "eu_rc_0001",    # Red Cross response vehicle
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["klein", "zimage"],
                    choices=["klein", "zimage"])
    ap.add_argument("--ids", nargs="*", default=None,
                    help="manifest item ids to benchmark (default: 4 picks)")
    ap.add_argument("--klein-steps", type=int, default=18)
    ap.add_argument("--klein-guidance", type=float, default=3.0)
    ap.add_argument("--zimage-steps", type=int, default=9)
    ap.add_argument("--zimage-guidance", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--matt", action="store_true",
                    help="also matte outputs and save checkerboard composites")
    return ap.parse_args()


def resolve_items(ids):
    from prompt_utils import all_items
    by_id = {it["id"]: it for it in all_items()}
    if ids is None:
        ids = BENCH_IDS
    items = []
    for iid in ids:
        if iid not in by_id:
            sys.exit(f"unknown id: {iid}")
        items.append(by_id[iid])
    return items


def free_pipe(pipe):
    del pipe
    gc.collect()
    try:
        import torch
        torch.mps.empty_cache()
    except Exception:
        pass


def run_model(name, items, steps, guidance, size, matt):
    import torch
    from diffusers import Flux2KleinPipeline, ZImagePipeline

    if name == "klein":
        model_dir = PROJECT_DIR / "meshmaker" / "local" / "models" / "FLUX.2-klein-4B"
        pipe = Flux2KleinPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    else:
        model_dir = PROJECT_DIR / "meshmaker" / "local" / "models" / "Z-Image-Turbo"
        pipe = ZImagePipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe.to("mps")

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for it in items:
        iid = it["id"]
        prompt = clean_prompt(it["prompt"], iid)
        seed = 1234  # fixed seed so both models get identical noise budgets
        gen = torch.Generator("cpu").manual_seed(seed)

        # Warmup (MPS kernel compilation etc.) — not timed.
        pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
             width=size, height=size, generator=gen)

        t0 = time.time()
        out = pipe(prompt=prompt, num_inference_steps=steps,
                   guidance_scale=guidance, width=size, height=size,
                   generator=torch.Generator("cpu").manual_seed(seed))
        dt = time.time() - t0
        img = out.images[0]
        img.convert("RGB").save(out_dir / f"{iid}.raw.png")

        entry = {"seconds": round(dt, 1), "steps": steps,
                 "guidance": guidance, "size": size, "seed": seed}
        if matt:
            rgba, stats = matte_image(img)
            entry["matting"] = stats
            entry["matte_ok"] = has_object(rgba)
            composite_checker(rgba).save(out_dir / f"{iid}.debug.png")
        results[iid] = entry
        print(f"[{name}] {iid}: {dt:.1f}s"
              + (f" fg={entry['matting']['fg_fraction']:.2f}" if matt else ""))

    free_pipe(pipe)
    return results


def main():
    args = parse_args()
    items = resolve_items(args.ids)
    for it in items:
        print(f"bench item: {it['id']}")

    all_results = {}
    for name in args.models:
        if name == "klein":
            all_results[name] = run_model(
                name, items, args.klein_steps, args.klein_guidance,
                args.size, args.matt)
        else:
            all_results[name] = run_model(
                name, items, args.zimage_steps, args.zimage_guidance,
                args.size, args.matt)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "items": [it["id"] for it in items],
        "results": all_results,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    for name, res in all_results.items():
        times = [e["seconds"] for e in res.values()]
        print(f"{name}: {sum(times) / len(times):.1f}s/img avg "
              f"({', '.join(f'{t:.0f}s' for t in times)})")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
