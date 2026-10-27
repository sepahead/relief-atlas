"""Head-to-head T2I benchmark: Qwen-Image-2.1 vs Z-Image-Turbo on MPS.

Generates the same cleaned manifest prompts with both models, times each
image (after one warmup run per model), saves raw outputs side by side under
state/bench_t2i/<model>/<item_id>.raw.png for visual QA, and optionally runs
the offline matting pass to compare matte-ability (fg fraction / emptiness).

Run with: meshmaker/local/imgenv/bin/python scripts/local/bench_t2i.py
"""

import argparse
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matting import composite_checker, has_object, reference_rgba  # noqa: E402
from prompt_utils import PROJECT_DIR, clean_prompt  # noqa: E402
from pipeline_utils import STATE_DIR, write_json  # noqa: E402

OUTPUT_DIR = STATE_DIR / "bench_t2i"

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
    ap.add_argument("--models", nargs="*", default=["qwen", "zimage"],
                    choices=["qwen", "zimage"])
    ap.add_argument("--ids", nargs="*", default=None,
                    help="manifest item ids to benchmark (default: 4 picks)")
    ap.add_argument("--qwen-steps", type=int, default=40)
    ap.add_argument("--qwen-guidance", type=float, default=1.0)
    ap.add_argument("--zimage-steps", type=int, default=9)
    ap.add_argument("--zimage-guidance", type=float, default=0.0)
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
    from diffusers import QwenImage21Pipeline, ZImagePipeline

    if name == "qwen":
        model_dir = PROJECT_DIR / "meshmaker" / "local" / "models" / "Qwen-Image-2.1"
        pipe = QwenImage21Pipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16,
                                                  local_files_only=True)
    else:
        model_dir = PROJECT_DIR / "meshmaker" / "local" / "models" / "Z-Image-Turbo"
        pipe = ZImagePipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16,
                                             local_files_only=True)
    pipe.to("mps")
    guidance_args = ({"true_cfg_scale": guidance,
                      "negative_prompt": "cropped object, extra objects, blurry image" if guidance > 1 else None}
                     if name == "qwen" else {"guidance_scale": guidance})

    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    warmed_up = False
    for it in items:
        iid = it["id"]
        prompt = clean_prompt(it["prompt"], iid, transparent=name == "qwen")
        seed = 1234  # fixed seed so both models get identical noise budgets
        gen = torch.Generator("cpu").manual_seed(seed)

        # Warmup (MPS kernel compilation etc.) — not timed.
        if not warmed_up:
            with torch.inference_mode():
                pipe(prompt=prompt, num_inference_steps=steps, **guidance_args,
                     width=size, height=size, generator=gen)
            torch.mps.synchronize()
            warmed_up = True

        t0 = time.time()
        with torch.inference_mode():
            out = pipe(prompt=prompt, num_inference_steps=steps,
                       **guidance_args, width=size, height=size,
                       generator=torch.Generator("cpu").manual_seed(seed))
        torch.mps.synchronize()
        dt = time.time() - t0
        img = out.images[0]
        img.convert("RGB").save(out_dir / f"{iid}.raw.png")

        entry = {"seconds": round(dt, 1), "steps": steps,
                 "guidance": guidance, "size": size, "seed": seed}
        if matt:
            rgba, stats = reference_rgba(img)
            entry["matting"] = stats
            entry["matte_ok"] = bool(has_object(rgba))
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
        if name == "qwen":
            all_results[name] = run_model(
                name, items, args.qwen_steps, args.qwen_guidance,
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
    write_json(OUTPUT_DIR / "summary.json", summary)
    print("\n=== summary ===")
    for name, res in all_results.items():
        times = [e["seconds"] for e in res.values()]
        print(f"{name}: {sum(times) / len(times):.1f}s/img avg "
              f"({', '.join(f'{t:.0f}s' for t in times)})")
    print(f"outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
