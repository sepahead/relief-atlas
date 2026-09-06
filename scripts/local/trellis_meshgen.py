"""Phase 2: TRELLIS.2-4B image-to-3D on Apple Silicon (mesh + 3DGS).

Reads the RGBA reference images produced by flux_imagegen.py and exports, per
item: a PBR-textured GLB mesh and a 3D Gaussian Splatting PLY synthesized from
TRELLIS.2's decoded voxel grid. Runs fully offline against local weights.

Run with: meshmaker/local/trellis-mac/.venv/bin/python scripts/local/trellis_meshgen.py
"""

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Extends the macOS GPU watchdog timeout (Metal-debugger side effect), the
# documented mitigation for decoder kernels being killed on Apple Silicon.
os.environ.setdefault("MTL_CAPTURE_ENABLED", "1")

from pathlib import Path  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
TRELLIS_MAC = PROJECT_DIR / "meshmaker" / "local" / "trellis-mac"
MODEL_DIR = PROJECT_DIR / "meshmaker" / "local" / "models" / "TRELLIS.2-4B"

try:
    import flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

sys.path.insert(0, str(TRELLIS_MAC / "TRELLIS.2"))
sys.path.insert(0, str(TRELLIS_MAC))
sys.path.append(str(TRELLIS_MAC / "stubs"))
sys.path.insert(0, str(SCRIPT_DIR))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

# Neutralize the gated RMBG-2.0 background remover before pipeline load.
# All our inputs are RGBA with a real alpha channel, so it is never invoked.
import trellis2.pipelines.rembg as _rembg_mod  # noqa: E402


class _NoRembg:
    def __init__(self, *a, **k):
        pass

    def to(self, device):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        raise RuntimeError("background removal requested but RMBG bypass is active")


_rembg_mod.BiRefNet = _NoRembg

from gs_export import (  # noqa: E402
    export_gaussians_from_mesh_with_voxel, tmp_sibling,
)
import content_policy  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR_NAME, clean_name, clean_prompt, item_seed,
    load_manifest, mesh_complete,
)

OUTPUT_DIR = PROJECT_DIR / OUTPUT_DIR_NAME
STATE_DIR = PROJECT_DIR / "state"
IMAGE_MODEL_ID = "local/flux.2-klein-4b"
MESH_MODEL_ID = "local/trellis.2-4b"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geographies", nargs="*", default=list(MANIFESTS),
                    choices=list(MANIFESTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit item ids (overrides geographies/limit)")
    ap.add_argument("--pipeline-type", default="512",
                    choices=["512", "1024", "1024_cascade"])
    ap.add_argument("--texture-size", type=int, default=1024,
                    choices=[512, 1024, 2048])
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    content_policy.add_policy_args(ap)
    ap.add_argument("--ignore-image-qa", action="store_true",
                    help="mesh images that failed or lack an image QA verdict; "
                         "the resulting meshes are then uncertified")
    return ap.parse_args()


def save_json_atomic(path, data):
    tmp = tmp_sibling(path)
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def export_glb(mesh_out, glb_path, texture_size):
    """Bake PBR textures and export GLB. Mirrors trellis-mac/generate.py."""
    import torch

    use_metal = False
    try:
        import o_voxel.postprocess
        backend = getattr(o_voxel.postprocess, "_BACKEND", None)
        use_metal = backend == "metal" and getattr(o_voxel.postprocess, "_HAS_DR", False)
        if use_metal and not getattr(o_voxel.postprocess, "_HAS_FLEX_GEMM", False):
            import torch.nn.functional as _F_gs

            def _gs3d_fix(feats, coords, shape, grid, mode="trilinear"):
                B, C = shape[0], shape[1]
                D, H, W = shape[2], shape[3], shape[4]
                dense_vol = torch.zeros(B, C, D, H, W, dtype=feats.dtype, device=feats.device)
                dense_vol[coords[:, 0].long(), :, coords[:, 1].long(),
                          coords[:, 2].long(), coords[:, 3].long()] = feats
                grid_norm = torch.stack([
                    grid[..., 2] / (W - 1) * 2 - 1,
                    grid[..., 1] / (H - 1) * 2 - 1,
                    grid[..., 0] / (D - 1) * 2 - 1,
                ], dim=-1).reshape(B, 1, 1, -1, 3)
                sampled = _F_gs.grid_sample(dense_vol, grid_norm, mode="bilinear",
                                            align_corners=True, padding_mode="border")
                M = grid.shape[1]
                return sampled.reshape(B, C, M).permute(0, 2, 1).reshape(B * M, C)

            o_voxel.postprocess._grid_sample_3d = _gs3d_fix
    except (ImportError, AttributeError):
        use_metal = False

    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()

    if use_metal:
        try:
            import fast_simplification
            import o_voxel

            target_faces = min(200000, len(faces))
            if len(faces) > target_faces:
                ratio = 1.0 - (target_faces / len(faces))
                sv, sf = fast_simplification.simplify(verts, faces, ratio)
                simp_v = torch.from_numpy(sv).float()
                simp_f = torch.from_numpy(sf.astype("int32"))
            else:
                simp_v = mesh_out.vertices.cpu().float()
                simp_f = mesh_out.faces.cpu()

            glb = o_voxel.postprocess.to_glb(
                vertices=simp_v, faces=simp_f,
                attr_volume=mesh_out.attrs.cpu(),
                coords=mesh_out.coords.cpu(),
                attr_layout=mesh_out.layout,
                voxel_size=mesh_out.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target_faces,
                texture_size=texture_size,
            )
            # file_type is explicit: glb_path is a transient sibling name and
            # must never rely on extension sniffing.
            glb.export(str(glb_path), file_type="glb")
            return "metal"
        except RuntimeError as e:
            print(f"    metal bake failed ({e}); falling back to KDTree baker")

    # KDTree fallback baker from trellis-mac/backends
    from backends.texture_baker import bake_texture, export_glb_with_texture, uv_unwrap

    target_faces = min(200000, len(faces))
    bake_v, bake_f = verts, faces
    if len(faces) > target_faces:
        import fast_simplification
        ratio = 1.0 - (target_faces / len(faces))
        bake_v, bake_f = fast_simplification.simplify(verts, faces, ratio)

    new_verts, new_faces, uvs, _ = uv_unwrap(bake_v, bake_f)
    base_color_img, mr_img, _mask = bake_texture(
        new_verts, new_faces, uvs,
        mesh_out.coords.cpu().float().numpy(), mesh_out.attrs.cpu().float().numpy(),
        mesh_out.origin.cpu().float().numpy(), mesh_out.voxel_size,
        texture_size=texture_size,
    )
    export_glb_with_texture(new_verts, new_faces, uvs, base_color_img, mr_img, str(glb_path))
    return "kdtree"


def main():
    args = parse_args()

    items = []
    if args.ids:
        by_id = {it["id"]: it for geo in MANIFESTS for it in load_manifest(geo)}
        missing = [i for i in args.ids if i not in by_id]
        if missing:
            print(f"unknown ids: {missing}")
            return
        items = [dict(by_id[i]) for i in args.ids]
        for it in items:
            it["prompt_clean"] = clean_prompt(it["prompt"], it["id"])
    else:
        for geo in args.geographies:
            for it in load_manifest(geo):
                it = dict(it)
                it["prompt_clean"] = clean_prompt(it["prompt"], it["id"])
                items.append(it)
        # Tier selection precedes --limit; see flux_imagegen.py for why.
        items = content_policy.select_tier(items, args.only_policy)
        if args.limit:
            items = items[: args.limit]

    retries = {}
    retries_path = STATE_DIR / "qa_retries.json"
    if retries_path.exists():
        retries = json.loads(retries_path.read_text())

    def load_state(name):
        path = STATE_DIR / name
        return json.loads(path.read_text()) if path.exists() else {}

    # The mesh phase used to gate only on "does a PNG exist", which made the
    # image QA gate purely advisory: a QA-failed or NSFW-flagged frame would
    # still be turned into a shipped asset. Require a passing verdict.
    image_qa = load_state("qa_images.json")
    quarantine = load_state("qa_quarantine.json")

    # Content policy: refuse blocked subjects even if an image somehow exists.
    items, rejected, overrides = content_policy.screen_from_args(
        items, args, phase="mesh_gen")
    print(content_policy.describe_screen(items, rejected, overrides))

    todo = []
    n_unverified = n_quarantined = 0
    for it in items:
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        png = out_dir / f"{it['id']}.png"
        if not png.exists():
            continue  # image phase hasn't produced it yet
        if it["id"] in quarantine:
            n_quarantined += 1
            continue
        if not args.ignore_image_qa and not image_qa.get(it["id"], {}).get("pass"):
            # No verdict is treated exactly like a failed verdict: an
            # unscreened image must not become a shipped asset.
            n_unverified += 1
            continue
        if not mesh_complete(out_dir, it["id"]):
            todo.append(it)
    print(f"{len(items)} items, {len(todo)} need meshes"
          + (f", {n_unverified} blocked by image QA" if n_unverified else "")
          + (f", {n_quarantined} quarantined" if n_quarantined else ""))
    if n_unverified and not args.ignore_image_qa:
        print("  run qa.py --mode images to certify them, or pass "
              "--ignore-image-qa to bypass (uncertified)")
    if not todo:
        return

    import torch
    from PIL import Image as PILImage
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    print(f"Loading pipeline from {MODEL_DIR} ...")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_DIR))
    pipeline.to(torch.device("mps"))
    print(f"Loaded in {time.time() - t0:.0f}s, device=mps")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    failures_path = STATE_DIR / "local_mesh_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}

    done = 0
    # Meshes actually written under a policy override; see flux_imagegen.py.
    generated_overrides = []
    for it in todo:
        item_id = it["id"]
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / item_id
        glb_path = out_dir / f"{item_id}.glb"
        png_path = out_dir / f"{item_id}.png"

        t0 = time.time()
        seed = args.seed if args.seed is not None else (
            item_seed(item_id) + 7919 * retries.get(item_id, 0))
        try:
            img = PILImage.open(png_path)
            outputs = pipeline.run(img, seed=seed, pipeline_type=args.pipeline_type)
            mesh_out = outputs[0] if isinstance(outputs, list) else outputs

            verts = mesh_out.vertices.cpu().numpy()
            if len(verts) == 0:
                raise RuntimeError("empty mesh (possible GPU watchdog kill)")

            t_gen = time.time() - t0
            # Export to transient sibling names and atomically promote, so a
            # crash can never leave a truncated GLB/3DGS that resume logic
            # accepts. tmp_sibling keeps the real suffix, which the exporters
            # need in order to pick the right container.
            glb_tmp = tmp_sibling(glb_path)
            bake_backend = export_glb(mesh_out, glb_tmp, args.texture_size)
            # Condensed 3DGS: emits BOTH containers when the spz package is
            # installed -- .spz (~10x smaller than PLY, preferred) plus .splat
            # (32 B/splat, universal web fallback). Without spz, .splat only.
            # The raw float32 PLY is never kept.
            gs_paths, n_gaussians = export_gaussians_from_mesh_with_voxel(
                mesh_out, str(out_dir / item_id))
            if not n_gaussians:
                raise RuntimeError("3DGS export produced no gaussians")
            # Promote only once the GLB and every 3DGS container are on disk.
            glb_tmp.replace(glb_path)
            for gs_final in gs_paths:
                tmp_sibling(gs_final).replace(gs_final)
            t_total = time.time() - t0

            metadata = {
                "id": item_id,
                "category": it["category"],
                "geography": it["geography"],
                "name": clean_name(it.get("name", "")),
                "prompt": it["prompt_clean"],
                "policy": it["policy"],
                "image_qa": {k: image_qa.get(item_id, {}).get(k)
                             for k in ("pass", "nsfw", "sharpness")},
                "image_model": IMAGE_MODEL_ID,
                "mesh_model": MESH_MODEL_ID,
                "seed": seed,
                "pipeline_type": args.pipeline_type,
                "texture_size": args.texture_size,
                "bake_backend": bake_backend,
                "vertices": int(len(verts)),
                "triangles": int(len(mesh_out.faces)),
                "gaussians": int(n_gaussians),
                "gs_format": "+".join(
                    p.suffix.lstrip(".") for p in gs_paths),
                "generation_seconds": round(t_gen, 1),
                "total_seconds": round(t_total, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            image_meta_path = out_dir / "image_meta.json"
            if image_meta_path.exists():
                metadata["image_meta"] = json.loads(image_meta_path.read_text())
            save_json_atomic(out_dir / "metadata.json", metadata)
            failures.pop(item_id, None)
            done += 1
            if it["policy"]["override"]:
                generated_overrides.append(it)
            print(f"[{done}/{len(todo)}] {item_id}: {len(verts):,} verts, "
                  f"{n_gaussians:,} gaussians, {t_total:.0f}s ({bake_backend})")
        except Exception as e:
            failures[item_id] = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            print(f"[{done}/{len(todo)}] {item_id} FAILED: {e}")
            # Drop half-written transients so a 10K run cannot accumulate them.
            for stale in out_dir.glob(f".{item_id}.tmp.*"):
                stale.unlink(missing_ok=True)
        finally:
            save_json_atomic(failures_path, failures)

    if generated_overrides:
        path = content_policy.record_overrides(generated_overrides,
                                              phase="mesh_gen")
        print(f"** {len(generated_overrides)} mesh(es) generated under a policy "
              f"override; logged to {path}")

    print(f"finished: {done} meshes, {len(failures)} failures")


if __name__ == "__main__":
    main()
