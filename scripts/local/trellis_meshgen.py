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

from gs_export import export_gaussians_from_mesh_with_voxel  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR_NAME, clean_prompt, item_seed, load_manifest,
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
    ap.add_argument("--pipeline-type", default="512",
                    choices=["512", "1024", "1024_cascade"])
    ap.add_argument("--texture-size", type=int, default=1024,
                    choices=[512, 1024, 2048])
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    return ap.parse_args()


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
            glb.export(str(glb_path))
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
    for geo in args.geographies:
        for it in load_manifest(geo):
            it = dict(it)
            it["prompt_clean"] = clean_prompt(it["prompt"], it["id"])
            items.append(it)
    if args.limit:
        items = items[: args.limit]

    todo = []
    for it in items:
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        glb = out_dir / f"{it['id']}.glb"
        png = out_dir / f"{it['id']}.png"
        if not png.exists():
            continue  # image phase hasn't produced it yet
        complete = (
            glb.exists() and glb.stat().st_size > 1024
            and (out_dir / f"{it['id']}.ply").exists()
            and (out_dir / "metadata.json").exists()
        )
        if not complete:
            todo.append(it)
    print(f"{len(items)} items, {len(todo)} need meshes")
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
    for it in todo:
        item_id = it["id"]
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / item_id
        glb_path = out_dir / f"{item_id}.glb"
        ply_path = out_dir / f"{item_id}.ply"
        png_path = out_dir / f"{item_id}.png"

        t0 = time.time()
        seed = args.seed if args.seed is not None else item_seed(item_id)
        try:
            img = PILImage.open(png_path)
            outputs = pipeline.run(img, seed=seed, pipeline_type=args.pipeline_type)
            mesh_out = outputs[0] if isinstance(outputs, list) else outputs

            verts = mesh_out.vertices.cpu().numpy()
            if len(verts) == 0:
                raise RuntimeError("empty mesh (possible GPU watchdog kill)")

            t_gen = time.time() - t0
            bake_backend = export_glb(mesh_out, glb_path, args.texture_size)
            n_gaussians = export_gaussians_from_mesh_with_voxel(mesh_out, str(ply_path))
            t_total = time.time() - t0

            metadata = {
                "id": item_id,
                "category": it["category"],
                "geography": it["geography"],
                "name": it.get("name", ""),
                "prompt": it["prompt_clean"],
                "image_model": IMAGE_MODEL_ID,
                "mesh_model": MESH_MODEL_ID,
                "seed": seed,
                "pipeline_type": args.pipeline_type,
                "texture_size": args.texture_size,
                "bake_backend": bake_backend,
                "vertices": int(len(verts)),
                "triangles": int(len(mesh_out.faces)),
                "gaussians": int(n_gaussians),
                "generation_seconds": round(t_gen, 1),
                "total_seconds": round(t_total, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            image_meta_path = out_dir / "image_meta.json"
            if image_meta_path.exists():
                metadata["image_meta"] = json.loads(image_meta_path.read_text())
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            failures.pop(item_id, None)
            done += 1
            print(f"[{done}/{len(todo)}] {item_id}: {len(verts):,} verts, "
                  f"{n_gaussians:,} gaussians, {t_total:.0f}s ({bake_backend})")
        except Exception as e:
            failures[item_id] = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            print(f"[{done}/{len(todo)}] {item_id} FAILED: {e}")
        finally:
            failures_path.write_text(json.dumps(failures, indent=2))

    print(f"finished: {done} meshes, {len(failures)} failures")


if __name__ == "__main__":
    main()
