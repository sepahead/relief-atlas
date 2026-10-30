"""Phase 2: TRELLIS.2-4B image-to-3D on Apple Silicon (mesh + 3DGS).

Reads the RGBA reference images produced by qwen_imagegen.py and exports, per
item: a PBR-textured GLB mesh and compact 3D Gaussian splats synthesized from
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
    export_gaussians_from_mesh_with_voxel, gltf_vertices, tmp_sibling,
)
import content_policy  # noqa: E402
from generation_prompts import is_optimized_prompt  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR, STATE_DIR, clean_name, clean_prompt, item_seed,
    mesh_complete, select_manifest_items,
)
from pipeline_utils import (  # noqa: E402
    GS_EXTENSIONS, GS_COLOR_ENCODINGS, file_sha256, generation_provenance, image_model_matches, source_image_matches,
    durable_replace, source_image_metadata, verdict_matches, write_json,
    is_infrastructure_error, gpu_lock,
)
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
    ap.add_argument("--mesh-steps", type=int, default=None,
                    help="override sampling steps for structure, shape and texture")
    ap.add_argument("--target-faces", type=int, default=200000,
                    help="triangle target before texture baking")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    ap.add_argument("--gs-format", choices=["spz", "splat", "both"], default="spz",
                    help="Gaussian containers to retain; SPZ falls back to SPLAT if unavailable")
    content_policy.add_policy_args(ap)
    ap.add_argument("--ignore-image-qa", action="store_true",
                    help="mesh images that failed or lack an image QA verdict; "
                         "the resulting meshes are then uncertified")
    args = ap.parse_args()
    if args.target_faces < 1 or (args.mesh_steps is not None and args.mesh_steps < 1):
        ap.error("mesh steps and target faces must be positive")
    return args


def save_json_atomic(path, data):
    write_json(path, data)


def export_glb(mesh_out, glb_path, texture_size, target_faces=200000):
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
            import o_voxel

            target_faces = min(target_faces, len(faces))
            # to_glb builds a BVH of this mesh before its own simplification,
            # then projects baked texels back to that original surface. An
            # external pre-decimation discarded that reference and sampled
            # attributes at positions displaced from the decoded surface.
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_out.vertices.cpu().float(), faces=mesh_out.faces.cpu(),
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
            if is_infrastructure_error(e):
                raise
            print(f"    metal bake failed ({e}); falling back to KDTree baker")

    # KDTree fallback baker from trellis-mac/backends
    from backends.texture_baker import bake_texture, export_glb_with_texture, uv_unwrap

    target_faces = min(target_faces, len(faces))
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
    export_glb_with_texture(gltf_vertices(new_verts), new_faces, uvs,
                            base_color_img, mr_img, str(glb_path))
    return "kdtree"


def main():
    args = parse_args()
    items = select_manifest_items(args.geographies, args.ids, args.limit, args.only_policy)
    for it in items:
        it["prompt_clean"] = clean_prompt(it["prompt"], it["id"], transparent=True)

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
        if not args.ignore_image_qa and (not image_model_matches(out_dir, it["id"])
                or not verdict_matches(image_qa.get(it["id"]), out_dir, it["id"], "images")):
            # No verdict is treated exactly like a failed verdict: an
            # unscreened image must not become a shipped asset.
            n_unverified += 1
            continue
        if not (mesh_complete(out_dir, it["id"]) and source_image_matches(out_dir, it["id"])):
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
    provenance = generation_provenance("trellis", "decoder", "dino")
    sampling = {name: {**getattr(pipeline, f"{name}_sampler_params"),
                       **({"steps": args.mesh_steps} if args.mesh_steps is not None else {})}
                for name in ("sparse_structure", "shape_slat", "tex_slat")}

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    failures_path = STATE_DIR / "local_mesh_failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}

    done = 0
    # Meshes actually written under a policy override; see qwen_imagegen.py.
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
            image_meta = source_image_metadata(out_dir, item_id)
            with PILImage.open(png_path) as img, torch.inference_mode():
                outputs = pipeline.run(img, seed=seed, pipeline_type=args.pipeline_type,
                    **{f"{name}_sampler_params": params for name, params in sampling.items()})
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
            bake_backend = export_glb(mesh_out, glb_tmp, args.texture_size, args.target_faces)
            # Retain only the requested containers. Compact SPZ is the default;
            # without the Niantic binding the exporter falls back to SPLAT.
            gs_paths, n_gaussians = export_gaussians_from_mesh_with_voxel(
                mesh_out, str(out_dir / item_id), gs_format=args.gs_format)
            if not n_gaussians:
                raise RuntimeError("3DGS export produced no gaussians")
            # Promote only once the GLB and every 3DGS container are on disk.
            # Metadata is the commit marker: a crash during promotion must
            # leave this item incomplete, even when replacing an older decode.
            (out_dir / "metadata.json").unlink(missing_ok=True)
            durable_replace(glb_tmp, glb_path)
            for gs_final in gs_paths:
                durable_replace(tmp_sibling(gs_final), gs_final)
            kept = {p.suffix.lstrip(".") for p in gs_paths}
            for ext in GS_EXTENSIONS:
                if ext not in kept:
                    (out_dir / f"{item_id}.{ext}").unlink(missing_ok=True)
            # Systemic emblem transfer: reference-anchored decal projection for
            # items whose PNG carries insignia blobs (red cross, roundels,
            # lettering). No-op without blobs; best-effort so it never fails
            # a freshly decoded item. Runs before the metadata commit marker.
            emblem_report = None
            try:
                import emblem_project as _emblem
                emblem_report = _emblem.try_emblem_fix(out_dir, item_id)
                if emblem_report.get("applied"):
                    print(f"    emblem projection: "
                          f"{emblem_report['projection']['projected_texels']:,} texels, "
                          f"{emblem_report['gaussians'].get('patched_gaussians', 0):,} gaussians")
            except Exception as _e:
                emblem_report = {"applied": False, "note": f"emblem hook error: {_e}"}
            t_total = time.time() - t0

            metadata = {
                "id": item_id,
                "category": it["category"],
                "geography": it["geography"],
                "name": clean_name(it.get("name", "")),
                "prompt": image_meta.get("prompt"),
                "manifest_prompt": it["prompt_clean"],
                "reference_prompt_recipe_version": image_meta.get("prompt_recipe_version"),
                "manifest_prompt_recipe_version": it.get("generation_prompt", {}).get("version") if is_optimized_prompt(it["prompt"]) else None,
                "reference_provenance": "recorded" if image_meta else "unknown",
                "policy": it["policy"],
                "image_qa": {k: image_qa.get(item_id, {}).get(k)
                             for k in ("pass", "qa_version", "nsfw", "nsfw_screened", "sharpness")},
                "source_image_sha256": file_sha256(png_path),
                "image_model": image_meta.get("image_model"),
                "mesh_model": MESH_MODEL_ID,
                "provenance": provenance,
                "seed": seed,
                "pipeline_type": args.pipeline_type,
                "texture_size": args.texture_size,
                "target_faces": args.target_faces,
                "sampling": sampling,
                "bake_backend": bake_backend,
                "mesh_coordinate_system": "right-handed-y-up",
                "gs_coordinate_system": "right-handed-z-up",
                "vertices": int(len(verts)),
                "triangles": int(len(mesh_out.faces)),
                "gaussians": int(n_gaussians),
                "gs_format": "+".join(
                    p.suffix.lstrip(".") for p in gs_paths),
                "gs_color_encodings": {p.suffix.lstrip("."): GS_COLOR_ENCODINGS[p.suffix.lstrip(".")]
                                       for p in gs_paths},
                "generation_seconds": round(t_gen, 1),
                "total_seconds": round(t_total, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emblem_project": emblem_report,
            }
            if image_meta:
                metadata["image_meta"] = image_meta
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
            if is_infrastructure_error(e):
                if generated_overrides:
                    content_policy.record_overrides(generated_overrides, phase="mesh_gen")
                raise RuntimeError(f"Mesh generation infrastructure failed for {item_id}; assets retained") from e
        finally:
            save_json_atomic(failures_path, failures)

    if generated_overrides:
        path = content_policy.record_overrides(generated_overrides,
                                              phase="mesh_gen")
        print(f"** {len(generated_overrides)} mesh(es) generated under a policy "
              f"override; logged to {path}")

    print(f"finished: {done} meshes, {len(failures)} failures")


if __name__ == "__main__":
    with gpu_lock():
        main()
