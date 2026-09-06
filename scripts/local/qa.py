"""Phase 1.5/2.5: QA gate between and after the generation phases.

Image mode (run with imgenv python): screens every item against the dual-use
content policy, then validates every matted PNG — alpha coverage,
fragmentation, backdrop uniformity, sharpness, near-duplicate subjects, and
NSFW screening (Falconsai/nsfw_image_detection). Failures bump
state/qa_retries.json so the deterministic per-item seed is perturbed and the
image phase regenerates the item.

The safety screens are fail-CLOSED and enforcing:
  * the NSFW model is loaded eagerly, so a missing or broken detector aborts
    the run instead of silently passing every image unscreened;
  * an image over the NSFW threshold is deleted, never left on disk;
  * after MAX_QA_RETRIES rejections an item is quarantined in
    state/qa_quarantine.json so the image phase stops regenerating it.
Content-policy rejections are recorded but never delete anything.

Mesh mode (run with trellis venv python): validates every GLB — loads, bbox
aspect sanity, texture whiteness (residual floor-slab regression), magenta
backdrop residue, degenerate triangles, and now a turntable pass: the mesh is
polygon-rasterised from 10 camera positions around the object (8-azimuth ring
plus top/bottom poles) and each silhouette is checked for coverage and
view-to-view consistency. A montage of every view is written to
<id>.qa_mesh_views.png so a human can eyeball the whole object in one image.
Mesh failures delete the outputs so the mesh phase retries with the perturbed
seed.

Gaussians mode (run with trellis venv python): judges the 3DGS containers
(.spz preferred, .splat fallback) that the mesh phase exports. It follows the
same turntable protocol as the mesh pass: the gaussian cloud is rasterised
from 10 surrounding views (8-azimuth ring plus top/bottom poles), and the
silhouettes are checked for coverage, solidity (ghost/transparent shells),
enclosed holes and view consistency. Native primitive cues — non-finite
positions, degenerate scales, dead splats, bad colors — are checked directly
on the cloud (the no-reference route from 3DGS-QA / AAAI 2026). A montage of
all 10 views is written to <id>.qa_views.png. Failures delete the 3DGS
containers and metadata so the mesh phase regenerates with the perturbed seed.

Verdicts land in state/qa_images.json / state/qa_meshes.json /
state/qa_gaussians.json.

Run with:
    imgenv/bin/python scripts/local/qa.py --mode images
    trellis-mac/.venv/bin/python scripts/local/qa.py --mode meshes
    trellis-mac/.venv/bin/python scripts/local/qa.py --mode gaussians
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_policy  # noqa: E402
from prompt_utils import (  # noqa: E402
    GS_EXTENSIONS, MANIFESTS, PROJECT_DIR, load_manifest,
)

OUTPUT_DIR = PROJECT_DIR / "outputs_relief"
STATE_DIR = PROJECT_DIR / "state"

# Image thresholds
FG_MIN, FG_MAX = 0.03, 0.75          # object coverage of frame
MAIN_PART_MIN = 0.90                 # largest connected component share of fg
BORDER_TOUCH_MAX = 0.02              # fraction of perimeter pixels that are fg
SHARPNESS_MIN = 55.0                 # Laplacian variance on masked gray
DUP_HAMMING = 6                      # phash distance for near-duplicate flag
NSFW_THRESHOLD = 0.50
MAX_QA_RETRIES = 3                   # regeneration attempts before quarantine

# Failure kinds a fresh sample can plausibly fix. Rejected artifacts for these
# are deleted and the item is retried with a perturbed seed. NSFW is included:
# a different sample of the same benign prompt is usually clean, and the
# offending file is deleted either way.
REGENERABLE_PREFIXES = ("fg_fraction", "blurry", "near-duplicate",
                        "fragmented", "touches border", "NSFW")


def _regenerable(reasons) -> bool:
    return any(r.startswith(REGENERABLE_PREFIXES) for r in reasons)

# Mesh thresholds
ASPECT_MIN, ASPECT_MAX = 0.15, 8.0   # height / footprint
DIM_RATIO_MAX = 20.0                 # max/min bbox dimension
WHITE_TEX_MAX = 0.35                 # near-white texture fraction (slab regression)
MAGENTA_TEX_MAX = 0.02               # saturated magenta residue in texture

# 3DGS / mesh turntable-view thresholds. Calibrated against the first nine
# completed TRELLIS.2 outputs so a healthy decode passes with margin while
# broken geometry (missing side, ghost shell, enclosed holes, view
# inconsistency) fails and forces regeneration.
GS_VIEW_RES = 128            # raster/raycast resolution (pixels, square)
GS_MAX_GAUSSIANS = 200_000   # stride-subsampled cloud size for the raster
GS_AZIMUTHS = 8              # turntable positions around the object
GS_ELEVATIONS = (0.45, math.pi / 2, -math.pi / 2)
#                             -> 10 lenses: 8-azimuth ring ~26deg up, plus
#                             top and bottom poles (poles catch missing
#                             undersides and hollow shells)
GS_FOCAL = 1.5 * GS_VIEW_RES
GS_KERNEL = 5                # per-splat raster kernel (GS_KERNEL x GS_KERNEL)

GS_COV_MIN = 0.015           # every view must show at least this frame share
GS_COV_MAX = 0.95            # nothing may flood the whole frame
GS_SOLID_MIN = 0.35          # mean opacity weight inside the silhouette
GS_HOLE_MIN_SHARE = 0.005    # enclosed gap must be >= this share of silhouette
GS_HOLE_MAX = 0.05           # total enclosed-hole area / silhouette
GS_HOLE_APERTURE_MAX = 0.30  # a single-view aperture above this is a defect
# Holes between GS_HOLE_MAX and GS_HOLE_APERTURE_MAX are tolerated when they
# appear in exactly one view (an antenna dish, ring or handle seen from one
# side) but fail when two or more views expose them (hollow shell).
GS_IOU_MIN = 0.20            # adjacent-view silhouette overlap (view consistency)

GS_NAN_TOL = 0.0             # any non-finite position is a failure
GS_BAD_COLOR_MAX = 0.10      # share of gaussians with non-finite colors
GS_DEAD_MAX = 0.90           # share of gaussians with opacity ~0 after sigmoid
GS_DEG_SCALE_MAX = 0.50      # share of gaussians with degenerate scale

SH_C0 = 0.2820947917738949   # zeroth SH basis value (gs_export.C0)
MESH_RAY_MAX_FACES = 150_000  # decimate meshes above this before rasterising
MESH_RAY_DECIMATE = 0.95      # face fraction dropped by quadric decimation

GS_MONTAGE = "qa_views.png"
MESH_MONTAGE = "qa_mesh_views.png"


def all_items(geographies=None):
    geos = geographies or list(MANIFESTS)
    return [it for geo in geos for it in load_manifest(geo)]


def select_items(args):
    """The items this QA pass covers, honouring the same scoping as the phases.

    qa.py used to always walk all 10,079 rows, so `run_all.py --limit 2` still
    re-examined the entire corpus. The selection flags mirror flux_imagegen so
    the orchestrator can forward one set of arguments.
    """
    if args.ids:
        wanted = set(args.ids)
        return [it for it in all_items() if it["id"] in wanted]
    items = all_items(args.geographies)
    items = content_policy.select_tier(items, args.only_policy)
    if args.limit:
        items = items[: args.limit]
    return items


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path, data):
    """Atomic write using the pipeline's hidden-transient convention.

    `with_suffix(".tmp")` turned qa_images.json into a visible sibling named
    qa_images.tmp, which neither matches the `.<name>.tmp` form the rest of the
    pipeline uses nor the glob verify_local.py sweeps for stale transients.
    """
    content_policy.write_json(path, data)


def bump_retry(retries, item_id) -> bool:
    """Count a rejection. Returns False once the item has exhausted its
    retries, so callers can quarantine it instead of looping forever."""
    retries[item_id] = retries.get(item_id, 0) + 1
    return retries[item_id] < MAX_QA_RETRIES


def quarantine_if_exhausted(retries, quarantine, item_id, reasons) -> bool:
    """Bump the retry counter; quarantine once the item is out of attempts.

    Returns True if the item was newly quarantined. Shared by both QA modes so
    the ledger entry has one shape.
    """
    if bump_retry(retries, item_id):
        return False
    quarantine[item_id] = {
        "reasons": reasons,
        "retries": retries[item_id],
        "at": datetime.now(timezone.utc).isoformat(),
    }
    return True


# ---------------------------------------------------------------- image mode

def largest_component_share(alpha_bool):
    from scipy import ndimage
    total = int(alpha_bool.sum())
    if total == 0:
        return 0.0
    lbl, n = ndimage.label(alpha_bool)
    if n == 0:
        return 0.0
    sizes = np.bincount(lbl.ravel())[1:]
    return float(sizes.max() / total)


def image_scores(png: Path):
    rgba = np.asarray(Image.open(png).convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3] / 255.0
    fg = alpha > 0.5
    h, w = fg.shape
    perimeter = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])

    gray = rgba[..., :3].mean(axis=-1)
    gy, gx = np.gradient(gray)
    lap = gx**2 + gy**2
    sharpness = float(lap[fg].mean()) if fg.any() else 0.0

    try:
        import imagehash
        phash = str(imagehash.phash(Image.open(png)))
    except Exception:
        phash = None

    return {
        "fg_fraction": round(float(fg.mean()), 4),
        "main_part_share": round(largest_component_share(fg), 4),
        "border_touch": round(float(perimeter.mean()), 4),
        "sharpness": round(sharpness, 1),
        "phash": phash,
    }


def image_pass(scores, category_peers):
    reasons = []
    if not (FG_MIN <= scores["fg_fraction"] <= FG_MAX):
        reasons.append(f"fg_fraction {scores['fg_fraction']}")
    if scores["main_part_share"] < MAIN_PART_MIN:
        reasons.append(f"fragmented matte {scores['main_part_share']}")
    if scores["border_touch"] > BORDER_TOUCH_MAX:
        reasons.append(f"touches border {scores['border_touch']}")
    if scores["sharpness"] < SHARPNESS_MIN:
        reasons.append(f"blurry {scores['sharpness']}")
    if scores.get("phash") and category_peers:
        import imagehash
        h = imagehash.hex_to_hash(scores["phash"])
        for peer in category_peers:
            if h - imagehash.hex_to_hash(peer) <= DUP_HAMMING:
                reasons.append("near-duplicate subject")
                break
    return reasons


NSFW_DIR = PROJECT_DIR / "meshmaker" / "local" / "models" / "nsfw_image_detection"


class NsfwGuard:
    """Fail-closed NSFW screen.

    The model is loaded in __init__ rather than lazily on first score(), so a
    missing or unloadable detector raises where the caller can still refuse to
    proceed. Lazy loading previously made construction unconditionally succeed,
    which turned the caller's error handling into dead code and let every
    image through unscreened.
    """

    def __init__(self):
        if not (NSFW_DIR / "config.json").exists():
            raise FileNotFoundError(f"NSFW model missing at {NSFW_DIR}")
        os_environ_offline()
        from transformers import pipeline
        self.pipe = pipeline(
            "image-classification",
            model=str(NSFW_DIR),
            device="cpu",
        )

    def score(self, png: Path) -> float:
        out = self.pipe(str(png))
        return next(r["score"] for r in out if r["label"] == "nsfw")


def os_environ_offline():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def wipe_image(d: Path, item_id: str):
    (d / f"{item_id}.png").unlink(missing_ok=True)
    (d / f"{item_id}.raw.jpg").unlink(missing_ok=True)
    (d / f"{item_id}.debug.png").unlink(missing_ok=True)


def run_images(args):
    items = select_items(args)
    verdicts = load_json(STATE_DIR / "qa_images.json")
    retries = load_json(STATE_DIR / "qa_retries.json")
    quarantine = load_json(STATE_DIR / "qa_quarantine.json")

    # Content policy runs before any pixel is examined: an item that must not
    # exist should not be judged on whether it is well-lit. screen_from_args
    # tags every item with its verdict, so `items` below still covers the whole
    # selection and only the refused ones acquire a policy failure reason.
    permitted, rejected, overrides = content_policy.screen_from_args(
        items, args, phase="image_qa")
    policy_reasons = {
        it["id"]: f"policy {it['policy']['classification']} "
                  f"({it['policy']['category']})"
        for it in rejected
    }
    print(content_policy.describe_screen(permitted, rejected, overrides))
    # --only-policy narrows the QA pass to the same tier the generator ran, so
    # an audit run does not re-QA the other ten thousand items.
    if args.only_policy:
        keep = {it["id"] for it in permitted} | set(policy_reasons)
        items = [it for it in items if it["id"] in keep]

    # Fail closed: without a working detector we cannot certify anything, so
    # abort rather than emit an unscreened dataset. --no-nsfw is the explicit,
    # recorded opt-out.
    guard = None
    if args.no_nsfw:
        print("WARNING: --no-nsfw given; images are NOT safety-screened")
    else:
        guard = NsfwGuard()

    # peer phashes per category for duplicate detection among ALREADY-passed items
    peers = {}
    for it in items:
        v = verdicts.get(it["id"])
        if v and v.get("pass") and v.get("phash"):
            peers.setdefault(it["category"], []).append(v["phash"])

    n_fail = 0
    n_quarantined = 0
    for it in items:
        d = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        png = d / f"{it['id']}.png"
        if not png.exists():
            continue
        prev = verdicts.get(it["id"], {})
        if prev.get("pass") and not args.rescan:
            continue
        s = image_scores(png)
        reasons = image_pass(s, peers.get(it["category"], []))

        policy_reason = policy_reasons.get(it["id"])
        if policy_reason:
            reasons.append(policy_reason)

        nsfw = None
        if guard is not None:
            nsfw = guard.score(png)
            if nsfw > NSFW_THRESHOLD:
                reasons.append(f"NSFW {nsfw:.2f}")

        passed = not reasons
        verdicts[it["id"]] = {
            "pass": passed, **s,
            "nsfw": round(nsfw, 4) if nsfw is not None else None,
            "policy": it["policy"],
            "reasons": reasons,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if passed:
            if s.get("phash"):
                peers.setdefault(it["category"], []).append(s["phash"])
        else:
            n_fail += 1
            if policy_reason and not _regenerable(reasons):
                # Policy failures are inherent to the prompt: regenerating
                # cannot help, and we do not delete existing artifacts.
                quarantine[it["id"]] = {
                    "reasons": reasons, "retries": retries.get(it["id"], 0),
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            else:
                # Everything else is treated as worth another sample. Rejected
                # content is removed unconditionally, even once the item is out
                # of retries: quarantine stops regeneration, it does not license
                # keeping the artifact.
                if not _regenerable(reasons):
                    # A reason we do not recognise must not become a silent
                    # no-op: without this the item would keep its failing PNG,
                    # never bump its seed and never be quarantined, so it would
                    # sit in permanent limbo invisible to every counter.
                    print(f"  warning: unclassified QA reason(s) {reasons}; "
                          f"treating as regenerable")
                wipe_image(d, it["id"])
                if quarantine_if_exhausted(retries, quarantine, it["id"], reasons):
                    n_quarantined += 1
        print(f"{it['id']}: {'PASS' if passed else 'FAIL ' + '; '.join(reasons)}")
        save_json(STATE_DIR / "qa_images.json", verdicts)

    save_json(STATE_DIR / "qa_retries.json", retries)
    save_json(STATE_DIR / "qa_quarantine.json", quarantine)
    print(f"image QA: {len(verdicts)} verdicts, {n_fail} new failures, "
          f"{n_quarantined} newly quarantined ({len(quarantine)} total)")


# ---------------------------------------------------------------- mesh mode

def glb_scores(glb_path: Path):
    import trimesh
    m = trimesh.load(glb_path, force="scene")
    geoms = list(m.geometry.values()) if hasattr(m, "geometry") else [m]
    verts = np.vstack([g.vertices for g in geoms])
    ext = verts.max(axis=0) - verts.min(axis=0)
    footprint = float(np.prod(np.sort(ext)[:2]) ** 0.5) + 1e-9
    aspect = float(ext[2] / footprint) if len(ext) == 3 else 1.0

    tex_white = tex_magenta = 0.0
    degenerate = 0.0
    for g in geoms:
        if hasattr(g, "visual") and getattr(g.visual, "material", None) is not None:
            img = getattr(g.visual.material, "baseColorTexture", None)
            if img is not None:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32)
                white = np.all(arr > 240, axis=-1).mean()
                mx = arr.max(axis=-1); mn = arr.min(axis=-1)
                sat = (mx - mn) / (mx + 1e-6)
                r, gr, b = arr[..., 0], arr[..., 1], arr[..., 2]
                magenta = ((r > 120) & (b > 120) & (gr < 90) & (sat > 0.45)).mean()
                tex_white = max(tex_white, float(white))
                tex_magenta = max(tex_magenta, float(magenta))
        if hasattr(g, "area_faces") and len(g.area_faces):
            degenerate = max(degenerate, float((g.area_faces < 1e-12).mean()))
    return {
        "vertices": int(len(verts)),
        "bbox_ext": [round(float(e), 3) for e in ext],
        "aspect": round(aspect, 3),
        "dim_ratio": round(float(ext.max() / (ext.min() + 1e-9)), 2),
        "tex_white": round(tex_white, 4),
        "tex_magenta": round(tex_magenta, 4),
        "degenerate_frac": round(degenerate, 5),
    }, geoms


def mesh_pass(s):
    reasons = []
    if s["vertices"] == 0:
        reasons.append("empty mesh")
    if not (ASPECT_MIN <= s["aspect"] <= ASPECT_MAX):
        reasons.append(f"aspect {s['aspect']}")
    if s["dim_ratio"] > DIM_RATIO_MAX:
        reasons.append(f"dim_ratio {s['dim_ratio']}")
    if s["tex_white"] > WHITE_TEX_MAX:
        reasons.append(f"white slab? tex_white {s['tex_white']}")
    if s["tex_magenta"] > MAGENTA_TEX_MAX:
        reasons.append(f"magenta residue {s['tex_magenta']}")
    return reasons


def mesh_silhouettes(geoms, res=96):
    """Rasterize the mesh silhouette from every turntable camera.

    Projects the (decimated) triangles with the same pinhole rig as the
    gaussian pass and polygon-fills them — no ray casting, so dense
    TRELLIS.2 meshes cost fractions of a second per view instead of
    minutes. Returns (masks, shades): per-view bool silhouette and
    per-view uint8 montage images, in turntable order.
    """
    import trimesh
    from PIL import Image, ImageDraw
    mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no faces")
    if len(mesh.faces) > MESH_RAY_MAX_FACES:
        mesh = mesh.simplify_quadric_decimation(MESH_RAY_DECIMATE)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    center, radius = _obj_rig(V)
    focal = 1.5 * res
    masks, shades = [], []
    for cam, right, up, forward in _turntable_cameras(center, radius):
        rel = V - np.asarray(cam)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = rel @ forward                  # depth along the optical axis
            x = rel @ right
            y = rel @ up
            px = x / z * focal + res / 2.0
            py = -y / z * focal + res / 2.0
        P = np.stack([px, py], axis=1)
        good = np.isfinite(V).all(axis=1)      # legacy GLBs can carry NaNs
        img = Image.new("L", (res, res), 0)
        dr = ImageDraw.Draw(img)
        zmin = 0.05 * radius  # drop anything touching the camera plane
        keep = [k for k, (i0, i1, i2) in enumerate(F)
                if good[i0] and good[i1] and good[i2]
                and z[i0] >= zmin and z[i1] >= zmin and z[i2] >= zmin]
        if keep:
            zf = np.array([z[F[k]].mean() for k in keep])
            lo, hi = float(zf.min()), float(zf.max())
            span = (hi - lo) or 1.0
            for k, shade in zip(keep, (255 * (1 - (zf - lo) / span)).astype(int)):
                i0, i1, i2 = F[k]
                a, b, c = P[i0], P[i1], P[i2]
                dr.polygon([(a[0], a[1]), (b[0], b[1]), (c[0], c[1])],
                           fill=int(shade))
        m = np.asarray(img) > 0
        masks.append(m)
        shades.append(np.asarray(img))
    return masks, shades


def mesh_view_reasons(stats):
    reasons = []
    if stats["coverage_min"] < GS_COV_MIN:
        reasons.append(f"empty view coverage {stats['coverage_min']}")
    # Enclosed silhouette holes are recorded but do not gate the mesh pass:
    # at mesh level they are usually genuine object topology (trigger guards,
    # quadcopter frames, antenna apertures, rotor hubs), and the definitive
    # hollow-shell detection is the opacity-solidity gate on the gaussian
    # side. Coverage and view consistency remain hard gates.
    if stats["iou_min"] < GS_IOU_MIN:
        reasons.append(f"inconsistent views iou {stats['iou_min']}")
    return reasons


def run_meshes(args):
    # Mesh QA judges geometry that already exists, so it does not re-run the
    # policy gate; select_items only scopes which items it looks at.
    items = select_items(args)
    verdicts = load_json(STATE_DIR / "qa_meshes.json")
    retries = load_json(STATE_DIR / "qa_retries.json")
    quarantine = load_json(STATE_DIR / "qa_quarantine.json")
    n_fail = 0
    n_quarantined = 0
    for it in items:
        d = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        glb = d / f"{it['id']}.glb"
        if not glb.exists():
            continue
        prev = verdicts.get(it["id"], {})
        if prev.get("pass") and not args.rescan:
            continue
        view_stats = {}
        try:
            s, geoms = glb_scores(glb)
            reasons = mesh_pass(s)
            try:
                masks, shades = mesh_silhouettes(geoms)
                view_stats = silhouette_view_metrics(masks)
                reasons += mesh_view_reasons(view_stats)
                write_view_montage(
                    shades,
                    d / f"{it['id']}.{MESH_MONTAGE}",
                    f"{it['id']} mesh turntable", 128,
                    [str(i + 1) for i in range(len(masks))])
            except Exception as ve:
                view_stats = {"view_error": str(ve)}
        except Exception as e:
            s, reasons = {}, [f"load error: {e}"]
        passed = not reasons
        verdicts[it["id"]] = {"pass": passed, **s, "reasons": reasons,
                              "at": datetime.now(timezone.utc).isoformat()}
        if not passed:
            n_fail += 1
            if any(r.startswith(("white slab", "magenta", "empty", "load error"))
                   for r in reasons):
                # geometry-level regression: wipe mesh artifacts so the mesh
                # phase retries with the perturbed seed. Every 3DGS container
                # goes, otherwise a stale splat from the rejected geometry
                # outlives the mesh it was decoded with.
                (d / f"{it['id']}.glb").unlink(missing_ok=True)
                for ext in GS_EXTENSIONS:
                    (d / f"{it['id']}.{ext}").unlink(missing_ok=True)
                if getattr(args, "no_wipe", False):
                    continue
                (d / f"{it['id']}.{GS_MONTAGE}").unlink(missing_ok=True)
                (d / f"{it['id']}.{MESH_MONTAGE}").unlink(missing_ok=True)
                (d / "metadata.json").unlink(missing_ok=True)
                if quarantine_if_exhausted(retries, quarantine, it["id"], reasons):
                    n_quarantined += 1
        print(f"{it['id']}: {'PASS' if passed else 'FAIL ' + '; '.join(reasons)}")
        save_json(STATE_DIR / "qa_meshes.json", verdicts)
    save_json(STATE_DIR / "qa_retries.json", retries)
    save_json(STATE_DIR / "qa_quarantine.json", quarantine)
    print(f"mesh QA: {len(verdicts)} verdicts, {n_fail} new failures, "
          f"{n_quarantined} newly quarantined ({len(quarantine)} total)")


# ------------------------------------------------- 3DGS (spz/splat) mode

# The turntable protocol shared by the mesh ray-cast pass and the gaussian
# raster pass: cameras orbit the object centroid at two elevations, matching
# the 360-degree orbit protocol used for subjective 3DGS evaluation in
# 3DGS-QA (Wan et al., AAAI 2026) — 10 lenses: an 8-position ring plus top
# and bottom poles. Seeing the object from every side in a fixed order is
# exactly the "rotate the graphics" imperfection scan.


def _turntable_cameras(center, radius, n_az=GS_AZIMUTHS,
                       elevations=GS_ELEVATIONS):
    """Camera rig: exactly 10 lenses from n_az ring elevations + the poles.

    Views are ordered azimuth-major (all azimuths at the lowest ring
    elevation, then the next), with the top and bottom poles last. At a pole
    every azimuth collapses to the same camera, so each pole contributes one
    view, not n_az duplicates.
    """
    center = np.asarray(center, dtype=np.float64)
    cams = []
    up_ref = np.array([0.0, 0.0, 1.0])
    half_pi = math.pi / 2.0
    for el in elevations:
        if abs(abs(el) - half_pi) < 1e-6:
            azs = [0.0]                    # pole: single lens
        else:
            azs = [2.0 * math.pi * i / n_az for i in range(n_az)]
        for az in azs:
            cam = center + radius * np.array([
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el)])
            forward = center - cam
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, up_ref)
            nrm = np.linalg.norm(right)
            right = np.array([1.0, 0.0, 0.0]) if nrm < 1e-6 else right / nrm
            up = np.cross(right, forward)
            cams.append((cam, right, up, forward))
    return cams


def _obj_rig(points):
    """Centroid + orbit radius that keeps the whole object in frame."""
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    ext = pts.max(axis=0) - pts.min(axis=0)
    radius = float(np.linalg.norm(ext)) * 1.25 + 1e-3
    return center, radius


# ------------------------------------------------------------- cloud loading

def _load_spz_gaussians(path):
    """Niantic binding: flat arrays, scales are logs, rotations xyzw,
    colors RGB 0..1, alphas pre-sigmoid logits. Convert to the internal
    g dict (rotations wxyz, f_dc SH DC, opacity logit)."""
    import spz
    cloud = spz.load_spz(str(path))
    n = cloud.num_points
    if n == 0:
        raise ValueError(f"empty spz: {path}")
    pos = np.asarray(cloud.positions).reshape(n, 3)
    rot = np.asarray(cloud.rotations).reshape(n, 4)  # xyzw
    rgb = np.asarray(cloud.colors).reshape(n, 3)
    return {
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "scale": np.asarray(cloud.scales).reshape(n, 3).astype(np.float32),
        "rot": np.stack([rot[:, 3], rot[:, 0], rot[:, 1], rot[:, 2]],
                        axis=1).astype(np.float32),  # -> wxyz
        "f_dc": ((np.clip(rgb, 0.0, 1.0) - 0.5) / SH_C0).astype(np.float32),
        "opacity": np.asarray(cloud.alphas).reshape(n, 1).astype(np.float32),
    }


def _load_splat_gaussians(path):
    """antimatter15 .splat: 32 B/splat — pos f32 xyz, scale f32 xyz,
    color+alpha u8, rotation u8 (w,x,y,z, 0 bias / 128 scale)."""
    buf = np.fromfile(path, dtype=np.uint8).reshape(-1, 32)
    n = len(buf)
    if n == 0:
        raise ValueError(f"empty splat: {path}")
    pos = buf[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scl = buf[:, 12:24].copy().view(np.float32).reshape(n, 3)
    rot = (buf[:, 28:32].astype(np.float32) - 128.0) / 128.0  # wxyz
    norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = rot / np.clip(norm, 1e-8, None)
    rgb = buf[:, 24:27].astype(np.float32) / 255.0
    alpha = buf[:, 27].astype(np.float32) / 255.0
    alpha = np.clip(alpha, 1e-6, 1.0 - 1e-6)
    return {
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "scale": np.log(np.clip(scl, 1e-8, None)),
        "rot": rot.astype(np.float32),
        "f_dc": ((rgb - 0.5) / SH_C0).astype(np.float32),
        "opacity": np.log(alpha / (1.0 - alpha)).reshape(n, 1).astype(np.float32),
    }


def _load_ply_gaussians(path):
    """Canonical INRIA PLY (62 f32 per vertex) — legacy fallback."""
    raw = path.read_bytes()
    header, _, payload = raw.partition(b"end_header\n")
    m = re.search(rb"element vertex (\d+)", header)
    if not m:
        raise ValueError(f"unparseable ply header: {path}")
    n = int(m.group(1))
    arr = np.frombuffer(payload, dtype=np.float32).reshape(n, 62)
    if n == 0:
        raise ValueError(f"empty ply: {path}")
    return {
        "x": arr[:, 0], "y": arr[:, 1], "z": arr[:, 2],
        "scale": arr[:, 55:58],
        "rot": arr[:, 58:62],
        "f_dc": arr[:, 6:9],
        "opacity": arr[:, 54:55],
    }


def load_gaussians(item_dir, item_id):
    """First existing 3DGS container in preference order: spz, splat, ply.

    Returns (gaussians_dict, container_extension) or (None, None)."""
    d = Path(item_dir)
    for ext in ("spz", "splat", "ply"):
        p = d / f"{item_id}.{ext}"
        if not p.exists():
            continue
        loader = {"spz": _load_spz_gaussians,
                  "splat": _load_splat_gaussians,
                  "ply": _load_ply_gaussians}[ext]
        return loader(p), ext
    return None, None


# --------------------------------------------------- native primitive checks

def gaussian_primitive_stats(g):
    """GSOQA-style cues read straight off the cloud, no rendering needed:
    non-finite positions, degenerate scales, dead splats, bad colors.
    Returns (stats dict, reasons list)."""
    pos = np.stack([g["x"], g["y"], g["z"]], axis=1)
    finite = np.isfinite(pos).all(axis=1)
    nan_frac = float((~finite).mean())

    scale = np.exp(np.clip(np.asarray(g["scale"], dtype=np.float32), -30, 30))
    deg_frac = float(((~np.isfinite(scale)).any(axis=1)
                      | (scale <= 0).any(axis=1)
                      | (scale > 100).any(axis=1)).mean())

    alpha = 1.0 / (1.0 + np.exp(-np.clip(
        np.asarray(g["opacity"], dtype=np.float32).ravel(), -30, 30)))
    dead_frac = float((alpha < 0.01).mean())
    mean_alpha = float(alpha.mean())

    rgb = np.asarray(g["f_dc"], dtype=np.float32) * SH_C0 + 0.5
    col_frac = float((~np.isfinite(rgb)).any(axis=1).mean())

    stats = {
        "gaussians": int(len(pos)),
        "nan_frac": round(nan_frac, 6),
        "degenerate_scale_frac": round(deg_frac, 5),
        "dead_frac": round(dead_frac, 5),
        "mean_opacity": round(mean_alpha, 4),
        "bad_color_frac": round(col_frac, 6),
    }
    reasons = []
    if nan_frac > GS_NAN_TOL:
        reasons.append(f"non-finite positions {nan_frac:.4f}")
    if deg_frac > GS_DEG_SCALE_MAX:
        reasons.append(f"degenerate scales {deg_frac:.2f}")
    if dead_frac > GS_DEAD_MAX:
        reasons.append(f"dead splats {dead_frac:.2f}")
    if col_frac > GS_BAD_COLOR_MAX:
        reasons.append(f"bad colors {col_frac:.4f}")
    return stats, reasons


# ------------------------------------------------------- gaussian rasterizer

def _rot_matrices_wxyz(quat):
    """(n, 4) wxyz unit quaternions -> (n, 3, 3) rotation matrices."""
    import torch
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    r = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * z + w * y)], dim=1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x)], dim=1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)], dim=1),
    ], dim=1)
    return r


def render_splat_views(g, res=GS_VIEW_RES, max_gaussians=GS_MAX_GAUSSIANS):
    """Rasterise the gaussian cloud from every turntable camera (CPU torch).

    Each gaussian is splatted with a small 2D gaussian kernel whose radius
    comes from its projected covariance, accumulated order-independently with
    scatter_add (coverage weight + colour weight). Returns
    (masks, weight_maps, colour_views): per-view bool silhouette, f32 weight
    map and uint8 RGB view, in turntable order.
    """
    import torch
    n = len(g["x"])
    if n == 0:
        return None

    # Deterministic stride subsampling keeps the raster fast and stable while
    # preserving spatial structure (a broken side stays broken after the cut).
    if n > max_gaussians:
        idx = np.linspace(0, n - 1, max_gaussians).astype(np.int64)
        idx = np.unique(idx)
    else:
        idx = None

    def pick(a):
        a = np.asarray(a, dtype=np.float32)
        return a[idx] if idx is not None else a

    pos = np.stack([pick(g["x"]), pick(g["y"]), pick(g["z"])], axis=1)
    scale = np.exp(np.clip(pick(g["scale"]), -12, 12))
    quat = pick(g["rot"])
    rgb = np.clip(pick(g["f_dc"]) * SH_C0 + 0.5, 0.0, 1.0)
    alpha = 1.0 / (1.0 + np.exp(-np.clip(
        pick(g["opacity"]).ravel(), -30, 30)))
    m = len(pos)

    center, radius = _obj_rig(pos)
    cams = _turntable_cameras(center, radius)
    focal = float(GS_FOCAL)
    offs = [(dy, dx) for dy in range(-(GS_KERNEL // 2), GS_KERNEL // 2 + 1)
            for dx in range(-(GS_KERNEL // 2), GS_KERNEL // 2 + 1)]

    pos_t = torch.from_numpy(pos)
    scale_t = torch.from_numpy(scale)
    rgb_t = torch.from_numpy(rgb)
    alpha_t = torch.from_numpy(alpha)
    R = _rot_matrices_wxyz(torch.from_numpy(quat))
    S = scale_t.unsqueeze(2) * R          # scale rows of R
    cov3d = torch.bmm(S, R.transpose(1, 2))  # R diag(s)^2 R^T

    masks, weights, colours = [], [], []
    pix_area = res * res
    for cam, right, up, forward in cams:
        Rc = np.stack([right, up, forward]).astype(np.float32)  # world->cam
        Rc_t = torch.from_numpy(Rc)
        dcam = (pos_t - torch.from_numpy(np.asarray(cam, np.float32))) @ Rc_t.T
        zc = dcam[:, 2]
        ok = (zc > 0.05) & (zc < 1e3) & torch.isfinite(dcam).all(dim=1)
        zc_safe = torch.clamp(zc, 0.05, None)
        u = focal * dcam[:, 0] / zc_safe + res / 2
        v = focal * dcam[:, 1] / zc_safe + res / 2

        # 2D screen covariance: top-left 2x2 of Rc cov3d Rc^T scaled by (f/z)^2.
        # @ broadcasts the 3x3 camera rotation against the (n,3,3) cloud.
        cov_cam = Rc_t @ cov3d @ Rc_t.T
        fz = focal / zc_safe
        a11 = cov_cam[:, 0, 0] * fz * fz
        a12 = cov_cam[:, 0, 1] * fz * fz
        a22 = cov_cam[:, 1, 1] * fz * fz
        trace = a11 + a22
        det = a11 * a22 - a12 * a12
        lam = (trace + torch.sqrt(torch.clamp(trace * trace - 4 * det,
                                              min=0.0))) / 2.0
        sigma = torch.clamp(1.0 * torch.sqrt(torch.clamp(lam, min=0.0)),
                            0.4, 2.0)

        ui = u.round().long()
        vi = v.round().long()
        wmap = torch.zeros(pix_area, dtype=torch.float32)
        cmap = torch.zeros(pix_area, 3, dtype=torch.float32)
        for dy, dx in offs:
            pu = ui + dx
            pv = vi + dy
            in_img = ok & (pu >= 0) & (pu < res) & (pv >= 0) & (pv < res)
            pix = (pv * res + pu).clamp(0, pix_area - 1)
            wk = torch.exp(-(dx * dx + dy * dy)
                           / (2.0 * sigma * sigma)) * alpha_t
            wk = wk * in_img.float()
            wmap.scatter_add_(0, pix, wk)
            cmap.scatter_add_(0, pix.unsqueeze(1).expand(m, 3),
                              wk.unsqueeze(1) * rgb_t)
        masks.append((wmap > 0.05).numpy().reshape(res, res))
        weights.append(wmap.numpy().reshape(res, res))
        col = cmap / torch.clamp(wmap.unsqueeze(1), 1e-6, None)
        colours.append((col.clamp(0, 1) * 255).byte().numpy().reshape(res, res, 3))
    return masks, weights, colours


# ------------------------------------------------------------- view metrics

def enclosed_hole_frac(mask):
    """Enclosed background share of a silhouette (pure numpy, no scipy).

    Flood-fills the background from the image border; anything background
    that the flood cannot reach is enclosed by the object. Only components at
    least GS_HOLE_MIN_SHARE of the silhouette count, so real object cutouts
    (wheel arches, trusses) do not read as defects.
    """
    total = int(mask.sum())
    if total == 0:
        return 0.0
    h, w = mask.shape
    bg = ~mask
    # Seed the flood with border background pixels.
    border = np.zeros_like(bg)
    border[0, :] = bg[0, :]
    border[-1, :] = bg[-1, :]
    border[:, 0] = bg[:, 0]
    border[:, -1] = bg[:, -1]
    frontier = set(zip(*np.nonzero(border)))
    interior = set(zip(*np.nonzero(bg))) - frontier
    reachable = set(frontier)
    while frontier:
        nxt = set()
        for y, x in frontier:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (y + dy, x + dx)
                if nb in interior:
                    interior.discard(nb)
                    nxt.add(nb)
        frontier = nxt
    # interior now holds only enclosed background pixels; group them into
    # connected components (BFS over the small remaining set).
    min_px = GS_HOLE_MIN_SHARE * total
    big = 0
    while interior:
        start = interior.pop()
        comp, queue = 0, [start]
        while queue:
            y, x = queue.pop()
            comp += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (y + dy, x + dx)
                if nb in interior:
                    interior.discard(nb)
                    queue.append(nb)
        if comp >= min_px:
            big += comp
    return float(big / total)


def _dilate(mask, px=2):
    """Binary dilation by px pixels (square kernel) via PIL MaxFilter."""
    from PIL import Image, ImageFilter
    k = 2 * px + 1
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MaxFilter(k))) > 0


def silhouette_view_metrics(masks, weight_maps=None, colours=None):
    """Coverage, solidity, enclosed holes and view consistency across the
    turntable silhouettes. Returns a stats dict (no verdict)."""
    n = len(masks)
    coverages = np.array([float(m.mean()) for m in masks])

    solidities = []
    if weight_maps:
        for m, w in zip(masks, weight_maps):
            total = int(m.sum())
            solidities.append(float((w[m] > 0.5).sum() / total) if total else 0.0)

    hole_fracs = [enclosed_hole_frac(m) for m in masks]

    # View consistency is only meaningful between neighbouring cameras.
    # Poles (and any second ring) are not neighbours of the ring views, so
    # compare consecutive azimuths within each elevation ring only. Masks are
    # dilated first: a slender object (knife, drone) legitimately shifts a
    # couple of pixels between adjacent cameras, and raw overlap would fail
    # healthy assets while real defects (missing side, ghost shell) still
    # fall far below the threshold.
    ious = []
    for start in range(0, n, GS_AZIMUTHS):
        ring = list(range(start, min(start + GS_AZIMUTHS, n)))
        if len(ring) < 3:
            continue  # poles / partial rings have no meaningful neighbour
        dil = [_dilate(m) for m in masks]
        for i in ring:
            a, b = dil[i], dil[(i - start + 1) % len(ring) + start]
            inter = int((a & b).sum())
            union = int((a | b).sum())
            ious.append(inter / union if union else 0.0)

    stats = {
        "coverage_min": round(float(coverages.min()), 4),
        "coverage_mean": round(float(coverages.mean()), 4),
        "coverage_std": round(float(coverages.std()), 4),
        "solidity_min": round(min(solidities), 4) if solidities else None,
        "holes_max": round(max(hole_fracs), 4),
        "holes_over_count": int(sum(1 for h in hole_fracs if h > GS_HOLE_MAX)),
        "iou_min": round(min(ious), 4),
        "views": n,
    }
    if colours:
        stats["brightness_mean"] = round(
            float(np.mean([c.mean() for c in colours])), 1)
    return stats


def gaussians_view_reasons(stats):
    reasons = []
    if stats["coverage_min"] < GS_COV_MIN:
        reasons.append(f"empty view coverage {stats['coverage_min']}")
    if stats["coverage_mean"] > GS_COV_MAX:
        reasons.append(f"overfull frame coverage {stats['coverage_mean']}")
    if stats.get("solidity_min") is not None and stats["solidity_min"] < GS_SOLID_MIN:
        reasons.append(f"ghost shell solidity {stats['solidity_min']}")
    if stats["holes_max"] > GS_HOLE_MAX and (
            stats.get("holes_over_count", 0) >= 2
            or stats["holes_max"] > GS_HOLE_APERTURE_MAX):
        reasons.append(f"holes {stats['holes_max']}")
    if stats["iou_min"] < GS_IOU_MIN:
        reasons.append(f"inconsistent views iou {stats['iou_min']}")
    return reasons


def write_view_montage(views, path, title, res, labels=None):
    """Contact sheet of every view in turntable order (row-major, azimuth-major).

    views: list of HxWx3 uint8 or HxW float/bool arrays."""
    from PIL import Image, ImageDraw
    n = len(views)
    cols = 4
    rows = (n + cols - 1) // cols
    bar = 22
    img = Image.new("RGB", (cols * res, rows * res + bar), (18, 18, 22))
    d = ImageDraw.Draw(img)
    d.text((6, 4), title, fill=(220, 220, 220))
    for i, v in enumerate(views):
        a = np.asarray(v)
        if a.ndim == 2:
            if a.dtype == np.bool_:
                a = (a.astype(np.float32) * 255).astype(np.uint8)
            elif a.dtype != np.uint8:
                a = np.clip(a * 255, 0, 255).astype(np.uint8)
            a = np.repeat(a[:, :, None], 3, axis=2)
        tile = Image.fromarray(a).resize((res, res), Image.NEAREST)
        x, y = (i % cols) * res, (i // cols) * res + bar
        img.paste(tile, (x, y))
        d.text((x + 3, y + 3), str(i + 1) if labels is None else labels[i],
               fill=(255, 255, 120))
    img.save(path)


# ------------------------------------------------------------ gaussians run

def wipe_gaussians(d: Path, item_id: str):
    """Delete the 3DGS containers + metadata so the mesh phase sees the item
    as incomplete and regenerates it with the perturbed retry seed."""
    for ext in GS_EXTENSIONS:
        (d / f"{item_id}.{ext}").unlink(missing_ok=True)
    (d / f"{item_id}.{GS_MONTAGE}").unlink(missing_ok=True)
    (d / "metadata.json").unlink(missing_ok=True)


def run_gaussians(args):
    """Turntable + primitive QA on the 3DGS containers.

    Judges geometry that already exists, so like the mesh pass it does not
    re-run the policy gate. Failures wipe the containers so the mesh phase
    regenerates; repeated failures quarantine the item."""
    items = select_items(args)
    verdicts = load_json(STATE_DIR / "qa_gaussians.json")
    retries = load_json(STATE_DIR / "qa_retries.json")
    quarantine = load_json(STATE_DIR / "qa_quarantine.json")
    n_fail = 0
    n_quarantined = 0
    for it in items:
        d = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        g, ext = load_gaussians(d, it["id"])
        if g is None:
            continue
        prev = verdicts.get(it["id"], {})
        if prev.get("pass") and not args.rescan:
            continue
        view_stats = {}
        montage_ok = False
        try:
            prim_stats, reasons = gaussian_primitive_stats(g)
            rendered = render_splat_views(g)
            if rendered is None:
                reasons.append("no renderable gaussians")
            else:
                masks, weights, colours = rendered
                view_stats = silhouette_view_metrics(masks, weights, colours)
                reasons += gaussians_view_reasons(view_stats)
                labels = [f"{i+1}" for i in range(len(masks))]
                write_view_montage(
                    colours, d / f"{it['id']}.{GS_MONTAGE}",
                    f"{it['id']} 3DGS turntable", GS_VIEW_RES, labels)
                montage_ok = True
        except Exception as e:
            prim_stats = {}
            reasons = [f"load error: {e}"]
        passed = not reasons
        verdicts[it["id"]] = {
            "pass": passed, "container": ext,
            **prim_stats, **view_stats,
            "reasons": reasons,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if not passed:
            n_fail += 1
            if getattr(args, "no_wipe", False):
                pass  # calibration mode: keep artifacts for human inspection
            else:
                # Any failure here is worth another decode: wipe containers so
                # the mesh phase regenerates with a perturbed seed.
                wipe_gaussians(d, it["id"])
                if quarantine_if_exhausted(retries, quarantine, it["id"],
                                           reasons):
                    n_quarantined += 1
        print(f"{it['id']}: {'PASS' if passed else 'FAIL ' + '; '.join(reasons)}")
        save_json(STATE_DIR / "qa_gaussians.json", verdicts)
    save_json(STATE_DIR / "qa_retries.json", retries)
    save_json(STATE_DIR / "qa_quarantine.json", quarantine)
    print(f"gaussians QA: {len(verdicts)} verdicts, {n_fail} new failures, "
          f"{n_quarantined} newly quarantined ({len(quarantine)} total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["images", "meshes", "gaussians"], required=True)
    ap.add_argument("--geographies", nargs="*", default=None,
                    choices=list(MANIFESTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit item ids (overrides geographies/limit)")
    ap.add_argument("--rescan", action="store_true",
                    help="re-check items that already have verdicts")
    ap.add_argument("--no-wipe", action="store_true",
                    help="record failures without deleting outputs "
                         "(calibration / human review mode)")
    ap.add_argument("--no-nsfw", action="store_true",
                    help="opt out of the NSFW screen; the run is then "
                         "explicitly uncertified")
    content_policy.add_policy_args(ap)
    args = ap.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    {"images": run_images, "meshes": run_meshes,
     "gaussians": run_gaussians}[args.mode](args)


if __name__ == "__main__":
    main()
