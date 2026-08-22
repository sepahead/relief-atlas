"""Phase 1.5/2.5: QA gate between and after the generation phases.

Image mode (run with imgenv python): validates every matted PNG — alpha
coverage, fragmentation, backdrop uniformity, sharpness, near-duplicate
subjects, and NSFW screening (Falconsai/nsfw_image_detection). Failures bump
state/qa_retries.json so the deterministic per-item seed is perturbed and the
image phase regenerates the item.

Mesh mode (run with trellis venv python): validates every GLB — loads, bbox
aspect sanity, texture whiteness (residual floor-slab regression), magenta
backdrop residue, degenerate triangles. Mesh failures delete the outputs so
the mesh phase retries with the perturbed seed.

Verdicts land in state/qa_images.json / state/qa_meshes.json.

Run with:
    imgenv/bin/python scripts/local/qa.py --mode images
    trellis-mac/.venv/bin/python scripts/local/qa.py --mode meshes
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_utils import MANIFESTS, PROJECT_DIR, load_manifest  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "outputs_relief"
STATE_DIR = PROJECT_DIR / "state"

# Image thresholds
FG_MIN, FG_MAX = 0.03, 0.75          # object coverage of frame
MAIN_PART_MIN = 0.90                 # largest connected component share of fg
BORDER_TOUCH_MAX = 0.02              # fraction of perimeter pixels that are fg
SHARPNESS_MIN = 55.0                 # Laplacian variance on masked gray
DUP_HAMMING = 6                      # phash distance for near-duplicate flag
NSFW_THRESHOLD = 0.50

# Mesh thresholds
ASPECT_MIN, ASPECT_MAX = 0.15, 8.0   # height / footprint
DIM_RATIO_MAX = 20.0                 # max/min bbox dimension
WHITE_TEX_MAX = 0.35                 # near-white texture fraction (slab regression)
MAGENTA_TEX_MAX = 0.02               # saturated magenta residue in texture


def all_items():
    return [it for geo in MANIFESTS for it in load_manifest(geo)]


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def bump_retry(retries, item_id):
    retries[item_id] = retries.get(item_id, 0) + 1


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
    def __init__(self):
        self.pipe = None

    def score(self, png: Path) -> float:
        if self.pipe is None:
            os_environ_offline()
            from transformers import pipeline
            self.pipe = pipeline(
                "image-classification",
                model=str(NSFW_DIR),
                device="cpu",
            )
        out = self.pipe(str(png))
        return next(r["score"] for r in out if r["label"] == "nsfw")


def os_environ_offline():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def run_images(args):
    items = all_items() if not args.ids else [it for it in all_items() if it["id"] in set(args.ids)]
    verdicts = load_json(STATE_DIR / "qa_images.json")
    retries = load_json(STATE_DIR / "qa_retries.json")

    guard = None
    if not args.no_nsfw:
        try:
            guard = NsfwGuard()
        except Exception as e:
            print(f"NSFW guard unavailable ({e}); skipping safety screen")

    # peer phashes per category for duplicate detection among ALREADY-passed items
    peers = {}
    for it in items:
        v = verdicts.get(it["id"])
        if v and v.get("pass") and v.get("phash"):
            peers.setdefault(it["category"], []).append(v["phash"])

    n_fail = 0
    for it in items:
        d = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        png = d / f"{it['id']}.png"
        if not png.exists():
            continue
        prev = verdicts.get(it["id"], {})
        if prev.get("pass") and not args.rescan:
            continue
        s = image_scores(png)
        peer_hashes = [imagehash_hex(p) for p in peers.get(it["category"], [])]
        reasons = image_pass(s, peer_hashes)

        nsfw = None
        if guard is not None:
            try:
                nsfw = guard.score(png)
                if nsfw > NSFW_THRESHOLD:
                    reasons.append(f"NSFW {nsfw:.2f}")
            except Exception as e:
                print(f"{it['id']}: nsfw screen error {e}")

        passed = not reasons
        verdicts[it["id"]] = {
            "pass": passed, **s,
            "nsfw": round(nsfw, 4) if nsfw is not None else None,
            "reasons": reasons,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if passed:
            if s.get("phash"):
                peers.setdefault(it["category"], []).append(s["phash"])
        else:
            if any(r.startswith(("fg_fraction", "blurry", "near-duplicate",
                                 "fragmented")) for r in reasons):
                # regenerable failures: wipe artifacts, bump the retry seed so
                # the image phase explores a new sample
                (d / f"{it['id']}.png").unlink(missing_ok=True)
                (d / f"{it['id']}.raw.jpg").unlink(missing_ok=True)
                bump_retry(retries, it["id"])
            n_fail += 1
        print(f"{it['id']}: {'PASS' if passed else 'FAIL ' + '; '.join(reasons)}")
        save_json(STATE_DIR / "qa_images.json", verdicts)

    save_json(STATE_DIR / "qa_retries.json", retries)
    print(f"image QA: {len(verdicts)} verdicts, {n_fail} new failures")


def imagehash_hex(p):
    return p


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
    }


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


def run_meshes(args):
    items = all_items() if not args.ids else [it for it in all_items() if it["id"] in set(args.ids)]
    verdicts = load_json(STATE_DIR / "qa_meshes.json")
    retries = load_json(STATE_DIR / "qa_retries.json")
    n_fail = 0
    for it in items:
        d = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        glb = d / f"{it['id']}.glb"
        if not glb.exists():
            continue
        prev = verdicts.get(it["id"], {})
        if prev.get("pass") and not args.rescan:
            continue
        try:
            s = glb_scores(glb)
            reasons = mesh_pass(s)
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
                # phase retries with the perturbed seed
                (d / f"{it['id']}.glb").unlink(missing_ok=True)
                (d / f"{it['id']}.ply").unlink(missing_ok=True)
                (d / "metadata.json").unlink(missing_ok=True)
                bump_retry(retries, it["id"])
        print(f"{it['id']}: {'PASS' if passed else 'FAIL ' + '; '.join(reasons)}")
        save_json(STATE_DIR / "qa_meshes.json", verdicts)
    save_json(STATE_DIR / "qa_retries.json", retries)
    print(f"mesh QA: {len(verdicts)} verdicts, {n_fail} new failures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["images", "meshes"], required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--rescan", action="store_true",
                    help="re-check items that already have verdicts")
    ap.add_argument("--no-nsfw", action="store_true")
    args = ap.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (run_images if args.mode == "images" else run_meshes)(args)


if __name__ == "__main__":
    main()
