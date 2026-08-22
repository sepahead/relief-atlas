"""Backdrop matting: convert FLUX.2 studio shots to RGBA for TRELLIS.2.

TRELLIS.2 skips its (gated) RMBG-2.0 background remover when the input image
already carries a real alpha channel, so we derive alpha ourselves.

History:
  v1  white backdrop + contact shadows + white flood-fill -> floor puddles
      leaked into alpha; TRELLIS.2 extruded them into white slab geometry.
  v2  solid magenta backdrop + chroma hysteresis -> magenta bounce light on
      the object sits close to the backdrop in color space, so global
      smoothstep alpha turned low-contrast object interiors semi-transparent.
  v3  light neutral gray backdrop + BiRefNet saliency matting (transformers,
      MPS) as the primary method; the chroma hysteresis matte remains as a
      fallback. Alpha is binary in the core and softened only within a narrow
      boundary band, so object interiors can never go translucent.

Deterministic; needs numpy/scipy/PIL (+ transformers/torch for BiRefNet).
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

BORDER_PX = 4            # border ring thickness used to estimate backdrop color
T_LO = 26.0              # chroma: color distance below this = certainly background
T_HI = 64.0              # chroma: color distance above this = certainly foreground
MIN_PART_FRAC = 0.0004   # drop detached foreground specks below this image fraction
MAX_HOLE_FRAC = 0.005    # fill enclosed holes below this fraction; larger stay transparent
FEATHER_PX = 1.0         # alpha edge feather radius
SOFT_BAND_PX = 2         # width of the boundary band that keeps partial alpha

_BIREFNET = None         # lazy-loaded model cache


def estimate_backdrop(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Median color of the border ring + its noise std."""
    ring = np.concatenate([
        arr[:BORDER_PX].reshape(-1, 3),
        arr[-BORDER_PX:].reshape(-1, 3),
        arr[:, :BORDER_PX].reshape(-1, 3),
        arr[:, -BORDER_PX:].reshape(-1, 3),
    ])
    return np.median(ring, axis=0), float(ring.std())


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _border_seed(certain_bg: np.ndarray) -> np.ndarray:
    seed = np.zeros_like(certain_bg)
    seed[0] = certain_bg[0]
    seed[-1] = certain_bg[-1]
    seed[:, 0] |= certain_bg[:, 0]
    seed[:, -1] |= certain_bg[:, -1]
    return seed


def _border_touch_mask(mask: np.ndarray) -> np.ndarray:
    touch = np.zeros_like(mask)
    touch[0] = mask[0]
    touch[-1] = mask[-1]
    touch[:, 0] |= mask[:, 0]
    touch[:, -1] |= mask[:, -1]
    return touch


# ------------------------------------------------------------------ methods

_BIREFNET_DIR = Path(__file__).resolve().parents[2] / "meshmaker" / "local" / "models" / "BiRefNet"
# MPS triggers a float64 op inside the backbone; CPU at ~7 s/frame is fine.
_BIREFNET_DEVICE = os.environ.get("MATTING_DEVICE", "cpu")


def birefnet_alpha(img: Image.Image) -> np.ndarray:
    """Saliency matte from local BiRefNet weights, float alpha in [0, 1]."""
    global _BIREFNET
    import torch

    if _BIREFNET is None:
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            str(_BIREFNET_DIR), trust_remote_code=True, local_files_only=True)
        model = model.float().eval()
        _BIREFNET = model.to(_BIREFNET_DEVICE)

    im = img.convert("RGB").resize((1024, 1024), Image.LANCZOS)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(_BIREFNET_DEVICE)
    with torch.no_grad():
        preds = _BIREFNET(t)
        p = preds[-1] if isinstance(preds, list) else preds
        small = torch.sigmoid(p)[0, 0].cpu().numpy()
    up = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize(
        (img.width, img.height), Image.LANCZOS), dtype=np.float32) / 255.0
    return np.clip(up, 0.0, 1.0)


def chroma_alpha(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Chroma hysteresis fallback. Returns (soft_alpha, fg_mask, bg_std)."""
    bg_color, bg_std = estimate_backdrop(rgb)
    dist = np.abs(rgb - bg_color).max(axis=-1)

    # Flood fill from certainly-background border pixels through everything up
    # to T_HI, so vignettes or residual gradients never survive as blobs.
    certain_bg = dist < T_LO
    passable = dist < T_HI
    seed = _border_seed(certain_bg)
    if not seed.any():
        seed = _border_seed(dist < max(48.0, 4 * bg_std))
    bg_mask = ndimage.binary_propagation(seed, mask=passable)

    soft = _smoothstep((dist - T_LO) / (T_HI - T_LO))
    soft[bg_mask] = 0.0
    return soft, ~bg_mask, bg_std


# ------------------------------------------------------- shared postprocess

def _finish_rgba(rgb: np.ndarray, fg_soft: np.ndarray,
                 bg_color: np.ndarray | None) -> tuple[Image.Image, dict]:
    h, w = rgb.shape[:2]
    n_px = h * w
    fg = fg_soft > 0.5

    # Drop tiny detached specks (stray debris dots become floating mesh blobs).
    lbl, n_parts = ndimage.label(fg)
    specks_removed = 0
    parts_kept = 0
    if n_parts:
        sizes = np.bincount(lbl.ravel())[1:]
        keep_ids = np.nonzero(sizes >= MIN_PART_FRAC * n_px)[0] + 1
        specks_removed = n_parts - len(keep_ids)
        parts_kept = len(keep_ids)
        fg = np.isin(lbl, keep_ids)

    # Fill small enclosed holes (pinholes -> noisy cavities); larger enclosed
    # gaps are real see-through space and stay transparent.
    holes_filled = 0
    inv_lbl, n_inv = ndimage.label(~fg)
    if n_inv:
        border_ids = set(np.unique(inv_lbl[_border_touch_mask(inv_lbl)]).tolist())
        isizes = np.bincount(inv_lbl.ravel())[1:]
        small_enclosed = [
            i + 1 for i in range(n_inv)
            if (i + 1) not in border_ids and isizes[i] < MAX_HOLE_FRAC * n_px
        ]
        holes_filled = len(small_enclosed)
        if small_enclosed:
            fg |= np.isin(inv_lbl, small_enclosed)

    # Binary core; partial alpha only inside a narrow boundary band so object
    # interiors can never turn translucent (the v2 failure mode).
    inner = ndimage.binary_erosion(fg, iterations=1)
    outer = ndimage.binary_dilation(fg, iterations=SOFT_BAND_PX)
    band = outer & ~inner
    alpha = np.where(fg, 1.0, 0.0)
    alpha[band] = np.maximum(alpha[band], fg_soft[band])

    if FEATHER_PX > 0:
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(FEATHER_PX))
        alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
        alpha[~outer] = 0.0
        alpha[inner] = np.maximum(alpha[inner], 1.0)

    # Despill: unmix only genuine soft-edge pixels against the backdrop.
    if bg_color is not None:
        edge_band = (alpha > 0.2) & (alpha < 0.97)
        a = alpha[..., None]
        safe_a = np.clip(a, 0.35, 1.0)
        unmixed = (rgb - (1.0 - safe_a) * bg_color) / safe_a
        out_rgb = np.where(edge_band[..., None], np.clip(unmixed, 0, 255), rgb)
    else:
        out_rgb = rgb

    rgba = np.dstack([out_rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    stats = {
        "fg_fraction": round(float((alpha > 0.5).mean()), 4),
        "parts_kept": int(parts_kept),
        "specks_removed": int(specks_removed),
        "holes_filled": int(holes_filled),
    }
    return Image.fromarray(rgba, "RGBA"), stats


def matte_image(img: Image.Image, method: str = "auto") -> tuple[Image.Image, dict]:
    """Matte a studio shot to RGBA. Returns (rgba_image, stats).

    method: 'birefnet' | 'chroma' | 'auto' (birefnet, falling back to chroma
    on any model error).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    use_birefnet = method in ("birefnet", "auto")
    if use_birefnet:
        try:
            fg_soft = birefnet_alpha(img)
            out, stats = _finish_rgba(rgb, fg_soft, None)
            stats["method"] = "birefnet"
            return out, stats
        except Exception as e:
            if method == "birefnet":
                raise
            print(f"birefnet unavailable ({e}); falling back to chroma")
    soft, fg, bg_std = chroma_alpha(rgb)
    bg_color, _ = estimate_backdrop(rgb)
    out, stats = _finish_rgba(rgb, np.where(fg, soft, 0.0), bg_color)
    stats["method"] = "chroma"
    stats["backdrop_std"] = round(bg_std, 2)
    return out, stats


# Back-compat alias used by earlier revisions of the runners.
def solid_bg_to_rgba(img: Image.Image) -> tuple[Image.Image, dict]:
    return matte_image(img, method="chroma")


def has_object(rgba: Image.Image, min_frac: float = 0.01) -> bool:
    """True if the alpha mask covers at least min_frac of the image."""
    alpha = np.asarray(rgba)[:, :, 3]
    return (alpha > 127).mean() >= min_frac


def composite_checker(rgba: Image.Image, cell: int = 32) -> Image.Image:
    """Composite RGBA over a gray checkerboard for visual QA."""
    arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // cell) + (xx // cell)) % 2 * 40 + 180).astype(np.float32)
    out = arr[..., :3] * (arr[..., 3:] / 255.0) + checker[..., None] * (1 - arr[..., 3:] / 255.0)
    return Image.fromarray(out.astype(np.uint8), "RGB")


if __name__ == "__main__":
    import json
    import sys

    out, st = matte_image(Image.open(sys.argv[1]))
    out.save(sys.argv[2])
    print(json.dumps(st, indent=2))
    if len(sys.argv) > 3:
        composite_checker(out).save(sys.argv[3])
