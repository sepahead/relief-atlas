"""Solid-backdrop matting: convert FLUX.2 studio shots to RGBA for TRELLIS.2.

TRELLIS.2 skips its (gated) RMBG-2.0 background remover when the input image
already carries a real alpha channel, so we derive alpha ourselves.

The v1 recipe (pure-white backdrop + contact shadows + white flood-fill)
leaked an opaque floor puddle into the alpha channel, which TRELLIS.2
faithfully extruded into white slab geometry under every asset. The prompt
template now forces a flat uniform solid magenta backdrop with no floor plane
and no cast shadows; this module estimates the backdrop color from the border
ring and mattes with hysteresis thresholds, morphological cleanup, small-hole
filling and edge despill against the estimated backdrop color.

Deterministic; needs only numpy/scipy/PIL.
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

BORDER_PX = 4            # border ring thickness used to estimate backdrop color
T_LO = 26.0              # color distance below this = certainly background
T_HI = 80.0              # color distance above this = certainly foreground
MIN_PART_FRAC = 0.0004   # drop detached foreground specks below this image fraction
MAX_HOLE_FRAC = 0.005    # fill enclosed holes below this fraction; larger stay transparent
FEATHER_PX = 1.0         # alpha edge feather radius


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


def solid_bg_to_rgba(img: Image.Image) -> tuple[Image.Image, dict]:
    """Matte an image shot on the standard solid backdrop to RGBA.

    Returns (rgba_image, stats).
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    n_px = h * w

    bg_color, bg_std = estimate_backdrop(rgb)
    dist = np.abs(rgb - bg_color).max(axis=-1)

    # Hysteresis background region: geodesic dilation (flood fill) from
    # certainly-background border pixels through everything up to T_HI, so
    # gentle vignettes or residual shadow gradients near the base can never
    # survive as an opaque blob.
    certain_bg = dist < T_LO
    passable = dist < T_HI
    seed = _border_seed(certain_bg)
    if not seed.any():  # heavily graded backdrop: relax the seed threshold
        seed = _border_seed(dist < max(48.0, 4 * bg_std))
    bg_mask = ndimage.binary_propagation(seed, mask=passable)

    fg = ~bg_mask

    # Drop tiny detached specks (stray debris dots become floating mesh blobs).
    lbl, n_parts = ndimage.label(fg)
    specks_removed = 0
    if n_parts:
        sizes = np.bincount(lbl.ravel())[1:]
        keep_ids = np.nonzero(sizes >= MIN_PART_FRAC * n_px)[0] + 1
        specks_removed = n_parts - len(keep_ids)
        fg = np.isin(lbl, keep_ids)

    # Fill small enclosed holes (pinholes between rocks -> noisy cavities);
    # larger enclosed gaps are real see-through space and stay transparent.
    holes_filled = 0
    hlbl, n_holes = ndimage.label(bg_mask)
    if n_holes:
        border_ids = set(np.unique(hlbl[_border_touch_mask(hlbl)]).tolist())
        hsizes = np.bincount(hlbl.ravel())[1:]
        small_enclosed = [
            i + 1 for i in range(n_holes)
            if (i + 1) not in border_ids and hsizes[i] < MAX_HOLE_FRAC * n_px
        ]
        holes_filled = len(small_enclosed)
        if small_enclosed:
            fg |= np.isin(hlbl, small_enclosed)

    # Soft alpha from distance, masked to the cleaned foreground support
    # (dilated a little so anti-aliased edge pixels keep their partial alpha).
    soft = _smoothstep((dist - T_LO) / (T_HI - T_LO))
    support = ndimage.binary_dilation(fg, iterations=2)
    alpha = np.where(support, soft, 0.0)
    alpha[fg & (dist >= T_HI)] = 1.0

    if FEATHER_PX > 0:
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(FEATHER_PX))
        alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
        alpha[~support] = 0.0
    else:
        alpha = np.clip(alpha, 0.0, 1.0)

    # Despill: unmix only genuine soft-edge pixels (the anti-aliased boundary
    # band against the backdrop). Interior and transparent pixels pass through
    # untouched — aggressive global unmixing corrodes object colors.
    edge_band = (alpha > 0.2) & (alpha < 0.97)
    a = alpha[..., None]
    safe_a = np.clip(a, 0.35, 1.0)
    unmixed = (rgb - (1.0 - safe_a) * bg_color) / safe_a
    out_rgb = np.where(edge_band[..., None], np.clip(unmixed, 0, 255), rgb)

    rgba = np.dstack([out_rgb.astype(np.uint8), (alpha * 255).astype(np.uint8)])
    stats = {
        "backdrop_rgb": [int(c) for c in bg_color],
        "backdrop_std": round(bg_std, 2),
        "fg_fraction": round(float((alpha > 0.5).mean()), 4),
        "parts_kept": int(len(keep_ids)) if n_parts else 0,
        "specks_removed": int(specks_removed),
        "holes_filled": int(holes_filled),
    }
    return Image.fromarray(rgba, "RGBA"), stats


def _border_touch_mask(mask: np.ndarray) -> np.ndarray:
    touch = np.zeros_like(mask)
    touch[0] = mask[0]
    touch[-1] = mask[-1]
    touch[:, 0] |= mask[:, 0]
    touch[:, -1] |= mask[:, -1]
    return touch


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

    out, st = solid_bg_to_rgba(Image.open(sys.argv[1]))
    out.save(sys.argv[2])
    print(json.dumps(st, indent=2))
    if len(sys.argv) > 3:
        composite_checker(out).save(sys.argv[3])
