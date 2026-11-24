"""Reference-anchored emblem transfer for TRELLIS.2 meshes.

TRELLIS.2 texture lifting smooths small high-frequency decals (red crosses,
roundels, insignia, lettering) into the base paint, even when the reference
PNG shows them crisply. Retrying with a new seed does not fix that: it is a
resolution prior of the texture latent, not a bad sample.

This stage projects true reference pixels back onto the camera-visible faces
of the baked GLB texture. It is systemic, not per-item hand work:

* Detection gate: small concentrated high-contrast insignia blobs (saturated
  crosses/roundels, thin dark unit codes on uniform light panels) well inside
  the reference foreground. Windows, open doorways, roof lightbars, full-length
  body stripes, tyres and specular lamps are rejected by aspect, bbox fill,
  linear span, foreground interior margin and host-panel annular uniformity.
  No blobs -> byte-identical no-op, so non-emblem items are untouched.
* Camera fit: core-mask silhouette IoU search over azimuth x elevation with
  explicit mirror/antipode symmetry augmentation proposes distinct candidate
  poses; candidates are re-ranked using studio headlight + depth-cavity ambient
  occlusion (`grey_opening` front-hull recess) across multi-cue luminance,
  high-pass structure, edge gradients and opponent chromaticity. Ambiguous
  poses are never projected.
* Projection: per-pixel ray/face intersection with rendered face-id + position
  passes, full-surface depth-discontinuity & rim erosion, incidence-angle
  obliquity gating (<= 45 deg), per-blob 3D surface normal concentration and
  depth-continuity checks, shared-UV island conflict rejection, and local
  surround luminance matching. Preserves RGBA baseColorTexture alpha when
  present.
* Atomic verification & promotion: both GLB and 3DGS containers are staged to
  temporary siblings and promoted via `durable_replace` only after a real 5-view
  `moderngl` before/after render verification confirms positive decal presence
  in the fitted pose and zero bleed onto opposing, orthogonal or roof surfaces.

Runs fully offline with numpy/PIL/moderngl/trimesh (mesh runtime).
"""

import argparse
import contextlib
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pipeline_utils import GS_EXTENSIONS, durable_replace, file_sha256, write_json


@contextlib.contextmanager
def _mgl_context():
    """Create and deterministically release a standalone OpenGL 3.3 context."""
    import moderngl
    ctx = moderngl.create_standalone_context(require=330)
    try:
        with ctx:
            yield ctx
    finally:
        try:
            ctx.release()
        except Exception:
            pass

# Blob gate thresholds
BLOB_MIN_FRAC = 0.0005
BLOB_MAX_FRAC = 0.05
DARK_BLOB_MAX_FRAC = 0.015
DARK_STROKE_MAX_FRAC = 0.012
BLOB_MAX_ASPECT = 2.6
BLOB_MIN_FILL = 0.22
BLOB_MAX_SPAN = 0.22
BLOB_EDGE_MARGIN_FRAC = 0.015
BLOB_FG_ENCLOSURE_MIN = 0.90
BLOB_RING_STD_MAX = 28.0
BLOB_CONTRAST_MIN = 140.0
BLOB_INTERNAL_LUMA_STD_MAX = 15.0
DARK_RING_LUMA_MIN = 185.0

# Second-pass multi-tone badge detector (crests, shields, roundels, lettering
# on neutral, painted or dark host panels).
PANEL_BADGE_SIGMA_FRAC = 0.022
PANEL_BADGE_BG_DEV_MAX = 42.0
PANEL_BADGE_DEV_MIN = 48.0
PANEL_BADGE_MAX_FRAC = 0.035
PANEL_BADGE_MAX_ASPECT = 2.8
PANEL_BADGE_MAX_SPAN = 0.18
PANEL_BADGE_ENCLOSURE_MIN = 0.80
PANEL_BADGE_RING_CLEAN_MIN = 0.20
PANEL_BADGE_RING_STD_MAX = 16.5
PANEL_BADGE_CONTRAST_MIN = 80.0
PANEL_BADGE_CHROMA_MIN = 0.24
PANEL_BADGE_HUE_SEP_DEG = 32.0
PANEL_BADGE_CHROMA_SHARE_MIN = 0.08

# Decal-restricted projection and incidence/coherence gates
DECAL_FEATHER_MIN = 0.02
MAX_PROJECTED_TEXEL_SHARE = 0.25
MAX_INCIDENCE_DEG = 78.0
FEATHER_INCIDENCE_DEG = 70.0
BLOB_COVERAGE_MIN = 0.65
BLOB_NORMAL_CONC_MIN = 0.92
ROOF_NORMAL_Z_MAX = 0.85
FLOOR_NORMAL_Z_MIN = -0.55
CORE_ASPECT_TOL = 0.12
CORE_ALIGNED_IOU_MIN = 0.78

# Write-back & verification thresholds
WRITEBACK_TOLERANCE = 12.0
FIT_IOU_MIN = 0.5
TRANSFER_MATCH_MIN = 0.5
MULTIVIEW_BLEED_MAX = 0.0005
ROOF_BLEED_MAX = 0.001
RENDER_SIZE = 1024

# Metadata record version.
# v4 adds true-insignia annular gating, cavity-shaded multi-cue pose fitting
# with mandatory symmetry candidates, 3D surface obliquity/planarity/depth
# continuity checks, atomic 3DGS staging, and 5-view render verification.
EMBLEM_SCHEMA = 4


# ------------------------------------------------------------ blob detection

def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage
        return ndimage.binary_dilation(mask, iterations=radius)
    except ImportError:
        img = Image.fromarray((np.asarray(mask, bool) * 255).astype(np.uint8))
        for _ in range(radius):
            img = img.filter(ImageFilter.MaxFilter(3))
        return np.asarray(img) > 0


def _fg_distance(fg: np.ndarray) -> np.ndarray:
    fg = np.asarray(fg, dtype=bool)
    if not (~fg).any():
        return np.full(fg.shape, 999.0, dtype=np.float32)
    try:
        from scipy import ndimage
        return ndimage.distance_transform_edt(fg).astype(np.float32)
    except ImportError:
        dist = np.zeros(fg.shape, dtype=np.float32)
        cur = Image.fromarray((fg * 255).astype(np.uint8))
        for step in range(1, 20):
            arr = np.asarray(cur) > 0
            if not arr.any():
                break
            dist[arr] = float(step)
            cur = cur.filter(ImageFilter.MinFilter(3))
        return dist


def _gauss_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    try:
        from scipy import ndimage
        return ndimage.gaussian_filter(a, sigma=sigma, mode="nearest").astype(np.float32)
    except ImportError:
        lo, hi = float(a.min()), float(a.max())
        span = max(hi - lo, 1e-6)
        img = Image.fromarray(np.clip((a - lo) / span * 255.0, 0, 255).astype(np.uint8))
        out = np.asarray(img.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
        return out / 255.0 * span + lo


def _hue_deg(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6.0,
                 np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0))
    return (h * 60.0).astype(np.float32)


def _local_panel_background(rgb: np.ndarray, fg: np.ndarray, sigma: float):
    """Robust masked low-pass estimate of the host-panel colour around each pixel.

    Two reweighting passes drop strongly deviating pixels (insignia, windows,
    trim) from the estimate so a badge never dilutes its own background.
    Returns `(bg_rgb, deviation)`.
    """
    fgf = fg.astype(np.float32)
    weight = fgf.copy()
    bg = rgb.copy()
    dev = np.zeros(fg.shape, np.float32)
    for _ in range(3):
        den = _gauss_blur_2d(weight, sigma)
        safe = np.maximum(den, 1e-3)
        bg = np.stack([_gauss_blur_2d(rgb[..., c] * weight, sigma) / safe
                       for c in range(3)], axis=-1)
        # Where every neighbour was rejected, fall back to the plain fg average.
        den_fg = np.maximum(_gauss_blur_2d(fgf, sigma), 1e-3)
        plain = np.stack([_gauss_blur_2d(rgb[..., c] * fgf, sigma) / den_fg
                          for c in range(3)], axis=-1)
        weak = (den < 0.08 * np.maximum(den_fg, 1e-3))[..., None]
        bg = np.where(weak, plain, bg)
        dev = np.linalg.norm(rgb - bg, axis=-1)
        weight = fgf * (dev < PANEL_BADGE_BG_DEV_MAX)
    return bg, dev


def _component_slices(lbl: np.ndarray, n: int) -> dict:
    """Bounding slices `{label: ((y0, y1), (x0, x1))}` for labels 1..n."""
    out = {}
    if n <= 0:
        return out
    try:
        from scipy import ndimage
        for i, sl in enumerate(ndimage.find_objects(lbl, max_label=n), start=1):
            if sl is not None:
                out[i] = ((sl[0].start, sl[0].stop), (sl[1].start, sl[1].stop))
        return out
    except ImportError:
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lbl == i)
            if len(ys):
                out[i] = ((int(ys.min()), int(ys.max()) + 1), (int(xs.min()), int(xs.max()) + 1))
        return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background pockets fully enclosed by `mask` (crop-local)."""
    m = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage
        return ndimage.binary_fill_holes(m)
    except ImportError:
        pad = np.pad(~m, 1, constant_values=True)
        lbl, _n, _s = _label_fallback(pad)
        outside = lbl == lbl[0, 0]
        return ~outside[1:-1, 1:-1]


def _closed_components(binary: np.ndarray, fg: np.ndarray, grow: int):
    """Close small gaps, label, and return hole-filled per-component masks."""
    closed = _dilate_bool(binary, grow) if grow > 0 else np.asarray(binary, bool)
    if grow > 0:
        img = Image.fromarray((closed * 255).astype(np.uint8))
        for _ in range(grow):
            img = img.filter(ImageFilter.MinFilter(3))
        closed = np.asarray(img) > 0
    closed &= fg
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(closed)
    except ImportError:
        lbl, n, _sizes = _label_fallback(closed)
    for i, ((y_a, y_b), (x_a, x_b)) in sorted(_component_slices(lbl, n).items()):
        comp = np.zeros(fg.shape, bool)
        comp[y_a:y_b, x_a:x_b] = (_fill_holes(lbl[y_a:y_b, x_a:x_b] == i)
                                  & fg[y_a:y_b, x_a:x_b])
        yield comp


def _panel_badge_entries(rgb, fg, fg_dist, fg_extent, edge_margin, has_bg, n_px, taken):
    """Second-pass detector for multi-tone crests, shields, roundels and lettering.

    The single-tone pass needs a saturated or dark stroke on a neutral, uniform
    panel. Real insignia are often multi-coloured (seal with gold/red/white
    interior), sit on painted bodies (navy lettering on an orange boom, a
    roundel on a red fuselage) or on dark armour (gold shield on grey), or are
    studio-lit to a pastel tone (red cross on a sloped white hood).
    """
    sigma = max(6.0, PANEL_BADGE_SIGMA_FRAC * fg_extent)
    bg, dev = _local_panel_background(rgb, fg, sigma)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    bg_luma = 0.299 * bg[..., 0] + 0.587 * bg[..., 1] + 0.114 * bg[..., 2]
    hue = _hue_deg(rgb)
    bg_hue = _hue_deg(bg)
    bg_mx = bg.max(axis=-1)
    bg_chr = (bg_mx - bg.min(axis=-1)) / np.maximum(bg_mx, 1e-6)
    hd = np.abs(hue - bg_hue) % 360.0
    hd = np.minimum(hd, 360.0 - hd)

    interior = fg & (fg_dist >= edge_margin if has_bg else fg)
    c_chrom = (interior & (dev >= PANEL_BADGE_DEV_MIN)
               & (chroma >= PANEL_BADGE_CHROMA_MIN) & (mx >= 45.0)
               & ((bg_chr < 0.18) | (hd >= PANEL_BADGE_HUE_SEP_DEG)))
    c_bright = interior & (dev >= 65.0) & ((luma - bg_luma) >= 60.0) & (luma >= 155.0)
    c_pdark = interior & (bg_chr >= 0.45) & (bg_luma >= 100.0) & (dev >= 85.0) & (luma <= 70.0)
    cand = c_chrom | c_bright | c_pdark

    px_min = max(24, int(round(0.0004 * fg.shape[0] * fg.shape[1])))
    is_full_res = max(fg.shape) >= 512

    def eval_comp(comp, claimed):
        px = int(comp.sum())
        frac = float(px) / n_px
        if px < px_min or frac < BLOB_MIN_FRAC:
            return None, "small"
        if frac > PANEL_BADGE_MAX_FRAC:
            return None, "large"
        ys, xs = np.nonzero(comp)
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        span_px = max(w, h)
        if is_full_res and span_px < 18:
            return None, "small"
        if has_bg and span_px / max(fg_extent, 1.0) > PANEL_BADGE_MAX_SPAN:
            return None, "large"
        aspect = span_px / max(min(w, h), 1)
        fill = float(px) / float(max(w * h, 1))
        if aspect > PANEL_BADGE_MAX_ASPECT or fill < 0.24:
            return None, "shape"
        if has_bg and float(fg_dist[comp].min()) < edge_margin:
            return None, "edge"
        if int((comp & claimed).sum()) > 0.10 * px:
            return None, "taken"

        raw = comp & (dev >= 45.0)
        if int(raw.sum()) < max(12, int(round(0.25 * px_min))):
            return None, "sparse"

        r_in = max(3, int(round(0.08 * span_px)))
        r_out = r_in + max(6, int(round(0.24 * span_px)))
        ann = _dilate_bool(comp, r_out) & ~_dilate_bool(comp, r_in)
        if not ann.any():
            return None, "ring"
        if has_bg and float(fg[ann].mean()) < PANEL_BADGE_ENCLOSURE_MIN:
            return None, "enclosure"
        ann_fg = ann & fg
        ring = ann_fg & ~cand & (dev < PANEL_BADGE_BG_DEV_MAX) & ~taken
        rc = float(ring.sum()) / float(max(ann_fg.sum(), 1))
        if int(ring.sum()) < 24 or rc < PANEL_BADGE_RING_CLEAN_MIN:
            return None, "ring_clutter"
        if is_full_res and rc < 0.45 and int(ring.sum()) < 120:
            return None, "ring_clutter"

        r_std = float(luma[ring].std())
        r_cstd = float(rgb[ring].std(axis=0).max())
        if r_std > (PANEL_BADGE_RING_STD_MAX if rc >= 0.45 else 15.5):
            return None, "ring_std"

        panel = np.median(rgb[ring], axis=0)
        p_luma = float(0.299 * panel[0] + 0.587 * panel[1] + 0.114 * panel[2])
        if p_luma < 28.0:
            return None, "dark_panel"
        p_mx = float(panel.max())
        p_chr = (p_mx - float(panel.min())) / max(p_mx, 1e-6)
        p_hue = float(_hue_deg(panel[None, :])[0])

        if p_chr < 0.45:
            if rc < 0.35 or r_cstd > PANEL_BADGE_RING_STD_MAX:
                return None, "neutral_ring"
        else:
            if r_std > 13.5 or (rc < 0.55 and r_cstd > 22.0):
                return None, "painted_ring"

        raw_rgb = rgb[raw]
        raw_luma = luma[raw]
        raw_chr = chroma[raw]
        raw_hue = hue[raw]
        raw_hd = np.abs(raw_hue - p_hue) % 360.0
        raw_hd = np.minimum(raw_hd, 360.0 - raw_hd)
        dp = np.linalg.norm(raw_rgb - panel[None, :], axis=-1)
        dp_mean = float(dp.mean())
        dp_p90 = float(np.percentile(dp, 90))
        luma_spread = float(np.percentile(raw_luma, 90) - np.percentile(raw_luma, 10))
        comp_luma_p90 = float(np.percentile(luma[comp], 90))

        redness = raw_rgb[:, 0] - 0.5 * (raw_rgb[:, 1] + raw_rgb[:, 2])
        is_type_a = (p_luma >= 210.0 and p_chr < 0.12 and aspect <= 1.6
                     and float((redness >= 32.0).mean()) >= 0.55
                     and PANEL_BADGE_CONTRAST_MIN <= dp_mean <= 145.0)

        chrom_px = (raw_chr >= 0.28) & (mx[raw] >= 70.0) & ((p_chr < 0.18) | (raw_hd >= 35.0))
        has_chrom = int(chrom_px.sum()) >= max(16.0, PANEL_BADGE_CHROMA_SHARE_MIN * len(raw_luma))
        has_light = float(((raw_luma >= 155.0) & (
            (raw_luma - p_luma >= 35.0) if p_luma < 170.0 else (raw_chr < 0.22)
        )).sum()) >= max(16.0, 0.10 * len(raw_luma))
        has_dark = float(((raw_luma <= 85.0) & (p_luma - raw_luma >= 45.0)).sum()) >= max(16.0, 0.10 * len(raw_luma))
        warm_px = chrom_px & ((raw_hue <= 75.0) | (raw_hue >= 325.0))
        cool_px = chrom_px & (raw_hue >= 165.0) & (raw_hue <= 270.0)
        has_dual_hue = (float(warm_px.mean()) >= 0.15 and float(cool_px.mean()) >= 0.15)
        is_type_b = (has_chrom and (has_light or has_dark)
                     and (comp_luma_p90 >= 160.0 or has_dual_hue)
                     and luma_spread >= 50.0 and dp_p90 >= 175.0)

        has_very_dark = float((raw_luma <= 70.0).mean()) >= 0.15
        is_type_c = (p_chr >= 0.45 and p_luma >= 85.0
                     and dp_mean >= 180.0 and dp_p90 >= 220.0
                     and (has_chrom or (has_light and has_very_dark)))

        if not (is_type_a or is_type_b or is_type_c):
            return None, "content"

        return {
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "pixels": px,
            "fraction": round(frac, 5),
            "kind": "badge",
            "panel_rgb": [int(round(v)) for v in panel.tolist()],
        }, "ok"

    out = []
    claimed = taken.copy()
    retry = []
    for grow in (3, 2):
        for comp in _closed_components(cand, fg, grow):
            rec, why = eval_comp(comp, claimed)
            if rec is not None:
                out.append((rec, comp))
                claimed |= comp
            elif grow == 2 and why in ("large", "shape", "edge") and int(comp.sum()) >= 4 * px_min:
                retry.append(comp)

    for comp in retry:
        img = Image.fromarray(((cand & comp) * 255).astype(np.uint8))
        for _ in range(2):
            img = img.filter(ImageFilter.MinFilter(3))
        for _ in range(2):
            img = img.filter(ImageFilter.MaxFilter(3))
        core = (np.asarray(img) > 0) & comp
        for sub in _closed_components(core, fg, 2):
            rec_s, _ = eval_comp(sub, claimed)
            if rec_s is not None:
                out.append((rec_s, sub))
                claimed |= sub
    return out


def _oriented_extent(ys: np.ndarray, xs: np.ndarray):
    """Return (major_span, minor_span, major_unit_vec) along PCA axes of a mask."""
    if len(ys) < 2:
        return 1.0, 1.0, np.array([1.0, 0.0], dtype=np.float64)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.einsum("ij,ik->jk", centered, centered) / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v_major = eigvecs[:, 1]
    v_minor = eigvecs[:, 0]
    if v_major[0] < 0.0:
        v_major = -v_major
    proj_maj = np.einsum("ij,j->i", centered, v_major)
    proj_min = np.einsum("ij,j->i", centered, v_minor)
    major = max(1.0, float(proj_maj.max() - proj_maj.min()) + 1.0)
    minor = max(1.0, float(proj_min.max() - proj_min.min()) + 1.0)
    return major, minor, v_major


def _panel_text_entries(rgb, fg, fg_dist, fg_extent, edge_margin, has_bg, n_px, taken):
    """Third-pass detector for crisp multi-character lettering bands on uniform panels.

    Catches white-on-painted ('FEUERWEHR', 'POLIZEI', 'CRUZ ROJA', 'THW', 'DSNS'),
    dark-on-light/painted ('THW', 'DRK', 'UNHCR', 'ECHO'), and colored-on-white
    word markings whose aspect ratio (>2.8) or per-letter area bypasses compact
    badge detectors. Requires >=2 collinear thin character strokes of consistent
    height and color surrounded by a uniform host panel, automatically rejecting
    solid stripes, windows, shutters, grilles, and lattice trusses.
    """
    h_img, w_img = fg.shape
    # Use a tighter local background kernel (~8 px at 1024) than large circular
    # badges (~22 px) so lettering painted inside narrow vehicle stripes or below
    # windshields is measured against its actual supporting paint, not distant trim.
    sigma = max(3.5, 0.008 * fg_extent)
    bg, dev = _local_panel_background(rgb, fg, sigma)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    bg_luma = 0.299 * bg[..., 0] + 0.587 * bg[..., 1] + 0.114 * bg[..., 2]
    bg_mx = bg.max(axis=-1)
    bg_chr = (bg_mx - bg.min(axis=-1)) / np.maximum(bg_mx, 1e-6)

    interior = fg & (fg_dist >= edge_margin if has_bg else fg) & ~taken
    polarities = [
        interior & (dev >= 55.0) & ((luma - bg_luma) >= 45.0) & (luma >= 165.0),
        interior & (dev >= 55.0) & ((bg_luma - luma) >= 45.0) & (luma <= 105.0) & (bg_luma >= 115.0),
        interior & (dev >= 65.0) & (chroma >= 0.35) & (mx >= 80.0) & (bg_chr < 0.22) & (bg_luma >= 165.0),
    ]

    glyph_px_min = max(10, int(round(0.000025 * h_img * w_img)))
    glyph_px_max = max(glyph_px_min * 4, int(round(0.004 * n_px)))
    min_ch = 7 if max(h_img, w_img) >= 512 else 5
    max_ch = max(48, int(round(0.085 * fg_extent)))
    max_cw = max(64, int(round(0.095 * fg_extent)))
    word_px_min = max(90, int(round(0.00020 * h_img * w_img)))
    struct8 = np.ones((3, 3), dtype=bool)

    out = []
    claimed = taken.copy()

    for ink in polarities:
        if int(ink.sum()) < word_px_min:
            continue
        try:
            from scipy import ndimage
            lbl, n_c = ndimage.label(ink, structure=struct8)
        except ImportError:
            lbl, n_c, _ = _label_fallback(ink)
        if n_c < 2:
            continue
        slices = _component_slices(lbl, n_c)
        glyphs = []
        for cid, ((ya, yb), (xa, xb)) in slices.items():
            crop = lbl[ya:yb, xa:xb] == cid
            px = int(crop.sum())
            if px < glyph_px_min or px > glyph_px_max:
                continue
            ch = yb - ya
            cw = xb - xa
            if ch < min_ch or ch > max_ch or cw > max_cw:
                continue
            stroke = max_inscribed_radius(crop)
            g_fill = float(px) / float(max(cw * ch, 1))
            is_bar = (g_fill > 0.68 and (stroke / max(min(cw, ch), 1)) > 0.34)
            if is_bar:
                # Allow slender vertical bar strokes (capital "I" in POLIZEI, ICRC,
                # PUNAINEN RISTI, including condensed/perspective-foreshortened side
                # lettering up to ~11:1) to join real letters; solid square/wide blocks
                # are rejected and clusters must still contain >=2 non-bar glyphs.
                if not (ch >= 2.2 * cw and ch / max(cw, 1) <= 11.0
                        and stroke <= max(5.0, 0.012 * fg_extent)):
                    continue
            else:
                if max(cw, ch) / max(min(cw, ch), 1) > 4.5:
                    continue
                if stroke > max(5.5, 0.38 * min(cw, ch)) or stroke > 0.015 * fg_extent:
                    continue
            if has_bg and float(fg_dist[ya:yb, xa:xb][crop].min()) < edge_margin:
                continue
            g_rgb = rgb[ya:yb, xa:xb][crop].mean(axis=0)
            glyphs.append({
                "cid": cid, "ya": ya, "yb": yb, "xa": xa, "xb": xb,
                "h": float(ch), "w": float(cw), "px": px,
                "cx": 0.5 * (xa + xb - 1), "cy": 0.5 * (ya + yb - 1),
                "rgb": g_rgb, "is_bar": is_bar,
            })

        ng = len(glyphs)
        if ng < 3:
            continue
        adj = [[] for _ in range(ng)]
        for i in range(ng):
            gi = glyphs[i]
            for j in range(i + 1, ng):
                gj = glyphs[j]
                if min(gi["h"], gj["h"]) / max(gi["h"], gj["h"]) < 0.48:
                    continue
                if float(np.linalg.norm(gi["rgb"] - gj["rgb"])) > 65.0:
                    continue
                avg_h = 0.5 * (gi["h"] + gj["h"])
                gap_x = max(0.0, max(gi["xa"], gj["xa"]) - min(gi["xb"], gj["xb"]))
                gap_y = max(0.0, max(gi["ya"], gj["ya"]) - min(gi["yb"], gj["yb"]))
                if (gap_x <= 0.90 * avg_h
                        and abs(gi["cy"] - gj["cy"]) <= 0.65 * avg_h
                        and math.hypot(gap_x, gap_y) <= 0.95 * avg_h):
                    adj[i].append((j, math.hypot(gap_x, gap_y)))
                    adj[j].append((i, math.hypot(gap_x, gap_y)))

        seen = [False] * ng
        for start in range(ng):
            if seen[start] or not adj[start]:
                continue
            stack = [start]
            seen[start] = True
            cluster = []
            max_pair_gap = 0.0
            while stack:
                u = stack.pop()
                cluster.append(u)
                for v, gdist in adj[u]:
                    if gdist > max_pair_gap:
                        max_pair_gap = gdist
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            if len(cluster) < 3:
                continue
            if sum(1 for idx in cluster if not glyphs[idx]["is_bar"]) < 2:
                continue
            hs = np.array([glyphs[idx]["h"] for idx in cluster], dtype=np.float64)
            med_h = float(np.median(hs))
            total_px = sum(glyphs[idx]["px"] for idx in cluster)
            if total_px < word_px_min or float(total_px) / n_px > 0.04:
                continue

            centers = np.array([[glyphs[idx]["cx"], glyphs[idx]["cy"]] for idx in cluster], dtype=np.float64)
            c_mean = centers.mean(axis=0)
            rel_c = centers - c_mean
            cov = np.einsum("ij,ik->jk", rel_c, rel_c) / len(cluster)
            _, evecs = np.linalg.eigh(cov)
            axis = evecs[:, 1]
            perp = evecs[:, 0]
            if axis[0] < 0.0:
                axis = -axis
            if abs(axis[1] / max(abs(axis[0]), 1e-6)) > 0.72:
                continue
            resid = np.abs(np.einsum("ij,j->i", rel_c, perp))
            if float(resid.max()) > 0.45 * med_h or float(np.sqrt((resid ** 2).mean())) > 0.25 * med_h:
                continue

            cids = [glyphs[idx]["cid"] for idx in cluster]
            ya_c = min(glyphs[idx]["ya"] for idx in cluster)
            yb_c = max(glyphs[idx]["yb"] for idx in cluster)
            xa_c = min(glyphs[idx]["xa"] for idx in cluster)
            xb_c = max(glyphs[idx]["xb"] for idx in cluster)
            raw_crop = np.isin(lbl[ya_c:yb_c, xa_c:xb_c], cids)
            rys_l, rxs_l = np.nonzero(raw_crop)
            # Measure word span along the glyph-center baseline (`axis`, `perp`)
            # rather than raw pixel-mass PCA, which tilts ~12 deg on short 3-letter
            # acronyms with top-heavy/bottom-heavy letters (e.g. "THW", "PCK").
            pts_xy = np.column_stack([rxs_l.astype(np.float64) + xa_c,
                                      rys_l.astype(np.float64) + ya_c]) - c_mean[None, :]
            proj_ax = np.einsum("ij,j->i", pts_xy, axis)
            proj_pe = np.einsum("ij,j->i", pts_xy, perp)
            major = max(1.0, float(proj_ax.max() - proj_ax.min()) + 1.0)
            minor = max(1.0, float(proj_pe.max() - proj_pe.min()) + 1.0)
            if major / max(minor, 1.0) < 1.75:
                continue
            if has_bg and (major / max(fg_extent, 1.0) > 0.40 or minor / max(fg_extent, 1.0) > 0.10):
                continue

            # Bridge inter-letter gaps tightly so the full word stays one coherent
            # 3D planar decal during camera-to-UV projection and local alignment.
            bridge = max(2, min(14, int(math.ceil(0.52 * max(max_pair_gap, 0.28 * med_h))) + 1))
            pad_b = bridge + 2
            by0, by1 = max(0, ya_c - pad_b), min(h_img, yb_c + pad_b)
            bx0, bx1 = max(0, xa_c - pad_b), min(w_img, xb_c + pad_b)
            sub_raw = np.zeros((by1 - by0, bx1 - bx0), dtype=bool)
            sub_raw[ya_c - by0:yb_c - by0, xa_c - bx0:xb_c - bx0] = raw_crop
            bridged = _dilate_bool(sub_raw, bridge)
            erode_steps = max(1, bridge - 2)
            b_img = Image.fromarray((bridged * 255).astype(np.uint8))
            for _ in range(erode_steps):
                b_img = b_img.filter(ImageFilter.MinFilter(3))
            sub_band = ((np.asarray(b_img) > 0) | sub_raw) & fg[by0:by1, bx0:bx1]
            word_band = np.zeros(fg.shape, dtype=bool)
            word_band[by0:by1, bx0:bx1] = sub_band
            raw_full = np.zeros(fg.shape, dtype=bool)
            raw_full[ya_c:yb_c, xa_c:xb_c] = raw_crop

            wb_px = int(word_band.sum())
            if wb_px < word_px_min or float(wb_px) / n_px > 0.05:
                continue
            if int((word_band & claimed).sum()) > 0.10 * wb_px:
                continue

            # Evaluate the tight immediate host panel right around and between
            # the letters (so lettering on narrow stripes or below windshields
            # is judged on its supporting paint, while cluttered rubble/hardware
            # fails uniformity and panel-inlier dominance).
            tight_r = max(4, min(10, int(round(0.28 * med_h))))
            ann = _dilate_bool(word_band, tight_r) & ~_dilate_bool(raw_full, 1)
            if not ann.any():
                continue
            if has_bg and float(fg[ann].mean()) < 0.88:
                continue
            ann_fg = ann & fg & ~ink & ~claimed
            if int(ann_fg.sum()) < 32:
                continue
            panel = np.median(rgb[ann_fg], axis=0)
            p_dists = np.linalg.norm(rgb[ann_fg] - panel[None, :], axis=-1)
            inliers = p_dists <= 42.0
            inlier_frac = float(inliers.mean())
            if inlier_frac < 0.62 or int(inliers.sum()) < 32:
                continue
            inlier_rgb = rgb[ann_fg][inliers]
            inlier_luma = luma[ann_fg][inliers]
            if float(inlier_luma.std()) > 16.0 or float(inlier_rgb.std(axis=0).max()) > 24.0:
                continue
            # Also require the outer halo not to be high-variance rocky/rubble noise.
            outer_r = max(tight_r + 3, int(round(0.45 * med_h)))
            outer_ann = _dilate_bool(word_band, outer_r) & ~_dilate_bool(word_band, 2) & fg
            if outer_ann.any():
                outer_inlier = float((np.linalg.norm(rgb[outer_ann] - panel[None, :], axis=-1) <= 50.0).mean())
                if outer_inlier < 0.42 and float(luma[outer_ann].std()) > 34.0:
                    continue
            ink_rgb = rgb[raw_full]
            if float(np.linalg.norm(ink_rgb.mean(axis=0) - panel)) < 110.0:
                continue
            if float(ink_rgb.std(axis=0).max()) > 45.0:
                continue

            wys, wxs = np.nonzero(word_band)
            rec = {
                "bbox": [int(wxs.min()), int(wys.min()), int(wxs.max()), int(wys.max())],
                "pixels": wb_px,
                "fraction": round(float(wb_px) / n_px, 5),
                "kind": "text",
                "glyphs": int(len(cluster)),
                "panel_rgb": [int(round(v)) for v in panel.tolist()],
            }
            out.append((rec, word_band))
            claimed |= word_band
    return out


def _geodesic_reconstruct(seed: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Pixels of `allowed` 8-connected to `seed` (seed itself always kept)."""
    seed = np.asarray(seed, bool)
    allowed = np.asarray(allowed, bool) | seed
    try:
        from scipy import ndimage
        lbl, _n = ndimage.label(allowed, structure=np.ones((3, 3), bool))
        ids = np.unique(lbl[seed])
        ids = ids[ids > 0]
        return np.isin(lbl, ids)
    except ImportError:
        cur = seed.copy()
        while True:
            nxt = _dilate_bool(cur, 1) & allowed
            if int(nxt.sum()) == int(cur.sum()):
                return cur
            cur = nxt


def _emblem_blob_components(rgba: Image.Image):
    """Internal insignia detector returning (union_mask, comp_masks, records)."""
    arr = np.asarray(rgba.convert("RGBA"))
    fg = arr[..., 3] > 127
    if not fg.any():
        return np.zeros(fg.shape, bool), [], []
    has_bg = bool((~fg).any())
    fg_dist = _fg_distance(fg)
    rgb = arr[..., :3].astype(np.float32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    saturated = (chroma > 0.5) & (mx > 80) & fg
    dark = (mx < 90) & fg
    candidate = saturated | dark

    # Drop isolated 1-px specks with a 3x3 erosion before connected-component labelling.
    mask_img = Image.fromarray((candidate * 255).astype(np.uint8))
    opened = np.asarray(mask_img.filter(ImageFilter.MinFilter(3))) > 0
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(opened)
        sizes = np.bincount(lbl.ravel())
    except ImportError:
        lbl, n, sizes = _label_fallback(opened)

    n_px = float(fg.sum())
    fys, fxs = np.nonzero(fg)
    fg_extent = float(max(fys.max() - fys.min() + 1, fxs.max() - fxs.min() + 1))
    edge_margin = max(4.0, BLOB_EDGE_MARGIN_FRAC * fg_extent)

    entries = []
    for i in range(1, n + 1):
        px_count = int(sizes[i]) if len(sizes) > i else 0
        frac = float(px_count) / n_px
        if frac < BLOB_MIN_FRAC:
            continue
        comp = lbl == i
        kind = "saturated" if (saturated[comp].mean() > 0.5) else "dark"
        cap = BLOB_MAX_FRAC if kind == "saturated" else DARK_BLOB_MAX_FRAC
        if frac > cap:
            continue

        ys, xs = np.nonzero(comp)
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        aspect = max(w, h) / max(min(w, h), 1)
        fill = float(px_count) / float(max(w * h, 1))
        span = max(w, h) / max(fg_extent, 1.0)
        if aspect > BLOB_MAX_ASPECT or fill < BLOB_MIN_FILL:
            continue
        if has_bg and span > BLOB_MAX_SPAN:
            continue
        if has_bg and float(fg_dist[comp].min()) < edge_margin:
            continue

        # Host-panel annular context: a genuine insignia sits on a uniform,
        # contrasting painted/fabric panel well inside the object silhouette.
        r_in = max(2, int(round(0.003 * max(fg.shape))))
        r_out = max(r_in + 4, int(round(0.012 * max(fg.shape))))
        annulus = _dilate_bool(comp, r_out) & ~_dilate_bool(comp, r_in)
        if has_bg and annulus.any() and float(fg[annulus].mean()) < BLOB_FG_ENCLOSURE_MIN:
            continue
        ring = annulus & fg & ~candidate
        if int(ring.sum()) < 16:
            continue
        ring_luma = luma[ring]
        ring_mean = float(ring_luma.mean())
        ring_std = float(ring_luma.std())
        color_diff = float(np.linalg.norm(rgb[comp].mean(axis=0) - rgb[ring].mean(axis=0)))
        blob_luma_std = float(luma[comp].std())
        if (ring_std > BLOB_RING_STD_MAX
                or color_diff < BLOB_CONTRAST_MIN
                or blob_luma_std > BLOB_INTERNAL_LUMA_STD_MAX):
            continue

        stroke = max_inscribed_radius(comp[ys.min():ys.max() + 1,
                                           xs.min():xs.max() + 1])
        if fill > 0.68 and (stroke / max(min(w, h), 1)) > 0.34:
            continue
        if kind == "dark":
            if stroke > DARK_STROKE_MAX_FRAC * fg_extent or ring_mean < DARK_RING_LUMA_MIN:
                continue

        comp_full = _dilate_bool(comp, 1) & candidate
        ys_f, xs_f = np.nonzero(comp_full)
        rec = {
            "bbox": [int(xs_f.min()), int(ys_f.min()), int(xs_f.max()), int(ys_f.max())],
            "pixels": int(comp_full.sum()),
            "fraction": round(float(comp_full.sum()) / n_px, 5),
            "kind": kind,
        }
        if kind == "dark":
            rec["stroke_radius_px"] = int(stroke)
        entries.append((rec, comp_full))

    # Pass 1 (`entries` so far) holds saturated/dark components, which can include
    # bold individual letters of large colored/dark inscriptions (e.g. red
    # "RØDE KORS" on white) whose narrower sibling letters ("E", "S", "I", "L")
    # exceed BLOB_MAX_ASPECT or fall below BLOB_MIN_FRAC. Lock Pass-2 badges and
    # non-glyph Pass-1 emblems first, then let Pass-3 multi-character text bands
    # supersede fragmented Pass-1 single letters into one complete word decal.
    taken = np.zeros(fg.shape, bool)
    for _rec, comp_full in entries:
        taken |= comp_full
    badge_entries = _panel_badge_entries(
        rgb, fg, fg_dist, fg_extent, edge_margin, has_bg, n_px, taken)
    entries.extend(badge_entries)

    glyph_px_max = max(40, int(round(0.004 * n_px)))
    max_ch = max(48, int(round(0.085 * fg_extent)))
    max_cw = max(64, int(round(0.095 * fg_extent)))
    locked_taken = np.zeros(fg.shape, bool)
    absorbable_idx = set()
    for idx, (rec, comp_m) in enumerate(entries):
        if rec.get("kind") in ("saturated", "dark"):
            x0, y0, x1, y1 = rec["bbox"]
            cw, ch = x1 - x0 + 1, y1 - y0 + 1
            stroke = max_inscribed_radius(comp_m[y0:y1 + 1, x0:x1 + 1])
            if (rec["pixels"] <= glyph_px_max
                    and ch <= max_ch and cw <= max_cw
                    and stroke <= max(5.5, 0.38 * min(cw, ch))
                    and stroke <= 0.015 * fg_extent):
                absorbable_idx.add(idx)
                continue
        locked_taken |= comp_m

    text_entries = _panel_text_entries(
        rgb, fg, fg_dist, fg_extent, edge_margin, has_bg, n_px, locked_taken)
    superseded = set()
    accepted_text = []
    for t_rec, t_mask in text_entries:
        t_px = max(1, int(t_mask.sum()))
        overlap_non_absorbed = 0
        hits = []
        for idx in absorbable_idx:
            _, p_mask = entries[idx]
            inter = int((t_mask & p_mask).sum())
            if inter == 0:
                continue
            p_px = max(1, int(p_mask.sum()))
            if inter / p_px >= 0.80:
                hits.append(idx)
            else:
                overlap_non_absorbed += inter
        if overlap_non_absorbed > 0.10 * t_px:
            continue
        superseded.update(hits)
        accepted_text.append((t_rec, t_mask))

    if superseded:
        entries = [item for idx, item in enumerate(entries) if idx not in superseded]
    entries.extend(accepted_text)

    entries.sort(key=lambda item: (item[0]["bbox"][1], item[0]["bbox"][0]))
    union_mask = np.zeros(fg.shape, bool)
    comp_masks = []
    records = []
    for rec, comp in entries:
        union_mask |= comp
        comp_masks.append(comp)
        records.append(rec)
    return union_mask, comp_masks, records


def emblem_blobs(rgba: Image.Image):
    """Find small concentrated insignia/lettering blobs in the reference.

    Returns (mask, records): bool foreground mask of blob pixels and a list
    of {bbox, pixels, fraction, kind} dicts. Empty records mean "no emblems, no-op".
    """
    mask, _comp_masks, records = _emblem_blob_components(rgba)
    return mask, records


def _label_fallback(binary):
    """4-connected labelling without scipy (small images only)."""
    h, w = binary.shape
    lbl = np.zeros((h, w), dtype=np.int32)
    n = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or lbl[y, x]:
                continue
            n += 1
            stack = [(y, x)]
            lbl[y, x] = n
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not lbl[ny, nx]:
                        lbl[ny, nx] = n
                        stack.append((ny, nx))
    sizes = np.bincount(lbl.ravel())
    return lbl, n, sizes


def max_inscribed_radius(binary):
    """Largest inscribed disc radius (px) of a bool mask: half its stroke width."""
    b = np.pad(np.asarray(binary, bool), 1)
    if not b.any():
        return 0.0
    try:
        from scipy import ndimage
        return float(ndimage.distance_transform_edt(b).max())
    except ImportError:
        img = Image.fromarray((b * 255).astype(np.uint8))
        radius = 0
        while np.asarray(img).any():
            img = img.filter(ImageFilter.MinFilter(3))
            radius += 1
        return float(radius)


# ------------------------------------------------------------ camera fitting

def _core_silhouette_mask(mask: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Morphological opening that strips thin guy ropes, antennas and fringe."""
    m = np.asarray(mask, dtype=bool)
    if int(m.sum()) < 64 or kernel < 3:
        return m
    img = Image.fromarray((m * 255).astype(np.uint8))
    opened = np.asarray(
        img.filter(ImageFilter.MinFilter(kernel)).filter(ImageFilter.MaxFilter(kernel))
    ) > 127
    return opened if int(opened.sum()) >= 64 else m


def _bbox_normalize(mask):
    """Translate/scale a bool mask to a canonical square by its bbox."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((64, 64), bool)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = mask[y0:y1, x0:x1]
    side = max(x1 - x0, y1 - y0)
    canvas = np.zeros((side, side), bool)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = crop
    return np.asarray(Image.fromarray((canvas * 255).astype(np.uint8)).resize((64, 64))) > 127


def _orbit_pose(center, radius, az, el):
    """Camera (cam, right, up, forward) on the QA orbit; az/el in radians."""
    cam = center + radius * np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el)])
    forward = center - cam
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    nrm = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if nrm < 1e-6 else right / nrm
    up = np.cross(right, forward)
    return cam, right, up, forward


def _orbit_cameras(center, radius, azimuths, elevations):
    cams = []
    for el in elevations:
        for az in azimuths:
            cam, right, up, forward = _orbit_pose(center, radius, az, el)
            cams.append((float(az), float(el), cam, right, up, forward))
    return cams


def pose_separation_deg(a, b):
    """Angle in degrees between two orbit viewing directions (az, el) in radians."""
    def unit(az, el):
        return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                         math.sin(el)])
    u, v = unit(*a), unit(*b)
    cos = float(np.clip((u * v).sum(), -1.0, 1.0))
    return math.degrees(math.acos(cos))


SIL_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
void main() { gl_Position = view_projection * vec4(position, 1.0); }
"""
SIL_FRAG = """#version 330
out vec4 color;
void main() { color = vec4(1.0); }
"""
# Studio-lit + camera-depth pass for appearance disambiguation. Writing linear
# camera depth into alpha enables greyscale morphological cavity occlusion so
# open tent doorways and recessed cab interiors darken like studio photography.
TEX_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
in vec3 normal;
in vec2 texcoord;
out vec2 uv;
out vec3 v_norm;
out vec3 v_pos;
void main() {
    uv = texcoord;
    v_norm = normal;
    v_pos = position;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""
TEX_FRAG = """#version 330
uniform sampler2D base_color;
uniform vec3 cam_pos;
uniform vec3 forward_dir;
uniform vec3 light_dir;
in vec2 uv;
in vec3 v_norm;
in vec3 v_pos;
out vec4 color;
void main() {
    vec3 base = texture(base_color, uv).rgb;
    vec3 n = normalize(v_norm);
    vec3 v = normalize(cam_pos - v_pos);
    if (dot(n, v) < 0.0) n = -n;
    float ndv = max(dot(n, v), 0.0);
    float ndl = max(dot(n, normalize(light_dir)), 0.0);
    float shade = 0.42 + 0.30 * ndv + 0.28 * ndl;
    float z_cam = dot(v_pos - cam_pos, forward_dir);
    color = vec4(clamp(base * shade, 0.0, 1.0), z_cam);
}
"""


def _silhouette_renderer(ctx, geoms_zup, size):
    """Compile a clay-silhouette program; returns (program, framebuffer, vaos)."""
    program = ctx.program(vertex_shader=SIL_VERT, fragment_shader=SIL_FRAG)
    framebuffer = ctx.simple_framebuffer((size, size), components=4)
    vaos = []
    for mesh in geoms_zup:
        buf = ctx.buffer(np.asarray(mesh.vertices, dtype="f4").tobytes())
        idx = ctx.buffer(np.asarray(mesh.faces, dtype="u4").tobytes())
        vaos.append(ctx.vertex_array(program, [(buf, "3f", "position")],
                                     idx, index_element_size=4))
    return program, framebuffer, vaos


def _vp_matrix(cam, right, up, forward, size, *, focal_ndc=3.0):
    view = np.eye(4)
    view[:3, :3] = np.array([right, up, -forward])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        view[:3, 3] = -(view[:3, :3] * np.asarray(cam, dtype=np.float64)[None, :]).sum(axis=1)
    near, far = 0.001, 50.0
    proj = np.zeros((4, 4))
    proj[0, 0] = proj[1, 1] = float(focal_ndc)
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return proj @ view


CANDIDATE_IOU_MARGIN = 0.08
MAX_POSE_CANDIDATES = 8
DISTINCT_POSE_DEG = 30.0
SILHOUETTE_MARGIN = 0.02
APPEARANCE_MARGIN = 0.06
APPEARANCE_MIN = 0.30
APPEARANCE_SIZE = 256
APPEARANCE_GRID = 64


def _appearance_crop(rgb, mask, n=APPEARANCE_GRID):
    """Crop RGB + coverage to the mask bbox, pad square, resize to n x n."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return np.zeros((n, n, 3), np.float32), np.zeros((n, n), bool)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = max(y1 - y0, x1 - x0)
    canvas = np.zeros((side, side, 3), np.uint8)
    cover = np.zeros((side, side), np.uint8)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = np.asarray(rgb)[y0:y1, x0:x1, :3]
    cover[oy:oy + y1 - y0, ox:ox + x1 - x0] = np.asarray(mask)[y0:y1, x0:x1] * 255
    small = (Image.fromarray(canvas).resize((n, n), Image.Resampling.BILINEAR)
             .filter(ImageFilter.GaussianBlur(1)))
    cov = np.asarray(Image.fromarray(cover).resize((n, n), Image.Resampling.BILINEAR)) > 200
    return np.asarray(small).astype(np.float32), cov


def _ncc_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-6 else 0.0


def luminance_ncc(a, mask_a, b, mask_b):
    """Normalised luminance correlation over the shared coverage (-1..1)."""
    both = np.asarray(mask_a) & np.asarray(mask_b)
    if both.sum() < 16:
        return -1.0
    weights = np.array([0.299, 0.587, 0.114], np.float64)
    la = (np.asarray(a, np.float64)[both] * weights).sum(axis=-1)
    lb = (np.asarray(b, np.float64)[both] * weights).sum(axis=-1)
    return _ncc_1d(la, lb)


def appearance_similarity(a, mask_a, b, mask_b):
    """Multi-cue correlation: shaded luma + high-pass structure + edges + chroma."""
    both = np.asarray(mask_a, dtype=bool) & np.asarray(mask_b, dtype=bool)
    if int(both.sum()) < 24:
        return -1.0
    eroded = np.asarray(
        Image.fromarray((both * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3))
    ) > 0
    m = eroded if int(eroded.sum()) >= 20 else both

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    weights = np.array([0.299, 0.587, 0.114], np.float64)
    la = (aa * weights).sum(axis=-1)
    lb = (bb * weights).sum(axis=-1)

    bla = np.asarray(
        Image.fromarray(np.clip(la, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)),
        dtype=np.float64)
    blb = np.asarray(
        Image.fromarray(np.clip(lb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)),
        dtype=np.float64)
    hpa = la - bla
    hpb = lb - blb

    gya, gxa = np.gradient(la)
    gyb, gxb = np.gradient(lb)
    ga = np.hypot(gxa, gya)
    gb = np.hypot(gxb, gyb)

    s_luma = _ncc_1d(la[m], lb[m])
    s_hp = _ncc_1d(hpa[m], hpb[m])
    s_grad = _ncc_1d(ga[m], gb[m])

    crg_a = aa[..., 0] - 0.5 * (aa[..., 1] + aa[..., 2])
    crg_b = bb[..., 0] - 0.5 * (bb[..., 1] + bb[..., 2])
    cby_a = aa[..., 2] - 0.5 * (aa[..., 0] + aa[..., 1])
    cby_b = bb[..., 2] - 0.5 * (bb[..., 0] + bb[..., 1])
    n_m = int(m.sum())
    sal_min = max(6, int(round(0.0015 * n_m)))
    chr_terms = []
    for ca, cb in ((crg_a, crg_b), (cby_a, cby_b)):
        std_a = float(ca[m].std())
        med_a = float(np.median(ca[m]))
        sal_a = (np.abs(ca - med_a) > 20.0) & m
        has_sal_a = int(sal_a.sum()) >= sal_min
        if std_a > 3.5 or has_sal_a:
            std_b = float(cb[m].std())
            # Never omit chroma when the reference has a chromatic feature and a
            # candidate view is blank/achromatic (e.g. an unmarked rear gable):
            # missing chroma must score 0.0 rather than falling back to luma-only.
            s_glob = _ncc_1d(ca[m], cb[m]) if std_b > 1.2 else 0.0
            if has_sal_a:
                roi = _dilate_bool(sal_a, 5) & m
                s_loc = (_ncc_1d(ca[roi], cb[roi])
                         if int(roi.sum()) >= 12 and float(cb[roi].std()) > 2.0
                         else 0.0)
                dev_a = float(np.abs(ca[sal_a] - med_a).mean())
                if dev_a > 25.0:
                    # Broad-neighborhood signed-chroma presence: when open
                    # interior lighting shifts silhouette elevation by a few
                    # degrees, the emblem on the true front face can shift by
                    # >5 crop pixels yet remain strongly chromatic inside the
                    # panel quadrant while a blank antipode has zero chroma.
                    broad_r = max(8, m.shape[0] // 5)
                    broad_roi = _dilate_bool(sal_a, broad_r) & m
                    sgn = 1.0 if float(np.median(ca[sal_a] - med_a)) >= 0.0 else -1.0
                    med_b = float(np.median(cb[m]))
                    vals_b = sgn * (cb[broad_roi] - med_b)
                    k_sal = min(len(vals_b), max(1, int(sal_a.sum())))
                    top_b = (float(np.mean(np.partition(vals_b, -k_sal)[-k_sal:]))
                             if k_sal > 0 else 0.0)
                    pres = float(np.clip(top_b / max(0.65 * dev_a, 1e-6), 0.0, 1.0))
                    s_c = 0.35 * s_glob + 0.35 * s_loc + 0.30 * (2.0 * pres - 1.0)
                else:
                    s_c = 0.50 * s_glob + 0.50 * s_loc
            else:
                s_c = s_glob
            chr_terms.append((max(std_a, 4.0), s_c))
    if chr_terms:
        w_sum = sum(w for w, _ in chr_terms)
        s_chr = sum(w * s for w, s in chr_terms) / max(w_sum, 1e-6)
        return 0.36 * s_luma + 0.28 * s_hp + 0.12 * s_grad + 0.24 * s_chr
    return 0.45 * s_luma + 0.35 * s_hp + 0.20 * s_grad


def _geom_textures(geoms_zup):
    """Base-colour image per geometry, or None when any geometry lacks one."""
    out = []
    for g in geoms_zup:
        material = getattr(getattr(g, "visual", None), "material", None)
        tex = getattr(material, "baseColorTexture", None)
        if tex is None:
            tex = getattr(material, "image", None)
        uv = getattr(getattr(g, "visual", None), "uv", None)
        if tex is None or uv is None or len(uv) != len(g.vertices):
            return None
        out.append((tex.convert("RGB"), np.asarray(uv, dtype=np.float32)))
    return out


def _cavity_ao_rgb(raw_rgba_z: np.ndarray, obj_span: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply greyscale-opening front-hull cavity occlusion to a float32 render."""
    fg = raw_rgba_z[..., 3] > 1e-4
    if not fg.any():
        return np.zeros((*fg.shape, 3), dtype=np.uint8), fg
    z_far = float(raw_rgba_z[..., 3][fg].max() + max(obj_span, 1e-3))
    z_fill = np.where(fg, raw_rgba_z[..., 3], z_far)
    k = max(15, int(round(0.33 * raw_rgba_z.shape[0])) | 1)
    try:
        from scipy import ndimage
        z_env = ndimage.grey_opening(z_fill, size=(k, k))
    except ImportError:
        z_min = float(z_fill.min())
        span = max(float(z_fill.max() - z_min), 1e-4)
        z_u8 = Image.fromarray(np.clip((z_fill - z_min) / span * 255.0, 0, 255).astype(np.uint8))
        opened = z_u8.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))
        z_env = np.asarray(opened, dtype=np.float32) / 255.0 * span + z_min
    recess = np.maximum(0.0, raw_rgba_z[..., 3] - z_env)
    tau = max(0.06 * obj_span, 1e-4)
    ao = np.where(fg, np.clip(np.exp(-recess / tau), 0.18, 1.0), 0.0)
    rgb = np.clip(raw_rgba_z[..., :3] * ao[..., None] * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb, fg


def _appearance_scores(geoms_zup, textures, ref_rgba, center, radius, poses,
                       *, obj_span=1.0):
    """Cavity-shaded multi-cue similarity vs the reference for each pose."""
    import moderngl
    ref = np.asarray(ref_rgba.convert("RGBA"))
    ref_core = _core_silhouette_mask(ref[..., 3] > 127)
    ref_c, ref_m = _appearance_crop(ref[..., :3], ref_core)
    size = APPEARANCE_SIZE
    scores = []
    with _mgl_context() as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        program = ctx.program(vertex_shader=TEX_VERT, fragment_shader=TEX_FRAG)
        ftex = ctx.texture((size, size), 4, dtype="f4")
        framebuffer = ctx.framebuffer([ftex], ctx.depth_renderbuffer((size, size)))
        framebuffer.use()
        draws = []
        for g, (tex, uv) in zip(geoms_zup, textures):
            pixels = np.asarray(tex)[::-1].copy()
            gpu_tex = ctx.texture(tex.size, 3, pixels.tobytes(), alignment=1)
            gpu_tex.build_mipmaps()
            gpu_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            v_pos = np.asarray(g.vertices, dtype=np.float32)
            try:
                v_nrm = np.asarray(g.vertex_normals, dtype=np.float32)
                if v_nrm.shape != v_pos.shape:
                    raise ValueError("normal shape mismatch")
            except Exception:
                v_nrm = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(v_pos), 1))
            buf = ctx.buffer(np.column_stack([v_pos, v_nrm, uv]).astype("f4").tobytes())
            idx = ctx.buffer(np.asarray(g.faces, dtype="u4").tobytes())
            vao = ctx.vertex_array(
                program, [(buf, "3f 3f 2f", "position", "normal", "texcoord")],
                idx, index_element_size=4)
            draws.append((vao, gpu_tex))
        for item in poses:
            if len(item) == 3:
                az, el, focal_ndc = item
            else:
                az, el = item
                focal_ndc = 3.0
            cam, right, up, fwd = _orbit_pose(center, radius, az, el)
            vp = _vp_matrix(cam, right, up, fwd, size, focal_ndc=focal_ndc)
            program["view_projection"].write(vp.T.astype("f4").tobytes())
            program["cam_pos"].value = tuple(float(v) for v in cam)
            program["forward_dir"].value = tuple(float(v) for v in fwd)
            ldir = -fwd + 0.35 * up + 0.20 * right
            program["light_dir"].value = tuple(float(v) for v in ldir)
            framebuffer.clear(0, 0, 0, 0, depth=1)
            for vao, gpu_tex in draws:
                gpu_tex.use(0)
                vao.render()
            raw = np.frombuffer(ftex.read(alignment=1), dtype=np.float32).reshape(size, size, 4)[::-1]
            rgb_u8, fg = _cavity_ao_rgb(raw, obj_span)
            got_c, got_m = _appearance_crop(rgb_u8, _core_silhouette_mask(fg))
            scores.append(appearance_similarity(ref_c, ref_m, got_c, got_m))
    return scores


def select_pose(candidates, *, distinct_deg=DISTINCT_POSE_DEG):
    """Choose the pose and decide whether it is trustworthy."""
    if not candidates:
        raise ValueError("no pose candidates")
    use_ncc = all(c.get("ncc") is not None for c in candidates)
    key = "ncc" if use_ncc else "iou"
    ranked = sorted(candidates, key=lambda c: (-c[key], -c["iou"], c["az"], c["el"]))
    winner = ranked[0]
    runner = next((c for c in ranked[1:]
                   if pose_separation_deg((c["az"], c["el"]),
                                          (winner["az"], winner["el"])) >= distinct_deg),
                  None)
    if use_ncc:
        margin = winner["ncc"] - (runner["ncc"] if runner else -1.0)
        ambiguous = winner["ncc"] < APPEARANCE_MIN or margin < APPEARANCE_MARGIN
    else:
        margin = winner["iou"] - (runner["iou"] if runner else 0.0)
        ambiguous = margin < SILHOUETTE_MARGIN
    return winner, runner, bool(ambiguous), "appearance" if use_ncc else "silhouette"


def fit_reference_camera(geoms_zup, ref_alpha, *, size=128, ref_rgba=None):
    """Fit the reference viewpoint: core-silhouette search + symmetry seeds + shaded multi-cue rank."""
    import moderngl
    ref = _core_silhouette_mask(np.asarray(ref_alpha.resize((size, size))) > 127)
    ref_n = _bbox_normalize(ref)
    verts = np.vstack([np.asarray(g.vertices) for g in geoms_zup])
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    obj_span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    radius = float(np.linalg.norm(verts - center, axis=1).max()) * math.sqrt(10) * 1.08
    azimuths = [math.radians(a) for a in range(0, 360, 15)]
    elevations = [math.radians(e) for e in (2, 8, 16, 26, 38, 52)]

    def _eval_silhouette_ious(pose_list):
        with _mgl_context() as s_ctx:
            s_ctx.enable(moderngl.DEPTH_TEST)
            prog, fb, vaos_s = _silhouette_renderer(s_ctx, geoms_zup, size)
            fb.use()
            out_ious = []
            for az_p, el_p, f_p in pose_list:
                vp_p = _vp_matrix(*_orbit_pose(center, radius, az_p, el_p), size,
                                  focal_ndc=f_p)
                prog["view_projection"].write(vp_p.T.astype("f4").tobytes())
                fb.clear(0, 0, 0, 1, depth=1)
                for vao_s in vaos_s:
                    vao_s.render()
                got = _core_silhouette_mask(np.asarray(Image.frombytes(
                    "RGB", (size, size),
                    fb.read(components=3, alignment=1),
                ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("L")) > 127)
                got_n = _bbox_normalize(got)
                union = float((got_n | ref_n).sum())
                out_ious.append(float((got_n & ref_n).sum()) / union if union else 0.0)
            return out_ious

    with _mgl_context() as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        program, framebuffer, vaos = _silhouette_renderer(ctx, geoms_zup, size)
        framebuffer.use()

        def silhouette_iou(az, el, focal_ndc=3.0):
            vp = _vp_matrix(*_orbit_pose(center, radius, az, el), size,
                            focal_ndc=focal_ndc)
            program["view_projection"].write(vp.T.astype("f4").tobytes())
            framebuffer.clear(0, 0, 0, 1, depth=1)
            for vao in vaos:
                vao.render()
            got = _core_silhouette_mask(np.asarray(Image.frombytes(
                "RGB", (size, size),
                framebuffer.read(components=3, alignment=1),
            ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("L")) > 127)
            got_n = _bbox_normalize(got)
            union = float((got_n | ref_n).sum())
            return float((got_n & ref_n).sum()) / union if union else 0.0

        coarse = [{"az": az, "el": el, "iou": silhouette_iou(az, el), "focal_ndc": 3.0}
                  for az, el, *_ in _orbit_cameras(center, radius, azimuths, elevations)]
        best_iou = max(c["iou"] for c in coarse)
        seeds = []
        for c in sorted(coarse, key=lambda c: (-c["iou"], c["az"], c["el"])):
            if c["iou"] < best_iou - CANDIDATE_IOU_MARGIN or len(seeds) >= 6:
                break
            if all(pose_separation_deg((c["az"], c["el"]), (s["az"], s["el"])) >= DISTINCT_POSE_DEG
                   for s in seeds):
                seeds.append(c)

        # Mandatory symmetry counterparts: box-like vehicles, crates and tents
        # have near-equal silhouettes under antipode (az + pi) and mirror (-az,
        # pi - az) views; include them explicitly before fine refinement.
        aug_seeds = list(seeds)
        for s in list(seeds)[:2]:
            for sym_az in ((s["az"] + math.pi) % (2.0 * math.pi),
                           (-s["az"]) % (2.0 * math.pi),
                           (math.pi - s["az"]) % (2.0 * math.pi)):
                if all(pose_separation_deg((sym_az, s["el"]),
                                           (x["az"], x["el"])) >= DISTINCT_POSE_DEG
                       for x in aug_seeds):
                    aug_seeds.append({
                        "az": sym_az, "el": s["el"],
                        "iou": silhouette_iou(sym_az, s["el"]), "focal_ndc": 3.0,
                    })

        candidates = []
        for seed in aug_seeds:
            best = dict(seed)
            for daz in range(-10, 11, 2):
                for del_ in range(-6, 7, 2):
                    az2 = (seed["az"] + math.radians(daz)) % (2.0 * math.pi)
                    el2 = min(math.radians(80), max(math.radians(1), seed["el"] + math.radians(del_)))
                    iou2 = silhouette_iou(az2, el2, 3.0)
                    if iou2 > best["iou"]:
                        best = {"az": az2, "el": el2, "iou": iou2, "focal_ndc": 3.0}
            for f_try in (2.5, 3.6):
                iou_f = silhouette_iou(best["az"], best["el"], f_try)
                if iou_f > best["iou"] + 0.008:
                    best = {**best, "iou": iou_f, "focal_ndc": f_try}
            if (best["iou"] >= best_iou - (CANDIDATE_IOU_MARGIN + 0.02)
                    and all(pose_separation_deg((best["az"], best["el"]),
                                                (x["az"], x["el"])) >= 25.0
                            for x in candidates)):
                candidates.append(best)
        if not candidates:
            candidates = [max(coarse, key=lambda c: c["iou"])]
        candidates = sorted(candidates, key=lambda c: -c["iou"])[:MAX_POSE_CANDIDATES]

    textures = _geom_textures(geoms_zup) if ref_rgba is not None else None
    if textures:
        scores = _appearance_scores(
            geoms_zup, textures, ref_rgba, center, radius,
            [(c["az"], c["el"], c.get("focal_ndc", 3.0)) for c in candidates],
            obj_span=obj_span)
        for c, s in zip(candidates, scores):
            c["ncc"] = s

        # Symmetrically refine the top distinct candidate hemispheres BEFORE
        # judging ambiguity: on barrel-vaulted tents or box bodies, silhouette
        # search alone can leave elevation off by 12-16 deg so both front and
        # rear hemispheres look falsely mediocre until windows/emblems align.
        iou_floor = max(FIT_IOU_MIN, best_iou - 0.08)
        top_indices = sorted(
            range(len(candidates)),
            key=lambda idx_c: (-candidates[idx_c]["ncc"], -candidates[idx_c]["iou"]),
        )[:min(3, len(candidates))]
        hemi_probes = []
        hemi_owners = []
        for idx_c in top_indices:
            cand_c = candidates[idx_c]
            f_c = float(cand_c.get("focal_ndc", 3.0))
            for daz in (-15, -10, -5, 0, 5, 10, 15):
                for del_ in (-10, -5, 0, 5, 10, 15, 20):
                    az_p = (cand_c["az"] + math.radians(daz)) % (2.0 * math.pi)
                    el_p = min(math.radians(80), max(math.radians(1), cand_c["el"] + math.radians(del_)))
                    if pose_separation_deg((az_p, el_p), (cand_c["az"], cand_c["el"])) > 24.0:
                        continue
                    if any(
                        other_i != idx_c
                        and pose_separation_deg(
                            (az_p, el_p),
                            (candidates[other_i]["az"], candidates[other_i]["el"]),
                        ) < 25.0
                        for other_i in range(len(candidates))
                    ):
                        continue
                    hemi_probes.append((az_p, el_p, f_c))
                    hemi_owners.append(idx_c)
        if hemi_probes:
            h_ious = _eval_silhouette_ious(hemi_probes)
            h_apps = _appearance_scores(
                geoms_zup, textures, ref_rgba, center, radius, hemi_probes, obj_span=obj_span)
            best_by_cand = {
                idx_c: (
                    0.68 * float(candidates[idx_c]["ncc"]) + 0.32 * float(candidates[idx_c]["iou"]),
                    dict(candidates[idx_c]),
                )
                for idx_c in top_indices
            }
            for idx_c, (az_p, el_p, f_p), iou_p, app_p in zip(
                hemi_owners, hemi_probes, h_ious, h_apps
            ):
                if iou_p < iou_floor:
                    continue
                comb_p = 0.68 * float(app_p) + 0.32 * float(iou_p)
                if comb_p > best_by_cand[idx_c][0] + 0.005:
                    best_by_cand[idx_c] = (
                        comb_p,
                        {"az": az_p, "el": el_p, "iou": iou_p,
                         "focal_ndc": f_p, "ncc": app_p},
                    )
            for idx_c, (_comb_b, ref_cand) in best_by_cand.items():
                candidates[idx_c] = ref_cand

    winner, runner, ambiguous, basis = select_pose(candidates)

    # Fine 1-degree + focal-length polish around an unambiguous winning pose.
    if textures and not ambiguous and "ncc" in winner:
        win_idx = candidates.index(winner)
        iou_floor = max(FIT_IOU_MIN, best_iou - 0.08)
        base_comb = 0.68 * float(winner["ncc"]) + 0.32 * float(winner["iou"])
        best_ref = dict(winner)
        best_comb = base_comb

        stage2 = []
        for daz in (-3, -2, -1, 0, 1, 2, 3):
            for del_ in (-3, -2, -1, 0, 1, 2, 3):
                az_p = (best_ref["az"] + math.radians(daz)) % (2.0 * math.pi)
                el_p = min(math.radians(80), max(math.radians(1), best_ref["el"] + math.radians(del_)))
                for f_p in (2.6, 3.0, 3.4):
                    stage2.append((az_p, el_p, f_p))
        s2_ious = _eval_silhouette_ious(stage2)
        s2_apps = _appearance_scores(
            geoms_zup, textures, ref_rgba, center, radius, stage2, obj_span=obj_span)
        for (az_p, el_p, f_p), iou_p, app_p in zip(stage2, s2_ious, s2_apps):
            if iou_p < iou_floor:
                continue
            comb_p = 0.68 * float(app_p) + 0.32 * float(iou_p)
            if comb_p > best_comb:
                best_comb = comb_p
                best_ref = {"az": az_p, "el": el_p, "iou": iou_p,
                            "focal_ndc": f_p, "ncc": app_p}

        if best_comb > base_comb + 0.005:
            candidates[win_idx] = best_ref
            winner, runner, ambiguous, basis = select_pose(candidates)
    cam, right, up, forward = _orbit_pose(center, radius, winner["az"], winner["el"])
    return {
        "azimuth_deg": round(math.degrees(winner["az"]) % 360.0, 2),
        "elevation_deg": round(math.degrees(winner["el"]), 2),
        "focal_ndc": round(float(winner.get("focal_ndc", 3.0)), 3),
        "iou": round(float(winner["iou"]), 4),
        "runner_up_iou": round(float(runner["iou"]), 4) if runner else None,
        "appearance_ncc": round(float(winner["ncc"]), 4) if "ncc" in winner else None,
        "runner_up_ncc": (round(float(runner["ncc"]), 4)
                          if runner and "ncc" in runner else None),
        "runner_up_azimuth_deg": (round(math.degrees(runner["az"]) % 360.0, 2)
                                  if runner else None),
        "ambiguity_basis": basis,
        "ambiguous": ambiguous,
        "candidates": len(candidates),
        "center": [float(v) for v in center],
        "radius": float(radius),
        "cam": [float(v) for v in cam],
        "right": [float(v) for v in right],
        "up": [float(v) for v in up],
        "forward": [float(v) for v in forward],
    }


def pose_rejection(fit):
    """Reason the fitted pose must not drive a projection, or None."""
    if fit["iou"] < FIT_IOU_MIN:
        return f"pose fit too poor (iou {fit['iou']}); skipped"
    if fit.get("ambiguous"):
        if fit.get("ambiguity_basis") == "appearance":
            return (f"pose ambiguous (appearance ncc {fit.get('appearance_ncc')} vs "
                    f"{fit.get('runner_up_ncc')} at azimuth "
                    f"{fit.get('runner_up_azimuth_deg')}); skipped")
        return (f"pose ambiguous (silhouette iou {fit['iou']} vs "
                f"{fit.get('runner_up_iou')}, no texture to disambiguate); skipped")
    return None


POSE_RECORD_KEYS = ("azimuth_deg", "elevation_deg", "focal_ndc", "iou",
                    "runner_up_iou", "appearance_ncc", "runner_up_ncc",
                    "runner_up_azimuth_deg", "ambiguity_basis", "ambiguous",
                    "candidates")


def pose_record(fit):
    """Metadata summary of the pose decision (no camera vectors)."""
    return {k: fit.get(k) for k in POSE_RECORD_KEYS}


# ------------------------------------------------------------ projection

ID_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
in vec3 a_face_id;
flat out vec3 v_face_id;
void main() {
    v_face_id = a_face_id;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""
ID_FRAG = """#version 330
flat in vec3 v_face_id;
out vec4 color0;
void main() { color0 = vec4(v_face_id, 1.0); }
"""
POS_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
out vec3 wpos;
void main() { wpos = position; gl_Position = view_projection * vec4(position, 1.0); }
"""
POS_FRAG = """#version 330
in vec3 wpos;
out vec4 color0;
void main() { color0 = vec4(wpos, 1.0); }
"""


def decal_weights(blob_mask):
    """Blurred, dilated emblem footprint in reference space (float 0..1)."""
    m = np.asarray(blob_mask).astype(bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    grow = max(2, int(round(0.004 * max(m.shape[-2:]))))
    img = Image.fromarray((m * 255).astype(np.uint8))
    for _ in range(grow):
        img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(max(1.0, grow / 2.0)))
    return np.asarray(img).astype(np.float32) / 255.0


def badge_patch_weights(blob_comps, ref_rgb: np.ndarray, ref_alpha: np.ndarray,
                        *, text_slots=None):
    """Build per-blob badge + host-panel patch weights and stroke/owner maps.

    Returns `(footprint, core_patch_mask, stroke_mask, owner_map)`:
    - Absorbs concentric closed badge rings/frames that lie strictly inside a
      local padded box around each accepted emblem (excluding passing body
      stripes that cross the box border).
    - Gives full `1.0` core weight to enclosed badge interiors (e.g. the 4 white
      quadrants inside a circled red cross) and the immediate uniform host-panel
      collar (`||ref_rgb - panel_rgb|| <= 38`) so smeared TRELLIS emblem halos
      are cleanly restored to host-panel colour without touching windows/trim.
    """
    h, w = ref_alpha.shape
    footprint = np.zeros((h, w), dtype=np.float32)
    core_patch = np.zeros((h, w), dtype=bool)
    stroke_mask = np.zeros((h, w), dtype=bool)
    owner_map = np.zeros((h, w), dtype=np.int32)
    if not blob_comps:
        return footprint, core_patch, stroke_mask, owner_map

    mx_all = ref_rgb.max(axis=-1)
    mn_all = ref_rgb.min(axis=-1)
    sat_all = mx_all - mn_all
    chroma_all = sat_all / np.maximum(mx_all, 1e-6)
    union_blobs = np.zeros((h, w), dtype=bool)
    for _slot, comp in blob_comps:
        union_blobs |= comp

    r_out_global = max(6, int(round(0.012 * max(h, w))))
    for slot, comp in blob_comps:
        bys, bxs = np.nonzero(comp)
        if not len(bys):
            continue
        y0, y1 = int(bys.min()), int(bys.max())
        x0, x1 = int(bxs.min()), int(bxs.max())
        major_s, minor_s, _ = _oriented_extent(bys, bxs)
        is_text_band = ((slot in text_slots) if text_slots is not None and slot in text_slots
                        else (major_s / max(minor_s, 1.0)) > 2.8)
        if is_text_band:
            span = minor_s
            pad_core = max(6, int(round(0.38 * span)))
            pad_outer = max(10, int(round(0.58 * span)))
        else:
            span = max(y1 - y0 + 1, x1 - x0 + 1)
            pad_core = max(12, int(round(0.82 * span)))
            pad_outer = max(16, int(round(1.00 * span)))
        pad_box = pad_outer + 3
        cy0, cy1 = max(0, y0 - pad_box), min(h, y1 + 1 + pad_box)
        cx0, cx1 = max(0, x0 - pad_box), min(w, x1 + 1 + pad_box)

        loc_comp = comp[cy0:cy1, cx0:cx1]
        loc_fg = ref_alpha[cy0:cy1, cx0:cx1]
        loc_rgb = ref_rgb[cy0:cy1, cx0:cx1]
        loc_chr = chroma_all[cy0:cy1, cx0:cx1]
        loc_sat = sat_all[cy0:cy1, cx0:cx1]
        loc_mx = mx_all[cy0:cy1, cx0:cx1]
        loc_union = union_blobs[cy0:cy1, cx0:cx1]

        init_ann = (_dilate_bool(loc_comp, max(r_out_global, int(round(0.22 * span))))
                    & ~_dilate_bool(loc_comp, 3) & loc_fg & ~loc_union)
        if int(init_ann.sum()) >= 8:
            init_panel = np.median(loc_rgb[init_ann], axis=0)
        else:
            init_panel = np.array([235.0, 235.0, 235.0], dtype=np.float32)
        init_pdiff = np.linalg.norm(loc_rgb - init_panel[None, None, :], axis=-1)

        # Closed outer badge frame/circle (including anti-aliased thin strokes):
        # contrasting strokes within 0.30*span whose connected component stays
        # strictly inside a tight ring box (so passing body stripes are excluded).
        # Wide text bands skip concentric-circle frame search.
        badge_frame = np.zeros_like(loc_comp)
        if not is_text_band:
            is_sat_blob = float(loc_chr[loc_comp].mean()) > 0.35
            ring_pad = max(5, int(round(0.30 * span)))
            ring_box = ring_pad + 4
            ry0_b = max(0, (y0 - cy0) - ring_box)
            ry1_b = min(cy1 - cy0, (y1 - cy0) + 1 + ring_box)
            rx0_b = max(0, (x0 - cx0) - ring_box)
            rx1_b = min(cx1 - cx0, (x1 - cx0) + 1 + ring_box)
            ring_sub_mask = np.zeros_like(loc_comp)
            ring_sub_mask[ry0_b:ry1_b, rx0_b:rx1_b] = True

            ring_zone = _dilate_bool(loc_comp, ring_pad) & loc_fg & ring_sub_mask
            if is_sat_blob:
                raw_fc = (loc_sat >= 16.0) & (loc_chr > 0.10) & (loc_mx > 65.0) & (init_pdiff >= 32.0)
            else:
                raw_fc = ((loc_mx < 110.0) | (init_pdiff >= 42.0)) & (init_pdiff >= 32.0)
            frame_cand = raw_fc & loc_fg & ring_sub_mask
            try:
                from scipy import ndimage as _ndi
                f_lbl, n_fc = _ndi.label(frame_cand)
            except ImportError:
                f_lbl, n_fc, _ = _label_fallback(frame_cand)
            if n_fc > 0:
                sub_lbl = f_lbl[ry0_b:ry1_b, rx0_b:rx1_b]
                border_ids = set(sub_lbl[0, :].tolist() + sub_lbl[-1, :].tolist()
                                 + sub_lbl[:, 0].tolist() + sub_lbl[:, -1].tolist())
                border_ids.discard(0)
                for fid_c in range(1, n_fc + 1):
                    if fid_c not in border_ids:
                        comp_fc = (f_lbl == fid_c)
                        if (comp_fc & ring_zone).any():
                            badge_frame |= (comp_fc & ring_zone)

        badge_all = _fill_holes(loc_comp | badge_frame) & loc_fg
        annulus_pad = max(r_out_global, int(round(0.18 * span)))
        annulus = (_dilate_bool(badge_all, annulus_pad)
                   & ~_dilate_bool(badge_all, 2)
                   & loc_fg & ~loc_union)
        if int(annulus.sum()) >= 8:
            panel_rgb = np.median(loc_rgb[annulus], axis=0)
        else:
            glob_ann = (_dilate_bool(comp, r_out_global)
                        & ~_dilate_bool(comp, 2) & ref_alpha & ~union_blobs)
            if int(glob_ann.sum()) >= 8:
                panel_rgb = np.median(ref_rgb[glob_ann], axis=0)
            else:
                panel_rgb = np.array([235.0, 235.0, 235.0], dtype=np.float32)

        insignia = _dilate_bool(badge_all, 2) & loc_fg
        panel_dist = np.linalg.norm(loc_rgb - panel_rgb[None, None, :], axis=-1)
        is_panel = (panel_dist <= 38.0) & loc_fg
        # 2-px guard margin around non-panel features (stripes, windows, vents)
        # outside the immediate badge interior, so wider smudge-clearing collars
        # never bleed onto passing livery stripes or adjacent trim.
        inner_badge_zone = _dilate_bool(
            badge_all, max(3 if is_text_band else 4, int(round((0.20 if is_text_band else 0.28) * span)))
        ) & loc_fg
        non_panel_guard = _dilate_bool(~is_panel, 2) & ~inner_badge_zone
        is_panel_collar = is_panel & ~non_panel_guard

        disc_core = _dilate_bool(badge_all, pad_core) & loc_fg
        disc_outer = _dilate_bool(badge_all, pad_outer) & loc_fg

        loc_core = insignia | (disc_core & is_panel_collar)
        loc_allow = insignia | (disc_outer & is_panel_collar)
        blur_r = max(1.5, (0.08 if is_text_band else 0.045) * span)
        blurred = np.asarray(
            Image.fromarray((loc_allow * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur_r)),
            dtype=np.float32,
        ) / 255.0
        loc_w = np.where(loc_core, 1.0, blurred)
        loc_w = np.where(insignia | is_panel_collar, loc_w, 0.0)

        prev_w = footprint[cy0:cy1, cx0:cx1]
        upd = loc_w > prev_w
        footprint[cy0:cy1, cx0:cx1] = np.where(upd, loc_w, prev_w)
        owner_map[cy0:cy1, cx0:cx1] = np.where(upd, int(slot), owner_map[cy0:cy1, cx0:cx1])
        core_patch[cy0:cy1, cx0:cx1] |= loc_core
        stroke_mask[cy0:cy1, cx0:cx1] |= badge_all

    return footprint, core_patch, stroke_mask, owner_map


# ------------------------------------------------------------ shared texels
UV_BIN_GRID = 128
SHARED_TEXEL_TOL_FRAC = 0.01
SHARED_TEXEL_PAIR_BUDGET = 8_000_000


def uv_face_index(faces, uv, grid=UV_BIN_GRID):
    """Bucket faces by their UV bounding box for texel -> face lookups."""
    f = np.asarray(faces, dtype=np.int64)
    uv = np.nan_to_num(np.asarray(uv, dtype=np.float64), nan=0.0,
                       posinf=0.0, neginf=0.0)
    n = len(f)
    while True:
        if not n:
            return {"grid": grid, "starts": np.zeros(grid * grid + 1, np.int64),
                    "faces": np.zeros(0, np.int32)}
        uvf = uv[f]
        bu0 = np.clip((uvf[..., 0].min(axis=1) * grid).astype(np.int64), 0, grid - 1)
        bu1 = np.clip((uvf[..., 0].max(axis=1) * grid).astype(np.int64), 0, grid - 1)
        bv0 = np.clip(((1.0 - uvf[..., 1].max(axis=1)) * grid).astype(np.int64), 0, grid - 1)
        bv1 = np.clip(((1.0 - uvf[..., 1].min(axis=1)) * grid).astype(np.int64), 0, grid - 1)
        w = bu1 - bu0 + 1
        h = bv1 - bv0 + 1
        cnt = w * h
        total = int(cnt.sum())
        if total <= SHARED_TEXEL_PAIR_BUDGET or grid <= 16:
            break
        grid = max(16, grid // 2)
    starts_f = np.repeat(np.cumsum(cnt) - cnt, cnt)
    local = np.arange(total, dtype=np.int64) - starts_f
    w_rep = np.repeat(w, cnt)
    bins = ((np.repeat(bv0, cnt) + local // w_rep) * grid
            + np.repeat(bu0, cnt) + local % w_rep)
    order = np.argsort(bins, kind="stable")
    bins = bins[order]
    counts = np.bincount(bins, minlength=grid * grid)
    starts = np.zeros(grid * grid + 1, np.int64)
    starts[1:] = np.cumsum(counts)
    faces_sorted = np.repeat(np.arange(n, dtype=np.int32), cnt)[order]
    return {"grid": grid, "starts": starts, "faces": faces_sorted}


def shared_surface_texels(index, faces, vertices, uv, tu, tv, tw, th, hits, tol):
    """Flag candidate texels that another, far-away surface also samples."""
    tu = np.asarray(tu, dtype=np.int64)
    tv = np.asarray(tv, dtype=np.int64)
    n = len(tu)
    if not n or not len(index["faces"]):
        return np.zeros(n, bool), 0
    grid = index["grid"]
    u = (tu + 0.5) / float(tw)
    v = 1.0 - (tv + 0.5) / float(th)
    flat = (np.clip((v * grid).astype(np.int64), 0, grid - 1) * grid
            + np.clip((u * grid).astype(np.int64), 0, grid - 1))
    lo = index["starts"][flat]
    counts = index["starts"][flat + 1] - lo
    pairs = int(counts.sum())
    if pairs > SHARED_TEXEL_PAIR_BUDGET:
        return np.zeros(n, bool), pairs
    rows = np.repeat(np.arange(n), counts)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    face_rows = index["faces"][np.repeat(lo, counts) + (np.arange(pairs) - offsets)]
    f = np.asarray(faces, dtype=np.int64)[face_rows]
    uvf = np.nan_to_num(np.asarray(uv, dtype=np.float64), nan=0.0)[f]
    pu, pv = u[rows], v[rows]
    x0, y0 = uvf[:, 0, 0], uvf[:, 0, 1]
    x1, y1 = uvf[:, 1, 0], uvf[:, 1, 1]
    x2, y2 = uvf[:, 2, 0], uvf[:, 2, 1]
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    ok = np.abs(det) > 1e-12
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        l0 = ((y1 - y2) * (pu - x2) + (x2 - x1) * (pv - y2)) / det
        l1 = ((y2 - y0) * (pu - x2) + (x0 - x2) * (pv - y2)) / det
    l2 = 1.0 - l0 - l1
    inside = ok & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    if not inside.any():
        return np.zeros(n, bool), pairs
    vtx = np.asarray(vertices, dtype=np.float64)[f]
    with np.errstate(invalid="ignore", over="ignore"):
        point = (vtx[:, 0] * l0[:, None] + vtx[:, 1] * l1[:, None]
                 + vtx[:, 2] * l2[:, None])
        dist = np.linalg.norm(point - np.asarray(hits, dtype=np.float64)[rows], axis=1)
    far = inside & np.isfinite(dist) & (dist > tol)
    conflict = np.zeros(n, bool)
    conflict[rows[far]] = True
    return conflict, pairs


def _pack_output_texture(rgb_float: np.ndarray, orig_alpha: np.ndarray | None) -> Image.Image:
    rgb_u8 = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
    if orig_alpha is not None:
        return Image.fromarray(np.concatenate([rgb_u8, orig_alpha[..., None]], axis=-1), "RGBA")
    return Image.fromarray(rgb_u8, "RGB")


def _depth_step_mask(t_grid: np.ndarray, tol: float) -> np.ndarray:
    """Detect pixels sitting on 3D depth discontinuities (doorways, rims)."""
    valid = np.isfinite(t_grid)
    if not valid.any():
        return np.zeros(t_grid.shape, bool)
    try:
        from scipy import ndimage
        t_hi = ndimage.maximum_filter(np.where(valid, t_grid, -np.inf), size=3)
        t_lo = ndimage.minimum_filter(np.where(valid, t_grid, np.inf), size=3)
        all_v = ndimage.minimum_filter(valid.astype(np.uint8), size=3) > 0
        return (~all_v) | ((t_hi - t_lo) > tol)
    except ImportError:
        vis_img = Image.fromarray((valid * 255).astype(np.uint8))
        return ~(np.asarray(vis_img.filter(ImageFilter.MinFilter(3))) > 0)


def _bilinear_grid_2d(grid_2d: np.ndarray, xf: np.ndarray, yf: np.ndarray) -> np.ndarray:
    """Bilinearly sample a 2D float array `(H, W)` at continuous `(xf, yf)`."""
    h, w = grid_2d.shape[:2]
    x0 = np.floor(xf).astype(np.int64).clip(0, max(w - 2, 0))
    y0 = np.floor(yf).astype(np.int64).clip(0, max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = np.clip(xf - x0, 0.0, 1.0)
    fy = np.clip(yf - y0, 0.0, 1.0)
    return (
        grid_2d[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + grid_2d[y0, x1] * fx * (1.0 - fy)
        + grid_2d[y1, x0] * (1.0 - fx) * fy
        + grid_2d[y1, x1] * fx * fy
    )


def _clip_poly_halfspace(poly: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Clip an `(M, D)` polygon against linear half-space `vals >= 0`."""
    m = len(poly)
    if not m:
        return poly
    out = []
    for i in range(m):
        j = (i + 1) % m
        pa, va = poly[i], float(vals[i])
        pb, vb = poly[j], float(vals[j])
        if va >= 0.0:
            if vb >= 0.0:
                out.append(pb)
            else:
                t_c = va / max(va - vb, 1e-12)
                out.append(pa + t_c * (pb - pa))
        elif vb >= 0.0:
            t_c = va / min(va - vb, -1e-12)
            out.append(pa + t_c * (pb - pa))
            out.append(pb)
    if not out:
        return np.empty((0, poly.shape[1]), dtype=poly.dtype)
    return np.asarray(out, dtype=poly.dtype)


def _apply_shift_with_collar_fallback(
    rxf0: np.ndarray, ryf0: np.ndarray,
    rx0_i: np.ndarray, ry0_i: np.ndarray,
    shift_maps, ref_alpha: np.ndarray,
    fg_dist_ref: np.ndarray = None,
    stroke_mask: np.ndarray = None,
    footprint: np.ndarray = None,
):
    """Apply per-blob `(sx, sy)` shift, falling back to unshifted `(rxf0, ryf0)`
    on outer host-panel collar texels whose shifted sample exits the safe
    reference silhouette interior (`fg_dist < 2.0`)."""
    if shift_maps is None:
        return rxf0, ryf0
    sx_map, sy_map = shift_maps
    rxf = rxf0 + sx_map[ry0_i, rx0_i]
    ryf = ryf0 + sy_map[ry0_i, rx0_i]
    if fg_dist_ref is None or stroke_mask is None:
        return rxf, ryf
    h_ref, w_ref = ref_alpha.shape
    in_b = (rxf >= 0.0) & (rxf < w_ref) & (ryf >= 0.0) & (ryf < h_ref)
    rxi = np.clip(rxf.astype(np.int64), 0, w_ref - 1)
    ryi = np.clip(ryf.astype(np.int64), 0, h_ref - 1)
    ok_shifted = in_b & (fg_dist_ref[ryi, rxi] >= 2.0)
    in_b0 = (rxf0 >= 0.0) & (rxf0 < w_ref) & (ryf0 >= 0.0) & (ryf0 < h_ref)
    is_collar = (~stroke_mask[ry0_i, rx0_i]) & in_b0 & (fg_dist_ref[ry0_i, rx0_i] >= 1.5)
    if footprint is not None:
        is_collar &= footprint[ry0_i, rx0_i] > DECAL_FEATHER_MIN
    fb = is_collar & ~ok_shifted
    if np.any(fb):
        rxf = np.where(fb, rxf0, rxf)
        ryf = np.where(fb, ryf0, ryf)
    return rxf, ryf


def _gather_uv_decal_texels(
    F: np.ndarray, V: np.ndarray, UV: np.ndarray, tw: int, th: int,
    cam: np.ndarray, right: np.ndarray, up: np.ndarray, forward: np.ndarray,
    focal: float, size: int, radius: float,
    scale: float, ox: float, oy: float, ref_alpha: np.ndarray,
    pos_raw: np.ndarray, vis_screen: np.ndarray, surface_feather: np.ndarray,
    footprint: np.ndarray, owner_map: np.ndarray, blob_planes: dict,
    cos_min: float, cos_full: float, plane_tol: float,
    shift_maps=None, fg_dist_ref: np.ndarray = None,
    stroke_mask: np.ndarray = None,
):
    """Inverse UV-to-3D-to-reference gather so 100% of decal UV texels are filled.

    Forward scattering from a `512x512` render onto a `2048x2048` or `4096x4096`
    atlas leaves 75-90% of UV texels inside each emblem triangle unwritten,
    interlacing TRELLIS's blurry/shifted baked insignia with projected pixels
    ("Venetian-blind" comb stripes and double-vision ghosting). Clipping each
    coplanar candidate triangle to the decal frustum and rasterizing every UV
    texel `(tu + 0.5, tv + 0.5)` directly into continuous reference coordinates
    `(rxf, ryf)` fills 100% of emblem texels at native reference resolution.
    """
    active_base = (owner_map > 0) if shift_maps is not None else (footprint > DECAL_FEATHER_MIN)
    fp_ys, fp_xs = np.nonzero(active_base)
    if not len(fp_ys) or not len(F):
        return None

    h_ref, w_ref = ref_alpha.shape
    rx_lo, rx_hi = float(fp_xs.min() - 2), float(fp_xs.max() + 3)
    ry_lo, ry_hi = float(fp_ys.min() - 2), float(fp_ys.max() + 3)
    s_inv = 1.0 / max(scale, 1e-6)
    sx_lo, sx_hi = (rx_lo - ox) * s_inv, (rx_hi - ox) * s_inv
    sy_lo, sy_hi = (ry_lo - oy) * s_inv, (ry_hi - oy) * s_inv
    half = size * 0.5

    rel_v = V - cam[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xc = rel_v @ right
        yc = rel_v @ up
        zc = rel_v @ forward

    h0 = zc - 1e-4
    h1 = xc * focal + (half - sx_lo) * zc
    h2 = -xc * focal + (sx_hi - half) * zc
    h3 = -yc * focal + (half - sy_lo) * zc
    h4 = yc * focal + (sy_hi - half) * zc

    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    in_frust = (
        ((h0[i0] >= 0) | (h0[i1] >= 0) | (h0[i2] >= 0))
        & ((h1[i0] >= 0) | (h1[i1] >= 0) | (h1[i2] >= 0))
        & ((h2[i0] >= 0) | (h2[i1] >= 0) | (h2[i2] >= 0))
        & ((h3[i0] >= 0) | (h3[i1] >= 0) | (h3[i2] >= 0))
        & ((h4[i0] >= 0) | (h4[i1] >= 0) | (h4[i2] >= 0))
    )
    cand_f = np.nonzero(in_frust)[0]
    if not len(cand_f):
        return None

    tv0 = V[F[cand_f, 0]]
    tv1 = V[F[cand_f, 1]]
    tv2 = V[F[cand_f, 2]]
    fn = np.cross(tv1 - tv0, tv2 - tv0)
    fn_len = np.linalg.norm(fn, axis=1)
    ok_fn = fn_len > 1e-12
    cand_f = cand_f[ok_fn]
    if not len(cand_f):
        return None
    tv0, tv1, tv2 = tv0[ok_fn], tv1[ok_fn], tv2[ok_fn]
    fn = fn[ok_fn] / fn_len[ok_fn, None]

    # Keep triangles coplanar with at least one accepted blob surface plane.
    coplanar_tri = np.zeros(len(cand_f), dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _slot, (u_n, c_3d) in blob_planes.items():
            nd = np.abs(fn @ u_n)
            d0 = (tv0 - c_3d[None, :]) @ u_n
            d1 = (tv1 - c_3d[None, :]) @ u_n
            d2 = (tv2 - c_3d[None, :]) @ u_n
            straddle = (d0 * d1 <= 0.0) | (d1 * d2 <= 0.0) | (d0 * d2 <= 0.0)
            min_d = np.minimum(np.abs(d0), np.minimum(np.abs(d1), np.abs(d2)))
            coplanar_tri |= (nd >= 0.58) & (straddle | (min_d <= plane_tol))
    cand_f = cand_f[coplanar_tri]
    if not len(cand_f):
        return None
    tv0, tv1, tv2, fn = tv0[coplanar_tri], tv1[coplanar_tri], tv2[coplanar_tri], fn[coplanar_tri]

    tu_chunks, tv_chunks, prio_chunks, p3d_chunks, n_chunks = [], [], [], [], []
    for k, fi in enumerate(cand_f):
        vi = F[fi]
        u_tri = UV[vi, 0] * tw
        v_tri = (1.0 - UV[vi, 1]) * th
        e1_u, e1_v = float(u_tri[1] - u_tri[0]), float(v_tri[1] - v_tri[0])
        e2_u, e2_v = float(u_tri[2] - u_tri[0]), float(v_tri[2] - v_tri[0])
        det_uv = e1_u * e2_v - e1_v * e2_u
        if abs(det_uv) <= 1e-8:
            continue

        poly = np.column_stack([xc[vi], yc[vi], zc[vi], u_tri, v_tri])
        poly = _clip_poly_halfspace(poly, poly[:, 2] - 1e-4)
        if not len(poly):
            continue
        poly = _clip_poly_halfspace(poly, poly[:, 0] * focal + (half - sx_lo) * poly[:, 2])
        if not len(poly):
            continue
        poly = _clip_poly_halfspace(poly, -poly[:, 0] * focal + (sx_hi - half) * poly[:, 2])
        if not len(poly):
            continue
        poly = _clip_poly_halfspace(poly, -poly[:, 1] * focal + (half - sy_lo) * poly[:, 2])
        if not len(poly):
            continue
        poly = _clip_poly_halfspace(poly, poly[:, 1] * focal + (sy_hi - half) * poly[:, 2])
        if not len(poly):
            continue

        tu0 = max(0, int(math.floor(float(poly[:, 3].min()) - 1.0)))
        tu1 = min(tw - 1, int(math.ceil(float(poly[:, 3].max()) + 1.0)))
        tv0_i = max(0, int(math.floor(float(poly[:, 4].min()) - 1.0)))
        tv1_i = min(th - 1, int(math.ceil(float(poly[:, 4].max()) + 1.0)))
        if tu1 < tu0 or tv1_i < tv0_i:
            continue

        gv, gu = np.mgrid[tv0_i:tv1_i + 1, tu0:tu1 + 1]
        gu_f = gu.ravel()
        gv_f = gv.ravel()
        px_u = (gu_f + 0.5) - float(u_tri[0])
        py_v = (gv_f + 0.5) - float(v_tri[0])
        inv_det = 1.0 / det_uv
        lam1 = (px_u * e2_v - py_v * e2_u) * inv_det
        lam2 = (e1_u * py_v - e1_v * px_u) * inv_det
        lam0 = 1.0 - lam1 - lam2

        inside = (lam0 >= -1e-5) & (lam1 >= -1e-5) & (lam2 >= -1e-5)
        # 1-texel seam gutter prevents GL_LINEAR chart-border bleed across UV cuts.
        lc0 = np.maximum(lam0, 0.0)
        lc1 = np.maximum(lam1, 0.0)
        lc2 = np.maximum(lam2, 0.0)
        lc_sum = np.maximum(lc0 + lc1 + lc2, 1e-12)
        lc0 /= lc_sum
        lc1 /= lc_sum
        lc2 /= lc_sum
        pc_u = lc0 * float(u_tri[0]) + lc1 * float(u_tri[1]) + lc2 * float(u_tri[2])
        pc_v = lc0 * float(v_tri[0]) + lc1 * float(v_tri[1]) + lc2 * float(v_tri[2])
        dist_uv2 = ((gu_f + 0.5) - pc_u) ** 2 + ((gv_f + 0.5) - pc_v) ** 2
        gutter = (~inside) & (dist_uv2 <= 1.15)
        keep_t = inside | gutter
        if not bool(keep_t.any()):
            continue

        b0 = np.where(inside, lam0, lc0)[keep_t]
        b1 = np.where(inside, lam1, lc1)[keep_t]
        b2 = np.where(inside, lam2, lc2)[keep_t]
        p3d = (b0[:, None] * tv0[k][None, :]
               + b1[:, None] * tv1[k][None, :]
               + b2[:, None] * tv2[k][None, :])
        prio = np.where(inside[keep_t], 0, 1).astype(np.int8)

        tu_chunks.append(gu_f[keep_t].astype(np.int64))
        tv_chunks.append(gv_f[keep_t].astype(np.int64))
        prio_chunks.append(prio)
        p3d_chunks.append(p3d)
        n_chunks.append(np.broadcast_to(fn[k][None, :], (len(p3d), 3)).copy())

    if not tu_chunks:
        return None

    tu_g = np.concatenate(tu_chunks)
    tv_g = np.concatenate(tv_chunks)
    prio_g = np.concatenate(prio_chunks)
    p3d_g = np.vstack(p3d_chunks)
    fn_g = np.vstack(n_chunks)

    rel_p = p3d_g - cam[None, :]
    t_g = np.linalg.norm(rel_p, axis=1)
    ray_g = rel_p / np.maximum(t_g[:, None], 1e-12)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z_g = rel_p @ forward
        sx_f = (rel_p @ right) / np.maximum(z_g, 1e-9) * focal + half
        sy_f = -(rel_p @ up) / np.maximum(z_g, 1e-9) * focal + half
    rxf0_g = sx_f * scale + ox
    ryf0_g = sy_f * scale + oy
    rx0_i = np.clip(rxf0_g.astype(np.int64), 0, w_ref - 1)
    ry0_i = np.clip(ryf0_g.astype(np.int64), 0, h_ref - 1)
    rxf_g, ryf_g = _apply_shift_with_collar_fallback(
        rxf0_g, ryf0_g, rx0_i, ry0_i, shift_maps, ref_alpha,
        fg_dist_ref=fg_dist_ref, stroke_mask=stroke_mask, footprint=footprint)

    in_b = ((z_g > 1e-4)
            & (sx_f >= 0.5) & (sx_f <= size - 0.5)
            & (sy_f >= 0.5) & (sy_f <= size - 0.5)
            & (rxf0_g >= -64.0) & (rxf0_g < w_ref + 64.0)
            & (ryf0_g >= -64.0) & (ryf0_g < h_ref + 64.0)
            & (rxf_g >= 0.0) & (rxf_g < w_ref)
            & (ryf_g >= 0.0) & (ryf_g < h_ref))
    if not bool(in_b.any()):
        return None

    tu_g, tv_g, prio_g, p3d_g, fn_g, t_g, ray_g = (
        tu_g[in_b], tv_g[in_b], prio_g[in_b], p3d_g[in_b], fn_g[in_b], t_g[in_b], ray_g[in_b])
    sx_f, sy_f, rxf_g, ryf_g = sx_f[in_b], sy_f[in_b], rxf_g[in_b], ryf_g[in_b]
    rx0_i, ry0_i = rx0_i[in_b], ry0_i[in_b]

    rx_i = np.clip(rxf_g.astype(np.int64), 0, w_ref - 1)
    ry_i = np.clip(ryf_g.astype(np.int64), 0, h_ref - 1)
    on_fg = ref_alpha[ry_i, rx_i]
    if not bool(on_fg.any()):
        return None
    tu_g, tv_g, prio_g, p3d_g, fn_g, t_g, ray_g = (
        tu_g[on_fg], tv_g[on_fg], prio_g[on_fg], p3d_g[on_fg], fn_g[on_fg], t_g[on_fg], ray_g[on_fg])
    sx_f, sy_f, rxf_g, ryf_g, rx_i, ry_i = (
        sx_f[on_fg], sy_f[on_fg], rxf_g[on_fg], ryf_g[on_fg], rx_i[on_fg], ry_i[on_fg])
    rx0_i, ry0_i = rx0_i[on_fg], ry0_i[on_fg]

    # 3x3 hardware depth-buffer agreement rejects occluded faces behind foreground parts.
    px_i = np.clip(sx_f.astype(np.int64), 0, size - 1)
    py_i = np.clip(sy_f.astype(np.int64), 0, size - 1)
    min_d3 = np.full(len(tu_g), np.inf, dtype=np.float64)
    for dy_s in (-1, 0, 1):
        py_n = np.clip(py_i + dy_s, 0, size - 1)
        for dx_s in (-1, 0, 1):
            px_n = np.clip(px_i + dx_s, 0, size - 1)
            v_ok = vis_screen[py_n, px_n]
            if not bool(v_ok.any()):
                continue
            d_n = np.linalg.norm(p3d_g[v_ok] - pos_raw[py_n[v_ok], px_n[v_ok]], axis=1)
            min_d3[v_ok] = np.minimum(min_d3[v_ok], d_n)
    vis_ok = min_d3 <= (radius * 0.014 + 1e-5)
    if not bool(vis_ok.any()):
        return None

    tu_g, tv_g, prio_g, p3d_g, fn_g, t_g, ray_g = (
        tu_g[vis_ok], tv_g[vis_ok], prio_g[vis_ok], p3d_g[vis_ok], fn_g[vis_ok], t_g[vis_ok], ray_g[vis_ok])
    sx_f, sy_f, rxf_g, ryf_g, rx_i, ry_i = (
        sx_f[vis_ok], sy_f[vis_ok], rxf_g[vis_ok], ryf_g[vis_ok], rx_i[vis_ok], ry_i[vis_ok])
    rx0_i, ry0_i = rx0_i[vis_ok], ry0_i[vis_ok]

    n_cam_g = np.where((ray_g * fn_g).sum(axis=1, keepdims=True) > 0.0, -fn_g, fn_g)
    cos_inc_g = np.abs(np.einsum("ij,ij->i", ray_g, n_cam_g))
    inc_w_g = np.clip((cos_inc_g - cos_min) / max(cos_full - cos_min, 1e-6), 0.0, 1.0)

    owners_g = owner_map[ry0_i, rx0_i]
    coplanar_g = np.zeros(len(tu_g), dtype=bool)
    for slot, (u_n, c_3d) in blob_planes.items():
        sel_s = owners_g == slot
        if not bool(sel_s.any()):
            continue
        ndot = np.einsum("ij,j->i", n_cam_g[sel_s], u_n)
        pdist = np.abs(np.einsum("ij,j->i", p3d_g[sel_s] - c_3d[None, :], u_n))
        coplanar_g[sel_s] = (ndot >= 0.60) & (pdist <= plane_tol)

    fp_w = _bilinear_grid_2d(footprint, rxf_g, ryf_g)
    sf_w = _bilinear_grid_2d(surface_feather, sx_f - 0.5, sy_f - 0.5)
    on_dec = (fp_w > DECAL_FEATHER_MIN) & (cos_inc_g >= cos_min) & coplanar_g
    if not bool(on_dec.any()):
        return None

    wgt_g = sf_w[on_dec] * inc_w_g[on_dec] * fp_w[on_dec]
    keep_w = wgt_g > 0.01
    if not bool(keep_w.any()):
        return None
    idx_k = np.nonzero(on_dec)[0][keep_w]
    return (
        tu_g[idx_k], tv_g[idx_k], prio_g[idx_k], t_g[idx_k],
        p3d_g[idx_k], n_cam_g[idx_k],
        rxf_g[idx_k], ryf_g[idx_k], ry_i[idx_k], rx_i[idx_k],
        wgt_g[keep_w].astype(np.float32),
    )


def project_reference(geoms_zup, texture_img, ref_rgba, fit, *, blob_mask=None,
                      blob_records=None, size=RENDER_SIZE, out_state: dict = None):
    """Write true reference pixels into validated 3D emblem decal texels only."""
    import moderngl
    ref = ref_rgba.convert("RGBA")
    ref_rgb = np.asarray(ref.convert("RGB")).astype(np.float32)
    ref_alpha = np.asarray(ref)[:, :, 3] > 127
    fg_dist_ref = _fg_distance(ref_alpha)
    orig_alpha = np.asarray(texture_img)[:, :, 3].copy() if texture_img.mode == "RGBA" else None
    tex = np.asarray(texture_img.convert("RGB")).astype(np.float32)
    th, tw = tex.shape[:2]

    if blob_mask is None or not np.asarray(blob_mask, dtype=bool).any():
        return texture_img.copy(), {
            "projected_texels": 0, "visible_faces": 0,
            "note": "no emblem footprint; refusing whole-surface projection",
        }

    faces = []
    face_uv = []
    base = 0
    all_v = np.vstack([np.asarray(g.vertices, dtype=np.float64) for g in geoms_zup])
    for g in geoms_zup:
        f = np.asarray(g.faces, dtype=np.int64) + base
        base += len(g.vertices)
        faces.append(f)
        uv = np.asarray(getattr(g.visual, "uv", np.zeros((len(g.vertices), 2))), dtype=np.float64)
        face_uv.append(uv)
    F = np.vstack(faces)
    UV = np.vstack(face_uv)
    V = all_v
    n_faces = len(F)

    cam = np.array(fit["cam"], dtype=np.float64)
    right = np.array(fit["right"], dtype=np.float64)
    up = np.array(fit["up"], dtype=np.float64)
    forward = np.array(fit["forward"], dtype=np.float64)
    focal_ndc = float(fit.get("focal_ndc", 3.0))
    focal = 0.5 * focal_ndc * size

    vp = _vp_matrix(cam, right, up, forward, size, focal_ndc=focal_ndc).T.astype("f4").tobytes()
    with _mgl_context() as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        id_program = ctx.program(vertex_shader=ID_VERT, fragment_shader=ID_FRAG)
        id_program["view_projection"].write(vp)
        color_tex = ctx.texture((size, size), 3)
        fb_id = ctx.framebuffer([color_tex], ctx.depth_renderbuffer((size, size)))
        fb_id.use()
        fb_id.clear(0, 0, 0, 0, depth=1)
        fid_offset = 0
        for g in geoms_zup:
            gv = np.asarray(g.vertices, dtype=np.float32)
            gf = np.asarray(g.faces, dtype=np.int64)
            nf = len(gf)
            if not nf:
                continue
            encs = np.arange(fid_offset + 1, fid_offset + nf + 1, dtype=np.int64)
            fid_offset += nf
            id_rgb = np.stack([
                ((encs >> 16) & 255) / 255.0,
                ((encs >> 8) & 255) / 255.0,
                (encs & 255) / 255.0,
            ], axis=1).astype(np.float32)
            tri_pos = gv[gf].reshape(nf * 3, 3)
            tri_id = np.repeat(id_rgb, 3, axis=0)
            vbo = ctx.buffer(np.hstack([tri_pos, tri_id]).astype("f4").tobytes())
            vao = ctx.vertex_array(id_program, [(vbo, "3f 3f", "position", "a_face_id")])
            vao.render(moderngl.TRIANGLES)
            vao.release()
            vbo.release()
        id_raw = np.asarray(Image.frombytes(
            "RGB", (size, size),
            color_tex.read(alignment=1),
        ).transpose(Image.Transpose.FLIP_TOP_BOTTOM))

        pos_program = ctx.program(vertex_shader=POS_VERT, fragment_shader=POS_FRAG)
        pos_program["view_projection"].write(vp)
        pos_tex = ctx.texture((size, size), 4, dtype="f4")
        fb_pos = ctx.framebuffer([pos_tex], ctx.depth_renderbuffer((size, size)))
        fb_pos.use()
        fb_pos.clear(0, 0, 0, 0, depth=1)
        for g in geoms_zup:
            buf = ctx.buffer(np.asarray(g.vertices, dtype="f4").tobytes())
            idx = ctx.buffer(np.asarray(g.faces, dtype="u4").tobytes())
            vao_p = ctx.vertex_array(
                pos_program, [(buf, "3f", "position")], idx,
                index_element_size=4)
            vao_p.render()
            vao_p.release()
            idx.release()
            buf.release()
        pos_raw = np.frombuffer(pos_tex.read(alignment=1),
                                dtype=np.float32).reshape(size, size, 4)[:, :, :3]
        pos_raw = np.asarray(pos_raw, dtype=np.float64)[::-1]

    face_of = ((id_raw[:, :, 0].astype(np.int64) << 16) |
               (id_raw[:, :, 1].astype(np.int64) << 8) |
               (id_raw[:, :, 2].astype(np.int64))) - 1
    valid = (face_of >= 0) & (face_of < n_faces)
    ys, xs = np.nonzero(valid)
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0, "visible_faces": 0,
                                    "note": "no faces in view"}

    fids = face_of[ys, xs]
    v0 = V[F[fids, 0]]
    v1 = V[F[fids, 1]]
    v2 = V[F[fids, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n_len = np.linalg.norm(n, axis=1)
    keep_n = n_len > 1e-12
    ys, xs, fids = ys[keep_n], xs[keep_n], fids[keep_n]
    v0, v1, v2 = v0[keep_n], v1[keep_n], v2[keep_n]
    n = (n[keep_n].T / n_len[keep_n]).T

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denom = (n * forward[None, :]).sum(axis=1)
    ok_d = np.isfinite(denom) & (np.abs(denom) > 1e-9)
    ys, xs, fids = ys[ok_d], xs[ok_d], fids[ok_d]
    v0, v1, v2, n = v0[ok_d], v1[ok_d], v2[ok_d], n[ok_d]

    dx = (xs + 0.5 - size / 2.0) / focal
    dy = -((ys + 0.5 - size / 2.0) / focal)
    ray = (right[None, :] * dx[:, None] + up[None, :] * dy[:, None] + forward[None, :])
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    denom2 = np.einsum("ij,ij->i", ray, n)
    ok2 = np.abs(denom2) > 1e-9
    ys, xs, fids = ys[ok2], xs[ok2], fids[ok2]
    v0, v1, v2, n, ray, denom2 = v0[ok2], v1[ok2], v2[ok2], n[ok2], ray[ok2], denom2[ok2]

    t = np.einsum("ij,ij->i", v0 - cam, n) / denom2
    hit = cam[None, :] + ray * t[:, None]
    ren = pos_raw[ys, xs]
    agree = np.linalg.norm(hit - ren, axis=1) < (fit["radius"] * 2e-3 + 1e-6)
    ys, xs, fids, hit, t, n, ray = (
        ys[agree], xs[agree], fids[agree], hit[agree], t[agree], n[agree], ray[agree])
    v0, v1, v2 = v0[agree], v1[agree], v2[agree]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0, "visible_faces": 0,
                                    "note": "depth test rejected everything"}

    e1 = v1 - v0
    e2 = v2 - v0
    w = hit - v0
    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    dw0 = np.einsum("ij,ij->i", w, e1)
    dw1 = np.einsum("ij,ij->i", w, e2)
    det = d00 * d11 - d01 * d01
    ok3 = np.abs(det) > 1e-18
    ys, xs, fids, t, hit, n, ray = (
        ys[ok3], xs[ok3], fids[ok3], t[ok3], hit[ok3], n[ok3], ray[ok3])
    lam1 = (d11[ok3] * dw0[ok3] - d01[ok3] * dw1[ok3]) / det[ok3]
    lam2 = (d00[ok3] * dw1[ok3] - d01[ok3] * dw0[ok3]) / det[ok3]
    lam0 = 1.0 - lam1 - lam2
    inside = ((lam0 >= -1e-6) & (lam1 >= -1e-6) & (lam2 >= -1e-6)
              & (lam0 <= 1 + 1e-6) & (lam1 <= 1 + 1e-6) & (lam2 <= 1 + 1e-6))
    ys, xs, fids, t, hit, n, ray = (
        ys[inside], xs[inside], fids[inside], t[inside], hit[inside], n[inside], ray[inside])
    lam0, lam1, lam2 = lam0[inside], lam1[inside], lam2[inside]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0, "visible_faces": 0,
                                    "note": "no valid barycentric hits"}

    uv = (UV[F[fids, 0]] * lam0[:, None] + UV[F[fids, 1]] * lam1[:, None]
          + UV[F[fids, 2]] * lam2[:, None])
    tu = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
    tv = np.clip(((1.0 - uv[:, 1]) * th).astype(np.int64), 0, th - 1)

    # Full-surface depth-step and rim erosion BEFORE decal restriction:
    # use hardware float depth (pos_raw) on all rasterized pixels so isolated
    # 1-px face-ID interpolation drops on sliver edges never carve false
    # MinFilter(5) depth-step craters in the middle of a continuous panel.
    valid_hw = (id_raw[:, :, 0] | id_raw[:, :, 1] | id_raw[:, :, 2]) > 0
    t_grid = np.full((size, size), np.inf, dtype=np.float64)
    if bool(valid_hw.any()):
        t_grid[valid_hw] = np.linalg.norm(pos_raw[valid_hw] - cam[None, :], axis=1)
    step_edge = _depth_step_mask(t_grid, float(fit["radius"]) * 0.015)
    vis_full = np.zeros((size, size), dtype=np.uint8)
    vis_full[valid_hw] = 255
    vis_full[step_edge] = 0
    eroded_full = np.asarray(Image.fromarray(vis_full).filter(ImageFilter.MinFilter(5))) > 0
    surface_feather = np.asarray(
        Image.fromarray((eroded_full * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
        dtype=np.float32) / 255.0

    # Core-mask frame mapping + appearance-assisted sub-frame alignment.
    ren_mask = np.zeros((size, size), bool)
    ren_mask[ys, xs] = True
    ren_core = _core_silhouette_mask(ren_mask, kernel=9)
    ref_core = _core_silhouette_mask(ref_alpha, kernel=9)
    rys, rxs = np.nonzero(ref_core)
    scale, ox, oy = _frame_map(ren_core, (rys, rxs), ref_alpha.shape, size)
    ren_rgb_frame = np.zeros((size, size, 3), dtype=np.float32)
    ren_rgb_frame[ys, xs] = tex[tv, tu]
    scale, ox, oy = _refine_frame_map(
        ren_rgb_frame, ren_core, ref_rgb, ref_core, scale, ox, oy)

    cy_s, cx_s = np.nonzero(ren_core)
    if len(cy_s) and len(rys):
        ren_asp = (cx_s.max() - cx_s.min() + 1) / max(cy_s.max() - cy_s.min() + 1, 1)
        ref_asp = (rxs.max() - rxs.min() + 1) / max(rys.max() - rys.min() + 1, 1)
        if abs(math.log(max(ren_asp, 1e-6) / max(ref_asp, 1e-6))) > CORE_ASPECT_TOL:
            return texture_img.copy(), {
                "projected_texels": 0, "visible_faces": int(len(set(fids.tolist()))),
                "note": f"core silhouette aspect mismatch ({ren_asp:.3f} vs {ref_asp:.3f}); refusing",
            }
        # Aligned silhouette IoU in render coordinates.
        gy, gx = np.mgrid[0:size, 0:size]
        grx = np.clip(((gx + 0.5) * scale + ox).astype(int), 0, ref_alpha.shape[1] - 1)
        gry = np.clip(((gy + 0.5) * scale + oy).astype(int), 0, ref_alpha.shape[0] - 1)
        in_bounds = (((gx + 0.5) * scale + ox >= 0)
                     & ((gx + 0.5) * scale + ox < ref_alpha.shape[1])
                     & ((gy + 0.5) * scale + oy >= 0)
                     & ((gy + 0.5) * scale + oy < ref_alpha.shape[0]))
        ref_in_ren = in_bounds & ref_core[gry, grx]
        union_c = float((ren_core | ref_in_ren).sum())
        aligned_iou = float((ren_core & ref_in_ren).sum()) / union_c if union_c else 0.0
        if aligned_iou < CORE_ALIGNED_IOU_MIN:
            return texture_img.copy(), {
                "projected_texels": 0, "visible_faces": int(len(set(fids.tolist()))),
                "aligned_iou": round(aligned_iou, 4),
                "note": f"aligned silhouette IoU too low ({aligned_iou:.3f}); refusing",
            }

    rxf = (xs + 0.5) * scale + ox
    ryf = (ys + 0.5) * scale + oy
    ok4 = (rxf >= 0) & (rxf < ref_rgb.shape[1]) & (ryf >= 0) & (ryf < ref_rgb.shape[0])
    inside_ref_alpha = np.zeros(len(ys), bool)
    ok_idx = np.nonzero(ok4)[0]
    inside_ref_alpha[ok_idx] = ref_alpha[
        np.clip(ryf[ok_idx].astype(int), 0, ref_rgb.shape[0] - 1),
        np.clip(rxf[ok_idx].astype(int), 0, ref_rgb.shape[1] - 1)]
    use = ok4 & inside_ref_alpha
    ys, xs, tu, tv, t, hit, n, ray = (
        ys[use], xs[use], tu[use], tv[use], t[use], hit[use], n[use], ray[use])
    rxf, ryf = rxf[use], ryf[use]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0,
                                    "visible_faces": int(len(set(fids.tolist()))),
                                    "note": "no overlap with reference foreground"}

    # Orient hit normals toward the camera and compute incidence angle weight.
    n_cam = np.where((ray * n).sum(axis=1, keepdims=True) > 0.0, -n, n)
    cos_inc = np.abs(np.einsum("ij,ij->i", ray, n_cam))
    cos_min = math.cos(math.radians(MAX_INCIDENCE_DEG))
    cos_full = math.cos(math.radians(FEATHER_INCIDENCE_DEG))
    inc_w = np.clip((cos_inc - cos_min) / max(cos_full - cos_min, 1e-6), 0.0, 1.0)

    # Per-blob 3D surface coherence gate: validate each blob independently
    # before admitting it into the decal footprint.
    bm_bool = np.asarray(blob_mask, dtype=bool)
    try:
        from scipy import ndimage as _ndi
        b_lbl, n_blobs = _ndi.label(bm_bool)
    except ImportError:
        b_lbl, n_blobs, _ = _label_fallback(bm_bool)

    ry_all = np.clip(ryf.astype(int), 0, bm_bool.shape[0] - 1)
    rx_all = np.clip(rxf.astype(int), 0, bm_bool.shape[1] - 1)
    hit_blob = b_lbl[ry_all, rx_all]
    accepted_blob_mask = np.zeros_like(bm_bool)
    accepted_comps = []
    text_slots = set()
    blob_planes = {}
    accepted_blobs = 0
    rejected_reasons = []
    text_boxes_ref = np.zeros_like(bm_bool)
    if blob_records:
        rh_b, rw_b = bm_bool.shape
        for rec in blob_records:
            if isinstance(rec, dict) and rec.get("kind") == "text" and rec.get("bbox"):
                bx0, by0, bx1, by1 = rec["bbox"]
                text_boxes_ref[max(0, int(by0)):min(rh_b, int(by1) + 1),
                               max(0, int(bx0)):min(rw_b, int(bx1) + 1)] = True

    z_min_obj = float(V[:, 2].min())
    z_span_obj = max(float(V[:, 2].max()) - z_min_obj, 1e-6)

    for bi in range(1, n_blobs + 1):
        comp_ref = b_lbl == bi
        expected_ren_px = float(comp_ref.sum()) / max(scale * scale, 1e-6)
        in_b = (hit_blob == bi) & (cos_inc >= cos_min) & (surface_feather[ys, xs] > 0.1)
        if expected_ren_px > 1.0 and (float(in_b.sum()) / expected_ren_px) < BLOB_COVERAGE_MIN:
            rejected_reasons.append(f"blob#{bi}:grazing_or_rim_occluded")
            continue
        if int(in_b.sum()) < 6:
            rejected_reasons.append(f"blob#{bi}:too_few_hits")
            continue
        mean_n = n_cam[in_b].mean(axis=0)
        conc = float(np.linalg.norm(mean_n))
        if conc < BLOB_NORMAL_CONC_MIN:
            rejected_reasons.append(f"blob#{bi}:nonplanar_normals({conc:.2f})")
            continue
        unit_n = mean_n / max(conc, 1e-9)
        if float(unit_n[2]) < FLOOR_NORMAL_Z_MIN:
            rejected_reasons.append(f"blob#{bi}:floor_surface(nz={unit_n[2]:.2f})")
            continue
        hit_z_frac = (float(hit[in_b, 2].mean()) - z_min_obj) / z_span_obj
        if (float(fit.get("elevation_deg", 0.0)) < 25.0
                and float(unit_n[2]) > ROOF_NORMAL_Z_MAX
                and hit_z_frac > 0.80):
            rejected_reasons.append(
                f"blob#{bi}:roof_surface(nz={unit_n[2]:.2f},zf={hit_z_frac:.2f})")
            continue
        bys, bxs = np.nonzero(comp_ref)
        b_span_ren = max(bys.max() - bys.min() + 1, bxs.max() - bxs.min() + 1) / max(scale, 1e-6)
        t_b = t[in_b]
        t_span = float(np.percentile(t_b, 95) - np.percentile(t_b, 5))
        max_planar_t = 1.6 * b_span_ren * float(t_b.mean()) / max(focal, 1.0) + 0.015 * float(fit["radius"])
        if t_span > max_planar_t:
            rejected_reasons.append(f"blob#{bi}:depth_jump({t_span:.4f}>{max_planar_t:.4f})")
            continue
        # Only skip rewriting if BOTH strokes and immediate/wide surround are already crisp.
        near_ring_ref = _dilate_bool(comp_ref, 6) & ~comp_ref & ref_alpha
        in_ring_h = near_ring_ref[ry_all, rx_all] & (cos_inc >= cos_min)
        stroke_err = float(np.abs(tex[tv[in_b], tu[in_b]] - ref_rgb[ry_all[in_b], rx_all[in_b]]).mean())
        ring_err = (float(np.abs(tex[tv[in_ring_h], tu[in_ring_h]]
                                 - ref_rgb[ry_all[in_ring_h], rx_all[in_ring_h]]).mean())
                    if int(in_ring_h.sum()) >= 6 else 999.0)
        major_b, minor_b, _ = _oriented_extent(bys, bxs)
        is_text_b = (bool((comp_ref & text_boxes_ref).any())
                     or (major_b / max(minor_b, 1.0)) > 2.8)
        wide_pad = (max(8, int(round(0.50 * minor_b))) if is_text_b
                    else max(8, int(round(0.75 * max(bys.max() - bys.min() + 1, bxs.max() - bxs.min() + 1)))))
        wide_ring_ref = _dilate_bool(comp_ref, wide_pad) & ~comp_ref & ref_alpha
        if near_ring_ref.any():
            p_med = np.median(ref_rgb[near_ring_ref], axis=0)
            wide_ring_ref &= np.linalg.norm(ref_rgb - p_med[None, None, :], axis=-1) <= 35.0
        in_wide_h = wide_ring_ref[ry_all, rx_all] & (cos_inc >= cos_min)
        wide_p95 = (float(np.percentile(
            np.abs(tex[tv[in_wide_h], tu[in_wide_h]] - ref_rgb[ry_all[in_wide_h], rx_all[in_wide_h]]).max(axis=-1),
            95)) if int(in_wide_h.sum()) >= 12 else 0.0)
        if stroke_err <= 10.0 and ring_err <= 10.0 and wide_p95 <= 22.0:
            rejected_reasons.append(f"blob#{bi}:already_crisp")
            continue
        accepted_blobs += 1
        slot = accepted_blobs
        accepted_blob_mask |= comp_ref
        accepted_comps.append((slot, comp_ref))
        if is_text_b:
            text_slots.add(slot)
        blob_planes[slot] = (unit_n, hit[in_b].mean(axis=0))

    if not accepted_blob_mask.any():
        return texture_img.copy(), {
            "projected_texels": 0,
            "accepted_blobs": 0,
            "rejected_blobs": rejected_reasons,
            "visible_faces": int(len(set(fids.tolist()))),
            "note": "no blob passed 3D surface continuity gates",
        }

    footprint, core_patch_mask, stroke_mask, owner_ref_map = badge_patch_weights(
        accepted_comps, ref_rgb, ref_alpha, text_slots=text_slots)
    accepted_blob_mask = stroke_mask if stroke_mask.any() else accepted_blob_mask

    plane_tol = 0.042 * float(fit["radius"])
    slot_shifts = {}
    for slot, comp_ref in accepted_comps:
        u_n, c_3d = blob_planes[slot]
        ndot_all = np.einsum("ij,j->i", n_cam, u_n)
        pdist_all = np.abs(np.einsum("ij,j->i", hit - c_3d[None, :], u_n))
        cop_all = (ndot_all >= 0.72) & (pdist_all <= plane_tol) & (cos_inc >= cos_min)
        vis_plane_s = np.zeros((size, size), dtype=bool)
        vis_plane_s[ys[cop_all], xs[cop_all]] = True
        vis_plane_s &= eroded_full
        ox_s, oy_s = _refine_local_decal_offset(
            ren_rgb_frame, vis_plane_s, ref_rgb, ref_alpha, comp_ref, scale, ox, oy,
            is_text=(slot in text_slots))
        slot_shifts[slot] = (float(ox_s - ox), float(oy_s - oy))

    owner_map, shift_maps = _build_per_blob_shift_maps(
        accepted_comps, owner_ref_map, slot_shifts)
    rxf_adj, ryf_adj = _apply_shift_with_collar_fallback(
        rxf, ryf, rx_all, ry_all, shift_maps, ref_alpha,
        fg_dist_ref=fg_dist_ref, stroke_mask=stroke_mask, footprint=footprint)
    in_adj = ((rxf_adj >= 0.0) & (rxf_adj < ref_rgb.shape[1])
              & (ryf_adj >= 0.0) & (ryf_adj < ref_rgb.shape[0]))
    ry_adj = np.clip(ryf_adj.astype(int), 0, bm_bool.shape[0] - 1)
    rx_adj = np.clip(rxf_adj.astype(int), 0, bm_bool.shape[1] - 1)
    in_adj &= ref_alpha[ry_adj, rx_adj]

    owners_hit = owner_map[ry_all, rx_all]
    coplanar = np.zeros(len(ys), dtype=bool)
    for slot, (u_n, c_3d) in blob_planes.items():
        sel_s = owners_hit == slot
        if not bool(sel_s.any()):
            continue
        ndot = np.einsum("ij,j->i", n_cam[sel_s], u_n)
        pdist = np.abs(np.einsum("ij,j->i", hit[sel_s] - c_3d[None, :], u_n))
        coplanar[sel_s] = (ndot >= 0.60) & (pdist <= plane_tol)

    on_decal = (in_adj & (footprint[ry_adj, rx_adj] > DECAL_FEATHER_MIN)
                & (cos_inc >= cos_min) & coplanar)
    ys_fwd, xs_fwd = ys[on_decal], xs[on_decal]
    tu_fwd, tv_fwd = tu[on_decal], tv[on_decal]
    t_fwd, hit_fwd, n_fwd, inc_fwd = (
        t[on_decal], hit[on_decal], n_cam[on_decal], inc_w[on_decal])
    rxf_fwd, ryf_fwd, ry_fwd, rx_fwd = (
        rxf_adj[on_decal], ryf_adj[on_decal], ry_adj[on_decal], rx_adj[on_decal])
    wgt_fwd = surface_feather[ys_fwd, xs_fwd] * inc_fwd * footprint[ry_fwd, rx_fwd]
    keep_fwd = wgt_fwd > 0.01
    tu_fwd, tv_fwd = tu_fwd[keep_fwd], tv_fwd[keep_fwd]
    t_fwd, hit_fwd, n_fwd = t_fwd[keep_fwd], hit_fwd[keep_fwd], n_fwd[keep_fwd]
    rxf_fwd, ryf_fwd = rxf_fwd[keep_fwd], ryf_fwd[keep_fwd]
    ry_fwd, rx_fwd, wgt_fwd = ry_fwd[keep_fwd], rx_fwd[keep_fwd], wgt_fwd[keep_fwd]
    prio_fwd = np.full(len(tu_fwd), 2, dtype=np.int8)

    gathered = _gather_uv_decal_texels(
        F, V, UV, tw, th,
        cam, right, up, forward, focal, size, float(fit["radius"]),
        scale, ox, oy, ref_alpha,
        pos_raw, np.isfinite(t_grid), surface_feather,
        footprint, owner_map, blob_planes,
        cos_min, cos_full, plane_tol,
        shift_maps=shift_maps,
        fg_dist_ref=fg_dist_ref,
        stroke_mask=stroke_mask,
    )
    if gathered is not None:
        (tu_uv, tv_uv, prio_uv, t_uv, hit_uv, n_uv,
         rxf_uv, ryf_uv, ry_uv, rx_uv, wgt_uv) = gathered
        if len(tu_fwd):
            tu = np.concatenate([tu_uv, tu_fwd])
            tv = np.concatenate([tv_uv, tv_fwd])
            prio = np.concatenate([prio_uv, prio_fwd])
            t = np.concatenate([t_uv, t_fwd])
            hit = np.vstack([hit_uv, hit_fwd])
            n_cam = np.vstack([n_uv, n_fwd])
            rxf = np.concatenate([rxf_uv, rxf_fwd])
            ryf = np.concatenate([ryf_uv, ryf_fwd])
            ry_i = np.concatenate([ry_uv, ry_fwd])
            rx_i = np.concatenate([rx_uv, rx_fwd])
            wgt = np.concatenate([wgt_uv, wgt_fwd])
        else:
            tu, tv, prio, t, hit, n_cam = tu_uv, tv_uv, prio_uv, t_uv, hit_uv, n_uv
            rxf, ryf, ry_i, rx_i, wgt = rxf_uv, ryf_uv, ry_uv, rx_uv, wgt_uv
    else:
        tu, tv, prio, t, hit, n_cam = tu_fwd, tv_fwd, prio_fwd, t_fwd, hit_fwd, n_fwd
        rxf, ryf, ry_i, rx_i, wgt = rxf_fwd, ryf_fwd, ry_fwd, rx_fwd, wgt_fwd

    if not len(tu):
        return texture_img.copy(), {
            "projected_texels": 0,
            "visible_faces": int(len(set(fids.tolist()))),
            "note": ("emblem footprint not visible from the fitted pose"
                     if not bool(on_decal.any())
                     else "feather band empty after rim/obliquity gating"),
        }

    x0 = np.floor(rxf).astype(int).clip(0, ref_rgb.shape[1] - 2)
    y0 = np.floor(ryf).astype(int).clip(0, ref_rgb.shape[0] - 2)
    fx, fy = rxf - x0, ryf - y0
    sample = (ref_rgb[y0, x0] * ((1 - fx) * (1 - fy))[:, None] +
              ref_rgb[y0, x0 + 1] * (fx * (1 - fy))[:, None] +
              ref_rgb[y0 + 1, x0] * ((1 - fx) * fy)[:, None] +
              ref_rgb[y0 + 1, x0 + 1] * (fx * fy)[:, None])

    # Match local surround brightness per panel/slot on clean host-panel pixels
    # using a scalar luminance gain (filtering out smeared emblem hues in the
    # baked texture so a red TRELLIS smudge can never pink-tint white badge quadrants).
    in_core_blob = accepted_blob_mask[ry_i, rx_i]
    owners_sample = np.where(
        owner_ref_map[ry_i, rx_i] > 0,
        owner_ref_map[ry_i, rx_i],
        owner_map[ry_i, rx_i],
    )
    mx_s, mn_s = sample.max(axis=1), sample.min(axis=1)
    sat_s = (mx_s - mn_s) / np.maximum(mx_s, 1e-6)
    emblem_lock = np.where(
        in_core_blob | (mx_s < 95.0),
        1.0,
        np.clip((sat_s - 0.22) / 0.22, 0.0, 1.0),
    )
    for slot, _comp_ref in accepted_comps:
        slot_sel = owners_sample == slot
        ring_sel = slot_sel & ~in_core_blob
        if int(ring_sel.sum()) < 8:
            continue
        baked_cand = tex[tv[ring_sel], tu[ring_sel]]
        ref_cand = sample[ring_sel]
        ref_mx, ref_mn = ref_cand.max(axis=1), ref_cand.min(axis=1)
        ref_chr = (ref_mx - ref_mn) / np.maximum(ref_mx, 1e-6)
        bak_mx, bak_mn = baked_cand.max(axis=1), baked_cand.min(axis=1)
        bak_chr = (bak_mx - bak_mn) / np.maximum(bak_mx, 1e-6)
        clean_bak = np.abs(bak_chr - np.median(ref_chr)) < 0.16
        if int(clean_bak.sum()) < 8:
            continue
        b_luma = float(np.median(0.299 * baked_cand[clean_bak, 0]
                                 + 0.587 * baked_cand[clean_bak, 1]
                                 + 0.114 * baked_cand[clean_bak, 2]))
        r_luma = max(float(np.median(0.299 * ref_cand[clean_bak, 0]
                                     + 0.587 * ref_cand[clean_bak, 1]
                                     + 0.114 * ref_cand[clean_bak, 2])), 16.0)
        luma_gain = float(np.clip(b_luma / r_luma, 0.88, 1.12))
        adj = np.clip(sample[slot_sel] * luma_gain, 0.0, 255.0)
        el_slot = emblem_lock[slot_sel, None]
        sample[slot_sel] = sample[slot_sel] * el_slot + adj * (1.0 - el_slot)

    out = tex.copy()
    order = np.lexsort((t, prio, tv * tw + tu))
    seen_keys = set()
    keep_list = []
    for idx in order:
        key = (int(tv[idx]), int(tu[idx]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        keep_list.append(idx)
    idxs = np.array(keep_list, dtype=np.int64)
    keys = np.stack([tv[idxs], tu[idxs]], axis=1)
    if len(keys) > MAX_PROJECTED_TEXEL_SHARE * tw * th:
        return texture_img.copy(), {
            "projected_texels": 0,
            "texel_share": round(float(len(keys)) / (tw * th), 4),
            "visible_faces": int(len(set(fids.tolist()))),
            "note": "emblem footprint covers most of the texture; refusing",
        }

    conflict, _pairs = shared_surface_texels(
        uv_face_index(F, UV), F, V, UV, keys[:, 1], keys[:, 0], tw, th,
        hit[idxs], fit["radius"] * SHARED_TEXEL_TOL_FRAC)
    skipped = int(conflict.sum())
    if skipped:
        idxs, keys = idxs[~conflict], keys[~conflict]
        if not len(idxs):
            return texture_img.copy(), {
                "projected_texels": 0, "texel_share": 0.0,
                "shared_texels_skipped": skipped,
                "visible_faces": int(len(set(fids.tolist()))),
                "note": "every candidate texel is shared with another surface; refusing",
            }

    weights = wgt[idxs]
    out[keys[:, 0], keys[:, 1]] = (
        sample[idxs] * weights[:, None]
        + out[keys[:, 0], keys[:, 1]] * (1.0 - weights[:, None]))

    w_max = float(weights.max()) if len(weights) else 0.0
    w_thresh = max(0.40, w_max * 0.75) if w_max >= 0.20 else 0.50
    strong = weights >= w_thresh
    if strong.any():
        expected = (
            sample[idxs][strong] * weights[strong, None]
            + tex[keys[strong, 0], keys[strong, 1]] * (1.0 - weights[strong, None]))
        err = np.abs(out[keys[strong, 0], keys[strong, 1]] - expected).max(axis=1)
        writeback = float((err <= WRITEBACK_TOLERANCE).mean())
    else:
        writeback = 0.0

    if out_state is not None:
        out_state["accepted_blob_mask"] = accepted_blob_mask
        out_state["patch_footprint"] = footprint
        out_state["core_patch_mask"] = core_patch_mask
        out_state["accepted_hits"] = hit[idxs].copy()
        out_state["accepted_normals"] = n_cam[idxs].copy()
        out_state["shift_maps"] = shift_maps
        out_state["stroke_mask"] = stroke_mask

    report = {
        "projected_texels": int(len(keys)),
        "texel_share": round(float(len(keys)) / (tw * th), 4),
        "blob_pixels": int(accepted_blob_mask.sum()),
        "accepted_blobs": int(accepted_blobs),
        "rejected_blobs": rejected_reasons,
        "shared_texels_skipped": skipped,
        "writeback_match": round(writeback, 4),
        "strong_count": int(strong.sum()),
        "weights_max": round(float(weights.max()), 4) if len(weights) else 0.0,
        "weights_mean": round(float(weights.mean()), 4) if len(weights) else 0.0,
        "visible_faces": int(len(set(fids.tolist()))),
        "frame_scale": round(float(scale), 4),
        "frame_offset": [round(float(ox), 2), round(float(oy), 2)],
        "blob_shifts": {
            str(k): [round(v[0], 1), round(v[1], 1)]
            for k, v in slot_shifts.items()
            if abs(v[0]) >= 0.5 or abs(v[1]) >= 0.5
        },
    }
    return _pack_output_texture(out, orig_alpha), report


def _frame_map(ren_mask, ref_nonzero, ref_shape, size):
    """Uniform scale + offset mapping render pixels onto reference pixels."""
    rys, rxs = ref_nonzero
    ys, xs = np.nonzero(ren_mask)
    if not len(ys) or not len(rys):
        return 1.0, 0.0, 0.0
    ren_h = ys.max() - ys.min() + 1
    ren_w = xs.max() - xs.min() + 1
    ref_h = rys.max() - rys.min() + 1
    ref_w = rxs.max() - rxs.min() + 1
    scale = min(ref_h / max(ren_h, 1), ref_w / max(ren_w, 1))
    ox = (rxs.min() + rxs.max() + 1) / 2 - (xs.min() + xs.max() + 1) / 2 * scale
    oy = (rys.min() + rys.max() + 1) / 2 - (ys.min() + ys.max() + 1) / 2 * scale
    return float(scale), float(ox), float(oy)


def _refine_frame_map(ren_rgb, ren_core, ref_rgb, ref_core, scale, ox, oy):
    """Refine (scale, ox, oy) against baked mesh features when texture detail exists."""
    ys, xs = np.nonzero(ren_core)
    if len(ys) < 64:
        return scale, ox, oy
    if float(ren_rgb[ren_core].std()) < 4.0 or float(ref_rgb[ref_core].std()) < 4.0:
        return scale, ox, oy

    size = ren_core.shape[0]
    step = max(1, size // 128)
    rc_s = ren_core[::step, ::step]
    rr_s = ren_rgb[::step, ::step]
    gh, gw = rc_s.shape
    gy, gx = np.mgrid[0:gh, 0:gw]
    px = (gx + 0.5) * step
    py = (gy + 0.5) * step
    cx = float((xs.min() + xs.max() + 1) * 0.5)
    cy = float((ys.min() + ys.max() + 1) * 0.5)

    ref_blur = np.asarray(
        Image.fromarray(np.clip(ref_rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
        dtype=np.float32)
    h_ref, w_ref = ref_core.shape
    l_ren = 0.299 * rr_s[..., 0] + 0.587 * rr_s[..., 1] + 0.114 * rr_s[..., 2]
    c_ren = rr_s[..., 0] - 0.5 * (rr_s[..., 1] + rr_s[..., 2])
    l_ref_full = 0.299 * ref_blur[..., 0] + 0.587 * ref_blur[..., 1] + 0.114 * ref_blur[..., 2]
    c_ref_full = ref_blur[..., 0] - 0.5 * (ref_blur[..., 1] + ref_blur[..., 2])

    def _eval(s_t, ox_t, oy_t):
        rx = np.rint(px * s_t + ox_t).astype(np.int64)
        ry = np.rint(py * s_t + oy_t).astype(np.int64)
        ib = (rx >= 0) & (rx < w_ref) & (ry >= 0) & (ry < h_ref)
        rxc = np.clip(rx, 0, w_ref - 1)
        ryc = np.clip(ry, 0, h_ref - 1)
        ref_m = ib & ref_core[ryc, rxc]
        uni = float((rc_s | ref_m).sum())
        if uni < 16.0:
            return 0.0, -1.0
        iou = float((rc_s & ref_m).sum()) / uni
        both = rc_s & ref_m
        if int(both.sum()) < 32:
            return iou, -1.0
        s_l = _ncc_1d(l_ren[both], l_ref_full[ryc[both], rxc[both]])
        cr_a = c_ren[both]
        cr_b = c_ref_full[ryc[both], rxc[both]]
        if float(cr_a.std()) > 2.0 and float(cr_b.std()) > 3.5:
            s_c = _ncc_1d(cr_a, cr_b)
            app = 0.72 * s_l + 0.28 * s_c
        else:
            app = s_l
        return iou, app

    base_iou, base_app = _eval(scale, ox, oy)
    base_comb = 0.65 * base_app + 0.35 * base_iou
    best = (scale, ox, oy)
    best_comb = base_comb
    shift_span = max(8, int(round(0.015 * max(h_ref, w_ref))))
    shift_step = max(2, shift_span // 3)
    shifts = list(range(-shift_span, shift_span + 1, shift_step))
    if 0 not in shifts:
        shifts.append(0)
    for s_mul in (0.97, 0.985, 1.0, 1.015, 1.03):
        s_t = scale * s_mul
        ox_c = ox + cx * (scale - s_t)
        oy_c = oy + cy * (scale - s_t)
        for dox in shifts:
            for doy in shifts:
                iou_t, app_t = _eval(s_t, ox_c + dox, oy_c + doy)
                if iou_t < base_iou - 0.025:
                    continue
                comb_t = 0.65 * app_t + 0.35 * iou_t
                if comb_t > best_comb:
                    best_comb = comb_t
                    best = (float(s_t), float(ox_c + dox), float(oy_c + doy))
    if best_comb > base_comb + 0.012:
        return best
    return scale, ox, oy


def _refine_local_decal_offset(ren_rgb, ren_vis_mask, ref_rgb, ref_alpha,
                               blob_mask, scale, ox, oy, *, is_text: bool = False):
    """Snap `(ox, oy)` locally to TRELLIS's baked panel features around one emblem.

    Whole-object silhouette registration (`_frame_map` / `_refine_frame_map`) can
    leave a 0.5-4.5% non-rigid 2D-to-3D perspective offset on individual oblique
    panels (front hood, cab alcove, tent gable), causing a projected insignia to
    land beside TRELLIS's centered 3D embossed/baked feature. Correlating local
    opponent chroma + luminance on coplanar panel pixels snaps each emblem
    independently onto its 3D panel feature while leaving blank panels untouched.
    """
    bm = np.asarray(blob_mask, dtype=bool)
    bys, bxs = np.nonzero(bm)
    if not len(bys):
        return ox, oy
    ys_s, xs_s = np.nonzero(ren_vis_mask)
    if len(ys_s) < 28:
        return ox, oy

    h_ref, w_ref = ref_alpha.shape
    major_ref, minor_ref, _ = _oriented_extent(bys, bxs)
    if is_text or (major_ref / max(minor_ref, 1.0)) > 2.8:
        max_shift = min(
            max(10, int(round(0.045 * max(h_ref, w_ref)))),
            max(10, int(round(0.75 * minor_ref))),
        )
        pad_x = max(24, int(round(0.35 * major_ref)) + max_shift)
        pad_y = max(20, int(round(1.00 * minor_ref)) + max_shift)
    else:
        span_ref = max(int(bys.max() - bys.min() + 1), int(bxs.max() - bxs.min() + 1))
        max_shift = min(
            max(10, int(round(0.045 * max(h_ref, w_ref)))),
            max(10, int(round(0.75 * span_ref))),
        )
        pad_x = pad_y = max(28, int(round(1.15 * span_ref)) + max_shift)
    rx_init = (xs_s + 0.5) * scale + ox
    ry_init = (ys_s + 0.5) * scale + oy
    in_roi = (
        (rx_init >= float(bxs.min() - pad_x))
        & (rx_init <= float(bxs.max() + pad_x))
        & (ry_init >= float(bys.min() - pad_y))
        & (ry_init <= float(bys.max() + pad_y))
    )
    if int(in_roi.sum()) < 28:
        return ox, oy

    ys_r, xs_r = ys_s[in_roi], xs_s[in_roi]
    rx0, ry0 = rx_init[in_roi], ry_init[in_roi]
    rr = ren_rgb[ys_r, xs_r]
    l_r = 0.299 * rr[:, 0] + 0.587 * rr[:, 1] + 0.114 * rr[:, 2]
    crg_r = rr[:, 0] - 0.5 * (rr[:, 1] + rr[:, 2])
    cby_r = rr[:, 2] - 0.5 * (rr[:, 0] + rr[:, 1])
    sat_r = rr.max(axis=1) - rr.min(axis=1)
    if float(l_r.std()) < 2.2 and float(sat_r.std()) < 2.2:
        return ox, oy

    ref_blur = np.asarray(
        Image.fromarray(np.clip(ref_rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
        dtype=np.float32,
    )
    l_ref_full = 0.299 * ref_blur[..., 0] + 0.587 * ref_blur[..., 1] + 0.114 * ref_blur[..., 2]
    crg_ref_full = ref_blur[..., 0] - 0.5 * (ref_blur[..., 1] + ref_blur[..., 2])
    cby_ref_full = ref_blur[..., 2] - 0.5 * (ref_blur[..., 0] + ref_blur[..., 1])
    sat_ref_full = ref_blur.max(axis=-1) - ref_blur.min(axis=-1)

    def _eval_shift(dox: int, doy: int):
        rxi = np.rint(rx0 + dox).astype(np.int64)
        ryi = np.rint(ry0 + doy).astype(np.int64)
        ib = (rxi >= 0) & (rxi < w_ref) & (ryi >= 0) & (ryi < h_ref)
        rxc = np.clip(rxi, 0, w_ref - 1)
        ryc = np.clip(ryi, 0, h_ref - 1)
        fg = ib & ref_alpha[ryc, rxc]
        if int(fg.sum()) < max(20, int(0.80 * len(rx0))):
            return -2.0, -2.0
        s_l = _ncc_1d(l_r[fg], l_ref_full[ryc[fg], rxc[fg]])
        c_terms = []
        for c_mesh, c_ref_im in (
            (crg_r[fg], crg_ref_full[ryc[fg], rxc[fg]]),
            (cby_r[fg], cby_ref_full[ryc[fg], rxc[fg]]),
            (sat_r[fg], sat_ref_full[ryc[fg], rxc[fg]]),
        ):
            if float(c_mesh.std()) > 2.5 and float(c_ref_im.std()) > 2.5:
                c_terms.append(_ncc_1d(c_mesh, c_ref_im))
        if c_terms:
            s_c = float(max(c_terms))
            ncc = 0.32 * s_l + 0.68 * s_c
        else:
            ncc = s_l
        pen = 0.020 * ((dox / max_shift) ** 2 + (doy / max_shift) ** 2)
        return ncc - pen, ncc

    base_score, _ = _eval_shift(0, 0)
    best_score = base_score
    best_ncc = -2.0
    best_d = (0, 0)
    step_c = max(2, max_shift // 8)
    coarse = list(range(-max_shift, max_shift + 1, step_c))
    if 0 not in coarse:
        coarse.append(0)
    for dox in coarse:
        for doy in coarse:
            sc, nc = _eval_shift(dox, doy)
            if sc > best_score:
                best_score, best_ncc, best_d = sc, nc, (dox, doy)

    cx_d, cy_d = best_d
    for dox in range(max(-max_shift, cx_d - step_c), min(max_shift, cx_d + step_c) + 1):
        for doy in range(max(-max_shift, cy_d - step_c), min(max_shift, cy_d + step_c) + 1):
            sc, nc = _eval_shift(dox, doy)
            if sc > best_score:
                best_score, best_ncc, best_d = sc, nc, (dox, doy)

    if best_d != (0, 0) and best_score > base_score + 0.030 and best_ncc >= 0.35:
        return float(ox + best_d[0]), float(oy + best_d[1])
    return ox, oy


def _build_per_blob_shift_maps(accepted_comps, owner_ref_map: np.ndarray, slot_shifts: dict):
    """Map base-frame coordinates `(ry0, rx0)` to `(owner_slot, (dox_map, doy_map))`."""
    h, w = owner_ref_map.shape
    sx_map = np.zeros((h, w), dtype=np.float32)
    sy_map = np.zeros((h, w), dtype=np.float32)
    if not slot_shifts or all(abs(dx) < 0.5 and abs(dy) < 0.5 for dx, dy in slot_shifts.values()):
        return owner_ref_map, (sx_map, sy_map)

    owner_base = np.zeros((h, w), dtype=np.int32)
    best_dist2 = np.full((h, w), np.inf, dtype=np.float32)
    for slot, comp_ref in accepted_comps:
        dox_f, doy_f = slot_shifts.get(slot, (0.0, 0.0))
        dx_i = int(round(dox_f))
        dy_i = int(round(doy_f))
        dom_ref = _dilate_bool(owner_ref_map == slot, 6)
        ys_d, xs_d = np.nonzero(dom_ref)
        if not len(ys_d):
            continue
        bys, bxs = np.nonzero(comp_ref)
        cy_b = float(bys.mean()) - doy_f
        cx_b = float(bxs.mean()) - dox_f
        yb = ys_d - dy_i
        xb = xs_d - dx_i
        ok_b = (yb >= 0) & (yb < h) & (xb >= 0) & (xb < w)
        yb, xb = yb[ok_b], xb[ok_b]
        if not len(yb):
            continue
        d2 = ((yb - cy_b) ** 2 + (xb - cx_b) ** 2).astype(np.float32)
        win = d2 < best_dist2[yb, xb]
        yw, xw = yb[win], xb[win]
        best_dist2[yw, xw] = d2[win]
        owner_base[yw, xw] = int(slot)
        sx_map[yw, xw] = float(dox_f)
        sy_map[yw, xw] = float(doy_f)
    return owner_base, (sx_map, sy_map)


# ------------------------------------------------------------ gaussian patch
C0 = 0.2820947917738949


def patch_gaussian_colors(item_dir: Path, item_id: str, ref_rgba: Image.Image,
                          fit, blob_mask=None, *, frame_map=None,
                          accepted_hits=None, patch_footprint=None,
                          core_patch_mask=None, shift_maps=None,
                          stroke_mask=None,
                          commit: bool = True, staged_out: list = None) -> dict:
    """Recolor 3D-validated emblem gaussians; optionally stage without promoting."""
    if blob_mask is None or not np.asarray(blob_mask).any():
        return {"patched_gaussians": 0, "visible_gaussians": 0,
                "note": "no blob mask; refusing whole-side repaint"}
    ref = ref_rgba.convert("RGBA")
    ref_rgb = np.asarray(ref.convert("RGB")).astype(np.float32)
    ref_alpha = np.asarray(ref)[:, :, 3] > 127
    fg_dist_ref = _fg_distance(ref_alpha)
    cam = np.array(fit["cam"], dtype=np.float64)
    right = np.array(fit["right"], dtype=np.float64)
    up = np.array(fit["up"], dtype=np.float64)
    forward = np.array(fit["forward"], dtype=np.float64)
    size = RENDER_SIZE
    focal_ndc = float(fit.get("focal_ndc", 3.0))
    focal = 0.5 * focal_ndc * size

    containers = []
    for ext in GS_EXTENSIONS:
        p = item_dir / f"{item_id}.{ext}"
        if p.is_file() and p.stat().st_size > 0:
            containers.append((ext, p))
    if not containers:
        return {"patched_gaussians": 0, "visible_gaussians": 0, "note": "no containers"}

    try:
        from qa import load_gaussians as _load
    except Exception:
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "qa", str(Path(__file__).resolve().parent / "qa.py"))
            _qa = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_qa)
            _load = _qa.load_gaussians
        except Exception as e:
            return {"patched_gaussians": 0, "visible_gaussians": 0,
                    "note": f"gaussian loader unavailable: {e}"}

    g, _ext = _load(item_dir, item_id)
    if g is None or not len(g["x"]):
        return {"patched_gaussians": 0, "visible_gaussians": 0, "note": "empty cloud"}

    pos = np.stack([np.asarray(g["x"], dtype=np.float64),
                    np.asarray(g["y"], dtype=np.float64),
                    np.asarray(g["z"], dtype=np.float64)], axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rel = pos - cam[None, :]
        z = (rel * forward[None, :]).sum(axis=1)
        x = (rel * right[None, :]).sum(axis=1)
        y = (rel * up[None, :]).sum(axis=1)
    ok = np.isfinite(z) & (z > 1e-6)
    px = np.full(len(pos), -1.0)
    py = np.full(len(pos), -1.0)
    px[ok] = x[ok] / z[ok] * focal + size / 2.0
    py[ok] = -y[ok] / z[ok] * focal + size / 2.0
    in_img = ok & (px >= 0) & (px < size) & (py >= 0) & (py < size)
    idx = np.nonzero(in_img)[0]
    if not len(idx):
        return {"patched_gaussians": 0, "visible_gaussians": 0,
                "note": "no gaussians in fitted view"}
    visible = int(len(idx))

    if frame_map is not None:
        scale, ox, oy = float(frame_map[0]), float(frame_map[1]), float(frame_map[2])
    else:
        ren_mask = np.zeros((size, size), bool)
        ren_mask[py[idx].astype(int).clip(0, size - 1),
                 px[idx].astype(int).clip(0, size - 1)] = True
        rys, rxs = np.nonzero(_core_silhouette_mask(ref_alpha, kernel=9))
        scale, ox, oy = _frame_map(ren_mask, (rys, rxs), ref_alpha.shape, size)

    bm = np.asarray(blob_mask, dtype=bool)
    if bm.shape != ref_alpha.shape:
        bm = np.zeros_like(ref_alpha, bool)
    footprint = (np.asarray(patch_footprint, dtype=np.float32)
                 if patch_footprint is not None and np.asarray(patch_footprint).shape == ref_alpha.shape
                 else decal_weights(bm))

    rxf0 = (px[idx] + 0.5) * scale + ox
    ryf0 = (py[idx] + 0.5) * scale + oy
    if shift_maps is not None:
        xi0 = np.clip(rxf0.astype(int), 0, ref_rgb.shape[1] - 1)
        yi0 = np.clip(ryf0.astype(int), 0, ref_rgb.shape[0] - 1)
        sm_eff = stroke_mask if stroke_mask is not None else bm
        rxf, ryf = _apply_shift_with_collar_fallback(
            rxf0, ryf0, xi0, yi0, shift_maps, ref_alpha,
            fg_dist_ref=fg_dist_ref, stroke_mask=sm_eff, footprint=footprint)
    else:
        rxf = rxf0
        ryf = ryf0
    ok4 = (rxf >= 0) & (rxf < ref_rgb.shape[1]) & (ryf >= 0) & (ryf < ref_rgb.shape[0])
    xi = rxf.astype(int).clip(0, ref_rgb.shape[1] - 1)
    yi = ryf.astype(int).clip(0, ref_rgb.shape[0] - 1)
    on_fg = np.zeros(len(idx), bool)
    on_fg[ok4] = ref_alpha[yi[ok4], xi[ok4]]

    w_all = np.zeros(len(idx), dtype=np.float32)
    w_all[ok4] = footprint[yi[ok4], xi[ok4]]
    use_g = on_fg & (w_all > DECAL_FEATHER_MIN)
    sel = idx[use_g]
    if not len(sel):
        return {"patched_gaussians": 0, "visible_gaussians": visible,
                "note": "visible gaussians miss emblem footprint"}

    _keep = None
    depth_note = "depth test skipped"
    try:
        vox_r = np.exp(np.asarray(g["scale"], dtype=np.float64)[sel][:, 0]) / 0.55
    except Exception:
        vox_r = np.full(len(sel), 0.005, dtype=np.float64)

    if accepted_hits is not None and len(accepted_hits) > 0:
        try:
            from scipy.spatial import cKDTree
            dists, _ = cKDTree(np.asarray(accepted_hits, dtype=np.float64)).query(pos[sel], k=1)
        except ImportError:
            diff = pos[sel][:, None, :] - np.asarray(accepted_hits, dtype=np.float64)[None, :, :]
            dists = np.linalg.norm(diff, axis=-1).min(axis=1)
        tol = float(fit["radius"]) * 6e-3 + 2.5 * vox_r
        _keep = np.isfinite(dists) & (dists <= tol)
        sel = sel[_keep]
        depth_note = f"3d_patch_bound {int(_keep.sum())}/{len(_keep)}"
        if not len(sel):
            return {"patched_gaussians": 0, "visible_gaussians": visible,
                    "note": f"no gaussians near accepted 3D patch ({depth_note})"}
    else:
        try:
            _size = 512
            _vp = _vp_matrix(cam, right, up, forward, _size,
                             focal_ndc=focal_ndc).T.astype("f4").tobytes()
            with _mgl_context() as _ctx:
                import moderngl as _mgl
                _prog = _ctx.program(vertex_shader=POS_VERT, fragment_shader=POS_FRAG)
                _prog["view_projection"].write(_vp)
                _pt = _ctx.texture((_size, _size), 4, dtype="f4")
                _fb = _ctx.framebuffer([_pt], _ctx.depth_renderbuffer((_size, _size)))
                _fb.use()
                _ctx.enable(_mgl.DEPTH_TEST)
                _fb.clear(0, 0, 0, 0, depth=1)
                for _g in load_geoms_zup(item_dir / f"{item_id}.glb"):
                    _b = _ctx.buffer(np.asarray(_g.vertices, dtype="f4").tobytes())
                    _ix = _ctx.buffer(np.asarray(_g.faces, dtype="u4").tobytes())
                    _ctx.vertex_array(
                        _prog, [(_b, "3f", "position")], _ix,
                        index_element_size=4).render()
                _praw = np.frombuffer(_pt.read(alignment=1),
                                      dtype=np.float32).reshape(_size, _size, 4)[:, :, :3]
                _praw = np.asarray(_praw, dtype=np.float64)[::-1]
            _f = 0.5 * focal_ndc * _size
            _rel = pos[sel] - cam[None, :]
            _zz = (_rel * forward[None, :]).sum(axis=1)
            _xx = (_rel * right[None, :]).sum(axis=1)
            _yy = (_rel * up[None, :]).sum(axis=1)
            _gx = ((_xx / np.maximum(_zz, 1e-9)) * _f + _size / 2.0).astype(int).clip(0, _size - 1)
            _gy = ((-_yy / np.maximum(_zz, 1e-9)) * _f + _size / 2.0).astype(int).clip(0, _size - 1)
            _surf = _praw[_gy, _gx]
            _has = np.abs(_surf).sum(axis=-1) > 1e-9
            _dist = np.linalg.norm(pos[sel] - _surf, axis=1)
            _tol = float(fit["radius"]) * 5e-3 + 2.0 * vox_r
            _keep = _has & np.isfinite(_dist) & (_dist <= _tol)
            sel = sel[_keep]
            depth_note = f"depth kept {int(_keep.sum())}/{len(_keep)}"
            if not len(sel):
                return {"patched_gaussians": 0, "visible_gaussians": visible,
                        "note": f"all emblem gaussians occluded ({depth_note})"}
        except Exception as _e:
            depth_note = f"depth test unavailable: {_e}"

    sxi = (rxf[use_g][_keep] if _keep is not None else rxf[use_g]).astype(float)
    syi = (ryf[use_g][_keep] if _keep is not None else ryf[use_g]).astype(float)
    w_g = (w_all[use_g][_keep] if _keep is not None else w_all[use_g]).astype(np.float32)

    x0 = np.floor(sxi).astype(int).clip(0, ref_rgb.shape[1] - 2)
    y0 = np.floor(syi).astype(int).clip(0, ref_rgb.shape[0] - 2)
    fx_, fy_ = sxi - x0, syi - y0
    sample = (ref_rgb[y0, x0] * ((1 - fx_) * (1 - fy_))[:, None] +
              ref_rgb[y0, x0 + 1] * (fx_ * (1 - fy_))[:, None] +
              ref_rgb[y0 + 1, x0] * ((1 - fx_) * fy_)[:, None] +
              ref_rgb[y0 + 1, x0 + 1] * (fx_ * fy_)[:, None]) / 255.0

    # Recolor emblem strokes and validated badge interiors cleanly while
    # feathering the outer collar against existing splat RGB.
    core_m = (np.asarray(core_patch_mask, dtype=bool)
              if core_patch_mask is not None and np.asarray(core_patch_mask).shape == ref_alpha.shape
              else bm)
    on_core = core_m[np.clip(syi.astype(int), 0, core_m.shape[0] - 1),
                     np.clip(sxi.astype(int), 0, core_m.shape[1] - 1)]
    w_g = np.where(on_core, np.maximum(w_g, 0.90), w_g * 0.50)
    keep = w_g > 0.05
    sel, sample, w_g = sel[keep], sample[keep], w_g[keep]
    if not len(sel):
        return {"patched_gaussians": 0, "visible_gaussians": visible,
                "note": "no weighted emblem gaussians"}

    try:
        from gs_export import write_spz as _wspz, write_splat as _wsplat, \
            write_ply as _wply, tmp_sibling as _tmp
    except Exception as e:
        return {"patched_gaussians": 0, "visible_gaussians": visible,
                "note": f"gs writers unavailable: {e}"}

    base = {"x": np.asarray(g["x"]), "y": np.asarray(g["y"]),
            "z": np.asarray(g["z"]), "scale": np.asarray(g["scale"]),
            "rot": np.asarray(g["rot"]), "f_dc": np.asarray(g["f_dc"]).copy(),
            "opacity": np.asarray(g["opacity"])}
    if "sh_degree" in g:
        base["sh_degree"] = g["sh_degree"]
    old_rgb = np.clip(base["f_dc"][sel] * C0 + 0.5, 0.0, 1.0)
    blended_rgb = np.clip(sample * w_g[:, None] + old_rgb * (1.0 - w_g[:, None]), 0.0, 1.0)
    base["f_dc"][sel] = ((blended_rgb - 0.5) / C0).astype(np.float32)
    if "f_rest" not in g:
        base["f_rest"] = np.zeros((len(base["x"]), 45), dtype=np.float32)
    else:
        base["f_rest"] = np.asarray(g["f_rest"]).copy()
        if len(base["f_rest"]) == len(base["x"]) and base["f_rest"].ndim == 2 and base["f_rest"].shape[1] > 0:
            base["f_rest"][sel] *= (1.0 - w_g)[:, None]

    staged_pairs = []
    try:
        for ext, final in containers:
            tmp = _tmp(final)
            if ext == "spz":
                n = _wspz(str(tmp), base)
                if not n:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"SPZ writer failed for {final.name}")
            elif ext == "splat":
                _wsplat(str(tmp), base)
            else:
                _wply(str(tmp), base)
            if commit:
                durable_replace(tmp, final)
            else:
                staged_pairs.append((tmp, final))
    except Exception:
        _cleanup_staged(staged_pairs)
        raise
    if staged_out is not None:
        staged_out.extend(staged_pairs)

    patched = int(len(sel))
    return {"patched_gaussians": patched, "visible_gaussians": visible,
            "gaussian_share": round(patched / max(len(base["x"]), 1), 5),
            "depth": depth_note}


# ------------------------------------------------------------ GLB surgery

def replace_glb_texture(glb_path: Path, texture_index: int, png_bytes: bytes):
    """Swap one embedded PNG inside a GLB while preserving glTF 2.0 4-byte alignment."""
    data = glb_path.read_bytes()
    magic, version, glb_len = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    off = 12
    chunks = []
    while off < len(data):
        ln, ctype = struct.unpack("<II", data[off:off + 8])
        chunks.append((ctype, data[off + 8:off + 8 + ln]))
        off += 8 + ln
    js = json.loads(chunks[0][1].decode("utf-8"))
    bin_blob = bytearray()
    for ctype, payload in chunks[1:]:
        if ctype == 0x004E4942:
            bin_blob += payload
            break
    images = js.get("images", [])
    if texture_index >= len(images):
        raise ValueError(f"texture index {texture_index} out of range ({len(images)} images)")
    buffer_views = js["bufferViews"]
    bv = buffer_views[images[texture_index]["bufferView"]]
    start = int(bv.get("byteOffset", 0))
    old_len = int(bv["byteLength"])
    old_end = start + old_len
    # Consume existing inter-view zero padding up to the next bufferView so
    # repeated texture swaps neither accumulate padding bytes nor shift
    # subsequent FLOAT32/UINT32 accessor views off 4-byte alignment. Also
    # self-heals GLBs previously shifted by earlier unpadded replacements.
    next_start = min(
        (int(other.get("byteOffset", 0)) for other in buffer_views
         if other is not bv and int(other.get("byteOffset", 0)) >= old_end),
        default=len(bin_blob),
    )
    overlaps_gap = any(
        other is not bv
        and int(other.get("byteOffset", 0)) < next_start
        and int(other.get("byteOffset", 0)) + int(other["byteLength"]) > old_end
        for other in buffer_views
    )
    if (not overlaps_gap and next_start > old_end
            and all(b == 0 for b in bin_blob[old_end:next_start])):
        old_end = next_start
    pad = (-(start + len(png_bytes))) % 4
    new_chunk = png_bytes + b"\x00" * pad
    new_blob = bytes(bin_blob[:start]) + new_chunk + bytes(bin_blob[old_end:])
    delta = len(new_chunk) - (old_end - start)
    bv["byteLength"] = len(png_bytes)
    for other in buffer_views:
        if other is not bv and int(other.get("byteOffset", 0)) > start:
            other["byteOffset"] = int(other.get("byteOffset", 0)) + delta
    # Trim any stale trailing zeros beyond the last bufferView, then pad the
    # BIN chunk to a 4-byte boundary as required by glTF 2.0 section 4.4.
    used_end = max(
        (int(other.get("byteOffset", 0)) + int(other["byteLength"]) for other in buffer_views),
        default=len(new_blob),
    )
    if used_end <= len(new_blob) and all(b == 0 for b in new_blob[used_end:]):
        new_blob = new_blob[:used_end]
    bin_pad = (-len(new_blob)) % 4
    if bin_pad:
        new_blob += b"\x00" * bin_pad
    js["buffers"][0]["byteLength"] = len(new_blob)
    js_raw = json.dumps(js, separators=(",", ":")).encode("utf-8")
    js_pad = (-len(js_raw)) % 4
    header_len = 12 + 8 + len(js_raw) + js_pad + 8 + len(new_blob)
    out = struct.pack("<III", 0x46546C67, 2, header_len)
    out += struct.pack("<II", len(js_raw) + js_pad, 0x4E4F534A)
    out += js_raw + b" " * js_pad
    out += struct.pack("<II", len(new_blob), 0x004E4942)
    out += bytes(new_blob)
    return out


def load_geoms_zup(glb_path: Path):
    """Load GLB geometry in the shared Z-up frame used by the QA rig."""
    import trimesh
    scene = trimesh.load(glb_path, force="scene")
    geoms = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh.apply_transform(transform)
        mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                       [0, 1, 0, 0], [0, 0, 0, 1]]))
        geoms.append(mesh)
    if not geoms:
        raise ValueError(f"No triangular geometry in {glb_path}")
    return geoms


def glb_basecolor_image_index(glb_path: Path) -> int:
    """Return the GLB images[] index backing the first baseColorTexture."""
    data = glb_path.read_bytes()
    ln = struct.unpack("<I", data[12:16])[0]
    js = json.loads(data[20:20 + ln].decode("utf-8"))
    for mat in js.get("materials", []):
        tex = ((mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture") or {})
        if "index" in tex:
            textures = js.get("textures", [])
            if tex["index"] < len(textures):
                img = textures[tex["index"]].get("source")
                if img is not None:
                    return int(img)
    return 0


def base_color_texture(geoms):
    """Return (texture_image, texture_index) preserving RGBA alpha when present."""
    for g in geoms:
        material = getattr(getattr(g, "visual", None), "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            texture = getattr(material, "image", None)
        if texture is not None:
            mode = "RGBA" if texture.mode == "RGBA" else "RGB"
            return texture.convert(mode), 0
    raise ValueError("no base color texture found")


# ------------------------------------------------------------ 3D verification

VERIFY_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
in vec2 texcoord;
out vec2 uv;
out vec3 v_pos;
void main() {
    uv = texcoord;
    v_pos = position;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""
VERIFY_FRAG = """#version 330
uniform sampler2D base_color;
in vec2 uv;
in vec3 v_pos;
layout(location = 0) out vec4 out_col;
layout(location = 1) out vec4 out_pos;
void main() {
    out_col = vec4(texture(base_color, uv).rgb, 1.0);
    out_pos = vec4(v_pos, 1.0);
}
"""


def _verify_multiview_render(geoms_zup, old_texture: Image.Image,
                             new_texture: Image.Image, ref_rgba: Image.Image,
                             blob_mask: np.ndarray, fit: dict,
                             proj_report: dict, accepted_hits=None,
                             patch_footprint=None, shift_maps=None) -> dict:
    """Render before/after textures from 5 views to verify presence and zero bleed."""
    import moderngl
    size = 384
    center = np.asarray(fit["center"], dtype=np.float64)
    radius = float(fit["radius"])
    az0 = math.radians(float(fit["azimuth_deg"]))
    el0 = math.radians(float(fit["elevation_deg"]))
    focal_ndc = float(fit.get("focal_ndc", 3.0))

    old_rgb_px = np.asarray(old_texture.convert("RGB"))[::-1].copy()
    new_rgb_px = np.asarray(new_texture.convert("RGB"))[::-1].copy()
    tw, th = old_texture.size

    views = [
        ("target", az0, el0),
        ("ortho_pos", (az0 + 0.5 * math.pi) % (2.0 * math.pi), el0),
        ("opposing", (az0 + math.pi) % (2.0 * math.pi), el0),
        ("ortho_neg", (az0 + 1.5 * math.pi) % (2.0 * math.pi), el0),
    ]
    if float(fit["elevation_deg"]) < 30.0:
        views.append(("roof", az0, math.radians(85.0)))

    with _mgl_context() as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        prog = ctx.program(vertex_shader=VERIFY_VERT, fragment_shader=VERIFY_FRAG)
        col_tex = ctx.texture((size, size), 4, dtype="f4")
        pos_tex = ctx.texture((size, size), 4, dtype="f4")
        fb = ctx.framebuffer([col_tex, pos_tex], ctx.depth_renderbuffer((size, size)))
        fb.use()

        tex_old_gpu = ctx.texture((tw, th), 3, old_rgb_px.tobytes(), alignment=1)
        tex_new_gpu = ctx.texture((tw, th), 3, new_rgb_px.tobytes(), alignment=1)
        for gt in (tex_old_gpu, tex_new_gpu):
            gt.filter = (moderngl.LINEAR, moderngl.LINEAR)

        vaos = []
        for g in geoms_zup:
            uv = np.asarray(g.visual.uv, dtype=np.float32)
            buf = ctx.buffer(np.column_stack([np.asarray(g.vertices, np.float32), uv]).astype("f4").tobytes())
            idx = ctx.buffer(np.asarray(g.faces, dtype="u4").tobytes())
            vaos.append(ctx.vertex_array(prog, [(buf, "3f 2f", "position", "texcoord")],
                                         idx, index_element_size=4))

        def render_pair(az, el):
            vp = _vp_matrix(*_orbit_pose(center, radius, az, el), size,
                            focal_ndc=focal_ndc).T.astype("f4").tobytes()
            prog["view_projection"].write(vp)
            outs = []
            for gt in (tex_old_gpu, tex_new_gpu):
                fb.clear(0, 0, 0, 0, depth=1)
                gt.use(0)
                for vao in vaos:
                    vao.render()
                c = np.frombuffer(col_tex.read(alignment=1), dtype=np.float32).reshape(size, size, 4)[::-1]
                p = np.frombuffer(pos_tex.read(alignment=1), dtype=np.float32).reshape(size, size, 4)[::-1]
                outs.append((c, p))
            return outs[0], outs[1]

        hit_tree = None
        if accepted_hits is not None and len(accepted_hits) > 0:
            try:
                from scipy.spatial import cKDTree
                hit_tree = cKDTree(np.asarray(accepted_hits, dtype=np.float64))
            except ImportError:
                hit_tree = None

        spatial_tol = 0.025 * radius
        opposing_bleed = 0.0
        roof_bleed = 0.0
        spatial_bleed = 0.0
        target_collateral = 0.0
        target_presence = 0.0

        bm = np.asarray(blob_mask, dtype=bool)
        footprint = ((np.asarray(patch_footprint, dtype=np.float32) > 0.005)
                     if patch_footprint is not None and np.asarray(patch_footprint).shape == bm.shape
                     else (decal_weights(bm) > 0.005))
        ref_arr = np.asarray(ref_rgba.convert("RGB"), dtype=np.float32) / 255.0
        # Scale frame_map from RENDER_SIZE to verification size.
        ratio_sz = float(RENDER_SIZE) / float(size)
        f_scale = float(proj_report.get("frame_scale", 1.0)) * ratio_sz
        f_ox, f_oy = proj_report.get("frame_offset", [0.0, 0.0])

        for name, az, el in views:
            (c_old, p_old), (c_new, _p_new) = render_pair(az, el)
            fg = c_new[..., 3] > 0.5
            n_fg = max(int(fg.sum()), 1)
            diff = np.abs(c_new[..., :3] - c_old[..., :3]).max(axis=-1)
            changed = fg & (diff > (4.0 / 255.0))

            if hit_tree is not None and changed.any():
                pts = p_old[..., :3][changed].astype(np.float64)
                d3, _ = hit_tree.query(pts, k=1)
                spatial_bleed = max(spatial_bleed, float((d3 > spatial_tol).sum()) / n_fg)
            elif accepted_hits is not None and len(accepted_hits) > 0 and changed.any():
                pts = p_old[..., :3][changed].astype(np.float64)
                ah = np.asarray(accepted_hits, dtype=np.float64)
                if len(ah) > 512:
                    ah = ah[np.linspace(0, len(ah) - 1, 512).astype(int)]
                d3 = np.linalg.norm(pts[:, None, :] - ah[None, :, :], axis=-1).min(axis=1)
                spatial_bleed = max(spatial_bleed, float((d3 > spatial_tol).sum()) / n_fg)

            if name == "target":
                step = max(1, int(round(f_scale)))
                fp_tol = _dilate_bool(footprint, step)
                gy, gx = np.mgrid[0:size, 0:size]
                rx0_f = (gx + 0.5) * f_scale + f_ox
                ry0_f = (gy + 0.5) * f_scale + f_oy
                rx0 = np.clip(rx0_f.astype(int), 0, bm.shape[1] - 1)
                ry0 = np.clip(ry0_f.astype(int), 0, bm.shape[0] - 1)
                if shift_maps is not None:
                    sx_map, sy_map = shift_maps
                    rx = np.clip((rx0_f + sx_map[ry0, rx0]).astype(int), 0, bm.shape[1] - 1)
                    ry = np.clip((ry0_f + sy_map[ry0, rx0]).astype(int), 0, bm.shape[0] - 1)
                else:
                    rx = rx0
                    ry = ry0
                outside = fg & (~fp_tol[ry, rx]) & (~fp_tol[ry0, rx0])
                if outside.any():
                    target_collateral = float((changed & outside).sum()) / n_fg
                on_b = fg & bm[ry, rx]
                if on_b.any():
                    # 1-pixel tolerance in render space accounts for subpixel rasterization.
                    h_ref, w_ref = bm.shape
                    c_new_b = c_new[..., :3][on_b]
                    min_errs = []
                    for dy in (-step, 0, step):
                        for dx in (-step, 0, step):
                            rys = np.clip(ry[on_b] + dy, 0, h_ref - 1)
                            rxs = np.clip(rx[on_b] + dx, 0, w_ref - 1)
                            min_errs.append(np.abs(c_new_b - ref_arr[rys, rxs]).max(axis=-1))
                    err = np.min(np.column_stack(min_errs), axis=1)
                    target_presence = float((err <= 0.30).mean())
            elif name == "opposing":
                # Exclude pixels that belong to the front 3D patch seen through
                # an open tent doorway from the rear.
                if hit_tree is not None and changed.any():
                    pts = p_old[..., :3][changed].astype(np.float64)
                    d3, _ = hit_tree.query(pts, k=1)
                    opposing_bleed = float((d3 > spatial_tol).sum()) / n_fg
                else:
                    opposing_bleed = float(changed.sum()) / n_fg
            elif name == "roof":
                if hit_tree is not None and changed.any():
                    pts = p_old[..., :3][changed].astype(np.float64)
                    d3, _ = hit_tree.query(pts, k=1)
                    roof_bleed = float((d3 > spatial_tol).sum()) / n_fg
                else:
                    roof_bleed = float(changed.sum()) / n_fg

        ok = (
            target_presence >= TRANSFER_MATCH_MIN
            and target_collateral <= MULTIVIEW_BLEED_MAX
            and spatial_bleed <= MULTIVIEW_BLEED_MAX
            and opposing_bleed <= MULTIVIEW_BLEED_MAX
            and roof_bleed <= ROOF_BLEED_MAX
        )
        reasons = []
        if target_presence < TRANSFER_MATCH_MIN:
            reasons.append(f"target_presence={target_presence:.3f}")
        if target_collateral > MULTIVIEW_BLEED_MAX:
            reasons.append(f"target_collateral={target_collateral:.5f}")
        if spatial_bleed > MULTIVIEW_BLEED_MAX:
            reasons.append(f"spatial_bleed={spatial_bleed:.5f}")
        if opposing_bleed > MULTIVIEW_BLEED_MAX:
            reasons.append(f"opposing_bleed={opposing_bleed:.5f}")
        if roof_bleed > ROOF_BLEED_MAX:
            reasons.append(f"roof_bleed={roof_bleed:.5f}")
        return {
            "multiview_checked": True,
            "multiview_ok": bool(ok),
            "target_presence": round(target_presence, 4),
            "target_collateral_bleed": round(target_collateral, 6),
            "spatial_bleed": round(spatial_bleed, 6),
            "opposing_bleed": round(opposing_bleed, 6),
            "roof_bleed": round(roof_bleed, 6),
            "multiview_note": "; ".join(reasons),
        }


def verify_transfer(new_texture: Image.Image, ref_rgba: Image.Image,
                    blob_mask, proj_report: dict, *,
                    geoms_zup=None, old_texture: Image.Image = None,
                    fit: dict = None, accepted_hits=None,
                    patch_footprint=None, shift_maps=None) -> dict:
    """Verify emblem transfer via write-back and optional 5-view 3D rendering."""
    try:
        blob_px = int(np.asarray(blob_mask).sum())
        got = int(proj_report.get("projected_texels", 0) or 0)
        ratio = (got / blob_px) if blob_px else 0.0
        writeback = proj_report.get("writeback_match")
        if writeback is not None:
            ok = got > 0 and float(writeback) >= TRANSFER_MATCH_MIN
        else:
            ok = ratio >= TRANSFER_MATCH_MIN

        arr = np.asarray(new_texture.convert("RGB")).astype(np.float32)
        mx, mn = arr.max(axis=-1), arr.min(axis=-1)
        chroma = (mx - mn) / np.maximum(mx, 1e-6)
        hue_px = int((((chroma > 0.5) & (mx > 80)) | (mx < 90)).sum())
        if not hue_px:
            ok = False

        mv = {"multiview_checked": False}
        if ok and geoms_zup is not None and old_texture is not None and fit is not None:
            mv = _verify_multiview_render(
                geoms_zup, old_texture, new_texture, ref_rgba,
                blob_mask, fit, proj_report, accepted_hits=accepted_hits,
                patch_footprint=patch_footprint, shift_maps=shift_maps)
            if not mv.get("multiview_ok"):
                ok = False

        note = ""
        if not ok:
            note = mv.get("multiview_note") or "transfer below threshold; pose or blend suspect"
        return {
            "ok": bool(ok),
            "match_ratio": round(float(ratio), 4),
            "writeback_match": writeback,
            "hue_pixels": hue_px,
            "blob_pixels": blob_px,
            "threshold": TRANSFER_MATCH_MIN,
            **mv,
            "note": note,
        }
    except Exception as e:
        return {"ok": False, "match_ratio": 0.0, "multiview_checked": False,
                "note": f"verify failed: {e}"}


# ------------------------------------------------------------ item entry point

def _cleanup_staged(staged_pairs):
    for tmp, _final in staged_pairs:
        Path(tmp).unlink(missing_ok=True)


def process_item(item_dir: Path, item_id: str, *, dry_run=False):
    """Run the emblem stage for one item directory. Returns a report dict."""
    png_path = item_dir / f"{item_id}.png"
    glb_path = item_dir / f"{item_id}.glb"
    meta_path = item_dir / "metadata.json"

    def _persist_unapplied(rep):
        if dry_run or not meta_path.is_file():
            return
        meta = json.loads(meta_path.read_text())
        if not (meta.get("emblem_project") or {}).get("applied"):
            meta["emblem_project"] = {
                "schema": EMBLEM_SCHEMA,
                **rep,
                "note": rep.get("note", "no emblem blobs"),
            }
            write_json(meta_path, meta)

    with Image.open(png_path) as img:
        ref = img.convert("RGBA")
    blob_mask, records = emblem_blobs(ref)
    report = {"blobs": len(records), "records": records, "applied": False}
    if not records:
        _persist_unapplied(report)
        return report
    geoms = load_geoms_zup(glb_path)
    try:
        fit = fit_reference_camera(
            geoms, Image.fromarray(np.asarray(ref)[:, :, 3]).convert("L"),
            ref_rgba=ref)
    except RuntimeError as e:
        report["note"] = f"camera fit unavailable: {e}"
        _persist_unapplied(report)
        return report
    report["fit_iou"] = fit["iou"]
    report["camera"] = {k: fit[k] for k in ("azimuth_deg", "elevation_deg")}
    report["pose"] = pose_record(fit)
    rejection = pose_rejection(fit)
    if rejection:
        report["note"] = rejection
        _persist_unapplied(report)
        return report

    texture, _ = base_color_texture(geoms)
    state = {}
    new_texture, proj = project_reference(
        geoms, texture, ref, fit, blob_mask=blob_mask, blob_records=records,
        out_state=state)
    report["projection"] = proj
    if dry_run or not proj.get("projected_texels"):
        report["note"] = proj.get("note", "dry run or nothing to project")
        _persist_unapplied(report)
        return report

    accepted_mask = state.get("accepted_blob_mask", blob_mask)
    accepted_hits = state.get("accepted_hits")
    patch_fp = state.get("patch_footprint")
    core_pm = state.get("core_patch_mask")
    shift_maps = state.get("shift_maps")
    report["verification"] = verify_transfer(
        new_texture, ref, accepted_mask, proj,
        geoms_zup=geoms, old_texture=texture, fit=fit,
        accepted_hits=accepted_hits, patch_footprint=patch_fp,
        shift_maps=shift_maps)
    if not report["verification"].get("ok"):
        report["note"] = report["verification"].get("note")
        _persist_unapplied(report)
        return report

    buf = io.BytesIO()
    new_texture.save(buf, format="PNG")
    tex_index = glb_basecolor_image_index(glb_path)
    new_glb = replace_glb_texture(glb_path, tex_index, buf.getvalue())

    staged_gs = []
    frame_map = (proj["frame_scale"], proj["frame_offset"][0], proj["frame_offset"][1])
    gauss = patch_gaussian_colors(
        item_dir, item_id, ref, fit,
        blob_mask=accepted_mask, frame_map=frame_map,
        accepted_hits=accepted_hits, patch_footprint=patch_fp,
        core_patch_mask=core_pm, shift_maps=shift_maps,
        stroke_mask=state.get("stroke_mask"),
        commit=False, staged_out=staged_gs)
    report["gaussians"] = gauss
    report["texture_index"] = tex_index

    tmp_glb = glb_path.with_name(f".{item_id}.tmp.emblem.glb")
    try:
        tmp_glb.write_bytes(new_glb)
        durable_replace(tmp_glb, glb_path)
        for gs_tmp, gs_final in staged_gs:
            durable_replace(gs_tmp, gs_final)
    except Exception:
        tmp_glb.unlink(missing_ok=True)
        _cleanup_staged(staged_gs)
        raise

    gs_hashes = {
        p.name: file_sha256(p)
        for p in (item_dir / f"{item_id}.{ext}" for ext in GS_EXTENSIONS)
        if p.is_file()
    }
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"id": item_id}
    meta.pop("mesh_artifacts", None)
    meta.pop("gs_artifacts", None)
    meta["emblem_project"] = {
        "schema": EMBLEM_SCHEMA,
        "applied": True,
        "blobs": len(records),
        "records": records,
        "fit_iou": fit["iou"],
        "camera": report["camera"],
        "pose": report["pose"],
        "projection": proj,
        "gaussians": gauss,
        "verification": report["verification"],
        "texture_index": tex_index,
        "glb_sha256": file_sha256(glb_path),
        "gs_sha256": gs_hashes,
    }
    try:
        from prompt_utils import STATE_DIR as _SD
        for _name in ("qa_meshes.json", "qa_gaussians.json"):
            _p = _SD / _name
            if _p.is_file():
                _v = json.loads(_p.read_text())
                if item_id in _v:
                    _v.pop(item_id, None)
                    write_json(_p, _v)
    except Exception:
        pass
    write_json(meta_path, meta)
    report["applied"] = True
    return report


def try_emblem_fix(item_dir: Path, item_id: str):
    """Best-effort pipeline hook: atomic, verified, no-op without blobs, never silently splits GLB/3DGS."""
    staged_gs = []
    tmp_glb = None
    glb_committed = False
    try:
        png_path = item_dir / f"{item_id}.png"
        glb_path = item_dir / f"{item_id}.glb"
        with Image.open(png_path) as img:
            ref = img.convert("RGBA")
        blob_mask, records = emblem_blobs(ref)
        if not records:
            return {"schema": EMBLEM_SCHEMA, "applied": False, "blobs": 0,
                    "records": [], "note": "no emblem blobs"}
        geoms = load_geoms_zup(glb_path)
        try:
            fit = fit_reference_camera(
                geoms, Image.fromarray(np.asarray(ref)[:, :, 3]).convert("L"),
                ref_rgba=ref)
        except RuntimeError as e:
            return {"schema": EMBLEM_SCHEMA, "applied": False, "blobs": len(records),
                    "records": records, "note": f"camera fit unavailable: {e}"}
        cam_rec = {k: fit[k] for k in ("azimuth_deg", "elevation_deg")}
        rejection = pose_rejection(fit)
        if rejection:
            return {"schema": EMBLEM_SCHEMA, "applied": False, "blobs": len(records),
                    "records": records, "fit_iou": fit["iou"], "camera": cam_rec,
                    "pose": pose_record(fit), "note": rejection}
        texture, _ = base_color_texture(geoms)
        state = {}
        new_texture, proj = project_reference(
            geoms, texture, ref, fit, blob_mask=blob_mask, blob_records=records,
            out_state=state)
        if not proj.get("projected_texels"):
            return {"schema": EMBLEM_SCHEMA, "applied": False, "blobs": len(records),
                    "records": records, "fit_iou": fit["iou"], "camera": cam_rec,
                    "pose": pose_record(fit), "projection": proj,
                    "note": proj.get("note", "nothing to project")}

        accepted_mask = state.get("accepted_blob_mask", blob_mask)
        accepted_hits = state.get("accepted_hits")
        patch_fp = state.get("patch_footprint")
        core_pm = state.get("core_patch_mask")
        shift_maps = state.get("shift_maps")
        verification = verify_transfer(
            new_texture, ref, accepted_mask, proj,
            geoms_zup=geoms, old_texture=texture, fit=fit,
            accepted_hits=accepted_hits, patch_footprint=patch_fp,
            shift_maps=shift_maps)
        if not verification.get("ok"):
            return {"schema": EMBLEM_SCHEMA, "applied": False, "blobs": len(records),
                    "records": records, "fit_iou": fit["iou"], "camera": cam_rec,
                    "pose": pose_record(fit), "projection": proj,
                    "verification": verification,
                    "note": verification.get("note")}

        buf = io.BytesIO()
        new_texture.save(buf, format="PNG")
        tex_index = glb_basecolor_image_index(glb_path)
        new_glb = replace_glb_texture(glb_path, tex_index, buf.getvalue())

        frame_map = (proj["frame_scale"], proj["frame_offset"][0], proj["frame_offset"][1])
        gauss = patch_gaussian_colors(
            item_dir, item_id, ref, fit,
            blob_mask=accepted_mask, frame_map=frame_map,
            accepted_hits=accepted_hits, patch_footprint=patch_fp,
            core_patch_mask=core_pm, shift_maps=shift_maps,
            stroke_mask=state.get("stroke_mask"),
            commit=False, staged_out=staged_gs)

        tmp_glb = glb_path.with_name(f".{item_id}.tmp.emblem.glb")
        tmp_glb.write_bytes(new_glb)
        durable_replace(tmp_glb, glb_path)
        glb_committed = True
        for gs_tmp, gs_final in staged_gs:
            durable_replace(gs_tmp, gs_final)
        staged_gs.clear()

        gs_hashes = {
            p.name: file_sha256(p)
            for p in (item_dir / f"{item_id}.{ext}" for ext in GS_EXTENSIONS)
            if p.is_file()
        }
        return {
            "schema": EMBLEM_SCHEMA, "applied": True, "blobs": len(records),
            "records": records, "fit_iou": fit["iou"],
            "camera": cam_rec,
            "pose": pose_record(fit),
            "projection": proj, "gaussians": gauss,
            "verification": verification, "texture_index": tex_index,
            "glb_sha256": file_sha256(glb_path),
            "gs_sha256": gs_hashes,
        }
    except Exception as e:
        if tmp_glb is not None:
            tmp_glb.unlink(missing_ok=True)
        _cleanup_staged(staged_gs)
        if glb_committed:
            # GLB was already mutated on disk; never swallow a subsequent GS
            # commit failure into applied=False metadata.
            raise RuntimeError(f"emblem commit failed after GLB replacement: {e}") from e
        return {"schema": EMBLEM_SCHEMA, "applied": False, "note": f"emblem hook skipped: {e}"}


def audit_item(item_dir: Path, item_id: str) -> dict:
    """Refit the pose of an applied record and compare it with the recorded one."""
    meta = json.loads((item_dir / "metadata.json").read_text())
    rec = meta.get("emblem_project") or {}
    if not rec.get("applied"):
        return {"verdict": "not_applied"}
    recorded = rec.get("camera") or {}
    with Image.open(item_dir / f"{item_id}.png") as img:
        ref = img.convert("RGBA")
    geoms = load_geoms_zup(item_dir / f"{item_id}.glb")
    fit = fit_reference_camera(
        geoms, Image.fromarray(np.asarray(ref)[:, :, 3]).convert("L"), ref_rgba=ref)
    out = {"schema": rec.get("schema"), "recorded": recorded, "refit": pose_record(fit)}
    if not isinstance(rec.get("schema"), int) or rec["schema"] < EMBLEM_SCHEMA:
        out["legacy_schema"] = True
    if "azimuth_deg" not in recorded:
        out["verdict"] = "unknown_recorded_pose"
        return out
    sep = pose_separation_deg(
        (math.radians(recorded["azimuth_deg"]), math.radians(recorded["elevation_deg"])),
        (math.radians(fit["azimuth_deg"]), math.radians(fit["elevation_deg"])))
    out["separation_deg"] = round(sep, 1)
    if sep >= DISTINCT_POSE_DEG:
        out["verdict"] = "wrong_pose"
    elif fit.get("ambiguous"):
        out["verdict"] = "ambiguous"
    elif out.get("legacy_schema"):
        out["verdict"] = "legacy_schema"
    else:
        out["verdict"] = "consistent"
    return out


def audit_main(manifests, output_dir, ids, category=None):
    """Print one audit line per applied record; exit 1 if any is suspect."""
    rows = []
    for item_id, it in sorted(manifests.items()):
        if ids and item_id not in ids:
            continue
        d = output_dir / it["geography"] / (category or it["category"]) / item_id
        if not (d / "metadata.json").is_file() or not (d / f"{item_id}.glb").is_file():
            continue
        try:
            rec = json.loads((d / "metadata.json").read_text()).get("emblem_project") or {}
        except Exception:
            continue
        if not rec.get("applied"):
            continue
        try:
            result = audit_item(d, item_id)
        except Exception as e:
            result = {"verdict": "error", "note": str(e)}
        rows.append((item_id, result))
        print(f"{item_id}: {json.dumps(result)}", flush=True)
    suspect = [i for i, r in rows
               if r["verdict"] in ("wrong_pose", "ambiguous", "legacy_schema", "error")]
    print(f"emblem audit: {len(rows)} applied records, {len(suspect)} suspect"
          + (f": {' '.join(suspect)}" if suspect else ""))
    return 1 if suspect else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--all", action="store_true",
                    help="backfill every item whose reference has emblem blobs "
                         "but whose metadata lacks a current emblem_project record")
    ap.add_argument("--geography", default="germany")
    ap.add_argument("--category", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="read-only: refit the pose of every applied record (or --ids) "
                         "and report wrong-pose, ambiguous or legacy-schema records")
    args = ap.parse_args(argv)
    from prompt_utils import MANIFESTS, OUTPUT_DIR, load_manifest
    manifests = {}
    for geo in MANIFESTS:
        for it in load_manifest(geo):
            row = dict(it)
            row.setdefault("geography", geo)
            manifests[row["id"]] = row
    targets = list(args.ids)
    if args.audit:
        return audit_main(manifests, OUTPUT_DIR, targets, args.category)
    if args.all:
        for item_id, it in sorted(manifests.items()):
            geo = it["geography"]
            cat = args.category or it["category"]
            d = OUTPUT_DIR / geo / cat / item_id
            if not (d / f"{item_id}.glb").is_file():
                continue
            try:
                meta = json.loads((d / "metadata.json").read_text())
            except Exception:
                meta = {}
            _rec = meta.get("emblem_project") if isinstance(meta, dict) else None
            if (isinstance(_rec, dict) and _rec.get("applied")
                    and isinstance(_rec.get("schema"), int) and _rec["schema"] >= 2):
                # Never double-paint over an already modified GLB; legacy-schema
                # items must be regenerated or restored before re-projection.
                try:
                    if _rec.get("glb_sha256") == file_sha256(d / f"{item_id}.glb"):
                        continue
                except Exception:
                    pass
            try:
                with Image.open(d / f"{item_id}.png") as _img:
                    _mask, _recs = emblem_blobs(_img.convert("RGBA"))
            except Exception:
                continue
            if _recs:
                targets.append(item_id)
        print(f"emblem backfill: {len(targets)} items with blobs and no record")
    if not targets:
        print("emblem backfill: nothing to do")
        return 0
    code = 0
    for item_id in targets:
        it = manifests.get(item_id)
        if it is None:
            print(f"{item_id}: unknown id")
            code = 1
            continue
        geo = it["geography"]
        cat = args.category or it["category"]
        d = OUTPUT_DIR / geo / cat / item_id
        try:
            report = process_item(d, item_id, dry_run=args.dry_run)
        except Exception as e:
            print(f"{item_id}: FAILED {e}")
            code = 1
            continue
        print(f"{item_id}: {json.dumps(report)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
