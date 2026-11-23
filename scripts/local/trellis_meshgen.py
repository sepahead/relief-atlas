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
# Do NOT force MTL_CAPTURE_ENABLED=1 here: chunked decoder/upsample kernels
# avoid watchdog timeouts without Metal's debugger capture layer, which retains
# tens of GiB of non-PyTorch command/resource allocations across multi-item runs.

from pathlib import Path  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
TRELLIS_MAC = PROJECT_DIR / "meshmaker" / "local" / "trellis-mac"
MODEL_DIR = PROJECT_DIR / "meshmaker" / "local" / "models" / "TRELLIS.2-4B"

try:
    import flex_gemm as _flex_gemm  # noqa: F401
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
except (ImportError, RuntimeError):
    _flex_gemm = None
    os.environ.setdefault("SPARSE_CONV_BACKEND", "none")

sys.path.insert(0, str(TRELLIS_MAC / "TRELLIS.2"))
sys.path.insert(0, str(TRELLIS_MAC))
sys.path.append(str(TRELLIS_MAC / "stubs"))
sys.path.insert(0, str(SCRIPT_DIR))

import argparse  # noqa: E402
import gc  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


class _NoRembg:
    def __init__(self, *a, **k):
        pass

    def to(self, device):
        return self

    def cpu(self):
        return self

    def __call__(self, image):
        raise RuntimeError("background removal requested but RMBG bypass is active")


# Neutralize the gated RMBG-2.0 background remover before pipeline load when
# TRELLIS.2 is importable; keep lightweight test imports working without it.
try:
    import trellis2.pipelines.rembg as _rembg_mod  # noqa: E402
    _rembg_mod.BiRefNet = _NoRembg
except ImportError:
    _rembg_mod = None

from gs_export import (  # noqa: E402
    export_gaussians_from_mesh_with_voxel, gltf_vertices, tmp_sibling,
)
import content_policy  # noqa: E402
from generation_prompts import is_optimized_prompt  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR, SELECTION_ORDERS, STATE_DIR, clean_name, clean_prompt, item_seed,
    mesh_complete, select_manifest_items,
)
from pipeline_utils import (  # noqa: E402
    GS_EXTENSIONS, GS_COLOR_ENCODINGS, file_sha256, generation_provenance,
    image_model_matches, source_image_matches, durable_replace,
    source_image_metadata, verdict_matches, write_json,
    is_infrastructure_error, gpu_lock,
)

MESH_MODEL_ID = "local/trellis.2-4b"
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")
TEXTURE_SIZES = (512, 1024, 2048, 4096)

_MPS_PATCHES_INSTALLED = False


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--geographies", nargs="*", default=list(MANIFESTS),
                    choices=list(MANIFESTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit item ids (overrides geographies/limit)")
    ap.add_argument("--order", choices=list(SELECTION_ORDERS), default="interleaved")
    ap.add_argument("--pipeline-type", default="1024_cascade",
                    choices=list(PIPELINE_TYPES))
    ap.add_argument("--texture-size", type=int, default=2048,
                    choices=list(TEXTURE_SIZES))
    ap.add_argument("--mesh-steps", type=int, default=None,
                    help="override sampling steps for structure, shape and texture")
    ap.add_argument("--target-faces", type=int, default=300000,
                    help="triangle target before texture baking")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the per-item deterministic seed")
    ap.add_argument("--gs-format", choices=["spz", "splat", "both"], default="spz",
                    help="Gaussian containers to retain; SPZ falls back to SPLAT if unavailable")
    content_policy.add_policy_args(ap)
    ap.add_argument("--ignore-image-qa", action="store_true",
                    help="mesh images that failed or lack an image QA verdict; "
                         "the resulting meshes are then uncertified")
    args = ap.parse_args(argv)
    if args.target_faces < 1 or (args.mesh_steps is not None and args.mesh_steps < 1):
        ap.error("mesh steps and target faces must be positive")
    return args


def save_json_atomic(path, data):
    write_json(path, data)


# ------------------------------------------------------------ reference cleaning

def _fill_holes_2d(mask):
    """Return bool mask with enclosed 2D interior holes filled."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    if not m.any() or m.all():
        return m.copy()
    try:
        from scipy import ndimage
        return ndimage.binary_fill_holes(m)
    except ImportError:
        from collections import deque
        h, w = m.shape
        ext = np.zeros((h, w), dtype=bool)
        q = deque()
        for x in range(w):
            if not m[0, x] and not ext[0, x]:
                ext[0, x] = True
                q.append((0, x))
            if not m[h - 1, x] and not ext[h - 1, x]:
                ext[h - 1, x] = True
                q.append((h - 1, x))
        for y in range(h):
            if not m[y, 0] and not ext[y, 0]:
                ext[y, 0] = True
                q.append((y, 0))
            if not m[y, w - 1] and not ext[y, w - 1]:
                ext[y, w - 1] = True
                q.append((y, w - 1))
        while q:
            cy, cx = q.popleft()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and not m[ny, nx] and not ext[ny, nx]:
                    ext[ny, nx] = True
                    q.append((ny, nx))
        return ~ext


def clean_reference_rgba(image):
    """In-memory RGBA cleanup before TRELLIS conditioning (never touches disk).

    1. Clamps residual Qwen VAE / matting background wisps (`alpha < 40` outside
       a 1px anti-aliasing ring around opaque core `alpha >= 242`) regardless of
       faint magenta/blue VAE tint.
    2. Uses topological hole-filling of `core` to protect enclosed semi-transparent
       interior features (cabin windows, visors, translucent reservoirs) while
       stripping exterior semi-transparent studio contact-shadow skirts and grey
       cast ground shadows along the lower silhouette boundary.
    3. Drops detached semi-transparent shadow islands that do not connect to any
       opaque core component.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    arr = np.array(image.convert("RGBA"), copy=True)
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    core = alpha >= 242
    core_aa = np.asarray(
        Image.fromarray((core * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3))
    ) > 0
    enclosed = _fill_holes_2d(core)

    noise = (~core_aa) & (alpha < 40)
    rgb_max = rgb.max(axis=-1)
    rgb_min = rgb.min(axis=-1)
    chroma = rgb_max.astype(np.int16) - rgb_min.astype(np.int16)

    exterior_fringe = (~enclosed) & (~core_aa) & (alpha >= 40) & (alpha < 242)
    dark_skirt = exterior_fringe & (rgb_max < 96) & (chroma < 36)

    ys_c, _xs_c = np.nonzero(core)
    if len(ys_c):
        y0_c, y1_c = int(ys_c.min()), int(ys_c.max())
        y_floor_band = y0_c + int(0.75 * max(y1_c - y0_c + 1, 1))
        yy = np.arange(alpha.shape[0], dtype=np.int32)[:, None]
        cast_shadow = (
            exterior_fringe
            & (yy >= y_floor_band)
            & (alpha < 200)
            & (rgb_max < 150)
            & (chroma < 32)
        )
    else:
        cast_shadow = np.zeros_like(core)

    alpha[noise | dark_skirt | cast_shadow] = 0

    if int(core.sum()) >= 256:
        try:
            from scipy import ndimage
            lbl, n_comp = ndimage.label(alpha > 0)
            if n_comp > 1:
                has_core = np.bincount(lbl.ravel(), weights=core.ravel().astype(np.float32)) > 0
                has_core[0] = False
                alpha[~has_core[lbl]] = 0
        except ImportError:
            pass

    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")


def clean_reference_for_trellis(image, *, margin_ratio: float = 0.06):
    """Clean RGBA alpha, crop with symmetric context margin, premultiply to RGB."""
    import numpy as np
    from PIL import Image

    rgba = clean_reference_rgba(image)
    max_size = max(rgba.size)
    if max_size > 1024:
        scale = 1024.0 / float(max_size)
        rgba = rgba.resize(
            (max(1, int(round(rgba.width * scale))),
             max(1, int(round(rgba.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    arr = np.asarray(rgba)
    alpha = arr[..., 3]
    ys, xs = np.nonzero(alpha > 64)
    if not len(ys):
        ys, xs = np.nonzero(alpha > 0)
    if len(ys):
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    else:
        x0, y0, x1, y1 = 0, 0, rgba.width - 1, rgba.height - 1

    span = max(x1 - x0 + 1, y1 - y0 + 1, 1)
    pad = max(2, int(round(span * float(margin_ratio)))) if margin_ratio > 0 else 0
    side = span + 2 * pad
    cx = 0.5 * (x0 + x1 + 1)
    cy = 0.5 * (y0 + y1 + 1)
    left = int(math.floor(cx - 0.5 * side))
    top = int(math.floor(cy - 0.5 * side))
    cropped = rgba.crop((left, top, left + side, top + side))

    crop_f = np.asarray(cropped, dtype=np.float32) / 255.0
    rgb_premult = np.clip(crop_f[..., :3] * crop_f[..., 3:4] * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb_premult, "RGB")


# ------------------------------------------------------------ MPS runtime patches

def _free_mps_memory():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _varlen_sdpa(q, k, v, q_seqlen, kv_seqlen, *, q_chunk: int | None = None):
    """Exact variable-length attention in O(1) Metal workspace or adaptive SDPA.

    Upstream `full_attn.py` zero-pads sequences into `(B, max_q, H, C)` and
    calls `F.scaled_dot_product_attention` without a mask, both corrupting
    multi-sequence softmax and materializing multi-tens-of-GB `N x N` matrices
    on MPS at `1024_cascade` / `1536_cascade` token counts (`N <= 49,152`).
    """
    import torch
    import torch.nn.functional as F

    q_lens = q_seqlen.tolist() if hasattr(q_seqlen, "tolist") else [int(x) for x in q_seqlen]
    kv_lens = (q_lens if kv_seqlen is q_seqlen else
               (kv_seqlen.tolist() if hasattr(kv_seqlen, "tolist") else [int(x) for x in kv_seqlen]))
    max_q = max(q_lens) if q_lens else 0
    max_kv = max(kv_lens) if kv_lens else 0
    if max_q == 0 or max_kv == 0:
        return torch.zeros((q.shape[0], q.shape[1], v.shape[-1]),
                           device=q.device, dtype=q.dtype)

    flex_fwd = getattr(getattr(getattr(_flex_gemm, "kernels", None), "cuda", None),
                       "sparse_attention_fwd", None)
    if (flex_fwd is not None
            and max_q <= 2048 and max_kv <= 2048
            and q.device.type in ("mps", "cuda")
            and q.shape[-1] == v.shape[-1]
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        cu_q = torch.tensor([0, *itertools.accumulate(q_lens)],
                            device=q.device, dtype=torch.int32)
        cu_kv = (cu_q if kv_lens is q_lens else
                 torch.tensor([0, *itertools.accumulate(kv_lens)],
                              device=q.device, dtype=torch.int32))
        return flex_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(),
            cu_q, cu_kv, max_q, max_kv, 1.0 / math.sqrt(float(q.shape[-1]))
        )

    num_heads = max(int(q.shape[1]), 1)
    out_parts = []
    q_off = 0
    kv_off = 0
    for ql, kvl in zip(q_lens, kv_lens):
        if ql == 0:
            kv_off += kvl
            continue
        if kvl == 0:
            out_parts.append(torch.zeros((ql, q.shape[1], v.shape[-1]),
                                         device=q.device, dtype=q.dtype))
            q_off += ql
            continue
        qb = q[q_off:q_off + ql].permute(1, 0, 2).unsqueeze(0)
        kb = k[kv_off:kv_off + kvl].permute(1, 0, 2).unsqueeze(0)
        vb = v[kv_off:kv_off + kvl].permute(1, 0, 2).unsqueeze(0)
        step = (int(q_chunk) if q_chunk is not None else
                max(1024, min(ql, 268_435_456 // max(num_heads * kvl, 1))))
        if ql <= step:
            ob = F.scaled_dot_product_attention(qb, kb, vb)
            out_parts.append(ob.squeeze(0).permute(1, 0, 2))
        else:
            chunks = []
            for s in range(0, ql, step):
                ob = F.scaled_dot_product_attention(qb[:, :, s:s + step], kb, vb)
                chunks.append(ob.squeeze(0).permute(1, 0, 2))
            out_parts.append(torch.cat(chunks, dim=0))
        q_off += ql
        kv_off += kvl
    return torch.cat(out_parts, dim=0)


def install_trellis_mps_patches(*, force: bool = False) -> bool:
    """Install Metal sparse-attention, cascade memory, and RGBA-clean patches."""
    global _MPS_PATCHES_INSTALLED
    if _MPS_PATCHES_INSTALLED and not force:
        return True

    try:
        import torch
        from PIL import Image
        import trellis2.modules.sparse.basic  # Load basic first to populate VarLenTensor in lazy sparse __init__
        import trellis2.modules.sparse as sp_pkg
        import trellis2.modules.sparse.attention as attn_pkg
        import trellis2.modules.sparse.attention.full_attn as full_mod
        import trellis2.modules.sparse.attention.windowed_attn as win_mod
        import trellis2.modules.sparse.attention.modules as attn_modules
        from trellis2.modules.sparse import SparseTensor, VarLenTensor, config as sp_cfg
        from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
        try:
            from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
        except (ImportError, ModuleNotFoundError, AttributeError):
            FlowEulerSampler = None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False

    orig_reduce = getattr(VarLenTensor, "reduce", None)
    if orig_reduce is not None and not getattr(VarLenTensor, "_relief_mps_reduce", False):
        def _fast_reduce(self, op: str, dim=None, keepdim: bool = False):
            if isinstance(dim, int):
                dim = (dim,)
            if op == "mean":
                red = self.feats.mean(dim=dim, keepdim=keepdim)
            elif op == "sum":
                red = self.feats.sum(dim=dim, keepdim=keepdim)
            elif op == "prod":
                red = self.feats.prod(dim=dim, keepdim=keepdim)
            else:
                raise ValueError(f"Unsupported reduce operation: {op}")
            if dim is None or 0 in dim:
                return red
            if len(self.layout) == 1:
                if op == "mean":
                    return red.mean(dim=0, keepdim=True)
                elif op == "sum":
                    return red.sum(dim=0, keepdim=True)
                elif op == "prod":
                    return red.prod(dim=0, keepdim=True)
            return torch.segment_reduce(red, reduce=op, lengths=self.seqlen)

        VarLenTensor.reduce = _fast_reduce
        VarLenTensor._relief_mps_reduce = True

    if FlowEulerSampler is not None and not getattr(FlowEulerSampler, "_relief_mps_euler", False):
        import numpy as _np
        from easydict import EasyDict as _edict
        from tqdm import tqdm as _tqdm

        def _pruned_euler_sample(
            self,
            model,
            noise,
            cond=None,
            steps: int = 50,
            rescale_t: float = 1.0,
            verbose: bool = True,
            tqdm_desc: str = "Sampling",
            **kwargs,
        ):
            sample = noise
            t_seq = _np.linspace(1, 0, steps + 1)
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
            t_seq = t_seq.tolist()
            t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
            ret = _edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
            for t, t_prev in _tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
                sample = out.pred_x_prev
            ret.samples = sample
            return ret

        FlowEulerSampler.sample = _pruned_euler_sample
        FlowEulerSampler._relief_mps_euler = True

    orig_full = full_mod.sparse_scaled_dot_product_attention
    orig_win_self = win_mod.sparse_windowed_scaled_dot_product_self_attention
    orig_win_cross = win_mod.sparse_windowed_scaled_dot_product_cross_attention

    def _patched_full_sdpa(*args, **kwargs):
        if sp_cfg.ATTN not in ("sdpa", "naive"):
            return orig_full(*args, **kwargs)
        num_all_args = len(args) + len(kwargs)
        if num_all_args == 1:
            qkv = args[0] if len(args) > 0 else kwargs["qkv"]
            s = qkv
            q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
            kv_seqlen = q_seqlen
            q, k, v = qkv.feats.unbind(dim=1)
            N = L = None
        elif num_all_args == 2:
            q_in = args[0] if len(args) > 0 else kwargs["q"]
            kv_in = args[1] if len(args) > 1 else kwargs["kv"]
            if isinstance(q_in, VarLenTensor):
                s = q_in
                q_seqlen = [q_in.layout[i].stop - q_in.layout[i].start for i in range(q_in.shape[0])]
                q = q_in.feats
                N = L = None
            else:
                s = None
                N, L, H, C = q_in.shape
                q_seqlen = [L] * N
                q = q_in.reshape(N * L, H, C)
            if isinstance(kv_in, VarLenTensor):
                kv_seqlen = [kv_in.layout[i].stop - kv_in.layout[i].start for i in range(kv_in.shape[0])]
                kv_f = kv_in.feats
            else:
                Nb, Lb, _, Hb, Cb = kv_in.shape
                kv_seqlen = [Lb] * Nb
                kv_f = kv_in.reshape(Nb * Lb, 2, Hb, Cb)
            k, v = kv_f.unbind(dim=1)
        elif num_all_args == 3:
            q_in = args[0] if len(args) > 0 else kwargs["q"]
            k_in = args[1] if len(args) > 1 else kwargs["k"]
            v_in = args[2] if len(args) > 2 else kwargs["v"]
            if isinstance(q_in, VarLenTensor):
                s = q_in
                q_seqlen = [q_in.layout[i].stop - q_in.layout[i].start for i in range(q_in.shape[0])]
                q = q_in.feats
                N = L = None
            else:
                s = None
                N, L, H, CI = q_in.shape
                q_seqlen = [L] * N
                q = q_in.reshape(N * L, H, CI)
            if isinstance(k_in, VarLenTensor):
                kv_seqlen = [k_in.layout[i].stop - k_in.layout[i].start for i in range(k_in.shape[0])]
                k = k_in.feats
                v = v_in.feats
            else:
                Nb, Lb, Hb, CI = k_in.shape
                CO = v_in.shape[-1]
                kv_seqlen = [Lb] * Nb
                k = k_in.reshape(Nb * Lb, Hb, CI)
                v = v_in.reshape(Nb * Lb, Hb, CO)
        else:
            return orig_full(*args, **kwargs)

        out = _varlen_sdpa(q, k, v, q_seqlen, kv_seqlen)
        return s.replace(out) if s is not None else out.reshape(N, L, out.shape[-2], -1)

    def _patched_win_self(qkv, window_size, shift_window=(0, 0, 0)):
        if sp_cfg.ATTN not in ("sdpa", "naive"):
            return orig_win_self(qkv, window_size, shift_window=shift_window)
        cache_name = f"windowed_attention_{window_size}_{shift_window}"
        cache = qkv.get_spatial_cache(cache_name)
        if cache is None:
            cache = win_mod.calc_window_partition(qkv, window_size, shift_window)
            qkv.register_spatial_cache(cache_name, cache)
        fwd_indices, bwd_indices, seq_lens, _ = cache
        q, k, v = qkv.feats[fwd_indices].unbind(dim=1)
        lens = seq_lens.tolist() if hasattr(seq_lens, "tolist") else list(seq_lens)
        out = _varlen_sdpa(q, k, v, lens, lens)
        return qkv.replace(out[bwd_indices])

    def _patched_win_cross(q, kv, q_window_size, kv_window_size,
                           q_shift_window=(0, 0, 0), kv_shift_window=(0, 0, 0)):
        if sp_cfg.ATTN not in ("sdpa", "naive"):
            return orig_win_cross(q, kv, q_window_size, kv_window_size,
                                  q_shift_window=q_shift_window,
                                  kv_shift_window=kv_shift_window)
        q_name = f"windowed_attention_{q_window_size}_{q_shift_window}"
        q_cache = q.get_spatial_cache(q_name)
        if q_cache is None:
            q_cache = win_mod.calc_window_partition(q, q_window_size, q_shift_window)
            q.register_spatial_cache(q_name, q_cache)
        q_fwd, q_bwd, q_lens_t, _ = q_cache

        kv_name = f"windowed_attention_{kv_window_size}_{kv_shift_window}"
        kv_cache = kv.get_spatial_cache(kv_name)
        if kv_cache is None:
            kv_cache = win_mod.calc_window_partition(kv, kv_window_size, kv_shift_window)
            kv.register_spatial_cache(kv_name, kv_cache)
        kv_fwd, _kv_bwd, kv_lens_t, _ = kv_cache

        qf = q.feats[q_fwd]
        kf, vf = kv.feats[kv_fwd].unbind(dim=1)
        q_lens = q_lens_t.tolist() if hasattr(q_lens_t, "tolist") else list(q_lens_t)
        kv_lens = kv_lens_t.tolist() if hasattr(kv_lens_t, "tolist") else list(kv_lens_t)
        out = _varlen_sdpa(qf, kf, vf, q_lens, kv_lens)
        return q.replace(out[q_bwd])

    for mod in (full_mod, attn_modules, attn_pkg, sp_pkg):
        if hasattr(mod, "sparse_scaled_dot_product_attention"):
            setattr(mod, "sparse_scaled_dot_product_attention", _patched_full_sdpa)
    for mod in (win_mod, attn_modules, attn_pkg, sp_pkg):
        if hasattr(mod, "sparse_windowed_scaled_dot_product_self_attention"):
            setattr(mod, "sparse_windowed_scaled_dot_product_self_attention", _patched_win_self)
        if hasattr(mod, "sparse_windowed_scaled_dot_product_cross_attention"):
            setattr(mod, "sparse_windowed_scaled_dot_product_cross_attention", _patched_win_cross)

    def _patched_preprocess_image(self, input_img):
        import numpy as np
        has_alpha = False
        if input_img.mode == "RGBA":
            alpha = np.asarray(input_img)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            rgba = input_img
        else:
            max_size = max(input_img.size)
            scale = min(1.0, 1024.0 / float(max_size))
            work = input_img
            if scale < 1.0:
                work = work.resize(
                    (int(work.width * scale), int(work.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            work = work.convert("RGB")
            if self.low_vram:
                self.rembg_model.to(self.device)
            rgba = self.rembg_model(work)
            if self.low_vram:
                self.rembg_model.cpu()
                _free_mps_memory()
        return clean_reference_for_trellis(rgba)

    def _patched_sample_shape_slat_cascade(
        self, lr_cond, cond, flow_model_lr, flow_model,
        lr_resolution, resolution, coords, sampler_params={}, max_num_tokens=49152,
    ):
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=self.device),
            coords=coords,
        )
        sp_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr, noise, **lr_cond, **sp_params,
            verbose=True, tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        del noise
        if self.low_vram:
            flow_model_lr.cpu()
            _free_mps_memory()
        std = torch.tensor(self.shape_slat_normalization["std"], device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization["mean"], device=slat.device)[None]
        slat = slat * std + mean

        if self.low_vram:
            self.models["shape_slat_decoder"].to(self.device)
            self.models["shape_slat_decoder"].low_vram = True
        hr_coords = self.models["shape_slat_decoder"].upsample(slat, upsample_times=4)
        del slat
        if self.low_vram:
            self.models["shape_slat_decoder"].cpu()
            self.models["shape_slat_decoder"].low_vram = False
        _free_mps_memory()

        token_budget = (
            min(max_num_tokens, 26624)
            if getattr(getattr(self, "device", None), "type", None) == "mps"
            else max_num_tokens
        )
        min_hr_res = 512
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            del quant_coords
            if coords.shape[0] <= token_budget or hr_resolution <= min_hr_res:
                if hr_resolution != resolution:
                    print(
                        f"Adaptive cascade resolution {resolution} -> {hr_resolution} "
                        f"({coords.shape[0]:,} tokens <= {token_budget:,} budget)."
                    )
                break
            hr_resolution -= 128
        del hr_coords
        _free_mps_memory()

        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords,
        )
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model, noise, **cond, **sp_params,
            verbose=True, tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        del noise, coords
        if self.low_vram:
            flow_model.cpu()
        _free_mps_memory()

        std = torch.tensor(self.shape_slat_normalization["std"], device=slat.device)[None]
        mean = torch.tensor(self.shape_slat_normalization["mean"], device=slat.device)[None]
        slat = slat * std + mean
        return slat, hr_resolution

    def _wrap_stage_cleanup(fn):
        def _wrapped(self, *a, **kw):
            out = fn(self, *a, **kw)
            if getattr(getattr(self, "device", None), "type", None) == "mps":
                _free_mps_memory()
            return out
        return _wrapped

    if not getattr(Trellis2ImageTo3DPipeline, "_relief_mps_patched", False) or force:
        if hasattr(Trellis2ImageTo3DPipeline, "preprocess_image"):
            Trellis2ImageTo3DPipeline.preprocess_image = _patched_preprocess_image
        if hasattr(Trellis2ImageTo3DPipeline, "sample_shape_slat_cascade"):
            Trellis2ImageTo3DPipeline.sample_shape_slat_cascade = _patched_sample_shape_slat_cascade
        for _stage_name in (
            "get_cond", "sample_sparse_structure", "sample_shape_slat",
            "decode_shape_slat", "sample_tex_slat", "decode_tex_slat", "run",
        ):
            if hasattr(Trellis2ImageTo3DPipeline, _stage_name):
                _orig_m = getattr(Trellis2ImageTo3DPipeline, _stage_name)
                setattr(Trellis2ImageTo3DPipeline, _stage_name, _wrap_stage_cleanup(_orig_m))
        Trellis2ImageTo3DPipeline._relief_mps_patched = True

    _MPS_PATCHES_INSTALLED = True
    return True


# ------------------------------------------------------------ GLB export

def _make_hole_safe_grid_sampler(orig_gs3d, has_flex: bool):
    """Wrap o_voxel `_grid_sample_3d` so hole-filled faces never bake as `[0,0,0]`.

    `o_voxel.postprocess.to_glb` runs `mesh.fill_holes()` before baking, creating
    watertight cap/underbody triangles that can sit > 1 voxel from any active
    `coords` entry. `_flex_grid_sample_3d` returns exact `[0, 0, 0, ...]` where
    all 8 trilinear neighbours are unoccupied. Appending an occupancy channel
    preserves active surface trilinear sampling identically while interpolating
    zero-occupancy queries from the nearest decoded surface voxels via cKDTree.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F_gs
    from scipy.spatial import cKDTree

    tree_cache = {}

    def _fill_unoccupied(out, occ, feats, coords, grid):
        B = out.shape[0]
        for b in range(B):
            empty = occ[b] <= 1e-5
            if not bool(empty.any()):
                continue
            if b not in tree_cache:
                sel = coords[:, 0] == b
                pts = coords[sel, 1:4].detach().cpu().numpy().astype(np.float32) + 0.5
                f_np = feats[sel].detach().cpu().numpy().astype(np.float32)
                tree_cache[b] = (cKDTree(pts) if len(pts) else None, f_np)
            tree, f_np = tree_cache[b]
            if tree is None or not len(f_np):
                continue
            q_pts = grid[b, empty].detach().cpu().numpy().astype(np.float32)
            k = min(8, len(f_np))
            dists, nn_idx = tree.query(q_pts, k=k)
            if k == 1:
                dists = dists[:, None]
                nn_idx = nn_idx[:, None]
            w = 1.0 / np.maximum(dists, 1e-3)
            w /= w.sum(axis=1, keepdims=True)
            filled = (f_np[nn_idx] * w[..., None]).sum(axis=1)
            out[b, empty] = torch.from_numpy(filled).to(device=out.device, dtype=out.dtype)
        return out

    def _gs3d_safe(feats, coords, shape, grid, mode="trilinear"):
        B, C = int(shape[0]), int(shape[1])
        ones = torch.ones((feats.shape[0], 1), dtype=feats.dtype, device=feats.device)
        feats_occ = torch.cat([feats, ones], dim=1)
        shape_occ = torch.Size([B, C + 1, *shape[2:]])

        if has_flex:
            sampled = orig_gs3d(feats_occ, coords, shape_occ, grid, mode=mode)
            out = sampled[..., :C].clone()
            occ = sampled[..., C]
            return _fill_unoccupied(out, occ, feats, coords, grid)

        D, H, W = int(shape[2]), int(shape[3]), int(shape[4])
        dense_vol = torch.zeros(B, C + 1, D, H, W, dtype=feats.dtype, device=feats.device)
        dense_vol[coords[:, 0].long(), :, coords[:, 1].long(),
                  coords[:, 2].long(), coords[:, 3].long()] = feats_occ
        # Voxel centres sit at `idx + 0.5`; subtract 0.5 for align_corners=True.
        g = grid - 0.5
        grid_norm = torch.stack([
            g[..., 2] / max(W - 1, 1) * 2 - 1,
            g[..., 1] / max(H - 1, 1) * 2 - 1,
            g[..., 0] / max(D - 1, 1) * 2 - 1,
        ], dim=-1).reshape(B, 1, 1, -1, 3)
        sampled = F_gs.grid_sample(dense_vol, grid_norm, mode="bilinear",
                                   align_corners=True, padding_mode="border")
        M = grid.shape[1]
        sampled = sampled.reshape(B, C + 1, M).permute(0, 2, 1)
        occ = sampled[..., C]
        nz = occ > 1e-5
        out = torch.zeros((B, M, C), dtype=feats.dtype, device=feats.device)
        if bool(nz.any()):
            out[nz] = sampled[..., :C][nz] / occ[nz].unsqueeze(-1)
        return _fill_unoccupied(out, occ, feats, coords, grid)

    return _gs3d_safe


def _export_glb_kdtree_fallback(verts, faces, mesh_out, glb_path, texture_size, target_faces):
    """CPU/xatlas fallback matching o_voxel sRGB/alpha/UV/material conventions."""
    import cv2
    import numpy as np
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree
    from backends.texture_baker import _rasterize_uv_triangles, uv_unwrap

    target_faces = min(target_faces, len(faces))
    bake_v, bake_f = verts, faces
    if len(faces) > target_faces:
        import fast_simplification
        ratio = 1.0 - (target_faces / len(faces))
        bake_v, bake_f = fast_simplification.simplify(verts, faces, ratio)

    new_verts, new_faces, uvs, _ = uv_unwrap(bake_v, bake_f)
    positions, mask = _rasterize_uv_triangles(new_verts, new_faces, uvs, texture_size)

    coords_np = mesh_out.coords.cpu().float().numpy()
    attrs_np = mesh_out.attrs.cpu().float().numpy()
    origin_np = (mesh_out.origin.cpu().float().numpy()
                 if hasattr(mesh_out.origin, "cpu")
                 else np.asarray(mesh_out.origin, dtype=np.float32))
    voxel_size = float(mesh_out.voxel_size)
    voxel_world = coords_np * voxel_size + origin_np + voxel_size * 0.5

    tree = cKDTree(voxel_world)
    q_pts = positions[mask]
    k = min(8, len(voxel_world))
    dists, indices = tree.query(q_pts, k=k, workers=-1)
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]
    weights = 1.0 / (dists + voxel_size * 0.1)
    far = dists > (voxel_size * 2.5)
    weights_masked = np.where(far, 0.0, weights)
    w_sum = weights_masked.sum(axis=1, keepdims=True)
    # Fall back to unclipped inverse-distance weights on hole-filled triangles.
    use_w = np.where(w_sum > 0, weights_masked, weights)
    use_w /= np.maximum(use_w.sum(axis=1, keepdims=True), 1e-10)
    sampled = (attrs_np[indices] * use_w[..., None]).sum(axis=1)

    layout = mesh_out.layout
    H = W = texture_size
    full = np.zeros((H, W, attrs_np.shape[1]), dtype=np.float32)
    full[mask] = sampled
    mask_inv = (~mask).astype(np.uint8)

    # `tex_slat_decoder` already outputs display sRGB in [0, 1]; never apply a
    # second 1/2.2 gamma curve.
    base_color = np.clip(full[..., layout["base_color"]] * 255.0, 0, 255).astype(np.uint8)
    metallic = np.clip(full[..., layout["metallic"]] * 255.0, 0, 255).astype(np.uint8)
    roughness = np.clip(full[..., layout["roughness"]] * 255.0, 0, 255).astype(np.uint8)
    alpha = (np.clip(full[..., layout["alpha"]] * 255.0, 0, 255).astype(np.uint8)
             if "alpha" in layout else np.full((H, W, 1), 255, dtype=np.uint8))

    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    # Match the Metal baker (`o_voxel.postprocess.to_glb`), which always emits
    # opaque PBR materials so solid opaque assets never hit alpha-blend sorting
    # artifacts in viewers.
    alpha_mode = "OPAQUE"
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    # `_rasterize_uv_triangles` writes V=0 to row 0 (top), whereas trimesh
    # interprets V=0 as the bottom of the image; flip V just like o_voxel.
    uvs_gltf = np.column_stack([uvs[:, 0], 1.0 - uvs[:, 1]])
    mesh = trimesh.Trimesh(
        vertices=gltf_vertices(new_verts),
        faces=new_faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_gltf, material=material),
    )
    mesh.export(str(glb_path), file_type="glb")
    return "kdtree"


def sanitize_voxel_albedo(mesh_out) -> int:
    """Lift unphysical `[0,0,0]` void base_color voxels prior to GLB & 3DGS export.

    TRELLIS.2 premultiplies RGBA references against `[0,0,0]` black, which can
    cause `tex_slat_decoder` to predict `base_color < 0.04` (`uint8 == [0,0,0]`)
    on occluded underbodies or dark tyre contact rims. Physical matte carbon
    rubber and chassis steel sit at sRGB `>= 0.06..0.20`; interpolating void
    voxels from nearby non-void voxels (clamped to realistic dark underbody
    reflectance across occluded gaps) eliminates black-hole voids in both GLB
    textures and SPZ/SPLAT Gaussian clouds while preserving dark tyres.
    """
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    layout = getattr(mesh_out, "layout", None)
    attrs = getattr(mesh_out, "attrs", None)
    coords = getattr(mesh_out, "coords", None)
    if layout is None or "base_color" not in layout or attrs is None or len(attrs) == 0:
        return 0

    bc_sl = layout["base_color"]
    bc = attrs[:, bc_sl].detach().cpu().float().numpy()
    bc_max = bc.max(axis=1)
    void_mask = bc_max < 0.045
    soft_dark = (bc_max >= 0.045) & (bc_max < 0.065)
    if not bool(void_mask.any()) and not bool(soft_dark.any()):
        return 0

    bc_fixed = bc.copy()
    if bool(void_mask.any()):
        good_mask = bc_max >= 0.065
        if bool(good_mask.any()) and coords is not None and len(coords) == len(bc):
            c_np = coords.detach().cpu().float().numpy()
            pts = c_np[:, 1:4] if c_np.shape[1] >= 4 else c_np[:, :3]
            tree = cKDTree(pts[good_mask])
            k = min(12, int(good_mask.sum()))
            dists, nn_idx = tree.query(pts[void_mask], k=k)
            if k == 1:
                dists = dists[:, None]
                nn_idx = nn_idx[:, None]
            w = 1.0 / np.maximum(dists, 0.25)
            w /= w.sum(axis=1, keepdims=True)
            interp = (bc[good_mask][nn_idx] * w[..., None]).sum(axis=1)
            near_t = np.clip((4.0 - dists[:, 0]) / 2.5, 0.0, 1.0)[:, None]
            cap_hi = 0.20 + 0.14 * near_t
            blended = interp * near_t + np.minimum(interp, 0.20) * (1.0 - near_t)
            floor_grey = np.full_like(interp, 0.11) * (1.0 - 0.3 * near_t)
            bc_fixed[void_mask] = np.clip(np.maximum(blended, floor_grey), 0.075, cap_hi)
        else:
            bc_fixed[void_mask] = 0.09

    if bool(soft_dark.any()):
        target_mx = 0.060 + 0.25 * (bc_max[soft_dark] - 0.045)
        gain = target_mx / np.maximum(bc_max[soft_dark], 1e-6)
        bc_fixed[soft_dark] = np.clip(bc_fixed[soft_dark] * gain[:, None], 0.0, 1.0)

    new_attrs = attrs.clone()
    new_attrs[:, bc_sl] = torch.from_numpy(bc_fixed).to(device=attrs.device, dtype=attrs.dtype)
    mesh_out.attrs = new_attrs
    return int(void_mask.sum())


def _adjust_remesh_dc_kwargs(kw: dict, max_res: int = 384) -> dict:
    """Cap narrow-band DC resolution while expanding grid scale to prevent AABB clipping.

    o_voxel.postprocess.to_glb computes scale = (orig_res + 3*band) / orig_res * base_scale
    before calling _remesh_narrow_band_dc. Capping resolution from 1024 to 384 without
    updating scale leaves a grid half-extent of +/-0.501465 while a 384^3 voxel has
    eps = 0.002612, slicing open any surface touching +/-0.500 (vehicle roofs, wheel
    bottoms, trailer sidewalls). Recomputing scale with 4*band margin at max_res keeps
    the entire [-0.5, 0.5]^3 volume strictly inside the active narrow band.
    """
    out = dict(kw)
    if "resolution" in out and int(out["resolution"]) > max_res:
        orig_res = float(out["resolution"])
        band = float(out.get("band", 1.0))
        out["resolution"] = int(max_res)
        if "scale" in out:
            orig_scale = float(out["scale"])
            base_scale = orig_scale * orig_res / (orig_res + 3.0 * band)
            out["scale"] = (float(max_res) + 4.0 * band) / float(max_res) * base_scale
    return out


def export_glb(mesh_out, glb_path, texture_size, target_faces=300000):
    """Bake PBR textures and export watertight GLB. Mirrors trellis-mac/generate.py."""
    import warnings

    use_metal = False
    pp_mod = None
    orig_gs3d = None
    orig_dc = None
    orig_uv = None
    mesh_backend_cls = None
    try:
        import o_voxel.postprocess as pp_mod
        backend = getattr(pp_mod, "_BACKEND", None)
        use_metal = backend == "metal" and getattr(pp_mod, "_HAS_DR", False)
        if use_metal:
            orig_gs3d = pp_mod._grid_sample_3d
            has_flex = bool(getattr(pp_mod, "_HAS_FLEX_GEMM", False))
            pp_mod._grid_sample_3d = _make_hole_safe_grid_sampler(orig_gs3d, has_flex)
            orig_dc = getattr(pp_mod, "_remesh_narrow_band_dc", None)
            if orig_dc is not None:
                def _capped_narrow_band_dc(*a, **kw):
                    kw = _adjust_remesh_dc_kwargs(kw, max_res=384)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*torch.cross.*")
                        return orig_dc(*a, **kw)
                pp_mod._remesh_narrow_band_dc = _capped_narrow_band_dc
            mesh_backend_cls = getattr(pp_mod, "_MeshBackend", None)
            if mesh_backend_cls is not None and hasattr(mesh_backend_cls, "uv_unwrap"):
                orig_uv = mesh_backend_cls.uv_unwrap

                def _clean_then_uv_unwrap(self, *a, **kw):
                    self.remove_duplicate_faces()
                    if hasattr(self, "remove_degenerate_faces"):
                        self.remove_degenerate_faces()
                    self.repair_non_manifold_edges()
                    self.remove_small_connected_components(2e-4)
                    self.fill_holes(max_hole_perimeter=0.50)
                    self.unify_face_orientations()
                    return orig_uv(self, *a, **kw)

                mesh_backend_cls.uv_unwrap = _clean_then_uv_unwrap
    except (ImportError, AttributeError):
        use_metal = False

    verts = mesh_out.vertices.cpu().numpy()
    faces = mesh_out.faces.cpu().numpy()

    if use_metal:
        try:
            import o_voxel

            target_faces = min(target_faces, len(faces))
            # to_glb builds a BVH of the full-resolution decoded mesh before
            # narrow-band Dual Contouring remeshing (`remesh=True`), sealing
            # <2-voxel thin-shell pinholes so QEM simplification cannot tear
            # hollow tent roofs/gables while projecting baked texels back to the
            # original 1024 surface BVH.
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh_out.vertices.cpu().float(), faces=mesh_out.faces.cpu(),
                attr_volume=mesh_out.attrs.cpu(),
                coords=mesh_out.coords.cpu(),
                attr_layout=mesh_out.layout,
                voxel_size=mesh_out.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target_faces,
                texture_size=texture_size,
                remesh=(orig_dc is not None),
                remesh_band=1.0,
                remesh_project=0.5,
            )
            # file_type is explicit: glb_path is a transient sibling name and
            # must never rely on extension sniffing.
            glb.export(str(glb_path), file_type="glb")
            return "metal"
        except RuntimeError as e:
            if is_infrastructure_error(e):
                raise
            print(f"    metal bake failed ({e}); falling back to KDTree baker")
        finally:
            if pp_mod is not None and orig_gs3d is not None:
                pp_mod._grid_sample_3d = orig_gs3d
            if pp_mod is not None and orig_dc is not None:
                pp_mod._remesh_narrow_band_dc = orig_dc
            if mesh_backend_cls is not None and orig_uv is not None:
                mesh_backend_cls.uv_unwrap = orig_uv
            _free_mps_memory()

    return _export_glb_kdtree_fallback(
        verts, faces, mesh_out, glb_path, texture_size, target_faces)


MAX_ITEMS_PER_SUBWORKER = 2


def _build_subworker_args(args, chunk_ids: list[str]) -> list[str]:
    """Build deterministic CLI arguments for a bounded worker subprocess."""
    cmd = [
        "--pipeline-type", str(args.pipeline_type),
        "--texture-size", str(args.texture_size),
        "--target-faces", str(args.target_faces),
        "--gs-format", str(args.gs_format),
    ]
    if args.mesh_steps is not None:
        cmd += ["--mesh-steps", str(args.mesh_steps)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if getattr(args, "allow_restricted", False):
        cmd.append("--allow-restricted")
    if getattr(args, "allow_blocked", False):
        cmd.append("--allow-blocked")
    if getattr(args, "only_policy", None):
        cmd += ["--only-policy", str(args.only_policy)]
    if getattr(args, "ignore_image_qa", False):
        cmd.append("--ignore-image-qa")
    cmd += ["--ids", *chunk_ids]
    return cmd


def _collect_todo(args, *, quiet: bool = False):
    items = select_manifest_items(args.geographies, args.ids, args.limit, args.only_policy,
                                  order=getattr(args, "order", "interleaved"))
    for it in items:
        it["prompt_clean"] = clean_prompt(it["prompt"], it["id"], transparent=True)

    def load_state(name):
        path = STATE_DIR / name
        return json.loads(path.read_text()) if path.exists() else {}

    image_qa = load_state("qa_images.json")
    quarantine = load_state("qa_quarantine.json")

    if quiet:
        items, _rejected, _overrides = content_policy.screen_items(
            content_policy.select_tier(items, getattr(args, "only_policy", None)),
            getattr(args, "allow_restricted", False),
            getattr(args, "allow_blocked", False),
        ), [], []
        items = items[0]
    else:
        items, rejected, overrides = content_policy.screen_from_args(
            items, args, phase="mesh_gen")
        print(content_policy.describe_screen(items, rejected, overrides))

    todo = []
    n_unverified = n_quarantined = 0
    for it in items:
        out_dir = OUTPUT_DIR / it["geography"] / it["category"] / it["id"]
        png = out_dir / f"{it['id']}.png"
        if not png.exists():
            continue
        if it["id"] in quarantine:
            n_quarantined += 1
            continue
        if not args.ignore_image_qa and (not image_model_matches(out_dir, it["id"])
                or not verdict_matches(image_qa.get(it["id"]), out_dir, it["id"], "images")):
            n_unverified += 1
            continue
        if not (mesh_complete(out_dir, it["id"]) and source_image_matches(out_dir, it["id"])):
            todo.append(it)
    if not quiet:
        print(f"{len(items)} items, {len(todo)} need meshes"
              + (f", {n_unverified} blocked by image QA" if n_unverified else "")
              + (f", {n_quarantined} quarantined" if n_quarantined else ""))
        if n_unverified and not args.ignore_image_qa:
            print("  run qa.py --mode images to certify them, or pass "
                  "--ignore-image-qa to bypass (uncertified)")
    return todo, image_qa


def main():
    args = parse_args()
    retries = {}
    retries_path = STATE_DIR / "qa_retries.json"
    if retries_path.exists():
        retries = json.loads(retries_path.read_text())

    todo, image_qa = _collect_todo(args, quiet=False)
    if not todo:
        return

    import torch
    from PIL import Image as PILImage
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    install_trellis_mps_patches()

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

            sanitize_voxel_albedo(mesh_out)
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
                mesh_out, str(out_dir / item_id), gs_format=args.gs_format,
                glb_path=glb_tmp)
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
            raw_verts = int(len(verts))
            raw_faces = int(len(mesh_out.faces))
            # Release heavy TRELLIS.2 voxel/mesh tensors and Metal cache before
            # loading the GLB + 3DGS + moderngl emblem pass so peak unified RAM
            # does not double-stack across the two stages.
            del mesh_out, outputs, verts
            _free_mps_memory()
            # Systemic emblem transfer: reference-anchored decal projection for
            # items whose PNG carries insignia blobs (red cross, roundels,
            # lettering). No-op without blobs; best-effort so it never fails
            # a freshly decoded item (except a post-commit GLB/3DGS split).
            emblem_report = None
            try:
                import emblem_project as _emblem
                emblem_report = _emblem.try_emblem_fix(out_dir, item_id)
                if emblem_report.get("applied"):
                    print(f"    emblem projection: "
                          f"{emblem_report['projection']['projected_texels']:,} texels, "
                          f"{emblem_report['gaussians'].get('patched_gaussians', 0):,} gaussians")
            except RuntimeError as _e:
                if "after GLB replacement" in str(_e):
                    raise
                emblem_report = {"applied": False, "note": f"emblem hook error: {_e}"}
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
                "vertices": raw_verts,
                "triangles": raw_faces,
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
            print(f"[{done}/{len(todo)}] {item_id}: {raw_verts:,} verts, "
                  f"{n_gaussians:,} gaussians, {t_total:.0f}s ({bake_backend})")
            _free_mps_memory()
        except Exception as e:
            failures[item_id] = {"error": str(e), "at": datetime.now(timezone.utc).isoformat()}
            print(f"[{done}/{len(todo)}] {item_id} FAILED: {e}")
            # Drop half-written transients so a 10K run cannot accumulate them.
            for stale in out_dir.glob(f".{item_id}.tmp.*"):
                stale.unlink(missing_ok=True)
            _free_mps_memory()
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


def run_cli():
    args = parse_args()
    if os.environ.get("RELIEF_MESH_SUBWORKER") != "1":
        todo, _ = _collect_todo(args, quiet=True)
        if len(todo) > MAX_ITEMS_PER_SUBWORKER:
            import subprocess
            print(
                f"Isolating {len(todo)} mesh items into {MAX_ITEMS_PER_SUBWORKER}-item "
                f"worker subprocesses to reclaim native Metal heaps between chunks."
            )
            env = {**os.environ, "RELIEF_MESH_SUBWORKER": "1"}
            for off in range(0, len(todo), MAX_ITEMS_PER_SUBWORKER):
                chunk_ids = [it["id"] for it in todo[off:off + MAX_ITEMS_PER_SUBWORKER]]
                sub_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *_build_subworker_args(args, chunk_ids),
                ]
                proc = subprocess.run(sub_cmd, env=env)
                if proc.returncode != 0:
                    raise SystemExit(proc.returncode)
            return
    with gpu_lock():
        main()


if __name__ == "__main__":
    run_cli()
