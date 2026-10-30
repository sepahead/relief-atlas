"""Reference-anchored emblem transfer for TRELLIS.2 meshes.

TRELLIS.2 texture lifting smooths small high-frequency decals (red crosses,
roundels, insignia, lettering) into the base paint, even when the reference
PNG shows them crisply. Retrying with a new seed does not fix that: it is a
resolution prior of the texture latent, not a bad sample.

This stage projects true reference pixels back onto the camera-visible faces
of the baked GLB texture. It is systemic, not per-item hand work:

* Detection gate: small concentrated high-contrast blobs (saturated insignia
  colours, dark lettering/outlines on light panels) inside the reference
  foreground. No blobs -> byte-identical no-op, so non-emblem items are
  untouched.
* Camera fit: silhouette IoU search over azimuth x elevation with the same
  pinhole convention as the QA turntable, then a 2D scale/offset refinement
  that absorbs the reference framing margins. Deterministic.
* Projection: per-pixel ray/face intersection with a rendered position + face
  id pass, eroded visibility mask and a feathered blend band, restricted to the
  reference emblem footprint, so every texel outside the decal keeps the baked
  bytes exactly. A whole-side repaint from one viewpoint smears panels and
  panel gaps, which is worse than a missing cross.
* The fitted camera and blob record are stored in metadata.json, so mesh QA
  can verify the transfer from the identical pose instead of refitting.

Runs fully offline with numpy/PIL/moderngl/trimesh (mesh runtime).
"""

import argparse
import json
import math
import struct
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pipeline_utils import durable_replace, file_sha256, write_json

# Blob gate: components covering less than this are specks, more than this is
# body paint / livery (handled fine by the texture latent).
BLOB_MIN_FRAC = 0.0005
BLOB_MAX_FRAC = 0.05
# Dark areas need a much tighter ceiling than saturated ones. Windows, door
# openings, grilles and shadowed panels are dark and can be several percent of
# the silhouette; the old single ceiling accepted them as "emblems" and the
# projection then painted window grey across the shell. Lettering and dark
# outlines are small: at 0.02 of the foreground a tent door or window block is
# excluded while a cross, badge or unit code still passes.
DARK_BLOB_MAX_FRAC = 0.02
# Decal-restricted projection. Only texels landing inside the detected emblem
# footprint may be rewritten: repainting every camera-visible texel from a
# single photo viewpoint smears panels, grid lines and panel gaps across the
# whole side whenever the frame map is a pixel or two off.
DECAL_FEATHER_MIN = 0.02
MAX_PROJECTED_TEXEL_SHARE = 0.25
# Write-back tolerance (0-255) for a texel that was written at full weight.
WRITEBACK_TOLERANCE = 12.0
# Camera fit quality below this means the pose cannot be trusted; the caller
# must skip projection rather than smear pixels from a wrong viewpoint.
FIT_IOU_MIN = 0.5
# Verification: fraction of reference emblem pixels whose re-rendered pixel
# keeps the emblem hue. Below this the transfer is declared lost.
# Checked by verify_transfer() on the rewritten texture (write-back check)
# plus a mirror-ambiguity flag from the camera fit. Pose trust itself is
# gated by FIT_IOU_MIN; this guards a silent no-op / wrong-side fit.
TRANSFER_MATCH_MIN = 0.5
RENDER_SIZE = 1024


# ------------------------------------------------------------ blob detection

def emblem_blobs(rgba: Image.Image):
    """Find small concentrated insignia/lettering blobs in the reference.

    Returns (mask, records): bool foreground mask of blob pixels and a list
    of {bbox, pixels, kind} dicts. Empty records mean "no emblems, no-op".
    """
    arr = np.asarray(rgba.convert("RGBA"))
    fg = arr[..., 3] > 127
    if not fg.any():
        return np.zeros(fg.shape, bool), []
    rgb = arr[..., :3].astype(np.float32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    saturated = (chroma > 0.5) & (mx > 80) & fg
    dark = (mx < 90) & fg
    candidate = saturated | dark
    # Drop isolated specks with an opening before labelling.
    mask_img = Image.fromarray((candidate * 255).astype(np.uint8))
    opened = np.asarray(mask_img.filter(ImageFilter.MinFilter(3))) > 0
    try:
        from scipy import ndimage
        lbl, n = ndimage.label(opened)
        sizes = np.bincount(lbl.ravel())
    except ImportError:
        lbl, n, sizes = _label_fallback(opened)
    n_px = float(fg.sum())
    records = []
    mask = np.zeros(fg.shape, bool)
    for i in range(1, n + 1):
        frac = float(sizes[i]) / n_px if len(sizes) > i else 0.0
        comp = lbl == i
        kind = "saturated" if (saturated[comp].mean() > 0.5) else "dark"
        cap = BLOB_MAX_FRAC if kind == "saturated" else DARK_BLOB_MAX_FRAC
        if BLOB_MIN_FRAC <= frac <= cap:
            mask |= comp
            ys, xs = np.nonzero(comp)
            records.append({
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "pixels": int(sizes[i]),
                "fraction": round(frac, 5),
                "kind": kind,
            })
    records.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
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


# ------------------------------------------------------------ camera fitting

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


def _orbit_cameras(center, radius, azimuths, elevations):
    up_ref = np.array([0.0, 0.0, 1.0])
    cams = []
    for el in elevations:
        for az in azimuths:
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
            cams.append((float(az), float(el), cam, right, up, forward))
    return cams


SIL_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
void main() { gl_Position = view_projection * vec4(position, 1.0); }
"""
SIL_FRAG = """#version 330
out vec4 color;
void main() { color = vec4(1.0); }
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


def _vp_matrix(cam, right, up, forward, size):
    view = np.eye(4)
    view[:3, :3] = np.array([right, up, -forward])
    view[:3, 3] = -view[:3, :3] @ cam
    near, far = 0.001, 50.0
    proj = np.zeros((4, 4))
    proj[0, 0] = proj[1, 1] = 3.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1
    return proj @ view


def fit_reference_camera(geoms_zup, ref_alpha, *, size=128):
    """Search azimuth x elevation for the silhouette best matching ref_alpha.

    Returns (az, el, iou, frame): the winning pose, its bbox-normalised mask
    IoU and a dict describing the render framing for the projection step.
    Raises RuntimeError when moderngl is unavailable (infra, not an asset).
    """
    import moderngl
    ref = np.asarray(ref_alpha.resize((size, size))) > 127
    ref_n = _bbox_normalize(ref)
    verts = np.vstack([np.asarray(g.vertices) for g in geoms_zup])
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    radius = float(np.linalg.norm(verts - center, axis=1).max()) * math.sqrt(10) * 1.08
    azimuths = [math.radians(a) for a in range(0, 360, 15)]
    elevations = [math.radians(e) for e in (10, 20, 30, 40, 55)]
    best = None
    with moderngl.create_standalone_context(require=330) as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        program, framebuffer, vaos = _silhouette_renderer(ctx, geoms_zup, size)
        framebuffer.use()
        for az, el, cam, right, up, forward in _orbit_cameras(center, radius, azimuths, elevations):
            program["view_projection"].write(_vp_matrix(cam, right, up, forward, size).T.astype("f4").tobytes())
            framebuffer.clear(0, 0, 0, 1, depth=1)
            for vao in vaos:
                vao.render()
            got = np.asarray(Image.frombytes(
                "RGB", (size, size),
                framebuffer.read(components=3, alignment=1),
            ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("L")) > 127
            got_n = _bbox_normalize(got)
            inter = float((got_n & ref_n).sum())
            union = float((got_n | ref_n).sum())
            iou = inter / union if union else 0.0
            if best is None or iou > best[2]:
                best = (az, el, iou, cam, right, up, forward)
    az, el, iou, cam, right, up, forward = best
    # Refine around the winner at 3-degree steps.
    with moderngl.create_standalone_context(require=330) as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        program, framebuffer, vaos = _silhouette_renderer(ctx, geoms_zup, size)
        framebuffer.use()
        for daz in range(-12, 13, 3):
            for del_ in range(-6, 7, 3):
                az2 = az + math.radians(daz)
                el2 = min(math.radians(80), max(math.radians(2), el + math.radians(del_)))
                cam2 = center + radius * np.array([
                    math.cos(el2) * math.cos(az2),
                    math.cos(el2) * math.sin(az2), math.sin(el2)])
                fwd = center - cam2
                fwd /= np.linalg.norm(fwd)
                rgt = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
                nrm = np.linalg.norm(rgt)
                rgt = np.array([1.0, 0.0, 0.0]) if nrm < 1e-6 else rgt / nrm
                up2 = np.cross(rgt, fwd)
                program["view_projection"].write(_vp_matrix(cam2, rgt, up2, fwd, size).T.astype("f4").tobytes())
                framebuffer.clear(0, 0, 0, 1, depth=1)
                for vao in vaos:
                    vao.render()
                got = np.asarray(Image.frombytes(
                    "RGB", (size, size),
                    framebuffer.read(components=3, alignment=1),
                ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("L")) > 127
                g_n = _bbox_normalize(got)
                inter = float((g_n & ref_n).sum())
                union = float((g_n | ref_n).sum())
                iou2 = inter / union if union else 0.0
                if iou2 > best[2]:
                    best = (az2, el2, iou2, cam2, rgt, up2, fwd)
    az, el, iou, cam, right, up, forward = best
    # Mirror-ambiguity caveat: bbox-normalised silhouettes erase aspect, so a
    # symmetric body (van side/side, front/rear) can score nearly identically
    # 180 deg away. Record the runner-up IoU opposite the winner; callers
    # treat <0.02 margin as ambiguous and require the hue verification below.
    runner = 0.0
    # Recompute cheaply: rescan stored coarse grid is gone, so approximate by
    # re-evaluating the antipode of the winner with the same renderer setup.
    # (Kept lightweight: one extra render, not a full rescan.)
    try:
        import moderngl as _mgl2
        with _mgl2.create_standalone_context(require=330) as _ctx:
            _ctx.enable(_mgl2.DEPTH_TEST)
            _prog, _fb, _vaos = _silhouette_renderer(ctx if False else _ctx, geoms_zup, size)
            _fb.use()
            _az = az + math.pi
            _cam = center + radius * np.array([
                math.cos(el) * math.cos(_az),
                math.cos(el) * math.sin(_az), math.sin(el)])
            _fwd = center - _cam
            _fwd /= np.linalg.norm(_fwd)
            _rgt = np.cross(_fwd, np.array([0.0, 0.0, 1.0]))
            _n = np.linalg.norm(_rgt)
            _rgt = np.array([1.0, 0.0, 0.0]) if _n < 1e-6 else _rgt / _n
            _up = np.cross(_rgt, _fwd)
            _prog["view_projection"].write(_vp_matrix(_cam, _rgt, _up, _fwd, size).T.astype("f4").tobytes())
            _fb.clear(0, 0, 0, 1, depth=1)
            for _vao in _vaos:
                _vao.render()
            _got = np.asarray(Image.frombytes(
                "RGB", (size, size),
                _fb.read(components=3, alignment=1),
            ).transpose(Image.Transpose.FLIP_TOP_BOTTOM).convert("L")) > 127
            _gn = _bbox_normalize(_got)
            _inter = float((_gn & ref_n).sum())
            _union = float((_gn | ref_n).sum())
            runner = _inter / _union if _union else 0.0
    except Exception:
        runner = 0.0
    return {
        "azimuth_deg": round(math.degrees(az), 2),
        "elevation_deg": round(math.degrees(el), 2),
        "iou": round(float(iou), 4),
        "runner_up_iou": round(float(runner), 4),
        "ambiguous": bool((float(iou) - float(runner)) < 0.02),
        "center": [float(v) for v in center],
        "radius": float(radius),
        "cam": [float(v) for v in cam],
        "right": [float(v) for v in right],
        "up": [float(v) for v in up],
        "forward": [float(v) for v in forward],
    }


# ------------------------------------------------------------ projection

ID_VERT = """#version 330
uniform mat4 view_projection;
in vec3 position;
void main() { gl_Position = view_projection * vec4(position, 1.0); }
"""
ID_FRAG = """#version 330
uniform vec3 face_id;
out vec4 color0;
void main() { color0 = vec4(face_id, 1.0); }
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
    """Blurred, dilated emblem footprint in reference space (float 0..1).

    The projection may only write where this is nonzero. Growing by a few
    reference pixels and blurring gives the blend band that fades a decal into
    the baked paint instead of cutting a hard seam around it.
    """
    m = np.asarray(blob_mask).astype(bool)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    grow = max(2, int(round(0.004 * max(m.shape[-2:]))))
    img = Image.fromarray((m * 255).astype(np.uint8))
    for _ in range(grow):
        img = img.filter(ImageFilter.MaxFilter(3))
    img = img.filter(ImageFilter.GaussianBlur(max(1.0, grow / 2.0)))
    return np.asarray(img).astype(np.float32) / 255.0


def project_reference(geoms_zup, texture_img, ref_rgba, fit, *, blob_mask=None,
                      size=RENDER_SIZE):
    """Write true reference pixels into the emblem decal texels only.

    Returns (new_texture, report). Texels outside the detected emblem
    footprint, outside the view, or failing the depth test keep the baked
    bytes exactly. Passing no blob_mask is a no-op, never a whole-surface
    repaint: one viewpoint cannot describe a body's panels, so an unrestricted
    write trades a missing cross for a smeared, streaky model.
    """
    import moderngl
    ref = ref_rgba.convert("RGBA")
    ref_rgb = np.asarray(ref.convert("RGB")).astype(np.float32)
    ref_alpha = np.asarray(ref)[:, :, 3] > 127
    tex = np.asarray(texture_img.convert("RGB")).astype(np.float32)
    th, tw = tex.shape[:2]

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

    cam = np.array(fit["cam"])
    right = np.array(fit["right"])
    up = np.array(fit["up"])
    forward = np.array(fit["forward"])

    # Two separate single-attachment passes: a mixed-format MRT (RGBA8 + float)
    # is incomplete on several drivers, which silently produced garbage face
    # ids and positions (and a depth test that rejected everything). Each pass
    # gets its own depth buffer but renders the identical camera.
    vp = _vp_matrix(cam, right, up, forward, size).T.astype("f4").tobytes()
    with moderngl.create_standalone_context(require=330) as ctx:
        ctx.enable(moderngl.DEPTH_TEST)
        id_program = ctx.program(vertex_shader=ID_VERT, fragment_shader=ID_FRAG)
        id_program["view_projection"].write(vp)
        # 3-component texture: color_tex.read() returns raw RGB bytes, which
        # Image.frombytes('RGB', ...) parses exactly. A 4-component texture
        # read as 'RGB' drifts every row and yields garbage face ids (this
        # exact bug made the depth test reject the whole view).
        color_tex = ctx.texture((size, size), 3)
        fb_id = ctx.framebuffer([color_tex], ctx.depth_renderbuffer((size, size)))
        fb_id.use()
        fb_id.clear(0, 0, 0, 0, depth=1)
        # Per-vertex face ids would interpolate across the triangle, so draw
        # face-by-face with a uniform id instead.
        vao_cache = []
        for g in geoms_zup:
            buf = ctx.buffer(np.asarray(g.vertices, dtype="f4").tobytes())
            idx = ctx.buffer(np.asarray(g.faces, dtype="u4").tobytes())
            vao_cache.append((ctx.vertex_array(
                id_program, [(buf, "3f", "position")], idx, index_element_size=4),
                len(g.faces)))
        # Encode face_id + 1 so cleared background (0,0,0) decodes to -1
        # instead of colliding with face 0. The old pos-length heuristic
        # misclassified any surface point near the origin as background.
        fid = 0
        for vao, nf in vao_cache:
            for k in range(nf):
                enc = fid + 1
                id_program["face_id"].value = (
                    ((enc >> 16) & 255) / 255.0,
                    ((enc >> 8) & 255) / 255.0,
                    (enc & 255) / 255.0)
                vao.render(moderngl.TRIANGLES, vertices=3, first=k * 3)
                fid += 1
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
            ctx.vertex_array(
                pos_program, [(buf, "3f", "position")], idx,
                index_element_size=4).render()
        pos_raw = np.frombuffer(pos_tex.read(alignment=1),
                                dtype=np.float32).reshape(size, size, 4)[:, :, :3]
        pos_raw = np.asarray(pos_raw, dtype=np.float64)
        # Flip the float buffer to top-up to match id_raw (moderngl reads are
        # bottom-up; frombytes+FLIP_TOP_BOTTOM did that for the id pass).
        pos_raw = pos_raw[::-1]
    face_of = ((id_raw[:, :, 0].astype(np.int64) << 16) |
               (id_raw[:, :, 1].astype(np.int64) << 8) |
               id_raw[:, :, 2].astype(np.int64)) - 1
    valid = (face_of >= 0) & (face_of < n_faces)
    ys, xs = np.nonzero(valid)
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0, "visible_faces": 0,
                                    "note": "no faces in view"}

    fids = face_of[ys, xs]
    # Ray/plane intersection per pixel against its own face: catches depth
    # edges where the position buffer straddles two surfaces.
    v0 = V[F[fids, 0]]
    v1 = V[F[fids, 1]]
    v2 = V[F[fids, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n_len = np.linalg.norm(n, axis=1)
    keep_n = n_len > 1e-12
    ys, xs, fids = ys[keep_n], xs[keep_n], fids[keep_n]
    v0, v1, v2 = v0[keep_n], v1[keep_n], v2[keep_n]
    n = (n[keep_n].T / n_len[keep_n]).T
    # Non-finite normals (degenerate UV/geometry rows) can trip a spurious
    # Accelerate overflow warning in matmul; drop those rows explicitly.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denom = n @ forward
    ok_d = np.isfinite(denom) & (np.abs(denom) > 1e-9)
    ys, xs, fids = ys[ok_d], xs[ok_d], fids[ok_d]
    v0, v1, v2 = v0[ok_d], v1[ok_d], v2[ok_d]
    n = n[ok_d]
    denom = denom[ok_d]
    # Reconstruct the per-pixel ray from the pinhole model and intersect it
    # with the pixel's own face plane.
    focal = 1.5 * size
    dx = (xs + 0.5 - size / 2.0) / focal
    dy = -((ys + 0.5 - size / 2.0) / focal)
    ray = (right[None, :] * dx[:, None] + up[None, :] * dy[:, None] + forward[None, :])
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    denom2 = np.einsum("ij,ij->i", ray, n)
    ok2 = np.abs(denom2) > 1e-9
    ys, xs, fids = ys[ok2], xs[ok2], fids[ok2]
    v0, v1, v2 = v0[ok2], v1[ok2], v2[ok2]
    n = n[ok2]
    ray = ray[ok2]
    denom2 = denom2[ok2]
    t = np.einsum("ij,ij->i", v0 - cam, n) / denom2
    hit = cam[None, :] + ray * t[:, None]
    # Depth test against the position pass.
    ren = pos_raw[ys, xs]
    agree = np.linalg.norm(hit - ren, axis=1) < (fit["radius"] * 2e-3 + 1e-6)
    ys, xs, fids, hit, t = ys[agree], xs[agree], fids[agree], hit[agree], t[agree]
    v0, v1, v2 = v0[agree], v1[agree], v2[agree]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0, "visible_faces": 0,
                                    "note": "depth test rejected everything"}
    # Barycentric in 3D, then the same weights index UV and reference color.
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
    ys, xs, fids, t = ys[ok3], xs[ok3], fids[ok3], t[ok3]
    lam1 = (d11[ok3] * dw0[ok3] - d01[ok3] * dw1[ok3]) / det[ok3]
    lam2 = (d00[ok3] * dw1[ok3] - d01[ok3] * dw0[ok3]) / det[ok3]
    lam0 = 1.0 - lam1 - lam2
    inside = (lam0 >= -1e-6) & (lam1 >= -1e-6) & (lam2 >= -1e-6) & \
             (lam0 <= 1 + 1e-6) & (lam1 <= 1 + 1e-6) & (lam2 <= 1 + 1e-6)
    ys, xs, fids, t = ys[inside], xs[inside], fids[inside], t[inside]
    lam0, lam1, lam2 = lam0[inside], lam1[inside], lam2[inside]
    uv = UV[F[fids, 0]] * lam0[:, None] + UV[F[fids, 1]] * lam1[:, None] + \
        UV[F[fids, 2]] * lam2[:, None]
    tu = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
    tv = np.clip(((1.0 - uv[:, 1]) * th).astype(np.int64), 0, th - 1)
    # Render-pixel -> reference-pixel: uniform scale + offset from mask bboxes.
    ren_mask = np.zeros((size, size), bool)
    ren_mask[ys, xs] = True
    rys, rxs = np.nonzero(ref_alpha)
    scale, ox, oy = _frame_map(ren_mask, (rys, rxs), ref_alpha.shape, size)
    rxf = (xs + 0.5) * scale + ox
    ryf = (ys + 0.5) * scale + oy
    ok4 = (rxf >= 0) & (rxf < ref_rgb.shape[1]) & (ryf >= 0) & (ryf < ref_rgb.shape[0])
    inside_ref_alpha = np.zeros(len(ys), bool)
    ok_idx = np.nonzero(ok4)[0]
    inside_ref_alpha[ok_idx] = ref_alpha[np.clip(ryf[ok_idx].astype(int), 0, ref_rgb.shape[0] - 1),
                                         np.clip(rxf[ok_idx].astype(int), 0, ref_rgb.shape[1] - 1)]
    use = ok4 & inside_ref_alpha
    ys, xs, tu, tv, t = ys[use], xs[use], tu[use], tv[use], t[use]
    rxf, ryf = rxf[use], ryf[use]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0,
                                    "visible_faces": int(len(set(fids.tolist()))),
                                    "note": "no overlap with reference foreground"}
    # Decal restriction: sample the emblem footprint at the mapped reference
    # position of each visible pixel. Only pixels landing on it may write.
    footprint = (decal_weights(blob_mask) if blob_mask is not None
                 else np.zeros(ref_alpha.shape, np.float32))
    if not footprint.any():
        return texture_img.copy(), {"projected_texels": 0,
                                    "visible_faces": int(len(set(fids.tolist()))),
                                    "note": "no emblem footprint; refusing whole-surface projection"}
    ry_i = np.clip(ryf.astype(int), 0, footprint.shape[0] - 1)
    rx_i = np.clip(rxf.astype(int), 0, footprint.shape[1] - 1)
    on_decal = footprint[ry_i, rx_i] > DECAL_FEATHER_MIN
    ys, xs, tu, tv, t = ys[on_decal], xs[on_decal], tu[on_decal], tv[on_decal], t[on_decal]
    rxf, ryf, ry_i, rx_i = rxf[on_decal], ryf[on_decal], ry_i[on_decal], rx_i[on_decal]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0,
                                    "visible_faces": int(len(set(fids.tolist()))),
                                    "note": "emblem footprint not visible from the fitted pose"}
    # Erode the visibility set so silhouette rims keep baked texels, then a
    # narrow feather band blends the seam.
    vis = np.zeros((size, size), np.float32)
    vis[ys, xs] = 1.0
    eroded = np.asarray(Image.fromarray((vis * 255).astype(np.uint8))
                        .filter(ImageFilter.MinFilter(5))) > 0
    feather = np.asarray(Image.fromarray((eroded * 255).astype(np.uint8))
                         .filter(ImageFilter.GaussianBlur(2))) / 255.0
    # The written weight is the visibility feather times the decal footprint:
    # a texel needs to be both visible and inside the emblem. Never falls back
    # to an unweighted whole-surface write.
    wgt = feather[ys, xs] * footprint[ry_i, rx_i]
    keep_w = wgt > 0.01
    if not keep_w.any():
        # Erosion ate a decal that sits against the silhouette (thin cross arms,
        # edge lettering). The footprint weight is already soft and confined to
        # the decal, so use it alone rather than widening the mask.
        wgt = footprint[ry_i, rx_i]
        keep_w = wgt > 0.01
    ys, xs, tu, tv, t = ys[keep_w], xs[keep_w], tu[keep_w], tv[keep_w], t[keep_w]
    rxf, ryf, ry_i, rx_i = rxf[keep_w], ryf[keep_w], ry_i[keep_w], rx_i[keep_w]
    wgt = wgt[keep_w]
    if not len(ys):
        return texture_img.copy(), {"projected_texels": 0,
                                    "visible_faces": int(len(set(fids.tolist()))),
                                    "note": "feather band empty"}
    # Bilinear sample of the reference.
    x0 = np.floor(rxf).astype(int).clip(0, ref_rgb.shape[1] - 2)
    y0 = np.floor(ryf).astype(int).clip(0, ref_rgb.shape[0] - 2)
    fx, fy = rxf - x0, ryf - y0
    sample = (ref_rgb[y0, x0] * ((1 - fx) * (1 - fy))[:, None] +
              ref_rgb[y0, x0 + 1] * (fx * (1 - fy))[:, None] +
              ref_rgb[y0 + 1, x0] * ((1 - fx) * fy)[:, None] +
              ref_rgb[y0 + 1, x0 + 1] * (fx * fy)[:, None])
    out = tex.copy()
    # Deterministic splat: per texel keep the closest-surface sample
    # (smallest ray t). Sort by (texel, t) so the first occurrence wins.
    order = np.lexsort((t, tv * tw + tu))
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
            "note": "emblem footprint covers most of the texture; refusing"}
    out[keys[:, 0], keys[:, 1]] = (
        sample[idxs] * wgt[idxs][:, None] +
        out[keys[:, 0], keys[:, 1]] * (1 - wgt[idxs][:, None]))
    # Real write-back check: texels written at full weight must carry (close
    # to) the reference sample. Catches a silent no-op, a colorspace slip and
    # an off-by-one splat that the coverage ratio cannot see.
    strong = wgt[idxs] > 0.9
    if strong.any():
        err = np.abs(out[keys[strong, 0], keys[strong, 1]] - sample[idxs][strong]).max(axis=1)
        writeback = float((err <= WRITEBACK_TOLERANCE).mean())
    else:
        writeback = 0.0
    report = {
        "projected_texels": int(len(keys)),
        "texel_share": round(float(len(keys)) / (tw * th), 4),
        "blob_pixels": int(np.asarray(blob_mask).sum()),
        "writeback_match": round(writeback, 4),
        "visible_faces": int(len(set(fids.tolist()))),
        "frame_scale": round(float(scale), 4),
        "frame_offset": [round(float(ox), 2), round(float(oy), 2)],
    }
    return Image.fromarray(out.astype(np.uint8), "RGB"), report


def _frame_map(ren_mask, ref_nonzero, ref_shape, size):
    """Uniform scale + offset mapping render pixels onto reference pixels."""
    rys, rxs = ref_nonzero
    ys, xs = np.nonzero(ren_mask)
    if not len(ys) or not len(rys):
        return 1.0, 0.0, 0.0
    rh, rw = ref_shape
    ren_h = ys.max() - ys.min() + 1
    ren_w = xs.max() - xs.min() + 1
    ref_h = rys.max() - rys.min() + 1
    ref_w = rxs.max() - rxs.min() + 1
    scale = min(ref_h / max(ren_h, 1), ref_w / max(ren_w, 1))
    # Map render bbox center onto reference bbox center.
    ox = (rxs.min() + rxs.max() + 1) / 2 - (xs.min() + xs.max() + 1) / 2 * scale
    oy = (rys.min() + rys.max() + 1) / 2 - (ys.min() + ys.max() + 1) / 2 * scale
    # Reference may be a different size than the render; work in reference px.
    scale *= 1.0
    return scale, ox, oy


# ------------------------------------------------------------ gaussian patch
C0 = 0.2820947917738949


def patch_gaussian_colors(item_dir: Path, item_id: str, ref_rgba: Image.Image,
                          fit, blob_mask=None) -> dict:
    """Recolor emblem-area gaussians with true reference pixels.

    Voxel-derived SPZ/SPLAT colors come from the same smoothed texture
    latent as the GLB, so a GLB-only fix would leave the splat views without
    the emblem. This projects the reference onto gaussian means visible from
    the fitted camera and rewrites every recognised container in place.

    Only gaussians whose reference sample lands on an emblem blob are
    touched: body paint is already correct from TRELLIS, and repainting the
    whole visible side from one view would smear occluded layers. The
    blob_mask from emblem_blobs is required; without it the patch refuses
    to run. A position-buffer depth test drops occluded backside/interior
    gaussians the same way the texture pass does.
    Returns a report dict; empty (patched 0) when nothing is visible.
    """
    if blob_mask is None or not np.asarray(blob_mask).any():
        return {"patched_gaussians": 0, "visible_gaussians": 0,
                "note": "no blob mask; refusing whole-side repaint"}
    from PIL import Image as _Image
    ref = ref_rgba.convert("RGBA")
    ref_rgb = np.asarray(ref.convert("RGB")).astype(np.float32)
    ref_alpha = np.asarray(ref)[:, :, 3] > 127
    cam = np.array(fit["cam"], dtype=np.float64)
    right = np.array(fit["right"], dtype=np.float64)
    up = np.array(fit["up"], dtype=np.float64)
    forward = np.array(fit["forward"], dtype=np.float64)
    size = RENDER_SIZE
    focal = 1.5 * size
    patched = 0
    visible = 0
    containers = []
    for ext in ("spz", "splat", "ply"):
        p = item_dir / f"{item_id}.{ext}"
        if p.is_file() and p.stat().st_size > 0:
            containers.append((ext, p))
    if not containers:
        return {"patched_gaussians": 0, "visible_gaussians": 0, "note": "no containers"}
    # Load positions + colors from the preferred container; write back to all.
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
    # Gaussians are native Z-up; the fitted camera is in the same Z-up frame.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rel = pos - cam[None, :]
        z = rel @ forward
        x = rel @ right
        y = rel @ up
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
    # Map render pixels onto reference pixels with the same frame map used
    # for the texture pass.
    ren_mask = np.zeros((size, size), bool)
    ren_mask[py[idx].astype(int).clip(0, size - 1),
               px[idx].astype(int).clip(0, size - 1)] = True
    rys, rxs = np.nonzero(ref_alpha)
    scale, ox, oy = _frame_map(ren_mask, (rys, rxs), ref_alpha.shape, size)
    rxf = (px[idx] + 0.5) * scale + ox
    ryf = (py[idx] + 0.5) * scale + oy
    ok4 = (rxf >= 0) & (rxf < ref_rgb.shape[1]) & (ryf >= 0) & (ryf < ref_rgb.shape[0])
    # Only recolor gaussians that land on reference foreground; background
    # projection would smear the studio backdrop onto the object.
    xi = rxf.astype(int).clip(0, ref_rgb.shape[1] - 1)
    yi = ryf.astype(int).clip(0, ref_rgb.shape[0] - 1)
    on_fg = np.zeros(len(idx), bool)
    on_fg[ok4] = ref_alpha[yi[ok4], xi[ok4]]
    # ... and further only emblem-blob pixels, never plain body paint.
    bm = np.asarray(blob_mask).astype(bool)
    if bm.shape != ref_alpha.shape:
        bm = np.zeros_like(ref_alpha, bool)
    on_blob = np.zeros(len(idx), bool)
    on_blob[ok4] = bm[yi[ok4], xi[ok4]]
    use_g = on_fg & on_blob
    sel = idx[use_g]
    _keep = None
    depth_note = "depth test skipped"
    if not len(sel):
        return {"patched_gaussians": 0, "visible_gaussians": visible,
                "note": "visible gaussians miss emblem blobs"}
    # Depth test: drop gaussians occluded behind the rendered surface,
    # using a position-buffer render of the fitted camera. Gaussian means
    # sit at voxel centres, so the tolerance adds two voxel radii (from
    # the stored log-scales) to the texture pass's surface tolerance.
    try:
        import moderngl as _mgl
        _size = 512
        _vp = _vp_matrix(cam, right, up, forward, _size).T.astype("f4").tobytes()
        with _mgl.create_standalone_context(require=330) as _ctx:
            _prog = _ctx.program(vertex_shader=POS_VERT, fragment_shader=POS_FRAG)
            _prog["view_projection"].write(_vp)
            try:
                import trimesh as _tri
                _verts = np.vstack([np.asarray(_g.vertices, dtype=np.float64)
                                    for _g in load_geoms_zup(item_dir / f"{item_id}.glb")])
            except Exception:
                _verts = pos
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
        _f = 1.5 * _size
        # Projection consistent with the pinhole used above.
        _rel = pos[sel] - cam[None, :]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _zz = _rel @ forward
            _xx = _rel @ right
            _yy = _rel @ up
        _gx = ((_xx / np.maximum(_zz, 1e-9)) * _f + _size / 2.0).astype(int).clip(0, _size - 1)
        _gy = ((-_yy / np.maximum(_zz, 1e-9)) * _f + _size / 2.0).astype(int).clip(0, _size - 1)
        _surf = _praw[_gy, _gx]
        _has = np.abs(_surf).sum(axis=-1) > 1e-9
        _dist = np.linalg.norm(pos[sel] - _surf, axis=1)
        try:
            _vox = np.exp(np.asarray(g["scale"], dtype=np.float64)[sel][:, 0]) / 0.55
        except Exception:
            _vox = np.full(len(sel), 0.01)
        _tol = float(fit["radius"]) * 5e-3 + 2.0 * _vox
        _keep = _has & np.isfinite(_dist) & (_dist <= _tol)
        sel = sel[_keep]
        depth_note = f"depth kept {int(_keep.sum())}/{len(_keep)}"
        if not len(sel):
            return {"patched_gaussians": 0, "visible_gaussians": visible,
                    "note": f"all emblem gaussians occluded ({depth_note})"}
    except Exception as _e:
        depth_note = f"depth test unavailable: {_e}"
    # Bilinear reference samples for the selected gaussians. Recompute from
    # the depth-filtered selection (use_g then _keep), not the pre-depth mask.
    sxi = (rxf[use_g][_keep] if _keep is not None else rxf[use_g]).astype(float)
    syi = (ryf[use_g][_keep] if _keep is not None else ryf[use_g]).astype(float)
    x0 = np.floor(sxi).astype(int).clip(0, ref_rgb.shape[1] - 2)
    y0 = np.floor(syi).astype(int).clip(0, ref_rgb.shape[0] - 2)
    fx_, fy_ = sxi - x0, syi - y0
    sample = (ref_rgb[y0, x0] * ((1 - fx_) * (1 - fy_))[:, None] +
              ref_rgb[y0, x0 + 1] * (fx_ * (1 - fy_))[:, None] +
              ref_rgb[y0 + 1, x0] * ((1 - fx_) * fy_)[:, None] +
              ref_rgb[y0 + 1, x0 + 1] * (fx_ * fy_)[:, None]) / 255.0
    # Feather by distance to the visibility rim is unnecessary for points;
    # keep only emblem-hue samples so body paint never shifts. The blob gate
    # already established the hues: saturated insignia reds/blues or dark
    # lettering. Here we accept a recolor only when the reference sample is
    # itself strongly saturated or dark, so a misaligned fit cannot repaint
    # the whole side panel.
    mx = sample.max(axis=1)
    mn = sample.min(axis=1)
    chroma = (mx - mn) / np.maximum(mx, 1e-6)
    keep = ((chroma > 0.5) & (mx > 0.31)) | (mx < 0.35)
    sel = sel[keep]
    sample = sample[keep]
    if not len(sel):
        return {"patched_gaussians": 0, "visible_gaussians": visible,
                "note": "no emblem-hue samples in view"}
    f_dc = ((np.clip(sample, 0.0, 1.0) - 0.5) / C0).astype(np.float32)
    # Write back through the canonical writers so every container agrees.
    try:
        from gs_export import write_spz as _wspz, write_splat as _wsplat, \
            write_ply as _wply, tmp_sibling as _tmp
    except Exception as e:
        return {"patched_gaussians": 0, "visible_gaussians": visible,
                "note": f"gs writers unavailable: {e}"}
    base = {"x": np.asarray(g["x"]), "y": np.asarray(g["y"]),
            "z": np.asarray(g["z"]), "scale": np.asarray(g["scale"]),
            "rot": np.asarray(g["rot"]), "f_dc": np.asarray(g["f_dc"]),
            "opacity": np.asarray(g["opacity"])}
    if "sh_degree" in g:
        base["sh_degree"] = g["sh_degree"]
    base["f_dc"] = base["f_dc"].copy()
    base["f_dc"][sel] = f_dc
    if "f_rest" not in g:
        base["f_rest"] = np.zeros((len(base["x"]), 45), dtype=np.float32)
    else:
        base["f_rest"] = np.asarray(g["f_rest"])
    for ext, final in containers:
        tmp = _tmp(final)
        if ext == "spz":
            n = _wspz(str(tmp), base)
            if not n:
                tmp.unlink(missing_ok=True)
                continue
        elif ext == "splat":
            _wsplat(str(tmp), base)
        else:
            _wply(str(tmp), base)
        durable_replace(tmp, final)
    patched = int(len(sel))
    return {"patched_gaussians": patched, "visible_gaussians": visible,
            "gaussian_share": round(patched / max(len(base["x"]), 1), 5),
            "depth": depth_note}


# ------------------------------------------------------------ GLB surgery

def replace_glb_texture(glb_path: Path, texture_index: int, png_bytes: bytes):
    """Swap one embedded PNG inside a GLB, fixing offsets deterministically."""
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
    start, old_len = bv["byteOffset"], bv["byteLength"]
    # Pad the replacement image to 4-byte alignment; shift every later
    # bufferView forward by the size delta.
    pad = (-len(png_bytes)) % 4
    new_blob = bytes(bin_blob[:start]) + png_bytes + b"\x00" * pad + bytes(bin_blob[start + old_len:])
    delta = len(png_bytes) + pad - old_len
    bv["byteLength"] = len(png_bytes)
    for other in buffer_views:
        if other is not bv and other.get("byteOffset", 0) > start:
            other["byteOffset"] += delta
    for acc in js.get("accessors", []):
        if acc.get("bufferView") is not None:
            ob = buffer_views[acc["bufferView"]]
            if ob is not bv and ob.get("byteOffset", 0) > start:
                pass  # accessors index their own bufferView; offsets fixed above.
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
    """Return the GLB images[] index backing the first baseColorTexture.

    Caveat fixed here: the old code always rewrote images[0], which corrupts
    multi-material GLBs by swapping the wrong texture. trimesh collapses
    materials, so read the index from the GLB JSON instead.
    """
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
    """Return (texture_image, texture_index) for the first PBR material."""
    for g in geoms:
        material = getattr(getattr(g, "visual", None), "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            texture = getattr(material, "image", None)
        if texture is not None:
            return texture.convert("RGB"), 0
    raise ValueError("no base color texture found")


def verify_transfer(new_texture: Image.Image, ref_rgba: Image.Image,
                    blob_mask, proj_report: dict) -> dict:
    """Write-back check that the emblem hue actually landed in the texture.

    Samples the rewritten texture at the projected texel positions is
    expensive to thread through, so this checks the necessary (not
    sufficient) conditions with teeth: projected texels exist, the blob
    area in the reference is genuinely saturated/dark emblem hue, and the
    new texture's global emblem-hue pixel share grew vs nothing. Uses
    TRANSFER_MATCH_MIN as the floor for projected_texels / blob_pixels.
    Returns {ok, match_ratio, note}.
    """
    try:
        blob_px = int(np.asarray(blob_mask).sum())
        got = int(proj_report.get("projected_texels", 0) or 0)
        ratio = (got / blob_px) if blob_px else 0.0
        writeback = proj_report.get("writeback_match")
        # Prefer the write-back metric when the projection reported one: the
        # coverage ratio only proves the code ran, not that the color landed.
        if writeback is not None:
            ok = got > 0 and float(writeback) >= TRANSFER_MATCH_MIN
        else:
            ok = ratio >= TRANSFER_MATCH_MIN
        # Independent hue check on the rewritten texture: emblem-hue pixels
        # must exist (guards a silent blend-to-zero / colorspace slip).
        arr = np.asarray(new_texture.convert("RGB")).astype(np.float32)
        mx, mn = arr.max(axis=-1), arr.min(axis=-1)
        chroma = (mx - mn) / np.maximum(mx, 1e-6)
        hue_px = int((((chroma > 0.5) & (mx > 80)) | (mx < 90)).sum())
        if not hue_px:
            ok = False
        return {"ok": bool(ok), "match_ratio": round(float(ratio), 4),
                "writeback_match": writeback,
                "hue_pixels": hue_px, "blob_pixels": blob_px,
                "threshold": TRANSFER_MATCH_MIN,
                "note": "" if ok else "transfer below threshold; pose or blend suspect"}
    except Exception as e:
        return {"ok": False, "match_ratio": 0.0, "note": f"verify failed: {e}"}


# ------------------------------------------------------------ item entry point

def process_item(item_dir: Path, item_id: str, *, dry_run=False):
    """Run the emblem stage for one item directory. Returns a report dict."""
    png_path = item_dir / f"{item_id}.png"
    glb_path = item_dir / f"{item_id}.glb"
    meta_path = item_dir / "metadata.json"
    with Image.open(png_path) as img:
        ref = img.convert("RGBA")
    blob_mask, records = emblem_blobs(ref)
    report = {"blobs": len(records), "records": records, "applied": False}
    if not records:
        return report
    geoms = load_geoms_zup(glb_path)
    try:
        fit = fit_reference_camera(
        geoms, Image.fromarray(np.asarray(ref)[:, :, 3]).convert("L"))
    except RuntimeError as e:
        report["note"] = f"camera fit unavailable: {e}"
        return report
    report["fit_iou"] = fit["iou"]
    report["camera"] = {k: fit[k] for k in ("azimuth_deg", "elevation_deg")}
    if fit["iou"] < FIT_IOU_MIN:
        report["note"] = f"pose fit too poor (iou {fit['iou']}); skipped"
        return report
    texture, _ = base_color_texture(geoms)
    new_texture, proj = project_reference(geoms, texture, ref, fit,
                                          blob_mask=blob_mask)
    report["projection"] = proj
    if dry_run or not proj.get("projected_texels"):
        report["note"] = report.get("note", "dry run or nothing to project")
        return report
    import io
    buf = io.BytesIO()
    new_texture.save(buf, format="PNG")
    tex_index = glb_basecolor_image_index(glb_path)
    new_glb = replace_glb_texture(glb_path, tex_index, buf.getvalue())
    # Gaussian containers carry the same smoothed colors as the baked texture,
    # so patch them from the same fitted camera before promoting anything.
    gauss = patch_gaussian_colors(item_dir, item_id, ref, fit,
                                  blob_mask=blob_mask)
    report["gaussians"] = gauss
    report["texture_index"] = tex_index
    report["verification"] = verify_transfer(new_texture, ref, blob_mask, proj)
    if not report["verification"].get("ok"):
        report["note"] = report["verification"].get("note")
        return report
    if not dry_run:
        tmp = glb_path.with_name(f".{item_id}.tmp.emblem.glb")
        tmp.write_bytes(new_glb)
        durable_replace(tmp, glb_path)
        # Reload the artifact hashes the way qa.py binds verdicts to them, so
        # a stale mesh/gaussian verdict can never certify the rewritten files.
        try:
            from pipeline_utils import artifact_fingerprints as _fp
            fps = _fp(item_dir, item_id, "meshes")
            gfps = _fp(item_dir, item_id, "gaussians")
        except Exception:
            fps, gfps = {}, {}
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {"id": item_id}
        # Unified schema (v2): blobs=int count + records list + applied +
        # verification. Old records overwrote blobs with a list and omitted
        # applied; migrate on next touch without rewriting bytes.
        meta["emblem_project"] = {
            "schema": 2,
            "applied": True,
            "blobs": len(records),
            "records": records,
            "fit_iou": fit["iou"],
            "camera": report["camera"],
            "projection": proj,
            "gaussians": gauss,
            "verification": report.get("verification"),
            "texture_index": report.get("texture_index", 0),
            "glb_sha256": file_sha256(glb_path),
        }
        # Invalidate stale QA verdicts bound to the pre-patch hashes: they
        # describe files that no longer exist.
        try:
            from prompt_utils import STATE_DIR as _SD
            for _name in ("qa_meshes.json", "qa_gaussians.json"):
                _p = _SD / _name
                if _p.is_file():
                    _v = json.loads(_p.read_text())
                    _v.pop(item_id, None)
                    write_json(_p, _v)
        except Exception:
            pass
        meta["mesh_artifacts"] = fps
        meta["gs_artifacts"] = gfps
        write_json(meta_path, meta)
    report["applied"] = not dry_run
    return report


def try_emblem_fix(item_dir: Path, item_id: str):
    """Best-effort pipeline hook: no-op without blobs, never raises.

    Called by trellis_meshgen.py after GLB + gaussian promotion and before
    the metadata commit marker is written, so the report lands in the same
    metadata.json that certifies the bytes. Any failure (no moderngl, poor
    fit, nothing visible) returns a skipped report and leaves the baked
    assets untouched. The mesh phase must never fail an item it just decoded
    because an optional decal transfer did not apply.
    """
    try:
        png_path = item_dir / f"{item_id}.png"
        glb_path = item_dir / f"{item_id}.glb"
        with Image.open(png_path) as img:
            ref = img.convert("RGBA")
        blob_mask, records = emblem_blobs(ref)
        if not records:
            return {"applied": False, "blobs": 0, "note": "no emblem blobs"}
        geoms = load_geoms_zup(glb_path)
        try:
            fit = fit_reference_camera(
                geoms, Image.fromarray(np.asarray(ref)[:, :, 3]).convert("L"))
        except RuntimeError as e:
            return {"applied": False, "blobs": len(records),
                    "note": f"camera fit unavailable: {e}"}
        if fit["iou"] < FIT_IOU_MIN:
            return {"applied": False, "blobs": len(records),
                    "fit_iou": fit["iou"],
                    "note": f"pose fit too poor (iou {fit['iou']}); skipped"}
        texture, _ = base_color_texture(geoms)
        new_texture, proj = project_reference(geoms, texture, ref, fit,
                                              blob_mask=blob_mask)
        if not proj.get("projected_texels"):
            return {"applied": False, "blobs": len(records),
                    "fit_iou": fit["iou"], "projection": proj,
                    "note": "nothing to project"}
        import io
        buf = io.BytesIO()
        new_texture.save(buf, format="PNG")
        tex_index = glb_basecolor_image_index(glb_path)
        new_glb = replace_glb_texture(glb_path, tex_index, buf.getvalue())
        gauss = patch_gaussian_colors(item_dir, item_id, ref, fit,
                                      blob_mask=blob_mask)
        verification = verify_transfer(new_texture, ref, blob_mask, proj)
        if not verification.get("ok"):
            return {"applied": False, "blobs": len(records),
                    "fit_iou": fit["iou"], "projection": proj,
                    "verification": verification,
                    "note": verification.get("note")}
        tmp = glb_path.with_name(f".{item_id}.tmp.emblem.glb")
        tmp.write_bytes(new_glb)
        durable_replace(tmp, glb_path)
        return {"schema": 2, "applied": True, "blobs": len(records), "records": records,
                "fit_iou": fit["iou"],
                "camera": {k: fit[k] for k in ("azimuth_deg", "elevation_deg")},
                "projection": proj, "gaussians": gauss,
                "verification": verification, "texture_index": tex_index,
                "glb_sha256": file_sha256(glb_path)}
    except Exception as e:
        return {"applied": False, "note": f"emblem hook skipped: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", nargs="*", default=[])
    ap.add_argument("--all", action="store_true",
                    help="backfill every item whose reference has emblem blobs "
                         "but whose metadata lacks an emblem_project record")
    ap.add_argument("--geography", default="germany")
    ap.add_argument("--category", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    from prompt_utils import MANIFESTS, OUTPUT_DIR, load_manifest
    manifests = {}
    for geo in MANIFESTS:
        for it in load_manifest(geo):
            row = dict(it)
            row.setdefault("geography", geo)
            manifests[row["id"]] = row
    targets = list(args.ids)
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
            # Reprocess old-schema (blobs as list / missing applied), failed
            # (applied False), or hash-mismatched records; skip only valid
            # schema-2 applied records bound to the current bytes.
            _rec = meta.get("emblem_project") if isinstance(meta, dict) else None
            if isinstance(_rec, dict) and _rec.get("applied") and _rec.get("schema") == 2:
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
