"""Export TRELLIS.2 decoded voxels as a 3D Gaussian Splatting PLY.

TRELLIS.2 decodes to a mesh plus a sparse voxel grid of PBR attributes
(MeshWithVoxel). There is no gaussian decoder in TRELLIS.2, so we convert the
voxel grid directly: one gaussian per active voxel, centered in the voxel,
sized to the voxel, colored by the decoded base color. The result is a
standard INRIA-format 3DGS PLY that loads in any splat viewer.
"""

from pathlib import Path

import numpy as np

C0 = 0.2820947917738949  # zeroth SH basis value


def tmp_sibling(final_path) -> Path:
    """Transient sibling path used to atomically promote an export.

    `a/b.glb` -> `a/.b.tmp.glb`. The real suffix is preserved on purpose:
    exporters infer the container format from the extension, so the naive
    `b.glb.tmp` makes trimesh raise "unsupported export format: tmp" and
    silently kills the whole mesh phase. The leading dot marks the file as
    transient so a stale one is never mistaken for an output.

    Callers write to this path, then `tmp_sibling(final).replace(final)`.
    """
    final_path = Path(final_path)
    return final_path.with_name(f".{final_path.stem}.tmp{final_path.suffix}")


def voxels_to_gaussians(coords, attrs, origin, voxel_size, layout,
                        opacity_channel: bool = True) -> dict:
    """Build gaussian arrays from a MeshWithVoxel's voxel data.

    Args:
        coords: (N, 4) int tensor/array [batch, x, y, z]
        attrs: (N, C) float tensor/array of voxel attributes
        origin: (3,) world position of voxel (0,0,0)
        voxel_size: scalar voxel edge length
        layout: dict of attribute-name -> slice (base_color, metallic, roughness, alpha)
    """
    coords = np.asarray(coords)
    attrs = np.asarray(attrs, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    vs = float(voxel_size)

    spatial = coords[:, 1:4] if coords.ndim == 2 and coords.shape[1] >= 4 else coords
    xyz = origin + (spatial.astype(np.float32) + 0.5) * vs

    rgb = np.clip(attrs[:, layout["base_color"]], 0.0, 1.0)
    f_dc = (rgb - 0.5) / C0

    if opacity_channel and "alpha" in layout:
        a = np.clip(attrs[:, layout["alpha"]], 0.0, 1.0)
        # only keep voxels the model considers occupied
        keep = (a > 0.05).ravel()
        xyz, f_dc, a = xyz[keep], f_dc[keep], a[keep]
        opacity = np.log(a / (1.0 - a + 1e-6) + 1e-6)  # inverse sigmoid
    else:
        opacity = np.full((len(xyz), 1), 8.0, dtype=np.float32)

    n = len(xyz)
    return {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "f_dc": f_dc,
        "f_rest": np.zeros((n, 45), dtype=np.float32),
        "opacity": opacity.reshape(n, 1),
        "scale": np.full((n, 3), np.log(vs * 0.55), dtype=np.float32),
        "rot": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
    }


# Canonical INRIA 3DGS vertex layout, in order: position, normal, SH DC,
# SH rest, opacity, scale, rotation quaternion. 62 float32 per gaussian.
#
# The normals matter for a subtle reason. This file previously listed 59
# properties in the header (omitting nx/ny/nz) while allocating a 62-wide
# buffer, so every written .ply declared 59 floats per vertex and contained 62 —
# readers resynchronised onto garbage after the first gaussian and the file was
# unusable. The buffer width was right and the property list was short, so the
# normals are restored here rather than the stride being trimmed: 62 is also
# what splat viewers expect from an INRIA-layout PLY.
PLY_PROPS = (
    ["x", "y", "z"]
    + ["nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + [f"f_rest_{i}" for i in range(45)]
    + ["opacity"]
    + [f"scale_{i}" for i in range(3)]
    + [f"rot_{i}" for i in range(4)]
)
PLY_STRIDE = len(PLY_PROPS)  # 62


def write_ply_to(fp, g: dict) -> int:
    """Write gaussians as a standard 3DGS PLY into a binary file object.

    Single implementation of the INRIA layout; write_ply() is the path-taking
    wrapper. The two used to be byte-identical copies, which is exactly the
    shape of duplication where one copy quietly stops matching the other.

    Normals are written as zeros: voxel-derived gaussians have no meaningful
    surface normal, and 3DGS renderers ignore the field. It is present so the
    header and the payload agree on 62 floats per vertex.
    """
    n = len(g["x"])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in PLY_PROPS)
        + "end_header\n"
    )

    buf = np.zeros((n, PLY_STRIDE), dtype=np.float32)
    buf[:, 0:3] = np.stack([g["x"], g["y"], g["z"]], axis=1)
    # 3:6 are the normals, left at zero.
    buf[:, 6:9] = g["f_dc"]
    buf[:, 9:54] = g["f_rest"]
    buf[:, 54:55] = g["opacity"]
    buf[:, 55:58] = g["scale"]
    buf[:, 58:62] = g["rot"]

    fp.write(header.encode("ascii"))
    fp.write(buf.tobytes())
    return n


def write_ply(path: str, g: dict) -> int:
    """Write gaussians as a standard 3DGS PLY (float32, binary little endian)."""
    with open(path, "wb") as f:
        return write_ply_to(f, g)


def write_splat(path: str, g: dict) -> int:
    """Write gaussians in the antimatter15 .splat format: 32 bytes/splat.

    Layout per splat: position float32 xyz, scale float32 xyz, color rgba u8,
    rotation quaternion u8 (w,x,y,z with 0 bias, 128 scale). ~6x smaller than
    the float32 PLY and loads directly in web splat viewers.
    """
    n = len(g["x"])
    buf = np.empty((n, 32), dtype=np.uint8)

    pos = np.stack([g["x"], g["y"], g["z"]], axis=1).astype(np.float32)
    buf[:, 0:12] = pos.view(np.uint8).reshape(n, 12)

    scl = np.exp(np.stack([g["scale"][:, 0], g["scale"][:, 1], g["scale"][:, 2]], axis=1)).astype(np.float32)
    buf[:, 12:24] = scl.view(np.uint8).reshape(n, 12)

    rgba = np.empty((n, 4), dtype=np.uint8)
    rgb = np.clip(g["f_dc"] * C0 + 0.5, 0.0, 1.0)
    rgba[:, 0:3] = (rgb * 255).round().astype(np.uint8)
    rgba[:, 3:4] = np.clip(255 / (1 + np.exp(-g["opacity"])), 0, 255).astype(np.uint8)
    buf[:, 24:28] = rgba

    # .splat wants (w,x,y,z); ours is identity (1,0,0,0) from voxels_to_gaussians
    rot = np.clip(g["rot"][:, [3, 0, 1, 2]] * 128 + 128, 0, 255).astype(np.uint8)
    buf[:, 28:32] = rot

    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return n


def write_spz(path: str, g: dict) -> int:
    """Write gaussians as .spz (Niantic compressed format, ~10x smaller than PLY).

    Requires the official Niantic `spz` python binding (GaussianCloud +
    save_spz; installed from github.com/nianticlabs/spz). Returns 0 if the
    package is unavailable or the pack fails, so callers always have a .splat
    fallback. Any exception is swallowed deliberately: an optional container
    must never take an item down.

    The old body of this function is worth remembering. It set
    `GaussianCloud().albedos/opacities` and called `saveSplatToPath` -- an API
    no released spz binding ever had. The import succeeded, the attribute
    access raised, every call fell back to .splat, and zero .spz files were
    ever produced; the defect was invisible because the fallback hid it. The
    real binding takes flat arrays:

      positions  (n*3) xyz
      scales     (n*3) log scale -- voxels_to_gaussians already stores logs
      rotations  (n*4) xyzw    -- internal order is wxyz, so columns rotate
      colors     (n*3) base RGB (SH DC), 0..1
      alphas     (n,)  opacity BEFORE sigmoid -- voxels_to_gaussians stores
                 the inverse sigmoid (logit), which is exactly this
    """
    try:
        import spz  # type: ignore
    except Exception:
        return 0
    n = len(g["x"])
    try:
        cloud = spz.GaussianCloud()
        cloud.positions = np.stack([g["x"], g["y"], g["z"]],
                                   axis=1).astype(np.float32).reshape(-1)
        cloud.scales = np.asarray(g["scale"], dtype=np.float32).reshape(-1)
        rot = np.asarray(g["rot"], dtype=np.float32)
        # wxyz -> xyzw: spz wants (x,y,z,w) per point.
        cloud.rotations = np.stack(
            [rot[:, 1], rot[:, 2], rot[:, 3], rot[:, 0]],
            axis=1).astype(np.float32).reshape(-1)
        cloud.colors = np.clip(np.asarray(g["f_dc"], dtype=np.float32) * C0
                               + 0.5, 0.0, 1.0).reshape(-1)
        cloud.alphas = np.asarray(g["opacity"], dtype=np.float32).reshape(-1)
        ok = spz.save_spz(cloud, spz.PackOptions(), path)
    except Exception:
        return 0
    return n if ok else 0


def export_gaussians_from_mesh_with_voxel(mesh_out, path_stem: str,
                                          condensed: bool = True):
    """Convenience wrapper taking a TRELLIS.2 MeshWithVoxel object.

    Condensed mode writes BOTH containers so every item carries the compact
    format and a universally viewable one:

      <path_stem>.spz   Niantic compressed container (~10x smaller than PLY)
      <path_stem>.splat universal 32 B/splat web format

    When the `spz` package is unavailable only the .splat is written. With
    condensed=False the raw float32 PLY (248 B/splat) is written instead of
    the two condensed containers.

    Every branch writes to `tmp_sibling(final)` and leaves promotion to the
    caller, so the mesh and the 3DGS files appear together or not at all.

    Returns (written_paths: tuple[Path, ...], gaussian_count: int). The paths
    are the final (pre-promotion) names, ordered .spz before .splat when both
    were written.
    """
    g = voxels_to_gaussians(
        coords=mesh_out.coords.cpu().numpy(),
        attrs=mesh_out.attrs.cpu().numpy(),
        origin=mesh_out.origin.cpu().numpy(),
        voxel_size=mesh_out.voxel_size,
        layout=mesh_out.layout,
    )
    if not condensed:
        ply_path = Path(f"{path_stem}.ply")
        return (ply_path,), write_ply(str(tmp_sibling(ply_path)), g)

    written = []
    spz_path = Path(f"{path_stem}.spz")
    spz_tmp = tmp_sibling(spz_path)
    n_spz = write_spz(str(spz_tmp), g)
    if n_spz:
        written.append(spz_path)
    else:
        # Package missing or pack failed: drop any partial file, keep the
        # .splat-only path. The .splat count below is the source of truth.
        spz_tmp.unlink(missing_ok=True)

    splat_path = Path(f"{path_stem}.splat")
    n_splat = write_splat(str(tmp_sibling(splat_path)), g)
    written.append(splat_path)  # universal fallback: always present
    return tuple(written), n_splat
