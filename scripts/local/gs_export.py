"""Export TRELLIS.2 decoded voxels as a 3D Gaussian Splatting PLY.

TRELLIS.2 decodes to a mesh plus a sparse voxel grid of PBR attributes
(MeshWithVoxel). There is no gaussian decoder in TRELLIS.2, so we convert the
voxel grid directly: one gaussian per active voxel, centered in the voxel,
sized to the voxel, colored by the decoded base color. The result is a
standard INRIA-format 3DGS PLY that loads in any splat viewer.
"""

import numpy as np

C0 = 0.2820947917738949  # zeroth SH basis value


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


def write_ply(path: str, g: dict) -> int:
    """Write gaussians as a standard 3DGS PLY (float32, binary little endian)."""
    props = ["x", "y", "z"]
    props += [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(45)]
    props += ["opacity"]
    props += [f"scale_{i}" for i in range(3)]
    props += [f"rot_{i}" for i in range(4)]

    n = len(g["x"])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )

    buf = np.empty((n, 62), dtype=np.float32)
    buf[:, 0:3] = np.stack([g["x"], g["y"], g["z"]], axis=1)
    buf[:, 3:6] = g["f_dc"]
    buf[:, 6:51] = g["f_rest"]
    buf[:, 51:52] = g["opacity"]
    buf[:, 52:55] = g["scale"]
    buf[:, 55:59] = g["rot"]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(buf.tobytes())
    return n


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


def write_ply_gz(path: str, g: dict) -> int:
    """gzip-compressed 3DGS PLY (~2-3x smaller); still standard after decompress."""
    import gzip
    import io

    raw = io.BytesIO()
    n = write_ply_to(raw, g)
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw.getvalue())
    return n


def write_ply_to(fp, g: dict) -> int:
    """write_ply into a file object (shared by write_ply and write_ply_gz)."""
    props = ["x", "y", "z"]
    props += [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(45)]
    props += ["opacity"]
    props += [f"scale_{i}" for i in range(3)]
    props += [f"rot_{i}" for i in range(4)]

    n = len(g["x"])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )

    buf = np.empty((n, 62), dtype=np.float32)
    buf[:, 0:3] = np.stack([g["x"], g["y"], g["z"]], axis=1)
    buf[:, 3:6] = g["f_dc"]
    buf[:, 6:51] = g["f_rest"]
    buf[:, 51:52] = g["opacity"]
    buf[:, 52:55] = g["scale"]
    buf[:, 55:59] = g["rot"]

    fp.write(header.encode("ascii"))
    fp.write(buf.tobytes())
    return n


def write_spz(path: str, g: dict) -> int:
    """Write gaussians as .spz (Niantic compressed format, ~10x smaller than PLY).

    Requires the `spz` python package. Returns 0 if unavailable.
    """
    try:
        import spz  # type: ignore
    except ImportError:
        return 0
    n = len(g["x"])
    gs = spz.GaussianCloud()
    gs.num_points = n
    gs.positions = np.stack([g["x"], g["y"], g["z"]], axis=1).astype(np.float32)
    gs.scales = np.exp(g["scale"]).astype(np.float32)
    gs.rotations = g["rot"].astype(np.float32)
    gs.albedos = np.clip(g["f_dc"] * C0 + 0.5, 0.0, 1.0).astype(np.float32)
    gs.albedos = (gs.albedos - 0.5) / 0.15  # spz stores SH0-style coefficients
    gs.opacities = (1 / (1 + np.exp(-g["opacity"]))).astype(np.float32).reshape(n)
    ok = spz.saveSplatToPath(gs, path)
    return n if ok else 0


def export_gaussians_from_mesh_with_voxel(mesh_out, path_stem: str,
                                          condensed: bool = True):
    """Convenience wrapper taking a TRELLIS.2 MeshWithVoxel object.

    Writes a condensed 3DGS file to `<path_stem>.spz` (preferred, ~10x smaller
    than PLY) or `<path_stem>.splat` (32 B/splat fallback), always to a
    `<final>.tmp` name so the caller can atomically promote it. The raw
    float32 PLY (248 B/splat) is only written when condensed=False.

    Returns (final_path: str, gaussian_count: int).
    """
    g = voxels_to_gaussians(
        coords=mesh_out.coords.cpu().numpy(),
        attrs=mesh_out.attrs.cpu().numpy(),
        origin=mesh_out.origin.cpu().numpy(),
        voxel_size=mesh_out.voxel_size,
        layout=mesh_out.layout,
    )
    if not condensed:
        return write_ply(path_stem + ".ply", g), len(g["x"])
    spz_path = path_stem + ".spz"
    n = write_spz(spz_path + ".tmp", g)
    if n:
        return spz_path, n
    splat_path = path_stem + ".splat"
    return splat_path, write_splat(splat_path + ".tmp", g)
