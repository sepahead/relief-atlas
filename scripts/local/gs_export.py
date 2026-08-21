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
    n = len(g["x"])
    props = ["x", "y", "z"]
    props += [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(45)]
    props += ["opacity"]
    props += [f"scale_{i}" for i in range(3)]
    props += [f"rot_{i}" for i in range(4)]

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


def export_gaussians_from_mesh_with_voxel(mesh_out, path: str) -> int:
    """Convenience wrapper taking a TRELLIS.2 MeshWithVoxel object."""
    g = voxels_to_gaussians(
        coords=mesh_out.coords.cpu().numpy(),
        attrs=mesh_out.attrs.cpu().numpy(),
        origin=mesh_out.origin.cpu().numpy(),
        voxel_size=mesh_out.voxel_size,
        layout=mesh_out.layout,
    )
    return write_ply(path, g)
