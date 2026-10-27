"""Render existing GLBs for visual review without loading generation models.

Uses the complete exported mesh and its base-color texture, a depth buffer and
diffuse studio lighting. These are review images, not additional QA verdicts or
a full metallic/roughness PBR simulation. Requires moderngl in the mesh runtime.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_utils import file_sha256, write_json
from qa import _turntable_cameras, write_view_montage


VERTEX_SHADER = """#version 330
uniform mat4 view_projection;
in vec3 position;
in vec3 normal;
in vec2 texcoord;
out vec3 world_position;
out vec3 world_normal;
out vec2 uv;
void main() {
    world_position = position;
    world_normal = normal;
    uv = texcoord;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""
FRAGMENT_SHADER = """#version 330
uniform sampler2D base_color;
uniform vec4 color_factor;
uniform vec3 eye;
uniform bool clay;
in vec3 world_position;
in vec3 world_normal;
in vec2 uv;
out vec4 color;
void main() {
    vec3 albedo = clay ? vec3(0.65) : texture(base_color, uv).rgb * color_factor.rgb;
    vec3 n = normalize(world_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 light = normalize(eye - world_position);
    float diffuse = 0.45 + 0.55 * max(dot(n, light), 0.0);
    color = vec4(pow(pow(albedo, vec3(2.2)) * diffuse, vec3(1.0 / 2.2)), 1.0);
}
"""


def camera_matrix(cam, right, up, forward, radius):
    """Pinhole projection matching the QA orbit, for a square viewport."""
    view = np.eye(4)
    view[:3, :3] = np.array([right, up, -forward])
    view[:3, 3] = -view[:3, :3] @ cam
    near, far = 0.01 * radius, 10 * radius
    projection = np.zeros((4, 4))
    projection[0, 0] = projection[1, 1] = 3.0
    projection[2, 2] = -(far + near) / (far - near)
    projection[2, 3] = -2 * far * near / (far - near)
    projection[3, 2] = -1
    return projection @ view


def render_glb(path, *, size=512):
    """Return textured and clay views of every transformed scene instance."""
    import moderngl
    import trimesh

    scene = trimesh.load(path, force="scene")
    geoms = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh.apply_transform(transform)
        # GLB Y-up -> the shared native Z-up camera rig.
        mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                       [0, 1, 0, 0], [0, 0, 0, 1]]))
        geoms.append(mesh)
    if not geoms:
        raise ValueError(f"No triangular geometry in {path}")
    vertices = np.vstack([g.vertices for g in geoms])
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    # The QA rig was designed for silhouette statistics. Review must include
    # every tip even for asymmetric objects; fit a sphere to the actual FOV.
    radius = float(np.linalg.norm(vertices - center, axis=1).max()) * np.sqrt(10) * 1.08
    with moderngl.create_standalone_context(require=330) as ctx:
        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        framebuffer = ctx.simple_framebuffer((size, size), components=4)
        framebuffer.use()
        ctx.enable(moderngl.DEPTH_TEST)
        draws = []
        for mesh in geoms:
            uv = getattr(mesh.visual, "uv", None)
            if uv is None:
                uv = np.zeros((len(mesh.vertices), 2))
            material = getattr(mesh.visual, "material", None)
            texture = getattr(material, "baseColorTexture", None)
            if texture is None:
                texture = getattr(material, "image", None)
            if texture is None:
                texture = Image.new("RGB", (1, 1), (210, 210, 210))
            texture = texture.convert("RGB")
            # OpenGL's first uploaded row is the bottom of the texture.
            pixels = np.asarray(texture)[::-1].copy()
            gpu_texture = ctx.texture(texture.size, 3, pixels.tobytes(), alignment=1)
            gpu_texture.build_mipmaps()
            gpu_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            factor = getattr(material, "baseColorFactor", None)
            factor = np.ones(4) if factor is None else np.asarray(factor) / 255.0
            buffer = ctx.buffer(np.column_stack([mesh.vertices, mesh.vertex_normals, uv])
                                .astype("f4").tobytes())
            indices = ctx.buffer(np.asarray(mesh.faces, dtype="u4").tobytes())
            vao = ctx.vertex_array(program, [(buffer, "3f 3f 2f", "position", "normal", "texcoord")],
                                   indices, index_element_size=4)
            draws.append((vao, gpu_texture, tuple(factor)))
        textured, clay = [], []
        for cam, right, up, forward in _turntable_cameras(center, radius):
            matrix = camera_matrix(cam, right, up, forward, radius)
            program["view_projection"].write(matrix.T.astype("f4").tobytes())
            program["eye"].value = tuple(cam)
            for is_clay, output in [(False, textured), (True, clay)]:
                framebuffer.clear(0.10, 0.12, 0.15, 1, depth=1)
                program["clay"].value = is_clay
                for vao, texture, factor in draws:
                    texture.use(0)
                    program["color_factor"].value = factor
                    vao.render()
                output.append(np.asarray(Image.frombytes("RGB", (size, size),
                    framebuffer.read(components=3, alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)))
        renderer = ctx.info["GL_RENDERER"]
    return textured, clay, {"renderer": renderer, "vertices": sum(len(g.vertices) for g in geoms),
                             "triangles": sum(len(g.faces) for g in geoms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("glbs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size < 32 or args.size > 2048:
        parser.error("review size must be between 32 and 2048 pixels per view")
    if len({path.stem for path in args.glbs}) != len(args.glbs):
        parser.error("use separate review directories for GLBs with the same item ID")
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for path in args.glbs:
        textured, clay, info = render_glb(path, size=args.size)
        for mode, views in [("textured", textured), ("clay", clay)]:
            destination = args.output / f"{path.stem}.{mode}.png"
            write_view_montage(views, destination, f"{path.stem} | {mode} | full GLB, depth tested",
                               args.size)
        record = {"path": str(path.resolve()), "sha256": file_sha256(path), **info}
        reference = path.with_suffix(".png")
        if reference.is_file():
            with Image.open(reference) as original:
                rgba = original.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (225, 225, 225, 255))
                Image.alpha_composite(background, rgba).convert("RGB").save(
                    args.output / f"{path.stem}.reference.png")
            record["reference"] = {"path": str(reference.resolve()), "sha256": file_sha256(reference)}
        report.append(record)
        print(f"{path.stem}: rendered {info['triangles']:,} triangles from 10 views", flush=True)
    write_json(args.output / "rendered_assets.json", {
        "at": datetime.now(timezone.utc).isoformat(), "size": args.size,
        "method": "full GLB, depth buffer, base color and diffuse lighting; no PBR or QA verdict",
        "items": report})


if __name__ == "__main__":
    main()
