"""Build an illustrative radial cargo-drone guide; never a certified CAD asset.

The selected recipe must explicitly choose the flat eight-boom layout. The
render and part inventory guide native Qwen editing; they are not Qwen output.
No models, network traffic or production generation are involved.
"""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import content_policy
from generation_prompts import digest
from pipeline_utils import durable_replace, file_sha256, write_json
from prompt_utils import select_manifest_items


def extruded_outline(outline, height):
    """Closed cap-and-wall mesh without optional polygon/boolean packages."""
    import numpy as np
    import trimesh

    outline = np.asarray(outline, dtype=float)
    n = len(outline)
    vertices = np.vstack([np.column_stack([outline, np.full(n, -height / 2)]),
                          np.column_stack([outline, np.full(n, height / 2)]),
                          [*outline.mean(0), -height / 2], [*outline.mean(0), height / 2]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(2 * n, j, i), (2 * n + 1, n + i, n + j),
                  (i, j, n + j), (i, n + j, n + i)]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def build_parts(specification):
    import numpy as np
    import trimesh

    if (specification.get("profile") != "cargo_flat_octocopter"
            or specification.get("rotor_count") != 8 or specification.get("arm_count") != 8
            or specification.get("rotor_layout") != "radial"):
        raise ValueError("Guide requires an explicit flat eight-boom cargo recipe")
    parts = []

    def add(name, mesh, colour):
        parts.append((name, mesh, colour))

    def box(name, size, centre, colour):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(centre)
        add(name, mesh, colour)

    def rod(name, start, end, radius, colour):
        a, b = np.asarray(start), np.asarray(end)
        mesh = trimesh.creation.cylinder(radius=radius, height=float(np.linalg.norm(b - a)), sections=24)
        mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], b - a))
        mesh.apply_translation((a + b) / 2)
        add(name, mesh, colour)

    def bevel_box(name, size, centre, corner, colour):
        hx, hy = size[0] / 2, size[1] / 2
        outline = [(-hx + corner, -hy), (hx - corner, -hy), (hx, -hy + corner), (hx, hy - corner),
                   (hx - corner, hy), (-hx + corner, hy), (-hx, hy - corner), (-hx, -hy + corner)]
        mesh = extruded_outline(outline, size[2])
        mesh.apply_translation(centre)
        add(name, mesh, colour)

    dark, metal, pod = (0.12, 0.15, 0.18, 1), (0.62, 0.65, 0.68, 1), (0.86, 0.86, 0.82, 1)
    bevel_box("central_frame", (0.46, 0.42, 0.12), (0, 0, 0.30), 0.055, dark)
    bevel_box("lower_frame_plate", (0.47, 0.43, 0.018), (0, 0, 0.245), 0.055, metal)
    bevel_box("cargo_pod", (0.36, 0.34, 0.28), (0, 0, 0.06), 0.025, pod)
    bevel_box("cargo_lid_rim", (0.38, 0.36, 0.018), (0, 0, 0.197), 0.025, metal)
    box("cargo_mount", (0.26, 0.24, 0.04), (0, 0, 0.22), dark)
    for x in (-0.10, 0.10):
        box(f"cargo_latch_{x}", (0.025, 0.014, 0.055), (x, -0.175, 0.11), dark)
        bevel_box(f"battery_{x}", (0.15, 0.29, 0.06), (x, 0, 0.39), 0.014, dark)
    for y in (-0.075, 0.075):
        box(f"battery_strap_{y}", (0.36, 0.018, 0.008), (0, y, 0.42), metal)
    blade_outline = [(0.015, -0.007), (0.06, -0.015), (0.18, -0.020), (0.29, -0.008),
                     (0.316, 0.001), (0.306, 0.010), (0.22, 0.024), (0.10, 0.025), (0.020, 0.009)]
    motors = []
    for i in range(8):
        theta = 2 * math.pi * i / 8
        direction = np.array([math.cos(theta), math.sin(theta), 0.0])
        centre = 1.04 * direction
        rod(f"boom_{i}", 0.18 * direction + [0, 0, 0.30], centre + [0, 0, 0.30], 0.023, dark)
        rod(f"boom_collar_{i}", 0.24 * direction + [0, 0, 0.30], 0.31 * direction + [0, 0, 0.30], 0.030, metal)
        rod(f"motor_{i}", centre + [0, 0, 0.30], centre + [0, 0, 0.39], 0.061, dark)
        rod(f"motor_base_{i}", centre + [0, 0, 0.30], centre + [0, 0, 0.315], 0.067, metal)
        rod(f"motor_cap_{i}", centre + [0, 0, 0.385], centre + [0, 0, 0.405], 0.061, metal)
        rod(f"hub_{i}", centre + [0, 0, 0.40], centre + [0, 0, 0.45], 0.020, dark)
        blade = extruded_outline(blade_outline, 0.009)
        opposite = blade.copy()
        opposite.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1]))
        propeller = trimesh.util.concatenate([blade, opposite])
        propeller.apply_transform(trimesh.transformations.rotation_matrix(theta + 0.4, [0, 0, 1]))
        propeller.apply_translation(centre + [0, 0, 0.44])
        add(f"propeller_{i}", propeller, dark)
        motors.append({"index": i, "centre": list(centre + [0, 0, 0.345]), "shaft_axis": [0, 0, 1],
                       "hub_centre": list(centre + [0, 0, 0.44])})
    for side in (-1, 1):
        x = side * 0.27
        for y in (-0.19, 0.19):
            rod(f"leg_{side}_{y}", (side * 0.18, y, 0.27), (x, y, -0.17), 0.016, dark)
        rod(f"skid_{side}", (x, -0.43, -0.17), (x, 0.43, -0.17), 0.019, dark)
    return parts, motors


def render_guide(parts, size=1024):
    import moderngl
    import numpy as np
    from PIL import Image
    from review_assets import VERTEX_SHADER, FRAGMENT_SHADER, camera_matrix

    vertices = np.vstack([mesh.vertices for _name, mesh, _colour in parts])
    centre = (vertices.min(0) + vertices.max(0)) / 2
    distance = float(np.linalg.norm(vertices - centre, axis=1).max()) * math.sqrt(10) * 1.3
    azimuth, elevation = math.radians(25), math.radians(65)
    eye = centre + distance * np.array([math.sin(azimuth) * math.cos(elevation),
                                      -math.cos(azimuth) * math.cos(elevation), math.sin(elevation)])
    forward = (centre - eye) / np.linalg.norm(centre - eye)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    with moderngl.create_standalone_context(require=330) as ctx:
        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        framebuffer = ctx.simple_framebuffer((size, size), components=4)
        framebuffer.use()
        framebuffer.clear(0, 0, 0, 0, depth=1)
        ctx.enable(moderngl.DEPTH_TEST)
        program["view_projection"].write(camera_matrix(eye, right, up, forward, distance).T.astype("f4").tobytes())
        program["eye"].value = tuple(eye)
        program["clay"].value = False
        texture = ctx.texture((1, 1), 3, bytes((255, 255, 255)))
        texture.use(0)
        for _name, mesh, colour in parts:
            data = np.column_stack([mesh.vertices, mesh.vertex_normals, np.zeros((len(mesh.vertices), 2))])
            buffer = ctx.buffer(data.astype("f4").tobytes())
            indices = ctx.buffer(np.asarray(mesh.faces, dtype="u4").tobytes())
            vao = ctx.vertex_array(program, [(buffer, "3f 3f 2f", "position", "normal", "texcoord")],
                                   indices, index_element_size=4)
            program["color_factor"].value = colour
            vao.render()
            vao.release()
            indices.release()
            buffer.release()
        result = Image.frombytes("RGBA", (size, size), framebuffer.read(components=4, alignment=1))
        result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        renderer = ctx.info["GL_RENDERER"]
    return result, renderer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", required=True, help="one default-policy manifest item")
    parser.add_argument("--output", type=Path, required=True, help="new PNG path; JSON and illustrative GLB are adjacent")
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a new .png path")
    if any(args.output.with_suffix(ext).exists() for ext in (".png", ".json", ".glb")):
        parser.error("guide outputs already exist; choose a new path")
    rows, _refused = content_policy.screen_items(select_manifest_items(ids=[args.id]), False, False)
    if len(rows) != 1:
        parser.error("guide builder requires one default-policy eligible item")
    item = rows[0]
    specification = item.get("generation_prompt", {}).get("specifications", {}).get("prompt", {})
    parts, motors = build_parts(specification)
    image, renderer = render_guide(parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import trimesh
    scene = trimesh.Scene()
    native_to_glb = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for name, mesh, colour in parts:
        mesh = mesh.copy()
        mesh.apply_transform(native_to_glb)
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=colour))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    glb = args.output.with_suffix(".glb")
    temporary = glb.with_suffix(".tmp.glb")
    temporary.write_bytes(scene.export(file_type="glb"))
    durable_replace(temporary, glb)
    temporary = args.output.with_suffix(".tmp.png")
    image.save(temporary)
    durable_replace(temporary, args.output)
    write_json(args.output.with_suffix(".json"), {
        "schema_version": 1, "construction_version": 2, "kind": "authored_geometry_guide", "id": item["id"],
        "at": datetime.now(timezone.utc).isoformat(), "image_sha256": file_sha256(args.output),
        "guide_mesh_sha256": file_sha256(glb), "generator_sha256": file_sha256(__file__),
        "manifest_prompt_sha256": digest(item["prompt"]), "policy": item["policy"],
        "specification": specification, "motors": motors, "part_names": [name for name, _mesh, _colour in parts],
        "scale": "illustrative normalized units; no metric, collision or engineering certification",
        "coordinates": "motor inventory Z-up; illustrative GLB Y-up", "renderer": renderer,
        "camera": {"azimuth_degrees": 25, "elevation_degrees": 65, "image_size": list(image.size)}})
    print(f"Guide committed: {args.output}; {len(motors)} motor positions; illustrative geometry only")


if __name__ == "__main__":
    main()
