"""Reference-anchored emblem transfer must be a no-op without emblems and
lossless to geometry when it fires: the GLB rewrite may only swap the texture
bytes, never the mesh, and detection must be deterministic.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local"))


class TestEmblemBlobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"emblem tests require {name}")
        import emblem_project
        cls.mod = emblem_project

    def blank(self, size=256, color=(200, 200, 200)):
        from PIL import Image
        return Image.new("RGBA", (size, size), color + (255,))

    def test_plain_reference_is_noop(self):
        mask, records = self.mod.emblem_blobs(self.blank())
        self.assertEqual(records, [])
        self.assertFalse(mask.any())

    def test_red_cross_is_detected(self):
        from PIL import Image, ImageDraw
        img = self.blank()
        d = ImageDraw.Draw(img)
        d.ellipse([78, 78, 178, 178], fill=(255, 255, 255, 255))
        d.rectangle([118, 92, 138, 164], fill=(220, 20, 20, 255))
        d.rectangle([92, 118, 164, 138], fill=(220, 20, 20, 255))
        mask, records = self.mod.emblem_blobs(img)
        saturated = [r for r in records if r["kind"] == "saturated"]
        self.assertTrue(saturated, "a red cross on white must trip the gate")
        cross = max(saturated, key=lambda r: r["pixels"])
        x0, y0, x1, y1 = cross["bbox"]
        self.assertLess(x0, 128, "blob must span the cross centre")
        self.assertGreater(x1, 128, "blob must span the cross centre")
        self.assertLess(y0, 128, "blob must span the cross centre")
        self.assertGreater(y1, 128, "blob must span the cross centre")

    def test_body_paint_is_not_an_emblem(self):
        from PIL import Image
        img = self.blank(color=(220, 30, 30))
        mask, records = self.mod.emblem_blobs(img)
        self.assertEqual(records, [])

    def test_detection_is_deterministic(self):
        from PIL import Image, ImageDraw
        img = self.blank()
        d = ImageDraw.Draw(img)
        d.rectangle([100, 100, 140, 140], fill=(200, 10, 10, 255))
        first = self.mod.emblem_blobs(img)[1]
        second = self.mod.emblem_blobs(img)[1]
        self.assertEqual(first, second)


class TestGlbSurgery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL", "trimesh"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"GLB surgery tests require {name}")
        import emblem_project
        cls.mod = emblem_project

    def make_glb(self, directory):
        import numpy as np
        import trimesh
        from PIL import Image
        box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        uv = np.asarray(box.vertices[:, :2])
        uv = (uv - uv.min(axis=0)) / (uv.max(axis=0) - uv.min(axis=0) + 1e-9)
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            baseColorTexture=Image.new("RGB", (64, 64), (200, 30, 30)))
        box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        path = Path(directory) / "box.glb"
        box.scene().export(path)
        return path

    def test_texture_swap_preserves_geometry(self):
        import io
        import json
        import struct
        import trimesh
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_glb(tmp)
            before = trimesh.load(path, force="mesh")
            green = io.BytesIO()
            Image.new("RGB", (32, 32), (20, 200, 20)).save(green, format="PNG")
            out = self.mod.replace_glb_texture(path, 0, green.getvalue())
            path.write_bytes(out)
            after = trimesh.load(path, force="mesh")
            self.assertTrue(bool((np_vertices(before) == np_vertices(after)).all()))
            self.assertEqual(np_faces(before).tolist(), np_faces(after).tolist())
            ln = struct.unpack("<I", out[12:16])[0]
            js = json.loads(out[20:20 + ln].decode("utf-8"))
            self.assertEqual(js["buffers"][0]["byteLength"] % 4, 0)
            tex = getattr(after.visual.material, "baseColorTexture")
            px = tex.convert("RGB").getpixel((4, 4))
            self.assertGreater(px[1], 150)

    def test_texture_index_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_glb(tmp)
            with self.assertRaises(ValueError):
                self.mod.replace_glb_texture(path, 7, b"PNG")


def np_vertices(mesh):
    import numpy as np
    return np.asarray(mesh.vertices)


def np_faces(mesh):
    import numpy as np
    return np.asarray(mesh.faces)


class TestFrameMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("numpy") is None:
            raise unittest.SkipTest("frame map tests require numpy")
        import emblem_project
        cls.mod = emblem_project

    def test_identity_frames(self):
        import numpy as np
        ren = np.zeros((100, 100), bool)
        ren[20:80, 20:80] = True
        ref = np.zeros((100, 100), bool)
        ref[20:80, 20:80] = True
        scale, ox, oy = self.mod._frame_map(ren, np.nonzero(ref), ref.shape, 100)
        self.assertAlmostEqual(scale, 1.0, places=6)
        self.assertAlmostEqual(ox, 0.0, places=6)
        self.assertAlmostEqual(oy, 0.0, places=6)

    def test_scaled_frames(self):
        import numpy as np
        ren = np.zeros((100, 100), bool)
        ren[20:80, 20:80] = True
        ref = np.zeros((200, 200), bool)
        ref[40:160, 40:160] = True
        scale, ox, oy = self.mod._frame_map(ren, np.nonzero(ref), ref.shape, 100)
        self.assertAlmostEqual(scale, 2.0, places=6)


class TestVerifyTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"verify tests require {name}")
        import emblem_project
        cls.mod = emblem_project

    def test_verify_uses_threshold(self):
        import numpy as np
        from PIL import Image
        tex = Image.new("RGB", (32, 32), (200, 30, 30))
        ref = Image.new("RGBA", (32, 32), (200, 200, 200, 255))
        mask = np.zeros((32, 32), bool)
        mask[10:20, 10:20] = True
        ok_report = {"projected_texels": 80}
        v = self.mod.verify_transfer(tex, ref, mask, ok_report)
        self.assertTrue(v["ok"])
        self.assertGreaterEqual(v["match_ratio"], 0.5)
        bad = self.mod.verify_transfer(tex, ref, mask, {"projected_texels": 0})
        self.assertFalse(bad["ok"])

    def test_face_id_offset_decodes_background(self):
        import numpy as np
        # enc = fid + 1; background 0 decodes to -1, face 0 to 0.
        raw = np.array([[[0, 0, 0], [0, 0, 1], [1, 0, 0]]], dtype=np.uint8)
        face_of = ((raw[:, :, 0].astype(int) << 16) |
                   (raw[:, :, 1].astype(int) << 8) |
                   raw[:, :, 2].astype(int)) - 1
        self.assertEqual(face_of[0, 0], -1)
        self.assertEqual(face_of[0, 1], 0)

    def test_closest_t_wins_dedup(self):
        import numpy as np
        tu = np.array([5, 5])
        tv = np.array([7, 7])
        t = np.array([3.0, 1.0])
        tw = 16
        order = np.lexsort((t, tv * tw + tu))
        seen, keep = set(), []
        for idx in order:
            key = (int(tv[idx]), int(tu[idx]))
            if key in seen:
                continue
            seen.add(key)
            keep.append(idx)
        self.assertEqual(keep, [1])


if __name__ == "__main__":
    unittest.main()
