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

    def test_dark_window_is_not_an_emblem(self):
        """Door openings and window blocks are dark but bigger than lettering."""
        from PIL import ImageDraw
        img = self.blank()
        d = ImageDraw.Draw(img)
        # ~5.5% of the opaque area: a tent door / truck window block.
        d.rectangle([88, 88, 168, 168], fill=(30, 30, 32, 255))
        mask, records = self.mod.emblem_blobs(img)
        self.assertEqual(records, [], "a large dark opening must not be a decal")
        self.assertFalse(mask.any())

    def test_dark_lettering_is_an_emblem(self):
        from PIL import ImageDraw
        img = self.blank(color=(230, 230, 225))
        d = ImageDraw.Draw(img)
        # A two-glyph unit code ("H I") drawn with 4 px strokes.
        d.rectangle([110, 116, 113, 134], fill=(25, 25, 28, 255))
        d.rectangle([122, 116, 125, 134], fill=(25, 25, 28, 255))
        d.rectangle([110, 124, 125, 127], fill=(25, 25, 28, 255))
        d.rectangle([134, 116, 137, 134], fill=(25, 25, 28, 255))
        _, records = self.mod.emblem_blobs(img)
        self.assertTrue([r for r in records if r["kind"] == "dark"],
                        "small dark unit codes must still trip the gate")

    def test_small_solid_window_is_not_an_emblem(self):
        """A window under the area cap is still solid, not a stroke."""
        from PIL import ImageDraw
        img = self.blank(size=512)
        d = ImageDraw.Draw(img)
        # ~0.7% of the silhouette, like each side window of a DRK tent.
        d.rectangle([300, 200, 322, 280], fill=(35, 38, 40, 255))
        mask, records = self.mod.emblem_blobs(img)
        self.assertEqual([r for r in records if r["kind"] == "dark"], [],
                         "a solid dark window must not be projected as a decal")

    def test_inscribed_radius_measures_stroke(self):
        import numpy as np
        bar = np.zeros((40, 40), bool)
        bar[10:30, 18:22] = True
        self.assertLessEqual(self.mod.max_inscribed_radius(bar), 2.5)
        block = np.zeros((40, 40), bool)
        block[5:35, 5:35] = True
        self.assertGreaterEqual(self.mod.max_inscribed_radius(block), 14.0)

    def test_detection_is_deterministic(self):
        from PIL import Image, ImageDraw
        img = self.blank()
        d = ImageDraw.Draw(img)
        d.rectangle([100, 100, 140, 140], fill=(200, 10, 10, 255))
        first = self.mod.emblem_blobs(img)[1]
        second = self.mod.emblem_blobs(img)[1]
        self.assertEqual(first, second)

    def test_diagonal_body_stripe_is_rejected(self):
        """A long livery stripe has low bbox fill / large span and must not be an emblem."""
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([60, 140, 452, 380], fill=(235, 235, 235, 255))
        # Diagonal stripe: square-ish bbox (180x180) but low fill (< 0.22).
        d.line([(120, 180), (300, 360)], fill=(220, 20, 20, 255), width=12)
        _mask, records = self.mod.emblem_blobs(img)
        self.assertEqual(records, [], "diagonal vehicle livery stripes must not trip blob gate")

    def test_silhouette_rim_lightbar_is_rejected(self):
        """Saturated lamps touching the silhouette boundary must be rejected."""
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([100, 160, 412, 380], fill=(235, 235, 235, 255))
        # Blue roof lightbar flush with top silhouette edge y=160..176.
        d.rectangle([220, 160, 265, 178], fill=(20, 60, 230, 255))
        _mask, records = self.mod.emblem_blobs(img)
        self.assertEqual(records, [], "roof edge lightbars must not be projected as decals")


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


class TestDecalFootprint(unittest.TestCase):
    """The projection may only write inside the emblem footprint."""

    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"decal tests require {name}")
        import emblem_project
        cls.mod = emblem_project

    def test_footprint_is_confined_to_the_blob(self):
        import numpy as np
        blob = np.zeros((128, 128), bool)
        blob[60:68, 60:68] = True
        w = self.mod.decal_weights(blob)
        self.assertGreater(w[64, 64], 0.9)
        self.assertEqual(float(w[:30, :30].max()), 0.0)
        self.assertEqual(float(w[100:, 100:].max()), 0.0)
        # The blend band exists but stays local to the decal.
        self.assertGreater(float(w[56:72, 56:72].max()), 0.0)

    def test_empty_mask_has_no_footprint(self):
        import numpy as np
        self.assertFalse(self.mod.decal_weights(np.zeros((16, 16), bool)).any())


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

    def test_writeback_metric_decides(self):
        from PIL import Image
        tex = Image.new("RGB", (32, 32), (200, 30, 30))
        ref = Image.new("RGBA", (32, 32), (200, 200, 200, 255))
        import numpy as np
        mask = np.zeros((32, 32), bool)
        mask[10:20, 10:20] = True
        good = self.mod.verify_transfer(tex, ref, mask, {
            "projected_texels": 4, "writeback_match": 0.97})
        self.assertTrue(good["ok"])
        bad = self.mod.verify_transfer(tex, ref, mask, {
            "projected_texels": 4, "writeback_match": 0.10})
        self.assertFalse(bad["ok"], "a low write-back match must refuse the write")

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


class TestSharedUvTexels(unittest.TestCase):
    """A texel reused by a distant panel must not receive the decal."""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("numpy") is None:
            raise unittest.SkipTest("shared-texel tests require numpy")
        import emblem_project
        cls.mod = emblem_project

    @staticmethod
    def two_panels(z_gap=5.0):
        """Two unit quads five world units apart sharing one UV rect."""
        import numpy as np
        V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [0, 0, z_gap], [1, 0, z_gap], [1, 1, z_gap], [0, 1, z_gap]], float)
        F = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]] * 2, float)
        return V, F, UV

    def test_duplicated_island_is_refused(self):
        import numpy as np
        V, F, UV = self.two_panels()
        index = self.mod.uv_face_index(F, UV, grid=8)
        conflict, pairs = self.mod.shared_surface_texels(
            index, F, V, UV, np.array([5]), np.array([5]), 8, 8,
            np.array([[0.6875, 0.3125, 0.0]]), 0.01)
        self.assertGreater(pairs, 0)
        self.assertTrue(bool(conflict[0]),
                        "a texel shared with a panel 5 units away must be skipped")

    def test_single_surface_is_not_a_conflict(self):
        import numpy as np
        V, F, UV = self.two_panels()
        F1 = F[:2]
        index = self.mod.uv_face_index(F1, UV, grid=8)
        conflict, _ = self.mod.shared_surface_texels(
            index, F1, V, UV, np.array([5]), np.array([5]), 8, 8,
            np.array([[0.6875, 0.3125, 0.0]]), 0.01)
        self.assertFalse(bool(conflict[0]),
                         "adjacent triangles of one continuous surface must pass")

    def test_texel_outside_any_face_is_not_flagged(self):
        import numpy as np
        V, F, UV = self.two_panels()
        F1, UV1 = F[:2], UV * 0.2
        index = self.mod.uv_face_index(F1, UV1, grid=8)
        conflict, pairs = self.mod.shared_surface_texels(
            index, F1, V, UV1, np.array([7]), np.array([1]), 8, 8,
            np.array([[0.5, 0.5, 0.0]]), 0.01)
        self.assertEqual(int(pairs), 0)
        self.assertFalse(bool(conflict[0]))


class TestPoseSelection(unittest.TestCase):
    """Mirrored/antipodal silhouettes must be resolved or refused, never guessed."""

    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"pose tests require {name}")
        import emblem_project
        cls.mod = emblem_project

    @staticmethod
    def cand(az, el, iou, ncc=None):
        import math
        c = {"az": math.radians(az), "el": math.radians(el), "iou": iou}
        if ncc is not None:
            c["ncc"] = ncc
        return c

    def test_appearance_overrides_a_better_mirrored_silhouette(self):
        # Shape of the ger_drk_0092 failure: the mirror had the higher IoU.
        cands = [self.cand(318, 4, 0.94, 0.23), self.cand(36, 4, 0.93, 0.68),
                 self.cand(210, 4, 0.92, 0.04)]
        winner, runner, ambiguous, basis = self.mod.select_pose(cands)
        self.assertEqual(basis, "appearance")
        self.assertAlmostEqual(winner["az"], cands[1]["az"])
        self.assertAlmostEqual(runner["az"], cands[0]["az"])
        self.assertFalse(ambiguous)

    def test_close_appearance_scores_are_ambiguous(self):
        cands = [self.cand(36, 4, 0.93, 0.50), self.cand(216, 4, 0.93, 0.48)]
        self.assertTrue(self.mod.select_pose(cands)[2])

    def test_weak_appearance_is_ambiguous_even_without_rival(self):
        self.assertTrue(self.mod.select_pose([self.cand(36, 4, 0.93, 0.05)])[2])

    def test_neighbouring_pose_is_not_a_rival(self):
        cands = [self.cand(36, 4, 0.93, 0.60), self.cand(45, 4, 0.92, 0.59)]
        winner, runner, ambiguous, _ = self.mod.select_pose(cands)
        self.assertIsNone(runner)
        self.assertFalse(ambiguous)

    def test_silhouette_only_tie_is_ambiguous(self):
        cands = [self.cand(36, 4, 0.930), self.cand(318, 4, 0.925)]
        winner, runner, ambiguous, basis = self.mod.select_pose(cands)
        self.assertEqual(basis, "silhouette")
        self.assertTrue(ambiguous)

    def test_rejection_reports_ambiguity(self):
        fit = {"iou": 0.9, "ambiguous": True, "ambiguity_basis": "appearance",
               "appearance_ncc": 0.5, "runner_up_ncc": 0.48, "runner_up_azimuth_deg": 216.0}
        self.assertIn("ambiguous", self.mod.pose_rejection(fit))
        self.assertIn("too poor", self.mod.pose_rejection({"iou": 0.2}))
        self.assertIsNone(self.mod.pose_rejection({"iou": 0.9, "ambiguous": False}))

    def test_luminance_ncc_detects_mirrored_layout(self):
        import numpy as np
        img = np.full((64, 64, 3), 230.0, np.float32)
        img[20:40, 5:20] = 30.0  # door on the left
        mask = np.ones((64, 64), bool)
        self.assertGreater(self.mod.luminance_ncc(img, mask, img, mask), 0.99)
        mirrored = img[:, ::-1]
        self.assertLess(self.mod.luminance_ncc(img, mask, mirrored, mask), 0.0)

    def test_pose_separation(self):
        import math
        self.assertAlmostEqual(self.mod.pose_separation_deg(
            (math.radians(10), 0.0), (math.radians(350), 0.0)), 20.0, places=5)

    def test_cavity_ao_darkens_open_doorway_without_dimming_outer_wall(self):
        import numpy as np
        raw = np.zeros((128, 128, 4), dtype=np.float32)
        # Flat exterior wall at camera depth z = 2.0 with bright white base colour.
        raw[16:112, 16:112, :3] = 0.90
        raw[16:112, 16:112, 3] = 2.0
        # Narrow open doorway at x=56..72 seeing an interior wall at z = 2.6.
        raw[40:108, 56:72, 3] = 2.6
        rgb, fg = self.mod._cavity_ao_rgb(raw, obj_span=1.0)
        self.assertTrue(fg[64, 32])
        self.assertGreater(int(rgb[64, 32, 0]), 220, "exterior wall must remain unattenuated")
        self.assertLess(int(rgb[64, 64, 0]), 80, "recessed doorway interior must receive cavity AO")

    def test_rgba_basecolor_alpha_is_preserved(self):
        import numpy as np
        from PIL import Image
        alpha = np.full((16, 16), 255, dtype=np.uint8)
        alpha[4:8, 4:8] = 96
        rgb = np.full((16, 16, 3), 180.0, dtype=np.float32)
        packed = self.mod._pack_output_texture(rgb, alpha)
        self.assertEqual(packed.mode, "RGBA")
        self.assertEqual(int(np.asarray(packed)[6, 6, 3]), 96)
        self.assertEqual(int(np.asarray(packed)[0, 0, 3]), 255)


class TestQaEmblemAndTrellisClean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"pipeline regression tests require {name}")

    def test_qa_rejects_legacy_emblem_schema_and_hash_mismatch(self):
        import qa
        from pipeline_utils import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            glb = d / "item.glb"
            glb.write_bytes(b"glb-bytes")
            h = file_sha256(glb)
            legacy = {"emblem_project": {"applied": True, "schema": 2, "glb_sha256": h,
                                         "verification": {"ok": True, "multiview_checked": True}}}
            self.assertTrue(any("legacy emblem_project schema" in r
                                for r in qa.emblem_qa_reasons(d, "item", legacy)))
            valid = {"emblem_project": {
                "applied": True,
                "schema": 4,
                "glb_sha256": h,
                "pose": {"ambiguous": False},
                "verification": {"ok": True, "multiview_checked": True},
            }}
            self.assertEqual(qa.emblem_qa_reasons(d, "item", valid), [])
            glb.write_bytes(b"mutated-glb")
            self.assertIn("emblem_project glb_sha256 mismatch", qa.emblem_qa_reasons(d, "item", valid))

    def test_qa_catches_gaussian_count_mismatch(self):
        import qa
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "item.spz").write_bytes(b"spz")
            meta = {"gaussians": 728813}
            reasons = qa.gaussian_metadata_reasons(meta, {"gaussians": 711469}, d, "item")
            self.assertTrue(any("count mismatch" in r for r in reasons))
            self.assertEqual(qa.gaussian_metadata_reasons(meta, {"gaussians": 728813}, d, "item"), [])

    def test_hole_safe_grid_sampler_fills_zero_occupancy_queries(self):
        for name in ("torch", "scipy"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"hole-safe sampler test requires {name}")
        import torch
        import trellis_meshgen as tm

        feats = torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float32)
        coords = torch.tensor([[0, 2, 2, 2]], dtype=torch.int32)
        grid = torch.tensor([[[2.5, 2.5, 2.5], [7.5, 7.5, 7.5]]], dtype=torch.float32)

        def fake_flex(f_occ, _coords, _shape, _grid, mode="trilinear"):
            # Active query [2.5,2.5,2.5] gets exact f_occ; far query [7.5,7.5,7.5] gets 0.
            z = torch.zeros_like(f_occ[0])
            return torch.stack([f_occ[0], z], dim=0).unsqueeze(0)

        safe_flex = tm._make_hole_safe_grid_sampler(fake_flex, has_flex=True)
        out_flex = safe_flex(feats, coords, torch.Size([1, 3, 8, 8, 8]), grid)
        self.assertEqual(tuple(out_flex.shape), (1, 2, 3))
        self.assertTrue(torch.allclose(out_flex[0, 0], feats[0], atol=1e-5))
        self.assertTrue(torch.allclose(out_flex[0, 1], feats[0], atol=1e-5))

        safe_dense = tm._make_hole_safe_grid_sampler(None, has_flex=False)
        out_dense = safe_dense(feats, coords, torch.Size([1, 3, 8, 8, 8]), grid)
        self.assertEqual(tuple(out_dense.shape), (1, 2, 3))
        self.assertTrue(torch.allclose(out_dense[0, 0], feats[0], atol=1e-5))
        self.assertTrue(torch.allclose(out_dense[0, 1], feats[0], atol=1e-5))

    def test_trellis_reference_cleaning_strips_noise_and_shadow_keeps_tyres(self):
        import numpy as np
        from PIL import Image
        import trellis_meshgen as tm

        arr = np.zeros((256, 256, 4), dtype=np.uint8)
        # Residual VAE background noise (`alpha = 4`, magenta tint).
        arr[:, :] = [165, 55, 205, 4]
        # Vehicle body (`alpha = 255`, white).
        arr[60:170, 50:206] = [235, 235, 235, 255]
        # Opaque rubber tyre (`alpha = 255`, dark neutral).
        arr[150:190, 75:115] = [22, 22, 24, 255]
        # Semi-transparent ground shadow slab (`alpha = 150`, dark neutral).
        arr[192:225, 45:210] = [10, 10, 12, 150]

        cleaned_rgba = np.asarray(tm.clean_reference_rgba(Image.fromarray(arr, "RGBA")))
        self.assertEqual(int(cleaned_rgba[10, 10, 3]), 0, "residual alpha<24 noise must be zeroed")
        self.assertEqual(int(cleaned_rgba[205, 130, 3]), 0, "ground shadow slab must be stripped")
        self.assertEqual(int(cleaned_rgba[170, 95, 3]), 255, "opaque dark tyre must be preserved")

        prepped = np.asarray(tm.clean_reference_for_trellis(Image.fromarray(arr, "RGBA")))
        self.assertEqual(prepped.shape[2], 3)
        # Outer 2px border must be pure black margin (no clipped foreground).
        self.assertEqual(int(prepped[:2, :, :].max()), 0)
        self.assertEqual(int(prepped[-2:, :, :].max()), 0)

    def test_preflight_storage_decouples_texture_size_from_3dgs(self):
        import preflight
        b_spz_1k = preflight.storage_estimate(100, "spz", "512", 1024)
        b_spz_2k = preflight.storage_estimate(100, "spz", "512", 2048)
        b_splat_1k = preflight.storage_estimate(100, "splat", "512", 1024)
        b_splat_2k = preflight.storage_estimate(100, "splat", "512", 2048)
        # Increasing 2D texture size from 1024 -> 2048 adds the exact same GLB
        # texture delta regardless of whether the 3DGS container is SPZ or SPLAT.
        self.assertEqual(b_spz_2k - b_spz_1k, b_splat_2k - b_splat_1k)

    def test_badge_patch_fills_enclosed_quadrants_absorbs_ring_and_skips_stripe(self):
        import numpy as np
        from PIL import Image, ImageDraw
        import emblem_project as ep

        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # White tent/truck panel with a dark window on the right and a long red stripe below.
        d.rectangle([20, 20, 236, 236], fill=(242, 242, 240, 255))
        d.rectangle([168, 70, 210, 150], fill=(32, 35, 38, 255))  # adjacent dark window
        d.rectangle([25, 162, 230, 172], fill=(215, 22, 22, 255))  # passing red livery stripe
        # Red cross inside a thin red circular frame at centre (110, 110).
        d.ellipse([78, 78, 142, 142], outline=(215, 22, 22, 255), width=2)
        d.rectangle([104, 88, 116, 132], fill=(215, 22, 22, 255))
        d.rectangle([88, 104, 132, 116], fill=(215, 22, 22, 255))

        arr = np.asarray(img)
        ref_rgb = arr[..., :3].astype(np.float32)
        ref_alpha = arr[..., 3] > 127
        _union, comps, _recs = ep._emblem_blob_components(img)
        self.assertTrue(comps, "cross inside circle must be detected")

        fp, core_pm, stroke_m, _owners = ep.badge_patch_weights(
            [(1, comps[0])], ref_rgb, ref_alpha)
        # White interior quadrant (95, 95) inside the badge must receive full 1.0 weight.
        self.assertAlmostEqual(float(fp[95, 95]), 1.0, places=5)
        self.assertTrue(bool(core_pm[95, 95]))
        # Thin outer circle ring at top (79, 110) must be absorbed into stroke_mask.
        self.assertTrue(bool(stroke_m[78:82, 110].any()), "closed circle ring must be absorbed")
        # Passing livery stripe at (167, 50) and dark window at (110, 180) must have 0 weight.
        self.assertEqual(float(fp[167, 50]), 0.0, "passing livery stripe must not be absorbed")
        self.assertEqual(float(fp[110, 180]), 0.0, "adjacent dark window must be excluded")

    def test_trellis_clean_preserves_enclosed_window_and_strips_vae_tinted_shadow(self):
        import numpy as np
        from PIL import Image
        import trellis_meshgen as tm

        arr = np.zeros((256, 256, 4), dtype=np.uint8)
        # Purple-tinted Qwen VAE background wisp (`alpha = 26`, chroma = 21).
        arr[:, :] = [15, 15, 36, 26]
        # Opaque white ambulance body (`alpha = 255`).
        arr[50:180, 50:206] = [238, 238, 238, 255]
        # Semi-transparent cabin window enclosed inside the opaque body (`alpha = 140`).
        arr[75:115, 85:135] = [40, 48, 56, 140]
        # Exterior soft grey cast shadow below body (`alpha = 160`, neutral grey).
        arr[182:215, 55:200] = [110, 112, 115, 160]

        cleaned = np.asarray(tm.clean_reference_rgba(Image.fromarray(arr, "RGBA")))
        self.assertEqual(int(cleaned[10, 10, 3]), 0, "purple VAE low-alpha wisp must be stripped")
        self.assertEqual(int(cleaned[95, 110, 3]), 140, "enclosed cabin window alpha must be preserved")
        self.assertEqual(int(cleaned[195, 120, 3]), 0, "exterior grey cast shadow must be stripped")

    def test_sanitize_voxel_albedo_lifts_black_void_without_whitening_underbody(self):
        for name in ("torch", "scipy"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"voxel albedo test requires {name}")
        import torch
        import trellis_meshgen as tm

        class DummyMesh:
            layout = {"base_color": slice(0, 3), "roughness": slice(3, 4)}
            coords = torch.tensor([
                [0, 10, 10, 20],  # bright white upper body
                [0, 10, 10, 2],   # dark tyre sidewall
                [0, 10, 10, 1],   # adjacent void tyre contact
                [0, 30, 30, 0],   # far occluded void underbody slab
            ], dtype=torch.int32)
            attrs = torch.tensor([
                [0.92, 0.92, 0.92, 0.5],
                [0.14, 0.14, 0.15, 0.8],
                [0.001, 0.001, 0.001, 0.9],
                [0.0, 0.0, 0.0, 0.95],
            ], dtype=torch.float32)

        m = DummyMesh()
        n_fixed = tm.sanitize_voxel_albedo(m)
        self.assertEqual(n_fixed, 2)
        bc = m.attrs[:, :3].numpy()
        # Upper white body and dark tyre sidewall remain untouched.
        self.assertAlmostEqual(float(bc[0, 0]), 0.92, places=4)
        self.assertAlmostEqual(float(bc[1, 0]), 0.14, places=4)
        # Void voxels are lifted out of [0,0,0] into dark charcoal range [0.075, 0.34], never white.
        self.assertGreaterEqual(float(bc[2].min()), 0.075)
        self.assertLessEqual(float(bc[2].max()), 0.34)
        self.assertGreaterEqual(float(bc[3].min()), 0.075)
        self.assertLessEqual(float(bc[3].max()), 0.21)

    def test_high_res_defaults_and_void_black_qa_gate(self):
        import offline_run
        import preflight
        import qa
        import run_all
        import trellis_meshgen as tm

        prof = offline_run.load_profile()
        self.assertEqual(prof["pipeline_type"], "1024_cascade")
        self.assertEqual(prof["texture_size"], 2048)
        self.assertEqual(prof["target_faces"], 300000)
        self.assertEqual(prof["guided_pipeline_type"], "1536_cascade")

        ra = run_all.parse_args([])
        self.assertEqual((ra.pipeline_type, ra.texture_size, ra.target_faces),
                         ("1024_cascade", 2048, 300000))
        ta = tm.parse_args([])
        self.assertEqual((ta.pipeline_type, ta.texture_size, ta.target_faces),
                         ("1024_cascade", 2048, 300000))
        self.assertEqual(preflight.storage_estimate(10),
                         preflight.storage_estimate(10, "spz", "1024_cascade", 2048))

        scores = {"vertices": 1000, "aspect": 1.0, "dim_ratio": 2.0,
                  "tex_white": 0.1, "tex_magenta": 0.0, "tex_void_black": 0.12,
                  "degenerate_frac": 0.0}
        self.assertTrue(any("void black" in r for r in qa.mesh_pass(scores)))

    def test_mgl_context_releases_gpu_objects_and_mtl_capture_not_forced(self):
        import os
        env_before = os.environ.pop("MTL_CAPTURE_ENABLED", None)
        try:
            import importlib
            import trellis_meshgen as tm
            importlib.reload(tm)
            self.assertNotIn(
                "MTL_CAPTURE_ENABLED", os.environ,
                "trellis_meshgen must not force MTL_CAPTURE_ENABLED=1 (causes multi-item Metal leaks)",
            )
        finally:
            if env_before is not None:
                os.environ["MTL_CAPTURE_ENABLED"] = env_before

        if importlib.util.find_spec("moderngl") is None:
            return
        import emblem_project as ep
        with ep._mgl_context() as ctx:
            self.assertNotEqual(type(ctx.mglo).__name__, "InvalidObject")
        self.assertEqual(type(ctx.mglo).__name__ == "InvalidObject", True)

    def test_inverse_uv_gather_eliminates_highres_comb_stripes_and_halos(self):
        for name in ("moderngl", "trimesh"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"inverse UV gather test requires {name}")
        import numpy as np
        import trimesh
        from PIL import Image, ImageDraw
        import emblem_project as ep

        # Front-facing Z-up panel at x = +0.5 spanning y, z in [-0.4, +0.4].
        V = np.array([
            [0.5, -0.4, -0.4],
            [0.5,  0.4, -0.4],
            [0.5,  0.4,  0.4],
            [0.5, -0.4,  0.4],
        ], dtype=np.float64)
        F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        UV = np.array([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]], dtype=np.float64)

        # Old 2048x2048 texture with a blurry/shifted green-blue smudge where forward
        # 512x512 scattering would leave >75% of texels unwritten.
        old_tex = Image.new("RGB", (2048, 2048), (236, 236, 234))
        dt = ImageDraw.Draw(old_tex)
        dt.ellipse([930, 930, 1180, 1180], fill=(40, 160, 90))

        mesh = trimesh.Trimesh(
            vertices=V, faces=F, process=False,
            visual=trimesh.visual.TextureVisuals(
                uv=UV,
                material=trimesh.visual.material.PBRMaterial(baseColorTexture=old_tex),
            ),
        )

        # 2048x2048 reference RGBA with crisp red cross + circle at panel centre.
        ref = Image.new("RGBA", (2048, 2048), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ref)
        dr.rectangle([256, 256, 1792, 1792], fill=(238, 238, 236, 255))
        dr.ellipse([900, 900, 1148, 1148], outline=(215, 20, 20, 255), width=14)
        dr.rectangle([1000, 936, 1048, 1112], fill=(215, 20, 20, 255))
        dr.rectangle([936, 1000, 1112, 1048], fill=(215, 20, 20, 255))

        blob_mask, recs = ep.emblem_blobs(ref)
        self.assertTrue(recs)
        cam, right, up, fwd = ep._orbit_pose(np.array([0.5, 0.0, 0.0]), 0.65, 0.0, 0.0)
        fit = {
            "azimuth_deg": 0.0, "elevation_deg": 0.0, "focal_ndc": 3.0,
            "iou": 0.99, "ambiguous": False,
            "center": [0.5, 0.0, 0.0], "radius": 0.65,
            "cam": list(cam), "right": list(right), "up": list(up), "forward": list(fwd),
        }
        new_tex, rep = ep.project_reference([mesh], old_tex, ref, fit, blob_mask=blob_mask)
        self.assertGreater(rep["projected_texels"], 30000,
                           "inverse UV gather must populate full 2048x2048 patch, not sparse 512 rays")
        out_arr = np.asarray(new_tex)
        # Central 16x16 texel block of the red cross must have 100% solid red fill (zero green comb stripes).
        core_block = out_arr[1016:1032, 1016:1032]
        self.assertTrue(bool(((core_block[..., 0] > 180) & (core_block[..., 1] < 50)).all()),
                        "every UV texel inside the emblem core must be written without Venetian-blind gaps")

    def test_gs_surface_gap_supplement_fills_thin_shell_pinholes(self):
        for name in ("trimesh", "scipy"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"GS gap supplement test requires {name}")
        import numpy as np
        import trimesh
        from PIL import Image
        import gs_export

        # Build a dense ring of voxels on z=0 plane in [-0.2, 0.2]^2 leaving a central hole |x|,|y| < 0.08.
        vs = 0.01
        xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 41), np.linspace(-0.2, 0.2, 41))
        xs, ys = xs.ravel(), ys.ravel()
        ring = (np.abs(xs) >= 0.08) | (np.abs(ys) >= 0.08)
        xs, ys = xs[ring], ys[ring]
        n = len(xs)
        g = {
            "x": xs.astype(np.float32),
            "y": ys.astype(np.float32),
            "z": np.zeros(n, dtype=np.float32),
            "f_dc": np.zeros((n, 3), dtype=np.float32),
            "f_rest": np.zeros((n, 45), dtype=np.float32),
            "opacity": np.full((n, 1), 5.0, dtype=np.float32),
            "scale": np.full((n, 3), np.log(vs * 0.55), dtype=np.float32),
            "rot": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        }

        # Watertight GLB bridging the central [-0.08, 0.08]^2 hole on z=0 (Z-up -> Y-up via gltf_vertices).
        gv, gf = [], []
        grid_p = np.linspace(-0.10, 0.10, 9)
        for i in range(8):
            for j in range(8):
                b = len(gv)
                gv.extend([
                    [grid_p[i], grid_p[j], 0.0],
                    [grid_p[i + 1], grid_p[j], 0.0],
                    [grid_p[i + 1], grid_p[j + 1], 0.0],
                    [grid_p[i], grid_p[j + 1], 0.0],
                ])
                gf.extend([[b, b + 1, b + 2], [b, b + 2, b + 3]])
        gv_yup = gs_export.gltf_vertices(np.asarray(gv, dtype=np.float32))
        uv = np.full((len(gv_yup), 2), 0.5, dtype=np.float32)
        mat = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (16, 16), (220, 40, 40)))
        tm_mesh = trimesh.Trimesh(
            vertices=gv_yup, faces=np.asarray(gf, dtype=np.int64), process=False,
            visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
        )
        with tempfile.TemporaryDirectory() as td:
            glb_p = Path(td) / "bridge.glb"
            tm_mesh.export(str(glb_p), file_type="glb")
            g2 = gs_export._supplement_surface_gaps_from_glb(g, glb_p, vs)
            self.assertGreater(len(g2["x"]), n, "central thin-shell pinhole must receive gap-filling splats")
            added_x = g2["x"][n:]
            added_y = g2["y"][n:]
            self.assertTrue(bool(((np.abs(added_x) < 0.08) & (np.abs(added_y) < 0.08)).any()))

    def test_subworker_chunking_args_roundtrip(self):
        import trellis_meshgen as tm
        orig = tm.parse_args([
            "--pipeline-type", "1536_cascade",
            "--texture-size", "4096",
            "--target-faces", "350000",
            "--mesh-steps", "14",
            "--seed", "42",
            "--gs-format", "both",
            "--allow-restricted",
            "--ignore-image-qa",
        ])
        rebuilt = tm.parse_args(tm._build_subworker_args(orig, ["ger_drk_0092", "ger_drk_0088"]))
        self.assertEqual(rebuilt.ids, ["ger_drk_0092", "ger_drk_0088"])
        self.assertEqual(rebuilt.pipeline_type, "1536_cascade")
        self.assertEqual(rebuilt.texture_size, 4096)
        self.assertEqual(rebuilt.target_faces, 350000)
        self.assertEqual(rebuilt.mesh_steps, 14)
        self.assertEqual(rebuilt.seed, 42)
        self.assertEqual(rebuilt.gs_format, "both")
        self.assertTrue(rebuilt.allow_restricted)
        self.assertTrue(rebuilt.ignore_image_qa)

    def test_appearance_similarity_breaks_blank_antipode_under_vertical_shift(self):
        import numpy as np
        import emblem_project as ep

        mask = np.ones((96, 96), dtype=bool)
        ref = np.full((96, 96, 3), 218.0, dtype=np.float32)
        # Small red cross in upper gable of ref (y=18..26, x=44..52).
        ref[18:26, 47:50] = [215.0, 24.0, 24.0]
        ref[21:24, 44:53] = [215.0, 24.0, 24.0]

        # Front candidate: same red cross shifted 9 px downward (outside 5-px roi).
        front = np.full((96, 96, 3), 216.0, dtype=np.float32)
        front[27:35, 47:50] = [205.0, 30.0, 30.0]
        front[30:33, 44:53] = [205.0, 30.0, 30.0]

        # Blank antipodal rear gable: zero red pixels anywhere.
        rear = np.full((96, 96, 3), 216.0, dtype=np.float32)

        s_front = ep.appearance_similarity(ref, mask, front, mask)
        s_rear = ep.appearance_similarity(ref, mask, rear, mask)
        self.assertGreater(
            s_front - s_rear,
            ep.APPEARANCE_MARGIN + 0.05,
            f"front with shifted red cross ({s_front:.4f}) must beat blank rear ({s_rear:.4f}) well above APPEARANCE_MARGIN",
        )

    def test_badge_patch_weights_captures_antialiased_circle_without_stripe_bleed(self):
        import numpy as np
        import emblem_project as ep

        ref_rgb = np.full((256, 256, 3), 215.0, dtype=np.float32)
        ref_alpha = np.ones((256, 256), dtype=bool)
        comp = np.zeros((256, 256), dtype=bool)
        comp[116:140, 124:132] = True
        comp[124:132, 116:140] = True
        ref_rgb[comp] = [210.0, 22.0, 22.0]

        # Anti-aliased thin circular border (rgb=[167,129,126], chr~0.245) at r=16.
        yy, xx = np.mgrid[0:256, 0:256]
        dist = np.hypot(yy - 127.5, xx - 127.5)
        aa_ring = (dist >= 15.2) & (dist <= 16.8)
        ref_rgb[aa_ring] = [167.0, 129.0, 126.0]

        # Passing full-width red livery stripe below the badge (y=156..164) must stay untouched.
        stripe = np.zeros((256, 256), dtype=bool)
        stripe[156:164, 20:236] = True
        ref_rgb[stripe] = [215.0, 25.0, 25.0]

        fp, _core, _stroke, _owner = ep.badge_patch_weights([(1, comp)], ref_rgb, ref_alpha)
        self.assertTrue(bool((fp[aa_ring] >= 0.99).all()),
                        "anti-aliased thin badge circle must not leave fp=0 pinholes")
        self.assertTrue(bool((fp[stripe] == 0.0).all()),
                        "passing livery stripe must remain 100% excluded")

    def test_multitone_crest_detection_across_neutral_painted_and_dark_panels(self):
        import numpy as np
        from PIL import Image, ImageDraw
        import emblem_project as ep

        panels = {
            "neutral": (232, 232, 230, 255),
            "painted_orange": (235, 118, 22, 255),
            "painted_red": (208, 30, 28, 255),
            "dark_armor": (62, 66, 70, 255),
        }
        cx, cy = 256, 256
        for name, bg_rgba in panels.items():
            with self.subTest(panel=name, has_crest=False):
                plain = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                ImageDraw.Draw(plain).rectangle([32, 32, 480, 480], fill=bg_rgba)
                _u, comps_plain, recs_plain = ep._emblem_blob_components(plain)
                self.assertEqual(recs_plain, [], f"plain {name} panel must not false-trigger")
                self.assertEqual(comps_plain, [])

            with self.subTest(panel=name, has_crest=True):
                img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)
                d.rectangle([32, 32, 480, 480], fill=bg_rgba)
                # Multi-tone circular/shield crest with bright perimeter ring + dual-tone inner motif.
                d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26],
                          fill=(242, 242, 240, 255), outline=(24, 34, 52, 255), width=3)
                d.polygon([(cx, cy - 17), (cx + 15, cy), (cx, cy + 17), (cx - 15, cy)],
                          fill=(235, 192, 32, 255))
                d.rectangle([cx - 4, cy - 11, cx + 4, cy + 11], fill=(24, 62, 165, 255))
                d.rectangle([cx - 11, cy - 4, cx + 11, cy + 4], fill=(24, 62, 165, 255))

                union, comps, recs = ep._emblem_blob_components(img)
                self.assertTrue(len(recs) >= 1 and len(comps) >= 1,
                                f"multi-tone crest on {name} panel must be detected")
                self.assertTrue(bool(union[cy - 26:cy + 27, cx - 26:cx + 27].any()),
                                f"detected crest mask on {name} panel must overlap crest center")
                # Keep detections localized to the crest rather than swallowing the whole host panel.
                self.assertLess(int(union.sum()), 10000,
                                f"crest mask on {name} panel leaked into host panel")

    def test_apply_shift_with_collar_fallback_preserves_border_collar(self):
        import numpy as np
        import emblem_project as ep

        h, w = 64, 64
        ref_alpha = np.zeros((h, w), dtype=bool)
        ref_alpha[12:52, 12:52] = True
        fg_dist_ref = ep._fg_distance(ref_alpha)

        stroke_mask = np.zeros((h, w), dtype=bool)
        stroke_mask[24:40, 29:35] = True
        stroke_mask[29:35, 24:40] = True

        footprint = np.zeros((h, w), dtype=np.float32)
        footprint[14:51, 14:51] = 1.0

        sx_map = np.full((h, w), 8.0, dtype=np.float32)
        sy_map = np.zeros((h, w), dtype=np.float32)

        # Samples: (0) stroke pixel shifted safely inside, (1) interior collar shifted safely inside,
        # (2) near-edge outer collar shifted across fg boundary (x=49 -> 57, outside fg),
        # (3) outer collar shifted completely out of image bounds (sx=30 -> 79 >= w).
        rxf0 = np.array([32.0, 40.0, 49.0, 49.0], dtype=np.float32)
        ryf0 = np.array([32.0, 32.0, 32.0, 32.0], dtype=np.float32)
        rx0_i = rxf0.astype(np.int64)
        ry0_i = ryf0.astype(np.int64)

        rxf, ryf = ep._apply_shift_with_collar_fallback(
            rxf0, ryf0, rx0_i, ry0_i,
            (sx_map, sy_map), ref_alpha,
            fg_dist_ref=fg_dist_ref, stroke_mask=stroke_mask, footprint=footprint,
        )
        np.testing.assert_allclose(rxf[:3], [40.0, 48.0, 49.0])
        np.testing.assert_allclose(ryf[:3], [32.0, 32.0, 32.0])

        sx_oob = np.full((h, w), 30.0, dtype=np.float32)
        rxf_oob, ryf_oob = ep._apply_shift_with_collar_fallback(
            rxf0[2:3], ryf0[2:3], rx0_i[2:3], ry0_i[2:3],
            (sx_oob, np.zeros_like(sx_oob)), ref_alpha,
            fg_dist_ref=fg_dist_ref, stroke_mask=stroke_mask, footprint=footprint,
        )
        np.testing.assert_allclose(rxf_oob, [49.0])
        np.testing.assert_allclose(ryf_oob, [32.0])

    def test_solid_lamp_and_door_handle_blocks_rejected_while_cross_detected(self):
        import numpy as np
        from PIL import Image, ImageDraw
        import emblem_project as ep

        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([32, 32, 480, 480], fill=(236, 236, 234, 255))

        # Genuine thin-armed red cross (fill < 0.68, stroke/min_dim <= 0.34) must be detected.
        d.rectangle([134, 152, 146, 208], fill=(215, 22, 22, 255))
        d.rectangle([112, 174, 168, 186], fill=(215, 22, 22, 255))

        # Solid rectangular red side flasher lamp and solid dark vertical door handle
        # (high fill and high inscribed-radius ratio) must both be rejected.
        d.rectangle([280, 160, 314, 194], fill=(215, 22, 22, 255))
        d.rectangle([360, 260, 377, 296], fill=(20, 22, 26, 255))

        union, comps, recs = ep._emblem_blob_components(img)
        self.assertEqual(len(recs), 1, f"expected only the cross badge, got {recs}")
        self.assertEqual(len(comps), 1)
        x0, y0, x1, y1 = recs[0]["bbox"]
        self.assertGreaterEqual(x0, 110)
        self.assertGreaterEqual(y0, 150)
        self.assertLessEqual(x1, 170)
        self.assertLessEqual(y1, 210)
        self.assertFalse(bool(union[160:195, 280:315].any()), "solid red lamp must not trigger")
        self.assertFalse(bool(union[260:297, 360:378].any()), "solid dark door handle must not trigger")

    def test_replace_glb_texture_preserves_and_self_heals_gltf_4byte_alignment(self):
        import io
        import json
        import struct
        import tempfile
        from pathlib import Path
        import numpy as np
        from PIL import Image
        import emblem_project as ep
        import qa

        try:
            import trimesh
        except ImportError:
            trimesh = None

        with tempfile.TemporaryDirectory() as root:
            glb_path = Path(root) / "aligned.glb"
            img0 = Image.new("RGBA", (32, 32), (200, 40, 40, 255))
            if trimesh is not None:
                box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
                uv = np.zeros((len(box.vertices), 2), dtype=np.float32)
                mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img0, alphaMode="OPAQUE")
                box.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
                box.export(str(glb_path), file_type="glb")
            else:
                # Minimal valid glTF 2.0 GLB with an image bufferView followed by a FLOAT32 VEC3 accessor.
                buf0 = io.BytesIO()
                img0.save(buf0, format="PNG")
                png0 = buf0.getvalue()
                pad0 = (-len(png0)) % 4
                verts_bin = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
                bin_blob = png0 + b"\x00" * pad0 + verts_bin
                bin_blob += b"\x00" * ((-len(bin_blob)) % 4)
                js0 = {
                    "asset": {"version": "2.0"},
                    "buffers": [{"byteLength": len(bin_blob)}],
                    "bufferViews": [
                        {"buffer": 0, "byteOffset": 0, "byteLength": len(png0)},
                        {"buffer": 0, "byteOffset": len(png0) + pad0, "byteLength": len(verts_bin)},
                    ],
                    "accessors": [{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}],
                    "images": [{"bufferView": 0, "mimeType": "image/png"}],
                    "textures": [{"source": 0}],
                    "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
                }
                js_raw = json.dumps(js0, separators=(",", ":")).encode("utf-8")
                js_pad = (-len(js_raw)) % 4
                hdr_len = 12 + 8 + len(js_raw) + js_pad + 8 + len(bin_blob)
                glb_path.write_bytes(
                    struct.pack("<III", 0x46546C67, 2, hdr_len)
                    + struct.pack("<II", len(js_raw) + js_pad, 0x4E4F534A)
                    + js_raw + b" " * js_pad
                    + struct.pack("<II", len(bin_blob), 0x004E4942)
                    + bin_blob
                )
            self.assertEqual(qa.glb_alignment_errors(glb_path), [])

            # Repeated texture swaps of varying lengths mod 4 must remain 4-byte aligned.
            tex_idx = ep.glb_basecolor_image_index(glb_path)
            for idx, color in enumerate([(10, 20, 30, 255), (250, 240, 230, 255), (15, 180, 95, 255)]):
                buf = io.BytesIO()
                im = Image.new("RGBA", (32 + idx, 32), color)
                im.save(buf, format="PNG")
                raw_png = buf.getvalue()
                patched = ep.replace_glb_texture(glb_path, tex_idx, raw_png)
                glb_path.write_bytes(patched)
                self.assertEqual(qa.glb_alignment_errors(glb_path), [], f"iteration {idx} misaligned")
                if trimesh is not None:
                    scores, _ = qa.glb_scores(glb_path)
                    self.assertTrue(scores["glb_aligned"])

            # Deliberately corrupt alignment (simulating pre-fix replace_glb_texture:
            # 3 extra zero bytes inserted after image bufferView, shifting later views by +3).
            data = bytearray(glb_path.read_bytes())
            jlen, _ = struct.unpack("<II", data[12:20])
            js = json.loads(data[20:20 + jlen].decode("utf-8"))
            boff = 20 + jlen
            blen, _ = struct.unpack("<II", data[boff:boff + 8])
            bin_blob = bytearray(data[boff + 8:boff + 8 + blen])
            bvs = js["bufferViews"]
            bv = bvs[js["images"][tex_idx]["bufferView"]]
            start = int(bv.get("byteOffset", 0))
            end = start + int(bv["byteLength"])
            corrupt_blob = bytes(bin_blob[:end]) + b"\x00\x00\x00" + bytes(bin_blob[end:])
            for other in bvs:
                if other is not bv and int(other.get("byteOffset", 0)) > start:
                    other["byteOffset"] = int(other.get("byteOffset", 0)) + 3
            js["buffers"][0]["byteLength"] = len(corrupt_blob)
            js_raw = json.dumps(js, separators=(",", ":")).encode("utf-8")
            js_pad = (-len(js_raw)) % 4
            hdr_len = 12 + 8 + len(js_raw) + js_pad + 8 + len(corrupt_blob)
            corrupt = (struct.pack("<III", 0x46546C67, 2, hdr_len)
                       + struct.pack("<II", len(js_raw) + js_pad, 0x4E4F534A)
                       + js_raw + b" " * js_pad
                       + struct.pack("<II", len(corrupt_blob), 0x004E4942)
                       + corrupt_blob)
            glb_path.write_bytes(corrupt)
            # QA must catch the corruption cleanly.
            self.assertTrue(qa.glb_alignment_errors(glb_path))
            if trimesh is not None:
                bad_scores, _ = qa.glb_scores(glb_path)
                self.assertFalse(bad_scores["glb_aligned"])
                self.assertTrue(any("glTF alignment error" in r for r in qa.mesh_pass(bad_scores)))
            # Fixed replace_glb_texture must self-heal the corrupted GLB back to 4-byte alignment.
            buf = io.BytesIO()
            Image.new("RGBA", (32, 32), (80, 90, 100, 255)).save(buf, format="PNG")
            healed = ep.replace_glb_texture(glb_path, tex_idx, buf.getvalue())
            glb_path.write_bytes(healed)
            self.assertEqual(qa.glb_alignment_errors(glb_path), [])
            if trimesh is not None:
                healed_scores, _ = qa.glb_scores(glb_path)
                self.assertTrue(healed_scores["glb_aligned"])

    def test_adjust_remesh_dc_kwargs_prevents_aabb_clipping_at_capped_resolution(self):
        import trellis_meshgen as tm

        # Mirrors o_voxel.postprocess.to_glb at resolution=1024, band=1.0, base_scale=1.0.
        orig_kw = {"resolution": 1024, "band": 1.0, "scale": (1024.0 + 3.0) / 1024.0}
        adj = tm._adjust_remesh_dc_kwargs(orig_kw, max_res=384)
        self.assertEqual(adj["resolution"], 384)
        eps = adj["scale"] / adj["resolution"]
        # Grid half-extent must strictly exceed 0.5 + band * eps so faces touching +/-0.500 never clip.
        self.assertGreater(0.5 * adj["scale"], 0.5 + adj["band"] * eps)
        # Uncapped resolutions are left untouched.
        small = {"resolution": 256, "band": 1.0, "scale": 1.01}
        self.assertEqual(tm._adjust_remesh_dc_kwargs(small, max_res=384), small)

    def test_qa_welded_mesh_silhouettes_prevent_uv_seam_tears(self):
        import numpy as np
        import qa
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh unavailable")

        # Build a watertight cube with deliberately unwelded vertices (every triangle
        # uses distinct vertex indices, simulating dense UV-split GLB charts).
        base = trimesh.creation.box(extents=(0.8, 0.8, 0.8)).subdivide().subdivide()
        unwelded_v = base.vertices[base.faces].reshape(-1, 3)
        unwelded_f = np.arange(len(unwelded_v), dtype=np.int64).reshape(-1, 3)
        unwelded = trimesh.Trimesh(vertices=unwelded_v, faces=unwelded_f, process=False)
        # Force decimation branch by temporarily lowering MESH_RAY_MAX_FACES.
        orig_max = qa.MESH_RAY_MAX_FACES
        try:
            qa.MESH_RAY_MAX_FACES = 100
            masks, _ = qa.mesh_silhouettes([unwelded], res=64)
        finally:
            qa.MESH_RAY_MAX_FACES = orig_max
        stats = qa.silhouette_view_metrics(masks)
        self.assertEqual(stats["holes_max"], 0.0, f"welded decimation tore holes: {stats}")
        self.assertGreater(stats["coverage_min"], 0.15)

    def test_qa_front_slab_splat_views_occlude_back_wall(self):
        import numpy as np
        import qa
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        grid = np.linspace(-0.25, 0.25, 25, dtype=np.float32)
        gy, gz = np.meshgrid(grid, grid)
        n_plane = gy.size
        # Front plane at x = +0.35 (red), back plane at x = -0.35 (blue).
        x = np.concatenate([np.full(n_plane, 0.35, np.float32), np.full(n_plane, -0.35, np.float32)])
        y = np.concatenate([gy.ravel(), gy.ravel()])
        z = np.concatenate([gz.ravel(), gz.ravel()])
        rgb = np.zeros((2 * n_plane, 3), dtype=np.float32)
        rgb[:n_plane, 0] = 1.0  # red front
        rgb[n_plane:, 2] = 1.0  # blue back
        cloud = {
            "x": x, "y": y, "z": z,
            "scale": np.full((2 * n_plane, 3), np.log(0.025), dtype=np.float32),
            "rot": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2 * n_plane, 1)),
            "opacity": np.full((2 * n_plane, 1), 4.0, dtype=np.float32),
            "f_dc": ((rgb - 0.5) / qa.SH_C0).astype(np.float32),
        }
        _masks_g, _w, cols = qa.render_splat_views(cloud, res=64)
        # View 0 sits on the +X side looking at -X, so center pixels must be overwhelmingly red (front), not purple!
        center = cols[0][28:36, 28:36]
        self.assertGreater(float(center[..., 0].mean()), 200.0)
        self.assertLess(float(center[..., 2].mean()), 40.0)

    def test_panel_text_detector_catches_white_and_dark_lettering_with_tight_collar(self):
        import numpy as np
        from PIL import Image, ImageDraw
        import emblem_project as ep

        # Synthetic 512x512 red vehicle body with crisp white block capitals ("THW" / "FEUER")
        # painted along the side panel, plus a long solid white stripe below that must NOT trigger.
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([32, 32, 480, 480], fill=(205, 22, 22, 255))

        # Draw 6 crisp block capital letters (F-E-U-E-R-W) with 3-px strokes on the upper red panel.
        base_y = 140
        h_c = 28
        xs = [110, 136, 162, 188, 214, 240]
        # F
        d.rectangle([xs[0], base_y, xs[0] + 4, base_y + h_c], fill=(248, 248, 248, 255))
        d.rectangle([xs[0], base_y, xs[0] + 17, base_y + 4], fill=(248, 248, 248, 255))
        d.rectangle([xs[0], base_y + 12, xs[0] + 14, base_y + 16], fill=(248, 248, 248, 255))
        # E
        for x in (xs[1], xs[3]):
            d.rectangle([x, base_y, x + 4, base_y + h_c], fill=(248, 248, 248, 255))
            d.rectangle([x, base_y, x + 17, base_y + 4], fill=(248, 248, 248, 255))
            d.rectangle([x, base_y + 12, x + 14, base_y + 16], fill=(248, 248, 248, 255))
            d.rectangle([x, base_y + h_c - 4, x + 17, base_y + h_c], fill=(248, 248, 248, 255))
        # U
        d.rectangle([xs[2], base_y, xs[2] + 4, base_y + h_c], fill=(248, 248, 248, 255))
        d.rectangle([xs[2] + 13, base_y, xs[2] + 17, base_y + h_c], fill=(248, 248, 248, 255))
        d.rectangle([xs[2], base_y + h_c - 4, xs[2] + 17, base_y + h_c], fill=(248, 248, 248, 255))
        # H
        d.rectangle([xs[4], base_y, xs[4] + 4, base_y + h_c], fill=(248, 248, 248, 255))
        d.rectangle([xs[4] + 13, base_y, xs[4] + 17, base_y + h_c], fill=(248, 248, 248, 255))
        d.rectangle([xs[4], base_y + 12, xs[4] + 17, base_y + 16], fill=(248, 248, 248, 255))
        # T
        d.rectangle([xs[5], base_y, xs[5] + 18, base_y + 4], fill=(248, 248, 248, 255))
        d.rectangle([xs[5] + 7, base_y, xs[5] + 11, base_y + h_c], fill=(248, 248, 248, 255))

        # Long solid white livery stripe far below must NOT be detected as text.
        d.rectangle([40, 340, 470, 362], fill=(248, 248, 248, 255))

        union, comps, recs = ep._emblem_blob_components(img)
        text_recs = [r for r in recs if r["kind"] == "text"]
        self.assertEqual(len(text_recs), 1, f"expected 1 text band, got {recs}")
        self.assertGreaterEqual(text_recs[0]["glyphs"], 5)
        self.assertFalse(bool(union[338:364, :].any()), "solid livery stripe must not trigger text detector")

        # Text-band collar in badge_patch_weights must hug character height (~28px),
        # never expanding to a giant circle of radius 0.82 * word_width (~130px).
        arr = np.asarray(img)
        fp, _core, _stroke, _owners = ep.badge_patch_weights(
            [(1, comps[0])], arr[..., :3].astype(np.float32), arr[..., 3] > 127)
        self.assertEqual(float(fp[base_y - 60, 180]), 0.0, "text collar must not blow up vertically to word width")
        self.assertEqual(float(fp[base_y + h_c + 60, 180]), 0.0, "text collar must not blow up vertically to word width")
        self.assertGreater(float(fp[base_y + 10, 112]), 0.9, "letter strokes must retain full projection weight")

    def test_large_saturated_word_absorbs_pass1_letters_and_rescues_narrow_glyphs(self):
        from PIL import Image, ImageDraw
        import emblem_project as ep

        # Bold red word on white panel (e.g. "RØDE KORS") where wide letters exceed
        # Pass-1 BLOB_MIN_FRAC while a narrow "I" exceeds BLOB_MAX_ASPECT (2.6).
        # Pass 3 must unify all glyphs into one coherent text band without dropping
        # the narrow letter, while leaving an adjacent large red cross untouched.
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([28, 28, 484, 484], fill=(240, 240, 238, 255))

        red = (210, 24, 24, 255)
        # Adjacent large red cross on the left panel (must remain its own saturated emblem).
        d.rectangle([82, 260, 102, 340], fill=red)
        d.rectangle([52, 290, 132, 310], fill=red)

        # 5 large red capitals ("H-E-F-U-I") with ~36px height; wide letters cross
        # Pass-1 area threshold, while trailing narrow "I" (7x36 -> aspect 5.1,
        # solid bar) fails Pass-1 and cannot be covered merely by interior bridging.
        y0, h_c = 130, 36
        # H
        d.rectangle([170, y0, 176, y0 + h_c], fill=red)
        d.rectangle([188, y0, 194, y0 + h_c], fill=red)
        d.rectangle([170, y0 + 15, 194, y0 + 21], fill=red)
        # E
        d.rectangle([204, y0, 210, y0 + h_c], fill=red)
        d.rectangle([204, y0, 224, y0 + 6], fill=red)
        d.rectangle([204, y0 + 15, 221, y0 + 21], fill=red)
        d.rectangle([204, y0 + h_c - 6, 224, y0 + h_c], fill=red)
        # F
        d.rectangle([234, y0, 240, y0 + h_c], fill=red)
        d.rectangle([234, y0, 254, y0 + 6], fill=red)
        d.rectangle([234, y0 + 15, 251, y0 + 21], fill=red)
        # U
        d.rectangle([264, y0, 270, y0 + h_c], fill=red)
        d.rectangle([282, y0, 288, y0 + h_c], fill=red)
        d.rectangle([264, y0 + h_c - 6, 288, y0 + h_c], fill=red)
        # Trailing narrow I (aspect > 7.5, condensed perspective vertical bar — like "I" in POLIZEI)
        d.rectangle([299, y0, 303, y0 + h_c], fill=red)

        # Row of 4 identical vertical louver bars below must NOT trigger text (0 non-bar glyphs).
        for bx in (200, 218, 236, 254):
            d.rectangle([bx, 310, bx + 6, 346], fill=red)

        union, _comps, recs = ep._emblem_blob_components(img)
        text_recs = [r for r in recs if r["kind"] == "text"]
        sat_recs = [r for r in recs if r["kind"] == "saturated"]
        self.assertEqual(len(text_recs), 1, f"expected 1 unified text band, got {recs}")
        self.assertEqual(text_recs[0]["glyphs"], 5, f"trailing narrow 'I' must be rescued: {text_recs[0]}")
        self.assertTrue(bool(union[y0 + 18, 302]), "trailing 'I' stroke must be covered by unified text band")
        self.assertFalse(bool(union[310:347, 200:261].any()), "louver bars with 0 non-bar glyphs must not trigger")
        self.assertEqual(len(sat_recs), 1, f"only the red cross should remain saturated, got {sat_recs}")

    def test_short_3letter_text_slot_uses_tight_collar_and_void_black_respects_source(self):
        import numpy as np
        from PIL import Image, ImageDraw
        import emblem_project as ep
        import qa

        # Short 3-letter word ("THW" ~ 80x34 -> aspect ~2.35 <= 2.8) passed via text_slots={1}
        # must keep a tight character-height collar and NOT absorb nearby frame strokes.
        img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([32, 32, 480, 480], fill=(24, 64, 150, 255))
        y0, h_c = 200, 34
        # T, H, W (~80px wide total -> aspect 80/34 = 2.35 <= 2.8)
        d.rectangle([200, y0, 222, y0 + 5], fill=(248, 248, 248, 255))
        d.rectangle([209, y0, 214, y0 + h_c], fill=(248, 248, 248, 255))
        d.rectangle([230, y0, 235, y0 + h_c], fill=(248, 248, 248, 255))
        d.rectangle([247, y0, 252, y0 + h_c], fill=(248, 248, 248, 255))
        d.rectangle([230, y0 + 14, 252, y0 + 19], fill=(248, 248, 248, 255))
        d.rectangle([260, y0, 265, y0 + h_c], fill=(248, 248, 248, 255))
        d.rectangle([275, y0, 280, y0 + h_c], fill=(248, 248, 248, 255))
        d.rectangle([260, y0 + h_c - 5, 280, y0 + h_c], fill=(248, 248, 248, 255))

        arr = np.asarray(img)
        _u, comps, recs = ep._emblem_blob_components(img)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["kind"], "text")
        fp, _core, _stroke, _owners = ep.badge_patch_weights(
            [(1, comps[0])], arr[..., :3].astype(np.float32), arr[..., 3] > 127,
            text_slots={1})
        self.assertEqual(float(fp[y0 - 45, 240]), 0.0, "3-letter text collar must scale to char height, not word width")
        self.assertEqual(float(fp[y0 + h_c + 45, 240]), 0.0)

        # Reference-relative void-black check in mesh_pass.
        base_mesh = {
            "vertices": 1000, "aspect": 0.6, "dim_ratio": 2.0,
            "tex_white": 0.1, "tex_magenta": 0.0, "degenerate_frac": 0.0,
            "glb_aligned": True,
        }
        self.assertEqual(qa.mesh_pass({**base_mesh, "tex_void_black": 0.08, "source_void_black_frac": 0.06}), [])
        self.assertIn("void black texture 0.09", qa.mesh_pass({**base_mesh, "tex_void_black": 0.09, "source_void_black_frac": 0.01}))


if __name__ == "__main__":
    unittest.main()
