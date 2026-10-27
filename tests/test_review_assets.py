"""Visual-review regressions: real depth occlusion and complete framing."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local"))


class TestGlbReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "PIL", "trimesh", "moderngl"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"GLB review requires {name}")
        import moderngl
        try:
            with moderngl.create_standalone_context(require=330):
                pass
        except Exception as exc:
            raise unittest.SkipTest(f"OpenGL review context unavailable: {exc}")
        import review_assets
        cls.review = review_assets

    def test_nearest_surface_wins_regardless_of_scene_draw_order(self):
        import numpy as np
        import trimesh
        from PIL import Image

        boxes = []
        for x, color in [(0.15, (255, 0, 0)), (-0.15, (0, 0, 255))]:
            box = trimesh.creation.box(extents=(0.05, 1, 1))
            box.apply_translation((x, 0, 0))
            box.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(box.vertices), 2)),
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.new("RGB", (2, 2), color), alphaMode="OPAQUE"))
            boxes.append(box)
        with tempfile.TemporaryDirectory() as root:
            results = []
            for order in (boxes, boxes[::-1]):
                path = Path(root) / "overlap.glb"
                trimesh.Scene(order).export(path)
                textured, _, _ = self.review.render_glb(path, size=96)
                results.append(textured[0])
        # First orbit camera looks from +X: the red surface must occlude blue.
        pixel = results[0][48, 48]
        self.assertGreater(int(pixel[0]), int(pixel[2]) + 100)
        np.testing.assert_array_equal(*results)

    def test_long_asymmetric_asset_stays_inside_every_review_frame(self):
        import numpy as np
        import trimesh

        body = trimesh.creation.box(extents=(0.8, 1.0, 1.0))
        tail = trimesh.creation.box(extents=(3.0, 0.1, 0.1))
        tail.apply_translation((1.7, 0.4, 0))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "asymmetric.glb"
            trimesh.Scene([body, tail]).export(path)
            _, clay, _ = self.review.render_glb(path, size=128)
        for view in clay:
            foreground = view[..., 0] > 90
            self.assertTrue(foreground.any())
            self.assertFalse(np.any(foreground[[0, -1]]))
            self.assertFalse(np.any(foreground[:, [0, -1]]))


if __name__ == "__main__":
    unittest.main()
