"""Repair known RGB-in-DC exports while preserving sources and provenance."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local"))


class TestSpzRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("numpy", "spz"):
            if importlib.util.find_spec(name) is None:
                raise unittest.SkipTest(f"SPZ repair requires {name}")
        import repair_spz_colors
        cls.repair = staticmethod(repair_spz_colors.repair_spz)

    def fixture(self, root):
        import spz
        import numpy as np
        from pipeline_utils import write_json, file_sha256
        folder = Path(root) / "original"
        folder.mkdir()
        native = spz.GaussianCloud()
        native.positions = np.zeros(9, dtype=np.float32)
        native.scales = np.full(9, -3.0, dtype=np.float32)
        native.rotations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), 3)
        native.colors = np.repeat(np.array([0, 0.5, 1], dtype=np.float32), 3)
        native.alphas = np.full(3, 2.0, dtype=np.float32)
        source = folder / "asset.spz"
        self.assertTrue(spz.save_spz(native, spz.PackOptions(), str(source)))
        (folder / "asset.glb").write_bytes(b"unchanged geometry and textures")
        (folder / "asset.png").write_bytes(b"unchanged source reference")
        write_json(folder / "metadata.json", {"id": "asset", "mesh_model": "local/trellis.2-4b",
            "image_model": "local/qwen-image-2.1", "source_image_sha256": file_sha256(folder / "asset.png")})
        return source

    def test_repaired_copy_has_native_colours_and_preserves_source(self):
        import numpy as np
        import spz
        from gs_export import C0
        from pipeline_utils import load_json
        with tempfile.TemporaryDirectory() as root:
            source = self.fixture(root)
            original = source.read_bytes()
            target = Path(root) / "repaired"
            self.repair(source, target, confirmed_legacy_rgb=True)
            actual = np.asarray(spz.load_spz(str(target / source.name)).colors).reshape(3, 3) * C0 + 0.5
            expected = np.repeat(np.array([[0], [0.5], [1]], dtype=np.float32), 3, axis=1)
            np.testing.assert_allclose(actual, expected, atol=0.02)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual((target / "asset.glb").read_bytes(), (source.parent / "asset.glb").read_bytes())
            self.assertEqual((target / "asset.png").read_bytes(), (source.parent / "asset.png").read_bytes())
            metadata = load_json(target / "metadata.json")
            self.assertEqual(metadata["image_model"], "local/qwen-image-2.1")
            self.assertEqual(metadata["gs_color_encodings"]["spz"], "sh_dc")
            self.assertEqual(metadata["repairs"][0]["kind"], "spz_rgb_to_sh_dc")

    def test_unconfirmed_conversion_creates_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.fixture(root)
            target = Path(root) / "repaired"
            with self.assertRaisesRegex(ValueError, "Confirm"):
                self.repair(source, target)
            self.assertFalse(target.exists())

    def test_verified_encoding_is_not_converted_twice(self):
        from pipeline_utils import load_json, write_json
        with tempfile.TemporaryDirectory() as root:
            source = self.fixture(root)
            metadata = load_json(source.parent / "metadata.json")
            metadata["gs_color_encodings"] = {"spz": "sh_dc"}
            write_json(source.parent / "metadata.json", metadata)
            target = Path(root) / "repaired"
            with self.assertRaisesRegex(ValueError, "already tagged"):
                self.repair(source, target, confirmed_legacy_rgb=True)
            self.assertFalse(target.exists())

    def test_interrupted_copy_never_exposes_a_commit_marker(self):
        import shutil
        import repair_spz_colors

        def interrupted_copy(source_dir, destination, *, ignore=None):
            destination.mkdir()
            names = [path.name for path in source_dir.iterdir()]
            excluded = set(ignore(str(source_dir), names)) if ignore else set()
            # Exercise the worst ordering: the source marker is encountered
            # before an artifact copy fails (e.g. disk full or power loss).
            for name in ["metadata.json", "asset.glb"]:
                if name not in excluded:
                    shutil.copy2(source_dir / name, destination / name)
            raise OSError("copy interrupted")

        with tempfile.TemporaryDirectory() as root:
            source = self.fixture(root)
            target = Path(root) / "repaired"
            original_marker = (source.parent / "metadata.json").read_bytes()
            with patch.object(repair_spz_colors.shutil, "copytree", side_effect=interrupted_copy):
                with self.assertRaisesRegex(OSError, "copy interrupted"):
                    self.repair(source, target, confirmed_legacy_rgb=True)
            self.assertFalse((target / "metadata.json").exists())
            self.assertEqual((source.parent / "metadata.json").read_bytes(), original_marker)


if __name__ == "__main__":
    unittest.main()
