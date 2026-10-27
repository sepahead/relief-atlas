"""A geometry guide stays distinct from a generated reference and binds its input."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local"))

import build_structure_guide as guide
import qwen_imagegen as qwen
from generation_prompts import digest
from pipeline_utils import file_sha256, write_json


class TestStructureGuide(unittest.TestCase):
    def fixture(self, root, *, override=False):
        from PIL import Image
        source = Path(root) / "layout.png"
        Image.new("RGBA", (32, 32), (30, 60, 90, 180)).save(source)
        spec = {"profile": "cargo_flat_octocopter", "rotor_layout": "radial", "arm_count": 8, "rotor_count": 8}
        item = {"id": "a", "prompt": "cargo octocopter", "generation_prompt": {"specifications": {"prompt": spec}}}
        record = {"schema_version": 1, "kind": "authored_geometry_guide", "id": "a",
                  "image_sha256": file_sha256(source), "manifest_prompt_sha256": digest(item["prompt"]),
                  "specification": spec, "motors": [{"index": i} for i in range(8)], "policy": {"override": override}}
        write_json(source.with_suffix(".json"), record)
        return source, item, record

    def test_native_input_keeps_alpha_and_records_the_actual_guide_kind(self):
        with tempfile.TemporaryDirectory() as root:
            source, item, _record = self.fixture(root)
            args = qwen.parse_args(["--ids", "a", "--structure-reference", str(source)])
            result = qwen.structure_provenance(source, item, Path(root) / "new/a.png")
            inputs = qwen.inference_inputs(args, "Subject: cargo octocopter.\nSurface: light service wear.")
            self.assertEqual(result["mode"], "native_structure_reference")
            self.assertEqual(result["edit_source_kind"], "authored_geometry_guide")
            self.assertEqual(result["edit_source_metadata_sha256"], file_sha256(source.with_suffix(".json")))
            self.assertEqual(inputs["image"].mode, "RGBA")
            self.assertEqual(inputs["image"].getpixel((0, 0))[3], 180)
            self.assertIn("component count, component positions", inputs["prompt"])
            self.assertIn("Surface: light service wear.", inputs["prompt"])
            self.assertNotIn("View:", inputs["prompt"])

    def test_guide_refuses_changed_item_prompt_image_or_layout(self):
        for change in ("id", "prompt", "pixels", "layout"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                source, item, _record = self.fixture(root)
                if change == "pixels":
                    source.write_bytes(b"changed")
                elif change == "layout":
                    item["generation_prompt"]["specifications"]["prompt"]["profile"] = "coaxial"
                else:
                    item[change] = "changed"
                with self.assertRaisesRegex(ValueError, "hash-bound|eight-boom"):
                    qwen.structure_provenance(source, item, Path(root) / "new/a.png")

    def test_guide_cannot_overwrite_source_or_launder_an_override(self):
        with tempfile.TemporaryDirectory() as root:
            source, item, _record = self.fixture(root, override=True)
            with self.assertRaisesRegex(ValueError, "preserve the source"):
                qwen.structure_provenance(source, item, source, override=True)
            with self.assertRaisesRegex(ValueError, "default-policy"):
                qwen.structure_provenance(source, item, Path(root) / "new/a.png")

    def test_guide_interrupted_commit_uses_the_same_source_bound_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            source, item, _record = self.fixture(root)
            output = Path(root) / "new/a.png"
            record = qwen.structure_provenance(source, item, output)
            write_json(output.parent / ".reference_edit.json", record)
            output.write_bytes(source.read_bytes())
            self.assertEqual(qwen.structure_provenance(source, item, output), record)
            write_json(output.parent / ".reference_edit.json", {"different": "guide"})
            with self.assertRaisesRegex(ValueError, "another pending"):
                qwen.structure_provenance(source, item, output)

    def test_native_modes_are_mutually_exclusive_and_selection_is_bounded(self):
        for args in (["--structure-reference", "layout.png"],
                     ["--ids", "a", "b", "--structure-reference", "layout.png"],
                     ["--ids", "a", "--structure-reference", "layout.png", "--reframe-reference", "a.png"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                qwen.parse_args(args)

    @unittest.skipUnless(importlib.util.find_spec("trimesh"), "trimesh unavailable")
    def test_constructed_motor_and_hub_axes_match_and_rotor_disks_have_clearance(self):
        import numpy as np
        spec = {"profile": "cargo_flat_octocopter", "rotor_layout": "radial", "arm_count": 8, "rotor_count": 8}
        parts, motors = guide.build_parts(spec)
        self.assertEqual(len(motors), 8)
        self.assertEqual(len([p for p in parts if p[0].startswith("boom_") and p[0][5:].isdigit()]), 8)
        self.assertEqual(len([p for p in parts if p[0].startswith("propeller_")]), 8)
        meshes = {name: mesh for name, mesh, _colour in parts}
        for motor in motors:
            np.testing.assert_allclose(motor["centre"][:2], motor["hub_centre"][:2])
            np.testing.assert_allclose(motor["shaft_axis"], [0, 0, 1])
            index = motor["index"]
            np.testing.assert_allclose(meshes[f"motor_{index}"].centroid, motor["centre"], atol=1e-9)
            np.testing.assert_allclose(meshes[f"hub_{index}"].centroid[:2], motor["centre"][:2], atol=1e-9)
        centres = np.array([m["centre"] for m in motors])
        distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=2)
        radii = np.array([np.linalg.norm(meshes[f"propeller_{i}"].vertices[:, :2] - c[:2], axis=1).max()
                          for i, c in enumerate(centres)])
        off_diagonal = ~np.eye(len(centres), dtype=bool)
        self.assertTrue((distances[off_diagonal] > (radii[:, None] + radii[None, :])[off_diagonal]).all())
        with self.assertRaisesRegex(ValueError, "explicit flat eight-boom"):
            guide.build_parts({**spec, "rotor_layout": "coaxial"})
        for name, mesh, _colour in parts:
            if name.startswith("propeller_"):
                self.assertTrue(mesh.is_watertight)
                self.assertGreater(mesh.volume, 0)


if __name__ == "__main__":
    unittest.main()
