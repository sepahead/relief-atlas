"""Native reference edits preserve source identity, alpha and bounded selection."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/local"))

import qwen_imagegen as qwen
from pipeline_utils import IMAGE_MODEL_ID, file_sha256, source_image_metadata, write_json


class TestReferenceReframe(unittest.TestCase):
    def source(self, root, *, override=False):
        from PIL import Image
        folder = Path(root) / "original"
        folder.mkdir()
        path = folder / "a.png"
        image = Image.new("RGBA", (32, 32), (20, 40, 60, 0))
        image.paste((80, 100, 120, 170), (8, 8, 24, 24))
        image.save(path)
        write_json(folder / "image_meta.json", {"id": "a", "image_model": IMAGE_MODEL_ID,
            "image_sha256": file_sha256(path), "policy": {"override": override}})
        return path

    def test_edit_passes_the_original_rgba_to_qwen_and_records_source_hashes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            before = source.read_bytes()
            args = qwen.parse_args(["--ids", "a", "--reframe-reference", str(source),
                                    "--width", "1536", "--height", "1024"])
            record = qwen.reframe_provenance(source, "a", Path(root) / "edited/a.png")
            inputs = qwen.inference_inputs(args, "cargo delivery drone")
            with Image.open(source) as original:
                self.assertEqual(inputs["image"].tobytes(), original.tobytes())
            self.assertEqual(inputs["image"].mode, "RGBA")
            self.assertEqual((inputs["width"], inputs["height"]), (1536, 1024))
            self.assertIn("cargo delivery drone", inputs["prompt"])
            self.assertEqual(record["edit_source_image_sha256"], file_sha256(source))
            self.assertEqual(record["edit_source_metadata_sha256"], file_sha256(source.parent / "image_meta.json"))
            self.assertEqual(source.read_bytes(), before)

    def test_text_generation_has_no_native_edit_input(self):
        inputs = qwen.inference_inputs(qwen.parse_args([]), "water pump")
        self.assertNotIn("image", inputs)
        self.assertEqual(inputs["prompt"], "water pump")

    def test_reframing_does_not_append_a_conflicting_generation_camera_or_layout(self):
        from generation_prompts import compose_prompt
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            args = qwen.parse_args(["--ids", "a", "--reframe-reference", str(source)])
            inputs = qwen.inference_inputs(args, compose_prompt("cargo octocopter with sealed pod", transparent=True))
            self.assertIn("framing only", inputs["prompt"])
            self.assertIn("preserve the source subject, component positions", inputs["prompt"])
            self.assertNotIn("View:", inputs["prompt"])
            self.assertNotIn("Geometry:", inputs["prompt"])

    def test_reframe_refuses_in_place_or_an_unrelated_existing_output(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            with self.assertRaisesRegex(ValueError, "separate output"):
                qwen.reframe_provenance(source, "a", source)
            output = Path(root) / "existing/a.png"
            output.parent.mkdir()
            output.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "another reference"):
                qwen.reframe_provenance(source, "a", output)

    def test_changed_source_or_another_model_cannot_be_reframed_as_verified_qwen(self):
        for change in ("bytes", "model", "id"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                source = self.source(root)
                metadata = source_image_metadata(source.parent, "a")
                if change == "bytes":
                    source.write_bytes(b"replaced reference")
                else:
                    metadata["image_model" if change == "model" else "id"] = "another"
                    write_json(source.parent / "image_meta.json", metadata)
                with self.assertRaisesRegex(ValueError, "hash-verified"):
                    qwen.reframe_provenance(source, "a", Path(root) / "edited/a.png")

    def test_interrupted_edit_resumes_only_with_matching_durable_intent(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root)
            output = Path(root) / "edited/a.png"
            record = qwen.reframe_provenance(source, "a", output)
            # A crash after PNG promotion but before metadata commit leaves
            # this input-bound intent, distinguishing it from an unrelated PNG.
            write_json(output.parent / ".reference_edit.json", record)
            output.write_bytes(source.read_bytes())
            self.assertEqual(qwen.reframe_provenance(source, "a", output), record)
            committed = {**record, "id": "a", "image_model": IMAGE_MODEL_ID,
                         "image_sha256": file_sha256(output), "policy": {"override": False}}
            write_json(output.parent / "image_meta.json", committed)
            (output.parent / ".reference_edit.json").unlink()
            self.assertEqual(qwen.reframe_provenance(source, "a", output), record)
            write_json(output.parent / ".reference_edit.json", {"another": "operation"})
            with self.assertRaisesRegex(ValueError, "another pending edit"):
                qwen.reframe_provenance(source, "a", output)

    def test_generator_recovers_after_png_promotion_before_metadata_commit(self):
        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        class Pipeline:
            def to(self, device):
                return self

            def __call__(self, **inputs):
                return SimpleNamespace(images=[inputs["image"].resize((inputs["width"], inputs["height"]))])

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = self.source(root)
            original = source.read_bytes()
            args = qwen.parse_args(["--ids", "a", "--reframe-reference", str(source),
                                    "--width", "512", "--height", "256"])
            row = {"id": "a", "name": "water pump", "prompt": "water pump",
                   "geography": "general", "category": "equipment"}
            output = root / "outputs/general/equipment/a"
            original_save = qwen.save_json_atomic

            def interrupt_commit(path, data):
                if path.name == "image_meta.json":
                    raise OSError("simulated metadata commit interruption")
                original_save(path, data)

            fake_torch = SimpleNamespace(bfloat16="bf16", Generator=Generator,
                                         inference_mode=contextlib.nullcontext)
            fake_diffusers = SimpleNamespace(QwenImage21Pipeline=SimpleNamespace(
                from_pretrained=lambda *a, **kw: Pipeline()))
            with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}), \
                 patch.object(qwen, "OUTPUT_DIR", root / "outputs"), \
                 patch.object(qwen, "STATE_DIR", root / "state"), \
                 patch.object(qwen, "parse_args", return_value=args), \
                 patch.object(qwen, "select_manifest_items", return_value=[row]), \
                 patch.object(qwen, "generation_provenance", return_value={"test_runtime": True}), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(qwen, "save_json_atomic", side_effect=interrupt_commit):
                    with self.assertRaisesRegex(RuntimeError, "infrastructure failed"):
                        qwen.main()
                self.assertTrue((output / "a.png").exists())
                self.assertFalse((output / "image_meta.json").exists())
                self.assertTrue((output / ".reference_edit.json").exists())
                qwen.main()
            metadata = source_image_metadata(output, "a")
            self.assertEqual(metadata["mode"], "native_reference_reframe")
            self.assertEqual(metadata["edit_source_image_sha256"], file_sha256(source))
            self.assertEqual((metadata["width"], metadata["height"]), (512, 256))
            self.assertFalse((output / ".reference_edit.json").exists())
            self.assertEqual(source.read_bytes(), original)

    def test_pending_edit_recovers_with_a_stale_or_interrupted_metadata_sidecar(self):
        # QA may be interrupted after removing a rejected PNG but before
        # removing its metadata. A subsequent edit may itself stop after PNG
        # promotion. Its durable source-bound intent still owns that output.
        for sidecar in ('{"image_sha256": "previous-generation"}', '{"interrupted":'):
            with self.subTest(sidecar=sidecar), tempfile.TemporaryDirectory() as root:
                source = self.source(root)
                output = Path(root) / "edited/a.png"
                record = qwen.reframe_provenance(source, "a", output)
                write_json(output.parent / ".reference_edit.json", record)
                output.write_bytes(source.read_bytes())
                (output.parent / "image_meta.json").write_text(sidecar)
                self.assertEqual(qwen.reframe_provenance(source, "a", output), record)
                (output.parent / ".reference_edit.json").unlink()
                with self.assertRaisesRegex(ValueError, "another reference"):
                    qwen.reframe_provenance(source, "a", output)

    def test_source_override_cannot_be_laundered_into_default_policy(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.source(root, override=True)
            destination = Path(root) / "edited/a.png"
            with self.assertRaisesRegex(ValueError, "default-policy"):
                qwen.reframe_provenance(source, "a", destination)
            self.assertEqual(qwen.reframe_provenance(source, "a", destination, override=True)["mode"],
                             "native_reference_reframe")

    def test_reframe_requires_one_explicit_id_and_valid_dimensions(self):
        for args in (["--reframe-reference", "a.png"],
                     ["--ids", "a", "b", "--reframe-reference", "a.png"],
                     ["--width", "0"], ["--height", "257"], ["--steps", "0"],
                     ["--guidance", "nan"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    qwen.parse_args(args)


if __name__ == "__main__":
    unittest.main()
