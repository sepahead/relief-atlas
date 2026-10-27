"""Exercise production stage/commit logic with tiny fake models and exports.

All files are temporary. Fake QA is explicitly marked and only tests orchestration;
these tests neither load weights nor establish image, mesh or Gaussian quality.
"""

from collections import Counter
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

import pipeline_utils as pu  # noqa: E402
import prompt_utils  # noqa: E402
import qwen_imagegen as qwen  # noqa: E402
import run_all  # noqa: E402
from gs_export import tmp_sibling  # noqa: E402


class PowerCut(BaseException):
    """Bypass normal error cleanup, as a killed process would."""


class GenerationHarness(contextlib.ExitStack):
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.output, self.state = self.root / "outputs", self.root / "state"
        self.folder = self.output / "general/equipment/a"
        self.row = {"id": "a", "name": "water pump", "prompt": "portable water pump",
                    "geography": "general", "category": "equipment"}
        self.calls, self.interruption = Counter(), None

    def hook(self, boundary):
        if self.interruption == boundary:
            self.interruption = None
            raise PowerCut(boundary)

    def __enter__(self):
        super().__enter__()
        harness = self

        class Generator:
            def __init__(self, _device):
                pass

            def manual_seed(self, _seed):
                return self

        class ImagePipeline:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                harness.calls["image_loads"] += 1
                return cls()

            def to(self, _device):
                return self

            def __call__(self, **_kwargs):
                from PIL import Image
                harness.calls["image_samples"] += 1
                harness.hook("image_sampling")
                image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                image.paste((40, 90, 120, 255), (64, 64, 192, 192))
                return SimpleNamespace(images=[image])

        class MeshPipeline:
            sparse_structure_sampler_params = shape_slat_sampler_params = tex_slat_sampler_params = {}

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                harness.calls["mesh_loads"] += 1
                return cls()

            def to(self, _device):
                return self

            def run(self, _image, **_kwargs):
                harness.calls["mesh_samples"] += 1
                harness.hook("mesh_sampling")
                vertices = SimpleNamespace()
                vertices.cpu = lambda: vertices
                vertices.numpy = lambda: [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
                return SimpleNamespace(vertices=vertices, faces=[[0, 1, 2]])

        trellis, pipelines, rembg = (ModuleType(name) for name in
                                    ("trellis2", "trellis2.pipelines", "trellis2.pipelines.rembg"))
        trellis.__path__ = pipelines.__path__ = []
        trellis.pipelines, pipelines.rembg = pipelines, rembg
        pipeline_module = ModuleType("trellis2.pipelines.trellis2_image_to_3d")
        pipeline_module.Trellis2ImageTo3DPipeline = MeshPipeline
        modules = {"torch": SimpleNamespace(bfloat16="bf16", Generator=Generator, device=str,
                                           inference_mode=contextlib.nullcontext),
                   "diffusers": SimpleNamespace(QwenImage21Pipeline=ImagePipeline),
                   "flex_gemm": ModuleType("flex_gemm"), "trellis2": trellis,
                   "trellis2.pipelines": pipelines, "trellis2.pipelines.rembg": rembg,
                   pipeline_module.__name__: pipeline_module}
        self.enter_context(patch.dict(sys.modules, modules))
        self.enter_context(patch.object(sys, "path", list(sys.path)))
        # Loading the production CLI sets environment defaults; isolate them too.
        import os
        self.enter_context(patch.dict(os.environ))
        spec = importlib.util.spec_from_file_location("_recovery_meshgen", ROOT / "scripts/local/trellis_meshgen.py")
        self.mesh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mesh)
        for module in (qwen, self.mesh, run_all):
            self.enter_context(patch.object(module, "OUTPUT_DIR", self.output))
            self.enter_context(patch.object(module, "STATE_DIR", self.state))
            self.enter_context(patch.object(module, "select_manifest_items", side_effect=lambda *_a: [dict(self.row)]))
        for module in (qwen, self.mesh):
            self.enter_context(patch.object(module, "generation_provenance", return_value={"test_fixture": True}))
            self.enter_context(patch.object(module, "durable_replace", side_effect=self.promote))
            self.enter_context(patch.object(module, "save_json_atomic", side_effect=self.commit))
        self.enter_context(patch.object(self.mesh, "export_glb", side_effect=self.export_glb))
        self.enter_context(patch.object(self.mesh, "export_gaussians_from_mesh_with_voxel", side_effect=self.export_gaussians))
        self.enter_context(patch.object(run_all, "pipeline_lock", side_effect=lambda: pu.pipeline_lock(self.state)))
        self.enter_context(patch.object(run_all, "run", side_effect=self.dispatch))
        self.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return self

    def promote(self, temporary, final):
        pu.durable_replace(temporary, final)
        boundary = {".png": "png_promotion", ".glb": "glb_promotion", ".splat": "gaussian_promotion"}
        self.hook(boundary[Path(final).suffix])

    def commit(self, path, data):
        boundary = {"image_meta.json": "image_metadata", "metadata.json": "mesh_metadata"}.get(Path(path).name)
        self.hook(str(boundary) + "_commit")
        pu.write_json(path, {**data, "test_fixture": True})
        self.hook(str(boundary) + "_committed")

    def export_glb(self, _mesh, path, _texture_size, _target_faces):
        Path(path).write_bytes(b"partial")
        self.hook("glb_write")
        Path(path).write_bytes(b"glTF" + b"test-fixture" * 512)
        return "test_fixture"

    def export_gaussians(self, _mesh, stem, **_kwargs):
        final = Path(stem + ".splat")
        tmp_sibling(final).write_bytes(b"partial")
        self.hook("gaussian_write")
        tmp_sibling(final).write_bytes(b"test-fixture" * 64)
        return [final], 64

    def qa(self, mode):
        self.hook("qa_" + mode)
        verdict = {"pass": True, "qa_version": pu.QA_VERSION, "test_fixture": True,
                   "artifacts": pu.artifact_fingerprints(self.folder, "a", mode)}
        if mode == "images":
            verdict.update(nsfw=0.0, nsfw_screened=True)
        pu.write_json(self.state / f"qa_{mode}.json", {"a": verdict})
        self.hook("qa_" + mode + "_committed")

    def dispatch(self, command):
        script = Path(command[1]).name
        with patch.object(sys, "argv", list(map(str, command[1:]))):
            if script == "qwen_imagegen.py":
                qwen.main()
            elif script == "trellis_meshgen.py":
                self.mesh.main()
            elif script == "qa.py":
                self.qa(command[command.index("--mode") + 1])
            else:
                raise AssertionError(f"Unexpected stage: {script}")

    def reference(self):
        with patch.object(sys, "argv", ["qwen_imagegen.py", "--ids", "a", "--size", "256"]):
            qwen.main()

    def run(self):
        return run_all.main(["--ids", "a", "--batch-size", "1", "--skip-preflight", "--image-size", "256"])

    def certified(self):
        return pu.item_certified(self.folder, "a", run_all.ledgers())


class TestProductionRecovery(unittest.TestCase):
    def test_committed_image_without_any_glb_is_reused_with_or_without_image_qa(self):
        for image_qa in (False, True):
            with self.subTest(image_qa=image_qa), tempfile.TemporaryDirectory() as root, GenerationHarness(root) as job:
                job.reference()
                before = pu.file_sha256(job.folder / "a.png")
                if image_qa:
                    job.qa("images")
                job.calls.clear()
                self.assertEqual(job.run(), 0)
                self.assertEqual(job.calls["image_loads"], 0)
                self.assertEqual(job.calls["mesh_samples"], 1)
                self.assertEqual(pu.file_sha256(job.folder / "a.png"), before)
                self.assertTrue(job.certified())

    def test_partial_geometry_and_missing_or_stale_qa_have_distinct_recovery(self):
        cases = ("glb_only", "gaussians_only", "metadata_only", "no_metadata", "old_source", "missing_qa", "stale_qa")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root, GenerationHarness(root) as job:
                self.assertEqual(job.run(), 0)
                reference_hash = pu.file_sha256(job.folder / "a.png")
                for name in {
                    "glb_only": ["a.splat", "metadata.json"],
                    "gaussians_only": ["a.glb", "metadata.json"],
                    "metadata_only": ["a.glb", "a.splat"],
                    "no_metadata": ["metadata.json"],
                }.get(case, []):
                    (job.folder / name).unlink()
                if case == "old_source":
                    metadata = pu.load_json(job.folder / "metadata.json")
                    pu.write_json(job.folder / "metadata.json", {**metadata, "source_image_sha256": "old-png"})
                if case in ("missing_qa", "stale_qa"):
                    for mode in ("images", "meshes", "gaussians"):
                        path = job.state / f"qa_{mode}.json"
                        if case == "missing_qa":
                            path.unlink()
                        else:
                            data = pu.load_json(path)
                            data["a"]["qa_version"] -= 1
                            pu.write_json(path, data)
                self.assertFalse(job.certified())
                job.calls.clear()
                self.assertEqual(job.run(), 0)
                self.assertEqual(job.calls["image_loads"], 0)
                self.assertEqual(job.calls["mesh_samples"], int(case not in ("missing_qa", "stale_qa")))
                self.assertEqual(pu.file_sha256(job.folder / "a.png"), reference_hash)
                self.assertTrue(job.certified())

    def test_interrupted_generation_exports_and_qa_resume_at_committed_stages(self):
        boundaries = ("image_sampling", "png_promotion", "image_metadata_commit", "image_metadata_committed",
                      "mesh_sampling", "glb_write", "gaussian_write", "glb_promotion", "gaussian_promotion",
                      "mesh_metadata_commit", "mesh_metadata_committed", "qa_images", "qa_images_committed",
                      "qa_meshes", "qa_meshes_committed", "qa_gaussians", "qa_gaussians_committed")
        image_restarts = {"image_sampling", "png_promotion", "image_metadata_commit"}
        mesh_restarts = {"mesh_sampling", "glb_write", "gaussian_write", "glb_promotion",
                         "gaussian_promotion", "mesh_metadata_commit"}
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as root, GenerationHarness(root) as job:
                job.interruption = boundary
                with self.assertRaises(PowerCut):
                    job.run()
                self.assertIsNone(job.interruption, "The injected failure must have been reached")
                self.assertEqual(job.run(), 0)
                self.assertTrue(prompt_utils.mesh_complete(job.folder, "a"))
                self.assertTrue(job.certified())
                self.assertEqual(job.calls["image_samples"], 1 + int(boundary in image_restarts))
                self.assertEqual(job.calls["mesh_samples"], 1 + int(boundary in mesh_restarts))
                # Once complete, another invocation must not load either model.
                before = dict(job.calls)
                self.assertEqual(job.run(), 0)
                self.assertEqual(dict(job.calls), before)


if __name__ == "__main__":
    unittest.main()
