"""Regression tests for scoped runs, live-artifact certification and readiness."""

import contextlib
import io
import importlib
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))

import content_policy
import checkpoint_download
import pipeline_utils as pu
import pending
import preflight
import prepare_models
import prompt_utils
import release_manifest
import run_all


def item(item_id="a", category="equipment"):
    return {"id": item_id, "name": "water pump trailer", "prompt": "water pump trailer",
            "category": category, "geography": "general"}


def make_asset(root, it):
    d = Path(root) / it["geography"] / it["category"] / it["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{it['id']}.png").write_bytes(b"reference image")
    (d / f"{it['id']}.glb").write_bytes(b"glTF" + b"x" * 4096)
    (d / f"{it['id']}.splat").write_bytes(b"x" * 64)
    pu.write_json(d / "image_meta.json", {
        "image_model": pu.IMAGE_MODEL_ID,
        "image_sha256": pu.file_sha256(d / f"{it['id']}.png")})
    pu.write_json(d / "metadata.json", {
        "id": it["id"], "source_image_sha256": pu.file_sha256(d / f"{it['id']}.png"),
        "policy": {"override": False}})
    return d


def ledgers(d, item_id="a"):
    result = {}
    for mode in ("images", "meshes", "gaussians"):
        v = {"qa_version": pu.QA_VERSION, "pass": True,
             "artifacts": pu.artifact_fingerprints(d, item_id, mode)}
        if mode == "images":
            v.update(nsfw=0.0, nsfw_screened=True)
        result[mode] = {item_id: v}
    return result


def exclusion(it, d, data):
    return release_manifest.exclusion_reason(
        it, d, image_qa=data["images"], mesh_qa=data["meshes"],
        gs_qa=data["gaussians"], quarantine=())


class TestArtifactCertification(unittest.TestCase):
    def test_replaced_assets_invalidate_their_saved_pass(self):
        for suffix, reason in (("png", "no current image-QA pass"),
                               ("glb", "no current mesh-QA pass"),
                               ("splat", "no current 3DGS-QA pass")):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as root:
                it = item()
                d = make_asset(root, it)
                data = ledgers(d)
                self.assertIsNone(exclusion(it, d, data))
                path = d / f"a.{suffix}"
                # Same size, changed bytes: existence/size checks cannot detect it.
                path.write_bytes(b"z" * path.stat().st_size)
                self.assertEqual(exclusion(it, d, data), reason)

    def test_unscreened_image_cannot_ship(self):
        with tempfile.TemporaryDirectory() as root:
            it = item()
            d = make_asset(root, it)
            data = ledgers(d)
            data["images"]["a"].update(nsfw=None, nsfw_screened=False)
            self.assertEqual(exclusion(it, d, data), "no current image-QA pass")

    def test_new_image_pass_cannot_certify_an_old_decode(self):
        with tempfile.TemporaryDirectory() as root:
            it = item()
            d = make_asset(root, it)
            data = ledgers(d)
            (d / "a.png").write_bytes(b"new reference")
            pu.write_json(d / "image_meta.json", {"image_model": pu.IMAGE_MODEL_ID,
                          "image_sha256": pu.file_sha256(d / "a.png")})
            data["images"]["a"]["artifacts"] = pu.artifact_fingerprints(d, "a", "images")
            self.assertEqual(exclusion(it, d, data), "mesh source image mismatch")
            self.assertFalse(run_all.phase_complete(it, run_all.parse_args([]), data))

    def test_review_montages_do_not_invalidate_certification(self):
        with tempfile.TemporaryDirectory() as root:
            d = make_asset(root, item())
            data = ledgers(d)
            (d / "a.qa_views.png").write_bytes(b"new montage")
            self.assertTrue(pu.item_certified(d, "a", data))

    def test_unknown_files_cannot_be_release_deliverables(self):
        for name in ("run.log", "api_keys.txt", "notes.json", "a.raw.jpg"):
            self.assertFalse(release_manifest.is_deliverable(Path(name)))

    def test_release_cannot_include_uninspected_neighboring_assets(self):
        with tempfile.TemporaryDirectory() as root:
            it = item()
            d = make_asset(root, it)
            (d / "another_asset.png").write_bytes(b"uninspected image")
            (d / "another_asset.glb").write_bytes(b"uninspected mesh")
            record = release_manifest.describe_item(it, d)
            self.assertNotIn("another_asset.png", record["files"])
            self.assertNotIn("another_asset.glb", record["files"])

    def test_malformed_metadata_is_excluded_without_aborting_release(self):
        for value in ([], {"policy": []}):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as root:
                it = item()
                d = make_asset(root, it)
                pu.write_json(d / "metadata.json", value)
                self.assertEqual(exclusion(it, d, ledgers(d)), "unreadable metadata.json")

    def test_reference_provenance_must_match_the_image_before_it_ships(self):
        with tempfile.TemporaryDirectory() as root:
            it = item()
            d = make_asset(root, it)
            record = {"image_sha256": pu.file_sha256(d / "a.png"),
                      "image_model": "actual model", "prompt": "actual prompt"}
            pu.write_json(d / "image_meta.json", record)
            self.assertEqual(pu.source_image_metadata(d, "a"), record)
            self.assertIn("image_meta.json", release_manifest.describe_item(it, d)["files"])
            (d / "a.png").write_bytes(b"another reference")
            self.assertEqual(pu.source_image_metadata(d, "a"), {})
            self.assertNotIn("image_meta.json", release_manifest.describe_item(it, d)["files"])

    def test_atomic_json_failure_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "state.json"
            pu.write_json(path, {"good": True})
            with self.assertRaises(ValueError):
                pu.write_json(path, {"bad": float("nan")})
            self.assertEqual(pu.load_json(path), {"good": True})
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_second_runner_cannot_take_the_same_lock(self):
        with tempfile.TemporaryDirectory() as root:
            with pu.pipeline_lock(root):
                with self.assertRaises(RuntimeError):
                    with pu.pipeline_lock(root):
                        self.fail("second runner acquired the lock")
            with pu.pipeline_lock(root):
                pass


class TestSelectionAndRunner(unittest.TestCase):
    def test_transparent_prompt_uses_qwen_alpha_instructions_without_a_backdrop(self):
        prompt = prompt_utils.clean_prompt("water pump trailer", "a", transparent=True)
        self.assertTrue(prompt.startswith("This is an RGBA image with transparency."))
        self.assertIn("background is transparent", prompt)
        self.assertIn("10% empty margin on every side", prompt)
        self.assertNotIn("gray seamless background", prompt)

    def test_old_image_model_cannot_satisfy_resume_or_release(self):
        with tempfile.TemporaryDirectory() as root:
            it = item()
            d = make_asset(root, it)
            pu.write_json(d / "image_meta.json", {"image_model": "previous-model",
                "image_sha256": pu.file_sha256(d / "a.png")})
            data = ledgers(d)
            self.assertFalse(pu.item_certified(d, "a", data))
            self.assertEqual(exclusion(it, d, data), "reference image is not from the configured model")
            with patch.object(run_all, "OUTPUT_DIR", Path(root)):
                self.assertFalse(run_all.phase_complete(it, run_all.parse_args([]), data))

    def test_default_manifest_prompts_do_not_repeat(self):
        permitted, _ = content_policy.screen_items(prompt_utils.all_items(), False, False)
        prompts = [prompt_utils.clean_prompt(it["prompt"], it["id"]) for it in permitted]
        self.assertEqual(len(set(prompts)), len(prompts))

    def test_aircraft_variation_is_not_selected_by_its_cargo(self):
        row = prompt_utils.select_manifest_items(ids=["ger_uav_0001"])[0]
        prompt = prompt_utils.clean_prompt(row["prompt"], row["id"])
        self.assertIn("exactly eight separate motor/rotor units", prompt)
        self.assertIn("parallel vertical shafts", prompt)
        self.assertNotIn("strapped to the pallet", prompt)
        self.assertNotIn("hydraulic cylinders", prompt)
        self.assertNotIn("arms folded inward", prompt)

    def test_explicit_ids_override_geography_and_limit(self):
        result = prompt_utils.select_manifest_items(["germany"], ["gen_dis_0002", "eu_rc_0001"], 1)
        self.assertEqual([it["id"] for it in result], ["gen_dis_0002", "eu_rc_0001"])

    def test_unknown_id_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown item IDs"):
            prompt_utils.select_manifest_items(ids=["typo"])

    def test_tier_selection_precedes_limit(self):
        selected = prompt_utils.select_manifest_items(limit=2, only_policy="blocked")
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(content_policy.classify_item(it).classification == "blocked" for it in selected))

    def test_every_batch_phase_receives_exactly_the_same_ids(self):
        args = run_all.parse_args(["--allow-restricted", "--pipeline-type", "1024_cascade",
            "--texture-size", "2048", "--mesh-steps", "16", "--target-faces", "500000"])
        calls = []
        with patch.object(run_all, "run", side_effect=lambda cmd: calls.append(list(map(str, cmd)))), \
             patch.object(run_all, "ledgers", return_value={"images": {}, "meshes": {}, "gaussians": {}}), \
             patch.object(run_all, "load_json", return_value={}), \
             patch.object(run_all, "verdict_matches", return_value=True), \
             patch.object(run_all, "phase_complete", return_value=True):
            run_all.process_batch([item("a"), item("b")], args)
        self.assertEqual(len(calls), 5)
        for command in calls:
            start = command.index("--ids") + 1
            end = next((i for i in range(start, len(command)) if command[i].startswith("--")), len(command))
            self.assertEqual(command[start:end], ["a", "b"])
            self.assertIn("--allow-restricted", command)
        mesh_command = next(cmd for cmd in calls if Path(cmd[1]).name == "trellis_meshgen.py")
        for option, value in [("--pipeline-type", "1024_cascade"), ("--texture-size", "2048"),
                              ("--mesh-steps", "16"), ("--target-faces", "500000")]:
            self.assertEqual(mesh_command[mesh_command.index(option) + 1], value)


class TestModelReadiness(unittest.TestCase):
    def test_a_dropped_shard_resumes_before_the_next_file_starts(self):
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface_hub unavailable")
        spec = {"repo_id": "test/model", "revision": "pinned", "directory": "model",
                "allow_patterns": ["*.json", "*.safetensors"],
                "checkpoint_sha256": {"first.safetensors": "a" * 64,
                                      "second.safetensors": "b" * 64}}
        attempts = {}
        order = []
        def transfer(spec, root, name, token, **kwargs):
            order.append(name)
            attempts[name] = attempts.get(name, 0) + 1
            if name == "first.safetensors" and attempts[name] == 1:
                raise RuntimeError("Connection dropped; resume data retained")
        with tempfile.TemporaryDirectory() as root, \
             patch.object(huggingface_hub, "snapshot_download"), \
             patch.object(checkpoint_download, "download_checkpoint", side_effect=transfer), \
             patch.object(checkpoint_download.time, "sleep"), \
             contextlib.redirect_stdout(io.StringIO()):
            checkpoint_download.download_snapshot(spec, root)
        self.assertEqual(attempts, {"first.safetensors": 2, "second.safetensors": 1})
        self.assertEqual(order, ["first.safetensors", "first.safetensors", "second.safetensors"])

    def test_nested_shard_download_checks_pinned_digest_and_keeps_its_directory(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests unavailable")
        spec = {"repo_id": "test/model", "revision": "pinned", "directory": "model"}
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        checkpoint = struct.pack("<Q", len(header)) + header + b"\0" * 4
        digest = hashlib.sha256(checkpoint).hexdigest()
        with tempfile.TemporaryDirectory() as root:
            class Response:
                history = []
                url = "https://cdn.example/blob"
                headers = {"x-linked-etag": digest, "x-linked-size": str(len(checkpoint))}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def raise_for_status(self): pass
            class Input(io.StringIO):
                def close(self):
                    self.saved = self.getvalue()
                    super().close()
            class Process:
                stdout = []
                def __init__(self): self.stdin = Input()
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self):
                    lines = self.stdin.saved.splitlines()
                    folder = Path(next(line.strip()[4:] for line in lines if line.strip().startswith("dir=")))
                    output = next(line.strip()[4:] for line in lines if line.strip().startswith("out="))
                    (folder / output).write_bytes(checkpoint)
                    return 0
            with patch.object(requests, "get", return_value=Response()) as get, \
                 patch.object(checkpoint_download.shutil, "which", return_value="aria2c"), \
                 patch.object(checkpoint_download.subprocess, "Popen",
                               side_effect=lambda *a, **k: Process()) as popen:
                checkpoint_download.download_checkpoint(spec, root, "text_encoder/shard.safetensors",
                    stage_config=False, expected_sha256=digest,
                    alternative_urls=["https://modelscope.cn/official-shard"])
                self.assertIn("/text_encoder/shard.safetensors", get.call_args.args[0])
                command = popen.call_args.args[0]
                self.assertIn("--split=1", command)
                self.assertIn("--max-connection-per-server=1", command)
                self.assertIn("--max-download-limit=256K", command)
                self.assertIn("--auto-save-interval=10", command)
                self.assertIn("--disk-cache=0", command)
                checkpoint_download.download_checkpoint(spec, root, "text_encoder/shard.safetensors",
                    stage_config=False, expected_sha256=digest, speed_limit_kib=None)
                unlimited_command = popen.call_args.args[0]
                self.assertNotIn("--max-download-limit=256K", unlimited_command)
                self.assertNotIn("--max-overall-download-limit=256K", unlimited_command)
            self.assertEqual((Path(root) / "model/text_encoder/shard.safetensors").read_bytes(), checkpoint)
            self.assertFalse((Path(root) / "model/shard.safetensors").exists())

    def test_server_digest_cannot_override_the_pinned_model_digest(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests unavailable")
        class Response:
            history = []
            headers = {"x-linked-etag": "a" * 64}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
        spec = {"repo_id": "test/model", "revision": "pinned", "directory": "model"}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(requests, "get", return_value=Response()), \
             patch.object(checkpoint_download.shutil, "which", return_value="aria2c"):
            with self.assertRaisesRegex(RuntimeError, "differs from the pinned catalog"):
                checkpoint_download.download_checkpoint(spec, root, "part.safetensors",
                                                       stage_config=False, expected_sha256="b" * 64)

    def test_http_errors_cannot_expose_signed_download_urls(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests unavailable")
        spec = {"repo_id": "test/model", "revision": "pinned", "directory": "model"}
        with tempfile.TemporaryDirectory() as root, \
             patch.object(checkpoint_download.shutil, "which", return_value="aria2c"), \
             patch.object(requests, "get", side_effect=requests.HTTPError(
                 "https://cdn.example/file?signature=SECRET")):
            with self.assertRaisesRegex(RuntimeError, "Official checkpoint request failed") as raised:
                checkpoint_download.download_checkpoint(spec, root, token="test-token")
            self.assertIn("HTTPError", str(raised.exception))
            self.assertIn("[URL]", str(raised.exception))
            self.assertNotIn("cdn.example", str(raised.exception))
            self.assertNotIn("SECRET", str(raised.exception))
            self.assertNotIn("test-token", str(raised.exception))

    def test_bad_download_checksum_cannot_replace_existing_model_or_config(self):
        try:
            import requests
        except ImportError:
            self.skipTest("requests unavailable")
        spec = {"repo_id": "test/model", "revision": "pinned", "directory": "model"}
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "model"
            folder.mkdir()
            (folder / "model.safetensors").write_bytes(b"previous weights")
            (folder / "config.json").write_text('{"previous":true}')
            class Response:
                content = b'{}'
                history = []
                url = "https://cdn.example/checkpoint"
                headers = {"x-linked-etag": hashlib.sha256(b"good").hexdigest(), "x-linked-size": "4"}
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def raise_for_status(self): pass
            class Process:
                stdin = io.StringIO()
                stdout = ["request https://cdn.example/file?signature=SECRET\n"]
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def wait(self):
                    name = ".model.safetensors." + hashlib.sha256(b"good").hexdigest()[:12] + ".download"
                    (folder / name).write_bytes(b"evil")
                    return 0
            output = io.StringIO()
            with patch.object(requests, "get", return_value=Response()), \
                 patch.object(checkpoint_download.shutil, "which", return_value="aria2c"), \
                 patch.object(checkpoint_download.subprocess, "Popen", return_value=Process()), \
                 contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    checkpoint_download.download_checkpoint(spec, root, token="test-token")
            self.assertEqual((folder / "model.safetensors").read_bytes(), b"previous weights")
            self.assertEqual((folder / "config.json").read_text(), '{"previous":true}')
            self.assertNotIn("SECRET", output.getvalue())
            self.assertNotIn("test-token", output.getvalue())

    def test_truncated_checkpoint_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "weights.safetensors"
            header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
            prepare_models.validate_safetensors(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                prepare_models.validate_safetensors(path)

    def test_config_without_weights_is_not_ready(self):
        with tempfile.TemporaryDirectory() as root:
            d = Path(root) / "model"
            d.mkdir()
            (d / "config.json").write_text("{}")
            errors = prepare_models.model_errors({"repo_id": "test/model", "directory": "model",
                "required": ["config.json"], "weight_directories": ["."]}, root)
            self.assertTrue(any("No weights" in error for error in errors))

    def test_missing_dino_is_a_preflight_failure(self):
        with patch.object(preflight, "model_errors", side_effect=lambda spec: ["missing DINO"]
                          if spec.get("gated") else []), \
             patch.object(preflight, "probe_environment", return_value={"errors": [], "spz": True}), \
             patch.object(preflight.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**12})()):
            report = preflight.check(ids=["gen_dis_0001"])
        self.assertFalse(report["ready"])
        self.assertIn("missing DINO", report["errors"])

    def test_compact_storage_budget_is_smaller(self):
        self.assertLess(preflight.storage_estimate(9509, "spz"),
                        preflight.storage_estimate(9509, "both"))

    def test_completed_items_do_not_consume_the_resume_disk_budget(self):
        with patch.object(preflight, "model_errors", return_value=[]), \
             patch.object(preflight, "probe_environment", return_value={"errors": [], "spz": True}), \
             patch.object(preflight.shutil, "disk_usage", return_value=type("Disk", (), {"free": 1_000_000})()):
            report = preflight.check(ids=["gen_dis_0001"], pending_count=0)
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual(report["disk"]["budget_bytes"], 0)


class TestUnattendedCompletion(unittest.TestCase):
    def test_raw_complete_without_current_gates_is_pending(self):
        with tempfile.TemporaryDirectory() as root:
            it = item()
            d = make_asset(root, it)
            result = pending.classify([it], root, certified=True)
            self.assertEqual(result[pending.PENDING], ["a"])
            result = pending.classify([it], root, certified=True, ledgers=ledgers(d))
            self.assertEqual(result[pending.DONE], ["a"])

    def test_quarantine_prevents_a_success_exit(self):
        buckets = {bucket: [] for bucket in pending.BUCKETS}
        buckets[pending.QUARANTINED] = ["a"]
        with patch.object(pending, "report", return_value=buckets), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pending.main(["--certified", "--require-complete"]), 1)

    def test_driver_stops_at_preflight_and_never_calls_generation(self):
        with tempfile.TemporaryDirectory() as root:
            stub = Path(root) / "python3"
            log = Path(root) / "calls"
            stub.write_text('#!/bin/bash\n' +
                            'printf "%s\\n" "$*" >> "$TEST_CALL_LOG"\n' +
                            'exit 1\n')
            stub.chmod(0o755)
            env = {**os.environ, "PATH": root + os.pathsep + os.environ["PATH"],
                   "RELIEF_STATE_DIR": root, "TEST_CALL_LOG": str(log)}
            result = subprocess.run(["bash", str(pu.PROJECT_DIR / "scripts/local/run_forever.sh")],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("--preflight-only", calls[0])


class TestImageQaRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            for module in ("numpy", "PIL", "scipy", "imagehash"):
                importlib.import_module(module)
        except ImportError:
            raise unittest.SkipTest("image QA dependencies unavailable")
        import qa
        cls.qa = qa

    def make_png(self, path, full=False):
        import numpy as np
        from PIL import Image
        arr = np.zeros((128, 128, 4), dtype=np.uint8)
        yy, xx = np.mgrid[:128, :128]
        arr[..., :3] = ((((xx // 2 + yy // 2) % 2) * 200 + 20)[..., None])
        arr[..., 3] = 255 if full else 0
        if not full:
            arr[32:96, 32:96, 3] = 255
        Image.fromarray(arr).save(path)

    def test_localized_clipped_tip_is_rejected_even_when_perimeter_fraction_is_small(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "thin_tip.png"
            pixels = np.zeros((128, 128, 4), dtype=np.uint8)
            pixels[32:96, 32:96] = (220, 220, 220, 255)
            pixels[63, :33] = (220, 220, 220, 255)
            Image.fromarray(pixels).save(path)
            scores = self.qa.image_scores(path)
        self.assertLess(scores["border_touch"], self.qa.BORDER_TOUCH_MAX)
        self.assertTrue(any(r.startswith("touches border") for r in self.qa.image_pass(scores, [])))

    def test_infrastructure_failures_are_distinct_from_content_rejections(self):
        self.assertTrue(pu.is_infrastructure_error(RuntimeError("MPS backend out of memory")))
        self.assertTrue(pu.is_infrastructure_error(BrokenPipeError("Broken pipe")))
        self.assertFalse(pu.is_infrastructure_error(RuntimeError("empty backdrop mask")))

    def run_qa(self, output, state, selected, corpus, *, no_wipe=False):
        args = type("Args", (), dict(rescan=True, no_nsfw=False, no_wipe=no_wipe,
                                     allow_restricted=False, allow_blocked=False, only_policy=None))()
        with patch.object(self.qa, "OUTPUT_DIR", Path(output)), \
             patch.object(self.qa, "STATE_DIR", Path(state)), \
             patch.object(self.qa, "select_items", return_value=selected), \
             patch.object(self.qa, "all_items", return_value=corpus), \
             patch.object(self.qa, "NsfwGuard") as guard, \
             patch.object(content_policy, "STATE_DIR", Path(state)), \
             contextlib.redirect_stdout(io.StringIO()):
            guard.return_value.score.return_value = 0.0
            self.qa.run_images(args)
        return pu.load_json(Path(state) / "qa_images.json")

    def test_rescan_does_not_reject_itself(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            d = make_asset(output, item())
            self.make_png(d / "a.png")
            data = ledgers(d)["images"]
            data["a"].update(self.qa.image_scores(d / "a.png"))
            pu.write_json(state / "qa_images.json", data)
            result = self.run_qa(output, state, [item()], [item()])
            self.assertTrue(result["a"]["pass"], result["a"]["reasons"])
            self.assertTrue((d / "a.png").exists())

    def test_another_batch_remains_a_duplicate_reference(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            a, b = item("a"), item("b")
            for it in (a, b):
                self.make_png(make_asset(output, it) / f"{it['id']}.png")
            self.run_qa(output, state, [a], [a, b])
            result = self.run_qa(output, state, [b], [a, b], no_wipe=True)
            self.assertTrue(result["a"]["pass"])
            self.assertIn("near-duplicate subject", result["b"]["reasons"])

    def test_rejected_reference_invalidates_its_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            d = make_asset(output, item())
            self.make_png(d / "a.png", full=True)
            result = self.run_qa(output, state, [item()], [item()])
            self.assertFalse(result["a"]["pass"])
            for name in ("a.png", "a.glb", "a.splat", "metadata.json"):
                self.assertFalse((d / name).exists(), name)

    def test_no_wipe_preserves_rejected_images_and_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            d = make_asset(output, item())
            self.make_png(d / "a.png", full=True)
            result = self.run_qa(output, state, [item()], [item()], no_wipe=True)
            self.assertFalse(result["a"]["pass"])
            self.assertTrue((d / "a.png").exists())
            self.assertTrue((d / "a.glb").exists())
            self.assertEqual(pu.load_json(state / "qa_retries.json"), {})


class TestMattingAndGaussians(unittest.TestCase):
    def test_native_qwen_alpha_is_preserved_without_rematting(self):
        import matting
        import numpy as np
        from PIL import Image
        pixels = np.zeros((16, 16, 4), dtype=np.uint8)
        pixels[4:12, 4:12] = (40, 80, 120, 170)
        image = Image.fromarray(pixels)
        with patch.object(matting, "matte_image", side_effect=AssertionError("native alpha re-matted")):
            result, stats = matting.reference_rgba(image)
        np.testing.assert_array_equal(np.asarray(result), pixels)
        self.assertEqual(stats["method"], "native_rgba")
        self.assertEqual(stats["fg_fraction"], 0.25)

    def test_opaque_image_still_uses_matting_fallback(self):
        import matting
        from PIL import Image
        opaque = Image.new("RGBA", (8, 8), (40, 80, 120, 255))
        expected = (opaque, {"method": "fallback"})
        with patch.object(matting, "matte_image", return_value=expected) as matte:
            self.assertEqual(matting.reference_rgba(opaque), expected)
            matte.assert_called_once_with(opaque)

    def test_gltf_conversion_matches_the_metal_baker_and_preserves_input(self):
        import numpy as np
        from gs_export import gltf_vertices
        vertices = np.array([[1., 2., 3.], [0., 0., 1.]])
        original = vertices.copy()
        converted = gltf_vertices(vertices)
        np.testing.assert_array_equal(converted, [[1., 3., -2.], [0., 1., 0.]])
        np.testing.assert_array_equal(vertices, original)
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), np.linalg.norm(converted, axis=1))

    @classmethod
    def setUpClass(cls):
        try:
            for module in ("numpy", "scipy"):
                importlib.import_module(module)
            import matting
            import qa
            import gs_export
        except ImportError:
            raise unittest.SkipTest("numpy/scipy/PIL unavailable")
        cls.matting, cls.qa, cls.gs = matting, qa, gs_export

    def test_integer_label_border_is_a_boolean_pixel_mask(self):
        import numpy as np
        labels = np.ones((8, 8), dtype=np.int32)
        labels[3:5, 3:5] = 2
        mask = self.matting._border_touch_mask(labels)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(set(labels[mask]), {1})
        self.assertFalse(mask[3, 3])

    def test_small_enclosed_hole_is_filled(self):
        import numpy as np
        rgb = np.zeros((64, 64, 3), dtype=np.float32)
        fg = np.zeros((64, 64), dtype=np.float32)
        fg[8:56, 8:56] = 1
        fg[30:33, 30:33] = 0
        image, stats = self.matting._finish_rgba(rgb, fg, None)
        self.assertEqual(stats["holes_filled"], 1)
        self.assertEqual(np.asarray(image)[31, 31, 3], 255)

    def test_white_objects_and_white_backdrop_residue_are_distinguished(self):
        scores = {"vertices": 1000, "aspect": 1.0, "dim_ratio": 2.0,
                  "tex_white": 0.8, "tex_magenta": 0.0, "degenerate_frac": 0.0}
        self.assertTrue(any("white slab" in r for r in self.qa.mesh_pass(scores)))
        self.assertEqual(self.qa.mesh_pass({**scores, "source_white_frac": 0.75}), [])
        self.assertTrue(any("white slab" in r for r in self.qa.mesh_pass(
            {**scores, "source_white_frac": 0.1})))

    def test_degenerate_triangles_are_a_mesh_defect(self):
        scores = {"vertices": 1000, "aspect": 1.0, "dim_ratio": 2.0,
                  "tex_white": 0.0, "tex_magenta": 0.0, "degenerate_frac": 0.1}
        self.assertTrue(any("degenerate triangles" in r for r in self.qa.mesh_pass(scores)))

    def test_splat_preserves_quaternion_component_order(self):
        import numpy as np
        quat = np.array([[math.sqrt(0.5), math.sqrt(0.5), 0, 0]], dtype=np.float32)
        g = {"x": np.zeros(1), "y": np.zeros(1), "z": np.zeros(1),
             "scale": np.log([[1., 2., 3.]]), "f_dc": np.zeros((1, 3)),
             "opacity": np.ones((1, 1)), "rot": quat}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "g.splat"
            self.gs.write_splat(path, g)
            loaded = self.qa._load_splat_gaussians(path)
        np.testing.assert_allclose(loaded["rot"], quat, atol=0.02)

    def test_rotated_covariance_uses_squared_scales(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        scale = torch.tensor([[2., 3., 1.]])
        quat = torch.tensor([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]])
        result = self.qa.gaussian_covariance(scale, quat)
        torch.testing.assert_close(result[0], torch.diag(torch.tensor([9., 4., 1.])), atol=1e-5, rtol=1e-5)

    def test_corrupt_gaussian_container_records_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            state.mkdir()
            d = make_asset(output, item())
            (d / "a.splat").write_bytes(b"truncated")
            args = type("Args", (), {"rescan": True, "no_wipe": True})()
            with patch.object(self.qa, "OUTPUT_DIR", output), patch.object(self.qa, "STATE_DIR", state), \
                 patch.object(self.qa, "select_items", return_value=[item()]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.qa.run_gaussians(args)
            result = pu.load_json(state / "qa_gaussians.json")["a"]
            self.assertFalse(result["pass"])
            self.assertTrue(any(r.startswith("load error") for r in result["reasons"]))

    def test_renderer_failure_keeps_assets_and_retry_budget(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            state.mkdir()
            d = make_asset(output, item())
            args = type("Args", (), {"rescan": True, "no_wipe": False})()
            with patch.object(self.qa, "OUTPUT_DIR", output), patch.object(self.qa, "STATE_DIR", state), \
                 patch.object(self.qa, "select_items", return_value=[item()]), \
                 patch.object(self.qa, "load_gaussians", return_value=({}, "splat")), \
                 patch.object(self.qa, "gaussian_primitive_stats", return_value=({"gaussians": 2}, [])), \
                 patch.object(self.qa, "render_splat_views", side_effect=RuntimeError("GPU unavailable")), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "infrastructure failed"):
                    self.qa.run_gaussians(args)
            result = pu.load_json(state / "qa_gaussians.json")["a"]
            self.assertEqual(result["view_error"], "GPU unavailable")
            self.assertFalse(pu.verdict_matches(result, d, "a", "gaussians"))
            self.assertTrue((d / "a.splat").is_file())
            self.assertTrue((d / "metadata.json").is_file())
            self.assertEqual(pu.load_json(state / "qa_retries.json"), {})

    def test_legacy_spz_encoding_requires_repair_without_wiping_or_retries(self):
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            state.mkdir()
            d = make_asset(output, item())
            (d / "a.spz").write_bytes(b"legacy SPZ")
            metadata = pu.load_json(d / "metadata.json")
            metadata["mesh_model"] = "local/trellis.2-4b"
            pu.write_json(d / "metadata.json", metadata)
            pu.write_json(state / "qa_retries.json", {"a": 1})
            args = type("Args", (), {"rescan": True, "no_wipe": False})()
            with patch.object(self.qa, "OUTPUT_DIR", output), patch.object(self.qa, "STATE_DIR", state), \
                 patch.object(self.qa, "select_items", return_value=[item()]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "colour encoding is unverified"):
                    self.qa.run_gaussians(args)
            self.assertFalse(pu.load_json(state / "qa_gaussians.json")["a"]["pass"])
            self.assertEqual(pu.load_json(state / "qa_retries.json"), {"a": 1})
            self.assertTrue((d / "a.spz").exists())
            self.assertTrue((d / "a.glb").exists())

    def test_sampled_render_failure_is_confirmed_on_the_complete_cloud(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as root:
            output, state = Path(root) / "outputs", Path(root) / "state"
            d = make_asset(output, item())
            args = type("Args", (), {"rescan": True, "no_wipe": False})()
            stats = {"coverage_min": 0.2, "coverage_mean": 0.4, "solidity_min": 0.98,
                     "holes_max": 0.31, "holes_over_count": 1, "iou_min": 0.8, "views": 10}
            with patch.object(self.qa, "OUTPUT_DIR", output), patch.object(self.qa, "STATE_DIR", state), \
                 patch.object(self.qa, "select_items", return_value=[item()]), \
                 patch.object(self.qa, "GS_MAX_GAUSSIANS", 5), \
                 patch.object(self.qa, "load_gaussians", return_value=({"x": np.zeros(6)}, "splat")), \
                 patch.object(self.qa, "gaussian_primitive_stats", return_value=({"gaussians": 6}, [])), \
                 patch.object(self.qa, "render_splat_views", return_value=([], [], [])) as render, \
                 patch.object(self.qa, "silhouette_view_metrics", side_effect=[stats, {**stats, "holes_max": 0.29}]), \
                 patch.object(self.qa, "write_view_montage"), contextlib.redirect_stdout(io.StringIO()):
                self.qa.run_gaussians(args)
            self.assertEqual(render.call_count, 2)
            self.assertEqual(render.call_args.kwargs, {"max_gaussians": 6})
            result = pu.load_json(state / "qa_gaussians.json")["a"]
            self.assertTrue(result["pass"])
            self.assertTrue(result["refined_render"])
            self.assertEqual(result["rendered_gaussians"], 6)
            self.assertTrue((d / "a.splat").is_file())
            self.assertEqual(pu.load_json(state / "qa_retries.json"), {})


if __name__ == "__main__":
    unittest.main()
