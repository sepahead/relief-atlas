"""Controller invariants plus real subprocess pause/resume/crash recovery."""

import contextlib
import io
import itertools
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
sys.path.insert(0, str(ROOT / "tests"))

import offline_run as offline  # noqa: E402
import offline_fixture as fixture  # noqa: E402
import pipeline_utils as pu  # noqa: E402
import prompt_utils  # noqa: E402
import run_all  # noqa: E402

NATIVE = sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file()


def make_plan(root, ids=None):
    root = Path(root)
    sources = root / "sources"
    sources.mkdir(parents=True)
    rows = [{"id": i, "name": "water pump", "prompt": "portable water pump",
             "geography": "general", "category": "equipment"} for i in ("a", "b")]
    rows += [{"id": "blocked", "name": "rifle", "prompt": "G36 assault rifle",
              "geography": "general", "category": "equipment"}]
    for geo, filename in prompt_utils.MANIFESTS.items():
        pu.write_json(sources / filename, {"meshes": rows if geo == "general" else []})
    profile = offline.load_profile(overrides={"ids": ids, "output_dir": str(root / "outputs"),
        "state_dir": str(root / "pipeline"), "retry_seconds": 0, "max_item_attempts": 2})
    control = root / "control"
    plan = offline.prepare_plan(control, profile, manifest_dir=sources)
    offline.change_control(control, "running", plan["run_id"])
    return control, plan, sources


def wait_until(predicate, *, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for fixture state")


class TestOfflinePlan(unittest.TestCase):
    def test_freezes_only_the_selected_permitted_rows_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, sources = make_plan(root, ["a", "blocked"])
            self.assertEqual(plan["target_ids"], ["a"])
            self.assertEqual(plan["refused_ids"], ["blocked"])
            backend = offline.Backend(plan)
            self.assertEqual([i["id"] for i in backend.items], ["a"])
            pu.write_json(sources / prompt_utils.MANIFESTS["general"], {"meshes": []})
            offline.verify_plan(plan)  # Live catalogue edits cannot broaden/shrink this run.
            snapshot = Path(plan["manifest_dir"]) / prompt_utils.MANIFESTS["general"]
            pu.write_json(snapshot, {"meshes": []})
            with self.assertRaisesRegex(ValueError, "Saved manifest changed"):
                offline.verify_plan(plan)
            self.assertFalse((control / "status.json").exists())

    def test_snapshot_environment_reaches_the_real_selector_and_never_widens_ids(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root, ["b"])
            backend = offline.Backend(plan)
            env = backend.env(control)
            code = "import prompt_utils; print(' '.join(i['id'] for i in prompt_utils.all_items()))"
            env["PYTHONPATH"] = str(ROOT / "scripts/local")
            result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "b")
            command = [str(x) for x in backend.command(backend.items[0])]
            self.assertEqual(command[command.index("--ids") + 1], "b")
            self.assertNotIn("--allow-restricted", command)
            self.assertNotIn("--skip-image-qa", command)
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_status_is_read_only_and_stale_pid_is_not_a_live_worker(self):
        with tempfile.TemporaryDirectory() as root:
            control = Path(root) / "missing"
            self.assertFalse(offline.status_data(control)["worker_running"])
            self.assertFalse(control.exists())
            control, plan, _sources = make_plan(Path(root) / "fixture")
            pu.write_json(control / "status.json", {"phase": "generating", "pid": os.getpid()})
            offline.change_control(control, "paused", plan["run_id"])
            self.assertEqual(offline.status_data(control)["phase"], "paused")
            self.assertFalse(offline.status_data(control)["worker_running"])

    def test_worker_cannot_overwrite_a_later_user_pause_or_resume(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            revision = pu.load_json(control / "control.json")["revision"]
            offline.change_control(control, "paused", plan["run_id"])
            self.assertFalse(offline.finish_control(control, plan, "completed", revision))
            self.assertEqual(pu.load_json(control / "control.json")["desired"], "paused")
            offline.change_control(control, "running", plan["run_id"])
            self.assertFalse(offline.finish_control(control, plan, "attention", revision))
            self.assertEqual(pu.load_json(control / "control.json")["desired"], "running")

    def test_startup_failure_cannot_overwrite_a_later_pause_or_resume(self):
        for desired in ("paused", "running"):
            with self.subTest(desired=desired), tempfile.TemporaryDirectory() as root:
                control, plan, _sources = make_plan(root, ["a"])

                def interrupted_startup():
                    offline.change_control(control, desired, plan["run_id"])
                    raise RuntimeError("simulated startup interruption")

                with patch.object(offline, "require_network_guard", side_effect=interrupted_startup):
                    code = offline.worker_main(control)
                self.assertEqual(pu.load_json(control / "control.json")["desired"], desired)
                self.assertEqual(code, 75 if desired == "running" else 0)

    def test_uncommitted_new_plan_cannot_replace_the_saved_selection(self):
        with tempfile.TemporaryDirectory() as root:
            control, original, sources = make_plan(root, ["a"])
            profile = {**original["profile"], "ids": ["b"]}
            candidate = offline.prepare_plan(control, profile, manifest_dir=sources)
            self.assertNotEqual(original["run_id"], candidate["run_id"])
            self.assertEqual(offline.saved_plan(control)["target_ids"], ["a"])
            offline.change_control(control, "running", candidate["run_id"])
            self.assertEqual(offline.saved_plan(control)["target_ids"], ["b"])

    def test_login_definition_precedes_committed_running_intent(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root, ["a"])
            offline.change_control(control, "paused", plan["run_id"])
            events = []

            def prepare(_control):
                self.assertEqual(pu.load_json(control / "control.json")["desired"], "paused")
                events.append("definition_durable")

            def start(_control):
                self.assertEqual(pu.load_json(control / "control.json")["desired"], "running")
                events.append("start_service")

            with patch.object(offline, "prepare_service", side_effect=prepare), \
                 patch.object(offline, "start_service", side_effect=start), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(offline.main(["--control-dir", str(control), "start"]), 0)
            self.assertEqual(events, ["definition_durable", "start_service"])

    def test_uninstall_preserves_outputs_and_unrelated_agents(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root, ["a"])
            home = Path(root) / "home"
            agents = home / "Library/LaunchAgents"
            agents.mkdir(parents=True)
            owned = agents / (offline.job_label(control) + ".plist")
            owned.write_text("owned service")
            other = agents / "another.service.plist"
            other.write_text("other service")
            output = Path(plan["profile"]["output_dir"])
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("existing asset")
            with patch.object(offline.Path, "home", return_value=home), \
                 patch.object(offline.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(offline.main(["--control-dir", str(control), "uninstall"]), 0)
            self.assertFalse(owned.exists())
            self.assertEqual(other.read_text(), "other service")
            self.assertEqual(marker.read_text(), "existing asset")
            self.assertEqual(pu.load_json(control / "control.json")["desired"], "paused")

    def test_a_later_resume_cancels_service_removal(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root, ["a"])

            def resume(_control, **_kwargs):
                offline.change_control(control, "running", plan["run_id"])

            with patch.object(offline, "show_status", side_effect=resume), \
                 patch.object(offline.subprocess, "run") as service, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(offline.main(["--control-dir", str(control), "uninstall"]), 0)
            service.assert_not_called()
            self.assertEqual(pu.load_json(control / "control.json")["desired"], "running")

    def test_supervisor_enforces_offline_and_cleans_its_process_group(self):
        with tempfile.TemporaryDirectory() as root:
            data = offline.launch_payload(Path(root))
            self.assertEqual(data["ProgramArguments"][0], "/usr/bin/sandbox-exec")
            self.assertIn("(deny network*)", " ".join(data["ProgramArguments"]))
            self.assertEqual(data["KeepAlive"], {"SuccessfulExit": False})
            self.assertFalse(data["AbandonProcessGroup"])
            self.assertTrue(data["RunAtLoad"])
            self.assertNotIn("NetworkState", data)

    def test_invalid_profile_does_not_fall_back_to_a_full_run(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "not found"):
                offline.load_profile(Path(root) / "missing.json")
            path = Path(root) / "bad.json"
            pu.write_json(path, {"allow_blocked": True})
            with self.assertRaisesRegex(ValueError, "Unknown offline"):
                offline.load_profile(path)

    def test_battery_hysteresis_waits_without_starting_a_stage(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            worker = offline.Worker(control, plan, fixture.FixtureBackend(plan))
            power = [{"percent": 10, "ac": False}, {"percent": 20, "ac": False}, {"percent": 30, "ac": False}]
            with patch.object(worker, "battery", side_effect=power), patch.object(worker, "delay") as delay:
                worker.ready_resources()
            self.assertEqual(delay.call_count, 2)
            self.assertFalse(worker.power_wait)
            self.assertIsNone(worker.child)

    def test_battery_hysteresis_survives_restart_and_manual_pause(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            worker = offline.Worker(control, plan, fixture.FixtureBackend(plan))
            with patch.object(worker, "battery", return_value={"percent": 10, "ac": False}), \
                 patch.object(worker, "delay", side_effect=offline.Interrupted):
                with self.assertRaises(offline.Interrupted):
                    worker.ready_resources()
            worker.update("paused")
            offline.change_control(control, "running", plan["run_id"])
            resumed = offline.Worker(control, plan, fixture.FixtureBackend(plan))
            self.assertTrue(resumed.power_wait)
            power = [{"percent": 20, "ac": False}, {"percent": 30, "ac": False}]
            with patch.object(resumed, "battery", side_effect=power), patch.object(resumed, "delay") as delay:
                resumed.ready_resources(check_disk=False)
            delay.assert_called_once_with(10)
            self.assertFalse(resumed.power_wait)

    def test_active_low_battery_is_saved_before_interrupting_the_child(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            worker = offline.Worker(control, plan, fixture.FixtureBackend(plan))
            clock = itertools.count(step=10)
            with patch.object(offline.subprocess, "Popen") as popen, \
                 patch.object(worker, "battery", return_value={"percent": 10, "ac": False}), \
                 patch.object(offline.time, "monotonic", side_effect=lambda: next(clock)), \
                 patch.object(offline.time, "sleep"):
                popen.return_value.poll.return_value = None
                with self.assertRaises(offline.LowPower):
                    worker.execute([sys.executable, "-c", "pass"], "generating")
            status = pu.load_json(control / "status.json")
            self.assertTrue(status["power_wait"])
            self.assertEqual(status["phase"], "waiting_for_power")
            self.assertIsNotNone(worker.child)  # Cleanup follows the durable update.

    def test_gpu_wait_rechecks_power_before_starting_a_stage(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            worker = offline.Worker(control, plan, fixture.FixtureBackend(plan))
            power = [{"percent": 50, "ac": False}, {"percent": 10, "ac": False}, {"percent": 30, "ac": False}]
            with patch.object(offline, "gpu_lock", side_effect=[RuntimeError("busy"), contextlib.nullcontext()]), \
                 patch.object(worker, "battery", side_effect=power), \
                 patch.object(offline.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**12})()), \
                 patch.object(worker, "delay") as delay, patch.object(worker, "stop_children"), \
                 patch.object(offline.subprocess, "Popen"):
                with worker.lease():
                    self.assertFalse(worker.power_wait)
            self.assertEqual([call.args[0] for call in delay.call_args_list], [5, 10])

    def test_logs_include_supervisor_startup_errors_even_without_a_run_log(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            (control / "supervisor.log").write_text("old-line\n" * 4000 + "latest startup error\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(offline.main(["--control-dir", str(control), "logs"]), 0)
            self.assertIn("latest startup error", output.getvalue())
            self.assertLess(len(output.getvalue()), 17000)
            (Path(plan["directory"]) / "run.log").write_text("saved stage checkpoint\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                offline.main(["--control-dir", str(control), "logs"])
            self.assertIn("saved stage checkpoint", output.getvalue())
            self.assertIn("latest startup error", output.getvalue())

    def test_network_guard_requires_actual_socket_denial(self):
        with patch("socket.socket") as socket:
            with self.assertRaisesRegex(RuntimeError, "isolation is not active"):
                offline.require_network_guard()
            self.assertTrue(socket.return_value.bind.called)

    def test_low_disk_wait_does_not_launch_or_delete_anything(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root)
            worker = offline.Worker(control, plan, fixture.FixtureBackend(plan))
            low = type("Disk", (), {"free": 0})()
            enough = type("Disk", (), {"free": 10**12})()
            with patch.object(worker, "battery", return_value=None), \
                 patch.object(worker, "delay") as delay, \
                 patch.object(offline.shutil, "disk_usage", side_effect=[low, enough]):
                worker.ready_resources()
            self.assertEqual(delay.call_count, 1)
            self.assertIsNone(worker.child)
            self.assertFalse((Path(plan["profile"]["output_dir"])).exists())

    def test_gpu_lease_allows_its_group_children_but_excludes_another_job(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "gpu.lock")
            child_code = f"import pipeline_utils as p\nwith p.gpu_lock({path!r}): print('nested')"
            source = ("import subprocess,sys,pipeline_utils as p\n"
                      f"with p.gpu_lock({path!r}):\n"
                      f" a=subprocess.run([sys.executable,'-c',{child_code!r}],capture_output=True,text=True)\n"
                      f" b=subprocess.run([sys.executable,'-c',{child_code!r}],start_new_session=True,capture_output=True,text=True)\n"
                      " print(a.returncode,b.returncode,a.stdout.strip())\n")
            env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts/local")}
            result = subprocess.run([sys.executable, "-c", source], start_new_session=True,
                                    env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "0 1 nested")

    def test_native_guide_is_only_propagated_for_one_id_and_reuses_a_valid_image(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_all.parse_args(["--structure-reference", "guide.png", "--ids", "a", "b"])
        args = run_all.parse_args(["--ids", "a", "--phase", "images", "--structure-reference", "guide.png"])
        row = {"id": "a", "geography": "general", "category": "equipment"}
        with patch.object(run_all, "image_model_matches", return_value=True), \
             patch.object(run_all, "run") as invoke, \
             patch.object(run_all, "ledgers", return_value={"images": {"a": {}}}), \
             patch.object(run_all, "verdict_matches", return_value=True), \
             patch.object(run_all, "load_json", return_value={}):
            run_all.process_batch([row], args)
        self.assertEqual(invoke.call_count, 1)
        self.assertIn("qa.py", str(invoke.call_args.args[0][1]))


@unittest.skipUnless(NATIVE, "macOS network sandbox required")
class TestOfflineProcesses(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.control, self.plan, self.sources = make_plan(self.temporary.name, ["a"])
        self.workers = []

    def tearDown(self):
        offline.change_control(self.control, "paused", self.plan["run_id"])
        for proc in self.workers:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait(timeout=10)
        self.temporary.cleanup()

    def start_worker(self):
        command = ["/usr/bin/sandbox-exec", "-p", offline.NETWORK_PROFILE,
                   sys.executable, str(ROOT / "tests/offline_fixture.py"), "worker", str(self.control)]
        proc = subprocess.Popen(command, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.workers.append(proc)
        return proc

    def test_pause_kills_grandchildren_and_resume_preserves_the_finished_image(self):
        (self.control / "slow").touch()
        first = self.start_worker()
        child = wait_until(lambda: pu.load_json(self.control / "fixture_child.json"))
        image_hash = pu.file_sha256(Path(self.plan["profile"]["output_dir"]) / "general/equipment/a/a.png")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(offline.main(["--control-dir", str(self.control), "pause"]), 0)
        self.assertEqual(first.wait(timeout=10), 0)
        wait_until(lambda: not any(p == child["pid"] for p in offline.group_members(child["group"])))
        self.assertEqual(pu.load_json(Path(self.plan["directory"]) / "progress.json").get("attempts", {}), {})
        (self.control / "allow_finish").touch()
        offline.change_control(self.control, "running", self.plan["run_id"])
        resumed = self.start_worker()
        self.assertEqual(resumed.wait(timeout=20), 0)
        state = pu.load_json(self.control / "status.json")
        self.assertEqual(state["phase"], "completed")
        self.assertEqual(state["counts"]["done"], 1)
        self.assertEqual(pu.load_json(Path(self.plan["profile"]["state_dir"]) / "image_count.json"), {"a": 1})
        self.assertEqual(pu.file_sha256(Path(self.plan["profile"]["output_dir"]) / "general/equipment/a/a.png"), image_hash)

    def test_duplicate_worker_cannot_start_another_inference(self):
        (self.control / "slow").touch()
        self.start_worker()
        wait_until(lambda: pu.load_json(self.control / "fixture_child.json"))
        duplicate = self.start_worker()
        self.assertEqual(duplicate.wait(timeout=10), 0)
        self.assertEqual(pu.load_json(Path(self.plan["profile"]["state_dir"]) / "stage_count.json"), {"a": 1})

    def test_pause_on_second_item_keeps_first_complete_and_finishes_the_saved_selection(self):
        profile = {**self.plan["profile"], "ids": ["a", "b"]}
        self.plan = offline.prepare_plan(self.control, profile, manifest_dir=self.sources)
        offline.change_control(self.control, "running", self.plan["run_id"])
        (self.control / "slow").touch()
        (self.control / "slow_item").write_text("b")
        proc = self.start_worker()
        wait_until(lambda: pu.load_json(self.control / "fixture_child.json"))
        first_folder = Path(profile["output_dir"]) / "general/equipment/a"
        original = {p.name: pu.file_sha256(p) for p in first_folder.iterdir() if p.is_file()}
        self.assertEqual(pu.load_json(self.control / "status.json")["counts"]["done"], 1)
        offline.change_control(self.control, "paused", self.plan["run_id"])
        self.assertEqual(proc.wait(timeout=10), 0)
        (self.control / "allow_finish").touch()
        offline.change_control(self.control, "running", self.plan["run_id"])
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        state = Path(profile["state_dir"])
        self.assertEqual(pu.load_json(state / "stage_count.json"), {"a": 1, "b": 2})
        self.assertEqual(pu.load_json(state / "image_count.json"), {"a": 1, "b": 1})
        self.assertEqual({p.name: pu.file_sha256(p) for p in first_folder.iterdir() if p.is_file()}, original)
        self.assertEqual(pu.load_json(self.control / "status.json")["counts"]["done"], 2)
        release = pu.load_json(Path(self.plan["directory"]) / "release_manifest.json")
        self.assertEqual({row["id"] for row in release["items"]}, {"a", "b"})

    def test_interrupted_final_release_reuses_every_finished_stage(self):
        (self.control / "slow_release").touch()
        proc = self.start_worker()
        wait_until(lambda: pu.load_json(self.control / "fixture_release.json"))
        offline.change_control(self.control, "paused", self.plan["run_id"])
        self.assertEqual(proc.wait(timeout=10), 0)
        self.assertFalse((Path(self.plan["directory"]) / "release_manifest.json").exists())
        (self.control / "allow_release").touch()
        offline.change_control(self.control, "running", self.plan["run_id"])
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        self.assertEqual(pu.load_json(Path(self.plan["profile"]["state_dir"]) / "stage_count.json"), {"a": 1})
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "completed")
        self.assertTrue((Path(self.plan["directory"]) / "release_manifest.json").is_file())

    def test_each_geometry_preset_is_checked_before_its_first_item(self):
        filename = self.sources / prompt_utils.MANIFESTS["general"]
        document = pu.load_json(filename)
        document["meshes"][1]["generation_prompt"] = {
            "specifications": {"prompt": {"profile": "cargo_flat_octocopter"}}}
        document["meshes"].append({**document["meshes"][0], "id": "c"})
        pu.write_json(filename, document)
        profile = {**self.plan["profile"], "ids": ["a", "b", "c"]}
        self.plan = offline.prepare_plan(self.control, profile, manifest_dir=self.sources)
        offline.change_control(self.control, "running", self.plan["run_id"])
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        state = Path(profile["state_dir"])
        self.assertEqual(pu.load_json(state / "preflight_calls.json"), {"ids": ["a", "b"]})
        self.assertEqual(pu.load_json(state / "stage_count.json"), {"a": 1, "b": 1, "c": 1})
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "completed")

    def test_failed_second_preset_keeps_the_first_asset_and_starts_no_second_item(self):
        filename = self.sources / prompt_utils.MANIFESTS["general"]
        document = pu.load_json(filename)
        document["meshes"][1]["generation_prompt"] = {
            "specifications": {"prompt": {"profile": "cargo_flat_octocopter"}}}
        pu.write_json(filename, document)
        profile = {**self.plan["profile"], "ids": ["a", "b"]}
        self.plan = offline.prepare_plan(self.control, profile, manifest_dir=self.sources)
        offline.change_control(self.control, "running", self.plan["run_id"])
        (self.control / "bad_preflight").touch()
        (self.control / "bad_preflight_item").write_text("b")
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        state = Path(profile["state_dir"])
        self.assertEqual(pu.load_json(state / "preflight_calls.json"), {"ids": ["a", "b"]})
        self.assertEqual(pu.load_json(state / "stage_count.json"), {"a": 1})
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "needs_attention")

    def test_pause_is_responsive_during_a_large_saved_asset_scan(self):
        (self.control / "slow_scan").touch()
        proc = self.start_worker()
        wait_until(lambda: pu.load_json(self.control / "status.json").get("phase") == "scanning")
        offline.change_control(self.control, "paused", self.plan["run_id"])
        self.assertEqual(proc.wait(timeout=5), 0)
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "paused")
        self.assertFalse((Path(self.plan["profile"]["state_dir"]) / "stage_count.json").exists())

    def test_pause_escalates_only_its_owned_unresponsive_child(self):
        (self.control / "slow").touch()
        (self.control / "ignore_term").touch()
        proc = self.start_worker()
        child = wait_until(lambda: pu.load_json(self.control / "fixture_child.json"))
        wait_until(lambda: (self.control / "ignoring_ready").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            offline.main(["--control-dir", str(self.control), "pause"])
        self.assertEqual(proc.wait(timeout=10), 0)
        self.assertNotIn(child["pid"], offline.group_members(child["group"]))

    def test_content_quarantine_is_never_relabelled_as_completion(self):
        (self.control / "quarantine").touch()
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        state = pu.load_json(self.control / "status.json")
        self.assertEqual(state["phase"], "needs_attention")
        self.assertEqual(state["counts"]["quarantined"], 1)
        self.assertEqual(pu.load_json(self.control / "control.json")["desired"], "attention")

    def test_failures_are_bounded_and_missing_preflight_never_starts_a_model(self):
        (self.control / "bad_preflight").touch()
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "needs_attention")
        self.assertFalse((Path(self.plan["profile"]["state_dir"]) / "stage_count.json").exists())
        (self.control / "bad_preflight").unlink()
        (self.control / "fail").touch()
        offline.change_control(self.control, "running", self.plan["run_id"], retry=True)
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        self.assertEqual(pu.load_json(Path(self.plan["profile"]["state_dir"]) / "stage_count.json"), {"a": 2})
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "needs_attention")

    def test_corrupted_completed_artifact_is_revalidated_not_skipped(self):
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        folder = Path(self.plan["profile"]["output_dir"]) / "general/equipment/a"
        (folder / "a.glb").write_bytes(b"changed geometry")
        offline.change_control(self.control, "running", self.plan["run_id"])
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        self.assertEqual(pu.load_json(Path(self.plan["profile"]["state_dir"]) / "stage_count.json"), {"a": 2})
        self.assertEqual(pu.load_json(Path(self.plan["profile"]["state_dir"]) / "image_count.json"), {"a": 1})
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "completed")

    def test_finished_work_does_not_wait_for_new_inference_scratch_space(self):
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        profile = {**self.plan["profile"], "disk_reserve_gb": 10**9}
        self.plan = offline.prepare_plan(self.control, profile, manifest_dir=self.sources)
        offline.change_control(self.control, "running", self.plan["run_id"])
        proc = self.start_worker()
        self.assertEqual(proc.wait(timeout=20), 0)
        self.assertEqual(pu.load_json(self.control / "status.json")["phase"], "completed")
        self.assertEqual(pu.load_json(Path(profile["state_dir"]) / "stage_count.json"), {"a": 1})


@unittest.skipUnless(NATIVE, "macOS launchd required")
class TestLaunchdRecovery(unittest.TestCase):
    def test_crash_cleans_the_group_and_login_respects_pause(self):
        with tempfile.TemporaryDirectory() as root:
            control, plan, _sources = make_plan(root, ["a"])
            script = Path(root) / "supervised_fixture.py"
            script.write_text(
                "import sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(ROOT / 'scripts/local')!r})\n"
                f"sys.path.insert(0, {str(ROOT / 'tests')!r})\n"
                "import offline_run as o\nfrom offline_fixture import FixtureBackend\n"
                "c=Path(sys.argv[sys.argv.index('--control-dir')+1])\n"
                "original=o.gpu_lock\no.gpu_lock=lambda:original(c/'fixture_gpu.lock')\n"
                "raise SystemExit(o.worker_main(c,FixtureBackend,poll=0.05))\n")
            payload = offline.launch_payload(control, script=script)
            payload["Label"] = "org.reliefatlas.test." + uuid.uuid4().hex
            payload["ThrottleInterval"] = 1
            plist = Path(root) / "fixture.plist"
            offline.write_plist(plist, payload)
            service = f"gui/{os.getuid()}/{payload['Label']}"

            def launch():
                result = subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode())

            def stop():
                subprocess.run(["launchctl", "bootout", service], capture_output=True)
                # bootout acknowledges removal before launchd necessarily
                # unregisters the job. A fresh login occurs after that teardown;
                # wait for it before reusing the label in this same GUI session.
                wait_until(lambda: subprocess.run(["launchctl", "print", service],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0)

            try:
                (control / "slow").touch()
                launch()
                child = wait_until(lambda: pu.load_json(control / "fixture_child.json"))
                worker = pu.load_json(control / "status.json")["pid"]
                (control / "allow_finish").touch()
                self.assertEqual(os.getpgid(worker), worker)
                os.kill(worker, signal.SIGKILL)
                wait_until(lambda: pu.load_json(control / "status.json").get("phase") == "completed", timeout=30)
                self.assertEqual(pu.load_json(Path(plan["profile"]["state_dir"]) / "image_count.json"), {"a": 1})
                self.assertNotIn(child["pid"], offline.group_members(child["group"]))
                stop()
                offline.change_control(control, "paused", plan["run_id"])
                counts = pu.load_json(Path(plan["profile"]["state_dir"]) / "stage_count.json")
                launch()  # Equivalent RunAtLoad path after logout/reboot/login.
                time.sleep(0.5)
                self.assertFalse(offline.lock_busy(control / ".worker.lock"))
                self.assertEqual(pu.load_json(Path(plan["profile"]["state_dir"]) / "stage_count.json"), counts)
                self.assertEqual(pu.load_json(control / "control.json")["desired"], "paused")
            finally:
                stop()


if __name__ == "__main__":
    unittest.main()
