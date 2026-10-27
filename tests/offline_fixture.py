"""Isolated dummy stages for controller tests. Never invokes a model or API."""

import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

import offline_run as offline  # noqa: E402
from pipeline_utils import (  # noqa: E402
    IMAGE_MODEL_ID, QA_VERSION, artifact_fingerprints, file_sha256, load_json, write_json,
)


def image(folder, item_id, state):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{item_id}.png"
    if not path.exists():
        path.write_bytes(b"controller-test-image; not real inference")
        count = state / "image_count.json"
        data = load_json(count)
        data[item_id] = data.get(item_id, 0) + 1
        write_json(count, data)
    write_json(folder / "image_meta.json", {"id": item_id, "image_model": IMAGE_MODEL_ID,
        "image_sha256": file_sha256(path), "test_fixture": True, "policy": {"override": False}})


def finish(folder, item_id, state):
    image(folder, item_id, state)
    (folder / f"{item_id}.glb").write_bytes(b"glTF" + b"fixture" * 600)
    (folder / f"{item_id}.splat").write_bytes(b"fixture" * 64)
    write_json(folder / "metadata.json", {"id": item_id, "test_fixture": True,
        "source_image_sha256": file_sha256(folder / f"{item_id}.png"), "policy": {"override": False}})
    for mode in ("images", "meshes", "gaussians"):
        path = state / f"qa_{mode}.json"
        data = load_json(path)
        data[item_id] = {"pass": True, "qa_version": QA_VERSION, "test_fixture": True,
                         "artifacts": artifact_fingerprints(folder, item_id, mode)}
        if mode == "images":
            data[item_id].update(nsfw=0.0, nsfw_screened=True)
        write_json(path, data)


class FixtureBackend(offline.Backend):
    def classify(self, items=None, on_item=None):
        control = self.directory.parent.parent
        if (control / "slow_scan").exists():
            for number in range(1000):
                if on_item:
                    on_item(number, 1000)
                time.sleep(0.01)
        return super().classify(items, on_item=on_item)

    def command(self, item, *, preflight=False, guide=None):
        return [sys.executable, "-u", __file__, "preflight" if preflight else "generate",
                str(self.directory.parent.parent), item["id"]]

    def prepare_guide(self, item, execute):
        return None

    def release(self, execute):
        # Use the real, scoped release accounting on deliberately fake test assets.
        control = self.directory.parent.parent
        if (control / "slow_release").exists() and not (control / "allow_release").exists():
            execute([sys.executable, "-u", __file__, "release_wait", str(control), "all"], "release")
        return super().release(execute)


def stage(action, control, item_id):
    control = Path(control)
    plan = load_json(control / "plan.json")
    backend = FixtureBackend(plan)
    state = backend.state
    state.mkdir(parents=True, exist_ok=True)
    if action == "release_wait":
        write_json(control / "fixture_release.json", {"pid": os.getpid(), "group": os.getpgrp()})
        while not (control / "allow_release").exists():
            time.sleep(0.05)
        return 0
    if action == "preflight":
        calls_path = state / "preflight_calls.json"
        calls = load_json(calls_path, {"ids": []})
        calls["ids"].append(item_id)
        write_json(calls_path, calls)
        bad_item = control / "bad_preflight_item"
        fail = ((control / "bad_preflight").exists()
                and (not bad_item.exists() or bad_item.read_text().strip() == item_id))
        write_json(state / "preflight.json", {"ready": not fail, "errors": ["missing fixture dependency"] if fail else []})
        return int(fail)
    attempts = load_json(state / "stage_count.json")
    attempts[item_id] = attempts.get(item_id, 0) + 1
    write_json(state / "stage_count.json", attempts)
    folder = backend.output / "general/equipment" / item_id
    image(folder, item_id, state)
    write_json(control / "fixture_phase.json", {"item": item_id, "pid": os.getpid(), "group": os.getpgrp()})
    if (control / "quarantine").exists():
        write_json(state / "qa_quarantine.json", {item_id: {"reason": "fixture content failure"}})
        return 1
    if (control / "fail").exists():
        return 1
    slow_item = control / "slow_item"
    if ((control / "slow").exists() and not (control / "allow_finish").exists()
            and (not slow_item.exists() or slow_item.read_text().strip() == item_id)):
        code = "import time; time.sleep(120)"
        if (control / "ignore_term").exists():
            code = ("import time,signal;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                    f"Path({str(control / 'ignoring_ready')!r}).touch();time.sleep(120)")
        child = subprocess.Popen([sys.executable, "-c", code])
        write_json(control / "fixture_child.json", {"pid": child.pid, "group": os.getpgrp()})
        while not (control / "allow_finish").exists():
            time.sleep(0.05)
        child.terminate()
        child.wait(timeout=5)
    finish(folder, item_id, state)
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        control = Path(sys.argv[2])
        original = offline.gpu_lock
        offline.gpu_lock = lambda: original(control / "fixture_gpu.lock")
        raise SystemExit(offline.worker_main(control, FixtureBackend, poll=0.05))
    raise SystemExit(stage(*sys.argv[1:]))
