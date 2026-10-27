"""Offline start/pause/resume controller, supervised by the user's macOS launchd.

Selection and settings are snapshotted before starting. Only verified artifacts
count as finished; a process checkpoint is never an asset certification.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from pipeline_utils import (
    PROJECT_DIR, LOCAL_DIR, durable_replace, file_sha256, gpu_lock, load_json, write_json,
)

DEFAULT_CONTROL = PROJECT_DIR / "state" / "offline_controller"
DEFAULT_PROFILE = PROJECT_DIR / "config" / "offline_run.json"
NETWORK_PROFILE = "(version 1)(allow default)(deny network*)"
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "PIP_NO_INDEX": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1",
}
GENERATION_KEYS = ("pipeline_type", "texture_size", "mesh_steps", "target_faces", "gs_format",
                   "image_steps", "image_guidance", "image_size")


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def lock_busy(path):
    """Observe the OS lock without trusting stale PID files or creating files."""
    try:
        with Path(path).open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(stream, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    return False


def load_profile(path=DEFAULT_PROFILE, overrides=None):
    if not Path(path).is_file():
        raise ValueError(f"Offline profile not found: {path}")
    defaults = load_json(DEFAULT_PROFILE)
    profile = load_json(path)
    if not isinstance(profile, dict):
        raise ValueError("Offline profile must be a JSON object")
    if set(profile) - set(defaults):
        raise ValueError("Unknown offline profile option(s): " + ", ".join(sorted(set(profile) - set(defaults))))
    profile = {**defaults, **profile, **(overrides or {})}
    if profile["version"] != 1:
        raise ValueError("Unsupported offline profile version")
    for key in ("image_steps", "target_faces", "max_item_attempts"):
        if type(profile[key]) is not int or profile[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("limit", "mesh_steps"):
        if profile[key] is not None and (type(profile[key]) is not int or profile[key] < 1):
            raise ValueError(f"{key} must be positive or null")
    for key in ("pipeline_type", "guided_pipeline_type"):
        if profile[key] not in ("512", "1024", "1024_cascade"):
            raise ValueError(f"Invalid {key}")
    if profile["texture_size"] not in (512, 1024, 2048) or profile["gs_format"] not in ("spz", "splat", "both"):
        raise ValueError("Invalid texture size or Gaussian format")
    if type(profile["image_size"]) is not int or profile["image_size"] < 256 or profile["image_size"] % 16:
        raise ValueError("image_size must be at least 256 and divisible by 16")
    for key in ("image_guidance", "disk_reserve_gb", "retry_seconds"):
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if type(profile["structure_guides"]) is not bool:
        raise ValueError("structure_guides must be boolean")
    if (any(type(profile[k]) is not int for k in ("low_battery_percent", "resume_battery_percent"))
            or not 0 <= profile["low_battery_percent"] < profile["resume_battery_percent"] <= 100):
        raise ValueError("Battery thresholds must satisfy 0 <= low < resume <= 100")
    from prompt_utils import MANIFESTS
    if not isinstance(profile["geographies"], list) or not profile["geographies"] or set(profile["geographies"]) - set(MANIFESTS):
        raise ValueError("Invalid geographies")
    if profile["ids"] is not None and (not isinstance(profile["ids"], list) or not profile["ids"]
                                      or any(not isinstance(i, str) for i in profile["ids"])):
        raise ValueError("ids must be a nonempty list or null")
    for key in ("output_dir", "state_dir"):
        profile[key] = str((PROJECT_DIR / profile[key]).resolve())
    return profile


def source_selection(profile, manifest_dir=None):
    import content_policy
    from prompt_utils import MANIFESTS

    manifest_dir = Path(manifest_dir or PROJECT_DIR / "manifests")
    documents, source_hashes, rows = {}, {}, []
    for geo, filename in MANIFESTS.items():
        raw = (manifest_dir / filename).read_bytes()
        document = json.loads(raw)
        if not isinstance(document.get("meshes"), list):
            raise ValueError(f"Invalid manifest: {filename}")
        documents[geo] = document
        source_hashes[filename] = hashlib.sha256(raw).hexdigest()
        rows += document["meshes"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate item IDs across source manifests")
    if profile["ids"]:
        by_id = {r["id"]: r for r in rows}
        unknown = set(profile["ids"]) - set(by_id)
        if unknown:
            raise ValueError("Unknown item IDs: " + ", ".join(sorted(unknown)))
        selected = [by_id[i] for i in dict.fromkeys(profile["ids"])]
    else:
        selected = [r for r in rows if r["geography"] in profile["geographies"]]
        if profile["limit"]:
            selected = selected[:profile["limit"]]
    permitted, refused = content_policy.screen_items([dict(r) for r in selected], False, False)
    return documents, source_hashes, permitted, refused


def prepare_plan(control, profile, *, manifest_dir=None):
    """Freeze only the selected permitted rows, in every phase's manifest format."""
    import content_policy
    from prompt_utils import MANIFESTS
    documents, source_hashes, permitted, refused = source_selection(profile, manifest_dir)
    targets = {r["id"] for r in permitted}
    if not targets:
        raise ValueError("The selection contains no default-policy eligible items")
    for row in permitted:
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", row[key]) for key in ("id", "geography", "category")):
            raise ValueError(f"Unsafe output identity: {row['id']}")
    run_id = uuid.uuid4().hex
    directory = Path(control) / "runs" / run_id
    snapshots = directory / "manifests"
    hashes = {}
    for geo, filename in MANIFESTS.items():
        document = {**documents[geo], "meshes": [r for r in documents[geo]["meshes"] if r["id"] in targets]}
        document.update(total_items=len(document["meshes"]), offline_run_id=run_id)
        write_json(snapshots / filename, document)
        hashes[filename] = file_sha256(snapshots / filename)
    plan = {"version": 1, "run_id": run_id, "created_at": now(), "profile": profile,
            "directory": str(directory), "manifest_dir": str(snapshots), "manifest_sha256": hashes,
            "source_manifest_sha256": source_hashes, "target_ids": [r["id"] for r in permitted],
            "refused_ids": [r["id"] for r in refused], "policy_version": content_policy.POLICY_VERSION}
    write_json(directory / "plan.json", plan)
    write_json(Path(control) / "plan.json", plan)
    return plan


def verify_plan(plan):
    if plan.get("version") != 1 or not plan.get("target_ids"):
        raise ValueError("Missing or unsupported saved run; use Start to prepare a run")
    from prompt_utils import MANIFESTS
    if set(plan["manifest_sha256"]) != set(MANIFESTS.values()):
        raise ValueError("Saved manifest inventory is incomplete")
    for filename, expected in plan["manifest_sha256"].items():
        if file_sha256(Path(plan["manifest_dir"]) / filename) != expected:
            raise ValueError(f"Saved manifest changed: {filename}; original run is preserved")


def saved_plan(control):
    """The committed control record selects a plan, even if a new preparation died."""
    control = Path(control)
    run_id = load_json(control / "control.json").get("run_id")
    if run_id:
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid saved run identity")
        path = control / "runs" / run_id / "plan.json"
        if not path.is_file():
            raise ValueError("The committed run plan is missing; existing assets are preserved")
        return load_json(path)
    return load_json(control / "plan.json")


def change_control(control, desired, run_id=None, *, retry=False):
    control = Path(control)
    with exclusive(control / ".control.lock"):
        old = load_json(control / "control.json")
        data = {"desired": desired, "run_id": run_id or old.get("run_id"), "at": now(),
                "revision": old.get("revision", 0) + 1,
                "retry_epoch": old.get("retry_epoch", 0) + int(retry)}
        write_json(control / "control.json", data)
    return data


def finish_control(control, plan, desired, revision=None):
    with exclusive(Path(control) / ".control.lock"):
        data = load_json(Path(control) / "control.json")
        if (data.get("desired") == "running" and data.get("run_id") == plan["run_id"]
                and (revision is None or data.get("revision") == revision)):
            data.update(desired=desired, at=now(), revision=data.get("revision", 0) + 1)
            write_json(Path(control) / "control.json", data)
            return True
    return False


def job_label(control):
    suffix = hashlib.sha256(str(Path(control).resolve()).encode()).hexdigest()[:12]
    return "org.reliefatlas.offline." + suffix


def launch_payload(control, *, python=None, script=None):
    control = Path(control).resolve()
    return {
        "Label": job_label(control),
        "ProgramArguments": ["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
                             str(python or sys.executable), "-u", str(script or Path(__file__).resolve()),
                             "--control-dir", str(control), "_worker"],
        "WorkingDirectory": str(PROJECT_DIR),
        "EnvironmentVariables": {**OFFLINE_ENV, "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
        "StandardOutPath": str(control / "supervisor.log"), "StandardErrorPath": str(control / "supervisor.log"),
        "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 30,
        "ExitTimeOut": 20, "AbandonProcessGroup": False, "ProcessType": "Standard",
    }


def write_plist(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(plistlib.dumps(payload))
    try:
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_service(control):
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("The offline background service requires macOS and sandbox-exec")
    control = Path(control)
    path = Path.home() / "Library/LaunchAgents" / (job_label(control) + ".plist")
    write_plist(path, launch_payload(control))
    return path


def start_service(control):
    control = Path(control)
    path = prepare_service(control)
    service = f"gui/{os.getuid()}/{job_label(control)}"
    existing = subprocess.run(["launchctl", "print", service], capture_output=True, text=True)
    loaded = existing.returncode == 0
    state = re.search(r"^\s*state = (.+)$", existing.stdout, re.M)
    subprocess.run(["launchctl", "enable", service], check=True, capture_output=True)
    if not loaded:
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True, capture_output=True)
    elif not lock_busy(control / ".worker.lock") and not (state and state[1] == "running"):
        # No -k: a second click must never kill a live worker or its GPU work.
        subprocess.run(["launchctl", "kickstart", service], check=True, capture_output=True)


def group_members(group):
    """PID-only process information; never read or print another process's argv."""
    if sys.platform == "darwin":
        # /bin/ps is setuid on macOS and cannot execute inside a sandbox.
        # libproc exposes the owned group directly, without inspecting argv.
        import ctypes
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_listpids.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
        library.proc_listpids.restype = ctypes.c_int
        library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        library.proc_pidinfo.restype = ctypes.c_int
        size = library.proc_listpids(2, group, None, 0)  # PROC_PGRP_ONLY
        if size < 0:
            raise OSError(ctypes.get_errno(), "Cannot inspect this worker's process group")
        buffer = (ctypes.c_int * (size // ctypes.sizeof(ctypes.c_int) + 64))()
        used = library.proc_listpids(2, group, buffer, ctypes.sizeof(buffer))
        if used < 0:
            raise OSError(ctypes.get_errno(), "Cannot inspect this worker's process group")
        result = []
        for pid in buffer[:used // ctypes.sizeof(ctypes.c_int)]:
            if pid <= 0 or pid == os.getpid():
                continue
            info = ctypes.create_string_buffer(256)
            count = library.proc_pidinfo(pid, 3, 0, info, len(info))  # PROC_PIDTBSDINFO
            # proc_bsdinfo begins with uint32 flags, uint32 status. Ignore
            # zombies, which no longer hold GPU work and await their reaper.
            if count >= 8 and ctypes.c_uint32.from_buffer(info, 4).value == 5:
                continue
            try:
                if os.getpgid(pid) == group:
                    result.append(pid)
            except ProcessLookupError:
                pass
        return result
    with subprocess.Popen(["/bin/ps", "-axo", "pid=,pgid=,stat="], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True) as proc:
        output, _ = proc.communicate(timeout=5)
        excluded = {os.getpid(), proc.pid}
    return [int(parts[0]) for line in output.splitlines() if len(parts := line.split()) == 3
            and int(parts[1]) == group and int(parts[0]) not in excluded and "Z" not in parts[2]]


class Interrupted(Exception):
    pass


class LowPower(Interrupted):
    pass


def require_network_guard():
    """Prove socket use is denied, using loopback bind only (no packets sent)."""
    import errno
    import socket

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
    except OSError as exc:
        if exc.errno in (errno.EPERM, errno.EACCES):
            return
        raise RuntimeError("Cannot verify the offline network guard") from exc
    finally:
        if sock:
            sock.close()
    raise RuntimeError("Network isolation is not active; launch with Start.command or atlas start")


class Backend:
    def __init__(self, plan):
        self.plan, self.profile = plan, plan["profile"]
        self.state = Path(self.profile["state_dir"])
        self.output = Path(self.profile["output_dir"])
        self.directory = Path(plan["directory"])
        self.python = LOCAL_DIR / "imgenv/bin/python"
        self.mesh_python = LOCAL_DIR / "trellis-mac/.venv/bin/python"
        self.items = []
        for filename in plan["manifest_sha256"]:
            self.items += load_json(Path(plan["manifest_dir"]) / filename)["meshes"]
        by_id = {i["id"]: i for i in self.items}
        if set(by_id) != set(plan["target_ids"]) or len(by_id) != len(self.items):
            raise ValueError("Saved selection does not match the saved manifests")
        self.items = [by_id[i] for i in plan["target_ids"]]

    def env(self, control):
        return {**os.environ, **OFFLINE_ENV, "RELIEF_OUTPUT_DIR": str(self.output),
                "RELIEF_STATE_DIR": str(self.state), "RELIEF_MANIFEST_DIR": self.plan["manifest_dir"],
                "RELIEF_PHASE_FILE": str(Path(control) / "phase.json"), "RELIEF_RUN_ID": self.plan["run_id"]}

    def classify(self, items=None, on_item=None):
        import pending
        ledgers = {m: load_json(self.state / f"qa_{m}.json") for m in ("images", "meshes", "gaussians")}
        items = self.items if items is None else items
        quarantine = load_json(self.state / "qa_quarantine.json")
        result = {key: [] for key in pending.BUCKETS}
        for number, item in enumerate(items):
            if on_item:
                on_item(number, len(items))
            buckets = pending.classify([item], self.output, quarantine=quarantine,
                                       certified=True, ledgers=ledgers)
            for key, values in buckets.items():
                result[key].extend(values)
        if on_item:
            on_item(len(items), len(items))
        return result

    def settings(self, item):
        result = dict(self.profile)
        spec = item.get("generation_prompt", {}).get("specifications", {}).get("prompt", {})
        if result["structure_guides"] and spec.get("profile") == "cargo_flat_octocopter":
            result["pipeline_type"] = result["guided_pipeline_type"]
        return result

    def command(self, item, *, preflight=False, guide=None):
        settings = self.settings(item)
        command = [self.python, PROJECT_DIR / "scripts/local/run_all.py", "--ids", item["id"], "--batch-size", "1"]
        for key in GENERATION_KEYS:
            if settings[key] is not None:
                command += ["--" + key.replace("_", "-"), str(settings[key])]
        command += ["--preflight-only" if preflight else "--skip-preflight"]
        if guide:
            command += ["--structure-reference", str(guide)]
        return command

    def prepare_guide(self, item, execute):
        spec = item.get("generation_prompt", {}).get("specifications", {}).get("prompt", {})
        if not self.profile["structure_guides"] or spec.get("profile") != "cargo_flat_octocopter":
            return None
        from generation_prompts import digest
        source_hash = digest(item["prompt"])
        directory = self.directory / "guides" / item["id"]
        for number in range(1000):
            path = directory / f"{source_hash[:16]}_{number:03d}.png"
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                record = load_json(sidecar)
                if (record.get("kind") == "authored_geometry_guide" and record.get("id") == item["id"]
                        and record.get("manifest_prompt_sha256") == source_hash
                        and record.get("specification") == spec and path.is_file()
                        and file_sha256(path) == record.get("image_sha256")):
                    return path
            if not any(path.with_suffix(ext).exists() for ext in (".png", ".json", ".glb")):
                code = execute([self.mesh_python, PROJECT_DIR / "scripts/local/build_structure_guide.py",
                                "--id", item["id"], "--output", path], "structure_guide")
                if code:
                    raise RuntimeError(f"Guide generation failed for {item['id']}; see the run log")
                if not sidecar.is_file():
                    raise RuntimeError("Guide builder returned without a commit marker")
                return path
        raise RuntimeError("Too many interrupted guide files; preserved for inspection")

    def release(self, execute):
        path = self.directory / "release_manifest.json"
        code = execute([sys.executable, PROJECT_DIR / "scripts/local/release_manifest.py", "--output", path], "release")
        if code:
            raise RuntimeError("Release accounting failed; see the run log")
        release = load_json(path)
        ids = {i["id"] for i in release.get("items", [])}
        if ids != set(self.plan["target_ids"]):
            raise RuntimeError(f"Release includes {len(ids)}/{len(self.plan['target_ids'])} selected items; see {path}")
        return path


class Worker:
    def __init__(self, control, plan, backend=None, *, poll=0.5):
        self.control, self.plan = Path(control), plan
        self.backend = backend or Backend(plan)
        self.profile = plan["profile"]
        self.poll = poll
        self.signalled = False
        self.child = None
        self.awake = None
        self.group = os.getpgrp()
        old = load_json(self.control / "status.json")
        counts = old.get("counts", {}) if old.get("run_id") == plan["run_id"] else {}
        self.status = {"run_id": plan["run_id"], "pid": os.getpid(), "counts": counts, "item": None}
        self.progress_path = Path(plan["directory"]) / "progress.json"
        self.progress = load_json(self.progress_path) or {"attempts": {}, "errors": {}, "retry_epoch": 0}
        self.log_path = Path(plan["directory"]) / "run.log"
        self.last_heartbeat = 0.0
        self.power_wait = (old.get("run_id") == plan["run_id"]
                           and bool(old.get("power_wait", old.get("phase") == "waiting_for_power")))
        self.control_revision = None
        self.internal_stop = False

    def wanted(self):
        data = load_json(self.control / "control.json")
        running = data.get("desired") == "running" and data.get("run_id") == self.plan["run_id"]
        if running:
            self.control_revision = data.get("revision")
        return running

    def finish(self, desired):
        saved = finish_control(self.control, self.plan, desired, self.control_revision)
        return 75 if not saved and self.wanted() else 0

    def update(self, phase=None, **extra):
        self.status.update(extra)
        if phase:
            self.status["phase"] = phase
        self.status["at"] = now()
        self.status["power_wait"] = self.power_wait
        write_json(self.control / "status.json", self.status)
        self.last_heartbeat = time.monotonic()

    def signal(self, _number, _frame):
        if not self.internal_stop:
            self.signalled = True

    def stop_children(self, *, keep_awake=False):
        if self.group != os.getpid() or os.getpgrp() != self.group:
            raise RuntimeError("Refusing to signal a process group not owned by this worker")
        self.internal_stop = True
        try:
            excluded = {self.awake.pid} if keep_awake and self.awake and self.awake.poll() is None else set()
            if not keep_awake and (self.child or self.awake):
                os.killpg(self.group, signal.SIGTERM)
            members = [pid for pid in group_members(self.group) if pid not in excluded]
            if keep_awake:
                for pid in members:
                    try:
                        if os.getpgid(pid) == self.group:
                            os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 8
            while members and time.monotonic() < deadline:
                if self.child:
                    self.child.poll()
                if self.awake:
                    self.awake.poll()
                time.sleep(0.05)
                members = [pid for pid in group_members(self.group) if pid not in excluded]
            for pid in members:
                try:
                    if os.getpgid(pid) == self.group:
                        os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if self.child:
                self.child.wait(timeout=5)
            self.child = None
            if self.awake:
                self.awake.poll()
        finally:
            self.internal_stop = False

    def checkpoint(self):
        if self.signalled or not self.wanted():
            raise Interrupted()

    def delay(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.checkpoint()
            time.sleep(min(self.poll, max(0, deadline - time.monotonic())))
            if time.monotonic() - self.last_heartbeat >= 5:
                self.update()

    def execute(self, command, phase):
        self.checkpoint()
        self.update(phase)
        write_json(self.control / "phase.json", {"at": now(), "run_id": self.plan["run_id"], "stage": phase})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", buffering=1) as log:
            log.write(f"\n[{now()}] {phase}: {self.status.get('item') or 'run'}\n")
            log.flush()
            # Inherit the launchd job's process group, so crash cleanup includes
            # grandchildren and preflight probes, not just the Python parent.
            self.child = subprocess.Popen([str(c) for c in command], cwd=PROJECT_DIR,
                                          env=self.backend.env(self.control), stdout=log, stderr=subprocess.STDOUT)
            power_check = time.monotonic()
            while self.child.poll() is None:
                self.checkpoint()
                if time.monotonic() - power_check >= 15:
                    power_check = time.monotonic()
                    power = self.battery()
                    if power and not power["ac"] and power["percent"] <= self.profile["low_battery_percent"]:
                        # Persist the recovery threshold before child cleanup,
                        # which may itself be interrupted by shutdown.
                        self.power_wait = True
                        self.update("waiting_for_power", battery_percent=power["percent"])
                        raise LowPower()
                if time.monotonic() - self.last_heartbeat >= 5:
                    self.update()
                time.sleep(self.poll)
            code = self.child.wait()
            self.child = None
        self.checkpoint()  # A pause during child exit is not a failed attempt.
        return code

    def battery(self):
        if sys.platform != "darwin":
            return None
        try:
            result = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        percent = re.search(r"(\d+)%;", result.stdout)
        if result.returncode or not percent:
            return None
        return {"percent": int(percent[1]), "ac": "AC Power" in result.stdout}

    def ready_resources(self, *, check_disk=True):
        while True:
            self.checkpoint()
            power = self.battery()
            if power and not power["ac"]:
                threshold = self.profile["resume_battery_percent"] if self.power_wait else self.profile["low_battery_percent"]
                self.power_wait = power["percent"] <= threshold
            else:
                self.power_wait = False
            if self.power_wait:
                if self.awake and self.awake.poll() is None:
                    self.awake.terminate()
                    self.awake.wait(timeout=5)
                self.update("waiting_for_power", battery_percent=power["percent"])
                self.delay(10)
                continue
            if not check_disk:
                return
            path = Path(self.profile["output_dir"])
            while not path.exists():
                path = path.parent
            free = shutil.disk_usage(path).free
            if free < self.profile["disk_reserve_gb"] * 1e9:
                if self.awake and self.awake.poll() is None:
                    self.awake.terminate()
                    self.awake.wait(timeout=5)
                self.update("waiting_for_disk", free_bytes=free)
                self.delay(10)
                continue
            if lock_busy(Path(self.profile["state_dir"]) / ".pipeline.lock"):
                self.update("waiting_for_pipeline")
                self.delay(5)
                continue
            return

    @contextmanager
    def lease(self):
        while True:
            self.checkpoint()
            # Power, disk and pipeline ownership may change during a GPU wait.
            self.ready_resources()
            try:
                context = gpu_lock()
                context.__enter__()
            except RuntimeError:
                self.update("waiting_for_gpu")
                self.delay(5)
                continue
            break
        try:
            if sys.platform == "darwin" and (not self.awake or self.awake.poll() is not None):
                self.awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            yield
        finally:
            try:
                self.stop_children(keep_awake=sys.exc_info()[0] is None)
            finally:
                context.__exit__(None, None, None)

    def counts(self, buckets):
        self.status["counts"] = {key: len(value) for key, value in buckets.items()}
        self.status["counts"]["total"] = len(self.plan["target_ids"])

    def scan(self, items=None):
        def on_item(number, total):
            self.checkpoint()
            self.status.update(scan_checked=number, scan_total=total)
            if time.monotonic() - self.last_heartbeat >= 5:
                self.update()
        return self.backend.classify(items, on_item=on_item)

    def run(self):
        if self.group != os.getpid():
            raise RuntimeError("Worker must have its own process group; start it with atlas")
        previous = {s: signal.signal(s, self.signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            self.checkpoint()
            self.ready_resources(check_disk=False)
            verify_plan(self.plan)
            self.update("scanning")
            buckets = self.scan()
            self.counts(buckets)
            if buckets["refused"]:
                raise RuntimeError("The saved selection is no longer eligible under the current policy")
            data = load_json(self.control / "control.json")
            if data.get("retry_epoch", 0) > self.progress.get("retry_epoch", 0):
                self.progress.update(attempts={}, errors={}, retry_epoch=data["retry_epoch"])
            by_id = {i["id"]: i for i in self.backend.items}
            queue = [by_id[i] for i in buckets["pending"]]
            certified = set(buckets["done"])
            quarantined = set(buckets["quarantined"])
            checked_presets = set()
            for item in queue:
                item_id = item["id"]
                self.status["item"] = item_id
                while self.progress["attempts"].get(item_id, 0) < self.profile["max_item_attempts"]:
                    self.ready_resources()
                    with self.lease():
                        settings = self.backend.settings(item)
                        preset = tuple(settings[key] for key in ("pipeline_type", "texture_size", "gs_format"))
                        if preset not in checked_presets:
                            code = self.execute(self.backend.command(item, preflight=True), "preflight")
                            report = load_json(Path(self.profile["state_dir"]) / "preflight.json")
                            if code or not report.get("ready"):
                                raise RuntimeError("Offline preflight failed; see " + str(Path(self.profile["state_dir"]) / "preflight.json"))
                            checked_presets.add(preset)
                        guide = self.backend.prepare_guide(item, self.execute)
                        code = self.execute(self.backend.command(item, guide=guide), "generating")
                    latest = self.scan([item])
                    if item_id in latest["done"]:
                        certified.add(item_id)
                        self.progress["errors"].pop(item_id, None)
                        break
                    if item_id in latest["quarantined"]:
                        quarantined.add(item_id)
                        break
                    if lock_busy(Path(self.profile["state_dir"]) / ".pipeline.lock"):
                        self.update("waiting_for_pipeline")
                        self.delay(5)
                        continue
                    self.progress["attempts"][item_id] = self.progress["attempts"].get(item_id, 0) + 1
                    self.progress["errors"][item_id] = f"Stage returned {code}; deliverable still lacks current certification"
                    write_json(self.progress_path, self.progress)
                    self.update("retry_wait", attempt=self.progress["attempts"][item_id])
                    self.delay(self.profile["retry_seconds"])
                write_json(self.progress_path, self.progress)
                self.status["counts"] = {"done": len(certified), "quarantined": len(quarantined), "refused": 0,
                    "pending": len(self.plan["target_ids"]) - len(certified) - len(quarantined),
                    "total": len(self.plan["target_ids"])}
                self.update("item_checked")
            self.status["item"] = None
            self.update("verifying")
            final = self.scan()
            self.counts(final)
            if final["pending"] or final["quarantined"] or final["refused"]:
                self.update("needs_attention", error="Some items remain incomplete; quarantine and failures are preserved",
                            problem_ids=final["pending"] + final["quarantined"] + final["refused"])
                return self.finish("attention")
            release = self.backend.release(self.execute)
            self.update("completed", release=str(release), verified_at=now())
            return self.finish("completed")
        except LowPower:
            self.stop_children()
            self.power_wait = True
            self.update("waiting_for_power")
            return 75 if self.wanted() else 0
        except Interrupted:
            self.stop_children()
            desired = load_json(self.control / "control.json").get("desired")
            self.update("paused" if desired != "running" else "interrupted")
            return 75 if desired == "running" else 0
        except Exception as exc:
            self.stop_children()
            self.update("needs_attention", error=str(exc))
            return self.finish("attention")
        finally:
            if self.awake and self.awake.poll() is None:
                self.awake.terminate()
                self.awake.wait(timeout=5)
            for number, handler in previous.items():
                signal.signal(number, handler)


def worker_main(control, backend_factory=Backend, *, poll=0.5):
    control = Path(control)
    if load_json(control / "control.json").get("desired") != "running":
        return 0
    control.mkdir(parents=True, exist_ok=True)
    with (control / ".worker.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        request = load_json(control / "control.json")
        if request.get("desired") != "running":
            return 0
        try:
            require_network_guard()
            plan = saved_plan(control)
            worker = Worker(control, plan, backend_factory(plan), poll=poll)
            return worker.run()
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            write_json(control / "status.json", {"phase": "needs_attention", "at": now(), "error": str(exc),
                "run_id": request.get("run_id")})
            if finish_control(control, request, "attention", request.get("revision")):
                return 0
            # Startup can fail before Worker owns the request. The same
            # revision check still protects a later Pause or Resume click.
            return 75 if load_json(control / "control.json").get("desired") == "running" else 0


def status_data(control):
    control = Path(control)
    data = load_json(control / "status.json")
    wanted = load_json(control / "control.json")
    plan = saved_plan(control)
    if data.get("run_id") != plan.get("run_id"):
        data = {}
    active = lock_busy(control / ".worker.lock")
    data.update(desired=wanted.get("desired", "not_started"), worker_running=active,
                run_id=plan.get("run_id"), run_directory=plan.get("directory"))
    if not active:
        if wanted.get("desired") == "paused":
            data["phase"] = "paused"
        elif wanted.get("desired") == "running":
            data["phase"] = "recovering"
    phase = load_json(control / "phase.json")
    if (active and phase.get("run_id") == plan.get("run_id")
            and data.get("phase") in ("preflight", "structure_guide", "generating", "release")):
        data["active_stage"] = phase
    return data


def show_status(control, *, json_output=False):
    data = status_data(control)
    if json_output:
        print(json.dumps(data, indent=2))
        return
    print("Atlas: " + data.get("phase", "not started").replace("_", " "))
    counts = data.get("counts", {})
    if counts:
        print(f"Certified {counts.get('done', 0)}/{counts.get('total', 0)} | "
              f"pending {counts.get('pending', 0)} | quarantined {counts.get('quarantined', 0)}")
    if data.get("item") and data["worker_running"]:
        stage = data.get("active_stage", {})
        print("Current: " + data["item"] + " / " + str(stage.get("mode") or stage.get("stage") or data.get("phase")))
    if data.get("phase") in ("scanning", "verifying") and data.get("scan_total"):
        print(f"Checking saved files: {data.get('scan_checked', 0)}/{data['scan_total']}")
    if data.get("error"):
        print(("" if data.get("phase") == "needs_attention" else "Previous issue: ") + data["error"])
    if data.get("run_directory"):
        print("Saved run: " + data["run_directory"])
    else:
        print("Start.command starts the configured default-policy collection offline.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control-dir", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("action", choices=["start", "pause", "resume", "status", "plan", "logs", "uninstall", "_worker"])
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--state-dir")
    parser.add_argument("--new-run", action="store_true", help="start a new frozen plan; preserve all existing assets")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    control = args.control_dir.resolve()
    try:
        if args.action == "_worker":
            return worker_main(control)
        if args.action == "status":
            show_status(control, json_output=args.json)
            return 0
        if args.action == "logs":
            plan = saved_plan(control)
            paths = ([Path(plan["directory"]) / "run.log"] if plan else []) + [control / "supervisor.log"]
            found = False
            for path in paths:
                if not path.is_file():
                    continue
                found = True
                print(f"--- {path.name} ---")
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16000))
                    print(stream.read().decode(errors="replace"))
            if not found:
                print("No run log yet.")
            return 0
        if args.action in ("pause", "uninstall"):
            with exclusive(control / ".commands.lock"):
                if not (control / "plan.json").exists() and args.action == "pause":
                    print("Nothing is running.")
                    return 0
                pause_request = change_control(control, "paused")
            deadline = time.monotonic() + 18
            while (lock_busy(control / ".worker.lock") and time.monotonic() < deadline
                   and load_json(control / "control.json").get("desired") == "paused"):
                time.sleep(0.1)
            show_status(control, json_output=args.json)
            if args.action == "uninstall":
                with exclusive(control / ".commands.lock"):
                    latest = load_json(control / "control.json")
                    if latest.get("revision") != pause_request["revision"]:
                        print("A later command superseded removal; the service is preserved.")
                        return 0
                    service = f"gui/{os.getuid()}/{job_label(control)}"
                    loaded = subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0
                    if loaded:
                        subprocess.run(["launchctl", "bootout", service], capture_output=True, check=True)
                    (Path.home() / "Library/LaunchAgents" / (job_label(control) + ".plist")).unlink(missing_ok=True)
                    print("Offline service removed. Saved assets and run are preserved; Start reinstalls it.")
            return 0
        overrides = {k: v for k, v in {"ids": args.ids, "limit": args.limit,
                      "output_dir": args.output_dir, "state_dir": args.state_dir}.items() if v is not None}
        plan = saved_plan(control)
        if args.action == "plan":
            if plan and not args.new_run and not overrides and args.profile == DEFAULT_PROFILE:
                print(json.dumps({"saved_run": plan["run_id"], "items": len(plan["target_ids"]),
                                  "profile": plan["profile"], "created_at": plan["created_at"]}, indent=2))
            else:
                profile = load_profile(args.profile, overrides)
                _documents, _hashes, permitted, refused = source_selection(profile)
                print(json.dumps({"items": len(permitted), "policy_excluded": len(refused), "profile": profile,
                                  "inference_started": False}, indent=2))
            return 0
        with exclusive(control / ".commands.lock"):
            plan = saved_plan(control)
            if plan and not args.new_run and (overrides or args.profile != DEFAULT_PROFILE):
                raise ValueError("Resume uses the saved selection/settings; use start --new-run to change them")
            if args.new_run and (args.action != "start" or lock_busy(control / ".worker.lock")):
                raise ValueError("Pause the worker before start --new-run; existing assets will be preserved")
            if not plan or args.new_run:
                profile = load_profile(args.profile, overrides)
                plan = prepare_plan(control, profile)
            verify_plan(plan)
            old = load_json(control / "control.json")
            # The login definition is durable before running intent is
            # committed, so a power cut before bootstrap can recover on login.
            prepare_service(control)
            change_control(control, "running", plan["run_id"], retry=old.get("desired") == "attention")
            start_service(control)
        print(f"Offline controller started: {len(plan['target_ids'])} selected items. Safe to close this terminal.")
        print("Pause.command pauses; Resume.command continues; Status.command shows progress.")
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print("Atlas: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
