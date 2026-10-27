"""Resume this workspace's download/pilot and its passive health monitor on macOS.

Use --install-login to resume automatically after reboot and user login.
The preparation worker reuses on-disk chunks, checkpoints and artifact-bound QA.
"""

import argparse
import os
import plistlib
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pipeline_utils import PROJECT_DIR, STATE_DIR, durable_replace

PREPARATION_LABEL = "org.reliefatlas.qwen-preparation"
MONITOR_LABEL = "org.reliefatlas.qwen-health"


def payload(label, script, log, *, keep_awake=False):
    arguments = [sys.executable, "-u", str(script)]
    if keep_awake:
        arguments = ["/usr/bin/caffeinate", "-i", *arguments]
    return {
        "Label": label, "ProgramArguments": arguments,
        "WorkingDirectory": str(PROJECT_DIR),
        "EnvironmentVariables": {"PATH": os.environ["PATH"], "PYTHONUNBUFFERED": "1"},
        "StandardOutPath": str(log), "StandardErrorPath": str(log),
        "RunAtLoad": True, "ProcessType": "Background", "ExitTimeOut": 30,
    }


def write_plist(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name,
                                     suffix=".tmp", delete=False) as fp:
        temporary = Path(fp.name)
        fp.write(plistlib.dumps(data))
    try:
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stop_service(label):
    service = f"gui/{os.getuid()}/{label}"
    exists = subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0
    if exists:
        subprocess.run(["launchctl", "bootout", service], check=True)


def stop_old_monitor(script):
    pid_file = STATE_DIR / "qwen_migration/health_monitor.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text())
    except ValueError:
        return
    command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True).stdout
    if str(script) in command:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-login", action="store_true",
                        help="register both jobs in your user LaunchAgents directory")
    parser.add_argument("--status", action="store_true", help="show service status without restarting")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("This preparation service requires macOS")
    if args.status:
        for label in (PREPARATION_LABEL, MONITOR_LABEL):
            result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{label}"],
                                    capture_output=True, text=True)
            fields = {}
            for line in result.stdout.splitlines():
                for key in ("state =", "pid =", "last exit code ="):
                    if line.strip().startswith(key):
                        fields.setdefault(key, line.strip())
            print(label + ": " + ("; ".join(fields.values()) if fields else "not loaded"))
        return 0
    directory = STATE_DIR / "qwen_migration"
    preparation = STATE_DIR / "prepare_qwen_and_pilot.py"
    monitor = directory / "monitor_health.py"
    for script in (preparation, monitor):
        if not script.is_file():
            parser.error(f"Missing workspace preparation script: {script}")
    destination = (Path.home() / "Library/LaunchAgents" if args.install_login else directory)
    # After installation, ordinary resumes use the installed job definitions.
    if not args.install_login and (Path.home() / "Library/LaunchAgents" / (PREPARATION_LABEL + ".plist")).exists():
        destination = Path.home() / "Library/LaunchAgents"
    jobs = (
        (PREPARATION_LABEL, preparation, STATE_DIR / "prepare_qwen_and_pilot.log", True),
        (MONITOR_LABEL, monitor, directory / "health_monitor.log", False),
    )
    for label, _, _, _ in jobs:
        stop_service(label)
    stop_old_monitor(monitor)
    for label, script, log, keep_awake in jobs:
        path = destination / (label + ".plist")
        write_plist(path, payload(label, script, log, keep_awake=keep_awake))
        subprocess.run(["plutil", "-lint", str(path)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["launchctl", "enable", f"gui/{os.getuid()}/{label}"], check=True)
        subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True)
    print("Preparation and health monitoring resumed from disk.")
    if destination == Path.home() / "Library/LaunchAgents":
        print("Both services will resume automatically after reboot and login.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
