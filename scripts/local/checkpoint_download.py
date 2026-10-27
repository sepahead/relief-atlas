"""Resumable official-HF transfer for large checkpoints on slow connections.

Authentication and signed CDN URLs remain in memory. A validated checksum is
required before a checkpoint replaces the final filename.
"""

import hashlib
import re
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit


def _safe_request_error(exc):
    """Keep useful transport causes while removing URLs and credentials."""
    detail = re.sub(r"https?://[^\s\"'<>]+", "[URL]", str(exc))
    detail = re.sub(
        r"(?i)(authorization|bearer|token|signature|x-amz-[\w-]+)(\s*[=:]\s*)[^\s,)'\"]+",
        r"\1\2[REDACTED]", detail,
    )
    return detail[:500]


def _download_checkpoint(spec, root, filename, token, stage_config, expected_sha256,
                         alternative_urls, speed_limit_kib):
    import requests
    from prepare_models import validate_safetensors
    from pipeline_utils import durable_replace

    if not shutil.which("aria2c"):
        raise RuntimeError("Resumable download needs aria2c; use the standard downloader otherwise")
    relative_name = Path(filename)
    if relative_name.is_absolute() or ".." in relative_name.parts:
        raise ValueError("Checkpoint filename must stay inside its model directory")
    folder = Path(root) / spec["directory"] / relative_name.parent
    filename = relative_name.name
    folder.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}"

    # Keep the checkpoint paired with the configuration at the same revision.
    staged = []
    for name in (("config.json", "LICENSE.md") if stage_config else ()):
        with requests.get(f"{base}/{name}", headers=headers, timeout=(30, 120)) as response:
            response.raise_for_status()
            temporary = folder / f".{name}.download"
            temporary.write_bytes(response.content)
            staged.append((temporary, folder / name))

    with requests.get(f"{base}/{relative_name.as_posix()}?download=true", headers=headers,
                      stream=True, allow_redirects=False, timeout=(30, 120)) as response:
        response.raise_for_status()
        origin = response.history[0] if response.history else response
        expected = origin.headers.get("x-linked-etag", "").strip('"')
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("The official server did not supply a checkpoint SHA256")
        if expected_sha256 and expected != expected_sha256:
            raise RuntimeError("Official checkpoint digest differs from the pinned catalog")
        size = int(origin.headers.get("x-linked-size") or response.headers["content-length"])
        signed_url = response.headers.get("location", response.url)
        # Identify progress by checkpoint digest, independently of expiring
        # URLs or CDN routing changes after a restart. Preserve older transfers.
        legacy_name = Path(urlsplit(signed_url).path).name
        partial_name = f".{filename}.{expected[:12]}.download"
    partial = folder / partial_name
    legacy = folder / legacy_name if legacy_name not in ("", ".", "..", filename) else None
    resume_map = Path(str(partial) + ".aria2")
    if legacy is not None and legacy != partial:
        if not partial.exists() and legacy.exists():
            durable_replace(legacy, partial)
        legacy_map = Path(str(legacy) + ".aria2")
        if partial.exists() and not resume_map.exists() and legacy_map.exists():
            durable_replace(legacy_map, resume_map)
    command = ["aria2c", "--input-file=-", "--auto-file-renaming=false",
               "--allow-overwrite=true", "--continue=true", "--async-dns=false",
               "--auto-save-interval=10", "--disk-cache=0",
               "--max-concurrent-downloads=1", "--max-connection-per-server=1", "--split=1",
               "--file-allocation=none",
               "--check-integrity=true", f"--checksum=sha-256={expected}",
               "--summary-interval=60", "--console-log-level=warn", "--max-tries=3",
               "--retry-wait=60", "--timeout=120", "--connect-timeout=30",
               "--check-certificate=true"]
    if speed_limit_kib is not None:
        if speed_limit_kib <= 0:
            raise ValueError("speed_limit_kib must be positive or None")
        limit = f"{speed_limit_kib}K"
        command[command.index("--file-allocation=none"):command.index("--file-allocation=none")] = [
            f"--max-overall-download-limit={limit}", f"--max-download-limit={limit}"]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True) as process:
        # --out on the command line is ignored with --input-file. Per-file
        # options also keep capabilities and credentials out of process argv.
        urls = "\t".join([*alternative_urls, signed_url])
        process.stdin.write(f"{urls}\n  dir={folder}\n  out={partial_name}\n")
        process.stdin.close()
        for line in process.stdout:
            print(re.sub(r"https?://[^\s]+", "[download URL]", line), end="", flush=True)
        if process.wait():
            raise RuntimeError("Checkpoint transfer failed; resume data retained")
    if partial.stat().st_size != size:
        raise RuntimeError("Checkpoint transfer is incomplete; original weights retained")
    digest = hashlib.sha256()
    with partial.open("rb") as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError("Checkpoint SHA256 mismatch; original weights retained")
    validate_safetensors(partial)
    for temporary, final in staged:
        durable_replace(temporary, final)
    durable_replace(partial, folder / filename)
    print(f"{spec['repo_id']}/{relative_name}: official checkpoint SHA256 verified", flush=True)


def download_checkpoint(spec, root, filename="model.safetensors", token=None, *,
                        stage_config=True, expected_sha256=None, alternative_urls=(),
                        speed_limit_kib=256):
    """Download without exposing signed capabilities through HTTP errors."""
    import requests

    try:
        _download_checkpoint(spec, root, filename, token, stage_config, expected_sha256,
                             alternative_urls, speed_limit_kib)
    except requests.RequestException as exc:
        detail = _safe_request_error(exc)
        raise RuntimeError(f"Official checkpoint request failed ({type(exc).__name__}: {detail}); "
                           "retry to resume the transfer") from None


def download_snapshot(spec, root, token=None, force=False, *, speed_limit_kib=256):
    """Download one checkpoint at a time with one connection and slow retries."""
    from huggingface_hub import snapshot_download
    from pipeline_utils import file_sha256

    directory = Path(root) / spec["directory"]
    metadata_names = [name for name in spec.get("required", []) if not name.endswith(".safetensors")]
    def cached_at_revision(name):
        metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
        if not (directory / name).is_file() or not metadata.is_file():
            return False
        lines = metadata.read_text().splitlines()
        return bool(lines) and lines[0] == spec["revision"]
    if force or not metadata_names or not all(cached_at_revision(name) for name in metadata_names):
        snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                          local_dir=directory,
                          allow_patterns=spec["allow_patterns"], ignore_patterns=["*.safetensors"],
                          max_workers=1, force_download=force)

    def transfer(pair):
        name, expected = pair
        final = Path(root) / spec["directory"] / name
        if not force and final.is_file() and file_sha256(final) == expected:
            return
        for attempt in range(1, 65):
            try:
                # Refresh expiring HF capabilities while keeping every saved
                # chunk. Finish this file before opening the next transfer.
                download_checkpoint(spec, root, name, token, stage_config=False,
                                    expected_sha256=expected, speed_limit_kib=speed_limit_kib)
                return
            except RuntimeError as exc:
                if attempt == 64:
                    raise
                print(f"{name}: {exc}; resuming attempt {attempt + 1}", flush=True)
                time.sleep(min(600, attempt * 60))

    for pair in spec["checkpoint_sha256"].items():
        transfer(pair)
