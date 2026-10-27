"""Dependency-free paths, atomic state, and artifact-bound QA contracts."""

import hashlib
import json
import math
import os
import tempfile
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_DIR / "meshmaker" / "local"
OUTPUT_DIR = Path(os.environ.get("RELIEF_OUTPUT_DIR", PROJECT_DIR / "outputs_relief")).resolve()
STATE_DIR = Path(os.environ.get("RELIEF_STATE_DIR", PROJECT_DIR / "state")).resolve()
MODEL_DIR = LOCAL_DIR / "models"
GS_EXTENSIONS = ("spz", "splat", "ply")
QA_VERSION = 6
GS_COLOR_ENCODINGS = {"spz": "sh_dc", "splat": "rgb", "ply": "sh_dc"}
IMAGE_MODEL_ID = "local/qwen-image-2.1"


def is_infrastructure_error(exc):
    """Resource/transport failures cannot be repaired by resampling content."""
    return isinstance(exc, (OSError, MemoryError)) or any(
        marker in str(exc).lower() for marker in
        ("out of memory", "failed to allocate", "cannot allocate", "gpu watchdog", "device lost"))


def sync_directory(path):
    """Persist a rename/commit marker as well as the file's contents."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_replace(temporary, final):
    temporary, final = Path(temporary), Path(final)
    with temporary.open("rb") as fp:
        os.fsync(fp.fileno())
    temporary.replace(final)
    sync_directory(final.parent)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read state {path}: {exc}") from exc


def write_json(path, data):
    """Write and fsync a unique sibling before atomically replacing state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                dir=path.parent, prefix=f".{path.stem}.", suffix=f".tmp{path.suffix}",
                delete=False) as fp:
            tmp = Path(fp.name)
            json.dump(data, fp, indent=2, allow_nan=False)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        tmp.replace(path)
        sync_directory(path.parent)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@lru_cache(maxsize=4096)
def _sha256(path, size, mtime_ns, ctime_ns, inode):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    path = Path(path)
    st = path.stat()
    return _sha256(str(path.resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


@lru_cache(maxsize=4)
def generation_provenance(*model_names):
    """Record the prepared checkpoint identities and code actually executing."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "torchvision", "diffusers", "transformers", "numpy", "scipy", "spz"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    manifest = load_json(MODEL_DIR / "model_manifest.json")
    models = {name: manifest[name] for name in model_names if name in manifest}
    files = sorted((PROJECT_DIR / "scripts" / "local").glob("*.py"))
    files += sorted((PROJECT_DIR / "config").glob("local_*.json"))
    files += [PROJECT_DIR / "config" / "reference_library.json"]
    code_hashes = {str(p.relative_to(PROJECT_DIR)): file_sha256(p) for p in files}
    return {"device": "mps", "runtime": versions, "prepared_models": models,
            "pipeline_sha256": hashlib.sha256(json.dumps(code_hashes, sort_keys=True).encode()).hexdigest()}


def artifact_fingerprints(item_dir, item_id, mode):
    """Content hashes for the inputs and outputs inspected by one QA stage.

    Metadata and the source PNG bind geometry verdicts to their generation.
    Review montages are deliberately excluded: producing a montage cannot
    invalidate the verdict that just produced it.
    """
    d = Path(item_dir)
    files = [d / f"{item_id}.png"]
    if mode in ("meshes", "gaussians"):
        files += [d / f"{item_id}.glb", d / "metadata.json"]
    if mode == "gaussians":
        containers = [d / f"{item_id}.{ext}" for ext in GS_EXTENSIONS
                      if (d / f"{item_id}.{ext}").is_file()]
        if not containers:
            return None
        files += containers
    elif mode not in ("images", "meshes"):
        raise ValueError(f"Unknown QA mode: {mode}")
    try:
        if any(not p.is_file() or p.stat().st_size == 0 for p in files):
            return None
        return {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                for p in files}
    except OSError:
        return None


def verdict_matches(verdict, item_dir, item_id, mode, *, require_nsfw=True):
    if not isinstance(verdict, dict) or verdict.get("qa_version") != QA_VERSION:
        return False
    if not verdict.get("pass") or verdict.get("view_error"):
        return False
    if mode == "images" and require_nsfw:
        score = verdict.get("nsfw")
        if (not verdict.get("nsfw_screened") or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 0.5):
            return False
    expected = artifact_fingerprints(item_dir, item_id, mode)
    return expected is not None and verdict.get("artifacts") == expected


def source_image_matches(item_dir, item_id):
    d = Path(item_dir)
    try:
        meta = load_json(d / "metadata.json")
        return (isinstance(meta, dict)
                and meta.get("source_image_sha256") == file_sha256(d / f"{item_id}.png"))
    except (ValueError, OSError):
        return False


def source_image_metadata(item_dir, item_id):
    """Return reference provenance only when it belongs to the current PNG."""
    d = Path(item_dir)
    try:
        data = load_json(d / "image_meta.json")
        if (isinstance(data, dict)
                and data.get("image_sha256") == file_sha256(d / f"{item_id}.png")):
            return data
    except (ValueError, OSError):
        pass
    return {}


def image_model_matches(item_dir, item_id):
    return source_image_metadata(item_dir, item_id).get("image_model") == IMAGE_MODEL_ID


def item_certified(item_dir, item_id, ledgers, *, require_gs=True):
    if not source_image_matches(item_dir, item_id) or not image_model_matches(item_dir, item_id):
        return False
    modes = ("images", "meshes", "gaussians") if require_gs else ("images", "meshes")
    return all(verdict_matches(ledgers.get(mode, {}).get(item_id), item_dir, item_id, mode)
               for mode in modes)


@contextmanager
def gpu_lock(path=None):
    """Serialize model CLI processes; a managed job may lend its lease to children.

    Only descendants in the lease holder's own process group can inherit it.
    The OS lock, rather than a saved PID alone, establishes live ownership.
    """
    import fcntl

    path = Path(path) if path is not None else PROJECT_DIR / "state" / ".gpu.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as fp:
        acquired = False
        try:
            try:
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as exc:
                fp.seek(0)
                try:
                    owner = int(fp.read().strip())
                    nested = owner == os.getpgrp() and os.getpgid(owner) == owner
                except (ValueError, ProcessLookupError, PermissionError):
                    nested = False
                if not nested:
                    raise RuntimeError("Another local generation process owns the GPU lease") from exc
            if acquired:
                fp.seek(0)
                fp.truncate()
                fp.write(str(os.getpid()))
                fp.flush()
            yield
        finally:
            if acquired:
                fcntl.flock(fp, fcntl.LOCK_UN)


@contextmanager
def pipeline_lock(state_dir=STATE_DIR):
    """The OS releases this lock even if the orchestrator crashes."""
    import fcntl
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / ".pipeline.lock").open("a+") as fp:
        try:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another pipeline run owns {state_dir / '.pipeline.lock'}") from exc
        try:
            fp.seek(0)
            fp.truncate()
            fp.write(str(os.getpid()))
            fp.flush()
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)
