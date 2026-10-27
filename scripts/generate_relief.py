#!/usr/bin/env python3
"""
generate_relief.py — Dual-provider mesh generation for disaster relief dataset.

Supports both fal.ai and Runware APIs with multi-key rotation and budget tracking.

Usage:
  python generate_relief.py                           # All providers, all geographies
  python generate_relief.py --provider fal             # Only fal.ai
  python generate_relief.py --geography germany        # Only Germany
  python generate_relief.py --workers-per-key 15       # 15 workers per key
  python generate_relief.py --dry-run                  # Show queue stats only
  python generate_relief.py --retry-failed             # Retry previously failed items
"""

import argparse
import fcntl
import json
import os
import random
import requests
import signal
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import multiprocessing as mp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MANIFEST_DIR = PROJECT_DIR / "manifests"
CONFIG_DIR = PROJECT_DIR / "config"
OUTPUT_DIR = PROJECT_DIR / "outputs_relief"
STATE_DIR = PROJECT_DIR / "state"
QUEUE_FILE = STATE_DIR / "queue.json"
LOCK_FILE = STATE_DIR / "queue.lock"
SPENDING_FILE = STATE_DIR / "spending.json"
SPENDING_LOCK = STATE_DIR / "spending.lock"
FAILED_FILE = STATE_DIR / "failed.json"
OVERRIDE_LOCK = STATE_DIR / "content_policy_overrides.lock"

for d in [OUTPUT_DIR, STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# The dual-use content policy lives beside the local pipeline, but it is not
# specific to it: it screens manifest TEXT, so it applies to any generator
# reading the same manifests. This path had no gate at all, which meant the
# README's primary Usage section would push all 30 blocked weapon items and 540
# restricted items through a paid image API, while the docs claimed the policy
# flags were "accepted by every phase".
sys.path.insert(0, str(SCRIPT_DIR / "local"))
import content_policy  # noqa: E402
from generation_prompts import normalize_generation_prompt  # noqa: E402

# ---------------------------------------------------------------------------
# API Key Configuration
# ---------------------------------------------------------------------------

@dataclass
class APIKeyConfig:
    provider: str       # "fal" or "runware"
    name: str           # e.g. "sepehr_fal_1"
    key: str            # actual API key string
    max_budget: float   # USD budget ceiling
    spent: float = 0.0  # tracked spending

    @property
    def remaining(self) -> float:
        return max(0.0, self.max_budget - self.spent)

    @property
    def is_exhausted(self) -> bool:
        return self.spent >= self.max_budget * 0.90  # 90% threshold


def parse_api_keys(filepath: Path | None = None) -> list[APIKeyConfig]:
    """Parse multi-provider, multi-key config file."""
    if filepath is None:
        filepath = CONFIG_DIR / "api_keys.txt"
    if not filepath.exists():
        print(f"ERROR: API keys file not found: {filepath}")
        print(f"Copy {CONFIG_DIR / 'api_keys.example.txt'} to {filepath} and add your keys.")
        sys.exit(1)

    keys = []
    current_provider = None
    current_name = None
    current_budget = 100.0  # default

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Section header: # === fal.ai keys ===
            if line.startswith("# ==="):
                header = line.lower()
                if "fal" in header:
                    current_provider = "fal"
                elif "runware" in header:
                    current_provider = "runware"
                continue

            # Comment: key name or budget
            if line.startswith("#"):
                content = line.lstrip("#").strip()
                if content.startswith("max_budget="):
                    try:
                        current_budget = float(content.split("=")[1])
                    except ValueError:
                        pass
                elif not content.startswith("="):
                    current_name = content
                continue

            # Actual key line
            if current_provider and line and not line.startswith("#"):
                key_name = current_name or f"{current_provider}_{len(keys)+1}"
                keys.append(APIKeyConfig(
                    provider=current_provider,
                    name=key_name,
                    key=line,
                    max_budget=current_budget,
                ))
                current_name = None
                current_budget = 100.0

    return keys


# ---------------------------------------------------------------------------
# Spending Ledger
# ---------------------------------------------------------------------------

def load_spending() -> dict:
    if SPENDING_FILE.exists():
        with open(SPENDING_FILE) as f:
            return json.load(f)
    return {"version": 1, "keys": {}}


def save_spending(ledger: dict):
    with open(SPENDING_LOCK, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            with open(SPENDING_FILE, "w") as f:
                json.dump(ledger, f, indent=2)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def record_spending(key_name: str, provider: str, cost: float, stage: str, item_id: str,
                   max_budget: float = 0.0):
    with open(SPENDING_LOCK, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            ledger = load_spending()
            if key_name not in ledger["keys"]:
                ledger["keys"][key_name] = {
                    "provider": provider,
                    "max_budget": max_budget,
                    "total_spent": 0.0,
                    "request_count": 0,
                    "last_updated": "",
                    "history": [],
                }
            entry = ledger["keys"][key_name]
            if max_budget > 0:
                entry["max_budget"] = max_budget
            entry["total_spent"] += cost
            entry["request_count"] += 1
            entry["last_updated"] = datetime.now().isoformat()
            entry["history"].append({
                "item_id": item_id,
                "stage": stage,
                "cost": cost,
                "ts": datetime.now().isoformat(),
            })
            # Keep history manageable (last 1000 entries)
            if len(entry["history"]) > 1000:
                entry["history"] = entry["history"][-1000:]
            with open(SPENDING_FILE, "w") as f:
                json.dump(ledger, f, indent=2)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def sync_key_spending(keys: list[APIKeyConfig]):
    """Load spending from ledger into key configs."""
    ledger = load_spending()
    for key in keys:
        if key.name in ledger.get("keys", {}):
            key.spent = ledger["keys"][key.name].get("total_spent", 0.0)


# ---------------------------------------------------------------------------
# Failed Items Tracking
# ---------------------------------------------------------------------------

def load_failed() -> list[dict]:
    if FAILED_FILE.exists():
        with open(FAILED_FILE) as f:
            return json.load(f).get("failures", [])
    return []


def save_failed(failures: list[dict]):
    with open(FAILED_FILE, "w") as f:
        json.dump({"failures": failures}, f, indent=2)


def record_failure(item_id: str, error: str, provider: str, key_name: str, retry_count: int = 0):
    """Record a failed item (file-locked for process safety)."""
    with open(str(LOCK_FILE) + ".fail", "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            failures = load_failed()
            failures.append({
                "id": item_id,
                "error": error[:200],
                "provider": provider,
                "key": key_name,
                "timestamp": datetime.now().isoformat(),
                "retry_count": retry_count,
                "max_retries": 3,
            })
            save_failed(failures)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Queue Management (file-locked, process-safe)
# ---------------------------------------------------------------------------

def get_next_mesh() -> Optional[dict]:
    """Get next mesh from queue (process-safe)."""
    lock_path = str(LOCK_FILE)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            with open(QUEUE_FILE) as f:
                queue = json.load(f)
            if not queue:
                return None
            mesh = queue.pop(0)
            with open(QUEUE_FILE, "w") as f:
                json.dump(queue, f)
            return mesh
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def requeue_item(item: dict):
    """Put item back at end of queue."""
    lock_path = str(LOCK_FILE)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            with open(QUEUE_FILE) as f:
                queue = json.load(f)
            queue.append(item)
            with open(QUEUE_FILE, "w") as f:
                json.dump(queue, f)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def record_override(item: dict):
    """Append an override-GENERATED item to the ledger, process-safely.

    content_policy.record_overrides() read-modify-writes the whole file, which
    is fine for the single-process local phases but not for 15 workers per key.
    The fcntl lock mirrors record_failure().
    """
    with open(OVERRIDE_LOCK, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            content_policy.record_overrides([item], phase="cloud_gen")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def build_queue(geography: str | None = None, retry_failed: bool = False,
                allow_restricted: bool = False, allow_blocked: bool = False,
                only_policy: str | None = None) -> int:
    """Build queue from manifests minus completed items, minus policy refusals."""
    all_items = []

    if retry_failed:
        failures = load_failed()
        retryable = [f for f in failures if f.get("retry_count", 0) < f.get("max_retries", 3)]
        # Re-queue failed items by ID — we need their full data from manifests
        failed_ids = {f["id"] for f in retryable}
        for manifest_file in sorted(MANIFEST_DIR.glob("relief_manifest_*.json")):
            with open(manifest_file) as f:
                data = json.load(f)
            for item in data["meshes"]:
                if item["id"] in failed_ids:
                    all_items.append(item)
        # NB: the screen below applies to this branch too — retrying a failure
        # must not be a way around the gate — so the count is reported after it,
        # not here, where it would claim items the screen goes on to refuse.
        print(f"Found {len(all_items)} failed items to screen")
    else:
        # Load all manifest items
        for manifest_file in sorted(MANIFEST_DIR.glob("relief_manifest_*.json")):
            with open(manifest_file) as f:
                data = json.load(f)
            for item in data["meshes"]:
                if geography and item.get("geography") != geography:
                    continue
                all_items.append(item)

    # Content policy BEFORE anything is queued, so a weapon prompt never
    # reaches a paid API and no refused item can consume budget. Refusals are
    # logged to state/content_policy_skipped.json, and each surviving item
    # carries its verdict into the queue so the worker can stamp it onto the
    # asset's metadata.json.
    all_items = content_policy.select_tier(all_items, only_policy)
    permitted, rejected = content_policy.screen_items(
        all_items, allow_restricted, allow_blocked)
    if rejected:
        content_policy.record_skipped(rejected, phase="cloud_gen")
    overrides = [it for it in permitted if it["policy"]["override"]]
    print(content_policy.describe_screen(permitted, rejected, overrides))
    all_items = permitted

    # Find completed items (GLB exists and >1KB)
    completed = set()
    if OUTPUT_DIR.exists():
        for glb in OUTPUT_DIR.rglob("*.glb"):
            if glb.stat().st_size > 1024:
                completed.add(glb.parent.name)  # folder name = item id

    # Build queue: pending items only
    queue = [item for item in all_items if item["id"] not in completed]

    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f)

    return len(queue)


# ---------------------------------------------------------------------------
# Provider Abstraction
# ---------------------------------------------------------------------------

# Estimated costs per call (for fal.ai which doesn't report cost per request)
FAL_IMAGE_COST = 0.133   # GPT-Image-1.5 HQ 1024x1024
FAL_MESH_COST = 0.30      # Trellis 2 or Hunyuan3D



def enhance_prompt(base: str) -> str:
    """Use the shared reference recipe without contradictory style suffixes."""
    return normalize_generation_prompt(base)


class ProviderBase(ABC):
    @abstractmethod
    def generate_image(self, prompt: str) -> tuple[str, float]:
        """Returns (image_url, cost_usd)."""

    @abstractmethod
    def generate_mesh(self, image_url: str, prompt: str) -> tuple[str, float]:
        """Returns (glb_url, cost_usd)."""


class FalProvider(ProviderBase):
    IMAGE_MODEL = "fal-ai/gpt-image-1.5"
    MESH_MODELS = [
        # Trellis 2: higher guidance for hard-surface disaster equipment
        ("fal-ai/trellis-2", "image_url", {
            "resolution": 1024,
            "texture_size": 2048,
            "ss_guidance_strength": 8.0,  # stricter geometry for hard-surface objects
            "ss_sampling_steps": 16,
            "slat_guidance_strength": 3.0,
            "slat_sampling_steps": 16,
        }),
        ("fal-ai/hunyuan3d-v3/image-to-3d", "input_image_url", {}),
    ]

    def __init__(self, api_key: str):
        os.environ["FAL_KEY"] = api_key
        import fal_client
        self.fal = fal_client

    def generate_image(self, prompt: str) -> tuple[str, float]:
        result = self.fal.subscribe(
            self.IMAGE_MODEL,
            arguments={
                "prompt": prompt,
                "image_size": "1024x1024",
                "num_images": 1,
                "background": "opaque",
                "quality": "high",
                "output_format": "png",
            },
        )
        images = result.get("images", [])
        if not images or not images[0].get("url"):
            raise RuntimeError("No image URL in fal response")
        return images[0]["url"], FAL_IMAGE_COST

    def generate_mesh(self, image_url: str, prompt: str) -> tuple[str, float]:
        for model_id, param_name, extra in self.MESH_MODELS:
            try:
                args = {param_name: image_url}
                args.update(extra)
                result = self.fal.subscribe(model_id, arguments=args)
                glb_url = (
                    result.get("glb_url")
                    or result.get("model_mesh", {}).get("url")
                    or result.get("model_glb", {}).get("url")
                )
                if glb_url:
                    return glb_url, FAL_MESH_COST
            except Exception:
                continue
        raise RuntimeError("No GLB URL from any fal mesh model")


class RunwareProvider(ProviderBase):
    REST_URL = "https://api.runware.ai/v1"
    IMAGE_MODEL = "openai:gpt-image@2"
    MESH_MODELS = [
        "microsoft:trellis-2@4b",
        "tripo:v3.1@0",
        "tencent:hunyuan-3d@3.1-rapid",
    ]
    MAX_RETRIES = 3
    BACKOFF_BASE = 5  # seconds

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _post_with_retry(self, payload: list[dict], timeout: int = 300, label: str = "") -> dict:
        """POST with exponential backoff for rate limits."""
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.REST_URL, json=payload,
                    headers=self.headers, timeout=timeout,
                )
                if resp.status_code == 402:
                    raise RuntimeError("Runware quota exhausted (402)")
                if resp.status_code == 429:
                    if attempt < self.MAX_RETRIES:
                        wait = self.BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 2)
                        print(f"  [Runware] Rate limited, waiting {wait:.0f}s (attempt {attempt+1})", flush=True)
                        time.sleep(wait)
                        continue
                    raise RuntimeError("Runware rate limited after retries (429)")
                resp.raise_for_status()
                data = resp.json()
                results = data.get("data", [])
                if not results:
                    raise RuntimeError(f"Empty response from Runware: {data}")
                return results[0]
            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES:
                    wait = self.BACKOFF_BASE * (2 ** attempt)
                    print(f"  [Runware] Timeout on {label}, retrying in {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Runware timeout after {self.MAX_RETRIES} retries on {label}")
        raise RuntimeError(f"Runware exhausted all retries on {label}")

    def generate_image(self, prompt: str) -> tuple[str, float]:
        task_uuid = str(uuid.uuid4())
        payload = [{
            "taskType": "imageInference",
            "taskUUID": task_uuid,
            "model": self.IMAGE_MODEL,
            "positivePrompt": prompt,
            "width": 1024,
            "height": 1024,
            "includeCost": True,
            "deliveryMethod": "sync",
        }]
        result = self._post_with_retry(payload, timeout=120, label="image")
        image_url = result.get("imageURL")
        if not image_url:
            raise RuntimeError(f"No imageURL in Runware response: {result}")
        cost = result.get("cost", 0.05)
        return image_url, cost

    # Model-specific optimal settings based on API docs and research.
    # Trellis 2: image (singular), settings for resolution/texture/guidance.
    # Tripo v3.1: images (array), settings for geometry/texture quality.
    # Hunyuan3D: images (array), minimal settings.
    MODEL_SETTINGS = {
        "microsoft:trellis-2@4b": {
            "inputs_key": "image",     # singular string
            "settings": {
                "resolution": 1024,
                "textureSize": 2048,
                "textureFormat": "WEBP",
                "remesh": True,
                "shapeSlat": {"steps": 16, "guidanceStrength": 8.0},
                "sparseStructure": {"steps": 16},
                "texSlat": {"steps": 16},
            },
        },
        "tripo:v3.1@0": {
            "inputs_key": "images",    # array
            "settings": {
                "geometryQuality": "detailed",
                "textureQuality": "detailed",
                "pbr": True,
                "texture": True,
                "imageAutoFix": True,
            },
        },
        "tencent:hunyuan-3d@3.1-rapid": {
            "inputs_key": "images",    # array
            "settings": {},
        },
    }

    def generate_mesh(self, image_url: str, prompt: str) -> tuple[str, float]:
        for model_id in self.MESH_MODELS:
            try:
                task_uuid = str(uuid.uuid4())
                model_cfg = self.MODEL_SETTINGS.get(model_id, {})
                inputs_key = model_cfg.get("inputs_key", "images")
                settings = model_cfg.get("settings", {})

                # Build inputs based on model requirements
                if inputs_key == "image":
                    inputs = {"image": image_url}  # Trellis 2: singular string
                else:
                    inputs = {"images": [image_url]}  # Tripo/Hunyuan: array

                payload = [{
                    "taskType": "3dInference",
                    "taskUUID": task_uuid,
                    "model": model_id,
                    "inputs": inputs,
                    "positivePrompt": prompt[:1024],  # Tripo has 1024 char limit
                    "negativePrompt": "blurry, low quality, distorted geometry, extra parts, floating pieces",
                    "outputFormat": "GLB",
                    "includeCost": True,
                    "deliveryMethod": "sync",
                }]
                if settings:
                    payload[0]["settings"] = settings

                result = self._post_with_retry(payload, timeout=600, label=f"3d:{model_id}")
                # Handle both nested 'outputs.files' and flat 'modelURL' formats
                files = result.get("outputs", {}).get("files", [])
                glb_url = None
                if files and files[0].get("url"):
                    glb_url = files[0]["url"]
                elif result.get("modelURL"):
                    glb_url = result["modelURL"]
                elif result.get("model_url"):
                    glb_url = result["model_url"]
                if glb_url:
                    cost = result.get("cost", 0.30)
                    return glb_url, cost
            except RuntimeError:
                raise  # re-raise quota/rate errors
            except Exception as e:
                print(f"  [Runware] {model_id} failed: {e}", flush=True)
                continue
        raise RuntimeError("No GLB URL from any Runware mesh model")


def create_provider(provider_type: str, api_key: str) -> ProviderBase:
    if provider_type == "fal":
        return FalProvider(api_key)
    elif provider_type == "runware":
        return RunwareProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


# ---------------------------------------------------------------------------
# Worker Process
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, timeout: int = 120, max_retries: int = 3):
    """Download a file from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            # Write to temp file first, then rename (atomic)
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            tmp_path.rename(dest)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt) + random.uniform(0, 2)
                print(f"  [Download] Retry {attempt+1}/{max_retries} for {dest.name}: {e}", flush=True)
                time.sleep(wait)
            else:
                raise


def worker_process(worker_id: int, key_config: dict, stats_queue: mp.Queue, shutdown_event: mp.Event):
    """Worker process for a single API key."""
    provider_type = key_config["provider"]
    key_name = key_config["name"]
    api_key = key_config["key"]

    print(f"[W{worker_id}] Started ({key_name}, {provider_type})", flush=True)

    try:
        provider = create_provider(provider_type, api_key)
    except Exception as e:
        print(f"[W{worker_id}] Failed to init provider: {e}", flush=True)
        stats_queue.put(("error", worker_id, key_name, str(e)))
        return

    count = 0
    while not shutdown_event.is_set():
        # Get next item
        item = get_next_mesh()
        if item is None:
            print(f"[W{worker_id}] Queue empty, exiting", flush=True)
            break

        item_id = item["id"]
        category = item.get("category", "")
        geography = item.get("geography", "")

        # Output path
        output_path = OUTPUT_DIR / geography / category / item_id
        glb_file = output_path / f"{item_id}.glb"
        png_file = output_path / f"{item_id}.png"

        # Skip if already done
        if glb_file.exists() and glb_file.stat().st_size > 1024:
            stats_queue.put(("skip", worker_id, key_name, item_id))
            continue

        # Budget check before API call
        key_budget = key_config.get("max_budget", 0)
        if key_budget > 0:
            ledger = load_spending()
            key_data = ledger.get("keys", {}).get(key_name, {})
            spent_so_far = key_data.get("total_spent", 0)
            if spent_so_far >= key_budget * 0.90:
                print(f"[W{worker_id}] Key {key_name} near budget (${spent_so_far:.2f}/${key_budget:.0f}), stopping",
                      flush=True)
                stats_queue.put(("exhausted", worker_id, key_name, item_id))
                requeue_item(item)
                break

        print(f"[W{worker_id}] {item_id}", flush=True)

        try:
            # Generate image
            prompt = enhance_prompt(item["prompt"])
            image_url, img_cost = provider.generate_image(prompt)
            record_spending(key_name, provider_type, img_cost, "image", item_id,
                           max_budget=key_config.get("max_budget", 0))

            # Generate mesh (try models in order)
            glb_url, mesh_cost = provider.generate_mesh(image_url, item["prompt"])
            record_spending(key_name, provider_type, mesh_cost, "mesh", item_id,
                           max_budget=key_config.get("max_budget", 0))

            # Download files
            output_path.mkdir(parents=True, exist_ok=True)

            try:
                download_file(image_url, png_file)
            except Exception:
                pass  # PNG is optional

            download_file(glb_url, glb_file)

            # Sanity check: size and GLB magic bytes
            if glb_file.stat().st_size < 1024:
                glb_file.unlink(missing_ok=True)
                raise RuntimeError("GLB file too small (corrupt)")
            with open(glb_file, "rb") as f:
                header = f.read(4)
            if header[:4] != b"glTF":
                glb_file.unlink(missing_ok=True)
                raise RuntimeError(f"GLB file missing glTF header (got {header!r})")

            # Save metadata
            metadata = {
                "id": item_id,
                "category": category,
                "geography": geography,
                "organization": item.get("organization", ""),
                "prompt": prompt,
                "provider": provider_type,
                "image_model": provider.IMAGE_MODEL if hasattr(provider, "IMAGE_MODEL") else "unknown",
                # The policy verdict travels with the asset, as in the local
                # pipeline: policy.override is the only durable trace on the
                # asset itself that a flag was needed to produce it, so a
                # release can filter on it without consulting run logs.
                #
                # Indexed, not .get(): every queued item passed through
                # screen_items, so an unstamped item here means the screen was
                # bypassed. A KeyError is the right outcome for that — defaulting
                # to {} would write an asset whose provenance silently reads as
                # "no override".
                "policy": item["policy"],
                "timestamp": datetime.now().isoformat(),
            }
            with open(output_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            count += 1
            stats_queue.put(("ok", worker_id, key_name, item_id))

            # Logged only now that the artifacts are on disk and the item is
            # already counted as a success: being permitted is not the same as
            # being generated, and a ledger write must not be able to turn a
            # completed asset into a failure that gets requeued.
            if item["policy"].get("override"):
                try:
                    record_override(item)
                except Exception as e:
                    print(f"[W{worker_id}] WARNING: override ledger write "
                          f"failed for {item_id}: {e}", flush=True)

        except RuntimeError as e:
            err = str(e).lower()
            if any(x in err for x in ["credit", "quota", "exceeded", "insufficient", "exhausted"]):
                print(f"[W{worker_id}] Key exhausted!", flush=True)
                stats_queue.put(("exhausted", worker_id, key_name, item_id))
                record_failure(item_id, str(e), provider_type, key_name)
                requeue_item(item)
                break
            else:
                print(f"[W{worker_id}] FAIL {item_id}: {e}", flush=True)
                stats_queue.put(("fail", worker_id, key_name, item_id))
                record_failure(item_id, str(e), provider_type, key_name)
                requeue_item(item)

        except Exception as e:
            print(f"[W{worker_id}] ERROR {item_id}: {e}", flush=True)
            stats_queue.put(("fail", worker_id, key_name, item_id))
            record_failure(item_id, str(e), provider_type, key_name)
            requeue_item(item)

    print(f"[W{worker_id}] Done ({count} generated)", flush=True)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

def monitor_loop(stats_queue: mp.Queue, shutdown_event: mp.Event, total_keys: int):
    """Main process: collect stats and display progress."""
    stats = {"ok": 0, "skip": 0, "fail": 0, "exhausted": 0, "error": 0}
    finished_keys = 0

    while finished_keys < total_keys and not shutdown_event.is_set():
        try:
            msg = stats_queue.get(timeout=5)
            status = msg[0]
            stats[status] = stats.get(status, 0) + 1

            if status in ("exhausted", "error"):
                finished_keys += 1

            total = stats["ok"] + stats["skip"] + stats["fail"]
            print(
                f"\r[Monitor] OK:{stats['ok']} Skip:{stats['skip']} "
                f"Fail:{stats['fail']} Exhausted:{stats['exhausted']} "
                f"| Keys done:{finished_keys}/{total_keys}",
                end="", flush=True,
            )
        except Exception:
            # Timeout — check queue; if empty, workers will exit naturally
            if QUEUE_FILE.exists():
                with open(QUEUE_FILE) as f:
                    remaining = len(json.load(f))
                if remaining == 0:
                    # Wait a bit for workers to finish their last items and exit
                    time.sleep(10)
                    break

    print(f"\n[Monitor] Final: {stats}", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate disaster relief meshes")
    parser.add_argument("--provider", choices=["fal", "runware", "all"], default="all",
                        help="Which API provider to use")
    parser.add_argument("--geography", choices=["germany", "eu", "ukraine", "general", "all"],
                        default="all", help="Which geography to generate")
    parser.add_argument("--workers-per-key", type=int, default=15,
                        help="Worker processes per API key")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build queue and show stats, don't generate")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry previously failed items")
    content_policy.add_policy_args(parser)
    args = parser.parse_args()

    # Parse keys
    all_keys = parse_api_keys()
    sync_key_spending(all_keys)

    # Filter by provider
    if args.provider != "all":
        all_keys = [k for k in all_keys if k.provider == args.provider]

    # Filter exhausted keys
    active_keys = [k for k in all_keys if not k.is_exhausted]
    exhausted_keys = [k for k in all_keys if k.is_exhausted]

    print(f"API Keys: {len(active_keys)} active, {len(exhausted_keys)} exhausted")
    for k in active_keys:
        print(f"  [{k.provider}] {k.name}: budget ${k.max_budget:.0f}, "
              f"spent ${k.spent:.2f}, remaining ${k.remaining:.2f}")
    for k in exhausted_keys:
        print(f"  [{k.provider}] {k.name}: EXHAUSTED (${k.spent:.2f}/${k.max_budget:.0f})")

    if not active_keys:
        print("ERROR: No active API keys. Check config/api_keys.txt")
        sys.exit(1)

    # Build queue
    geo = None if args.geography == "all" else args.geography
    queue_size = build_queue(geography=geo, retry_failed=args.retry_failed,
                             allow_restricted=args.allow_restricted,
                             allow_blocked=args.allow_blocked,
                             only_policy=args.only_policy)

    # Count completed items for display
    completed_count = 0
    if OUTPUT_DIR.exists():
        for glb in OUTPUT_DIR.rglob("*.glb"):
            if glb.stat().st_size > 1024:
                completed_count += 1

    if args.geography == "all":
        print(f"\nQueue: {queue_size} items pending ({completed_count} completed)")
    else:
        print(f"\nQueue: {queue_size} {args.geography} items pending")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"Would generate {queue_size} items with {len(active_keys)} keys")
        est_per_item = 0.43  # average cost
        est_cost = queue_size * est_per_item
        print(f"Estimated cost: ~${est_cost:,.0f} (at ~${est_per_item}/item)")
        print(f"Estimated time: ~{queue_size * 45 / (len(active_keys) * args.workers_per_key) / 3600:.1f} hours")
        return

    # Shutdown handling
    shutdown_event = mp.Event()
    stats_queue = mp.Queue()

    def handle_signal(sig, frame):
        print("\n[Main] Shutdown requested, finishing current items...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Spawn workers
    processes = []
    worker_id = 0
    for key in active_keys:
        key_dict = {
            "provider": key.provider,
            "name": key.name,
            "key": key.key,
            "max_budget": key.max_budget,
        }
        for _ in range(args.workers_per_key):
            worker_id += 1
            p = mp.Process(
                target=worker_process,
                args=(worker_id, key_dict, stats_queue, shutdown_event),
            )
            p.start()
            processes.append(p)

    print(f"\nLaunched {len(processes)} workers across {len(active_keys)} keys")

    # Monitor
    stats = monitor_loop(stats_queue, shutdown_event, len(processes))

    # Cleanup
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Final stats
    remaining = 0
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE) as f:
            remaining = len(json.load(f))

    print(f"\n=== Generation Complete ===")
    print(f"Generated: {stats.get('ok', 0)}")
    print(f"Skipped: {stats.get('skip', 0)}")
    print(f"Failed: {stats.get('fail', 0)}")
    print(f"Remaining: {remaining}")

    # Print spending
    ledger = load_spending()
    print(f"\n=== Spending Summary ===")
    for key_name, data in ledger.get("keys", {}).items():
        print(f"  {key_name}: ${data.get('total_spent', 0):.2f} "
              f"({data.get('request_count', 0)} requests)")


if __name__ == "__main__":
    main()
