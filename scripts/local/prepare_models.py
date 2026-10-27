"""Download pinned local checkpoints and wire TRELLIS for offline use.

Run with imgenv's Python. DINOv3 requires an approved Hugging Face account;
credentials come from the standard HF login and are never written to this repo.
Existing weights are validated before deciding whether a download is needed.
"""

import argparse
import json
import os
import struct
from pathlib import Path

from pipeline_utils import MODEL_DIR, PROJECT_DIR, file_sha256, load_json, write_json

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ["HF_HUB_DISABLE_XET"] = "1"

CATALOG_PATH = PROJECT_DIR / "config" / "local_models.json"


def catalog():
    return load_json(CATALOG_PATH)["models"]


def validate_safetensors(path):
    """Check header and payload bounds without loading multi-GB tensors."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as fp:
        raw = fp.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        if not 0 < header_len < min(size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header length: {path}")
        tensors = json.loads(fp.read(header_len))
    payload = size - 8 - header_len
    entries = [v for k, v in tensors.items() if k != "__metadata__"]
    if not entries:
        raise ValueError(f"Empty checkpoint: {path}")
    offsets = sorted(v["data_offsets"] for v in entries)
    cursor = 0
    for start, end in offsets:
        if start != cursor or not start <= end <= payload:
            raise ValueError(f"Truncated or inconsistent tensor payload: {path}")
        cursor = end
    if cursor != payload:
        raise ValueError(f"Unaccounted checkpoint payload: {path}")


def model_errors(spec, root=MODEL_DIR):
    d = Path(root) / spec["directory"]
    errors = []
    for name in spec["required"]:
        p = d / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f"Missing {p}")
        elif p.suffix == ".json":
            try:
                load_json(p)
            except ValueError as exc:
                errors.append(str(exc))
    checkpoints = set()
    for directory in spec["weight_directories"]:
        weights = list((d / directory).glob("*.safetensors"))
        bins = list((d / directory).glob("*.bin"))
        if not weights and not bins:
            errors.append(f"No weights in {d / directory}")
        checkpoints.update(weights)
        for index in (d / directory).glob("*.safetensors.index.json"):
            try:
                for shard in set(load_json(index)["weight_map"].values()):
                    if not (index.parent / shard).is_file():
                        errors.append(f"Missing weight shard {index.parent / shard}")
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"Invalid weight index {index}: {exc}")
    checkpoints.update(d / name for name in spec["required"] if name.endswith(".safetensors"))
    for path in checkpoints:
        if path.is_file():
            try:
                validate_safetensors(path)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(str(exc))
    if spec["repo_id"] == "Qwen/Qwen-Image-2.1":
        for name, expected in (("model_index.json", "QwenImage21Pipeline"),
                               ("transformer/config.json", "QwenImage21Transformer2DModel"),
                               ("vae/config.json", "AutoencoderKLQwenImage21")):
            if (d / name).is_file():
                try:
                    if load_json(d / name).get("_class_name") != expected:
                        errors.append(f"Qwen-Image-2.1 requires {expected} in {name}")
                except (ValueError, AttributeError) as exc:
                    errors.append(str(exc))
    if spec["repo_id"] == "microsoft/TRELLIS.2-4B" and (d / "pipeline.json").is_file():
        try:
            models = load_json(d / "pipeline.json")["args"]["models"]
            for name, checkpoint in models.items():
                if "TRELLIS-image-large" in checkpoint:
                    checkpoint = "ckpts/ss_dec_conv3d_16l8_fp16"
                for suffix in (".json", ".safetensors"):
                    if not (d / (checkpoint + suffix)).is_file():
                        errors.append(f"Missing {name}: {d / (checkpoint + suffix)}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"Invalid TRELLIS pipeline: {exc}")
    if spec["repo_id"] == "facebook/dinov3-vitl16-pretrain-lvd1689m" and (d / "config.json").is_file():
        try:
            config = load_json(d / "config.json")
            expected = {"model_type": "dinov3_vit", "hidden_size": 1024,
                        "num_hidden_layers": 24, "patch_size": 16}
            if any(config.get(key) != value for key, value in expected.items()):
                errors.append("DINO checkpoint is not the ViT-L/16 architecture required by TRELLIS.2")
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))
    return errors


def wire_offline(root=MODEL_DIR):
    root = Path(root)
    path = root / "TRELLIS.2-4B" / "pipeline.json"
    if not path.is_file():
        return
    data = load_json(path)
    data["args"]["models"]["sparse_structure_decoder"] = "ckpts/ss_dec_conv3d_16l8_fp16"
    data["args"]["image_cond_model"]["args"]["model_name"] = str(root / "DINOv3-vitl16")
    write_json(path, data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(catalog()), default=None)
    ap.add_argument("--force", action="store_true", help="refresh an existing snapshot at the pinned revision")
    ap.add_argument("--offline", action="store_true", help="validate without downloading")
    ap.add_argument("--fingerprint", action="store_true", help="record SHA256 for all local checkpoint files")
    ap.add_argument("--resumable-dino", "--parallel-dino", dest="resumable_dino", action="store_true",
                    help="use resumable aria2 transfer for DINO; credentials remain in memory")
    ap.add_argument("--resumable-qwen", "--parallel-qwen", dest="resumable_qwen", action="store_true",
                    help="resume Qwen with one connection and one file at a time (256 KiB/s cap by default)")
    ap.add_argument("--unlimited-qwen", action="store_true",
                    help="remove the speed cap from the resumable Qwen transfer")
    args = ap.parse_args(argv)
    specs = catalog()
    manifest_path = MODEL_DIR / "model_manifest.json"
    manifest = load_json(manifest_path)
    failures = []
    names = list(args.only or specs)
    # TRELLIS's pipeline references the decoder in a separate model repo. Fetch
    # that dependency first so a fresh setup succeeds on its first invocation.
    if "trellis" in names:
        names = ["decoder", *[name for name in names if name != "decoder"]]
    for name in names:
        spec = specs[name]
        errors = model_errors(spec)
        fetched = False
        if (errors or args.force) and not args.offline:
            try:
                if name == "qwen" and args.resumable_qwen:
                    from checkpoint_download import download_snapshot
                    # Qwen is public; approved-account credentials are needed
                    # only for gated checkpoints such as DINOv3.
                    download_snapshot(spec, MODEL_DIR, force=args.force,
                                      speed_limit_kib=None if args.unlimited_qwen else 256)
                elif name == "dino" and args.resumable_dino:
                    from huggingface_hub import get_token
                    from checkpoint_download import download_checkpoint
                    download_checkpoint(spec, MODEL_DIR, token=get_token())
                else:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"],
                                      local_dir=MODEL_DIR / spec["directory"],
                                      allow_patterns=spec["allow_patterns"], max_workers=1)
                fetched = True
                errors = model_errors(spec)
            except Exception as exc:
                errors = [f"{type(exc).__name__}: {exc}"]
                if spec.get("gated"):
                    errors.append(f"Approve access at https://huggingface.co/{spec['repo_id']} "
                                  "and run hf auth login, then rerun --only dino.")
        if errors:
            manifest[name] = {**manifest.get(name, {}), "repo_id": spec["repo_id"],
                              "requested_revision": spec["revision"],
                              "validated": False, "errors": errors}
            failures += errors
            print(f"{name}: NOT READY\n  " + "\n  ".join(errors), flush=True)
            continue
        previous = manifest.get(name, {})
        record = {"repo_id": spec["repo_id"], "directory": spec["directory"],
                  "requested_revision": spec["revision"],
                  "downloaded_revision": spec["revision"] if fetched else previous.get("downloaded_revision"),
                  "validated": True}
        if fetched and name == "qwen" and args.resumable_qwen:
            record["weight_source"] = "official checkpoints with pinned SHA256; resumable single-connection transfer"
        if args.fingerprint or spec.get("checkpoint_sha256"):
            d = MODEL_DIR / spec["directory"]
            weights = sorted(set(p for p in d.rglob("*.safetensors") if ".cache" not in p.parts))
            record["checkpoint_sha256"] = {str(p.relative_to(d)): file_sha256(p) for p in weights}
            mismatches = [f"Official checkpoint SHA256 mismatch: {d / path}"
                          for path, expected in spec.get("checkpoint_sha256", {}).items()
                          if record["checkpoint_sha256"].get(path) != expected]
            if mismatches:
                record.update(validated=False, errors=mismatches)
                manifest[name] = record
                failures += mismatches
                print(f"{name}: NOT READY\n  " + "\n  ".join(mismatches), flush=True)
                continue
        elif "checkpoint_sha256" in previous:
            record["checkpoint_sha256"] = previous["checkpoint_sha256"]
        manifest[name] = record
        print(f"{name}: READY", flush=True)
    if not failures:
        wire_offline()
    write_json(manifest_path, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
