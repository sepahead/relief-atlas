"""Repair a confirmed legacy local RGB-in-DC SPZ export into an isolated copy.

An untagged SPZ cannot identify this error by itself. Use only with known writer
provenance: ordinary valid SPZ files already contain SH DC and must not undergo
this conversion. Source assets are preserved. The copy needs fresh QA.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np

from gs_export import C0, tmp_sibling, write_spz
from pipeline_utils import (GS_COLOR_ENCODINGS, file_sha256, load_json, sync_directory,
                            durable_replace, write_json)
from qa import _load_spz_gaussians


def repair_spz(source, destination, *, confirmed_legacy_rgb=False):
    """Copy the complete item, repair its SPZ, verify native decoding and commit metadata."""
    if not confirmed_legacy_rgb:
        raise ValueError("Confirm the legacy RGB-in-DC writer from provenance before conversion")
    source, destination = Path(source), Path(destination)
    if source.suffix != ".spz":
        raise ValueError("Source must be an SPZ file")
    if destination.exists() or source.parent.resolve() in destination.resolve().parents:
        raise ValueError("Choose a new output directory outside the source item")
    metadata = load_json(source.parent / "metadata.json")
    if metadata.get("mesh_model") != "local/trellis.2-4b":
        raise ValueError("Only confirmed local TRELLIS exports are supported")
    if metadata.get("gs_color_encodings", {}).get("spz") == "sh_dc":
        raise ValueError("Source is already tagged with the corrected SH DC encoding")
    cloud = _load_spz_gaussians(source)
    if cloud["sh_degree"] != 0:
        raise ValueError("Legacy local repair supports degree-zero voxel Gaussians only")
    legacy_rgb = cloud["f_dc"]
    if not np.isfinite(legacy_rgb).all() or legacy_rgb.min() < -0.04 or legacy_rgb.max() > 1.04:
        raise ValueError("Source values do not match the confirmed legacy RGB convention")
    # The old exporter wrote RGB into the SH field. The native decoder gives
    # those same numerical values back, with inherited packing quantization.
    expected_rgb = np.clip(legacy_rgb, 0, 1)
    cloud["f_dc"] = (expected_rgb - 0.5) / C0
    source_hash = file_sha256(source)
    source_dir = source.parent.resolve()
    # Never expose the source commit marker in a partly copied destination.
    # copytree may be interrupted before it returns, so removing it afterwards
    # would leave a crash window. Nested collection metadata is preserved.
    shutil.copytree(source_dir, destination,
                    ignore=lambda directory, names: ["metadata.json"]
                    if Path(directory) == source_dir else [])
    marker = destination / "metadata.json"
    sync_directory(destination)
    final = destination / source.name
    temporary = tmp_sibling(final)
    if write_spz(temporary, cloud) != len(legacy_rgb):
        raise RuntimeError("SPZ repair could not be written; source retained, copy remains uncommitted")
    import spz
    native = spz.load_spz(str(temporary))
    actual_rgb = np.asarray(native.colors).reshape(-1, 3) * C0 + 0.5
    if not np.allclose(actual_rgb, expected_rgb, atol=0.01, rtol=0):
        raise RuntimeError("Native SPZ decoder did not reproduce repaired RGB; source retained")
    durable_replace(temporary, final)
    # Sync copied artifact data before publishing metadata as the commit marker.
    for path in destination.iterdir():
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    sync_directory(destination)
    repair = {"at": datetime.now(timezone.utc).isoformat(), "kind": "spz_rgb_to_sh_dc",
              "source": str(source.resolve()), "source_sha256": source_hash,
              "output_sha256": file_sha256(final), "inherited_rgb_quantization": True,
              "code_sha256": file_sha256(Path(__file__)),
              "native_decoder_max_rgb_error": float(np.max(np.abs(actual_rgb - expected_rgb)))}
    metadata["gs_color_encodings"] = {
        ext: encoding for ext, encoding in GS_COLOR_ENCODINGS.items()
        if (destination / f"{source.stem}.{ext}").is_file()}
    metadata.setdefault("repairs", []).append(repair)
    write_json(marker, metadata)
    return repair


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new isolated item directory")
    parser.add_argument("--confirm-legacy-rgb", action="store_true",
                        help="the source writer is known to have stored RGB in the SH DC field")
    args = parser.parse_args()
    repair = repair_spz(args.source, args.output, confirmed_legacy_rgb=args.confirm_legacy_rgb)
    print(f"Repaired {args.output / args.source.name}; native RGB max error "
          f"{repair['native_decoder_max_rgb_error']:.5f}; fresh QA required")


if __name__ == "__main__":
    main()
