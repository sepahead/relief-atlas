"""Derive the shippable set from the ledgers, not from what happens to be on disk.

Every gate in this pipeline records a verdict, and until now nothing read them
back: state/qa_meshes.json and state/qa_gaussians.json were written by qa.py and
consumed only by verify_local.py's progress display. "What ships" was therefore
whatever files existed under outputs_relief/ -- which is not the same set, and
the difference is not academic:

  * outputs_relief/germany/legacy_weapon_g36_rifle/ holds an 18.5 MB assault
    rifle GLB, one of 27 `blocked`-tier weapon meshes generated before the
    content policy existed. Mesh QA even passed them, because mesh QA judges
    geometry and deliberately does not re-screen subject matter.
  * a mesh that failed the aspect / dim_ratio / view-consistency checks used to
    be left on disk with its metadata intact, so mesh_complete() called it done.

So the release set is computed here, positively: an item ships only if the
policy allows it *on its own merits*, no override was involved, the deliverable
is complete, and all three gates recorded a PASS under the current QA contract.
Anything else is excluded with an attributable reason.

Fail-closed throughout. A missing verdict is treated exactly like a failing one,
and a verdict from an older QA contract (prompt_utils.QA_VERSION) does not count
-- a stale pass would otherwise let an asset skip every live gate.

Stdlib only, so it runs on the system interpreter without the model workspace.

    python3 scripts/local/release_manifest.py                  # report + write
    python3 scripts/local/release_manifest.py --verbose        # list exclusions
    python3 scripts/local/release_manifest.py --dry-run        # no file written
    python3 scripts/local/release_manifest.py --allow-missing-gs-qa
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import content_policy  # noqa: E402
from prompt_utils import (  # noqa: E402
    GS_EXTENSIONS, MANIFESTS, OUTPUT_DIR, QA_VERSION, STATE_DIR,
    load_manifest, mesh_complete, verdict_is_current,
)
from pipeline_utils import (  # noqa: E402
    file_sha256, image_model_matches, source_image_matches, source_image_metadata, verdict_matches,
)
DEFAULT_OUTPUT = STATE_DIR / "release_manifest.json"


def load_state(name: str, state_dir=None) -> dict:
    p = Path(state_dir or STATE_DIR) / name
    return json.loads(p.read_text()) if p.exists() else {}


# Review artifacts that live in the item directory but are not deliverables:
# the QA contact sheets (.gitignore excludes qa_*.png) and the pre-matte frame
# kept so matting can be re-tuned without paying generation cost again.
NON_DELIVERABLE_SUFFIXES = (".qa_views.png", ".qa_mesh_views.png",
                            ".raw.jpg", ".debug.png")


def is_deliverable(path: Path) -> bool:
    if path.name.startswith("."):
        return False
    return (not path.name.endswith(NON_DELIVERABLE_SUFFIXES)
            and (path.name in ("metadata.json", "image_meta.json")
                 or path.suffix in (".glb", ".png", ".spz", ".splat", ".ply")))


def _gate_ok(verdicts: dict, item_id: str) -> bool:
    """A gate is satisfied only by a current-contract PASS that actually ran.

    Absent verdict, stale verdict and failing verdict are all the same answer:
    this asset has not been certified by the gate that is running today.

    `view_error` is rejected too. When the turntable fails to render, qa.py
    records the error and still passes the item -- reasonably, since a QA
    infrastructure failure is not an asset defect, and failing there would wipe
    every mesh the moment trimesh broke. But it means the coverage and
    view-consistency checks provably did not execute, so the verdict is not a
    certification and must not be treated as one here.
    """
    v = verdicts.get(item_id)
    if not (verdict_is_current(v) and v.get("pass")):
        return False
    return not v.get("view_error")


def exclusion_reason(item, item_dir, *, image_qa, mesh_qa, gs_qa, quarantine,
                     require_gs_qa=True):
    """Why this item must not ship, or None if it may.

    Order is policy -> existence -> provenance -> completeness -> quarantine ->
    the three gates in the order they run. Policy comes first so that a refused
    subject is attributed to the policy even when it was never generated;
    otherwise a reader of the report cannot tell refusal from absence.

    Note this is deliberately stricter than pending.classify(), which lets a
    complete deliverable outrank a quarantine entry. The two answer different
    questions: pending asks "is there work left to do", where a finished asset is
    finished; release asks "may this ship", where a gate having given up on an
    item is disqualifying regardless of what is on disk.
    """
    verdict = content_policy.classify_item(item)
    if verdict.classification != content_policy.ALLOWED:
        return f"policy {verdict.classification} ({verdict.category})"

    if not item_dir.is_dir():
        return "not generated"

    # Defence in depth: the policy above judges the manifest row, this catches an
    # asset whose own metadata admits it needed a flag. The two disagree only if
    # the policy families changed after generation, which is exactly when you
    # want to hear about it.
    #
    # The `exists()` guard is safe only because mesh_complete() below
    # independently requires metadata.json, so an asset with no metadata can
    # never reach a shippable verdict by skipping this check. That coupling is
    # load-bearing: do not relax mesh_complete()'s metadata requirement without
    # making this check unconditional.
    meta = item_dir / "metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            if not isinstance(data, dict) or not isinstance(data.get("policy", {}), dict):
                return "unreadable metadata.json"
            if data.get("policy", {}).get("override"):
                return "override-generated"
        except (json.JSONDecodeError, OSError):
            return "unreadable metadata.json"

    if not mesh_complete(item_dir, item["id"]):
        return "incomplete deliverable"
    if item["id"] in quarantine:
        return "quarantined"
    if not verdict_matches(image_qa.get(item["id"]), item_dir, item["id"], "images"):
        return "no current image-QA pass"
    if not image_model_matches(item_dir, item["id"]):
        return "reference image is not from the configured model"
    if not source_image_matches(item_dir, item["id"]):
        return "mesh source image mismatch"
    if not verdict_matches(mesh_qa.get(item["id"]), item_dir, item["id"], "meshes"):
        return "no current mesh-QA pass"
    if require_gs_qa and not verdict_matches(gs_qa.get(item["id"]), item_dir, item["id"], "gaussians"):
        return "no current 3DGS-QA pass"
    return None


def describe_item(item, item_dir) -> dict:
    """The record that goes into the release manifest."""
    allowed = {f"{item['id']}.{ext}" for ext in ("png", "glb", *GS_EXTENSIONS)}
    allowed.add("metadata.json")
    if source_image_metadata(item_dir, item["id"]):
        allowed.add("image_meta.json")
    files = {p.name: p.stat().st_size
             for p in sorted(item_dir.iterdir())
             if p.is_file() and p.name in allowed and is_deliverable(p)}
    # No guard needed: exclusion_reason() has already parsed this file and
    # returned None, so it exists and is valid JSON.
    meta = json.loads((item_dir / "metadata.json").read_text())
    return {
        "id": item["id"],
        "geography": item["geography"],
        "category": item["category"],
        "name": meta.get("name") or item.get("name", ""),
        "gs_format": meta.get("gs_format"),
        "vertices": meta.get("vertices"),
        "gaussians": meta.get("gaussians"),
        "bytes": sum(files.values()),
        "files": files,
        "sha256": {name: file_sha256(item_dir / name) for name in files},
    }


def build(output_dir=None, state_dir=None, require_gs_qa=True):
    """Return (included_records, excluded_by_id_reason)."""
    output_dir = Path(output_dir or OUTPUT_DIR)
    image_qa = load_state("qa_images.json", state_dir)
    mesh_qa = load_state("qa_meshes.json", state_dir)
    gs_qa = load_state("qa_gaussians.json", state_dir)
    quarantine = load_state("qa_quarantine.json", state_dir)

    included, excluded = [], {}
    for geo in MANIFESTS:
        for item in load_manifest(geo):
            d = output_dir / geo / item["category"] / item["id"]
            reason = exclusion_reason(
                item, d, image_qa=image_qa, mesh_qa=mesh_qa, gs_qa=gs_qa,
                quarantine=quarantine, require_gs_qa=require_gs_qa)
            if reason:
                excluded[item["id"]] = reason
            else:
                included.append(describe_item(item, d))
    return included, excluded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"where to write the manifest (default {DEFAULT_OUTPUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report without writing the manifest")
    ap.add_argument("--verbose", action="store_true",
                    help="list the excluded item ids under each reason")
    ap.add_argument("--allow-missing-gs-qa", action="store_true",
                    help="ship items whose 3DGS containers have no QA verdict "
                         "(for runs where phase 2.6 has not been run); the "
                         "manifest records that this was relaxed")
    args = ap.parse_args(argv)

    included, excluded = build(require_gs_qa=not args.allow_missing_gs_qa)
    by_reason = Counter(excluded.values())

    total = len(included) + len(excluded)
    total_bytes = sum(r["bytes"] for r in included)
    if args.allow_missing_gs_qa:
        print("** RELAXED: --allow-missing-gs-qa given; items ship without a "
              "3DGS QA verdict and this release is not fully certified")
    print(f"release: {len(included)} of {total} items ship "
          f"({total_bytes / 1e9:.2f} GB), {len(excluded)} excluded "
          f"(policy v{content_policy.POLICY_VERSION}, QA v{QA_VERSION})")
    for reason, n in by_reason.most_common():
        print(f"  {n:6d}  {reason}")
        if args.verbose:
            ids = sorted(i for i, r in excluded.items() if r == reason)
            for item_id in ids[:40]:
                print(f"          {item_id}")
            if len(ids) > 40:
                print(f"          ... and {len(ids) - 40} more")

    if args.dry_run:
        print("release: dry run, nothing written")
        return 0

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "policy_version": content_policy.POLICY_VERSION,
        "qa_version": QA_VERSION,
        "gs_extensions": list(GS_EXTENSIONS),
        "require_gs_qa": not args.allow_missing_gs_qa,
        "counts": {
            "manifest_total": total,
            "included": len(included),
            "excluded": len(excluded),
            "bytes": total_bytes,
        },
        "excluded_by_reason": dict(by_reason),
        "excluded": excluded,
        "items": included,
    }
    content_policy.write_json(args.output, payload)
    print(f"release: wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
