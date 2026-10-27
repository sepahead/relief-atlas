"""Progress report for the local generation run.

Counts items with completed images and completed meshes (GLB + 3DGS) across all
manifest geographies, then reports what the pipeline's gates are doing: QA
verdicts, quarantined items, content-policy refusals, any assets that only exist
because someone passed an override flag, and — the more alarming inverse — any
refused subject with artifacts on disk that NO override ever authorised, which is
what pre-policy output looks like.

The 3DGS column counts an item once any container the mesh phase can emit is
present -- .spz (preferred), .splat (universal fallback) or .ply (only with
condensed=False) -- via the shared prompt_utils.GS_EXTENSIONS. When the spz
package is installed the mesh phase writes .spz AND .splat per item; counting
one extension alone reports zero forever.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_policy  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR, STATE_DIR, has_gaussians, load_manifest,
    verdict_is_current,
)


def load(name: str):
    p = STATE_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def report_progress():
    """Per-geography image/mesh/3DGS completion."""
    total_items = total_img = total_mesh = total_gs = 0
    for geo in MANIFESTS:
        items = load_manifest(geo)
        n_img = n_mesh = n_gs = 0
        for it in items:
            d = OUTPUT_DIR / geo / it["category"] / it["id"]
            if (d / f"{it['id']}.png").exists():
                n_img += 1
            glb = d / f"{it['id']}.glb"
            if glb.exists() and glb.stat().st_size > 1024:
                n_mesh += 1
            if has_gaussians(d, it["id"]):
                n_gs += 1
        total_items += len(items)
        total_img += n_img
        total_mesh += n_mesh
        total_gs += n_gs
        print(f"{geo:10s} {len(items):6d} items | images {n_img:6d} | "
              f"meshes {n_mesh:6d} | 3DGS {n_gs:6d}")

    print(f"{'TOTAL':10s} {total_items:6d} items | images {total_img:6d} | "
          f"meshes {total_mesh:6d} | 3DGS {total_gs:6d}")
    return total_items


def report_qa():
    """QA verdicts and the quarantine, which otherwise stay invisible.

    A large quarantine is the signal that the generator is looping on items it
    cannot satisfy, so it belongs in the progress report rather than only in a
    JSON file nobody opens.
    """
    for mode, name in (("image", "qa_images.json"), ("mesh", "qa_meshes.json"),
                       ("gaussians", "qa_gaussians.json")):
        verdicts = load(name)
        if not verdicts:
            continue
        passed = sum(1 for v in verdicts.values() if v.get("pass"))
        stale = sum(not verdict_is_current(v) for v in verdicts.values())
        print(f"{mode} QA: {len(verdicts):6d} verdicts | pass {passed:6d} | "
              f"fail {len(verdicts) - passed:6d} | stale {stale:6d}")

    quarantine = load("qa_quarantine.json")
    if quarantine:
        print(f"quarantined: {len(quarantine)} item(s) QA gave up on "
              f"(state/qa_quarantine.json)")

    retries = load("qa_retries.json")
    if retries:
        active = sum(1 for n in retries.values() if n)
        print(f"retry seeds bumped: {active} item(s)")

    for name, label in (("local_image_failures.json", "image"),
                        ("local_mesh_failures.json", "mesh")):
        fails = load(name)
        if fails:
            print(f"{label} failures: {len(fails)} ({name})")


def report_policy():
    """Content-policy standing, including override provenance.

    Overrides are reported loudly and unconditionally: an override run produces
    assets that a default run would not, and that fact has to survive until
    someone decides what ships.
    """
    rep = load("content_policy_report.json")
    if rep:
        counts = rep.get("counts", {})
        print(f"content policy v{rep.get('policy_version', '?')}: "
              + " | ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    else:
        print("content policy: no report yet "
              "(run: python3 scripts/local/content_policy.py --report)")

    skipped = load("content_policy_skipped.json")
    if skipped:
        print(f"policy refusals logged: {len(skipped)} item(s)")

    overrides = load("content_policy_overrides.json")
    if overrides:
        by_class = {}
        for entry in overrides.values():
            c = entry.get("classification", "?")
            by_class[c] = by_class.get(c, 0) + 1
        detail = ", ".join(f"{n} {c}" for c, n in sorted(by_class.items()))
        print(f"** OVERRIDE: {len(overrides)} item(s) generated outside the "
              f"default policy ({detail})")
        print("   these are not part of a default run; see "
              "state/content_policy_overrides.json")


def report_output_policy():
    """Re-screen what is actually ON DISK, not just what the gate permitted.

    The policy reports and refuses but never deletes existing artifacts, which
    is the right call — and it means anything produced BEFORE the gate existed
    sits in outputs_relief unmentioned by every counter. Three restricted
    legacy_* items were found exactly this way.

    An override run is accounted for by its ledger. Artifacts that the policy
    refuses and that NO ledger entry authorises are pre-policy output: nobody
    ever passed a flag for them, so they must not ride along into a release
    unnoticed.
    """
    if not OUTPUT_DIR.exists():
        return
    overrides = load("content_policy_overrides.json")
    unaccounted = []
    for geo in MANIFESTS:
        for it in load_manifest(geo):
            d = OUTPUT_DIR / geo / it["category"] / it["id"]
            if not d.is_dir():
                continue
            artifacts = [p for p in d.iterdir() if p.is_file()]
            if not artifacts:
                continue
            verdict = content_policy.classify_item(it)
            if verdict.classification == content_policy.ALLOWED:
                continue
            if it["id"] in overrides:
                continue  # an override run owns this one; report_policy says so
            unaccounted.append((verdict, it["id"], len(artifacts)))

    if not unaccounted:
        return
    n_blocked = sum(1 for v, _, _ in unaccounted
                    if v.classification == content_policy.BLOCKED)
    print(f"** UNACCOUNTED: {len(unaccounted)} item(s) have artifacts on disk "
          f"that the policy refuses and no override authorised "
          f"({n_blocked} blocked) — pre-policy output, review before release:")
    for verdict, item_id, n_files in sorted(unaccounted, key=lambda r: r[1]):
        print(f"   {verdict.classification:10s} {verdict.category:20s} "
              f"{item_id:32s} {n_files} file(s)")


def report_stale():
    if not OUTPUT_DIR.exists():
        return
    stale = sorted(OUTPUT_DIR.glob("*/*/*/.*.tmp.*"))
    if stale:
        print(f"warning: {len(stale)} stale transient export(s), "
              f"e.g. {stale[0].name} — safe to delete")
    # Finder droppings are not dataset artifacts and would otherwise be uploaded
    # alongside the meshes.
    junk = sorted(OUTPUT_DIR.rglob(".DS_Store"))
    if junk:
        print(f"warning: {len(junk)} .DS_Store file(s) under {OUTPUT_DIR.name}/ "
              f"— safe to delete")


def main():
    report_progress()
    print()
    report_qa()
    print()
    report_policy()
    report_output_policy()
    report_stale()


if __name__ == "__main__":
    main()
