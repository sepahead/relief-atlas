"""Pending-work accounting for the unattended local run (run_forever.sh).

Termination is the whole point of the unattended driver, so this count has to
agree with the pipeline about what "done" means and what "will never finish"
means. It lives here, as an importable module with tests, rather than as a
heredoc inside run_forever.sh -- the heredoc version drifted twice:

  * it hardcoded `.ply` as the 3DGS container, so when the condensed export
    switched to .spz/.splat the count could never reach zero and the driver
    re-launched a full no-op pass every 60 seconds indefinitely;
  * it treated an item with no PNG as "not pending" (commented "the image phase
    has not produced it yet"), which is exactly backwards. An item whose image
    generation failed, or whose PNG the QA gate had just deleted, became
    invisible to the count. Measured on a fresh workspace: pending=14 while
    10,047 items had no image at all -- so the driver printed "all done" and
    exited having produced 9 assets out of 10,079.

The rules, stated once:

  * DONE when prompt_utils.mesh_complete() says so: GLB + a 3DGS container +
    metadata.json. Same predicate the mesh phase resumes on.
  * REFUSED when the content policy refuses the item under the same flags the
    run was launched with. 570 of the 10,079 rows are refused by default, so a
    count that ignores the policy can never reach zero.
  * QUARANTINED when the QA gate has given up (state/qa_quarantine.json). This
    is the only "will never progress" state, because it is the only one the
    pipeline itself declares terminal.
  * everything else is PENDING -- including items with no image yet, and items
    with a failed or absent QA verdict. That is work the next pass attempts.

Transient hard failures (state/local_*_failures.json) are deliberately NOT
consulted by classify(): an item stays in whatever bucket its artifacts imply,
because a recorded failure is usually a GPU watchdog kill that a rerun fixes.
They are surfaced by summarize() so a stuck run is attributable. A genuinely
stuck item therefore holds the count above zero, which is what the driver's
no-progress circuit breaker is for: stopping and saying so beats declaring
success.

Scoping flags mirror the generation phases, because the count has to be over the
same set the run covers -- `run_forever.sh --geographies germany` generating only
Germany while this counted all four manifests would never reach zero, and the
driver would report failure on a run that succeeded. Note that no scoping makes
`--phase images` or `--limit` reachable: neither can ever complete a mesh, so
run_forever.sh is for full runs only.

    python3 scripts/local/pending.py                    # human summary
    python3 scripts/local/pending.py --count            # just the number
    python3 scripts/local/pending.py --json             # full breakdown
    python3 scripts/local/pending.py --allow-restricted  # match the run's flags
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import content_policy  # noqa: E402
from prompt_utils import (  # noqa: E402
    MANIFESTS, OUTPUT_DIR, STATE_DIR, mesh_complete, select_manifest_items,
)
from pipeline_utils import item_certified, load_json  # noqa: E402

DONE = "done"
PENDING = "pending"
REFUSED = "refused"
QUARANTINED = "quarantined"
BUCKETS = (PENDING, DONE, REFUSED, QUARANTINED)


def load_state(name: str, state_dir=None):
    p = Path(state_dir or STATE_DIR) / name
    return load_json(p)


def item_dir(output_dir, item) -> Path:
    return Path(output_dir) / item["geography"] / item["category"] / item["id"]


def classify(items, output_dir, *, quarantine=(), allow_restricted=False,
             allow_blocked=False, only_policy=None, certified=False, ledgers=None) -> dict:
    """Bucket every item into pending / done / refused / quarantined.

    Returns {bucket: [item_id, ...]}. The only impurity is the filesystem probe
    behind mesh_complete(), so tests drive this with a temp directory and
    literal state rather than a 40GB workspace.

    Items are copied before screening: screen_items() stamps a "policy" key on
    whatever it is handed, and this function must not mutate a caller's
    manifest rows.
    """
    items = [dict(it) for it in items]
    items = content_policy.select_tier(items, only_policy)
    permitted, refused = content_policy.screen_items(
        items, allow_restricted, allow_blocked)

    quarantine = set(quarantine)
    buckets = {b: [] for b in BUCKETS}
    buckets[REFUSED] = [it["id"] for it in refused]
    for it in permitted:
        # DONE is tested before QUARANTINED on purpose: a complete deliverable
        # is complete regardless of what the ledger remembers about earlier
        # attempts, and reporting it as unreachable would be a lie.
        d = item_dir(output_dir, it)
        if (mesh_complete(d, it["id"])
                and (not certified or item_certified(d, it["id"], ledgers or {}))):
            buckets[DONE].append(it["id"])
        elif it["id"] in quarantine:
            buckets[QUARANTINED].append(it["id"])
        else:
            buckets[PENDING].append(it["id"])
    return buckets


def select_items(geographies=None, ids=None, limit=None, only_policy=None) -> list:
    """The manifest rows this count covers, scoped like the phases scope theirs."""
    return select_manifest_items(geographies, ids, limit, only_policy)


def report(output_dir=None, state_dir=None, geographies=None, ids=None,
           limit=None, **kw) -> dict:
    """Classify the selected manifest rows against the workspace on disk."""
    quarantine = load_state("qa_quarantine.json", state_dir)
    ledgers = {mode: load_state(f"qa_{mode}.json", state_dir)
               for mode in ("images", "meshes", "gaussians")}
    items = select_items(geographies, ids, limit, kw.get("only_policy"))
    return classify(items, output_dir or OUTPUT_DIR,
                    quarantine=quarantine, ledgers=ledgers, **kw)


def summarize(buckets, state_dir=None) -> str:
    total = sum(len(v) for v in buckets.values())
    parts = [f"{len(buckets[b])} {b}" for b in BUCKETS]
    line = f"{total} manifest items: " + " | ".join(parts)
    extra = []
    for name, label in (("local_image_failures.json", "image"),
                        ("local_mesh_failures.json", "mesh")):
        fails = load_state(name, state_dir)
        if fails:
            # Counted inside `pending`, not as a separate terminal state: a
            # watchdog kill is worth retrying. Surfaced here so a stuck run is
            # attributable rather than mysterious.
            extra.append(f"{len(fails)} recorded {label} failure(s)")
    if extra:
        line += "\n  " + "; ".join(extra) + " (still counted as pending)"
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="How much work the local pipeline still has to do.")
    ap.add_argument("--count", action="store_true",
                    help="print only the pending count (for shell drivers)")
    ap.add_argument("--json", action="store_true",
                    help="print the full per-bucket breakdown as JSON")
    ap.add_argument("--certified", action="store_true",
                    help="count done only after all three current artifact-bound QA passes")
    ap.add_argument("--require-complete", action="store_true",
                    help="exit nonzero if pending or quarantined items remain")
    ap.add_argument("--limit", type=int, default=None)
    # Mirrors qwen_imagegen / trellis_meshgen so a scoped run's driver counts the
    # same set the run covers.
    ap.add_argument("--geographies", nargs="*", default=None,
                    choices=list(MANIFESTS))
    ap.add_argument("--ids", nargs="*", default=None,
                    help="explicit item ids (overrides --geographies)")
    content_policy.add_policy_args(ap)
    # parse_known_args so run_forever.sh can forward run_all.py's whole argv
    # verbatim; selection and policy flags are shared, while generation settings
    # such as --texture-size have no effect on work accounting.
    args, _ignored = ap.parse_known_args(argv)

    buckets = report(geographies=args.geographies, ids=args.ids, limit=args.limit,
                     certified=args.certified,
                     allow_restricted=args.allow_restricted,
                     allow_blocked=args.allow_blocked,
                     only_policy=args.only_policy)
    if args.count:
        print(len(buckets[PENDING]))
    elif args.json:
        print(json.dumps({b: buckets[b] for b in BUCKETS}, indent=2))
    else:
        print(summarize(buckets))
    return 1 if args.require_complete and (buckets[PENDING] or buckets[QUARANTINED]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
