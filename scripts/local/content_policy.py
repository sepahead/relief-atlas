"""Content policy gate: dual-use screening for the relief-atlas pipeline.

The manifests describe civil-protection and humanitarian-response assets, but
some rows -- notably the `legacy_*` rows inherited from the project's earlier
Bundeswehr mesh sets -- describe military materiel: small arms, artillery,
armoured fighting vehicles, air-defence systems, combat aircraft and armed
personnel. The image and mesh QA gates only measure geometry and pixels, so
they are blind to that. This module screens the *text* of an item before any
pixels are generated.

Three verdicts:

    allowed     civil-protection, humanitarian or dual-use-neutral subject
    restricted  military platform or military personnel with no weapon
                function (skipped unless the caller passes allow_restricted)
    blocked     weapon, weapon platform or munition; never generated

Evaluation ORDER is the substance of the policy, not an implementation detail:

    1. Unambiguous weapon nouns are matched first, so no amount of
       humanitarian framing can launder an assault rifle.
    1b. A weapon match is discarded when it sits inside a benign compound
       phrase from WEAPON_EXEMPTIONS -- a wildfire "water bomber", a
       "torpedo-shaped" rescue ROV, a "landmine detector". The exemption is
       scoped to the matched span, so an item naming both a landmine detector
       and an assault rifle is still blocked on the rifle.
    2. The humanitarian allowlist is matched second, so demining, EOD/UXO
       clearance and personal protective equipment keep their mine- and
       ordnance-related vocabulary without being misread as weapons.
    3. Military platforms and combatants are matched last.

Two rules for editing the families. Keep the bare words `mine`, `ordnance`,
`explosive`, `demolition`, `bomb`, `IED` and `UXO` out of the weapon family --
those are exactly the words a mine-clearance or bomb-disposal item
legitimately uses, so only compound weapon forms belong there (`landmine`,
`explosive charge`, ...). And keep WEAPON_EXEMPTIONS to specific compound
phrases, never bare nouns: an exemption can only ever narrow the weapon
family, so a loose entry there silently permits real weapons. SELFTEST pins
both directions.

The gate reports and refuses; it never deletes existing artifacts.

Overrides are explicit and recorded. `--allow-restricted` permits the
restricted tier; `--allow-blocked` permits the blocked tier too (and implies
restricted, since blocked is the strictly more severe verdict). Anything
generated under an override is stamped `policy.override = true` in its
metadata and appended to state/content_policy_overrides.json, so an override
run can always be told apart from a default run and filtered out of a release.
`--only-policy` restricts a run to one tier, which is how you audit what the
gate is catching without generating the other 9,509 items.

Audit the manifests:
    python3 scripts/local/content_policy.py --report
List what a tier actually contains:
    python3 scripts/local/content_policy.py --list blocked
Check one string:
    python3 scripts/local/content_policy.py --check "Pearson mine roller"
Verify the policy against its regression table:
    python3 scripts/local/content_policy.py --selftest
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_DIR / "state"

# Bump when the families below change so stored verdicts stay interpretable.
POLICY_VERSION = 1

ALLOWED = "allowed"
RESTRICTED = "restricted"
BLOCKED = "blocked"

# --only-policy targets. "flagged" means anything the gate did not wave through.
POLICY_TIERS = (BLOCKED, RESTRICTED, "flagged")

# ---------------------------------------------------------------------------
# 1. Weapons, weapon platforms and munitions -> blocked.
#    Compound forms only for anything a demining/EOD item would also say.
# ---------------------------------------------------------------------------
WEAPON_PATTERNS = [
    # small arms
    r"\b(?:assault|battle|sniper|service)\s+rifles?\b",
    r"\brifles?\b",
    r"\bcarbines?\b",
    r"\b(?:sub)?machine\s?guns?\b",
    r"\bfirearms?\b|\bhandguns?\b|\brevolvers?\b|\bsidearms?\b|\bshotguns?\b",
    # "pistol grip" is a power-tool feature, not a weapon
    r"\bpistols?\b(?!\s+grip)",
    r"\bsnipers?\b|\bgun\s?ships?\b",
    r"\bPDW\b|\bSMG\b",
    # munitions
    r"\bgrenades?\b",
    r"\bwarheads?\b|\bcluster\s+munitions?\b",
    r"\bammunition\b|\bammo\b|\bshell\s+casings?\b|\bbandolier\b",
    r"\blandmines?\b|\banti-?personnel\s+mines?\b|\bnaval\s+mines?\b",
    r"\b(?:explosive|demolition|shaped)\s+charges?\b",
    # "torpedo-shaped" describes rescue ROV hulls; see WEAPON_EXEMPTIONS
    r"\btorpedo(?:es)?\b|\bdepth\s+charges?\b",
    # indirect fire and heavy weapons.  Bare "mortar" is masonry mortar in the
    # rubble rows so it needs a weaponised form; "water cannon" is a fire
    # appliance and is waived by WEAPON_EXEMPTIONS.
    r"\bartillery\b|\bhowitzers?\b|\bautocannons?\b",
    r"\bcannons?\b",
    r"\bmortars?\s+(?:system|tube|round|shell|bomb|carrier|team|launcher|pit)\b",
    r"\b\d+\s?mm\s+mortars?\b",
    r"\bmissiles?\b|\brocket\s+(?:launcher|artillery|pod)s?\b",
    r"\bMANPADS\b|\bsurface-to-air\b|\bair-to-air\b",
    r"\banti-?tank\b|\banti-?aircraft\b",
    r"\bflamethrowers?\b",
    r"\b(?:gun|weapon|cannon|missile)\s+turrets?\b",
    # bladed weapons
    r"\bbayonets?\b|\bcombat\s+knives?\b|\bcombat\s+knife\b",
    # armed platforms: an MBT/IFV/SPH/air-defence launcher is a weapon system
    r"\bmain\s+battle\s+tanks?\b|\bMBT\b",
    r"\binfantry\s+fighting\s+vehicles?\b|\bIFV\b",
    r"\bself-?propelled\s+howitzers?\b",
    r"\bair[- ]defen[cs]e\s+(?:system|launcher|battery|vehicle)s?\b|\bC-?RAM\b",
    r"\battack\s+helicopters?\b|\bfighter\s+jets?\b|\bcombat\s+aircraft\b",
    # every "bomber" in these manifests is a wildfire water bomber; waived below
    r"\bbombers?\b",
    # generic
    r"\bweapons?\b|\barmaments?\b",
    # named systems seen in the legacy Bundeswehr sets
    r"\bPanzerfaust\b|\bPzH\s?2000\b|\bLeopard\s?2\b|\bMarder\b",
    r"\bPuma\s+(?:IFV|infantry)\b|\bGepard\b|\bOzelot\b|\bMANTIS\b",
    r"\bIRIS-?T\b|\bStinger\b|\bPatriot\s+(?:launcher|missile|battery|system|PAC)\b",
    r"\bEurofighter\b|\bTornado\s+(?:IDS|ECR|jet|fighter|bomber)\b",
    r"\bTiger\s+attack\s+helicopter\b",
    r"\bG36\b|\bHK\s?416\b|\bMG\s?[35]\b|\bMP\s?7\b|\bAK-?47\b|\bM4A1\b|\bP8\s+pistol\b",
]

# ---------------------------------------------------------------------------
# 1b. Benign compound phrases that contain a weapon noun.  A weapon match is
#     waived only when it lies WITHIN one of these spans, so naming a landmine
#     detector cannot launder a rifle mentioned elsewhere in the same item.
#     Every entry must be a specific compound phrase, never a bare noun.
#     These come from real manifest rows: 72 wildfire water bombers, 44
#     torpedo-shaped rescue ROVs, 52 landmine detectors, and the
#     Bundespolizei water cannon trucks.
# ---------------------------------------------------------------------------
WEAPON_EXEMPTIONS = [
    # rescEU Canadair CL-415 wildfire aircraft
    r"\b(?:water|wildfire|fire|aerial)\s+bombers?\b",
    # underwater search-and-rescue ROV hull shape
    r"\btorpedo[-\s]shaped\b",
    # fire-service and public-order water projector, not a gun
    r"\bwater\s+cannons?\b",
    # mine action: detection, clearance and marking of existing hazards
    r"\blandmines?\s+(?:detect\w+|clearance|clearing|survey|marking|"
    r"awareness|risk|hazard)\b",
    r"\b(?:detector|detection|clearance|survey)\s+(?:for|of)\s+landmines?\b",
    r"\banti-?(?:tank|personnel)\s+mines?\s+(?:detect\w+|clearance|clearing|"
    r"survey|marking|roller|flail|plough|plow|awareness)\b",
    r"\banti-?(?:tank|personnel)\s+mine\s+(?:detect\w+|clearance|clearing)\b",
    # cooking appliance, not ordnance
    r"\brocket\s+stoves?\b",
]

# ---------------------------------------------------------------------------
# 2. Humanitarian allowlist -> allowed, and it neutralises the mine/ordnance
#    vocabulary that mine-clearance and bomb-disposal items legitimately use.
# ---------------------------------------------------------------------------
HUMANITARIAN_PATTERNS = [
    r"\bde-?mining\b|\bmine\s+clearance\b|\bmine[- ]clearing\b",
    r"\bmine\s+(?:roller|flail|plough|plow|detector|detection|rake|sweeper|"
    r"trailer|marking)\b",
    # "landmine" is one word, so the \bmine rules above cannot reach it
    r"\blandmines?\s+(?:detect\w+|clearance|clearing|survey|marking|"
    r"awareness|risk|hazard)\b",
    r"\bexplosive\s+ordnance\s+disposal\b|\bEOD\b|\bUXO\b",
    r"\bunexploded\s+ordnance\b|\bordnance\s+disposal\b",
    r"\bbomb\s+disposal\b|\bbomb\s+suit\b|\bblast\s+blanket\b",
    r"\bbody\s+armou?r\b|\bballistic\s+(?:vest|helmet|shield|plate)s?\b",
    r"\bprotective\s+(?:visor|apron)\b|\bmine[- ]resistant\b",
]

# ---------------------------------------------------------------------------
# 3. Unarmed military platforms and personnel -> restricted.
# ---------------------------------------------------------------------------
MILITARY_PATTERNS = [
    r"\barmou?red\s+personnel\s+carriers?\b|\bAPC\b",
    r"\barmou?red\s+(?:vehicle|car|truck|hull)s?\b",
    r"\bBoxer\s+(?:APC|GTK|armou?red)\b|\bFuchs\b|\bWiesel\b|\bFennek\b|\bDingo\b",
    r"\bBundeswehr\b|\bWehrmacht\b|\bLuftwaffe\b",
    r"\bmilitary\s+(?:vehicle|truck|transport|convoy|uniform|personnel|"
    r"materiel|equipment|aircraft|helmet|base|camp)s?\b",
]

COMBATANT_PATTERNS = [
    r"\bsoldiers?\b|\binfantry(?:man|men)?\b|\briflem[ae]n\b",
    r"\bparatroopers?\b|\bmachine\s+gunners?\b|\bcommandos?\b",
    r"\bspecial\s+forces\b|\bcombat\s+(?:uniform|fatigues|helmet)\b",
    r"\bflecktarn\b",
]

# Categories/ids carried over from the earlier military mesh sets.
LEGACY_MILITARY_RE = re.compile(
    r"\blegacy_(?:infantry|warzone|bundeswehr|military)", re.I)


def _compile(patterns):
    return re.compile("|".join(patterns), re.I)


WEAPON_RE = _compile(WEAPON_PATTERNS)
EXEMPTION_RE = _compile(WEAPON_EXEMPTIONS)

# Matched after the weapon family; first match wins.  See the module docstring
# on why the weapon family must precede the humanitarian allowlist.
FAMILIES = (
    (ALLOWED, "humanitarian", _compile(HUMANITARIAN_PATTERNS)),
    (RESTRICTED, "military_platform", _compile(MILITARY_PATTERNS)),
    (RESTRICTED, "combatant", _compile(COMBATANT_PATTERNS)),
)


def _weapon_match(text: str):
    """First weapon mention that is not inside a benign compound phrase.

    Scans every weapon hit rather than stopping at the first, so a waived
    phrase ("landmine detector") cannot suppress a real weapon named later in
    the same description. The exemption spans are only computed once a weapon
    hit exists, which is under 3% of the corpus.
    """
    exempt = None
    for m in WEAPON_RE.finditer(text):
        if exempt is None:
            exempt = [x.span() for x in EXEMPTION_RE.finditer(text)]
        if not any(s <= m.start() and m.end() <= e for s, e in exempt):
            return m
    return None


@dataclass(frozen=True)
class Verdict:
    classification: str
    category: str
    match: str

    def permitted(self, allow_restricted: bool,
                  allow_blocked: bool = False) -> bool:
        """Whether this item may be generated.

        allow_blocked implies allow_restricted: blocked is the strictly more
        severe verdict, so permitting an armed platform while still refusing an
        unarmed transport would be incoherent. Both default to False, so the
        gate is closed unless a caller opts out in so many words.
        """
        if self.classification == BLOCKED:
            return allow_blocked
        if self.classification == RESTRICTED:
            return allow_restricted or allow_blocked
        return True

    def as_dict(self) -> dict:
        # `override` is always present, even before screen_items decides it, so
        # every consumer can read policy["override"] without a KeyError guard.
        return {
            "classification": self.classification,
            "category": self.category,
            "match": self.match,
            "policy_version": POLICY_VERSION,
            "override": False,
        }

    def summary(self) -> str:
        return f"{self.classification} ({self.category}: {self.match})"


def classify_text(text: str) -> Verdict:
    """Classify a free-text subject description."""
    weapon = _weapon_match(text)
    if weapon:
        return Verdict(BLOCKED, "weapon", weapon.group(0).strip())
    for classification, category, pattern in FAMILIES:
        m = pattern.search(text)
        if m:
            return Verdict(classification, category, m.group(0).strip())
    return Verdict(ALLOWED, "unclassified", "")


def classify_item(item: dict) -> Verdict:
    """Classify a manifest row from its name, prompt, category and id."""
    verdict = classify_text(f"{item.get('name', '')} {item.get('prompt', '')}")
    # An explicit humanitarian match wins over the provenance heuristic: a
    # demining rig from the legacy set is still a demining rig.
    if verdict.classification == ALLOWED and verdict.category != "humanitarian":
        legacy = LEGACY_MILITARY_RE.search(
            f"{item.get('category', '')} {item.get('id', '')}")
        if legacy:
            return Verdict(RESTRICTED, "legacy_military_set", legacy.group(0))
    return verdict


def screen_items(items: list[dict], allow_restricted: bool,
                 allow_blocked: bool = False):
    """Split items into (permitted, rejected), tagging each with its verdict.

    Every item gains a "policy" key so downstream metadata can record the
    classification that let it through, including `override` — true when a
    non-allowed item was permitted anyway. That flag is the only durable trace
    on the asset itself that it came from an override run, so a release can
    filter on it without consulting run logs.
    """
    permitted, rejected = [], []
    for it in items:
        verdict = classify_item(it)
        ok = verdict.permitted(allow_restricted, allow_blocked)
        entry = verdict.as_dict()
        entry["override"] = bool(ok and verdict.classification != ALLOWED)
        it["policy"] = entry
        (permitted if ok else rejected).append(it)
    return permitted, rejected


def select_tier(items: list[dict], tier: str | None) -> list[dict]:
    """Narrow a run to one policy tier, for auditing what the gate catches.

    Tags every inspected item with its verdict so the caller can screen the
    survivors without reclassifying them.
    """
    if not tier:
        return items
    keep = []
    for it in items:
        verdict = classify_item(it)
        it["policy"] = verdict.as_dict()
        hit = (verdict.classification == tier
               or (tier == "flagged" and verdict.classification != ALLOWED))
        if hit:
            keep.append(it)
    return keep


# ---------------------------------------------------------------------------
# Shared CLI surface.
#
# Each phase runs as its own process, so the policy flags have to be declared
# four times over. They used to be, and they drifted: the image phase honoured
# --allow-restricted while the QA phase did not, so restricted items were
# generated and then quarantined as policy failures by the very next step.
# Declaring the flags once here, and forwarding them with policy_argv(), makes
# that particular bug unrepresentable.
# ---------------------------------------------------------------------------

def add_policy_args(ap):
    """Register the policy flags every phase must expose identically."""
    g = ap.add_argument_group("content policy")
    g.add_argument("--allow-restricted", action="store_true",
                   help="permit 'restricted' items (unarmed military "
                        "platforms and personnel)")
    g.add_argument("--allow-blocked", action="store_true",
                   help="permit 'blocked' items (weapons, munitions, armed "
                        "platforms); implies --allow-restricted. Generated "
                        "assets are stamped policy.override=true and logged "
                        "to state/content_policy_overrides.json")
    g.add_argument("--only-policy", choices=POLICY_TIERS, default=None,
                   metavar="TIER",
                   help=f"run only items in one tier {POLICY_TIERS} — for "
                        "auditing the gate rather than producing the dataset")
    return ap


def policy_argv(args) -> list[str]:
    """The policy flags as argv, so an orchestrator can forward them verbatim."""
    argv = []
    if getattr(args, "allow_restricted", False):
        argv.append("--allow-restricted")
    if getattr(args, "allow_blocked", False):
        argv.append("--allow-blocked")
    tier = getattr(args, "only_policy", None)
    if tier:
        argv += ["--only-policy", tier]
    return argv


def screen_from_args(items: list[dict], args, phase: str):
    """Apply --only-policy, then the gate. Logs refusals.

    Returns (permitted, rejected, overrides). `overrides` is the subset of
    `permitted` that only got through because of a flag — reported so a caller
    can say out loud that the run is not a default one.

    Deliberately does NOT write the override ledger: being permitted is not the
    same as being generated, and callers screen every item before filtering out
    the ones that are already done, quarantined, or about to fail. Generators
    call record_overrides() once they have actually written something.
    """
    items = select_tier(items, getattr(args, "only_policy", None))
    permitted, rejected = screen_items(
        items,
        getattr(args, "allow_restricted", False),
        getattr(args, "allow_blocked", False),
    )
    if rejected:
        record_skipped(rejected, phase)
    overrides = [it for it in permitted if it["policy"]["override"]]
    return permitted, rejected, overrides


def describe_screen(permitted, rejected, overrides) -> str:
    """One-line, honest summary of what the gate just did."""
    parts = [f"content policy: {len(permitted)} permitted"]
    if rejected:
        n_blocked = sum(1 for r in rejected
                        if r["policy"]["classification"] == BLOCKED)
        parts.append(f"{len(rejected)} refused ({n_blocked} blocked)")
    if overrides:
        n_blocked = sum(1 for r in overrides
                        if r["policy"]["classification"] == BLOCKED)
        parts.append(f"OVERRIDE: {len(overrides)} non-allowed items permitted "
                     f"({n_blocked} blocked)")
    return ", ".join(parts)


def write_json(path: Path, data):
    """Atomic write, mirroring the convention used across the pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def record_skipped(rejected: list[dict], phase: str):
    """Log policy refusals to state/content_policy_skipped.json."""
    return _append_log("content_policy_skipped.json", rejected, phase)


def record_overrides(overrides: list[dict], phase: str):
    """Log override-GENERATED items to state/content_policy_overrides.json.

    The ledger is the audit trail for a non-default run: it names every asset
    that exists only because someone passed --allow-restricted or
    --allow-blocked, so those files can be found again and pulled from a release
    without re-deriving the policy.

    Call this with items whose artifacts were actually written, not with
    everything the gate permitted — otherwise the ledger claims assets that were
    never produced.
    """
    return _append_log("content_policy_overrides.json", overrides, phase)


def _append_log(filename: str, items: list[dict], phase: str):
    path = STATE_DIR / filename
    log = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    for it in items:
        log[it["id"]] = {**it["policy"], "phase": phase, "at": now}
    write_json(path, log)
    return path


def report():
    """Audit every manifest and write state/content_policy_report.json."""
    from prompt_utils import MANIFESTS, load_manifest

    counts = Counter()
    by_category = Counter()
    flagged = {}
    for geo in MANIFESTS:
        for it in load_manifest(geo):
            verdict = classify_item(it)
            counts[verdict.classification] += 1
            if verdict.classification != ALLOWED:
                by_category[verdict.category] += 1
                flagged[it["id"]] = {
                    "geography": geo,
                    "name": it.get("name", ""),
                    **verdict.as_dict(),
                }

    total = sum(counts.values())
    path = STATE_DIR / "content_policy_report.json"
    write_json(path, {
        "policy_version": POLICY_VERSION,
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_items": total,
        "counts": dict(counts),
        "by_category": dict(by_category),
        "flagged": flagged,
    })

    print(f"{total} items screened (policy v{POLICY_VERSION})")
    for name in (ALLOWED, RESTRICTED, BLOCKED):
        print(f"  {name:10s} {counts[name]:6d}")
    if by_category:
        print("flagged by category:")
        for cat, n in by_category.most_common():
            print(f"  {cat:22s} {n:6d}")
    print(f"report: {path}")


# ---------------------------------------------------------------------------
# Regression table.  Every entry is a phrase drawn from, or deliberately
# adversarial to, the real manifests.  The false-positive rows exist because
# each one WAS misclassified at some point; the false-negative rows exist so a
# careless widening of WEAPON_EXEMPTIONS fails loudly instead of silently
# permitting materiel.
# ---------------------------------------------------------------------------
SELFTEST = [
    # -- benign subjects that contain weapon vocabulary -------------------
    (ALLOWED, "rescEU Canadair CL-415 wildfire response water bomber aircraft"),
    (ALLOWED, "underwater ROV for search and rescue, torpedo-shaped with "
              "thrusters and camera"),
    (ALLOWED, "Schiebel AN19/2 handheld metal detector for landmine detection"),
    (ALLOWED, "Bundespolizei water cannon truck for public order and flood response"),
    (ALLOWED, "Pearson Engineering mine roller, triggers pressure-activated "
              "mines safely"),
    (ALLOWED, "EOD disposal robot for unexploded ordnance, safe disposal of UXO"),
    (ALLOWED, "anti-tank mine clearance flail vehicle for humanitarian demining"),
    (ALLOWED, "anti-personnel mine detection robot with ground penetrating radar"),
    (ALLOWED, "portable rocket stove for field kitchen cooking"),
    (ALLOWED, "collapsed building rubble pile, shattered brick and mortar chunks"),
    (ALLOWED, "water tank, 5000 litre potable water storage bladder"),
    (ALLOWED, "cordless demolition drill with pistol grip handle"),
    (ALLOWED, "maritime search and rescue vessel, coastal lifeboat"),
    (ALLOWED, "psychosocial support tent, safe space for counseling"),
    (ALLOWED, "ballistic vest and protective visor for aid workers"),
    # -- an exemption must not launder a weapon named elsewhere -----------
    (BLOCKED, "handheld landmine detector carried by a soldier with a G36 "
              "assault rifle"),
    (BLOCKED, "water bomber aircraft escorted by a Tornado IDS fighter"),
    (BLOCKED, "torpedo-shaped ROV stored beside a rack of ammunition"),
    # -- unambiguous materiel --------------------------------------------
    (BLOCKED, "Heckler & Koch G36 assault rifle, standard issue service rifle"),
    (BLOCKED, "MG5 general purpose machine gun on bipod"),
    (BLOCKED, "MP7 personal defence weapon"),
    (BLOCKED, "P8 pistol sidearm in holster"),
    (BLOCKED, "bayonet and combat knife set"),
    (BLOCKED, "Panzerfaust 3 anti-tank recoilless launcher"),
    (BLOCKED, "PzH 2000 self-propelled howitzer artillery"),
    (BLOCKED, "Patriot air defence missile launcher"),
    (BLOCKED, "Leopard 2A7 main battle tank"),
    (BLOCKED, "Puma IFV infantry fighting vehicle"),
    (BLOCKED, "Eurofighter multirole combat aircraft"),
    (BLOCKED, "Tiger attack helicopter"),
    (BLOCKED, "120mm mortar tube and mortar rounds"),
    # -- dual-use, disclosed but not generated by default ----------------
    (RESTRICTED, "Bundeswehr soldier in combat uniform, full body character"),
    (RESTRICTED, "Luftwaffe ground crew with safety vest and marshalling wands"),
    (RESTRICTED, "Boxer GTK armoured personnel carrier, unarmed transport variant"),
    (RESTRICTED, "military truck for disaster logistics convoy"),
]


def selftest() -> int:
    """Check the policy against SELFTEST. Returns the number of failures."""
    failures = 0
    for expected, text in SELFTEST:
        got = classify_text(text)
        if got.classification != expected:
            failures += 1
            print(f"FAIL expected {expected:10s} got {got.summary()}")
            print(f"       {text}")
    n = len(SELFTEST)
    print(f"selftest: {n - failures}/{n} passed (policy v{POLICY_VERSION})")
    return failures


def list_tier(tier: str, show_prompt: bool = False):
    """Print every manifest row in one tier, so the gate can be eyeballed.

    This is the cheap way to audit a classification: read what got caught
    before deciding whether the pattern families are right.
    """
    from prompt_utils import MANIFESTS, load_manifest

    rows = []
    for geo in MANIFESTS:
        rows.extend(select_tier([dict(it) for it in load_manifest(geo)], tier))
    for it in rows:
        p = it["policy"]
        print(f"{p['classification']:10s} {p['category']:20s} "
              f"{it['id']:24s} {it.get('name', '')[:56]}")
        print(f"{'':32s} matched: {p['match']!r}")
        if show_prompt:
            print(f"{'':32s} prompt: {it.get('prompt', '')[:300]}")
    print(f"\n{len(rows)} item(s) in tier {tier!r} (policy v{POLICY_VERSION})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="audit all manifests and write the policy report")
    ap.add_argument("--list", choices=POLICY_TIERS, default=None, metavar="TIER",
                    help=f"list every manifest row in one tier {POLICY_TIERS}")
    ap.add_argument("--prompts", action="store_true",
                    help="with --list, also print each item's prompt")
    ap.add_argument("--check", nargs="+", default=None,
                    help="classify one or more literal strings")
    ap.add_argument("--selftest", action="store_true",
                    help="check the policy against its regression table")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(1 if selftest() else 0)
    if args.list:
        list_tier(args.list, show_prompt=args.prompts)
        return
    if args.check:
        for text in args.check:
            print(f"{classify_text(text).summary()}  <- {text!r}")
    if args.report or not args.check:
        report()


if __name__ == "__main__":
    main()
