"""Manifest loading and prompt/name sanitation for the local pipeline."""

import hashlib
import json
import os
import re
from pathlib import Path

from generation_prompts import is_optimized_prompt, normalize_generation_prompt
from pipeline_utils import (
    GS_EXTENSIONS, OUTPUT_DIR as OUTPUT_DIR, PROJECT_DIR, QA_VERSION, STATE_DIR as STATE_DIR,
)

MANIFEST_DIR = Path(os.environ.get("RELIEF_MANIFEST_DIR", PROJECT_DIR / "manifests")).resolve()
OUTPUT_DIR_NAME = "outputs_relief"

# 3DGS containers the mesh phase can emit, most-preferred first. Every place
# that counts, completes or cleans a 3DGS output must read this list. It lives
# here, in the one dependency-free module every script already imports, because
# the alternative has already cost us a bug: run_forever.sh kept a private copy
# that said `.ply`, so when the condensed export switched to .spz/.splat its
# "meshes pending" count could never reach zero and the unattended multi-day
# driver looped forever without producing anything.

# QA contract version. Bump whenever a gate's thresholds or measurements change
# in a way that makes an older verdict uninterpretable, so stored verdicts are
# re-judged instead of being trusted forever.
#
# It lives here for the same reason GS_EXTENSIONS does: qa.py imports numpy and
# PIL at module scope, so anything that only needs to reason ABOUT verdicts
# (release_manifest.py, tooling, tests on a bare interpreter) cannot import the
# constant from there.
#
# Version 2 is the first version to be recorded at all, which is the point. At
# v1 (unstamped) state/qa_images.json held 57 failures measuring fg_fraction=1.0
# from the pre-BiRefNet matting era, indistinguishable from current verdicts
# except by timestamp -- and those obsolete failures had already charged retry
# budget against items, permanently perturbing the "deterministic" per-item seed.
# v3 binds passes to artifact hashes and records whether NSFW screening ran.
# v5 rejects localized frame clipping and measures mesh coverage independently
# of face ordering and depth-preview shading.
# v6 reads SPZ colours as SH DC, matching external viewers, and prevents
# unverified legacy local SPZ encodings from being recertified automatically.


def verdict_is_current(verdict) -> bool:
    """Whether a stored QA verdict was produced by the current QA contract.

    Unstamped (pre-v2) verdicts are NOT current: they were measured by code that
    no longer exists, so trusting them would admit assets no live gate ever saw.
    """
    return isinstance(verdict, dict) and verdict.get("qa_version") == QA_VERSION


def current_verdicts(verdicts: dict) -> dict:
    """Drop verdicts from an older QA contract, so callers re-judge them."""
    return {k: v for k, v in (verdicts or {}).items() if verdict_is_current(v)}


def has_gaussians(item_dir, item_id: str) -> bool:
    """Whether any recognised 3DGS container exists for an item."""
    d = Path(item_dir)
    return any((d / f"{item_id}.{ext}").is_file()
               and (d / f"{item_id}.{ext}").stat().st_size > 0
               for ext in GS_EXTENSIONS)


def mesh_complete(item_dir, item_id: str, min_glb_bytes: int = 1024) -> bool:
    """Whether an item has a full mesh deliverable: GLB + 3DGS + metadata.

    Shared by the mesh phase's resume check, the progress report and the
    unattended driver, so "done" means the same thing to all three.
    """
    d = Path(item_dir)
    glb = d / f"{item_id}.glb"
    return (
        glb.exists() and glb.stat().st_size > min_glb_bytes
        and has_gaussians(d, item_id)
        and (d / "metadata.json").is_file()
    )

MANIFESTS = {
    "germany": "relief_manifest_germany.json",
    "eu": "relief_manifest_eu.json",
    "ukraine": "relief_manifest_ukraine.json",
    "general": "relief_manifest_general.json",
}


def stable_hash(text: str) -> int:
    """Deterministic 32-bit hash so seeds/framings survive restarts."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def item_seed(item_id: str) -> int:
    """Per-item generator seed derived from the id (stable across reruns)."""
    return stable_hash(f"relief-atlas/{item_id}") % (2**31 - 1)


# Camera framings rotated per item: identical manifest subjects (the template
# splices the same subject into many rows) still render as visibly different
# assets, and a slight elevation shows top surfaces to the image-to-3D model.
FRAMINGS = [
    "a slightly elevated three-quarter front-left view",
    "a slightly elevated three-quarter front-right view",
    "a straight-on eye-level left side view",
    "a straight-on eye-level right side view",
    "a three-quarter rear-left view",
    "a three-quarter rear-right view",
]

ROTORCRAFT_RE = re.compile(
    r"\b(quadcopter|hexacopter|octocopter|multirotor|drone|uav|vtol)\b", re.I)
TRANSLUCENT_PROP_RE = re.compile(
    r"translucent(?: polycarbonate)? propeller blades?", re.I)
ROTOR_CLAUSE = ("All propellers are fully stationary and parked, each rotor "
                "showing crisp distinct rigid blades with sharp silhouettes, "
                "no motion blur and no spinning-disc effect.")

# Manifest templates splice one shared subject sentence into dozens of rows
# (e.g. 52 items are all "collapsed building rubble pile"). After cleaning,
# those rows would produce byte-identical prompts that differ only by camera
# framing, collapsing dataset variety. These per-category variation clauses
# give every duplicate row a deterministic, visually distinct appearance that
# stays plausible for its asset class.
VARIATION_GROUPS = [
    (re.compile(r"rubble|debris|demolition", re.I), [
        "This particular pile is dominated by large broken concrete slabs with rusted rebar jutting out at angles.",
        "This particular pile consists mainly of shattered red brick and mortar chunks mixed with crushed concrete.",
        "This particular pile mixes concrete chunks with splintered wooden beams and twisted corrugated metal sheets.",
        "This particular pile is mostly fine pulverized concrete gravel with a few large tilted slab fragments on top.",
        "This particular pile is pale freshly-broken concrete lightly dusted with gray powder.",
        "This particular pile is dark weathered debris streaked with rain stains and old soot.",
    ]),
    (re.compile(r"tent|shelter|canopy", re.I), [
        "This particular unit is a dome-style tent with curved pole arches.",
        "This particular unit is an A-frame ridge tent with straight sloped walls.",
        "This particular unit is a rectangular cabin-style tent with near-vertical walls and a peaked roof.",
        "This particular unit is a tunnel-style tent with a rounded half-cylinder profile.",
        "This particular unit has an attached front vestibule awning over the entrance.",
    ]),
    (re.compile(r"flood barrier|gabion|mitigation barrier|barrier wall", re.I), [
        "This particular section is assembled from stacked aluminum panels with visible interlocking bolts.",
        "This particular section combines metal frames with tan sandbags layered along the base.",
        "This particular section uses dark rubberized modular blocks in a staggered brick pattern.",
        "This particular section is a taller two-tier configuration with cross bracing on the back side.",
    ]),
    (re.compile(r"pallet|cache|kit\b|supplies|blanket|hygiene|bedding", re.I), [
        "This particular load is stacked in neat cardboard boxes with printed labels, shrink-wrapped.",
        "This particular load is arranged in open plastic crates showing contents, strapped to the pallet.",
        "This particular load is wrapped in blue tarpaulin with tension straps.",
        "This particular load is stacked loose with rolled goods on top of boxed items.",
    ]),
    (re.compile(r"generator|pump|purification|x-ray|concentrator|terminal|repeater|"
                r"radar|laboratory|decontamination|siren", re.I), [
        "This particular unit is housed in a rugged yellow polymer case with black corner bumpers.",
        "This particular unit is mounted in an open steel frame with exposed cabling on one side.",
        "This particular unit is a compact suitcase-style enclosure with recessed handles and latches.",
        "This particular unit sits on a wheeled trolley chassis with a telescoping pull handle.",
        "This particular unit has an attached control panel with an array of indicator lights and dials.",
    ]),
    (re.compile(r"truck|van|ambulance|vehicle|carrier|transporter|bus|tanker|"
                r"bulldozer|excavator|loader|tractor|crane|robot", re.I), [
        "This particular vehicle has finely textured body panels with clearly visible seams and recessed fasteners.",
        "This particular vehicle has contrasting protective trim around exposed panel edges.",
        "This particular vehicle carries a roof-mounted light bar and antenna mast.",
        "This particular vehicle has a reinforced bull bar front bumper and auxiliary driving lamps.",
        "This particular vehicle has rear-mounted storage racks loaded with strapped equipment cases.",
    ]),
    (re.compile(r"drone|quadcopter|hexacopter|octocopter|multirotor|uav|aircraft|plane|helicopter|vtol", re.I), [
        "This particular aircraft has narrow reflective high-visibility strips along the airframe edges.",
        "This particular aircraft has contrasting safety markings around access panels.",
        "This particular aircraft has clearly defined panel seams and recessed fastening points.",
        "This particular aircraft has dark protective edge trim and crisp contrasting identification bands.",
    ]),
    (re.compile(r"bridge", re.I), [
        "This particular bridge section has an open truss lattice deck surface.",
        "This particular bridge section has a solid non-slip plate deck with raised edges.",
        "This particular bridge section includes fold-down support legs at both ends.",
    ]),
    (re.compile(r"container|housing unit|mobile clinic|field hospital|command|hub", re.I), [
        "This particular unit is a standard ISO container shape with corrugated steel walls.",
        "This particular unit has smooth flat sandwich-panel walls and a slightly wider footprint than a shipping container.",
        "This particular unit has an extendable slide-out section doubling its interior width.",
        "This particular unit has a rooftop solar panel array and cable ducts along one wall.",
    ]),
]
FALLBACK_VARIATIONS = [
    "This particular unit has a fine matte surface texture.",
    "This particular unit has a subtle satin surface texture.",
    "This particular unit has a lightly stippled surface texture.",
    "This particular unit has a smooth low-sheen surface texture.",
]

STUDIO_SPLIT = "The object stands alone"


def _studio_tail(framing: str, transparent=False) -> str:
    """Controlled studio framing that replaces the manifest boilerplate.

    The original boilerplate demanded a pure white seamless backdrop under
    softbox lighting 'with gentle contact shadows'. That floor + shadow
    gradient survived matting and TRELLIS.2 extruded it into white slab
    geometry. We instead demand a flat uniform light neutral gray backdrop
    with no floor plane at all: gray bounce light stays hue-neutral on the
    subject (saturated backdrops tinted it), matches the neutral-studio
    distribution TRELLIS.2 was trained on, and BiRefNet mattes it cleanly.
    """
    if transparent:
        return (
            f"The object stands alone, perfectly centered, fully visible and uncropped, "
            f"photographed from {framing}. Leave at least a 10% empty margin on every side, "
            f"including all propeller tips, antennas and protruding parts. "
            f"The image has alpha channel and the background "
            f"is transparent. No floor plane, horizon line, cast shadow or backdrop. "
            f"Even diffuse studio lighting, crisp edges, deep focus keeping the entire object "
            f"sharp. Professional product reference photograph for 3D model generation. "
            f"Medium-format camera, 85mm lens. No watermark, no extra objects, no text overlays."
        )
    return (
        f"The object stands alone, perfectly centered, fully visible and "
        f"uncropped, photographed from {framing}, isolated on a completely "
        f"flat uniform light neutral gray seamless background that fills the "
        f"entire frame edge to edge, with no horizon line and no floor "
        f"plane. Even diffuse shadowless studio lighting, crisp edges, deep "
        f"focus keeping the entire object sharp. Professional product "
        f"reference photograph for 3D model generation. Medium-format "
        f"camera, 85mm lens. No watermark, no logos, no extra objects, no "
        f"text overlays."
    )


def clean_name(name: str) -> str:
    """Strip template variant text leaked into names: 'Base Name (It carries...)'."""
    name = re.split(r"\s*\(", name, maxsplit=1)[0]
    return name.strip().rstrip(",.")


VEHICLE_RE = re.compile(
    r"\b(truck|trucks|vehicle|van|chassis|trailer|bus|car|cars|ambulance|transporter|"
    r"jeep|apc|ifv|tank|unimog|kat|sprinter|crane|excavator|bulldozer|loader|tractor|"
    r"motorcycle|boat|ship|helicopter|aircraft|plane|drone|uav|rover|cart|wagon|"
    r"forklift|engine|ladder|bus)\b",
    re.I,
)

VARIANT_BLOCKS = [
    "It is equipped for flood response with water pump attachments bolted to the "
    "side panels, extended mud guards, an elevated air intake snorkel on the cab, "
    "and a sandbag rack welded to the roof.",
    "It carries earthquake rescue gear: hydraulic rescue tools (jaws of life) "
    "mounted visibly on the exterior, structural shoring timber strapped to the "
    "frame, and a debris hook on the rear.",
    "It carries a standard multipurpose disaster relief loadout with basic "
    "equipment, clearly visible organization markings, and a utility rack loaded "
    "with mixed supplies.",
    "It is equipped for Oder river flood defense with a sandbag loader attachment "
    "on the front and a reinforced bumper guard.",
    "It is configured for winter operations with tire chains wrapped around the "
    "wheels, a heated equipment enclosure on top, antifreeze container markings "
    "on the side, and cold-weather insulation wrap on exposed pipes.",
    "It is configured for storm response with wind-rated anchor points visible "
    "on the corners, a folded rain cover tarp on top, reinforced structural "
    "bracing added to the frame, and a drainage pump attached at the base.",
    "It was deployed during the Ahrweiler 2021 flood, with waterline staining "
    "visible on the lower body and emergency flashers still active.",
    "It has just returned from a disaster site, with fresh mud splatter on the "
    "lower body and wet surfaces reflecting studio light, emergency equipment "
    "still deployed.",
]

SURFACE_RE = re.compile(r"The surface (?:shows|is) ([^.:]+): ([^.]*)\.")

# Variation clauses that describe finish/wear. A manifest prompt usually
# already carries its own "The surface is/shows ..." sentence, so appending
# one of these on top produced self-contradicting prompts such as "pristine
# factory-new: no scratches" followed by "heavy weathering: rain stains,
# scuffed edges". Image models resolve that by splitting the difference, which is
# neither state. Detecting the overlap lets the per-item clause win outright.
WEAR_CLAUSE_RE = re.compile(
    r"\b(?:factory finish|recently-repainted|pristine|weather(?:ed|ing)?|"
    r"wear|scratch(?:es|ed)?|scuff(?:ed|s)?|faded|sun-bleached|oxidi[sz]ed|"
    r"rain stain|mud splatter|chipped paint|grime|dusted|dust settled|"
    r"touch-up|freshly-broken|soot|clean factory)\b", re.I)

NEUTRAL_SURFACE = {
    "factory-new": "The surface is pristine factory-new: clean intact finish, no scratches or damage.",
    "post-deployment": "The surface shows heavy post-deployment wear: mud and grime caked in "
    "crevices, chipped paint, minor dents, and stains.",
    "field-used": "The surface shows field-used operational wear: dirt and dust on lower "
    "surfaces, scuff marks on handles and edges, and faded decals from sun exposure.",
    "well-maintained": "The surface is well-maintained: recently cleaned but showing honest "
    "service wear, touch-up paint on scratches, and minor rust in recesses.",
}


def _neutral_surface(m: re.Match) -> str:
    cond = m.group(1).lower()
    for key, repl in NEUTRAL_SURFACE.items():
        if key in cond:
            return repl
    return ""


TEMPLATE_START_RE = re.compile(r"\.\s+It (?:is|carries|has|was) ")

PART_WORDS_RE = re.compile(
    r"\b(tires?|tracks?|wheels?|wheel wells|body panels?|engine bay|undercarriage|"
    r"bumpers?|cab|snorkel|windshield|exhaust|suspension|engine|bilge|deck|"
    r"hood|bonnet|grille|axles?|mud guards?|flashers|wading|studded|tread)\b",
    re.I,
)


def _variation_choices(subject: str) -> list[str]:
    # Match the asset noun, rather than incidental cargo or operating context.
    # A cargo drone delivering supplies must not receive a pallet variation.
    asset = subject.split(",", 1)[0]
    for pattern, clauses in VARIATION_GROUPS:
        if pattern.search(asset):
            return clauses
    return FALLBACK_VARIATIONS


def _variation_clause(subject: str, item_id: str) -> str:
    clauses = _variation_choices(subject)
    index = _subject_ordinals().get(item_id, stable_hash(f"relief-atlas/var/{item_id}"))
    return clauses[index % len(clauses)]


_SUBJECT_COUNTS: dict[str, int] | None = None
_SUBJECT_ORDINALS: dict[str, int] | None = None


def _subject_key(prompt: str) -> str:
    p = re.sub(r"\)\s*The surface is", ". The surface is", prompt.strip())
    p = TRANSLUCENT_PROP_RE.sub("solid matte carbon-fiber propeller blades", p)
    return TEMPLATE_START_RE.split(p, maxsplit=1)[0]


def _subject_ordinals() -> dict[str, int]:
    """Allocate combinations without hash collisions within a shared subject."""
    global _SUBJECT_ORDINALS
    if _SUBJECT_ORDINALS is None:
        groups: dict[str, list[str]] = {}
        for it in all_items():
            subject = _subject_key(it["prompt"])
            groups.setdefault(subject, []).append(it["id"])
        _SUBJECT_ORDINALS = {item_id: index for ids in groups.values()
                             for index, item_id in enumerate(sorted(ids))}
    return _SUBJECT_ORDINALS


def _subject_counts() -> dict[str, int]:
    """How many manifest rows share each raw subject sentence (lazy, cached)."""
    global _SUBJECT_COUNTS
    if _SUBJECT_COUNTS is None:
        counts: dict[str, int] = {}
        for it in all_items():
            subj = _subject_key(it["prompt"])
            counts[subj] = counts.get(subj, 0) + 1
        _SUBJECT_COUNTS = counts
    return _SUBJECT_COUNTS


def clean_prompt(prompt: str, item_id: str | None = None, *, transparent=False) -> str:
    """Normalize a manifest prompt for isolated product-reference generation.

    Replaces the white-backdrop studio boilerplate with the controlled solid
    neutral gray framing, strips vehicle-only variant text spliced into non-vehicle
    items, and hardens rotorcraft prompts (solid props, stationary blades) so
    thin translucent parts matte cleanly and TRELLIS.2 never receives a
    motion-blurred prop disc.
    """
    if is_optimized_prompt(prompt):
        return normalize_generation_prompt(prompt, transparent=transparent)
    p = prompt.strip()
    p = re.sub(r"\)\s*The surface is", ". The surface is", p)
    p = TRANSLUCENT_PROP_RE.sub("solid matte carbon-fiber propeller blades", p)

    # Drop everything from the studio boilerplate to the end; it is rebuilt
    # deterministically below.
    if STUDIO_SPLIT in p:
        p = p[: p.index(STUDIO_SPLIT)]

    subject = TEMPLATE_START_RE.split(p, maxsplit=1)[0]
    is_vehicle = bool(VEHICLE_RE.search(subject))
    # Aircraft/rotorcraft match VEHICLE_RE, but the spliced variant blocks are
    # ground-vehicle gear (mud guards, tire chains, sandbag racks) — nonsense
    # on a quadcopter. Strip them for anything that flies.
    flies = bool(re.search(
        r"\b(quadcopter|hexacopter|octocopter|multirotor|drone|uav|vtol|"
        r"helicopter|fixed-wing|aircraft|plane)\b", subject, re.I))

    if not is_vehicle or flies:
        for block in VARIANT_BLOCKS:
            p = p.replace(block, "")
        p = SURFACE_RE.sub(_neutral_surface, p)
        sentences = re.split(r"(?<=\.)\s+", p)
        sentences = [
            s for s in sentences
            if not (s.startswith("It ") and PART_WORDS_RE.search(s))
        ]
        p = " ".join(sentences)
        p = p.replace("..", ".")
    if flies:
        p = p.replace("visible hydraulic cylinders and hoses, ", "")
        p = p.replace("heavy-gauge welded steel construction", "lightweight aerospace alloy and composite construction")

    if ROTORCRAFT_RE.search(p):
        p = f"{p.rstrip('. ')}. {ROTOR_CLAUSE}"

    # Shared-subject rows get one deterministic appearance variation so the
    # dataset does not collapse to N near-identical assets.
    framing = FRAMINGS[stable_hash(item_id or prompt) % len(FRAMINGS)]
    if item_id and _subject_counts().get(subject, 0) > 1:
        index = _subject_ordinals().get(item_id, stable_hash(f"relief-atlas/var/{item_id}"))
        choices = _variation_choices(subject)
        clause = _variation_clause(subject, item_id)
        framing = FRAMINGS[(index // len(choices)) % len(FRAMINGS)]
        condition = (index // (len(choices) * len(FRAMINGS))) % 4
        if re.search(r"rubble|debris|demolition", subject.split(",", 1)[0], re.I):
            p = SURFACE_RE.sub("", p)
            conditions = [
                "The fragments form a low broad mound with a loose irregular perimeter.",
                "The fragments form a tall compact mound with clustered edges.",
                "The fragments form a lopsided heap with one noticeably higher side.",
                "The fragments form a layered heap with clearly separated tiers.",
            ]
        else:
            conditions = list(NEUTRAL_SURFACE.values())
            p = SURFACE_RE.sub("", p)
        p = f"{p.rstrip('. ').strip()}. {clause} {conditions[condition]}"
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r"[.\s]+$", "", p)
    result = f"{p}.\n{_studio_tail(framing, transparent)}"
    return f"This is an RGBA image with transparency. {result}" if transparent else result


def load_manifest(geography: str) -> list[dict]:
    path = MANIFEST_DIR / MANIFESTS[geography]
    with open(path) as f:
        data = json.load(f)
    return data["meshes"]


def all_items() -> list[dict]:
    items = []
    for geo in MANIFESTS:
        items.extend(load_manifest(geo))
    return items


def select_manifest_items(geographies=None, ids=None, limit=None, only_policy=None):
    """One selection contract for generators, QA, accounting and orchestration.

    Explicit IDs preserve requested order and override geography/limit. Tier
    filtering precedes the limit, as documented by the generation CLI.
    """
    import content_policy

    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if ids:
        by_id = {it["id"]: it for it in all_items()}
        missing = [item_id for item_id in ids if item_id not in by_id]
        if missing:
            raise ValueError(f"Unknown item IDs: {', '.join(missing)}")
        items = [dict(by_id[item_id]) for item_id in dict.fromkeys(ids)]
    else:
        geographies = list(MANIFESTS) if geographies is None else geographies
        unknown = set(geographies) - set(MANIFESTS)
        if unknown:
            raise ValueError(f"Unknown geographies: {', '.join(sorted(unknown))}")
        items = [dict(it) for geo in geographies for it in load_manifest(geo)]
    items = content_policy.select_tier(items, only_policy)
    return items[:limit] if limit is not None and not ids else items


if __name__ == "__main__":
    total = 0
    for geo, fname in MANIFESTS.items():
        items = load_manifest(geo)
        total += len(items)
        print(f"{geo}: {len(items)} items")
    print(f"total: {total}")
    sample = load_manifest("germany")[5]
    print("clean name:", clean_name(sample["name"]))
