"""Manifest loading and prompt/name sanitation for the local pipeline."""

import hashlib
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_DIR / "manifests"
OUTPUT_DIR_NAME = "outputs_relief"

# 3DGS containers the mesh phase can emit, most-preferred first. Every place
# that counts, completes or cleans a 3DGS output must read this list. It lives
# here, in the one dependency-free module every script already imports, because
# the alternative has already cost us a bug: run_forever.sh kept a private copy
# that said `.ply`, so when the condensed export switched to .spz/.splat its
# "meshes pending" count could never reach zero and the unattended multi-day
# driver looped forever without producing anything.
GS_EXTENSIONS = ("spz", "splat", "ply")


def has_gaussians(item_dir, item_id: str) -> bool:
    """Whether any recognised 3DGS container exists for an item."""
    d = Path(item_dir)
    return any((d / f"{item_id}.{ext}").exists() for ext in GS_EXTENSIONS)


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
        and (d / "metadata.json").exists()
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
    "a slightly elevated three-quarter front view",
    "a straight-on eye-level side view",
    "a three-quarter front view",
    "a slightly elevated side view",
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
        "This particular vehicle has a clean recently-repainted body with crisp markings.",
        "This particular vehicle shows heavy field use: chipped paint, mud splatter on the lower panels and dusty windows.",
        "This particular vehicle carries a roof-mounted light bar and antenna mast.",
        "This particular vehicle has a reinforced bull bar front bumper and auxiliary driving lamps.",
        "This particular vehicle has rear-mounted storage racks loaded with strapped equipment cases.",
    ]),
    (re.compile(r"drone|quadcopter|aircraft|plane|helicopter|vtol", re.I), [
        "This particular aircraft has a matte gray airframe with orange high-visibility accents.",
        "This particular aircraft has a white body with red stripe markings and dark sensor gimbal.",
        "This particular aircraft has a compact folding-arm design with the arms folded inward.",
        "This particular aircraft has oversized bulbous sensor housings under the nose.",
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
    "This particular unit has a clean factory finish with crisp unmarked surfaces.",
    "This particular unit shows light service wear: fine scratches, slightly faded color and dust settled in crevices.",
    "This particular unit shows heavy weathering: rain stains, scuffed edges and a dull oxidized sheen.",
    "This particular unit has a few worn touch-up patches and a slightly sun-bleached finish.",
]

STUDIO_SPLIT = "The object stands alone"


def _studio_tail(framing: str) -> str:
    """Controlled studio framing that replaces the manifest boilerplate.

    The original boilerplate demanded a pure white seamless backdrop under
    softbox lighting 'with gentle contact shadows'. That floor + shadow
    gradient survived matting and TRELLIS.2 extruded it into white slab
    geometry. We instead demand a flat uniform light neutral gray backdrop
    with no floor plane at all: gray bounce light stays hue-neutral on the
    subject (saturated backdrops tinted it), matches the neutral-studio
    distribution TRELLIS.2 was trained on, and BiRefNet mattes it cleanly.
    """
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
# scuffed edges". FLUX resolves that by splitting the difference, which is
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


def _variation_clause(subject: str, item_id: str) -> str:
    """Deterministic per-item appearance variation for a shared subject."""
    for pattern, clauses in VARIATION_GROUPS:
        if pattern.search(subject):
            return clauses[stable_hash(f"relief-atlas/var/{item_id}") % len(clauses)]
    return FALLBACK_VARIATIONS[
        stable_hash(f"relief-atlas/var/{item_id}") % len(FALLBACK_VARIATIONS)]


_SUBJECT_COUNTS: dict[str, int] | None = None


def _subject_counts() -> dict[str, int]:
    """How many manifest rows share each raw subject sentence (lazy, cached)."""
    global _SUBJECT_COUNTS
    if _SUBJECT_COUNTS is None:
        counts: dict[str, int] = {}
        for it in all_items():
            subj = TEMPLATE_START_RE.split(it["prompt"].strip(), maxsplit=1)[0]
            counts[subj] = counts.get(subj, 0) + 1
        _SUBJECT_COUNTS = counts
    return _SUBJECT_COUNTS


def clean_prompt(prompt: str, item_id: str | None = None) -> str:
    """Normalize a manifest prompt for local FLUX.2 generation.

    Replaces the white-backdrop studio boilerplate with the controlled solid
    magenta framing, strips vehicle-only variant text spliced into non-vehicle
    items, and hardens rotorcraft prompts (solid props, stationary blades) so
    thin translucent parts matte cleanly and TRELLIS.2 never receives a
    motion-blurred prop disc.
    """
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

    if ROTORCRAFT_RE.search(p):
        p = f"{p.rstrip('. ')}. {ROTOR_CLAUSE}"

    # Shared-subject rows get one deterministic appearance variation so the
    # dataset does not collapse to N near-identical assets.
    if item_id and _subject_counts().get(subject, 0) > 1:
        clause = _variation_clause(subject, item_id)
        # Both the manifest sentence and the variation clause describe the
        # finish: drop the generic manifest one so they cannot contradict.
        # The variation clause is the per-item signal, so it wins.
        if WEAR_CLAUSE_RE.search(clause):
            p = SURFACE_RE.sub("", p)
        p = f"{p.rstrip('. ').strip()}. {clause}"

    framing = FRAMINGS[stable_hash(item_id or prompt) % len(FRAMINGS)]
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r"[.\s]+$", "", p)
    return f"{p}.\n{_studio_tail(framing)}"


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


if __name__ == "__main__":
    total = 0
    for geo, fname in MANIFESTS.items():
        items = load_manifest(geo)
        total += len(items)
        print(f"{geo}: {len(items)} items")
    print(f"total: {total}")
    sample = load_manifest("germany")[5]
    print("clean name:", clean_name(sample["name"]))
