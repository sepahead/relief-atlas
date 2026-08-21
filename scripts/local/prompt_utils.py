"""Manifest loading and prompt/name sanitation for the local pipeline."""

import hashlib
import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
MANIFEST_DIR = PROJECT_DIR / "manifests"
OUTPUT_DIR_NAME = "outputs_relief"

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
