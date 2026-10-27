"""Shared, versioned image-reference prompts for local and cloud generation.

Prompt recipes describe the primary asset before its cargo or operating context.
Rewriting preserves the original fields and never changes IDs, policy or outputs.
These constraints guide image generation; they do not certify mechanical accuracy.
"""

import hashlib
import re

from prompt_specs import component_role, object_phrase, reference_evidence, resolve_spec

PROMPT_VERSION = 2
RGBA_PREFIX = "This is an RGBA image with transparency. "
GRAY_BACKGROUND = (
    "Background: flat uniform light neutral gray, with no studio floor, horizon, "
    "pedestal or cast backdrop shadow."
)
ALPHA_BACKGROUND = (
    "Background: The image has alpha channel and the background is transparent. "
    "No studio floor, horizon, pedestal or cast backdrop shadow."
)

# Match the asset noun, rather than equipment carried by it. In particular,
# 'cargo delivery drone ... medical supplies' is not a pallet or medical device.
FAMILY_PATTERNS = [
    ("multirotor", r"\b(?:quadcopter|hexacopter|octocopter|multirotor|drone|uav)\b"),
    ("helicopter", r"\b(?:helicopter|rotorcraft)\b"),
    ("aircraft", r"\b(?:aircraft|airplane|aeroplane|fixed.wing|glider|jet)\b"),
    ("boat", r"\b(?:boat|ship|zodiac|vessel|kayak|canoe|dinghy|frigate|submarine)\b"),
    ("figure", r"\b(?:soldier|infantryman|technician|firefighter|person|figurine|mannequin)\b"),
    ("robot", r"\b(?:robot|ugv|rover|manipulator|gripper|robotic arm)\b"),
    ("rubble", r"\b(?:rubble|debris pile|collapsed building|demolition pile)\b"),
    ("supply_set", r"\b(?:pallet|cache|kit|supplies|distribution point|crate stack)\b"),
    ("tent", r"\b(?:tent|shelter|canopy|air.beam)\b"),
    ("bridge", r"\b(?:bridge|gangway|footbridge)\b"),
    ("barrier", r"\b(?:barrier|gabion|flood gate|dam bladder)\b"),
    ("container", r"\b(?:shipping container|iso container|containerized|housing unit|clinic module|hospital module)\b"),
    ("machinery", r"\b(?:excavator|bulldozer|tractor|crane|loader|forklift)\b"),
    ("vehicle", r"\b(?:truck|van|ambulance|vehicle|chassis|trailer|bus|car|tanker|"
                r"excavator|bulldozer|tractor|crane|loader|motorcycle|forklift|main battle tank|apc|ifv)\b"),
]

GEOMETRY = {
    "generic": "Preserve the specified shape, proportions, material and pose. Parts meet at their real joints; repeated components have consistent size and alignment.",
    "multirotor": "A rigid, fully deployed airframe with a clear central body and distinct arm-to-motor connections. Every propeller hub is concentric with its own motor shaft. Preserve the specified rotor arrangement.",
    "helicopter": "A coherent fuselage with correctly connected landing gear and rotor mounts. Preserve the authored single-main, tandem or coaxial arrangement. Every hub aligns with its own shaft; where a tail rotor is specified, retain the main and tail rotors' different intended axes. All blades are stationary, rigid and sharply defined.",
    "aircraft": "Preserve the specified wing, tail and fuselage arrangement. Paired wings and landing-gear mounts are aligned consistently with the airframe. Propellers, when specified, are stationary with distinct rigid blades.",
    "boat": "One coherent hull with continuous hull surfaces and physically attached propulsion and deck fittings. Keep the specified vessel type and proportions; show the complete hull outside water.",
    "figure": "Preserve the requested clothing, equipment and pose. Anatomically coherent limbs and joints, distinct hands and feet, complete head and body within the frame. Equipment attaches to the stated straps or mounts.",
    "robot": "Keep the stated chassis, locomotion and joint arrangement. Linkages connect at real pivots, actuators align with their joints, and repeated wheels or tracks remain consistent. Preserve the requested articulated pose.",
    "rubble": "One bounded rubble pile with irregular, separately readable fragments and naturally supported contacts. Preserve the described debris materials and exposed reinforcement; no surrounding disaster landscape.",
    "supply_set": "The explicitly described kit or load forms one compact, organized set. Keep each specified component recognizable and supported; use a pallet, tray or crate only when the subject calls for one.",
    "tent": "Preserve the specified shelter design. A continuous fabric skin follows coherent supporting arches, poles or air beams. Seams follow the fabric structure. Interiors appear only through genuine entrances or windows, never through missing wall patches.",
    "bridge": "Preserve the specified bridge section and support system. Deck, beams and braces connect at real structural joints; repeated spans and parallel rails align consistently. No invented roadway or landscape.",
    "barrier": "Preserve the specified barrier materials and module arrangement. Adjacent panels or blocks connect coherently, with aligned joints and supported bases; no unrelated terrain or floodwater.",
    "container": "Preserve the specified module, footprint and wall construction. Straight structural edges, consistent door and window frames and intact wall panels. Interior fittings appear only through genuine openings, without cutaway walls unless explicitly requested.",
    "vehicle": "Preserve the stated chassis and body type. Straight axles, aligned wheels, correctly centered wheel hubs and a continuous body shell. Doors and service panels are closed unless the subject explicitly requires them open. Interior fittings appear through genuine windows or openings, without cutaways or missing body panels.",
    "fastener": "Preserve the specified fastener type, head profile, proportions and material. Threaded features share the intended axis, with continuous thread pitch and an accurately centered drive recess or bore. Keep holes open and mating faces flat.",
    "connector": "Preserve the stated connector standard, housing outline and contact arrangement. Contacts align in consistent rows with their specified pitch; ports, keyways and mounting features remain distinct and unobstructed.",
    "motor": "Preserve the specified linear or rotary actuator type. Shafts, bearing seats and guides align with their intended axes and mounts; retain any explicitly specified offset transmission. Keep the complete housing and terminals connected.",
    "electronics": "Preserve the specified component or package outline, pin arrangement and material. Repeated contacts have consistent spacing and connect coherently to their housing; retain explicitly requested sockets, leads and mounting features.",
    "primitive": "Preserve the exact specified geometric shape, dimensions and proportions. Planar faces remain flat, straight edges remain straight and intended curved surfaces remain regular. Retain requested openings, deformations and markings without adding product-design features.",
    "animal": "Preserve the specified species, anatomy and pose. Species-appropriate limbs, eyes and appendages attach coherently to the body. Show the complete animal with distinct features and no extra limbs.",
    "fabric": "Preserve the described garment or deformable object and its pose. Continuous material follows believable folds, seams and thickness; retain explicitly requested openings, hems, layers and patterns.",
    "machinery": "Preserve the specified mobile or stationary base and the described boom, tool and support arrangement. Structural links connect at real pivots. Any actuators align with their mounts; keep the stated tracks, wheels or base plates rather than substituting another chassis.",
    "gripper": "Keep the stated standalone gripper and its jaws, mounting interface and connected drive mechanism.",
    "vessel": "Preserve the stated vessel shape, cavity, wall thickness, rim, lid and handle configuration.",
    "food": "Keep the specified food identity, edible material, natural shape and whole or cut condition.",
    "toy_animal": "Preserve the stated animal figurine, toy material, species and pose.",
    "object_set": "Keep the requested objects complete and separately readable in the authored arrangement.",
}

SURFACES = {
    "factory-new": "The surface is factory-new: intact material colour and clean panel boundaries, with no artificial grime or damage.",
    "field-used": "The surface shows controlled field use: light dust and localized handling scuffs at exposed edges; material colour, seams and emblems remain readable.",
    "post-deployment": "The surface shows post-deployment wear: localized soil in recesses and abrasion on exposed edges, with continuous panels and intact structural parts.",
    "well-maintained": "The surface is well-maintained: clean material colour with slight service wear at handles and fasteners; clear seams and markings.",
}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_optimized_prompt(prompt):
    body = prompt.removeprefix(RGBA_PREFIX)
    return body.startswith("Subject: ") and "\nGeometry: " in body and "\nBackground: " in body


def subject_family(subject, *, name="", category=""):
    # The first noun phrase is authoritative. Only fall back to the name when
    # it does not match: supply/cargo words later in the description are context.
    part = component_role(subject, name=name, category=category)
    if part:
        return part
    primary = object_phrase(subject)
    identity = primary + " " + name.split(" (", 1)[0]
    fixed_wing = r"\b(?:fixed.wing|airplane|aeroplane|glider)\b"
    if re.search(fixed_wing, identity, re.I) or (
            re.search(r"\b(?:drone|uav|aircraft)\b", identity, re.I) and re.search(fixed_wing, subject, re.I)):
        return "aircraft"
    if re.search(r"\b(?:helicopter|black hornet)\b", identity, re.I):
        return "helicopter"
    for candidate in (primary, name.split(" (", 1)[0]):
        text = candidate.lower()
        if re.search(r"\b(?:fixed.wing|airplane|aeroplane|glider)\b", text):
            return "aircraft"
        if re.search(r"\b(?:helicopter|black hornet)\b", text):
            return "helicopter"
        for family, pattern in FAMILY_PATTERNS:
            if re.search(pattern, text, re.I):
                return family
    if "infantry" in category:
        return "figure"
    for prefix, family in (("industrial_fasteners", "fastener"), ("electronics_connectors", "connector"),
                           ("motors_actuators", "motor"), ("electronics_", "electronics"),
                           ("primitive_", "primitive"), ("animal", "animal"),
                           ("fabric", "fabric"), ("deformable", "fabric")):
        if category.startswith(prefix):
            return family
    return "generic"


def strip_capture_style(prompt):
    """Remove photography boilerplate, retaining authored materials and pose."""
    prompt = prompt.split("The object stands alone", 1)[0]
    prompt = prompt.removeprefix(RGBA_PREFIX).strip()
    style = re.compile(
        r"^(?:photorealistic(?: render(?:ing)?)?|product photography(?: style)?|"
        r"professional (?:product photography|studio lighting)|studio lighting|"
        r"(?:clean |pure |seamless )?(?:white|gr[ae]y)(?: studio| seamless)? background|"
        r"isolated(?: object)?|soft shadows|no shadows(?: on background)?|"
        r"high resolution(?: [48]K)?|[48]K|ultra high resolution [48]K)$", re.I)
    parts = [p.strip(" .") for p in re.split(r",\s*", prompt)]
    parts = [p for p in parts if p and not style.fullmatch(p) and not re.fullmatch(
        r"(?:variation|length variant|v)\s*\d+|(?:detailed )?macro(?: shot| view| photography)?|"
        r"(?:industrial standard|realistic texture|detailed texture|engineering part|industrial construction part|workshop context but isolated)", p, re.I)]
    return ", ".join(parts).rstrip(" .")


def rotor_spec(subject):
    lower = subject.lower()
    count = next((n for term, n in (("octocopter", 8), ("hexacopter", 6), ("quadcopter", 4))
                  if term in lower), None)
    if count is None:
        explicit = re.search(r"\b(four|4|six|6|eight|8)[ -](?:rotor|motor)", lower)
        if explicit:
            count = {"four": 4, "4": 4, "six": 6, "6": 6, "eight": 8, "8": 8}[explicit.group(1)]
    coaxial = bool(re.search(r"\b(?:coaxial|stacked rotors|tandem|tilt.rotor|tiltrotor)\b", lower))
    return count, coaxial


def rotor_geometry(subject):
    """State rotor counts only when authored; never invent a generic drone type."""
    count, coaxial = rotor_spec(subject)
    parts = []
    if count:
        parts.append(f"Exactly {count} motor/rotor units; every unit has a real motor housing and propeller hub.")
    else:
        parts.append(GEOMETRY["multirotor"])
    if not coaxial:
        parts.append("Keep the authored arm and rotor layout. Each hub is concentric with its own motor "
                     "shaft and each motor housing connects to its actual arm mount. Matching components "
                     "have consistent scale. A rotor count alone does not specify the arm count or layout.")
    else:
        parts.append("Keep the explicitly specified coaxial, tandem or tilting arrangement; "
                     "align each hub to its own shaft without substituting a flat radial layout.")
    parts.append("All propellers are stationary with distinct rigid blades. Any described payload remains "
                 "secured to its intended mount, clear of all rotors.")
    return " ".join(parts)


def compose_prompt(subject, *, name="", category="", ordinal=0, condition=None, transparent=False, specification=None):
    subject = strip_capture_style(subject)
    family = subject_family(subject, name=name, category=category)
    specification = specification or resolve_spec(subject, name=name, category=category)
    geometry = rotor_geometry(subject) if family == "multirotor" else GEOMETRY[family]
    geometry = specification.get("geometry") or geometry
    rotor_count, coaxial = rotor_spec(subject) if family == "multirotor" else (None, False)
    if specification.get("rotor_count"):
        rotor_count = specification["rotor_count"]
        coaxial = specification.get("rotor_layout") == "coaxial"
    if rotor_count:
        words = {4: "four", 6: "six", 8: "eight"}
        layout = {4: "quadcopter", 6: "hexacopter", 8: "octocopter"}[rotor_count]
        article = "An" if rotor_count == 8 else "A"
        subject = (f"{article} {layout} with exactly {words[rotor_count]} separate motor/rotor units. "
                   f"{subject[0].upper() + subject[1:]}")
    if family == "tent" and re.search(r"\b(?:inflatable|air.beam)\b", subject, re.I):
        geometry += " Inflated main air beams remain distinct from any explicitly specified rigid spacers or fittings."
    if family in ("multirotor", "helicopter", "aircraft"):
        geometry += " Use the described aerospace materials, without ground-vehicle attachments."
    # Bounded viewpoint variation differentiates repeated catalogue rows without
    # changing tent topology, adding unrelated cargo or deforming a chassis.
    view_index = ordinal // 4 if condition else ordinal
    side = "left" if (view_index // 9) % 2 == 0 else "right"
    azimuth = 25 + 3 * (view_index % 9)
    elevation = (55 if family == "multirotor" and not coaxial else 20) + 5 * ((view_index // 18) % 4)
    surface = SURFACES.get(condition, "The surface retains the specified materials, colours, finish and wear; avoid invented damage or decorations.")
    if family == "rubble":
        surface = ["Fresh fractures and distinct concrete, masonry and metal surfaces.",
                   "Localized dust follows fragment surfaces without hiding their edges.",
                   "Localized soil and weathering follow recesses between fragments.",
                   "Weathered fragments retain readable material boundaries."][ordinal % 4]
        surface = "The surface shows " + surface[0].lower() + surface[1:]
    elif family == "tent" and condition:
        surface = {
            "factory-new": "The surface is clean intact fabric with broad continuous colour and crisp structural seams.",
            "field-used": "The surface shows light fabric creases and localized dirt along the lower hem; the canopy remains intact.",
            "post-deployment": "The surface shows localized dirt and handling wear along seams and the lower hem, without random holes or black streaks across the canopy.",
            "well-maintained": "The surface is clean, tensioned fabric with slight handling creases and intact seams.",
        }[condition]
    view = (f"one elevated three-quarter front-{side} view, about {azimuth} degrees "
            f"around from the front and {elevation} degrees above level. Show depth and the "
            "mounting relationships between parts; avoid overlapping silhouettes where possible.")
    if family == "multirotor" and rotor_count and not coaxial:
        view = (f"A high three-quarter front-{side} view, {azimuth} degrees around from the front "
                f"and looking down at about {elevation} degrees, clearly "
                f"showing all {rotor_count} separate motor housings around the body. The complete "
                "rotor pattern is readable, with no motor hidden behind the central body.")
    if specification.get("view"):
        view = (specification["view"] + f" Show one view from the {side}, about {azimuth} degrees around "
                f"and {elevation} degrees above level, while retaining any explicitly requested pose or view.")
    specified_view = re.search(r"\b(?:top.down|overhead|top view|front view|side view|rear view)\b", subject, re.I)
    if specified_view:
        view = f"Preserve the specified {specified_view.group(0)}; show the entire subject clearly."
    elif category in ("target_marker", "target_surface", "zone_marker", "calibration"):
        view = "Face the specified target surface squarely, with its complete outline and undistorted printed pattern visible."
    lines = [f"Subject: {subject}.", f"Geometry: {geometry}", f"Surface: {surface}", f"View: {view}",
             "Capture: the complete subject or explicitly described set fits inside the central "
             "70 percent of the canvas, centered and uncropped. Leave at least a 10% empty margin "
             "on every side, including all outer tips and protrusions. Even diffuse neutral studio lighting, deep focus and "
             "crisp edges. Coherent material regions and clean boundaries between specified patterns, without baked lighting or harsh "
             "specular highlights. Preserve requested emblems and inscriptions clearly; omit "
             "unrelated text, watermarks, measurement arrows and extra props. Preserve functional bases and supports.",
             ALPHA_BACKGROUND if transparent else GRAY_BACKGROUND]
    result = "\n".join(lines)
    return RGBA_PREFIX + result if transparent else result


def normalize_generation_prompt(prompt, *, transparent=False, category=""):
    """Idempotent backend adaptation; never append a contradictory style suffix."""
    if not is_optimized_prompt(prompt):
        return compose_prompt(prompt, category=category, transparent=transparent)
    lines = prompt.removeprefix(RGBA_PREFIX).splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Background: "):
            lines[index] = ALPHA_BACKGROUND if transparent else GRAY_BACKGROUND
    result = "\n".join(lines)
    return RGBA_PREFIX + result if transparent else result


def rewrite_item(item, *, base_subject=None, ordinal=0):
    """Rewrite every generation field while retaining its original source text."""
    existing = item.get("generation_prompt", {})
    if existing and existing.get("version") not in (1, PROMPT_VERSION):
        raise ValueError(f"Unsupported prompt recipe on {item.get('id')}")
    sources = existing.get("sources") or {key: item[key] for key in
              ("prompt", "text_prompt", "image_prompt") if isinstance(item.get(key), str)}
    if not sources:
        return dict(item)
    original = sources.get("prompt", sources.get("text_prompt", ""))
    condition = next((key for key in SURFACES if key in original.lower()), None)
    result = dict(item)
    specifications = {}
    for field, source in sources.items():
        subject = strip_capture_style(base_subject if base_subject is not None and field == "prompt" else source)
        specification = resolve_spec(subject, name=item.get("name", ""), category=item.get("category", ""), item=item)
        specification["reference_evidence"] = reference_evidence(specification)
        specifications[field] = specification
        result[field] = compose_prompt(subject, name=item.get("name", ""), category=item.get("category", ""),
                                       ordinal=ordinal, condition=condition, specification=specification)
    result["generation_prompt"] = {"version": PROMPT_VERSION, "sources": sources,
        "source_sha256": {key: digest(value) for key, value in sources.items()},
        "subject": base_subject or strip_capture_style(original), "ordinal": ordinal,
        "family": subject_family(base_subject or original, name=item.get("name", ""),
                                 category=item.get("category", "")), "specifications": specifications}
    return result
