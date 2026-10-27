"""Item-level visual specifications; separate components, assemblies and examples.

External references are local evidence records. This module never uses the network.
An authored shape or chosen design interpretation is not labelled an exact product.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from shape_specs import shape_spec

LIBRARY = Path(__file__).resolve().parents[2] / "config/reference_library.json"


@lru_cache(maxsize=1)
def reference_library():
    return json.loads(LIBRARY.read_text())


def reference_evidence(specification):
    """Bind cited source records, without upgrading a type example to a replica."""
    library = reference_library()["sources"]
    records = {}
    for key in specification["sources"]:
        source = library[key]  # An unknown citation is a recipe error.
        records[key] = {"url": source["url"], "record_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "visual_example_reviewed": bool(source.get("visual_review"))}
    return records


def object_phrase(subject):
    parts = [part.strip() for part in subject.split(",") if part.strip()]
    if len(parts) > 1 and re.fullmatch(
            r"(?:raw industrial material sample|heavy duty hydraulic part|industrial pneumatic component|"
            r"scientific lab equipment|industrial safety equipment|industrial automation sensor|electronic component|electrical connector)",
            parts[0], re.I):
        return parts[1]
    return re.split(r"\b(?:with|carrying|equipped|for)\b", parts[0] if parts else subject,
                    maxsplit=1, flags=re.I)[0].strip()


def component_role(subject, *, category="", name=""):
    primary = object_phrase(subject).lower()
    identity = primary + " " + name.lower().split(" (", 1)[0]
    if category == "multi_object" or re.search(r"\b(?:swarm|fleet|set of|stack of|group of)\b", identity):
        return "object_set"
    if category == "motors_actuators" or re.search(r"\b(?:brushless|stepper|servo|dc|geared|outrunner)\b.*\bmotor\b", primary):
        return "motor"
    if re.search(r"\b(?:gripper|end.effector|suction cup)\b", primary) or category == "robot_part":
        return "gripper"
    if category.startswith("electronics_ics"):
        return "electronics"
    if not category.startswith("primitive_") and re.search(r"\b(?:cup|mug|bowl|jar|bucket|drinking glass|storage box|storage container)\b", primary):
        return "vessel"
    if category == "container" and re.search(r"\b(?:box|bin|container)\b", primary):
        return "vessel"
    edible = re.search(r"\b(?:croissant|bagel|bread|pastry|fruit|apple|banana|lemon)\b", primary)
    if (edible and not re.search(r"\b(?:iphone|ipad|macbook|cube|sphere|cone|cylinder|marker)\b", primary)) or (
            category in ("food", "food_extra", "fruit") and re.search(r"\borange\b", primary)):
        return "food"
    if category == "animal" and re.search(r"\b(?:toy|figurine)\b", subject, re.I):
        return "toy_animal"
    return None


def _spec(profile, geometry, *, sources=(), scope="authored_description", checks=(), view=None, description=None, **extra):
    return {"profile": profile, "geometry": geometry, "sources": list(sources), "scope": scope,
            "checks": list(checks), "view": view, "description": description, **extra}


def resolve_spec(subject, *, name="", category="", item=None):
    item = item or {}
    primary = object_phrase(subject)
    text = (primary + " " + name.split(" (", 1)[0]).lower()
    full = subject.lower()
    role = component_role(subject, category=category, name=name)
    base_name = name.split(" (", 1)[0]
    if base_name == "French Red Cross Ambulance":
        return _spec("french_red_cross_ambulance", "A two-axle ambulance van with aligned wheel hubs, a smooth "
                     "continuous white patient-compartment body, correctly framed cab glazing and closed patient doors. "
                     "Large red-cross emblems lie flat on the white body panels; French identification stays on those panels. "
                     "Medical fittings remain inside the normal body rather than replacing exterior panels.",
                     sources=["sprinter_ambulance"], scope="ambulance_family_example",
                     checks=["continuous exterior panels", "aligned axles", "red cross shape readable", "no invented equipment openings"],
                     design_choice="A panel-van ambulance body is used as a generic visual interpretation; no particular French fleet chassis is asserted.")
    if base_name == "Field Hospital Tent Large":
        return _spec("large_medical_tent", "One large deployed medical tent shell with a consistent aluminium "
                     "support-frame layout, opaque continuous pale fabric, structural seams, a genuine entrance and "
                     "rectangular window panels. The multi-ward, operating and pharmacy functions are inside; "
                     "exterior fabric walls remain intact rather than becoming transparent cutaways.",
                     sources=["lanco_sg200"], checks=["consistent support frame", "continuous canopy", "real door/window openings"],
                     scope="chosen_medical_frame_family_example",
                     design_choice="A modular aluminium-frame shelter is a visual interpretation of the generic field hospital; SG200 is a frame example, not a claim about the hospital's size or exact product.")
    if base_name == "Earthquake Rubble Pile":
        return _spec("concrete_rubble", "One bounded irregular heap of broken concrete slabs and debris. "
                     "Fragments have readable thickness and fracture faces; exposed rebar emerges from concrete "
                     "rather than floating independently. Pieces meet at plausible contacts and retain the irregular perimeter.",
                     checks=["distinct slab thickness", "rebar attached to fragments", "bounded pile"],
                     scope="authored_debris_requires_scene_reference")
    if role == "motor":
        if "linear actuator" in text:
            form = shape_spec(primary, category)
            return _spec(**form, checks=["linear output rod aligned to guide", "authored extension and mounts"])
        geometry = ("One standalone motor or actuator, with a coherent housing, mounting features and output "
                    "shaft or rod. Keep the described axis, terminals and cable exit visible; depict the component itself.")
        if "brushless" in text or "outrunner" in text:
            geometry = ("One compact brushless motor: cylindrical rotor bell concentric with its stator and central "
                        "shaft, a mounting base and attached electrical leads. Show the motor as a standalone component.")
        return _spec("standalone_motor", geometry, checks=["one component", "shaft concentric with housing", "requested shaft/connector view"],
                     view="A three-quarter component view showing the output shaft, mounting base and cable exit; retain any explicitly requested orientation.")
    if role == "gripper":
        if "franka" in full or "panda" in full or "franka" in name.lower():
            return _spec("franka_parallel_hand", "The end-effector alone: compact white and gray housing, two opposed "
                         "parallel fingers with dark gripping pads, and its mounting flange. Fingers connect to the "
                         "housing and face each other across one open grasping gap.", sources=["franka_hand"], scope="model_family_reference",
                         checks=["two opposed fingers", "open grasping gap", "end-effector, not a whole robot arm"])
        if "suction" in text:
            return _spec("suction_end_effector", "A standalone suction end-effector with the specified cup count, "
                         "an open compliant cup rim, connected vacuum fitting and mounting interface.", checks=["specified cup count", "open cup rim"])
        return _spec("standalone_gripper", "The described gripper alone, with its stated jaw or finger arrangement, "
                     "mounting interface and a clear grasping gap. Fingers remain connected to their guides or joints.",
                     checks=["requested jaw count and pose", "clear grasping gap"])
    if category == "electronics_ics":
        package = re.search(r"\b(DIP|SOIC|SOP|TSSOP|QFN|QFP|BGA|LGA)\s*[-_]?\s*(\d+)\b", subject, re.I)
        if package:
            kind, count = package.group(1).upper(), int(package.group(2))
            if kind in ("DIP", "SOIC", "SOP", "TSSOP") and count % 2 == 0:
                shape = "through-hole legs" if kind == "DIP" else "surface-mount gull-wing leads"
                return _spec("ic_" + kind.lower(), f"A closed molded {kind}{count} package with exactly {count} {shape}, "
                             f"in two parallel rows of {count // 2} along its long sides. Preserve the package orientation "
                             "mark and lead spacing; the semiconductor die stays inside its package.", sources=["ti_package_families"],
                             scope="package_family_with_authored_pin_count", checks=[f"{count} leads total", f"{count // 2} per long side", "closed package body"],
                             view="An elevated view showing the package body, end orientation mark and both lead rows.")
            if kind in ("QFN", "QFP"):
                contacts = "edge contact pads without protruding legs" if kind == "QFN" else "formed leads around the perimeter"
                return _spec("ic_" + kind.lower(), f"A closed {kind}{count} package with the specified {count} "
                             f"terminals as {contacts}. Keep the package outline and terminal layout distinct.",
                             sources=["ti_package_families"], scope="package_family_with_authored_pin_count", checks=[f"{count} terminals", "correct leaded or leadless package"])
            return _spec("ic_" + kind.lower(), f"A closed {kind}{count} package with its specified underside contact "
                         "array. Keep the body and actual ball-grid or land-grid contact type distinct.", sources=["ti_package_families"],
                         scope="package_family_with_authored_pin_count", checks=["requested contact type and count", "no invented side leads"])
        if re.search(r"\bTO\s*[-_]?220\b", subject, re.I):
            return _spec("to220", "A packaged TO-220 component: a closed black molded body, flat metal mounting "
                         "tab with one hole and three aligned leads exiting one edge. The die remains inside.",
                         sources=["ti_package_families"], scope="package_family_example", checks=["mounting tab", "three leads", "closed molded body"])
        if re.search(r"\bBGA\b", subject, re.I):
            return _spec("ic_bga", "One closed ball-grid-array package with its contact balls arranged on the "
                         "underside, not protruding from its sides. Keep the authored package outline; no terminal count is inferred.",
                         sources=["ti_package_families"], scope="package_family_reference",
                         checks=["underside ball contacts", "no invented side leads"], view="An underside oblique view showing the contact array and package thickness.")
    if re.search(r"fly\s*cart\s*30", full):
        return _spec("flycart30_coaxial", "Four deployed arms, each ending in a coaxial pair of motor/rotor units: "
                     "eight rotors in four paired stacks. Upper and lower hubs share each local axis and occupy "
                     "different heights. Keep the specified cargo system clear of all propellers.", sources=["dji_flycart30"],
                     scope="named_model_reference", checks=["four arms", "four coaxial pairs", "eight rotors"],
                     rotor_layout="coaxial", rotor_count=8, arm_count=4)
    if "octocopter" in full and "cargo" in full and not re.search(
            r"\b(?:coaxial|stacked|tandem|tilt.rotor|tiltrotor|four arms|4 arms)\b", full):
        return _spec("cargo_flat_octocopter", "Exactly eight equal radial booms, one separate motor housing and "
                     "stationary propeller at each boom end, around one compact central frame. Keep eight complete "
                     "assemblies in a coherent ring, with parallel vertical shafts and hubs at a common height. "
                     "Each propeller hub is concentric with its own motor shaft. A closed cargo pod "
                     "is secured below the centre, clear of the rotor ring and landing supports.", sources=["freefly_alta8_manual"],
                     scope="chosen_layout_example", checks=["eight booms", "eight complete motor/rotor units", "concentric hubs", "cargo clear of rotors"],
                     view="A high oblique view that makes all eight separate boom-end motor housings countable.",
                     rotor_layout="radial", rotor_count=8, arm_count=8,
                     design_choice="The generic cargo octocopter uses an ALTA8-like eight-boom layout; it is not labelled as an ALTA8 replica.")
    if "luf" in text and re.search(r"\b60\b", text):
        return _spec("luf60", "A compact tracked firefighting robot with one large circular ducted ventilator "
                     "mounted above its low chassis. The fan has a coherent ring, hub, blades and protective grille; "
                     "water-fog fittings belong to the fan assembly. Keep both tracks attached and the fan clear of the body.",
                     sources=["luf60_operator"], scope="named_model_reference",
                     checks=["large ducted fan", "tracked base", "fan connected to chassis"])
    if "brokk" in text:
        return _spec("brokk_demolition_family", "A compact remotely operated demolition machine with a tracked "
                     "base, articulated hydraulic arm and the specified breaker attachment. Hydraulic cylinders "
                     "and hoses follow the arm joints, and the machine's stabilizers connect to its base.",
                     sources=["brokk170"], scope="manufacturer_family_example",
                     checks=["tracked chassis", "connected tool arm", "requested breaker attachment"],
                     design_choice="Brokk170 illustrates the family; the generic manifest name does not establish an exact model or measured dimensions.")
    if re.search(r"\b(?:franka|panda)\b", text) and "arm" in text:
        return _spec("franka_arm", "Keep the authored Franka arm as a complete connected articulated robot, with "
                     "its specified seven-joint arrangement, mounting base and requested end-effector. Preserve "
                     "the alternating housing/link proportions rather than replacing it with a generic six-axis arm.",
                     sources=["franka_arm"], scope="manufacturer_arm_family_example",
                     checks=["seven articulated joints", "correct arm versus hand identity"],
                     design_choice="Research3 documents the seven-axis arm family; a Panda row keeps its authored Panda identity rather than being renamed Research3.")
    if role == "vessel":
        if re.search(r"\b(?:mug|cup|bowl|glass|bucket|jar)\b", primary, re.I):
            return _spec("hollow_household_vessel", "A household vessel with a continuous wall, defined rim and "
                         "real interior cavity. Keep the stated handle, lid, material and opening; preserve an "
                         "explicitly inverted, tilted or sideways pose.", checks=["open cavity where specified", "requested handle and pose"],
                         view="A three-quarter view making the rim, cavity and any handle visible, while preserving the authored pose.")
        return _spec("small_storage_container", "A small storage container with the specified lid or open top, "
                     "wall thickness, rim and closure features. Keep the intended cavity and opening state.", checks=["correct lid/opening state", "household-scale container"])
    if role == "toy_animal":
        return _spec("molded_animal_figurine", "A manufactured animal figurine with the specified species, "
                     "pose and simplified coherent anatomy. Preserve its molded toy material and painted "
                     "features, with the correct species-specific limbs and appendages.", checks=["specified toy material", "species-appropriate anatomy"])
    if role == "food":
        return _spec("food_object", "Preserve the specific food form, edible surface and stated cut or whole "
                     "condition. Keep natural irregularity and the specified rind, skin, crust or crumb.", checks=["food identity", "specified whole/cut condition"])
    if role == "object_set":
        if "swarm" in text and re.search(r"\b5\b", subject) and "quadcopter" in full:
            return _spec("five_quadcopter_set", "Exactly five separate complete quadcopters, each with its own "
                         "central body, four arms and four motor/propeller units. Arrange the five aircraft as an "
                         "uncluttered group with gaps between airframes; do not share arms or rotors between aircraft.",
                         checks=["five separate aircraft", "four rotors on each aircraft", "no fused airframes"],
                         view="A high oblique view showing all five airframes and their separate rotor layouts.")
        return _spec("explicit_object_set", "The explicitly requested objects remain separate complete objects "
                     "in their stated arrangement, with readable boundaries and correct contact relationships. "
                     "Do not fuse the set into one machine or substitute one object for the entire collection.",
                     checks=["requested object count", "each object complete", "correct arrangement and contacts"])
    if base_name == "UNHCR Family Tent":
        return _spec("unhcr_family_tent", "One family shelter with a rectangular main sleeping volume, "
                     "a pitched ridge roof and a vestibule at each end. Preserve the authored pale fabric "
                     "and blue stripe, coherent poles, tied guy lines, defined entrance flaps and continuous "
                     "skin; the main sleeping floor is distinct from the vestibules.",
                     sources=["unhcr_family_tent"], scope="humanitarian_tent_family_reference",
                     checks=["ridge roof and two vestibules", "continuous tent skin", "authored markings"],
                     design_choice="The cited eco-design specification is without logos. The row's blue stripe is retained from its authored description; no official logo layout is inferred.")
    if base_name == "Inflatable Emergency Shelter":
        return _spec("inflatable_shelter", "One deployed air-beam shelter with inflated main supporting "
                     "arches and a continuous opaque fabric skin, coherent seams and genuine entrance and "
                     "window openings. Any described rigid spacers remain secondary fittings, not a substituted pole frame.",
                     sources=["lanco_azf"], scope="inflatable_shelter_family_example",
                     checks=["inflatable main frame", "continuous fabric skin", "genuine openings"])
    form = shape_spec(primary, category)
    if form:
        sources = ["nsk_bearing_types"] if form["profile"] == "bearing" else []
        if category == "ycb_object":
            sources.append("ycb_examples")
        return _spec(**form, sources=sources, scope="authored_type_with_family_example" if sources else "authored_type_geometry",
                     checks=["authored identity and dimensions", "specified material and pose", "type-specific geometry"])
    if category == "ycb_object":
        return _spec("ycb_authored_object", "Preserve the specifically named everyday object, its authored "
                     "package shape, material and label layout. Keep actual openings, closures and functional "
                     "features coherent; do not substitute a generic object solely from its colour.",
                     sources=["ycb_examples"], scope="collection_example_requires_exact_item_check",
                     checks=["exact named object", "package geometry and closure", "authored label layout"])
    return _spec("authored_" + (category or "subject"), None, checks=["authored identity", "authored proportions, material and pose"],
                 scope="authored_description_requires_type_reference")
