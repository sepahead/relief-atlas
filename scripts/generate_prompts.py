#!/usr/bin/env python3
"""
generate_prompts.py — Generates 10,000 disaster relief & recovery mesh items.

Produces 4 manifest JSON files:
  - relief_manifest_germany.json  (~4,000 items)
  - relief_manifest_eu.json       (~2,500 items)
  - relief_manifest_ukraine.json  (~2,500 items)
  - relief_manifest_general.json  (~1,000 items)

Usage:
  python generate_prompts.py [--output-dir manifests/]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "manifests"
sys.path.insert(0, str(Path(__file__).resolve().parent / "local"))
from generation_prompts import PROMPT_VERSION, is_optimized_prompt, rewrite_item  # noqa: E402
from pipeline_utils import write_json  # noqa: E402


# ---------------------------------------------------------------------------
# Data: base objects per category
# ---------------------------------------------------------------------------

# Each base object: (name, base_prompt, tags)
# Variants are generated from CONFIG_VARIANTS, CONDITION_VARIANTS, SCENE_VARIANTS

CONFIG_VARIANTS = [
    # Each variant describes VISUAL modifications the model can actually render
    "It is equipped for flood response with water pump attachments bolted to the side panels, extended mud guards, an elevated air intake snorkel on the cab, and a sandbag rack welded to the roof.",
    "It carries earthquake rescue gear: hydraulic rescue tools (jaws of life) mounted visibly on the exterior, structural shoring timber strapped to the frame, and a debris hook on the rear.",
    "It is configured for winter operations with tire chains wrapped around the wheels, a heated equipment enclosure on top, antifreeze container markings on the side, and cold-weather insulation wrap on exposed pipes.",
    "It is deployed for wildfire response with heat shield panels riveted to the sides, ember guard mesh over the air intake, an orange fire-retardant tank on the rear, and scorched paint around the exhaust.",
    "It is configured for storm response with wind-rated anchor points visible on the corners, a folded rain cover tarp on top, reinforced structural bracing added to the frame, and a drainage pump attached at the base.",
    "It carries a standard multipurpose disaster relief loadout with basic equipment, clearly visible organization markings, and a utility rack loaded with mixed supplies.",
]

CONDITION_VARIANTS = [
    # Each variant describes specific wear/patina/material state
    "The surface is pristine factory-new: fresh paint with no scratches, glossy finish, protective film still on instrument displays, and zero wear on tires and tracks.",
    "The surface shows field-used operational wear: dirt and dust coating on lower surfaces, scuff marks on bumpers and handles, faded decals from sun exposure, and tire tread worn to half depth.",
    "The surface shows heavy post-deployment wear: mud caked in crevices and wheel wells, paint chips revealing bare primer underneath, minor dents on body panels, oil stains near the engine bay, and cracked rubber seals.",
    "The surface is well-maintained: recently washed but showing honest service wear, touch-up paint visible on scratches, clean interior, tires at half tread, and minor surface rust on the undercarriage.",
]

SCENE_VARIANTS = [
    "It has just returned from a disaster site, with fresh mud splatter on the lower body and wet surfaces reflecting studio light, emergency equipment still deployed.",
    "It is in transit configuration: travel covers on sensitive equipment, tie-down straps visible, clean from a recent wash before deployment.",
    "It is at a staging area with all equipment fully deployed and ready, doors and hatches open showing the interior, all indicator lights and systems active.",
    "It is in depot storage: protective covers partially removed, some dust on upper surfaces, batteries disconnected, long-term storage markings visible on the hull.",
]

# Historical template text is retained as source provenance and for stable ID
# expansion. rewrite_item emits the actual versioned image-reference recipe;
# this old backdrop, depth-of-field and attachment boilerplate is not sent to
# generation models.
PROMPT_SUFFIX = (
    "The object stands alone on a clean pure white seamless backdrop, centered in frame, "
    "under even softbox studio lighting with gentle contact shadows. "
    "Product reference photograph for 3D model generation. "
    "Medium-format camera, 85mm lens, moderate depth of field. "
    "No watermark, no logos, no extra objects, no text overlays."
)


def make_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:04d}"


def enrich_prompt(base: str) -> str:
    """Add material/texture detail to base prompts based on object keywords.
    This helps the image model produce realistic surfaces for 3D conversion."""
    additions = []
    lower = base.lower()

    # Vehicles
    if any(w in lower for w in ["truck", "van", "vehicle", "ambulance", "chassis",
                                "transport", "tanker", "bus"]):
        if "steel" not in lower:
            additions.append("painted steel body panels with aluminum checker-plate lower sections")
        if "glass" not in lower and "window" not in lower:
            additions.append("tinted safety glass windows")
        if "rubber" not in lower and "tire" not in lower:
            additions.append("black rubber tires with visible tread pattern")
    elif any(w in lower for w in ["excavator", "crane", "loader", "bulldozer",
                                   "tracked", "machinery", "heavy"]):
        if "hydraulic" not in lower:
            additions.append("visible hydraulic cylinders and hoses")
        if "steel" not in lower:
            additions.append("heavy-gauge welded steel construction with yellow safety paint")
    elif any(w in lower for w in ["tent", "shelter", "canopy"]):
        if "canvas" not in lower and "fabric" not in lower:
            additions.append("heavy-duty ripstop canvas fabric with aluminum pole frame")
    elif any(w in lower for w in ["drone", "uav", "quadcopter", "hexacopter",
                                   "fixed-wing"]):
        if "carbon" not in lower:
            additions.append("carbon fiber frame arms with matte black plastic housings")
        if "propeller" not in lower and "rotor" not in lower:
            additions.append("translucent polycarbonate propeller blades")
    elif any(w in lower for w in ["robot", "ugv", "tracked robot",
                                   "manipulator"]):
        if "aluminum" not in lower:
            additions.append("machined aluminum chassis with anodized surface")
        if "rubber" not in lower and "track" not in lower:
            additions.append("rubber caterpillar tracks with steel drive sprockets")
    elif any(w in lower for w in ["boat", "zodiac", "inflatable"]):
        if "pvc" not in lower and "hypalon" not in lower:
            additions.append("reinforced hypalon rubber inflatable tubes")
    elif any(w in lower for w in ["generator", "power"]):
        if "steel" not in lower:
            additions.append("powder-coated steel enclosure with ventilation louvers")
    elif any(w in lower for w in ["crate", "container", "pallet", "box",
                                   "supply"]):
        if "plastic" not in lower and "steel" not in lower:
            additions.append("rotation-molded polyethylene with recessed handles")
    elif any(w in lower for w in ["pump", "filtration", "water"]):
        if "stainless" not in lower:
            additions.append("stainless steel piping with blue industrial valve handles")
    elif any(w in lower for w in ["mine", "demining", "eod"]):
        additions.append("matte olive drab finish with high-visibility safety markings")
    elif any(w in lower for w in ["medical", "hospital", "clinic"]):
        if "stainless" not in lower:
            additions.append("medical-grade stainless steel fixtures and white laminate surfaces")

    if additions:
        return base + ", " + ", ".join(additions)
    return base


def build_items(prefix: str, category: str, geography: str, organization: str,
                objects: list[tuple[str, str, list[str]]],
                target: int,
                extra_variants: list[str] | None = None) -> list[dict]:
    """
    Expand base objects with variants until target count is reached.
    Each object: (name, base_prompt, tags_list)
    Uses cartesian product of (object × config_variant × condition × scene) to avoid duplicates.
    """
    items = []
    idx = 1
    all_config_variants = list(CONFIG_VARIANTS)
    if extra_variants:
        all_config_variants.extend(extra_variants)

    seen_prompts = set()
    subject_ordinals = {}

    # Generate all unique combinations in nested loops
    for pass_num in range(100):  # safety limit
        for obj_idx, (name, base_prompt, tags) in enumerate(objects):
            for var_idx, config_var in enumerate(all_config_variants):
                for cond_idx, cond_var in enumerate(CONDITION_VARIANTS):
                    if len(items) >= target:
                        break

                    # Build variant string
                    variant_parts = [config_var, cond_var]
                    if pass_num > 0:
                        # Add scene variant to differentiate passes
                        scene_idx = pass_num - 1
                        if scene_idx < len(SCENE_VARIANTS):
                            variant_parts.append(SCENE_VARIANTS[scene_idx])
                        else:
                            variant_parts.append(f"Variation set {pass_num + 1}.")

                    # Assemble with natural sentence flow (not comma-separated keywords)
                    variant_str = " ".join(variant_parts)
                    full_prompt = f"{enrich_prompt(base_prompt)}. {variant_str} {PROMPT_SUFFIX}"

                    # Skip duplicates
                    if full_prompt in seen_prompts:
                        continue
                    seen_prompts.add(full_prompt)

                    row = {
                        "id": make_id(prefix, idx),
                        "category": category,
                        "geography": geography,
                        "organization": organization,
                        "name": f"{name} ({variant_str})",
                        "prompt": full_prompt,
                        "tags": tags,
                    }
                    ordinal = subject_ordinals.get(name, 0)
                    items.append(rewrite_item(row, base_subject=base_prompt, ordinal=ordinal))
                    subject_ordinals[name] = ordinal + 1
                    idx += 1

                if len(items) >= target:
                    break
            if len(items) >= target:
                break
        if len(items) >= target:
            break

    return items[:target]


# ---------------------------------------------------------------------------
# GERMANY (~4,000)
# ---------------------------------------------------------------------------

GER_DRK_OBJECTS = [
    ("DRK Mobile Blood Donation Unit", "German Red Cross (DRK) mobile blood donation unit, Mercedes Sprinter chassis, red cross markings, white with red stripe, medical equipment visible inside", ["medical", "vehicle", "red_cross"]),
    ("DRK Ambulance Mercedes Sprinter", "DRK ambulance vehicle, Mercedes Sprinter, white with red cross markings, emergency lights, fully equipped medical interior", ["medical", "vehicle", "red_cross"]),
    ("DRK Field Hospital Tent Module", "DRK modular field hospital tent, large white medical tent with red cross emblem, internal lighting, medical beds visible", ["medical", "shelter", "red_cross"]),
    ("DRK Emergency Supply Crate Stack", "DRK emergency supply crates, stackable red and white containers with Red Cross logo, medical supplies and food rations", ["supplies", "red_cross"]),
    ("DRK Rescue Boat Zodiac", "DRK water rescue inflatable zodiac boat, red and orange, outboard motor, rescue equipment on board", ["rescue", "water", "red_cross"]),
    ("DRK First Aid Station Module", "DRK mobile first aid station, portable medical treatment module, white with red cross, triage equipment visible", ["medical", "red_cross"]),
    ("DRK Blood Mobile Unit", "DRK bloodmobile, large white truck with red cross markings, refrigerated blood storage interior, donation beds", ["medical", "vehicle", "red_cross"]),
    ("DRK Emergency Blanket Pallet", "DRK emergency thermal blanket pallet, shrink-wrapped stack of grey and silver emergency blankets on pallet", ["supplies", "red_cross"]),
    ("DRK Portable Stretcher System", "DRK folding emergency stretcher system, aluminum frame with orange canvas, wheeled base, medical transport", ["medical", "red_cross"]),
    ("DRK Mobile Clinic Container", "DRK mobile clinic shipping container, converted 20ft container with medical interior, white with red cross", ["medical", "shelter", "red_cross"]),
    ("DRK Command Vehicle VW Crafter", "DRK incident command vehicle, VW Crafter van, white with red cross markings, communications antenna, command interior", ["vehicle", "command", "red_cross"]),
    ("DRK Disaster Response Kit Pallet", "DRK pre-packed disaster response kit on pallet, includes tents, blankets, medical supplies, shrink-wrapped", ["supplies", "red_cross"]),
    ("DRK Water Purification Trailer", "DRK mobile water purification trailer, white trailer with filtration system, DRK markings, produces potable water", ["water", "vehicle", "red_cross"]),
    ("DRK Kitchen Field Unit", "DRK field kitchen unit, mobile cooking facility in white container with red cross, serves hot meals at disaster sites", ["supplies", "red_cross"]),
    ("DRK Generator Trailer", "DRK emergency power generator on trailer, yellow diesel generator with DRK markings, cable distribution system", ["power", "vehicle", "red_cross"]),
]

GER_THW_OBJECTS = [
    ("THW Excavator Caterpillar", "Technisches Hilfswerk (THW) excavator, Caterpillar 320, orange with THW blue markings, tracked, disaster debris removal", ["heavy_machinery", "thw"]),
    ("THW Mobile Crane Liebherr", "THW mobile crane, Liebherr LTM series, blue and orange THW livery, telescopic boom, rescue operations", ["heavy_machinery", "thw"]),
    ("THW Pontoon Bridge Section", "THW modular pontoon bridge section, steel floating bridge component, blue THW markings, rapid deployment", ["infrastructure", "thw"]),
    ("THW Water Purification Plant", "THW mobile water purification plant, containerized water treatment system, THW blue, produces drinking water", ["water", "thw"]),
    ("THW Lighting Tower", "THW mobile lighting tower, telescopic mast with LED floodlights, blue THW markings, nighttime disaster illumination", ["power", "thw"]),
    ("THW Generator Truck MAN", "THW emergency power generator truck, MAN TGM chassis, blue THW livery, large diesel generator in container body", ["power", "vehicle", "thw"]),
    ("THW Sandbag Filling Machine", "THW automated sandbag filling machine, portable conveyor system for rapid sandbag production, THW blue", ["infrastructure", "thw"]),
    ("THW Command Vehicle", "THW incident command vehicle, Mercedes Sprinter, blue with THW logo, communications equipment, mobile command center", ["vehicle", "command", "thw"]),
    ("THW Rescue Boat", "THW water rescue boat, rigid-hull inflatable, blue and orange THW markings, jet drive, flood operations", ["rescue", "water", "thw"]),
    ("THW Modular Bridge Kit", "THW emergency bridge kit, prefabricated steel bridge sections on flatbed, rapid deployment for damaged infrastructure", ["infrastructure", "thw"]),
    ("THW Concrete Saw Unit", "THW portable concrete cutting station, diamond blade saw system for urban search and rescue, THW blue", ["rescue", "thw"]),
    ("THW Pump Unit Submersible", "THW high-capacity submersible pump unit, portable flood pump with hoses, THW blue, disaster dewatering", ["water", "thw"]),
    ("THW Welding Generator Unit", "THW mobile welding and cutting station, generator-powered welder with gas cutting equipment, THW blue", ["repair", "thw"]),
    ("THW Debris Conveyor System", "THW portable debris conveyor belt system, modular belt for clearing rubble, THW blue and orange", ["heavy_machinery", "thw"]),
    ("THW Personnel Transport Bus", "THW personnel transport bus, Mercedes Citaro in THW blue, seats rescue workers for deployment", ["vehicle", "thw"]),
    ("THW Logistics Truck", "THW logistics truck, MAN TGX box truck in THW blue, carries equipment and supplies to disaster sites", ["vehicle", "supplies", "thw"]),
    ("THW Mobile Workshop Container", "THW mobile workshop container, equipped with tools for field repairs, THW blue markings", ["repair", "thw"]),
    ("THW Aerial Survey Drone", "THW survey drone, DJI Matrice with thermal camera, used for disaster area mapping and survivor detection", ["drone", "thw"]),
]

GER_BW_RELIEF_OBJECTS = [
    ("Bundeswehr MAN Multi FSA Transport", "Bundeswehr disaster relief MAN Multi FSA truck, military transport in olive green, carrying relief supplies, German flag markings", ["vehicle", "bundeswehr", "relief"]),
    ("Bundeswehr Unimog U5000 Relief", "Bundeswehr Unimog U5000, off-road disaster relief vehicle, olive green, extreme terrain capability, supply transport", ["vehicle", "bundeswehr", "relief"]),
    ("Bundeswehr Field Kitchen FK250", "Bundeswehr mobile field kitchen trailer FK250, olive green, large-scale meal preparation for disaster victims", ["supplies", "bundeswehr", "relief"]),
    ("Bundeswehr Water Tanker Truck", "Bundeswehr water tanker truck, 15000 liter capacity, olive green with white relief markings, potable water distribution", ["water", "vehicle", "bundeswehr"]),
    ("Bundeswehr Mobile Hospital Module", "Bundeswehr Role 2 mobile hospital module, expandable container hospital, olive green, surgical capability", ["medical", "bundeswehr", "relief"]),
    ("Bundeswehr CH-53 Relief Helicopter", "Bundeswehr CH-53GA heavy lift helicopter in disaster relief configuration, olive green, external cargo hook, supply airdrop", ["aircraft", "bundeswehr", "relief"]),
    ("Bundeswehr Temporary Bridge Bailey", "Bundeswehr Bailey-type temporary bridge kit, military engineer bridge sections on trucks, rapid infrastructure repair", ["infrastructure", "bundeswehr"]),
    ("Bundeswehr Engineer Excavator", "Bundeswehr military engineer excavator, armored cab, olive green, debris clearance and earthworks for relief", ["heavy_machinery", "bundeswehr"]),
    ("Bundeswehr Fuel Tanker", "Bundeswehr fuel tanker truck, olive green, mobile fuel supply for relief vehicle fleet", ["vehicle", "bundeswehr", "supplies"]),
    ("Bundeswehr Recovery Vehicle", "Bundeswehr heavy recovery vehicle, MAN HX chassis, olive green, crane and winch for vehicle recovery", ["vehicle", "bundeswehr", "rescue"]),
    ("Bundeswehr NH90 Medevac", "Bundeswehr NH90 helicopter in medevac configuration, olive green with red cross, medical evacuation from disaster zones", ["aircraft", "medical", "bundeswehr"]),
    ("Bundeswehr Floating Bridge M3", "Bundeswehr M3 amphibious floating bridge vehicle, olive green, rapid river crossing for relief operations", ["infrastructure", "bundeswehr", "water"]),
    ("Bundeswehr Communication Truck", "Bundeswehr mobile communications center truck, satellite uplink, olive green, disaster area coordination", ["vehicle", "command", "bundeswehr"]),
    ("Bundeswehr Water Purification Unit", "Bundeswehr mobile water purification system, containerized, olive green, field water treatment plant", ["water", "bundeswehr"]),
]

GER_POLIZEI_OBJECTS = [
    ("Bundespolizei Patrol Vehicle VW", "Bundespolizei patrol vehicle, VW Passat, blue and silver German police livery, light bar, disaster response role", ["vehicle", "police"]),
    ("Bundespolizei Helicopter EC135", "Bundespolizei EC135 helicopter, blue and white police livery, search and rescue equipment, thermal camera", ["aircraft", "police", "rescue"]),
    ("Bundespolizei Water Cannon Truck", "Bundespolizei water cannon vehicle, MAN chassis, blue-silver livery, crowd control and fire suppression capability", ["vehicle", "police"]),
    ("Bundespolizei Command Vehicle", "Bundespolizei mobile command center, large van with communications mast, blue-silver, incident coordination", ["vehicle", "command", "police"]),
    ("Bundespolizei Rescue Boat", "Bundespolizei water rescue boat, rigid hull, blue-silver markings, patrol and rescue on rivers and lakes", ["rescue", "water", "police"]),
    ("Bundespolizei Communication Truck", "Bundespolizei communications relay truck, satellite and radio systems, blue-silver, disaster area network", ["vehicle", "police"]),
    ("Bundespolizei Helicopter EC155", "Bundespolizei EC155 helicopter, blue-silver, VIP transport and heavy lift rescue operations", ["aircraft", "police", "rescue"]),
    ("Bundespolizei Transporter Mercedes", "Bundespolizei Mercedes Sprinter transporter, blue-silver, personnel and equipment transport for relief", ["vehicle", "police"]),
    ("Bundespolizei Mobile Barricade System", "Bundespolizei portable barricade system, modular steel barriers, blue markings, disaster area perimeter control", ["infrastructure", "police"]),
    ("Bundespolizei Drone Unit", "Bundespolizei surveillance drone system, DJI Matrice with camera, blue case, aerial disaster assessment", ["drone", "police"]),
]

GER_FEUERWEHR_OBJECTS = [
    ("Feuerwehr Fire Engine HLF 20", "German Feuerwehr HLF 20 fire engine, Mercedes Atego, red with reflective stripes, combined rescue and fire suppression", ["vehicle", "fire", "rescue"]),
    ("Feuerwehr Aerial Ladder DLK 23", "Feuerwehr DLK 23 aerial ladder truck, Mercedes, red, 23-meter telescopic ladder with rescue basket", ["vehicle", "fire", "rescue"]),
    ("Feuerwehr Tanker TLF 2000", "Feuerwehr TLF 2000 water tanker, red, 2000 liter water and foam tank, wildfire and disaster response", ["vehicle", "fire", "water"]),
    ("Feuerwehr Rescue Truck RW", "Feuerwehr RW rescue truck, red, heavy rescue equipment including hydraulic cutters and spreaders", ["vehicle", "fire", "rescue"]),
    ("Feuerwehr Hazmat Vehicle GW-G", "Feuerwehr GW-Gefahrgut hazmat response vehicle, red, chemical spill and contamination response equipment", ["vehicle", "fire", "hazmat"]),
    ("Feuerwehr Pump Truck SW 2000", "Feuerwehr SW 2000 pump truck, red, high-capacity water pump system with 2000m of hose", ["vehicle", "fire", "water"]),
    ("Feuerwehr Command Car ELW", "Feuerwehr ELW command vehicle, red, mobile incident command post with communications", ["vehicle", "fire", "command"]),
    ("Feuerwehr Boat Fire Rescue", "Feuerwehr fire rescue boat, red with reflective markings, water pump and rescue equipment", ["rescue", "water", "fire"]),
    ("Feuerwehr Mobile Pump Unit", "Feuerwehr portable pump unit, trailer-mounted high-capacity pump for flood dewatering", ["water", "fire"]),
    ("Feuerwehr Lighting Trailer", "Feuerwehr mobile lighting trailer, generator-powered LED floodlights for nighttime operations", ["power", "fire"]),
    ("Feuerwehr Decon Shower Unit", "Feuerwehr decontamination shower unit, portable shower tent with water heating, hazmat decontamination", ["hazmat", "fire"]),
    ("Feuerwehr Rescue Saw System", "Feuerwehr emergency rescue saw system, portable concrete and metal cutting tools for urban search and rescue", ["rescue", "fire"]),
    ("Feuerwehr Ventilation Unit", "Feuerwehr mobile ventilation fan unit, positive pressure ventilation system for smoke clearing", ["fire"]),
    ("Feuerwehr Foam Tender", "Feuerwehr foam tender truck, carries foam concentrate for large-scale fire suppression", ["vehicle", "fire"]),
    ("Feuerwehr Wildfire Truck", "Feuerwehr specialized wildfire response truck, off-road capable, brush guards, water cannon", ["vehicle", "fire"]),
]

GER_ROBOTS_OBJECTS = [
    ("Rescue Ground Robot UGV", "tracked ground rescue robot UGV, rugged all-terrain vehicle with camera arm and gripper, urban search and rescue in collapsed buildings", ["robot", "rescue"]),
    ("Snake Robot Inspection System", "articulated snake robot for structural inspection, long flexible body with multiple joints and cameras, searches through rubble and narrow passages", ["robot", "rescue"]),
    ("Underwater ROV Rescue", "underwater remotely operated vehicle ROV, torpedo-shaped with thrusters and camera, underwater search and rescue operations", ["robot", "water", "rescue"]),
    ("Structural Inspection Robot", "wall-climbing structural inspection robot, magnetic tracks for steel structures, assesses damage after earthquakes", ["robot", "rescue"]),
    ("EOD Disposal Robot", "explosive ordnance disposal robot, tracked platform with precision robotic arm and camera, bomb disposal in disaster zones", ["robot", "rescue"]),
    ("Debris Clearing Robot", "heavy-duty debris clearing robot, bulldozer-style tracked robot with articulated arm, clears collapsed building rubble", ["robot", "heavy_machinery"]),
    ("Firefighting Robot LUF 60", "LUF 60 firefighting robot, tracked unmanned fire suppression vehicle with water cannon, enters dangerous fire zones", ["robot", "fire"]),
    ("Medical Evacuation Robot", "autonomous medical evacuation robot, tracked platform with stretcher bay, extracts casualties from dangerous areas", ["robot", "medical", "rescue"]),
    ("Pipe Inspection Crawler", "pipe inspection crawler robot, small tracked vehicle for sewer and pipeline inspection after floods and earthquakes", ["robot", "rescue"]),
    ("Chemical Detection Robot", "chemical hazard detection robot, tracked platform with gas sensors and spectrometer, maps contamination in disaster zones", ["robot", "hazmat"]),
    ("Aerial Rescue Drone Heavy", "heavy-lift aerial rescue drone, hexacopter with cargo winch, delivers medical supplies and extracts small items", ["robot", "drone", "rescue"]),
    ("Demolition Robot Brokk", "Brokk remote demolition robot, tracked with hydraulic breaker attachment, controlled building demolition after disasters", ["robot", "heavy_machinery"]),
    ("Logistics Transport Robot", "autonomous logistics transport robot, wheeled platform for moving supplies in disaster areas, follows GPS waypoints", ["robot", "supplies"]),
    ("Thermal Search Robot", "thermal imaging search robot, small tracked vehicle with FLIR camera, locates survivors in smoke and darkness", ["robot", "rescue"]),
]

GER_UAV_OBJECTS = [
    ("Cargo Delivery Drone", "heavy-lift cargo delivery drone, octocopter design with cargo pod, delivers medical supplies and food to isolated disaster areas", ["drone", "supplies"]),
    ("Surveillance Drone Fixed-Wing", "fixed-wing surveillance drone, long-endurance UAV with high-resolution camera, wide-area disaster damage assessment", ["drone", "surveillance"]),
    ("Medical Delivery Drone", "medical delivery drone, quadcopter with insulated medical supply pod, delivers blood, vaccines, and medication", ["drone", "medical"]),
    ("Mapping Drone LiDAR", "LiDAR mapping drone, hexacopter with laser scanner, creates 3D terrain maps of disaster zones for response planning", ["drone", "surveillance"]),
    ("Search and Rescue UAV", "search and rescue UAV, quadcopter with thermal camera and loudspeaker, locates and communicates with trapped survivors", ["drone", "rescue"]),
    ("Tethered Relay Drone", "tethered relay drone, quadcopter connected to ground power supply, provides continuous communications relay over disaster area", ["drone", "communications"]),
    ("Fire Detection Drone", "wildfire detection drone, fixed-wing UAV with infrared sensor, early fire detection and hotspot mapping", ["drone", "fire"]),
    ("Flood Monitoring Drone", "flood monitoring drone, quadcopter with multispectral camera, tracks water levels and flood progression", ["drone", "water"]),
    ("Infrastructure Inspection UAV", "infrastructure inspection UAV, small quadcopter with zoom camera, inspects bridges and buildings for structural damage", ["drone", "surveillance"]),
    ("Night Search Drone", "night search drone, quadcopter with high-intensity LED array and thermal camera, nighttime survivor location", ["drone", "rescue"]),
    ("Swarm Mapping Drone System", "drone swarm system, 5 small coordinated quadcopters for rapid parallel mapping of disaster zones", ["drone", "surveillance"]),
    ("Supply Airdrop Drone", "large supply airdrop drone, VTOL fixed-wing with parachute delivery system, drops supplies to inaccessible areas", ["drone", "supplies"]),
]

GER_INFRA_OBJECTS = [
    ("Temporary Shelter Container", "emergency temporary shelter container, converted shipping container with insulation, windows, and bunk beds, disaster housing", ["shelter", "infrastructure"]),
    ("Water Storage Tank 10000L", "emergency water storage tank, 10000 liter collapsible bladder tank, potable water reserve for disaster areas", ["water", "infrastructure"]),
    ("Emergency Power Generator", "emergency diesel power generator, containerized 100kVA unit, provides electricity to disaster relief camps", ["power", "infrastructure"]),
    ("Mobile Communications Mast", "mobile communications mast, telescopic antenna tower on trailer, restores cell and radio communications after disasters", ["communications", "infrastructure"]),
    ("Mobile Medical Clinic", "mobile medical clinic module, expandable container with examination room and pharmacy, disaster area healthcare", ["medical", "infrastructure"]),
    ("Decontamination Station", "portable decontamination station, shower and cleaning facility for hazmat response, modular tent system", ["hazmat", "infrastructure"]),
    ("Emergency Lighting System", "emergency area lighting system, portable LED tower lights on tripods, illuminates disaster response operations", ["power", "infrastructure"]),
    ("Temporary Latrine Block", "temporary latrine block, portable toilet facility with waste treatment, disaster area sanitation", ["infrastructure"]),
    ("Field Kitchen Tent", "large field kitchen tent, canvas tent with cooking equipment, prepares meals for disaster victims and responders", ["supplies", "infrastructure"]),
    ("Fuel Storage Bladder", "collapsible fuel storage bladder, 5000 liter portable diesel tank, fuel reserve for relief vehicles and generators", ["supplies", "infrastructure"]),
]

GER_EXTRA_VARIANTS = [
    "It was deployed during the Ahrweiler 2021 flood, with waterline staining visible on the lower body and emergency flashers still active.",
    "It is equipped for Oder river flood defense with a sandbag loader attachment on the front and a reinforced bumper guard.",
    "It is a Black Forest storm response variant, with a roof-mounted spotlight bar and a chainsaw holder mounted on the side.",
    "It is a North Sea coastal flood variant with corrosion-resistant zinc coating on all metalwork and marine-grade stainless steel fittings.",
    "It is an urban earthquake response variant with a dust filtration system on the roof and structural crack-monitoring sensors mounted on the frame.",
]


def generate_germany() -> list[dict]:
    all_items = []

    all_items.extend(build_items(
        "ger_drk", "drk", "germany", "DRK (German Red Cross)",
        GER_DRK_OBJECTS, 500, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_thw", "thw", "germany", "THW (Technisches Hilfswerk)",
        GER_THW_OBJECTS, 600, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_bw", "bundeswehr_relief", "germany", "Bundeswehr Disaster Relief",
        GER_BW_RELIEF_OBJECTS, 500, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_bp", "bundespolizei", "germany", "Bundespolizei",
        GER_POLIZEI_OBJECTS, 400, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_fw", "feuerwehr", "germany", "Feuerwehr (Fire Department)",
        GER_FEUERWEHR_OBJECTS, 600, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_bot", "rescue_robots", "germany", "German Rescue Robotics",
        GER_ROBOTS_OBJECTS, 500, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_uav", "uav", "germany", "German Relief UAVs",
        GER_UAV_OBJECTS, 500, GER_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ger_inf", "infrastructure", "germany", "German Relief Infrastructure",
        GER_INFRA_OBJECTS, 400, GER_EXTRA_VARIANTS
    ))

    return all_items



# ---------------------------------------------------------------------------
# EU (~2,500)
# ---------------------------------------------------------------------------

EU_CIVIL_OBJECTS = [
    ("rescEU Firefighting Aircraft", "rescEU Canadair CL-415 firefighting aircraft, yellow and blue EU markings, water bomber for wildfire response", ["aircraft", "fire", "eu"]),
    ("rescEU Medical Evacuation Jet", "rescEU medical evacuation jet, white with EU stars, mobile ICU interior, cross-border patient transport", ["aircraft", "medical", "eu"]),
    ("rescEU Water Pumping Unit", "rescEU high-capacity water pumping unit, trailer-mounted, EU blue markings, flood response across member states", ["water", "infrastructure", "eu"]),
    ("rescEU Field Hospital Module", "rescEU emergency field hospital, modular tent complex with EU flag, surgical and ICU capability", ["medical", "shelter", "eu"]),
    ("rescEU Shelter Kit Pallet", "rescEU emergency shelter kit, pallet of tents and bedding for displaced persons, EU markings", ["shelter", "supplies", "eu"]),
    ("rescEU Detection Equipment", "rescEU urban search and rescue detection equipment, acoustic and seismic sensors for locating survivors", ["rescue", "eu"]),
    ("rescEU Chemical Response Unit", "rescEU CBRN response unit, containerized chemical detection and decontamination system, EU markings", ["hazmat", "eu"]),
    ("rescEU Transport Truck", "rescEU logistics transport truck, EU blue with yellow stars, carries relief supplies across borders", ["vehicle", "supplies", "eu"]),
    ("rescEU Drone Fleet", "rescEU mapping drone fleet, set of quadcopters with cameras for disaster damage assessment, EU cases", ["drone", "eu"]),
    ("rescEU Mobile Lab", "rescEU mobile laboratory unit, containerized lab for water quality testing and disease surveillance", ["medical", "eu"]),
    ("rescEU Power Generator", "rescEU emergency power generator, containerized unit with distribution panels, EU markings", ["power", "eu"]),
    ("rescEU Communication Hub", "rescEU mobile communications hub, satellite uplink and radio relay system on trailer", ["communications", "eu"]),
]

EU_REDCROSS_OBJECTS = [
    ("French Red Cross Ambulance", "French Red Cross (Croix-Rouge) ambulance, white with red cross, French markings, equipped medical interior", ["medical", "vehicle", "red_cross", "france"]),
    ("Italian Red Cross Mobile Clinic", "Italian Red Cross (Croce Rossa) mobile clinic truck, white with red cross, Italian markings, medical examination room", ["medical", "vehicle", "red_cross", "italy"]),
    ("Dutch Red Cross Response Van", "Netherlands Red Cross (Rode Kruis) response van, white with red cross, Dutch markings, emergency supplies", ["vehicle", "supplies", "red_cross", "netherlands"]),
    ("Swedish Red Cross Field Unit", "Swedish Red Cross (Röda Korset) field response unit, white with red cross, Nordic markings, winter-equipped", ["vehicle", "red_cross", "sweden"]),
    ("Polish Red Cross Aid Truck", "Polish Red Cross (PCK) aid distribution truck, white with red cross, Polish markings, supplies transport", ["vehicle", "supplies", "red_cross", "poland"]),
    ("Spanish Red Cross Ambulance", "Spanish Red Cross (Cruz Roja) ambulance, white with red cross, Spanish markings, advanced life support", ["vehicle", "medical", "red_cross", "spain"]),
    ("Austrian Red Cross Rescue Vehicle", "Austrian Red Cross (Rotes Kreuz) rescue vehicle, white with red cross, Austrian markings, mountain rescue equipped", ["vehicle", "rescue", "red_cross", "austria"]),
    ("Belgian Red Cross Response Unit", "Belgian Red Cross (Rode Kruis) emergency response unit, white with red cross, Belgian markings", ["vehicle", "red_cross", "belgium"]),
    ("Finnish Red Cross Aid Module", "Finnish Red Cross (Punainen Risti) modular aid station, containerized distribution center, Finnish markings", ["supplies", "red_cross", "finland"]),
    ("Danish Red Cross Mobile Unit", "Danish Red Cross (Røde Kors) mobile response unit, white van with red cross, Danish markings", ["vehicle", "red_cross", "denmark"]),
    ("Red Cross Blood Mobile EU", "European Red Cross bloodmobile, white with red cross and EU stars, mobile blood donation and testing", ["medical", "vehicle", "red_cross"]),
    ("Red Cross Water Unit", "European Red Cross water and sanitation unit, portable water treatment system, red cross markings", ["water", "red_cross"]),
]

EU_ECHO_OBJECTS = [
    ("ECHO Supply Airdrop Package", "EU ECHO humanitarian supply airdrop package, palletized with parachute, food and medical supplies for isolated communities", ["supplies", "eu"]),
    ("ECHO Water Sanitation Kit", "ECHO water and sanitation kit, portable water treatment and distribution system for humanitarian emergencies", ["water", "eu"]),
    ("ECHO Food Distribution Point", "ECHO food distribution center module, tent with tables and scales, organized food rationing for displaced populations", ["supplies", "eu"]),
    ("ECHO Shelter Materials Pallet", "ECHO emergency shelter materials pallet, tarps, rope, tools, and fixings for temporary housing", ["shelter", "supplies", "eu"]),
    ("ECHO Telecoms Emergency Kit", "ECHO emergency telecommunications kit, portable satellite terminal and radio set for disaster area communications", ["communications", "eu"]),
    ("ECHO Medical Supply Container", "ECHO medical supply container, temperature-controlled, pre-positioned medicines and medical devices", ["medical", "supplies", "eu"]),
    ("ECHO Winterization Kit", "ECHO winterization kit pallet, insulated blankets, heating equipment, warm clothing for cold-climate emergencies", ["supplies", "shelter", "eu"]),
    ("ECHO Education Kit", "ECHO education in emergencies kit, portable classroom materials, school supplies, and temporary learning space tent", ["supplies", "eu"]),
    ("ECHO Protection Kit", "ECHO protection and GBV response kit, safe space tent with support materials for vulnerable populations", ["supplies", "eu"]),
    ("ECHO Cash Transfer Module", "ECHO cash and voucher assistance module, mobile registration and distribution system for humanitarian cash transfers", ["supplies", "eu"]),
]

EU_MEMBER_STATE_OBJECTS = [
    ("France Securite Civile Helicopter", "France Securite Civile EC225 helicopter, red and yellow, search and rescue and disaster response", ["aircraft", "rescue", "france"]),
    ("France Securite Civile Fire Truck", "France Securite Civile wildfire response truck, red, off-road, water cannon and foam system", ["vehicle", "fire", "france"]),
    ("Italy Protezione Civile Boat", "Italy Protezione Civile rescue boat, blue and white, Mediterranean disaster response", ["rescue", "water", "italy"]),
    ("Italy Protezione Civile Mobile Hospital", "Italy Protezione Civile mobile hospital, expandable container, medical and surgical capability", ["medical", "shelter", "italy"]),
    ("Netherlands Flood Barrier", "Netherlands mobile flood barrier system, modular aluminum water barrier, rapid deployment flood defense", ["infrastructure", "water", "netherlands"]),
    ("Netherlands Dewatering Pump", "Netherlands high-capacity dewatering pump unit, submersible pump on trailer, Dutch engineering", ["water", "infrastructure", "netherlands"]),
    ("Sweden MSB Fire Truck", "Sweden MSB (Myndigheten för samhällsskydd) forest fire truck, red, off-road capable, Nordic winter-ready", ["vehicle", "fire", "sweden"]),
    ("Sweden MSB Rescue Boat", "Sweden MSB ice rescue boat, orange, operates in Baltic ice conditions, cold water rescue", ["rescue", "water", "sweden"]),
    ("Poland Fire Brigade Unit", "Poland Państwowa Straż Pożarna fire and rescue unit, red, urban search and rescue equipped", ["vehicle", "fire", "rescue", "poland"]),
    ("Spain UME Fire Truck", "Spain UME (Unidad Militar de Emergencias) fire truck, green and red, military emergency unit, wildfire response", ["vehicle", "fire", "spain"]),
    ("Spain UME Helicopter", "Spain UME emergency helicopter, green, rescue hoist and water bucket for wildfire operations", ["aircraft", "fire", "rescue", "spain"]),
    ("Austria Bergrettung Vehicle", "Austria Bergrettung mountain rescue vehicle, white and red, specialized alpine rescue equipment", ["vehicle", "rescue", "austria"]),
    ("Finland VSR Rescue Vehicle", "Finland Vapaaehtoinen pelastuspalvelu volunteer rescue vehicle, orange, Nordic forest rescue", ["vehicle", "rescue", "finland"]),
    ("Denmark Emergency Management Vehicle", "Denmark Beredskabsstyrelsen emergency vehicle, red and white, Danish emergency management agency", ["vehicle", "rescue", "denmark"]),
    ("Belgium Civil Protection Truck", "Belgium Civil Protection heavy rescue truck, orange, urban search and rescue and CBRN response", ["vehicle", "rescue", "belgium"]),
]

EU_SPECIAL_OBJECTS = [
    ("Copernicus EMS Satellite Terminal", "Copernicus Emergency Management Service satellite ground terminal, portable antenna system, receives satellite imagery for disaster mapping", ["communications", "eu"]),
    ("ERCC Coordination Center Module", "EU Emergency Response Coordination Centre (ERCC) mobile module, command container with video walls and communications", ["command", "eu"]),
    ("rescEU Medical Stockpile Container", "rescEU medical stockpile container, pre-positioned pandemic response supplies, ventilators and PPE", ["medical", "supplies", "eu"]),
    ("EU CBRN Detection Vehicle", "EU CBRN detection vehicle, equipped with chemical biological radiological nuclear sensors, mobile laboratory", ["hazmat", "vehicle", "eu"]),
    ("EU Border Aid Station", "EU border humanitarian aid station, tent complex for refugee reception, medical screening and supplies", ["supplies", "shelter", "eu"]),
    ("EU Disaster Assessment Drone", "EU Copernicus disaster assessment drone, fixed-wing UAV with multispectral camera, damage mapping", ["drone", "eu"]),
    ("EU Emergency Radio Network", "EU emergency radio network base station, portable VHF/UHF repeater system for disaster communications", ["communications", "eu"]),
    ("EU Water Testing Mobile Lab", "EU mobile water quality testing laboratory, containerized lab with analysis equipment", ["water", "eu"]),
]

EU_EXTRA_VARIANTS = [
    "It is a Mediterranean earthquake variant with dust-resistant air filters and reinforced suspension built for rubble terrain.",
    "It is an Iberian wildfire variant with orange heat-reflective coating on the roof and ember-proof mesh on all air intakes.",
    "It is a Central European flood variant with sealed electrical compartments and a raised exhaust pipe for deep wading.",
    "It is a Nordic winter storm variant with a heated windshield, studded winter tires, and reflective yellow safety striping along the body.",
    "It is a Balkan disaster relief variant with UN/EU blue markings and multilingual signage panels on both sides.",
    "It is a Baltic coastal emergency variant with anti-corrosion treatment on all metalwork and amphibious capability modifications.",
]


def generate_eu() -> list[dict]:
    all_items = []

    all_items.extend(build_items(
        "eu_cp", "civil_protection", "eu", "EU Civil Protection / rescEU",
        EU_CIVIL_OBJECTS, 600, EU_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "eu_rc", "red_cross_eu", "eu", "European Red Cross Societies",
        EU_REDCROSS_OBJECTS, 500, EU_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "eu_echo", "echo", "eu", "ECHO Humanitarian Aid",
        EU_ECHO_OBJECTS, 500, EU_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "eu_ms", "member_state", "eu", "EU Member State Relief",
        EU_MEMBER_STATE_OBJECTS, 600, EU_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "eu_sp", "specialized", "eu", "EU Specialized Equipment",
        EU_SPECIAL_OBJECTS, 300, EU_EXTRA_VARIANTS
    ))

    return all_items


# ---------------------------------------------------------------------------
# UKRAINE (~2,500)
# ---------------------------------------------------------------------------

UA_RECOVERY_OBJECTS = [
    ("Debris Removal Excavator", "heavy excavator for post-conflict debris removal, yellow with protective cab, clearing collapsed buildings in Ukraine", ["heavy_machinery", "recovery", "ukraine"]),
    ("Demolition Crane Truck", "mobile demolition crane truck, telescopic boom with wrecking ball and grapple, urban clearing operations", ["heavy_machinery", "recovery", "ukraine"]),
    ("Concrete Crusher Machine", "mobile concrete crusher, jaw crusher on tracked chassis, recycles building rubble for reconstruction", ["heavy_machinery", "recovery", "ukraine"]),
    ("Waste Management Truck", "waste management and debris haulage truck, tipper body, transports rubble from conflict zones", ["vehicle", "recovery", "ukraine"]),
    ("Skid Steer Loader", "compact skid steer loader with multiple attachments, clears narrow urban passages of debris", ["heavy_machinery", "recovery", "ukraine"]),
    ("Front Loader Bulldozer", "wheel loader front-end loader, large bucket for moving debris and earth in reconstruction areas", ["heavy_machinery", "recovery", "ukraine"]),
    ("Rubble Sorting Conveyor", "portable rubble sorting conveyor system, separates debris by material type for recycling and disposal", ["infrastructure", "recovery", "ukraine"]),
    ("Demolition Robot", "remote-controlled demolition robot, tracked with hydraulic breaker, safely demolishes damaged structures", ["robot", "recovery", "ukraine"]),
    ("Crane Truck Heavy", "heavy crane truck, Liebherr-type mobile crane, lifts large structural elements during cleanup", ["heavy_machinery", "recovery", "ukraine"]),
    ("Dump Truck Fleet", "fleet dump truck, heavy payload capacity, hauls construction debris from urban recovery sites", ["vehicle", "recovery", "ukraine"]),
]

UA_REBUILD_OBJECTS = [
    ("Modular Housing Unit", "prefabricated modular housing unit, single-story container-based dwelling with insulation and utilities, temporary housing for displaced families", ["shelter", "rebuilding", "ukraine"]),
    ("Prefab School Module", "prefabricated school classroom module, container-based with desks and heating, restores education in conflict areas", ["infrastructure", "rebuilding", "ukraine"]),
    ("Temporary Power Station", "mobile temporary power station, containerized diesel generator with switchgear, restores electricity to neighborhoods", ["power", "infrastructure", "ukraine"]),
    ("Water Treatment Plant Mobile", "mobile water treatment plant, containerized system with filtration and chlorination, restores clean water supply", ["water", "infrastructure", "ukraine"]),
    ("Construction Crane Tower", "tower crane for reconstruction projects, fixed-base lattice crane, rebuilding residential areas", ["heavy_machinery", "rebuilding", "ukraine"]),
    ("Concrete Mixer Truck", "concrete mixer truck, rotating drum, delivers ready-mix concrete for reconstruction projects", ["vehicle", "rebuilding", "ukraine"]),
    ("Portable Bridge Section", "portable emergency bridge section, prefabricated steel bridge for restoring road connections", ["infrastructure", "rebuilding", "ukraine"]),
    ("Heating Station Mobile", "mobile district heating station, boiler unit on trailer, provides heat to buildings during winter recovery", ["power", "infrastructure", "ukraine"]),
    ("Pipe Laying Machine", "trenching and pipe laying machine, restores water and sewage infrastructure in damaged areas", ["heavy_machinery", "rebuilding", "ukraine"]),
    ("Electrical Repair Truck", "electrical grid repair truck, bucket truck with transformer and cable equipment, restores power lines", ["vehicle", "power", "rebuilding", "ukraine"]),
    ("Window Repair Unit", "mobile window and glazing repair unit, van with glass stocks and tools, repairs buildings damaged by conflict", ["vehicle", "rebuilding", "ukraine"]),
    ("Roofing Repair Kit", "emergency roofing repair kit, pallet of tarps, plywood, and fasteners for damaged building roof sealing", ["supplies", "rebuilding", "ukraine"]),
]

UA_HUMANITARIAN_OBJECTS = [
    ("Aid Distribution Center Tent", "large humanitarian aid distribution center tent, organized shelving and counter system, distributes food and supplies to displaced populations", ["supplies", "shelter", "ukraine"]),
    ("Food Convoy Truck", "humanitarian food convoy truck, box truck with refrigeration, delivers food supplies to conflict-affected areas", ["vehicle", "supplies", "ukraine"]),
    ("Mobile Medical Clinic Ukraine", "mobile medical clinic container, examination and pharmacy modules, serves conflict-affected communities", ["medical", "ukraine"]),
    ("Winterization Supply Pallet", "winterization supply pallet, insulated blankets, warm clothing, portable heaters for Ukrainian winter relief", ["supplies", "ukraine"]),
    ("Generator Shipment Container", "container of portable generators, multiple small diesel generators for household power restoration", ["power", "supplies", "ukraine"]),
    ("Medical Supply Truck", "medical supply distribution truck, temperature-controlled, delivers medicines and medical equipment to hospitals", ["vehicle", "medical", "supplies", "ukraine"]),
    ("Water Distribution Tanker", "water distribution tanker truck, delivers potable water to areas with damaged water infrastructure", ["vehicle", "water", "ukraine"]),
    ("Blanket and Bedding Pallet", "pallet of emergency blankets and bedding, shrink-wrapped, thermal blankets and sleeping bags for displaced persons", ["supplies", "ukraine"]),
    ("Hygiene Kit Pallet", "hygiene kit pallet, boxes of soap, sanitizer, menstrual products, and basic hygiene supplies", ["supplies", "ukraine"]),
    ("Psychosocial Support Tent", "psychosocial support tent, safe space for counseling and mental health support, conflict trauma care", ["medical", "shelter", "ukraine"]),
]

UA_DEMINING_OBJECTS = [
    ("DOK-ING Demining Vehicle", "DOK-ING MV-4 mine clearing vehicle, tracked armored machine with flail and tiller, mechanical demining in Ukrainian fields", ["vehicle", "demining", "ukraine"]),
    ("Pearson Mine Roller", "Pearson Engineering mine roller system, front-mounted on armored vehicle, triggers pressure-activated mines safely", ["vehicle", "demining", "ukraine"]),
    ("Handheld Mine Detector", "handheld metal detector for landmine detection, Schiebel AN19/2 type, operator sweeps ground for buried mines", ["demining", "ukraine"]),
    ("Mechanical Demining Machine", "mechanical demining machine, tracked vehicle with rotating flail chains, clears vegetation and detonates mines", ["heavy_machinery", "demining", "ukraine"]),
    ("EOD Robot Ukraine", "EOD disposal robot for unexploded ordnance, tracked platform with precision arm, safe disposal of UXO in Ukraine", ["robot", "demining", "ukraine"]),
    ("Demining Survey Drone", "demining survey drone, quadcopter with metal detection sensor and camera, aerial survey of suspected mine areas", ["drone", "demining", "ukraine"]),
    ("Mine Marking System", "mine marking and fencing system, portable signs and tape dispensers for marking hazardous areas", ["demining", "ukraine"]),
    ("Armored Demining Tractor", "armored tractor with mine plow attachment, V-hull protection, clears agricultural land of mines", ["vehicle", "demining", "ukraine"]),
    ("Ground Penetrating Radar Unit", "ground penetrating radar survey unit, cart-mounted GPR for detecting buried ordnance and mines", ["demining", "ukraine"]),
    ("Demining Command Vehicle", "demining operations command vehicle, van with mapping and coordination equipment, manages demining teams", ["vehicle", "command", "demining", "ukraine"]),
]

UA_EMERGENCY_OBJECTS = [
    ("DSNS Fire Truck Ukraine", "DSNS (State Emergency Service of Ukraine) fire truck, red and white, Ukrainian emergency services, urban firefighting", ["vehicle", "fire", "ukraine"]),
    ("DSNS Rescue Vehicle", "DSNS urban search and rescue vehicle, orange and white, hydraulic rescue tools and detection equipment", ["vehicle", "rescue", "ukraine"]),
    ("Ukrainian Red Cross Ambulance", "Ukrainian Red Cross ambulance, white with red cross, medical response in conflict zones", ["vehicle", "medical", "red_cross", "ukraine"]),
    ("Ukraine National Guard Relief Truck", "Ukraine National Guard logistics and relief truck, green, delivers supplies to front-line communities", ["vehicle", "supplies", "ukraine"]),
    ("DSNS Demining Team Vehicle", "DSNS pyrotechnic team vehicle, orange van with detection equipment, UXO and mine clearance operations", ["vehicle", "demining", "ukraine"]),
    ("Ukrainian Volunteer Aid Van", "Ukrainian volunteer organization aid van, civilian vehicle converted for aid delivery, grassroots relief effort", ["vehicle", "supplies", "ukraine"]),
    ("DSNS Chemical Response Unit", "DSNS chemical and radiological response unit, detection equipment and protective suits, CBRN monitoring", ["hazmat", "vehicle", "ukraine"]),
    ("Ukrainian Ambulance Fleet", "Ukrainian emergency ambulance, Ford Transit type, marked with Ukrainian medical symbols, frontline medical response", ["vehicle", "medical", "ukraine"]),
    ("DSNS Pontoon Bridge Unit", "DSNS emergency pontoon bridge unit, portable bridge system for restoring crossings destroyed in conflict", ["infrastructure", "ukraine"]),
    ("Ukrainian Drone Recon Unit", "Ukrainian emergency services reconnaissance drone, quadcopter for damage assessment and survivor search", ["drone", "rescue", "ukraine"]),
]

UA_EXTRA_VARIANTS = [
    "It operates in Kyiv Oblast, bearing the Ukrainian trident emblem on the doors and sand-colored camouflage netting draped over the roof.",
    "It is a Kharkiv reconstruction variant with blast-resistant window film and reinforced cab protection plates bolted to the frame.",
    "It operates in the Donbas region with dust storm protection covers over all openings and a desert tan paint scheme.",
    "It is a Kherson flood recovery variant with waterproofed electrical systems and a bilge pump mounted on the deck.",
    "It operates in the Zaporizhzhia area with a radiation monitoring sensor pod on the roof and an anti-contamination wash-down system.",
    "It is an Odesa region humanitarian variant with multilingual aid signage and a refrigerated compartment for temperature-sensitive medicines.",
    "It is a Chernihiv deconfliction variant with high-visibility white paint and blue UN-style markings for safe passage identification.",
]


def generate_ukraine() -> list[dict]:
    all_items = []

    all_items.extend(build_items(
        "ua_rec", "recovery", "ukraine", "Conflict Recovery / Debris Removal",
        UA_RECOVERY_OBJECTS, 500, UA_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ua_rbl", "rebuilding", "ukraine", "Rebuilding / Reconstruction",
        UA_REBUILD_OBJECTS, 600, UA_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ua_hum", "humanitarian", "ukraine", "Humanitarian Aid / Convoys",
        UA_HUMANITARIAN_OBJECTS, 500, UA_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ua_mine", "demining", "ukraine", "Demining / EOD",
        UA_DEMINING_OBJECTS, 500, UA_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "ua_em", "emergency_services", "ukraine", "DSNS / Emergency Services",
        UA_EMERGENCY_OBJECTS, 400, UA_EXTRA_VARIANTS
    ))

    return all_items


# ---------------------------------------------------------------------------
# GENERAL (~1,000)
# ---------------------------------------------------------------------------

GEN_DISASTER_OBJECTS = [
    ("Earthquake Rubble Pile", "collapsed building rubble pile, concrete debris with exposed rebar, earthquake disaster scene", ["disaster_scene", "earthquake"]),
    ("Flood Barrier Wall", "modular flood barrier wall, aluminum panels and sandbags, urban flood defense system", ["infrastructure", "flood"]),
    ("Wildfire Equipment Cache", "wildfire firefighting equipment cache, hoses, pumps, Pulaski tools, and fire shelters staged for deployment", ["fire", "supplies"]),
    ("Hurricane Response Kit", "hurricane emergency response kit, plywood boards, tarps, generators, and supplies on pallet", ["supplies", "storm"]),
    ("Tsunami Warning Siren", "tsunami warning siren pole, solar-powered with battery backup, coastal early warning system", ["infrastructure", "tsunami"]),
    ("Landslide Mitigation Barrier", "landslide mitigation barrier, steel mesh and gabion wall, slope stabilization system", ["infrastructure", "landslide"]),
    ("Flood Pump Station", "emergency flood pump station, multiple submersible pumps with discharge hoses, urban flood drainage", ["water", "infrastructure"]),
    ("Earthquake Sensor Station", "seismic monitoring station, ground sensor with communications mast, early earthquake warning network", ["infrastructure", "earthquake"]),
    ("Wildfire Watchtower", "portable wildfire watchtower, telescopic observation platform with camera and radio", ["infrastructure", "fire"]),
    ("Storm Surge Barrier Gate", "portable storm surge barrier gate, modular flood gate system for coastal protection", ["infrastructure", "storm"]),
]

GEN_MEDICAL_OBJECTS = [
    ("Field Hospital Tent Large", "large field hospital tent, multi-ward medical facility with operating theater and pharmacy, disaster area healthcare", ["medical", "shelter"]),
    ("Triage System Module", "emergency triage system, color-coded tent sections with stretchers and medical assessment equipment", ["medical"]),
    ("Portable X-Ray Unit", "portable digital X-ray machine, battery-powered with digital detector, field diagnostic imaging", ["medical"]),
    ("Oxygen Concentrator Unit", "portable oxygen concentrator station, multiple units with distribution manifold, field respiratory support", ["medical"]),
    ("Vaccine Cold Chain Unit", "vaccine cold chain storage unit, solar-powered refrigerator with temperature monitoring, maintains vaccine supply", ["medical"]),
    ("Surgical Kit Field", "field surgical instrument kit, stainless steel tray with complete surgical set, emergency operations", ["medical"]),
    ("Dialysis Field Unit", "portable dialysis machine, battery-operated with water purification, field renal replacement therapy", ["medical"]),
    ("ECMO Transport Unit", "transportable ECMO life support unit, mobile extracorporeal membrane oxygenation for critical patients", ["medical"]),
    ("Lab Blood Analysis Unit", "portable blood analysis laboratory, compact analyzer for blood typing and basic panels, field diagnostics", ["medical"]),
    ("Dental Emergency Kit", "emergency dental treatment kit, portable dental chair and instruments, field dental care", ["medical"]),
]

GEN_SHELTER_OBJECTS = [
    ("UNHCR Family Tent", "UNHCR standard family tent, white canvas with blue stripe, 16 square meters, refugee shelter", ["shelter"]),
    ("Container Housing Unit", "converted shipping container housing unit, insulated with windows and door, temporary residential", ["shelter", "infrastructure"]),
    ("Inflatable Emergency Shelter", "inflatable emergency shelter, rapidly deployable air-beam tent structure, disaster housing", ["shelter"]),
    ("Emergency Blanket Distribution", "emergency thermal blanket distribution point, pallets of silver mylar blankets, mass distribution", ["supplies", "shelter"]),
    ("Camp Infrastructure Kit", "refugee camp infrastructure kit, tent frames, ground sheets, drainage pipes, and latrine components", ["shelter", "infrastructure"]),
    ("Winter Tent Insulated", "insulated winter tent, heavy-duty canvas with thermal lining and stove port, cold-weather shelter", ["shelter"]),
    ("Transitional Shelter Frame", "transitional shelter frame kit, timber and steel frame with corrugated metal roofing, semi-permanent housing", ["shelter", "infrastructure"]),
    ("Communal Kitchen Shelter", "communal kitchen shelter structure, large tent with cooking facilities for camp food preparation", ["shelter", "supplies"]),
]

GEN_EQUIPMENT_OBJECTS = [
    ("Water Purification Unit Portable", "portable water purification unit, backpack-size filtration system with UV treatment, personal water safety", ["water"]),
    ("Generator Diesel 5kVA", "portable diesel generator, 5kVA single-cylinder unit, household emergency power", ["power"]),
    ("Satellite Phone Terminal", "satellite phone terminal, portable BGAN terminal with data and voice, disaster area communications", ["communications"]),
    ("Portable Bridge Aluminum", "portable aluminum bridge section, lightweight pedestrian bridge for emergency crossings", ["infrastructure"]),
    ("Searchlight Portable LED", "portable LED searchlight, battery-powered with tripod, high-intensity area illumination", ["power"]),
    ("Sandbag Machine Portable", "portable sandbag filling machine, hopper and conveyor system, rapid flood defense production", ["infrastructure", "water"]),
    ("Portable Dam Bladder", "portable water-filled dam bladder, long inflatable tube for rapid flood barrier deployment", ["water", "infrastructure"]),
    ("Emergency Radio Handheld", "emergency handheld radio set, waterproof VHF transceiver for disaster response coordination", ["communications"]),
    ("Portable Solar Panel Array", "portable solar panel array, foldable panels with battery bank, off-grid power for relief operations", ["power"]),
    ("Emergency Siren PA System", "emergency siren and public address system, vehicle-mounted loudspeaker for evacuation warnings", ["communications"]),
]

GEN_EXTRA_VARIANTS = [
    "It is a Southeast Asia typhoon variant with tropical humidity-proof coating and bamboo reinforcement lashing points on the frame.",
    "It is a Caribbean hurricane variant with a Category 5 wind-rated anchoring system and impact-resistant polycarbonate panels.",
    "It is a South American earthquake variant with Spanish-language markings and high-altitude engine tuning for Andean operations.",
    "It is an African drought relief variant with solar-powered auxiliary systems on the roof and sand filtration on all moving parts.",
    "It is a Pacific island disaster variant with saltwater-resistant marine-grade hardware and outrigger stabilizer mounts.",
    "It is a Central Asian earthquake variant with extreme temperature insulation rated from minus 40 to plus 50 degrees Celsius and dustproof sealed electronics.",
    "It is a Middle East conflict relief variant with desert camouflage paint and blast-resistant laminated glass on all windows.",
]


def generate_general() -> list[dict]:
    all_items = []

    all_items.extend(build_items(
        "gen_dis", "natural_disaster", "general", "Natural Disaster Response",
        GEN_DISASTER_OBJECTS, 300, GEN_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "gen_med", "medical", "general", "Medical Supplies / Field Hospitals",
        GEN_MEDICAL_OBJECTS, 250, GEN_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "gen_shl", "shelter", "general", "Temporary Shelters",
        GEN_SHELTER_OBJECTS, 200, GEN_EXTRA_VARIANTS
    ))
    all_items.extend(build_items(
        "gen_equ", "equipment", "general", "Generic Relief Equipment",
        GEN_EQUIPMENT_OBJECTS, 250, GEN_EXTRA_VARIANTS
    ))

    return all_items


# ---------------------------------------------------------------------------
# Main: write manifests
# ---------------------------------------------------------------------------

def write_manifest(filepath: Path, items: list[dict], description: str, geography: str):
    # Source collections already integrated into the active Germany manifest
    # must survive regenerating its template-derived rows.
    if filepath.exists():
        previous = json.loads(filepath.read_text())
        present = {item["id"] for item in items}
        items = [*items, *[rewrite_item(item) for item in previous.get("meshes", [])
                          if item.get("id", "").startswith("legacy_") and item["id"] not in present]]
    items = [item if is_optimized_prompt(item.get("prompt", "")) else rewrite_item(item) for item in items]
    manifest = {
        "version": "2.0",
        "description": description,
        "geography": geography,
        "total_items": len(items),
        "generated_at": datetime.now().isoformat(),
        "generation_prompt_version": PROMPT_VERSION,
        "meshes": items,
    }
    write_json(filepath, manifest)
    print(f"  {filepath.name}: {len(items)} items")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate disaster relief mesh prompts")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for manifests")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Generating manifests to: {out}")

    print("\nGenerating Germany items...")
    germany = generate_germany()

    print("Generating EU items...")
    eu = generate_eu()

    print("Generating Ukraine items...")
    ukraine = generate_ukraine()

    print("Generating General items...")
    general = generate_general()

    total = len(germany) + len(eu) + len(ukraine) + len(general)

    print(f"\nWriting manifests (total: {total:,} items):")
    write_manifest(out / "relief_manifest_germany.json", germany,
                   "Disaster Relief - Germany: DRK, THW, Bundeswehr, Police, Fire, Robots, UAVs",
                   "germany")
    write_manifest(out / "relief_manifest_eu.json", eu,
                   "Disaster Relief - EU: Civil Protection, Red Cross, ECHO, Member States",
                   "eu")
    write_manifest(out / "relief_manifest_ukraine.json", ukraine,
                   "Disaster Relief - Ukraine: Recovery, Rebuilding, Humanitarian, Demining, Emergency",
                   "ukraine")
    write_manifest(out / "relief_manifest_general.json", general,
                   "Disaster Relief - General: Natural Disasters, Medical, Shelters, Equipment",
                   "general")

    print(f"\nDone! Total items: {total:,}")
    print(f"Manifest files in: {out}")


if __name__ == "__main__":
    main()
