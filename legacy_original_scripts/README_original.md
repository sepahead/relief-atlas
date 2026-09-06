---
viewer: false
license: mit
language:
  - en
tags:
  - 3d-meshes
  - military
  - defense
  - bundeswehr
  - simulation
  - glb
  - synthetic-data
  - drones
  - vehicles
pretty_name: Defense Meshes
size_categories:
  - n<1K
---

# Defense Meshes: 3D Military & Defense Asset Dataset

A curated collection of **58 3D military meshes** (GLB format) covering Bundeswehr equipment, military vehicles, drones, weapons, and tactical gear. Designed for defense simulation environments, military training simulators, and robotics research.

Meshes are generated via text-to-image and image-to-3D pipelines. Text-to-image models include **Flux Schnell** and **GPT-Image-1.5**; image-to-3D models include **Trellis**, **Trellis 2** (primary), and **Hunyuan3D v3** (fallback). All 58 meshes include reference images.

> **Status**: Actively expanding. 22 additional meshes are defined in manifests and queued for generation. Target: 200+ defense assets.

## Dataset Overview

| # | Category | Count | Description |
|---|----------|------:|-------------|
| 1 | **bundeswehr_gear** | 8 | Tactical equipment: plate carrier, Flecktarn backpack, SEM 70 radio, MG5 machine gun, Gefechtshelm M92 helmet, field gear |
| 2 | **bundeswehr_aircraft** | 6 | Aircraft: NH90 TTH, Tiger UHT helicopter, Eurofighter Typhoon, Panavia Tornado IDS, CH-53G Stallion |
| 3 | **bundeswehr_airdefense** | 6 | Air defense systems: Patriot launcher, Ozelot/Wiesel 2, TRML-4D radar, IRIS-T SLM, MANTIS turret |
| 4 | **bundeswehr_vehicles** | 6 | Armored vehicles: Leopard 2A7 MBT, PzH 2000 SPH, Dingo 2 MRAP, Puma IFV, Boxer GTK |
| 5 | **weapon_modern** | 6 | Modern weapons: P8 pistol, G36 rifle, Panzerfaust 3, MP7 PDW, MG5 machine gun |
| 6 | **military_drone** | 5 | UAVs: Heron TP, ALADIN hand-launched, Vector VTOL, Luna NG, Mikado quad |
| 7 | **military_gear** | 5 | Tactical equipment: Vector binoculars, combat helmet, SEM 52 radio, Flecktarn backpack, combat boots |
| 8 | **military_vehicle** | 5 | Combat vehicles: PzH 2000 artillery, Boxer APC, Puma IFV, Leopard 2A7, Fennek recon |
| 9 | **weapon_cold** | 4 | Edged weapons: bayonet, KM2000 combat knife, trench shovel, tactical tomahawk |
| 10 | **bundeswehr_infantry_heer** | 3 | Heer (Army) infantry: Jäger light infantry, Recon scout, Gebirgsjäger alpine |
| 11 | **bundeswehr_logistics** | 3 | Logistics: field kitchen TFK250, Unimog U1300L, MAN KAT1 8×8 |
| 12 | **bundeswehr_infantry_luftwaffe** | 1 | Luftwaffe: ground crew technician |
| | **TOTAL** | **58** | |

### Planned Expansions (22 meshes in queue)

| Category | Planned | Description |
|----------|--------:|-------------|
| Infantry (Heer) | 4 | Fallschirmjäger paratrooper, KSK operator, Panzer crewman, Panzergrenadier |
| Infantry (Luftwaffe) | 2 | Eurofighter pilot, Objektschutz soldier |
| Marine / Navy | 4 | Combat swimmer (KSM), deck crew sailor, Marine officer (dress), Seebataillon boarding |
| Naval vessels | 3 | Corvette Braunschweig, Frigate Baden-Württemberg, Submarine Type 212A |
| Medical (Sanität) | 2 | Combat medic, Medical officer |
| Ceremonial / Cyber | 2 | Cyber Ops soldier, EloKa specialist |
| Logistics | 2 | Faltstrasse mat, 20L jerrycan |
| SKB / Guard | 3 | Feldjäger MP, logistics soldier, Wachbataillon guard |

## Dataset Structure

| File / Directory | Description |
|---|---|
| `README.md` | This file |
| `manifests/` | Mesh generation manifests with prompts and metadata |
| `scripts/` | Generation and management scripts |
| `outputs/bundeswehr/` | 33 Bundeswehr-specific meshes |
| `outputs/warzone/` | 25 tactical/military meshes |

Each mesh directory contains:
- `<mesh_name>.glb` — 3D mesh in GLB format (glTF 2.0 Binary)
- `<mesh_name>_reference.png` — Reference image used during generation

## Mesh Properties

| Property | Value |
|----------|-------|
| **Format** | GLB (glTF 2.0 Binary) |
| **Coordinate System** | Y-up, right-handed |
| **Scale** | Unit-normalized (~1 unit bounding box) |
| **Reference Images** | 58 of 58 meshes (100%) |

## Quick Start

```python
import trimesh
import json

# Load a single mesh
mesh = trimesh.load("outputs/bundeswehr/bundeswehr_vehicles/vehicle_Leopard_2A7/vehicle_Leopard_2A7.glb")
print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

# Load a manifest for batch processing
with open("manifests/mesh_manifest_bundeswehr.json") as f:
    manifest = json.load(f)

for item in manifest[:5]:
    print(f"  {item['id']}: {item.get('category', '?')}")
```

## Use Cases

1. **Military Simulation** — Training environments, wargaming, tactical planning
2. **Defense Robotics** — UGV/UAV interaction with military equipment
3. **VR/AR Training** — Immersive military equipment familiarization
4. **Synthetic Data** — Domain randomization for object detection and recognition
5. **Academic Research** — Defense-related computer vision and simulation

## Citation

```bibtex
@misc{defense-meshes-2026,
  author = {Mahmoudian, Sepehr},
  title = {Defense Meshes: A 3D Military Asset Dataset for Simulation},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Datasets},
  howpublished = {\url{https://huggingface.co/datasets/torusprime/defense-meshes}}
}
```

## License

This dataset is released under the **MIT License**.

## Acknowledgments

- 3D meshes generated using text-to-image models (Flux Schnell, GPT-Image-1.5) and image-to-3D models (Trellis, Trellis 2, Hunyuan3D v3)
- Bundeswehr equipment references based on publicly available specifications

## Links

- **Author**: [github.com/sepahead](https://github.com/sepahead)
- **Issues**: [github.com/sepahead/defense-meshes/issues](https://github.com/sepahead/defense-meshes/issues)
