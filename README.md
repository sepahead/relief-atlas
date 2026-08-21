# relief-atlas

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img src="assets/logo-light.svg" width="180" alt="RELIEF-ATLAS logo: an indexed contact sheet presents varied 3D mesh records over terrain contours.">
  </picture>
</p>

**10,000+ 3D mesh assets for disaster relief, humanitarian aid, and civil protection.**

AI-generated polygonal meshes (GLB) for use in robotics simulation, embodied AI training, and disaster response planning. Covers equipment and vehicles from Germany (DRK, THW, Feuerwehr, Bundeswehr), EU Civil Protection, Ukraine recovery operations, and global natural disaster response.

## Dataset

- **10,079 items** across 4 geography manifests
- Original generation: GPT-Image → Trellis 2 / Tripo v3.1 / Hunyuan3D pipeline (cloud)
- Local generation: FLUX.2 [klein] 4B → TRELLIS.2-4B on Apple Silicon (see below)
- Prompts optimized for image-to-3D: natural language, camera specifications, explicit constraints

## Geography

| Region | Items | Organizations |
|--------|-------|---------------|
| Germany | 4,079 | DRK, THW, Feuerwehr, Bundespolizei, Bundeswehr |
| EU | 2,500 | ECHO, Civil Protection, Red Cross, Member States |
| Ukraine | 2,500 | Conflict recovery, humanitarian aid, rebuilding |
| General | 1,000 | Global disaster response, natural disasters |

## Structure

```
scripts/              — Generation pipeline (prompts, mesh generation, verification)
scripts/local/        — On-device pipeline (FLUX.2 klein -> TRELLIS.2, mesh + 3DGS)
manifests/            — Item definitions with pre-generated prompts
outputs_relief/       — Generated 3D meshes (GLB + PNG + metadata)
legacy_original/      — Pre-existing meshes from earlier generation runs
legacy_original_scripts/ — Earlier generation scripts (reference)
config/               — API key configuration
```

## Setup

```bash
pip install -r requirements.txt
cp config/api_keys.example.txt config/api_keys.txt
# Edit config/api_keys.txt with your fal.ai and Runware keys
```

## Usage

```bash
# Verify setup
python scripts/generate_relief.py --dry-run

# Generate meshes (all providers)
python scripts/generate_relief.py

# Use only one provider
python scripts/generate_relief.py --provider fal
python scripts/generate_relief.py --provider runware

# Check progress
python scripts/verify_outputs.py
```

## Technical Details

- **Image generation**: GPT-Image-1.5/2 with structured natural language prompts
- **3D generation**: Trellis 2 (4B), Tripo v3.1, Hunyuan3D 3.1-Rapid
- **Trellis 2 settings**: `ss_guidance_strength: 8.0`, `resolution: 1024`, `textureSize: 2048`
- **Tripo v3.1 settings**: `geometryQuality: detailed`, `pbr: true`, `imageAutoFix: true`
- **Output**: GLB with PBR textures, reference PNG, metadata JSON

## Local generation pipeline (on-device)

The same 10,079 manifest items can be regenerated **entirely on-device** on
Apple Silicon (developed on M4 Max, 128GB) — no API keys, no cloud calls:

| Stage | Model | Runtime |
|-------|-------|---------|
| Text-to-image | FLUX.2 [klein] 4B (Apache-2.0) | diffusers on MPS |
| Matting | Flood-fill white-backdrop matting (no model) | numpy/PIL |
| Image-to-3D | TRELLIS.2-4B via [trellis-mac](https://github.com/shivampkumar/trellis-mac) | PyTorch MPS + Metal bake |

Each item produces **both** representations:

- `<item_id>.glb` — PBR-textured mesh (Metal-baked base color / metallic / roughness)
- `<item_id>.ply` — 3D Gaussian Splat converted from TRELLIS.2's decoded voxel grid (standard INRIA 3DGS layout)
- `<item_id>.png` — RGBA reference image (alpha lets TRELLIS.2 skip its gated background remover, so the run is fully offline)
- `metadata.json` — prompt, models, seed, timings, vertex/gaussian counts

```bash
# One-time workspace setup (~40GB of weights, no HF login required)
scripts/local/setup_local.sh

# Smoke test
python3 scripts/local/run_all.py --limit 2

# Full run (checkpointed; safe to relaunch)
python3 scripts/local/run_all.py

# Unattended multi-day run
nohup scripts/local/run_forever.sh > state/full_run.log 2>&1 &

# Progress report
python3 scripts/local/verify_local.py
```

Model weights live outside git in `meshmaker/local/models/` (~37GB);
`scripts/local/setup_local.sh` rebuilds the workspace from scratch.

## License

See individual asset metadata for licensing details.

## Sister Project

**[cobot-atlas](https://github.com/sepehrmn/cobot-atlas)** — 2,000+ meshes for robot simulation, manipulation research, and embodied AI ([DOI: 10.5281/zenodo.20697491](https://doi.org/10.5281/zenodo.20697491)).

## Citation

If you use relief-atlas in your research, please cite:

```bibtex
@dataset{relief_atlas_2026,
  author    = {Mahmoudian, Sepehr},
  title     = {relief-atlas: 10K+ 3D Mesh Assets for Disaster Relief and Civil Protection},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/sepehrmn/relief-atlas}
}
```
