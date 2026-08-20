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
- Generated via GPT-Image → Trellis 2 / Tripo v3.1 / Hunyuan3D pipeline
- Prompts optimized for image-to-3D: natural language, camera specifications, explicit constraints
- Dual API providers (fal.ai + Runware) with multi-key rotation and budget management

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
