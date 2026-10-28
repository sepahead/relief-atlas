# README diagram maintenance

The README has three authored vector figures, each with a different question:

| Figure | Question | Source contracts |
| --- | --- | --- |
| Local pipeline | Which models produce which files? | [run_all.py](../scripts/local/run_all.py), [qwen_imagegen.py](../scripts/local/qwen_imagegen.py), [matting.py](../scripts/local/matting.py), [trellis_meshgen.py](../scripts/local/trellis_meshgen.py), [gs_export.py](../scripts/local/gs_export.py) |
| Offline recovery | Which saved work is reused after interruption? | [offline_run.py](../scripts/local/offline_run.py), [pipeline_utils.py](../scripts/local/pipeline_utils.py), [offline-run.md](offline-run.md) |
| Artifact evidence | What makes an item eligible for the release? | [pipeline_utils.py](../scripts/local/pipeline_utils.py), [pending.py](../scripts/local/pending.py), [release_manifest.py](../scripts/local/release_manifest.py) |

Independent read-only council reviews covered model/data flow, selection and
policy, provenance and certification, interruption recovery, operator intent,
first-time readers, visual hierarchy, accessibility and mobile readability.
Three figures keep these questions distinct. Prompt-research details remain in
[prompt-design.md](prompt-design.md); the pipeline figure points to recorded
source prompts and research without claiming every subject is product-verified.

## Edit once, render every variant

Edit [scripts/render_readme_diagrams.py](../scripts/render_readme_diagrams.py),
then run from any directory using the script's actual path:

```bash
# From the repository root
python3 scripts/render_readme_diagrams.py
python3 scripts/render_readme_diagrams.py --check
python3 scripts/check_agent_index.py
git diff --check
```

The generator uses Python's standard library. It writes twelve SVGs under
`assets/diagrams/`; it does not load models, make network requests, install
packages, operate a service or change generated assets. `--check` reads the
expected variants and exits nonzero when a file is missing or differs from the
authored source. Keep the generated variants in Git so the README works offline.

Light/dark palettes share semantic roles: rose for saved scope and orchestration,
blue for transformations and checks, teal for evidence and release membership,
and amber for pauses or attention. Labels and shapes carry meaning alongside
colour. The rose index marks and restrained contours echo the repository logo.

The wide layouts use a 1120-unit canvas. Compact layouts use 480 units and stack
the content, including distinct sibling routes for GLB and Gaussian exports.
README `<picture>` sources select compact dark, compact light, wide dark, then
wide light as fallback. Every SVG has an opaque background, an accessible title
and description, and a system font stack. There are no scripts, remote fonts,
linked images or `foreignObject` elements.

## Preserve the meaning

- GLB and Gaussian containers come from the same TRELLIS mesh/voxel decode.
  Do not draw a GLB-to-Gaussian conversion or an independent splat-training stage.
- Preserve native alpha; matting applies to opaque output. Image QA and the
  required safety screen gate reconstruction.
- A committed image needs matching reference metadata and configured model
  identity. PNG existence alone is insufficient.
- Missing GLB, Gaussian container or geometry metadata reruns the complete 3D
  stage. Decoded voxels and in-memory denoising steps are not saved checkpoints.
- Complete exports can reuse their models' work while missing or stale QA is
  rerun. Content rejections may still require regeneration.
- A crashed worker can restart within the logged-in session when intent is
  running. After reboot, login is also required. A manual pause survives login,
  and Status does not start the worker.
- A resource wait, bounded content retries, quarantine and infrastructure
  failures have different outcomes; do not depict an unlimited retry loop.
- Current artifact-bound passes do not automatically establish release inclusion.
  Positive membership also checks eligibility, overrides, quarantine and
  deliverable completeness. Controller completion covers every saved target.
- `release_manifest.json` is local accounting. Publication, visual fidelity,
  metric scale, collisions and physical suitability are separate claims.
- Retain historical and Bundeswehr source provenance. These figures do not
  claim that original binaries have been imported or certified as new Qwen output.

Keep live progress, test counts, hardware readings and transfer percentages in
timestamped local reports rather than these figures.

## Inspect the rendered result

When `rsvg-convert` is already installed, render a preview without new packages:

```bash
mkdir -p state/readme_diagrams
rsvg-convert --width 1120 --output state/readme_diagrams/pipeline-wide.png \
  assets/diagrams/local-pipeline-light.svg
rsvg-convert --width 360 --output state/readme_diagrams/pipeline-mobile.png \
  assets/diagrams/local-pipeline-compact-dark.svg
```

Inspect all three figures in both themes and at desktop and phone widths. Check
text/card bounds, arrow routing, conditional labels, contrast and SVG titles,
then verify the README picture sources and alt text. Raster previews stay in
ignored `state/`; the vector files remain the deliverables.
