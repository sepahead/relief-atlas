# meshmaker workspace

This directory holds local generation tooling and runtime data. Use the
[top-level README](../README.md) for the maintained workflow and
[AGENTS.md](../AGENTS.md) for contributor instructions. Ignore rules apply to
specific paths; this note and some source files are tracked, so do not assume
everything under `meshmaker/` is excluded from git.

## Local runtime

```text
meshmaker/local/models/       Qwen, TRELLIS, decoder, DINOv3, matting and safety weights
meshmaker/local/imgenv/       Image-generation Python environment
meshmaker/local/trellis-mac/  Pinned upstream source and mesh environment
meshmaker/local/*.log         Local run logs
```

Required model weights total approximately 52.6 GB, including the 33.1 GB Qwen
bundle. Environments, caches, optional weights and outputs add space. These
runtime paths are ignored. Fresh setup uses `scripts/local/setup_local.sh` and
requires approved Hugging Face access for DINOv3; prepared inference uses local
files. Model and source pins live in `config/local_models.json` and
`config/local_runtime.json`.

The current checkpoint transfer policy is one file, one connection and a
256 KiB/s cap on the resumable aria2 path. See the main README for the distinction
between checkpoint transfer, package installation and workspace-only recovery.

## Earlier generation tooling and collections

The earlier fal.ai and Runware scripts may launch many paid jobs and read local
credentials. Inspect their behavior before execution. The maintained cloud
entry point is `scripts/generate_relief.py`; local generation is under
`scripts/local/`.

The Bundeswehr and other earlier assets are useful source collections intended
for integration, with original provenance preserved. `legacy_original/`,
`legacy_original_scripts/` and the associated manifests record that history.
Keep credentials, weights, generated binaries and transient logs out of source
releases. Use an explicitly defined collection and release manifest for asset
distribution; the current default release excludes policy overrides.
