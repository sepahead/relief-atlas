# Claude orientation

Read [AGENTS.md](AGENTS.md) first: it is the shared contract for all agents.
Use [README.md](README.md) for setup, commands, architecture and operating limits.
This file adds orientation rather than duplicating those instructions.

Never add Claude, an AI assistant or an agent as a commit or pull-request
co-author, and do not add generated-by lines. Follow the shared delivery gates.
The checked [agent index](agent-index.json) records entry documents and gates.

## Start with the right question

What does this task need to prove: a correct manifest selection, a usable runtime,
a generated asset, a current QA pass, or a distributable release? Find the code
that owns that claim before changing it. Existing output, a healthy heartbeat or
a passing unit suite alone cannot establish all five.

Check `git status --short` and read relevant files before editing. This checkout
may contain work already in progress. Preserve it and keep the change scoped.
Local weights, environment versions and `state/` reports are machine evidence,
not portable repository guarantees.

The Bundeswehr and other earlier collections are useful assets intended for
integration. Preserve them and their true provenance. Historical directory names
do not authorize deletion or exclusion from the project's future scope; current
default policy and planned collection inclusion are distinct.

## Quick navigation

- User workflow and storage: `README.md`.
- Shared invariants and operational boundaries: `AGENTS.md`.
- Pipeline entry point: `scripts/local/run_all.py`.
- Offline controls and recovery: `atlas`, the root `.command` files and
  [docs/offline-run.md](docs/offline-run.md).
- Common identity and certification rules: `scripts/local/pipeline_utils.py`.
- Reference recipes, item evidence and visual review: [docs/prompt-design.md](docs/prompt-design.md),
  `scripts/local/generation_prompts.py`, `prompt_specs.py`, `shape_specs.py`.
- Native Qwen reference edits and illustrative guides: `scripts/local/qwen_imagegen.py`,
  `build_structure_guide.py`. A guide is not a generated or certified asset.
- Quality and release: `scripts/local/qa.py`, `release_manifest.py`.
- Model/runtime pins: `config/local_models.json`, `config/local_runtime.json`.
- Tests: `tests/test_pipeline.py`, `test_gates.py`, `test_readiness.py`.

## Work in this session

Observe the latest pause/resume instruction and the authorized item selection.
Qwen checkpoint transfers use one connection, one file at a time and unlimited
speed, per the user's latest instruction. Read local status for
progress requests; do not open extra network connections or restart a paused
worker to answer them.

When councils are requested, use independent reviews with explicit questions,
then reconcile their findings yourself. The council is a way to challenge
assumptions, not a vote that overrides code, evidence or user intent.

Before a handoff, check the affected contract and report actual results. Separate
missing evidence from a failed check; never present an unrun pilot as ready.
