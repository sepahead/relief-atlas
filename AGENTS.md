# relief-atlas agent contract

## Authority and workflow

The owner authorizes agents to commit and push documentation changes to `main`.
Run these checks before each push:

- `python3 -m unittest discover -s tests`
- `python3 scripts/check_agent_index.py`
- `git diff --check`

Do not add AI attribution, co-author trailers, or generated-by lines.
Releases, tags, citation records, dataset publication, and repository settings remain owner actions.
Change the generation pipeline only on the owner's request.

## What this repository holds

- `manifests/` holds the item definitions with pre-generated prompts.
- `scripts/` holds the cloud generation pipeline. `scripts/local/` holds the on-device pipeline.
- `tests/` holds the pipeline self-tests.
- `meshmaker/` holds batch-submission and manifest tools. Git ignores its local model workspace.
- `outputs_relief/` and `legacy_original/` hold the metadata of 125 earlier meshes.
- `legacy_original_scripts/` keeps the earlier generation scripts for reference.
- `config/` holds a placeholder key file. Git ignores the real key file.

The mesh, image, and splat files left Git on 22 August 2026 (commit `3c8cc86`).
Generated dataset artifacts live outside Git.
Do not rewrite retained manifests or metadata.

## Facts to keep consistent

- The four geography manifests hold 10,079 items: Germany 4,079, EU 2,500, Ukraine 2,500, and general 1,000.
- Three earlier defense manifests hold 90 more items.
- The content policy refuses 30 items and gates 540 behind an explicit flag. The default run generates 9,509 items.
- Git tracks 125 metadata records and no mesh files.

Recheck each count against the manifests before you change it.

## Agent index

[`agent-index.json`](agent-index.json) restates the facts of this file for tools.
Update it when a gate or a named path changes, then run `python3 scripts/check_agent_index.py`.
