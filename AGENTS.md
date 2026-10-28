# Working on relief-atlas

This file is the shared repository contract for coding agents. Read [README.md](README.md)
for the user workflow; [CLAUDE.md](CLAUDE.md) adds Claude-specific orientation.
User instructions and current task authorization take precedence over this guide.

## Purpose and evidence

Turn humanitarian and civil-protection manifest items into reproducible reference
images, textured meshes and Gaussian containers. A useful asset needs a known
source, a complete deliverable and current QA evidence—not just a file on disk.

Reason from the failure being prevented:

1. Identify the consumer and the observable behavior that must change.
2. Read the implementation, configuration and relevant tests before designing a fix.
3. Challenge the assumption: is this an invariant, a measured result, an estimate,
   or a convenience inherited from an earlier backend?
4. Make the smallest coherent change across every producer and consumer of the contract.
5. Validate the affected behavior, then report the evidence and its limits.

Imports succeeding, preflight passing, a pilot finishing, human review and a full
release are different claims. Never substitute one for another. Manifest counts
are planned coverage. Automated QA does not establish metric scale, collision
quality, mass properties or physical suitability for a simulator.

Do not cite social-media posts. A post can be deleted or edited, and it can state an
opinion as a fact, which biases readers and agents and weakens scientific rigor. State
the substance in your own words with its date, cite the primary source, and do not link
the post or name its author.

## Delivery and the shared index

Work within the owner's current authorization. Documentation delivery uses
`main` after the gates pass; pipeline and retained-manifest changes require an
owner request. Preserve retained generated metadata. Releases, tags, citation
records, dataset publication and repository settings remain owner actions.
Follow the committer's signing configuration. Do not add AI attribution,
co-author trailers or generated-by lines to commits or pull requests.

Before each push, run the pipeline suite in an appropriate existing Python
environment, `python3 scripts/check_agent_index.py`, and `git diff --check`.
For runtime-sensitive changes, run the suite in both owning environments below.
No package installation or model download is needed to validate a prepared tree.

[agent-index.json](agent-index.json) restates this contract for tools. Keep its
entry documents, gates and named paths aligned with this guide and run its
checker when they change. Keep the original source collections and their IDs;
a request to rewrite prompts does not authorize rewriting historical metadata.

Input coverage remains 10,079 rows across the four geography manifests:
Germany 4,079, EU 2,500, Ukraine 2,500 and general 1,000. The three earlier
defense manifests contain 90 additional source rows. Git retains 125 earlier
mesh metadata records, with generated mesh/image/splat binaries outside Git.
Recheck counts against source before changing these facts; source coverage and
retained metadata are not evidence of a certified release.

## Find the owner of a change

| Concern | Source of truth |
| --- | --- |
| Local orchestration and selection propagation | `scripts/local/run_all.py` |
| README SVG diagrams and responsive variants | `scripts/render_readme_diagrams.py`, `assets/diagrams/`, `docs/readme-diagrams.md` |
| Offline start/pause/resume and login recovery | `atlas`, `*.command`, `scripts/local/offline_run.py`, `config/offline_run.json` |
| Paths, output identity, hashes, locks, atomic state, QA version | `scripts/local/pipeline_utils.py` |
| Manifest loading, deterministic seeds, deliverable completeness | `scripts/local/prompt_utils.py` |
| Versioned reference recipes and retained source prompts | `scripts/local/generation_prompts.py`, `scripts/rewrite_generation_prompts.py` |
| Object specifications, type constraints and research evidence | `scripts/local/prompt_specs.py`, `shape_specs.py`, `config/reference_library.json` |
| Text policy, defaults, override provenance | `scripts/local/content_policy.py` |
| Reference generation and alpha | `scripts/local/qwen_imagegen.py`, `matting.py` |
| Illustrative eight-boom guide and native input binding | `scripts/local/build_structure_guide.py`, `qwen_imagegen.py` |
| 3D decode, texture bake, Gaussian export | `scripts/local/trellis_meshgen.py`, `gs_export.py` |
| QA measurements, retries, quarantine | `scripts/local/qa.py` |
| Certification accounting and release membership | `pending.py`, `verify_local.py`, `release_manifest.py` |
| Model revisions and integrity | `config/local_models.json`, `prepare_models.py` |
| Runtime pins and GPU readiness | `config/local_runtime.json`, `requirements-*.txt`, `preflight.py` |
| Setup and unattended generation | `setup_local.sh`, `run_forever.sh` |
| Slow resumable transfers and macOS recovery | `checkpoint_download.py`, `resume_preparation.py` |
| Paid cloud backend | `scripts/generate_relief.py`, `scripts/verify_outputs.py` |

Python and shell script names without a directory in this table are under
`scripts/local/`; requirement files are at the repository root.
`legacy_original/` and its Bundeswehr and other earlier collections are valuable
source assets intended for integration. The directory name does not mean
disposable or out of scope. Preserve binaries, metadata, IDs and original model
provenance. Add imported collections through an explicit manifest/provenance
mapping; some earlier subjects already have active `legacy_` manifest rows,
which does not prove their original binaries are imported or certified. Do not
silently classify those assets as Qwen output or current QA-certified.
Current default policy eligibility is distinct from the user's intended broader
collection. A request to document that intention does not change the classifier.

`legacy_original_scripts/` records how those earlier assets were produced. `meshmaker/local/` contains
ignored environments, upstream source and weights; durable fixes belong in
tracked scripts or pinned configuration, not solely in a local upstream clone.
`state/` and generated outputs are ignored, machine-specific evidence.

## Contracts to preserve

- Apply the same IDs, geography, limit and policy selection to every phase.
  A phase must never broaden a bounded pilot into production.
- Defaults exclude restricted and blocked subjects. Keep explicit overrides
  recorded in policy ledgers and metadata. Release excludes override artifacts.
- Use the configured Qwen model identity. Do not restore FLUX Klein, silently
  swap models, or relabel historical outputs as Qwen results.
- Preserve `generation_prompt.sources` and source hashes when rewriting recipes.
  Policy screens source and current wording; a rewrite must not lower the
  original eligibility requirements. Prompt constraints are not measured proof
  of motor count, mechanical alignment or other subject-specific structure.
- Keep authored geometry guides distinct from Qwen output. Native structure edits
  bind the source PNG, sidecar, selected item and manifest prompt by hash. The
  illustrative guide GLB is not an asset deliverable or a metric product model.
- Preserve native RGBA. Opaque references use matting; geometry must bind to
  the actual source PNG hash and matching reference metadata.
- A deliverable includes GLB, a recognized Gaussian container and metadata.
  Certification additionally requires current QA verdicts bound to artifact
  hashes. Release membership is computed positively by `release_manifest.py`.
- Keep the NSFW screen fail-closed for certification. Missing renderers and
  infrastructure failures stop the run without deleting otherwise valid assets.
  Content defects may trigger bounded retries and quarantine.
- If QA measurements or thresholds change enough to invalidate old verdicts,
  update the shared `QA_VERSION` and review pruning, retry and release behavior.
  Do not erase quarantine or fabricate passes to obtain a successful exit.
- Use shared `GS_EXTENSIONS` and completion predicates. Default SPZ may fall
  back to SPLAT; accounting and disk estimates must recognize actual containers.
- GLB is Y-up; Gaussian coordinates are native Z-up. Position mapping to GLB
  is `[x, y, z] -> [x, z, -y]`; orientations must also be transformed when needed.
- Preserve atomic writes, synced commit markers and process locks. Interrupted
  GPU work may regenerate the current item; finished artifacts must be revalidated.
- Offline-controller runs use frozen manifest snapshots through
  `RELIEF_MANIFEST_DIR`. Do not broaden a saved run to live catalogue changes.
  Keep the network-denying launchd sandbox, shared GPU lease and owned-process
  cleanup intact. Status is read-only; creating controls does not start production.
- Model revision claims require evidence. Reuse checkpoints only after their
  integrity checks; preserve partial files and `.aria2` resume maps.

## Operational boundaries

Inspect the working tree before editing. Preserve existing user changes; do not
reset, clean, overwrite unrelated files, or commit unless requested.

The current Qwen checkpoint transfer policy is **one connection, one file at a
time, unlimited speed**, with slow retries, per the user's latest instruction.
Keep that policy unless the user changes it. Do not add concurrent probes,
mirrors, routing workarounds, unbounded retries or downloader parallelism. Setup
package/git traffic is a separate operation; checkpoint transfer settings are
not a global network limiter.

Respect the latest pause/resume request. A status query does not authorize a
restart. Inspect `state/preparation_status.json`, timestamped logs, launchd state
and saved maps together; a live heartbeat is not evidence of bytes received.
Do not infer download progress from sparse file length or allocated blocks.

Stay within the authorized selection. A four-item pilot does not authorize a
9,509-item production run. Documentation work does not authorize setup, model
transfers, paid API calls or launching inference. Authorized work should proceed
without repeated confirmation; do not invent approval steps.

Keep tokens, API keys and signed download URLs out of logs, diffs, command
arguments and git. Do not dump environment variables, credential files or an
unfiltered process table. `config/api_keys.txt` is local only.

## Validation and handoff

For documentation changes, verify local links, paths, option names and side
effects against source. Use harmless `--help` or `--dry-run` where appropriate;
do not bootstrap runtimes or load models just to validate prose.

For code changes, choose focused regressions that exercise the failure. The
lightweight test environment is separate from model inference:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt
.venv/bin/python -m unittest discover -s tests
```

For runtime-sensitive changes, use the owning existing environment:

```bash
meshmaker/local/imgenv/bin/python -m unittest discover -s tests
meshmaker/local/trellis-mac/.venv/bin/python -m unittest discover -s tests
```

Read skips; optional dependencies can leave a path untested. Neither suite proves
model inference or GPU compatibility. Use preflight and an isolated pilot when
that behavior is relevant and authorized. Explain missing environments rather
than starting a large installation during an unrelated task.

Finish with what changed, why it matters, what was checked and any remaining
uncertainty. Keep live download percentages and test counts in timestamped
reports, not permanent agent instructions.

## Councils when requested

Use independent read-only reviews of workflow, correctness and user experience.
Give each reviewer a concrete question and shared constraints; avoid overlapping
edits. Compare findings against source and resolve disagreements with evidence.
The primary agent owns the final integration. Do not introduce external debate
services or extra model downloads for a council.
