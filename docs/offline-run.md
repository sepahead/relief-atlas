# Offline operation and recovery

Double-click **Start.command** once to begin. Use **Pause.command** whenever you
want the GPU work to stop and **Resume.command** to continue. **Status.command**
only reports the saved run. These files work from Finder and from any terminal
directory without activating Python environments.

The [README offline handoff](../README.md#start-pause-and-resume-offline) includes
the complete status table, terminal examples, a network-denied readiness check,
storage guidance and local troubleshooting paths.

Start's default selection is the whole currently eligible active collection.
The default policy still excludes restricted and blocked rows. Original source
collections, binaries and metadata remain in the repository; this controller
does not change their eligibility or import status.

## What a pause preserves

Pause first saves your intent to disk, then stops only this job's process group.
It waits for child processes to exit and escalates unresponsive children in that
owned group. Model CLI processes also share a GPU lease, so a second local model
run cannot start concurrently with the managed job.

References, meshes, Gaussian containers and QA use the existing atomic commit
markers and artifact hashes. Resume rescans those artifacts; a stale PID or an
old percentage is not proof of completion. It keeps valid finished stages and
restarts an interrupted current stage. Intermediate denoising tensors live in
RAM and are not a resumable checkpoint.

A committed reference can resume independently of its geometry. If the image
and its hash-bound `image_meta.json` are present but the GLB is missing, the
image generator does no inference: image QA is completed if necessary, then
TRELLIS generates the mesh and Gaussians. If only a GLB exists, the Gaussian
container or final `metadata.json` is missing, the 3D stage runs again from the
saved reference. Decoded voxel data is held in RAM, so that stage regenerates
both exports. Fully committed geometry with missing or outdated QA is inspected
without reloading the geometry model; content rejections may require regeneration.

Hidden temporary files and raw frames do not count as commits. A PNG whose image
metadata is missing or does not match is regenerated. Geometry bound to another
PNG is regenerated from the current reference. Interrupted native edits can
recover with stale sidecars only when their durable intent matches the same
source image and metadata hashes. Unrelated references remain protected.

Completed earlier items remain intact when a later item stops. An interruption
during final release accounting reuses the certified assets and recreates the
release manifest. Completion still requires every saved target to be included.

Start also installs a user LaunchAgent. A running job returns after reboot and
login, and can continue with the terminal closed. A manual pause survives reboot
and login. The supervisor cleans up the job's child processes after a crash;
the next worker revalidates the on-disk state before continuing. No service runs
before user login.

## Offline execution

The macOS worker and its descendants run under `sandbox-exec` with network
operations denied. At startup the worker verifies that socket use is denied,
using a loopback bind that sends no packets. Hugging Face offline and telemetry
settings are also explicit. The controller never invokes setup, pip, checkpoint
preparation or the paid backend.

The existing image and mesh environments and cached checkpoints must be ready.
Preflight checks them in the actual worker environment. A missing model or
broken runtime produces **needs attention**, with a local report and log path.
It does not attempt to fetch a replacement model or install dependencies.

## Saved run

The first Start freezes selected manifest rows and generation settings. It writes
the snapshot before committing the plan, with source and snapshot hashes. Every
generation and QA subprocess receives the same snapshot directory. Catalogue or
configuration edits cannot silently broaden that run.

| Location | Purpose |
| --- | --- |
| `config/offline_run.json` | Defaults for the next new run |
| `state/offline_controller/plan.json` | Convenient copy of the latest prepared plan |
| `state/offline_controller/control.json` | Your latest start/pause/resume intent and committed run ID |
| `state/offline_controller/status.json` | Worker progress, counts and diagnostics |
| `state/offline_controller/phase.json` | Current generator or QA phase |
| `state/offline_controller/supervisor.log` | Service startup output, also shown by `atlas logs` |
| `state/offline_controller/runs/<run-id>/manifests/` | Hash-bound selected manifest snapshots |
| `state/offline_controller/runs/<run-id>/run.log` | Detailed run output |
| `state/offline_controller/runs/<run-id>/progress.json` | Bounded retry state |
| `state/offline_controller/runs/<run-id>/release_manifest.json` | Positive release membership after successful completion |
| `outputs_relief/`, `state/qa_*.json` | Default artifacts and existing certification ledgers |

The output and QA-state paths are configurable and saved in the plan. Existing
valid work in those paths is reused. The controller does not automatically move
or mix isolated pilot directories. To continue one of those, select its exact
output/state pair in a new run.

The committed run ID selects its immutable `runs/<run-id>/plan.json`; a power
cut during preparation of a replacement plan cannot change the active selection.

```bash
# Preview without starting anything
./atlas plan

# A new, bounded pilot; existing output folders are preserved
./atlas start --ids gen_dis_0001 ger_uav_0001 gen_med_0001 eu_rc_0001 \
  --output-dir outputs_offline_pilot --state-dir state/offline_pilot

# Later, any number of times
./atlas pause
./atlas resume
./atlas status
./atlas logs
```

If a plan already exists, Start and Resume keep its settings. To deliberately
change selection or configuration, pause first and use `./atlas start --new-run`
with the desired options. Old plans and all generated assets remain on disk.
Changing settings does not force already certified assets to regenerate; choose
a separate output/state pair for a controlled comparison.

## Automatic waiting and failures

- On battery at or below 15%, the worker stops active sampling and waits. It
  resumes after AC power returns or the battery exceeds 25%. The thresholds are
  configurable. This waiting state is saved before child cleanup and survives
  worker restart and manual pause. A manual pause takes precedence over recovery.
- Below the configured disk reserve, it waits without deleting existing files.
  It resumes after enough space becomes available.
- It waits when another pipeline owns the same QA-state lock or GPU lease.
  Power, disk space and pipeline ownership are rechecked during the GPU wait.
- An incomplete item gets a bounded number of attempts with a delay. Other
  pending items can still finish. Content quarantine remains in the QA ledger.
- Repeated failures, missing prerequisites, changed snapshots or quarantine are
  reported as **needs attention**, never as completed. Resume rechecks the
  environment and can retry controller failures; it does not erase QA quarantine.

The standard preset uses sequential item processing, native sampler defaults,
2048 textures and SPZ with the existing SPLAT fallback. The supported radial
cargo-octocopter recipe gets a cached authored guide and native `1024_cascade`
geometry. Guides are distinct from Qwen output and have their own input hashes.
Existing valid configured-model references are reused; guides apply when a fresh
reference is needed. Use a separate output/state pair to replace older references.
Each distinct geometry/texture/container preset receives its own prerequisite
check before its first pending item, including a switch to guided geometry.
The pipeline still cannot automatically establish exact product fidelity,
mechanical alignment, collision quality or suitability for a simulator.

## Remove automatic login recovery

```bash
./atlas uninstall
```

This pauses the job and removes only this controller's user LaunchAgent. Saved
plans, logs, outputs and QA remain. The next Start reinstalls the service.
