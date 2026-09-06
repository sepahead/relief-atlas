# meshmaker — untracked working directory

This directory is **not** part of the committed relief-atlas source. `.gitignore`
keeps everything under it out of version control except this note. It holds two
unrelated things that both happen to be large, local-only, or sensitive:

## 1. The local model workspace (`meshmaker/local/`)

The on-device pipeline in `scripts/local/` reads its weights and virtualenvs
from here:

```
meshmaker/local/models/       ~38GB of weights (FLUX.2, TRELLIS.2, BiRefNet, NSFW detector)
meshmaker/local/imgenv/       image-generation virtualenv (diffusers on MPS)
meshmaker/local/trellis-mac/  upstream trellis-mac clone + its venv
meshmaker/local/*.log         run logs
```

None of this is committed, and none of it needs to be: `scripts/local/setup_local.sh`
rebuilds the entire workspace from public model repositories, with no
HuggingFace login required. If you are setting up a fresh clone, run that script
rather than trying to recover anything from git history.

## 2. Legacy cloud generation tooling (`meshmaker/*.py`)

The batch/swarm launchers and manifests at the top level of this directory are
the earlier **paid-API** generation pipeline (fal.ai / Runware), kept on disk for
reference. They are untracked because they are:

- **cost-bearing** — the swarm launchers can fan out many parallel paid cloud
  jobs, and a stray invocation spends real money;
- **a credential risk** — they read an `api_keys.txt` from the working tree;
- **superseded** — the maintained pipeline is the fully local one under
  `scripts/local/`, which needs no API keys at all.

`legacy_original_scripts/` contains the committed, reference-only subset of this
same lineage.

## Release checklist

- [ ] `meshmaker/` is absent from any released source tree or archive.
- [ ] No API keys (`api_keys.txt`, `config/api_keys.txt`) are inside the repo tree.
- [ ] No generated assets or run logs from `meshmaker/` ship in a release.
- [ ] Any asset stamped `policy.override = true` in its `metadata.json` is
      deliberately included or deliberately removed — see the content policy
      section of the top-level [README](../README.md) and
      `state/content_policy_overrides.json`.
