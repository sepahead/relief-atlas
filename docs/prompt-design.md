# Prompt design and review

A reference image must explain the particular object to the reconstruction
model. Attractive lighting cannot compensate for an incorrect rotor count,
sealed gripper gap or missing tent wall. Start with the authored identity and
the features a reviewer must be able to see.

## Source, interpretation and evidence

Original prompt fields remain in `generation_prompt.sources`, with SHA256 hashes.
Recipes retain IDs, names, dimensions, physics metadata and policy eligibility.
The original wording participates in policy screening even when a rewrite
removes an inaccurate visual instruction.

[prompt_specs.py](../scripts/local/prompt_specs.py) resolves standalone parts,
assemblies, explicit sets and selected named models before broad type rules.
[shape_specs.py](../scripts/local/shape_specs.py) interprets the primary object
phrase. Payloads and operating context must not select another object's geometry.
Do not add chassis, wheels, poles or cargo from a keyword elsewhere in a sentence.

Research a named product using its manufacturer, operator or original dataset.
Inspect an actual photograph or diagram as well as the technical text. Match
the model, assembly state, material and pose before transferring a detail.
Record the URL, publisher, scope and actual review in
[reference_library.json](../config/reference_library.json). Leave
`visual_review` null when no useful image was inspected. Each recipe binds its
cited records by hash; changes to this library enter generation provenance.

Keep the distinction between:

- An authored mathematical or geometric definition, which needs no invented
  manufacturer. Preserve its dimensions, topology and material.
- A type or collection example, which explains a form without establishing an
  exact model, revision, footprint or fleet configuration.
- A named-model reference, which supports only the details actually documented.
- A chosen design interpretation for an unspecified subject, which must be
  explicit in the recipe rather than presented as a discovered fact.

For example, eight rotors can be eight independent boom-end assemblies or four
coaxial pairs. A Franka Hand reference does not describe the complete arm. An
inflatable shelter photograph does not establish an aluminium pole frame. A
collection photograph of YCB objects does not verify every row as a YCB replica.

## Make the necessary features visible

Put the object identity and critical counts first. Describe the relationships
that matter: concentric motor hubs, connected tool pivots, open connector ports,
two opposed gripping fingers, intact fabric or an actual vessel cavity. Respect
the requested component, material, colour and pose before choosing a camera.

Choose a view for the geometry under review. Motor shafts and cable exits need
a component view; an IC's contacts need a suitable lead or underside view; a
marker needs a square-on face. A multi-rotor layout benefits from a high view
that separates its motor housings. Use one view per reference and leave margin
around every protrusion. Preserve functional bases, supports and explicitly
requested groups.

Do not turn meaningless labels such as `variation 12` into invented dimensions
or topology. Repeated unspecified variants use bounded camera variation; the
source label remains in provenance. Preserve an explicit upside-down, open,
folded or articulated pose. Avoid contradictory backdrop instructions, motion
blur, generic attachment lists and photography keyword chains.

## Apply and inspect

From the repository root:

```bash
python3 scripts/rewrite_generation_prompts.py \
  --report state/prompt_audit.json \
  --review-guide state/prompt_review.json
python3 scripts/rewrite_generation_prompts.py \
  --apply --report state/prompt_rewrite.json
```

The first command performs no inference, checkpoint transfer or paid API calls.
Both commands validate every proposed document before any manifest is replaced.
They check original-source hashes, identity fields, active policy eligibility,
distinct default-eligible active prompts and idempotence. Writes use the shared
atomic JSON writer. Existing images, meshes and saved job snapshots keep their
original prompts and provenance.

The review inventory groups distinct subject definitions and lists every
affected field and row. External source coverage and reviewed visual examples
are separate counts. `exact_product_verified` and `generated_result_reviewed`
start false because compiling a recipe establishes neither claim. This file is
a generated worklist, not a persistent approval ledger; a new export resets
those fields. Keep real visual findings in the isolated pilot's review report,
bound to the actual PNG/GLB hashes.

Generate only the authorized pilot selection into a separate output/state tree.
Inspect the reference before reconstructing it; reject the wrong component
count or layout even if numerical image QA passes. Then inspect textured and
untextured mesh renders, including several views. Automated QA cannot establish
model identity, lettering correctness, mechanical alignment or simulator physics.
Record defects honestly and retain earlier outputs for comparison.

If a counted assembly repeatedly loses parts in text-only generation, stronger
conditioning may be necessary. The optional
[structure-guide builder](../scripts/local/build_structure_guide.py) constructs
the explicitly chosen eight-boom cargo layout, with concentric hubs and separated
rotor disks, for native Qwen image input. It does not infer another drone's layout
or stand in for an inspected generated result. The guide stays separate from
Qwen references; its JSON records the item, prompt hash, actual PNG hash,
generator hash, part inventory and illustrative scale. The generator's
`--structure-reference` path verifies that binding and uses the same durable
edit-intent recovery as reframing. Unsupported layouts stop explicitly.
