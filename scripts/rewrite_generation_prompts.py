#!/usr/bin/env python3
"""Rewrite generation manifests without inference, downloads or paid API calls."""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

import content_policy  # noqa: E402
from generation_prompts import PROMPT_VERSION, digest, rewrite_item, strip_capture_style  # noqa: E402
from pipeline_utils import file_sha256, write_json  # noqa: E402


def base_catalogue():
    """Read authored subjects, before the old keyword-based material enrichment."""
    module = ast.parse((ROOT / "scripts/generate_prompts.py").read_text())
    subjects = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id.endswith("_OBJECTS") for target in node.targets):
            continue
        for name, prompt, _tags in ast.literal_eval(node.value):
            if name in subjects and subjects[name] != prompt:
                raise ValueError(f"Ambiguous authored subject: {name}")
            subjects[name] = prompt
    return subjects


def manifest_paths():
    return sorted([*(ROOT / "manifests").glob("*.json"),
                   *(ROOT / "meshmaker").glob("*manifest*.json")])


def rewrite_document(document, catalogue):
    if not isinstance(document, dict) or not isinstance(document.get("meshes"), list):
        raise ValueError("Expected a manifest with a meshes list")
    result = dict(document)
    rewritten = []
    ordinals = defaultdict(int)
    for row in document["meshes"]:
        name = row.get("name", "").split(" (", 1)[0]
        base = catalogue.get(name) if row.get("geography") else None
        recipe = row.get("generation_prompt", {})
        source = recipe.get("sources", {}).get("prompt", row.get("prompt", row.get("text_prompt", "")))
        key = name if base else (row.get("category", ""), strip_capture_style(source))
        ordinal = recipe.get("ordinal", ordinals[key]) if recipe.get("version") == PROMPT_VERSION else ordinals[key]
        rewritten.append(rewrite_item(row, base_subject=base, ordinal=ordinal))
        ordinals[key] += 1
    result["meshes"] = rewritten
    result["generation_prompt_version"] = PROMPT_VERSION
    return result


def validate_document(before, after):
    """Preserve identity, authored source fields and current policy eligibility."""
    if len(before["meshes"]) != len(after["meshes"]):
        raise ValueError("Manifest row count changed")
    fields = {"prompt", "text_prompt", "image_prompt", "generation_prompt"}
    for old, new in zip(before["meshes"], after["meshes"]):
        if {k: v for k, v in old.items() if k not in fields} != {
                k: v for k, v in new.items() if k not in fields}:
            raise ValueError(f"Non-prompt fields changed for {old.get('id')}")
        recipe = new.get("generation_prompt", {})
        for key, source in recipe.get("sources", {}).items():
            expected = old.get("generation_prompt", {}).get("sources", {}).get(key, old.get(key))
            if source != expected or digest(source) != recipe["source_sha256"][key]:
                raise ValueError(f"Source prompt changed for {old.get('id')}")
        if old.get("geography") and content_policy.classify_item(old).classification != content_policy.classify_item(new).classification:
            raise ValueError(f"Policy classification changed for {old['id']}")


def review_guide(plans):
    """Inventory distinct visual definitions; never treat recipe coverage as QA."""
    definitions = {}
    profiles, scopes = Counter(), Counter()
    fields_with_sources = fields_with_visual_examples = 0
    for path, _before, document, _changed in plans:
        for row in document["meshes"]:
            for field, spec in row["generation_prompt"]["specifications"].items():
                subject = row[field].splitlines()[0].removeprefix("Subject: ")
                key = digest(json.dumps({"subject": subject, "category": row.get("category"),
                                         "specification": spec}, sort_keys=True))
                if key not in definitions:
                    definitions[key] = {"subject": subject, "category": row.get("category"),
                        "specification": spec, "occurrences": [],
                        "generated_result_reviewed": False,
                        "exact_product_verified": False}
                definitions[key]["occurrences"].append({"manifest": str(path.relative_to(ROOT)),
                    "id": row["id"], "field": field})
                profiles[spec["profile"]] += 1
                scopes[spec["scope"]] += 1
                evidence = spec["reference_evidence"]
                fields_with_sources += bool(evidence)
                fields_with_visual_examples += any(r["visual_example_reviewed"] for r in evidence.values())
    return {"at": datetime.now(timezone.utc).isoformat(), "prompt_version": PROMPT_VERSION,
            "meaning": "Prompt design evidence only. Family examples do not verify exact products; authored definitions are not web research. No generated result is approved by this inventory.",
            "distinct_definitions": len(definitions), "profiles": dict(profiles), "scopes": dict(scopes),
            "fields_with_external_sources": fields_with_sources,
            "fields_with_reviewed_visual_examples": fields_with_visual_examples,
            "definitions": list(definitions.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the checked prompts; default is report only")
    parser.add_argument("--report", type=Path, help="optional timestamped JSON audit report")
    parser.add_argument("--review-guide", type=Path, help="write distinct item specifications, evidence and visual-review checks")
    args = parser.parse_args(argv)
    catalogue = base_catalogue()
    plans = []
    counts = Counter()
    for path in manifest_paths():
        before = json.loads(path.read_text())
        after = rewrite_document(before, catalogue)
        validate_document(before, after)
        if rewrite_document(after, catalogue) != after:
            raise ValueError(f"Rewrite is not idempotent: {path}")
        changed = sum(old.get(key) != new.get(key) for old, new in zip(before["meshes"], after["meshes"])
                      for key in ("prompt", "text_prompt", "image_prompt") if key in old)
        fields = sum(len(row.get("generation_prompt", {}).get("sources", {})) for row in after["meshes"])
        changed_recipes = sum(old.get("generation_prompt") != new.get("generation_prompt")
                              for old, new in zip(before["meshes"], after["meshes"]))
        counts.update(manifests=1, rows=len(after["meshes"]), fields=fields, changed_fields=changed,
                      changed_recipes=changed_recipes, changed_documents=int(before != after))
        plans.append((path, before, after, changed))
    active = [row for path, _before, after, _changed in plans
              if path.name.startswith("relief_manifest_") for row in after["meshes"]]
    permitted, _ = content_policy.screen_items([dict(row) for row in active], False, False)
    prompts = [row["prompt"] for row in permitted]
    if len(set(prompts)) != len(prompts):
        raise ValueError("Rewritten default-eligible active prompts are not unique")
    # Validate the whole set before replacing any file.
    records = []
    for path, before, after, changed in plans:
        original_hash = file_sha256(path)
        if args.apply and before != after:
            write_json(path, after)
        records.append({"path": str(path.relative_to(ROOT)), "rows": len(after["meshes"]),
                        "changed_fields": changed, "before_sha256": original_hash,
                        "after_sha256": file_sha256(path) if args.apply else None})
    report = {"at": datetime.now(timezone.utc).isoformat(), "prompt_version": PROMPT_VERSION,
              "applied": args.apply, **counts, "files": records,
              "preserved": ["IDs", "row order", "names", "categories", "geography", "tags", "physics metadata", "original prompt fields", "active policy eligibility"],
              "inference_started": False, "generated_binaries_changed": False}
    report["active_policy_counts"] = dict(Counter(content_policy.classify_item(row).classification for row in active))
    report["unique_default_eligible_prompts"] = len(set(prompts))
    guide = review_guide(plans)
    report["research_coverage"] = {k: v for k, v in guide.items() if k not in ("definitions", "at", "prompt_version", "profiles", "scopes")}
    if args.review_guide:
        write_json(args.review_guide, guide)
    if args.report:
        write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "preserved")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
