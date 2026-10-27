"""Self-tests for the run-accounting and release machinery.

Companion to test_pipeline.py, which covers the policy families and the
exporters. Everything here corresponds to a defect that actually shipped, or to
a rule that had no enforcement at all:

  * the unattended driver's pending count treated an item with no image as "not
    pending", so it reported 14 items outstanding while 10,047 had never been
    generated, printed "all done" and exited;
  * mesh QA wiped artifacts for four of its seven failure reasons and nothing
    for the other three, leaving a rejected GLB on disk that mesh_complete()
    then reported as a finished deliverable;
  * QA verdicts carried no contract version, so verdicts measured by a retired
    matting implementation were indistinguishable from current ones;
  * nothing read the QA ledgers back, so "what ships" was whatever files existed
    -- including 27 blocked-tier weapon meshes that predate the content policy.

Stdlib + numpy only, so it gates a commit without the 40GB model workspace.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_LOCAL = REPO_ROOT / "scripts" / "local"
sys.path.insert(0, str(SCRIPTS_LOCAL))

import pending  # noqa: E402
import prompt_utils  # noqa: E402
import release_manifest  # noqa: E402
from pipeline_utils import IMAGE_MODEL_ID, artifact_fingerprints, file_sha256  # noqa: E402


def item(item_id, name="water pump trailer", geography="general",
         category="relief", prompt=""):
    return {"id": item_id, "name": name, "prompt": prompt,
            "geography": geography, "category": category}


def make_complete(root, it, *, gs_ext="spz", override=False):
    """Write a full deliverable: GLB + a 3DGS container + metadata.json."""
    d = Path(root) / it["geography"] / it["category"] / it["id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{it['id']}.glb").write_bytes(b"glTF" + b"0" * 4096)
    (d / f"{it['id']}.{gs_ext}").write_bytes(b"x" * 64)
    (d / f"{it['id']}.png").write_bytes(b"synthetic reference")
    (d / "image_meta.json").write_text(json.dumps({"image_model": IMAGE_MODEL_ID,
        "image_sha256": file_sha256(d / f"{it['id']}.png")}))
    (d / "metadata.json").write_text(json.dumps(
        {"id": it["id"], "policy": {"override": override},
         "source_image_sha256": file_sha256(d / f"{it['id']}.png")}))
    return d


def verdict(passed=True, version=None, *, d=None, item_id="a", mode=None):
    v = {"pass": passed, "reasons": [] if passed else ["synthetic"]}
    v["qa_version"] = prompt_utils.QA_VERSION if version is None else version
    if d is not None:
        v["artifacts"] = artifact_fingerprints(d, item_id, mode)
        if mode == "images":
            v.update(nsfw=0.0, nsfw_screened=True)
    return v


class TestQaVersioning(unittest.TestCase):
    """A verdict is only usable if the contract that produced it is current."""

    def test_unstamped_verdict_is_not_current(self):
        """The pre-v2 ledgers carry no version at all; that must not read as OK."""
        self.assertFalse(prompt_utils.verdict_is_current({"pass": True}))

    def test_superseded_version_is_not_current(self):
        self.assertFalse(prompt_utils.verdict_is_current(
            {"pass": True, "qa_version": prompt_utils.QA_VERSION - 1}))

    def test_current_version_is_current(self):
        self.assertTrue(prompt_utils.verdict_is_current(verdict()))

    def test_non_dict_is_not_current(self):
        for junk in (None, 1, "pass", []):
            self.assertFalse(prompt_utils.verdict_is_current(junk), junk)

    def test_current_verdicts_filters_the_ledger(self):
        ledger = {"a": verdict(), "b": {"pass": True},
                  "c": verdict(version=0)}
        self.assertEqual(set(prompt_utils.current_verdicts(ledger)), {"a"})

    def test_current_verdicts_tolerates_empty(self):
        self.assertEqual(prompt_utils.current_verdicts({}), {})
        self.assertEqual(prompt_utils.current_verdicts(None), {})


class TestPendingAccounting(unittest.TestCase):
    """What the unattended driver counts as outstanding work."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.out = Path(self._td.name)
        self.addCleanup(self._td.cleanup)

    def test_ungenerated_item_is_pending(self):
        """THE regression: no PNG, no mesh, nothing on disk -- that is work.

        The old count skipped these ("the image phase has not produced it yet"),
        which made every failed or QA-wiped image invisible and let the driver
        declare success with the dataset ungenerated.
        """
        buckets = pending.classify([item("a")], self.out)
        self.assertEqual(buckets[pending.PENDING], ["a"])
        self.assertEqual(buckets[pending.DONE], [])

    def test_complete_item_is_done(self):
        it = item("a")
        make_complete(self.out, it)
        buckets = pending.classify([it], self.out)
        self.assertEqual(buckets[pending.DONE], ["a"])
        self.assertEqual(buckets[pending.PENDING], [])

    def test_partial_deliverable_is_still_pending(self):
        """A GLB alone is not done; mesh_complete() is the shared predicate."""
        it = item("a")
        d = make_complete(self.out, it)
        (d / "a.spz").unlink()
        self.assertEqual(pending.classify([it], self.out)[pending.PENDING], ["a"])

    def test_policy_refusal_is_not_pending(self):
        """570 rows are refused by default. Counting them means never reaching 0."""
        buckets = pending.classify([item("w", name="G36 assault rifle")], self.out)
        self.assertEqual(buckets[pending.REFUSED], ["w"])
        self.assertEqual(buckets[pending.PENDING], [])

    def test_allow_restricted_moves_an_item_into_pending(self):
        it = item("m", name="military truck for disaster logistics convoy")
        self.assertEqual(
            pending.classify([it], self.out)[pending.REFUSED], ["m"])
        self.assertEqual(
            pending.classify([it], self.out,
                             allow_restricted=True)[pending.PENDING], ["m"])

    def test_quarantined_item_is_not_pending(self):
        """Quarantine is the only state the pipeline itself calls terminal."""
        buckets = pending.classify([item("a")], self.out, quarantine=["a"])
        self.assertEqual(buckets[pending.QUARANTINED], ["a"])
        self.assertEqual(buckets[pending.PENDING], [])

    def test_complete_item_beats_a_stale_quarantine_entry(self):
        """A finished deliverable is finished, whatever the ledger remembers."""
        it = item("a")
        make_complete(self.out, it)
        buckets = pending.classify([it], self.out, quarantine=["a"])
        self.assertEqual(buckets[pending.DONE], ["a"])
        self.assertEqual(buckets[pending.QUARANTINED], [])

    def test_every_item_lands_in_exactly_one_bucket(self):
        """The count is only meaningful if the buckets partition the manifest."""
        items = [item("a"), item("b", name="G36 assault rifle"),
                 item("c"), item("d")]
        make_complete(self.out, items[2])
        buckets = pending.classify(items, self.out, quarantine=["d"])
        seen = [i for b in pending.BUCKETS for i in buckets[b]]
        self.assertEqual(sorted(seen), ["a", "b", "c", "d"])
        self.assertEqual(len(seen), len(set(seen)))

    def test_classify_does_not_mutate_the_caller_rows(self):
        """screen_items() stamps a policy key onto whatever it is handed."""
        rows = [item("a")]
        pending.classify(rows, self.out)
        self.assertNotIn("policy", rows[0])

    def test_any_gs_container_counts_as_complete(self):
        for ext in prompt_utils.GS_EXTENSIONS:
            with self.subTest(ext=ext), tempfile.TemporaryDirectory() as td:
                it = item("a")
                make_complete(td, it, gs_ext=ext)
                self.assertEqual(
                    pending.classify([it], td)[pending.DONE], ["a"])


class TestReleaseGate(unittest.TestCase):
    """Which items may actually ship."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.out = Path(self._td.name)
        self.addCleanup(self._td.cleanup)

    def reason(self, it, d, *, image=True, mesh=True, gs=True,
               quarantine=(), require_gs_qa=True, version=None):
        def led(ok, mode):
            return {it["id"]: verdict(ok, version, d=d, item_id=it["id"], mode=mode)} if ok is not None else {}
        return release_manifest.exclusion_reason(
            it, d,
            image_qa=led(image, "images"), mesh_qa=led(mesh, "meshes"),
            gs_qa=led(gs, "gaussians"),
            quarantine=quarantine, require_gs_qa=require_gs_qa)

    def test_fully_certified_allowed_item_ships(self):
        it = item("a")
        d = make_complete(self.out, it)
        self.assertIsNone(self.reason(it, d))

    def test_blocked_item_never_ships_even_when_every_gate_passed(self):
        """The 27 weapon meshes on disk passed mesh QA. Geometry is not consent."""
        it = item("w", name="Heckler & Koch G36 assault rifle")
        d = make_complete(self.out, it)
        self.assertEqual(self.reason(it, d), "policy blocked (weapon)")

    def test_restricted_item_does_not_ship(self):
        it = item("m", name="Boxer GTK armoured personnel carrier")
        d = make_complete(self.out, it)
        self.assertTrue(self.reason(it, d).startswith("policy restricted"))

    def test_override_generated_asset_does_not_ship(self):
        """policy.override is the asset's own admission that a flag was needed."""
        it = item("a")
        d = make_complete(self.out, it, override=True)
        self.assertEqual(self.reason(it, d), "override-generated")

    def test_stale_verdict_does_not_certify(self):
        """A pass from a retired contract must not let an asset skip the gate."""
        it = item("a")
        d = make_complete(self.out, it)
        self.assertEqual(self.reason(it, d, version=0),
                         "no current image-QA pass")

    def test_missing_verdict_is_treated_as_a_failing_one(self):
        it = item("a")
        d = make_complete(self.out, it)
        self.assertEqual(self.reason(it, d, image=None),
                         "no current image-QA pass")

    def test_each_gate_is_required(self):
        it = item("a")
        d = make_complete(self.out, it)
        for kwargs, expected in (
            ({"image": False}, "no current image-QA pass"),
            ({"mesh": False}, "no current mesh-QA pass"),
            ({"gs": False}, "no current 3DGS-QA pass"),
        ):
            with self.subTest(**kwargs):
                self.assertEqual(self.reason(it, d, **kwargs), expected)

    def test_gs_qa_can_be_relaxed_explicitly(self):
        it = item("a")
        d = make_complete(self.out, it)
        self.assertIsNone(self.reason(it, d, gs=None, require_gs_qa=False))

    def test_incomplete_deliverable_does_not_ship(self):
        it = item("a")
        d = make_complete(self.out, it)
        (d / "a.spz").unlink()
        self.assertEqual(self.reason(it, d), "incomplete deliverable")

    def test_ungenerated_item_is_reported_as_such(self):
        it = item("a")
        d = self.out / it["geography"] / it["category"] / it["id"]
        self.assertEqual(self.reason(it, d), "not generated")

    def test_quarantined_item_does_not_ship(self):
        it = item("a")
        d = make_complete(self.out, it)
        self.assertEqual(self.reason(it, d, quarantine=["a"]), "quarantined")

    def test_unreadable_metadata_fails_closed(self):
        it = item("a")
        d = make_complete(self.out, it)
        (d / "metadata.json").write_text("{not json")
        self.assertEqual(self.reason(it, d), "unreadable metadata.json")

    def test_policy_is_judged_before_anything_else(self):
        """Precedence mirrors the pipeline: subject matter before pixels.

        A weapon that was never generated must still be attributed to the
        policy, or an audit of the report cannot tell refusal from absence.
        """
        it = item("w", name="G36 assault rifle")
        d = self.out / it["geography"] / it["category"] / it["id"]
        self.assertEqual(self.reason(it, d), "policy blocked (weapon)")


class TestMeshWipeIsTotal(unittest.TestCase):
    """A partial wipe leaves a rejected mesh looking like a finished one."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
            import PIL  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("numpy/PIL unavailable")
        import qa
        cls.qa = qa

    def test_wipe_mesh_removes_every_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            it = item("a")
            d = make_complete(td, it, gs_ext="spz")
            (d / "a.splat").write_bytes(b"x")
            (d / f"a.{self.qa.GS_MONTAGE}").write_bytes(b"x")
            (d / f"a.{self.qa.MESH_MONTAGE}").write_bytes(b"x")
            self.assertTrue(prompt_utils.mesh_complete(d, "a"))

            self.qa.wipe_mesh(d, "a")

            # The GLB in particular: leaving it behind is what made a failed
            # mesh report as a complete deliverable.
            self.assertFalse((d / "a.glb").exists())
            self.assertFalse((d / "metadata.json").exists())
            for ext in prompt_utils.GS_EXTENSIONS:
                self.assertFalse((d / f"a.{ext}").exists(), ext)
            self.assertFalse(prompt_utils.mesh_complete(d, "a"))

    def test_wipe_mesh_is_idempotent(self):
        """QA can revisit an item whose artifacts are already gone."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.qa.wipe_mesh(d, "a")
            self.qa.wipe_mesh(d, "a")

    def test_verdicts_are_stamped_with_the_contract_version(self):
        self.assertTrue(prompt_utils.verdict_is_current(
            self.qa.stamp({"pass": True})))

    def test_every_failure_reason_wipes(self):
        """The bug was a prefix tuple that matched only 4 of the 7 reasons.

        aspect / dim_ratio / inconsistent-views failures left the rejected GLB
        and its metadata on disk, so mesh_complete() called it a deliverable.
        """
        for reason in ("empty mesh", "aspect 12.4", "dim_ratio 31.0",
                       "white slab? tex_white 0.5", "magenta residue 0.1",
                       "inconsistent views iou 0.05",
                       "empty view coverage 0.001", "load error: boom"):
            with self.subTest(reason=reason):
                self.assertEqual(
                    self.qa.mesh_failure_action([reason], False), "wipe")

    def test_no_wipe_keeps_artifacts(self):
        """Calibration mode must preserve exactly what it exists to inspect.

        The old code deleted the GLB and every 3DGS container and only THEN
        consulted the flag.
        """
        self.assertEqual(
            self.qa.mesh_failure_action(["aspect 12.4"], True), "keep")

    def test_a_passing_mesh_is_never_wiped(self):
        self.assertEqual(self.qa.mesh_failure_action([], False), "keep")

    def test_certified_requires_a_current_pass(self):
        self.assertTrue(self.qa.certified(verdict(True)))
        self.assertFalse(self.qa.certified(verdict(False)))
        self.assertFalse(self.qa.certified({"pass": True}))  # unstamped
        self.assertFalse(self.qa.certified(verdict(True, version=0)))
        self.assertFalse(self.qa.certified({}))

    def test_justified_retries_keeps_only_live_charges(self):
        """A counter feeds item_seed + 7919 * retries, so an obsolete charge
        permanently moves an item off its deterministic seed."""
        kept = self.qa.justified_retries(
            {"current": 1, "obsolete": 2, "quarantined": 3},
            current_ids={"current"}, quarantine={"quarantined": {}})
        self.assertEqual(sorted(kept), ["current", "quarantined"])
        self.assertEqual(kept["current"], 1)

    def test_justified_retries_drops_everything_when_nothing_is_live(self):
        self.assertEqual(
            self.qa.justified_retries({"a": 1}, current_ids=set(),
                                      quarantine={}), {})


class TestNonDeliverables(unittest.TestCase):
    """A release manifest must not claim review artifacts as shipped bytes."""

    def test_qa_montages_and_raw_frames_are_not_deliverables(self):
        for name in ("a.qa_views.png", "a.qa_mesh_views.png", "a.raw.jpg",
                     "a.debug.png", ".a.tmp.glb"):
            self.assertFalse(release_manifest.is_deliverable(Path(name)), name)

    def test_real_outputs_are_deliverables(self):
        for name in ("a.glb", "a.spz", "a.splat", "a.png", "metadata.json"):
            self.assertTrue(release_manifest.is_deliverable(Path(name)), name)

    def test_release_record_excludes_review_artifacts_from_the_byte_total(self):
        with tempfile.TemporaryDirectory() as td:
            it = item("a")
            d = make_complete(td, it)
            (d / "a.qa_views.png").write_bytes(b"x" * 10_000)
            (d / "a.raw.jpg").write_bytes(b"x" * 10_000)
            record = release_manifest.describe_item(it, d)
            self.assertNotIn("a.qa_views.png", record["files"])
            self.assertNotIn("a.raw.jpg", record["files"])
            self.assertLess(record["bytes"], 20_000)


class TestUnrunGateIsNotACertification(unittest.TestCase):
    """qa.py records view_error and still passes the item, because a QA
    infrastructure failure is not an asset defect. But the coverage and
    view-consistency checks provably did not run, so it cannot certify."""

    def test_view_error_verdict_does_not_certify(self):
        v = verdict(True)
        v["view_error"] = "trimesh exploded"
        self.assertFalse(release_manifest._gate_ok({"a": v}, "a"))

    def test_clean_verdict_certifies(self):
        self.assertTrue(release_manifest._gate_ok({"a": verdict(True)}, "a"))

    def test_item_with_an_unrun_mesh_turntable_does_not_ship(self):
        with tempfile.TemporaryDirectory() as td:
            it = item("a")
            d = make_complete(td, it)
            mesh_v = verdict(True, d=d, mode="meshes")
            mesh_v["view_error"] = "boom"
            reason = release_manifest.exclusion_reason(
                it, d, image_qa={"a": verdict(d=d, mode="images")}, mesh_qa={"a": mesh_v},
                gs_qa={"a": verdict(d=d, mode="gaussians")}, quarantine=())
            self.assertEqual(reason, "no current mesh-QA pass")


if __name__ == "__main__":
    unittest.main(verbosity=2)
