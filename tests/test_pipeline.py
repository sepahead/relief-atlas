"""Self-tests for the local relief-atlas pipeline.

Runs with the system interpreter and needs only numpy/PIL from the stack, so it
can gate a commit without a 40GB model workspace:

    python3 -m unittest discover -s tests -v
    python3 tests/test_pipeline.py

Each test here corresponds to a defect that actually shipped. The point is not
coverage; it is that these specific mistakes cannot recur silently:

  * the policy families misclassifying humanitarian mine-action vocabulary,
  * an exemption laundering a weapon named elsewhere in the same row,
  * the policy flags drifting apart between phases, so --allow-restricted
    generated 540 items that the next step then quarantined,
  * the 3DGS extension list being copied per-script until one copy went stale
    and the unattended driver looped forever,
  * transient export names losing the real file suffix, which made trimesh
    refuse the container and killed the whole mesh phase.
"""

import argparse
import ast
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_LOCAL = REPO_ROOT / "scripts" / "local"
sys.path.insert(0, str(SCRIPTS_LOCAL))

import content_policy  # noqa: E402
import prompt_utils  # noqa: E402


class TestPolicyClassification(unittest.TestCase):
    """The regression table in content_policy.SELFTEST."""

    def test_regression_table(self):
        failures = []
        for expected, text in content_policy.SELFTEST:
            got = content_policy.classify_text(text)
            if got.classification != expected:
                failures.append(f"{text!r}: expected {expected}, got {got.summary()}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_humanitarian_vocabulary_survives(self):
        """Mine action keeps its mine/ordnance words without being read as a weapon."""
        for text in (
            "handheld metal detector for landmine detection",
            "explosive ordnance disposal robot for UXO clearance",
            "mine clearance flail vehicle for humanitarian demining",
            "bomb disposal suit with blast blanket",
        ):
            self.assertEqual(
                content_policy.classify_text(text).classification,
                content_policy.ALLOWED, text)

    def test_exemption_cannot_launder_a_weapon(self):
        """A benign compound phrase must not excuse a weapon named elsewhere."""
        for text in (
            "handheld landmine detector carried by a soldier with a G36 assault rifle",
            "water bomber aircraft escorted by a Tornado IDS fighter",
            "torpedo-shaped ROV stored beside a rack of ammunition",
        ):
            self.assertEqual(
                content_policy.classify_text(text).classification,
                content_policy.BLOCKED, text)

    def test_exemptions_are_compound_phrases_not_bare_nouns(self):
        """A bare-noun exemption would silently permit real weapons.

        Exemptions can only ever narrow the weapon family, so a loose entry is
        unsafe in a way that is invisible at the call site.
        """
        for pattern in content_policy.WEAPON_EXEMPTIONS:
            self.assertTrue(
                r"\s" in pattern or "[-" in pattern or "[" in pattern,
                f"exemption {pattern!r} looks like a bare noun")


class TestGateSemantics(unittest.TestCase):
    """permitted() / screen_items() behaviour, including overrides."""

    def _item(self, name, prompt="", **kw):
        return {"id": kw.get("id", "x"), "name": name, "prompt": prompt,
                "category": kw.get("category", "c")}

    def test_closed_by_default(self):
        v = content_policy.classify_text("G36 assault rifle")
        self.assertFalse(v.permitted(allow_restricted=False))
        self.assertFalse(v.permitted(allow_restricted=True))

    def test_allow_blocked_opens_blocked(self):
        v = content_policy.classify_text("G36 assault rifle")
        self.assertTrue(v.permitted(allow_restricted=False, allow_blocked=True))

    def test_allow_blocked_implies_restricted(self):
        """Permitting an armed platform while refusing an unarmed truck is incoherent."""
        v = content_policy.classify_text("military truck for disaster logistics")
        self.assertEqual(v.classification, content_policy.RESTRICTED)
        self.assertFalse(v.permitted(allow_restricted=False))
        self.assertTrue(v.permitted(allow_restricted=False, allow_blocked=True))

    def test_allowed_items_are_not_marked_override(self):
        items = [self._item("water pump trailer")]
        permitted, rejected = content_policy.screen_items(items, False)
        self.assertEqual(len(permitted), 1)
        self.assertEqual(rejected, [])
        self.assertFalse(permitted[0]["policy"]["override"])

    def test_override_is_stamped_on_permitted_non_allowed_items(self):
        """The stamp is the only trace on the asset itself that a flag was used."""
        items = [self._item("Leopard 2A7 main battle tank")]
        permitted, rejected = content_policy.screen_items(
            items, allow_restricted=False, allow_blocked=True)
        self.assertEqual(rejected, [])
        self.assertTrue(permitted[0]["policy"]["override"])
        self.assertEqual(permitted[0]["policy"]["classification"],
                         content_policy.BLOCKED)

    def test_refused_items_carry_their_verdict(self):
        items = [self._item("G36 assault rifle")]
        permitted, rejected = content_policy.screen_items(items, True)
        self.assertEqual(permitted, [])
        self.assertEqual(rejected[0]["policy"]["classification"],
                         content_policy.BLOCKED)
        self.assertFalse(rejected[0]["policy"]["override"])


class TestTierSelection(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"id": "a", "name": "water pump", "prompt": "", "category": "c"},
            {"id": "b", "name": "military truck", "prompt": "", "category": "c"},
            {"id": "c", "name": "G36 assault rifle", "prompt": "", "category": "c"},
        ]

    def test_none_is_a_passthrough(self):
        self.assertEqual(
            len(content_policy.select_tier(self.items, None)), 3)

    def test_selects_single_tier(self):
        self.assertEqual(
            [it["id"] for it in content_policy.select_tier(self.items, "blocked")],
            ["c"])
        self.assertEqual(
            [it["id"] for it in content_policy.select_tier(self.items, "restricted")],
            ["b"])

    def test_flagged_is_everything_not_allowed(self):
        self.assertEqual(
            sorted(it["id"] for it in
                   content_policy.select_tier(self.items, "flagged")),
            ["b", "c"])

    def test_idempotent(self):
        once = content_policy.select_tier(self.items, "blocked")
        twice = content_policy.select_tier(once, "blocked")
        self.assertEqual([it["id"] for it in once], [it["id"] for it in twice])


class TestPolicyFlagContract(unittest.TestCase):
    """The flags must round-trip, and every phase must expose them.

    This is the bug that motivated the shared helper: run_all.py declared
    --allow-restricted and forwarded it to the generators but not to qa.py, so
    restricted items were generated and then quarantined as policy failures by
    the next step in the same command.
    """

    def _parser(self):
        return content_policy.add_policy_args(argparse.ArgumentParser())

    def test_defaults_emit_no_flags(self):
        args = self._parser().parse_args([])
        self.assertEqual(content_policy.policy_argv(args), [])

    def test_every_flag_round_trips(self):
        for argv in (
            ["--allow-restricted"],
            ["--allow-blocked"],
            ["--allow-restricted", "--allow-blocked"],
            ["--only-policy", "blocked"],
            ["--allow-blocked", "--only-policy", "blocked"],
        ):
            with self.subTest(argv=argv):
                args = self._parser().parse_args(argv)
                emitted = content_policy.policy_argv(args)
                # Re-parsing what we emit must produce the same decisions.
                again = self._parser().parse_args(emitted)
                self.assertEqual(args.allow_restricted, again.allow_restricted)
                self.assertEqual(args.allow_blocked, again.allow_blocked)
                self.assertEqual(args.only_policy, again.only_policy)

    def test_only_policy_rejects_unknown_tier(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["--only-policy", "allowed"])

    def _calls_add_policy_args(self, filename):
        """True if the module calls content_policy.add_policy_args anywhere.

        Parsed rather than imported: flux_imagegen and trellis_meshgen import
        scipy / trellis2 at module level, which a bare interpreter lacks.
        """
        tree = ast.parse((SCRIPTS_LOCAL / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None)
                if name == "add_policy_args":
                    return True
        return False

    def test_all_phases_register_the_shared_flags(self):
        for filename in ("flux_imagegen.py", "qa.py", "trellis_meshgen.py",
                         "run_all.py"):
            with self.subTest(script=filename):
                self.assertTrue(
                    self._calls_add_policy_args(filename),
                    f"{filename} must use content_policy.add_policy_args so the "
                    f"policy flags cannot drift between phases")

    def test_orchestrator_forwards_policy_flags_to_qa(self):
        """run_all.py must hand the policy flags to qa.py, not just the generators."""
        source = (SCRIPTS_LOCAL / "run_all.py").read_text()
        self.assertIn("policy_argv", source)
        self.assertIn("qa_common", source)


class TestGaussianContainers(unittest.TestCase):
    """The shared extension list, and what 'complete' means."""

    def test_spz_and_splat_are_recognised(self):
        for ext in ("spz", "splat", "ply"):
            self.assertIn(ext, prompt_utils.GS_EXTENSIONS)

    def test_has_gaussians_accepts_any_container(self):
        for ext in prompt_utils.GS_EXTENSIONS:
            with tempfile.TemporaryDirectory() as td, self.subTest(ext=ext):
                d = Path(td)
                self.assertFalse(prompt_utils.has_gaussians(d, "item"))
                (d / f"item.{ext}").write_bytes(b"x")
                self.assertTrue(prompt_utils.has_gaussians(d, "item"))

    def test_mesh_complete_requires_glb_gaussians_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.assertFalse(prompt_utils.mesh_complete(d, "item"))

            (d / "item.glb").write_bytes(b"0" * 2048)
            self.assertFalse(prompt_utils.mesh_complete(d, "item"),
                             "GLB alone is not a complete deliverable")

            (d / "item.spz").write_bytes(b"x")
            self.assertFalse(prompt_utils.mesh_complete(d, "item"),
                             "metadata.json is part of the deliverable")

            (d / "metadata.json").write_text("{}")
            self.assertTrue(prompt_utils.mesh_complete(d, "item"))

    def test_mesh_complete_rejects_truncated_glb(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "item.glb").write_bytes(b"0" * 10)
            (d / "item.spz").write_bytes(b"x")
            (d / "metadata.json").write_text("{}")
            self.assertFalse(prompt_utils.mesh_complete(d, "item"))


class TestGsExport(unittest.TestCase):
    """Transient naming and the PLY layout."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("numpy unavailable")
        import gs_export
        cls.gs = gs_export

    def test_tmp_sibling_preserves_the_real_suffix(self):
        """Exporters infer the container from the extension.

        The naive `b.glb.tmp` made trimesh raise "unsupported export format:
        tmp" and take the entire mesh phase down with it.
        """
        for name, expected in (
            ("a/b.glb", ".b.tmp.glb"),
            ("a/b.spz", ".b.tmp.spz"),
            ("a/b.splat", ".b.tmp.splat"),
            ("a/b.ply", ".b.tmp.ply"),
        ):
            with self.subTest(name=name):
                tmp = self.gs.tmp_sibling(name)
                self.assertEqual(tmp.name, expected)
                self.assertEqual(tmp.suffix, Path(name).suffix)
                self.assertEqual(tmp.parent, Path(name).parent)

    def test_tmp_sibling_is_hidden_and_matches_the_sweep_glob(self):
        """verify_local.py sweeps '*/*/*/.*.tmp.*' for stale transients."""
        tmp = self.gs.tmp_sibling("g/c/i/item.glb")
        self.assertTrue(tmp.name.startswith("."))
        self.assertTrue(tmp.match(".*.tmp.*"))

    def _gaussians(self, n=4):
        import numpy as np
        return {
            "x": np.arange(n, dtype=np.float32),
            "y": np.zeros(n, dtype=np.float32),
            "z": np.zeros(n, dtype=np.float32),
            "f_dc": np.zeros((n, 3), dtype=np.float32),
            "f_rest": np.zeros((n, 45), dtype=np.float32),
            "opacity": np.zeros((n, 1), dtype=np.float32),
            "scale": np.full((n, 3), -1.0, dtype=np.float32),
            "rot": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        }

    def test_ply_roundtrip_header_and_payload_size(self):
        n = 4
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.ply"
            written = self.gs.write_ply(str(path), self._gaussians(n))
            self.assertEqual(written, n)

            blob = path.read_bytes()
            header, _, payload = blob.partition(b"end_header\n")
            self.assertTrue(header.startswith(b"ply\n"))
            self.assertIn(b"format binary_little_endian 1.0", header)
            self.assertIn(f"element vertex {n}".encode(), header)
            # The header's property count and the payload stride MUST agree.
            # They did not: 59 properties were declared and 62 floats written,
            # so readers resynchronised onto garbage after the first gaussian.
            self.assertEqual(header.count(b"property float "),
                             self.gs.PLY_STRIDE)
            self.assertEqual(len(payload), n * self.gs.PLY_STRIDE * 4)

    def test_ply_layout_is_canonical_inria_order(self):
        """position, normal, SH DC, SH rest, opacity, scale, rotation = 62."""
        self.assertEqual(self.gs.PLY_STRIDE, 62)
        self.assertEqual(self.gs.PLY_PROPS[:6], ["x", "y", "z", "nx", "ny", "nz"])
        self.assertEqual(self.gs.PLY_PROPS[6:9], ["f_dc_0", "f_dc_1", "f_dc_2"])
        self.assertEqual(self.gs.PLY_PROPS[54], "opacity")
        self.assertEqual(self.gs.PLY_PROPS[-4:],
                         ["rot_0", "rot_1", "rot_2", "rot_3"])

    def test_ply_payload_places_fields_where_the_header_says(self):
        """Read the binary back and confirm each field landed in its column."""
        import numpy as np
        n = 3
        g = self._gaussians(n)
        g["opacity"][:] = 0.25
        g["f_dc"][:] = 0.5
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.ply"
            self.gs.write_ply(str(path), g)
            payload = path.read_bytes().partition(b"end_header\n")[2]
            arr = np.frombuffer(payload, dtype=np.float32).reshape(n, 62)

            np.testing.assert_allclose(arr[:, 0], g["x"])
            np.testing.assert_allclose(arr[:, 3:6], 0.0)      # normals
            np.testing.assert_allclose(arr[:, 6:9], 0.5)      # f_dc
            np.testing.assert_allclose(arr[:, 54], 0.25)      # opacity
            np.testing.assert_allclose(arr[:, 55:58], -1.0)   # scale
            np.testing.assert_allclose(arr[:, 58], 1.0)       # rot w
            np.testing.assert_allclose(arr[:, 59:62], 0.0)    # rot xyz

    def test_write_ply_and_write_ply_to_agree(self):
        """They were byte-identical copies; keep them that way by construction."""
        import io
        g = self._gaussians()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.ply"
            self.gs.write_ply(str(path), g)
            buf = io.BytesIO()
            self.gs.write_ply_to(buf, g)
            self.assertEqual(path.read_bytes(), buf.getvalue())

    def test_splat_is_32_bytes_per_gaussian(self):
        n = 5
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.splat"
            self.assertEqual(self.gs.write_splat(str(path), self._gaussians(n)), n)
            self.assertEqual(path.stat().st_size, n * 32)

    @staticmethod
    def _spz_available():
        """Import guard: a broken third-party PyPI `spz` must count as absent."""
        try:
            import spz  # noqa: F401
            return True
        except Exception:
            return False

    def test_write_spz_roundtrips_on_the_real_niantic_api(self):
        """The original body called an API no spz binding ever had
        (GaussianCloud().albedos/opacities + saveSplatToPath), so the write
        always raised and the .splat fallback hid it: zero .spz ever shipped.
        Against the real binding the pack must save and load back the same
        cloud, with opacity surviving as a pre-sigmoid logit and colors as RGB.
        """
        import numpy as np
        if not self._spz_available():
            self.skipTest("spz (Niantic binding) not installed")
        import spz
        n = 4
        g = self._gaussians(n)
        rgb = np.array([1.0, 0.2, 0.0], dtype=np.float32)
        g["f_dc"][:] = (rgb - 0.5) / self.gs.C0
        g["opacity"][:] = 2.0  # logit -> sigmoid ~0.88
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "g.spz"
            self.assertEqual(self.gs.write_spz(str(path), g), n)
            cloud = spz.load_spz(str(path))
            self.assertEqual(cloud.num_points, n)
            self.assertEqual(np.asarray(cloud.positions).size, n * 3)
            # pack quantizes; sign/magnitude must survive, not exact bits
            np.testing.assert_allclose(np.asarray(cloud.alphas), 2.0, atol=0.25)
            np.testing.assert_allclose(
                np.asarray(cloud.colors).reshape(n, 3)[0], rgb, atol=0.06)

    def _stub_voxel(self, n=20):
        """Duck-typed MeshWithVoxel: tensors expose cpu().numpy()."""
        import types
        import numpy as np

        class Arr:
            def __init__(self, a):
                self._a = a

            def cpu(self):
                return self

            def numpy(self):
                return self._a

        coords = np.zeros((n, 4), dtype=np.int64)
        coords[:, 0] = 0
        coords[:, 1] = np.arange(n) % 5
        coords[:, 2] = (np.arange(n) // 5) % 5
        coords[:, 3] = np.arange(n) // 25
        attrs = np.full((n, 6), 0.5, dtype=np.float32)
        attrs[:, 5] = 0.9  # alpha channel
        return types.SimpleNamespace(
            coords=Arr(coords), attrs=Arr(attrs),
            origin=Arr(np.zeros(3, dtype=np.float32)), voxel_size=0.1,
            layout={"base_color": slice(0, 3), "metallic": 3,
                    "roughness": 4, "alpha": 5})

    def test_export_writes_both_containers_when_spz_available(self):
        """Every item gets the universal .splat; .spz joins it when the
        Niantic binding is installed. Both must be promotable from their
        transient sibling names."""
        spz_ok = self._spz_available()
        with tempfile.TemporaryDirectory() as td:
            stem = str(Path(td) / "item")
            paths, cnt = self.gs.export_gaussians_from_mesh_with_voxel(
                self._stub_voxel(), stem)
            self.assertGreater(cnt, 0)
            expected = [".splat"] if not spz_ok else [".spz", ".splat"]
            self.assertEqual([p.suffix for p in paths], expected)
            # pre-promotion names: transients on disk, finals not yet
            for p in paths:
                self.assertFalse(p.exists())
                self.assertTrue(self.gs.tmp_sibling(p).exists())
            for p in paths:
                self.gs.tmp_sibling(p).replace(p)
            for p in paths:
                self.assertGreater(p.stat().st_size, 0)


class TestPromptCleaning(unittest.TestCase):
    def test_deterministic(self):
        p = ("THW rescue truck, blue markings. The surface is pristine "
             "factory-new: clean intact finish. The object stands alone on a "
             "white background.")
        self.assertEqual(prompt_utils.clean_prompt(p, "id_1"),
                         prompt_utils.clean_prompt(p, "id_1"))

    def test_studio_boilerplate_is_replaced(self):
        p = ("THW rescue truck. " + prompt_utils.STUDIO_SPLIT
             + " on a pure white seamless backdrop with gentle contact shadows.")
        out = prompt_utils.clean_prompt(p, "id_1")
        self.assertNotIn("pure white seamless", out)
        self.assertNotIn("contact shadows", out)
        self.assertIn("neutral gray", out)
        self.assertIn("no floor", out)

    def test_wear_clause_detection(self):
        """Used to drop the manifest surface sentence when the per-item clause
        also describes finish, so the two cannot contradict each other."""
        for clause in prompt_utils.FALLBACK_VARIATIONS:
            self.assertTrue(prompt_utils.WEAR_CLAUSE_RE.search(clause), clause)

    def test_rotorcraft_gets_the_stationary_rotor_clause(self):
        out = prompt_utils.clean_prompt(
            "Flood monitoring quadcopter drone with camera.", "id_1")
        self.assertIn("stationary", out)

    def test_clean_name_strips_template_variant_text(self):
        """Manifest names splice a shared variant clause after the base name;
        the pipeline metadata must carry the base name, not the clause."""
        self.assertEqual(
            prompt_utils.clean_name(
                "DRK Mobile Blood Donation Unit (It is equipped for flood "
                "response with water pump attachments...)"),
            "DRK Mobile Blood Donation Unit")
        self.assertEqual(prompt_utils.clean_name("Simple Name"), "Simple Name")


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _healthy_gaussians(n=6000):
    """Cube cloud, dense voxel-style splats: identity rotation, log scale
    for a ~0.1 voxel, gray color, opacity logit ~2.2 (sigmoid ~0.9)."""
    import numpy as np
    side = int(round(n ** (1 / 3)))
    xs = np.linspace(-0.5, 0.5, side)
    gx, gy, gz = np.meshgrid(xs, xs, xs)
    pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32)
    pos = pos[:n]
    return {
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "scale": np.full((len(pos), 3), np.log(0.06), dtype=np.float32),
        "rot": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(pos), 1)),
        "f_dc": np.zeros((len(pos), 3), dtype=np.float32),
        "opacity": np.full((len(pos), 1), 2.2, dtype=np.float32),
    }


class TestGaussianQA(unittest.TestCase):
    """Turntable + primitive QA on the 3DGS containers.

    The numpy-level checks (primitive stats, silhouette metrics, montage,
    container loaders) run on a bare interpreter; the torch rasteriser and
    trimesh ray-cast tests self-skip without their stack and are exercised by
    running the suite under the trellis venv.
    """

    @classmethod
    def setUpClass(cls):
        if not (_have("numpy") and _have("PIL")):
            raise unittest.SkipTest("numpy/PIL unavailable")
        import qa
        cls.qa = qa

    def test_primitive_stats_pass_on_healthy_cloud(self):
        stats, reasons = self.qa.gaussian_primitive_stats(_healthy_gaussians())
        self.assertEqual(reasons, [])
        self.assertEqual(stats["nan_frac"], 0.0)
        self.assertGreater(stats["mean_opacity"], 0.8)

    def test_nan_positions_fail(self):
        import numpy as np
        g = _healthy_gaussians()
        g["x"][0] = np.nan
        _, reasons = self.qa.gaussian_primitive_stats(g)
        self.assertTrue(any("non-finite" in r for r in reasons))

    def test_dead_splats_fail(self):
        import numpy as np
        g = _healthy_gaussians()
        g["opacity"][:] = -20.0  # sigmoid ~= 0
        _, reasons = self.qa.gaussian_primitive_stats(g)
        self.assertTrue(any("dead splats" in r for r in reasons))

    def test_degenerate_scales_fail(self):
        import numpy as np
        g = _healthy_gaussians()
        g["scale"][:, 0] = 30.0  # exp -> astronomically large
        _, reasons = self.qa.gaussian_primitive_stats(g)
        self.assertTrue(any("degenerate scales" in r for r in reasons))

    def test_splat_loader_roundtrips_gs_export_writer(self):
        """write_splat -> _load_splat_gaussians must recover the cloud."""
        import gs_export
        import numpy as np
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "g.splat"
            gs_export.write_splat(str(p), _healthy_gaussians(100))
            g = self.qa._load_splat_gaussians(p)
            self.assertEqual(len(g["x"]), 100)
            np.testing.assert_allclose(g["x"][:3], np.asarray(g["x"])[:3])
            # gray f_dc=0 -> rgb 0.5 -> opacity logit ~2.2 recovered
            # u8 quantization: gray 0.5 +/- 1/255 survives, exact bits don't
            np.testing.assert_allclose(g["f_dc"], 0.0, atol=0.01)
            np.testing.assert_allclose(g["opacity"].ravel(), 2.2, atol=0.1)

    def _disc_mask(self, res=64, r=18, cx=None, cy=None, hole=False):
        import numpy as np
        ys, xs = np.mgrid[0:res, 0:res]
        cx = res / 2 if cx is None else cx
        cy = res / 2 if cy is None else cy
        m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        if hole:
            m &= (xs - cx) ** 2 + (ys - cy) ** 2 >= (r * 0.35) ** 2
        return m

    def test_silhouette_metrics_pass_on_solid_discs(self):
        import numpy as np
        masks = [self._disc_mask(cx=30 + 4 * i % 8) for i in range(16)]
        wmaps = [m.astype(np.float32) for m in masks]
        stats = self.qa.silhouette_view_metrics(masks, wmaps)
        self.assertGreater(stats["coverage_min"], self.qa.GS_COV_MIN)
        self.assertLess(stats["holes_max"], self.qa.GS_HOLE_MAX)
        self.assertGreater(stats["iou_min"], self.qa.GS_IOU_MIN)
        self.assertEqual(self.qa.gaussians_view_reasons(stats), [])

    def test_empty_view_fails_coverage(self):
        import numpy as np
        masks = [self._disc_mask() for _ in range(15)]
        masks.append(np.zeros((64, 64), dtype=bool))  # the missing side
        stats = self.qa.silhouette_view_metrics(masks)
        self.assertTrue(any("empty view" in r
                            for r in self.qa.gaussians_view_reasons(stats)))

    def test_enclosed_hole_fails(self):
        masks = [self._disc_mask(hole=True) for _ in range(16)]
        stats = self.qa.silhouette_view_metrics(masks)
        self.assertGreater(stats["holes_max"], self.qa.GS_HOLE_MAX)
        self.assertTrue(any("holes" in r
                            for r in self.qa.gaussians_view_reasons(stats)))

    def test_montage_writes_a_contact_sheet(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "views.png"
            views = [self._disc_mask() for _ in range(16)]
            self.qa.write_view_montage(views, p, "test", 64,
                                       [str(i) for i in range(16)])
            im = Image.open(p)
            self.assertEqual(im.size, (4 * 64, 4 * 64 + 22))
            self.assertEqual(im.mode, "RGB")

    def test_render_splat_views_turntable(self):
        """The torch rasteriser: the 10-lens turntable (8-azimuth ring +
        top/bottom poles), sane coverage, finite colour. Runs under the
        trellis venv; self-skips elsewhere."""
        if not _have("torch"):
            self.skipTest("torch unavailable")
        import numpy as np
        rendered = self.qa.render_splat_views(_healthy_gaussians(6000))
        self.assertIsNotNone(rendered)
        masks, weights, colours = rendered
        self.assertEqual(len(masks), 10)
        stats = self.qa.silhouette_view_metrics(masks, weights, colours)
        self.assertGreater(stats["coverage_min"], 0.02)
        self.assertLess(stats["coverage_mean"], 0.9)
        self.assertGreater(stats["solidity_min"], 0.3)
        for c in colours:
            self.assertTrue(np.isfinite(c).all())


class TestMeshViews(unittest.TestCase):
    """Turntable polygon-raster silhouettes for the mesh pass."""

    @classmethod
    def setUpClass(cls):
        if not (_have("numpy") and _have("trimesh")):
            raise unittest.SkipTest("numpy/trimesh unavailable")
        import qa
        cls.qa = qa

    def test_box_renders_from_all_views(self):
        import numpy as np
        import trimesh
        box = trimesh.creation.box(extents=(2.0, 1.0, 0.8))
        masks, shades = self.qa.mesh_silhouettes([box])
        self.assertEqual(len(masks), 10)
        covs = [float(m.mean()) for m in masks]
        self.assertGreater(min(covs), self.qa.GS_COV_MIN)
        self.assertLess(max(covs), self.qa.GS_COV_MAX)
        stats = self.qa.silhouette_view_metrics(masks)
        self.assertEqual(self.qa.mesh_view_reasons(stats), [])

    def test_degenerate_flat_mesh_fails_view_consistency(self):
        """A mesh collapsed to a single plane looks like a wall from some
        views and a line from others: adjacent-view silhouettes stop
        overlapping and the view-consistency check fires."""
        import numpy as np
        import trimesh
        box = trimesh.creation.box(extents=(2.0, 1.0, 0.8))
        centroids = box.triangles_center
        box.update_faces(np.abs(centroids[:, 0] - 1.0) < 0.01)  # +x face only
        masks, _ = self.qa.mesh_silhouettes([box])
        stats = self.qa.silhouette_view_metrics(masks)
        self.assertLess(stats["iou_min"], self.qa.GS_IOU_MIN)
        self.assertTrue(any("inconsistent" in r
                            for r in self.qa.mesh_view_reasons(stats)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
