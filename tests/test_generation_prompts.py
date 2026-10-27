"""Prompt regressions: mechanical structure, semantic identity and consumer parity."""

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

import content_policy  # noqa: E402
import generation_prompts as gp  # noqa: E402
import prompt_specs  # noqa: E402
from pipeline_utils import load_json  # noqa: E402

spec = importlib.util.spec_from_file_location("rewrite_prompts", ROOT / "scripts/rewrite_generation_prompts.py")
rewrite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewrite)


class TestGenerationPrompts(unittest.TestCase):
    def test_cargo_octocopter_has_explicit_count_mounts_and_motor_axes(self):
        prompt = gp.compose_prompt("heavy-lift cargo delivery drone, octocopter design with cargo pod, medical supplies", transparent=True)
        for expected in ("Exactly eight equal radial booms", "concentric", "parallel vertical shafts", "common height", "stationary"):
            self.assertIn(expected, prompt)
        for wrong in ("hydraulic cylinders", "welded steel", "tire chains", "sandbag rack", "spinning-disc effect"):
            self.assertNotIn(wrong, prompt)
        self.assertTrue(prompt.startswith(gp.RGBA_PREFIX))
        self.assertIn("Subject: An octocopter with exactly eight separate motor/rotor units", prompt)
        self.assertIn("all eight separate boom-end motor housings countable", prompt)
        self.assertIn("central 70 percent", prompt)

    def test_payload_does_not_select_the_asset_family(self):
        cases = [("cargo drone, medical supplies and a folded tent", "multirotor"),
                 ("rescue robot with a drone launch rack", "robot"),
                 ("wildfire equipment cache, fire shelters and pumps", "supply_set"),
                 ("blanket pallet, insulated blankets", "supply_set"),
                 ("mapping UAV, fixed-wing design", "aircraft"),
                 ("Black Hornet nano drone, camera payload", "helicopter")]
        for subject, family in cases:
            with self.subTest(subject=subject):
                self.assertEqual(gp.subject_family(subject), family)

    def test_coaxial_rotors_are_not_forced_into_one_flat_eight_arm_layout(self):
        prompt = gp.compose_prompt("cargo drone, coaxial octocopter with four arms")
        self.assertIn("Exactly 8 motor/rotor units", prompt)
        self.assertNotIn("The 8 arms", prompt)
        self.assertNotIn("one common horizontal plane", prompt)
        helicopter = gp.compose_prompt("rescue helicopter, tail rotor")
        self.assertIn("different intended axes", helicopter)
        self.assertNotIn("parallel and vertical", helicopter)

    def test_hardware_and_primitives_get_applicable_geometry(self):
        cases = [("Hex socket bolt M8", "industrial_fasteners", "thread/head designation"),
                 ("USB connector", "electronics_connectors", "contact count"),
                 ("rotary servo motor", "motors_actuators", "standalone motor"),
                 ("red cube, 5cm sides", "primitive_cube", "Planar faces remain flat")]
        for subject, category, expected in cases:
            self.assertIn(expected, gp.compose_prompt(subject, category=category))

    def test_repeated_tent_variants_keep_the_authored_design_and_one_condition(self):
        for ordinal in range(52):
            prompt = gp.compose_prompt("inflatable emergency shelter, air-beam tunnel tent", ordinal=ordinal, condition="field-used")
            self.assertIn("Inflated main air beams", prompt)
            self.assertNotIn("A-frame ridge", prompt)
            self.assertNotIn("wheels", prompt)
            self.assertEqual(prompt.count("The surface "), 1)

    def test_backend_adaptation_changes_only_the_background_and_is_idempotent(self):
        prompt = gp.compose_prompt("white ambulance with red cross markings")
        self.assertEqual(gp.normalize_generation_prompt(prompt), prompt)
        transparent = gp.normalize_generation_prompt(prompt, transparent=True)
        self.assertEqual(gp.normalize_generation_prompt(transparent, transparent=True), transparent)
        self.assertEqual(gp.normalize_generation_prompt(transparent), prompt)
        self.assertIn("red cross markings", transparent)
        self.assertNotIn("no logos", transparent)
        self.assertEqual(transparent.count("Geometry: "), 1)

    def test_all_six_image_prompt_consumers_preserve_the_shared_recipe(self):
        cases = [("scripts/generate_relief.py", "enhance_prompt"),
                 ("meshmaker/generate_meshes.py", "enhance_image_prompt"),
                 ("meshmaker/generate_meshes_parallel.py", "enhance_prompt"),
                 ("meshmaker/generate_meshes_robust.py", "enhance_prompt"),
                 ("meshmaker/batch_submit.py", "enhance_prompt"),
                 ("meshmaker/batch_submit_v2.py", "enhance_prompt")]
        prompt = gp.compose_prompt("mobile clinic container, white with red cross")
        for filename, name in cases:
            tree = ast.parse((ROOT / filename).read_text())
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            namespace = {"normalize_generation_prompt": gp.normalize_generation_prompt}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
            with self.subTest(filename=filename):
                self.assertEqual(namespace[name](prompt), prompt)

    def test_rewrite_preserves_source_pose_and_physics_metadata(self):
        row = {"id": "mug", "category": "pose_variation", "mass_kg": 0.2, "dimensions": "10 cm tall",
               "text_prompt": "Blue ceramic mug positioned upside down, resting on its rim, white background",
               "image_prompt": "Blue mug upside down, handle visible, photorealistic"}
        result = gp.rewrite_item(row)
        for field in ("text_prompt", "image_prompt"):
            self.assertIn("upside down", result[field])
            self.assertEqual(result["generation_prompt"]["sources"][field], row[field])
        self.assertEqual(result["mass_kg"], row["mass_kg"])
        self.assertEqual(result["dimensions"], row["dimensions"])
        self.assertEqual(gp.rewrite_item(result), result)

    def test_policy_screens_both_original_and_rewritten_wording(self):
        row = {"id": "weapon", "name": "asset", "category": "equipment", "prompt": "portable water pump",
               "generation_prompt": {"sources": {"prompt": "G36 assault rifle"}}}
        self.assertEqual(content_policy.classify_item(row).classification, content_policy.BLOCKED)
        row["generation_prompt"]["sources"]["prompt"] = "portable water pump"
        row["prompt"] = "G36 assault rifle"
        self.assertEqual(content_policy.classify_item(row).classification, content_policy.BLOCKED)

    def test_document_rewrite_preserves_identity_and_is_idempotent(self):
        before = {"version": "2.0", "meshes": [{"id": "a", "name": "Cargo Drone (variant)", "geography": "general",
                   "category": "uav", "tags": ["relief"], "prompt": "cargo drone, medical supplies. The object stands alone on white."}]}
        catalogue = {"Cargo Drone": "cargo drone, octocopter design with cargo pod"}
        after = rewrite.rewrite_document(before, catalogue)
        rewrite.validate_document(before, after)
        self.assertEqual(after, rewrite.rewrite_document(after, catalogue))
        self.assertEqual(after["meshes"][0]["id"], "a")
        self.assertIn("exactly eight separate motor/rotor units", after["meshes"][0]["prompt"])

    def test_standalone_drone_motor_and_franka_hand_do_not_gain_whole_machines(self):
        cases = [("brushless drone motor, connector view", "motors_actuators", "motor", "standalone_motor"),
                 ("Franka Emika Panda robot parallel gripper, two fingers", "robot_part", "gripper", "franka_parallel_hand")]
        for subject, category, family, profile in cases:
            out = gp.rewrite_item({"id": "part", "category": category, "prompt": subject})
            self.assertEqual(out["generation_prompt"]["family"], family)
            self.assertEqual(out["generation_prompt"]["specifications"]["prompt"]["profile"], profile)
            self.assertNotIn("View: A high three-quarter", out["prompt"])
            self.assertNotIn("locomotion", out["prompt"])

    def test_distinct_real_objects_are_not_flattened_by_their_category(self):
        cases = [("Golden croissant, curved French pastry", "primitive_crescent", "food_object"),
                 ("Red plastic storage container, lid open", "container", "small_storage_container"),
                 ("Open cardboard box with flaps up", "container", "small_storage_container"),
                 ("Red plastic hemisphere, half sphere bowl shape", "primitive_hemisphere", "primitive_hemisphere"),
                 ("Toy dog figurine, plastic dog", "animal", "molded_animal_figurine"),
                 ("water purification plant, pumps and filters", "infrastructure", "generator_pump"),
                 ("Heavy duty hydraulic part, hydraulic cylinder heavy duty, variation 1", "industrial_hydraulics", "linear_actuator")]
        for subject, category, profile in cases:
            spec = prompt_specs.resolve_spec(subject, category=category)
            with self.subTest(subject=subject):
                self.assertEqual(spec["profile"], profile)

    def test_rotor_count_is_not_an_arm_count_or_layout_invariant(self):
        prompt = gp.compose_prompt("generic survey drone, eight rotors")
        self.assertNotIn("equal length", prompt)
        self.assertNotIn("one common horizontal plane", prompt)
        self.assertNotIn("8 arms", prompt)
        row = gp.rewrite_item({"id": "flycart", "prompt": "DJI FlyCart 30 cargo drone"})
        prompt = row["prompt"]
        self.assertIn("Four deployed arms", prompt)
        self.assertIn("eight rotors in four paired stacks", prompt)
        self.assertNotIn("eight equal radial booms", prompt)

    def test_explicit_aircraft_set_retains_each_airframes_rotor_count(self):
        row = gp.rewrite_item({"id": "swarm", "name": "Swarm Mapping Drone System",
                               "prompt": "drone swarm system, 5 small coordinated quadcopters"})
        self.assertEqual(row["generation_prompt"]["family"], "object_set")
        self.assertIn("Exactly five separate complete quadcopters", row["prompt"])
        self.assertIn("four motor/propeller units", row["prompt"])

    def test_package_pin_count_and_closed_body_survive_a_bad_generic_source(self):
        out = gp.rewrite_item({"id": "ic", "category": "electronics_ics", "prompt":
                             "Electronic component DIP16 ic, black epoxy, metal pins, silicon chip, white background"})
        self.assertIn("exactly 16 through-hole legs", out["prompt"])
        self.assertIn("two parallel rows of 8", out["prompt"])
        self.assertIn("die stays inside", out["prompt"])
        self.assertIn("ti_package_families", out["generation_prompt"]["specifications"]["prompt"]["reference_evidence"])

    def test_background_style_and_meaningless_variation_labels_are_removed(self):
        prompt = gp.compose_prompt("black matte panel, variation 12, macro view, clean gray background")
        self.assertNotIn("variation 12", prompt)
        self.assertNotIn("macro view", prompt)
        self.assertNotIn("clean gray background", prompt)
        self.assertEqual(prompt.count("Background: "), 1)

    def test_unnamed_machines_are_not_claimed_as_exact_research_replicas(self):
        spec = prompt_specs.resolve_spec("Brokk remote demolition robot, hydraulic breaker")
        self.assertEqual(spec["scope"], "manufacturer_family_example")
        self.assertIn("does not establish an exact model", spec["design_choice"])
        spec = prompt_specs.resolve_spec("large field hospital tent", name="Field Hospital Tent Large")
        self.assertIn("lanco_sg200", spec["sources"])
        self.assertNotIn("lanco_azf", spec["sources"])

    def test_manifest_regeneration_preserves_integrated_legacy_rows(self):
        import contextlib
        import io
        import json
        spec = importlib.util.spec_from_file_location("prompt_catalogue_test", ROOT / "scripts/generate_prompts.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "relief_manifest_germany.json"
            legacy = {"id": "legacy_drone", "name": "quad", "prompt": "reconnaissance quadcopter", "category": "legacy_aircraft"}
            path.write_text(json.dumps({"meshes": [legacy]}))
            with contextlib.redirect_stdout(io.StringIO()):
                generator.write_manifest(path, [{"id": "new", "prompt": "water pump"}], "test", "germany")
            rows = load_json(path)["meshes"]
            self.assertEqual([row["id"] for row in rows], ["new", "legacy_drone"])
            self.assertEqual(rows[1]["generation_prompt"]["sources"]["prompt"], legacy["prompt"])


if __name__ == "__main__":
    unittest.main()
