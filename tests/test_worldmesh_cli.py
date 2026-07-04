from __future__ import annotations

import contextlib
import io
import os
import shlex
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

import worldmesh_cli
from checkpoint_requirements import CheckpointRequirement


class WorldMeshCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = Path(self.temp_dir.name)

    def _completed(self, args: list[str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args=args or [], returncode=0)

    def _bundle(self, output_dir: Path, scene_name: str = "scene_2rooms_compound") -> worldmesh_cli.JobBundle:
        return worldmesh_cli.JobBundle(
            scene_name=scene_name,
            job_dir=output_dir / "_job_inputs" / scene_name,
            scene_json=output_dir / "_job_inputs" / scene_name / f"{scene_name}.json",
            prompts_file=output_dir / "_job_inputs" / scene_name / "prompts.txt",
            theme="Quiet Scandinavian loft",
        )

    def test_wizard_uses_default_bundled_scene(self) -> None:
        output_dir = self.work_dir / "wizard_output"
        with mock.patch.object(worldmesh_cli, "DEFAULT_OUTPUT_DIR", output_dir), mock.patch(
            "worldmesh_cli.subprocess.run",
            return_value=self._completed(),
        ) as run_mock, mock.patch("worldmesh_cli.abort_if_missing_checkpoints"):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [],
                # Option 1 (defaults): select option, accept bundled theme, confirm start
                input="\n\n\n",
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("scene_4rooms_zigzag", result.output)
        command = run_mock.call_args.args[0]
        generated_output_dir = Path(command[command.index("--output-dir") + 1])
        prompts_file = generated_output_dir / "_job_inputs" / "scene_4rooms_zigzag" / "prompts.txt"
        self.assertTrue(prompts_file.exists())
        self.assertEqual(
            prompts_file.read_text(encoding="utf-8"),
            "Crazy, Surreal, Beautiful Bubblegum Reef\n",
        )

        self.assertIn("--api", command)
        self.assertIn("scene_4rooms_zigzag", command)
        self.assertIn("--placement-mode", command)
        self.assertIn("simple", command)

    def test_generate_dry_run_uses_scene12_theme_and_defaults(self) -> None:
        output_dir = self.work_dir / "generate_existing"
        with mock.patch("worldmesh_cli.abort_if_missing_checkpoints"):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--existing-scene",
                    "scene_12rooms_grid",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Beautiful elegant ancient Roman estate with animating mosaics", result.output)
        self.assertIn("--depth-threshold 0.7", result.output)
        self.assertIn("--min-bootstrap-attempts 1", result.output)
        self.assertIn("--fov-final 60", result.output)
        self.assertIn("--api", result.output)
        self.assertIn("--placement-mode simple", result.output)

        prompts_file = output_dir / "_job_inputs" / "scene_12rooms_grid" / "prompts.txt"
        self.assertTrue(prompts_file.exists())
        self.assertEqual(
            prompts_file.read_text(encoding="utf-8"),
            "Beautiful elegant ancient Roman estate with animating mosaics\n",
        )

    def test_generate_dry_run_custom_scene_uses_custom_theme(self) -> None:
        source_scene = worldmesh_cli.SCENES_DIR / "scene_4rooms_zigzag.json"
        custom_scene = self.work_dir / "my_custom_scene.json"
        shutil.copy2(source_scene, custom_scene)

        output_dir = self.work_dir / "generate_custom"
        with mock.patch("worldmesh_cli.abort_if_missing_checkpoints"):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--scene-json",
                    str(custom_scene),
                    "--theme",
                    "Quiet Scandinavian loft",
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Quiet Scandinavian loft", result.output)
        self.assertIn("my_custom_scene", result.output)
        self.assertIn("--placement-mode simple", result.output)

        prompts_file = output_dir / "_job_inputs" / "my_custom_scene" / "prompts.txt"
        self.assertTrue(prompts_file.exists())
        self.assertEqual(prompts_file.read_text(encoding="utf-8"), "Quiet Scandinavian loft\n")

    def test_generate_placement_mode_smart_override(self) -> None:
        output_dir = self.work_dir / "generate_smart"
        with mock.patch("worldmesh_cli.abort_if_missing_checkpoints"):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--existing-scene",
                    "scene_4rooms_zigzag",
                    "--output-dir",
                    str(output_dir),
                    "--placement-mode",
                    "smart",
                    "--dry-run",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--placement-mode smart", result.output)
        self.assertNotIn("--placement-mode simple", result.output)

    def test_build_resume_command_quotes_theme_and_uses_saved_bundle_json(self) -> None:
        output_dir = self.work_dir / "resume_output"
        bundle = worldmesh_cli.JobBundle(
            scene_name="scene_2rooms_compound",
            job_dir=output_dir / "_job_inputs" / "scene_2rooms_compound",
            scene_json=output_dir / "_job_inputs" / "scene_2rooms_compound" / "scene_2rooms_compound.json",
            prompts_file=output_dir / "_job_inputs" / "scene_2rooms_compound" / "prompts.txt",
            theme="Crazy, Surreal: Bubblegum Reef & chrome chairs",
        )

        resume_cmd = worldmesh_cli.build_resume_command(
            bundle,
            output_dir,
            api=True,
            reconstruction=False,
            depth_threshold=0.7,
            min_bootstrap_attempts=1,
            fov_final=60,
            placement_mode="simple",
            verbose=False,
        )

        self.assertIn(str(bundle.scene_json), resume_cmd)
        self.assertIn(shlex.quote(bundle.theme), resume_cmd)
        self.assertIn("--api", resume_cmd)
        self.assertIn("--depth-threshold 0.7", resume_cmd)

    def test_generate_writes_resume_file_with_saved_bundle_json_and_theme(self) -> None:
        output_dir = self.work_dir / "generate_resume_existing"
        with mock.patch("worldmesh_cli.abort_if_missing_checkpoints"), mock.patch(
            "worldmesh_cli.subprocess.run",
            return_value=self._completed(),
        ):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--existing-scene",
                    "scene_12rooms_grid",
                    "--output-dir",
                    str(output_dir),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        resume_text = (output_dir / "RESUME.txt").read_text(encoding="utf-8")
        self.assertIn(
            str(output_dir / "_job_inputs" / "scene_12rooms_grid" / "scene_12rooms_grid.json"),
            resume_text,
        )
        self.assertIn(
            shlex.quote("Beautiful elegant ancient Roman estate with animating mosaics"),
            resume_text,
        )
        self.assertIn("--output-dir", resume_text)
        self.assertNotIn("--existing-scene", resume_text)

    def test_generate_writes_resume_file_using_copied_job_json_for_custom_scene(self) -> None:
        source_scene = worldmesh_cli.SCENES_DIR / "scene_4rooms_zigzag.json"
        custom_scene = self.work_dir / "my custom scene.json"
        shutil.copy2(source_scene, custom_scene)
        output_dir = self.work_dir / "generate_resume_custom"

        with mock.patch("worldmesh_cli.abort_if_missing_checkpoints"), mock.patch(
            "worldmesh_cli.subprocess.run",
            return_value=self._completed(),
        ):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--scene-json",
                    str(custom_scene),
                    "--theme",
                    "Quiet Scandinavian loft",
                    "--output-dir",
                    str(output_dir),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        resume_text = (output_dir / "RESUME.txt").read_text(encoding="utf-8")
        copied_json = output_dir / "_job_inputs" / "my custom scene" / "my custom scene.json"
        self.assertIn(str(copied_json), resume_text)
        self.assertIn(shlex.quote("Quiet Scandinavian loft"), resume_text)
        self.assertNotIn(str(custom_scene), resume_text)

    def test_print_completion_box_uses_saved_viewer_command(self) -> None:
        output_dir = self.work_dir / "completed_scene"
        bundle = self._bundle(output_dir)
        scene_output = output_dir / bundle.scene_name
        nerfstudio_output = scene_output / "nerfstudio_output"
        nerfstudio_output.mkdir(parents=True)
        config_path = nerfstudio_output / "unnamed" / "depth-splatfacto" / "2026-04-15_165350" / "config.yml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("trainer: {}\n", encoding="utf-8")
        view_file = scene_output / "view_scene.txt"
        viewer_cmd = f"ns-viewer --load-config {config_path}"
        view_file.write_text(viewer_cmd + "\n", encoding="utf-8")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            worldmesh_cli.print_completion_box(output_dir, bundle)

        rendered = stdout.getvalue()
        self.assertIn(viewer_cmd, rendered)
        self.assertIn(str(nerfstudio_output), rendered)
        self.assertIn(str(view_file), rendered)
        self.assertNotIn("splatfacto/config.yml", rendered)

    def test_print_completion_box_falls_back_to_latest_nerfstudio_config(self) -> None:
        output_dir = self.work_dir / "completed_scene"
        bundle = self._bundle(output_dir)
        scene_output = output_dir / bundle.scene_name
        first_config = (
            scene_output
            / "nerfstudio_output"
            / "unnamed"
            / "depth-splatfacto"
            / "2026-04-15_164520"
            / "config.yml"
        )
        second_config = (
            scene_output
            / "nerfstudio_output"
            / "unnamed"
            / "depth-splatfacto"
            / "2026-04-15_165350"
            / "config.yml"
        )
        first_config.parent.mkdir(parents=True)
        second_config.parent.mkdir(parents=True)
        first_config.write_text("trainer: {}\n", encoding="utf-8")
        second_config.write_text("trainer: {}\n", encoding="utf-8")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            worldmesh_cli.print_completion_box(output_dir, bundle)

        rendered = stdout.getvalue()
        self.assertIn(f"ns-viewer --load-config {second_config}", rendered)
        self.assertNotIn(f"ns-viewer --load-config {first_config}", rendered)
        self.assertNotIn("splatfacto/config.yml", rendered)

    def test_print_completion_box_warns_when_viewer_command_missing(self) -> None:
        output_dir = self.work_dir / "completed_scene"
        bundle = self._bundle(output_dir)
        scene_output = output_dir / bundle.scene_name
        scene_output.mkdir(parents=True)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            worldmesh_cli.print_completion_box(output_dir, bundle)

        rendered = stdout.getvalue()
        self.assertIn("Viewer command unavailable", rendered)
        self.assertNotIn("splatfacto/config.yml", rendered)

    def test_layouts_create_api_requires_key_before_subprocess(self) -> None:
        with mock.patch("worldmesh_cli.subprocess.run") as run_mock, mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "layouts",
                    "create-api",
                    "--prompt",
                    "A four-room courtyard house",
                    "--output-dir",
                    str(self.work_dir / "api_layouts"),
                ],
            )

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Missing required setup", result.output)
        self.assertIn("ANTHROPIC_API_KEY", result.output)
        run_mock.assert_not_called()

    def test_make_api_layout_name_adds_suffix_when_timestamp_name_exists(self) -> None:
        output_dir = self.work_dir / "api_layouts"
        output_dir.mkdir()
        (output_dir / "scene_llm_20260415_163500.json").write_text("{}", encoding="utf-8")

        with mock.patch("worldmesh_cli.datetime") as datetime_mock:
            datetime_mock.now.return_value.strftime.return_value = "20260415_163500"
            name = worldmesh_cli.make_api_layout_name(output_dir)

        self.assertEqual(name, "scene_llm_20260415_163500_2")

    def test_layouts_create_api_generates_unique_name_and_verifies_outputs(self) -> None:
        output_dir = self.work_dir / "api_layouts"
        output_dir.mkdir()
        (output_dir / "scene_llm_20260415_163500.json").write_text("old", encoding="utf-8")
        (output_dir / "scene_llm_20260415_163500.png").write_text("old", encoding="utf-8")

        def fake_run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
            self.assertEqual(cwd, worldmesh_cli.REPO_ROOT)
            name = cmd[cmd.index("--name") + 1]
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (out_dir / f"{name}.json").write_text("{}", encoding="utf-8")
            (out_dir / f"{name}.png").write_text("png", encoding="utf-8")
            return self._completed(cmd)

        with mock.patch("worldmesh_cli.subprocess.run", side_effect=fake_run) as run_mock, mock.patch(
            "worldmesh_cli.datetime"
        ) as datetime_mock, mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=True,
        ):
            datetime_mock.now.return_value.strftime.return_value = "20260415_163500"
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "layouts",
                    "create-api",
                    "--prompt",
                    "A six-room apartment",
                    "--output-dir",
                    str(output_dir),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Created layout:", result.output)
        self.assertIn("scene_llm_20260415_163500_2.json", result.output)
        run_cmd = run_mock.call_args.args[0]
        self.assertIn("--name", run_cmd)
        self.assertIn("scene_llm_20260415_163500_2", run_cmd)
        self.assertTrue((output_dir / "scene_llm_20260415_163500_2.json").exists())
        self.assertTrue((output_dir / "scene_llm_20260415_163500_2.png").exists())

    def test_layouts_create_api_with_explicit_name_accepts_overwrite(self) -> None:
        output_dir = self.work_dir / "api_layouts_explicit"
        output_dir.mkdir()
        (output_dir / "manual_layout.json").write_text("old", encoding="utf-8")
        (output_dir / "manual_layout.png").write_text("old", encoding="utf-8")

        def fake_run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
            self.assertEqual(cwd, worldmesh_cli.REPO_ROOT)
            name = cmd[cmd.index("--name") + 1]
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            (out_dir / f"{name}.json").write_text("new", encoding="utf-8")
            (out_dir / f"{name}.png").write_text("new", encoding="utf-8")
            return self._completed(cmd)

        with mock.patch("worldmesh_cli.subprocess.run", side_effect=fake_run) as run_mock, mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key"},
            clear=True,
        ):
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "layouts",
                    "create-api",
                    "--prompt",
                    "A four-room courtyard house",
                    "--output-dir",
                    str(output_dir),
                    "--name",
                    "manual_layout",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("manual_layout.json", result.output)
        run_cmd = run_mock.call_args.args[0]
        self.assertIn("--name", run_cmd)
        self.assertIn("manual_layout", run_cmd)
        self.assertEqual(
            (output_dir / "manual_layout.json").read_text(encoding="utf-8"),
            "new",
        )

    def test_generate_missing_checkpoints_exits_before_subprocess(self) -> None:
        output_dir = self.work_dir / "missing_checkpoints"
        missing = [
            CheckpointRequirement(
                identifier="sam3d",
                name="SAM-3D-Objects pipeline config",
                stage="Stage 4: Extract Objects",
                install_commands=("hf download facebook/sam-3d-objects --local-dir sam-3d-objects/checkpoints/hf-download",),
                candidate_paths=(self.work_dir / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml",),
                present=False,
            )
        ]
        with mock.patch("worldmesh_cli.find_missing_cli_checkpoints", return_value=missing), mock.patch(
            "worldmesh_cli.subprocess.run"
        ) as run_mock:
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--existing-scene",
                    "scene_4rooms_zigzag",
                    "--output-dir",
                    str(output_dir),
                ],
            )

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Missing required checkpoints", result.output)
        self.assertIn("SAM-3D-Objects pipeline config", result.output)
        run_mock.assert_not_called()

    def test_generate_missing_comfy_api_key_exits_before_subprocess(self) -> None:
        output_dir = self.work_dir / "missing_comfy_key"
        missing = [
            CheckpointRequirement(
                identifier="comfy-api-key",
                name="COMFY_API_KEY",
                stage="Final Flux generation via ComfyOrg API",
                install_commands=('echo \'export COMFY_API_KEY="your_key_here"\' >> ~/.bashrc',),
                env_var="COMFY_API_KEY",
                present=False,
            )
        ]
        with mock.patch("worldmesh_cli.find_missing_cli_checkpoints", return_value=missing), mock.patch(
            "worldmesh_cli.subprocess.run"
        ) as run_mock:
            result = self.runner.invoke(
                worldmesh_cli.app,
                [
                    "generate",
                    "--existing-scene",
                    "scene_4rooms_zigzag",
                    "--output-dir",
                    str(output_dir),
                ],
            )

        self.assertEqual(result.exit_code, 1, result.output)
        self.assertIn("Missing required setup", result.output)
        self.assertIn("COMFY_API_KEY", result.output)
        run_mock.assert_not_called()

    def test_find_missing_cli_checkpoints_requires_comfy_key_only_for_final_api_generation(self) -> None:
        present_requirement = CheckpointRequirement(
            identifier="present",
            name="present",
            stage="test",
            install_commands=("./setup.sh",),
            present=True,
        )
        with mock.patch("worldmesh_cli.sam3_requirement", return_value=present_requirement), mock.patch(
            "worldmesh_cli.sam3d_requirement", return_value=present_requirement
        ), mock.patch("worldmesh_cli.depth_pro_requirement", return_value=present_requirement), mock.patch(
            "worldmesh_cli.workflow_model_requirements", return_value=[]
        ), mock.patch("worldmesh_cli.default_flux_workflow_paths", return_value=[]), mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            missing_masks = worldmesh_cli.find_missing_cli_checkpoints(
                phase=worldmesh_cli.GenerationPhase.masks,
                api=True,
                reconstruction=False,
            )
            missing_reconstruction = worldmesh_cli.find_missing_cli_checkpoints(
                phase=worldmesh_cli.GenerationPhase.generate,
                api=True,
                reconstruction=True,
            )
            missing_final = worldmesh_cli.find_missing_cli_checkpoints(
                phase=worldmesh_cli.GenerationPhase.generate,
                api=True,
                reconstruction=False,
            )
            missing_no_api = worldmesh_cli.find_missing_cli_checkpoints(
                phase=worldmesh_cli.GenerationPhase.generate,
                api=False,
                reconstruction=False,
            )

        self.assertFalse(any(req.name == "COMFY_API_KEY" for req in missing_masks))
        self.assertFalse(any(req.name == "COMFY_API_KEY" for req in missing_reconstruction))
        self.assertTrue(any(req.name == "COMFY_API_KEY" for req in missing_final))
        self.assertFalse(any(req.name == "COMFY_API_KEY" for req in missing_no_api))


if __name__ == "__main__":
    unittest.main()
