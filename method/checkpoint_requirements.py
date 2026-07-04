"""Helpers for validating required model checkpoints before running the pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODEL_KEY_TO_DIRS: dict[str, tuple[str, ...]] = {
    "unet_name": ("diffusion_models", "unet"),
    "clip_name": ("text_encoders", "clip"),
    "vae_name": ("vae",),
    "lora_name": ("loras",),
    "ckpt_name": ("checkpoints",),
}


KNOWN_MODEL_INSTALL_COMMANDS: dict[str, tuple[str, ...]] = {
    "flux-2-klein-9b-fp8.safetensors": (
        "conda activate worldmesh-comfy",
        "hf auth login",
        "hf download black-forest-labs/FLUX.2-klein-9b-fp8 "
        "flux-2-klein-9b-fp8.safetensors --local-dir comfyui/models/diffusion_models",
    ),
    "flux-2-klein-base-9b-fp8.safetensors": (
        "conda activate worldmesh-comfy",
        "hf auth login",
        "hf download black-forest-labs/FLUX.2-klein-base-9b-fp8 "
        "flux-2-klein-base-9b-fp8.safetensors --local-dir comfyui/models/diffusion_models",
    ),
    "flux2-vae.safetensors": (
        "./setup.sh",
    ),
    "full_encoder_small_decoder.safetensors": (
        "conda activate worldmesh-comfy",
        "hf download black-forest-labs/FLUX.2-small-decoder "
        "full_encoder_small_decoder.safetensors --local-dir comfyui/models/vae",
    ),
    "qwen_3_8b_fp8mixed.safetensors": (
        "./setup.sh",
    ),
}


@dataclass(frozen=True)
class CheckpointRequirement:
    """A single required setup artifact."""

    identifier: str
    name: str
    stage: str
    install_commands: tuple[str, ...]
    candidate_paths: tuple[Path, ...] = ()
    referenced_by: tuple[str, ...] = ()
    env_var: str | None = None
    present: bool | None = None

    def is_present(self) -> bool:
        if self.present is not None:
            return self.present
        if self.env_var:
            return bool(os.environ.get(self.env_var, "").strip())
        return any(path.exists() for path in self.candidate_paths)


def sam3_requirement(repo_root: Path, stage: str = "SAM3 segmentation / manual mask creation") -> CheckpointRequirement:
    repo_root = repo_root.resolve()
    return CheckpointRequirement(
        identifier="sam3",
        name="SAM3 checkpoint",
        stage=stage,
        candidate_paths=(repo_root / "sam3" / "checkpoints" / "sam3.pt",),
        install_commands=(
            "conda activate worldmesh-sam3",
            "hf auth login",
            "hf download facebook/sam3 --local-dir sam3/checkpoints",
        ),
    )


def sam3d_requirement(
    repo_root: Path,
    stage: str = "SAM-3D object reconstruction",
) -> CheckpointRequirement:
    repo_root = repo_root.resolve()
    return CheckpointRequirement(
        identifier="sam3d",
        name="SAM-3D-Objects pipeline config",
        stage=stage,
        candidate_paths=(repo_root / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml",),
        install_commands=(
            "conda activate worldmesh-sam3d-objects",
            "hf auth login",
            "hf download facebook/sam-3d-objects --local-dir sam-3d-objects/checkpoints/hf-download",
            "mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf",
            "rm -rf sam-3d-objects/checkpoints/hf-download",
        ),
    )


def depth_pro_requirement(repo_root: Path, stage: str = "Depth validation") -> CheckpointRequirement:
    repo_root = repo_root.resolve()
    return CheckpointRequirement(
        identifier="depth-pro",
        name="Depth Pro checkpoint",
        stage=stage,
        candidate_paths=(repo_root / "ml-depth-pro" / "checkpoints" / "depth_pro.pt",),
        install_commands=(
            "./setup.sh",
        ),
    )


def comfy_api_key_requirement(
    current_value: str | None = None,
    *,
    stage: str = "Final Flux generation via ComfyOrg API",
) -> CheckpointRequirement:
    return CheckpointRequirement(
        identifier="comfy-api-key",
        name="COMFY_API_KEY",
        stage=stage,
        install_commands=(
            'echo \'export COMFY_API_KEY="your_key_here"\' >> ~/.bashrc',
            "source ~/.bashrc",
        ),
        env_var="COMFY_API_KEY",
        present=bool((current_value or "").strip() or os.environ.get("COMFY_API_KEY", "").strip()),
    )


def anthropic_api_key_requirement(
    current_value: str | None = None,
    *,
    stage: str = "Anthropic layout generation",
) -> CheckpointRequirement:
    return CheckpointRequirement(
        identifier="anthropic-api-key",
        name="ANTHROPIC_API_KEY",
        stage=stage,
        install_commands=(
            'echo \'export ANTHROPIC_API_KEY="your_key_here"\' >> ~/.bashrc',
            "source ~/.bashrc",
        ),
        env_var="ANTHROPIC_API_KEY",
        present=bool((current_value or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()),
    )


def gpt_oss_20b_requirement(
    repo_root: Path,
    stage: str = "Local layout generation with gpt-oss-20b",
) -> CheckpointRequirement:
    repo_root = repo_root.resolve()
    return CheckpointRequirement(
        identifier="gpt-oss-20b",
        name="gpt-oss-20b checkpoint",
        stage=stage,
        candidate_paths=(repo_root / "comfyui" / "models" / "llm" / "gpt-oss-20b" / "config.json",),
        install_commands=(
            "./setup.sh",
        ),
    )


def default_flux_workflow_paths(
    repo_root: Path,
    *,
    api: bool,
    include_initial: bool = True,
    include_final: bool = True,
    model: str = "api",
) -> list[Path]:
    """Return the default workflow paths used by the pipeline.

    The ``model`` parameter selects the per-model iterative workflow when
    ``api`` is on and ``model != "api"`` (e.g. ``model="flux2-klein-9b-distilled"``).
    That extra path is appended so the preflight scans the workflow that the
    Stage-6 subprocess will actually use, and surfaces missing checkpoints
    (e.g. ``full_encoder_small_decoder.safetensors``) before expensive stages.
    """
    repo_root = repo_root.resolve()
    workflows_dir = repo_root / "comfyui" / "scenes_workflows"
    paths: list[Path] = []

    if include_initial:
        preferred_initial = workflows_dir / "image_flux2_klein_image_edit_9b_distilled_api.json"
        fallback_initial = workflows_dir / "initial_camera_depth_qwen_api.json"
        paths.append(preferred_initial if preferred_initial.exists() else fallback_initial)

    if include_final:
        if api:
            bootstrap = workflows_dir / "bootstrap_flux2_klein_9b_api_1376w.json"
            iterative = workflows_dir / "image_flux2_klein_image_edit_9b_base_1376w_api.json"
            if not bootstrap.exists():
                bootstrap = workflows_dir / "bootstrap_flux2_klein_9b_api.json"
            if not iterative.exists():
                iterative = workflows_dir / "image_flux2_klein_image_edit_9b_base_1302_api.json"
        else:
            bootstrap = workflows_dir / "bootstrap_flux2_klein_9b_api.json"
            iterative = workflows_dir / "image_flux2_klein_image_edit_9b_base_1302_api.json"

        paths.extend([bootstrap, iterative])

        # When --api is on and --model is something other than "api" (the
        # Nano Banana Pro default), append the per-model iterative workflow
        # from MODEL_REGISTRY so the preflight scans it too. Lazy import
        # avoids a circular dependency with flux_generation/cli.py.
        if api and model != "api":
            try:
                from flux_generation.config import MODEL_REGISTRY
            except ImportError:
                MODEL_REGISTRY = None
            if MODEL_REGISTRY is not None and model in MODEL_REGISTRY:
                paths.append(workflows_dir / MODEL_REGISTRY[model].workflow_filename)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def workflow_model_requirements(
    repo_root: Path,
    workflow_paths: Iterable[Path],
    *,
    stage: str = "Flux generation via ComfyUI",
) -> list[CheckpointRequirement]:
    """Build requirements for model files referenced by ComfyUI workflows."""
    repo_root = repo_root.resolve()
    collected: dict[str, dict[str, object]] = {}

    for workflow_path in workflow_paths:
        if workflow_path is None or not workflow_path.exists():
            continue

        data = json.loads(workflow_path.read_text(encoding="utf-8"))
        for key, filename in _iter_workflow_model_entries(data):
            dirs = MODEL_KEY_TO_DIRS.get(key)
            if not dirs:
                continue

            identifier = f"{key}:{filename}"
            candidate_paths = tuple(
                repo_root / "comfyui" / "models" / directory / filename
                for directory in dirs
            )
            item = collected.setdefault(
                identifier,
                {
                    "name": f"ComfyUI model file: {filename}",
                    "candidate_paths": candidate_paths,
                    "referenced_by": set(),
                    "install_commands": _install_commands_for_model(filename, dirs[0]),
                },
            )
            item["referenced_by"].add(workflow_path.name)

    requirements: list[CheckpointRequirement] = []
    for identifier, item in collected.items():
        requirements.append(
            CheckpointRequirement(
                identifier=identifier,
                name=item["name"],
                stage=stage,
                candidate_paths=item["candidate_paths"],
                install_commands=item["install_commands"],
                referenced_by=tuple(sorted(item["referenced_by"])),
            )
        )
    return sorted(requirements, key=lambda requirement: requirement.name)


def find_missing_requirements(
    requirements: Iterable[CheckpointRequirement],
) -> list[CheckpointRequirement]:
    """Return all missing requirements in stable order."""
    return [requirement for requirement in requirements if not requirement.is_present()]


def format_missing_checkpoints_error(
    missing: Iterable[CheckpointRequirement],
    *,
    heading: str = "Missing required checkpoints",
) -> str:
    """Render a clear user-facing error message for missing setup requirements."""
    missing_list = list(missing)
    if heading == "Missing required checkpoints" and any(req.env_var for req in missing_list):
        heading = "Missing required setup"
    lines = [heading, ""]

    for requirement in missing_list:
        lines.append(f"- {requirement.name}")
        if requirement.env_var:
            lines.append(f"  Expected env var: {requirement.env_var}")
        elif len(requirement.candidate_paths) == 1:
            lines.append(f"  Expected at: {requirement.candidate_paths[0]}")
        else:
            lines.append("  Expected at one of:")
            for path in requirement.candidate_paths:
                lines.append(f"    {path}")
        lines.append(f"  Needed for: {requirement.stage}")
        if requirement.referenced_by:
            lines.append(f"  Referenced by: {', '.join(requirement.referenced_by)}")
        lines.append("  Install:")
        for command in requirement.install_commands:
            lines.append(f"    {command}")
        lines.append("")

    lines.append("Generation cannot continue until the missing setup is available.")
    return "\n".join(lines)


def _iter_workflow_model_entries(node: object) -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in MODEL_KEY_TO_DIRS and isinstance(value, str) and value:
                yield key, value
            yield from _iter_workflow_model_entries(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_workflow_model_entries(item)


def _install_commands_for_model(filename: str, primary_dir: str) -> tuple[str, ...]:
    commands = KNOWN_MODEL_INSTALL_COMMANDS.get(filename)
    if commands is not None:
        return commands
    return (
        f"Download {filename} and place it in comfyui/models/{primary_dir}/",
    )
