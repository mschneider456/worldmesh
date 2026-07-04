#!/usr/bin/env python3
"""User-friendly WorldMesh CLI built with Typer."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent / "method"))
from checkpoint_requirements import (
    anthropic_api_key_requirement,
    comfy_api_key_requirement,
    default_flux_workflow_paths,
    depth_pro_requirement,
    find_missing_requirements,
    format_missing_checkpoints_error,
    gpt_oss_20b_requirement,
    sam3_requirement,
    sam3d_requirement,
    workflow_model_requirements,
)
from run_batch_pipeline import discover_scenes


app = typer.Typer(
    add_completion=False,
    help=(
        "Guided WorldMesh scene generation. Run without a subcommand to start "
        "the interactive wizard."
    ),
    no_args_is_help=False,
)
layouts_app = typer.Typer(
    help="Browse or create scene layouts before running generation.",
)
app.add_typer(layouts_app, name="layouts")


REPO_ROOT = Path(__file__).resolve().parent
SCENES_DIR = REPO_ROOT / "method" / "scenes"
PROMPTS_PATH = SCENES_DIR / "prompts.txt"

DEFAULT_EXISTING_SCENE = "scene_4rooms_zigzag"
DEFAULT_OUTPUT_DIR = Path("user_worlds")
DEFAULT_LAYOUT_OUTPUT_DIR = Path("user_layouts")
DEFAULT_PORT = 7860
DEFAULT_DEPTH_THRESHOLD = 0.7
DEFAULT_MIN_BOOTSTRAP_ATTEMPTS = 1
DEFAULT_FOV_FINAL = 60
DEFAULT_PLACEMENT_MODE = "simple"
DEFAULT_THEME = "Crazy, Surreal, Beautiful Bubblegum Reef"

# User-facing image-generation model names mapped to the internal
# (api_flag, --model value) pair forwarded to run_full_pipeline.py.
# Both currently set api=True because flux2-klein-9b-distilled uses the
# 1376w workflow stack that --api enables.
IMAGE_MODEL_NANO_BANANA_PRO = "nano-banana-pro"
IMAGE_MODEL_FLUX2_KLEIN_9B = "flux2-klein-9b"
IMAGE_MODEL_FLUX2_KLEIN_9B_BASE = "flux2-klein-9b-base"
IMAGE_MODEL_CHOICES: dict[str, tuple[bool, str]] = {
    IMAGE_MODEL_NANO_BANANA_PRO: (True, "api"),
    IMAGE_MODEL_FLUX2_KLEIN_9B: (True, "flux2-klein-9b-distilled"),
    IMAGE_MODEL_FLUX2_KLEIN_9B_BASE: (True, "flux2-klein-9b-base"),
}
DEFAULT_IMAGE_MODEL = IMAGE_MODEL_NANO_BANANA_PRO

# Layout LLM selection.
LAYOUT_MODEL_CLAUDE_OPUS_4_6 = "claude-opus-4-6"
LAYOUT_MODEL_GPT_OSS_20B = "gpt-oss-20b"
LAYOUT_MODEL_CHOICES = (LAYOUT_MODEL_CLAUDE_OPUS_4_6, LAYOUT_MODEL_GPT_OSS_20B)
DEFAULT_LAYOUT_MODEL = LAYOUT_MODEL_CLAUDE_OPUS_4_6
# Conda env that holds transformers + the gpt-oss-20b weights.
WORLDMESH_LLM_ENV = "worldmesh-llm"

_EXCLUDED_THEME_SUBSTRINGS = {
    "crystallized volcano",
    "subterranean ice cavern",
    "paper origami temple",
    "post-apocalyptic greenhouse cathedral",
}


class LayoutSource(str, Enum):
    bundled = "bundled"
    generated = "generated"
    api = "api"


class GenerationPhase(str, Enum):
    all = "all"
    masks = "masks"
    generate = "generate"


class LayoutShape(str, Enum):
    compound = "compound"
    corridor = "corridor"
    grid = "grid"
    linear = "linear"
    lshape = "lshape"
    plus = "plus"
    rectangle = "rectangle"
    tshape = "tshape"
    ushape = "ushape"
    zigzag = "zigzag"


@dataclass(frozen=True)
class SceneChoice:
    scene_name: str
    scene_json: Path
    theme: str


@dataclass(frozen=True)
class ResolvedScene:
    scene_name: str
    scene_json: Path
    theme: str
    source: LayoutSource
    description: str


@dataclass(frozen=True)
class JobBundle:
    scene_name: str
    job_dir: Path
    scene_json: Path
    prompts_file: Path
    theme: str


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def conda_env_exists(env_name: str) -> bool:
    """Return True if a conda env named env_name exists on this system."""
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError):
        return False
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts and parts[0] == env_name:
            return True
    return False


def format_missing_worldmesh_llm_env_error() -> str:
    return (
        "Missing required setup\n\n"
        "- worldmesh-llm conda env\n"
        f"  Expected name: {WORLDMESH_LLM_ENV}\n"
        "  Needed for: --layout-model gpt-oss-20b\n"
        "  Install:\n"
        "    ./setup.sh\n"
        "    # (creates worldmesh-llm, installs torch/transformers/kernels,\n"
        "    #  and downloads gpt-oss-20b)\n\n"
        "Generation cannot continue until the missing setup is available."
    )


def resolve_user_path(path: Path) -> Path:
    return path.expanduser().resolve()


def run_command(cmd: list[str], description: str) -> None:
    typer.echo(f"\n{description}")
    typer.echo(f"Command: {format_command(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


def find_missing_cli_checkpoints(
    *,
    phase: GenerationPhase,
    api: bool,
    reconstruction: bool,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> list:
    requirements = []

    if phase in (GenerationPhase.masks, GenerationPhase.all):
        requirements.append(sam3_requirement(REPO_ROOT))

    if phase in (GenerationPhase.generate, GenerationPhase.all):
        requirements.append(sam3d_requirement(REPO_ROOT))

    needs_flux = phase in (GenerationPhase.masks, GenerationPhase.all)
    if phase is GenerationPhase.generate and not reconstruction:
        needs_flux = True

    if needs_flux:
        _, model_key = IMAGE_MODEL_CHOICES[image_model]
        workflow_paths = default_flux_workflow_paths(
            REPO_ROOT,
            api=api,
            include_initial=phase in (GenerationPhase.masks, GenerationPhase.all),
            include_final=phase in (GenerationPhase.generate, GenerationPhase.all),
            model=model_key,
        )
        requirements.append(depth_pro_requirement(REPO_ROOT))
        requirements.extend(workflow_model_requirements(REPO_ROOT, workflow_paths))
        # COMFY_API_KEY is only required when the chosen image model actually
        # calls the ComfyOrg API. flux2-klein-9b is fully local.
        uses_comfy_api = image_model == IMAGE_MODEL_NANO_BANANA_PRO
        if api and uses_comfy_api and phase in (GenerationPhase.all, GenerationPhase.generate) and not reconstruction:
            requirements.append(
                comfy_api_key_requirement(stage="Final Flux generation via ComfyOrg API")
            )

    return find_missing_requirements(requirements)


def abort_if_missing_checkpoints(
    *,
    phase: GenerationPhase,
    api: bool,
    reconstruction: bool,
    dry_run: bool,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> None:
    missing = find_missing_cli_checkpoints(
        phase=phase,
        api=api,
        reconstruction=reconstruction,
        image_model=image_model,
    )
    if not missing:
        return

    message = format_missing_checkpoints_error(missing)
    color = typer.colors.YELLOW if dry_run else typer.colors.RED
    typer.secho("\n" + message, fg=color)
    if not dry_run:
        raise typer.Exit(1)


def load_scene_catalog() -> list[SceneChoice]:
    pairs = discover_scenes(SCENES_DIR, PROMPTS_PATH)
    return [
        SceneChoice(
            scene_name=scene_json.stem,
            scene_json=scene_json.resolve(),
            theme=theme,
        )
        for scene_json, theme in pairs
    ]


def scene_catalog_by_name() -> dict[str, SceneChoice]:
    return {scene.scene_name: scene for scene in load_scene_catalog()}


def print_scene_catalog(catalog: list[SceneChoice]) -> None:
    typer.secho("Available scene layouts", fg=typer.colors.CYAN, bold=True)
    typer.echo("2D floor plan previews are available as PNG files alongside each layout.\n")
    for index, scene in enumerate(catalog, start=1):
        suffix = "  [recommended]" if scene.scene_name == DEFAULT_EXISTING_SCENE else ""
        png_path = scene.scene_json.parent / f"{scene.scene_name}.png"
        png_note = f"  (preview: {png_path})" if png_path.exists() else ""
        typer.echo(f"{index:>2}. {scene.scene_name}{suffix}{png_note}")


def prompt_for_menu_choice(title: str, options: list[str], default_index: int) -> int:
    typer.secho(title, fg=typer.colors.CYAN, bold=True)
    for index, option in enumerate(options, start=1):
        marker = "  [default]" if index == default_index else ""
        typer.echo(f"{index}. {option}{marker}")

    while True:
        choice = typer.prompt("Enter a number", default=str(default_index)).strip()
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(options):
                return selected
        typer.secho("Please enter one of the numbers shown above.", fg=typer.colors.RED)


def prompt_for_existing_scene() -> Path:
    """Display the layout catalog (scenes + user_layouts) and return the chosen JSON path."""
    catalog_entries: list[tuple[str, Path]] = []

    for scene in load_scene_catalog():
        label = scene.scene_name
        if scene.scene_name == DEFAULT_EXISTING_SCENE:
            label += "  [recommended]"
        catalog_entries.append((label, scene.scene_json))

    user_scenes_dir = REPO_ROOT / DEFAULT_LAYOUT_OUTPUT_DIR
    if user_scenes_dir.exists():
        for json_path in sorted(user_scenes_dir.glob("*.json")):
            catalog_entries.append((f"{json_path.stem}  [user-created]", json_path.resolve()))

    typer.secho("\nAvailable scene layouts", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        "2D floor plan previews are available as PNG files alongside each layout.\n"
        "Open any .png file to see the room arrangement before making your choice.\n"
    )
    for idx, (label, json_path) in enumerate(catalog_entries, 1):
        png_path = json_path.with_suffix(".png")
        png_note = f"  →  {png_path}" if png_path.exists() else ""
        typer.echo(f"{idx:>2}. {label}{png_note}")

    default_index = next(
        (i for i, (_, p) in enumerate(catalog_entries, 1) if p.stem == DEFAULT_EXISTING_SCENE),
        1,
    )
    typer.secho("\nChoose a scene layout", fg=typer.colors.CYAN, bold=True)
    while True:
        choice = typer.prompt("Enter a number", default=str(default_index)).strip()
        if choice.isdigit():
            sel = int(choice)
            if 1 <= sel <= len(catalog_entries):
                _, selected_path = catalog_entries[sel - 1]
                open_preview_png(selected_path.with_suffix(".png"))
                return selected_path
        typer.secho("Please enter one of the numbers shown above.", fg=typer.colors.RED)


def load_curated_themes() -> list[str]:
    """Load themes from prompts.txt, excluding the four Crazy/Surreal entries."""
    if not PROMPTS_PATH.exists():
        return []
    themes = []
    for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if any(exc in line.lower() for exc in _EXCLUDED_THEME_SUBSTRINGS):
            continue
        themes.append(line)
    return themes


def make_run_name(theme: str) -> str:
    """Generate a short timestamped folder name from the theme string."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _STOPWORDS = {
        "a", "an", "the", "and", "with", "in", "of", "at", "by", "for",
        "from", "into", "on", "to", "or", "as", "its", "but", "so",
        "where", "all", "is", "are", "was", "were",
    }
    words = re.sub(r"[^\w\s]", "", theme.lower()).split()
    keywords = [w for w in words if w not in _STOPWORDS][:4]
    abbreviated = "-".join(keywords) if keywords else "scene"
    return f"{timestamp}_{abbreviated}"


def make_api_layout_name(output_dir: Path) -> str:
    """Generate a unique filesystem-safe layout name for API-created scenes."""
    base_name = f"scene_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = base_name
    suffix = 2

    while (
        (output_dir / f"{candidate}.json").exists()
        or (output_dir / f"{candidate}.png").exists()
    ):
        candidate = f"{base_name}_{suffix}"
        suffix += 1

    return candidate


def open_preview_png(png_path: Path) -> None:
    """Print the floor plan preview path and try to open it with the system viewer."""
    typer.echo(f"  Floor plan preview: {png_path}")
    if png_path.exists():
        try:
            subprocess.Popen(
                ["xdg-open", str(png_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def prompt_for_layout_model_menu() -> str:
    """Present a 2-option menu: claude-opus-4-6 (recommended) vs gpt-oss-20b (local)."""
    typer.secho("\nChoose a layout LLM:", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        "  1. claude-opus-4-6  [recommended]  -- best quality; uses the Anthropic API (requires ANTHROPIC_API_KEY).\n"
        "  2. gpt-oss-20b                     -- fully local in the worldmesh-llm env; no API key.\n"
    )
    options = [LAYOUT_MODEL_CLAUDE_OPUS_4_6, LAYOUT_MODEL_GPT_OSS_20B]
    while True:
        choice = typer.prompt("Enter a number", default="1").strip()
        if choice.isdigit():
            sel = int(choice)
            if 1 <= sel <= len(options):
                return options[sel - 1]
        typer.secho("Please enter 1 or 2.", fg=typer.colors.RED)


def prompt_for_image_model_menu() -> str:
    """Present a 3-option menu: Nano Banana Pro (recommended) vs flux2-klein-9b (fast local) vs flux2-klein-9b-base (slow but higher-quality local)."""
    typer.secho("\nChoose an image-generation model:", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        "  1. nano-banana-pro      [recommended]  -- best quality overall; uses ComfyOrg API (requires COMFY_API_KEY).\n"
        "  2. flux2-klein-9b                      -- fully local via ComfyUI; distilled UNet; no API key; fastest local option.\n"
        "  3. flux2-klein-9b-base                 -- fully local via ComfyUI; undistilled UNet; no API key; higher quality than\n"
        "                                            the distilled variant but ~5x slower per iterative camera (20 sampler\n"
        "                                            steps + CFG=5 vs 4 steps at CFG=1).\n"
    )
    options = [IMAGE_MODEL_NANO_BANANA_PRO, IMAGE_MODEL_FLUX2_KLEIN_9B, IMAGE_MODEL_FLUX2_KLEIN_9B_BASE]
    while True:
        choice = typer.prompt("Enter a number", default="1").strip()
        if choice.isdigit():
            sel = int(choice)
            if 1 <= sel <= len(options):
                return options[sel - 1]
        typer.secho("Please enter 1, 2, or 3.", fg=typer.colors.RED)


def prompt_for_theme_menu() -> str:
    """Present a numbered menu of curated themes; last option is free-text entry."""
    curated = load_curated_themes()

    def _norm(s: str) -> str:
        return s.strip().rstrip(".")

    themes = [DEFAULT_THEME] + [t for t in curated if _norm(t) != _norm(DEFAULT_THEME)]
    options = themes + ["Enter my own theme"]

    typer.secho("\nChoose a visual theme:", fg=typer.colors.CYAN, bold=True)
    typer.echo("The AI will use this theme to style every room in the scene.\n")
    for idx, opt in enumerate(options, 1):
        marker = "  [default]" if idx == 1 else ""
        typer.echo(f"{idx:>2}. {opt}{marker}")

    while True:
        choice = typer.prompt("\nEnter a number", default="1").strip()
        if choice.isdigit():
            sel = int(choice)
            if 1 <= sel <= len(options):
                if sel == len(options):
                    return typer.prompt("Your theme").strip()
                return options[sel - 1]
        typer.secho("Please enter one of the numbers shown above.", fg=typer.colors.RED)


def resolve_scene(
    existing_scene: Optional[str],
    scene_json: Optional[Path],
    theme: Optional[str],
    output_dir: Optional[Path] = None,
) -> ResolvedScene:
    if existing_scene and scene_json:
        raise typer.BadParameter(
            "Use either --existing-scene or --scene-json, not both."
        )

    if scene_json:
        resolved_json = resolve_user_path(scene_json)
        if not resolved_json.exists():
            raise typer.BadParameter(f"Scene JSON not found: {resolved_json}")
        return ResolvedScene(
            scene_name=resolved_json.stem,
            scene_json=resolved_json,
            theme=(theme or "").strip(),
            source=LayoutSource.generated,
            description=f"custom layout from {resolved_json}",
        )

    # Auto-resume: if no scene was specified explicitly but output_dir already
    # has a materialised job bundle, reuse that scene instead of silently
    # falling back to DEFAULT_EXISTING_SCENE (which would overwrite the
    # existing run with a different layout).
    if existing_scene is None and output_dir is not None:
        job_inputs_dir = output_dir / "_job_inputs"
        if job_inputs_dir.is_dir():
            scene_subdirs = sorted(
                entry for entry in job_inputs_dir.iterdir() if entry.is_dir()
            )
            matching = [
                sub for sub in scene_subdirs if (sub / f"{sub.name}.json").exists()
            ]
            if len(matching) == 1:
                job_dir = matching[0]
                resolved_json = (job_dir / f"{job_dir.name}.json").resolve()
                resolved_theme = (theme or "").strip()
                if not resolved_theme:
                    prompts_file = job_dir / "prompts.txt"
                    if prompts_file.exists():
                        resolved_theme = prompts_file.read_text(encoding="utf-8").strip()
                return ResolvedScene(
                    scene_name=job_dir.name,
                    scene_json=resolved_json,
                    theme=resolved_theme,
                    source=LayoutSource.generated,
                    description=f"resumed layout from {resolved_json}",
                )
            if len(matching) > 1:
                names = ", ".join(sub.name for sub in matching)
                raise typer.BadParameter(
                    f"{job_inputs_dir} contains multiple scene bundles ({names}). "
                    "Pass --scene-json explicitly to pick one."
                )
            if scene_subdirs and not matching:
                raise typer.BadParameter(
                    f"{job_inputs_dir} exists but contains no <name>/<name>.json bundle. "
                    "Pass --scene-json explicitly."
                )

    selected_name = existing_scene or DEFAULT_EXISTING_SCENE
    scene = scene_catalog_by_name().get(selected_name)
    if scene is None:
        known = ", ".join(choice.scene_name for choice in load_scene_catalog())
        raise typer.BadParameter(
            f"Unknown bundled layout '{selected_name}'. Choose one of: {known}"
        )

    resolved_theme = scene.theme if theme is None else theme.strip()
    return ResolvedScene(
        scene_name=scene.scene_name,
        scene_json=scene.scene_json,
        theme=resolved_theme,
        source=LayoutSource.bundled,
        description=f"bundled layout {scene.scene_name}",
    )


def materialize_job_bundle(
    scene: ResolvedScene,
    output_dir: Path,
) -> JobBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    job_dir = output_dir / "_job_inputs" / scene.scene_name
    job_dir.mkdir(parents=True, exist_ok=True)

    target_json = job_dir / f"{scene.scene_name}.json"
    if scene.scene_json.resolve() != target_json.resolve():
        shutil.copy2(scene.scene_json, target_json)

    prompts_file = job_dir / "prompts.txt"
    prompts_file.write_text(f"{scene.theme.strip()}\n", encoding="utf-8")

    return JobBundle(
        scene_name=scene.scene_name,
        job_dir=job_dir,
        scene_json=target_json,
        prompts_file=prompts_file,
        theme=scene.theme.strip(),
    )


def build_batch_command(
    bundle: JobBundle,
    output_dir: Path,
    phase: str,
    api: bool,
    port: int,
    reconstruction: bool,
    depth_threshold: float,
    min_bootstrap_attempts: int,
    fov_final: int,
    placement_mode: str = DEFAULT_PLACEMENT_MODE,
    verbose: bool = False,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "method" / "run_batch_pipeline.py"),
        "--scenes-dir",
        str(bundle.job_dir),
        "--output-dir",
        str(output_dir),
        "--phase",
        phase,
        "--scenes",
        bundle.scene_name,
        "--depth-threshold",
        str(depth_threshold),
        "--min-bootstrap-attempts",
        str(min_bootstrap_attempts),
        "--fov-final",
        str(fov_final),
        "--placement-mode",
        placement_mode,
        "--port",
        str(port),
    ]

    # image_model -> (api, internal model key) for the run_full_pipeline subprocess
    api_from_model, model_key = IMAGE_MODEL_CHOICES[image_model]
    effective_api = api and api_from_model
    if effective_api:
        cmd.append("--api")
        cmd.extend(["--model", model_key])
    else:
        cmd.append("--no-api")

    if reconstruction:
        cmd.append("--reconstruction")

    if verbose:
        cmd.append("--verbose")

    return cmd


def build_resume_command(
    bundle: JobBundle,
    output_dir: Path,
    *,
    api: bool,
    reconstruction: bool,
    depth_threshold: float,
    min_bootstrap_attempts: int,
    fov_final: int,
    placement_mode: str,
    verbose: bool = False,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> str:
    cmd = [
        "python",
        "worldmesh_cli.py",
        "generate",
        "--phase",
        GenerationPhase.generate.value,
        "--scene-json",
        str(bundle.scene_json),
        "--output-dir",
        str(output_dir),
        "--depth-threshold",
        str(depth_threshold),
        "--min-bootstrap-attempts",
        str(min_bootstrap_attempts),
        "--fov-final",
        str(fov_final),
        "--placement-mode",
        placement_mode,
        "--image-model",
        image_model,
    ]

    if bundle.theme:
        cmd.extend(["--theme", bundle.theme])

    if api:
        cmd.append("--api")
    else:
        cmd.append("--no-api")

    if reconstruction:
        cmd.append("--reconstruction")

    if verbose:
        cmd.append("--verbose")

    return format_command(cmd)


def print_generation_summary(
    scene: ResolvedScene,
    bundle: JobBundle,
    output_dir: Path,
    phase: str,
    api: bool,
    depth_threshold: float,
    min_bootstrap_attempts: int,
    fov_final: int,
    placement_mode: str,
    dry_run: bool,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> None:
    typer.secho("\nGeneration summary", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  source: {scene.description}")
    typer.echo(f"  scene JSON: {scene.scene_json}")
    typer.echo(f"  job bundle: {bundle.job_dir}")
    typer.echo(f"  output dir: {output_dir}")
    typer.echo(f"  phase: {phase}")
    typer.echo(f"  image model: {image_model}")
    typer.echo(f"  API mode: {'enabled' if api else 'disabled'}")
    typer.echo(f"  theme: {bundle.theme or '(generic photorealistic fallback)'}")
    typer.echo(f"  depth threshold: {depth_threshold}")
    typer.echo(f"  min bootstrap attempts: {min_bootstrap_attempts}")
    typer.echo(f"  final FOV: {fov_final}")
    typer.echo(f"  placement mode: {placement_mode}")
    typer.echo(f"  mode: {'dry run' if dry_run else 'execute now'}")


def print_two_phase_notice() -> None:
    typer.secho("\n" + "=" * 60, fg=typer.colors.YELLOW)
    typer.secho("IMPORTANT: Two-phase pipeline", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        "The pipeline will pause after generating initial renders to let you\n"
        "create object masks in a browser UI (Gradio). After the UI shows\n"
        "'All done!', confirm that you are happy with the masks. The UI should\n"
        "close and the pipeline should continue automatically."
    )
    typer.secho("=" * 60, fg=typer.colors.YELLOW)


def write_resume_file(
    output_dir: Path,
    bundle: JobBundle,
    *,
    api: bool,
    reconstruction: bool,
    depth_threshold: float,
    min_bootstrap_attempts: int,
    fov_final: int,
    placement_mode: str,
    verbose: bool = False,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> None:
    resume_cmd = build_resume_command(
        bundle,
        output_dir,
        api=api,
        reconstruction=reconstruction,
        depth_threshold=depth_threshold,
        min_bootstrap_attempts=min_bootstrap_attempts,
        fov_final=fov_final,
        placement_mode=placement_mode,
        verbose=verbose,
        image_model=image_model,
    )
    resume_path = output_dir / "RESUME.txt"
    resume_path.write_text(
        "If the pipeline does not continue automatically after mask confirmation, run:\n\n"
        f"  {resume_cmd}\n\n"
        "On a remote server, copy the entire run folder and run the same command there.\n",
        encoding="utf-8",
    )


def print_resume_box(
    output_dir: Path,
    bundle: JobBundle,
    *,
    api: bool,
    reconstruction: bool,
    depth_threshold: float,
    min_bootstrap_attempts: int,
    fov_final: int,
    placement_mode: str,
    verbose: bool = False,
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> None:
    resume_cmd = build_resume_command(
        bundle,
        output_dir,
        api=api,
        reconstruction=reconstruction,
        depth_threshold=depth_threshold,
        min_bootstrap_attempts=min_bootstrap_attempts,
        fov_final=fov_final,
        placement_mode=placement_mode,
        verbose=verbose,
        image_model=image_model,
    )
    typer.secho("\n" + "=" * 60, fg=typer.colors.CYAN)
    typer.secho("If the pipeline does not continue automatically, run:", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"\n  {resume_cmd}\n")
    typer.echo("On a remote server, copy the entire run folder and run the same command there.")
    typer.secho("=" * 60, fg=typer.colors.CYAN)


def resolve_viewer_command(scene_output: Path) -> tuple[Optional[str], Optional[Path]]:
    """Resolve the exact ns-viewer command for a completed scene."""
    view_file = scene_output / "view_scene.txt"
    if view_file.exists():
        viewer_cmd = view_file.read_text(encoding="utf-8").strip()
        if viewer_cmd:
            try:
                parts = shlex.split(viewer_cmd)
            except ValueError:
                return viewer_cmd, None

            if parts and parts[0] == "ns-viewer" and "--load-config" in parts:
                config_index = parts.index("--load-config") + 1
                if config_index < len(parts):
                    return viewer_cmd, Path(parts[config_index])
            return viewer_cmd, None

    nerfstudio_output = scene_output / "nerfstudio_output"
    if nerfstudio_output.exists():
        config_files = sorted(nerfstudio_output.rglob("config.yml"))
        if config_files:
            config_path = config_files[-1]
            return f"ns-viewer --load-config {config_path}", config_path

    return None, None


def print_completion_box(output_dir: Path, bundle: JobBundle) -> None:
    scene_output = output_dir / bundle.scene_name
    nerfstudio_output = scene_output / "nerfstudio_output"
    view_file = scene_output / "view_scene.txt"
    viewer_cmd, _ = resolve_viewer_command(scene_output)

    typer.secho("\n" + "=" * 60, fg=typer.colors.GREEN)
    typer.secho("Generation complete!", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"\n  Output:        {scene_output}")
    if nerfstudio_output.exists():
        typer.echo(f"  Trained scene: {nerfstudio_output}")
    typer.secho("\nView with nerfstudio:", fg=typer.colors.CYAN, bold=True)
    typer.echo("  conda activate worldmesh-nerfstudio")
    if viewer_cmd:
        typer.echo(f"  {viewer_cmd}")
        if view_file.exists():
            typer.echo(f"  # exact command also saved in {view_file}")
    else:
        typer.secho(
            "  Viewer command unavailable: no view_scene.txt or nerfstudio_output/**/config.yml found.",
            fg=typer.colors.YELLOW,
        )
    typer.secho("=" * 60, fg=typer.colors.GREEN)


def create_layout_via_script(
    *,
    shape: str,
    num_rooms: int,
    room_width: float,
    room_depth: float,
    jitter: float,
    seed: int,
    output_dir: Path,
    name: Optional[str],
) -> Path:
    resolved_output_dir = resolve_user_path(output_dir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "method" / "generate_scene_layout.py"),
        "--shape",
        shape,
        "--num-rooms",
        str(num_rooms),
        "--room-width",
        str(room_width),
        "--room-depth",
        str(room_depth),
        "--jitter",
        str(jitter),
        "--seed",
        str(seed),
        "--output-dir",
        str(resolved_output_dir),
    ]
    stem = name or f"scene_{num_rooms}rooms_{shape}"
    if name:
        cmd.extend(["--name", name])

    run_command(cmd, "Creating deterministic scene layout")
    scene_json = resolved_output_dir / f"{stem}.json"
    scene_png = resolved_output_dir / f"{stem}.png"
    if not scene_json.exists():
        typer.secho(
            f"Expected layout file was not created: {scene_json}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(f"Created layout: {scene_json}", fg=typer.colors.GREEN)
    if not scene_png.exists():
        typer.secho(
            "Layout JSON was created, but the PNG preview was not written. "
            "This usually means the current Python environment is missing "
            "matplotlib.",
            fg=typer.colors.YELLOW,
        )
    return scene_json


def create_layout_api_via_script(
    *,
    prompt_text: str,
    output_dir: Path,
    name: Optional[str],
    max_iterations: int,
    model: str,
    api_key: Optional[str],
    layout_model: str = DEFAULT_LAYOUT_MODEL,
) -> Path:
    if layout_model not in LAYOUT_MODEL_CHOICES:
        raise typer.BadParameter(
            f"Unknown --layout-model {layout_model!r}. Choose one of: "
            f"{', '.join(LAYOUT_MODEL_CHOICES)}"
        )

    resolved_output_dir = resolve_user_path(output_dir)
    effective_name = name or make_api_layout_name(resolved_output_dir)

    if layout_model == LAYOUT_MODEL_CLAUDE_OPUS_4_6:
        requirement = anthropic_api_key_requirement(
            api_key,
            stage="Anthropic layout generation",
        )
        if not requirement.is_present():
            typer.secho(
                format_missing_checkpoints_error([requirement]),
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        resolved_api_key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()

        cmd = [
            sys.executable,
            str(REPO_ROOT / "method" / "generate_scene_layout_api.py"),
            "--prompt",
            prompt_text,
            "--output-dir",
            str(resolved_output_dir),
            "--max-iterations",
            str(max_iterations),
            "--layout-model",
            layout_model,
            "--model",
            model,
            "--api-key",
            resolved_api_key,
            "--name",
            effective_name,
        ]
        description = "Creating Anthropic-assisted scene layout"
    else:  # gpt-oss-20b
        if not conda_env_exists(WORLDMESH_LLM_ENV):
            typer.secho(
                format_missing_worldmesh_llm_env_error(),
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        checkpoint_requirement = gpt_oss_20b_requirement(REPO_ROOT)
        if not checkpoint_requirement.is_present():
            typer.secho(
                format_missing_checkpoints_error([checkpoint_requirement]),
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            WORLDMESH_LLM_ENV,
            "python",
            str(REPO_ROOT / "method" / "generate_scene_layout_api.py"),
            "--prompt",
            prompt_text,
            "--output-dir",
            str(resolved_output_dir),
            "--max-iterations",
            str(max_iterations),
            "--layout-model",
            layout_model,
            "--name",
            effective_name,
        ]
        description = f"Creating gpt-oss-20b layout (env {WORLDMESH_LLM_ENV})"

    run_command(cmd, description)
    scene_json = resolved_output_dir / f"{effective_name}.json"
    scene_png = resolved_output_dir / f"{effective_name}.png"
    if not scene_json.exists():
        typer.secho(
            f"Expected API layout file was not created: {scene_json}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if not scene_png.exists():
        typer.secho(
            f"Expected API layout preview was not created: {scene_png}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(f"Created layout: {scene_json}", fg=typer.colors.GREEN)
    return scene_json


def run_guided_wizard() -> None:
    """Interactive wizard: choose a layout, pick a theme, generate."""
    typer.secho("\nWorldMesh Scene Generation", fg=typer.colors.GREEN, bold=True)
    typer.echo(
        "\nWorldMesh generates photorealistic multi-room 3D scenes.\n"
        "\nHow it works:\n"
        "  1. Choose or generate a floor plan layout which controls the room shapes and arrangement.\n"
        "  2. Choose a visual theme.\n"
        "  3. WorldMesh runs in two phases:\n"
        "       Phase 1 — Initial renders + mask creation: generates structure renders,\n"
        "                 then opens a browser UI where you select individual objects to be added to each room.\n"
        "       Phase 2 — Full generation: WorldMesh uses the underlying geometry to\n"
        "                 produce the final photorealistic scene."
    )

    top_choice = prompt_for_menu_choice(
        "\nHow would you like to set up your scene?",
        [
            "Default 4-room zigzag layout  [recommended]",
            "Choose an existing scene layout",
            "Create a new layout (procedural: choose shape, room count, and size)",
            "Create a layout from a text description  [Claude Opus 4.6 or local gpt-oss-20b]",
        ],
        default_index=1,
    )

    selected_scene_name: Optional[str] = None
    selected_scene_json: Optional[Path] = None

    if top_choice == 1:
        selected_scene_name = DEFAULT_EXISTING_SCENE
        typer.echo(
            f"\nUsing the default layout: {DEFAULT_EXISTING_SCENE}\n"
            "This is a 4-room floor plan in a zigzag arrangement — a good starting point."
        )
        open_preview_png(SCENES_DIR / f"{DEFAULT_EXISTING_SCENE}.png")

    elif top_choice == 2:
        selected_scene_json = prompt_for_existing_scene()

    elif top_choice == 3:
        typer.secho("\nProcedural layout generation", fg=typer.colors.CYAN, bold=True)
        typer.echo(
            "Choose the overall shape of the floor plan, the number of rooms, and the\n"
            "base room dimensions. A 2D floor plan image will be generated for you to review."
        )

        shapes = list(LayoutShape)
        default_shape_index = shapes.index(LayoutShape.zigzag) + 1
        shape_choice = prompt_for_menu_choice(
            "\nFloor plan shape",
            [s.value for s in shapes],
            default_index=default_shape_index,
        )
        chosen_shape = shapes[shape_choice - 1].value
        num_rooms = int(typer.prompt("Number of rooms", default=4))

        typer.echo(
            "\nRoom dimensions set the base size of each room in the floor plan (in meters).\n"
            "  Room width = the X dimension (left-right extent of each room)\n"
            "  Room depth = the Y dimension (front-back extent of each room)"
        )
        room_width = float(typer.prompt("Room width in meters", default=6.0))
        room_depth = float(typer.prompt("Room depth in meters", default=4.0))

        typer.echo(
            "\nJitter adds random size variation between rows and columns of rooms.\n"
            "  0.0 = all rooms the same size (clean, predictable)\n"
            "  0.5 = rooms vary up to ±50% in size\n"
            "  1.0 = maximum variation (more organic, less regular)"
        )
        jitter = float(typer.prompt("Room size jitter (0.0–1.0)", default=0.0))

        selected_scene_json = create_layout_via_script(
            shape=chosen_shape,
            num_rooms=num_rooms,
            room_width=room_width,
            room_depth=room_depth,
            jitter=jitter,
            seed=0,
            output_dir=DEFAULT_LAYOUT_OUTPUT_DIR,
            name=None,
        )
        open_preview_png(selected_scene_json.with_suffix(".png"))

        if not typer.confirm("\nContinue to scene generation with this layout?", default=True):
            typer.echo(f"Layout saved to: {selected_scene_json}")
            raise typer.Exit()

    else:
        # Option 4: LLM-generated layout (Claude Opus 4.6 or local gpt-oss-20b).
        chosen_layout_model = prompt_for_layout_model_menu()
        if chosen_layout_model == LAYOUT_MODEL_CLAUDE_OPUS_4_6:
            requirement = anthropic_api_key_requirement(
                os.environ.get("ANTHROPIC_API_KEY", ""),
                stage="Anthropic layout generation",
            )
            if not requirement.is_present():
                typer.secho("\n" + format_missing_checkpoints_error([requirement]), fg=typer.colors.RED)
                raise typer.Exit(1)
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        else:  # gpt-oss-20b
            if not conda_env_exists(WORLDMESH_LLM_ENV):
                typer.secho("\n" + format_missing_worldmesh_llm_env_error(), fg=typer.colors.RED)
                raise typer.Exit(1)
            checkpoint_requirement = gpt_oss_20b_requirement(REPO_ROOT)
            if not checkpoint_requirement.is_present():
                typer.secho("\n" + format_missing_checkpoints_error([checkpoint_requirement]), fg=typer.colors.RED)
                raise typer.Exit(1)
            api_key = ""

        typer.echo(
            "\nDescribe the floor plan you want and the LLM will generate it for you.\n"
            'Example: "6-room apartment with a central foyer"'
        )
        layout_prompt = typer.prompt("Your floor plan description")
        selected_scene_json = create_layout_api_via_script(
            prompt_text=layout_prompt,
            output_dir=DEFAULT_LAYOUT_OUTPUT_DIR,
            name=None,
            max_iterations=5,
            model="claude-opus-4-6",
            api_key=api_key,
            layout_model=chosen_layout_model,
        )
        open_preview_png(selected_scene_json.with_suffix(".png"))

        if not typer.confirm("\nContinue to scene generation with this layout?", default=True):
            typer.echo(f"Layout saved to: {selected_scene_json}")
            raise typer.Exit()

    selected_theme = prompt_for_theme_menu()
    selected_image_model = prompt_for_image_model_menu()
    output_dir = DEFAULT_OUTPUT_DIR / make_run_name(selected_theme)

    resolved_scene = resolve_scene(
        existing_scene=selected_scene_name,
        scene_json=selected_scene_json,
        theme=selected_theme,
    )
    resolved_output_dir = resolve_user_path(output_dir)
    bundle = materialize_job_bundle(resolved_scene, resolved_output_dir)
    cmd = build_batch_command(
        bundle=bundle,
        output_dir=resolved_output_dir,
        phase=GenerationPhase.all.value,
        api=True,
        port=DEFAULT_PORT,
        reconstruction=False,
        depth_threshold=DEFAULT_DEPTH_THRESHOLD,
        min_bootstrap_attempts=DEFAULT_MIN_BOOTSTRAP_ATTEMPTS,
        fov_final=DEFAULT_FOV_FINAL,
        placement_mode=DEFAULT_PLACEMENT_MODE,
        image_model=selected_image_model,
    )

    print_generation_summary(
        scene=resolved_scene,
        bundle=bundle,
        output_dir=resolved_output_dir,
        phase=GenerationPhase.all.value,
        api=True,
        depth_threshold=DEFAULT_DEPTH_THRESHOLD,
        min_bootstrap_attempts=DEFAULT_MIN_BOOTSTRAP_ATTEMPTS,
        fov_final=DEFAULT_FOV_FINAL,
        placement_mode=DEFAULT_PLACEMENT_MODE,
        dry_run=False,
        image_model=selected_image_model,
    )
    typer.echo(f"Command: {format_command(cmd)}")
    print_two_phase_notice()
    write_resume_file(
        resolved_output_dir,
        bundle,
        api=True,
        reconstruction=False,
        depth_threshold=DEFAULT_DEPTH_THRESHOLD,
        min_bootstrap_attempts=DEFAULT_MIN_BOOTSTRAP_ATTEMPTS,
        fov_final=DEFAULT_FOV_FINAL,
        placement_mode=DEFAULT_PLACEMENT_MODE,
        image_model=selected_image_model,
    )

    if not typer.confirm("\nStart generation now?", default=True):
        typer.echo(f"\nResume file written to: {resolved_output_dir / 'RESUME.txt'}")
        raise typer.Exit()

    abort_if_missing_checkpoints(
        phase=GenerationPhase.all,
        api=True,
        reconstruction=False,
        dry_run=False,
        image_model=selected_image_model,
    )
    run_command(cmd, "Launching the batch pipeline")
    print_completion_box(resolved_output_dir, bundle)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        run_guided_wizard()


@app.command()
def generate(
    existing_scene: Annotated[
        Optional[str],
        typer.Option(
            "--existing-scene",
            help=(
                "Bundled layout name from scenes. Defaults to "
                f"{DEFAULT_EXISTING_SCENE} if --scene-json is not provided."
            ),
        ),
    ] = None,
    scene_json: Annotated[
        Optional[Path],
        typer.Option(
            "--scene-json",
            help="Path to a custom scene layout JSON file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Root output directory for the generated scene.",
        ),
    ] = DEFAULT_OUTPUT_DIR,
    phase: Annotated[
        GenerationPhase,
        typer.Option(
            "--phase",
            case_sensitive=False,
            help="Which batch-pipeline phase to run.",
        ),
    ] = GenerationPhase.all,
    api: Annotated[
        bool,
        typer.Option(
            "--api/--no-api",
            help="Pass API mode through to run_full_pipeline.py via run_batch_pipeline.py.",
        ),
    ] = True,
    port: Annotated[
        int,
        typer.Option("--port", help="Port for the manual mask Gradio UI."),
    ] = DEFAULT_PORT,
    reconstruction: Annotated[
        bool,
        typer.Option(
            "--reconstruction",
            help="Stop after stage 4 object reconstruction.",
        ),
    ] = False,
    depth_threshold: Annotated[
        float,
        typer.Option(
            "--depth-threshold",
            help="Depth validation threshold passed through to the pipeline.",
        ),
    ] = DEFAULT_DEPTH_THRESHOLD,
    min_bootstrap_attempts: Annotated[
        int,
        typer.Option(
            "--min-bootstrap-attempts",
            help="Minimum bootstrap attempts before selecting the best image.",
        ),
    ] = DEFAULT_MIN_BOOTSTRAP_ATTEMPTS,
    fov_final: Annotated[
        int,
        typer.Option(
            "--fov-final",
            help="Final stage camera field of view.",
        ),
    ] = DEFAULT_FOV_FINAL,
    placement_mode: Annotated[
        str,
        typer.Option(
            "--placement-mode",
            help="Object placement mode: 'simple' (minimal, default) or 'smart' (full heuristics).",
        ),
    ] = DEFAULT_PLACEMENT_MODE,
    theme: Annotated[
        Optional[str],
        typer.Option(
            "--theme",
            help=(
                "Override the bundled showcase theme or provide one for a "
                "custom scene. Leave unset to use the bundled theme or the "
                "pipeline's generic fallback."
            ),
        ),
    ] = None,
    image_model: Annotated[
        str,
        typer.Option(
            "--image-model",
            help=(
                "Image-generation model. 'nano-banana-pro' (default, recommended) uses the "
                "ComfyOrg API and requires COMFY_API_KEY. 'flux2-klein-9b' runs locally via "
                "ComfyUI with the distilled UNet (fast). 'flux2-klein-9b-base' uses the "
                "undistilled UNet (higher quality, ~5x slower per iterative camera). The "
                "two local options need no API key."
            ),
        ),
    ] = DEFAULT_IMAGE_MODEL,
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            help=(
                "Internal model key alias (forwarded by RESUME.txt / run_batch_pipeline.py). "
                "Accepts values like 'api' or 'flux2-klein-9b-distilled' and translates them "
                "to the corresponding --image-model. If both are given, --image-model wins."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Prepare the one-scene job bundle and print the command without running it.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Stream subprocess output live instead of capturing it (useful for debugging failures).",
        ),
    ] = False,
) -> None:
    """Generate one scene through the existing batch pipeline."""
    # If the caller passed only --model X (e.g. from a RESUME.txt written by
    # run_batch_pipeline.py), translate it to the corresponding --image-model.
    if model is not None and image_model == DEFAULT_IMAGE_MODEL:
        inverse_lookup = {model_key: name for name, (_, model_key) in IMAGE_MODEL_CHOICES.items()}
        if model in inverse_lookup:
            image_model = inverse_lookup[model]
        else:
            raise typer.BadParameter(
                f"Unknown --model {model!r}. Known internal model keys: "
                f"{', '.join(sorted(inverse_lookup.keys()))}"
            )
    if image_model not in IMAGE_MODEL_CHOICES:
        raise typer.BadParameter(
            f"Unknown --image-model {image_model!r}. Choose one of: "
            f"{', '.join(sorted(IMAGE_MODEL_CHOICES.keys()))}"
        )
    resolved_output_dir = resolve_user_path(output_dir)
    resolved_scene = resolve_scene(
        existing_scene=existing_scene,
        scene_json=scene_json,
        theme=theme,
        output_dir=resolved_output_dir,
    )
    bundle = materialize_job_bundle(resolved_scene, resolved_output_dir)
    cmd = build_batch_command(
        bundle=bundle,
        output_dir=resolved_output_dir,
        phase=phase.value,
        api=api,
        port=port,
        reconstruction=reconstruction,
        depth_threshold=depth_threshold,
        min_bootstrap_attempts=min_bootstrap_attempts,
        fov_final=fov_final,
        placement_mode=placement_mode,
        verbose=verbose,
        image_model=image_model,
    )

    print_generation_summary(
        scene=resolved_scene,
        bundle=bundle,
        output_dir=resolved_output_dir,
        phase=phase.value,
        api=api,
        depth_threshold=depth_threshold,
        min_bootstrap_attempts=min_bootstrap_attempts,
        fov_final=fov_final,
        placement_mode=placement_mode,
        dry_run=dry_run,
        image_model=image_model,
    )
    typer.echo(f"Command: {format_command(cmd)}")

    abort_if_missing_checkpoints(
        phase=phase,
        api=api,
        reconstruction=reconstruction,
        dry_run=dry_run,
        image_model=image_model,
    )

    if dry_run:
        typer.secho("Dry run complete. No generation command was executed.", fg=typer.colors.YELLOW)
        return

    if phase.value in (GenerationPhase.all.value, GenerationPhase.masks.value):
        print_two_phase_notice()
    write_resume_file(
        resolved_output_dir,
        bundle,
        api=api,
        reconstruction=reconstruction,
        depth_threshold=depth_threshold,
        min_bootstrap_attempts=min_bootstrap_attempts,
        fov_final=fov_final,
        placement_mode=placement_mode,
        verbose=verbose,
        image_model=image_model,
    )
    run_command(cmd, "Launching the existing batch pipeline")
    if phase.value in (GenerationPhase.all.value, GenerationPhase.generate.value):
        print_completion_box(resolved_output_dir, bundle)


@layouts_app.command("list")
def layouts_list() -> None:
    """List bundled scene layouts and their paired showcase themes."""
    print_scene_catalog(load_scene_catalog())


@layouts_app.command("create")
def layouts_create(
    shape: Annotated[
        LayoutShape,
        typer.Option(
            "--shape",
            case_sensitive=False,
            help="Shape template used by generate_scene_layout.py.",
        ),
    ] = LayoutShape.zigzag,
    num_rooms: Annotated[
        int,
        typer.Option(
            "--num-rooms",
            min=1,
            help="Total number of rooms in the new layout.",
        ),
    ] = 4,
    room_width: Annotated[
        float,
        typer.Option(
            "--room-width",
            min=0.1,
            help="Base room width.",
        ),
    ] = 6.0,
    room_depth: Annotated[
        float,
        typer.Option(
            "--room-depth",
            min=0.1,
            help="Base room depth.",
        ),
    ] = 4.0,
    jitter: Annotated[
        float,
        typer.Option(
            "--jitter",
            min=0.0,
            max=1.0,
            help="Per-row and per-column size jitter.",
        ),
    ] = 0.0,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Seed for jitter and room naming.",
        ),
    ] = 0,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory where the new layout JSON and PNG should be written.",
        ),
    ] = DEFAULT_LAYOUT_OUTPUT_DIR,
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            help="Optional file stem. Defaults to scene_{num_rooms}rooms_{shape}.",
        ),
    ] = None,
) -> None:
    """Create a deterministic layout with generate_scene_layout.py."""
    create_layout_via_script(
        shape=shape.value,
        num_rooms=num_rooms,
        room_width=room_width,
        room_depth=room_depth,
        jitter=jitter,
        seed=seed,
        output_dir=output_dir,
        name=name,
    )


@layouts_app.command("create-api")
def layouts_create_api(
    prompt_text: Annotated[
        str,
        typer.Option(
            "--prompt",
            prompt="Describe the layout you want",
            help="Natural-language description for generate_scene_layout_api.py.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory where the generated layout should be written.",
        ),
    ] = DEFAULT_LAYOUT_OUTPUT_DIR,
    name: Annotated[
        Optional[str],
        typer.Option(
            "--name",
            help="Optional file stem. Defaults to the generator's scene_*rooms_llm name.",
        ),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            min=1,
            help="Maximum correction attempts before the layout generator gives up.",
        ),
    ] = 5,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Anthropic model id to pass through to generate_scene_layout_api.py.",
        ),
    ] = "claude-opus-4-6",
    layout_model: Annotated[
        str,
        typer.Option(
            "--layout-model",
            help=(
                "LLM to use. 'claude-opus-4-6' (default, recommended) calls the "
                "Anthropic API. 'gpt-oss-20b' loads the local model from the "
                "worldmesh-llm conda env."
            ),
        ),
    ] = DEFAULT_LAYOUT_MODEL,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            envvar="ANTHROPIC_API_KEY",
            help="Anthropic API key. Falls back to ANTHROPIC_API_KEY.",
            hide_input=True,
        ),
    ] = None,
) -> None:
    """Create a layout from text with generate_scene_layout_api.py."""
    create_layout_api_via_script(
        prompt_text=prompt_text,
        output_dir=output_dir,
        name=name,
        max_iterations=max_iterations,
        model=model,
        api_key=api_key,
        layout_model=layout_model,
    )


if __name__ == "__main__":
    app()
