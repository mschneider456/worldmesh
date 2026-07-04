#!/usr/bin/env python3
"""
Batch Scene Generation Pipeline

Processes multiple scenes from a folder, pairing scene JSONs with prompts from prompts.txt.
Runs in two phases:

  Phase 'masks' (local, interactive):
    - Runs stages 1-3 for all scenes
    - Launches combined Gradio UI for mask creation across all scenes

  Phase 'generate' (can run on remote server):
    - Resumes from stage 4 for each scene (masks already exist)
    - Runs reconstruction + stages 5-8

Usage:
    # Phase 1: Generate initial images and create masks
    python run_batch_pipeline.py --scenes-dir scenes/ --output-dir output/batch_001 --phase masks --api --fov-final 60

    # Phase 2: Run remaining pipeline (can be on remote server)
    python run_batch_pipeline.py --scenes-dir scenes/ --output-dir output/batch_001 --phase generate --api --fov-final 60

    # Process specific scenes only (for splitting across servers)
    python run_batch_pipeline.py --scenes-dir scenes/ --output-dir output/batch_001 --phase generate --scenes scene_3rooms_linear scene_4rooms_tshape --api
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def natural_sort_key(path: Path):
    """Sort key that treats numeric parts as integers (e.g. 3 < 12)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.stem)]


def format_command(cmd: List[str]) -> str:
    """Format a command for shell reuse."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def write_resume_file(resume_path: Path, resume_command: str):
    """Persist the fallback resume command for the batch run."""
    resume_path.write_text(
        "If the pipeline does not continue automatically after mask confirmation, run:\n\n"
        f"{resume_command}\n",
        encoding="utf-8",
    )


def build_worldmesh_resume_command(
    scene_json: Path,
    output_dir: Path,
    prompt: str,
    extra_args: List[str],
) -> str:
    """Build a one-scene resume command through the user-facing CLI."""
    cmd = [
        "python",
        "worldmesh_cli.py",
        "generate",
        "--phase",
        "generate",
        "--scene-json",
        str(scene_json),
        "--output-dir",
        str(output_dir),
    ]
    if prompt:
        cmd.extend(["--theme", prompt])
    cmd.extend(extra_args)
    return format_command(cmd)


def build_batch_resume_command(
    scenes: List[Tuple[Path, str]],
    scenes_dir: Path,
    output_dir: Path,
    extra_args: List[str],
) -> str:
    """Build a resume command for the current batch scope."""
    worldmesh_supported_flags = {
        "--api",
        "--no-api",
        "--model",
        "--reconstruction",
        "--depth-threshold",
        "--min-bootstrap-attempts",
        "--fov-final",
        "--placement-mode",
        "--verbose",
    }
    extra_flags = {arg for arg in extra_args if arg.startswith("-")}

    if len(scenes) == 1 and extra_flags.issubset(worldmesh_supported_flags):
        scene_json, prompt = scenes[0]
        return build_worldmesh_resume_command(scene_json, output_dir, prompt, extra_args)

    cmd = [
        "python",
        "run_batch_pipeline.py",
        "--scenes-dir",
        str(scenes_dir),
        "--output-dir",
        str(output_dir),
        "--phase",
        "generate",
    ]
    if scenes:
        cmd.extend(["--scenes", *[scene_json.stem for scene_json, _ in scenes]])
    cmd.extend(extra_args)
    return format_command(cmd)


def discover_scenes(scenes_dir: Path, prompts_path: Path, scene_filter: Optional[List[str]] = None) -> List[Tuple[Path, str]]:
    """
    Pair scene JSONs (natural sort order) with prompts (line by line).

    Returns list of (scene_json_path, prompt) tuples.
    """
    # Find all scene JSON files (natural sort so 3rooms < 12rooms)
    scene_jsons = sorted(scenes_dir.glob("*.json"), key=natural_sort_key)
    if not scene_jsons:
        print(f"Error: No JSON files found in {scenes_dir}", file=sys.stderr)
        sys.exit(1)

    # Read prompts
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]

    if len(prompts) < len(scene_jsons):
        print(f"Warning: {len(scene_jsons)} scenes but only {len(prompts)} prompts. Extra scenes will have no prompt.", file=sys.stderr)

    # Pair them
    pairs = []
    for i, scene_json in enumerate(scene_jsons):
        scene_name = scene_json.stem
        if scene_filter and scene_name not in scene_filter:
            continue
        prompt = prompts[i] if i < len(prompts) else ""
        pairs.append((scene_json, prompt))

    return pairs


def get_scene_output_dir(output_dir: Path, scene_json: Path) -> Path:
    """Get per-scene output directory."""
    return output_dir / scene_json.stem


def run_pipeline_for_scene(scene_json: Path, output_dir: Path, prompt: str, extra_args: List[str], skip_stages: Optional[List[int]] = None) -> bool:
    """Run run_full_pipeline.py for a single scene."""
    scene_output = get_scene_output_dir(output_dir, scene_json)
    scene_output.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "run_full_pipeline.py", "--scene-json", str(scene_json), "--output-dir", str(scene_output), "--theme", prompt]

    if skip_stages:
        cmd.extend(["--skip-stage"] + [str(s) for s in skip_stages])

    cmd.extend(extra_args)

    scene_name = scene_json.stem
    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"  JSON:   {scene_json}")
    print(f"  Output: {scene_output}")
    print(f"  Prompt: {prompt}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        print(f"WARNING: Pipeline failed for {scene_name} (exit code {result.returncode})", file=sys.stderr)
        return False
    return True


def build_batch_mask_config(scenes: List[Tuple[Path, str]], output_dir: Path) -> Path:
    """
    Build batch_mask_config.json mapping each scene's rooms to their
    initial_flux and extracted_objects directories.
    """
    config = []
    for scene_json, prompt in scenes:
        scene_output = get_scene_output_dir(output_dir, scene_json)
        initial_flux_dir = scene_output / "initial_flux"
        extracted_dir = scene_output / "extracted_objects"

        if not initial_flux_dir.exists():
            print(f"Warning: initial_flux dir not found for {scene_json.stem}, skipping", file=sys.stderr)
            continue

        # Discover rooms from initial_flux subdirectories
        rooms = []
        for subdir in sorted(initial_flux_dir.iterdir()):
            if subdir.is_dir():
                # Check for generated images
                if (subdir / "generated.png").exists() or list(subdir.glob("generated_0000*.png")) or list(subdir.glob("*.png")):
                    # Check if masks already exist
                    masks_meta = extracted_dir / subdir.name / "masks" / "metadata.json"
                    if masks_meta.exists():
                        with open(masks_meta) as f:
                            meta = json.load(f)
                        if meta.get("num_masks", 0) > 0:
                            continue  # Skip rooms with existing masks
                    rooms.append(subdir.name)

        if rooms:
            config.append({
                "scene_name": scene_json.stem,
                "input_dir": str(initial_flux_dir),
                "output_dir": str(extracted_dir),
                "rooms": rooms,
            })

    config_path = output_dir / "batch_mask_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    total_rooms = sum(len(entry["rooms"]) for entry in config)
    print(f"\nBatch mask config: {len(config)} scenes, {total_rooms} rooms needing masks")
    print(f"Saved to: {config_path}")
    return config_path


def phase_masks(scenes: List[Tuple[Path, str]], scenes_dir: Path, output_dir: Path, extra_args: List[str], port: int) -> bool:
    """
    Phase 'masks': Run stages 1-3 for all scenes, then launch combined Gradio UI.
    """
    print("\n" + "=" * 60)
    print("PHASE: masks")
    print("=" * 60)

    # Step 1: Run stages 1-3 for each scene (skip 4-8)
    # Also add --no-manual-masks since we handle masks separately
    mask_extra = ["--no-manual-masks"] + extra_args
    for scene_json, prompt in scenes:
        run_pipeline_for_scene(scene_json, output_dir, prompt, mask_extra, skip_stages=[4, 5, 6, 7, 8])

    # Step 2: Build batch config for Gradio
    config_path = build_batch_mask_config(scenes, output_dir)

    # Check if any rooms need masks
    with open(config_path) as f:
        config = json.load(f)
    total_rooms = sum(len(entry["rooms"]) for entry in config)

    if total_rooms == 0:
        print("\nAll rooms already have masks. Skipping Gradio UI.")
        return True

    # Step 3: Launch combined Gradio UI
    resume_command = build_batch_resume_command(scenes, scenes_dir, output_dir, extra_args)
    resume_path = output_dir / "RESUME.txt"
    write_resume_file(resume_path, resume_command)

    print(f"\nLaunching combined Gradio UI for {total_rooms} rooms across {len(config)} scenes...")
    print(f"Open http://localhost:{port} to create masks")
    print("After the UI shows 'All done!', confirm the masks to close Gradio and continue automatically.")
    print(f"If it does not continue automatically, run: {resume_command}")
    print(f"Saved fallback command to: {resume_path}\n")

    cmd = [
        "conda", "run", "-n", "worldmesh-sam3", "--live-stream", "python",
        str(Path(__file__).parent / "manual_mask_gradio.py"),
        "--batch-config", str(config_path),
        "--port", str(port),
        "--resume-command", resume_command,
        "--resume-file", str(resume_path),
    ]

    subprocess.run(cmd, cwd=str(Path(__file__).parent))

    # Verify some masks were saved
    masks_found = False
    for entry in config:
        extracted_dir = Path(entry["output_dir"])
        for room_id in entry["rooms"]:
            meta_path = extracted_dir / room_id / "masks" / "metadata.json"
            if meta_path.exists():
                masks_found = True
                break
        if masks_found:
            break

    if not masks_found:
        print("WARNING: No masks were saved. Run again and save masks before closing.", file=sys.stderr)
        return False

    print("\nPhase 'masks' complete. You can now:")
    print("  1. Copy output to remote server(s)")
    print(f"  2. Run: {resume_command}")
    return True


def phase_generate(scenes: List[Tuple[Path, str]], output_dir: Path, extra_args: List[str], reconstruction: bool = False) -> bool:
    """
    Phase 'generate': Resume pipeline from stage 4 for each scene.
    Pipeline state + existing masks ensure it resumes correctly.

    If reconstruction=True, stops after stage 4 (object extraction) — skips
    rendering and flux generation so you can inspect scene_with_all_objects.glb.
    """
    mode = "generate (reconstruction only)" if reconstruction else "generate"
    print("\n" + "=" * 60)
    print(f"PHASE: {mode}")
    print("=" * 60)

    successes = 0
    failures = 0

    for scene_json, prompt in scenes:
        skip = [5, 6, 7, 8] if reconstruction else None
        ok = run_pipeline_for_scene(scene_json, output_dir, prompt, extra_args, skip_stages=skip)
        if ok:
            successes += 1
        else:
            failures += 1

    print(f"\nPhase '{mode}' complete: {successes} succeeded, {failures} failed out of {len(scenes)} scenes")
    if reconstruction and successes > 0:
        print("Inspect scene_with_all_objects.glb in each scene's output directory before running full generate.")
    return failures == 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch Scene Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--scenes-dir", type=Path, required=True, help="Folder with *.json scene files and prompts.txt")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root output directory (each scene gets output_dir/<scene_name>/)")
    parser.add_argument("--phase", choices=["masks", "generate", "all"], default="all", help="Pipeline phase to run (default: all)")
    parser.add_argument("--scenes", nargs="+", default=None, help="Filter: only process these scene names (without .json extension)")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server (default: 7860)")
    parser.add_argument("--reconstruction", action="store_true", help="Stop after object extraction (stage 4) — skip rendering and flux generation")

    args, extra_args = parser.parse_known_args()

    # Validate
    if not args.scenes_dir.exists():
        print(f"Error: Scenes directory not found: {args.scenes_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover and pair scenes with prompts
    prompts_path = args.scenes_dir / "prompts.txt"
    scenes = discover_scenes(args.scenes_dir, prompts_path, args.scenes)

    if not scenes:
        print("Error: No scenes found to process", file=sys.stderr)
        return 1

    print(f"Found {len(scenes)} scenes to process:")
    for scene_json, prompt in scenes:
        print(f"  {scene_json.stem}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    # Run phases
    if args.phase in ("masks", "all"):
        ok = phase_masks(scenes, args.scenes_dir, args.output_dir, extra_args, args.port)
        if not ok and args.phase == "masks":
            return 1

    if args.phase in ("generate", "all"):
        ok = phase_generate(scenes, args.output_dir, extra_args, reconstruction=args.reconstruction)
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
