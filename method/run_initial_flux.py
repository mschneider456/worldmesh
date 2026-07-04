#!/usr/bin/env python3
"""Generate images from depth map inputs using Flux2-Klein Image Edit via ComfyUI.

Usage:
    python run_initial_flux.py --input depth/room_depth.png
    python run_initial_flux.py --input depth/room_depth.png --prompt "Bedroom and a lot of furniture. Photorealistic."
    python run_initial_flux.py --input depth/room_depth.png --output generated/my_room.png
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for flux_generation imports
sys.path.insert(0, str(Path(__file__).parent))

from checkpoint_requirements import (
    find_missing_requirements,
    format_missing_checkpoints_error,
    workflow_model_requirements,
)
from flux_generation.comfyui_server import ComfyUIServer
from flux_generation.comfyui_client import ComfyUIClient
from flux_generation.workflow_manager import load_workflow, prepare_initial_workflow


WORKFLOW_PATH = Path(__file__).parent.parent / "comfyui" / "scenes_workflows" / "image_flux2_klein_image_edit_9b_distilled_api.json"


async def run_generation(
    input_path: Path,
    output_path: Path,
    prompt: str | None = None,
    seed: int | None = None,
) -> None:
    """Run the Flux generation pipeline."""
    # Start ComfyUI server (auto-shutdown on exit)
    async with ComfyUIServer() as server:
        print(f"ComfyUI server ready at {server.base_url}")

        async with ComfyUIClient(server.host, server.port) as client:
            # Upload input image
            print(f"Uploading input image: {input_path}")
            uploaded_filename = await client.upload_image(input_path)
            print(f"  Uploaded as: {uploaded_filename}")

            # Load and prepare workflow
            print("Loading workflow...")
            workflow = load_workflow(WORKFLOW_PATH)
            workflow = prepare_initial_workflow(
                workflow,
                image_filename=uploaded_filename,
                prompt=prompt,
                seed=seed,
            )

            # Execute workflow
            print("Executing workflow (this may take a while)...")
            images = await client.generate(workflow, timeout=600.0)

            if not images:
                raise RuntimeError("No images generated")

            # Save first output image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(output_path)
            print(f"Saved output to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from depth map inputs using Flux2-Klein Image Edit via ComfyUI"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input PNG (depth map image)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/initial_flux.png"),
        help="Path to save generated image (default: outputs/initial_flux.png)",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Custom prompt (uses workflow default if not specified)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate workflow exists
    if not WORKFLOW_PATH.exists():
        print(f"Error: Workflow not found: {WORKFLOW_PATH}", file=sys.stderr)
        sys.exit(1)

    missing = find_missing_requirements(
        workflow_model_requirements(
            Path(__file__).parent.parent.resolve(),
            [WORKFLOW_PATH],
            stage="Initial Flux generation",
        )
    )
    if missing:
        print("\n" + format_missing_checkpoints_error(missing), file=sys.stderr)
        sys.exit(1)

    # Run generation
    try:
        asyncio.run(run_generation(
            input_path=args.input,
            output_path=args.output,
            prompt=args.prompt,
            seed=args.seed,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
