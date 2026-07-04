"""Command-line interface for Flux generation pipeline."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checkpoint_requirements import (
    comfy_api_key_requirement,
    depth_pro_requirement,
    find_missing_requirements,
    format_missing_checkpoints_error,
    workflow_model_requirements,
)
from .config import Config, MODEL_REGISTRY
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate photorealistic images from edge-based scene renderings using Flux2-Klein via ComfyUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-starts ComfyUI server)
  python flux_generate.py \\
      --input-dir output/one_scene/final_render_with_objects_floor_aligned_wall/images/master_bedroom \\
      --output-dir flux_generated/master_bedroom

  # Use existing server (skip auto-start)
  python flux_generate.py \\
      --input-dir output/one_scene/final_render_with_objects_floor_aligned_wall/images/master_bedroom \\
      --output-dir flux_generated/master_bedroom \\
      --no-auto-start

  # Custom ComfyUI path
  python flux_generate.py \\
      --input-dir output/one_scene/final_render_with_objects_floor_aligned_wall/images/master_bedroom \\
      --output-dir flux_generated/master_bedroom \\
      --comfyui-path /path/to/comfyui

  # With custom options
  python flux_generate.py \\
      --input-dir output/one_scene/final_render_with_objects_floor_aligned_wall/images/master_bedroom \\
      --output-dir flux_generated/master_bedroom \\
      --iou-threshold 0.3 \\
      --max-retries 4 \\
      --verbose

  # Override prompt
  python flux_generate.py \\
      --input-dir /path/to/images/room \\
      --output-dir /path/to/output \\
      --prompt "Professional interior photography of a modern living room..."
""",
    )

    # Required arguments
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_with_edges.png renderings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated images",
    )

    # ComfyUI connection
    parser.add_argument(
        "--comfyui-host",
        type=str,
        default="127.0.0.1",
        help="ComfyUI server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--comfyui-port",
        type=int,
        default=8188,
        help="ComfyUI server port (default: 8188)",
    )

    # Server management
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't auto-start ComfyUI (assume server is already running)",
    )
    parser.add_argument(
        "--comfyui-path",
        type=Path,
        default=None,
        help="Path to ComfyUI installation (auto-detected if not specified)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=120,
        help="Max time in seconds to wait for ComfyUI server to start (default: 120)",
    )

    # Workflow paths
    parser.add_argument(
        "--initial-workflow",
        type=Path,
        default=None,
        help="Path to initial_camera.json workflow (auto-detected if not specified)",
    )
    parser.add_argument(
        "--iterative-workflow",
        type=Path,
        default=None,
        help="Path to iterative_cameras.json workflow (auto-detected if not specified)",
    )
    parser.add_argument(
        "--qwen-workflow",
        type=Path,
        default=None,
        help="Path to Flux2-Klein Base iterative workflow (auto-detected if not specified)",
    )

    # Scene configuration
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=None,
        help="Path to scene JSON file (for opening detection in prompts)",
    )
    parser.add_argument(
        "--room-id",
        type=str,
        default=None,
        help="Room ID for opening detection (extracted from input-dir if not specified)",
    )

    # Validation settings
    parser.add_argument(
        "--use-edge-validation",
        action="store_true",
        help="Enable edge IoU validation (disabled by default)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Minimum Edge IoU score for validation (default: 0.3)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum attempts per image before marking failed (default: 2)",
    )
    parser.add_argument(
        "--bootstrap-max-retries",
        type=int,
        default=30,
        help="Maximum attempts for bootstrap image before abandoning room (default: 30)",
    )
    parser.add_argument(
        "--min-bootstrap-attempts",
        type=int,
        default=3,
        help="Minimum bootstrap attempts before selecting best passing one (default: 3)",
    )
    parser.add_argument(
        "--canny-low",
        type=float,
        default=0.1,
        help="Canny edge low threshold, normalized (default: 0.1)",
    )
    parser.add_argument(
        "--canny-high",
        type=float,
        default=0.3,
        help="Canny edge high threshold, normalized (default: 0.3)",
    )
    parser.add_argument(
        "--dilation-pixels",
        type=int,
        default=3,
        help="Pixel tolerance for edge alignment (default: 3)",
    )

    # Depth validation settings (Edge IoU on depth maps)
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.78,
        help="Minimum depth Edge IoU for validation pass (default: 0.78, higher is better)",
    )
    parser.add_argument(
        "--no-depth-validation",
        action="store_true",
        help="Disable depth-based validation, use edge-only",
    )
    parser.add_argument(
        "--depth-conda-env",
        type=str,
        default="worldmesh-depth-pro",
        help="Conda environment with Depth Pro installed (default: worldmesh-depth-pro)",
    )
    parser.add_argument(
        "--depth-canny-low",
        type=float,
        default=0.1,
        help="Lower threshold for depth edge detection (default: 0.1)",
    )
    parser.add_argument(
        "--depth-canny-high",
        type=float,
        default=0.3,
        help="Upper threshold for depth edge detection (default: 0.3)",
    )
    parser.add_argument(
        "--depth-dilation-pixels",
        type=int,
        default=21,
        help="Pixel tolerance for depth edge alignment (default: 21)",
    )
    parser.add_argument(
        "--depth-sharpen",
        type=float,
        default=4.0,
        help="Sharpening strength for estimated depth edges (default: 4.0, 1.0=none)",
    )
    parser.add_argument(
        "--depth-sharpen-gt",
        type=float,
        default=3.0,
        help="Sharpening strength for GT depth edges (default: 3.0, 1.0=none)",
    )
    parser.add_argument(
        "--depth-min-gradient-percentile",
        type=float,
        default=25.0,
        help="Filter depth edges below this GT gradient percentile (default: 25.0, 0=disabled)",
    )

    # Optional prompt override
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt (default: use prompt from workflow JSON)",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help="Theme for bootstrap prompt (e.g. \"a red and grim vampire castle\")",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="JSON file with per-camera prompts {camera_name: prompt}",
    )

    # Bootstrap control
    parser.add_argument(
        "--only-bootstrap",
        action="store_true",
        help="Generate camera 0 (bootstrap) only, then exit",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip camera 0, start from camera 1 (assumes generated_0000.png exists)",
    )
    parser.add_argument(
        "--only-camera",
        type=int,
        default=None,
        help="Generate only this camera ID using iterative workflow (requires existing bootstrap)",
    )
    parser.add_argument(
        "--bootstrap-cameras",
        type=int,
        nargs="+",
        default=None,
        help="Bootstrap camera IDs for style reference selection (default: [0])",
    )

    # API mode (Nano Banana Pro)
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use Nano Banana Pro (Gemini) for iterative generation and 1376w bootstrap workflows",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="api",
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            "Iterative-camera model. 'api' (default) = Nano Banana Pro (paid). "
            "'flux2-klein-9b-distilled' = fully local ComfyUI workflow (free)."
        ),
    )
    parser.add_argument(
        "--comfy-api-key",
        type=str,
        default=None,
        help="ComfyOrg API key for API nodes (Gemini, etc.). Falls back to COMFY_API_KEY env var.",
    )

    # Control flags
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip all validation, accept first generation attempt",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from progress file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Build config
    try:
        config = Config(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            comfyui_host=args.comfyui_host,
            comfyui_port=args.comfyui_port,
            auto_start_server=not args.no_auto_start,
            comfyui_path=args.comfyui_path,
            startup_timeout=args.startup_timeout,
            initial_workflow_path=args.initial_workflow,
            iterative_workflow_path=args.iterative_workflow,
            qwen_workflow_path=args.qwen_workflow,
            use_edge_validation=args.use_edge_validation,
            iou_threshold=args.iou_threshold,
            max_retries=args.max_retries,
            bootstrap_max_retries=args.bootstrap_max_retries,
            min_bootstrap_attempts=args.min_bootstrap_attempts,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            dilation_pixels=args.dilation_pixels,
            use_depth_validation=not args.no_depth_validation,
            depth_threshold=args.depth_threshold,
            depth_conda_env=args.depth_conda_env,
            depth_canny_low=args.depth_canny_low,
            depth_canny_high=args.depth_canny_high,
            depth_dilation_pixels=args.depth_dilation_pixels,
            depth_sharpen=args.depth_sharpen,
            depth_sharpen_gt=args.depth_sharpen_gt,
            depth_min_gradient_percentile=args.depth_min_gradient_percentile,
            api=args.api,
            model=args.model,
            comfy_api_key=args.comfy_api_key or os.environ.get("COMFY_API_KEY"),
            no_validation=args.no_validation,
            only_bootstrap=args.only_bootstrap,
            skip_bootstrap=args.skip_bootstrap,
            only_camera=args.only_camera,
            bootstrap_cameras=args.bootstrap_cameras,
            prompt=args.prompt,
            theme=args.theme,
            prompts_file=args.prompts_file,
            scene_json_path=args.scene_json,
            room_id=args.room_id,
            verbose=args.verbose,
            resume=not args.no_resume,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    workflow_paths = [
        path for path in (
            config.initial_workflow_path,
            config.iterative_workflow_path,
            config.qwen_workflow_path,
            config.bootstrap_workflow_path,
            config.model_workflow_path,
        )
        if path is not None
    ]
    requirements = workflow_model_requirements(
        repo_root,
        workflow_paths,
        stage="Flux generation via ComfyUI",
    )
    if config.use_depth_validation:
        requirements.append(depth_pro_requirement(repo_root, stage="Flux depth validation"))
    if config.api and config.model == "api" and not config.only_bootstrap:
        requirements.append(
            comfy_api_key_requirement(
                config.comfy_api_key,
                stage="Flux iterative generation via ComfyOrg API (Nano Banana Pro)",
            )
        )

    missing = find_missing_requirements(requirements)
    if missing:
        print("\n" + format_missing_checkpoints_error(missing), file=sys.stderr)
        return 1

    # Print config summary
    if args.verbose:
        print("Configuration:")
        print(f"  Input directory:  {config.input_dir}")
        print(f"  Output directory: {config.output_dir}")
        print(f"  Edges directory:  {config.edges_dir}")
        print(f"  ComfyUI:          {config.comfyui_host}:{config.comfyui_port}")
        print(f"  Auto-start server: {config.auto_start_server}")
        if config.auto_start_server:
            comfyui_path = config.comfyui_path or "(auto-detect)"
            print(f"  ComfyUI path:     {comfyui_path}")
            print(f"  Startup timeout:  {config.startup_timeout}s")
        print(f"  No validation:    {config.no_validation}")
        print(f"  Edge validation:  {config.use_edge_validation}")
        if config.use_edge_validation:
            print(f"  Edge IoU threshold: {config.iou_threshold}")
            print(f"  Dilation pixels:  {config.dilation_pixels}")
        print(f"  Max retries:      {config.max_retries}")
        print(f"  Depth validation: {config.use_depth_validation}")
        if config.use_depth_validation:
            print(f"  Depth Edge IoU threshold: {config.depth_threshold}")
            print(f"  Depth canny low:  {config.depth_canny_low}")
            print(f"  Depth canny high: {config.depth_canny_high}")
            print(f"  Depth dilation:   {config.depth_dilation_pixels}")
            print(f"  Depth conda env:  {config.depth_conda_env}")
        print(f"  Resume:           {config.resume}")
        print(f"  API mode:         {config.api}")
        print(f"  Model:            {config.model}")
        if config.api and config.model_workflow_path:
            print(f"  Model workflow:   {config.model_workflow_path}")
        if config.qwen_workflow_path:
            print(f"  Iterative workflow: {config.qwen_workflow_path}")
        if config.scene_json_path:
            print(f"  Scene JSON:       {config.scene_json_path}")
        print(f"  Room ID:          {config.room_id}")
        print()

    # Run pipeline
    try:
        asyncio.run(run_pipeline(config))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
