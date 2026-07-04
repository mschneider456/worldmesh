#!/usr/bin/env python3
"""
SAM3 Segmentation + SAM-3D-Objects Reconstruction Pipeline

Orchestrates the complete pipeline:
1. Segments objects in an image using SAM3
2. Reconstructs 3D meshes using SAM-3D-Objects
3. Transforms objects from camera space to world space
4. Merges reconstructed objects into the original room mesh
5. Re-renders from the original camera position

Usage:
    # Basic usage (without scale calibration)
    python run_pipeline.py \
        --input-image generations/Flux2_00013_.png \
        --room-mesh scene_output.glb \
        --camera-name bedroom_0001 \
        --output-dir output/

    # With wall-based scale calibration (recommended)
    python run_pipeline.py \
        --input-image outputs/one_scene/initial_flux.png \
        --room-mesh outputs/one_scene/scene_output_large.glb \
        --camera-name master_bedroom_0000 \
        --images-txt outputs/one_scene/renders/images.txt \
        --cameras-txt outputs/one_scene/renders/cameras.txt \
        --output-dir outputs/one_scene/extracted \
        --segmentation-map outputs/one_scene/renders/segmentation/master_bedroom_0000_seg.png \
        --segmentation-metadata outputs/one_scene/renders/segmentation_metadata.json \
        --scene-json outputs/one_scene/scene_layout.json \
        --room-id master_bedroom \
        --prompts "bed with bedding and pillows" nightstand lamp dresser

    # With manual scale factor (alternative)
    python run_pipeline.py \
        --input-image outputs/one_scene/initial_flux.png \
        --room-mesh outputs/one_scene/scene_output_large.glb \
        --camera-name master_bedroom_0000 \
        --output-dir outputs/one_scene/extracted \
        --scale-factor 2.5 \
        --prompts bed nightstand lamp

Each step runs in its appropriate conda environment:
    - Step 1 (SAM3): worldmesh-sam3
    - Step 2 (SAM-3D-Objects): worldmesh-sam3d-objects
    - Step 3 (Positioning): worldmesh
    - Step 4 (Rendering): worldmesh
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint_requirements import (
    find_missing_requirements,
    format_missing_checkpoints_error,
    sam3_requirement,
    sam3d_requirement,
)


def get_camera_pose_string(images_txt_path, camera_name):
    """
    Extract camera pose from COLMAP images.txt as 'qw,qx,qy,qz,tx,ty,tz' string.

    Args:
        images_txt_path: Path to COLMAP images.txt file
        camera_name: Camera name to search for (partial match)

    Returns:
        Pose string 'qw,qx,qy,qz,tx,ty,tz' or None if camera not found
    """
    with open(images_txt_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        parts = line.split()
        if len(parts) >= 10:
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            image_name = parts[9]
            if camera_name in image_name:
                qw, qx, qy, qz = parts[1], parts[2], parts[3], parts[4]
                tx, ty, tz = parts[5], parts[6], parts[7]
                return f"{qw},{qx},{qy},{qz},{tx},{ty},{tz}"

    return None


def run_step(step_name, command, cwd=None, env=None):
    """
    Run a pipeline step with colored output.

    Args:
        step_name: Name of the step for logging
        command: List of command arguments
        cwd: Working directory (optional)
        env: Environment variables (optional)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"RUNNING: {step_name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    if cwd:
        print(f"Working dir: {cwd}")
    print()

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True
        )
        print(f"\n[SUCCESS] {step_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"\n[ERROR] Command not found: {e}")
        return False


def count_object_masks(masks_dir):
    """
    Count object masks (non-doorway) from step1 metadata.

    Returns:
        tuple: (object_mask_indices, total_mask_count)
            object_mask_indices: list of index values for non-doorway masks
            total_mask_count: total number of masks including doorways
    """
    meta_path = Path(masks_dir) / "metadata.json"
    if not meta_path.exists():
        return [], 0

    with open(meta_path) as f:
        metadata = json.load(f)

    object_indices = []
    for mask_info in metadata.get('masks', []):
        if not mask_info.get('is_doorway', False):
            object_indices.append(mask_info['index'])

    return object_indices, len(metadata.get('masks', []))


def merge_batch_outputs(batch_dirs, objects_dir):
    """
    Merge per-object files and reconstruction summaries from batch subdirectories
    into the main objects directory.

    Args:
        batch_dirs: list of Path objects pointing to batch output directories
        objects_dir: main objects output directory
    """
    objects_dir = Path(objects_dir)
    objects_dir.mkdir(parents=True, exist_ok=True)

    merged_summary = {
        'objects': [],
        'failed': [],
        'skipped_doorway': [],
    }

    for batch_dir in batch_dirs:
        batch_dir = Path(batch_dir)
        if not batch_dir.exists():
            continue

        # Copy per-object files (*.glb, *.ply, *_pose.json)
        for pattern in ['*.glb', '*.ply', '*_pose.json']:
            for src in batch_dir.glob(pattern):
                # Skip combined scene files
                if src.name.startswith('scene_combined'):
                    continue
                dst = objects_dir / src.name
                shutil.copy2(src, dst)

        # Merge reconstruction summary
        summary_path = batch_dir / "reconstruction_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                batch_summary = json.load(f)
            merged_summary['objects'].extend(batch_summary.get('objects', []))
            merged_summary['failed'].extend(batch_summary.get('failed', []))
            merged_summary['skipped_doorway'].extend(batch_summary.get('skipped_doorway', []))
            # Carry over calibration and metadata from first batch
            if 'input_image' not in merged_summary:
                for key in ['input_image', 'masks_dir', 'calibration', 'doorway_overlap_threshold']:
                    if key in batch_summary:
                        merged_summary[key] = batch_summary[key]

    merged_summary['num_objects'] = len(merged_summary['objects'])
    merged_summary['combined_scene'] = {}  # No combined scene for batched runs
    merged_summary['batched'] = True
    merged_summary['num_batches'] = len(batch_dirs)

    # Save merged summary
    summary_path = objects_dir / "reconstruction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(merged_summary, f, indent=2)

    print(f"\n  Merged {len(batch_dirs)} batches: "
          f"{merged_summary['num_objects']} objects, "
          f"{len(merged_summary['failed'])} failed, "
          f"{len(merged_summary['skipped_doorway'])} skipped (doorway)")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 + SAM-3D-Objects Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input-image", required=True,
                        help="Input image path (e.g., generations/Flux2_00013_.png)")
    parser.add_argument("--depth-map", required=False,
                        help="Depth map path (optional, for future use)")
    parser.add_argument("--room-mesh", required=True,
                        help="Original room mesh (e.g., scene_output.glb)")
    parser.add_argument("--camera-name", required=True,
                        help="Camera name for pose lookup (e.g., bedroom_0001)")
    parser.add_argument("--images-txt", default="renders_structure_only/images.txt",
                        help="COLMAP images.txt path")
    parser.add_argument("--cameras-txt", default="renders_structure_only/cameras.txt",
                        help="COLMAP cameras.txt path")
    parser.add_argument("--output-dir", default="output",
                        help="Output directory (default: output)")

    # Step-specific options
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="SAM3 confidence threshold (default: 0.3)")
    parser.add_argument("--min-area", type=float, default=0.003,
                        help="Minimum mask area ratio for SAM3 (default: 0.003)")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Maximum objects to reconstruct (for testing)")
    parser.add_argument("--no-texture", action="store_true",
                        help="Skip texture baking in reconstruction")

    # Scale calibration options (for step 2)
    parser.add_argument("--segmentation-map", type=str, default=None,
                        help="Path to segmentation PNG from empty room render (for scale calibration)")
    parser.add_argument("--segmentation-metadata", type=str, default=None,
                        help="Path to segmentation_metadata.json")
    parser.add_argument("--scene-json", type=str, default=None,
                        help="Path to scene layout JSON")
    parser.add_argument("--room-id", type=str, default=None,
                        help="Room identifier (e.g., 'master_bedroom')")
    parser.add_argument("--scale-factor", type=float, default=None,
                        help="Manual scale correction factor (alternative to wall calibration)")
    parser.add_argument("--doorway-overlap-threshold", type=float, default=0.5,
                        help="Fraction of object overlapping doorway to trigger exclusion (default: 0.5)")
    parser.add_argument("--scale-boost", type=float, default=2.3,
                        help="Scale multiplier applied to all objects (default: 2.3)")
    parser.add_argument("--wall-calibration", action="store_true",
                        help="Enable wall-based scale calibration (default: off, uses fixed scale-boost instead)")

    # Skip steps
    parser.add_argument("--skip-step1", action="store_true",
                        help="Skip step 1 (segmentation)")
    parser.add_argument("--skip-step2", action="store_true",
                        help="Skip step 2 (reconstruction)")
    parser.add_argument("--skip-step3", action="store_true",
                        help="Skip step 3 (positioning)")
    parser.add_argument("--skip-step4", action="store_true",
                        help="Skip step 4 (rendering)")

    # Batching (GPU memory management)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Max masks per step2 invocation (default: 8). "
                             "Rooms with more masks are split into batches, each running "
                             "as a separate subprocess for full GPU memory cleanup between batches.")

    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Include AABB debug boxes in step3 output")

    # Placement mode
    parser.add_argument("--placement-mode", choices=["smart", "simple"], default="smart",
                        help="Object placement mode for step3: 'smart' (full heuristics) or 'simple' (minimal SAM3D-direct)")

    # Prompts
    parser.add_argument("--prompts", nargs="+",
                        default=["bed with bedding and pillows", "nightstand", "lamp",
                                 "dresser", "chair", "desk", "wardrobe", "plant with pot",
                                 "book", "bag", "shoes", "trash can", "laundry basket"],
                        help="Text prompts for SAM3 segmentation")

    args = parser.parse_args()

    # Get absolute paths
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    sam3d_dir = project_dir.parent / "sam-3d-objects"
    repo_root = project_dir.parent

    input_image = (project_dir / args.input_image).resolve()
    room_mesh = (project_dir / args.room_mesh).resolve()
    images_txt = (project_dir / args.images_txt).resolve()
    cameras_txt = (project_dir / args.cameras_txt).resolve()
    output_dir = (project_dir / args.output_dir).resolve()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / timestamp
    masks_dir = run_output_dir / "masks"
    objects_dir = run_output_dir / "objects"

    print("=" * 60)
    print("SAM3 Segmentation + SAM-3D-Objects Reconstruction Pipeline")
    print("=" * 60)
    print(f"\nInput image: {input_image}")
    print(f"Room mesh: {room_mesh}")
    print(f"Camera: {args.camera_name}")
    print(f"Output directory: {run_output_dir}")
    print(f"\nPrompts: {args.prompts}")

    # Validate inputs
    if not input_image.exists():
        print(f"\nERROR: Input image not found: {input_image}")
        return 1
    if not room_mesh.exists():
        print(f"\nERROR: Room mesh not found: {room_mesh}")
        return 1
    if not images_txt.exists():
        print(f"\nERROR: images.txt not found: {images_txt}")
        return 1
    if not cameras_txt.exists():
        print(f"\nERROR: cameras.txt not found: {cameras_txt}")
        return 1

    requirements = []
    sam3d_checkpoint = repo_root / "sam-3d-objects" / "checkpoints" / "hf" / "pipeline.yaml"
    if not args.skip_step1:
        requirements.append(sam3_requirement(repo_root, stage="Step 1: SAM3 segmentation"))
    if not args.skip_step2:
        requirements.append(sam3d_requirement(repo_root, stage="Step 2: SAM-3D-Objects reconstruction"))

    missing = find_missing_requirements(requirements)
    if missing:
        print("\n" + format_missing_checkpoints_error(missing), file=sys.stderr)
        return 1

    # Create output directory
    run_output_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Step 1: SAM3 Segmentation
    if not args.skip_step1:
        step1_cmd = [
            "conda", "run", "-n", "worldmesh-sam3", "--no-capture-output",
            "python", str(script_dir / "step1_segment_sam3.py"),
            "--input-image", str(input_image),
            "--output-dir", str(masks_dir),
            "--confidence", str(args.confidence),
            "--min-area", str(args.min_area),
        ]
        # Add prompts
        step1_cmd.extend(["--prompts"] + args.prompts)

        success = run_step("Step 1: SAM3 Segmentation", step1_cmd, cwd=str(project_dir))
        if not success:
            print("\nPipeline failed at Step 1")
            return 1
    else:
        print("\n[SKIPPED] Step 1: SAM3 Segmentation")
        # Assume masks exist from previous run
        if not masks_dir.exists():
            masks_dir = run_output_dir.parent / "masks"  # Try sibling
            if not masks_dir.exists():
                print(f"ERROR: No masks found at {masks_dir}")
                return 1

    # Step 2: SAM-3D-Objects Reconstruction
    if not args.skip_step2:
        # Build base step2 command args (shared across batches)
        step2_base_args = []

        # Add wall calibration arguments only when --wall-calibration is set
        if args.wall_calibration and args.segmentation_map and args.segmentation_metadata and args.scene_json and args.room_id:
            # Build camera pose string from images.txt
            camera_pose_str = get_camera_pose_string(images_txt, args.camera_name)
            if camera_pose_str:
                step2_base_args.extend([
                    "--wall-calibration",
                    "--segmentation-map", str((project_dir / args.segmentation_map).resolve()),
                    "--segmentation-metadata", str((project_dir / args.segmentation_metadata).resolve()),
                    "--scene-json", str((project_dir / args.scene_json).resolve()),
                    "--room-id", args.room_id,
                    f"--camera-pose={camera_pose_str}",  # Use = syntax to handle negative numbers
                ])
                print(f"Wall calibration enabled with camera pose: {camera_pose_str}")
            else:
                print(f"WARNING: Could not find camera pose for '{args.camera_name}' in images.txt")
                print("         Wall calibration will not be applied")
        elif args.scale_factor is not None:
            step2_base_args.extend(["--scale-factor", str(args.scale_factor)])
            print(f"Manual scale factor: {args.scale_factor}")

        if args.no_texture:
            step2_base_args.append("--no-texture")
        if args.max_objects is not None:
            step2_base_args.extend(["--max-objects", str(args.max_objects)])

        # Pass doorway overlap threshold for exclusion
        step2_base_args.extend(["--doorway-overlap-threshold", str(args.doorway_overlap_threshold)])

        # Pass scale boost
        step2_base_args.extend(["--scale-boost", str(args.scale_boost)])

        # Check if batching is needed
        object_indices, total_masks = count_object_masks(masks_dir)
        num_objects = len(object_indices)
        needs_batching = num_objects > args.batch_size

        if needs_batching:
            # Split into batches
            batches = []
            for i in range(0, num_objects, args.batch_size):
                batches.append(object_indices[i:i + args.batch_size])

            print(f"\n  Batching: {num_objects} object masks split into {len(batches)} batches "
                  f"(batch size {args.batch_size})")
            for bi, batch in enumerate(batches):
                print(f"    Batch {bi+1}: mask indices {batch}")

            batch_dirs = []
            any_success = False
            for bi, batch_indices in enumerate(batches):
                batch_dir = objects_dir / f"batch_{bi}"
                batch_dirs.append(batch_dir)

                indices_str = ",".join(str(idx) for idx in batch_indices)
                step2_cmd = [
                    "conda", "run", "-n", "worldmesh-sam3d-objects", "--no-capture-output",
                    "python", str(script_dir / "step2_reconstruct_sam3d.py"),
                    "--input-image", str(input_image),
                    "--masks-dir", str(masks_dir),
                    "--output-dir", str(batch_dir),
                    "--mask-indices", indices_str,
                    "--checkpoint", str(sam3d_checkpoint),
                ] + step2_base_args

                batch_success = run_step(
                    f"Step 2: SAM-3D-Objects Reconstruction (batch {bi+1}/{len(batches)})",
                    step2_cmd, cwd=str(sam3d_dir)
                )
                if batch_success:
                    any_success = True
                else:
                    print(f"\n  WARNING: Batch {bi+1} failed, continuing with remaining batches")

            if not any_success:
                print("\nPipeline failed at Step 2: all batches failed")
                return 1

            # Merge batch outputs into main objects directory
            print("\n  Merging batch outputs...")
            merge_batch_outputs(batch_dirs, objects_dir)

        else:
            # No batching needed — run step2 directly
            step2_cmd = [
                "conda", "run", "-n", "worldmesh-sam3d-objects", "--no-capture-output",
                "python", str(script_dir / "step2_reconstruct_sam3d.py"),
                "--input-image", str(input_image),
                "--masks-dir", str(masks_dir),
                "--output-dir", str(objects_dir),
                "--checkpoint", str(sam3d_checkpoint),
            ] + step2_base_args

            # SAM-3D-Objects needs to run from its directory
            success = run_step("Step 2: SAM-3D-Objects Reconstruction", step2_cmd, cwd=str(sam3d_dir))
            if not success:
                print("\nPipeline failed at Step 2")
                return 1
    else:
        print("\n[SKIPPED] Step 2: SAM-3D-Objects Reconstruction")

    # Step 3: Camera-to-World Transform and Scene Merging
    if not args.skip_step3:
        step3_cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(script_dir / "step3_position_merge.py"),
            "--objects-dir", str(objects_dir),
            "--room-mesh", str(room_mesh),
            "--images-txt", str(images_txt),
            "--camera-name", args.camera_name,
            "--output-dir", str(run_output_dir),
        ]
        if args.debug:
            step3_cmd.append("--debug")

        # Pass scene-json and room-id for wall-aware conflict resolution
        if args.scene_json and args.room_id:
            step3_cmd.extend([
                "--scene-json", str((project_dir / args.scene_json).resolve()),
                "--room-id", args.room_id,
            ])

        step3_cmd.extend(["--placement-mode", args.placement_mode])

        success = run_step("Step 3: Position and Merge", step3_cmd, cwd=str(project_dir))
        if not success:
            print("\nPipeline failed at Step 3")
            return 1
    else:
        print("\n[SKIPPED] Step 3: Position and Merge")

    # Step 4: Final Rendering
    if not args.skip_step4:
        step4_cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(script_dir / "step4_render_final.py"),
            "--scene-mesh", str(run_output_dir / "scene_with_objects.glb"),
            "--images-txt", str(images_txt),
            "--cameras-txt", str(cameras_txt),
            "--camera-name", args.camera_name,
            "--output-dir", str(run_output_dir),
            "--output-name", "final_render",
        ]

        success = run_step("Step 4: Final Rendering", step4_cmd, cwd=str(project_dir))
        if not success:
            print("\nPipeline failed at Step 4")
            return 1
    else:
        print("\n[SKIPPED] Step 4: Final Rendering")

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {run_output_dir}")
    print("\nGenerated files:")
    print(f"  masks/               - Segmentation masks from SAM3")
    print(f"  objects/             - Reconstructed 3D objects")
    print(f"  scene_with_objects.glb - Combined scene mesh")
    print(f"  objects_only.glb     - Objects without room")
    print(f"  final_render.png     - Rendered view from {args.camera_name}")
    print(f"  final_render_depth.png - Depth visualization")

    return 0


if __name__ == "__main__":
    sys.exit(main())
