"""
Generate depth + textured object conditioning images.

Combines:
- Structure-only depth (grayscale) for walls/floor/ceiling
- Textured RGB for furniture objects
- Optional: wall texture blending from projected bootstrap textures

Usage:
    python generate_depth_object_conditioning.py \
        --renders-dir output/final_flux_depth/renders_final \
        --structure-mesh output/final_flux_depth/structure_only.glb \
        --scene-json scene_layout_large.json

    # With wall texture blending:
    python generate_depth_object_conditioning.py \
        --renders-dir output/final_flux_depth/renders_final \
        --structure-mesh output/final_flux_depth/structure_only.glb \
        --scene-json scene_layout_large.json \
        --textured-mesh output/textured_scene.glb \
        --wall-texture-alpha 0.6
"""

from render_backend import configure_render_backend

# Set rendering backend BEFORE importing pyrender.
configure_render_backend()

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import pyrender
from pyrender.constants import RenderFlags


def _fix_scene_materials(scene):
    """Fix pyrender material defaults for correct flat rendering brightness.

    from_trimesh_scene sets incorrect baseColorFactor values:
    - Textured meshes: [0.4, 0.4, 0.4, 1.0] (darkens texture to 40%)
    - Untextured meshes: [0.3, 0.3, 0.3, 1.0] (darkens color to 30%)
    Set all to [1,1,1,1] for correct rendering with FLAT flag.
    """
    for node in scene.mesh_nodes:
        for prim in node.mesh.primitives:
            prim.material.baseColorFactor = np.array([1.0, 1.0, 1.0, 1.0])
            prim.material.metallicFactor = 0.0
            prim.material.roughnessFactor = 1.0


def _strip_textures(scene):
    """Remove UV texture maps from all materials, keeping vertex/face colors.

    Used to create a de-textured reference render from the same geometry as
    the textured mesh. Comparing textured vs de-textured renders of the SAME
    mesh avoids false positives from geometry differences (subdivision, splitting).
    """
    for node in scene.mesh_nodes:
        for prim in node.mesh.primitives:
            if prim.material.baseColorTexture is not None:
                prim.material.baseColorTexture = None


def load_colmap_cameras(cameras_txt: Path) -> dict:
    """Load camera intrinsics from COLMAP cameras.txt."""
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            # Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            # For PINHOLE: fx fy cx cy
            return {
                'width': int(parts[2]),
                'height': int(parts[3]),
                'fx': float(parts[4]),
                'fy': float(parts[5]),
                'cx': float(parts[6]),
                'cy': float(parts[7])
            }
    return {}


def load_colmap_images(images_txt: Path) -> dict:
    """Load camera extrinsics from COLMAP images.txt."""
    poses = {}
    with open(images_txt, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue

        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]  # e.g., "living_room/living_room_0000.png"

            poses[name] = {
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz]
            }

        i += 2  # Skip POINTS2D line

    return poses


def quaternion_to_rotation_matrix(qvec):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def colmap_to_camera_pose(qvec, tvec):
    """Convert COLMAP extrinsics (world-to-camera) to camera-to-world pose matrix."""
    R = quaternion_to_rotation_matrix(qvec)
    t = np.array(tvec)

    # COLMAP stores world-to-camera, we need camera-to-world
    R_c2w = R.T
    t_c2w = -R.T @ t

    pose = np.eye(4)
    pose[:3, :3] = R_c2w
    pose[:3, 3] = t_c2w
    return pose


def render_depth(scene, camera, pose, renderer):
    """Render depth map for given camera pose."""
    cam_node = scene.add(camera, pose=pose)
    # renderer.render returns (color, depth) - we only need depth
    result = renderer.render(scene, flags=RenderFlags.DEPTH_ONLY)
    scene.remove_node(cam_node)
    # Handle both tuple unpacking styles
    if isinstance(result, tuple):
        return result[1]  # depth is second element
    return result


def render_rgb(scene, camera, pose, renderer):
    """Render RGB image for given camera pose."""
    cam_node = scene.add(camera, pose=pose)
    color, _ = renderer.render(scene, flags=RenderFlags.FLAT)
    scene.remove_node(cam_node)
    return color


def create_depth_object_composite(
    rgb_image: np.ndarray,
    full_depth: np.ndarray,
    structure_depth: np.ndarray,
    depth_tolerance: float = 0.05,
    wall_texture_rgb: np.ndarray = None,
    wall_texture_alpha: float = 0.6,
    structure_rgb: np.ndarray = None,
) -> np.ndarray:
    """
    Create composite image: depth grayscale + textured objects + optional wall texture.

    Args:
        rgb_image: Full scene RGB (H, W, 3)
        full_depth: Full scene depth (H, W)
        structure_depth: Structure-only depth (H, W)
        depth_tolerance: Threshold for object detection (meters)
        wall_texture_rgb: Optional RGB from textured wall rendering (H, W, 3)
        wall_texture_alpha: Blend factor for wall textures (0=depth only, 1=full texture)
        structure_rgb: Optional untextured structure RGB for comparison-based mask detection (H, W, 3)

    Returns:
        Composite image (H, W, 3)
    """
    h, w = full_depth.shape

    # Create object mask: where full_depth is significantly closer than structure_depth
    # Objects are in front of (closer than) the structure behind them
    valid_full = full_depth > 0
    valid_struct = structure_depth > 0

    # Object pixels: full scene has something closer than structure would be
    # (or structure has no depth there, meaning we're looking through an opening)
    object_mask = valid_full & (
        (~valid_struct) |  # Structure has no depth (looking through opening - use RGB)
        (full_depth < structure_depth - depth_tolerance)  # Object is closer
    )

    # Convert structure depth to normalized grayscale
    # Use full scene depth range for consistent normalization
    depth_vis = np.zeros((h, w), dtype=np.float32)

    all_valid = full_depth > 0
    if all_valid.any():
        depth_min = full_depth[all_valid].min()
        depth_max = full_depth[all_valid].max()

        if depth_max > depth_min:
            # For structure pixels, normalize using full scene range
            # Invert: close = white (255), far = black (0)
            depth_vis[valid_struct] = 1.0 - (structure_depth[valid_struct] - depth_min) / (depth_max - depth_min)
            depth_vis = np.clip(depth_vis, 0, 1)

    depth_vis = (depth_vis * 255).astype(np.uint8)

    # Convert grayscale to RGB
    depth_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=2)

    # Composite: use depth background, overlay RGB objects
    composite = depth_rgb.copy()
    composite[object_mask] = rgb_image[object_mask]

    # Blend wall textures if provided
    if wall_texture_rgb is not None and wall_texture_alpha > 0:
        # Detect textured pixels by comparing textured render against untextured structure render.
        # When structure_rgb is available, even small differences indicate a projected texture.
        # Use a low threshold (3) so near-white textures aren't missed — the untextured
        # walls are [240,240,240], so a white texture (255,255,255) only differs by 15.
        if structure_rgb is not None:
            color_diff = np.abs(wall_texture_rgb.astype(np.int16) - structure_rgb.astype(np.int16)).max(axis=2)
            textured_wall_mask = color_diff > 3
        else:
            # Fallback: compare against flat gray (less reliable, needs higher threshold)
            flat_gray = np.array([240, 240, 240], dtype=np.uint8)
            color_diff = np.abs(wall_texture_rgb.astype(np.int16) - flat_gray.astype(np.int16)).max(axis=2)
            textured_wall_mask = color_diff > 20

        # Only apply to structure pixels (not objects, not background)
        wall_blend_mask = textured_wall_mask & valid_struct & ~object_mask

        if wall_blend_mask.any():
            # Alpha blend: composite = alpha * texture + (1-alpha) * depth
            alpha = wall_texture_alpha
            composite[wall_blend_mask] = (
                alpha * wall_texture_rgb[wall_blend_mask].astype(np.float32) +
                (1 - alpha) * composite[wall_blend_mask].astype(np.float32)
            ).astype(np.uint8)

    return composite


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth + textured object conditioning images'
    )
    parser.add_argument('--renders-dir', required=True,
                       help='Path to renders_final directory')
    parser.add_argument('--structure-mesh', required=True,
                       help='Path to structure_only.glb')
    parser.add_argument('--scene-json', required=True,
                       help='Path to scene JSON file')
    parser.add_argument('--depth-tolerance', type=float, default=0.05,
                       help='Depth tolerance for object detection (meters)')
    parser.add_argument('--textured-mesh', default=None,
                       help='Path to textured wall mesh (enables wall texture blending)')
    parser.add_argument('--wall-texture-alpha', type=float, default=0.6,
                       help='Blend factor for wall textures (0=depth only, 1=full texture, default: 0.6)')
    parser.add_argument('--rooms', nargs='*', default=None,
                       help='Specific rooms to process (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    renders_dir = Path(args.renders_dir)

    # Load camera info
    print(f"Loading camera parameters from {renders_dir / 'cameras.txt'}...")
    camera_params = load_colmap_cameras(renders_dir / 'cameras.txt')
    print(f"  Resolution: {camera_params['width']}x{camera_params['height']}")
    print(f"  Focal length: fx={camera_params['fx']:.1f}, fy={camera_params['fy']:.1f}")

    print(f"Loading camera poses from {renders_dir / 'images.txt'}...")
    image_poses = load_colmap_images(renders_dir / 'images.txt')
    print(f"  Found {len(image_poses)} camera poses")

    # Load structure mesh
    print(f"Loading structure mesh: {args.structure_mesh}")
    structure_mesh = trimesh.load(args.structure_mesh)

    # Create pyrender scene for structure-only depth
    if isinstance(structure_mesh, trimesh.Scene):
        pr_scene = pyrender.Scene.from_trimesh_scene(structure_mesh)
        _fix_scene_materials(pr_scene)
        print(f"  Loaded scene with {len(structure_mesh.geometry)} geometries")
    else:
        pr_scene = pyrender.Scene()
        pr_scene.add(pyrender.Mesh.from_trimesh(structure_mesh))
        _fix_scene_materials(pr_scene)
        print(f"  Loaded single mesh")

    # Load textured mesh for wall texture blending (optional)
    textured_pr_scene = None
    detex_pr_scene = None
    if args.textured_mesh:
        print(f"Loading textured mesh: {args.textured_mesh}")
        textured_mesh = trimesh.load(args.textured_mesh)
        if isinstance(textured_mesh, trimesh.Scene):
            textured_pr_scene = pyrender.Scene.from_trimesh_scene(textured_mesh)
            _fix_scene_materials(textured_pr_scene)
            # Create de-textured version from same mesh for comparison-based mask.
            # Same geometry guarantees pixel-perfect comparison — only actual
            # texture content causes differences (no false positives from
            # subdivision/splitting geometry changes vs structure mesh).
            detex_pr_scene = pyrender.Scene.from_trimesh_scene(textured_mesh)
            _fix_scene_materials(detex_pr_scene)
            _strip_textures(detex_pr_scene)
            print(f"  Loaded textured scene with {len(textured_mesh.geometry)} geometries")
            print(f"  Wall texture alpha: {args.wall_texture_alpha}")
        else:
            textured_pr_scene = pyrender.Scene()
            textured_pr_scene.add(pyrender.Mesh.from_trimesh(textured_mesh))
            _fix_scene_materials(textured_pr_scene)
            detex_pr_scene = pyrender.Scene()
            detex_pr_scene.add(pyrender.Mesh.from_trimesh(textured_mesh))
            _fix_scene_materials(detex_pr_scene)
            _strip_textures(detex_pr_scene)
            print(f"  Loaded textured single mesh")

    # Setup camera
    width = camera_params['width']
    height = camera_params['height']
    fx, fy = camera_params['fx'], camera_params['fy']

    # Calculate FOV from focal length
    yfov = 2 * np.arctan(height / (2 * fy))
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width/height)

    # Create renderer
    print(f"Creating offscreen renderer ({width}x{height})...")
    renderer = pyrender.OffscreenRenderer(width, height)

    # Load scene JSON for room list
    with open(args.scene_json, 'r') as f:
        scene_data = json.load(f)

    room_ids = args.rooms or [room['id'] for room in scene_data['rooms']]
    print(f"Processing rooms: {', '.join(room_ids)}")

    total_processed = 0
    total_skipped = 0

    for room_id in room_ids:
        print(f"\n{'='*60}")
        print(f"Processing room: {room_id}")
        print(f"{'='*60}")

        images_dir = renders_dir / 'images' / room_id
        depth_dir = renders_dir / 'depth' / room_id

        if not images_dir.exists():
            print(f"  Skipping - no images directory")
            continue

        # Find all RGB images (not *_with_edges.png or *_depth_objects.png)
        rgb_files = sorted([
            f for f in images_dir.glob('*.png')
            if not f.name.endswith('_with_edges.png')
            and not f.name.endswith('_depth_objects.png')
        ])

        print(f"  Found {len(rgb_files)} RGB images")

        for rgb_path in rgb_files:
            # Get pose for this image
            image_key = f"{room_id}/{rgb_path.name}"
            if image_key not in image_poses:
                if args.verbose:
                    print(f"  Skipping {rgb_path.name} - no pose found")
                total_skipped += 1
                continue

            pose_data = image_poses[image_key]
            pose = colmap_to_camera_pose(pose_data['qvec'], pose_data['tvec'])

            # Load RGB image
            rgb_image = np.array(Image.open(rgb_path).convert('RGB'))

            # Load full scene depth (raw .npy)
            # RGB is living_room_0000.png, depth is living_room_0000_depth.npy
            depth_npy_path = depth_dir / (rgb_path.stem + '_depth.npy')
            if not depth_npy_path.exists():
                if args.verbose:
                    print(f"  Skipping {rgb_path.name} - no depth .npy found")
                total_skipped += 1
                continue

            full_depth = np.load(depth_npy_path)

            # Render structure-only depth
            structure_depth = render_depth(pr_scene, camera, pose, renderer)

            # Render wall texture RGB if textured mesh is available
            wall_texture_rgb = None
            structure_rgb = None
            if textured_pr_scene is not None:
                wall_texture_rgb = render_rgb(textured_pr_scene, camera, pose, renderer)
                # Render de-textured version of same mesh for comparison.
                # Using detex (same geometry, textures stripped) instead of
                # structure mesh avoids false positives from geometry differences
                # (project_wall_texture.py subdivides/splits current room surfaces).
                structure_rgb = render_rgb(detex_pr_scene, camera, pose, renderer)

            # Create composite
            composite = create_depth_object_composite(
                rgb_image, full_depth, structure_depth,
                depth_tolerance=args.depth_tolerance,
                wall_texture_rgb=wall_texture_rgb,
                wall_texture_alpha=args.wall_texture_alpha,
                structure_rgb=structure_rgb,
            )

            # Save
            output_name = rgb_path.stem + '_depth_objects.png'
            output_path = images_dir / output_name
            Image.fromarray(composite).save(output_path)

            total_processed += 1
            if args.verbose:
                print(f"  Generated: {output_name}")
            elif total_processed % 10 == 0:
                print(f"  Processed {total_processed} images...")

    renderer.delete()

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Generated: {total_processed} depth+objects conditioning images")
    print(f"Skipped: {total_skipped} images (no pose or depth)")
    if textured_pr_scene is not None:
        print(f"Wall texture blending: enabled (alpha={args.wall_texture_alpha})")
    print(f"\nOutput location: {renders_dir}/images/*/*_depth_objects.png")


if __name__ == '__main__':
    main()
