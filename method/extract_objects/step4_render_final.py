#!/usr/bin/env python3
"""
Step 4: Final Rendering

Re-renders the combined scene from the original camera viewpoint.

Runs in the 'worldmesh' conda environment.

Usage:
    conda run -n worldmesh python step4_render_final.py \
        --scene-mesh ../output/scene_with_objects.glb \
        --images-txt ../renders_structure_only/images.txt \
        --cameras-txt ../renders_structure_only/cameras.txt \
        --camera-name bedroom_0001 \
        --output-dir ../output
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from render_backend import configure_render_backend

# Set rendering backend BEFORE importing pyrender.
RENDER_SELECTION = configure_render_backend()

import numpy as np

# Compatibility fix for numpy 2.0+ (pyrender uses deprecated np.infty)
if not hasattr(np, 'infty'):
    np.infty = np.inf

import trimesh
import pyrender
from PIL import Image
from scipy.spatial.transform import Rotation


def parse_colmap_cameras_txt(cameras_txt_path):
    """
    Parse COLMAP cameras.txt to get camera intrinsics.

    Returns:
        dict with camera intrinsics
    """
    cameras = {}

    with open(cameras_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                camera_id = int(parts[0])
                model = parts[1]

                if model == 'PINHOLE':
                    width = int(parts[2])
                    height = int(parts[3])
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])

                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy
                    }
            except (ValueError, IndexError):
                continue

    return cameras


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt file to extract camera poses.

    Returns:
        dict mapping image names to pose dicts
    """
    poses = {}

    with open(images_txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]

                base_name = Path(image_name).stem

                poses[base_name] = {
                    'image_id': image_id,
                    'quaternion': [qw, qx, qy, qz],
                    'translation': [tx, ty, tz],
                    'camera_id': camera_id,
                    'image_name': image_name
                }
            except (ValueError, IndexError):
                continue

    return poses


def colmap_to_camera_pose_matrix(quat_wxyz, translation):
    """
    Convert COLMAP pose to pyrender camera pose matrix.

    COLMAP stores world-to-camera: T_w2c
    PyRender uses camera-to-world (OpenGL convention)

    Returns:
        4x4 camera pose matrix (camera-to-world)
    """
    # scipy uses (x, y, z, w)
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    R_w2c = Rotation.from_quat(quat_xyzw).as_matrix()
    t_w2c = np.array(translation)

    # Convert to camera-to-world
    R_c2w = R_w2c.T
    camera_position = -R_c2w @ t_w2c

    # Build 4x4 pose matrix
    pose = np.eye(4)
    pose[:3, :3] = R_c2w
    pose[:3, 3] = camera_position

    return pose


def render_scene(scene_mesh, camera_pose, intrinsics, output_dir, output_name,
                 key_light_intensity=4.0, fill_light_intensity=2.5,
                 ambient_light_intensity=1.5):
    """
    Render scene from a specific camera pose.

    Args:
        scene_mesh: Trimesh scene or mesh
        camera_pose: 4x4 camera-to-world matrix
        intrinsics: Camera intrinsics dict
        output_dir: Output directory
        output_name: Base name for output files
        *_light_intensity: Lighting parameters
    """
    width = intrinsics['width']
    height = intrinsics['height']
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    # Create PyRender scene
    if isinstance(scene_mesh, trimesh.Scene):
        scene = pyrender.Scene.from_trimesh_scene(scene_mesh)
    else:
        mesh = pyrender.Mesh.from_trimesh(scene_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)

    # Create camera with exact intrinsics
    # Convert fx, fy to yfov
    yfov = 2 * np.arctan(height / (2 * fy))
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        znear=0.01, zfar=100.0
    )

    # Add camera at the specified pose
    cam_node = scene.add(camera, pose=camera_pose)

    # Add lighting
    # Key light - attached to camera
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=key_light_intensity)
    scene.add(key_light, pose=camera_pose)

    # Fill light - offset from camera
    fill_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=fill_light_intensity,
        innerConeAngle=np.pi/4,
        outerConeAngle=np.pi/3
    )
    fill_pose = camera_pose.copy()
    fill_pose[:3, 3] += camera_pose[:3, 0] * 0.5  # Offset to the right
    scene.add(fill_light, pose=fill_pose)

    # Ambient point light at scene center
    ambient_light = pyrender.PointLight(color=np.ones(3), intensity=ambient_light_intensity)
    scene.add(ambient_light, pose=np.eye(4))

    # Create renderer
    renderer = pyrender.OffscreenRenderer(width, height)

    # Render RGB and depth
    print("  Rendering RGB and depth...")
    flags = pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
    color, depth = renderer.render(scene, flags=flags)

    # Save RGB
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = output_dir / f"{output_name}.png"
    Image.fromarray(color).save(rgb_path)
    print(f"    Saved: {rgb_path}")

    # Save depth visualization (normalized, close=white, far=black)
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        depth_normalized = np.zeros_like(depth)
        near = np.percentile(valid_depth, 1)
        far = np.percentile(valid_depth, 99)
        mask = depth > 0
        depth_normalized[mask] = 1.0 - (depth[mask] - near) / (far - near + 1e-6)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_vis = (depth_normalized * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)

    depth_vis_path = output_dir / f"{output_name}_depth.png"
    Image.fromarray(depth_vis).save(depth_vis_path)
    print(f"    Saved: {depth_vis_path}")

    # Save raw depth as NPY
    depth_npy_path = output_dir / f"{output_name}_depth.npy"
    np.save(depth_npy_path, depth)
    print(f"    Saved: {depth_npy_path}")

    # Cleanup
    renderer.delete()

    return {
        'rgb_path': str(rgb_path),
        'depth_vis_path': str(depth_vis_path),
        'depth_npy_path': str(depth_npy_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Final Scene Rendering")
    parser.add_argument("--scene-mesh", required=True, help="Combined scene mesh (GLB)")
    parser.add_argument("--images-txt", required=True, help="COLMAP images.txt file")
    parser.add_argument("--cameras-txt", required=True, help="COLMAP cameras.txt file")
    parser.add_argument("--camera-name", required=True, help="Camera name (e.g., bedroom_0001)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--output-name", default="final_render", help="Output file base name")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4: Final Scene Rendering")
    print("=" * 60)
    print(f"Render backend: {RENDER_SELECTION.summary()}")

    # Parse camera intrinsics
    print(f"\nLoading camera intrinsics from: {args.cameras_txt}")
    cameras = parse_colmap_cameras_txt(args.cameras_txt)
    print(f"  Found {len(cameras)} camera(s)")

    # Parse camera poses
    print(f"\nLoading camera poses from: {args.images_txt}")
    poses = parse_colmap_images_txt(args.images_txt)
    print(f"  Found {len(poses)} poses")

    if args.camera_name not in poses:
        print(f"ERROR: Camera '{args.camera_name}' not found!")
        print(f"  Available cameras: {list(poses.keys())}")
        return 1

    pose_data = poses[args.camera_name]
    camera_id = pose_data['camera_id']

    if camera_id not in cameras:
        print(f"ERROR: Camera ID {camera_id} not found in cameras.txt!")
        return 1

    intrinsics = cameras[camera_id]
    print(f"\nUsing camera: {args.camera_name}")
    print(f"  Resolution: {intrinsics['width']}x{intrinsics['height']}")
    print(f"  Focal length: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
    print(f"  Principal point: cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")

    # Compute camera pose matrix
    camera_pose = colmap_to_camera_pose_matrix(
        pose_data['quaternion'], pose_data['translation']
    )
    camera_position = camera_pose[:3, 3]
    print(f"  Camera position: [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")

    # Load scene mesh
    print(f"\nLoading scene mesh: {args.scene_mesh}")
    scene_mesh = trimesh.load(args.scene_mesh)
    if isinstance(scene_mesh, trimesh.Scene):
        print(f"  Scene contains {len(scene_mesh.geometry)} geometries")
    else:
        print(f"  Mesh has {len(scene_mesh.vertices)} vertices, {len(scene_mesh.faces)} faces")

    # Render scene
    print("\nRendering scene...")
    result = render_scene(
        scene_mesh,
        camera_pose,
        intrinsics,
        args.output_dir,
        args.output_name
    )

    # Save render metadata
    metadata = {
        'camera_name': args.camera_name,
        'camera_pose': camera_pose.tolist(),
        'intrinsics': intrinsics,
        'outputs': result
    }
    meta_path = Path(args.output_dir) / f"{args.output_name}_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {meta_path}")

    print("\n" + "=" * 60)
    print("Rendering complete!")
    print(f"  RGB: {result['rgb_path']}")
    print(f"  Depth: {result['depth_vis_path']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
