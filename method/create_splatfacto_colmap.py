#!/usr/bin/env python3
"""Create nerfstudio splatfacto training data using native COLMAP format with coordinate fix.

Includes point cloud generation from depth maps for 3DGS initialization.
Optionally exports depth maps for depth-supervised training.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    # scipy uses (x, y, z, w) order
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_matrix()


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    r = Rotation.from_matrix(R)
    quat = r.as_quat()  # returns (x, y, z, w)
    return quat[3], quat[0], quat[1], quat[2]  # return as (w, x, y, z)


def get_image_dimensions(image_dir):
    """Get (width, height) of the smallest .png image in a directory.

    Returns the smallest resolution found, so that higher-res images (e.g. NB2 2x)
    are downscaled to match the original render/bootstrap resolution.
    """
    if not os.path.isdir(image_dir):
        return None
    min_size = None
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith('.png') and not fname.startswith('bootstrap'):
            img_path = os.path.join(image_dir, fname)
            if os.path.islink(img_path):
                img_path = os.path.realpath(img_path)
            with Image.open(img_path) as img:
                w, h = img.size
                if min_size is None or w * h < min_size[0] * min_size[1]:
                    min_size = (w, h)
    return min_size


def write_scaled_cameras_txt(src_cameras_txt, dst_cameras_txt, actual_width, actual_height):
    """Copy cameras.txt, scaling PINHOLE intrinsics if image dimensions differ.

    When generated images have different resolution than the renders (e.g., NB2
    outputs at 2x), the intrinsics must be scaled to match actual image dimensions
    so nerfstudio doesn't crash on dimension mismatch.

    Args:
        src_cameras_txt: Path to source cameras.txt
        dst_cameras_txt: Path to write (possibly scaled) cameras.txt
        actual_width: Actual image width in pixels
        actual_height: Actual image height in pixels

    Returns:
        Tuple of (scale_x, scale_y) applied to intrinsics
    """
    with open(src_cameras_txt, 'r') as f:
        lines = f.readlines()

    scale_x, scale_y = 1.0, 1.0
    output_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or not stripped:
            output_lines.append(line)
            continue
        parts = stripped.split()
        if len(parts) >= 8 and parts[1] == 'PINHOLE':
            cam_w = int(parts[2])
            cam_h = int(parts[3])
            if cam_w != actual_width or cam_h != actual_height:
                scale_x = actual_width / cam_w
                scale_y = actual_height / cam_h
                fx = float(parts[4]) * scale_x
                fy = float(parts[5]) * scale_y
                cx = float(parts[6]) * scale_x
                cy = float(parts[7]) * scale_y
                output_lines.append(
                    f"{parts[0]} PINHOLE {actual_width} {actual_height} {fx} {fy} {cx} {cy}\n"
                )
                print(f"  Scaled cameras.txt intrinsics: {cam_w}x{cam_h} -> "
                      f"{actual_width}x{actual_height} (scale {scale_x:.2f}x{scale_y:.2f})")
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)

    with open(dst_cameras_txt, 'w') as f:
        f.writelines(output_lines)

    return scale_x, scale_y


def count_valid_depth_pixels(depth_path):
    """Count non-zero pixels in a depth map without loading the full array into memory."""
    depth = np.load(depth_path)
    return int(np.count_nonzero(depth > 0))


def compute_subsample_factor(depth_paths, max_points=5_000_000):
    """
    Compute subsample factor so the total point cloud stays under max_points.

    Args:
        depth_paths: List of paths to .npy depth files
        max_points: Target maximum number of points

    Returns:
        subsample: Integer subsample factor (minimum 1)
    """
    total_valid = 0
    # Sample up to 5 images to estimate average valid pixels, then extrapolate
    sample_count = min(len(depth_paths), 5)
    for path in depth_paths[:sample_count]:
        total_valid += count_valid_depth_pixels(path)
    if sample_count < len(depth_paths):
        avg_per_image = total_valid / sample_count
        total_valid = int(avg_per_image * len(depth_paths))

    if total_valid <= max_points:
        return 1

    subsample = int(np.ceil(total_valid / max_points))
    return subsample


def depth_to_pointcloud(depth_path, rgb_path, pose, intrinsics, subsample=4):
    """
    Convert depth map to 3D points in world coordinates.

    Args:
        depth_path: Path to .npy depth file
        rgb_path: Path to RGB image for colors (optional, can be None)
        pose: Dict with 'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz' (original world-to-camera)
        intrinsics: Tuple of (fx, fy, cx, cy)
        subsample: Subsample factor to reduce point count

    Returns:
        points: Nx3 array of world coordinates
        colors: Nx3 array of RGB colors (0-255)
    """
    # Load depth map
    depth = np.load(depth_path)

    # Get valid pixels (depth > 0)
    v, u = np.where(depth > 0)
    d = depth[v, u]

    # Subsample to reduce point count
    indices = np.arange(0, len(u), subsample)
    u, v, d = u[indices], v[indices], d[indices]

    # Unproject to camera coordinates (OpenGL convention: -Z forward)
    fx, fy, cx, cy = intrinsics
    x_c = (u - cx) / fx * d
    y_c = (v - cy) / fy * d
    z_c = d  # Note: In OpenGL, positive Z is toward viewer

    # Our rendering uses OpenGL convention: camera looks along -Z
    # So actual points are at -z_c in camera space
    points_cam = np.stack([x_c, -y_c, -z_c], axis=1)  # Flip Y and Z for OpenGL

    # Get rotation matrix from original pose (world-to-camera)
    R_w2c = quaternion_to_rotation_matrix(
        float(pose['qw']), float(pose['qx']),
        float(pose['qy']), float(pose['qz'])
    )
    t_w2c = np.array([float(pose['tx']), float(pose['ty']), float(pose['tz'])])

    # Invert to get camera-to-world transform
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c  # Camera center in world coordinates

    # Transform points to world coordinates
    points_world = (R_c2w @ points_cam.T).T + t_c2w

    # Get colors from RGB image if available
    # Note: v, u are already subsampled above, so use them directly
    if rgb_path and os.path.exists(rgb_path):
        rgb_img = np.array(Image.open(rgb_path))
        # Get colors at the subsampled pixel locations
        colors = rgb_img[v, u]
        if len(colors.shape) > 1 and colors.shape[1] == 4:  # RGBA
            colors = colors[:, :3]
    else:
        # Default gray color
        colors = np.full((len(u), 3), 128, dtype=np.uint8)

    return points_world, colors


def transform_points_for_nerfstudio(points):
    """
    Transform points from Z-up world to Y-up world (matching pose conversion).

    Our world: Z-up (architectural)
    Nerfstudio world: Y-up (graphics)

    Transform: (x, y, z) -> (x, z, -y)
    """
    world_transform = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    return (world_transform @ points.T).T


def generate_pointcloud_from_depths(depth_dir, rgb_dir, poses, selected_indices, intrinsics,
                                     subsample=4, room_name="master_bedroom"):
    """
    Generate combined point cloud from multiple depth maps.

    Args:
        depth_dir: Directory containing depth .npy files
        rgb_dir: Directory containing RGB images
        poses: Dict of poses indexed by image number
        selected_indices: List of image indices to use
        intrinsics: Camera intrinsics (fx, fy, cx, cy)
        subsample: Subsampling factor per image
        room_name: Room name prefix for filenames

    Returns:
        all_points: Nx3 array of world coordinates (transformed to Y-up)
        all_colors: Nx3 array of RGB colors
    """
    all_points = []
    all_colors = []

    for idx in selected_indices:
        if idx not in poses:
            print(f"  Warning: No pose found for index {idx}, skipping")
            continue

        # Construct file paths
        depth_path = os.path.join(depth_dir, f"{room_name}_{idx:04d}_depth.npy")
        rgb_path = os.path.join(rgb_dir, f"{room_name}_{idx:04d}.png")

        if not os.path.exists(depth_path):
            print(f"  Warning: Depth file not found: {depth_path}")
            continue

        # Generate point cloud from this depth map
        points, colors = depth_to_pointcloud(
            depth_path, rgb_path, poses[idx], intrinsics, subsample
        )

        print(f"  Image {idx}: {len(points)} points")
        all_points.append(points)
        all_colors.append(colors)

    if not all_points:
        return np.array([]), np.array([])

    # Combine all points
    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    # Transform to nerfstudio world coordinates (Z-up to Y-up)
    all_points = transform_points_for_nerfstudio(all_points)

    return all_points, all_colors


def convert_depths_for_nerfstudio(depth_dir, output_dir, selected_indices, room_name="master_bedroom",
                                   output_format="npy", scale_factor=1.0):
    """
    Convert depth maps to nerfstudio format.

    Nerfstudio's colmap dataparser expects depth files in a 'depths' folder with names
    matching the image names (e.g., depths/0.png for images/0.png).

    Args:
        depth_dir: Directory containing depth .npy files (e.g., "master_bedroom_0004_depth.npy")
        output_dir: Output directory for converted depth maps
        selected_indices: List of image indices to convert
        room_name: Room name prefix in source depth filenames
        output_format: Output format - "npy" (direct copy, requires scale_factor=1.0 in config) or
                       "png" (16-bit PNG, can use depth_unit_scale_factor in config)
        scale_factor: Scale to apply before saving (e.g., 1000 for meters -> millimeters when using PNG)

    Returns:
        List of converted depth file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    converted_files = []

    for idx in selected_indices:
        # Source depth file: e.g., master_bedroom_0004_depth.npy
        src_path = os.path.join(depth_dir, f"{room_name}_{idx:04d}_depth.npy")

        if not os.path.exists(src_path):
            print(f"  Warning: Depth file not found: {src_path}")
            continue

        # Load depth map
        depth = np.load(src_path).astype(np.float32)

        if output_format == "npy":
            # Output as .npy file (nerfstudio can load these directly)
            # Note: colmap dataparser expects .png, so this requires modifying the dataparser
            # or using nerfstudio dataparser instead
            dst_path = os.path.join(output_dir, f"{idx}.npy")
            np.save(dst_path, depth)
        elif output_format == "png":
            # Output as 16-bit PNG
            # Apply scale factor (e.g., 1000 for meters -> millimeters)
            depth_scaled = (depth * scale_factor).astype(np.uint16)
            dst_path = os.path.join(output_dir, f"{idx}.png")
            cv2.imwrite(dst_path, depth_scaled)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

        converted_files.append(dst_path)
        print(f"  Converted depth {idx}: {src_path} -> {dst_path}")

    return converted_files


def write_points3d_txt(output_path, points, colors):
    """
    Write points to COLMAP points3D.txt format.

    Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
    """
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")

        for i, (pt, col) in enumerate(zip(points, colors), start=1):
            # POINT3D_ID X Y Z R G B ERROR (no track info needed)
            f.write(f"{i} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])} 0.0\n")


def convert_pose_for_nerfstudio(qw, qx, qy, qz, tx, ty, tz):
    """
    Convert pose from our rendering pipeline to nerfstudio/COLMAP convention.

    Our pipeline:
    - World: Z-up (architectural convention)
    - Camera: OpenGL (-Z forward, +Y up)
    - Stored as world-to-camera in COLMAP format

    Nerfstudio expects:
    - World: Y-up (graphics convention)
    - Camera: COLMAP (+Z forward, -Y up)

    We need to:
    1. Transform world coords from Z-up to Y-up (rotate -90° around X)
    2. Transform camera coords from OpenGL to COLMAP (flip Y and Z)
    """
    # Convert quaternion to rotation matrix (world-to-camera)
    R_w2c = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    t_w2c = np.array([tx, ty, tz])

    # World coordinate transform: Z-up to Y-up
    # This rotates -90° around X axis: (x, y, z) -> (x, z, -y)
    world_transform = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # Camera coordinate transform: OpenGL to COLMAP
    # OpenGL: -Z forward, +Y up -> COLMAP: +Z forward, -Y up
    camera_transform = np.diag([1, -1, -1])

    # Combined transformation:
    # R_new = camera_transform @ R_w2c @ world_transform.T
    # t_new = camera_transform @ t_w2c
    R_new = camera_transform @ R_w2c @ world_transform.T
    t_new = camera_transform @ t_w2c

    # Convert back to quaternion
    qw_new, qx_new, qy_new, qz_new = rotation_matrix_to_quaternion(R_new)

    return qw_new, qx_new, qy_new, qz_new, t_new[0], t_new[1], t_new[2]


def parse_colmap_images(images_txt_path, room_filter="master_bedroom"):
    """Parse COLMAP images.txt and return entries indexed by image number for specified room."""
    entries = {}

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # Skip comments and empty lines
        if line.startswith('#') or not line:
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        name = parts[9]

        if room_filter not in name:
            # Skip the POINTS2D line (empty in this case)
            if i < len(lines):
                i += 1
            continue

        # Extract index from filename (e.g., master_bedroom_0004.png -> 4)
        match = re.search(r'_(\d+)\.png$', name)
        if match:
            idx = int(match.group(1))
            # Store the full line (pose data)
            entries[idx] = {
                'qw': parts[1],
                'qx': parts[2],
                'qy': parts[3],
                'qz': parts[4],
                'tx': parts[5],
                'ty': parts[6],
                'tz': parts[7],
                'camera_id': parts[8],
                'original_name': name
            }

        # Skip the POINTS2D line
        if i < len(lines):
            i += 1

    return entries


def create_colmap_structure(best_dir, colmap_images_path, colmap_cameras_path, output_dir,
                            depth_dir=None, rgb_dir=None, intrinsics=None, subsample=4,
                            export_depths=False, depth_output_format="png", depth_scale=1000.0,
                            room_name="master_bedroom"):
    """
    Create COLMAP folder structure for nerfstudio with optional point cloud generation.

    Args:
        best_dir: Directory containing selected "best" images
        colmap_images_path: Path to original images.txt with poses
        colmap_cameras_path: Path to cameras.txt
        output_dir: Output directory for COLMAP structure
        depth_dir: Directory containing depth .npy files (optional, for point cloud)
        rgb_dir: Directory containing RGB images (optional, for point colors)
        intrinsics: Camera intrinsics (fx, fy, cx, cy) (required if depth_dir provided)
        subsample: Subsampling factor for point cloud (default 4)
        export_depths: Whether to export depth maps for depth-supervised training
        depth_output_format: Format for exported depths ("png" or "npy")
        depth_scale: Scale factor for depth export (e.g., 1000 for meters -> mm when using PNG)
        room_name: Name of the room to filter poses for
    """

    # Create directories (nerfstudio expects colmap/sparse/0/)
    sparse_dir = os.path.join(output_dir, 'colmap', 'sparse', '0')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Copy cameras.txt, scaling intrinsics if image resolution differs from render resolution
    actual_dims = get_image_dimensions(best_dir)
    dst_cameras = os.path.join(sparse_dir, 'cameras.txt')
    if actual_dims:
        scale_x, scale_y = write_scaled_cameras_txt(
            colmap_cameras_path, dst_cameras, actual_dims[0], actual_dims[1]
        )
        if scale_x == 1.0 and scale_y == 1.0:
            print(f"Copied cameras.txt (resolution matches)")
        else:
            print(f"Wrote scaled cameras.txt")
    else:
        shutil.copy(colmap_cameras_path, dst_cameras)
        print(f"Copied cameras.txt (no images found to check resolution)")

    # Parse poses for specified room
    poses = parse_colmap_images(colmap_images_path, room_filter=room_name)
    print(f"Found {len(poses)} {room_name} poses")

    # Get selected images from best folder (handle generated_XXXX.png or X.png naming)
    best_images = sorted([f for f in os.listdir(best_dir) if f.endswith('.png') and not f.startswith('bootstrap')])
    selected_indices = []
    for f in best_images:
        # Handle "generated_0004.png" format
        match = re.search(r'generated_(\d+)(?:_ref\d+)?\.png$', f)
        if match:
            selected_indices.append(int(match.group(1)))
        else:
            # Handle "4.png" format
            try:
                selected_indices.append(int(f.replace('.png', '')))
            except ValueError:
                pass
    print(f"Selected images: {selected_indices}")

    # Create filtered images.txt
    images_txt_path = os.path.join(sparse_dir, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(selected_indices)}\n")

        for new_id, idx in enumerate(selected_indices, start=1):
            if idx not in poses:
                print(f"Warning: No pose found for index {idx}")
                continue

            pose = poses[idx]
            # Use simple filename (matching symlinked images)
            filename = f"{idx}.png"

            # Convert from our pipeline convention to nerfstudio/COLMAP convention
            qw, qx, qy, qz, tx, ty, tz = convert_pose_for_nerfstudio(
                float(pose['qw']), float(pose['qx']), float(pose['qy']), float(pose['qz']),
                float(pose['tx']), float(pose['ty']), float(pose['tz'])
            )

            # Write pose line with converted coordinates
            f.write(f"{new_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {pose['camera_id']} {filename}\n")
            # Write empty POINTS2D line
            f.write("\n")

    print(f"Created images.txt with {len(selected_indices)} entries")

    # Generate point cloud from depth maps if available
    points3d_path = os.path.join(sparse_dir, 'points3D.txt')
    if depth_dir and intrinsics:
        print(f"\nGenerating point cloud from depth maps (subsample={subsample})...")
        points, colors = generate_pointcloud_from_depths(
            depth_dir=depth_dir,
            rgb_dir=rgb_dir,
            poses=poses,
            selected_indices=selected_indices,
            intrinsics=intrinsics,
            subsample=subsample,
            room_name=room_name
        )

        if len(points) > 0:
            write_points3d_txt(points3d_path, points, colors)
            print(f"Created points3D.txt with {len(points)} points")
        else:
            # Fallback to empty points3D.txt
            with open(points3d_path, 'w') as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                f.write("# Number of points: 0\n")
            print("Warning: No points generated, created empty points3D.txt")
    else:
        # Create empty points3D.txt
        with open(points3d_path, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
        print(f"Created empty points3D.txt (no depth_dir provided)")

    # Symlink images (rename generated_XXXX.png to X.png for COLMAP compatibility)
    # If an image doesn't match the target resolution, resize it instead of symlinking
    best_dir_abs = os.path.abspath(best_dir)
    for img in best_images:
        src = os.path.join(best_dir_abs, img)
        # Convert generated_0004.png to 4.png
        match = re.search(r'generated_(\d+)(?:_ref\d+)?\.png$', img)
        if match:
            dst_name = f"{int(match.group(1))}.png"
        else:
            dst_name = img
        dst = os.path.join(images_dir, dst_name)
        if os.path.exists(dst):
            os.remove(dst)
        # Check if image needs resizing to match target resolution
        if actual_dims:
            with Image.open(src) as im:
                if im.size != actual_dims:
                    im.resize(actual_dims, Image.LANCZOS).save(dst)
                    print(f"  Resized {img} from {im.size} to {actual_dims}")
                    continue
        os.symlink(src, dst)
    print(f"Linked {len(best_images)} images to {images_dir}")

    # Export depth maps for depth-supervised training
    if export_depths and depth_dir:
        depths_output_dir = os.path.join(output_dir, 'depths')
        print(f"\nExporting depth maps (format={depth_output_format}, scale={depth_scale})...")
        convert_depths_for_nerfstudio(
            depth_dir=depth_dir,
            output_dir=depths_output_dir,
            selected_indices=selected_indices,
            room_name=room_name,
            output_format=depth_output_format,
            scale_factor=depth_scale
        )
        print(f"Created depths/ folder with {len(selected_indices)} depth maps")

    return output_dir


def _compute_overlap_exclusions(poses, min_angle_deg=10):
    """
    Find camera indices that overlap with coverage cameras (indices 0 and 1).

    Extracts world positions from COLMAP poses, computes azimuths relative to
    the room center (midpoint of cameras 0 and 1), and returns indices whose
    azimuth is within min_angle_deg of either coverage camera.

    Args:
        poses: Dict mapping camera index to pose dict with qw/qx/qy/qz/tx/ty/tz
        min_angle_deg: Minimum angular separation in degrees

    Returns:
        Set of camera indices to exclude
    """
    if 0 not in poses or 1 not in poses:
        return set()

    # Extract world positions from COLMAP world-to-camera poses
    # camera_world_pos = -R^T @ t
    def world_position(pose):
        R = quaternion_to_rotation_matrix(
            float(pose['qw']), float(pose['qx']),
            float(pose['qy']), float(pose['qz'])
        )
        t = np.array([float(pose['tx']), float(pose['ty']), float(pose['tz'])])
        return -R.T @ t

    pos0 = world_position(poses[0])
    pos1 = world_position(poses[1])
    center = (pos0 + pos1) / 2

    # Azimuths of coverage cameras (XY plane, Z-up convention)
    cov_azimuths = []
    for p in (pos0, pos1):
        az = np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) % 360
        cov_azimuths.append(az)

    exclusions = set()
    for idx, pose in poses.items():
        if idx in (0, 1):
            continue
        p = world_position(pose)
        az = np.degrees(np.arctan2(p[1] - center[1], p[0] - center[0])) % 360
        for caz in cov_azimuths:
            diff = abs(az - caz)
            if diff > 180:
                diff = 360 - diff
            if diff < min_angle_deg:
                print(f"  Auto-excluding camera {idx} (azimuth {az:.1f}° within {min_angle_deg}° of coverage camera at {caz:.1f}°)")
                exclusions.add(idx)
                break

    return exclusions


def create_colmap_structure_all_rooms(pipeline_output, rooms, output_dir, intrinsics, subsample=8,
                                       max_points=5_000_000,
                                       export_depths=True, depth_output_format="png", depth_scale=1000.0,
                                       exclude_indices=None):
    """
    Create COLMAP folder structure for all rooms combined.
    """
    colmap_images_path = f"{pipeline_output}/renders_final/images.txt"
    colmap_cameras_path = f"{pipeline_output}/renders_final/cameras.txt"

    # Create directories
    sparse_dir = os.path.join(output_dir, 'colmap', 'sparse', '0')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Find actual image dimensions from generated images
    actual_dims = None
    for room_name in rooms:
        gen_dir = f"{pipeline_output}/flux_final/{room_name}/generated"
        actual_dims = get_image_dimensions(gen_dir)
        if actual_dims:
            break

    # Copy cameras.txt, scaling intrinsics if image resolution differs from render resolution
    dst_cameras = os.path.join(sparse_dir, 'cameras.txt')
    if actual_dims:
        scale_x, scale_y = write_scaled_cameras_txt(
            colmap_cameras_path, dst_cameras, actual_dims[0], actual_dims[1]
        )
        if scale_x == 1.0 and scale_y == 1.0:
            print(f"Copied cameras.txt (resolution matches)")
        else:
            print(f"Wrote scaled cameras.txt")
    else:
        shutil.copy(colmap_cameras_path, dst_cameras)
        print(f"Copied cameras.txt (no images found to check resolution)")

    # Collect all poses and images from all rooms
    all_poses = {}
    all_selected = []  # List of (room_name, idx, src_path)

    for room_name in rooms:
        best_dir = f"{pipeline_output}/flux_final/{room_name}/generated"
        if not os.path.exists(best_dir):
            print(f"Skipping {room_name} - no generated images found")
            continue

        # Parse poses for this room
        poses = parse_colmap_images(colmap_images_path, room_filter=room_name)
        print(f"Found {len(poses)} {room_name} poses")

        # Auto-exclude cameras overlapping with coverage cameras (indices 0, 1)
        auto_exclusions = _compute_overlap_exclusions(poses, min_angle_deg=10)
        room_exclude = set(exclude_indices) if exclude_indices else set()
        room_exclude |= auto_exclusions

        # Get generated images
        best_images = sorted([f for f in os.listdir(best_dir) if f.endswith('.png') and not f.startswith('bootstrap')])
        for f in best_images:
            match = re.search(r'generated_(\d+)(?:_ref\d+)?\.png$', f)
            if match:
                idx = int(match.group(1))
                if room_exclude and idx in room_exclude:
                    continue
                if idx in poses:
                    src_path = os.path.join(os.path.abspath(best_dir), f)
                    all_selected.append((room_name, idx, src_path))
                    all_poses[(room_name, idx)] = poses[idx]

    print(f"\nTotal images across all rooms: {len(all_selected)}")

    # Create images.txt with all images
    images_txt_path = os.path.join(sparse_dir, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(all_selected)}\n")

        for new_id, (room_name, idx, src_path) in enumerate(all_selected, start=1):
            pose = all_poses[(room_name, idx)]
            # Use unique filename: room_idx.png
            filename = f"{room_name}_{idx:04d}.png"

            # Convert pose
            qw, qx, qy, qz, tx, ty, tz = convert_pose_for_nerfstudio(
                float(pose['qw']), float(pose['qx']), float(pose['qy']), float(pose['qz']),
                float(pose['tx']), float(pose['ty']), float(pose['tz'])
            )

            f.write(f"{new_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {pose['camera_id']} {filename}\n")
            f.write("\n")

    print(f"Created images.txt with {len(all_selected)} entries")

    # Symlink all images (resize mismatched images instead of symlinking)
    for room_name, idx, src_path in all_selected:
        dst_name = f"{room_name}_{idx:04d}.png"
        dst = os.path.join(images_dir, dst_name)
        if os.path.exists(dst):
            os.remove(dst)
        # Check if image needs resizing to match target resolution
        if actual_dims:
            with Image.open(src_path) as im:
                if im.size != actual_dims:
                    im.resize(actual_dims, Image.LANCZOS).save(dst)
                    print(f"  Resized {dst_name} from {im.size} to {actual_dims}")
                    continue
        os.symlink(src_path, dst)
    print(f"Linked {len(all_selected)} images to {images_dir}")

    # Generate combined point cloud from all rooms
    points3d_path = os.path.join(sparse_dir, 'points3D.txt')
    all_points = []
    all_colors = []

    # Auto-compute subsample factor if requested
    if subsample == "auto" or subsample is None:
        # Collect all depth file paths
        all_depth_paths = []
        for room_name in rooms:
            depth_dir = f"{pipeline_output}/renders_final/depth/{room_name}"
            if not os.path.exists(depth_dir):
                continue
            room_indices = [idx for (r, idx, _) in all_selected if r == room_name]
            for idx in room_indices:
                dp = os.path.join(depth_dir, f"{room_name}_{idx:04d}_depth.npy")
                if os.path.exists(dp):
                    all_depth_paths.append(dp)
        subsample = compute_subsample_factor(all_depth_paths, max_points=max_points)
        print(f"\nAuto-computed subsample factor: {subsample} (targeting <= {max_points:,} points from {len(all_depth_paths)} depth maps)")

    for room_name in rooms:
        depth_dir = f"{pipeline_output}/renders_final/depth/{room_name}"
        rgb_dir = f"{pipeline_output}/renders_final/images/{room_name}"

        if not os.path.exists(depth_dir):
            continue

        # Get indices for this room
        room_indices = [idx for (r, idx, _) in all_selected if r == room_name]
        room_poses = {idx: all_poses[(room_name, idx)] for idx in room_indices if (room_name, idx) in all_poses}

        if room_indices and room_poses:
            print(f"\nGenerating point cloud for {room_name} ({len(room_indices)} images)...")
            points, colors = generate_pointcloud_from_depths(
                depth_dir=depth_dir,
                rgb_dir=rgb_dir,
                poses=room_poses,
                selected_indices=room_indices,
                intrinsics=intrinsics,
                subsample=subsample,
                room_name=room_name
            )
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)

    if all_points:
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        write_points3d_txt(points3d_path, combined_points, combined_colors)
        print(f"\nCreated points3D.txt with {len(combined_points)} total points")
    else:
        with open(points3d_path, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
        print("Created empty points3D.txt")

    # Export depth maps for all rooms
    if export_depths:
        depths_output_dir = os.path.join(output_dir, 'depths')
        os.makedirs(depths_output_dir, exist_ok=True)
        print(f"\nExporting depth maps (format={depth_output_format}, scale={depth_scale})...")

        for room_name, idx, _ in all_selected:
            depth_dir = f"{pipeline_output}/renders_final/depth/{room_name}"
            src_path = os.path.join(depth_dir, f"{room_name}_{idx:04d}_depth.npy")

            if not os.path.exists(src_path):
                continue

            depth = np.load(src_path).astype(np.float32)
            dst_name = f"{room_name}_{idx:04d}"

            if depth_output_format == "png":
                depth_scaled = (depth * depth_scale).astype(np.uint16)
                dst_path = os.path.join(depths_output_dir, f"{dst_name}.png")
                cv2.imwrite(dst_path, depth_scaled)
            else:
                dst_path = os.path.join(depths_output_dir, f"{dst_name}.npy")
                np.save(dst_path, depth)

        print(f"Created depths/ folder")

    return output_dir


def parse_cameras_txt(cameras_txt_path):
    """
    Parse camera intrinsics from a COLMAP cameras.txt file.

    Supports PINHOLE model: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy

    Returns:
        Tuple of (fx, fy, cx, cy)
    """
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 8 and parts[1] == 'PINHOLE':
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])
                return (fx, fy, cx, cy)

    raise ValueError(f"Could not parse PINHOLE intrinsics from {cameras_txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create nerfstudio splatfacto training data in COLMAP format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From pipeline output (auto-detects intrinsics from cameras.txt)
  python create_splatfacto_colmap.py \\
      --pipeline-output output/my_scene \\
      --rooms master_bedroom living_room dining_room

  # Custom output directory and subsample factor
  python create_splatfacto_colmap.py \\
      --pipeline-output output/my_scene \\
      --rooms master_bedroom \\
      --output-dir output/my_scene/custom_colmap \\
      --subsample 4

  # Disable depth export
  python create_splatfacto_colmap.py \\
      --pipeline-output output/my_scene \\
      --rooms master_bedroom living_room \\
      --no-export-depths
"""
    )

    parser.add_argument(
        "--pipeline-output",
        type=str,
        required=True,
        help="Base pipeline output directory (contains renders_final/, flux_final/)"
    )
    parser.add_argument(
        "--rooms",
        nargs="+",
        required=True,
        help="List of room names to include"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: {pipeline-output}/splatfacto_colmap_all)"
    )
    parser.add_argument(
        "--subsample",
        type=str,
        default="auto",
        help="Point cloud subsampling factor. Use 'auto' (default) to compute dynamically based on --max-points, or an integer for a fixed factor."
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5_000_000,
        help="Target maximum number of points when --subsample auto (default: 5000000)"
    )
    parser.add_argument(
        "--no-export-depths",
        action="store_true",
        help="Disable depth map export for depth-supervised training"
    )
    parser.add_argument(
        "--depth-format",
        type=str,
        default="png",
        choices=["png", "npy"],
        help="Depth output format (default: png)"
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Depth scale factor for PNG export (default: 1000.0, meters -> millimeters)"
    )
    parser.add_argument(
        "--exclude-indices",
        nargs="+",
        type=int,
        default=None,
        help="Camera indices to exclude per room (e.g., --exclude-indices 18 19 20 21)"
    )

    args = parser.parse_args()

    pipeline_output = args.pipeline_output
    rooms = args.rooms
    output_dir = args.output_dir or f"{pipeline_output}/splatfacto_colmap_all"
    subsample = "auto" if args.subsample == "auto" else int(args.subsample)
    max_points = args.max_points
    export_depths = not args.no_export_depths
    depth_output_format = args.depth_format
    depth_scale = args.depth_scale
    exclude_indices = set(args.exclude_indices) if args.exclude_indices else None

    # Auto-parse camera intrinsics from cameras.txt
    cameras_txt_path = f"{pipeline_output}/renders_final/cameras.txt"
    if not os.path.exists(cameras_txt_path):
        print(f"Error: cameras.txt not found at {cameras_txt_path}", file=sys.stderr)
        return 1

    intrinsics = parse_cameras_txt(cameras_txt_path)
    print(f"Parsed intrinsics from cameras.txt: fx={intrinsics[0]:.4f}, fy={intrinsics[1]:.4f}, cx={intrinsics[2]:.1f}, cy={intrinsics[3]:.1f}")

    # Create COLMAP structure with all rooms
    create_colmap_structure_all_rooms(
        pipeline_output=pipeline_output,
        rooms=rooms,
        output_dir=output_dir,
        intrinsics=intrinsics,
        subsample=subsample,
        max_points=max_points,
        export_depths=export_depths,
        depth_output_format=depth_output_format,
        depth_scale=depth_scale,
        exclude_indices=exclude_indices
    )

    print(f"\nData ready at: {output_dir}")
    print("\nFolder structure:")
    print(f"  {output_dir}/")
    print(f"  ├── colmap/sparse/0/")
    print(f"  │   ├── cameras.txt")
    print(f"  │   ├── images.txt")
    print(f"  │   └── points3D.txt (with 3D points from depth maps)")
    print(f"  ├── images/")
    print(f"  │   └── *.png")
    if export_depths:
        print(f"  └── depths/")
        print(f"      └── *.{depth_output_format} (depth maps for supervision)")

    print("\n" + "="*60)
    print("TRAINING COMMANDS")
    print("="*60)

    print("\n1. Standard splatfacto (RGB only):")
    print(f"   ns-train splatfacto --pipeline.model.camera-optimizer.mode off colmap --data {output_dir} --eval-mode all")

    if export_depths:
        print("\n2. Depth-supervised splatfacto (recommended for better geometry):")
        if depth_output_format == "png":
            print(f"   ns-train depth-splatfacto --pipeline.model.depth-loss-mult 0.7 --pipeline.model.camera-optimizer.mode off colmap --data {output_dir} --depth-unit-scale-factor 1e-3 --eval-mode all")
        else:
            print(f"   ns-train depth-splatfacto --pipeline.model.depth-loss-mult 0.7 --pipeline.model.camera-optimizer.mode off colmap --data {output_dir} --depth-unit-scale-factor 1.0 --eval-mode all")

        print("\n   Depth loss tuning tips:")
        print("   - Increase depth-loss-mult (0.2-0.5) if floaters persist")
        print("   - Decrease depth-loss-mult (0.05) if colors look washed out")
        print("   - Try depth-loss-type log_l1 for scale-invariant loss")

    return 0


if __name__ == "__main__":
    sys.exit(main())
