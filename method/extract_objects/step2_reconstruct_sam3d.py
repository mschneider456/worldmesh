#!/usr/bin/env python3
"""
Step 2: SAM-3D-Objects Mesh Reconstruction

Reconstructs 3D meshes from segmented masks using SAM-3D-Objects.
SAM-3D-Objects automatically estimates object poses (position, scale, rotation).
Outputs are in camera space.

Outputs per object:
- {index}_{label}.glb: Textured mesh
- {index}_{label}.ply: Gaussian splat
- {index}_{label}_pose.json: Pose (rotation, translation, scale)

Combined scene outputs:
- scene_combined.ply: Gaussian splat (normalized for viewing)
- scene_combined_posed.ply: Gaussian splat (camera space coordinates)
- scene_combined.glb: Mesh scene with all objects positioned correctly

Wall-Based Scale Calibration:
MoGe produces depth in arbitrary scale, not metric. To correct this, we can
compare MoGe's wall depth to expected wall depth from scene geometry:
  1. Use segmentation map to find wall pixels
  2. Exclude pixels occluded by furniture (using step1 masks)
  3. Sample MoGe's depth at valid wall pixels
  4. Compute expected depth from camera to wall plane
  5. Scale correction = expected_depth / moge_depth

Runs in the 'worldmesh-sam3d-objects' conda environment.

Usage:
    cd /mnt/hdd/scenes/sam-3d-objects
    conda run -n worldmesh-sam3d-objects python ../extract_objects/step2_reconstruct_sam3d.py \
        --input-image ../generations/Flux2_00013_.png \
        --masks-dir ../output/masks \
        --output-dir ../output/objects

With wall calibration:
    cd /mnt/hdd/scenes/sam-3d-objects && \\
    conda run -n worldmesh-sam3d-objects python ../extract_objects/step2_reconstruct_sam3d.py \\
        --input-image ../generations/Flux2-Klein-9b-base_00003_.png \\
        --masks-dir ../output/one_scene/masks_lower_threshold_descriptive2 \\
        --output-dir ../output/one_scene/objects_wall_calibrated \\
        --segmentation-map ../output/one_scene/renders_large/segmentation/master_bedroom/master_bedroom_0000.png \\
        --segmentation-metadata ../output/one_scene/renders_large/segmentation_metadata.json \\
        --scene-json ../scene_layout_large.json \\
        --room-id master_bedroom \\
        --camera-pose "0.5,-0.5,0.5,0.5,9.0,-1.6,0.05"
"""
import argparse
import gc
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint_requirements import (
    CheckpointRequirement,
    find_missing_requirements,
    format_missing_checkpoints_error,
)


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()


def load_inference_pipeline(checkpoint_path, compile_model=False):
    """
    Load SAM-3D-Objects inference pipeline and scene utilities.

    Returns:
        tuple: (inference, make_scene, ready_gaussian_for_video_rendering)
    """
    # Must set CUDA_HOME before imports
    os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
    os.environ["LIDRA_SKIP_INIT"] = "true"

    sys.path.insert(0, "notebook")
    from inference import Inference, make_scene, ready_gaussian_for_video_rendering

    resolved_checkpoint = Path(checkpoint_path).resolve()
    print(f"Loading SAM-3D-Objects pipeline from: {resolved_checkpoint}")
    inference = Inference(str(resolved_checkpoint), compile=compile_model)
    return inference, make_scene, ready_gaussian_for_video_rendering


def load_masks_from_dir(masks_dir):
    """
    Load masks and metadata from step1 output directory.

    Returns:
        List of dicts with mask, label, index, score, area, is_doorway
        metadata dict
    """
    masks_dir = Path(masks_dir)
    meta_path = masks_dir / "metadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path) as f:
        metadata = json.load(f)

    masks = []
    for mask_info in metadata['masks']:
        mask_path = masks_dir / mask_info['filename']
        mask_img = np.array(Image.open(mask_path))

        # Extract alpha channel as binary mask
        if mask_img.ndim == 3:
            mask = mask_img[..., 3] > 127
        else:
            mask = mask_img > 127

        masks.append({
            'mask': mask,
            'label': mask_info['label'],
            'index': mask_info['index'],
            'score': mask_info['score'],
            'area': mask_info['area'],
            'is_doorway': mask_info.get('is_doorway', False),
        })

    return masks, metadata


def compute_mask_overlap(mask1, mask2):
    """Compute fraction of mask1 that overlaps with mask2."""
    if mask1.sum() == 0:
        return 0.0
    intersection = np.logical_and(mask1, mask2).sum()
    return intersection / mask1.sum()


def should_exclude_object(obj_mask, doorway_masks, overlap_threshold=0.5):
    """
    Check if object should be excluded due to doorway overlap.

    An object is excluded if a significant portion of it overlaps with any
    doorway mask, indicating the object is visible through the doorway
    (i.e., in an adjacent room).

    Args:
        obj_mask: Object mask (boolean array)
        doorway_masks: List of doorway masks (boolean arrays)
        overlap_threshold: Fraction of object within doorway to trigger exclusion

    Returns:
        Tuple of (should_exclude: bool, overlap_fraction: float, doorway_index: int or None)
    """
    max_overlap = 0.0
    max_overlap_idx = None

    for idx, doorway_mask in enumerate(doorway_masks):
        overlap = compute_mask_overlap(obj_mask, doorway_mask)
        if overlap > max_overlap:
            max_overlap = overlap
            max_overlap_idx = idx

    should_exclude = max_overlap > overlap_threshold
    return should_exclude, max_overlap, max_overlap_idx


def parse_camera_pose(pose_string):
    """
    Parse camera pose from string format "qw,qx,qy,qz,tx,ty,tz".

    Args:
        pose_string: Comma-separated string of quaternion and translation

    Returns:
        tuple: (quaternion [w,x,y,z], translation [x,y,z])
    """
    parts = [float(x) for x in pose_string.split(',')]
    if len(parts) != 7:
        raise ValueError(f"Camera pose must have 7 values (qw,qx,qy,qz,tx,ty,tz), got {len(parts)}")
    quaternion = np.array(parts[:4])  # [w, x, y, z]
    translation = np.array(parts[4:])  # [x, y, z]
    return quaternion, translation


def compute_camera_position(quaternion, translation):
    """
    Convert COLMAP camera pose to world position.

    COLMAP stores world-to-camera transform. To get camera position in world:
    - Rotation R rotates world to camera frame
    - Translation t is applied after rotation
    - Camera position = -R^T @ t

    Args:
        quaternion: [w, x, y, z] rotation quaternion
        translation: [x, y, z] translation vector

    Returns:
        numpy array: Camera position in world coordinates [x, y, z]
    """
    # Convert quaternion to rotation matrix
    # scipy uses [x, y, z, w] format, COLMAP uses [w, x, y, z]
    quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    R = Rotation.from_quat(quat_scipy).as_matrix()

    # Camera position in world = -R^T @ t
    camera_pos = -R.T @ translation
    return camera_pos


def get_wall_position(scene_json, room_id, wall_name):
    """
    Get wall center position from scene JSON.

    Wall naming convention: "{room_id}_wall_{index}"
    where index corresponds to polygon edge (0-1, 1-2, 2-3, 3-0)

    Args:
        scene_json: Loaded scene JSON dict
        room_id: Room identifier (e.g., "master_bedroom")
        wall_name: Full wall name (e.g., "master_bedroom_wall_2")

    Returns:
        numpy array: Wall center position [x, y, z]
    """
    # Find the room
    room = None
    for r in scene_json['rooms']:
        if r['id'] == room_id:
            room = r
            break

    if room is None:
        raise ValueError(f"Room '{room_id}' not found in scene JSON")

    # Extract wall index from name (e.g., "master_bedroom_wall_2" -> 2)
    wall_idx = int(wall_name.split('_')[-1])

    polygon = room['floor_polygon']
    num_points = len(polygon)

    # Wall i connects point i to point (i+1) % n
    p1 = np.array(polygon[wall_idx])
    p2 = np.array(polygon[(wall_idx + 1) % num_points])

    # Wall center is midpoint at half ceiling height
    ceiling_height = room.get('ceiling_height', scene_json['metadata'].get('default_ceiling_height', 3.0))
    wall_center = np.array([
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2,
        ceiling_height / 2
    ])

    return wall_center


def get_wall_plane(scene_json, room_id, wall_name):
    """
    Get wall plane parameters (point + normal) from scene JSON.

    Args:
        scene_json: Loaded scene JSON dict
        room_id: Room identifier (e.g., "master_bedroom")
        wall_name: Full wall name (e.g., "master_bedroom_wall_2")

    Returns:
        tuple: (wall_center [x,y,z], wall_normal [x,y,z])
    """
    room = None
    for r in scene_json['rooms']:
        if r['id'] == room_id:
            room = r
            break

    if room is None:
        raise ValueError(f"Room '{room_id}' not found in scene JSON")

    wall_idx = int(wall_name.split('_')[-1])
    polygon = room['floor_polygon']
    num_points = len(polygon)

    p1 = np.array(polygon[wall_idx], dtype=float)
    p2 = np.array(polygon[(wall_idx + 1) % num_points], dtype=float)

    # Wall direction (along the wall)
    wall_dir = p2 - p1
    wall_dir = wall_dir / np.linalg.norm(wall_dir)

    # Normal is perpendicular to wall, pointing inward (counter-clockwise polygon)
    # For a counter-clockwise polygon, inward normal is (-dy, dx)
    wall_normal = np.array([-wall_dir[1], wall_dir[0], 0])

    # Wall center
    ceiling_height = room.get('ceiling_height', scene_json['metadata'].get('default_ceiling_height', 3.0))
    wall_center = np.array([
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2,
        ceiling_height / 2
    ])

    return wall_center, wall_normal


def compute_wall_scale_correction(
    pointmap,
    segmentation_map,
    seg_metadata,
    object_masks,
    room_id,
    scene_json,
    camera_pose,
    input_image=None,
    output_dir=None,
):
    """
    Compute scale correction by comparing MoGe's wall depth to expected depth.

    IMPORTANT: The segmentation map is from the empty room render, but MoGe
    processes the AI-generated image which has furniture. We must exclude
    pixels where furniture occludes the wall.

    Args:
        pointmap: MoGe's pointmap (H, W, 3) in camera space, Z = depth
        segmentation_map: Segmentation image (H, W, 3) from empty room render
        seg_metadata: Segmentation metadata dict with class_to_color
        object_masks: List of object masks from step1 (furniture)
        room_id: e.g., "master_bedroom"
        scene_json: Scene layout dict
        camera_pose: Tuple of (quaternion, translation)

    Returns:
        float: Scale correction factor (multiply all scales/positions by this)
    """
    quaternion, translation = camera_pose

    # Compute camera forward direction in world space (for wall alignment scoring)
    quat_scipy = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    R_w2c = Rotation.from_quat(quat_scipy).as_matrix()
    R_c2w = R_w2c.T
    # COLMAP convention: +Z is forward in camera space
    camera_forward_world = R_c2w @ np.array([0.0, 0.0, 1.0])
    # Project to XY plane (horizontal facing direction)
    camera_forward_xy = camera_forward_world[:2]
    fwd_norm = np.linalg.norm(camera_forward_xy)
    if fwd_norm > 1e-6:
        camera_forward_xy = camera_forward_xy / fwd_norm

    # Create combined furniture mask (all objects from step1)
    furniture_mask = np.zeros(segmentation_map.shape[:2], dtype=bool)
    for obj_mask in object_masks:
        # Resize mask to match segmentation map if needed
        if obj_mask.shape != furniture_mask.shape:
            obj_mask_resized = cv2.resize(
                obj_mask.astype(np.uint8),
                (furniture_mask.shape[1], furniture_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            obj_mask_resized = obj_mask
        furniture_mask |= obj_mask_resized

    # Find wall pixels for this room
    # Build dict of wall_name -> color
    class_to_color = seg_metadata['class_to_color']
    wall_colors = {
        name: np.array(color)
        for name, color in class_to_color.items()
        if room_id in name and 'wall' in name
    }

    if not wall_colors:
        raise ValueError(f"No walls found for room '{room_id}' in segmentation metadata")

    print(f"\n  Wall calibration for room '{room_id}':")
    print(f"    Found {len(wall_colors)} walls: {list(wall_colors.keys())}")
    print(f"    Camera forward (XY): [{camera_forward_xy[0]:.3f}, {camera_forward_xy[1]:.3f}]")
    print(f"    Furniture mask covers {furniture_mask.sum()} pixels ({100*furniture_mask.mean():.1f}%)")

    # Score each wall by camera alignment and pixel count
    best_wall = None
    best_count = 0
    best_mask = None
    best_full_mask = None

    best_aligned_wall = None
    best_alignment = -1.0
    best_aligned_count = 0
    best_aligned_mask = None
    best_aligned_full_mask = None

    for wall_name, color in wall_colors.items():
        wall_mask = np.all(segmentation_map == color, axis=-1)
        # Exclude pixels occluded by furniture
        valid_wall_mask = wall_mask & ~furniture_mask
        count = valid_wall_mask.sum()

        # Compute alignment with camera forward direction
        _, wall_normal = get_wall_plane(scene_json, room_id, wall_name)
        alignment = np.dot(camera_forward_xy, wall_normal[:2])

        print(f"    {wall_name}: {wall_mask.sum()} total, {count} unoccluded, alignment={alignment:.3f}")

        # Track wall with most unoccluded pixels (fallback)
        if count > best_count:
            best_count = count
            best_wall = wall_name
            best_mask = valid_wall_mask
            best_full_mask = wall_mask

        # Track best-aligned wall (must be visible and have enough pixels)
        if alignment > 0 and count >= 100 and alignment > best_alignment:
            best_alignment = alignment
            best_aligned_wall = wall_name
            best_aligned_count = count
            best_aligned_mask = valid_wall_mask
            best_aligned_full_mask = wall_mask

    # Prefer the best-aligned wall; fall back to max-unoccluded if alignment-based has too few pixels
    if best_aligned_wall is not None and best_aligned_count >= 100:
        print(f"    Selected: {best_aligned_wall} ({best_aligned_count} pixels, alignment={best_alignment:.3f}) [camera-facing]")
        best_wall = best_aligned_wall
        best_count = best_aligned_count
        best_mask = best_aligned_mask
        best_full_mask = best_aligned_full_mask
    else:
        if best_count < 100:
            raise ValueError(f"Not enough unoccluded wall pixels ({best_count}) for calibration")
        print(f"    Selected: {best_wall} ({best_count} pixels) [fallback: max unoccluded]")

    # Erode wall mask by 25% border to avoid floor/ceiling/adjacent wall edge pixels
    mask_ys, mask_xs = np.where(best_mask)
    if len(mask_ys) > 0:
        bbox_h = mask_ys.max() - mask_ys.min() + 1
        bbox_w = mask_xs.max() - mask_xs.min() + 1
        erode_ky = max(1, int(bbox_h * 0.10))
        erode_kx = max(1, int(bbox_w * 0.10))
        kernel = np.ones((erode_ky, erode_kx), np.uint8)
        best_mask = cv2.erode(best_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        best_full_mask = cv2.erode(best_full_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        eroded_count = best_mask.sum()
        print(f"    After 10% border erosion: {eroded_count} pixels (kernel {erode_kx}x{erode_ky})")
        if eroded_count < 50:
            raise ValueError(f"Not enough pixels after erosion ({eroded_count}) for calibration")

    # Save debug visualization of wall pixels used for calibration
    if input_image is not None and output_dir is not None:
        debug_img = input_image.copy()
        h, w = debug_img.shape[:2]
        seg_h, seg_w = segmentation_map.shape[:2]

        if (seg_h, seg_w) != (h, w):
            def _resize_mask(m):
                return cv2.resize(
                    m.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            vis_used = _resize_mask(best_mask)
            vis_excluded = _resize_mask(best_full_mask & furniture_mask)
        else:
            vis_used = best_mask
            vis_excluded = best_full_mask & furniture_mask

        # Green overlay for pixels used in calibration
        debug_img[vis_used] = (
            debug_img[vis_used].astype(np.float32) * 0.4
            + np.array([0, 255, 0], dtype=np.float32) * 0.6
        ).astype(np.uint8)
        # Red overlay for wall pixels excluded by furniture occlusion
        debug_img[vis_excluded] = (
            debug_img[vis_excluded].astype(np.float32) * 0.4
            + np.array([255, 0, 0], dtype=np.float32) * 0.6
        ).astype(np.uint8)

        debug_path = Path(output_dir) / "wall_calibration_debug.png"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(debug_img).save(str(debug_path))
        print(f"    Saved debug image: {debug_path}")

    # Sample MoGe's depth at VALID wall pixels only
    # Pointmap is (H, W, 3), Z coordinate is depth
    # Resize pointmap to match segmentation map if needed
    if pointmap.shape[:2] != segmentation_map.shape[:2]:
        pointmap_resized = cv2.resize(
            pointmap,
            (segmentation_map.shape[1], segmentation_map.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        pointmap_resized = pointmap

    wall_depths = pointmap_resized[best_mask, 2]  # Z coordinate = depth

    # Filter out invalid depths (NaN, negative, zero)
    valid_depths = wall_depths[np.isfinite(wall_depths) & (wall_depths > 0)]
    if len(valid_depths) < 50:
        raise ValueError(f"Not enough valid depth values ({len(valid_depths)}) for calibration")

    moge_wall_depth = np.median(valid_depths)

    # Compute expected depth from camera to wall plane
    camera_pos = compute_camera_position(quaternion, translation)
    wall_center, wall_normal = get_wall_plane(scene_json, room_id, best_wall)

    # Distance from point to plane: |n · (p - p0)| / |n|
    # Since we want depth along camera view direction, use direct distance to wall plane
    # The camera looks along its -Z axis in camera space (after COLMAP convention)
    # For simplicity, use perpendicular distance to wall plane

    # Vector from wall to camera
    cam_to_wall = wall_center - camera_pos

    # Project onto wall normal to get perpendicular distance
    # (positive if camera is on the normal side of wall, which it should be inside room)
    expected_wall_depth = np.abs(np.dot(cam_to_wall, wall_normal[:2].tolist() + [0]) if wall_normal[2] == 0
                                  else np.dot(cam_to_wall, wall_normal))

    # If perpendicular distance is tiny, use euclidean distance to wall center
    if expected_wall_depth < 0.5:
        expected_wall_depth = np.linalg.norm(cam_to_wall)

    scale_correction = expected_wall_depth / moge_wall_depth

    print(f"\n  Scale calibration results:")
    print(f"    Camera position: [{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}]")
    print(f"    Wall center: [{wall_center[0]:.2f}, {wall_center[1]:.2f}, {wall_center[2]:.2f}]")
    print(f"    MoGe wall depth (median): {moge_wall_depth:.2f}")
    print(f"    Expected wall depth: {expected_wall_depth:.2f}m")
    print(f"    Scale correction: {scale_correction:.2f}x")

    return scale_correction


def reconstruct_object(inference, image, mask, with_texture=True, fov_x=None):
    """
    Reconstruct a single object using SAM-3D-Objects.

    Args:
        inference: SAM-3D-Objects Inference instance
        image: RGB image as numpy array
        mask: Binary mask as numpy array
        with_texture: Whether to bake textures onto mesh
        fov_x: Optional horizontal FOV in degrees (improves scale accuracy)

    Returns:
        dict with reconstruction outputs including:
        - 'glb': Trimesh object (textured mesh)
        - 'gs': Gaussian splat object
        - 'rotation': (1, 4) quaternion [w, x, y, z]
        - 'translation': (1, 3) position in camera space
        - 'scale': (1, 3) scale factors
    """
    # Create RGBA image with mask as alpha
    rgba = inference.merge_mask_to_rgba(image, mask)

    # Run reconstruction
    output = inference._pipeline.run(
        rgba,
        None,  # mask
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=with_texture,
        with_layout_postprocess=False,
        use_vertex_color=not with_texture,
        pointmap=None,
        fov_x=fov_x,
    )

    return output


def save_object_outputs(output, output_dir, index, label):
    """
    Save reconstruction outputs for a single object.

    Saves:
    - GLB mesh file
    - PLY Gaussian splat file
    - JSON pose file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{index:02d}_{label.replace(' ', '_')}"

    # Save GLB mesh
    if output.get('glb') is not None:
        glb_path = output_dir / f"{prefix}.glb"
        output['glb'].export(str(glb_path))
        print(f"    Saved mesh: {glb_path.name}")

    # Save Gaussian splat
    if output.get('gs') is not None:
        ply_path = output_dir / f"{prefix}.ply"
        output['gs'].save_ply(str(ply_path))
        print(f"    Saved splat: {ply_path.name}")

    # Save pose information
    pose_data = {}

    if output.get('rotation') is not None:
        rot = output['rotation'].cpu().numpy().tolist()
        pose_data['rotation'] = rot  # [w, x, y, z] quaternion

    if output.get('translation') is not None:
        trans = output['translation'].cpu().numpy().tolist()
        pose_data['translation'] = trans  # [x, y, z] position

    if output.get('scale') is not None:
        scale = output['scale'].cpu().numpy().tolist()
        pose_data['scale'] = scale  # [sx, sy, sz] scale factors

    pose_path = output_dir / f"{prefix}_pose.json"
    with open(pose_path, 'w') as f:
        json.dump(pose_data, f, indent=2)
    print(f"    Saved pose: {pose_path.name}")

    return {
        'glb_path': str(output_dir / f"{prefix}.glb"),
        'ply_path': str(output_dir / f"{prefix}.ply"),
        'pose_path': str(pose_path),
        'pose': pose_data
    }


def transform_mesh_preserve_visual(mesh, rotation_quat, translation, scale, yup_to_zup):
    """
    Transform a mesh while preserving its visual (texture or vertex colors).

    Args:
        mesh: trimesh.Trimesh object
        rotation_quat: quaternion for rotation
        translation: translation vector
        scale: scale factors
        yup_to_zup: matrix to convert from y-up to z-up

    Returns:
        Transformed trimesh.Trimesh with visual preserved
    """
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    # Convert vertices from y-up (GLB) to z-up for pose transformation
    vertices_yup = np.array(mesh.vertices, dtype=np.float32)
    vertices_zup = vertices_yup @ yup_to_zup

    # Apply pose transformation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vertices_tensor = torch.tensor(vertices_zup, dtype=torch.float32).to(device)
    rotation_quat = rotation_quat.to(device)
    translation = translation.to(device)
    scale = scale.to(device)

    R_l2c = quaternion_to_matrix(rotation_quat)
    l2c_transform = compose_transform(scale=scale, rotation=R_l2c, translation=translation)
    transformed = l2c_transform.transform_points(vertices_tensor.unsqueeze(0)).squeeze(0)
    transformed_np = transformed.cpu().numpy()

    # Create new mesh with transformed vertices but same faces and visual
    new_mesh = trimesh.Trimesh(
        vertices=transformed_np,
        faces=mesh.faces.copy(),
        process=False
    )

    # Copy the visual (texture or vertex colors)
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        # Textured mesh - copy the TextureVisuals
        new_mesh.visual = mesh.visual.copy()
    elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # Vertex colored mesh
        new_mesh.visual.vertex_colors = mesh.visual.vertex_colors.copy()

    return new_mesh


def combine_scene(results_with_outputs, output_dir, make_scene, ready_gaussian_for_video_rendering):
    """
    Combine all reconstructed objects into a single scene.

    Creates:
    - scene_combined.ply: Gaussian splat (normalized for viewing)
    - scene_combined_posed.ply: Gaussian splat (camera space coordinates)
    - scene_combined.glb: Mesh scene with textures

    Args:
        results_with_outputs: List of dicts with 'output', 'label', 'index' keys
        output_dir: Directory to save combined scene
        make_scene: Function from inference.py
        ready_gaussian_for_video_rendering: Function from inference.py
    """
    output_dir = Path(output_dir)

    # Inverse rotation matrix to convert from y-up (GLB) back to z-up (original)
    # The GLB export applies: vertices @ [[1,0,0],[0,0,-1],[0,1,0]] (z-up to y-up)
    # This reverses it: (x,y,z) -> (x,z,-y) back to original z-up coordinates
    yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

    # Combine Gaussian splats
    print("\n  Combining Gaussian splats...")
    outputs_with_gs = [r['output'] for r in results_with_outputs if r['output'].get('gaussian') is not None]

    if outputs_with_gs:
        try:
            scene_gs = make_scene(*deepcopy(outputs_with_gs))

            # Save posed version (in scene/camera coordinates)
            posed_path = output_dir / "scene_combined_posed.ply"
            scene_gs.save_ply(str(posed_path))
            print(f"    Saved: {posed_path.name} (camera space)")

            # Save normalized version (for viewing)
            scene_gs_normalized = ready_gaussian_for_video_rendering(
                deepcopy(scene_gs),
                fix_alignment=True
            )
            normalized_path = output_dir / "scene_combined.ply"
            scene_gs_normalized.save_ply(str(normalized_path))
            print(f"    Saved: {normalized_path.name} (normalized for viewing)")

            del scene_gs, scene_gs_normalized
            clear_gpu_memory()
        except torch.cuda.OutOfMemoryError:
            print("    ERROR: Out of GPU memory during Gaussian combination - skipping")
            clear_gpu_memory()
    else:
        print("    No Gaussian outputs available")

    # Combine meshes
    print("\n  Combining meshes...")
    scene = trimesh.Scene()
    mesh_count = 0

    for r in results_with_outputs:
        output = r['output']
        if output.get('glb') is None:
            continue

        label = r['label']
        index = r['index']
        mesh = output['glb']

        try:
            transformed_mesh = transform_mesh_preserve_visual(
                mesh,
                output['rotation'],
                output['translation'],
                output['scale'],
                yup_to_zup
            )
            node_name = f"{index:02d}_{label.replace(' ', '_')}"
            scene.add_geometry(transformed_mesh, node_name=node_name)
            mesh_count += 1
            print(f"    Added: {node_name} ({len(transformed_mesh.vertices)} verts)")
        except Exception as e:
            print(f"    ERROR transforming mesh for '{label}': {e}")
            continue

    if mesh_count > 0:
        glb_path = output_dir / "scene_combined.glb"
        scene.export(str(glb_path))
        print(f"\n    Saved: {glb_path.name} ({mesh_count} objects)")
    else:
        print("    No meshes available to combine")


def main():
    parser = argparse.ArgumentParser(description="SAM-3D-Objects Reconstruction")
    parser.add_argument("--input-image", required=True, help="Input image path")
    parser.add_argument("--masks-dir", required=True, help="Directory with masks from step1")
    parser.add_argument("--output-dir", required=True, help="Output directory for reconstructed objects")
    parser.add_argument("--checkpoint", default="checkpoints/hf/pipeline.yaml",
                        help="Path to SAM-3D-Objects checkpoint config")
    parser.add_argument("--no-texture", action="store_true",
                        help="Skip texture baking (faster, vertex colors only)")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Maximum number of objects to process (for testing)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Camera horizontal FOV in degrees (improves scale accuracy)")

    # Wall-based scale calibration arguments
    parser.add_argument("--segmentation-map", type=str, default=None,
                        help="Path to segmentation PNG from empty room render")
    parser.add_argument("--segmentation-metadata", type=str, default=None,
                        help="Path to segmentation_metadata.json")
    parser.add_argument("--scene-json", type=str, default=None,
                        help="Path to scene layout JSON")
    parser.add_argument("--room-id", type=str, default=None,
                        help="Room identifier (e.g., 'master_bedroom')")
    parser.add_argument("--camera-pose", type=str, default=None,
                        help="COLMAP pose string 'qw,qx,qy,qz,tx,ty,tz'")
    parser.add_argument("--scale-factor", type=float, default=None,
                        help="Manual scale correction factor (alternative to wall calibration)")
    parser.add_argument("--scale-boost", type=float, default=2.3,
                        help="Scale multiplier applied to all objects (default: 2.3). "
                             "When wall calibration is disabled this is the total multiplier. "
                             "When wall calibration is enabled this is applied on top of the wall correction.")
    parser.add_argument("--wall-calibration", action="store_true",
                        help="Enable wall-based scale calibration (default: off, uses fixed scale-boost instead)")
    parser.add_argument("--doorway-overlap-threshold", type=float, default=0.5,
                        help="Fraction of object overlapping doorway to trigger exclusion (default: 0.5)")
    parser.add_argument("--mask-indices", type=str, default=None,
                        help="Comma-separated mask indices to process (for batching). "
                             "Only masks with these indices will be reconstructed.")

    args = parser.parse_args()

    # Ensure output_dir exists up front so the final summary write never fails
    # when every mask in the batch errored out (no save_object_outputs call).
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()

    missing = find_missing_requirements([
        CheckpointRequirement(
            identifier="sam3d-step2",
            name="SAM-3D-Objects pipeline config",
            stage="Step 2: SAM-3D-Objects reconstruction",
            candidate_paths=(checkpoint_path,),
            install_commands=(
                "conda activate worldmesh-sam3d-objects",
                "hf auth login",
                "hf download facebook/sam-3d-objects --local-dir sam-3d-objects/checkpoints/hf-download",
                "mv sam-3d-objects/checkpoints/hf-download/checkpoints sam-3d-objects/checkpoints/hf",
                "rm -rf sam-3d-objects/checkpoints/hf-download",
            ),
        )
    ])
    if missing:
        print("\n" + format_missing_checkpoints_error(missing), file=sys.stderr)
        return 1
    args.checkpoint = str(checkpoint_path)

    # Validate wall calibration arguments (all or none)
    wall_calib_args = [args.segmentation_map, args.segmentation_metadata,
                       args.scene_json, args.room_id, args.camera_pose]
    if args.wall_calibration:
        if not all(wall_calib_args):
            parser.error("Wall calibration requires all of: --segmentation-map, "
                         "--segmentation-metadata, --scene-json, --room-id, --camera-pose")

    use_wall_calibration = args.wall_calibration and all(wall_calib_args)

    print("=" * 60)
    print("Step 2: SAM-3D-Objects Reconstruction")
    print("=" * 60)

    # Load input image
    print(f"\nLoading image: {args.input_image}")
    image = np.array(Image.open(args.input_image))
    if image.shape[2] == 4:
        image = image[..., :3]  # Remove alpha if present
    print(f"  Image shape: {image.shape}")

    # Load masks from step1
    print(f"\nLoading masks from: {args.masks_dir}")
    masks, metadata = load_masks_from_dir(args.masks_dir)
    print(f"  Found {len(masks)} masks")

    # Separate doorway masks from object masks
    doorway_masks = [m for m in masks if m.get('is_doorway', False)]
    object_masks = [m for m in masks if not m.get('is_doorway', False)]

    print(f"  Doorway masks: {len(doorway_masks)}")
    print(f"  Object masks: {len(object_masks)}")

    if doorway_masks:
        print(f"  Doorway labels: {[m['label'] for m in doorway_masks]}")

    # Use only object masks for reconstruction
    masks = object_masks

    if args.mask_indices is not None:
        allowed_indices = set(int(x) for x in args.mask_indices.split(','))
        masks = [m for m in masks if m['index'] in allowed_indices]
        print(f"  Filtered to mask indices {sorted(allowed_indices)}: {len(masks)} masks")

    if args.max_objects is not None:
        masks = masks[:args.max_objects]
        print(f"  Limited to first {args.max_objects} objects")

    # Load wall calibration data if provided
    wall_calib_data = None
    if use_wall_calibration:
        print(f"\nLoading wall calibration data...")
        print(f"  Segmentation map: {args.segmentation_map}")
        print(f"  Scene JSON: {args.scene_json}")
        print(f"  Room ID: {args.room_id}")

        seg_map = np.array(Image.open(args.segmentation_map))
        if seg_map.ndim == 2:
            # Grayscale - convert to RGB
            seg_map = np.stack([seg_map, seg_map, seg_map], axis=-1)
        elif seg_map.shape[2] == 4:
            # RGBA - drop alpha
            seg_map = seg_map[:, :, :3]

        with open(args.segmentation_metadata) as f:
            seg_metadata = json.load(f)

        with open(args.scene_json) as f:
            scene_json = json.load(f)

        camera_pose = parse_camera_pose(args.camera_pose)
        print(f"  Camera pose: q=[{camera_pose[0][0]:.3f}, {camera_pose[0][1]:.3f}, {camera_pose[0][2]:.3f}, {camera_pose[0][3]:.3f}], "
              f"t=[{camera_pose[1][0]:.3f}, {camera_pose[1][1]:.3f}, {camera_pose[1][2]:.3f}]")

        # Extract object masks for furniture occlusion detection
        object_masks = [m['mask'] for m in masks]

        wall_calib_data = {
            'segmentation_map': seg_map,
            'seg_metadata': seg_metadata,
            'scene_json': scene_json,
            'room_id': args.room_id,
            'camera_pose': camera_pose,
            'object_masks': object_masks,
        }

    # Load inference pipeline
    print(f"\nLoading SAM-3D-Objects pipeline...")
    inference, make_scene, ready_gaussian_for_video_rendering = load_inference_pipeline(args.checkpoint)
    if args.fov is not None:
        print(f"  Using specified FOV: {args.fov}° (improves scale accuracy)")

    # Initialize scale correction (will be computed from first object's pointmap)
    scale_correction = args.scale_factor  # Use manual factor if provided
    pointmap_for_calibration = None

    # Prepare doorway masks for exclusion checking
    doorway_mask_arrays = [m['mask'] for m in doorway_masks]

    # Reconstruct each object
    print(f"\nReconstructing {len(masks)} objects...")
    results = []
    results_with_outputs = []  # For scene combination
    failed = []
    skipped_doorway = []

    for i, mask_data in enumerate(masks):
        label = mask_data['label']
        mask = mask_data['mask']
        idx = mask_data['index']

        print(f"\n  [{i+1}/{len(masks)}] Processing '{label}'...", flush=True)

        # Check for doorway overlap (object visible through doorway = in adjacent room)
        if doorway_mask_arrays:
            should_exclude, overlap_frac, doorway_idx = should_exclude_object(
                mask, doorway_mask_arrays, args.doorway_overlap_threshold
            )
            if should_exclude:
                doorway_label = doorway_masks[doorway_idx]['label'] if doorway_idx is not None else "doorway"
                print(f"    SKIPPED: {overlap_frac*100:.1f}% overlaps with '{doorway_label}' (threshold: {args.doorway_overlap_threshold*100:.0f}%)")
                skipped_doorway.append({
                    'index': idx,
                    'label': label,
                    'overlap_fraction': overlap_frac,
                    'doorway_label': doorway_label
                })
                continue

        # Clear GPU memory before each reconstruction to prevent accumulation
        clear_gpu_memory()

        try:
            # Run reconstruction
            output = reconstruct_object(
                inference, image, mask,
                with_texture=not args.no_texture,
                fov_x=args.fov
            )

            # Capture pointmap from first successful object for wall calibration
            if wall_calib_data is not None and scale_correction is None:
                if output.get('pointmap') is not None:
                    pointmap_for_calibration = output['pointmap'].numpy()
                    print(f"\n    Captured pointmap for wall calibration: {pointmap_for_calibration.shape}")

                    try:
                        scale_correction = compute_wall_scale_correction(
                            pointmap_for_calibration,
                            wall_calib_data['segmentation_map'],
                            wall_calib_data['seg_metadata'],
                            wall_calib_data['object_masks'],
                            wall_calib_data['room_id'],
                            wall_calib_data['scene_json'],
                            wall_calib_data['camera_pose'],
                            input_image=image,
                            output_dir=args.output_dir,
                        )
                    except Exception as e:
                        print(f"\n    WARNING: Wall calibration failed: {e}")
                        print("    Proceeding without scale correction")
                        scale_correction = None

            # Apply scale correction if computed
            if scale_correction is not None and scale_correction != 1.0:
                if output.get('translation') is not None:
                    output['translation'] = output['translation'] * scale_correction
                if output.get('scale') is not None:
                    output['scale'] = output['scale'] * scale_correction

            # Apply additional scale boost (to both size and position)
            if args.scale_boost != 1.0:
                if output.get('translation') is not None:
                    output['translation'] = output['translation'] * args.scale_boost
                if output.get('scale') is not None:
                    output['scale'] = output['scale'] * args.scale_boost

            # Compute effective total scale multiplier
            total_scale_mult = 1.0
            if scale_correction is not None and scale_correction != 1.0:
                total_scale_mult *= scale_correction
            if args.scale_boost != 1.0:
                total_scale_mult *= args.scale_boost

            # Print pose info with scaling breakdown
            if output.get('translation') is not None:
                t = output['translation'].cpu().numpy().flatten()
                print(f"    Position: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            if output.get('scale') is not None:
                s = output['scale'].cpu().numpy().flatten()
                print(f"    Scale: [{s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}]")
            scale_parts = []
            if scale_correction is not None and scale_correction != 1.0:
                scale_parts.append(f"wall calibration {scale_correction:.2f}x")
            if args.scale_boost != 1.0:
                scale_parts.append(f"boost {args.scale_boost:.2f}x")
            if scale_parts:
                print(f"    Scaling: {' * '.join(scale_parts)} = {total_scale_mult:.2f}x total")
            else:
                print(f"    Scaling: none (1.00x)")

            # Save outputs
            result = save_object_outputs(output, args.output_dir, idx, label)
            result['label'] = label
            result['index'] = idx
            result['scale_correction'] = scale_correction
            result['scale_boost'] = args.scale_boost
            result['total_scale_multiplier'] = total_scale_mult
            results.append(result)

            # Store output for scene combination
            results_with_outputs.append({
                'output': output,
                'label': label,
                'index': idx
            })

        except torch.cuda.OutOfMemoryError:
            print(f"    ERROR: Out of GPU memory - skipping")
            failed.append((idx, label, "OOM"))
            clear_gpu_memory()
            continue
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed.append((idx, label, str(e)))
            continue

    # Combine into scene
    if len(results_with_outputs) > 0:
        print("\n" + "-" * 60)
        print("Combining objects into scene...")
        combine_scene(
            results_with_outputs,
            args.output_dir,
            make_scene,
            ready_gaussian_for_video_rendering
        )

        # Clean up GPU memory after scene combination
        for r in results_with_outputs:
            del r['output']
        results_with_outputs.clear()
        clear_gpu_memory()

    # Save summary metadata
    output_dir = Path(args.output_dir)

    # Check which combined files were created
    combined_files = {}
    for fname, desc in [
        ("scene_combined.ply", "Gaussian splat (normalized)"),
        ("scene_combined_posed.ply", "Gaussian splat (camera space)"),
        ("scene_combined.glb", "Mesh scene"),
    ]:
        fpath = output_dir / fname
        if fpath.exists():
            combined_files[fname] = desc

    # Build calibration info for summary
    calibration_info = {}
    if scale_correction is not None:
        calibration_info['scale_correction'] = scale_correction
        calibration_info['method'] = 'wall_calibration' if use_wall_calibration else 'manual'
    if use_wall_calibration:
        calibration_info['room_id'] = args.room_id
        calibration_info['segmentation_map'] = args.segmentation_map
    calibration_info['scale_boost'] = args.scale_boost
    total_mult = calibration_info.get('scale_correction', 1.0) * args.scale_boost
    calibration_info['total_multiplier'] = total_mult

    summary = {
        'input_image': args.input_image,
        'masks_dir': args.masks_dir,
        'num_objects': len(results),
        'calibration': calibration_info if calibration_info else None,
        'objects': results,
        'failed': [{'index': idx, 'label': label, 'error': err} for idx, label, err in failed],
        'skipped_doorway': skipped_doorway,
        'doorway_overlap_threshold': args.doorway_overlap_threshold,
        'combined_scene': combined_files
    }

    summary_path = output_dir / "reconstruction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Reconstruction complete!")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(failed)}")
    if skipped_doorway:
        print(f"  Skipped (doorway overlap): {len(skipped_doorway)}")
        for skip_info in skipped_doorway:
            print(f"    - '{skip_info['label']}' ({skip_info['overlap_fraction']*100:.1f}% overlap with '{skip_info['doorway_label']}')")
    scale_parts = []
    if scale_correction is not None:
        scale_parts.append(f"wall calibration {scale_correction:.2f}x")
    else:
        scale_parts.append("no wall calibration")
    scale_parts.append(f"boost {args.scale_boost:.2f}x")
    total_mult = (scale_correction if scale_correction else 1.0) * args.scale_boost
    print(f"  Scaling: {' * '.join(scale_parts)} = {total_mult:.2f}x total")
    print(f"  Output directory: {args.output_dir}")
    if combined_files:
        print(f"\nCombined scene files:")
        for fname, desc in combined_files.items():
            print(f"  - {fname}: {desc}")
        print(f"\nView Gaussian splat at: https://playcanvas.com/supersplat")
    print(f"\nSummary: {summary_path}")
    print("=" * 60)

    # Return success if at least one object was reconstructed.
    # Partial failures (e.g. OOM on one mask) should not abort the pipeline.
    return 0 if len(results) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
