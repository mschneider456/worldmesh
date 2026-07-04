#!/usr/bin/env python3
"""
Step 3: Camera-to-World Transform and Scene Merging

Transforms reconstructed objects from camera space to world space using
the COLMAP camera pose, then merges them into the room mesh.

Object placement uses OBB (Oriented Bounding Box) for all operations:
1. Level each object so its most floor-aligned OBB face is horizontal
2. Sort objects by current OBB min Z (lowest first)
3. Drop each object using OBB-based XY overlap detection (SAT algorithm)
4. Stack objects on top of each other using OBB heights

Runs in the 'worldmesh' conda environment.

Usage:
    conda run -n worldmesh python step3_position_merge.py \
        --objects-dir ../output/objects \
        --room-mesh ../scene_output.glb \
        --images-txt ../renders_structure_only/images.txt \
        --camera-name bedroom_0001 \
        --output-dir ../output
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon, Point, box


def parse_colmap_images_txt(images_txt_path):
    """
    Parse COLMAP images.txt file to extract camera poses.

    COLMAP format (world-to-camera):
    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    Returns:
        dict mapping image names to pose dicts
    """
    poses = {}

    with open(images_txt_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
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

                # Extract base name (e.g., "bedroom_0001" from "bedroom/bedroom_0001.png")
                base_name = Path(image_name).stem

                poses[base_name] = {
                    'image_id': image_id,
                    'quaternion': [qw, qx, qy, qz],  # COLMAP uses (w, x, y, z)
                    'translation': [tx, ty, tz],
                    'camera_id': camera_id,
                    'image_name': image_name
                }
            except (ValueError, IndexError):
                continue

    return poses


def colmap_to_camera_to_world(quat_wxyz, translation):
    """
    Convert COLMAP pose (world-to-camera) to camera-to-world transform.

    COLMAP stores world-to-camera: T_w2c = [R | t]
    We want camera-to-world: T_c2w = [R.T | -R.T @ t]

    Args:
        quat_wxyz: Quaternion [w, x, y, z] (COLMAP convention)
        translation: Translation [tx, ty, tz]

    Returns:
        R_c2w: (3, 3) rotation matrix camera-to-world
        camera_position: (3,) camera position in world coordinates
    """
    # scipy uses (x, y, z, w) convention
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    R_w2c = Rotation.from_quat(quat_xyzw).as_matrix()
    t_w2c = np.array(translation)

    # Convert to camera-to-world
    R_c2w = R_w2c.T
    camera_position = -R_c2w @ t_w2c

    return R_c2w, camera_position


def apply_sam3d_pose(vertices, pose_data, yup_to_zup):
    """
    Apply SAM-3D-Objects pose to transform from local object space to camera space.

    The GLB mesh from SAM-3D-Objects is in local coordinates (centered at origin,
    y-up). The pose contains scale, rotation, and translation to camera space (z-up).

    Transform order (matching SAM-3D-Objects apply_transform):
    1. Convert y-up to z-up
    2. Apply scale
    3. Apply rotation (quaternion)
    4. Apply translation

    Args:
        vertices: (N, 3) numpy array in local object coords (y-up from GLB)
        pose_data: dict with 'rotation', 'translation', 'scale' from pose.json
        yup_to_zup: (3, 3) conversion matrix

    Returns:
        (N, 3) numpy array in camera space (z-up)
    """
    # Step 1: Convert from y-up (GLB) to z-up (SAM-3D-Objects internal)
    vertices_zup = vertices @ yup_to_zup

    # Extract pose parameters
    rotation = np.array(pose_data['rotation']).flatten()  # [w, x, y, z]
    translation = np.array(pose_data['translation']).flatten()  # [x, y, z]
    scale = np.array(pose_data['scale']).flatten()  # [sx, sy, sz]

    # Step 2: Apply scale
    vertices_scaled = vertices_zup * scale

    # Step 3: Apply rotation (quaternion [w, x, y, z] -> scipy uses [x, y, z, w])
    # PyTorch3D (used in SAM3D step2) uses row-vector convention: v @ R
    quat_scipy = [rotation[1], rotation[2], rotation[3], rotation[0]]
    R = Rotation.from_quat(quat_scipy).as_matrix()
    vertices_rotated = vertices_scaled @ R

    # Step 4: Apply translation
    vertices_camera = vertices_rotated + translation

    return vertices_camera


def transform_vertices_to_world(vertices_camera, R_c2w, camera_position):
    """
    Transform vertices from camera space to world space.

    Args:
        vertices_camera: (N, 3) array in SAM3D camera space (Z = depth/forward)
        R_c2w: (3, 3) camera-to-world rotation (expects OpenGL camera space: -Z forward)
        camera_position: (3,) camera position in world

    Returns:
        (N, 3) array of vertices in world coordinates
    """
    # SAM3D camera space: X right, Y up, Z forward (depth)
    # OpenGL camera space: X right, Y up, -Z forward
    # Flip both X and Z: this combines coordinate system conversion with 180° fix
    # (flipping X and Z together = 180° rotation around Y axis)
    vertices_opengl = vertices_camera.copy()
    vertices_opengl[:, 0] = -vertices_opengl[:, 0]
    vertices_opengl[:, 2] = -vertices_opengl[:, 2]

    # Apply camera-to-world transform
    # v_world = R_c2w @ v_camera + camera_position
    vertices_world = (R_c2w @ vertices_opengl.T).T + camera_position

    return vertices_world


def load_reconstruction_summary(objects_dir):
    """Load reconstruction summary from step2."""
    summary_path = Path(objects_dir) / "reconstruction_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def transform_mesh_to_world(mesh, pose_data, R_c2w, camera_position, yup_to_zup):
    """
    Transform a mesh from local object space to world space.

    Pipeline:
    1. Apply SAM-3D-Objects pose (local -> camera space)
    2. Apply camera-to-world transform (camera -> world space)

    Args:
        mesh: trimesh.Trimesh object (local coords, y-up)
        pose_data: dict with 'rotation', 'translation', 'scale' from pose.json
        R_c2w: camera-to-world rotation matrix
        camera_position: camera position in world
        yup_to_zup: y-up to z-up conversion matrix

    Returns:
        new_mesh: trimesh.Trimesh in world coordinates
    """
    vertices = np.array(mesh.vertices, dtype=np.float32)

    # Step 1: Apply SAM-3D-Objects pose (local -> camera space)
    vertices_camera = apply_sam3d_pose(vertices, pose_data, yup_to_zup)

    # Step 2: Apply camera-to-world transform
    vertices_world = transform_vertices_to_world(vertices_camera, R_c2w, camera_position)

    # Create new mesh with transformed vertices
    new_mesh = trimesh.Trimesh(
        vertices=vertices_world,
        faces=mesh.faces.copy(),
        process=False
    )

    # Preserve visual (texture or vertex colors)
    if hasattr(mesh, 'visual'):
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            # Textured mesh
            new_mesh.visual = mesh.visual.copy()
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            # Vertex colored mesh
            new_mesh.visual.vertex_colors = mesh.visual.vertex_colors.copy()

    return new_mesh


def _copy_visual(src, dst):
    """Copy visual properties between meshes."""
    if hasattr(src, 'visual'):
        if hasattr(src.visual, 'uv') and src.visual.uv is not None:
            dst.visual = src.visual.copy()
        elif hasattr(src.visual, 'vertex_colors') and src.visual.vertex_colors is not None:
            dst.visual.vertex_colors = src.visual.vertex_colors.copy()


def check_floor_penetration(mesh, floor_z=0.0):
    """Check if mesh goes below floor.

    Args:
        mesh: trimesh.Trimesh to check
        floor_z: Z coordinate of the floor

    Returns:
        float: Amount of penetration (positive if below floor, 0 otherwise)
    """
    min_z = mesh.vertices[:, 2].min()
    return max(0.0, floor_z - min_z)


# =============================================================================
# New OBB-based leveling and gravity drop functions
# =============================================================================


def align_mesh_to_axes(mesh, verbose=False):
    """
    Rotate mesh around Z-axis so OBB edges align with X/Y axes.

    This makes objects parallel to room walls (assuming axis-aligned rooms).
    Should be called BEFORE level_mesh_to_floor().

    Algorithm:
    1. Get OBB and extract its 3 principal axes from the rotation matrix
    2. Find the XY projection of each axis (ignore Z component)
    3. For the axis with the largest XY magnitude, compute the angle to the
       nearest cardinal direction (0°, 90°, 180°, 270°)
    4. Rotate the mesh around Z by that angle (around centroid)

    Args:
        mesh: trimesh.Trimesh in world coordinates
        verbose: If True, print diagnostic information

    Returns:
        New trimesh.Trimesh with axis-aligned orientation
    """
    try:
        obb = mesh.bounding_box_oriented
    except Exception:
        if verbose:
            print("      OBB computation failed, skipping alignment")
        return mesh

    # Get OBB principal axes
    rotation_matrix = obb.primitive.transform[:3, :3]

    # Find the axis with largest XY projection (most horizontal)
    best_axis_xy = None
    best_xy_magnitude = 0

    for i in range(3):
        axis = rotation_matrix[:, i]
        xy_magnitude = np.sqrt(axis[0]**2 + axis[1]**2)
        if xy_magnitude > best_xy_magnitude:
            best_xy_magnitude = xy_magnitude
            best_axis_xy = axis[:2]  # XY components only

    if best_xy_magnitude < 1e-6:
        # All axes are nearly vertical, no alignment needed
        if verbose:
            print("      All OBB axes vertical, skipping alignment")
        return mesh

    # Normalize XY projection
    best_axis_xy = best_axis_xy / np.linalg.norm(best_axis_xy)

    # Current angle in XY plane
    current_angle = np.arctan2(best_axis_xy[1], best_axis_xy[0])

    # Find nearest cardinal direction (0, 90, 180, -90 degrees)
    cardinal_angles = [0, np.pi/2, np.pi, -np.pi/2]
    angle_diffs = []
    for ca in cardinal_angles:
        diff = abs(current_angle - ca)
        # Handle wraparound
        diff = min(diff, 2*np.pi - diff)
        angle_diffs.append(diff)
    nearest_idx = np.argmin(angle_diffs)
    target_angle = cardinal_angles[nearest_idx]

    # Rotation needed
    rotation_angle = target_angle - current_angle

    # Normalize rotation to [-pi, pi]
    while rotation_angle > np.pi:
        rotation_angle -= 2*np.pi
    while rotation_angle < -np.pi:
        rotation_angle += 2*np.pi

    if verbose:
        print(f"      current_angle={np.degrees(current_angle):.1f}°, "
              f"target={np.degrees(target_angle):.1f}°, "
              f"rotation={np.degrees(rotation_angle):.1f}°")

    # Skip if already aligned (within 1 degree)
    if abs(rotation_angle) < np.radians(1):
        if verbose:
            print("      Already axis-aligned, skipping rotation")
        return mesh

    if verbose:
        print("      Applying axis alignment rotation...")

    # Create Z-axis rotation matrix
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
    rot_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    # Apply rotation around centroid
    centroid = mesh.vertices.mean(axis=0)
    new_vertices = mesh.vertices.copy()
    new_vertices = (new_vertices - centroid) @ rot_z.T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def _pca_pre_level(mesh, verbose=False):
    """
    Pre-level a mesh by aligning its principal axis with Z when appropriate.

    OBB-based leveling fails for near-cubical shapes (chandeliers, lamps,
    small tables) because the OBB axes are ambiguous.  PCA on the vertex
    cloud reliably finds the direction of most variance.  When that direction
    is roughly vertical and clearly dominant, we rotate it to align exactly
    with Z before OBB-based fine-tuning.

    Conditions for PCA alignment:
    - PC0 is within 45° of Z (object is roughly upright)
    - Eigenvalue ratio PC0/PC1 > 1.3 (PC0 is clearly dominant)
    - PC0 is more than 2° from Z (not already aligned)

    Args:
        mesh: trimesh.Trimesh in world coordinates
        verbose: If True, print diagnostic information

    Returns:
        New trimesh.Trimesh with PCA-aligned orientation, or original if skipped
    """
    verts = mesh.vertices
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    pc0 = eigenvectors[:, order[0]]
    ev_ratio = eigenvalues[order[0]] / eigenvalues[order[1]] if eigenvalues[order[1]] > 0 else 999

    # Point toward +Z
    if pc0[2] < 0:
        pc0 = -pc0

    z_alignment = abs(pc0[2])
    angle_from_z = np.degrees(np.arccos(np.clip(z_alignment, 0, 1)))

    if verbose:
        print(f"      PCA: PC0 {angle_from_z:.1f}° from Z, eigenvalue ratio {ev_ratio:.2f}")

    if angle_from_z > 45 or ev_ratio < 1.3 or angle_from_z < 2.0:
        if verbose:
            reason = "already aligned" if angle_from_z < 2.0 else (
                "PC0 not vertical" if angle_from_z > 45 else "weak dominance")
            print(f"      PCA: skipped ({reason})")
        return mesh

    if verbose:
        print(f"      PCA: aligning PC0 to Z ({angle_from_z:.1f}°)")

    up = np.array([0.0, 0.0, 1.0])
    rotation, _ = Rotation.align_vectors([up], [pc0])
    rot_matrix = rotation.as_matrix()

    centroid = verts.mean(axis=0)
    new_vertices = (verts - centroid) @ rot_matrix.T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def level_mesh_to_floor(mesh, name=None, verbose=False):
    """
    Rotate mesh so its most floor-aligned OBB face becomes horizontal.

    Algorithm:
    0. PCA pre-leveling: align the principal axis with Z for objects where
       the vertex distribution has a clear vertical dominant direction.
       This fixes near-cubical shapes where OBB axes are ambiguous.
       Skipped for wall-mounted (_w) objects.
    1. Get oriented bounding box (OBB)
    2. Extract the 3 principal axes from OBB transform
    3. Each axis defines 2 opposite faces; find the face normal most aligned with -Z
    4. Compute rotation to make that normal point exactly to -Z
    5. Apply rotation around mesh centroid

    Args:
        mesh: trimesh.Trimesh in world coordinates
        name: Object name; _w suffixed objects skip PCA pre-leveling
        verbose: If True, print diagnostic information

    Returns:
        New trimesh.Trimesh with leveled orientation
    """
    # PCA pre-leveling: correct gross tilt before OBB fine-tuning
    # Skip for wall-mounted (_w) objects — PCA would incorrectly rotate
    # flat objects (paintings, wall lamps) to align with Z.
    skip_pca = False
    if name:
        name_lower = name.lower().rstrip()
        if name_lower.endswith('_w'):
            skip_pca = True
            if verbose:
                print(f"      PCA: skipped (wall object '{name}')")
    if not skip_pca:
        mesh = _pca_pre_level(mesh, verbose=verbose)

    try:
        obb = mesh.bounding_box_oriented
    except Exception:
        if verbose:
            print("      OBB computation failed, skipping leveling")
        return mesh

    # OBB transform contains rotation and translation
    # The primitive's transform rotates from axis-aligned to oriented
    obb_transform = obb.primitive.transform
    rotation_matrix = obb_transform[:3, :3]

    # The 3 columns of rotation_matrix are the OBB's principal axes
    # Each axis is perpendicular to 2 opposite faces
    axes = [rotation_matrix[:, i] for i in range(3)]

    # Find which axis (or its negative) is most aligned with -Z (down)
    down = np.array([0, 0, -1])
    best_alignment = -1
    best_normal = None
    best_axis_idx = -1

    for i, axis in enumerate(axes):
        # Check both +axis and -axis (opposite faces)
        for sign in [1, -1]:
            normal = sign * axis
            alignment = np.dot(normal, down)  # cos(angle), 1.0 = perfect
            if alignment > best_alignment:
                best_alignment = alignment
                best_normal = normal
                best_axis_idx = i

    if verbose:
        angle_deg = np.degrees(np.arccos(np.clip(best_alignment, -1, 1)))
        print(f"      best_alignment={best_alignment:.4f} ({angle_deg:.1f}° from horizontal)")
        print(f"      best_axis={best_axis_idx}, normal={best_normal}")

    # If already well-aligned (within ~1 degree), skip rotation
    # Changed from 0.996 (~5°) to 0.9998 (~1°) to catch objects that are
    # slightly tilted but would have been skipped with the looser threshold
    if best_alignment > 0.9998:  # cos(1°) ≈ 0.9998
        if verbose:
            print("      Already level, skipping rotation")
        return mesh

    if verbose:
        print("      Applying leveling rotation...")

    # Compute rotation to align best_normal with down (-Z)
    # Using scipy's Rotation.align_vectors
    rotation, _ = Rotation.align_vectors([down], [best_normal])
    rot_matrix = rotation.as_matrix()

    # Apply rotation around mesh centroid
    centroid = mesh.vertices.mean(axis=0)
    new_vertices = mesh.vertices.copy()
    new_vertices = (new_vertices - centroid) @ rot_matrix.T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )

    _copy_visual(mesh, new_mesh)
    return new_mesh


def level_ceiling_object(mesh, verbose=False):
    """
    Rotate a ceiling-mounted mesh so it hangs straight down (vertical).

    Uses PCA on the vertex cloud to find the principal axis of variance
    (the direction the mesh extends most) and aligns it with Z.  PCA is
    more robust than OBB for near-cubical shapes like chandeliers where the
    OBB axes are ambiguous.

    Args:
        mesh: trimesh.Trimesh (already in world coords, may have wrong tilt)
        verbose: print diagnostics

    Returns:
        New trimesh.Trimesh with principal axis vertical
    """
    verts = mesh.vertices
    centered = verts - verts.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Largest eigenvalue = axis of most variance (hanging direction)
    order = np.argsort(eigenvalues)[::-1]
    pc0 = eigenvectors[:, order[0]]

    # Point toward +Z (preserve upward orientation)
    if pc0[2] < 0:
        pc0 = -pc0

    z_alignment = abs(pc0[2])
    angle_from_z = np.degrees(np.arccos(np.clip(z_alignment, 0, 1)))

    if verbose:
        print(f"      PC0 axis: [{pc0[0]:.3f}, {pc0[1]:.3f}, {pc0[2]:.3f}], "
              f"{angle_from_z:.1f}° from Z")

    # Already vertical enough (within 2°)
    if angle_from_z < 2.0:
        if verbose:
            print("      Already vertical, skipping")
        return mesh

    if verbose:
        print(f"      Rotating {angle_from_z:.1f}° to make principal axis vertical")

    up = np.array([0.0, 0.0, 1.0])
    rotation, _ = Rotation.align_vectors([up], [pc0])
    rot_matrix = rotation.as_matrix()

    centroid = mesh.vertices.mean(axis=0)
    new_vertices = mesh.vertices.copy()
    new_vertices = (new_vertices - centroid) @ rot_matrix.T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


# =============================================================================
# Shape-aware orientation utilities (from fix_meshes.py)
# =============================================================================


def ensure_pendant_down(mesh, verbose=False):
    """Ensure pendant lamp orientation: narrow at top (mount), wide at bottom (shade).

    Checks XY spread of top vs bottom vertices.
    If wider at top (upside down for a pendant lamp), flips 180 around X axis.
    """
    verts = mesh.vertices
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    z_range = z_max - z_min
    if z_range < 0.01:
        return mesh

    z_mid = (z_max + z_min) / 2

    # Compare XY spread of top 25% vs bottom 25%
    top_mask = verts[:, 2] > z_mid + z_range * 0.25
    bot_mask = verts[:, 2] < z_mid - z_range * 0.25

    if top_mask.sum() < 3 or bot_mask.sum() < 3:
        return mesh

    top_spread = np.std(verts[top_mask][:, :2])
    bot_spread = np.std(verts[bot_mask][:, :2])

    if verbose:
        print(f"      Pendant check: top_spread={top_spread:.3f}, bot_spread={bot_spread:.3f}")

    if top_spread > bot_spread * 1.3:
        # Upside down - flip Z through centroid
        centroid = verts.mean(axis=0)
        v = verts - centroid
        v[:, 2] *= -1
        new_mesh = trimesh.Trimesh(
            vertices=v + centroid, faces=mesh.faces.copy(), process=False
        )
        _copy_visual(mesh, new_mesh)
        if verbose:
            print(f"      Flipped pendant lamp (was upside down)")
        return new_mesh

    return mesh


def align_shortest_obb_to_z(mesh, verbose=False):
    """Align the OBB's shortest extent axis to Z (vertical).

    For disc-shaped objects where the thinnest dimension should be vertical.
    """
    obb = mesh.bounding_box_oriented
    T = obb.primitive.transform
    axes = T[:3, :3]  # columns are OBB axes
    extents = obb.primitive.extents

    shortest_idx = np.argmin(extents)
    shortest_axis = axes[:, shortest_idx].copy()

    # Ensure pointing upward
    if shortest_axis[2] < 0:
        shortest_axis = -shortest_axis

    angle_before = np.degrees(np.arccos(np.clip(abs(shortest_axis[2]), 0, 1)))
    if verbose:
        print(f"      OBB shortest extent: {extents[shortest_idx]:.3f}m, "
              f"axis angle from Z: {angle_before:.1f}")

    if angle_before < 5:
        if verbose:
            print(f"      Already aligned, skipping rotation")
        return mesh

    # Rotate shortest axis to align with +Z
    R, _ = Rotation.align_vectors([[0, 0, 1]], [shortest_axis])
    R_mat = R.as_matrix()

    centroid = mesh.vertices.mean(axis=0)
    v = mesh.vertices - centroid
    new_v = (R_mat @ v.T).T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_v, faces=mesh.faces.copy(), process=False
    )
    _copy_visual(mesh, new_mesh)

    # Verify
    if verbose:
        obb2 = new_mesh.bounding_box_oriented
        T2 = obb2.primitive.transform
        axes2 = T2[:3, :3]
        extents2 = obb2.primitive.extents
        si2 = np.argmin(extents2)
        sa2 = axes2[:, si2]
        angle_after = np.degrees(np.arccos(np.clip(abs(sa2[2]), 0, 1)))
        print(f"      After: shortest axis angle from Z: {angle_after:.1f}")

    return new_mesh


def _tracked_local_up_vector(raw_local_mesh, world_mesh):
    """Track SAM3D local Y-up through a world-space mesh.

    Returns:
        Tuple of (unit_up_vector or None, top_mask, bot_mask)
    """
    raw_verts = raw_local_mesh.vertices
    world_verts = world_mesh.vertices

    y_min, y_max = raw_verts[:, 1].min(), raw_verts[:, 1].max()
    y_range = y_max - y_min
    y_mid = (y_min + y_max) / 2
    top_mask = raw_verts[:, 1] > y_mid + y_range * 0.25
    bot_mask = raw_verts[:, 1] < y_mid - y_range * 0.25

    if y_range < 1e-6 or top_mask.sum() < 3 or bot_mask.sum() < 3:
        return None, top_mask, bot_mask

    top_centroid = world_verts[top_mask].mean(0)
    bot_centroid = world_verts[bot_mask].mean(0)

    local_up = top_centroid - bot_centroid
    norm = np.linalg.norm(local_up)
    if norm < 1e-6:
        return None, top_mask, bot_mask
    local_up /= norm

    if local_up[2] < 0:
        local_up = -local_up

    return local_up, top_mask, bot_mask


def _tracked_local_up_angle(raw_local_mesh, world_mesh):
    """Angle in degrees between tracked local Y-up and world Z."""
    local_up, _top_mask, _bot_mask = _tracked_local_up_vector(raw_local_mesh, world_mesh)
    if local_up is None:
        return None
    return np.degrees(np.arccos(np.clip(abs(local_up[2]), 0, 1)))


def _estimated_floor_lift(mesh, tilt_deg):
    """Approximate visible floor hover caused by residual bottom tilt."""
    extents = mesh.bounds[1] - mesh.bounds[0]
    footprint_span = max(extents[0], extents[1])
    return footprint_span * np.tan(np.radians(max(tilt_deg, 0.0)))


def align_local_up_to_z(raw_local_mesh, world_mesh, verbose=False):
    """Align the local Y-up direction to world Z using vertex tracking.

    More robust than OBB for near-cubic meshes where OBB axes are unstable.
    Tracks how the local Y axis (SAM3D convention: Y-up) maps through the
    full transform chain by comparing top/bottom vertex centroids.
    """
    local_up, top_mask, bot_mask = _tracked_local_up_vector(raw_local_mesh, world_mesh)
    if local_up is None:
        if verbose:
            if top_mask is None or bot_mask is None:
                print(f"      Degenerate local-up tracking, skipping")
            elif top_mask.sum() < 3 or bot_mask.sum() < 3:
                print(f"      Too few top/bot vertices for local-up tracking, skipping")
            else:
                print(f"      Degenerate local-up vector, skipping")
        return world_mesh

    angle = np.degrees(np.arccos(np.clip(abs(local_up[2]), 0, 1)))
    if verbose:
        print(f"      Local Y-up in world: {angle:.1f}° from Z")

    if angle < 2:
        if verbose:
            print(f"      Already aligned, skipping")
        return world_mesh

    R, _ = Rotation.align_vectors([[0, 0, 1]], [local_up])
    R_mat = R.as_matrix()

    world_verts = world_mesh.vertices
    centroid = world_verts.mean(axis=0)
    v = world_verts - centroid
    new_v = (R_mat @ v.T).T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_v, faces=world_mesh.faces.copy(), process=False
    )
    _copy_visual(world_mesh, new_mesh)

    # Verify
    if verbose:
        new_world = new_mesh.vertices
        up2 = new_world[top_mask].mean(0) - new_world[bot_mask].mean(0)
        up2 /= np.linalg.norm(up2)
        angle2 = np.degrees(np.arccos(np.clip(abs(up2[2]), 0, 1)))
        print(f"      After: {angle2:.1f}° from Z")

    return new_mesh


def level_wall_art(mesh, verbose=False):
    """Align a wall art OBB so its vertical edge is exactly world Z.

    A painting/TV is a thin box with two face-large extents (W and H) and
    one small extent (depth into the wall). The height extent should be the
    one most aligned with world Z in the OBB. This function:
      1. Finds the OBB axis most aligned with Z (height) and rotates to +Z
      2. Among the two remaining axes, picks the thinnest (depth/face normal);
         rotates around Z so it lies in the XY plane (i.e. horizontal)

    After this, the painting is hanging straight (no tilt) and its face
    normal is horizontal. The actual rotation around Z to face the wall is
    handled by place_wall_art().

    Thin, noisy wall art often needs more than one OBB refit: the first pass
    removes most of the pitch/roll, but the OBB axes can still drift a few
    degrees until the box is recomputed on the partially corrected mesh.
    Iterate a couple of times so the final vertical edge is actually aligned.
    """
    TARGET_TILT_DEG = 0.5
    MAX_REFINEMENT_ITERS = 3

    current_mesh = mesh

    for iter_idx in range(MAX_REFINEMENT_ITERS):
        obb = current_mesh.bounding_box_oriented
        T = obb.primitive.transform
        axes = T[:3, :3]
        extents = obb.primitive.extents

        # Height axis = OBB axis most aligned with world Z
        vert_idx = int(np.argmax(np.abs(axes[2, :])))
        height_axis = axes[:, vert_idx].copy()
        if height_axis[2] < 0:
            height_axis = -height_axis

        angle = np.degrees(np.arccos(np.clip(abs(height_axis[2]), 0, 1)))
        if verbose:
            pass_label = f" pass {iter_idx + 1}" if MAX_REFINEMENT_ITERS > 1 else ""
            print(f"      Wall-art OBB height axis {angle:.1f}° from Z{pass_label} "
                  f"(extents={extents[0]:.2f},{extents[1]:.2f},{extents[2]:.2f})")

        if angle < TARGET_TILT_DEG:
            return current_mesh

        R, _ = Rotation.align_vectors([[0, 0, 1]], [height_axis])
        R_mat = R.as_matrix()
        centroid = current_mesh.vertices.mean(axis=0)
        new_v = (current_mesh.vertices - centroid) @ R_mat.T + centroid

        new_mesh = trimesh.Trimesh(
            vertices=new_v, faces=current_mesh.faces.copy(), process=False
        )
        _copy_visual(current_mesh, new_mesh)
        current_mesh = new_mesh

    return current_mesh


def align_longest_obb_to_z(mesh, verbose=False):
    """Align the OBB's longest extent axis to Z (vertical).

    For elongated objects (tall pendant lamps, plants) where the longest
    dimension should be vertical.
    """
    obb = mesh.bounding_box_oriented
    T = obb.primitive.transform
    axes = T[:3, :3]
    extents = obb.primitive.extents

    longest_idx = np.argmax(extents)
    longest_axis = axes[:, longest_idx].copy()

    # Ensure pointing upward
    if longest_axis[2] < 0:
        longest_axis = -longest_axis

    angle_before = np.degrees(np.arccos(np.clip(abs(longest_axis[2]), 0, 1)))
    if verbose:
        print(f"      OBB longest extent: {extents[longest_idx]:.3f}m, "
              f"axis angle from Z: {angle_before:.1f}")

    if angle_before < 5:
        if verbose:
            print(f"      Already aligned, skipping rotation")
        return mesh

    R, _ = Rotation.align_vectors([[0, 0, 1]], [longest_axis])
    R_mat = R.as_matrix()

    centroid = mesh.vertices.mean(axis=0)
    v = mesh.vertices - centroid
    new_v = (R_mat @ v.T).T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_v, faces=mesh.faces.copy(), process=False
    )
    _copy_visual(mesh, new_mesh)

    # Verify
    if verbose:
        obb2 = new_mesh.bounding_box_oriented
        T2 = obb2.primitive.transform
        axes2 = T2[:3, :3]
        extents2 = obb2.primitive.extents
        li2 = np.argmax(extents2)
        la2 = axes2[:, li2]
        angle_after = np.degrees(np.arccos(np.clip(abs(la2[2]), 0, 1)))
        print(f"      After: longest axis angle from Z: {angle_after:.1f}")

    return new_mesh


def level_surface(mesh, which='top', pct=0.1, verbose=False):
    """Level the top or bottom surface of a mesh by fitting a plane.

    For ceiling objects: level the top (ceiling-facing) surface.
    For floor objects: level the bottom (floor-facing) surface.
    """
    verts = mesh.vertices
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    z_range = z_max - z_min

    if z_range < 0.01:
        return mesh

    if which == 'top':
        thresh = z_max - z_range * pct
        subset = verts[verts[:, 2] > thresh]
    else:
        thresh = z_min + z_range * pct
        subset = verts[verts[:, 2] < thresh]

    if len(subset) < 10:
        if verbose:
            print(f"      Too few {which} vertices ({len(subset)}), skipping surface leveling")
        return mesh

    centroid_sub = subset.mean(axis=0)
    centered = subset - centroid_sub
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Smallest eigenvalue eigenvector = surface normal
    normal = eigvecs[:, 0].copy()
    if normal[2] < 0:
        normal = -normal

    angle = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
    if verbose:
        print(f"      {which.capitalize()} surface tilt: {angle:.1f}° ({len(subset)} verts)")

    if angle < 2:
        if verbose:
            print(f"      Already level, skipping")
        return mesh

    R, _ = Rotation.align_vectors([[0, 0, 1]], [normal])
    R_mat = R.as_matrix()

    centroid = verts.mean(axis=0)
    v = verts - centroid
    new_v = (R_mat @ v.T).T + centroid

    new_mesh = trimesh.Trimesh(
        vertices=new_v, faces=mesh.faces.copy(), process=False
    )
    _copy_visual(mesh, new_mesh)

    # Verify
    if verbose:
        new_verts = new_mesh.vertices
        nz_min, nz_max = new_verts[:, 2].min(), new_verts[:, 2].max()
        nz_range = nz_max - nz_min
        if which == 'top':
            thresh2 = nz_max - nz_range * pct
            subset2 = new_verts[new_verts[:, 2] > thresh2]
        else:
            thresh2 = nz_min + nz_range * pct
            subset2 = new_verts[new_verts[:, 2] < thresh2]
        if len(subset2) > 10:
            c2 = subset2.mean(0)
            cov2 = np.cov((subset2 - c2).T)
            _, evec2 = np.linalg.eigh(cov2)
            n2 = evec2[:, 0]
            angle2 = np.degrees(np.arccos(np.clip(abs(n2[2]), 0, 1)))
            print(f"      After: {which} surface tilt: {angle2:.1f}°")

    return new_mesh


def _check_surface_tilt(mesh, which='bottom', pct=0.1):
    """Check how tilted the top or bottom surface is (degrees from horizontal).

    Returns angle in degrees, or 0.0 if not enough vertices.
    """
    verts = mesh.vertices
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    z_range = z_max - z_min
    if z_range < 0.01:
        return 0.0

    if which == 'top':
        thresh = z_max - z_range * pct
        subset = verts[verts[:, 2] > thresh]
    else:
        thresh = z_min + z_range * pct
        subset = verts[verts[:, 2] < thresh]

    if len(subset) < 10:
        return 0.0

    centered = subset - subset.mean(axis=0)
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    return np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))


def level_object_robust(mesh, raw_local_mesh=None, target='floor', name=None,
                        verbose=False):
    """Refine object orientation from SAM3D camera-to-world transform.

    The SAM3D + camera-to-world transform already gives us the roughly correct
    orientation (correct face pointing down/up). This function REFINES that
    alignment rather than replacing it:

    1. Apply existing leveling (PCA + OBB for floor, PCA for ceiling) which
       finds the most floor/ceiling-aligned face and makes it perfectly
       horizontal. This works because SAM3D already has the right face
       roughly aligned.
    2. Check surface tilt — if the standard approach failed (tilt > 15°),
       try align_local_up_to_z() as a fallback (tracks SAM3D's Y-up through
       the transform chain, more robust for near-cubic shapes).
    3. Apply iterative surface leveling to fine-tune.
    4. For ceiling objects, check for upside-down pendant orientation.

    Args:
        mesh: trimesh.Trimesh in world coordinates (from camera-to-world)
        raw_local_mesh: original local-space mesh (Y-up) for vertex tracking
        target: 'floor' or 'ceiling'
        name: object name for logging
        verbose: print diagnostics
    """
    which = 'bottom' if target == 'floor' else 'top'

    # Step 1: Apply existing leveling (refines SAM3D's roughly-correct face)
    if target == 'ceiling':
        if verbose:
            print(f"      Standard leveling: PCA (ceiling)")
        mesh = level_ceiling_object(mesh, verbose=verbose)
    else:
        if verbose:
            print(f"      Standard leveling: PCA + OBB (floor)")
        mesh = level_mesh_to_floor(mesh, name=name, verbose=verbose)

    # Step 2: Check if leveling succeeded
    tilt = _check_surface_tilt(mesh, which)
    if verbose:
        print(f"      Post-leveling {which} surface tilt: {tilt:.1f}°")

    if tilt > 15 and raw_local_mesh is not None:
        # Standard leveling failed for this object (likely near-cubic shape
        # where OBB/PCA picked wrong face). Try tracking SAM3D's local Y-up
        # through the transform chain — this preserves the SAM3D orientation
        # rather than relying on ambiguous OBB geometry.
        if verbose:
            print(f"      Tilt > 15° — trying align_local_up_to_z fallback")
        mesh_alt = align_local_up_to_z(raw_local_mesh, mesh, verbose=verbose)
        tilt_alt = _check_surface_tilt(mesh_alt, which)
        if verbose:
            print(f"      Fallback {which} surface tilt: {tilt_alt:.1f}°")
        if tilt_alt < tilt:
            mesh = mesh_alt
            if verbose:
                print(f"      Using fallback (improved from {tilt:.1f}° to {tilt_alt:.1f}°)")
        else:
            if verbose:
                print(f"      Keeping standard leveling (fallback not better)")

    # Step 3: Iterative surface refinement (small corrections)
    for _ in range(3):
        mesh = level_surface(mesh, which=which, verbose=verbose)

    # Step 4: For ceiling objects, check for upside-down pendant orientation
    if target == 'ceiling':
        mesh = ensure_pendant_down(mesh, verbose=verbose)

    return mesh


def smart_level_floor(mesh, raw_local_mesh=None, name=None, verbose=False):
    """Multi-candidate floor leveling with dimension-preservation filtering.

    Tries several leveling strategies (raw, surface fits at different
    percentiles, OBB) and picks the best one according to:
      - Lowest bottom-surface tilt
      - While preserving the sorted AABB extents (strict: <=15% relative
        change, loose: <=30% as fallback when strict would leave the
        object still tilted)
      - If raw_local_mesh is available: reject winners that rotate SAM3D's
        tracked local Y-up much farther away from the raw pose, which is a
        strong signal that a front/side face was mistaken for the base

    Candidates:
      - raw: no change
      - top10x3: level_surface(top, 10%) iterated 3 times (reliable for
        objects with a broad flat top like wardrobes)
      - bot25x3: level_surface(bottom, 25%) x3 (broad base sampling helps
        flat-bottomed objects with sparse real bottom vertices)
      - bot10x3: level_surface(bottom, 10%) x3 (standard)
      - bot5x3 : level_surface(bottom, 5%)  x3 (tight base sampling for
        chairs / legged objects where the true contact patch is small)
      - obb    : level_mesh_to_floor (OBB-based), reliable for objects
        leaning heavily or on their side

    Returns the refined mesh (shape unchanged if already level or no
    viable candidate improves on raw).
    """
    LOCAL_UP_WORSEN_TOL = 15.0
    MINOR_TILT_SKIP_DEG = 4.0
    MINOR_TILT_HOVER_TRIGGER = 0.025
    MINOR_TILT_TARGET_LIFT = 0.012
    MINOR_TILT_MIN_IMPROVE = 0.008
    MINOR_TILT_MAX_RD = 0.10

    def _sorted_ext(m):
        return sorted((m.bounds[1] - m.bounds[0]).tolist())

    def _rel_diff(a, b):
        return max(abs(x - y) / max(x, 1e-6) for x, y in zip(a, b))

    raw_tilt = _check_surface_tilt(mesh, 'bottom')
    raw_sorted = sorted((mesh.bounds[1] - mesh.bounds[0]).tolist())
    raw_up_angle = (_tracked_local_up_angle(raw_local_mesh, mesh)
                    if raw_local_mesh is not None else None)
    raw_lift = _estimated_floor_lift(mesh, raw_tilt)

    if raw_tilt < MINOR_TILT_SKIP_DEG:
        # Small angular tilt can still look wrong on large furniture because
        # a 1-2° lean across a wide footprint leaves feet visibly hovering.
        if raw_lift > MINOR_TILT_HOVER_TRIGGER:
            try:
                obb_mesh = level_mesh_to_floor(mesh, name=name)
                obb_tilt = _check_surface_tilt(obb_mesh, 'bottom')
                obb_rd = _rel_diff(raw_sorted, _sorted_ext(obb_mesh))
                obb_lift = _estimated_floor_lift(obb_mesh, obb_tilt)
                obb_up_angle = (_tracked_local_up_angle(raw_local_mesh, obb_mesh)
                                if raw_local_mesh is not None else None)
                if raw_up_angle is not None and obb_up_angle is not None:
                    obb_up_worsen = max(0.0, obb_up_angle - raw_up_angle)
                else:
                    obb_up_worsen = 0.0

                if (obb_rd <= MINOR_TILT_MAX_RD and
                        obb_lift <= MINOR_TILT_TARGET_LIFT and
                        raw_lift - obb_lift >= MINOR_TILT_MIN_IMPROVE and
                        obb_up_worsen <= LOCAL_UP_WORSEN_TOL):
                    if verbose:
                        print(f"      {name or ''}: minor tilt but visible lift "
                              f"({raw_lift:.3f}m) -> OBB de-skew to {obb_lift:.3f}m")
                    return obb_mesh
            except Exception:
                pass

        if verbose:
            print(f"      {name or ''}: already level ({raw_tilt:.1f}°, "
                  f"est_lift={raw_lift:.3f}m)")
        return mesh

    def _pick_best(scored_candidates):
        if not scored_candidates:
            return None

        valid_strict = [c for c in scored_candidates if c['rd'] <= 0.15]
        best_strict = min(valid_strict, key=lambda c: c['tilt']) if valid_strict else None

        if best_strict is None or best_strict['tilt'] > 5.0:
            valid_loose = [c for c in scored_candidates if c['rd'] <= 0.30]
            best_loose = min(valid_loose, key=lambda c: c['tilt']) if valid_loose else None
            if best_loose and (best_strict is None or best_loose['tilt'] < best_strict['tilt'] - 3):
                return best_loose
            if best_strict:
                return best_strict
            return scored_candidates[0]

        return best_strict

    candidates = [('raw', mesh)]

    m = mesh
    for _ in range(3):
        m = level_surface(m, which='top', pct=0.10)
    candidates.append(('top10x3', m))

    m = mesh
    for _ in range(3):
        m = level_surface(m, which='bottom', pct=0.25)
    candidates.append(('bot25x3', m))

    m = mesh
    for _ in range(3):
        m = level_surface(m, which='bottom', pct=0.10)
    candidates.append(('bot10x3', m))

    m = mesh
    for _ in range(3):
        m = level_surface(m, which='bottom', pct=0.05)
    candidates.append(('bot5x3', m))

    try:
        candidates.append(('obb', level_mesh_to_floor(mesh, name=name)))
    except Exception:
        pass

    scored = []
    for label, m in candidates:
        t = _check_surface_tilt(m, 'bottom')
        rd = _rel_diff(raw_sorted, _sorted_ext(m))
        up_angle = (_tracked_local_up_angle(raw_local_mesh, m)
                    if raw_local_mesh is not None else None)
        if raw_up_angle is not None and up_angle is not None:
            up_worsen = max(0.0, up_angle - raw_up_angle)
        else:
            up_worsen = 0.0
        scored.append({
            'label': label,
            'mesh': m,
            'tilt': t,
            'rd': rd,
            'up_angle': up_angle,
            'up_worsen': up_worsen,
        })

    best = _pick_best(scored)

    # Preserve the raw SAM3D "up" prior when surface-fitting would flip an
    # object onto a face/side that happens to be flatter than the real base.
    if (best is not None and raw_up_angle is not None and
            best['up_worsen'] > LOCAL_UP_WORSEN_TOL):
        safe_candidates = [c for c in scored if c['up_worsen'] <= LOCAL_UP_WORSEN_TOL]
        safe_best = _pick_best(safe_candidates)
        if safe_best is not None and safe_best['mesh'] is not best['mesh']:
            if verbose:
                print(f"      Rejecting {best['label']} — local-up worsened "
                      f"{best['up_worsen']:.1f}° (raw={raw_up_angle:.1f}°, "
                      f"cand={best['up_angle']:.1f}°); using {safe_best['label']}")
            best = safe_best

    if verbose:
        strict_ids = {id(c) for c in scored if c['rd'] <= 0.15}
        for c in scored:
            if c['label'] == best['label']:
                marker = " *"
            elif id(c) in strict_ids:
                marker = "  "
            else:
                marker = " X"
            extra = ""
            if raw_up_angle is not None and c['up_angle'] is not None:
                extra = (f" up={c['up_angle']:5.1f}°"
                         f" worsen={c['up_worsen']:5.1f}°")
            print(f"      {marker} {c['label']:10s}: tilt={c['tilt']:5.1f}° "
                  f"rd={c['rd']:.2%}{extra}")

    return best['mesh']


def classify_object_by_name(name):
    """Pre-leveling classification using only the object name/label.

    Returns 'ceiling', 'wall_art', 'window', or None (needs geometry check).
    This enables category-aware leveling before the full classify_object()
    which requires post-leveling geometry for flat_floor detection.
    """
    label_lower = name.lower().rstrip()
    if 'window' in label_lower:
        return 'window'
    if 'painting' in label_lower or 'tv' in label_lower or 'television' in label_lower:
        return 'wall_art'
    if label_lower.endswith('_w'):
        return 'wall_art'
    if label_lower.endswith('_c'):
        return 'ceiling'
    return None


def get_mesh_aabb(mesh):
    """
    Get axis-aligned bounding box as (min_point, max_point).

    Args:
        mesh: trimesh.Trimesh

    Returns:
        Tuple of (min_corner, max_corner) as numpy arrays
    """
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()


def get_mesh_obb(mesh):
    """
    Get oriented bounding box data for a mesh.

    Returns:
        dict with 'vertices', 'center', 'extents', 'axes' (3x3 rotation matrix),
        'min_z', 'max_z' for the OBB in world space
    """
    obb = mesh.bounding_box_oriented
    transform = obb.primitive.transform
    return {
        'vertices': obb.vertices.copy(),
        'center': transform[:3, 3].copy(),
        'axes': transform[:3, :3].copy(),  # columns are principal axes
        'extents': obb.primitive.extents.copy(),
        'min_z': obb.vertices[:, 2].min(),
        'max_z': obb.vertices[:, 2].max(),
    }


def diagnose_mesh_obb(mesh, name, verbose=True):
    """
    Compare AABB and OBB to detect mesh quality issues.
    Call this after each transformation step to track changes.

    Diagnostics help identify:
    - Meshes with outlier vertices (large extents)
    - Meshes that aren't axis-aligned (high AABB/OBB volume ratio)
    - PCA instability in OBB computation

    Args:
        mesh: trimesh.Trimesh to diagnose
        name: Descriptive name for logging
        verbose: If True, print diagnostic info

    Returns:
        dict with diagnostic metrics or None if OBB fails
    """
    bounds = mesh.bounds
    aabb_extents = bounds[1] - bounds[0]
    aabb_volume = np.prod(aabb_extents)

    try:
        obb = mesh.bounding_box_oriented
        obb_extents = obb.primitive.extents
        obb_volume = np.prod(obb_extents)
        obb_axes = obb.primitive.transform[:3, :3]

        # Avoid division by zero
        if obb_volume < 1e-10:
            volume_ratio = float('inf')
        else:
            volume_ratio = aabb_volume / obb_volume

        if verbose:
            print(f"    [{name}] AABB extents: [{aabb_extents[0]:.3f}, {aabb_extents[1]:.3f}, {aabb_extents[2]:.3f}]")
            print(f"    [{name}] OBB extents:  [{obb_extents[0]:.3f}, {obb_extents[1]:.3f}, {obb_extents[2]:.3f}]")
            print(f"    [{name}] Volume ratio (AABB/OBB): {volume_ratio:.2f}")

            # Check if OBB axes are axis-aligned (dot product with X/Y/Z should be ~1)
            # For each OBB axis, find the max alignment with any world axis
            x_alignment = max(abs(obb_axes[0, i]) for i in range(3))
            y_alignment = max(abs(obb_axes[1, i]) for i in range(3))
            z_alignment = max(abs(obb_axes[2, i]) for i in range(3))
            print(f"    [{name}] OBB axis alignment: X={x_alignment:.3f}, Y={y_alignment:.3f}, Z={z_alignment:.3f}")

            if volume_ratio > 1.5:
                print(f"    [{name}] WARNING: Large AABB/OBB ratio - mesh is NOT axis-aligned")
            if volume_ratio > 3.0:
                print(f"    [{name}] CRITICAL: Very large ratio - possible outlier vertices")
            if any(e > 5.0 for e in aabb_extents):
                print(f"    [{name}] WARNING: Extent > 5m - mesh may have outliers")

        return {
            'aabb_extents': aabb_extents,
            'obb_extents': obb_extents,
            'volume_ratio': volume_ratio,
            'is_aligned': volume_ratio < 1.2,
            'has_outliers': any(e > 5.0 for e in aabb_extents)
        }
    except Exception as e:
        if verbose:
            print(f"    [{name}] OBB computation failed: {e}")
        return None


def create_aabb_wireframe(aabb, name, color=None):
    """
    Create a semi-transparent box mesh from AABB for debug visualization.

    Uses trimesh.creation.box() to create a transparent box that shows
    the bounding box of an object in the scene.

    Args:
        aabb: Tuple of (min_point, max_point) as numpy arrays
        name: Name for the debug mesh (unused, for reference)
        color: RGBA color list [R, G, B, A], default red semi-transparent

    Returns:
        trimesh.Trimesh box mesh positioned at the AABB location
    """
    if color is None:
        color = [255, 0, 0, 64]  # Red, semi-transparent

    min_pt, max_pt = aabb
    min_pt = np.asarray(min_pt)
    max_pt = np.asarray(max_pt)

    extents = max_pt - min_pt
    center = (min_pt + max_pt) / 2

    # Create box at origin with correct extents
    box = trimesh.creation.box(extents=extents)

    # Translate to correct position
    box.vertices += center

    # Set semi-transparent color for all faces
    box.visual.face_colors = color

    return box


def create_obb_wireframe(mesh, name, color=None, use_aabb=False):
    """
    Create a semi-transparent bounding box mesh for debug visualization.

    Args:
        mesh: trimesh.Trimesh to create bounding box for
        name: Name for the debug mesh (unused, for reference)
        color: RGBA color list [R, G, B, A], default red semi-transparent
        use_aabb: If True, use AABB instead of OBB. Recommended after
                  axis alignment since OBB axes may not match world axes.

    Returns:
        trimesh.Trimesh box mesh representing the bounding box
    """
    if color is None:
        color = [255, 0, 0, 64]  # Red, semi-transparent

    # After axis alignment, AABB is more appropriate since:
    # 1. We've explicitly aligned the mesh to world X/Y/Z axes
    # 2. OBB's PCA-based axes may not align with world axes even for axis-aligned meshes
    # 3. AABB guarantees the box edges are parallel to world axes
    if use_aabb:
        aabb = get_mesh_aabb(mesh)
        return create_aabb_wireframe(aabb, name, color)

    try:
        obb = mesh.bounding_box_oriented
        # OBB is already a mesh with correct vertices/faces
        box = trimesh.Trimesh(
            vertices=obb.vertices.copy(),
            faces=obb.faces.copy(),
            process=False
        )
        box.visual.face_colors = color
        return box
    except Exception:
        # Fallback to AABB if OBB computation fails
        aabb = get_mesh_aabb(mesh)
        return create_aabb_wireframe(aabb, name, color)


def aabbs_overlap_xy(aabb1, aabb2):
    """
    Check if two AABBs overlap in XY plane.

    Args:
        aabb1: Tuple of (min_point, max_point)
        aabb2: Tuple of (min_point, max_point)

    Returns:
        bool: True if AABBs overlap in XY
    """
    min1, max1 = aabb1
    min2, max2 = aabb2
    return not (max1[0] < min2[0] or max2[0] < min1[0] or
                max1[1] < min2[1] or max2[1] < min1[1])


def _get_obb_corners_xy(obb):
    """
    Get the 4 unique XY corners of an OBB (projected onto XY plane).

    Args:
        obb: dict from get_mesh_obb() with 'vertices'

    Returns:
        numpy array of shape (4, 2) with XY coordinates of corners
    """
    # OBB has 8 vertices (box corners), project to XY and get unique 4
    vertices_xy = obb['vertices'][:, :2]
    # Round to avoid floating point issues when finding unique points
    rounded = np.round(vertices_xy, decimals=6)
    unique_xy = np.unique(rounded, axis=0)
    return unique_xy


def _project_corners_onto_axis(corners, axis):
    """
    Project 2D corners onto a 1D axis and return (min, max) interval.

    Args:
        corners: (N, 2) array of 2D points
        axis: (2,) unit vector

    Returns:
        tuple (min_projection, max_projection)
    """
    projections = corners @ axis
    return projections.min(), projections.max()


def _intervals_overlap(interval1, interval2):
    """Check if two 1D intervals overlap."""
    min1, max1 = interval1
    min2, max2 = interval2
    return not (max1 < min2 or max2 < min1)


def obbs_overlap_xy(obb1, obb2):
    """
    Check if two OBBs overlap when projected onto the XY plane.

    Uses Separating Axis Theorem (SAT) with 4 axes:
    - 2 axes from OBB1's local X and Y (projected to XY plane)
    - 2 axes from OBB2's local X and Y (projected to XY plane)

    If projections overlap on ALL 4 axes, the OBBs overlap.
    If projections are separate on ANY axis, OBBs don't overlap.

    Args:
        obb1: dict from get_mesh_obb() with 'axes', 'vertices'
        obb2: dict from get_mesh_obb() with 'axes', 'vertices'

    Returns:
        bool: True if OBBs overlap in XY plane
    """
    # Get XY corners for both OBBs
    corners1 = _get_obb_corners_xy(obb1)
    corners2 = _get_obb_corners_xy(obb2)

    # Get test axes from both OBBs
    # OBB axes are the columns of the rotation matrix
    # We only need XY components and normalize them
    test_axes = []

    for obb in [obb1, obb2]:
        axes_3d = obb['axes']
        # Take first two columns (local X and Y axes)
        for i in range(2):
            axis_xy = axes_3d[:2, i]  # XY components
            norm = np.linalg.norm(axis_xy)
            if norm > 1e-6:
                test_axes.append(axis_xy / norm)

    # Test all axes using SAT
    for axis in test_axes:
        interval1 = _project_corners_onto_axis(corners1, axis)
        interval2 = _project_corners_onto_axis(corners2, axis)
        if not _intervals_overlap(interval1, interval2):
            # Found a separating axis - no overlap
            return False

    # No separating axis found - OBBs overlap
    return True


def compute_drop_target_z(mesh_obb, placed_obbs, floor_z=0.0):
    """
    Compute the Z position where mesh should rest after dropping.

    Uses OBB for XY overlap detection and stacking height.

    Args:
        mesh_obb: dict from get_mesh_obb() for mesh being placed
        placed_obbs: List of OBB dicts for already-placed objects
        floor_z: Z coordinate of the floor

    Returns:
        float: Target Z for the bottom of the bounding box
    """
    target_z = floor_z

    for placed_obb in placed_obbs:
        if obbs_overlap_xy(mesh_obb, placed_obb):
            # If overlapping in XY, must rest on top of this object
            target_z = max(target_z, placed_obb['max_z'])

    return target_z


def drop_mesh_to_z(mesh, target_bottom_z):
    """
    Translate mesh so its bottom is at target_bottom_z.

    Args:
        mesh: trimesh.Trimesh to drop
        target_bottom_z: Target Z for the bottom of the bounding box

    Returns:
        New trimesh.Trimesh at the target position
    """
    # Use AABB min_z since mesh is axis-aligned after leveling
    current_min_z = mesh.bounds[0][2]
    offset = target_bottom_z - current_min_z

    new_vertices = mesh.vertices.copy()
    new_vertices[:, 2] += offset

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )

    _copy_visual(mesh, new_mesh)
    return new_mesh


def raise_mesh_to_z(mesh, target_top_z):
    """Translate mesh so its top is at target_top_z (for ceiling-mounted objects)."""
    current_max_z = mesh.bounds[1][2]
    offset = target_top_z - current_max_z

    new_vertices = mesh.vertices.copy()
    new_vertices[:, 2] += offset

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )

    _copy_visual(mesh, new_mesh)
    return new_mesh


# =============================================================================
# Interpenetration Resolution Functions
# =============================================================================


def get_room_bounds_xy(room_mesh):
    """
    Get the XY bounding box of the room (for wall collision detection).

    Args:
        room_mesh: The room mesh (trimesh.Trimesh or Scene)

    Returns:
        Tuple of (min_xy, max_xy) as numpy arrays [x, y]
    """
    if isinstance(room_mesh, trimesh.Scene):
        # Combine all geometry bounds
        all_vertices = []
        for geom in room_mesh.geometry.values():
            all_vertices.append(geom.vertices)
        vertices = np.vstack(all_vertices)
    else:
        vertices = room_mesh.vertices

    min_xy = vertices[:, :2].min(axis=0)
    max_xy = vertices[:, :2].max(axis=0)
    return min_xy, max_xy


# =============================================================================
# Floor-Only Placement with Wall-Aware Conflict Resolution
# =============================================================================


def load_room_geometry(scene_json_path, room_id):
    """
    Load floor polygon and wall thickness from scene JSON.

    Args:
        scene_json_path: Path to scene layout JSON
        room_id: Room identifier (e.g., 'master_bedroom')

    Returns:
        Tuple of (shapely.Polygon, wall_thickness) or (None, None) if not found
    """
    with open(scene_json_path) as f:
        scene = json.load(f)

    wall_thickness = scene.get('metadata', {}).get('wall_thickness', 0.1)

    # Find the room by ID
    for room in scene.get('rooms', []):
        if room.get('id') == room_id or room.get('name') == room_id:
            floor_polygon = room.get('floor_polygon', [])
            if floor_polygon:
                return Polygon(floor_polygon), wall_thickness

    return None, None


def find_nearest_wall_direction(point_xy, room_polygon):
    """
    Find unit vector pointing toward nearest wall from a point inside room.

    Args:
        point_xy: (x, y) point inside the room
        room_polygon: Shapely Polygon representing the room boundary

    Returns:
        numpy array [dx, dy] unit vector toward nearest wall
    """
    pt = Point(point_xy)
    boundary = room_polygon.exterior

    # Find nearest point on boundary
    nearest_pt = boundary.interpolate(boundary.project(pt))
    nearest_xy = np.array([nearest_pt.x, nearest_pt.y])

    # Direction from point toward nearest wall
    direction = nearest_xy - np.array(point_xy)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        # Point is on the wall, pick arbitrary direction
        return np.array([1.0, 0.0])
    return direction / norm


def compute_separation_push(fixed_aabb, moving_aabb, push_direction):
    """
    Compute distance to push moving object along direction to clear fixed object.

    Args:
        fixed_aabb: Tuple of (min_xy, max_xy) for stationary object
        moving_aabb: Tuple of (min_xy, max_xy) for object being pushed
        push_direction: Unit vector [dx, dy] in XY plane

    Returns:
        float: Distance to push to separate objects (0 if no overlap)
    """
    fixed_min, fixed_max = fixed_aabb
    moving_min, moving_max = moving_aabb

    # Check if there's XY overlap
    overlap_x = min(fixed_max[0], moving_max[0]) - max(fixed_min[0], moving_min[0])
    overlap_y = min(fixed_max[1], moving_max[1]) - max(fixed_min[1], moving_min[1])

    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0  # No overlap

    # Project AABB corners onto push direction to find required separation
    # We need to find the overlap interval along the push direction
    push_dir = np.array(push_direction[:2])

    # Get all corners of both AABBs
    fixed_corners = np.array([
        [fixed_min[0], fixed_min[1]],
        [fixed_max[0], fixed_min[1]],
        [fixed_min[0], fixed_max[1]],
        [fixed_max[0], fixed_max[1]]
    ])
    moving_corners = np.array([
        [moving_min[0], moving_min[1]],
        [moving_max[0], moving_min[1]],
        [moving_min[0], moving_max[1]],
        [moving_max[0], moving_max[1]]
    ])

    # Project onto push direction
    fixed_proj = fixed_corners @ push_dir
    moving_proj = moving_corners @ push_dir

    fixed_min_proj, fixed_max_proj = fixed_proj.min(), fixed_proj.max()
    moving_min_proj, moving_max_proj = moving_proj.min(), moving_proj.max()

    # Calculate separation distance along push direction
    # Moving object needs to clear fixed object's far edge
    if push_dir @ push_dir > 0:  # Pushing in positive direction
        separation = fixed_max_proj - moving_min_proj
    else:
        separation = moving_max_proj - fixed_min_proj

    # Add small epsilon for clearance
    return max(0.0, separation + 0.01)


def get_aabb_footprint(mesh):
    """
    Get the XY footprint area of a mesh's AABB.

    Args:
        mesh: trimesh.Trimesh

    Returns:
        float: XY area of bounding box
    """
    bounds = mesh.bounds
    return (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])


def get_mesh_aabb_xy(mesh):
    """
    Get mesh AABB as (min_xy, max_xy) tuple.

    Args:
        mesh: trimesh.Trimesh

    Returns:
        Tuple of (min_xy, max_xy) as 2D numpy arrays
    """
    bounds = mesh.bounds
    return bounds[0][:2].copy(), bounds[1][:2].copy()


def aabbs_overlap_xy_2d(aabb1, aabb2):
    """
    Check if two 2D AABBs overlap.

    Args:
        aabb1: Tuple of (min_xy, max_xy) as 2D arrays
        aabb2: Tuple of (min_xy, max_xy) as 2D arrays

    Returns:
        bool: True if AABBs overlap
    """
    min1, max1 = aabb1
    min2, max2 = aabb2
    return not (max1[0] <= min2[0] or max2[0] <= min1[0] or
                max1[1] <= min2[1] or max2[1] <= min1[1])


def clamp_mesh_to_polygon(mesh, inner_polygon, verbose=False):
    """
    Translate mesh minimally to be fully inside room polygon.

    Uses binary search along direction toward polygon centroid.

    Args:
        mesh: trimesh.Trimesh to clamp
        inner_polygon: Shapely Polygon (already inset from walls)
        verbose: Print debug info

    Returns:
        New mesh translated to be inside polygon, or original if already inside
    """
    aabb_min, aabb_max = get_mesh_aabb_xy(mesh)

    # Check if all corners are inside
    corners = [
        (aabb_min[0], aabb_min[1]),
        (aabb_max[0], aabb_min[1]),
        (aabb_min[0], aabb_max[1]),
        (aabb_max[0], aabb_max[1])
    ]

    all_inside = all(inner_polygon.contains(Point(c)) for c in corners)
    if all_inside:
        return mesh

    # Find centroid of mesh and polygon
    mesh_center = (aabb_min + aabb_max) / 2
    poly_centroid = np.array([inner_polygon.centroid.x, inner_polygon.centroid.y])

    # Direction toward polygon center
    direction = poly_centroid - mesh_center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        # Already at center, try arbitrary direction
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / norm

    # Binary search for minimum translation
    min_dist = 0.0
    max_dist = norm * 2  # Search up to 2x the distance to centroid

    for _ in range(20):  # Binary search iterations
        mid_dist = (min_dist + max_dist) / 2
        test_offset = direction * mid_dist

        test_corners = [
            (aabb_min[0] + test_offset[0], aabb_min[1] + test_offset[1]),
            (aabb_max[0] + test_offset[0], aabb_min[1] + test_offset[1]),
            (aabb_min[0] + test_offset[0], aabb_max[1] + test_offset[1]),
            (aabb_max[0] + test_offset[0], aabb_max[1] + test_offset[1])
        ]

        if all(inner_polygon.contains(Point(c)) for c in test_corners):
            max_dist = mid_dist
        else:
            min_dist = mid_dist

    # Use the found distance (max_dist is the first working distance)
    final_offset = direction * max_dist

    if verbose:
        print(f"      Clamping to polygon: offset=[{final_offset[0]:.3f}, {final_offset[1]:.3f}]")

    new_vertices = mesh.vertices.copy()
    new_vertices[:, 0] += final_offset[0]
    new_vertices[:, 1] += final_offset[1]

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def translate_mesh_xy(mesh, offset_xy):
    """
    Translate mesh by XY offset.

    Args:
        mesh: trimesh.Trimesh
        offset_xy: [dx, dy] translation

    Returns:
        New translated mesh
    """
    new_vertices = mesh.vertices.copy()
    new_vertices[:, 0] += offset_xy[0]
    new_vertices[:, 1] += offset_xy[1]

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def rotate_mesh_z_around_center(mesh, angle_rad):
    """Rotate mesh vertices around Z-axis, pivoting at XY center.

    Args:
        mesh: trimesh.Trimesh
        angle_rad: rotation angle in radians

    Returns:
        New mesh with rotated vertices and copied visuals
    """
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    center = mesh.vertices.mean(axis=0)
    new_vertices = (mesh.vertices - center) @ rot_z.T + center
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def orient_object_to_wall(mesh, wall_edge, name):
    """Rotate object so its appropriate face aligns parallel to wall.

    - Beds: align SMALLER XY dimension parallel to wall (headboard against wall)
    - Other objects: align LARGER XY dimension parallel to wall (back against wall)

    Args:
        mesh: trimesh.Trimesh (axis-aligned after leveling)
        wall_edge: dict with 'direction' (unit vector along wall)
        name: object name (used to detect bed objects)

    Returns:
        Rotated mesh, or original if already aligned
    """
    aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
    extent_x = aabb_max[0] - aabb_min[0]
    extent_y = aabb_max[1] - aabb_min[1]

    wall_dir = wall_edge['direction']
    wall_angle = np.arctan2(wall_dir[1], wall_dir[0])

    is_bed = 'bed' in name.lower()

    if is_bed:
        # Beds: smaller dimension parallel to wall (headboard touches wall)
        if extent_x <= extent_y:
            current_angle = 0.0  # X axis is smaller, align X to wall
        else:
            current_angle = np.pi / 2  # Y axis is smaller, align Y to wall
    else:
        # Other objects: larger dimension parallel to wall
        if extent_x >= extent_y:
            current_angle = 0.0  # X axis is larger, align X to wall
        else:
            current_angle = np.pi / 2  # Y axis is larger, align Y to wall

    # Compute rotation needed
    rotation = wall_angle - current_angle

    # Normalize to [-pi, pi]
    while rotation > np.pi:
        rotation -= 2 * np.pi
    while rotation < -np.pi:
        rotation += 2 * np.pi

    # Pick nearest 90-degree multiple to stay axis-aligned
    candidates = [0, np.pi / 2, np.pi, -np.pi / 2]
    best = min(candidates, key=lambda c: abs(rotation - c))
    rotation = best

    # Skip if already within 2 degrees
    if abs(rotation) < np.radians(2):
        return mesh

    return rotate_mesh_z_around_center(mesh, rotation)


def _mesh_footprint_polygon(mesh, padding=0.02):
    """Return a padded XY footprint polygon for overlap checks."""
    from shapely.geometry import MultiPoint

    pts_xy = mesh.vertices[:, :2]
    try:
        hull = MultiPoint(pts_xy).convex_hull
        if hull.is_empty:
            raise ValueError("empty hull")
    except Exception:
        b = mesh.bounds
        hull = box(b[0][0], b[0][1], b[1][0], b[1][1])

    if padding > 0:
        hull = hull.buffer(padding)
    return hull


def _mesh_support_polygon(mesh, padding=0.01, max_support_height=0.20):
    """Return an XY footprint from the mesh's lower support region."""
    from shapely.geometry import MultiPoint

    bounds = mesh.bounds
    height = float(bounds[1][2] - bounds[0][2])
    z_limit = bounds[0][2] + min(max_support_height, max(0.06, 0.25 * height))
    pts_xy = mesh.vertices[mesh.vertices[:, 2] <= z_limit][:, :2]
    if len(pts_xy) < 3:
        return _mesh_footprint_polygon(mesh, padding=padding)

    try:
        hull = MultiPoint(pts_xy).convex_hull
        if hull.is_empty:
            raise ValueError("empty support hull")
    except Exception:
        return _mesh_footprint_polygon(mesh, padding=padding)

    if padding > 0:
        hull = hull.buffer(padding)
    return hull


def _horizontal_obb_profile(mesh):
    """Estimate a mesh's horizontal width/depth axes from its OBB."""
    try:
        obb = mesh.bounding_box_oriented
        axes = obb.primitive.transform[:3, :3]
        extents = obb.primitive.extents
    except Exception:
        axes = None
        extents = None

    horizontals = []
    if axes is not None and extents is not None:
        for i in range(3):
            axis = axes[:, i]
            xy_norm = np.linalg.norm(axis[:2])
            if xy_norm > 0.5:
                horizontals.append({
                    'axis_xy': axis[:2] / xy_norm,
                    'extent': float(extents[i]),
                })

    if len(horizontals) >= 2:
        horizontals.sort(key=lambda item: item['extent'])
        height_extent = float(mesh.bounds[1][2] - mesh.bounds[0][2])
        return {
            'depth_axis': horizontals[0]['axis_xy'],
            'depth_extent': horizontals[0]['extent'],
            'width_axis': horizontals[-1]['axis_xy'],
            'width_extent': horizontals[-1]['extent'],
            'height_extent': height_extent,
        }

    bounds = mesh.bounds
    ext_xy = bounds[1][:2] - bounds[0][:2]
    if min(ext_xy) < 1e-6:
        return None
    if ext_xy[0] <= ext_xy[1]:
        depth_axis = np.array([1.0, 0.0])
        width_axis = np.array([0.0, 1.0])
        depth_extent = float(ext_xy[0])
        width_extent = float(ext_xy[1])
    else:
        depth_axis = np.array([0.0, 1.0])
        width_axis = np.array([1.0, 0.0])
        depth_extent = float(ext_xy[1])
        width_extent = float(ext_xy[0])
    return {
        'depth_axis': depth_axis,
        'depth_extent': depth_extent,
        'width_axis': width_axis,
        'width_extent': width_extent,
        'height_extent': float(bounds[1][2] - bounds[0][2]),
    }


def _mesh_back_distance_to_wall(mesh, wall_edge):
    """Perpendicular distance from wall line to the closest XY point of the mesh."""
    verts_xy = mesh.vertices[:, :2]
    inward = wall_edge['inward_normal']
    return float(((verts_xy - wall_edge['start']) @ inward).min())


def _mesh_interval_along_wall(mesh, wall_edge):
    """Projected [min, max] interval of a mesh along a wall edge."""
    verts_xy = mesh.vertices[:, :2]
    wall_dir = wall_edge['direction']
    projections = (verts_xy - wall_edge['start']) @ wall_dir
    return float(projections.min()), float(projections.max())


def _signed_angle_2d(a, b):
    """Signed angle rotating 2D vector a onto b."""
    cross = a[0] * b[1] - a[1] * b[0]
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.arctan2(cross, dot))


def _wall_alignment_kind(name):
    """Classify the wall-facing axis used for flush wall alignment."""
    if 'bed' in name.lower():
        return 'bed'
    return 'default'


def _wall_normal_axis(profile, wall_kind):
    """Return the horizontal axis that should face the wall normal."""
    if wall_kind == 'bed':
        return profile['width_axis']
    return profile['depth_axis']


def _wall_normal_extent(profile, wall_kind):
    """Return the object's horizontal extent perpendicular to the backing wall."""
    if wall_kind == 'bed':
        return profile['width_extent']
    return profile['depth_extent']


def _wall_alignment_error_deg(profile, wall_edge, wall_kind):
    """Angular error between the object's intended back face and a wall."""
    inward = wall_edge['inward_normal']
    normal_axis = _wall_normal_axis(profile, wall_kind)
    return float(np.degrees(
        np.arccos(np.clip(abs(np.dot(normal_axis, inward)), 0.0, 1.0))
    ))


def _select_reference_wall_edge(mesh, wall_edges, profile, wall_kind,
                                target_back_dist, max_align_deg):
    """Pick the most plausible supporting wall for a wall-adjacent object."""
    center = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2
    best = None

    for edge in wall_edges:
        align_error = _wall_alignment_error_deg(profile, edge, wall_kind)
        if align_error > max_align_deg:
            continue

        back_gap = _mesh_back_distance_to_wall(mesh, edge) - target_back_dist
        center_dist, _ = _point_to_segment_distance(center, edge['start'], edge['end'])
        score = (abs(back_gap), align_error, center_dist)

        if best is None or score < best['score']:
            best = {
                'wall_edge': edge,
                'back_gap': float(back_gap),
                'center_dist': float(center_dist),
                'align_error': float(align_error),
                'score': score,
            }

    return best


def _is_semantic_wall_object(name):
    """Return True for labels that are commonly expected to sit against a wall."""
    label = name.lower()
    wall_tokens = (
        'cabinet', 'wardrobe', 'dresser', 'drawer', 'bookshelf',
        'bookcase', 'nightdesk', 'nightstand', 'desk', 'sofa', 'bed',
        'console', 'sideboard', 'cupboard', 'shelf'
    )
    return any(token in label for token in wall_tokens)


def _opening_blocks_wall_object(mesh, opening, wall_edge, margin=0.08):
    """Return True if a wall opening blocks a flush wall placement."""
    obj_min_t, obj_max_t = _mesh_interval_along_wall(mesh, wall_edge)
    mesh_top_z = float(mesh.bounds[1][2])

    dist, _ = _point_to_segment_distance(
        opening['center_2d'], wall_edge['start'], wall_edge['end']
    )
    if dist > 0.3:
        return False

    center_t = np.dot(opening['center_2d'] - wall_edge['start'],
                      wall_edge['direction'])
    half_width = opening['width'] / 2 + margin
    if obj_min_t >= center_t + half_width or obj_max_t <= center_t - half_width:
        return False

    if opening.get('type') == 'window':
        z_min = opening.get('z_min')
        if z_min is not None and mesh_top_z <= z_min - 0.05:
            return False

    return True


def _mesh_hits_wall_opening(mesh, wall_edge, openings, margin=0.08):
    """Check whether a wall-flush placement would overlap a door/window opening."""
    for opening in openings:
        if _opening_blocks_wall_object(mesh, opening, wall_edge, margin=margin):
            return True

    return False


def _wall_slide_offsets_to_clear_openings(mesh, wall_edge, openings,
                                          max_slide=1.25, margin=0.08,
                                          edge_margin=0.05):
    """Return small along-wall shifts that move a flush object out of openings."""
    obj_min_t, obj_max_t = _mesh_interval_along_wall(mesh, wall_edge)
    obj_len = obj_max_t - obj_min_t
    wall_len = wall_edge['length']
    free_start = edge_margin
    free_end = wall_len - edge_margin

    if obj_len >= (free_end - free_start) - 1e-6:
        return []

    blocked = []
    for opening in openings:
        if not _opening_blocks_wall_object(mesh, opening, wall_edge, margin=margin):
            continue
        center_t = np.dot(opening['center_2d'] - wall_edge['start'],
                          wall_edge['direction'])
        half_width = opening['width'] / 2 + margin
        blocked.append((
            max(free_start, center_t - half_width),
            min(free_end, center_t + half_width),
        ))

    if not blocked:
        return [0.0]

    blocked.sort()
    merged = []
    for start, end in blocked:
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    free_intervals = []
    cursor = free_start
    for start, end in merged:
        if start > cursor + 1e-6:
            free_intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < free_end - 1e-6:
        free_intervals.append((cursor, free_end))

    center_t = 0.5 * (obj_min_t + obj_max_t)
    shifts = []
    for start, end in free_intervals:
        if end - start + 1e-6 < obj_len:
            continue
        center_min = start + 0.5 * obj_len
        center_max = end - 0.5 * obj_len
        target_center = np.clip(center_t, center_min, center_max)
        shift = float(target_center - center_t)
        if abs(shift) <= max_slide + 1e-6:
            shifts.append(shift)

    unique = []
    for shift in sorted(shifts, key=lambda v: (abs(v), v)):
        if not any(abs(shift - prev) < 1e-6 for prev in unique):
            unique.append(shift)
    return unique


def _is_bedside_companion_object(name, mesh, wall_edge, target_back_dist):
    """Return True for small side-table-like objects that can move with a bed."""
    label = name.lower()
    if not any(token in label for token in ('table', 'night', 'desk', 'cabinet', 'dresser')):
        return False

    profile = _horizontal_obb_profile(mesh)
    if profile is None:
        return False

    footprint_area = _mesh_footprint_polygon(mesh, padding=0.0).area
    back_gap = _mesh_back_distance_to_wall(mesh, wall_edge) - target_back_dist
    align_error = _wall_alignment_error_deg(profile, wall_edge, 'default')
    largest_extent = max(profile['depth_extent'], profile['width_extent'])

    return (
        footprint_area <= 0.85 and
        profile['height_extent'] <= 1.35 and
        profile['depth_extent'] <= 0.85 and
        largest_extent <= 1.25 and
        back_gap <= 0.75 and
        align_error <= 25.0
    )


def _shift_bedside_companions_with_bed(name, candidate_mesh, overlap_names,
                                       slide_shift, wall_edge, current_meshes,
                                       inner_polygon, openings,
                                       target_back_dist, overlap_eps=0.001):
    """Try moving small bedside companions together with a bed wall-slide."""
    moved = {}
    moved_fps = {
        name: _mesh_support_polygon(candidate_mesh, padding=0.0)
    }

    for other_name in overlap_names:
        other_mesh = current_meshes[other_name]
        if not _is_bedside_companion_object(
                other_name, other_mesh, wall_edge, target_back_dist):
            return None

        shifted = translate_mesh_xy(other_mesh, wall_edge['direction'] * slide_shift)
        shifted = clamp_mesh_to_polygon(shifted, inner_polygon)
        if _mesh_hits_wall_opening(shifted, wall_edge, openings):
            return None

        moved[other_name] = shifted
        moved_fps[other_name] = _mesh_support_polygon(shifted, padding=0.0)

    moved_names = [name] + list(moved.keys())
    for i, lhs in enumerate(moved_names):
        lhs_fp = moved_fps[lhs]
        for rhs in moved_names[i + 1:]:
            rhs_fp = moved_fps[rhs]
            if lhs_fp.intersects(rhs_fp) and lhs_fp.intersection(rhs_fp).area > overlap_eps:
                return None

    for moved_name, moved_fp in moved_fps.items():
        for other_name, other_mesh in current_meshes.items():
            if other_name == name or other_name in moved:
                continue
            other_fp = _mesh_support_polygon(other_mesh, padding=0.0)
            if moved_fp.intersects(other_fp) and moved_fp.intersection(other_fp).area > overlap_eps:
                return None

    return moved


def flush_likely_wall_objects_simple(object_meshes, reference_objects,
                                     room_polygon, wall_thickness,
                                     openings, verbose=False):
    """Make only clearly wall-adjacent simple-mode floor objects flush to walls.

    Uses the fitted SAM3D layout as a reference for wall affinity, then
    corrects the final placed floor objects if they are both:
      1. still close enough to the same wall to plausibly belong there, and
      2. semantically/geometrically the kind of object that normally sits
         with its back against a wall.

    The adjustment is conservative:
      - only modest Z-rotation corrections
      - only small along-wall slides when needed to clear an opening
      - skip any placement that would overlap another object or block an opening
    """
    if not object_meshes:
        return object_meshes

    SURFACE_GAP = 0.02
    REF_MAX_BACK_GAP = 0.38
    REF_MAX_CENTER_DIST = 1.05
    MAX_ALIGN_DEG = 18.0
    MAX_WALL_SLIDE = 1.25
    MIN_APPLY_GAP = 0.05
    MIN_APPLY_ANGLE = 2.0
    MIN_DEPTH_EXTENT = 0.14
    SHALLOW_RATIO_MAX = 0.60
    SHALLOW_MIN_WIDTH = 0.55
    SHALLOW_MIN_HEIGHT = 0.40
    OVERLAP_EPS = 0.001

    inner_margin = wall_thickness + SURFACE_GAP
    inner_polygon = room_polygon.buffer(-inner_margin)
    if inner_polygon.is_empty or not inner_polygon.is_valid:
        inner_polygon = room_polygon.buffer(-0.01)

    wall_edges = compute_wall_edges(room_polygon)
    target_back_dist = wall_thickness + SURFACE_GAP

    ref_map = {}
    for item in reference_objects:
        if len(item) >= 2:
            ref_map[item[0]] = item[1]

    current_meshes = {name: mesh for name, mesh in object_meshes}
    footprints = {
        name: _mesh_footprint_polygon(mesh, padding=0.02)
        for name, mesh in object_meshes
    }
    reference_pairs = [
        (name, ref_map.get(name, mesh))
        for name, mesh in object_meshes
    ]
    reference_blockers = [
        (name, ref_map.get(name, mesh))
        for name, mesh in object_meshes
        if classify_object_by_name(name) not in ('window', 'wall_art', 'ceiling')
    ]

    candidates = []
    for name, mesh in object_meshes:
        ref_mesh = ref_map.get(name)
        if ref_mesh is None:
            continue

        profile = _horizontal_obb_profile(mesh)
        ref_profile = _horizontal_obb_profile(ref_mesh)
        if profile is None or ref_profile is None:
            continue
        if ref_profile['depth_extent'] < MIN_DEPTH_EXTENT:
            continue

        wall_kind = _wall_alignment_kind(name)

        semantic_wall = _is_semantic_wall_object(name)
        geom_wall = (
            ref_profile['width_extent'] >= SHALLOW_MIN_WIDTH and
            ref_profile['height_extent'] >= SHALLOW_MIN_HEIGHT and
            ref_profile['depth_extent'] / max(ref_profile['width_extent'], 1e-6)
            <= SHALLOW_RATIO_MAX
        )
        if not (semantic_wall or geom_wall):
            continue

        ref_center = (ref_mesh.bounds[0][:2] + ref_mesh.bounds[1][:2]) / 2
        ref_wall = _select_reference_wall_edge(
            ref_mesh, wall_edges, ref_profile, wall_kind,
            target_back_dist, MAX_ALIGN_DEG,
        )
        if ref_wall is None:
            continue
        wall_edge = ref_wall['wall_edge']
        ref_center_dist = ref_wall['center_dist']
        max_center_dist = max(
            REF_MAX_CENTER_DIST,
            target_back_dist + 0.5 * _wall_normal_extent(ref_profile, wall_kind)
            + REF_MAX_BACK_GAP + 0.10,
        )
        if ref_center_dist > max_center_dist:
            continue

        blocking_count = count_objects_between(
            ref_center, wall_edge, reference_blockers, name
        )
        if blocking_count > 0:
            continue

        ref_back_gap = ref_wall['back_gap']
        cur_back_gap = _mesh_back_distance_to_wall(mesh, wall_edge) - target_back_dist
        if ref_back_gap > REF_MAX_BACK_GAP:
            continue

        normal_angle = _wall_alignment_error_deg(profile, wall_edge, wall_kind)
        if normal_angle > MAX_ALIGN_DEG:
            continue

        if (cur_back_gap < MIN_APPLY_GAP and normal_angle < MIN_APPLY_ANGLE):
            continue

        candidates.append({
            'name': name,
            'wall_edge': wall_edge,
            'wall_kind': wall_kind,
            'semantic_wall': semantic_wall,
            'ref_back_gap': ref_back_gap,
            'cur_back_gap': cur_back_gap,
            'normal_angle': normal_angle,
        })

    candidates.sort(key=lambda c: (
        0 if c['semantic_wall'] else 1,
        c['ref_back_gap'],
        -c['cur_back_gap'],
        -c['normal_angle'],
    ))

    applied = 0
    for cand in candidates:
        name = cand['name']
        mesh = current_meshes[name]
        wall_edge = cand['wall_edge']
        profile = _horizontal_obb_profile(mesh)
        if profile is None:
            continue

        inward = wall_edge['inward_normal']
        wall_kind = cand['wall_kind']
        normal_axis = _wall_normal_axis(profile, wall_kind)
        target_axis = inward.copy()
        if np.dot(normal_axis, target_axis) < 0:
            target_axis = -target_axis

        rotation = _signed_angle_2d(normal_axis, target_axis)
        rotation_deg = abs(np.degrees(rotation))
        if rotation_deg > MAX_ALIGN_DEG:
            continue

        candidate_mesh = mesh
        if rotation_deg >= 1.0:
            candidate_mesh = rotate_mesh_z_around_center(candidate_mesh, rotation)

        cur_back = _mesh_back_distance_to_wall(candidate_mesh, wall_edge)
        candidate_mesh = translate_mesh_xy(
            candidate_mesh, -inward * (cur_back - target_back_dist)
        )
        candidate_mesh = clamp_mesh_to_polygon(candidate_mesh, inner_polygon)

        # Clamping can nudge the object inward; try one more perpendicular
        # re-clamp to recover wall contact if that remains safe.
        post_clamp_back = _mesh_back_distance_to_wall(candidate_mesh, wall_edge)
        if post_clamp_back - target_back_dist > 0.01:
            repushed = translate_mesh_xy(
                candidate_mesh, -inward * (post_clamp_back - target_back_dist)
            )
            repushed = clamp_mesh_to_polygon(repushed, inner_polygon)
            if (_mesh_back_distance_to_wall(repushed, wall_edge)
                    < _mesh_back_distance_to_wall(candidate_mesh, wall_edge)):
                candidate_mesh = repushed

        variants = [(candidate_mesh, 0.0)]
        if _mesh_hits_wall_opening(candidate_mesh, wall_edge, openings):
            for shift in _wall_slide_offsets_to_clear_openings(
                    candidate_mesh, wall_edge, openings,
                    max_slide=MAX_WALL_SLIDE):
                if abs(shift) < 1e-6:
                    continue
                shifted = translate_mesh_xy(candidate_mesh, wall_edge['direction'] * shift)
                shifted = clamp_mesh_to_polygon(shifted, inner_polygon)
                shifted_back = _mesh_back_distance_to_wall(shifted, wall_edge)
                if abs(shifted_back - target_back_dist) > 0.01:
                    shifted = translate_mesh_xy(
                        shifted, -inward * (shifted_back - target_back_dist)
                    )
                    shifted = clamp_mesh_to_polygon(shifted, inner_polygon)
                variants.append((shifted, shift))

        accepted = False
        for variant_mesh, slide_shift in variants:
            if _mesh_hits_wall_opening(variant_mesh, wall_edge, openings):
                continue

            candidate_fp = _mesh_footprint_polygon(variant_mesh, padding=0.02)
            overlap_names = []
            for other_name, other_fp in footprints.items():
                if other_name == name:
                    continue
                if not candidate_fp.intersects(other_fp):
                    continue
                if candidate_fp.intersection(other_fp).area > OVERLAP_EPS:
                    overlap_names.append(other_name)

            companion_moves = {}
            if overlap_names:
                if wall_kind == 'bed' and abs(slide_shift) >= 0.01:
                    companion_moves = _shift_bedside_companions_with_bed(
                        name, variant_mesh, overlap_names, slide_shift,
                        wall_edge, current_meshes, inner_polygon, openings,
                        target_back_dist, overlap_eps=OVERLAP_EPS,
                    ) or {}
                if overlap_names and not companion_moves:
                    continue

            final_profile = _horizontal_obb_profile(variant_mesh)
            if final_profile is None:
                continue

            final_back_gap = (_mesh_back_distance_to_wall(variant_mesh, wall_edge)
                              - target_back_dist)
            final_angle = _wall_alignment_error_deg(final_profile, wall_edge, wall_kind)
            gap_improve = cand['cur_back_gap'] - final_back_gap
            angle_improve = cand['normal_angle'] - final_angle
            if gap_improve < 0.03 and angle_improve < 2.0:
                continue

            current_meshes[name] = variant_mesh
            footprints[name] = candidate_fp
            for other_name, other_mesh in companion_moves.items():
                current_meshes[other_name] = other_mesh
                footprints[other_name] = _mesh_footprint_polygon(
                    other_mesh, padding=0.02
                )
            applied += 1
            accepted = True

            if verbose:
                slide_msg = ""
                if abs(slide_shift) >= 0.01:
                    slide_msg = f", slide {slide_shift:+.2f}m"
                moved_msg = ""
                if companion_moves:
                    moved_msg = f", moved {', '.join(sorted(companion_moves))}"
                print(f"    {name}: wall-flush → wall #{wall_edge['edge_index']} "
                      f"(gap {cand['cur_back_gap']:.2f}m → {final_back_gap:.2f}m, "
                      f"angle {cand['normal_angle']:.1f}° → {final_angle:.1f}°"
                      f"{slide_msg}{moved_msg})")
            break

        if not accepted:
            continue

    if verbose:
        print(f"  Wall-flush refinement: {applied}/{len(candidates)} object(s) adjusted")

    return [(name, current_meshes[name]) for name, _mesh in object_meshes]


def compute_camera_depth(obj_center_xy, camera_position_xy, camera_forward_xy):
    """Compute signed depth of object along camera view direction.

    Args:
        obj_center_xy: [x, y] object center
        camera_position_xy: [x, y] camera position
        camera_forward_xy: [fx, fy] camera forward direction (unit vector)

    Returns:
        float: signed depth (positive = in front of camera)
    """
    diff = np.array(obj_center_xy) - np.array(camera_position_xy)
    return np.dot(diff, camera_forward_xy)


def find_perpendicular_walls(wall_idx, wall_edges, wall_capacities):
    """Find walls roughly perpendicular to source wall.

    Args:
        wall_idx: edge_index of source wall
        wall_edges: list of wall edge dicts
        wall_capacities: dict from compute_wall_capacities()

    Returns:
        list of edge indices sorted by distance from source wall midpoint
    """
    source_edge = None
    for e in wall_edges:
        if e['edge_index'] == wall_idx:
            source_edge = e
            break
    if source_edge is None:
        return []

    source_dir = source_edge['direction']
    source_mid = source_edge['midpoint']

    perp_walls = []
    for e in wall_edges:
        eidx = e['edge_index']
        if eidx == wall_idx or eidx not in wall_capacities:
            continue
        dot = abs(np.dot(source_dir, e['direction']))
        if dot < 0.3:  # roughly perpendicular
            dist = np.linalg.norm(e['midpoint'] - source_mid)
            perp_walls.append((eidx, dist))

    perp_walls.sort(key=lambda x: x[1])
    return [idx for idx, _ in perp_walls]


def spread_objects_along_wall(assignments, wall_idx, wall_capacities,
                               camera_position_xy, camera_forward_xy,
                               verbose=False):
    """Spread objects along wall to prevent stacking.

    Sorts objects by camera depth (nearest first) and places them
    sequentially into free intervals with spacing gaps.
    Only moves along wall direction; perpendicular position unchanged.

    Args:
        assignments: list of assignment dicts
        wall_idx: wall edge index to spread objects on
        wall_capacities: dict from compute_wall_capacities()
        camera_position_xy: [x, y] camera position (can be None)
        camera_forward_xy: [fx, fy] camera forward (can be None)
        verbose: print debug info
    """
    MIN_SPACING = 0.10  # 10cm gap between objects

    # Gather objects on this wall
    wall_objs = [(i, a) for i, a in enumerate(assignments)
                 if a['wall_idx'] == wall_idx and not a['is_center']]

    if len(wall_objs) < 2:
        return  # Nothing to spread

    cap = wall_capacities.get(wall_idx)
    if cap is None:
        return

    wall_edge = cap['wall_edge']
    wall_dir = wall_edge['direction']
    wall_start = wall_edge['start']

    # Check if any same-wall AABB overlaps exist
    has_overlap = False
    for ii in range(len(wall_objs)):
        aabb_i = get_mesh_aabb_xy(wall_objs[ii][1]['mesh'])
        for jj in range(ii + 1, len(wall_objs)):
            aabb_j = get_mesh_aabb_xy(wall_objs[jj][1]['mesh'])
            if aabbs_overlap_xy_2d(aabb_i, aabb_j):
                has_overlap = True
                break
        if has_overlap:
            break

    if not has_overlap:
        return  # No overlaps, skip spreading

    if verbose:
        names = [a['name'] for _, a in wall_objs]
        print(f"      Spreading {len(wall_objs)} objects on wall "
              f"#{wall_idx}: {names}")

    # Sort by camera depth (nearest camera first)
    if camera_position_xy is not None and camera_forward_xy is not None:
        def sort_key(item):
            _, a = item
            c = (a['mesh'].bounds[0][:2] + a['mesh'].bounds[1][:2]) / 2
            return compute_camera_depth(
                c, camera_position_xy, camera_forward_xy)
        wall_objs.sort(key=sort_key)

    # Compute object widths along wall
    obj_widths = []
    for _, a in wall_objs:
        aabb_min, aabb_max = get_mesh_aabb_xy(a['mesh'])
        corners = np.array([
            [aabb_min[0], aabb_min[1]],
            [aabb_max[0], aabb_min[1]],
            [aabb_min[0], aabb_max[1]],
            [aabb_max[0], aabb_max[1]]
        ])
        projs = corners @ wall_dir
        obj_widths.append(projs.max() - projs.min())

    # Total needed space
    total_needed = sum(obj_widths) + MIN_SPACING * (len(wall_objs) - 1)

    # Get free intervals on this wall
    free_intervals = cap['free_intervals']
    total_free = sum(end - start for start, end in free_intervals)

    if verbose and total_needed > total_free:
        print(f"        WARNING: need {total_needed:.2f}m but only "
              f"{total_free:.2f}m free")

    if not free_intervals:
        return

    # Use all free space as a single range
    all_free_start = free_intervals[0][0]
    all_free_end = free_intervals[-1][1]

    # Center the group within free space
    group_center = (all_free_start + all_free_end) / 2
    group_start = group_center - total_needed / 2
    group_start = max(group_start, all_free_start)

    cursor = group_start
    for (idx, a), width in zip(wall_objs, obj_widths):
        mesh = a['mesh']
        aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
        center = (aabb_min + aabb_max) / 2
        current_t = np.dot(center - wall_start, wall_dir)

        # Target position: cursor + half width
        target_t = cursor + width / 2

        # Slide along wall
        delta_t = target_t - current_t
        offset = wall_dir * delta_t

        mesh = translate_mesh_xy(mesh, offset)
        a['mesh'] = mesh

        if verbose:
            print(f"        {a['name']}: slid {delta_t:.3f}m along wall")

        cursor += width + MIN_SPACING


def avoid_doors(assignments, door_openings, inner_polygon, verbose=False):
    """Move objects out of door no-go zones.

    For each door, slides blocking objects along the wall direction
    using binary search to find the minimum slide distance.

    Args:
        assignments: list of assignment dicts
        door_openings: list of door opening dicts with 'nogo_polygon'
        inner_polygon: Shapely Polygon for room interior
        verbose: print debug info

    Returns:
        int: count of objects moved
    """
    moved_count = 0

    for door in door_openings:
        if 'nogo_polygon' not in door:
            continue
        nogo = door['nogo_polygon']

        for a in assignments:
            mesh = a['mesh']
            aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
            obj_box = box(aabb_min[0], aabb_min[1],
                          aabb_max[0], aabb_max[1])

            if not obj_box.intersects(nogo):
                continue

            if verbose:
                print(f"      {a['name']}: overlaps door no-go zone, "
                      "sliding along wall...")

            wall_dir = door['wall_dir']
            center = (aabb_min + aabb_max) / 2
            door_center = door['center_2d']

            side = np.dot(center - door_center, wall_dir)
            slide_dir = wall_dir if side >= 0 else -wall_dir

            # Binary search for minimum slide to clear no-go zone
            min_slide = 0.0
            max_slide = door['width'] + 1.0

            for _ in range(20):
                mid = (min_slide + max_slide) / 2
                test_mesh = translate_mesh_xy(mesh, slide_dir * mid)
                test_min, test_max = get_mesh_aabb_xy(test_mesh)
                test_box = box(test_min[0], test_min[1],
                               test_max[0], test_max[1])

                if not test_box.intersects(nogo):
                    max_slide = mid
                else:
                    min_slide = mid

            slide_amount = max_slide + 0.02

            # Helper: try a slide direction and return mesh if it clears
            def _try_slide(direction, amount):
                m = translate_mesh_xy(mesh, direction * amount)
                m_clamped = clamp_mesh_to_polygon(m, inner_polygon)
                c_min, c_max = get_mesh_aabb_xy(m_clamped)
                c_box = box(c_min[0], c_min[1], c_max[0], c_max[1])
                if not c_box.intersects(nogo):
                    return m_clamped
                return None

            result = None
            for mult in [1.0, 1.5, 2.0, 3.0]:
                result = _try_slide(slide_dir, slide_amount * mult)
                if result is not None:
                    break
            if result is not None:
                a['mesh'] = result
                moved_count += 1
                if verbose:
                    print(f"        -> slid along wall to clear door")
            else:
                # Try opposite direction
                result2 = None
                for mult in [1.0, 1.5, 2.0, 3.0]:
                    result2 = _try_slide(-slide_dir, slide_amount * mult)
                    if result2 is not None:
                        break
                if result2 is not None:
                    a['mesh'] = result2
                    moved_count += 1
                    if verbose:
                        print(f"        -> slid opposite direction "
                              "to clear door")
                else:
                    # Try pushing inward (away from wall)
                    inward = door['inward_normal']
                    pushed = False
                    for push_inward in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
                        result3 = _try_slide(inward, push_inward)
                        if result3 is not None:
                            a['mesh'] = result3
                            moved_count += 1
                            pushed = True
                            if verbose:
                                print(f"        -> pushed {push_inward:.1f}m "
                                      "inward from wall")
                            break

                    # If pure inward push failed, try combined slide + push
                    if not pushed:
                        for push_inward in [0.5, 1.0, 1.5]:
                            for slide_mult in [0.5, 1.0, -0.5, -1.0]:
                                combo_dir = (inward * push_inward
                                             + slide_dir * slide_mult)
                                combo_len = np.linalg.norm(combo_dir)
                                if combo_len < 0.01:
                                    continue
                                combo_dir_norm = combo_dir / combo_len
                                result4 = _try_slide(
                                    combo_dir_norm, combo_len)
                                if result4 is not None:
                                    a['mesh'] = result4
                                    moved_count += 1
                                    pushed = True
                                    if verbose:
                                        print(
                                            f"        -> combo: "
                                            f"{push_inward:.1f}m inward + "
                                            f"{slide_mult:.1f}m along wall")
                                    break
                            if pushed:
                                break

                    if not pushed and verbose:
                        print("        -> couldn't clear door, "
                              "keeping position")

    return moved_count


def reclamp_wall_objects(assignments, wall_thickness, verbose=False):
    """Re-push wall objects that drifted too far from their wall.

    After overlap resolution, some wall objects may have moved away
    from their assigned wall. This pushes them back toward the wall,
    but only to WALL_RECLAMP_DIST (not all the way). Also checks
    that the push doesn't create new overlaps with other objects.

    Args:
        assignments: list of assignment dicts
        wall_thickness: wall thickness in meters
        verbose: print debug info
    """
    WALL_RECLAMP_DIST = 0.15  # Max perpendicular drift before reclamp
    WALL_SURFACE_GAP = 0.02
    TARGET_DIST = WALL_RECLAMP_DIST  # Push back to this distance, not to wall

    for a_idx, a in enumerate(assignments):
        if a['is_center'] or a['wall_edge'] is None:
            continue

        wall_edge = a['wall_edge']
        inward = wall_edge['inward_normal']
        mesh = a['mesh']

        aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
        corners_xy = np.array([
            [aabb_min[0], aabb_min[1]],
            [aabb_max[0], aabb_min[1]],
            [aabb_min[0], aabb_max[1]],
            [aabb_max[0], aabb_max[1]]
        ])
        inward_dists = (corners_xy - wall_edge['start']) @ inward
        min_inward_dist = inward_dists.min()  # back of object

        wall_margin = wall_thickness + WALL_SURFACE_GAP
        drift = min_inward_dist - wall_margin

        if drift > WALL_RECLAMP_DIST:
            # Only push back to TARGET_DIST, not all the way to wall
            push_dist = drift - TARGET_DIST
            offset = -inward * push_dist
            test_mesh = translate_mesh_xy(mesh, offset)

            # Check if this would create new overlaps
            test_aabb = get_mesh_aabb_xy(test_mesh)
            creates_overlap = False
            for j, other in enumerate(assignments):
                if j == a_idx:
                    continue
                other_aabb = get_mesh_aabb_xy(other['mesh'])
                if aabbs_overlap_xy_2d(test_aabb, other_aabb):
                    creates_overlap = True
                    break

            if not creates_overlap:
                a['mesh'] = test_mesh
                if verbose:
                    print(f"      {a['name']}: reclamped to wall "
                          f"(drift={drift:.3f}m, pushed {push_dist:.3f}m)")
            elif verbose:
                print(f"      {a['name']}: skip reclamp "
                      f"(drift={drift:.3f}m, would cause overlap)")


def _is_in_any_nogo(mesh, door_openings):
    """Check if mesh AABB intersects any door nogo zone."""
    if not door_openings:
        return False
    aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
    obj_box = box(aabb_min[0], aabb_min[1], aabb_max[0], aabb_max[1])
    for door in door_openings:
        if 'nogo_polygon' not in door:
            continue
        if obj_box.intersects(door['nogo_polygon']):
            return True
    return False


def _find_valid_position(mesh, assignments, exclude_name,
                         inner_polygon, door_openings):
    """Grid-search for a valid position inside the room.

    Returns translated mesh at best valid position, or None.
    """
    MIN_SPACING = 0.10
    aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
    obj_half = (aabb_max - aabb_min) / 2
    obj_center = (aabb_min + aabb_max) / 2

    poly_bounds = inner_polygon.bounds  # (minx, miny, maxx, maxy)
    best_mesh = None
    best_dist = float('inf')

    for x in np.arange(poly_bounds[0] + obj_half[0],
                       poly_bounds[2] - obj_half[0], 0.25):
        for y in np.arange(poly_bounds[1] + obj_half[1],
                           poly_bounds[3] - obj_half[1], 0.25):
            candidate_center = np.array([x, y])
            offset = candidate_center - obj_center
            candidate = translate_mesh_xy(mesh, offset)
            candidate = clamp_mesh_to_polygon(candidate, inner_polygon)

            # Check nogo zones
            if _is_in_any_nogo(candidate, door_openings):
                continue

            # Check overlaps with other objects
            c_min, c_max = get_mesh_aabb_xy(candidate)
            has_overlap = False
            for a in assignments:
                if a['name'] == exclude_name:
                    continue
                a_min, a_max = get_mesh_aabb_xy(a['mesh'])
                if aabbs_overlap_xy_2d(
                    (c_min - MIN_SPACING, c_max + MIN_SPACING),
                    (a_min, a_max)):
                    has_overlap = True
                    break

            if has_overlap:
                continue

            dist = np.linalg.norm(offset)
            if dist < best_dist:
                best_dist = dist
                best_mesh = candidate

    return best_mesh


def force_separate_pair(assignments, i, j, inner_polygon, door_openings=None):
    """Force two objects apart along minimum overlap axis.

    Door-aware: never pushes objects into door nogo zones.
    If one direction would enter a nogo zone, the other object
    absorbs the full displacement. If both would, tries the
    perpendicular axis.
    """
    MIN_SPACING = 0.10

    a_i, a_j = assignments[i], assignments[j]
    aabb_i = get_mesh_aabb_xy(a_i['mesh'])
    aabb_j = get_mesh_aabb_xy(a_j['mesh'])

    center_i = (aabb_i[0] + aabb_i[1]) / 2
    center_j = (aabb_j[0] + aabb_j[1]) / 2

    overlap_x = (min(aabb_i[1][0], aabb_j[1][0]) -
                 max(aabb_i[0][0], aabb_j[0][0]))
    overlap_y = (min(aabb_i[1][1], aabb_j[1][1]) -
                 max(aabb_i[0][1], aabb_j[0][1]))

    if overlap_x <= 0 or overlap_y <= 0:
        return

    def _try_axis(overlap_val, axis_idx):
        """Try separating along given axis. Returns True if successful."""
        sign = 1.0 if (center_i[axis_idx] >= center_j[axis_idx]) else -1.0
        push_amount = MIN_SPACING / 2
        half = overlap_val / 2 + push_amount

        if axis_idx == 0:
            offset = np.array([sign * half, 0.0])
        else:
            offset = np.array([0.0, sign * half])

        # Try symmetric push (each moves half)
        mi = clamp_mesh_to_polygon(translate_mesh_xy(a_i['mesh'], offset), inner_polygon)
        mj = clamp_mesh_to_polygon(translate_mesh_xy(a_j['mesh'], -offset), inner_polygon)
        i_nogo = _is_in_any_nogo(mi, door_openings)
        j_nogo = _is_in_any_nogo(mj, door_openings)

        if not i_nogo and not j_nogo:
            a_i['mesh'] = mi
            a_j['mesh'] = mj
            return True

        # Asymmetric: only move the one that won't enter nogo
        if i_nogo and not j_nogo:
            mj_full = clamp_mesh_to_polygon(
                translate_mesh_xy(a_j['mesh'], -offset * 2), inner_polygon)
            if not _is_in_any_nogo(mj_full, door_openings):
                a_j['mesh'] = mj_full
                return True
        elif j_nogo and not i_nogo:
            mi_full = clamp_mesh_to_polygon(
                translate_mesh_xy(a_i['mesh'], offset * 2), inner_polygon)
            if not _is_in_any_nogo(mi_full, door_openings):
                a_i['mesh'] = mi_full
                return True

        return False

    # Try primary axis (minimum overlap), then secondary
    if overlap_x < overlap_y:
        if _try_axis(overlap_x, 0):
            return
        if _try_axis(overlap_y, 1):
            return
    else:
        if _try_axis(overlap_y, 1):
            return
        if _try_axis(overlap_x, 0):
            return

    # Fallback: symmetric push, but skip if it would push a safe object into nogo
    push_amount = MIN_SPACING / 2
    if overlap_x < overlap_y:
        sign = 1.0 if center_i[0] >= center_j[0] else -1.0
        offset = np.array([sign * (overlap_x / 2 + push_amount), 0.0])
    else:
        sign = 1.0 if center_i[1] >= center_j[1] else -1.0
        offset = np.array([0.0, sign * (overlap_y / 2 + push_amount)])

    mi_new = clamp_mesh_to_polygon(
        translate_mesh_xy(a_i['mesh'], offset), inner_polygon)
    mj_new = clamp_mesh_to_polygon(
        translate_mesh_xy(a_j['mesh'], -offset), inner_polygon)

    # Don't push a currently-safe object into a nogo zone
    i_was_safe = not _is_in_any_nogo(a_i['mesh'], door_openings)
    j_was_safe = not _is_in_any_nogo(a_j['mesh'], door_openings)
    i_would_nogo = _is_in_any_nogo(mi_new, door_openings)
    j_would_nogo = _is_in_any_nogo(mj_new, door_openings)

    if (i_was_safe and i_would_nogo) or (j_was_safe and j_would_nogo):
        return  # Skip: better to leave overlap than block a door

    a_i['mesh'] = mi_new
    a_j['mesh'] = mj_new


def _find_worst_overlap_pair(assignments):
    """Find the pair with the largest XY overlap area.

    Returns:
        (i, j) indices or None if no overlaps
    """
    worst_area = 0
    worst_pair = None

    for i in range(len(assignments)):
        aabb_i = get_mesh_aabb_xy(assignments[i]['mesh'])
        for j in range(i + 1, len(assignments)):
            aabb_j = get_mesh_aabb_xy(assignments[j]['mesh'])
            if not aabbs_overlap_xy_2d(aabb_i, aabb_j):
                continue
            ov_x = (min(aabb_i[1][0], aabb_j[1][0]) -
                    max(aabb_i[0][0], aabb_j[0][0]))
            ov_y = (min(aabb_i[1][1], aabb_j[1][1]) -
                    max(aabb_i[0][1], aabb_j[0][1]))
            area = ov_x * ov_y
            if area > worst_area:
                worst_area = area
                worst_pair = (i, j)

    return worst_pair


def compute_wall_affinity(mesh, room_polygon):
    """
    Determine which wall an object should be pushed toward based on its orientation.

    The object's "longer side" determines which pair of walls it faces:
    - Longer side along X → object faces +Y or -Y walls
    - Longer side along Y → object faces +X or -X walls

    Between the two candidate walls, the closer one is selected.

    Args:
        mesh: trimesh.Trimesh (axis-aligned after leveling)
        room_polygon: Shapely Polygon of room boundary

    Returns:
        dict with:
        - 'longer_axis': 'X' or 'Y'
        - 'target_wall': '+X', '-X', '+Y', or '-Y'
        - 'push_direction': np.array([dx, dy]) unit vector
        - 'distance_to_target_wall': float (for sorting)
    """
    aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
    width_x = aabb_max[0] - aabb_min[0]
    width_y = aabb_max[1] - aabb_min[1]
    center = (aabb_min + aabb_max) / 2

    # Get room bounds from polygon
    room_bounds = np.array(room_polygon.bounds)  # (minx, miny, maxx, maxy)
    room_min_x, room_min_y, room_max_x, room_max_y = room_bounds

    if width_x >= width_y:
        # Longer side along X → faces Y walls (perpendicular to short side)
        dist_plus_y = room_max_y - center[1]
        dist_minus_y = center[1] - room_min_y
        if dist_plus_y <= dist_minus_y:
            return {
                'longer_axis': 'X',
                'target_wall': '+Y',
                'push_direction': np.array([0.0, 1.0]),
                'distance_to_target_wall': dist_plus_y
            }
        else:
            return {
                'longer_axis': 'X',
                'target_wall': '-Y',
                'push_direction': np.array([0.0, -1.0]),
                'distance_to_target_wall': dist_minus_y
            }
    else:
        # Longer side along Y → faces X walls
        dist_plus_x = room_max_x - center[0]
        dist_minus_x = center[0] - room_min_x
        if dist_plus_x <= dist_minus_x:
            return {
                'longer_axis': 'Y',
                'target_wall': '+X',
                'push_direction': np.array([1.0, 0.0]),
                'distance_to_target_wall': dist_plus_x
            }
        else:
            return {
                'longer_axis': 'Y',
                'target_wall': '-X',
                'push_direction': np.array([-1.0, 0.0]),
                'distance_to_target_wall': dist_minus_x
            }


def is_mesh_inside_polygon(mesh, polygon):
    """Check if all AABB corners of mesh are inside the polygon."""
    aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
    corners = [
        (aabb_min[0], aabb_min[1]),
        (aabb_max[0], aabb_min[1]),
        (aabb_min[0], aabb_max[1]),
        (aabb_max[0], aabb_max[1])
    ]
    return all(polygon.contains(Point(c)) for c in corners)


def resolve_floor_conflicts(object_meshes, room_polygon, wall_thickness, max_iterations=100, verbose=False):
    """
    Resolve XY overlaps for floor-placed objects using "clear the middle" strategy.

    Strategy:
    1. When object A (smaller footprint) overlaps with blockers B, C:
       - First try to move B, C toward their preferred walls to make room for A
       - Preferred wall = wall that object's longer side faces AND is closer
       - Move objects closest to their wall first (to make space for others)
    2. Only if blockers can't move, move A toward its preferred wall
    3. Iterate until no overlaps or max_iterations

    Args:
        object_meshes: List of (name, mesh) tuples (already on floor)
        room_polygon: Shapely Polygon of room boundary
        wall_thickness: Wall thickness from scene JSON
        max_iterations: Maximum iterations
        verbose: Print progress

    Returns:
        List of (name, mesh) tuples with resolved positions
    """
    # Create inner polygon with margin from walls
    margin = 0.05  # 5cm margin
    inner_polygon = room_polygon.buffer(-(wall_thickness + margin))

    if inner_polygon.is_empty or not inner_polygon.is_valid:
        if verbose:
            print("      WARNING: Inner polygon invalid, using original room polygon")
        inner_polygon = room_polygon.buffer(-0.01)  # Minimal buffer

    # Work with mutable copies
    meshes = [(name, mesh.copy()) for name, mesh in object_meshes]
    epsilon = 0.01  # 1cm separation

    for iteration in range(max_iterations):
        # Build mesh_data with wall affinity for each object
        mesh_data = []
        for idx, (name, mesh) in enumerate(meshes):
            aabb = get_mesh_aabb_xy(mesh)
            affinity = compute_wall_affinity(mesh, room_polygon)
            mesh_data.append({
                'idx': idx,
                'name': name,
                'mesh': mesh,
                'aabb': aabb,
                'footprint': get_aabb_footprint(mesh),
                'affinity': affinity
            })

        # Find the smallest-footprint object that has overlaps ("incoming")
        # and collect ALL objects it overlaps with ("blockers")
        incoming = None
        incoming_idx = None
        blockers = []

        for i in range(len(mesh_data)):
            overlapping_indices = []
            for j in range(len(mesh_data)):
                if i != j and aabbs_overlap_xy_2d(mesh_data[i]['aabb'], mesh_data[j]['aabb']):
                    overlapping_indices.append(j)

            if overlapping_indices:
                # Check if this object should be the "incoming" (smaller footprint than at least one blocker)
                for j in overlapping_indices:
                    if mesh_data[i]['footprint'] < mesh_data[j]['footprint']:
                        # i is smaller than j, so i is the incoming object
                        if incoming is None or mesh_data[i]['footprint'] < incoming['footprint']:
                            incoming = mesh_data[i]
                            incoming_idx = i
                            blockers = [mesh_data[j] for j in overlapping_indices]
                        break

        if incoming is None:
            # No conflicts where incoming is smaller - check for any overlap
            for i in range(len(mesh_data)):
                for j in range(i + 1, len(mesh_data)):
                    if aabbs_overlap_xy_2d(mesh_data[i]['aabb'], mesh_data[j]['aabb']):
                        # Pick smaller as incoming
                        if mesh_data[i]['footprint'] <= mesh_data[j]['footprint']:
                            incoming = mesh_data[i]
                            incoming_idx = i
                            blockers = [mesh_data[j]]
                        else:
                            incoming = mesh_data[j]
                            incoming_idx = j
                            blockers = [mesh_data[i]]
                        break
                if incoming:
                    break

        if incoming is None:
            # No conflicts found
            if verbose:
                print(f"      No floor conflicts after {iteration + 1} iterations")
            break

        if verbose:
            blocker_names = [b['name'] for b in blockers]
            print(f"      Overlap: {incoming['name']} blocked by {blocker_names}")

        # Sort blockers by distance to their target wall (closest first)
        # Objects already near the wall move first to make space for others
        blockers.sort(key=lambda b: b['affinity']['distance_to_target_wall'])

        # PHASE 1: Try moving blockers toward their walls (closest to wall first)
        blocker_moved = False
        for blocker in blockers:
            push_dir = blocker['affinity']['push_direction']
            push_dist = compute_separation_push(incoming['aabb'], blocker['aabb'], push_dir)
            push_dist += epsilon  # Add clearance

            test_mesh = translate_mesh_xy(blocker['mesh'], push_dir * push_dist)

            # Check: inside room polygon?
            if not is_mesh_inside_polygon(test_mesh, inner_polygon):
                if verbose:
                    print(f"        {blocker['name']}: can't push to {blocker['affinity']['target_wall']} (hits wall)")
                continue

            # Check: no new collisions with other objects (except incoming)?
            causes_collision = False
            test_aabb = get_mesh_aabb_xy(test_mesh)
            for other in mesh_data:
                if other['idx'] == blocker['idx'] or other['idx'] == incoming_idx:
                    continue
                if aabbs_overlap_xy_2d(test_aabb, other['aabb']):
                    causes_collision = True
                    if verbose:
                        print(f"        {blocker['name']}: can't push (would hit {other['name']})")
                    break

            if not causes_collision:
                # Success! Move this blocker
                meshes[blocker['idx']] = (blocker['name'], test_mesh)
                if verbose:
                    print(f"        -> Moved {blocker['name']} toward {blocker['affinity']['target_wall']} "
                          f"by {push_dist:.3f}m to make room for {incoming['name']}")
                blocker_moved = True
                break

        if blocker_moved:
            continue  # Restart iteration - check if incoming is now clear

        # PHASE 2: No blocker could move - move incoming object toward ITS preferred wall
        if verbose:
            print(f"        No blocker could move, moving {incoming['name']} instead")

        push_dir = incoming['affinity']['push_direction']

        # Find minimum push to clear ALL remaining overlapping blockers
        max_push_dist = 0
        for blocker in blockers:
            dist = compute_separation_push(blocker['aabb'], incoming['aabb'], push_dir)
            max_push_dist = max(max_push_dist, dist)
        push_dist = max_push_dist + epsilon

        test_mesh = translate_mesh_xy(incoming['mesh'], push_dir * push_dist)

        if is_mesh_inside_polygon(test_mesh, inner_polygon):
            meshes[incoming_idx] = (incoming['name'], test_mesh)
            if verbose:
                print(f"        -> Moved {incoming['name']} toward {incoming['affinity']['target_wall']} "
                      f"by {push_dist:.3f}m")
        else:
            # Try perpendicular directions (+90°, -90°)
            perp_dirs = [
                np.array([-push_dir[1], push_dir[0]]),  # +90°
                np.array([push_dir[1], -push_dir[0]])   # -90°
            ]

            placed = False
            for perp_dir in perp_dirs:
                max_push_dist_perp = 0
                for blocker in blockers:
                    dist = compute_separation_push(blocker['aabb'], incoming['aabb'], perp_dir)
                    max_push_dist_perp = max(max_push_dist_perp, dist)
                push_dist_perp = max_push_dist_perp + epsilon

                test_mesh_perp = translate_mesh_xy(incoming['mesh'], perp_dir * push_dist_perp)

                if is_mesh_inside_polygon(test_mesh_perp, inner_polygon):
                    meshes[incoming_idx] = (incoming['name'], test_mesh_perp)
                    if verbose:
                        print(f"        -> Moved {incoming['name']} perpendicular by {push_dist_perp:.3f}m")
                    placed = True
                    break

            if not placed:
                # All directions blocked, clamp to room interior
                if verbose:
                    print(f"        -> All directions blocked, clamping {incoming['name']} to room interior")
                clamped = clamp_mesh_to_polygon(test_mesh, inner_polygon, verbose)
                meshes[incoming_idx] = (incoming['name'], clamped)

    else:
        if verbose:
            print(f"      WARNING: Max iterations ({max_iterations}) reached")

    # Final pass: ensure all objects are within inner polygon
    final_meshes = []
    for name, mesh in meshes:
        clamped = clamp_mesh_to_polygon(mesh, inner_polygon, verbose=False)
        if clamped is not mesh and verbose:
            print(f"      Final clamp: {name} adjusted to fit in room")
        final_meshes.append((name, clamped))

    return final_meshes


def get_dominant_direction_xy(center_a, center_b):
    """
    Compute the dominant XY direction from object A to object B.

    Returns the axis (0=X, 1=Y) and sign (+1 or -1) of the dominant direction.
    This is used to preserve spatial relationships when separating objects.

    Example: If B is primarily to the left (-X) of A, returns (0, -1)

    Args:
        center_a: Center of object A (3D array, only XY used)
        center_b: Center of object B (3D array, only XY used)

    Returns:
        Tuple of (axis_index, sign) where axis_index is 0 (X) or 1 (Y)
    """
    diff = center_b[:2] - center_a[:2]  # XY difference only

    # Find which axis has larger magnitude
    if abs(diff[0]) >= abs(diff[1]):
        # X is dominant
        return 0, (1 if diff[0] >= 0 else -1)
    else:
        # Y is dominant
        return 1, (1 if diff[1] >= 0 else -1)


def compute_aabb_overlap_xy(obb1, obb2):
    """
    Check if two OBBs overlap in XY plane (ignoring Z) and return penetration depths.

    After wall alignment + leveling, OBBs are axis-aligned, so we use AABB math.

    Args:
        obb1: dict from get_mesh_obb() with 'vertices'
        obb2: dict from get_mesh_obb() with 'vertices'

    Returns:
        None if no XY overlap, otherwise dict with:
        - 'overlap_x': penetration depth on X axis
        - 'overlap_y': penetration depth on Y axis
        - 'has_z_overlap': whether there's also Z overlap (true 3D collision)
    """
    # Get AABB from OBB vertices
    min1 = obb1['vertices'].min(axis=0)
    max1 = obb1['vertices'].max(axis=0)
    min2 = obb2['vertices'].min(axis=0)
    max2 = obb2['vertices'].max(axis=0)

    # Check XY overlap
    overlap_x = min(max1[0], max2[0]) - max(min1[0], min2[0])
    overlap_y = min(max1[1], max2[1]) - max(min1[1], min2[1])

    if overlap_x <= 0 or overlap_y <= 0:
        return None  # No XY overlap

    # Check Z overlap (for reporting purposes)
    overlap_z = min(max1[2], max2[2]) - max(min1[2], min2[2])

    return {
        'overlap_x': overlap_x,
        'overlap_y': overlap_y,
        'has_z_overlap': overlap_z > 0
    }


def is_within_room_bounds(mesh_obb, room_bounds_xy, margin=0.01):
    """
    Check if mesh OBB is within room XY bounds.

    Args:
        mesh_obb: OBB dict from get_mesh_obb()
        room_bounds_xy: Tuple of (min_xy, max_xy) from get_room_bounds_xy()
        margin: Small margin to allow objects near walls

    Returns:
        bool: True if object is within room bounds
    """
    min_xy = mesh_obb['vertices'][:, :2].min(axis=0)
    max_xy = mesh_obb['vertices'][:, :2].max(axis=0)

    room_min, room_max = room_bounds_xy

    return (min_xy[0] >= room_min[0] - margin and
            max_xy[0] <= room_max[0] + margin and
            min_xy[1] >= room_min[1] - margin and
            max_xy[1] <= room_max[1] + margin)


def clamp_to_room_bounds(mesh, room_bounds_xy, margin=0.01):
    """
    Translate mesh to be within room bounds if it's outside.

    Args:
        mesh: trimesh.Trimesh to clamp
        room_bounds_xy: Tuple of (min_xy, max_xy)
        margin: Margin from walls

    Returns:
        New mesh (translated if needed) or original if already inside
    """
    obb = get_mesh_obb(mesh)
    obj_min = obb['vertices'][:, :2].min(axis=0)
    obj_max = obb['vertices'][:, :2].max(axis=0)
    room_min, room_max = room_bounds_xy

    translation = np.array([0.0, 0.0])

    # Check and fix X bounds
    if obj_min[0] < room_min[0] - margin:
        translation[0] = (room_min[0] - margin) - obj_min[0]
    elif obj_max[0] > room_max[0] + margin:
        translation[0] = (room_max[0] + margin) - obj_max[0]

    # Check and fix Y bounds
    if obj_min[1] < room_min[1] - margin:
        translation[1] = (room_min[1] - margin) - obj_min[1]
    elif obj_max[1] > room_max[1] + margin:
        translation[1] = (room_max[1] + margin) - obj_max[1]

    if np.allclose(translation, 0):
        return mesh

    new_vertices = mesh.vertices.copy()
    new_vertices[:, 0] += translation[0]
    new_vertices[:, 1] += translation[1]

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def resolve_interpenetrations(object_meshes, room_bounds_xy, max_iterations=100, verbose=False):
    """
    Resolve all interpenetrations by moving objects apart on XY plane only.

    Strategy:
    1. For each overlapping pair (A, B), compute dominant direction from A to B
    2. Move B further along that dominant direction to separate (preserves "beside" relationships)
    3. If movement would push B outside room, try the other XY axis
    4. Clamp final positions to room bounds
    5. Iterate until no overlaps remain

    Args:
        object_meshes: List of (name, mesh) tuples
        room_bounds_xy: Tuple of (min_xy, max_xy) for room bounds
        max_iterations: Maximum iterations to prevent infinite loops
        verbose: Print progress

    Returns:
        List of (name, mesh) tuples with resolved positions
    """
    # Work with mutable copies
    meshes = [(name, mesh.copy()) for name, mesh in object_meshes]
    epsilon = 0.005  # 5mm separation gap

    for iteration in range(max_iterations):
        obbs = [(name, get_mesh_obb(mesh)) for name, mesh in meshes]

        found_overlap = False

        # Check all pairs for XY overlap (with Z overlap = true collision)
        for i in range(len(meshes)):
            for j in range(i + 1, len(meshes)):
                overlap = compute_aabb_overlap_xy(obbs[i][1], obbs[j][1])
                if overlap is None or not overlap['has_z_overlap']:
                    continue  # No true 3D collision

                found_overlap = True
                name_i, name_j = obbs[i][0], obbs[j][0]
                center_i = obbs[i][1]['center']
                center_j = obbs[j][1]['center']

                # Get dominant direction to preserve spatial relationship
                dom_axis, dom_sign = get_dominant_direction_xy(center_i, center_j)

                # Calculate movement amount along dominant axis
                if dom_axis == 0:
                    amount = overlap['overlap_x'] + epsilon
                else:
                    amount = overlap['overlap_y'] + epsilon

                # Try moving along dominant axis first
                test_vertices = meshes[j][1].vertices.copy()
                test_vertices[:, dom_axis] += dom_sign * amount

                test_mesh = trimesh.Trimesh(
                    vertices=test_vertices,
                    faces=meshes[j][1].faces.copy(),
                    process=False
                )
                test_obb = get_mesh_obb(test_mesh)

                # Check if this would push outside room bounds
                if is_within_room_bounds(test_obb, room_bounds_xy):
                    # Good - use this movement
                    axis_name = 'X' if dom_axis == 0 else 'Y'
                    if verbose:
                        print(f"      Overlap: {name_i} <-> {name_j}, "
                              f"moving {name_j} by {dom_sign * amount:.4f} on {axis_name} (dominant)")
                    _copy_visual(meshes[j][1], test_mesh)
                    meshes[j] = (name_j, test_mesh)
                else:
                    # Try the other axis
                    alt_axis = 1 - dom_axis
                    alt_sign = 1 if center_j[alt_axis] > center_i[alt_axis] else -1
                    alt_amount = (overlap['overlap_x'] if alt_axis == 0 else overlap['overlap_y']) + epsilon

                    test_vertices2 = meshes[j][1].vertices.copy()
                    test_vertices2[:, alt_axis] += alt_sign * alt_amount

                    test_mesh2 = trimesh.Trimesh(
                        vertices=test_vertices2,
                        faces=meshes[j][1].faces.copy(),
                        process=False
                    )
                    test_obb2 = get_mesh_obb(test_mesh2)

                    if is_within_room_bounds(test_obb2, room_bounds_xy):
                        axis_name = 'X' if alt_axis == 0 else 'Y'
                        if verbose:
                            print(f"      Overlap: {name_i} <-> {name_j}, "
                                  f"moving {name_j} by {alt_sign * alt_amount:.4f} on {axis_name} (alt, wall constraint)")
                        _copy_visual(meshes[j][1], test_mesh2)
                        meshes[j] = (name_j, test_mesh2)
                    else:
                        # Both axes would go outside - use dominant and clamp
                        if verbose:
                            print(f"      Overlap: {name_i} <-> {name_j}, "
                                  f"moving {name_j} on dominant axis + clamping to room")
                        _copy_visual(meshes[j][1], test_mesh)
                        clamped = clamp_to_room_bounds(test_mesh, room_bounds_xy)
                        meshes[j] = (name_j, clamped)

                # Restart pair checking after any move
                break
            if found_overlap:
                break

        if not found_overlap:
            if verbose:
                print(f"      No interpenetrations after {iteration + 1} iterations")
            break
    else:
        if verbose:
            print(f"      WARNING: Max iterations ({max_iterations}) reached")

    # Final pass: ensure all objects are within room bounds
    final_meshes = []
    for name, mesh in meshes:
        clamped = clamp_to_room_bounds(mesh, room_bounds_xy)
        if clamped is not mesh and verbose:
            print(f"      Clamped {name} to room bounds")
        final_meshes.append((name, clamped))

    return final_meshes


# =============================================================================
# Classification-Aware Multi-Phase Object Placement (Phase 5)
# =============================================================================


def load_room_openings(scene_json_path, room_id):
    """Load door and window openings from scene JSON.

    For each opening, computes:
    - 2D wall direction and inward normal
    - Opening center in 2D world coordinates
    - For doors: no-go zone polygon (1.0m + wall_thickness inward, width + 0.3m margin each side)
    - For windows: 3D z bounds (sill_height to sill_height + height)

    Returns:
        list of dicts with keys: type, wall_start, wall_end, position,
            width, height, sill_height, center_2d, inward_normal,
            wall_dir, z_min, z_max, nogo_polygon (for doors)
    """
    with open(scene_json_path) as f:
        scene = json.load(f)

    wall_thickness = scene.get('metadata', {}).get('wall_thickness', 0.15)

    # Find the room
    room = None
    for r in scene.get('rooms', []):
        if r.get('id') == room_id or r.get('name') == room_id:
            room = r
            break

    if room is None:
        return []

    floor_polygon = room.get('floor_polygon', [])
    openings_data = room.get('openings', [])
    result = []

    for opening in openings_data:
        opening_type = opening.get('type', 'door')
        wall_seg = opening.get('wall_segment', [])
        if len(wall_seg) != 2:
            continue

        position = opening.get('position', 0)
        width = opening.get('width', 0.9)
        height = opening.get('height', 2.1)
        sill_height = opening.get('sill_height', 0)

        wall_start = np.array(wall_seg[0], dtype=float)
        wall_end = np.array(wall_seg[1], dtype=float)

        # Check if wall_segment is reversed relative to polygon edge direction
        # (same logic as generate_scene.py find_openings_for_edge)
        reversed_dir = False
        n = len(floor_polygon)
        for i in range(n):
            edge_start = np.array(floor_polygon[i], dtype=float)
            edge_end = np.array(floor_polygon[(i + 1) % n], dtype=float)

            # Check forward match
            if (np.allclose(wall_start, edge_start, atol=0.01) and
                    np.allclose(wall_end, edge_end, atol=0.01)):
                break
            # Check reverse match
            if (np.allclose(wall_start, edge_end, atol=0.01) and
                    np.allclose(wall_end, edge_start, atol=0.01)):
                reversed_dir = True
                break

        # Compute wall direction
        wall_vec = wall_end - wall_start
        wall_length = np.linalg.norm(wall_vec)
        if wall_length < 1e-6:
            continue
        wall_dir = wall_vec / wall_length

        # Adjust position for reversed direction (same as generate_scene.py line 263-265)
        actual_position = position
        if reversed_dir:
            actual_position = wall_length - position - width

        # Inward normal: [-edge_dir_y, edge_dir_x] for CCW polygon
        # Must use polygon edge direction (not wall_segment direction)
        # When reversed, wall_dir is opposite to polygon edge dir, so negate
        if reversed_dir:
            inward_normal = np.array([wall_dir[1], -wall_dir[0]])
        else:
            inward_normal = np.array([-wall_dir[1], wall_dir[0]])

        # Opening center in 2D
        center_along_wall = actual_position + width / 2
        center_2d = wall_start + wall_dir * center_along_wall

        # Z bounds
        if opening_type == 'door':
            z_min = 0.0
            z_max = height
        else:  # window
            z_min = sill_height
            z_max = sill_height + height

        entry = {
            'type': opening_type,
            'wall_start': wall_start,
            'wall_end': wall_end,
            'wall_dir': wall_dir,
            'wall_length': wall_length,
            'position': actual_position,
            'width': width,
            'height': height,
            'sill_height': sill_height,
            'center_2d': center_2d,
            'inward_normal': inward_normal,
            'z_min': z_min,
            'z_max': z_max,
        }

        # For doors: compute no-go zone
        if opening_type == 'door':
            nogo_depth = 1.0 + wall_thickness
            nogo_half_width = width / 2 + 0.3

            c = center_2d
            w = wall_dir * nogo_half_width
            d = inward_normal * nogo_depth

            nogo_corners = [
                c - w,          # wall surface, left side
                c + w,          # wall surface, right side
                c + w + d,      # inward, right side
                c - w + d,      # inward, left side
            ]
            entry['nogo_polygon'] = Polygon(nogo_corners)

        result.append(entry)

    # Also check neighboring rooms for doors on shared walls
    # (a door may only be defined in the adjacent room's openings)
    n = len(floor_polygon)
    our_edges = []
    for i in range(n):
        our_edges.append((
            np.array(floor_polygon[i], dtype=float),
            np.array(floor_polygon[(i + 1) % n], dtype=float),
        ))

    for other_room in scene.get('rooms', []):
        if other_room.get('id') == room_id or other_room.get('name') == room_id:
            continue
        for opening in other_room.get('openings', []):
            if opening.get('type') != 'door':
                continue
            wall_seg = opening.get('wall_segment', [])
            if len(wall_seg) != 2:
                continue

            seg_start = np.array(wall_seg[0], dtype=float)
            seg_end = np.array(wall_seg[1], dtype=float)

            # Check if this door's wall segment matches one of our edges
            matched_edge = None
            for edge_start, edge_end in our_edges:
                if ((np.allclose(seg_start, edge_start, atol=0.01) and
                     np.allclose(seg_end, edge_end, atol=0.01)) or
                    (np.allclose(seg_start, edge_end, atol=0.01) and
                     np.allclose(seg_end, edge_start, atol=0.01))):
                    matched_edge = (edge_start, edge_end)
                    break

            if matched_edge is None:
                continue

            # Already have this door from our own openings?
            position = opening.get('position', 0)
            width = opening.get('width', 0.9)
            height = opening.get('height', 2.1)

            # Use OUR edge direction for consistent inward normal
            edge_vec = matched_edge[1] - matched_edge[0]
            edge_length = np.linalg.norm(edge_vec)
            if edge_length < 1e-6:
                continue
            edge_dir = edge_vec / edge_length

            # Inward normal for OUR room (CCW polygon convention)
            inward_normal = np.array([-edge_dir[1], edge_dir[0]])

            # Compute door center using the neighbor's wall_segment direction
            neighbor_vec = seg_end - seg_start
            neighbor_dir = neighbor_vec / np.linalg.norm(neighbor_vec)
            center_along_neighbor = position + width / 2
            center_2d = seg_start + neighbor_dir * center_along_neighbor

            # Check if we already have a door at this location
            duplicate = False
            for existing in result:
                if existing['type'] == 'door':
                    if np.linalg.norm(existing['center_2d'] - center_2d) < 0.5:
                        duplicate = True
                        break
            if duplicate:
                continue

            # Build nogo zone pointing into OUR room
            nogo_depth = 1.0 + wall_thickness
            nogo_half_width = width / 2 + 0.3
            w = edge_dir * nogo_half_width
            d = inward_normal * nogo_depth

            # Check which direction along our edge the center falls
            # (the neighbor's wall_dir may be reversed relative to ours)
            center_t = np.dot(center_2d - matched_edge[0], edge_dir)

            nogo_corners = [
                center_2d - w,
                center_2d + w,
                center_2d + w + d,
                center_2d - w + d,
            ]

            entry = {
                'type': 'door',
                'wall_start': matched_edge[0],
                'wall_end': matched_edge[1],
                'wall_dir': edge_dir,
                'wall_length': edge_length,
                'position': center_t - width / 2,
                'width': width,
                'height': height,
                'sill_height': 0,
                'center_2d': center_2d,
                'inward_normal': inward_normal,
                'z_min': 0.0,
                'z_max': height,
                'nogo_polygon': Polygon(nogo_corners),
                'from_neighbor': other_room.get('id', ''),
            }
            result.append(entry)

    return result


def compute_wall_edges(floor_polygon):
    """Extract wall edges from floor polygon vertices.

    Uses same convention as opening_visibility.py:get_wall_segments():
    inward_normal = [-edge_dir_y, edge_dir_x] for CCW polygon.

    Args:
        floor_polygon: Shapely Polygon of the room

    Returns:
        list of dicts with keys: start, end, direction, inward_normal,
            length, midpoint, edge_index
    """
    coords = list(floor_polygon.exterior.coords)[:-1]  # Remove closing duplicate
    n = len(coords)
    edges = []

    for i in range(n):
        start = np.array(coords[i], dtype=float)
        end = np.array(coords[(i + 1) % n], dtype=float)

        edge_vec = end - start
        length = np.linalg.norm(edge_vec)
        if length < 1e-6:
            continue

        direction = edge_vec / length
        inward_normal = np.array([-direction[1], direction[0]])

        edges.append({
            'start': start,
            'end': end,
            'direction': direction,
            'inward_normal': inward_normal,
            'length': length,
            'midpoint': (start + end) / 2,
            'edge_index': i,
        })

    return edges


def classify_object(name, label, mesh):
    """Classify object into placement category.

    Rules:
    - 'window': label contains "window" or "windows" (case-insensitive)
    - 'wall_art': label contains "painting", "tv", "television" OR label ends with "_w" suffix
    - 'ceiling': label ends with "_c" suffix (chandeliers, pendant lights, ceiling fans)
    - 'flat_floor': mesh height < 15% of max(width, depth) from AABB
    - 'floor_standing': everything else (default)

    Returns:
        str: 'window', 'wall_art', 'ceiling', 'flat_floor', or 'floor_standing'
    """
    label_lower = label.lower()

    # Window check
    if 'window' in label_lower:
        return 'window'

    # Wall art check
    if 'painting' in label_lower:
        return 'wall_art'
    if 'tv' in label_lower or 'television' in label_lower:
        return 'wall_art'
    if label_lower.endswith('_w'):
        return 'wall_art'

    # Ceiling check
    if label_lower.endswith('_c'):
        return 'ceiling'

    # Flat floor check: height < 15% of max(width, depth)
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    height = extents[2]  # Z extent
    max_xy = max(extents[0], extents[1])
    if max_xy > 0 and height < 0.15 * max_xy:
        return 'flat_floor'

    return 'floor_standing'


def _point_to_segment_distance(point, seg_start, seg_end):
    """Compute perpendicular distance from a 2D point to a line segment.

    Returns:
        (distance, projection_t) where projection_t is the parameter [0,1]
        along the segment for the closest point
    """
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-12:
        dist = np.linalg.norm(point - seg_start)
        return dist, 0.0

    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t = np.clip(t, 0, 1)

    closest = seg_start + t * seg_vec
    dist = np.linalg.norm(point - closest)
    return dist, t


def load_ceiling_height(scene_json_path, room_id):
    """Read ceiling_height from room definition in scene JSON.

    Returns:
        float: ceiling height (default 2.8 if not found)
    """
    with open(scene_json_path) as f:
        scene = json.load(f)

    default_height = scene.get('metadata', {}).get('default_ceiling_height', 2.8)

    for room in scene.get('rooms', []):
        if room.get('id') == room_id or room.get('name') == room_id:
            return room.get('ceiling_height', default_height)

    return default_height


def _find_nearest_wall(point_xy, wall_edges):
    """Find the nearest wall edge to a 2D point.

    Returns:
        (wall_edge_dict, distance, projection_t)
    """
    best_wall = None
    best_dist = float('inf')
    best_t = 0

    for edge in wall_edges:
        dist, t = _point_to_segment_distance(point_xy, edge['start'], edge['end'])
        if dist < best_dist:
            best_dist = dist
            best_wall = edge
            best_t = t

    return best_wall, best_dist, best_t


# ---- Phase A: Window Placement ----


def place_window_objects(window_objects, window_openings, wall_edges,
                         wall_thickness, verbose=False):
    """Place window object meshes into wall openings.

    For each window object, find the nearest unassigned window opening
    and transform the mesh to fit.

    Returns:
        Tuple of:
        - list of (name, placed_mesh, opening_dict) for successfully placed windows
        - list of opening dicts that have no window object assigned
    """
    if not window_openings:
        return [], []

    placed = []
    used_openings = set()

    for name, mesh in window_objects:
        # Find nearest unassigned window opening
        mesh_center = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2

        best_idx = None
        best_dist = float('inf')
        for idx, opening in enumerate(window_openings):
            if idx in used_openings:
                continue
            dist = np.linalg.norm(mesh_center - opening['center_2d'])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None:
            if verbose:
                print(f"    {name}: no unassigned window opening available, skipping")
            continue

        opening = window_openings[best_idx]
        used_openings.add(best_idx)

        # Transform mesh to fit in opening
        placed_mesh = _fit_mesh_to_opening(mesh, opening, wall_thickness)
        placed.append((name, placed_mesh, opening))

        if verbose:
            print(f"    {name}: placed in window opening at "
                  f"[{opening['center_2d'][0]:.2f}, {opening['center_2d'][1]:.2f}]")

    # Collect unmatched openings
    unmatched = [o for idx, o in enumerate(window_openings) if idx not in used_openings]

    return placed, unmatched


def _fit_mesh_to_opening(mesh, opening, wall_thickness):
    """Transform a mesh to fit inside a wall opening without shape distortion.

    Uses full OBB-based 3D alignment to correct both the depth direction AND
    any height tilt in the SAM3D mesh before applying anisotropic scaling.
    After proper alignment the wall and Z axes are independent, so scaling
    each independently fills the opening exactly without shearing faces.

    Steps:
    1. OBB: find height axis (most Z-aligned) and depth axis (thinnest horizontal)
    2. Full 3D rotation: height_axis → world Z, then depth_axis → inward_normal
    3. Anisotropic scale: wall extent → opening width, Z extent → opening height,
       normal extent → wall_thickness
    4. Position: bottom at z_min, centred on opening centre
    """
    wall_dir = opening['wall_dir']
    inward_normal = opening['inward_normal']

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2

    if extents[0] < 1e-6 or extents[1] < 1e-6 or extents[2] < 1e-6:
        return mesh

    # --- Step 1: OBB-based axis detection ---
    obb = mesh.bounding_box_oriented
    obb_axes = obb.primitive.transform[:3, :3]   # columns = OBB axis directions
    obb_extents = obb.primitive.extents

    # Height axis = OBB axis most aligned with world Z
    vert_idx = int(np.argmax(np.abs(obb_axes[2, :])))
    height_axis = obb_axes[:, vert_idx].copy()
    if height_axis[2] < 0:
        height_axis = -height_axis

    # Depth axis = thinnest of the two remaining (horizontal) OBB axes
    horiz = [i for i in range(3) if i != vert_idx]
    dep_idx = horiz[0] if obb_extents[horiz[0]] <= obb_extents[horiz[1]] else horiz[1]
    depth_axis = obb_axes[:, dep_idx].copy()
    inward_3d = np.array([inward_normal[0], inward_normal[1], 0.0])
    if np.dot(depth_axis, inward_3d) < 0:
        depth_axis = -depth_axis

    # --- Step 2: Full 3D rotation ---
    # 2a: Align height_axis → world Z (corrects any forward/backward tilt)
    R1, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [height_axis])

    # 2b: Rotate around Z to align depth axis with inward_normal
    depth_after_R1 = R1.apply(depth_axis)
    depth_xy = depth_after_R1[:2].copy()
    d_norm = np.linalg.norm(depth_xy)
    if d_norm > 1e-4:
        depth_xy /= d_norm
        angle_2d = (np.arctan2(inward_normal[1], inward_normal[0]) -
                    np.arctan2(depth_xy[1], depth_xy[0]))
        R2 = Rotation.from_euler('z', angle_2d)
    else:
        R2 = Rotation.identity()

    R_mat = (R2 * R1).as_matrix()
    verts = (mesh.vertices - center) @ R_mat.T

    # --- Step 3: Decompose and measure aligned extents ---
    xy = verts[:, :2]
    along_wall   = xy @ wall_dir
    along_normal = xy @ inward_normal
    z_vals       = verts[:, 2]

    cur_wall   = along_wall.max()   - along_wall.min()
    cur_normal = along_normal.max() - along_normal.min()
    cur_z      = z_vals.max()       - z_vals.min()

    if cur_wall < 1e-6 or cur_z < 1e-6:
        return mesh

    # --- Step 4: Anisotropic scale to fill opening ---
    sc_wall   = opening['width']  / cur_wall
    sc_z      = opening['height'] / cur_z
    sc_normal = wall_thickness / cur_normal if cur_normal > 1e-6 else 1.0

    ws = along_wall   * sc_wall
    ns = along_normal * sc_normal
    zs = z_vals       * sc_z

    new_verts = np.empty_like(verts)
    new_verts[:, 0] = ws * wall_dir[0] + ns * inward_normal[0]
    new_verts[:, 1] = ws * wall_dir[1] + ns * inward_normal[1]
    new_verts[:, 2] = zs

    # --- Step 5: Position at opening ---
    v_min = new_verts.min(axis=0)
    v_max = new_verts.max(axis=0)
    v_center_xy = (v_min[:2] + v_max[:2]) / 2

    target_xy = opening['center_2d'] + inward_normal * (wall_thickness / 2)
    new_verts[:, 0] += target_xy[0] - v_center_xy[0]
    new_verts[:, 1] += target_xy[1] - v_center_xy[1]
    new_verts[:, 2] += opening['z_min'] - v_min[2]

    new_mesh = trimesh.Trimesh(
        vertices=new_verts, faces=mesh.faces.copy(), process=False
    )
    _copy_visual(mesh, new_mesh)
    return new_mesh


def _fit_mesh_to_opening_simple(mesh, opening, wall_thickness):
    """Orient a window mesh to the wall and scale uniformly — simple mode.

    Unlike _fit_mesh_to_opening, this does NOT reshape the window to fill
    the opening exactly. Instead it:
    1. Applies the same full OBB-based 3D orientation (height → Z,
       depth → inward_normal) to correct SAM3D orientation noise.
    2. Scales uniformly in the wall × Z plane to fit *within* the opening
       while preserving the window's aspect ratio (opening adjusts to window).
    3. Scales depth independently to wall_thickness.
    4. Centers the window in the opening.
    """
    wall_dir = opening['wall_dir']
    inward_normal = opening['inward_normal']

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2

    if extents[0] < 1e-6 or extents[1] < 1e-6 or extents[2] < 1e-6:
        return mesh

    # OBB-based orientation (identical to _fit_mesh_to_opening)
    obb = mesh.bounding_box_oriented
    obb_axes = obb.primitive.transform[:3, :3]
    obb_extents = obb.primitive.extents

    vert_idx = int(np.argmax(np.abs(obb_axes[2, :])))
    height_axis = obb_axes[:, vert_idx].copy()
    if height_axis[2] < 0:
        height_axis = -height_axis

    horiz = [i for i in range(3) if i != vert_idx]
    dep_idx = horiz[0] if obb_extents[horiz[0]] <= obb_extents[horiz[1]] else horiz[1]
    depth_axis = obb_axes[:, dep_idx].copy()
    if np.dot(depth_axis, np.array([inward_normal[0], inward_normal[1], 0.0])) < 0:
        depth_axis = -depth_axis

    R1, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [height_axis])
    depth_after_R1 = R1.apply(depth_axis)
    depth_xy = depth_after_R1[:2].copy()
    d_norm = np.linalg.norm(depth_xy)
    if d_norm > 1e-4:
        depth_xy /= d_norm
        angle_2d = (np.arctan2(inward_normal[1], inward_normal[0]) -
                    np.arctan2(depth_xy[1], depth_xy[0]))
        R2 = Rotation.from_euler('z', angle_2d)
    else:
        R2 = Rotation.identity()

    R_mat = (R2 * R1).as_matrix()
    verts = (mesh.vertices - center) @ R_mat.T

    xy = verts[:, :2]
    along_wall   = xy @ wall_dir
    along_normal = xy @ inward_normal
    z_vals       = verts[:, 2]

    cur_wall   = along_wall.max()   - along_wall.min()
    cur_normal = along_normal.max() - along_normal.min()
    cur_z      = z_vals.max()       - z_vals.min()

    if cur_wall < 1e-6 or cur_z < 1e-6:
        return mesh

    # Uniform scale in visible plane: fit within opening, preserve aspect ratio
    sc_face   = min(opening['width'] / cur_wall, opening['height'] / cur_z)
    sc_normal = wall_thickness / cur_normal if cur_normal > 1e-6 else 1.0

    ws = along_wall   * sc_face
    ns = along_normal * sc_normal
    zs = z_vals       * sc_face

    new_verts = np.empty_like(verts)
    new_verts[:, 0] = ws * wall_dir[0] + ns * inward_normal[0]
    new_verts[:, 1] = ws * wall_dir[1] + ns * inward_normal[1]
    new_verts[:, 2] = zs

    # Centre in the opening (both XY and Z)
    v_min = new_verts.min(axis=0)
    v_max = new_verts.max(axis=0)
    v_center_xy = (v_min[:2] + v_max[:2]) / 2
    v_center_z  = (v_min[2]  + v_max[2])  / 2

    target_xy       = opening['center_2d'] + inward_normal * (wall_thickness / 2)
    target_z_center = (opening['z_min'] + opening['z_max']) / 2

    new_verts[:, 0] += target_xy[0]       - v_center_xy[0]
    new_verts[:, 1] += target_xy[1]       - v_center_xy[1]
    new_verts[:, 2] += target_z_center    - v_center_z

    out = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces.copy(), process=False)
    _copy_visual(mesh, out)
    return out


def fit_layout_to_room(objects_world, room_polygon, wall_thickness):
    """Scale the XY layout of objects outward so they fill the room without
    wall interpenetration.

    Finds the maximum uniform scale S around the room centroid such that every
    object's actual XY footprint (convex hull of vertex projections) stays
    within the room polygon minus a small wall clearance.  Because we maximise
    S, objects are pushed as far outward as possible — clearing the centre —
    while preserving the relative layout shape from SAM3D.

    Only centroid positions are translated; meshes are not individually scaled
    and orientations are unchanged.  Windows are excluded (placed at wall
    openings regardless of SAM3D position).
    """
    from shapely.geometry import MultiPoint
    from shapely.affinity import translate as shapely_translate

    # Buffer inward by wall_thickness (room polygon outer edge = wall outer face)
    # plus 5 cm clearance from the inner wall surface.
    margin = wall_thickness + 0.05
    inner_polygon = room_polygon.buffer(-margin)
    if inner_polygon.is_empty or not inner_polygon.is_valid:
        inner_polygon = room_polygon.buffer(-0.05)

    room_centroid = np.array(inner_polygon.centroid.coords[0])

    # Build centroid list and precompute centered footprint polygons,
    # excluding windows (placed at openings) and ceiling objects (not floor-placed).
    indices, centroids_xy, footprints = [], [], []
    for i, (name, mesh, mesh_raw) in enumerate(objects_world):
        if classify_object_by_name(name) in ('window', 'ceiling'):
            continue
        b = mesh.bounds
        c = (b[0][:2] + b[1][:2]) / 2
        # XY convex hull of all vertices, centered at origin for reuse
        pts_xy = mesh.vertices[:, :2]
        try:
            hull = MultiPoint(pts_xy).convex_hull
            fp_centered = shapely_translate(hull, -float(c[0]), -float(c[1]))
        except Exception:
            # Fallback: axis-aligned box from half-extents
            from shapely.geometry import box as shbox
            he = (b[1][:2] - b[0][:2]) / 2
            fp_centered = shbox(-he[0], -he[1], he[0], he[1])
        indices.append(i)
        centroids_xy.append(c)
        footprints.append(fp_centered)

    if not centroids_xy:
        return objects_world

    centroids_xy = np.array(centroids_xy)
    group_center = centroids_xy.mean(axis=0)
    centered = centroids_xy - group_center

    def all_fit(S):
        for k, fp in enumerate(footprints):
            new_c = room_centroid + S * centered[k]
            placed = shapely_translate(fp, float(new_c[0]), float(new_c[1]))
            if not inner_polygon.covers(placed):
                return False
        return True

    if not all_fit(0.0):
        print("    fit_layout_to_room: individual objects exceed room — leaving unchanged")
        return objects_world

    # Binary search over [0, 4] — allow up to 4× spread so tightly packed
    # SAM3D clusters can be pushed well toward the walls.
    lo, hi = 0.0, 4.0
    for _ in range(32):
        mid = (lo + hi) / 2
        if all_fit(mid):
            lo = mid
        else:
            hi = mid
    S = lo

    print(f"    Layout fit: scale={S:.3f}, "
          f"group_center=[{group_center[0]:.2f},{group_center[1]:.2f}] → "
          f"room_centroid=[{room_centroid[0]:.2f},{room_centroid[1]:.2f}]")

    # Translate each mesh by the XY delta between old and new centroid
    delta_map = {}
    for k, i in enumerate(indices):
        new_c_xy = room_centroid + S * centered[k]
        delta_map[i] = np.array([new_c_xy[0] - centroids_xy[k][0],
                                  new_c_xy[1] - centroids_xy[k][1],
                                  0.0])

    result = []
    for i, (name, mesh, mesh_raw) in enumerate(objects_world):
        if i not in delta_map:
            result.append((name, mesh, mesh_raw))
            continue
        delta = delta_map[i]
        if np.linalg.norm(delta[:2]) < 1e-4:
            result.append((name, mesh, mesh_raw))
            continue
        new_verts = mesh.vertices + delta
        new_mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces.copy(),
                                   process=False)
        _copy_visual(mesh, new_mesh)
        result.append((name, new_mesh, mesh_raw))

    return result


def declutter_objects(objects_world, room_polygon, wall_thickness):
    """Resolve object interpenetrations and push objects toward walls.

    Only activates when XY footprints overlap after fit_layout_to_room.
    Two phases:
      Phase 1 – pairwise repulsion: push overlapping pairs apart along their
                centroid-to-centroid axis until separation is achieved, while
                respecting the room boundary.
      Phase 2 – wall push: for each object (outermost first) binary-search the
                maximum outward translation that keeps it within the room and
                avoids creating new overlaps.

    Ceiling objects and windows are excluded entirely (neither moved nor used
    as blockers in the floor-object overlap checks).
    """
    from shapely.geometry import MultiPoint
    from shapely.affinity import translate as shp_translate
    from shapely.geometry import box as shbox

    # Same margin as fit_layout_to_room: wall outer face + 5 cm clearance.
    margin = wall_thickness + 0.05
    inner_polygon = room_polygon.buffer(-margin)
    if inner_polygon.is_empty or not inner_polygon.is_valid:
        inner_polygon = room_polygon.buffer(-0.05)
    room_centroid = np.array(inner_polygon.centroid.coords[0])

    # ── Build footprints for moveable (floor/flat/wall_art) objects ──────────
    moveable_pos = {}   # obj_index → current XY centroid [x, y]
    fp_at_origin = {}   # obj_index → shapely polygon centred at origin

    for i, (name, mesh, _mr) in enumerate(objects_world):
        cat = classify_object_by_name(name)
        if cat in ('ceiling', 'window'):
            continue
        b = mesh.bounds
        c = (b[0][:2] + b[1][:2]) / 2
        pts = mesh.vertices[:, :2]
        try:
            hull = MultiPoint(pts).convex_hull.buffer(0.02)   # 2 cm padding
            fp = shp_translate(hull, -float(c[0]), -float(c[1]))
        except Exception:
            he = (b[1][:2] - b[0][:2]) / 2
            fp = shbox(-he[0]-0.02, -he[1]-0.02, he[0]+0.02, he[1]+0.02)
        moveable_pos[i] = c.copy()
        fp_at_origin[i] = fp

    def placed(i):
        return shp_translate(fp_at_origin[i],
                              float(moveable_pos[i][0]),
                              float(moveable_pos[i][1]))

    def overlapping_pairs():
        """Return list of (area, i, j) for pairs with overlap > 1 cm²."""
        idxs = list(moveable_pos.keys())
        out = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ii, jj = idxs[a], idxs[b]
                fpi, fpj = placed(ii), placed(jj)
                if fpi.intersects(fpj):
                    inter = fpi.intersection(fpj)
                    if inter.area > 0.0001:
                        out.append((inter.area, ii, jj))
        return sorted(out, reverse=True)

    initial = overlapping_pairs()
    if not initial:
        return objects_world   # not cluttered – nothing to do

    print(f"    Declutter: {len(initial)} overlapping pair(s) detected")

    # ── Phase 1: pairwise repulsion ──────────────────────────────────────────
    step = 0.04   # push increment per iteration (4 cm)
    for _iter in range(400):
        pairs = overlapping_pairs()
        if not pairs:
            break
        moved = False
        for _area, ii, jj in pairs:
            sep = moveable_pos[ii] - moveable_pos[jj]
            dist = np.linalg.norm(sep)
            if dist < 0.01:
                angle = np.random.uniform(0, 2 * np.pi)
                sep = np.array([np.cos(angle), np.sin(angle)])
            else:
                sep /= dist
            # Push both objects apart in equal and opposite directions
            for idx, direction in [(ii, sep), (jj, -sep)]:
                new_pos = moveable_pos[idx] + direction * step
                if inner_polygon.covers(shp_translate(fp_at_origin[idx], *new_pos)):
                    moveable_pos[idx] = new_pos
                    moved = True
                else:
                    # Slide along boundary: try half-step
                    for t in (0.5, 0.25, 0.1):
                        np2 = moveable_pos[idx] + direction * step * t
                        if inner_polygon.covers(shp_translate(fp_at_origin[idx], *np2)):
                            moveable_pos[idx] = np2
                            moved = True
                            break
        if not moved:
            break

    remaining = len(overlapping_pairs())
    print(f"    Declutter: {remaining} overlap(s) remaining after repulsion")

    # ── Phase 2: push toward walls — only if centre is still cluttered ───────
    # "Cluttered centre": more than 40 % of moveable objects have their centroid
    # within the inner 45 % of the room's half-dimension from the room centroid.
    bounds = inner_polygon.bounds   # (minx, miny, maxx, maxy)
    room_half_dim = min(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2
    center_radius = room_half_dim * 0.45
    n_central = sum(
        1 for i in moveable_pos
        if np.linalg.norm(moveable_pos[i] - room_centroid) < center_radius
    )
    center_cluttered = n_central > max(1, 0.4 * len(moveable_pos))

    if not center_cluttered:
        print(f"    Declutter: centre clear ({n_central}/{len(moveable_pos)} objects "
              f"within {center_radius:.2f} m), skipping wall push")
    else:
        print(f"    Declutter: centre cluttered ({n_central}/{len(moveable_pos)} objects "
              f"within {center_radius:.2f} m), pushing toward walls")

    # Process outermost objects first so they claim their wall spots before
    # inner objects try to fill remaining space.
    sorted_by_r = sorted(
        moveable_pos.keys(),
        key=lambda i: -np.linalg.norm(moveable_pos[i] - room_centroid)
    )

    for ii in sorted_by_r:
        if not center_cluttered:
            break
        dir_out = moveable_pos[ii] - room_centroid
        r = np.linalg.norm(dir_out)
        if r < 0.05:
            continue   # at room centre – no clear wall direction
        dir_out /= r

        lo, hi = 0.0, 5.0
        for _ in range(28):
            mid = (lo + hi) / 2
            new_pos = moveable_pos[ii] + dir_out * mid
            fp_new = shp_translate(fp_at_origin[ii], *new_pos)
            if not inner_polygon.covers(fp_new):
                hi = mid
                continue
            blocked = any(
                fp_new.intersects(placed(jj)) and
                fp_new.intersection(placed(jj)).area > 0.001
                for jj in moveable_pos if jj != ii
            )
            if blocked:
                hi = mid
            else:
                lo = mid

        if lo > 0.005:
            moveable_pos[ii] = moveable_pos[ii] + dir_out * lo

    # ── Apply updated positions to meshes ────────────────────────────────────
    result = []
    for i, (name, mesh, mesh_raw) in enumerate(objects_world):
        if i not in moveable_pos:
            result.append((name, mesh, mesh_raw))
            continue
        b = mesh.bounds
        old_c = (b[0][:2] + b[1][:2]) / 2
        delta = np.array([moveable_pos[i][0] - old_c[0],
                          moveable_pos[i][1] - old_c[1],
                          0.0])
        if np.linalg.norm(delta[:2]) > 1e-4:
            new_verts = mesh.vertices + delta
            new_mesh = trimesh.Trimesh(vertices=new_verts,
                                       faces=mesh.faces.copy(),
                                       process=False)
            _copy_visual(mesh, new_mesh)
            result.append((name, new_mesh, mesh_raw))
        else:
            result.append((name, mesh, mesh_raw))

    return result


def clone_window_for_opening(source_mesh, source_opening, target_opening,
                             wall_thickness):
    """Scale and reposition a window mesh clone to fit a different opening.

    Uses anisotropic scaling: independently scales along wall direction,
    normal direction, and Z to fill the target opening completely.
    """
    mesh_copy = source_mesh.copy()

    src_bounds = source_mesh.bounds
    src_extents = src_bounds[1] - src_bounds[0]
    src_center = (src_bounds[0] + src_bounds[1]) / 2

    if src_extents[2] < 1e-6:
        return mesh_copy

    new_vertices = mesh_copy.vertices.copy()
    new_vertices = new_vertices - src_center

    # Rotate to align with target wall
    src_normal = source_opening['inward_normal']
    tgt_normal = target_opening['inward_normal']

    src_angle = np.arctan2(src_normal[1], src_normal[0])
    tgt_angle = np.arctan2(tgt_normal[1], tgt_normal[0])
    rotation = tgt_angle - src_angle

    cos_a, sin_a = np.cos(rotation), np.sin(rotation)
    rot_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    new_vertices = new_vertices @ rot_z.T

    # Anisotropic scaling in target wall's coordinate system
    tgt_wall_dir = target_opening['wall_dir']
    xy = new_vertices[:, :2]
    along_wall = xy @ tgt_wall_dir
    along_normal = xy @ tgt_normal
    z_vals = new_vertices[:, 2]

    cur_wall_ext = along_wall.max() - along_wall.min()
    cur_normal_ext = along_normal.max() - along_normal.min()
    cur_z_ext = z_vals.max() - z_vals.min()

    scale_wall = (target_opening['width'] / cur_wall_ext
                  if cur_wall_ext > 1e-6 else 1.0)
    scale_z = (target_opening['height'] / cur_z_ext
               if cur_z_ext > 1e-6 else 1.0)
    scale_normal = (wall_thickness / cur_normal_ext
                    if cur_normal_ext > 1e-6 else 1.0)

    along_wall_scaled = along_wall * scale_wall
    along_normal_scaled = along_normal * scale_normal
    z_scaled = z_vals * scale_z

    new_vertices[:, 0] = (along_wall_scaled * tgt_wall_dir[0] +
                           along_normal_scaled * tgt_normal[0])
    new_vertices[:, 1] = (along_wall_scaled * tgt_wall_dir[1] +
                           along_normal_scaled * tgt_normal[1])
    new_vertices[:, 2] = z_scaled

    # Position at target opening
    target_xy = target_opening['center_2d'] + tgt_normal * (wall_thickness / 2)

    v_min = new_vertices.min(axis=0)
    v_max = new_vertices.max(axis=0)
    v_center = (v_min + v_max) / 2

    new_vertices[:, 0] += target_xy[0] - v_center[0]
    new_vertices[:, 1] += target_xy[1] - v_center[1]
    new_vertices[:, 2] += target_opening['z_min'] - v_min[2]

    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh_copy.faces.copy(),
        process=False
    )
    _copy_visual(source_mesh, new_mesh)
    return new_mesh


def clone_windows_for_unmatched(placed_windows, unmatched_openings, wall_edges,
                                wall_thickness, scene_json_path, room_id,
                                verbose=False):
    """Clone windows for unmatched openings.

    Uses windows from this room first. If none available, skips.

    Returns:
        list of (name, placed_mesh) for cloned windows
    """
    if not unmatched_openings:
        return []

    if not placed_windows:
        if verbose:
            print(f"    No windows to clone for {len(unmatched_openings)} "
                  "unmatched openings")
        return []

    cloned = []
    # Use the first placed window as source
    source_name, source_mesh, source_opening = placed_windows[0]

    for idx, target_opening in enumerate(unmatched_openings):
        cloned_mesh = clone_window_for_opening(
            source_mesh, source_opening, target_opening, wall_thickness
        )
        if room_id:
            clone_name = f"cloned_window_{room_id}_{idx:02d}"
        else:
            clone_name = f"cloned_window_{idx:02d}"
        cloned.append((clone_name, cloned_mesh))

        if verbose:
            print(f"    Cloned window for opening at "
                  f"[{target_opening['center_2d'][0]:.2f}, "
                  f"{target_opening['center_2d'][1]:.2f}]")

    return cloned


# ---- Phase B: Floor-Standing Object Placement ----


def is_object_between(obj_center, wall_edge, other_objects):
    """Check if any other object's AABB intersects the perpendicular path
    from obj_center to the wall.

    Args:
        obj_center: [x, y] center of the object in question
        wall_edge: dict with 'start', 'end', 'inward_normal'
        other_objects: list of (name, mesh) tuples

    Returns:
        bool: True if another object is between this object and the wall
    """
    # Project object center onto the wall
    dist, t = _point_to_segment_distance(
        obj_center, wall_edge['start'], wall_edge['end']
    )
    wall_point = wall_edge['start'] + t * (wall_edge['end'] - wall_edge['start'])

    # Create a narrow box from object center to wall
    ray_start = obj_center
    ray_end = wall_point

    ray_min_x = min(ray_start[0], ray_end[0]) - 0.05
    ray_max_x = max(ray_start[0], ray_end[0]) + 0.05
    ray_min_y = min(ray_start[1], ray_end[1]) - 0.05
    ray_max_y = max(ray_start[1], ray_end[1]) + 0.05

    for name, mesh in other_objects:
        other_min, other_max = get_mesh_aabb_xy(mesh)
        other_center = (other_min + other_max) / 2

        # Skip if same object (by position)
        if np.linalg.norm(other_center - obj_center) < 0.01:
            continue

        # Check AABB overlap with ray box
        if (other_max[0] > ray_min_x and other_min[0] < ray_max_x and
                other_max[1] > ray_min_y and other_min[1] < ray_max_y):
            # Verify the other object is actually between (closer to wall)
            other_dist, _ = _point_to_segment_distance(
                other_center, wall_edge['start'], wall_edge['end']
            )
            if other_dist < dist:
                return True

    return False


def count_objects_between(obj_center, wall_edge, other_objects, obj_name):
    """Count how many other objects' AABBs intersect the perpendicular path
    from obj_center to the wall.

    Like is_object_between() but returns the count instead of bool,
    and identifies the source object by name instead of position.
    """
    dist, t = _point_to_segment_distance(
        obj_center, wall_edge['start'], wall_edge['end']
    )
    wall_point = wall_edge['start'] + t * (wall_edge['end'] - wall_edge['start'])

    ray_min_x = min(obj_center[0], wall_point[0]) - 0.05
    ray_max_x = max(obj_center[0], wall_point[0]) + 0.05
    ray_min_y = min(obj_center[1], wall_point[1]) - 0.05
    ray_max_y = max(obj_center[1], wall_point[1]) + 0.05

    count = 0
    for name, mesh in other_objects:
        if name == obj_name:
            continue
        other_min, other_max = get_mesh_aabb_xy(mesh)
        other_center = (other_min + other_max) / 2
        if (other_max[0] > ray_min_x and other_min[0] < ray_max_x and
                other_max[1] > ray_min_y and other_min[1] < ray_max_y):
            other_dist, _ = _point_to_segment_distance(
                other_center, wall_edge['start'], wall_edge['end']
            )
            if other_dist < dist:
                count += 1
    return count


def compute_wall_capacities(wall_edges, openings):
    """Compute available placement capacity for each wall.

    For each wall, determines:
    - Blocked intervals from openings (with margin each side)
    - Free intervals (wall length minus openings minus edge margins)
    - Total available length

    Args:
        wall_edges: list of wall edge dicts from compute_wall_edges()
        openings: list of opening dicts from load_room_openings()

    Returns:
        dict: edge_index -> {wall_edge, total_length, available_length,
              free_intervals, blocked_intervals}
    """
    EDGE_MARGIN = 0.10       # 10cm from wall corners
    OPENING_MARGIN = 0.15    # 15cm each side of doors/windows

    capacities = {}

    for edge in wall_edges:
        idx = edge['edge_index']
        wall_length = edge['length']

        # Find openings on this wall
        wall_openings = _get_openings_on_wall(edge, openings)

        # Build blocked intervals (along wall parameter in [0, wall_length])
        blocked = []
        for opening in wall_openings:
            # Project opening center onto wall
            _, t = _point_to_segment_distance(
                opening['center_2d'], edge['start'], edge['end']
            )
            center_along_wall = t * wall_length
            half_width = opening['width'] / 2
            block_start = center_along_wall - half_width - OPENING_MARGIN
            block_end = center_along_wall + half_width + OPENING_MARGIN
            blocked.append((max(0, block_start), min(wall_length, block_end)))

        # Sort and merge overlapping blocked intervals
        blocked.sort()
        merged_blocked = []
        for start, end in blocked:
            if merged_blocked and start <= merged_blocked[-1][1]:
                merged_blocked[-1] = (merged_blocked[-1][0],
                                      max(merged_blocked[-1][1], end))
            else:
                merged_blocked.append((start, end))

        # Compute free intervals (accounting for edge margins)
        free_intervals = []
        cursor = EDGE_MARGIN
        for block_start, block_end in merged_blocked:
            if cursor < block_start:
                free_intervals.append((cursor, block_start))
            cursor = max(cursor, block_end)
        if cursor < wall_length - EDGE_MARGIN:
            free_intervals.append((cursor, wall_length - EDGE_MARGIN))

        # Handle case where there are no blocked intervals
        if not merged_blocked and wall_length > 2 * EDGE_MARGIN:
            free_intervals = [(EDGE_MARGIN, wall_length - EDGE_MARGIN)]

        available_length = sum(end - start for start, end in free_intervals)

        capacities[idx] = {
            'wall_edge': edge,
            'total_length': wall_length,
            'available_length': available_length,
            'free_intervals': free_intervals,
            'blocked_intervals': merged_blocked,
        }

    return capacities


def assign_objects_to_walls_nearest(objects, wall_edges, wall_capacities,
                                    room_polygon, verbose=False):
    """Nearest-wall assignment with capacity overflow eviction.

    Each object is assigned to its single nearest wall (perpendicular distance).
    Then over-capacity walls evict farthest objects to center until all walls
    are within 90% of available length.

    This preserves reconstruction positions: objects near a wall stay on that
    wall, and only the farthest objects get evicted.

    Args:
        objects: list of (name, mesh) tuples
        wall_edges: list of wall edge dicts
        wall_capacities: dict from compute_wall_capacities()
        room_polygon: Shapely Polygon
        verbose: print debug info

    Returns:
        list of dicts with keys: name, mesh, wall_edge (or None),
            wall_idx (or None), is_center
    """
    WALL_CAPACITY_USAGE = 0.90  # Use 90% of available wall length

    if not wall_edges or not objects:
        return [{'name': n, 'mesh': m, 'wall_edge': None, 'wall_idx': None,
                 'is_center': True}
                for n, m in objects]

    MAX_WALL_ASSIGN_DIST = 1.5  # Don't assign objects >1.5m from any wall

    # Step 1: Compute nearest wall + distance for all objects
    raw_assignments = []
    for obj_idx, (name, mesh) in enumerate(objects):
        aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
        center = (aabb_min + aabb_max) / 2

        best_wall_idx = None
        best_dist = float('inf')

        for edge in wall_edges:
            wall_idx = edge['edge_index']
            if wall_idx not in wall_capacities:
                continue
            dist, _ = _point_to_segment_distance(
                center, edge['start'], edge['end']
            )
            if dist < best_dist:
                best_dist = dist
                best_wall_idx = wall_idx

        raw_assignments.append({
            'name': name, 'mesh': mesh,
            'center': center, 'best_wall_idx': best_wall_idx,
            'best_dist': best_dist,
        })

    # Step 1b: Apply distance threshold + path blocking
    other_objects = [(name, mesh) for name, mesh in objects]
    assignments = []
    for ra in raw_assignments:
        # Check 1: distance threshold
        if ra['best_dist'] > MAX_WALL_ASSIGN_DIST:
            assignments.append({
                'name': ra['name'], 'mesh': ra['mesh'],
                'wall_edge': None, 'wall_idx': None,
                'is_center': True,
                '_distance': ra['best_dist'],
            })
            if verbose:
                print(f"      {ra['name']}: too far from wall "
                      f"({ra['best_dist']:.2f}m > {MAX_WALL_ASSIGN_DIST}m) -> center")
            continue

        # Check 2: count objects between this one and the wall
        if ra['best_wall_idx'] is not None:
            edge = wall_capacities[ra['best_wall_idx']]['wall_edge']
            blocking_count = count_objects_between(
                ra['center'], edge, other_objects, ra['name'])
            if blocking_count >= 2:
                assignments.append({
                    'name': ra['name'], 'mesh': ra['mesh'],
                    'wall_edge': None, 'wall_idx': None,
                    'is_center': True,
                    '_distance': ra['best_dist'],
                })
                if verbose:
                    print(f"      {ra['name']}: {blocking_count} objects blocking "
                          f"path to wall -> center")
                continue

        # Normal wall assignment
        if ra['best_wall_idx'] is not None:
            edge = wall_capacities[ra['best_wall_idx']]['wall_edge']
            assignments.append({
                'name': ra['name'], 'mesh': ra['mesh'],
                'wall_edge': edge, 'wall_idx': ra['best_wall_idx'],
                'is_center': False,
                '_distance': ra['best_dist'],
            })
        else:
            assignments.append({
                'name': ra['name'], 'mesh': ra['mesh'],
                'wall_edge': None, 'wall_idx': None,
                'is_center': True,
                '_distance': float('inf'),
            })

    # Step 2: Compute object widths along their assigned wall
    for a in assignments:
        if a['is_center'] or a['wall_idx'] is None:
            a['_obj_width'] = 0.0
            continue
        wall_dir = a['wall_edge']['direction']
        aabb_min, aabb_max = get_mesh_aabb_xy(a['mesh'])
        corners = np.array([
            [aabb_min[0], aabb_min[1]],
            [aabb_max[0], aabb_min[1]],
            [aabb_min[0], aabb_max[1]],
            [aabb_max[0], aabb_max[1]]
        ])
        projections = corners @ wall_dir
        a['_obj_width'] = projections.max() - projections.min()

    # Step 3: Check capacity per wall, evict farthest objects until within limit
    max_eviction_rounds = len(objects)  # safety limit
    for _ in range(max_eviction_rounds):
        # Compute used capacity per wall
        wall_used = {}
        for a in assignments:
            if a['is_center'] or a['wall_idx'] is None:
                continue
            widx = a['wall_idx']
            wall_used[widx] = wall_used.get(widx, 0.0) + a['_obj_width']

        # Find an over-capacity wall
        over_wall = None
        for widx, used in wall_used.items():
            cap = wall_capacities[widx]['available_length'] * WALL_CAPACITY_USAGE
            if used > cap:
                over_wall = widx
                break

        if over_wall is None:
            break  # All walls within capacity

        # Evict the farthest-from-wall object on this wall
        farthest_idx = None
        farthest_dist = -1.0
        for idx, a in enumerate(assignments):
            if a['wall_idx'] == over_wall and not a['is_center']:
                if a['_distance'] > farthest_dist:
                    farthest_dist = a['_distance']
                    farthest_idx = idx

        if farthest_idx is not None:
            evicted = assignments[farthest_idx]

            # Try perpendicular walls before falling back to center
            placed_on_perp = False
            perp_walls = find_perpendicular_walls(
                over_wall, wall_edges, wall_capacities)
            for perp_idx in perp_walls:
                perp_cap = wall_capacities[perp_idx]
                # Check remaining capacity on perpendicular wall
                perp_used = sum(
                    a['_obj_width'] for a in assignments
                    if a['wall_idx'] == perp_idx and not a['is_center'])
                perp_avail = (perp_cap['available_length'] *
                              WALL_CAPACITY_USAGE - perp_used)
                if perp_avail >= evicted['_obj_width']:
                    # Reassign to perpendicular wall
                    perp_edge = perp_cap['wall_edge']
                    evicted['wall_edge'] = perp_edge
                    evicted['wall_idx'] = perp_idx
                    evicted['is_center'] = False
                    if verbose:
                        print(f"      Evicting {evicted['name']} from wall "
                              f"#{over_wall} -> perpendicular wall "
                              f"#{perp_idx}")
                    placed_on_perp = True
                    break

            if not placed_on_perp:
                if verbose:
                    print(f"      Evicting {evicted['name']} from wall "
                          f"#{over_wall} (dist={farthest_dist:.2f}m) "
                          f"-> center")
                evicted['is_center'] = True
                evicted['wall_edge'] = None
                evicted['wall_idx'] = None
        else:
            break  # Shouldn't happen, but safety

    # Clean up temp keys
    for a in assignments:
        a.pop('_distance', None)
        a.pop('_obj_width', None)

    return assignments


def _get_openings_on_wall(wall_edge, openings, tolerance=0.3):
    """Find all openings that are on a specific wall edge."""
    result = []
    for opening in openings:
        dist, t = _point_to_segment_distance(
            opening['center_2d'], wall_edge['start'], wall_edge['end']
        )
        if dist < tolerance and 0 <= t <= 1:
            result.append(opening)
    return result


def push_objects_to_walls(assignments, inner_polygon, wall_thickness,
                          verbose=False):
    """Push wall objects perpendicular to their assigned wall. Drop all to floor.

    Wall objects:
    - Push perpendicular to wall surface (keep reconstruction position along wall)
    - Drop to floor

    Center objects:
    - Drop to floor
    - Clamp to room polygon

    Args:
        assignments: list of assignment dicts from assign_objects_to_walls_nearest
        inner_polygon: Shapely Polygon (room inset)
        wall_thickness: wall thickness in meters
        verbose: print debug info
    """
    WALL_SURFACE_GAP = 0.02  # 2cm gap between object back and wall

    for a in assignments:
        mesh = a['mesh']

        if a['is_center'] or a['wall_edge'] is None:
            # Center object: drop to floor, clamp
            mesh = drop_mesh_to_z(mesh, 0.0)
            mesh = clamp_mesh_to_polygon(mesh, inner_polygon)
            a['mesh'] = mesh
            if verbose:
                center = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2
                print(f"      {a['name']}: center, dropped to floor at "
                      f"[{center[0]:.2f}, {center[1]:.2f}]")
            continue

        wall_edge = a['wall_edge']

        inward = wall_edge['inward_normal']

        # Push to wall surface (perpendicular only, no along-wall movement)
        # "back" = side closest to wall (smallest inward distance from wall line)
        aabb_min, aabb_max = get_mesh_aabb_xy(mesh)
        corners_xy = np.array([
            [aabb_min[0], aabb_min[1]],
            [aabb_max[0], aabb_min[1]],
            [aabb_min[0], aabb_max[1]],
            [aabb_max[0], aabb_max[1]]
        ])
        inward_dists = (corners_xy - wall_edge['start']) @ inward
        min_inward_dist = inward_dists.min()  # back of object

        wall_margin = wall_thickness + WALL_SURFACE_GAP
        push_dist = min_inward_dist - wall_margin
        offset_to_wall = -inward * push_dist

        mesh = translate_mesh_xy(mesh, offset_to_wall)

        # Drop to floor
        mesh = drop_mesh_to_z(mesh, 0.0)

        # Clamp to inner polygon (safety)
        mesh = clamp_mesh_to_polygon(mesh, inner_polygon)

        a['mesh'] = mesh
        if verbose:
            new_center = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2
            print(f"      {a['name']}: wall #{a['wall_idx']}, "
                  f"pushed to [{new_center[0]:.2f}, {new_center[1]:.2f}]")


def _get_wall_distance(assignment):
    """Helper: get perpendicular distance of object to its assigned wall."""
    if assignment['is_center'] or assignment['wall_edge'] is None:
        return float('inf')
    aabb_min, aabb_max = get_mesh_aabb_xy(assignment['mesh'])
    center = (aabb_min + aabb_max) / 2
    wall_edge = assignment['wall_edge']
    dist, _ = _point_to_segment_distance(
        center, wall_edge['start'], wall_edge['end']
    )
    return dist


def resolve_overlaps_simple(assignments, inner_polygon, verbose=False,
                            door_openings=None):
    """Pure repulsion overlap resolution with guaranteed zero overlaps.

    All objects (wall and center) get free 2D movement to resolve overlaps.
    No wall-parallel constraint — objects may drift slightly from walls.
    If still overlapping after MAX_ITERATIONS, demote farthest-from-wall
    object in overlapping pair to center and retry. Stall detection
    triggers force_separate_pair on worst pair. Final guarantee sweep
    ensures zero remaining overlaps.

    Args:
        assignments: list of assignment dicts
        inner_polygon: Shapely Polygon (room inset)
        verbose: print debug info

    Returns:
        Updated assignments with resolved positions
    """
    OVERLAP_DAMPING = 0.6
    MAX_OVERLAP_STEP = 0.20      # 20cm max displacement per iteration
    MAX_OVERLAP_ITERATIONS = 80
    MAX_DEMOTIONS = 5
    MIN_OBJECT_SPACING = 0.10    # 10cm minimum gap between objects
    FORCE_SEPARATE_THRESHOLD = 3  # Stall iterations before force-separate

    if len(assignments) < 2:
        return assignments

    for demotion_round in range(MAX_DEMOTIONS + 1):
        converged = False
        prev_overlap_count = float('inf')
        stall_counter = 0

        for iteration in range(MAX_OVERLAP_ITERATIONS):
            displacements = {a['name']: np.array([0.0, 0.0])
                             for a in assignments}
            overlap_count = 0

            # Check all pairs
            for i in range(len(assignments)):
                a_i = assignments[i]
                aabb_i = get_mesh_aabb_xy(a_i['mesh'])
                center_i = (aabb_i[0] + aabb_i[1]) / 2

                for j in range(i + 1, len(assignments)):
                    a_j = assignments[j]
                    aabb_j = get_mesh_aabb_xy(a_j['mesh'])

                    if not aabbs_overlap_xy_2d(aabb_i, aabb_j):
                        continue

                    overlap_count += 1
                    center_j = (aabb_j[0] + aabb_j[1]) / 2

                    overlap_x = (min(aabb_i[1][0], aabb_j[1][0]) -
                                 max(aabb_i[0][0], aabb_j[0][0]))
                    overlap_y = (min(aabb_i[1][1], aabb_j[1][1]) -
                                 max(aabb_i[0][1], aabb_j[0][1]))

                    # Separation direction: shorter overlap axis
                    if overlap_x < overlap_y:
                        sign = 1.0 if center_i[0] >= center_j[0] else -1.0
                        sep = np.array([
                            sign * (overlap_x / 2 + MIN_OBJECT_SPACING),
                            0.0])
                    else:
                        sign = 1.0 if center_i[1] >= center_j[1] else -1.0
                        sep = np.array([
                            0.0,
                            sign * (overlap_y / 2 + MIN_OBJECT_SPACING)])

                    displacements[a_i['name']] += sep * OVERLAP_DAMPING
                    displacements[a_j['name']] -= sep * OVERLAP_DAMPING

            if overlap_count == 0:
                if verbose:
                    print(f"      Overlap resolved at iteration {iteration}"
                          f" (demotion round {demotion_round})")
                converged = True
                break

            # Stall detection: if overlap count doesn't decrease
            if overlap_count >= prev_overlap_count:
                stall_counter += 1
            else:
                stall_counter = 0
            prev_overlap_count = overlap_count

            # If stalled, force-separate the worst pair
            if stall_counter >= FORCE_SEPARATE_THRESHOLD:
                worst = _find_worst_overlap_pair(assignments)
                if worst is not None:
                    if verbose:
                        print(f"      Stall detected at iteration "
                              f"{iteration}, force-separating "
                              f"{assignments[worst[0]]['name']} <-> "
                              f"{assignments[worst[1]]['name']}")
                    force_separate_pair(
                        assignments, worst[0], worst[1], inner_polygon,
                        door_openings)
                stall_counter = 0
                continue  # Skip normal displacement this iteration

            # Apply displacements — free 2D movement for all objects
            for a in assignments:
                d = displacements[a['name']]
                if np.linalg.norm(d) < 1e-4:
                    continue

                # Clamp magnitude
                mag = np.linalg.norm(d)
                if mag > MAX_OVERLAP_STEP:
                    d = d * (MAX_OVERLAP_STEP / mag)

                new_mesh = translate_mesh_xy(a['mesh'], d)
                new_mesh = clamp_mesh_to_polygon(new_mesh, inner_polygon)
                a['mesh'] = new_mesh

        if converged:
            break

        if demotion_round >= MAX_DEMOTIONS:
            if verbose:
                print(f"      WARNING: {MAX_DEMOTIONS} demotions exhausted")
            break

        # Find worst overlapping pair and demote the one farther from wall
        worst_pair = _find_worst_overlap_pair(assignments)

        if worst_pair is None:
            break

        # Demote a wall object to center (demoting center is a no-op)
        i, j = worst_pair
        a_i, a_j = assignments[i], assignments[j]

        # Prefer demoting the wall object (center demotion is meaningless)
        if not a_i['is_center'] and a_j['is_center']:
            demote_idx = i
        elif not a_j['is_center'] and a_i['is_center']:
            demote_idx = j
        elif not a_i['is_center'] and not a_j['is_center']:
            # Both wall objects: demote the one farther from its wall
            dist_i = _get_wall_distance(assignments[i])
            dist_j = _get_wall_distance(assignments[j])
            demote_idx = i if dist_i >= dist_j else j
        else:
            # Both center: nothing useful to demote, break
            if verbose:
                print(f"      Both {a_i['name']} and {a_j['name']} are "
                      "center objects, cannot demote further")
            break

        demoted = assignments[demote_idx]
        if verbose:
            print(f"      Demoting {demoted['name']} from wall to center "
                  f"(demotion {demotion_round + 1})")
        demoted['is_center'] = True
        demoted['wall_edge'] = None
        demoted['wall_idx'] = None

    # Final guarantee sweep: force-separate any remaining overlaps
    for sweep in range(20):
        worst = _find_worst_overlap_pair(assignments)
        if worst is None:
            break
        if verbose:
            print(f"      Final sweep {sweep}: force-separating "
                  f"{assignments[worst[0]]['name']} <-> "
                  f"{assignments[worst[1]]['name']}")
        force_separate_pair(assignments, worst[0], worst[1], inner_polygon,
                            door_openings)

    return assignments


def place_floor_objects(objects, wall_edges, openings, room_polygon,
                        wall_thickness, camera_forward_xy=None,
                        camera_position_xy=None, verbose=False):
    """Place floor-standing objects preserving reconstruction layout.

    Pipeline B1-B7:
    B1: Compute wall capacities
    B2: Nearest-wall assignment + capacity overflow (perpendicular wall first)
    B3: Push perpendicular to wall + drop to floor
    B3.5: Spread objects along wall (camera-depth ordered)
    B4: Door avoidance (BEFORE overlap resolution)
    B5: Overlap resolution (guaranteed zero overlaps, 10cm min gap)
    B5.5: Reclamp wall objects that drifted
    B6: Door avoidance (second pass, after overlap resolution)
    B6.5: Light overlap resolution (if B6 moved anything)
    B7: Final clamp + verify zero overlaps + verify no door blocks

    Returns:
        list of (name, placed_mesh)
    """
    if not objects:
        return []

    inner_margin = wall_thickness + 0.02
    inner_polygon = room_polygon.buffer(-inner_margin)
    if inner_polygon.is_empty or not inner_polygon.is_valid:
        inner_polygon = room_polygon.buffer(-0.01)

    door_openings = [o for o in openings if o['type'] == 'door']

    # B1: Compute wall capacities
    if verbose:
        print("    B1: Computing wall capacities...")
    wall_capacities = compute_wall_capacities(wall_edges, openings)
    for idx, cap in wall_capacities.items():
        if verbose:
            print(f"      wall #{idx}: total={cap['total_length']:.2f}m, "
                  f"available={cap['available_length']:.2f}m, "
                  f"free intervals={len(cap['free_intervals'])}")

    # B2: Nearest-wall assignment + capacity overflow
    if verbose:
        print("    B2: Assigning objects to nearest walls...")
    assignments = assign_objects_to_walls_nearest(
        objects, wall_edges, wall_capacities, room_polygon, verbose=verbose
    )
    for a in assignments:
        if verbose:
            wall_info = (f"wall #{a['wall_idx']}"
                         if a['wall_idx'] is not None else "center")
            print(f"      {a['name']}: {wall_info} "
                  f"(center={a['is_center']})")

    # B3: Push wall objects perpendicular to wall + drop to floor
    if verbose:
        print("    B3: Pushing objects to walls...")
    push_objects_to_walls(assignments, inner_polygon, wall_thickness,
                          verbose=verbose)

    # B3.5: Spread objects along wall (camera-depth ordered)
    if verbose:
        print("    B3.5: Spreading objects along walls...")
    wall_indices_with_objects = set(
        a['wall_idx'] for a in assignments
        if a['wall_idx'] is not None and not a['is_center'])
    for widx in wall_indices_with_objects:
        spread_objects_along_wall(
            assignments, widx, wall_capacities,
            camera_position_xy, camera_forward_xy,
            verbose=verbose)

    # B4: Door avoidance (before overlap resolution)
    if verbose:
        print("    B4: Door avoidance (first pass)...")
    avoid_doors(assignments, door_openings, inner_polygon, verbose=verbose)

    # B5: Overlap resolution (guaranteed zero overlaps, 10cm min gap)
    if verbose:
        print("    B5: Resolving overlaps (free 2D repulsion)...")
    assignments = resolve_overlaps_simple(assignments, inner_polygon,
                                         verbose=verbose,
                                         door_openings=door_openings)

    # B5.5: Reclamp wall objects that drifted
    if verbose:
        print("    B5.5: Reclamping drifted wall objects...")
    reclamp_wall_objects(assignments, wall_thickness, verbose=verbose)

    # B6: Door avoidance (second pass, after overlap resolution)
    if verbose:
        print("    B6: Door avoidance (second pass)...")
    moved = avoid_doors(assignments, door_openings, inner_polygon,
                        verbose=verbose)

    # B6.5: Light overlap resolution if B6 moved anything
    if moved > 0:
        if verbose:
            print(f"    B6.5: Light overlap resolution "
                  f"({moved} objects moved by B6)...")
        assignments = resolve_overlaps_simple(assignments, inner_polygon,
                                             verbose=verbose,
                                             door_openings=door_openings)

    # B7: Final clamp + fix remaining overlaps + fix remaining door blocks
    if verbose:
        print("    B7: Final clamp + verify...")

    # 1. Clamp all inside room
    for a in assignments:
        a['mesh'] = clamp_mesh_to_polygon(a['mesh'], inner_polygon)

    # 2. Fix overlaps (door-aware — can't push into doors)
    for sweep in range(20):
        worst = _find_worst_overlap_pair(assignments)
        if worst is None:
            break
        if verbose:
            print(f"      B7 fix overlap: force-separating "
                  f"{assignments[worst[0]]['name']} <-> "
                  f"{assignments[worst[1]]['name']}")
        force_separate_pair(assignments, worst[0], worst[1],
                            inner_polygon, door_openings)

    # 3. Final door avoidance pass
    avoid_doors(assignments, door_openings, inner_polygon, verbose=verbose)

    # 4. Fix overlaps introduced by door avoidance (door-aware)
    for sweep in range(20):
        worst = _find_worst_overlap_pair(assignments)
        if worst is None:
            break
        if verbose:
            print(f"      B7 post-door overlap fix: "
                  f"{assignments[worst[0]]['name']} <-> "
                  f"{assignments[worst[1]]['name']}")
        force_separate_pair(assignments, worst[0], worst[1],
                            inner_polygon, door_openings)

    # 5. Grid-search relocate any objects still blocking doors
    for door in door_openings:
        if 'nogo_polygon' not in door:
            continue
        nogo = door['nogo_polygon']
        for a in assignments:
            aabb_min, aabb_max = get_mesh_aabb_xy(a['mesh'])
            obj_box = box(aabb_min[0], aabb_min[1],
                          aabb_max[0], aabb_max[1])
            if obj_box.intersects(nogo):
                # Try inward push first (fast)
                inward = door['inward_normal']
                cleared = False
                for push_dist in [0.5, 1.0, 1.5, 2.0, 2.5]:
                    test = translate_mesh_xy(a['mesh'], inward * push_dist)
                    test = clamp_mesh_to_polygon(test, inner_polygon)
                    if not _is_in_any_nogo(test, door_openings):
                        a['mesh'] = test
                        a['is_center'] = True
                        a['wall_edge'] = None
                        a['wall_idx'] = None
                        cleared = True
                        if verbose:
                            print(f"      {a['name']}: pushed {push_dist:.1f}m "
                                  "inward to clear door")
                        break

                # Grid search (thorough)
                if not cleared:
                    relocated = _find_valid_position(
                        a['mesh'], assignments, a['name'],
                        inner_polygon, door_openings)
                    if relocated is not None:
                        a['mesh'] = relocated
                        a['is_center'] = True
                        a['wall_edge'] = None
                        a['wall_idx'] = None
                        cleared = True
                        if verbose:
                            print(f"      {a['name']}: grid-search relocated "
                                  "to clear door")

                if not cleared and verbose:
                    print(f"      WARNING: {a['name']} still blocking "
                          "door after all attempts")

    # 6. Final overlap fix after relocations (door-aware)
    for sweep in range(20):
        worst = _find_worst_overlap_pair(assignments)
        if worst is None:
            break
        force_separate_pair(assignments, worst[0], worst[1],
                            inner_polygon, door_openings)

    # 7. Safety clamp
    for a in assignments:
        a['mesh'] = clamp_mesh_to_polygon(a['mesh'], inner_polygon)

    # 7.5. Final door clearing (no overlap resolution after — avoids cycles)
    avoid_doors(assignments, door_openings, inner_polygon, verbose=verbose)
    for door in door_openings:
        if 'nogo_polygon' not in door:
            continue
        nogo = door['nogo_polygon']
        for a in assignments:
            aabb_min, aabb_max = get_mesh_aabb_xy(a['mesh'])
            obj_box = box(aabb_min[0], aabb_min[1],
                          aabb_max[0], aabb_max[1])
            if obj_box.intersects(nogo):
                inward = door['inward_normal']
                cleared = False
                for push_dist in [0.5, 1.0, 1.5, 2.0, 2.5]:
                    test = translate_mesh_xy(a['mesh'], inward * push_dist)
                    test = clamp_mesh_to_polygon(test, inner_polygon)
                    if not _is_in_any_nogo(test, door_openings):
                        a['mesh'] = test
                        a['is_center'] = True
                        a['wall_edge'] = None
                        a['wall_idx'] = None
                        cleared = True
                        if verbose:
                            print(f"      B7.5: {a['name']}: pushed "
                                  f"{push_dist:.1f}m inward to clear door")
                        break
                if not cleared:
                    relocated = _find_valid_position(
                        a['mesh'], assignments, a['name'],
                        inner_polygon, door_openings)
                    if relocated is not None:
                        a['mesh'] = relocated
                        a['is_center'] = True
                        a['wall_edge'] = None
                        a['wall_idx'] = None
                        if verbose:
                            print(f"      B7.5: {a['name']}: grid-search "
                                  "relocated to clear door")

    # 8. Verification
    remaining_overlap = _find_worst_overlap_pair(assignments)
    if remaining_overlap is not None:
        i, j = remaining_overlap
        if verbose:
            print(f"      WARNING: {assignments[i]['name']} and "
                  f"{assignments[j]['name']} still overlap after placement")

    for door in door_openings:
        if 'nogo_polygon' not in door:
            continue
        nogo = door['nogo_polygon']
        for a in assignments:
            aabb_min, aabb_max = get_mesh_aabb_xy(a['mesh'])
            obj_box = box(aabb_min[0], aabb_min[1],
                          aabb_max[0], aabb_max[1])
            if obj_box.intersects(nogo):
                if verbose:
                    print(f"      WARNING: {a['name']} still blocking "
                          "door after all attempts")

    final = []
    for a in assignments:
        final.append((a['name'], a['mesh']))

    return final


# ---- Phase C: Wall Art Placement ----


def place_wall_art(paintings, wall_edges, floor_objects, room_polygon,
                   wall_thickness, ceiling_height, openings, verbose=False):
    """Mount paintings/wall art on walls at appropriate heights.

    Algorithm:
    1. For each painting, find nearest wall
    2. Rotate so thinnest AABB dimension faces the wall (depth axis)
    3. Position against wall surface
    4. Mount at eye level (1.6m center), adjusted to clear furniture below
    5. Don't overlap with openings

    Returns:
        list of (name, placed_mesh)
    """
    if not paintings:
        return []

    door_openings = [o for o in openings if o['type'] == 'door']

    placed = []

    for name, mesh in paintings:
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        center = (bounds[0] + bounds[1]) / 2

        # Check if all 3 dimensions are similar (within 2x) -> floor instead
        # BUT: skip for explicit _w objects — user intent overrides shape heuristic
        sorted_ext = sorted(extents)
        is_explicit_wall = name.lower().rstrip().endswith('_w')
        if not is_explicit_wall and sorted_ext[2] < 2 * sorted_ext[0] and sorted_ext[0] > 0.01:
            mesh = drop_mesh_to_z(mesh, 0.0)
            inner = room_polygon.buffer(-(wall_thickness + 0.02))
            if not inner.is_empty:
                mesh = clamp_mesh_to_polygon(mesh, inner)
            placed.append((name, mesh))
            if verbose:
                print(f"    {name}: ambiguous shape, placed on floor")
            continue

        # Find nearest wall
        center_xy = center[:2]
        nearest_wall, dist, t = _find_nearest_wall(center_xy, wall_edges)

        if nearest_wall is None:
            mesh = drop_mesh_to_z(mesh, 0.0)
            placed.append((name, mesh))
            continue

        wall_dir = nearest_wall['direction']
        inward = nearest_wall['inward_normal']

        # Determine depth axis: thinnest AABB dimension in XY
        min_ext_idx = np.argmin(extents[:2])  # Only check X and Y
        if min_ext_idx == 2:
            # Z is thinnest (unusual for a painting) - pick thinner of X,Y
            min_ext_idx = 0 if extents[0] <= extents[1] else 1

        if min_ext_idx == 0:
            mesh_depth_dir = np.array([1.0, 0.0])
        else:
            mesh_depth_dir = np.array([0.0, 1.0])

        # Rotation to align depth with inward normal
        current_angle = np.arctan2(mesh_depth_dir[1], mesh_depth_dir[0])
        target_angle = np.arctan2(inward[1], inward[0])
        rotation = target_angle - current_angle

        cos_a, sin_a = np.cos(rotation), np.sin(rotation)
        rot_z = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])

        new_vertices = mesh.vertices.copy()
        new_vertices = (new_vertices - center) @ rot_z.T + center

        # Recompute bounds after rotation
        r_min = new_vertices.min(axis=0)
        r_max = new_vertices.max(axis=0)
        r_center = (r_min + r_max) / 2
        r_extents = r_max - r_min

        # Position along wall (keep projected position from reconstruction)
        proj_t = np.dot(r_center[:2] - nearest_wall['start'], wall_dir)

        # Compute art width along wall
        corners_2d = np.array([
            [r_min[0], r_min[1]],
            [r_max[0], r_min[1]],
            [r_min[0], r_max[1]],
            [r_max[0], r_max[1]]
        ])
        wall_projections = (corners_2d - nearest_wall['start']) @ wall_dir
        art_half_width = (wall_projections.max() - wall_projections.min()) / 2

        # Clamp along wall
        proj_t = np.clip(proj_t, art_half_width + 0.05,
                         nearest_wall['length'] - art_half_width - 0.05)

        wall_pos = nearest_wall['start'] + wall_dir * proj_t

        # Depth extent along inward normal
        corner_normal_dists = (corners_2d - nearest_wall['start']) @ inward
        depth_extent = corner_normal_dists.max() - corner_normal_dists.min()

        # Position center so back face touches wall surface
        inner_margin = wall_thickness + 0.02
        target_xy = wall_pos + inward * (inner_margin + depth_extent / 2)

        offset_x = target_xy[0] - r_center[0]
        offset_y = target_xy[1] - r_center[1]
        new_vertices[:, 0] += offset_x
        new_vertices[:, 1] += offset_y

        # Determine mounting height
        default_center_height = 1.6
        painting_half_height = r_extents[2] / 2

        # Check for furniture below this wall position
        max_furniture_top = 0.0
        for fname, fmesh in floor_objects:
            f_min, f_max = get_mesh_aabb_xy(fmesh)
            test_min = np.array([target_xy[0] - art_half_width,
                                 target_xy[1] - art_half_width])
            test_max = np.array([target_xy[0] + art_half_width,
                                 target_xy[1] + art_half_width])

            if (f_max[0] > test_min[0] and f_min[0] < test_max[0] and
                    f_max[1] > test_min[1] and f_min[1] < test_max[1]):
                furniture_top = fmesh.bounds[1][2]
                max_furniture_top = max(max_furniture_top, furniture_top)

        # Ensure bottom edge is at least 10cm above furniture
        min_bottom_z = max_furniture_top + 0.10

        target_center_z = default_center_height
        if target_center_z - painting_half_height < min_bottom_z:
            target_center_z = min_bottom_z + painting_half_height

        # Clamp top edge to ceiling - 10cm
        max_top_z = ceiling_height - 0.10
        if target_center_z + painting_half_height > max_top_z:
            target_center_z = max_top_z - painting_half_height

        offset_z = target_center_z - r_center[2]
        new_vertices[:, 2] += offset_z

        # Check overlap with openings on this wall
        for opening in openings:
            o_dist, o_t = _point_to_segment_distance(
                opening['center_2d'], nearest_wall['start'], nearest_wall['end']
            )
            if o_dist > 0.3:
                continue
            o_center_t = np.dot(
                opening['center_2d'] - nearest_wall['start'], wall_dir
            )
            o_min_t = o_center_t - opening['width'] / 2
            o_max_t = o_center_t + opening['width'] / 2
            art_min_t = proj_t - art_half_width
            art_max_t = proj_t + art_half_width

            if art_min_t < o_max_t and art_max_t > o_min_t:
                if proj_t < o_center_t:
                    shift = -(art_max_t - o_min_t + 0.05)
                else:
                    shift = o_max_t - art_min_t + 0.05
                new_vertices[:, 0] += wall_dir[0] * shift
                new_vertices[:, 1] += wall_dir[1] * shift
                break

        # Check if art falls within any door nogo zone
        for door in door_openings:
            if 'nogo_polygon' not in door:
                continue
            v_min = new_vertices.min(axis=0)
            v_max = new_vertices.max(axis=0)
            art_box = box(v_min[0], v_min[1], v_max[0], v_max[1])
            if not art_box.intersects(door['nogo_polygon']):
                continue
            # Only act if door is on this wall
            d_dist, _ = _point_to_segment_distance(
                door['center_2d'], nearest_wall['start'], nearest_wall['end'])
            if d_dist > 0.3:
                continue
            # Compute nogo extent along wall
            nogo_center_t = np.dot(
                door['center_2d'] - nearest_wall['start'], wall_dir)
            nogo_half = door['width'] / 2 + 0.3
            nogo_min_t = nogo_center_t - nogo_half
            nogo_max_t = nogo_center_t + nogo_half
            # Recompute current art position along wall
            art_center_t = np.dot(
                (v_min[:2] + v_max[:2]) / 2 - nearest_wall['start'], wall_dir)
            if art_center_t < nogo_center_t:
                nogo_shift = nogo_min_t - (art_center_t + art_half_width) - 0.05
            else:
                nogo_shift = nogo_max_t - (art_center_t - art_half_width) + 0.05
            new_vertices[:, 0] += wall_dir[0] * nogo_shift
            new_vertices[:, 1] += wall_dir[1] * nogo_shift
            if verbose:
                print(f"    {name}: shifted {abs(nogo_shift):.2f}m along wall "
                      "to clear door nogo zone")
            break

        new_mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=mesh.faces.copy(),
            process=False
        )
        _copy_visual(mesh, new_mesh)
        placed.append((name, new_mesh))

        if verbose:
            final_center = (new_mesh.bounds[0] + new_mesh.bounds[1]) / 2
            print(f"    {name}: mounted on wall #{nearest_wall['edge_index']} "
                  f"at height {final_center[2]:.2f}m")

    return placed


def _make_unique_scene_name(scene, base_name):
    """Return a stable scene name that does not collide with existing entries."""
    base = str(base_name or "geometry")
    if base not in scene.graph.nodes and base not in scene.geometry:
        return base

    suffix = 2
    while True:
        candidate = f"{base}__{suffix}"
        if candidate not in scene.graph.nodes and candidate not in scene.geometry:
            return candidate
        suffix += 1


def merge_meshes_into_scene(room_mesh, object_meshes, debug_meshes=None):
    """
    Create a trimesh Scene containing room and all objects.

    Args:
        room_mesh: Base room mesh (trimesh.Trimesh or Scene)
        object_meshes: List of (name, trimesh.Trimesh) tuples
        debug_meshes: Optional list of (name, trimesh.Trimesh) tuples for debug geometry

    Returns:
        trimesh.Scene with all meshes
    """
    scene = trimesh.Scene()

    # Add room mesh
    if isinstance(room_mesh, trimesh.Scene):
        # Preserve semantic node names from prior scene merges so later
        # passes can still identify windows after a GLB round-trip.
        for node_name in room_mesh.graph.nodes_geometry:
            transform, geom_name = room_mesh.graph[node_name]
            geom = room_mesh.geometry.get(geom_name)
            if geom is None:
                continue
            preserved_name = _make_unique_scene_name(scene, node_name)
            scene.add_geometry(
                geom,
                node_name=preserved_name,
                geom_name=preserved_name,
                transform=transform,
            )
    else:
        scene.add_geometry(room_mesh, node_name="room", geom_name="room")

    # Add object meshes
    for name, mesh in object_meshes:
        unique_name = _make_unique_scene_name(scene, name)
        scene.add_geometry(mesh, node_name=unique_name, geom_name=unique_name)

    # Add debug meshes if provided
    if debug_meshes:
        for name, mesh in debug_meshes:
            unique_name = _make_unique_scene_name(scene, name)
            scene.add_geometry(mesh, node_name=unique_name, geom_name=unique_name)

    return scene


def main():
    parser = argparse.ArgumentParser(description="Camera-to-World Transform and Scene Merging")
    parser.add_argument("--objects-dir", required=True, help="Directory with reconstructed objects from step2")
    parser.add_argument("--room-mesh", required=True, help="Original room mesh (GLB)")
    parser.add_argument("--images-txt", required=True, help="COLMAP images.txt file")
    parser.add_argument("--camera-name", required=True, help="Camera name (e.g., bedroom_0001)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Include OBB debug boxes in output")
    parser.add_argument("--scene-json", type=str, default=None,
                        help="Path to scene layout JSON (for room polygon)")
    parser.add_argument("--room-id", type=str, default=None,
                        help="Room identifier for polygon lookup")
    parser.add_argument("--placement-mode", choices=['smart', 'simple'], default='smart',
                        help="'smart': full heuristics (default). "
                             "'simple': minimal SAM3D-direct — level to floor/ceiling, "
                             "windows fit uniformly within opening (preserves aspect ratio).")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 3: Camera-to-World Transform and Scene Merging")
    print("=" * 60)

    # Parse COLMAP camera pose
    print(f"\nLoading camera poses from: {args.images_txt}")
    poses = parse_colmap_images_txt(args.images_txt)
    print(f"  Found {len(poses)} camera poses")

    if args.camera_name not in poses:
        print(f"ERROR: Camera '{args.camera_name}' not found!")
        print(f"  Available cameras: {list(poses.keys())}")
        return 1

    pose = poses[args.camera_name]
    print(f"\nUsing camera: {args.camera_name}")
    print(f"  Quaternion (wxyz): {pose['quaternion']}")
    print(f"  Translation: {pose['translation']}")

    # Compute camera-to-world transform
    R_c2w, camera_position = colmap_to_camera_to_world(
        pose['quaternion'], pose['translation']
    )
    print(f"  Camera position (world): [{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]")

    # Compute camera forward direction in XY plane for object placement
    camera_forward_3d = -R_c2w[:, 2]  # -Z column = forward direction
    camera_forward_xy = camera_forward_3d[:2].copy()
    norm = np.linalg.norm(camera_forward_xy)
    camera_forward_xy = camera_forward_xy / norm if norm > 1e-6 else None
    camera_position_xy = camera_position[:2].copy()

    # Y-up to Z-up conversion matrix
    # GLB export uses y-up, our world uses z-up
    # This transforms: (x, y, z)_yup -> (x, z, -y)_zup
    yup_to_zup = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)

    # Load reconstruction summary
    print(f"\nLoading reconstruction summary from: {args.objects_dir}")
    summary = load_reconstruction_summary(args.objects_dir)
    print(f"  Found {len(summary['objects'])} reconstructed objects")

    # Transform each object mesh to world space
    print("\nTransforming objects to world space...")
    objects_world = []
    objects_dir = Path(args.objects_dir)

    for obj in summary['objects']:
        prefix = f"{obj['index']:02d}_{obj['label'].replace(' ', '_')}"
        glb_path = objects_dir / f"{prefix}.glb"
        pose_path = objects_dir / f"{prefix}_pose.json"

        if not glb_path.exists():
            print(f"  WARNING: {glb_path.name} not found, skipping")
            continue

        if not pose_path.exists():
            print(f"  WARNING: {pose_path.name} not found, skipping")
            continue

        print(f"  Loading: {glb_path.name}...", end=" ", flush=True)

        # Load mesh (raw local, Y-up from SAM3D)
        mesh_raw = trimesh.load(str(glb_path), force='mesh')

        # Load SAM-3D-Objects pose
        with open(pose_path) as f:
            pose_data = json.load(f)

        # Transform to world space
        mesh_world = transform_mesh_to_world(
            mesh_raw, pose_data, R_c2w, camera_position, yup_to_zup
        )

        name = f"object_{obj['index']:02d}_{obj['label'].replace(' ', '_')}"
        objects_world.append((name, mesh_world, mesh_raw))

        # Print bounding box
        bounds = mesh_world.bounds
        center = (bounds[0] + bounds[1]) / 2
        print(f"center=[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    print(f"\n  Transformed {len(objects_world)} objects")

    # Diagnostic: Check mesh quality after world transform
    if args.debug:
        print("\n  [DIAGNOSTIC] After world transform:")
        for name, mesh, _raw in objects_world:
            diagnose_mesh_obb(mesh, f"{name} (after world transform)")

    # Load room mesh
    print(f"\nLoading room mesh: {args.room_mesh}")
    room_mesh = trimesh.load(args.room_mesh)
    if isinstance(room_mesh, trimesh.Scene):
        print(f"  Room contains {len(room_mesh.geometry)} geometries")
    else:
        print(f"  Room has {len(room_mesh.vertices)} vertices, {len(room_mesh.faces)} faces")

    # Create output directory early (needed for debug output)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # Object Placement Pipeline:
    # Transform → Wall Align → Level → Resolve Interpenetrations → Sort → Drop
    # ==========================================================================
    print("\nPlacing objects...")
    floor_z = 0.0

    # Get room bounds for wall collision detection
    room_bounds_xy = get_room_bounds_xy(room_mesh)
    print(f"  Room XY bounds: min=[{room_bounds_xy[0][0]:.2f}, {room_bounds_xy[0][1]:.2f}], "
          f"max=[{room_bounds_xy[1][0]:.2f}, {room_bounds_xy[1][1]:.2f}]")

    # Step 1a: Align to wall axes (rotate around Z to make OBB edges parallel to X/Y)
    print("  Step 1a: Aligning objects to wall axes...")
    objects_aligned = []
    for name, mesh, mesh_raw in objects_world:
        print(f"    {name}:")
        mesh_aligned = align_mesh_to_axes(mesh, verbose=True)
        objects_aligned.append((name, mesh_aligned, mesh_raw))
        if mesh_aligned is not mesh:
            print(f"      -> aligned")
        else:
            print(f"      -> unchanged")

    # Diagnostic: Check mesh quality after wall alignment
    if args.debug:
        print("\n  [DIAGNOSTIC] After wall alignment:")
        for name, mesh, _raw in objects_aligned:
            diagnose_mesh_obb(mesh, f"{name} (after wall align)")

    # Step 1b: Shape-aware leveling (floor objects only)
    # Windows/wall_art: skip — Phase A / Phase D handle their orientation
    # Ceiling: skip — leveled in Phase E right before positioning
    print("  Step 1b: Shape-aware leveling...")
    objects_leveled = []
    for name, mesh, mesh_raw in objects_aligned:
        pre_cat = classify_object_by_name(name)
        if pre_cat in ('window', 'wall_art', 'ceiling'):
            print(f"    {name}: skip leveling ({pre_cat})")
            objects_leveled.append((name, mesh, mesh_raw))
        else:
            print(f"    {name}:")
            mesh_leveled = level_object_robust(
                mesh, raw_local_mesh=mesh_raw, target='floor', name=name,
                verbose=True
            )
            objects_leveled.append((name, mesh_leveled, mesh_raw))
            if mesh_leveled is not mesh:
                print(f"      -> leveled")
            else:
                print(f"      -> unchanged")

    # Diagnostic: Check mesh quality after leveling
    if args.debug:
        print("\n  [DIAGNOSTIC] After leveling:")
        for name, mesh, _raw in objects_leveled:
            diagnose_mesh_obb(mesh, f"{name} (after leveling)")

    # DEBUG: Export leveled objects before interpenetration resolution
    if args.debug:
        print("\n  [DEBUG] Exporting aligned+leveled objects (before interpenetration resolution)...")
        leveled_scene = trimesh.Scene()
        # Add room mesh for context
        if isinstance(room_mesh, trimesh.Scene):
            for name, geom in room_mesh.geometry.items():
                leveled_scene.add_geometry(geom, node_name=f"room_{name}")
        else:
            leveled_scene.add_geometry(room_mesh, node_name="room")
        # Add leveled objects and their OBBs
        leveled_debug_meshes = []
        for name, mesh, _raw in objects_leveled:
            leveled_scene.add_geometry(mesh, node_name=name)
            # Add AABB visualization (use AABB since mesh is now axis-aligned)
            obb_mesh = create_obb_wireframe(mesh, f"obb_{name}", color=[0, 255, 0, 64], use_aabb=True)  # Green
            leveled_debug_meshes.append((f"obb_{name}", obb_mesh))
        for name, mesh in leveled_debug_meshes:
            leveled_scene.add_geometry(mesh, node_name=name)
        leveled_path = output_dir / "objects_leveled.glb"
        leveled_scene.export(str(leveled_path))
        print(f"    Saved: {leveled_path}")

    # ==========================================================================
    # Classification-Aware Multi-Phase Placement (or legacy fallback)
    # ==========================================================================

    # Load room geometry if available
    room_polygon = None
    wall_thickness = 0.1  # default
    if args.scene_json and args.room_id:
        print(f"  Loading room geometry from: {args.scene_json}")
        room_polygon, wall_thickness = load_room_geometry(args.scene_json, args.room_id)
        if room_polygon:
            print(f"    Room polygon loaded ({len(room_polygon.exterior.coords)} vertices)")
            print(f"    Wall thickness: {wall_thickness}m")
        else:
            print(f"    WARNING: Room '{args.room_id}' not found in scene JSON")

    if room_polygon and args.scene_json and args.room_id and args.placement_mode == 'simple':
        # ---- SIMPLE: Minimal SAM3D-direct placement ----
        # Use SAM3D output with minimal intervention (skips Steps 1a/1b entirely):
        #   - First: scale XY layout uniformly so all objects fit within room
        #   - Floor / flat-floor / wall_art: drop to floor only
        #   - Ceiling: align_local_up_to_z + ensure_pendant_down + raise to ceiling
        #   - Windows: OBB orientation + uniform scale (preserves aspect ratio, fits within opening)
        print("\n  SIMPLE placement mode (SAM3D-direct):")

        ceiling_height = load_ceiling_height(args.scene_json, args.room_id)
        inner_margin = wall_thickness + 0.02
        inner_polygon = room_polygon.buffer(-inner_margin)
        if inner_polygon.is_empty or not inner_polygon.is_valid:
            inner_polygon = room_polygon.buffer(-0.01)

        # Fit entire XY layout into room before per-object placement
        print("  Fitting XY layout to room...")
        objects_fitted = fit_layout_to_room(objects_world, room_polygon, wall_thickness)

        # Resolve interpenetrations and push toward walls (only if cluttered)
        print("  Decluttering...")
        objects_world = declutter_objects(objects_fitted, room_polygon, wall_thickness)

        openings = load_room_openings(args.scene_json, args.room_id)
        window_openings = [o for o in openings if o['type'] == 'window']
        used_win_openings = set()
        wall_edges = compute_wall_edges(room_polygon)

        # Two passes: floor/ceiling/window first, then wall_art (needs
        # floor objects for clash avoidance).
        placed_meshes = []
        placed_floor_pending = []
        wall_art_pending = []
        placed_windows = []

        for name, mesh, mesh_raw in objects_world:
            cat = classify_object_by_name(name)

            if cat == 'ceiling':
                # Same leveling as smart Phase E
                if mesh_raw is not None:
                    mesh = align_local_up_to_z(mesh_raw, mesh, verbose=True)
                tilt = _check_surface_tilt(mesh, 'top')
                if tilt < 30:
                    for _ in range(3):
                        mesh = level_surface(mesh, which='top', verbose=True)
                mesh = ensure_pendant_down(mesh, verbose=True)
                mesh = raise_mesh_to_z(mesh, ceiling_height)
                mesh = clamp_mesh_to_polygon(mesh, inner_polygon)
                print(f"    {name}: ceiling → raised to {ceiling_height:.2f}m")
                placed_meshes.append((name, mesh))

            elif cat == 'window':
                # Orient to nearest unmatched wall opening; preserve window's own proportions
                mesh_center_xy = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2
                best_idx, best_dist = None, float('inf')
                for idx, op in enumerate(window_openings):
                    if idx in used_win_openings:
                        continue
                    d = np.linalg.norm(mesh_center_xy - op['center_2d'])
                    if d < best_dist:
                        best_dist, best_idx = d, idx
                if best_idx is not None:
                    used_win_openings.add(best_idx)
                    mesh = _fit_mesh_to_opening(
                        mesh, window_openings[best_idx], wall_thickness)
                    op = window_openings[best_idx]
                    print(f"    {name}: window → opening at "
                          f"[{op['center_2d'][0]:.2f},{op['center_2d'][1]:.2f}]")
                    placed_windows.append((name, mesh, op))
                else:
                    print(f"    {name}: window → no opening found, kept as-is")
                    placed_meshes.append((name, mesh))

            elif cat == 'wall_art':
                # Defer wall art to a second pass so it can use the placed
                # floor objects for clash avoidance. Apply level_wall_art
                # now so the painting hangs straight, independent of SAM3D
                # tilt noise (aligns the OBB's vertical edge to world Z).
                mesh = level_wall_art(mesh, verbose=True)
                wall_art_pending.append((name, mesh))

            else:
                # Floor / flat_floor: multi-candidate smart leveling. Tries
                # multiple strategies (top/bottom surface fits at different
                # percentiles, OBB), filters by dimension preservation, and
                # picks the lowest-tilt survivor.
                raw_tilt = _check_surface_tilt(mesh, 'bottom')
                mesh = smart_level_floor(
                    mesh, raw_local_mesh=mesh_raw, name=name, verbose=True
                )
                mesh = drop_mesh_to_z(mesh, floor_z)
                print(f"    {name}: {cat} → smart_level (raw={raw_tilt:.1f}°) + dropped to floor")
                placed_floor_pending.append((name, mesh))

        unmatched_openings = [
            op for idx, op in enumerate(window_openings)
            if idx not in used_win_openings
        ]
        cloned_windows = clone_windows_for_unmatched(
            placed_windows, unmatched_openings, wall_edges, wall_thickness,
            args.scene_json, args.room_id, verbose=True
        )
        placed_meshes.extend((name, mesh) for name, mesh, _ in placed_windows)
        placed_meshes.extend(cloned_windows)
        print(f"    Placed {len(placed_windows)} windows, "
              f"cloned {len(cloned_windows)}")

        if placed_floor_pending:
            print("  Refining likely wall-adjacent floor objects...")
            placed_floor_for_art = flush_likely_wall_objects_simple(
                placed_floor_pending, objects_fitted, room_polygon,
                wall_thickness, openings, verbose=True,
            )
            for name, mesh in placed_floor_for_art:
                placed_meshes.append((name, mesh))
        else:
            placed_floor_for_art = []

        # Pass 2: wall art (uses placed floor objects for clash avoidance)
        if wall_art_pending:
            print(f"  Placing {len(wall_art_pending)} wall art objects...")
            placed_art = place_wall_art(
                wall_art_pending, wall_edges, placed_floor_for_art,
                room_polygon, wall_thickness, ceiling_height, openings,
                verbose=True,
            )
            for name, mesh in placed_art:
                placed_meshes.append((name, mesh))

        print(f"\n  Placement complete: {len(placed_meshes)} objects (simple mode)")

    elif room_polygon and args.scene_json and args.room_id:
        # ---- SMART: Classification-aware multi-phase placement ----
        print("\n  Classification-aware placement pipeline:")

        # Classify objects
        print("  Step 2: Classifying objects...")
        windows, paintings, flat_floors, floor_standing, ceiling_objects = [], [], [], [], []
        for name, mesh, mesh_raw in objects_leveled:
            label = name  # name contains label from reconstruction summary
            category = classify_object(name, label, mesh)
            print(f"    {name}: {category}")
            if category == 'window':
                windows.append((name, mesh))
            elif category == 'wall_art':
                paintings.append((name, mesh))
            elif category == 'ceiling':
                ceiling_objects.append((name, mesh, mesh_raw))
            elif category == 'flat_floor':
                flat_floors.append((name, mesh))
            else:
                floor_standing.append((name, mesh))

        print(f"    Classification: {len(windows)} windows, {len(paintings)} wall art, "
              f"{len(ceiling_objects)} ceiling, {len(flat_floors)} flat floor, "
              f"{len(floor_standing)} floor standing")

        # Load scene data
        print("  Step 3: Loading scene openings and wall edges...")
        openings = load_room_openings(args.scene_json, args.room_id)
        wall_edges = compute_wall_edges(room_polygon)
        door_openings = [o for o in openings if o['type'] == 'door']
        window_openings = [o for o in openings if o['type'] == 'window']
        print(f"    {len(openings)} openings ({len(door_openings)} doors, "
              f"{len(window_openings)} windows), {len(wall_edges)} wall edges")

        inner_margin = wall_thickness + 0.02
        inner_polygon = room_polygon.buffer(-inner_margin)
        if inner_polygon.is_empty or not inner_polygon.is_valid:
            inner_polygon = room_polygon.buffer(-0.01)

        # Phase A: Windows
        print("  Phase A: Placing windows...")
        placed_windows, unmatched_openings = place_window_objects(
            windows, window_openings, wall_edges, wall_thickness, verbose=True
        )
        cloned_windows = clone_windows_for_unmatched(
            placed_windows, unmatched_openings, wall_edges, wall_thickness,
            args.scene_json, args.room_id, verbose=True
        )
        # Strip the opening dict from placed_windows for final merge
        placed_windows_flat = [(n, m) for n, m, _ in placed_windows]
        print(f"    Placed {len(placed_windows_flat)} windows, "
              f"cloned {len(cloned_windows)}")

        # Phase B: Floor-standing objects
        print("  Phase B: Placing floor-standing objects...")
        placed_floor = place_floor_objects(
            floor_standing, wall_edges, openings, room_polygon,
            wall_thickness, camera_forward_xy=camera_forward_xy,
            camera_position_xy=camera_position_xy, verbose=True
        )
        print(f"    Placed {len(placed_floor)} floor-standing objects")

        # Phase C: Flat floor objects
        print("  Phase C: Placing flat floor objects...")
        placed_flat = []
        for name, mesh in flat_floors:
            mesh = drop_mesh_to_z(mesh, 0.0)
            mesh = clamp_mesh_to_polygon(mesh, inner_polygon)
            placed_flat.append((name, mesh))
            if args.debug:
                center = (mesh.bounds[0] + mesh.bounds[1]) / 2
                print(f"    {name}: dropped to floor at "
                      f"[{center[0]:.2f}, {center[1]:.2f}]")
        print(f"    Placed {len(placed_flat)} flat floor objects")

        # Note: flat floor objects (carpets, rugs) are not checked against
        # door nogo zones because they cannot physically block a doorway.

        # Phase D: Wall art / paintings
        print("  Phase D: Placing wall art...")
        ceiling_height = load_ceiling_height(args.scene_json, args.room_id)
        placed_art = place_wall_art(
            paintings, wall_edges, placed_floor + placed_flat,
            room_polygon, wall_thickness, ceiling_height, openings,
            verbose=True
        )
        print(f"    Placed {len(placed_art)} wall art objects")

        # Phase E: Ceiling objects
        # Use align_local_up_to_z to track SAM3D's Y-up through the
        # transform chain — this is more reliable than PCA (which fails
        # for disc-shaped chandeliers) or OBB (which is ambiguous for
        # near-cubic shapes). Then refine with level_surface.
        print("  Phase E: Placing ceiling objects...")
        placed_ceiling = []
        for name, mesh, mesh_raw in ceiling_objects:
            print(f"    {name}:")
            tilt_before = _check_surface_tilt(mesh, 'top')
            print(f"      Initial top surface tilt: {tilt_before:.1f}°")
            if mesh_raw is not None:
                mesh = align_local_up_to_z(mesh_raw, mesh, verbose=True)
            tilt_after_align = _check_surface_tilt(mesh, 'top')
            print(f"      Top tilt after align: {tilt_after_align:.1f}°")
            if tilt_after_align < 30:
                for i in range(3):
                    mesh = level_surface(mesh, which='top', verbose=True)
            else:
                print(f"      Skipping level_surface (tilt {tilt_after_align:.1f}° > 30° — irregular geometry, not real tilt)")
            mesh = ensure_pendant_down(mesh, verbose=True)
            mesh = raise_mesh_to_z(mesh, ceiling_height)
            mesh = clamp_mesh_to_polygon(mesh, inner_polygon)
            placed_ceiling.append((name, mesh))
        print(f"    Placed {len(placed_ceiling)} ceiling objects")

        # Merge all placed objects
        placed_meshes = (placed_windows_flat + cloned_windows +
                         placed_floor + placed_flat + placed_art +
                         placed_ceiling)
        print(f"\n  Placement complete: {len(placed_meshes)} objects total")

    else:
        # ---- LEGACY: Drop-to-floor + interpenetration resolution ----
        print("  Step 2: Dropping all objects to floor (legacy mode)...")
        objects_on_floor = []
        for name, mesh, _raw in objects_leveled:
            mesh_dropped = drop_mesh_to_z(mesh, floor_z)
            objects_on_floor.append((name, mesh_dropped))

            final_bounds = mesh_dropped.bounds
            center = (final_bounds[0] + final_bounds[1]) / 2
            base_z = final_bounds[0][2]
            print(f"    {name}: dropped to floor, base_z={base_z:.3f}, "
                  f"center=[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

        if args.debug:
            print("\n  [DIAGNOSTIC] After floor drop:")
            for name, mesh in objects_on_floor:
                diagnose_mesh_obb(mesh, f"{name} (after floor drop)")

        if room_polygon:
            print("  Step 3: Resolving floor conflicts (wall-aware)...")
            placed_meshes = resolve_floor_conflicts(
                objects_on_floor,
                room_polygon,
                wall_thickness,
                verbose=True
            )
        else:
            print("  Step 3: Resolving floor conflicts (AABB-based fallback)...")
            placed_meshes = resolve_interpenetrations(
                objects_on_floor,
                room_bounds_xy,
                verbose=True
            )

        print(f"  Placement complete for {len(placed_meshes)} objects")

    # Diagnostic: Check mesh quality after placement
    if args.debug:
        print("\n  [DIAGNOSTIC] After placement (final):")
        for name, mesh in placed_meshes:
            diagnose_mesh_obb(mesh, f"{name} (final)")

    # Create debug OBB visualizations if requested
    debug_meshes = None
    if args.debug:
        print("\nGenerating debug AABB boxes (axis-aligned after leveling)...")
        debug_meshes = []
        for name, mesh in placed_meshes:
            # Use AABB since mesh is now axis-aligned after leveling
            aabb_mesh = create_obb_wireframe(
                mesh,
                f"aabb_{name}",
                color=[255, 0, 0, 64],  # Red, semi-transparent
                use_aabb=True
            )
            debug_meshes.append((f"aabb_{name}", aabb_mesh))
            # Get AABB extents for logging (more meaningful after axis alignment)
            aabb_min, aabb_max = get_mesh_aabb(mesh)
            extents = aabb_max - aabb_min
            print(f"    {name}: AABB extents=[{extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}]")
        print(f"  Created {len(debug_meshes)} debug AABB boxes")

    # Create combined scene
    print("\nMerging objects into scene...")
    combined_scene = merge_meshes_into_scene(room_mesh, placed_meshes, debug_meshes)

    # Export combined scene
    output_path = output_dir / "scene_with_objects.glb"
    combined_scene.export(str(output_path))
    print(f"  Saved: {output_path}")
    print(f"  Total geometries: {len(combined_scene.geometry)}")

    # Also save objects-only scene (without room)
    objects_only = trimesh.Scene()
    for name, mesh in placed_meshes:
        objects_only.add_geometry(mesh, node_name=name)
    objects_only_path = output_dir / "objects_only.glb"
    objects_only.export(str(objects_only_path))
    print(f"  Saved objects only: {objects_only_path}")

    # Save transform info
    transform_info = {
        'camera_name': args.camera_name,
        'camera_pose': pose,
        'camera_position_world': camera_position.tolist(),
        'R_c2w': R_c2w.tolist(),
        'yup_to_zup': yup_to_zup.tolist(),
        'num_objects': len(placed_meshes),
        'objects': [name for name, _ in placed_meshes]
    }
    info_path = output_dir / "transform_info.json"
    with open(info_path, 'w') as f:
        json.dump(transform_info, f, indent=2)
    print(f"  Saved transform info: {info_path}")

    print("\n" + "=" * 60)
    print("Scene merging complete!")
    print(f"  Combined scene: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
