"""
Multi-view rendering for 3D Gaussian Splatting (3DGS) training.
Generates camera positions covering each room and exports renders + COLMAP format.
"""

from render_backend import configure_render_backend

# Set rendering backend BEFORE importing pyrender.
# Use RENDER_BACKEND=auto|egl|osmesa|pyglet. On headless Linux, auto selects
# the first EGL device that can initialize instead of assuming device 0 works.
configure_render_backend()

import json
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Optional
import pyrender
from pyrender.constants import RenderFlags
import cv2


def load_scene_data(json_path, glb_path):
    """Load both JSON scene description and 3D mesh."""
    with open(json_path, 'r') as f:
        scene_data = json.load(f)
    
    scene_mesh = trimesh.load(glb_path)
    
    return scene_data, scene_mesh


def compute_room_bounds(room_polygon):
    """Compute bounding box of room polygon."""
    polygon = np.array(room_polygon)
    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    center_x, center_y = polygon.mean(axis=0)
    
    return {
        'min': [min_x, min_y],
        'max': [max_x, max_y],
        'center': [center_x, center_y],
        'size': [max_x - min_x, max_y - min_y]
    }


def is_point_in_polygon(point, polygon):
    """Check if a 2D point is inside a polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def _sample_perimeter_positions(polygon, num_cameras, wall_offset, height,
                                center, look_at_height, camera_type=None):
    """
    Sample uniformly-spaced camera positions along a room's perimeter.

    Computes exact positions at equal arc-length intervals along the polygon
    boundary, offset inward by wall_offset. A half-spacing offset centers
    cameras within segments, naturally avoiding polygon corners (vertices)
    and eliminating near-duplicate cameras.

    Args:
        polygon: List of [x, y] vertices defining the room boundary
        num_cameras: Number of cameras to place
        wall_offset: Distance from wall to place camera (meters)
        height: Camera Z height (meters)
        center: [x, y] room center for look-at target
        look_at_height: Z height of look-at target (meters)
        camera_type: If not None, appended as third element in each tuple

    Returns:
        List of (position, look_at) or (position, look_at, camera_type) tuples
    """
    n_verts = len(polygon)
    # Build edges: start, end, direction, length, inward normal, cumulative distance
    edges = []
    cumulative = 0.0
    for i in range(n_verts):
        p1 = np.array(polygon[i], dtype=float)
        p2 = np.array(polygon[(i + 1) % n_verts], dtype=float)
        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)
        if edge_length < 1e-9:
            continue
        edge_dir = edge_vec / edge_length

        # Inward-pointing normal
        normal = np.array([-edge_dir[1], edge_dir[0]])
        test_point = (p1 + p2) / 2 + normal * 0.1
        if not is_point_in_polygon(test_point, polygon):
            normal = -normal

        edges.append({
            'start': p1,
            'dir': edge_dir,
            'normal': normal,
            'length': edge_length,
            'cum_start': cumulative,
        })
        cumulative += edge_length

    perimeter_length = cumulative
    if perimeter_length < 1e-9 or num_cameras <= 0:
        return []

    spacing = perimeter_length / num_cameras
    # Target distances with half-spacing offset to avoid corners
    target_distances = [spacing * (i + 0.5) for i in range(num_cameras)]

    cameras = []
    edge_idx = 0

    for target_d in target_distances:
        # Advance edge cursor to the edge containing target_d
        while edge_idx < len(edges) - 1:
            edge_end = edges[edge_idx]['cum_start'] + edges[edge_idx]['length']
            if target_d <= edge_end + 1e-9:
                break
            edge_idx += 1

        edge = edges[edge_idx]
        local_d = target_d - edge['cum_start']
        # Clamp to edge bounds
        local_d = max(0.0, min(local_d, edge['length']))

        cam_pos_2d = edge['start'] + edge['dir'] * local_d + edge['normal'] * wall_offset
        cam_pos = [cam_pos_2d[0], cam_pos_2d[1], height]
        look_at = [center[0], center[1], look_at_height]

        if camera_type is not None:
            cameras.append((cam_pos, look_at, camera_type))
        else:
            cameras.append((cam_pos, look_at))

    return cameras


def _filter_coverage_overlap(coverage_positions, regular_positions, polygon, min_angle_deg=10):
    """
    Drop perimeter cameras whose azimuth (relative to room center) is within
    min_angle_deg of any coverage camera.  This prevents near-duplicate views
    that cause ghosting in nerfstudio reconstructions.

    Args:
        coverage_positions: List of (position, look_at) for coverage cameras (indices 0, 1)
        regular_positions: List of (position, look_at) for perimeter cameras
        polygon: Room floor polygon [[x,y], ...]
        min_angle_deg: Minimum angular separation in degrees

    Returns:
        Filtered regular_positions list
    """
    if not coverage_positions or not regular_positions:
        return regular_positions

    bounds = compute_room_bounds(polygon)
    cx, cy = bounds['center']

    # Compute azimuths for coverage cameras
    coverage_azimuths = []
    for pos, _ in coverage_positions:
        az = np.degrees(np.arctan2(pos[1] - cy, pos[0] - cx)) % 360
        coverage_azimuths.append(az)

    filtered = []
    for pos, look_at in regular_positions:
        az = np.degrees(np.arctan2(pos[1] - cy, pos[0] - cx)) % 360
        too_close = False
        for caz in coverage_azimuths:
            diff = abs(az - caz)
            if diff > 180:
                diff = 360 - diff
            if diff < min_angle_deg:
                too_close = True
                break
        if too_close:
            print(f"  Dropping perimeter camera at azimuth {az:.1f}° (within {min_angle_deg}° of coverage camera at {caz:.1f}°)")
        else:
            filtered.append((pos, look_at))

    if len(filtered) < len(regular_positions):
        print(f"  Filtered {len(regular_positions) - len(filtered)} perimeter cameras overlapping with coverage cameras")

    return filtered


def generate_camera_positions(room_data, metadata, num_cameras=16,
                              camera_height=1.6, wall_offset=0.4):
    """
    Generate uniformly-spaced eye-level camera positions along the room perimeter.

    Elevated/overhead cameras are handled separately by generate_overhead_cameras().

    Args:
        room_data: Room information from JSON
        metadata: Scene metadata
        num_cameras: Maximum number of camera positions
        camera_height: Height of camera above floor (meters)
        wall_offset: Distance from wall to place camera (meters)

    Returns:
        List of (position, look_at) tuples
    """
    polygon = room_data['floor_polygon']
    bounds = compute_room_bounds(polygon)
    center = bounds['center']

    return _sample_perimeter_positions(
        polygon, num_cameras, wall_offset, height=camera_height,
        center=center, look_at_height=camera_height, camera_type=None
    )


def generate_central_corner_cameras(room_data, metadata, camera_height=1.6, offset_ratio=0.4):
    """
    Generate four cameras looking at diagonal corners from offset positions.

    Selects the two longest diagonal corner pairs and generates 2 cameras per pair.
    For a rectangle ABCD: first pair = (A,C), second pair = (B,D) → 4 cameras.

    These cameras provide optimal coverage for depth priors in diffusion models:
    - Each camera sees 2 walls with clear depth gradients
    - Corner provides a strong geometric anchor point
    - Together they cover all 4 walls from all diagonal angles
    - Cameras are offset from center toward the opposite corner for wider coverage

    Args:
        room_data: Room information from JSON
        metadata: Scene metadata
        camera_height: Height of camera above floor (meters)
        offset_ratio: How far to offset from center toward opposite corner (0-0.5)

    Returns:
        List of (position, look_at, camera_type) tuples where camera_type is 'central_corner'
    """
    polygon = room_data['floor_polygon']
    bounds = compute_room_bounds(polygon)
    center = np.array(bounds['center'])

    num_corners = len(polygon)

    if num_corners < 3:
        return []

    # Build all corner pairs sorted by distance (descending)
    pairs = []
    for i in range(num_corners):
        for j in range(i + 1, num_corners):
            dist = np.linalg.norm(np.array(polygon[i]) - np.array(polygon[j]))
            pairs.append((dist, i, j))
    pairs.sort(reverse=True)

    # Select up to 2 diagonal pairs using different corners
    selected_pairs = []
    used_corners = set()

    for dist, i, j in pairs:
        if len(selected_pairs) >= 2:
            break
        if len(selected_pairs) == 0:
            # First pair: always pick the longest diagonal
            selected_pairs.append((i, j))
            used_corners.add(i)
            used_corners.add(j)
        else:
            # Second pair: must use different corners than the first pair
            if i not in used_corners and j not in used_corners:
                selected_pairs.append((i, j))
                used_corners.add(i)
                used_corners.add(j)

    cameras = []

    for idx1, idx2 in selected_pairs:
        corner1 = np.array(polygon[idx1])
        corner2 = np.array(polygon[idx2])

        # Camera looking at corner 1, positioned offset toward corner 2
        offset1 = (corner2 - center) * offset_ratio
        cam_pos1 = center + offset1
        cam_pos1_3d = [cam_pos1[0], cam_pos1[1], camera_height]
        look_at1 = [corner1[0], corner1[1], camera_height]
        cameras.append((cam_pos1_3d, look_at1, 'central_corner'))

        # Camera looking at corner 2, positioned offset toward corner 1
        offset2 = (corner1 - center) * offset_ratio
        cam_pos2 = center + offset2
        cam_pos2_3d = [cam_pos2[0], cam_pos2[1], camera_height]
        look_at2 = [corner2[0], corner2[1], camera_height]
        cameras.append((cam_pos2_3d, look_at2, 'central_corner'))

    return cameras


def generate_overhead_cameras(room_data, metadata, num_cameras=8,
                              camera_height=1.6, wall_offset=0.4):
    """
    Generate uniformly-spaced elevated cameras along the room perimeter looking downward.

    Same perimeter placement as generate_camera_positions but at elevated height
    (camera_height + 0.8) looking down at eye level (camera_height). This provides
    overhead coverage without duplicating the eye-level views.

    Args:
        room_data: Room information from JSON
        metadata: Scene metadata
        num_cameras: Number of overhead camera positions
        camera_height: Base height (eye level); cameras placed at camera_height + 0.8
        wall_offset: Distance from wall to place camera (meters)

    Returns:
        List of (position, look_at, camera_type) tuples where camera_type is 'overhead'
    """
    polygon = room_data['floor_polygon']
    bounds = compute_room_bounds(polygon)
    center = bounds['center']
    elevated_height = camera_height + 0.8  # 2.4m by default
    look_at_height = camera_height  # Look at eye level

    return _sample_perimeter_positions(
        polygon, num_cameras, wall_offset, height=elevated_height,
        center=center, look_at_height=look_at_height, camera_type='overhead'
    )


def generate_room_coverage_camera(room_data, metadata, camera_height=1.6, wall_offset=0.05):
    """
    Generate a single camera pose that covers the entire room.

    Positions camera at the middle of the shortest wall, looking toward the
    opposite wall. This maximizes room coverage by showing 3 walls:
    - The opposite wall (perpendicular - strong depth anchor)
    - Both side walls (perspective convergence)

    This symmetric, centered view matches architectural photography conventions
    and provides clear depth structure for diffusion models.

    Args:
        room_data: Room dict with 'floor_polygon' and 'ceiling_height'
        metadata: Scene metadata with 'default_ceiling_height'
        camera_height: Height above floor (default 1.6m eye level)
        wall_offset: Distance from wall (default 0.05m, minimal safe margin)

    Returns:
        List of (position, look_at, camera_type) tuples where position/look_at are [x, y, z] lists
    """
    polygon = room_data['floor_polygon']
    bounds = compute_room_bounds(polygon)
    center = bounds['center']

    # Find the shortest wall by comparing room dimensions
    width = bounds['size'][0]   # x dimension
    depth = bounds['size'][1]   # y dimension

    if width <= depth:
        # Shortest walls are along X axis (at min_y and max_y)
        # Position at middle of the min_y wall, looking toward max_y
        camera_pos = [
            center[0],  # middle of wall
            bounds['min'][1] + wall_offset,  # offset from wall
            camera_height
        ]
        look_at = [
            center[0],  # look straight ahead
            bounds['max'][1],  # toward opposite wall
            camera_height
        ]
    else:
        # Shortest walls are along Y axis (at min_x and max_x)
        # Position at middle of the min_x wall, looking toward max_x
        camera_pos = [
            bounds['min'][0] + wall_offset,  # offset from wall
            center[1],  # middle of wall
            camera_height
        ]
        look_at = [
            bounds['max'][0],  # toward opposite wall
            center[1],  # look straight ahead
            camera_height
        ]

    return [(camera_pos, look_at, 'room_coverage')]


def generate_opposite_coverage_camera(room_data, metadata, camera_height=1.6, wall_offset=0.05):
    """
    Generate a coverage camera on the wall opposite to the primary coverage camera.

    If the primary coverage camera sits at the min side looking toward max,
    this places a camera at the max side looking toward min (and vice versa
    for the other axis). Together they cover all four walls.

    Args:
        room_data: Room dict with 'floor_polygon' and 'ceiling_height'
        metadata: Scene metadata with 'default_ceiling_height'
        camera_height: Height above floor (default 1.6m eye level)
        wall_offset: Distance from wall (default 0.05m, minimal safe margin)

    Returns:
        List of (position, look_at, camera_type) tuples
    """
    polygon = room_data['floor_polygon']
    bounds = compute_room_bounds(polygon)
    center = bounds['center']

    width = bounds['size'][0]   # x dimension
    depth = bounds['size'][1]   # y dimension

    if width <= depth:
        # Primary is at min_y looking toward max_y
        # Opposite is at max_y looking toward min_y
        camera_pos = [
            center[0],
            bounds['max'][1] - wall_offset,
            camera_height
        ]
        look_at = [
            center[0],
            bounds['min'][1],
            camera_height
        ]
    else:
        # Primary is at min_x looking toward max_x
        # Opposite is at max_x looking toward min_x
        camera_pos = [
            bounds['max'][0] - wall_offset,
            center[1],
            camera_height
        ]
        look_at = [
            bounds['min'][0],
            center[1],
            camera_height
        ]

    return [(camera_pos, look_at, 'room_coverage')]


def create_camera_matrix(position, look_at, up=[0, 0, 1]):
    """
    Create camera pose matrix (world to camera transform).
    
    Args:
        position: Camera position in world coordinates
        look_at: Point the camera is looking at
        up: Up vector
    
    Returns:
        4x4 camera pose matrix (OpenGL convention)
    """
    position = np.array(position)
    look_at = np.array(look_at)
    up = np.array(up)
    
    # Camera coordinate system (OpenGL: -Z forward, Y up, X right)
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    up_corrected = np.cross(right, forward)
    
    # Build rotation matrix (world to camera)
    # In OpenGL, camera looks down -Z, so forward maps to -Z
    R = np.array([
        right,
        up_corrected,
        -forward  # OpenGL convention
    ])
    
    # Translation (camera position in world)
    t = position
    
    # Build 4x4 pose matrix (camera-to-world)
    pose = np.eye(4)
    pose[:3, :3] = R.T  # Transpose because we want camera-to-world
    pose[:3, 3] = t
    
    return pose


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def export_colmap_cameras(output_dir, width, height, fx, fy, cx, cy):
    """
    Export COLMAP cameras.txt file with camera intrinsics.
    
    COLMAP format:
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # SIMPLE_PINHOLE: f, cx, cy
    # PINHOLE: fx, fy, cx, cy
    """
    cameras_path = Path(output_dir) / 'cameras.txt'
    
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
    
    print(f"Exported COLMAP cameras.txt to {cameras_path}")


def export_colmap_images(output_dir, image_data):
    """
    Export COLMAP images.txt file with camera extrinsics.
    
    COLMAP format:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    
    Args:
        image_data: List of dicts with keys: image_id, qvec, tvec, camera_id, name
    """
    images_path = Path(output_dir) / 'images.txt'
    
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_data)}\n")
        
        for img in image_data:
            # Line 1: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            qw, qx, qy, qz = img['qvec']
            tx, ty, tz = img['tvec']
            f.write(f"{img['image_id']} {qw} {qx} {qy} {qz} "
                   f"{tx} {ty} {tz} {img['camera_id']} {img['name']}\n")
            # Line 2: POINTS2D (empty for synthetic data)
            f.write("\n")
    
    print(f"Exported COLMAP images.txt to {images_path}")


def export_colmap_points3d(output_dir):
    """
    Export empty COLMAP points3D.txt file.
    For rendering-only, we don't have 3D points.
    """
    points_path = Path(output_dir) / 'points3D.txt'

    with open(points_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
               "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

    print(f"Exported COLMAP points3D.txt to {points_path}")


def extract_edges_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """
    Extract edges from a segmentation map by finding boundaries between regions.

    This produces geometrically exact edges since segmentation boundaries are
    ground truth, unlike Canny which can miss corners due to smooth gradients.

    Args:
        segmentation: RGB segmentation image as numpy array (H, W, 3)

    Returns:
        Edge map as numpy array (H, W) with values 0 or 255
    """
    h, w = segmentation.shape[:2]
    edges = np.zeros((h, w), dtype=np.uint8)

    # Check horizontal boundaries (compare pixel to its right neighbor)
    horiz_diff = np.any(segmentation[:, :-1] != segmentation[:, 1:], axis=2)
    edges[:, :-1] |= (horiz_diff * 255).astype(np.uint8)
    edges[:, 1:] |= (horiz_diff * 255).astype(np.uint8)

    # Check vertical boundaries (compare pixel to its bottom neighbor)
    vert_diff = np.any(segmentation[:-1, :] != segmentation[1:, :], axis=2)
    edges[:-1, :] |= (vert_diff * 255).astype(np.uint8)
    edges[1:, :] |= (vert_diff * 255).astype(np.uint8)

    return edges


def filter_scene_objects(
    scene_mesh: trimesh.Scene,
    scene_data: Dict,
    include_objects: bool = True
) -> trimesh.Scene:
    """
    Filter a scene to include or exclude furniture objects.

    Args:
        scene_mesh: Original Trimesh scene
        scene_data: Scene JSON data
        include_objects: If False, removes all furniture objects

    Returns:
        Filtered Trimesh scene
    """
    if include_objects:
        return scene_mesh

    if not isinstance(scene_mesh, trimesh.Scene):
        return scene_mesh

    # Build set of structural element names (walls, floors, ceilings)
    # These follow patterns: {room_id}_floor, {room_id}_ceiling, {room_id}_wall_N
    structural_names = set()
    for room in scene_data.get('rooms', []):
        room_id = room['id']
        structural_names.add(f"{room_id}_floor")
        structural_names.add(f"{room_id}_ceiling")
        # Add wall segments (up to a reasonable maximum)
        for i in range(100):
            structural_names.add(f"{room_id}_wall_{i}")

    # Create new scene with only structural elements
    filtered_scene = trimesh.Scene()

    for geom_name, geom in scene_mesh.geometry.items():
        # Only include if this is a structural element
        if geom_name not in structural_names:
            continue

        # Get transform for this geometry
        try:
            transform = scene_mesh.graph.get(geom_name)[0]
        except (KeyError, TypeError, ValueError):
            transform = np.eye(4)

        # Add to filtered scene
        filtered_scene.add_geometry(
            geom,
            node_name=geom_name,
            geom_name=geom_name,
            transform=transform
        )

    print(f"  Filtered scene: {len(scene_mesh.geometry)} -> {len(filtered_scene.geometry)} geometries (structure only)")

    return filtered_scene


def extract_object_meshes(
    scene_mesh: trimesh.Scene,
    scene_data: Dict,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """
    Extract non-structural (furniture/object) meshes from a scene in world space.

    Inverse of filter_scene_objects() — returns only the object geometries.

    Args:
        scene_mesh: Original Trimesh scene
        scene_data: Scene JSON data

    Returns:
        List of (name, trimesh.Trimesh) tuples with transforms applied
    """
    if not isinstance(scene_mesh, trimesh.Scene):
        return []

    # Build set of structural element names
    structural_names = set()
    for room in scene_data.get('rooms', []):
        room_id = room['id']
        structural_names.add(f"{room_id}_floor")
        structural_names.add(f"{room_id}_ceiling")
        for i in range(100):
            structural_names.add(f"{room_id}_wall_{i}")

    objects = []
    for geom_name, geom in scene_mesh.geometry.items():
        if geom_name in structural_names:
            continue

        # Get world-space transform
        try:
            transform = scene_mesh.graph.get(geom_name)[0]
        except (KeyError, TypeError, ValueError):
            transform = np.eye(4)

        # Apply transform to get world-space mesh
        world_mesh = geom.copy()
        world_mesh.apply_transform(transform)
        objects.append((geom_name, world_mesh))

    return objects


def check_camera_collision(
    cam_pos: np.ndarray,
    object_meshes: List[Tuple[str, trimesh.Trimesh]],
    padding: float = 0.05,
) -> List[str]:
    """
    Check if a camera position falls inside any object's axis-aligned bounding box.

    Uses AABB instead of trimesh.contains() because reconstructed meshes
    are not guaranteed watertight.

    Args:
        cam_pos: Camera position [x, y, z]
        object_meshes: List of (name, trimesh.Trimesh) in world space
        padding: Extra margin around AABBs (meters)

    Returns:
        List of colliding object names
    """
    collisions = []
    for name, mesh in object_meshes:
        bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        if bounds is None:
            continue
        bmin = bounds[0] - padding
        bmax = bounds[1] + padding
        if (bmin[0] <= cam_pos[0] <= bmax[0] and
                bmin[1] <= cam_pos[1] <= bmax[1] and
                bmin[2] <= cam_pos[2] <= bmax[2]):
            collisions.append(name)
    return collisions


def resolve_camera_collision(
    cam_pos: List[float],
    look_at: List[float],
    room_polygon: List[List[float]],
    object_meshes: List[Tuple[str, trimesh.Trimesh]],
    step_size: float = 0.05,
    max_steps: int = 20,
) -> Tuple[List[float], bool]:
    """
    Nudge a colliding camera toward the room center until it clears all objects.

    Tries the center direction first, then perpendicular directions (left/right
    along the wall). Preserves Z height.

    Args:
        cam_pos: Camera position [x, y, z]
        look_at: Look-at target [x, y, z]
        room_polygon: Room floor polygon [[x, y], ...]
        object_meshes: List of (name, trimesh.Trimesh) in world space
        step_size: Increment per step (meters)
        max_steps: Maximum number of steps (max displacement = step_size * max_steps)

    Returns:
        (new_position, was_adjusted) — new_position as [x, y, z] list
    """
    pos = np.array(cam_pos, dtype=float)
    bounds = compute_room_bounds(room_polygon)
    center_2d = np.array(bounds['center'], dtype=float)

    # Primary direction: toward room center (XY only)
    to_center = center_2d - pos[:2]
    dist = np.linalg.norm(to_center)
    if dist < 1e-9:
        return cam_pos, False
    to_center_dir = to_center / dist

    # Also try perpendicular directions (rotate ±90°)
    perp_left = np.array([-to_center_dir[1], to_center_dir[0]])
    perp_right = np.array([to_center_dir[1], -to_center_dir[0]])

    for direction in [to_center_dir, perp_left, perp_right]:
        for step in range(1, max_steps + 1):
            candidate_2d = pos[:2] + direction * step_size * step
            candidate = [candidate_2d[0], candidate_2d[1], pos[2]]

            # Must remain inside the room polygon
            if not is_point_in_polygon(candidate_2d, room_polygon):
                break  # This direction leaves the room, try next

            # Check collisions at candidate
            if not check_camera_collision(np.array(candidate), object_meshes):
                return candidate, True

    # All directions failed — keep original position (graceful degradation)
    return cam_pos, False


def avoid_object_collisions(
    camera_positions: List[Tuple],
    scene_mesh: trimesh.Scene,
    scene_data: Dict,
    room_data: Dict,
) -> List[Tuple]:
    """
    Check all camera positions for collisions with object meshes and resolve them.

    Args:
        camera_positions: List of (position, look_at) tuples
        scene_mesh: Full Trimesh scene (with objects)
        scene_data: Scene JSON data
        room_data: Current room dict with 'floor_polygon'

    Returns:
        Updated list of (position, look_at) tuples with collisions resolved
    """
    object_meshes = extract_object_meshes(scene_mesh, scene_data)
    if not object_meshes:
        return camera_positions

    polygon = room_data['floor_polygon']
    adjusted_count = 0
    result = []

    for cam_pos, look_at in camera_positions:
        collisions = check_camera_collision(np.array(cam_pos), object_meshes)
        if collisions:
            new_pos, was_adjusted = resolve_camera_collision(
                cam_pos, look_at, polygon, object_meshes
            )
            if was_adjusted:
                adjusted_count += 1
                print(f"    Camera at ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}) "
                      f"collided with {collisions} -> moved to "
                      f"({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")
            result.append((new_pos, look_at))
        else:
            result.append((cam_pos, look_at))

    if adjusted_count > 0:
        print(f"  Collision avoidance: adjusted {adjusted_count}/{len(camera_positions)} cameras")
    else:
        print(f"  Collision avoidance: no collisions detected")

    return result


def create_segmentation_palette(scene_mesh, scene_data):
    """
    Create a color palette for semantic segmentation.

    Assigns unique colors to:
    - Each room's floor, ceiling
    - Each wall segment (individual wall sides)
    - Each object instance

    Args:
        scene_mesh: Trimesh scene with named geometries
        scene_data: Scene JSON data

    Returns:
        dict: Mapping of geometry name to (color_rgb, class_id, class_name)
    """
    palette = {}
    class_id = 0

    # Predefined colors for structural elements
    structural_colors = {
        'floor': [139, 90, 43],      # Brown
        'ceiling': [200, 200, 200],  # Light gray
    }

    # Distinct colors for objects - using a perceptually distinct palette
    object_colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [0, 255, 128],    # Spring green
        [255, 0, 128],    # Rose
        [128, 255, 0],    # Lime
        [0, 128, 255],    # Sky blue
        [255, 128, 128],  # Light red
        [128, 255, 128],  # Light green
        [128, 128, 255],  # Light blue
        [255, 255, 128],  # Light yellow
    ]

    # Distinct colors for individual wall segments
    # Using varied hues to make each wall clearly distinguishable
    wall_segment_colors = [
        [255, 180, 180],  # Light red
        [180, 255, 180],  # Light green
        [180, 180, 255],  # Light blue
        [255, 255, 180],  # Light yellow
        [255, 180, 255],  # Light magenta
        [180, 255, 255],  # Light cyan
        [255, 220, 180],  # Peach
        [220, 180, 255],  # Lavender
        [180, 255, 220],  # Mint
        [255, 180, 220],  # Pink
        [220, 255, 180],  # Lime
        [180, 220, 255],  # Sky
        [240, 200, 200],  # Dusty rose
        [200, 240, 200],  # Sage
        [200, 200, 240],  # Periwinkle
        [240, 240, 200],  # Cream
    ]

    object_color_idx = 0
    wall_color_idx = 0

    if isinstance(scene_mesh, trimesh.Scene):
        for geom_name in scene_mesh.geometry.keys():
            # Determine the type of geometry
            if '_floor' in geom_name:
                room_id = geom_name.replace('_floor', '')
                color = structural_colors['floor']
                class_name = f"{room_id}_floor"
            elif '_ceiling' in geom_name:
                room_id = geom_name.replace('_ceiling', '')
                color = structural_colors['ceiling']
                class_name = f"{room_id}_ceiling"
            elif '_wall_' in geom_name:
                # Individual wall segment (e.g., "living_room_wall_0")
                color = wall_segment_colors[wall_color_idx % len(wall_segment_colors)]
                wall_color_idx += 1
                class_name = geom_name
            else:
                # Object - assign distinct color
                color = object_colors[object_color_idx % len(object_colors)]
                object_color_idx += 1
                class_name = geom_name  # e.g., "sofa_1_sofa"

            palette[geom_name] = {
                'color': color,
                'class_id': class_id,
                'class_name': class_name
            }
            class_id += 1

    return palette


def create_segmentation_scene(scene_mesh, palette):
    """
    Create a PyRender scene with flat-colored materials for segmentation.

    Args:
        scene_mesh: Trimesh scene
        palette: Color palette from create_segmentation_palette()

    Returns:
        pyrender.Scene with flat-colored meshes
    """
    seg_scene = pyrender.Scene(bg_color=[0, 0, 0, 255])  # Black background

    if isinstance(scene_mesh, trimesh.Scene):
        for geom_name, geom in scene_mesh.geometry.items():
            if geom_name in palette:
                color = palette[geom_name]['color']
                # Normalize color to 0-1 range
                color_normalized = [c / 255.0 for c in color] + [1.0]

                # Create a flat/unlit material
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=color_normalized,
                    metallicFactor=0.0,
                    roughnessFactor=1.0,
                    emissiveFactor=color_normalized[:3]  # Emit the color for flat look
                )

                # Get the transform for this geometry
                transform = scene_mesh.graph.get(geom_name)[0] if geom_name in scene_mesh.graph else np.eye(4)

                # Create mesh with the flat material
                mesh = pyrender.Mesh.from_trimesh(geom, material=material)
                seg_scene.add(mesh, pose=transform)
    else:
        # Single mesh - use a default color
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.5, 0.5, 0.5, 1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        mesh = pyrender.Mesh.from_trimesh(scene_mesh, material=material)
        seg_scene.add(mesh)

    return seg_scene


def export_segmentation_metadata(output_dir, palette):
    """
    Export segmentation metadata as JSON.

    Args:
        output_dir: Output directory
        palette: Color palette dict
    """
    metadata = {
        'classes': [],
        'color_to_class': {},
        'class_to_color': {}
    }

    for geom_name, info in palette.items():
        class_entry = {
            'class_id': info['class_id'],
            'class_name': info['class_name'],
            'geometry_name': geom_name,
            'color_rgb': info['color']
        }
        metadata['classes'].append(class_entry)

        # Create lookup tables
        color_key = f"{info['color'][0]}_{info['color'][1]}_{info['color'][2]}"
        metadata['color_to_class'][color_key] = info['class_name']
        metadata['class_to_color'][info['class_name']] = info['color']

    metadata_path = Path(output_dir) / 'segmentation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported segmentation metadata to {metadata_path}")
    return metadata


def render_scene(scene_mesh, camera_positions, output_dir, room_id,
                width=1920, height=1080, fov=60,
                key_light_intensity=4.0, fill_light_intensity=2.5,
                ambient_light_intensity=1.5,
                flat_lighting=False,
                seg_scene=None, seg_scene_structure=None,
                render_segmentation=True,
                render_edges=True):
    """
    Render scene from multiple camera positions using PyRender.

    Args:
        scene_mesh: Trimesh scene or mesh
        camera_positions: List of (position, look_at) tuples
        output_dir: Directory to save renders
        room_id: Room identifier
        width, height: Image resolution
        fov: Field of view in degrees
        seg_scene: Pre-built segmentation scene (optional)
        seg_scene_structure: Structure-only segmentation scene for wall edge overlay (optional)
        render_segmentation: Whether to render segmentation maps
        render_edges: Whether to render edge maps (from segmentation boundaries)

    Returns:
        List of image data dicts for COLMAP export
    """
    # Create PyRender scene from Trimesh
    if isinstance(scene_mesh, trimesh.Scene):
        scene = pyrender.Scene.from_trimesh_scene(scene_mesh)
    else:
        mesh = pyrender.Mesh.from_trimesh(scene_mesh)
        scene = pyrender.Scene()
        scene.add(mesh)
    
    # Camera intrinsics
    yfov = np.radians(fov)
    aspect_ratio = width / height
    fx = width / (2 * np.tan(yfov / 2) * aspect_ratio)
    fy = height / (2 * np.tan(yfov / 2))
    cx = width / 2
    cy = height / 2
    
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
    
    # Improved lighting setup with multiple sources
    # Key light (main directional)
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=key_light_intensity)
    
    # Fill light (softer, opposite side)
    fill_light = pyrender.SpotLight(
        color=np.ones(3), 
        intensity=fill_light_intensity,
        innerConeAngle=np.pi/4,
        outerConeAngle=np.pi/3
    )
    
    # Ambient light (overall scene illumination)
    ambient_light = pyrender.PointLight(color=np.ones(3), intensity=ambient_light_intensity)

    # Create renderer
    renderer = pyrender.OffscreenRenderer(width, height)

    # Prepare output directory
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images' / room_id
    depth_dir = output_dir / 'depth' / room_id
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Segmentation output directory
    if render_segmentation and seg_scene is not None:
        seg_dir = output_dir / 'segmentation' / room_id
        seg_dir.mkdir(parents=True, exist_ok=True)

    # Edge detection output directory
    if render_edges:
        edge_dir = output_dir / 'edges' / room_id
        edge_dir.mkdir(parents=True, exist_ok=True)

    image_data = []

    render_types = "color + depth"
    if render_segmentation and seg_scene is not None:
        render_types += " + segmentation"
    if render_edges:
        render_types += " + edges"
    print(f"\nRendering {len(camera_positions)} views for room '{room_id}' ({render_types})...")
    
    for idx, (cam_pos, look_at) in enumerate(camera_positions):
        # Create camera pose
        pose = create_camera_matrix(cam_pos, look_at)
        
        # Add camera to scene
        cam_node = scene.add(camera, pose=pose)

        # Add lighting (skip if flat_lighting - we'll use FLAT render flag instead)
        if not flat_lighting:
            # Key light attached to camera
            key_light_node = scene.add(key_light, pose=pose)

            # Fill light from opposite direction
            fill_pose = pose.copy()
            # Rotate 180 degrees around Z axis for opposite side lighting
            fill_rotation = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            fill_pose = fill_pose @ fill_rotation
            fill_light_node = scene.add(fill_light, pose=fill_pose)

            # Ambient light at room center
            ambient_pose = np.eye(4)
            ambient_pose[:3, 3] = look_at
            ambient_light_node = scene.add(ambient_light, pose=ambient_pose)

        # Render
        if flat_lighting:
            color, depth = renderer.render(scene, flags=RenderFlags.FLAT)
        else:
            color, depth = renderer.render(scene)

        # Remove camera and lights for next iteration
        scene.remove_node(cam_node)
        if not flat_lighting:
            scene.remove_node(key_light_node)
            scene.remove_node(fill_light_node)
            scene.remove_node(ambient_light_node)
        
        # Save color image
        image_name = f"{room_id}_{idx:04d}.png"
        image_path = images_dir / image_name
        Image.fromarray(color).save(image_path)

        # Save depth map
        depth_name = f"{room_id}_{idx:04d}_depth.png"
        depth_path = depth_dir / depth_name
        
        # Normalize depth for visualization (0-255)
        # Convention: close = white (255), far = black (0)
        depth_vis = depth.copy()
        valid_depth = depth_vis > 0  # Avoid division by zero
        if valid_depth.any():
            depth_min = depth_vis[valid_depth].min()
            depth_max = depth_vis[valid_depth].max()
            if depth_max > depth_min:
                # Invert: closest (depth_min) -> 1 (white), farthest (depth_max) -> 0 (black)
                depth_vis[valid_depth] = 1.0 - (depth_vis[valid_depth] - depth_min) / (depth_max - depth_min)
            depth_vis = (depth_vis * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
        
        Image.fromarray(depth_vis).save(depth_path)
        
        # Save raw depth as numpy array for training
        depth_raw_name = f"{room_id}_{idx:04d}_depth.npy"
        depth_raw_path = depth_dir / depth_raw_name
        np.save(depth_raw_path, depth)

        # Render segmentation map
        if render_segmentation and seg_scene is not None:
            # Add camera to segmentation scene
            seg_cam_node = seg_scene.add(camera, pose=pose)

            # Render with FLAT flag (no lighting)
            seg_color, _ = renderer.render(seg_scene, flags=RenderFlags.FLAT)

            # Remove camera
            seg_scene.remove_node(seg_cam_node)

            # Save segmentation map
            seg_name = f"{room_id}_{idx:04d}_seg.png"
            seg_path = seg_dir / seg_name
            Image.fromarray(seg_color).save(seg_path)

            # Extract edges from segmentation boundaries (geometrically exact)
            if render_edges:
                edges = extract_edges_from_segmentation(seg_color)
                edge_name = f"{room_id}_{idx:04d}_edges.png"
                edge_path = edge_dir / edge_name
                Image.fromarray(edges).save(edge_path)

        # Render wall structure edges and overlay on RGB (when flat_lighting enabled)
        if flat_lighting and seg_scene_structure is not None:
            # Render structure-only segmentation AND depth
            struct_cam_node = seg_scene_structure.add(camera, pose=pose)
            struct_seg_color, struct_depth = renderer.render(seg_scene_structure, flags=RenderFlags.FLAT)
            seg_scene_structure.remove_node(struct_cam_node)

            # Extract edges from structure segmentation
            structure_edges = extract_edges_from_segmentation(struct_seg_color)

            # Create occlusion mask: only show edges where structure is visible
            # (i.e., where structure depth matches full scene depth within tolerance)
            depth_tolerance = 0.05  # 5cm tolerance for depth comparison
            valid_depth = (depth > 0) & (struct_depth > 0)
            structure_visible = valid_depth & (np.abs(struct_depth - depth) < depth_tolerance)

            # Create composite: overlay black edges only where structure is visible
            composite = color.copy()
            edge_mask = (structure_edges > 0) & structure_visible
            composite[edge_mask] = [0, 0, 0]  # Black edges

            # Save composite image
            composite_name = f"{room_id}_{idx:04d}_with_edges.png"
            composite_path = images_dir / composite_name
            Image.fromarray(composite).save(composite_path)

        # Convert pose to COLMAP format (world-to-camera)
        # COLMAP uses world-to-camera, PyRender uses camera-to-world
        pose_w2c = np.linalg.inv(pose)
        R = pose_w2c[:3, :3]
        t = pose_w2c[:3, 3]
        
        # Convert rotation to quaternion
        qvec = rotation_matrix_to_quaternion(R)
        
        image_data.append({
            'image_id': idx + 1,
            'qvec': qvec,
            'tvec': t,
            'camera_id': 1,
            'name': f"{room_id}/{image_name}"
        })
        
        if (idx + 1) % 5 == 0:
            print(f"  Rendered {idx + 1}/{len(camera_positions)} views (color + depth)")
    
    renderer.delete()

    print(f"  Saved {len(camera_positions)} renders to {images_dir}")
    print(f"  Saved {len(camera_positions)} depth maps to {depth_dir}")
    if render_segmentation and seg_scene is not None:
        print(f"  Saved {len(camera_positions)} segmentation maps to {seg_dir}")
    if render_edges:
        print(f"  Saved {len(camera_positions)} edge maps to {edge_dir}")
    if flat_lighting and seg_scene_structure is not None:
        print(f"  Saved {len(camera_positions)} composite images with wall edges to {images_dir}")

    # Export camera intrinsics
    export_colmap_cameras(output_dir, width, height, fx, fy, cx, cy)

    return image_data, (fx, fy, cx, cy)


def create_summary_montage(output_dir, room_ids, render_segmentation=True):
    """
    Create summary montage images showing all rendered views at once.

    Generates:
    - Per-room montages (RGB, depth, segmentation grids)
    - Combined overview montage with samples from all rooms

    Args:
        output_dir: Output directory containing renders
        room_ids: List of room IDs that were rendered
        render_segmentation: Whether segmentation maps were rendered
    """
    output_dir = Path(output_dir)
    summary_dir = output_dir / 'summary'
    summary_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating summary montages...")
    print(f"{'='*60}")

    all_rgb_samples = []
    all_depth_samples = []
    all_seg_samples = []

    for room_id in room_ids:
        # Load all images for this room
        images_dir = output_dir / 'images' / room_id
        depth_dir = output_dir / 'depth' / room_id
        seg_dir = output_dir / 'segmentation' / room_id

        # Get sorted list of RGB images (exclude *_with_edges.png composite images)
        rgb_files = sorted([f for f in images_dir.glob('*.png') if not f.name.endswith('_with_edges.png')])
        if not rgb_files:
            continue

        # Load RGB images
        rgb_images = [Image.open(f) for f in rgb_files]

        # Load depth images (visualization PNGs)
        depth_files = sorted(depth_dir.glob('*_depth.png'))
        depth_images = [Image.open(f).convert('RGB') for f in depth_files]

        # Load segmentation images
        seg_images = []
        if render_segmentation and seg_dir.exists():
            seg_files = sorted(seg_dir.glob('*_seg.png'))
            seg_images = [Image.open(f) for f in seg_files]

        # Create per-room montages
        def create_grid(images, cols=4, thumb_width=480):
            """Create a grid montage from a list of images."""
            if not images:
                return None

            # Calculate thumbnail size maintaining aspect ratio
            orig_w, orig_h = images[0].size
            thumb_height = int(thumb_width * orig_h / orig_w)

            # Resize all images
            thumbs = [img.resize((thumb_width, thumb_height), Image.LANCZOS) for img in images]

            # Calculate grid dimensions
            rows = (len(thumbs) + cols - 1) // cols

            # Create output image
            grid_w = cols * thumb_width
            grid_h = rows * thumb_height
            grid = Image.new('RGB', (grid_w, grid_h), (32, 32, 32))

            # Paste thumbnails
            for idx, thumb in enumerate(thumbs):
                row = idx // cols
                col = idx % cols
                x = col * thumb_width
                y = row * thumb_height
                grid.paste(thumb, (x, y))

            return grid

        # Generate room montages
        cols = min(4, len(rgb_images))

        rgb_grid = create_grid(rgb_images, cols=cols)
        if rgb_grid:
            rgb_grid.save(summary_dir / f'{room_id}_rgb_montage.jpg', quality=90)
            print(f"  Saved {room_id} RGB montage ({len(rgb_images)} images)")

        depth_grid = create_grid(depth_images, cols=cols)
        if depth_grid:
            depth_grid.save(summary_dir / f'{room_id}_depth_montage.jpg', quality=90)
            print(f"  Saved {room_id} depth montage ({len(depth_images)} images)")

        if seg_images:
            seg_grid = create_grid(seg_images, cols=cols)
            if seg_grid:
                seg_grid.save(summary_dir / f'{room_id}_segmentation_montage.jpg', quality=90)
                print(f"  Saved {room_id} segmentation montage ({len(seg_images)} images)")

        # Collect samples for combined overview (first, middle, last from each room)
        for img_list, sample_list in [
            (rgb_images, all_rgb_samples),
            (depth_images, all_depth_samples),
            (seg_images, all_seg_samples),
        ]:
            if img_list:
                for i in [0, len(img_list)//2, -1]:
                    sample_list.append((room_id, img_list[i]))

    # Create combined overview with all render types side by side
    def create_combined_overview(rgb_samples, depth_samples, seg_samples, thumb_width=400):
        """Create a combined overview showing RGB, depth, seg for each sample."""
        if not rgb_samples:
            return None

        orig_w, orig_h = rgb_samples[0][1].size
        thumb_height = int(thumb_width * orig_h / orig_w)

        # Number of columns depends on what's available
        num_types = 1 + (1 if depth_samples else 0) + (1 if seg_samples else 0)
        num_samples = len(rgb_samples)

        grid_w = num_types * thumb_width
        grid_h = num_samples * thumb_height

        grid = Image.new('RGB', (grid_w, grid_h), (32, 32, 32))

        for row, ((room_id, rgb), (_, depth), (_, seg)) in enumerate(
            zip(rgb_samples,
                depth_samples if depth_samples else [(None, None)] * num_samples,
                seg_samples if seg_samples else [(None, None)] * num_samples)):

            y = row * thumb_height
            col = 0

            # RGB
            rgb_thumb = rgb.resize((thumb_width, thumb_height), Image.LANCZOS)
            grid.paste(rgb_thumb, (col * thumb_width, y))
            col += 1

            # Depth
            if depth:
                depth_thumb = depth.resize((thumb_width, thumb_height), Image.LANCZOS)
                grid.paste(depth_thumb, (col * thumb_width, y))
                col += 1

            # Segmentation
            if seg:
                seg_thumb = seg.resize((thumb_width, thumb_height), Image.LANCZOS)
                grid.paste(seg_thumb, (col * thumb_width, y))

        return grid

    # Create the combined overview
    if all_rgb_samples:
        combined = create_combined_overview(
            all_rgb_samples,
            all_depth_samples,
            all_seg_samples if render_segmentation else []
        )
        if combined:
            combined.save(summary_dir / 'combined_overview.jpg', quality=90)
            print(f"  Saved combined overview ({len(all_rgb_samples)} samples)")

    # Create a single "all views" montage for quick inspection
    all_rgb_images = []
    for room_id in room_ids:
        images_dir = output_dir / 'images' / room_id
        rgb_files = sorted([f for f in images_dir.glob('*.png') if not f.name.endswith('_with_edges.png')])
        all_rgb_images.extend([Image.open(f) for f in rgb_files])

    if all_rgb_images:
        # Determine optimal grid layout
        n = len(all_rgb_images)
        cols = int(np.ceil(np.sqrt(n * 16/9)))  # Prefer wider layout
        cols = max(4, min(8, cols))

        def create_grid_simple(images, cols, thumb_width=320):
            if not images:
                return None
            orig_w, orig_h = images[0].size
            thumb_height = int(thumb_width * orig_h / orig_w)
            thumbs = [img.resize((thumb_width, thumb_height), Image.LANCZOS) for img in images]
            rows = (len(thumbs) + cols - 1) // cols
            grid = Image.new('RGB', (cols * thumb_width, rows * thumb_height), (32, 32, 32))
            for idx, thumb in enumerate(thumbs):
                grid.paste(thumb, ((idx % cols) * thumb_width, (idx // cols) * thumb_height))
            return grid

        all_views_grid = create_grid_simple(all_rgb_images, cols=cols)
        if all_views_grid:
            all_views_grid.save(summary_dir / 'all_views.jpg', quality=85)
            print(f"  Saved all_views.jpg ({len(all_rgb_images)} images in {cols}-column grid)")

    print(f"\nSummary montages saved to: {summary_dir}/")
    return summary_dir


def main():
    """Main rendering pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Render multi-view images for 3DGS training with COLMAP export'
    )
    parser.add_argument('--scene-json', default='scene_layout.json',
                       help='Scene JSON file')
    parser.add_argument('--scene-mesh', default='scene_output.glb',
                       help='Scene 3D mesh file (GLB/OBJ)')
    parser.add_argument('--output-dir', default='renders',
                       help='Output directory for renders and COLMAP files')
    parser.add_argument('--num-cameras', type=int, default=16,
                       help='Number of eye-level camera views per room')
    parser.add_argument('--num-overhead', type=int, default=8,
                       help='Number of overhead (elevated) camera views per room')
    parser.add_argument('--width', type=int, default=1024,
                       help='Image width')
    parser.add_argument('--height', type=int, default=1024,
                       help='Image height')
    parser.add_argument('--fov', type=int, default=60,
                       help='Field of view in degrees')
    parser.add_argument('--camera-height', type=float, default=1.6,
                       help='Camera height above floor (meters)')
    parser.add_argument('--wall-offset', type=float, default=0.4,
                       help='Distance from wall for perimeter/overhead cameras (meters, default: 0.4)')
    parser.add_argument('--key-light-intensity', type=float, default=4.0,
                       help='Key light intensity (default: 4.0)')
    parser.add_argument('--fill-light-intensity', type=float, default=2.5,
                       help='Fill light intensity (default: 2.5)')
    parser.add_argument('--ambient-light-intensity', type=float, default=1.5,
                       help='Ambient light intensity (default: 1.5)')
    parser.add_argument('--no-segmentation', action='store_true',
                       help='Disable segmentation map rendering')
    parser.add_argument('--no-edges', action='store_true',
                       help='Disable edge detection output')
    parser.add_argument('--with-objects', action='store_true',
                       help='Include furniture objects in rendering (default: structure only)')
    parser.add_argument('--flat-lighting', action='store_true',
                       help='Use uniform lighting (no shading). Combine with --structure-mesh for wall edge overlay.')
    parser.add_argument('--structure-mesh', default=None,
                       help='Structure-only mesh for wall edge overlay (used with --flat-lighting)')
    parser.add_argument('--no-collision-avoidance', action='store_true',
                       help='Disable camera-object collision avoidance (for debugging)')

    args = parser.parse_args()

    render_segmentation = not args.no_segmentation
    render_edges = not args.no_edges
    include_objects = args.with_objects

    # Load scene data
    print(f"Loading scene from {args.scene_json} and {args.scene_mesh}...")
    scene_data, scene_mesh = load_scene_data(args.scene_json, args.scene_mesh)

    # Filter objects if requested
    if not include_objects:
        print("Filtering out furniture objects...")
        scene_mesh = filter_scene_objects(scene_mesh, scene_data, include_objects=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create segmentation palette and scene
    seg_scene = None
    palette = None
    if render_segmentation:
        print("Creating segmentation palette...")
        palette = create_segmentation_palette(scene_mesh, scene_data)
        print(f"  {len(palette)} classes defined")
        seg_scene = create_segmentation_scene(scene_mesh, palette)

    # Create structure-only segmentation scene for wall edge overlay
    seg_scene_structure = None
    if args.flat_lighting and args.structure_mesh:
        print(f"Loading structure mesh for wall edge overlay: {args.structure_mesh}")
        structure_mesh = trimesh.load(args.structure_mesh)
        structure_palette = create_segmentation_palette(structure_mesh, scene_data)
        seg_scene_structure = create_segmentation_scene(structure_mesh, structure_palette)
        print(f"  {len(structure_palette)} structure classes defined")

    all_image_data = []

    # Process each room
    for room in scene_data['rooms']:
        room_id = room['id']
        print(f"\n{'='*60}")
        print(f"Processing room: {room['name']} ({room_id})")
        print(f"{'='*60}")
        
        # Generate room coverage camera (always first - optimal full-room view)
        coverage_cameras = generate_room_coverage_camera(
            room,
            scene_data['metadata'],
            camera_height=args.camera_height
        )
        # Convert to same format (position, look_at) - strip the camera_type
        coverage_positions = [(pos, look_at) for pos, look_at, _ in coverage_cameras]

        # Generate opposite coverage camera (second - covers the other half)
        opposite_cameras = generate_opposite_coverage_camera(
            room,
            scene_data['metadata'],
            camera_height=args.camera_height
        )
        opposite_positions = [(pos, look_at) for pos, look_at, _ in opposite_cameras]

        # Generate overhead (elevated) cameras
        overhead_cameras = generate_overhead_cameras(
            room,
            scene_data['metadata'],
            num_cameras=args.num_overhead,
            camera_height=args.camera_height,
            wall_offset=args.wall_offset,
        )
        overhead_positions = [(pos, look_at) for pos, look_at, _ in overhead_cameras]

        # Generate regular eye-level camera positions
        regular_positions = generate_camera_positions(
            room,
            scene_data['metadata'],
            num_cameras=args.num_cameras,
            camera_height=args.camera_height,
            wall_offset=args.wall_offset,
        )

        # Drop perimeter cameras that overlap with coverage cameras (within 10°)
        regular_positions = _filter_coverage_overlap(
            coverage_positions + opposite_positions,
            regular_positions,
            room['floor_polygon'],
            min_angle_deg=10,
        )

        # Combine: coverage + opposite coverage first, then regular eye-level, then overhead last
        # Eye-level cameras are processed first so the flux pipeline has good references,
        # overhead cameras are deferred to the end
        camera_positions = coverage_positions + opposite_positions + regular_positions + overhead_positions

        # Avoid cameras inside reconstructed objects (e.g., windows near walls)
        if include_objects and not args.no_collision_avoidance:
            camera_positions = avoid_object_collisions(
                camera_positions, scene_mesh, scene_data, room
            )

        print(f"Generated {len(camera_positions)} camera positions "
              f"(2 coverage + {len(regular_positions)} regular + "
              f"{len(overhead_positions)} overhead [last])")
        
        # Render from each camera position
        image_data, intrinsics = render_scene(
            scene_mesh,
            camera_positions,
            output_dir,
            room_id,
            width=args.width,
            height=args.height,
            fov=args.fov,
            key_light_intensity=args.key_light_intensity,
            fill_light_intensity=args.fill_light_intensity,
            ambient_light_intensity=args.ambient_light_intensity,
            flat_lighting=args.flat_lighting,
            seg_scene=seg_scene,
            seg_scene_structure=seg_scene_structure,
            render_segmentation=render_segmentation,
            render_edges=render_edges
        )

        all_image_data.extend(image_data)
    
    # Export COLMAP format files (combined for all rooms)
    print(f"\n{'='*60}")
    print("Exporting COLMAP format files...")
    print(f"{'='*60}")

    export_colmap_images(output_dir, all_image_data)
    export_colmap_points3d(output_dir)

    # Export segmentation metadata
    if render_segmentation and palette is not None:
        export_segmentation_metadata(output_dir, palette)

    # Save camera positions as JSON for reference
    num_coverage_per_room = 2  # primary + opposite
    num_overhead_per_room = len(overhead_positions)
    num_regular_per_room = args.num_cameras
    camera_info = {
        'intrinsics': {
            'width': args.width,
            'height': args.height,
            'fx': intrinsics[0],
            'fy': intrinsics[1],
            'cx': intrinsics[2],
            'cy': intrinsics[3],
            'fov_deg': args.fov
        },
        'num_images': len(all_image_data),
        'num_rooms': len(scene_data['rooms']),
        'cameras_per_room': {
            'room_coverage': num_coverage_per_room,
            'overhead': num_overhead_per_room,
            'regular': num_regular_per_room,
            'total': num_coverage_per_room + num_overhead_per_room + num_regular_per_room,
            'note': (f'Cameras 0-1 are room coverage (eye-level, opposite walls). '
                     f'Cameras 2-{num_regular_per_room + 1} are regular eye-level multi-view cameras. '
                     f'Last {num_overhead_per_room} cameras are overhead cameras.')
        },
        'segmentation_enabled': render_segmentation,
        'edges_enabled': render_edges,
        'objects_included': include_objects,
        'flat_lighting': args.flat_lighting,
        'structure_mesh': args.structure_mesh
    }

    camera_info_path = output_dir / 'camera_info.json'
    with open(camera_info_path, 'w') as f:
        json.dump(camera_info, f, indent=2)

    print(f"\n{'='*60}")
    print("RENDERING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images rendered: {len(all_image_data)}")
    print(f"Total depth maps: {len(all_image_data)}")
    if render_segmentation:
        print(f"Total segmentation maps: {len(all_image_data)}")
    if render_edges:
        print(f"Total edge maps: {len(all_image_data)}")
    if not include_objects:
        print(f"Objects: EXCLUDED (structure only)")
    if args.flat_lighting:
        print(f"Lighting: FLAT (no shading, uniform base colors)")
        if args.structure_mesh:
            print(f"Wall edge overlay: ENABLED (*_with_edges.png)")
    print(f"Output directory: {output_dir}")
    print(f"\nCOLMAP files:")
    print(f"  - cameras.txt")
    print(f"  - images.txt")
    print(f"  - points3D.txt")
    print(f"  - camera_info.json")
    print(f"\nImages saved in: {output_dir / 'images'}/")
    if args.flat_lighting and args.structure_mesh:
        print(f"  - *_with_edges.png (flat RGB with black wall structure edges)")
    print(f"Depth maps saved in: {output_dir / 'depth'}/")
    print(f"  - PNG format (visualization)")
    print(f"  - NPY format (raw depth for training)")
    if render_segmentation:
        print(f"\nSegmentation maps saved in: {output_dir / 'segmentation'}/")
        print(f"  - PNG format (color-coded by class)")
        print(f"  - segmentation_metadata.json (class labels and colors)")
    if render_edges:
        print(f"\nEdge maps saved in: {output_dir / 'edges'}/")
        print(f"  - *_edges.png (from segmentation boundaries)")
    print(f"\nReady for 3DGS training with depth supervision!")

    # Generate summary montages
    room_ids = [room['id'] for room in scene_data['rooms']]
    create_summary_montage(output_dir, room_ids, render_segmentation)


if __name__ == '__main__':
    main()
