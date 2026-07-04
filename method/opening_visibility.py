"""
Opening visibility detection for Flux prompts.

Detects which doors and windows are visible from a camera position and classifies
them by wall direction (left/opposite/right) relative to camera view direction.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraPose:
    """Camera pose from COLMAP images.txt."""
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float

    @property
    def quaternion(self) -> np.ndarray:
        """Get quaternion as numpy array [qw, qx, qy, qz]."""
        return np.array([self.qw, self.qx, self.qy, self.qz])

    @property
    def translation(self) -> np.ndarray:
        """Get translation as numpy array [tx, ty, tz]."""
        return np.array([self.tx, self.ty, self.tz])


@dataclass
class WallSegment:
    """A wall segment defined by two floor polygon vertices."""
    start: np.ndarray  # [x, y]
    end: np.ndarray    # [x, y]
    inward_normal: np.ndarray  # [x, y] pointing into the room

    @property
    def center(self) -> np.ndarray:
        """Get center point of wall segment."""
        return (self.start + self.end) / 2

    @property
    def length(self) -> float:
        """Get length of wall segment."""
        return np.linalg.norm(self.end - self.start)


@dataclass
class Opening:
    """An opening (door or window) in a wall."""
    opening_type: str  # "door" or "window"
    wall_segment: List[List[float]]  # [[x1, y1], [x2, y2]]
    position: float  # distance along wall from start
    width: float
    height: float
    sill_height: Optional[float] = None  # only for windows


def extract_camera_pose(images_txt_path: Path, camera_name: str) -> Optional[CameraPose]:
    """
    Parse COLMAP images.txt to extract pose for a specific camera.

    Args:
        images_txt_path: Path to images.txt file
        camera_name: Camera name (e.g., "living_room_0000")

    Returns:
        CameraPose if found, None otherwise
    """
    if not images_txt_path.exists():
        return None

    with open(images_txt_path, 'r') as f:
        lines = f.readlines()

    # Skip comment lines and find matching camera
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 10:
            # Image line format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            name = parts[9]
            # Name format: "room_id/room_id_0000.png"
            stem = Path(name).stem  # "room_id_0000"

            if stem == camera_name:
                try:
                    return CameraPose(
                        qw=float(parts[1]),
                        qx=float(parts[2]),
                        qy=float(parts[3]),
                        qz=float(parts[4]),
                        tx=float(parts[5]),
                        ty=float(parts[6]),
                        tz=float(parts[7]),
                    )
                except (ValueError, IndexError):
                    pass

    return None


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix.

    Args:
        quat: Quaternion as [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = quat / np.linalg.norm(quat)

    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])

    return R


def extract_camera_view_direction(pose: CameraPose) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract camera position and view directions from COLMAP pose.

    COLMAP stores world-to-camera transform:
    - R_w2c: quaternion represents rotation from world to camera
    - t_w2c: translation in camera frame

    Camera position in world: camera_pos = -R_c2w @ t_w2c = -R_w2c.T @ t_w2c
    Camera forward in OpenGL convention: -Z axis in camera space

    Args:
        pose: Camera pose from COLMAP

    Returns:
        Tuple of (camera_position, forward_2d, right_2d) where:
        - camera_position: [x, y, z] in world coordinates
        - forward_2d: [x, y] normalized forward direction projected to XY plane
        - right_2d: [x, y] normalized right direction projected to XY plane
    """
    # Get rotation matrix (world-to-camera)
    R_w2c = quaternion_to_rotation_matrix(pose.quaternion)

    # Camera-to-world rotation is the transpose
    R_c2w = R_w2c.T

    # Camera position in world coordinates
    camera_pos = -R_c2w @ pose.translation

    # Forward direction in camera space is -Z (OpenGL convention)
    forward_camera = np.array([0.0, 0.0, -1.0])

    # Transform to world space
    forward_world = R_c2w @ forward_camera

    # Project to XY plane and normalize
    forward_2d = forward_world[:2]
    forward_2d_norm = np.linalg.norm(forward_2d)
    if forward_2d_norm > 1e-6:
        forward_2d = forward_2d / forward_2d_norm
    else:
        # Camera looking straight up/down, use X direction as fallback
        forward_2d = np.array([1.0, 0.0])

    # Right direction is 90 degrees clockwise from forward (in XY plane)
    # If forward is [fx, fy], right is [fy, -fx]
    right_2d = np.array([forward_2d[1], -forward_2d[0]])

    return camera_pos, forward_2d, right_2d


def get_wall_segments(floor_polygon: List[List[float]]) -> List[WallSegment]:
    """
    Extract wall segments from a room's floor polygon.

    Assumes counter-clockwise polygon ordering (when viewed from above),
    meaning the inward normal points to the right of each edge direction.

    Args:
        floor_polygon: List of [x, y] vertices defining the room boundary

    Returns:
        List of WallSegment objects
    """
    walls = []
    n = len(floor_polygon)

    for i in range(n):
        start = np.array(floor_polygon[i], dtype=float)
        end = np.array(floor_polygon[(i + 1) % n], dtype=float)

        # Edge direction
        edge_dir = end - start
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-6:
            continue

        edge_dir = edge_dir / edge_len

        # Inward normal is perpendicular to edge, pointing into the room
        # For counter-clockwise polygon: rotate edge direction 90 degrees counter-clockwise
        # This gives [-edge_dir_y, edge_dir_x]
        inward_normal = np.array([-edge_dir[1], edge_dir[0]])

        walls.append(WallSegment(
            start=start,
            end=end,
            inward_normal=inward_normal,
        ))

    return walls


def is_wall_visible(
    wall: WallSegment,
    camera_pos_2d: np.ndarray,
    forward_2d: np.ndarray
) -> bool:
    """
    Check if a wall is visible from the camera.

    A wall is visible if:
    1. Camera is on the inward side of the wall (can see the inside face)
    2. Camera is looking toward the wall (not completely away from it)

    Args:
        wall: Wall segment to check
        camera_pos_2d: Camera position [x, y] in XY plane
        forward_2d: Camera forward direction [x, y] normalized

    Returns:
        True if wall is visible
    """
    # Vector from wall center to camera
    to_camera = camera_pos_2d - wall.center

    # Camera must be on the inward side (positive dot product with inward normal)
    if np.dot(to_camera, wall.inward_normal) <= 0:
        return False

    # Camera must be looking toward the wall face
    # The wall face normal points inward, so camera should look opposite to it
    # (i.e., camera forward should have negative dot product with inward normal)
    # Use threshold > 0.3 to include side walls (perpendicular, dot=0) while
    # excluding walls mostly behind the camera
    if np.dot(forward_2d, wall.inward_normal) > 0.3:
        return False

    return True


def classify_wall_direction(
    wall: WallSegment,
    camera_pos_2d: np.ndarray,
    forward_2d: np.ndarray,
    right_2d: np.ndarray
) -> str:
    """
    Classify wall direction relative to camera view based on wall orientation.

    Uses the wall's inward normal (not the direction to its center) to determine
    if the wall is opposite, left, or right of the camera view.

    Args:
        wall: Wall segment with inward_normal attribute
        camera_pos_2d: Camera position [x, y] (unused but kept for API compatibility)
        forward_2d: Camera forward direction [x, y]
        right_2d: Camera right direction [x, y]

    Returns:
        "left", "opposite", or "right"
    """
    # Use wall's inward normal for orientation-based classification
    forward_dot = np.dot(forward_2d, wall.inward_normal)

    # If forward · inward_normal < -0.7 (within ~45° of directly facing), it's opposite
    if forward_dot < -0.7:
        return "opposite"

    # For side walls, use right direction to determine left vs right
    # If right · inward_normal > 0, wall normal points toward camera's right,
    # which means the wall itself is on the LEFT side of the view
    right_dot = np.dot(right_2d, wall.inward_normal)

    if right_dot > 0:
        return "left"
    else:
        return "right"


def wall_segment_matches(
    wall: WallSegment,
    opening_segment: List[List[float]],
    tolerance: float = 0.01
) -> bool:
    """
    Check if a wall segment matches an opening's wall_segment definition.

    The opening's wall_segment is defined as [[x1, y1], [x2, y2]] representing
    the wall edge. We need to match this to our WallSegment, accounting for
    possible direction reversal.

    Args:
        wall: WallSegment to check
        opening_segment: Opening's wall_segment [[x1, y1], [x2, y2]]
        tolerance: Distance tolerance for matching

    Returns:
        True if wall matches the opening's wall_segment
    """
    seg_start = np.array(opening_segment[0], dtype=float)
    seg_end = np.array(opening_segment[1], dtype=float)

    # Check both directions
    dist_forward = max(
        np.linalg.norm(wall.start - seg_start),
        np.linalg.norm(wall.end - seg_end)
    )
    dist_reverse = max(
        np.linalg.norm(wall.start - seg_end),
        np.linalg.norm(wall.end - seg_start)
    )

    return min(dist_forward, dist_reverse) < tolerance


def find_openings_on_wall(
    openings: List[Opening],
    wall: WallSegment
) -> List[Opening]:
    """
    Find all openings that are on a specific wall segment.

    Args:
        openings: List of all openings in the room
        wall: Wall segment to check

    Returns:
        List of openings on this wall
    """
    matched = []
    for opening in openings:
        if wall_segment_matches(wall, opening.wall_segment):
            matched.append(opening)
    return matched


def find_door_destination(
    opening_wall_segment: List[List[float]],
    current_room_id: str,
    all_rooms: List[Dict]
) -> Optional[str]:
    """
    Find which room a door leads to by checking for shared wall segments.

    A door connects two rooms if its wall_segment matches an edge in another
    room's floor_polygon.

    Args:
        opening_wall_segment: The door's wall_segment [[x1, y1], [x2, y2]]
        current_room_id: ID of the room containing the door
        all_rooms: List of all room definitions from scene JSON

    Returns:
        Name of destination room, or None if external door
    """
    seg_start = np.array(opening_wall_segment[0], dtype=float)
    seg_end = np.array(opening_wall_segment[1], dtype=float)
    tolerance = 0.01

    for room in all_rooms:
        if room['id'] == current_room_id:
            continue

        polygon = room['floor_polygon']
        n = len(polygon)

        for i in range(n):
            room_start = np.array(polygon[i], dtype=float)
            room_end = np.array(polygon[(i + 1) % n], dtype=float)

            # Check if segments match (either direction)
            dist_forward = max(
                np.linalg.norm(seg_start - room_start),
                np.linalg.norm(seg_end - room_end)
            )
            dist_reverse = max(
                np.linalg.norm(seg_start - room_end),
                np.linalg.norm(seg_end - room_start)
            )

            if min(dist_forward, dist_reverse) < tolerance:
                return room.get('name', room['id'])

    return None  # External door


def find_adjacent_room_openings(
    wall: WallSegment,
    current_room_id: str,
    all_rooms: List[Dict]
) -> List[Tuple[str, str]]:
    """
    Find openings from adjacent rooms that are on a shared wall.

    When two rooms share a wall, an opening (door/window) may only be defined
    in one room's openings list. This function finds such openings from
    the adjacent room's perspective.

    Args:
        wall: Wall segment from current room
        current_room_id: ID of the room we're viewing from
        all_rooms: List of all room definitions

    Returns:
        List of (opening_type, adjacent_room_name) tuples
    """
    tolerance = 0.01
    results = []

    for room in all_rooms:
        if room['id'] == current_room_id:
            continue

        polygon = room['floor_polygon']
        n = len(polygon)

        # Check if this room shares our wall
        for i in range(n):
            room_start = np.array(polygon[i], dtype=float)
            room_end = np.array(polygon[(i + 1) % n], dtype=float)

            # Check if wall segments match (either direction)
            dist_forward = max(
                np.linalg.norm(wall.start - room_start),
                np.linalg.norm(wall.end - room_end)
            )
            dist_reverse = max(
                np.linalg.norm(wall.start - room_end),
                np.linalg.norm(wall.end - room_start)
            )

            if min(dist_forward, dist_reverse) < tolerance:
                # This room shares our wall - check its openings
                for o in room.get('openings', []):
                    opening_segment = o.get('wall_segment', [])
                    if not opening_segment or len(opening_segment) != 2:
                        continue

                    # Check if the opening is on this shared wall
                    o_start = np.array(opening_segment[0], dtype=float)
                    o_end = np.array(opening_segment[1], dtype=float)

                    o_dist_forward = max(
                        np.linalg.norm(o_start - room_start),
                        np.linalg.norm(o_end - room_end)
                    )
                    o_dist_reverse = max(
                        np.linalg.norm(o_start - room_end),
                        np.linalg.norm(o_end - room_start)
                    )

                    if min(o_dist_forward, o_dist_reverse) < tolerance:
                        opening_type = o.get('type', 'door')
                        room_name = room.get('name', room['id'])
                        results.append((opening_type, room_name))

                return results  # Found the shared wall, no need to continue

    return results


def get_visible_openings_description(
    scene_json: Dict,
    room_id: str,
    camera_name: str,
    images_txt_path: Path,
    include_destinations: bool = True,
) -> str:
    """
    Generate a description of visible openings for Flux prompts.

    Args:
        scene_json: Full scene JSON dictionary
        room_id: Current room ID (e.g., "living_room")
        camera_name: Camera name (e.g., "living_room_0000")
        images_txt_path: Path to COLMAP images.txt file
        include_destinations: If False, doors are described as "open door"
            instead of "open door to {destination}"

    Returns:
        Description like "window on the left wall, door to Kitchen on the opposite wall"
        Returns empty string if no openings are visible.
    """
    # Find the room
    room = None
    for r in scene_json.get('rooms', []):
        if r['id'] == room_id:
            room = r
            break

    if room is None:
        return ""

    # Get camera pose
    pose = extract_camera_pose(images_txt_path, camera_name)
    if pose is None:
        return ""

    # Extract camera view direction
    camera_pos, forward_2d, right_2d = extract_camera_view_direction(pose)
    camera_pos_2d = camera_pos[:2]

    # Get wall segments
    floor_polygon = room.get('floor_polygon', [])
    if len(floor_polygon) < 3:
        return ""

    walls = get_wall_segments(floor_polygon)

    # Parse openings
    raw_openings = room.get('openings', [])
    openings = []
    for o in raw_openings:
        openings.append(Opening(
            opening_type=o.get('type', 'door'),
            wall_segment=o.get('wall_segment', []),
            position=o.get('position', 0),
            width=o.get('width', 1.0),
            height=o.get('height', 2.0),
            sill_height=o.get('sill_height'),
        ))

    # Collect visible openings by direction
    # Structure: {direction: [(type, destination_or_None), ...]}
    visible_by_direction: Dict[str, List[Tuple[str, Optional[str]]]] = {
        'left': [],
        'opposite': [],
        'right': [],
    }

    all_rooms = scene_json.get('rooms', [])

    for wall in walls:
        if not is_wall_visible(wall, camera_pos_2d, forward_2d):
            continue

        direction = classify_wall_direction(wall, camera_pos_2d, forward_2d, right_2d)
        wall_openings = find_openings_on_wall(openings, wall)

        # Also check for openings from adjacent rooms on shared walls
        adjacent_openings = find_adjacent_room_openings(wall, room_id, all_rooms)

        for opening in wall_openings:
            destination = None
            if opening.opening_type == 'door':
                destination = find_door_destination(
                    opening.wall_segment,
                    room_id,
                    all_rooms
                )

            visible_by_direction[direction].append(
                (opening.opening_type, destination)
            )

        # Add openings from adjacent rooms (destination is the adjacent room name)
        for opening_type, adjacent_room_name in adjacent_openings:
            visible_by_direction[direction].append(
                (opening_type, adjacent_room_name)
            )

    # Build description string
    parts = []

    for direction in ['left', 'opposite', 'right']:
        items = visible_by_direction[direction]
        if not items:
            continue

        # Group by type
        windows = [i for i in items if i[0] == 'window']
        doors = [i for i in items if i[0] == 'door']

        direction_parts = []

        # Windows
        if len(windows) == 1:
            direction_parts.append("window")
        elif len(windows) > 1:
            direction_parts.append(f"{len(windows)} windows")

        # Doors
        for _, destination in doors:
            if destination and include_destinations:
                direction_parts.append(f"open door to {destination}")
            else:
                direction_parts.append("open door")

        if direction_parts:
            items_str = " and ".join(direction_parts)
            parts.append(f"{items_str} on the {direction} wall")

    return ", ".join(parts)


def get_visible_windows_description(
    scene_json: Dict,
    room_id: str,
    camera_name: str,
    images_txt_path: Path,
) -> str:
    """
    Generate window-only description for Qwen prompts.

    Similar to get_visible_openings_description but:
    1. Filters to only windows (excludes doors)
    2. Uses plural "windows" when count > 1 for a wall
    3. Format: "{windows} on the {direction} wall" joined by ", " ending with "."

    Args:
        scene_json: Full scene JSON dictionary
        room_id: Current room ID (e.g., "living_room")
        camera_name: Camera name (e.g., "living_room_0000")
        images_txt_path: Path to COLMAP images.txt file

    Returns:
        Description like "window on the left wall." or
        "windows on the left wall, window on the opposite wall."
        Empty string if no windows visible.
    """
    # Find the room
    room = None
    for r in scene_json.get('rooms', []):
        if r['id'] == room_id:
            room = r
            break

    if room is None:
        return ""

    # Get camera pose
    pose = extract_camera_pose(images_txt_path, camera_name)
    if pose is None:
        return ""

    # Extract camera view direction
    camera_pos, forward_2d, right_2d = extract_camera_view_direction(pose)
    camera_pos_2d = camera_pos[:2]

    # Get wall segments
    floor_polygon = room.get('floor_polygon', [])
    if len(floor_polygon) < 3:
        return ""

    walls = get_wall_segments(floor_polygon)

    # Parse openings
    raw_openings = room.get('openings', [])
    openings = []
    for o in raw_openings:
        openings.append(Opening(
            opening_type=o.get('type', 'door'),
            wall_segment=o.get('wall_segment', []),
            position=o.get('position', 0),
            width=o.get('width', 1.0),
            height=o.get('height', 2.0),
            sill_height=o.get('sill_height'),
        ))

    # Count visible windows by direction
    # Structure: {direction: window_count}
    windows_by_direction: Dict[str, int] = {
        'left': 0,
        'opposite': 0,
        'right': 0,
    }

    all_rooms = scene_json.get('rooms', [])

    for wall in walls:
        if not is_wall_visible(wall, camera_pos_2d, forward_2d):
            continue

        direction = classify_wall_direction(wall, camera_pos_2d, forward_2d, right_2d)
        wall_openings = find_openings_on_wall(openings, wall)

        # Count windows on this wall
        for opening in wall_openings:
            if opening.opening_type == 'window':
                windows_by_direction[direction] += 1

        # Also check for windows from adjacent rooms on shared walls
        adjacent_openings = find_adjacent_room_openings(wall, room_id, all_rooms)
        for opening_type, _ in adjacent_openings:
            if opening_type == 'window':
                windows_by_direction[direction] += 1

    # Build description string
    parts = []

    for direction in ['left', 'opposite', 'right']:
        count = windows_by_direction[direction]
        if count == 0:
            continue
        elif count == 1:
            parts.append(f"window on the {direction} wall")
        else:
            parts.append(f"windows on the {direction} wall")

    if not parts:
        return ""

    return ", ".join(parts) + "."


def generate_prompts_for_room(
    scene_json: Dict,
    room_id: str,
    images_txt_path: Path,
    num_cameras: int,
    base_prompt: str = "",  # No longer used, kept for backward compatibility
    include_destinations: bool = True,
) -> Dict[str, str]:
    """
    Generate per-camera opening descriptions for a room.

    Note: This function now returns only opening descriptions, not full prompts.
    The base_prompt parameter is kept for backward compatibility but is ignored.
    Each method in the pipeline constructs the full prompt differently:
    - _generate_initial(): "Follow... {base_prompt} {opening_desc}"
    - _generate_iterative(): "generate reference_image2... {opening_desc}"

    Args:
        scene_json: Full scene JSON dictionary
        room_id: Room ID (e.g., "living_room")
        images_txt_path: Path to COLMAP images.txt file
        num_cameras: Number of cameras to generate prompts for
        base_prompt: Deprecated, kept for backward compatibility
        include_destinations: If False, door descriptions omit destination room names

    Returns:
        Dict mapping camera_name -> opening description (not full prompt)
    """
    prompts = {}

    for camera_idx in range(num_cameras):
        camera_name = f"{room_id}_{camera_idx:04d}"

        opening_desc = get_visible_openings_description(
            scene_json,
            room_id,
            camera_name,
            images_txt_path,
            include_destinations=include_destinations,
        )

        # Return just the opening description (can be empty string)
        prompts[camera_name] = opening_desc or ""

    return prompts


# Convenience functions for testing
def _test_with_scene(scene_json_path: str, images_txt_path: str, room_id: str, camera_name: str):
    """Test the opening visibility detection with a scene file."""
    with open(scene_json_path) as f:
        scene = json.load(f)

    desc = get_visible_openings_description(
        scene,
        room_id,
        camera_name,
        Path(images_txt_path),
    )

    print(f"Room: {room_id}")
    print(f"Camera: {camera_name}")
    print(f"Description: {desc if desc else '(no visible openings)'}")
    return desc


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python opening_visibility.py <scene.json> <images.txt> <room_id> <camera_name>")
        print("Example: python opening_visibility.py scene_layout.json renders/images.txt living_room living_room_0000")
        sys.exit(1)

    _test_with_scene(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
