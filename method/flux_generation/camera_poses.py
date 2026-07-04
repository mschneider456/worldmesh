"""COLMAP images.txt parser and quaternion utilities."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CameraPose:
    """Camera pose from COLMAP images.txt."""

    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str

    @property
    def quaternion(self) -> np.ndarray:
        """Get quaternion as numpy array [qw, qx, qy, qz]."""
        return np.array([self.qw, self.qx, self.qy, self.qz])

    @property
    def translation(self) -> np.ndarray:
        """Get translation as numpy array [tx, ty, tz]."""
        return np.array([self.tx, self.ty, self.tz])

    @property
    def room_name(self) -> str:
        """Extract room name from image path."""
        # Name is like "living_room/living_room_0000.png"
        parts = self.name.split("/")
        if len(parts) >= 1:
            return parts[0]
        return ""

    @property
    def image_index(self) -> int:
        """Extract image index from filename."""
        # Name is like "living_room/living_room_0000.png"
        filename = Path(self.name).stem  # "living_room_0000"
        match = re.search(r"_(\d+)$", filename)
        if match:
            return int(match.group(1))
        return -1


def parse_images_txt(images_txt_path: Path) -> Dict[str, Dict[int, CameraPose]]:
    """
    Parse COLMAP images.txt file.

    Returns:
        Dict mapping room_name -> {image_index -> CameraPose}
    """
    poses_by_room: Dict[str, Dict[int, CameraPose]] = {}

    with open(images_txt_path, "r") as f:
        lines = f.readlines()

    # Skip comment lines and empty lines (empty lines are the POINTS2D data)
    data_lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]

    # Process each image data line
    for line in data_lines:
        parts = line.split()

        if len(parts) >= 10:
            try:
                pose = CameraPose(
                    image_id=int(parts[0]),
                    qw=float(parts[1]),
                    qx=float(parts[2]),
                    qy=float(parts[3]),
                    qz=float(parts[4]),
                    tx=float(parts[5]),
                    ty=float(parts[6]),
                    tz=float(parts[7]),
                    camera_id=int(parts[8]),
                    name=parts[9],
                )

                room = pose.room_name
                if room not in poses_by_room:
                    poses_by_room[room] = {}

                poses_by_room[room][pose.image_index] = pose

            except (ValueError, IndexError):
                pass  # Skip malformed lines

    return poses_by_room


def quaternion_similarity(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute quaternion similarity using absolute dot product.

    Handles the double-cover property of quaternions (q and -q represent
    the same rotation).

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]

    Returns:
        Similarity score in [0, 1], where 1 means identical rotation
    """
    # Normalize quaternions
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)

    # Use absolute dot product to handle double-cover
    return abs(np.dot(q1_norm, q2_norm))


def find_best_reference(
    target_id: int,
    successful_ids: List[int],
    poses: Dict[int, CameraPose],
) -> Optional[int]:
    """
    Find the successful generation with the most similar camera rotation.

    Args:
        target_id: The image ID we're trying to generate
        successful_ids: List of already successfully generated image IDs
        poses: Dict mapping image_id -> CameraPose

    Returns:
        The ID of the best reference image, or None if no successful IDs
    """
    if not successful_ids or target_id not in poses:
        return None

    target_quat = poses[target_id].quaternion
    best_id = None
    best_sim = -1.0

    for ref_id in successful_ids:
        if ref_id not in poses:
            continue

        sim = quaternion_similarity(target_quat, poses[ref_id].quaternion)
        if sim > best_sim:
            best_id = ref_id
            best_sim = sim

    return best_id


def order_by_rotational_proximity(
    seed_ids: List[int],
    candidate_ids: List[int],
    poses: Dict[int, CameraPose],
) -> List[int]:
    """
    Order candidates by greedy nearest-neighbor from seed cameras.

    Starting from seed_ids as the "generated" set, repeatedly pick the
    candidate most rotationally similar to any camera in the generated set.
    This produces a gradual outward spiral from seed viewpoints.

    Candidates without pose data are appended at the end.
    """
    generated = set(seed_ids)
    remaining = [c for c in candidate_ids if c not in generated and c in poses]
    no_pose = [c for c in candidate_ids if c not in generated and c not in poses]
    ordered = []

    while remaining:
        best_id = None
        best_sim = -1.0
        for cid in remaining:
            for gid in generated:
                if gid not in poses:
                    continue
                sim = quaternion_similarity(poses[cid].quaternion, poses[gid].quaternion)
                if sim > best_sim:
                    best_sim = sim
                    best_id = cid
        if best_id is None:
            break
        ordered.append(best_id)
        generated.add(best_id)
        remaining.remove(best_id)

    return ordered + remaining + no_pose


def get_poses_for_room(
    images_txt_path: Path, room_name: str
) -> Dict[int, CameraPose]:
    """
    Get camera poses for a specific room.

    Args:
        images_txt_path: Path to images.txt
        room_name: Name of the room (e.g., "master_bedroom")

    Returns:
        Dict mapping image_index -> CameraPose for that room
    """
    all_poses = parse_images_txt(images_txt_path)
    return all_poses.get(room_name, {})
