#!/usr/bin/env python3
"""Fill unfilled window openings by cloning existing windows from the scene.

After the object extraction pipeline, some window openings may remain as holes
if no window objects were detected/reconstructed for a room. This script finds
all unfilled window openings across all rooms and fills each one by cloning the
best-matching existing window from anywhere in the scene.

Usage:
    python fill_missing_windows.py \
        --mesh scene_with_all_objects.glb \
        --scene-json scene_layout.json \
        [--output output.glb] \
        [--verbose]
"""

import argparse
import json
import logging
import sys

import numpy as np
import trimesh

from extract_objects.step3_position_merge import (
    clone_window_for_opening,
    load_room_openings,
)

logger = logging.getLogger(__name__)

MATCH_RADIUS_XY = 0.5  # meters
THICKNESS_TOLERANCE = 0.08
OPENING_SIZE_TOLERANCE = 0.20
WINDOW_LABEL_TOKENS = ("window", "cloned_window", "default_window")


def _iter_scene_mesh_entries(scene):
    """Yield scene meshes in world space with labels from nodes and geometry."""
    yielded = set()

    if isinstance(scene, trimesh.Scene):
        for node_name in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node_name]
            geom = scene.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh):
                continue
            mesh_world = geom.copy()
            if transform is not None and not np.allclose(transform, np.eye(4)):
                mesh_world.apply_transform(transform)
            yielded.add((node_name, geom_name))
            yield {
                'node_name': node_name,
                'geom_name': geom_name,
                'label': f"{node_name} {geom_name}".lower(),
                'mesh': mesh_world,
            }

    for geom_name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        if any(saved_geom_name == geom_name for _, saved_geom_name in yielded):
            continue
        yield {
            'node_name': None,
            'geom_name': geom_name,
            'label': str(geom_name).lower(),
            'mesh': geom,
        }


def _mesh_info(entry):
    """Compute bounds-derived metadata for a scene mesh entry."""
    verts = entry['mesh'].vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extents = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2
    return {
        **entry,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'bbox_center': bbox_center,
        'extents': extents,
    }


def _is_window_labeled(label):
    """Return True when a scene label clearly identifies a window mesh."""
    return any(token in label for token in WINDOW_LABEL_TOKENS)


def _matches_opening_spatially(entry, opening):
    """Check proximity and vertical overlap between a mesh and an opening."""
    dist_xy = np.linalg.norm(entry['bbox_center'][:2] - opening['_expected_center'][:2])
    if dist_xy > MATCH_RADIUS_XY:
        return False
    if entry['bbox_max'][2] < opening['z_min'] or entry['bbox_min'][2] > opening['z_max']:
        return False
    return True


def _matches_opening_by_geometry(entry, opening, wall_thickness):
    """Fallback matcher for flattened scenes that lost semantic window labels."""
    horizontal_extents = np.sort(entry['extents'][:2])
    thickness_ok = abs(horizontal_extents[0] - wall_thickness) <= THICKNESS_TOLERANCE
    width_ok = abs(horizontal_extents[1] - opening['width']) <= OPENING_SIZE_TOLERANCE
    height_ok = abs(entry['extents'][2] - opening['height']) <= OPENING_SIZE_TOLERANCE
    return thickness_ok and width_ok and height_ok


def _find_filled_window_for_opening(opening, named_entries, fallback_entries, wall_thickness):
    """Return the best source mesh that already fills the given opening."""
    named_matches = [
        entry for entry in named_entries
        if _matches_opening_spatially(entry, opening)
    ]
    if named_matches:
        named_matches.sort(
            key=lambda entry: np.linalg.norm(
                entry['bbox_center'][:2] - opening['_expected_center'][:2]
            )
        )
        return named_matches[0]

    fallback_matches = [
        entry for entry in fallback_entries
        if _matches_opening_spatially(entry, opening)
        and _matches_opening_by_geometry(entry, opening, wall_thickness)
    ]
    if not fallback_matches:
        return None

    fallback_matches.sort(
        key=lambda entry: np.linalg.norm(
            entry['bbox_center'][:2] - opening['_expected_center'][:2]
        )
    )
    return fallback_matches[0]


def collect_filled_and_unfilled(scene, scene_json_path):
    """Classify all window openings as filled or unfilled.

    For each window opening across all rooms, checks whether any geometry in
    the scene has its bounding-box center close enough to the opening's
    expected 3D center.

    Returns:
        (filled, unfilled) where:
            filled: list of (trimesh.Trimesh, opening_dict)
            unfilled: list of opening_dict
    """
    with open(scene_json_path) as f:
        scene_data = json.load(f)

    wall_thickness = scene_data.get('metadata', {}).get('wall_thickness', 0.15)
    rooms = scene_data.get('rooms', [])

    # Collect all window openings with their expected 3D centers
    all_window_openings = []
    for room in rooms:
        room_id = room.get('id', room.get('name', ''))
        openings = load_room_openings(scene_json_path, room_id)
        for op in openings:
            if op['type'] != 'window':
                continue
            # Expected 3D center of the opening
            center_xy = op['center_2d'] + op['inward_normal'] * (wall_thickness / 2)
            center_z = (op['z_min'] + op['z_max']) / 2
            op['_expected_center'] = np.array([center_xy[0], center_xy[1], center_z])
            op['_room_id'] = room_id
            all_window_openings.append(op)

    if not all_window_openings:
        return [], []

    mesh_entries = [_mesh_info(entry) for entry in _iter_scene_mesh_entries(scene)]
    named_entries = [entry for entry in mesh_entries if _is_window_labeled(entry['label'])]
    fallback_entries = [entry for entry in mesh_entries if not _is_window_labeled(entry['label'])]

    filled = []
    unfilled = []

    for op in all_window_openings:
        matched_entry = _find_filled_window_for_opening(
            op, named_entries, fallback_entries, wall_thickness
        )
        if matched_entry is None:
            unfilled.append(op)
            continue
        filled.append((matched_entry['mesh'], op))

    return filled, unfilled


def pick_best_source(target_opening, filled_windows, used_counts=None):
    """Pick a source window whose aspect ratio matches the target, with variety.

    Ranks candidates by aspect-ratio similarity and then, among those within
    20% of the best score, prefers sources that have been used the fewest
    times so far. This avoids always copying the same window when many
    openings need filling.

    Args:
        target_opening: opening dict to fill
        filled_windows: list of (mesh, source_opening)
        used_counts: optional dict{id(mesh) -> int} tracking reuse; updated
                     in-place when a source is chosen.
    """
    target_ratio = target_opening['width'] / max(target_opening['height'], 1e-6)
    scored = []
    for mesh, op in filled_windows:
        src_ratio = op['width'] / max(op['height'], 1e-6)
        diff = abs(src_ratio - target_ratio)
        scored.append((diff, mesh, op))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    best_diff = scored[0][0]
    tol = max(0.2, best_diff * 0.2)
    close = [s for s in scored if s[0] <= best_diff + tol]

    if used_counts is not None:
        close.sort(key=lambda s: (used_counts.get(id(s[1]), 0), s[0]))
        choice = close[0]
        used_counts[id(choice[1])] = used_counts.get(id(choice[1]), 0) + 1
    else:
        choice = close[0]

    return (choice[1], choice[2])


def fill_missing_windows(mesh_path, scene_json_path, output_path=None,
                         verbose=False):
    """Fill unfilled window openings by cloning existing windows.

    Args:
        mesh_path: Path to the scene GLB file
        scene_json_path: Path to the scene JSON
        output_path: Output path (defaults to overwriting mesh_path)
        verbose: Enable detailed logging
    """
    if output_path is None:
        output_path = mesh_path

    scene = trimesh.load(mesh_path)
    if not isinstance(scene, trimesh.Scene):
        logger.warning("Loaded mesh is not a Scene, skipping window fill")
        return

    with open(scene_json_path) as f:
        scene_data = json.load(f)
    wall_thickness = scene_data.get('metadata', {}).get('wall_thickness', 0.15)

    filled, unfilled = collect_filled_and_unfilled(scene, scene_json_path)

    if verbose:
        logger.info(f"Window openings: {len(filled)} filled, {len(unfilled)} unfilled")

    if not unfilled:
        logger.info("All window openings are filled, nothing to do")
        return

    if not filled:
        logger.warning("No filled windows found in scene, cannot clone")
        return

    added = 0
    used_counts = {}
    for idx, target_op in enumerate(unfilled):
        source_mesh, source_op = pick_best_source(
            target_op, filled, used_counts=used_counts
        )
        cloned = clone_window_for_opening(
            source_mesh, source_op, target_op, wall_thickness
        )
        room_id = target_op.get('_room_id', 'unknown')
        geom_name = f"default_window_{room_id}_{idx:02d}"
        scene.add_geometry(cloned, node_name=geom_name, geom_name=geom_name)
        added += 1

        if verbose:
            logger.info(
                f"Cloned window -> {geom_name} at "
                f"[{target_op['center_2d'][0]:.2f}, "
                f"{target_op['center_2d'][1]:.2f}]"
            )

    if added > 0:
        scene.export(output_path)
        logger.info(f"Added {added} cloned windows, saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill unfilled window openings by cloning existing windows"
    )
    parser.add_argument("--mesh", required=True, help="Path to scene GLB mesh")
    parser.add_argument("--scene-json", required=True, help="Path to scene JSON")
    parser.add_argument("--output", default=None,
                        help="Output path (defaults to overwriting --mesh)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    fill_missing_windows(
        mesh_path=args.mesh,
        scene_json_path=args.scene_json,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
