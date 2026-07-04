#!/usr/bin/env python3
"""
Replace reconstructed object meshes with their axis-aligned bounding boxes.

Takes a scene mesh (with reconstructed objects) and a structure-only mesh,
identifies object geometries (present in full scene but not in structure),
and replaces each with an AABB box of the same color.

Usage:
    python create_bbox_scene.py \
        --scene-mesh scene_with_all_objects.glb \
        --structure-mesh structure_only.glb \
        --output bbox_scene.glb
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def create_bbox_scene(scene_mesh_path: Path, structure_mesh_path: Path, output_path: Path) -> bool:
    """Replace object meshes with AABB boxes, keeping structure unchanged.

    Args:
        scene_mesh_path: Path to scene with reconstructed objects.
        structure_mesh_path: Path to structure-only mesh.
        output_path: Where to write the resulting GLB.

    Returns:
        True on success.
    """
    scene = trimesh.load(str(scene_mesh_path))
    structure = trimesh.load(str(structure_mesh_path))

    # Handle both Scene and single-mesh loads
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(geometry={'mesh': scene})
    if isinstance(structure, trimesh.Trimesh):
        structure = trimesh.Scene(geometry={'mesh': structure})

    structure_names = set(structure.geometry.keys())
    scene_names = set(scene.geometry.keys())
    object_names = scene_names - structure_names

    if not object_names:
        print("No object geometries found (scene and structure have identical geometry names).")
        print("Copying scene mesh as-is.")
        import shutil
        shutil.copy(str(scene_mesh_path), str(output_path))
        return True

    print(f"Found {len(object_names)} object geometries to replace with AABBs")

    # Build a new scene starting from the structure geometries
    new_scene = trimesh.Scene()

    # Copy structure geometries with their transforms
    for name in sorted(structure_names):
        if name in scene.geometry:
            geom = scene.geometry[name]
            transform = scene.graph.get(name)[0] if name in scene.graph else np.eye(4)
            # Get transform from scene graph
            try:
                transform = scene.graph[name][0]
            except (KeyError, TypeError):
                transform = np.eye(4)
            new_scene.add_geometry(geom, node_name=name, geom_name=name, transform=transform)

    # Replace each object geometry with its AABB box
    for name in sorted(object_names):
        geom = scene.geometry[name]

        # Get the transform for this geometry in the scene graph
        try:
            transform = scene.graph[name][0]
        except (KeyError, TypeError):
            transform = np.eye(4)

        # Compute world-space vertices
        vertices = geom.vertices.copy()
        if not np.allclose(transform, np.eye(4)):
            vertices_h = np.column_stack([vertices, np.ones(len(vertices))])
            vertices = (transform @ vertices_h.T).T[:, :3]

        # Compute AABB
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        extents = bbox_max - bbox_min
        center = (bbox_min + bbox_max) / 2.0

        # Clamp minimum extents to avoid degenerate boxes
        extents = np.maximum(extents, 0.01)

        # Create box mesh at origin, then position at AABB center
        box = trimesh.creation.box(extents=extents)

        # Copy visual color from original geometry
        color = _extract_color(geom)
        box.visual = trimesh.visual.ColorVisuals(
            mesh=box,
            face_colors=np.tile(color, (len(box.faces), 1))
        )

        # Place box at the AABB center (world space, no additional transform needed)
        box_transform = np.eye(4)
        box_transform[:3, 3] = center

        new_scene.add_geometry(box, node_name=name, geom_name=name, transform=box_transform)
        print(f"  {name}: extents={extents.round(3)}, center={center.round(3)}, color={color}")

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_scene.export(str(output_path))
    print(f"Saved bbox scene to {output_path}")
    return True


def _extract_color(geom: trimesh.Trimesh) -> np.ndarray:
    """Extract a representative RGBA color from a mesh geometry."""
    try:
        if hasattr(geom.visual, 'face_colors') and geom.visual.face_colors is not None:
            colors = geom.visual.face_colors
            if len(colors) > 0:
                return colors.mean(axis=0).astype(np.uint8)
    except Exception:
        pass

    try:
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            colors = geom.visual.vertex_colors
            if len(colors) > 0:
                return colors.mean(axis=0).astype(np.uint8)
    except Exception:
        pass

    try:
        if hasattr(geom.visual, 'main_color'):
            c = geom.visual.main_color
            if c is not None:
                return np.array(c, dtype=np.uint8)
    except Exception:
        pass

    # Fallback: gray
    return np.array([180, 180, 180, 255], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Replace reconstructed objects with AABB boxes"
    )
    parser.add_argument(
        "--scene-mesh", type=Path, required=True,
        help="Path to scene mesh with reconstructed objects"
    )
    parser.add_argument(
        "--structure-mesh", type=Path, required=True,
        help="Path to structure-only mesh"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for bbox scene"
    )

    args = parser.parse_args()

    if not args.scene_mesh.exists():
        print(f"Error: Scene mesh not found: {args.scene_mesh}", file=sys.stderr)
        return 1
    if not args.structure_mesh.exists():
        print(f"Error: Structure mesh not found: {args.structure_mesh}", file=sys.stderr)
        return 1

    success = create_bbox_scene(args.scene_mesh, args.structure_mesh, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
