#!/usr/bin/env python3
"""Combine two GLB files into a single GLB file."""

import argparse
import trimesh


def combine_glb_files(input_file1, input_file2, output_file):
    """Load two GLB files and combine them into a single scene."""
    print(f"Loading {input_file1}...")
    scene1 = trimesh.load(input_file1)

    print(f"Loading {input_file2}...")
    scene2 = trimesh.load(input_file2)

    # Convert scenes to Scene objects if they're meshes
    if isinstance(scene1, trimesh.Trimesh):
        scene1 = trimesh.Scene([scene1])
    if isinstance(scene2, trimesh.Trimesh):
        scene2 = trimesh.Scene([scene2])

    # Combine all geometry from both scenes
    combined_meshes = []

    if hasattr(scene1, 'geometry'):
        combined_meshes.extend(scene1.geometry.values())
    else:
        combined_meshes.append(scene1)

    if hasattr(scene2, 'geometry'):
        combined_meshes.extend(scene2.geometry.values())
    else:
        combined_meshes.append(scene2)

    # Create combined scene
    combined_scene = trimesh.Scene(combined_meshes)

    # Export to GLB
    print(f"Exporting combined scene to {output_file}...")
    combined_scene.export(output_file)

    print(f"✓ Successfully combined into {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two GLB files into one")
    parser.add_argument("input1", help="First GLB file")
    parser.add_argument("input2", help="Second GLB file")
    parser.add_argument("--output", "-o", required=True, help="Output GLB file")

    args = parser.parse_args()

    combine_glb_files(args.input1, args.input2, args.output)
