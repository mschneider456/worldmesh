# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Multi-object 3D scene reconstruction from a single image.

This script:
1. Loads an image and all its object masks
2. Reconstructs each object using SAM 3D Objects
3. Transforms each object to scene coordinates using estimated pose
4. Combines all objects into a single coherent 3D scene

The output scene, when viewed from the original camera position, should
resemble the input image with all objects correctly aligned.
"""
import sys
import os
import gc
from datetime import datetime
import numpy as np
import torch
import trimesh
from copy import deepcopy

# Add notebook to path for inference utilities
sys.path.append("notebook")
from inference import (
    Inference,
    load_image,
    load_masks,
    make_scene,
    ready_gaussian_for_video_rendering,
)

# Import transformation utilities from SAM 3D Objects
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform


def transform_mesh_to_scene(vertices, faces, rotation_quat, translation, scale):
    """
    Transform mesh vertices from local object space to scene/camera space.

    Uses the same transformation as SceneVisualizer.object_pointcloud:
    - Scale -> Rotate -> Translate

    Args:
        vertices: (N, 3) tensor of vertex positions in local coordinates
        faces: (M, 3) tensor of face indices
        rotation_quat: (1, 4) quaternion [w, x, y, z] for local-to-camera rotation
        translation: (1, 3) translation vector
        scale: (1, 3) scale factors

    Returns:
        transformed_vertices: (N, 3) numpy array in scene coordinates
        faces: (M, 3) numpy array of face indices
    """
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure vertices are tensors on the right device
    if not torch.is_tensor(vertices):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    vertices = vertices.to(device)

    # Move pose tensors to device
    rotation_quat = rotation_quat.to(device)
    translation = translation.to(device)
    scale = scale.to(device)

    # Convert quaternion to rotation matrix
    R_l2c = quaternion_to_matrix(rotation_quat)  # (1, 3, 3)

    # Create composed transform: Scale -> Rotate -> Translate
    l2c_transform = compose_transform(
        scale=scale,
        rotation=R_l2c,
        translation=translation
    )

    # Apply transformation
    # transform_points expects (N, 3) or (B, N, 3)
    transformed = l2c_transform.transform_points(vertices.unsqueeze(0))
    transformed = transformed.squeeze(0)

    # Convert to numpy
    transformed_np = transformed.cpu().numpy()
    faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces

    return transformed_np, faces_np


def move_output_to_cpu(output, keep_gaussian=True):
    """
    Move output tensors to CPU to free GPU memory.
    Only keeps essential data for scene reconstruction.
    """
    cpu_output = {}

    # Essential pose information
    # Keep on GPU if we're keeping gaussian (they need to be on same device)
    for key in ['rotation', 'translation', 'scale']:
        if key in output and output[key] is not None:
            if keep_gaussian:
                # Keep on GPU for make_scene compatibility
                cpu_output[key] = output[key].clone()
            else:
                cpu_output[key] = output[key].cpu().clone()

    # Mesh data (always move to CPU)
    if 'mesh' in output and output['mesh'] is not None:
        mesh_result = output['mesh'][0]
        cpu_output['mesh_vertices'] = mesh_result.vertices.cpu().clone()
        cpu_output['mesh_faces'] = mesh_result.faces.cpu().clone()

    # GLB (trimesh object, already on CPU)
    if 'glb' in output and output['glb'] is not None:
        cpu_output['glb'] = output['glb']

    # Gaussian splat (keep on GPU for make_scene, but can be large)
    if keep_gaussian and 'gaussian' in output and output['gaussian'] is not None:
        cpu_output['gaussian'] = output['gaussian']

    return cpu_output


def clear_gpu_memory():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()


def sample_texture_to_vertex_colors(mesh):
    """
    Convert a textured mesh to use vertex colors by sampling the texture at UV coordinates.

    Args:
        mesh: trimesh.Trimesh with TextureVisuals

    Returns:
        numpy array of vertex colors (N, 4) RGBA uint8
    """
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        return None

    # Get UV coordinates and texture
    uvs = mesh.visual.uv
    material = mesh.visual.material

    # Try to get the texture image
    texture = None
    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
        texture = np.array(material.baseColorTexture)
    elif hasattr(material, 'image') and material.image is not None:
        texture = np.array(material.image)

    if texture is None:
        return None

    # Sample texture at UV coordinates
    h, w = texture.shape[:2]
    # UV coordinates are in [0, 1], convert to pixel coordinates
    # Note: UV origin is typically bottom-left, image origin is top-left
    u = np.clip(uvs[:, 0], 0, 1)
    v = np.clip(uvs[:, 1], 0, 1)
    px = (u * (w - 1)).astype(int)
    py = ((1 - v) * (h - 1)).astype(int)  # Flip v for image coordinates

    # Sample colors
    if texture.ndim == 2:
        # Grayscale
        colors = texture[py, px]
        colors = np.stack([colors, colors, colors, np.full_like(colors, 255)], axis=-1)
    elif texture.shape[2] == 3:
        # RGB
        colors = texture[py, px]
        alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
        colors = np.concatenate([colors, alpha], axis=-1)
    else:
        # RGBA
        colors = texture[py, px]

    return colors.astype(np.uint8)


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


def merge_trimeshes(meshes, vertex_colors_list=None):
    """
    Combine multiple trimesh objects into a single mesh.

    Args:
        meshes: list of trimesh.Trimesh objects
        vertex_colors_list: optional list of (N, 4) vertex color arrays

    Returns:
        Combined trimesh.Trimesh object
    """
    if not meshes:
        return None

    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0

    for i, mesh in enumerate(meshes):
        all_vertices.append(mesh.vertices)
        all_faces.append(mesh.faces + vertex_offset)

        # Handle vertex colors if present
        if vertex_colors_list is not None and i < len(vertex_colors_list):
            all_colors.append(vertex_colors_list[i])
        elif hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            all_colors.append(mesh.visual.vertex_colors)

        vertex_offset += len(mesh.vertices)

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)

    combined = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False
    )

    # Apply vertex colors if we have them for all meshes
    if len(all_colors) == len(meshes):
        combined_colors = np.vstack(all_colors)
        combined.visual.vertex_colors = combined_colors

    return combined


def main():
    # Configuration
    image_folder = "notebook/images/shutterstock_stylish_kidsroom_1640806567"
    output_name = "scene_reconstructed"

    # Optional: limit number of objects for faster testing
    # Set to None to process all objects
    max_objects = None  # e.g., 5 for quick test, None for all

    # Memory management options
    skip_gaussian_scene = False  # Set True if running out of GPU memory
    save_individual_objects = True  # Save each object separately (useful for debugging)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, output_name)

    print("=" * 60)
    print("Multi-Object 3D Scene Reconstruction")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}/")

    # Load model
    print("\n[1/5] Loading SAM 3D Objects model...")
    inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)

    # Load image and masks
    print("\n[2/5] Loading image and masks...")
    image = load_image(f"{image_folder}/image.png")
    masks = load_masks(image_folder, extension=".png")

    if max_objects is not None:
        masks = masks[:max_objects]
        print(f"  Limited to first {max_objects} objects for testing")

    print(f"  Image shape: {image.shape}")
    print(f"  Number of objects: {len(masks)}")

    # Create output directories
    individual_dir = os.path.join(output_dir, "objects_individual")
    if save_individual_objects:
        os.makedirs(individual_dir, exist_ok=True)

    # Reconstruct each object
    print("\n[3/5] Reconstructing objects...")
    outputs = []
    failed_objects = []

    for i, mask in enumerate(masks):
        print(f"  Processing object {i+1}/{len(masks)}...", end=" ", flush=True)

        try:
            # Run inference
            rgba = inference.merge_mask_to_rgba(image, mask)
            output = inference._pipeline.run(
                rgba,
                None,
                seed=42,
                stage1_only=False,
                with_mesh_postprocess=True,
                with_texture_baking=True,   # Bake textures from Gaussian splat
                with_layout_postprocess=False,
                use_vertex_color=False,
            )

            # Save individual object if requested
            if save_individual_objects:
                if output.get("gs") is not None:
                    output["gs"].save_ply(os.path.join(individual_dir, f"object_{i:02d}.ply"))
                if output.get("glb") is not None:
                    output["glb"].export(os.path.join(individual_dir, f"object_{i:02d}.glb"))

            # Move output to CPU to free GPU memory
            output_cpu = move_output_to_cpu(output, keep_gaussian=not skip_gaussian_scene)
            outputs.append(output_cpu)

            # Print pose info
            t = output_cpu['translation'].cpu().numpy().flatten()
            s = output_cpu['scale'].cpu().numpy().flatten()[0]  # uniform scale
            print(f"pos=[{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}], scale={s:.3f}")

            # Clear GPU memory after each object
            del output
            clear_gpu_memory()

        except torch.cuda.OutOfMemoryError:
            print(f"OOM - skipping")
            failed_objects.append(i)
            clear_gpu_memory()
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            failed_objects.append(i)
            continue

    if failed_objects:
        print(f"\n  Warning: {len(failed_objects)} objects failed: {failed_objects}")

    # Create combined Gaussian splat scene
    print("\n[4/5] Combining Gaussian splats...")
    if skip_gaussian_scene:
        print("  Skipped (skip_gaussian_scene=True to save memory)")
        print(f"  Individual Gaussian splats saved in {individual_dir}/")
    else:
        try:
            # Filter outputs that have gaussians
            outputs_with_gs = [o for o in outputs if o.get("gaussian") is not None]
            if outputs_with_gs:
                scene_gs = make_scene(*deepcopy(outputs_with_gs))

                # Save posed version (in scene coordinates)
                scene_gs.save_ply(f"{output_prefix}_posed.ply")
                print(f"  Saved: {output_prefix}_posed.ply (scene coordinates)")

                # Save normalized version (for viewing)
                scene_gs_normalized = ready_gaussian_for_video_rendering(
                    deepcopy(scene_gs),
                    fix_alignment=True
                )
                scene_gs_normalized.save_ply(f"{output_prefix}.ply")
                print(f"  Saved: {output_prefix}.ply (normalized for viewing)")

                # Clean up
                del scene_gs, scene_gs_normalized
                clear_gpu_memory()
            else:
                print("  No Gaussian outputs available")
        except torch.cuda.OutOfMemoryError:
            print("  OOM during Gaussian combination - skipping")
            print("  Try setting skip_gaussian_scene=True and use individual objects")
            clear_gpu_memory()

    # Create combined mesh scene
    print("\n[5/5] Combining meshes...")

    # Inverse rotation matrix to convert from y-up (GLB) back to z-up (original)
    # The GLB export applies: vertices @ [[1,0,0],[0,0,-1],[0,1,0]] (z-up to y-up)
    # This reverses it: (x,y,z) -> (x,z,-y) back to original z-up coordinates
    yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

    # Lists for Scene-based output (preserves textures) and merged output (vertex colors)
    scene_meshes_textured = []  # For GLB Scene export
    scene_meshes_colored = []   # For merged OBJ export

    for i, output in enumerate(outputs):
        if output.get("glb") is None:
            print(f"  Object {i}: No GLB mesh available, skipping")
            continue

        mesh = output["glb"]

        # Transform mesh while preserving visual (texture)
        transformed_mesh = transform_mesh_preserve_visual(
            mesh,
            output["rotation"],
            output["translation"],
            output["scale"],
            yup_to_zup
        )
        scene_meshes_textured.append((f"object_{i:02d}", transformed_mesh))

        # Also create a vertex-colored version for merged export
        vertex_colors = sample_texture_to_vertex_colors(mesh)
        colored_mesh = trimesh.Trimesh(
            vertices=transformed_mesh.vertices.copy(),
            faces=transformed_mesh.faces.copy(),
            process=False
        )
        if vertex_colors is not None:
            colored_mesh.visual.vertex_colors = vertex_colors
        scene_meshes_colored.append(colored_mesh)

        # Clear memory after each mesh transformation
        clear_gpu_memory()
        print(f"  Object {i}: {len(transformed_mesh.vertices)} vertices, {len(transformed_mesh.faces)} faces")

    # Save as Scene-based GLB (preserves individual textures)
    if scene_meshes_textured:
        scene = trimesh.Scene()
        for name, mesh in scene_meshes_textured:
            scene.add_geometry(mesh, node_name=name)
        scene.export(f"{output_prefix}.glb")
        print(f"\n  Scene GLB saved: {output_prefix}.glb")
        print(f"  Contains {len(scene_meshes_textured)} textured objects")

    # Also save merged mesh with vertex colors as OBJ
    if scene_meshes_colored:
        combined_mesh = merge_trimeshes(scene_meshes_colored)
        if combined_mesh is not None:
            combined_mesh.export(f"{output_prefix}.obj")
            print(f"  Merged OBJ saved: {output_prefix}.obj")
            print(f"  Total: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")
    else:
        print("\n  Warning: No meshes were generated")

    print("\n" + "=" * 60)
    print("Scene reconstruction complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}/")
    print(f"\nFiles:")
    print(f"  - {output_name}.ply           : Gaussian splat (normalized, for viewing)")
    print(f"  - {output_name}_posed.ply     : Gaussian splat (scene coordinates)")
    print(f"  - {output_name}.glb           : Scene with textured meshes (multi-object)")
    print(f"  - {output_name}.obj           : Merged mesh with vertex colors")
    print(f"  - objects_individual/         : Individual object files")
    print(f"\nView the Gaussian splat at: https://playcanvas.com/supersplat")
    print(f"View the GLB/OBJ in any 3D viewer (Blender, MeshLab, etc.)")


if __name__ == "__main__":
    main()
