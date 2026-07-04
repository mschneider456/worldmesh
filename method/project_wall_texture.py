"""
Project generated image onto structural 3D mesh as projective texture.

After bootstrap generation (camera 0), projects the full generated image
onto all structural geometry (walls, ceiling, floor) using the known camera
intrinsics and extrinsics. Each mesh vertex is projected to its corresponding
pixel via pinhole camera model.

Uses the structure_only.glb mesh which preserves segment names like
"room_0_wall_0", "room_0_ceiling", "room_0_floor".

The output is used by generate_depth_object_conditioning.py to render
textured walls from each camera viewpoint.

Usage:
    python project_wall_texture.py \
        --generated-image output/flux_final/room_0/generated/generated_0000.png \
        --structure-mesh output/structure_only.glb \
        --cameras-txt output/renders_final/cameras.txt \
        --images-txt output/renders_final/images.txt \
        --camera-name room_0_0000 \
        --room-id room_0 \
        --output-mesh output/textured_structure.glb
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh


def load_colmap_cameras(cameras_txt: Path) -> dict:
    """Load camera intrinsics from COLMAP cameras.txt."""
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            return {
                'width': int(parts[2]),
                'height': int(parts[3]),
                'fx': float(parts[4]),
                'fy': float(parts[5]),
                'cx': float(parts[6]),
                'cy': float(parts[7])
            }
    return {}


def load_colmap_images(images_txt: Path) -> dict:
    """Load camera extrinsics from COLMAP images.txt."""
    poses = {}
    with open(images_txt, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = parts[9]

            poses[name] = {
                'qvec': [qw, qx, qy, qz],
                'tvec': [tx, ty, tz]
            }

        i += 2  # Skip POINTS2D line

    return poses


def quaternion_to_rotation_matrix(qvec):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def colmap_to_camera_pose(qvec, tvec):
    """Convert COLMAP extrinsics (world-to-camera) to camera-to-world pose matrix."""
    R = quaternion_to_rotation_matrix(qvec)
    t = np.array(tvec)

    R_c2w = R.T
    t_c2w = -R.T @ t

    pose = np.eye(4)
    pose[:3, :3] = R_c2w
    pose[:3, 3] = t_c2w
    return pose


def project_vertices_to_uv(vertices_world, camera_c2w, fx, fy, cx, cy, width, height):
    """
    Project world-space vertices to UV coordinates in the bootstrap image.

    Args:
        vertices_world: (N, 3) vertices in world space
        camera_c2w: 4x4 camera-to-world matrix
        fx, fy, cx, cy: Camera intrinsics
        width, height: Image dimensions

    Returns:
        (N, 2) UV coordinates in [0, 1] range (u=horizontal, v=vertical)
    """
    w2c = np.linalg.inv(camera_c2w)

    # Transform to camera space
    ones = np.ones((vertices_world.shape[0], 1))
    verts_h = np.hstack([vertices_world, ones])
    cam_coords = (w2c @ verts_h.T).T[:, :3]

    # OpenGL convention: camera looks down -Z, Y is up, X is right
    # Project to pixel coordinates
    # Avoid division by zero
    z = -cam_coords[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)

    px = fx * cam_coords[:, 0] / z + cx
    py = fy * (-cam_coords[:, 1]) / z + cy

    # Convert to UV [0, 1]
    # OpenGL/glTF texture convention: V=0 at bottom of image, V=1 at top.
    # Pixel coordinate py=0 is at image top, so flip V.
    u = px / width
    v = 1.0 - (py / height)

    return np.stack([u, v], axis=1), z


def subdivide_mesh(mesh, max_edge_length=0.3):
    """Subdivide mesh faces until all edges are below max_edge_length.

    Critical for ceiling geometry (often just 2 triangles) where too few
    vertices cause severe perspective-correct texture distortion.
    """
    verts, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=max_edge_length
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def split_by_camera_visibility(mesh, transform, camera_pos):
    """Split mesh into camera-facing and non-camera-facing submeshes.

    Uses face normal dot product with (camera - face_centroid) direction.
    Only camera-facing faces should be textured to avoid backface mirroring
    and artifacts from outer/top/bottom wall faces.

    Args:
        mesh: trimesh.Trimesh in local space
        transform: 4x4 local-to-world transform
        camera_pos: (3,) camera position in world space

    Returns:
        (facing_mesh, other_mesh) both in local space, or (mesh, None)
        if all faces are facing.
    """
    if len(mesh.faces) == 0:
        return mesh, None

    R = transform[:3, :3]
    normals_world = (R @ np.array(mesh.face_normals).T).T

    verts_local = np.array(mesh.vertices)
    ones = np.ones((verts_local.shape[0], 1))
    verts_world = (transform @ np.hstack([verts_local, ones]).T).T[:, :3]

    face_verts = verts_world[mesh.faces]  # (F, 3, 3)
    face_centroids = face_verts.mean(axis=1)  # (F, 3)
    to_camera = camera_pos - face_centroids
    dots = np.sum(normals_world * to_camera, axis=1)

    facing_mask = dots > 0

    if facing_mask.all():
        return mesh, None
    if not facing_mask.any():
        return None, mesh

    facing_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces[facing_mask], process=False
    )
    other_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces[~facing_mask], process=False
    )

    return facing_mesh, other_mesh


def get_surface_geometry_names(scene_mesh, room_id):
    """Get all structural geometry names (walls, ceiling, floor) for a room.

    Handles meshes from previous texture passes by matching _back/_edge suffixed
    names while skipping already-textured (_textured) geometry.
    """
    names = []
    if isinstance(scene_mesh, trimesh.Scene):
        for geom_name in scene_mesh.geometry.keys():
            if geom_name.endswith('_textured'):
                continue  # Already has texture from previous pass
            # Strip all _back/_edge suffixes to find the original base name
            # (after multiple texture passes, names accumulate like
            # room_0_floor_edge_back_edge)
            base = geom_name
            while True:
                stripped = False
                for suffix in ('_back', '_edge'):
                    if base.endswith(suffix):
                        base = base[:-len(suffix)]
                        stripped = True
                        break
                if not stripped:
                    break
            if (base.startswith(f"{room_id}_wall_") or
                    base == f"{room_id}_ceiling" or
                    base == f"{room_id}_floor"):
                names.append(geom_name)
    return sorted(names)


def check_occlusion(uvs, depths, scene_depth, width, height, tolerance=0.10):
    """Check which vertices are occluded by objects using depth comparison.

    For each vertex, compares its camera-space depth against the full-scene
    rendered depth map (which includes objects). If the scene depth at that
    pixel is significantly less than the vertex depth, an object is between
    the camera and the wall.

    Args:
        uvs: (N, 2) UV coordinates in [0, 1] range
        depths: (N,) camera-space depths (positive distance along view axis)
        scene_depth: (H, W) numpy array of rendered scene depth values
        width: Image width in pixels
        height: Image height in pixels
        tolerance: Depth tolerance in meters (default 0.10m)

    Returns:
        (N,) boolean mask: True = not occluded (visible), False = occluded
    """
    # Convert UVs back to pixel coordinates
    px = (uvs[:, 0] * width).astype(int)
    py = ((1.0 - uvs[:, 1]) * height).astype(int)

    # Clamp to image bounds
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    # Look up scene depth at each pixel
    scene_z = scene_depth[py, px]

    # A vertex is occluded if: scene rendered something closer than the vertex
    # scene_z == 0 means no geometry was rendered (e.g. door/window opening) -> not occluded
    # scene_z > 0 and scene_z < vertex_depth - tolerance -> occluded by object
    occluded = (scene_z > 0) & (scene_z < depths - tolerance)

    return ~occluded


def project_wall_texture(
    generated_image_path: Path,
    structure_mesh_path: Path,
    cameras_txt: Path,
    images_txt: Path,
    camera_name: str,
    room_id: str,
    output_mesh_path: Path,
    visibility_threshold: float = 0.1,
    scene_depth_path: Path = None,
    occlusion_tolerance: float = 0.10,
    uv_margin: float = 0.05,
    verbose: bool = False,
):
    """
    Project the bootstrap generated image onto structural meshes as textures.

    Projects the full generated image onto all structural geometry (walls,
    ceiling, floor) using the known camera intrinsics and extrinsics.
    Each vertex is projected to its corresponding pixel in the image.

    Uses the structure_only.glb mesh which preserves segment names like
    "room_0_wall_0", "room_0_ceiling", "room_0_floor".

    Args:
        generated_image_path: Path to bootstrap generated image (e.g., generated_0000.png)
        structure_mesh_path: Path to structure_only.glb (has segment names)
        cameras_txt: Path to COLMAP cameras.txt
        images_txt: Path to COLMAP images.txt
        camera_name: Camera name (e.g., "room_0_0000")
        room_id: Room ID (e.g., "room_0")
        output_mesh_path: Path for output textured mesh
        visibility_threshold: Min fraction of faces in frustum to texture a surface
        scene_depth_path: Path to scene depth .npy (with objects) for occlusion detection
        occlusion_tolerance: Depth tolerance in meters for occlusion check (default 0.10)
        verbose: Verbose output
    """
    # Load inputs
    print(f"Loading generated image: {generated_image_path}")
    gen_image = np.array(Image.open(generated_image_path).convert('RGB'))
    img_h, img_w = gen_image.shape[:2]

    # Load scene depth for occlusion detection (optional)
    scene_depth = None
    if scene_depth_path is not None:
        scene_depth_path = Path(scene_depth_path)
        if scene_depth_path.exists():
            scene_depth = np.load(scene_depth_path)
            print(f"  Loaded scene depth for occlusion detection: {scene_depth_path}")
            if verbose:
                print(f"    Depth shape: {scene_depth.shape}, range: [{scene_depth.min():.2f}, {scene_depth.max():.2f}]m, tolerance: {occlusion_tolerance}m")
        else:
            print(f"  WARNING: Scene depth not found: {scene_depth_path}, skipping occlusion detection")

    # Load camera parameters
    print(f"Loading camera parameters...")
    camera_params = load_colmap_cameras(cameras_txt)
    image_poses = load_colmap_images(images_txt)

    fx, fy = camera_params['fx'], camera_params['fy']
    cx, cy = camera_params['cx'], camera_params['cy']
    width, height = camera_params['width'], camera_params['height']

    # Find the camera pose
    image_key = f"{room_id}/{camera_name}.png"
    if image_key not in image_poses:
        raise ValueError(f"Camera pose not found for {image_key}")

    pose_data = image_poses[image_key]
    c2w = colmap_to_camera_pose(pose_data['qvec'], pose_data['tvec'])
    print(f"  Camera position: {c2w[:3, 3]}")

    # Load structure mesh (has proper wall segment names like "home_office_wall_0")
    print(f"Loading structure mesh: {structure_mesh_path}")
    scene_mesh = trimesh.load(structure_mesh_path)

    if not isinstance(scene_mesh, trimesh.Scene):
        raise ValueError("Expected a trimesh.Scene, got a single mesh")

    # Get wall and ceiling geometries for this room
    wall_names = get_surface_geometry_names(scene_mesh, room_id)
    print(f"  Found {len(wall_names)} surface segments for {room_id}")

    if not wall_names:
        print("  WARNING: No surface segments found. Check that the structure mesh has named wall/ceiling geometries.")
        all_names = sorted(scene_mesh.geometry.keys())
        print(f"  Available geometries ({len(all_names)}): {all_names[:10]}...")
        output_mesh_path = Path(output_mesh_path)
        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        scene_mesh.export(str(output_mesh_path))
        print(f"  Saved untextured mesh: {output_mesh_path}")
        return 0

    # Create the texture as a PIL Image for trimesh
    texture_pil = Image.fromarray(gen_image)

    camera_pos = c2w[:3, 3]

    textured_count = 0
    skipped_count = 0
    all_sampled_colors = []  # Collect pixel colors from textured regions for fallback

    for wall_name in wall_names:
        geom = scene_mesh.geometry[wall_name]

        # Get world-space transform (apply scene graph transform)
        transform = np.eye(4)
        try:
            # Get the transform from the scene graph
            node_path = scene_mesh.graph.transforms.get(wall_name)
            if node_path is not None:
                transform = node_path
            else:
                # Try getting it via the graph
                transform_tuple = scene_mesh.graph.get(wall_name)
                if transform_tuple is not None:
                    transform = transform_tuple[0]
        except Exception:
            pass  # Use identity transform

        # Save original color for untextured faces
        original_color = None
        if hasattr(geom.visual, 'main_color'):
            original_color = geom.visual.main_color

        # Subdivide for better texture interpolation (critical for ceiling
        # which often has only 2 triangles, causing severe distortion)
        subdiv_geom = subdivide_mesh(geom, max_edge_length=0.3)
        if verbose and len(subdiv_geom.faces) > len(geom.faces):
            print(f"  {wall_name}: subdivided {len(geom.faces)} → {len(subdiv_geom.faces)} faces")

        # Split into camera-facing and non-camera-facing faces.
        # Only camera-facing faces get textured — this avoids:
        # - Backface mirroring artifacts when orbiting in GLB viewers
        # - Incorrect UVs on outer/top/bottom wall box faces
        facing_mesh, back_mesh = split_by_camera_visibility(
            subdiv_geom, transform, camera_pos
        )

        if facing_mesh is None or len(facing_mesh.faces) == 0:
            if verbose:
                print(f"  {wall_name}: skipped (no camera-facing faces)")
            skipped_count += 1
            continue

        # Compute world-space vertices for facing mesh
        vertices_local = np.array(facing_mesh.vertices)
        ones = np.ones((vertices_local.shape[0], 1))
        vertices_h = np.hstack([vertices_local, ones])
        vertices_world = (transform @ vertices_h.T).T[:, :3]

        # Project vertices to UV
        uvs, depths = project_vertices_to_uv(
            vertices_world, c2w, fx, fy, cx, cy, width, height
        )

        # Per-vertex validity: in front of camera and UV inside [-margin, 1+margin].
        # A small margin (default 5%) allows corner faces whose vertices project
        # just barely outside the image to still be textured — the UVs are clamped
        # to [0, 1] before texture application, which repeats edge pixels for the
        # small overshoot (acceptable for corners, unlike wildly out-of-bounds UVs).
        valid_vertex = (
            (depths > 0) &
            (uvs[:, 0] >= -uv_margin) & (uvs[:, 0] <= 1 + uv_margin) &
            (uvs[:, 1] >= -uv_margin) & (uvs[:, 1] <= 1 + uv_margin)
        )

        # Occlusion check: exclude vertices where objects are between camera and wall
        if scene_depth is not None:
            not_occluded = check_occlusion(uvs, depths, scene_depth, width, height, tolerance=occlusion_tolerance)
            occluded_count = valid_vertex.sum() - (valid_vertex & not_occluded).sum()
            valid_vertex = valid_vertex & not_occluded
            if verbose and occluded_count > 0:
                print(f"  {wall_name}: {occluded_count} vertices occluded by objects")

        # Per-face: only texture faces where ALL 3 vertices have valid UVs.
        # Faces with any out-of-bounds vertex would get distorted by clamping
        # (vertices projecting to UV values like -24 or +25 get clamped to 0/1,
        # stretching edge pixels across large areas).
        faces = np.array(facing_mesh.faces)
        face_valid = valid_vertex[faces].all(axis=1)

        valid_face_count = face_valid.sum()
        total_face_count = len(faces)

        if valid_face_count == 0:
            if verbose:
                valid_verts = valid_vertex.sum()
                print(f"  {wall_name}: skipped (0/{total_face_count} faces fully in frustum, "
                      f"{valid_verts}/{len(valid_vertex)} valid vertices)")
            skipped_count += 1
            continue

        visibility_ratio = valid_face_count / total_face_count

        if visibility_ratio < visibility_threshold:
            if verbose:
                print(f"  {wall_name}: skipped (visibility {visibility_ratio:.1%} < {visibility_threshold:.0%})")
            skipped_count += 1
            continue

        # Split facing mesh into texturable (all verts valid) and frustum-edge faces
        if face_valid.all():
            texturable_mesh = facing_mesh
            texturable_uvs = np.clip(uvs, 0.0, 1.0)
            frustum_edge_mesh = None
        else:
            valid_faces = faces[face_valid]
            # Compact: remap to only used vertices (avoids extreme UV values
            # for unused vertices in GLB export)
            used_verts = np.unique(valid_faces.flatten())
            vert_map = np.full(len(facing_mesh.vertices), -1, dtype=int)
            vert_map[used_verts] = np.arange(len(used_verts))

            texturable_mesh = trimesh.Trimesh(
                vertices=np.array(facing_mesh.vertices)[used_verts],
                faces=vert_map[valid_faces],
                process=False,
            )
            texturable_uvs = np.clip(uvs[used_verts], 0.0, 1.0)

            frustum_edge_mesh = trimesh.Trimesh(
                vertices=facing_mesh.vertices,
                faces=faces[~face_valid],
                process=False,
            )

        # Apply texture to texturable submesh (UVs already clamped to [0, 1])
        texturable_mesh.visual = trimesh.visual.TextureVisuals(
            uv=texturable_uvs,
            image=texture_pil,
        )

        # Sample pixel colors from textured UVs for average-color fallback
        sample_px = np.clip((texturable_uvs[:, 0] * img_w).astype(int), 0, img_w - 1)
        sample_py = np.clip(((1.0 - texturable_uvs[:, 1]) * img_h).astype(int), 0, img_h - 1)
        sampled = gen_image[sample_py, sample_px]  # (N, 3)
        all_sampled_colors.append(sampled)

        # Replace original geometry with textured + untextured parts
        scene_mesh.delete_geometry(wall_name)

        # Add textured submesh
        textured_name = f"{wall_name}_textured"
        scene_mesh.add_geometry(
            texturable_mesh, geom_name=textured_name, transform=transform
        )

        # Add frustum-edge faces (camera-facing but outside frustum) as untextured
        if frustum_edge_mesh is not None and len(frustum_edge_mesh.faces) > 0:
            if original_color is not None:
                frustum_edge_mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=frustum_edge_mesh, face_colors=original_color
                )
            edge_name = f"{wall_name}_edge"
            scene_mesh.add_geometry(
                frustum_edge_mesh, geom_name=edge_name, transform=transform
            )

        # Add non-camera-facing faces as untextured
        if back_mesh is not None and len(back_mesh.faces) > 0:
            if original_color is not None:
                back_mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=back_mesh, face_colors=original_color
                )
            back_name = f"{wall_name}_back"
            scene_mesh.add_geometry(
                back_mesh, geom_name=back_name, transform=transform
            )

        textured_count += 1
        if verbose:
            edge_count = len(frustum_edge_mesh.faces) if frustum_edge_mesh and len(frustum_edge_mesh.faces) > 0 else 0
            back_count = len(back_mesh.faces) if back_mesh and len(back_mesh.faces) > 0 else 0
            print(f"  {wall_name}: textured {len(texturable_mesh.faces)}/{total_face_count} faces "
                  f"(visibility {visibility_ratio:.1%})"
                  f"{f', {edge_count} frustum-edge' if edge_count else ''}"
                  f"{f', {back_count} back' if back_count else ''}")

    print(f"\nTextured {textured_count} surfaces, skipped {skipped_count} (not visible)")

    # Apply average-color fallback to remaining untextured _edge/_back faces
    if all_sampled_colors:
        avg_color = np.concatenate(all_sampled_colors, axis=0).mean(axis=0).astype(np.uint8)
        avg_rgba = np.array([avg_color[0], avg_color[1], avg_color[2], 255], dtype=np.uint8)
        fallback_count = 0
        for geom_name in list(scene_mesh.geometry.keys()):
            if geom_name.endswith('_edge') or geom_name.endswith('_back'):
                geom = scene_mesh.geometry[geom_name]
                geom.visual = trimesh.visual.ColorVisuals(
                    mesh=geom, face_colors=avg_rgba
                )
                fallback_count += 1
        if fallback_count > 0:
            print(f"Applied average color fallback ({avg_color}) to {fallback_count} untextured submeshes")

    # Export
    output_mesh_path = Path(output_mesh_path)
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(str(output_mesh_path))
    print(f"Saved textured mesh: {output_mesh_path}")

    return textured_count


def main():
    parser = argparse.ArgumentParser(
        description='Project bootstrap wall textures onto 3D mesh'
    )
    parser.add_argument('--generated-image', required=True, type=Path,
                       help='Path to bootstrap generated image')
    parser.add_argument('--structure-mesh', required=True, type=Path,
                       help='Path to structure_only.glb (has wall segment names)')
    parser.add_argument('--cameras-txt', required=True, type=Path,
                       help='Path to COLMAP cameras.txt')
    parser.add_argument('--images-txt', required=True, type=Path,
                       help='Path to COLMAP images.txt')
    parser.add_argument('--camera-name', required=True, type=str,
                       help='Camera name (e.g., room_0_0000)')
    parser.add_argument('--room-id', required=True, type=str,
                       help='Room ID (e.g., room_0)')
    parser.add_argument('--output-mesh', required=True, type=Path,
                       help='Output path for textured mesh')
    parser.add_argument('--visibility-threshold', type=float, default=0.1,
                       help='Min fraction of faces in frustum to texture a surface (default: 0.1)')
    parser.add_argument('--scene-depth', type=Path, default=None,
                       help='Path to scene depth .npy (with objects) for occlusion detection')
    parser.add_argument('--occlusion-tolerance', type=float, default=0.10,
                       help='Depth tolerance in meters for occlusion check (default: 0.10)')
    parser.add_argument('--uv-margin', type=float, default=0.05,
                       help='UV margin beyond [0,1] to include corner faces (default: 0.05)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    project_wall_texture(
        generated_image_path=args.generated_image,
        structure_mesh_path=args.structure_mesh,
        cameras_txt=args.cameras_txt,
        images_txt=args.images_txt,
        camera_name=args.camera_name,
        room_id=args.room_id,
        output_mesh_path=args.output_mesh,
        visibility_threshold=args.visibility_threshold,
        scene_depth_path=args.scene_depth,
        occlusion_tolerance=args.occlusion_tolerance,
        uv_margin=args.uv_margin,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
