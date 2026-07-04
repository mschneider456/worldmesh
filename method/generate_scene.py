"""
Generate 3D room scene from JSON description using Trimesh.
Rooms are created with walls, floors, and ceilings.
Objects are represented as arrow-shaped wedges with clear front/back orientation.
"""

import json
import numpy as np
import trimesh
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from typing import List, Dict, Tuple, Optional


def load_scene_json(filepath):
    """Load and parse scene description from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_wall_mesh(polygon_coords, height, thickness=0.15):
    """
    Create walls from a 2D polygon by extruding both inner and outer boundaries.

    Args:
        polygon_coords: List of [x, y] coordinates defining the room floor
        height: Wall height in meters
        thickness: Wall thickness in meters

    Returns:
        trimesh.Trimesh: Combined wall mesh
    """
    # Create outer and inner polygons
    outer_poly = Polygon(polygon_coords)
    inner_poly = outer_poly.buffer(-thickness)

    # Extrude the difference to get walls
    wall_polygon = outer_poly.difference(inner_poly)

    # Extrude to 3D
    wall_mesh = trimesh.creation.extrude_polygon(wall_polygon, height=height)

    return wall_mesh


def points_match(p1: List[float], p2: List[float], tolerance: float = 0.01) -> bool:
    """Check if two 2D points are approximately equal."""
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance


def edges_are_coincident(
    edge1_start: List[float],
    edge1_end: List[float],
    edge2_start: List[float],
    edge2_end: List[float],
    tolerance: float = 0.01
) -> Tuple[bool, bool]:
    """
    Check if two edges are coincident (same line segment, possibly reversed).

    Args:
        edge1_start, edge1_end: First edge endpoints
        edge2_start, edge2_end: Second edge endpoints
        tolerance: Coordinate matching tolerance

    Returns:
        Tuple of (is_coincident, is_reversed)
    """
    # Check forward match
    forward = (
        points_match(edge1_start, edge2_start, tolerance) and
        points_match(edge1_end, edge2_end, tolerance)
    )
    # Check reverse match
    reverse = (
        points_match(edge1_start, edge2_end, tolerance) and
        points_match(edge1_end, edge2_start, tolerance)
    )
    return (forward or reverse, reverse)


def find_shared_edges(room_id: str, polygon_coords: List[List[float]],
                      all_rooms: List[Dict], tolerance: float = 0.01) -> set:
    """
    Find which edges of a room are shared with another room.

    Returns a set of edge indices (0-based) that are shared walls.
    """
    shared = set()
    n = len(polygon_coords)

    for i in range(n):
        p1 = polygon_coords[i]
        p2 = polygon_coords[(i + 1) % n]

        for other_room in all_rooms:
            if other_room['id'] == room_id:
                continue
            other_poly = other_room['floor_polygon']
            m = len(other_poly)
            for j in range(m):
                op1 = other_poly[j]
                op2 = other_poly[(j + 1) % m]
                is_coincident, _ = edges_are_coincident(p1, p2, op1, op2, tolerance)
                if is_coincident:
                    shared.add(i)
                    break
            if i in shared:
                break

    return shared


def find_shared_wall_openings(
    room_id: str,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    all_rooms: List[Dict],
    tolerance: float = 0.01
) -> List[Dict]:
    """
    Find openings from adjacent rooms that share a wall with the given edge.

    When rooms share a wall (e.g., living room and bedroom), a door defined
    in one room should also cut through the adjacent room's wall.

    Args:
        room_id: ID of the current room (to exclude from search)
        edge_start: Start point of the wall edge [x, y]
        edge_end: End point of the wall edge [x, y]
        all_rooms: List of all room definitions from JSON
        tolerance: Coordinate matching tolerance

    Returns:
        List of openings from adjacent rooms that affect this edge
    """
    shared_openings = []

    for other_room in all_rooms:
        # Skip the current room
        if other_room['id'] == room_id:
            continue

        other_polygon = other_room['floor_polygon']
        other_openings = other_room.get('openings', [])

        if not other_openings:
            continue

        # Check each edge of the other room's polygon
        n = len(other_polygon)
        for i in range(n):
            other_edge_start = other_polygon[i]
            other_edge_end = other_polygon[(i + 1) % n]

            # Check if this edge is coincident with our edge
            is_coincident, is_reversed = edges_are_coincident(
                edge_start.tolist(), edge_end.tolist(),
                other_edge_start, other_edge_end,
                tolerance
            )

            if is_coincident:
                # Found a shared wall - check for openings on this edge in the other room
                for opening in other_openings:
                    wall_seg = opening.get('wall_segment', [])
                    if len(wall_seg) != 2:
                        continue

                    # Check if this opening is on the shared edge
                    opening_on_edge, opening_reversed = edges_are_coincident(
                        other_edge_start, other_edge_end,
                        wall_seg[0], wall_seg[1],
                        tolerance
                    )

                    if opening_on_edge:
                        # This opening from the adjacent room affects our wall
                        opening_copy = opening.copy()
                        # Determine if position needs to be reversed relative to our edge
                        # If the shared wall is reversed AND the opening wasn't already reversed,
                        # or vice versa, we need to reverse the position
                        opening_copy['_reversed'] = is_reversed != opening_reversed
                        opening_copy['_from_room'] = other_room['id']
                        shared_openings.append(opening_copy)

    return shared_openings


def find_openings_for_edge(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    openings: List[Dict],
    tolerance: float = 0.01
) -> List[Dict]:
    """
    Find all openings that belong to a specific wall edge.

    Args:
        edge_start: Start point of the wall edge [x, y]
        edge_end: End point of the wall edge [x, y]
        openings: List of opening definitions from JSON
        tolerance: Coordinate matching tolerance

    Returns:
        List of openings that match this edge
    """
    matching_openings = []

    for opening in openings:
        wall_seg = opening.get('wall_segment', [])
        if len(wall_seg) != 2:
            continue

        seg_start = wall_seg[0]
        seg_end = wall_seg[1]

        # Check if opening's wall_segment matches this edge (in either direction)
        forward_match = (
            points_match(seg_start, edge_start.tolist(), tolerance) and
            points_match(seg_end, edge_end.tolist(), tolerance)
        )
        reverse_match = (
            points_match(seg_start, edge_end.tolist(), tolerance) and
            points_match(seg_end, edge_start.tolist(), tolerance)
        )

        if forward_match or reverse_match:
            # Store direction info for position calculation
            opening_copy = opening.copy()
            opening_copy['_reversed'] = reverse_match
            matching_openings.append(opening_copy)

    return matching_openings


def create_opening_cutout(
    opening: Dict,
    wall_start: np.ndarray,
    wall_end: np.ndarray,
    wall_thickness: float
) -> Tuple[trimesh.Trimesh, float, float]:
    """
    Create a box mesh to cut out an opening (door/window) from a wall.

    Args:
        opening: Opening definition from JSON
        wall_start: Start point of the wall edge [x, y]
        wall_end: End point of the wall edge [x, y]
        wall_thickness: Wall thickness for cutout depth

    Returns:
        Tuple of (cutout_mesh, z_start, z_end) for the opening
    """
    # Get opening parameters
    opening_type = opening.get('type', 'door')
    position_along_wall = opening.get('position', 0)
    width = opening.get('width', 0.9)
    height = opening.get('height', 2.1)
    sill_height = opening.get('sill_height', 0)  # Only for windows

    # If the opening was matched in reverse direction, adjust position
    if opening.get('_reversed', False):
        wall_length = np.linalg.norm(wall_end - wall_start)
        position_along_wall = wall_length - position_along_wall - width

    # Calculate wall direction and normal
    wall_vec = wall_end - wall_start
    wall_length = np.linalg.norm(wall_vec)
    wall_dir = wall_vec / wall_length

    # Normal pointing inward (perpendicular, rotated 90 degrees clockwise for CCW polygons)
    wall_normal = np.array([-wall_dir[1], wall_dir[0]])

    # Calculate center position of the opening
    opening_center_along_wall = position_along_wall + width / 2
    opening_center_2d = wall_start + wall_dir * opening_center_along_wall

    # Move center to middle of wall thickness
    opening_center_2d = opening_center_2d + wall_normal * (wall_thickness / 2)

    # Determine Z range for cutout
    if opening_type == 'door':
        z_start = 0
        z_end = height
    else:  # window
        z_start = sill_height
        z_end = sill_height + height

    z_center = (z_start + z_end) / 2
    z_height = z_end - z_start

    # Create the cutout box
    # Box dimensions: width along wall, extra depth through wall, height
    cutout_depth = wall_thickness * 3  # Extra depth to ensure clean cut
    cutout_box = trimesh.creation.box([width, cutout_depth, z_height])

    # Calculate rotation angle to align with wall
    angle = np.arctan2(wall_dir[1], wall_dir[0])
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
    cutout_box.apply_transform(rotation_matrix)

    # Translate to final position
    cutout_box.apply_translation([opening_center_2d[0], opening_center_2d[1], z_center])

    return cutout_box, z_start, z_end


def create_wall_segments_with_openings(
    polygon_coords: List[List[float]],
    height: float,
    thickness: float,
    openings: List[Dict],
    room_id: str = None,
    all_rooms: List[Dict] = None,
    shared_edges: set = None
) -> List[Tuple[trimesh.Trimesh, int]]:
    """
    Create individual wall segments with door/window cutouts.

    Each wall segment is a separate mesh, allowing distinct colors in segmentation.
    Openings (doors/windows) are cut out using boolean difference operations.

    This function also handles shared walls: if an adjacent room has a door/window
    on a shared wall, it will be cut from this room's wall as well.

    Shared walls use half thickness so that two adjacent rooms together form
    one full-thickness wall instead of double.

    Args:
        polygon_coords: List of [x, y] coordinates defining the room floor
        height: Wall height in meters
        thickness: Wall thickness in meters (full thickness for exterior walls)
        openings: List of opening definitions from room JSON
        room_id: ID of the current room (for shared wall detection)
        all_rooms: List of all room definitions (for shared wall detection)
        shared_edges: Set of edge indices that are shared with other rooms

    Returns:
        list: List of (wall_mesh, wall_index) tuples for each wall segment
    """
    wall_segments = []
    n = len(polygon_coords)
    if shared_edges is None:
        shared_edges = set()

    for i in range(n):
        # Get the two corners of this wall segment
        p1 = np.array(polygon_coords[i])
        p2 = np.array(polygon_coords[(i + 1) % n])

        # Use half thickness for shared walls to avoid double-thick walls
        edge_thickness = thickness / 2 if i in shared_edges else thickness

        # Calculate wall direction and normal
        wall_vec = p2 - p1
        wall_length = np.linalg.norm(wall_vec)
        wall_dir = wall_vec / wall_length

        # Normal pointing inward (perpendicular, rotated 90 degrees clockwise)
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Create wall rectangle vertices (outer edge to inner edge)
        outer_p1 = p1
        outer_p2 = p2
        inner_p1 = p1 + normal * edge_thickness
        inner_p2 = p2 + normal * edge_thickness

        # Create polygon for this wall segment
        wall_poly_coords = [
            tuple(outer_p1),
            tuple(outer_p2),
            tuple(inner_p2),
            tuple(inner_p1),
        ]

        wall_poly = Polygon(wall_poly_coords)

        # Extrude to 3D
        wall_mesh = trimesh.creation.extrude_polygon(wall_poly, height=height)

        # Find openings for this wall segment (from this room)
        edge_openings = find_openings_for_edge(p1, p2, openings)

        # Also find openings from adjacent rooms on shared walls
        if room_id and all_rooms:
            shared_openings = find_shared_wall_openings(room_id, p1, p2, all_rooms)
            edge_openings.extend(shared_openings)

        # Apply cutouts for each opening
        for opening in edge_openings:
            try:
                cutout_box, z_start, z_end = create_opening_cutout(
                    opening, p1, p2, edge_thickness
                )

                # Perform boolean difference
                wall_mesh = wall_mesh.difference(cutout_box, engine='manifold')

                opening_type = opening.get('type', 'door')
                opening_width = opening.get('width', 0.9)
                from_room = opening.get('_from_room', None)
                if from_room:
                    print(f"    Cut {opening_type} (w={opening_width}m) in wall segment {i} [from shared wall with {from_room}]")
                else:
                    print(f"    Cut {opening_type} (w={opening_width}m) in wall segment {i}")

            except Exception as e:
                print(f"    Warning: Failed to cut opening in wall {i}: {e}")

        wall_segments.append((wall_mesh, i))

    return wall_segments


def create_wall_segments(polygon_coords, height, thickness=0.15, shared_edges=None):
    """
    Create individual wall segments for each side of the room.

    Each wall segment is a separate mesh, allowing distinct colors in segmentation.
    Shared walls use half thickness so adjacent rooms form one full-thickness wall.

    Args:
        polygon_coords: List of [x, y] coordinates defining the room floor
        height: Wall height in meters
        thickness: Wall thickness in meters
        shared_edges: Set of edge indices that are shared with other rooms

    Returns:
        list: List of (wall_mesh, wall_index) tuples for each wall segment
    """
    wall_segments = []
    n = len(polygon_coords)
    if shared_edges is None:
        shared_edges = set()

    for i in range(n):
        # Get the two corners of this wall segment
        p1 = np.array(polygon_coords[i])
        p2 = np.array(polygon_coords[(i + 1) % n])

        # Use half thickness for shared walls to avoid double-thick walls
        edge_thickness = thickness / 2 if i in shared_edges else thickness

        # Calculate wall direction and normal
        wall_vec = p2 - p1
        wall_length = np.linalg.norm(wall_vec)
        wall_dir = wall_vec / wall_length

        # Normal pointing inward (perpendicular, rotated 90 degrees clockwise)
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Create wall rectangle vertices (outer edge to inner edge)
        outer_p1 = p1
        outer_p2 = p2
        inner_p1 = p1 + normal * edge_thickness
        inner_p2 = p2 + normal * edge_thickness

        # Create polygon for this wall segment
        wall_poly_coords = [
            tuple(outer_p1),
            tuple(outer_p2),
            tuple(inner_p2),
            tuple(inner_p1),
        ]

        wall_poly = Polygon(wall_poly_coords)

        # Extrude to 3D
        wall_mesh = trimesh.creation.extrude_polygon(wall_poly, height=height)

        wall_segments.append((wall_mesh, i))

    return wall_segments


def create_floor_mesh(polygon_coords):
    """Create floor mesh from 2D polygon."""
    poly = Polygon(polygon_coords)
    floor_mesh = trimesh.creation.extrude_polygon(poly, height=0.01)
    return floor_mesh


def create_ceiling_mesh(polygon_coords, height):
    """Create ceiling mesh from 2D polygon at specified height."""
    poly = Polygon(polygon_coords)
    ceiling_mesh = trimesh.creation.extrude_polygon(poly, height=0.01)
    
    # Translate to ceiling height
    ceiling_mesh.apply_translation([0, 0, height - 0.01])
    
    return ceiling_mesh


def create_object_placeholder(position, dimensions, rotation=0):
    """
    Create an arrow-shaped wedge as placeholder for furniture.

    The shape has a clear front (pointed) and back (flat) orientation,
    making it easy for image diffusion models to determine direction.
    The front points in the +Y direction before rotation is applied.

    Args:
        position: [x, y, z] position
        dimensions: [width, depth, height] of bounding box
        rotation: Rotation in degrees around Z-axis

    Returns:
        trimesh.Trimesh: Arrow-shaped wedge mesh
    """
    width = dimensions[0]
    depth = dimensions[1]
    height = dimensions[2]

    # Create arrow/wedge shape vertices (top-down view, front points toward +Y)
    # Shape: Pentagon with pointed front and flat back
    #
    #         (0, depth/2)          <- front point
    #            /\
    #           /  \
    #  (-w/2, 0)    (w/2, 0)        <- side corners (widest point)
    #         |      |
    #         |______|
    # (-w/2, -d/2)  (w/2, -d/2)     <- back corners
    #
    arrow_vertices_2d = [
        (0, depth / 2),              # Front point
        (width / 2, 0),              # Right side
        (width / 2, -depth / 2),     # Back right
        (-width / 2, -depth / 2),    # Back left
        (-width / 2, 0),             # Left side
    ]

    # Create polygon and extrude to 3D
    arrow_polygon = Polygon(arrow_vertices_2d)
    arrow_mesh = trimesh.creation.extrude_polygon(arrow_polygon, height=height)

    # Apply rotation around Z-axis
    if rotation != 0:
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(rotation), [0, 0, 1]
        )
        arrow_mesh.apply_transform(rotation_matrix)

    # Move to final position
    arrow_mesh.apply_translation(position)

    return arrow_mesh


def create_room_geometry(room_data, metadata, all_rooms=None):
    """
    Create complete geometry for a single room.

    Handles door and window cutouts if openings are defined in room_data.
    Also handles shared walls: doors/windows from adjacent rooms are automatically
    cut from shared walls.

    Args:
        room_data: Room definition from JSON
        metadata: Scene metadata (wall thickness, ceiling height, etc.)
        all_rooms: List of all room definitions (for shared wall detection)

    Returns:
        dict: {'wall_segments': [(mesh, idx), ...], 'floor': mesh, 'ceiling': mesh}
    """
    polygon = room_data['floor_polygon']
    height = room_data.get('ceiling_height', metadata['default_ceiling_height'])
    thickness = metadata['wall_thickness']
    openings = room_data.get('openings', [])
    room_id = room_data.get('id', None)

    # Find which edges are shared with other rooms (for half-thickness walls)
    shared_edges = set()
    if all_rooms is not None and len(all_rooms) > 1 and room_id:
        shared_edges = find_shared_edges(room_id, polygon, all_rooms)

    # Check if we need to process openings (either from this room or shared walls)
    has_own_openings = bool(openings)
    has_potential_shared = all_rooms is not None and len(all_rooms) > 1

    # Create room components - walls as individual segments for distinct segmentation colors
    if has_own_openings or has_potential_shared:
        # Use function that handles door/window cutouts and shared walls
        wall_segments = create_wall_segments_with_openings(
            polygon, height, thickness, openings,
            room_id=room_id, all_rooms=all_rooms,
            shared_edges=shared_edges
        )
    else:
        # No openings and no shared walls to check
        wall_segments = create_wall_segments(polygon, height, thickness,
                                             shared_edges=shared_edges)

    floor = create_floor_mesh(polygon)
    ceiling = create_ceiling_mesh(polygon, height)

    return {
        'wall_segments': wall_segments,
        'floor': floor,
        'ceiling': ceiling
    }


def assign_object_colors(objects):
    """
    Assign distinct colors to different object types.
    
    Returns:
        dict: Mapping of object type to RGB color
    """
    # Predefined color palette for common furniture types
    color_map = {
        'sofa': [100, 149, 237],      # Cornflower blue
        'chair': [218, 165, 32],       # Goldenrod
        'table': [139, 69, 19],        # Saddle brown
        'coffee_table': [160, 82, 45], # Sienna
        'dining_table': [139, 69, 19], # Saddle brown
        'bed': [176, 196, 222],        # Light steel blue
        'nightstand': [205, 133, 63],  # Peru
        'wardrobe': [139, 90, 43],     # Tan
        'tv_stand': [85, 107, 47],     # Dark olive green
        'refrigerator': [192, 192, 192], # Silver
        'desk': [160, 82, 45],         # Sienna
        'bookshelf': [139, 69, 19],    # Saddle brown
    }
    
    # Default color for unknown types
    default_color = [128, 128, 128]  # Gray
    
    return {obj_type: color_map.get(obj_type, default_color) 
            for obj_type in set(obj['type'] for obj in objects)}


def generate_scene(json_filepath, output_path='scene_output.glb', include_objects=True):
    """
    Main function to generate complete 3D scene from JSON.

    Args:
        json_filepath: Path to JSON scene description
        output_path: Path for output mesh file
        include_objects: Whether to include furniture objects (default True)
    """
    # Load scene data
    scene_data = load_scene_json(json_filepath)
    metadata = scene_data['metadata']
    rooms = scene_data['rooms']
    objects = scene_data['objects']

    # Create Trimesh scene
    scene = trimesh.Scene()

    # Generate room geometry
    print(f"Generating {len(rooms)} rooms...")
    for room in rooms:
        room_id = room['id']
        openings = room.get('openings', [])
        if openings:
            print(f"  Room '{room_id}': {len(openings)} openings to cut...")
        else:
            print(f"  Room '{room_id}': checking for shared wall openings...")
        room_geom = create_room_geometry(room, metadata, all_rooms=rooms)

        # Add wall segments individually for distinct segmentation colors
        for wall_mesh, wall_idx in room_geom['wall_segments']:
            scene.add_geometry(
                wall_mesh,
                node_name=f"{room_id}_wall_{wall_idx}",
                geom_name=f"{room_id}_wall_{wall_idx}",
                metadata={'color': [240, 240, 240, 255]}  # Light gray walls
            )

        scene.add_geometry(
            room_geom['floor'],
            node_name=f"{room_id}_floor",
            geom_name=f"{room_id}_floor",
            metadata={'color': [205, 170, 125, 255]}  # Tan floor
        )

        scene.add_geometry(
            room_geom['ceiling'],
            node_name=f"{room_id}_ceiling",
            geom_name=f"{room_id}_ceiling",
            metadata={'color': [255, 255, 255, 255]}  # White ceiling
        )

    # Generate object placeholders (if enabled)
    if include_objects:
        # Get color mapping for object types
        object_colors = assign_object_colors(objects)

        print(f"Generating {len(objects)} object placeholders...")
        for obj in objects:
            obj_id = obj['id']
            obj_type = obj['type']
            position = obj['position']
            dimensions = obj['dimensions']
            rotation = obj.get('rotation', 0)

            # Create placeholder pillar
            placeholder = create_object_placeholder(position, dimensions, rotation)

            # Get color for this object type
            color = object_colors[obj_type] + [255]  # Add alpha channel

            # Add to scene
            scene.add_geometry(
                placeholder,
                node_name=f"{obj_id}_{obj_type}",
                geom_name=f"{obj_id}_{obj_type}",
                metadata={'color': color, 'object_type': obj_type}
            )
    else:
        print("Skipping objects (--no-objects flag set)")
    
    # Export scene
    print(f"Exporting scene to {output_path}...")
    scene.export(output_path)
    
    # Print statistics
    print("\n=== Scene Statistics ===")
    print(f"Total geometries: {len(scene.geometry)}")
    print(f"Total triangles: {sum(m.faces.shape[0] for m in scene.geometry.values())}")
    print(f"Total vertices: {sum(m.vertices.shape[0] for m in scene.geometry.values())}")
    print(f"\nRooms: {len(rooms)}")

    if include_objects:
        print(f"Objects: {len(objects)}")

        # Print object type summary
        object_counts = {}
        for obj in objects:
            obj_type = obj['type']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        print("\n=== Object Inventory ===")
        for obj_type, count in sorted(object_counts.items()):
            print(f"  {obj_type}: {count}")
    else:
        print("Objects: 0 (excluded)")
    
    print(f"\n✓ Scene exported successfully to {output_path}")
    print(f"  Open with: trimesh.load('{output_path}').show()")
    
    return scene


def main():
    """Generate scene from JSON and optionally display it."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate 3D scene from JSON description')
    parser.add_argument('--input', default='scene_layout.json',
                       help='Input JSON file (default: scene_layout.json)')
    parser.add_argument('--output', default='scene_output.glb',
                       help='Output mesh file (default: scene_output.glb)')
    parser.add_argument('--show', action='store_true',
                       help='Display scene after generation')
    parser.add_argument('--no-objects', action='store_true',
                       help='Generate scene without furniture objects (structure only)')

    args = parser.parse_args()

    # Generate scene
    include_objects = not args.no_objects
    scene = generate_scene(args.input, args.output, include_objects=include_objects)

    # Optionally display
    if args.show:
        print("\nOpening viewer...")
        scene.show()


if __name__ == '__main__':
    main()
