"""
Simple 2D top-down visualization of the scene layout.
Useful for quick debugging and verification of room/object positions.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import numpy as np


def load_scene_json(filepath):
    """Load scene description from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def visualize_scene_2d(json_filepath, output_path='scene_layout_2d.png'):
    """
    Create a 2D top-down view of the scene layout.
    
    Args:
        json_filepath: Path to JSON scene description
        output_path: Path for output image
    """
    # Load scene data
    scene_data = load_scene_json(json_filepath)
    rooms = scene_data['rooms']
    objects = scene_data['objects']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palettes
    room_colors = ['#FFE5E5', '#E5F5FF', '#F0FFE5', '#FFF5E5', '#F5E5FF']
    object_colors = {
        'sofa': '#6495ED',
        'chair': '#DAA520',
        'table': '#8B4513',
        'coffee_table': '#A0522D',
        'dining_table': '#8B4513',
        'bed': '#B0C4DE',
        'nightstand': '#CD853F',
        'wardrobe': '#8B5A2B',
        'tv_stand': '#556B2F',
        'refrigerator': '#C0C0C0',
        'desk': '#A0522D',
        'bookshelf': '#8B4513',
    }
    
    # Draw rooms
    for i, room in enumerate(rooms):
        polygon = room['floor_polygon']
        poly_array = np.array(polygon + [polygon[0]])  # Close polygon
        
        # Fill room
        room_color = room_colors[i % len(room_colors)]
        ax.fill(poly_array[:, 0], poly_array[:, 1], 
                color=room_color, alpha=0.3, edgecolor='black', linewidth=2)
        
        # Add room label
        center_x = np.mean([p[0] for p in polygon])
        center_y = np.mean([p[1] for p in polygon])
        ax.text(center_x, center_y, room['name'], 
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Draw wall thickness indicator (just for visualization)
        thickness = scene_data['metadata']['wall_thickness']
        wall_poly_array = np.array(polygon + [polygon[0]])
        ax.plot(wall_poly_array[:, 0], wall_poly_array[:, 1], 
                'k-', linewidth=3, label='Walls' if i == 0 else '')
        
        # Draw doors and windows
        for opening in room.get('openings', []):
            wall_segment = opening['wall_segment']
            position_along_wall = opening['position']
            width = opening['width']
            
            # Calculate position on wall
            wall_start = np.array(wall_segment[0])
            wall_end = np.array(wall_segment[1])
            wall_vec = wall_end - wall_start
            wall_len = np.linalg.norm(wall_vec)
            wall_dir = wall_vec / wall_len
            
            # Opening position
            opening_start = wall_start + wall_dir * position_along_wall
            opening_end = opening_start + wall_dir * width
            
            if opening['type'] == 'door':
                ax.plot([opening_start[0], opening_end[0]], 
                       [opening_start[1], opening_end[1]], 
                       'r-', linewidth=4, label='Door' if i == 0 else '')
                # Draw door swing arc
                ax.add_patch(patches.Arc(opening_start, width, width, 
                                        angle=np.degrees(np.arctan2(wall_dir[1], wall_dir[0])),
                                        theta1=0, theta2=90, 
                                        color='red', linestyle='--', linewidth=1))
            else:  # window
                ax.plot([opening_start[0], opening_end[0]], 
                       [opening_start[1], opening_end[1]], 
                       'b-', linewidth=4, label='Window' if i == 0 else '')
    
    # Draw objects
    legend_objects = set()
    for obj in objects:
        obj_type = obj['type']
        position = obj['position']
        dimensions = obj['dimensions']
        rotation = obj.get('rotation', 0)
        
        # Get color
        color = object_colors.get(obj_type, '#808080')
        
        # Draw rectangle for object footprint
        x, y = position[0], position[1]
        width, depth = dimensions[0], dimensions[1]
        
        # Create rotated rectangle
        angle_rad = np.radians(rotation)
        rect = Rectangle(
            (x - width/2, y - depth/2), width, depth,
            angle=rotation,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add object label
        label_text = f"{obj_type}\n({obj['id']})"
        ax.text(x, y, label_text, ha='center', va='center', 
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, 
                         edgecolor='black', alpha=0.9))
        
        # Add to legend (only once per type)
        if obj_type not in legend_objects:
            legend_objects.add(obj_type)
            ax.plot([], [], 's', color=color, markersize=10, 
                   label=obj_type.replace('_', ' ').title())
    
    # Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Scene Layout - Top-Down View', fontsize=16, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
             borderaxespad=0, fontsize=9)
    
    # Add coordinate system indicator
    ax.annotate('', xy=(0.5, 0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.3, 0.3, 'Origin', fontsize=9, color='red')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D visualization saved to: {output_path}")
    
    return fig, ax


def main():
    """Generate 2D visualization from JSON."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 2D top-down scene visualization')
    parser.add_argument('--input', default='scene_layout.json',
                       help='Input JSON file (default: scene_layout.json)')
    parser.add_argument('--output', default='scene_layout_2d.png',
                       help='Output image file (default: scene_layout_2d.png)')
    parser.add_argument('--show', action='store_true',
                       help='Display visualization after generation')
    
    args = parser.parse_args()
    
    # Generate visualization
    fig, ax = visualize_scene_2d(args.input, args.output)
    
    # Optionally display
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    main()
