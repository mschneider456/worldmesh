# Scene Layout Constraints

This document describes the rules that every valid scene layout (`scenes/*.json` style) must satisfy. These rules are:

1. Enforced programmatically by `scene_constraints.py` (`validate_scene`, `assert_valid`).
2. Produced by `generate_scene_layout.py`, which calls `assert_valid` before writing output.

Each rule has a short `rule_id` (used by the validator) and a longer description.

---

## Metadata (`metadata_fields`)

The top-level `metadata` block must contain:

- `units = "meters"`
- `coordinate_system = "z_up"`
- `wall_thickness = 0.15`
- `default_ceiling_height = 2.8`

## Room geometry

### `polygon_is_rectangle`

Every `floor_polygon` must be an axis-aligned rectangle with exactly 4 vertices and right angles. Each edge must be either horizontal or vertical.

### `polygon_is_ccw`

Vertices must be listed counter-clockwise (positive signed area) when viewed from above. The expected pattern is `[SW, SE, NE, NW]`.

### `ceiling_height_matches`

Every room's `ceiling_height` must equal `metadata.default_ceiling_height` (2.8 m).

## Layout

### `rooms_disjoint`

No two rooms may have overlapping interiors. Rooms may touch along a wall but never overlap in area.

### `shared_walls_exact`

When two rooms touch, the shared region must be a complete single wall of both rooms (not a partial overlap). Concretely: the intersection of their boundaries is a line segment that is an entire edge of each rectangle, and each room lists that edge in its CCW polygon traversal in opposite directions.

## Openings: doors

### `door_on_shared_wall`

A door may only appear on an **interior** wall — a wall shared with a neighboring room. Doors are never placed on exterior walls. (Hence scenes houses have no explicit "front door"; the houses are enclosed.)

### `door_not_duplicated`

Each shared wall corresponds to **at most one physical door**. By convention the door is listed in the `openings` of only one of the two rooms sharing the wall. If both rooms happen to list a door on the same shared wall, the two entries must agree on width/height/position (they describe the same door). `generate_scene_layout.py` writes the door in exactly one room to keep outputs clean.

### `graph_connected`

Build an undirected graph where each node is a room and each door is an edge between the two rooms it connects. This graph must be connected — every room must be reachable from every other room via doors.

### Door dimensions (`opening_dims_valid`)

- `width = 0.9`
- `height = 2.1`

### Door position

Position along the wall is always centered: `position = (wall_length - 0.9) / 2`. Examples:

| Wall length | Position |
|-------------|----------|
| 4 m         | 1.55 m   |
| 5 m         | 2.05 m   |
| 6 m         | 2.55 m   |

## Openings: windows

### `window_on_exterior_wall`

A window may only appear on an **exterior** wall — a wall not shared with any other room. Windows are never placed between rooms.

### Window dimensions (`opening_dims_valid`)

- `height = 1.2`
- `sill_height = 0.9`
- `width ∈ {1.2, 1.5}` — 1.5 is used on long walls (≥ 5 m), 1.2 on shorter walls (< 5 m).

### Window position

Centered: `position = (wall_length - width) / 2`. Examples:

| Wall length | Window width | Position |
|-------------|--------------|----------|
| 4 m         | 1.2 m        | 1.4 m    |
| 5 m         | 1.5 m        | 1.75 m   |
| 6 m         | 1.5 m        | 2.25 m   |

## Openings: general

### `opening_fits_wall`

For every opening: `0 ≤ position` and `position + width ≤ wall_length`.

### Wall segment orientation

The `wall_segment` stored in an opening must be oriented in the same direction as the polygon's counter-clockwise traversal. A shared wall therefore appears with opposite orientations in the two neighboring rooms' polygons.

## Objects

The `objects` field is currently always `[]` in scenes and in output from `generate_scene_layout.py`. Furniture is added downstream by the pipeline, not at layout authoring time.

---

## Rule id → section quick reference

| Rule id                      | Section                      |
|------------------------------|------------------------------|
| `metadata_fields`            | Metadata                     |
| `polygon_is_rectangle`       | Room geometry                |
| `polygon_is_ccw`             | Room geometry                |
| `ceiling_height_matches`     | Room geometry                |
| `rooms_disjoint`             | Layout                       |
| `shared_walls_exact`         | Layout                       |
| `door_on_shared_wall`        | Openings: doors              |
| `window_on_exterior_wall`    | Openings: windows            |
| `door_not_duplicated`        | Openings: doors              |
| `opening_fits_wall`          | Openings: general            |
| `graph_connected`            | Openings: doors              |
| `opening_dims_valid`         | Openings: doors / windows    |
