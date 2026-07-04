"""
Programmatically generate scene layouts in the scenes/ format.

Given a shape template (linear, grid, lshape, ushape, tshape, plus, zigzag,
corridor, rectangle, compound) and a room count, build a valid scene JSON +
matching PNG (rendered via visualize_2d.visualize_scene_2d).

The result is validated with scene_constraints.assert_valid before writing.

Example:
    python generate_scene_layout.py \
        --shape ushape \
        --num-rooms 5 \
        --room-width 6 --room-depth 4 \
        --output-dir scenes_generated

Rules honored (see SCENE_CONSTRAINTS.md):
- Rooms are axis-aligned rectangles on a uniform cell grid.
- Columns and rows can be jittered independently so that walls stay aligned.
- Doors are placed on every shared (interior) wall, width 0.9 centered.
  Each door is listed in exactly one room (the one with the lower index).
- Windows are placed on every exterior wall, width 1.5 on walls ≥ 5 m and
  1.2 on shorter walls, centered.
- A themed room-name pool is seeded-shuffled; the cell closest to the origin
  gets the foyer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Callable

from scene_constraints import assert_valid, validate_scene

# ---------------------------------------------------------------------------
# Shape templates: each returns a sorted list of (row, col) occupied cells.
# All cells are on a uniform integer grid; metric coordinates come later.
# ---------------------------------------------------------------------------


def shape_linear(n: int) -> list[tuple[int, int]]:
    if n < 1:
        raise ValueError("linear requires n >= 1")
    return [(0, c) for c in range(n)]


def shape_grid(n: int) -> list[tuple[int, int]]:
    if n < 1:
        raise ValueError("grid requires n >= 1")
    # Prefer a complete rectangular factorization closest to square.
    best = None
    for rows in range(1, int(math.isqrt(n)) + 1):
        if n % rows == 0:
            cols = n // rows
            score = abs(rows - cols)
            if best is None or score < best[0]:
                best = (score, rows, cols)
    if best is None:
        rows = int(math.isqrt(n)) or 1
        cols = math.ceil(n / rows)
        return [(r, c) for r in range(rows) for c in range(cols)][:n]
    _, rows, cols = best
    return [(r, c) for r in range(rows) for c in range(cols)]


def shape_rectangle(n: int) -> list[tuple[int, int]]:
    # Alias for grid with an additional preference for wider (2xN or 3xN).
    if n < 2:
        raise ValueError("rectangle requires n >= 2")
    best = None
    for rows in (2, 3, 1):
        if n % rows == 0:
            cols = n // rows
            if best is None or abs(rows - cols) < abs(best[1] - best[2]):
                best = (0, rows, cols)
    if best is None:
        return shape_grid(n)
    _, rows, cols = best
    return [(r, c) for r in range(rows) for c in range(cols)]


def shape_lshape(n: int) -> list[tuple[int, int]]:
    if n < 3:
        raise ValueError("lshape requires n >= 3")
    # Two arms sharing a corner: w + h - 1 = n, minimize |w - h|.
    w = (n + 1) // 2
    h = n + 1 - w
    cells = [(0, c) for c in range(w)]
    cells += [(r, 0) for r in range(1, h)]
    return sorted(set(cells))


def shape_ushape(n: int) -> list[tuple[int, int]]:
    if n < 5:
        raise ValueError("ushape requires n >= 5")
    # Pick base b ≥ 3 and arms (l, r) with l + r + b = n, preferring l == r
    # then the smallest base.
    best = None
    for b in range(3, n - 1):
        rem = n - b
        if rem < 2:
            continue
        l = rem // 2
        r = rem - l
        if r < 1:
            continue
        score = abs(l - r) * 10 + (b - 3)
        if best is None or score < best[0]:
            best = (score, b, l, r)
    if best is None:
        raise ValueError(f"ushape({n}) cannot be built")
    _, b, l, r = best
    cells = [(0, x) for x in range(b)]
    cells += [(y, 0) for y in range(1, l + 1)]
    cells += [(y, b - 1) for y in range(1, r + 1)]
    return sorted(set(cells))


def shape_tshape(n: int) -> list[tuple[int, int]]:
    if n < 4:
        raise ValueError("tshape requires n >= 4")
    bw = 3  # fixed bar width; stem extends from its centre
    sh = n - bw
    if sh < 1:
        raise ValueError(f"tshape({n}) needs at least 4 rooms")
    cells = [(0, c) for c in range(bw)]
    cells += [(r, 1) for r in range(1, sh + 1)]
    return sorted(set(cells))


def shape_plus(n: int) -> list[tuple[int, int]]:
    if n < 5 or (n - 1) % 4 != 0:
        raise ValueError("plus requires n in {5, 9, 13, 17, ...}")
    k = (n - 1) // 4
    cells = [(k, k)]
    for i in range(1, k + 1):
        cells.append((k, k - i))
        cells.append((k, k + i))
        cells.append((k - i, k))
        cells.append((k + i, k))
    return sorted(set(cells))


def shape_zigzag(n: int) -> list[tuple[int, int]]:
    if n < 2:
        raise ValueError("zigzag requires n >= 2")
    cells = []
    r = c = 0
    for i in range(n):
        cells.append((r, c))
        if i % 2 == 0:
            c += 1
        else:
            r += 1
    return sorted(set(cells))


def shape_corridor(n: int) -> list[tuple[int, int]]:
    if n < 4:
        raise ValueError("corridor requires n >= 4")
    # 3-cell horizontal bar with a stem extending upward from the middle.
    # Middle-of-bar cell is the hallway; side-of-bar cells are rooms; stem
    # cells are rooms reached via the hallway.
    cells = [(0, 0), (0, 1), (0, 2)]
    for i in range(1, n - 2):
        cells.append((i, 1))
    return sorted(set(cells))


def shape_compound(n: int, rng: random.Random) -> list[tuple[int, int]]:
    if n < 1:
        raise ValueError("compound requires n >= 1")
    cells: set[tuple[int, int]] = {(0, 0)}
    while len(cells) < n:
        candidates: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb not in cells:
                    candidates.add(nb)
        if not candidates:
            break
        choice = rng.choice(sorted(candidates))
        cells.add(choice)
    return sorted(cells)


SHAPES: dict[str, Callable] = {
    "linear": shape_linear,
    "grid": shape_grid,
    "rectangle": shape_rectangle,
    "lshape": shape_lshape,
    "ushape": shape_ushape,
    "tshape": shape_tshape,
    "plus": shape_plus,
    "zigzag": shape_zigzag,
    "corridor": shape_corridor,
    # compound takes rng, wrapped below
}


def _normalize_cells(
    cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Translate so the minimum row/col is 0 and sort."""
    if not cells:
        return cells
    min_r = min(c[0] for c in cells)
    min_c = min(c[1] for c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def run_shape(shape: str, n: int, rng: random.Random) -> list[tuple[int, int]]:
    if shape == "compound":
        return _normalize_cells(shape_compound(n, rng))
    if shape not in SHAPES:
        raise ValueError(
            f"unknown shape {shape!r}; choose from "
            f"{sorted(list(SHAPES.keys()) + ['compound'])}"
        )
    return _normalize_cells(SHAPES[shape](n))


# ---------------------------------------------------------------------------
# Cell → metric coordinates
# ---------------------------------------------------------------------------


def compute_axis_sizes(
    num: int, base: float, jitter: float, rng: random.Random
) -> list[float]:
    sizes: list[float] = []
    for _ in range(num):
        if jitter > 0:
            delta = rng.uniform(-jitter * base, jitter * base)
            raw = base + delta
            # Round to nearest 0.5 m and clamp to [4, 6].
            size = round(raw * 2) / 2
        else:
            size = float(base)
        size = max(4.0, min(6.0, size))
        sizes.append(size)
    return sizes


def cell_rect(
    row: int, col: int, col_widths: list[float], row_heights: list[float]
) -> list[list[float]]:
    x0 = sum(col_widths[:col])
    x1 = x0 + col_widths[col]
    y0 = sum(row_heights[:row])
    y1 = y0 + row_heights[row]
    # CCW: SW, SE, NE, NW.
    return [
        [round(x0, 2), round(y0, 2)],
        [round(x1, 2), round(y0, 2)],
        [round(x1, 2), round(y1, 2)],
        [round(x0, 2), round(y1, 2)],
    ]


# ---------------------------------------------------------------------------
# Room name assignment
# ---------------------------------------------------------------------------

ROOM_POOL = [
    "living_room",
    "dining_room",
    "kitchen",
    "master_bedroom",
    "guest_bedroom",
    "family_room",
    "home_office",
    "study",
    "library",
    "home_gym",
    "sunroom",
    "game_room",
    "music_room",
    "art_studio",
    "conservatory",
    "media_room",
    "mudroom",
    "laundry_room",
    "wine_cellar",
    "spa_room",
    "dressing_room",
    "breakfast_room",
    "nursery",
    "hallway",
    "sitting_room",
    "lounge",
]


def title_case(rid: str) -> str:
    return rid.replace("_", " ").title()


def assign_names(
    cells: list[tuple[int, int]], rng: random.Random
) -> list[tuple[str, str]]:
    origin_cell = min(cells, key=lambda c: (c[0] + c[1], c[0], c[1]))
    pool = list(ROOM_POOL)
    rng.shuffle(pool)
    names: dict[tuple[int, int], tuple[str, str]] = {}
    names[origin_cell] = ("foyer", "Foyer")
    idx = 0
    for cell in cells:
        if cell == origin_cell:
            continue
        if idx < len(pool):
            rid = pool[idx]
            idx += 1
        else:
            rid = f"room_{cells.index(cell)}"
        names[cell] = (rid, title_case(rid))
    # Return in the same order as cells
    return [names[c] for c in cells]


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------


def edge_length(edge: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return math.hypot(edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])


def build_scene(
    shape: str,
    num_rooms: int,
    room_width: float,
    room_depth: float,
    jitter: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    cells = run_shape(shape, num_rooms, rng)
    if not cells:
        raise ValueError("shape produced no cells")

    max_row = max(c[0] for c in cells)
    max_col = max(c[1] for c in cells)
    col_widths = compute_axis_sizes(max_col + 1, room_width, jitter, rng)
    row_heights = compute_axis_sizes(max_row + 1, room_depth, jitter, rng)

    polygons = [cell_rect(r, c, col_widths, row_heights) for (r, c) in cells]
    name_pairs = assign_names(cells, rng)

    rooms = []
    for i, (cell, poly, (rid, rname)) in enumerate(zip(cells, polygons, name_pairs)):
        rooms.append(
            {
                "id": rid,
                "name": rname,
                "floor_polygon": poly,
                "ceiling_height": 2.8,
                "openings": [],
                "_cell": cell,
                "_polygon_points": [tuple(p) for p in poly],
            }
        )

    cell_to_idx = {cell: i for i, cell in enumerate(cells)}

    # CCW edges per room (order: S, E, N, W).
    def room_ccw_edges(idx):
        p = rooms[idx]["_polygon_points"]
        return [
            (p[0], p[1]),  # south
            (p[1], p[2]),  # east
            (p[2], p[3]),  # north
            (p[3], p[0]),  # west
        ]

    # Find shared walls via cell adjacency.
    # For each cell, check east neighbor (same row, col+1) and north neighbor
    # (row+1, same col). This covers all adjacency pairs exactly once.
    shared_walls: list[tuple[int, int, str]] = []  # (low_idx, high_idx, direction)
    for cell, i in cell_to_idx.items():
        r, c = cell
        east = (r, c + 1)
        if east in cell_to_idx:
            j = cell_to_idx[east]
            shared_walls.append((min(i, j), max(i, j), "EW"))
        north = (r + 1, c)
        if north in cell_to_idx:
            j = cell_to_idx[north]
            shared_walls.append((min(i, j), max(i, j), "NS"))

    # Build interior edge set keyed by (room_idx, frozenset(edge_points)).
    def edge_key(edge):
        a, b = edge
        if a <= b:
            return (a, b)
        return (b, a)

    interior_keys: set[tuple[int, tuple]] = set()
    for i, j, direction in shared_walls:
        # Determine the CCW edge in each room for this shared wall.
        if direction == "EW":
            # i and j are in the same row, j is east of i
            # Left room (i if i.col < j.col): east edge
            # Right room (j): west edge
            if rooms[i]["_cell"][1] < rooms[j]["_cell"][1]:
                left, right = i, j
            else:
                left, right = j, i
            left_edge = room_ccw_edges(left)[1]  # east
            right_edge = room_ccw_edges(right)[3]  # west
            # Place door in the room with the lower global index.
            door_room = min(i, j)
            door_edge = left_edge if door_room == left else right_edge
        else:  # "NS"
            if rooms[i]["_cell"][0] < rooms[j]["_cell"][0]:
                below, above = i, j
            else:
                below, above = j, i
            below_edge = room_ccw_edges(below)[2]  # north
            above_edge = room_ccw_edges(above)[0]  # south
            door_room = min(i, j)
            door_edge = below_edge if door_room == below else above_edge
            left_edge = below_edge
            right_edge = above_edge
            left, right = below, above

        interior_keys.add((left, edge_key(left_edge)))
        interior_keys.add((right, edge_key(right_edge)))

        wall_len = edge_length(door_edge)
        door_width = 0.9
        if wall_len < door_width + 0.3:
            # Too tight for a door — skip, and accept the validation error
            # later (shouldn't happen with default 4-6 m rooms).
            continue
        position = round((wall_len - door_width) / 2, 2)
        rooms[door_room]["openings"].append(
            {
                "type": "door",
                "wall_segment": [list(door_edge[0]), list(door_edge[1])],
                "position": position,
                "width": door_width,
                "height": 2.1,
            }
        )

    # Place windows on every exterior edge.
    for idx, room in enumerate(rooms):
        for edge in room_ccw_edges(idx):
            ekey = edge_key(edge)
            if (idx, ekey) in interior_keys:
                continue
            wall_len = edge_length(edge)
            win_width = 1.5 if wall_len >= 5.0 - 1e-6 else 1.2
            if wall_len < win_width + 0.3:
                continue
            position = round((wall_len - win_width) / 2, 2)
            room["openings"].append(
                {
                    "type": "window",
                    "wall_segment": [list(edge[0]), list(edge[1])],
                    "position": position,
                    "width": win_width,
                    "height": 1.2,
                    "sill_height": 0.9,
                }
            )

    # Strip internal scratch fields.
    for room in rooms:
        room.pop("_cell", None)
        room.pop("_polygon_points", None)

    scene = {
        "metadata": {
            "units": "meters",
            "coordinate_system": "z_up",
            "wall_thickness": 0.15,
            "default_ceiling_height": 2.8,
        },
        "rooms": rooms,
        "objects": [],
    }

    assert_valid(scene)
    return scene


# ---------------------------------------------------------------------------
# Output: JSON + PNG
# ---------------------------------------------------------------------------


def _dump_json_pretty(scene: dict) -> str:
    """
    Produce JSON in the compact-per-opening / compact-per-polygon style used
    by scenes/. This is cosmetic — json.load() parses both formats.
    """
    def compact_list(obj):
        # For a list of 2-lists of numbers (polygon), print inline.
        return json.dumps(obj, separators=(", ", ": "))

    lines = []
    lines.append("{")
    lines.append('  "metadata": ' + json.dumps(scene["metadata"], indent=2).replace(
        "\n", "\n  "
    ) + ",")
    lines.append('  "rooms": [')
    room_strs = []
    for room in scene["rooms"]:
        parts = []
        parts.append("    {")
        parts.append(f'      "id": {json.dumps(room["id"])},')
        parts.append(f'      "name": {json.dumps(room["name"])},')
        parts.append(f'      "floor_polygon": {compact_list(room["floor_polygon"])},')
        parts.append(f'      "ceiling_height": {room["ceiling_height"]},')
        if room["openings"]:
            parts.append('      "openings": [')
            op_strs = [
                "        " + json.dumps(op, separators=(", ", ": "))
                for op in room["openings"]
            ]
            parts.append(",\n".join(op_strs))
            parts.append("      ]")
        else:
            parts.append('      "openings": []')
        parts.append("    }")
        room_strs.append("\n".join(parts))
    lines.append(",\n".join(room_strs))
    lines.append("  ],")
    lines.append('  "objects": []')
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_scene(scene: dict, json_path: str, png_path: str) -> bool:
    """Write JSON and PNG. Returns True iff the PNG was rendered."""
    with open(json_path, "w") as f:
        f.write(_dump_json_pretty(scene))

    # Lazy import so the generator can run without matplotlib if the caller
    # only wants JSON.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualize_2d import visualize_scene_2d

        visualize_scene_2d(json_path, png_path)
        plt.close("all")
        return True
    except Exception as e:
        print(f"PNG rendering failed ({e}); JSON written to {json_path}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a scene layout JSON + PNG in the scenes/ format.",
    )
    parser.add_argument(
        "--shape",
        required=True,
        choices=sorted(list(SHAPES.keys()) + ["compound"]),
        help="Layout shape template.",
    )
    parser.add_argument(
        "--num-rooms",
        type=int,
        required=True,
        help="Total number of rooms.",
    )
    parser.add_argument(
        "--room-width",
        type=float,
        default=6.0,
        help="Base room width (x-axis / column size) in meters. Default 6.",
    )
    parser.add_argument(
        "--room-depth",
        type=float,
        default=4.0,
        help="Base room depth (y-axis / row size) in meters. Default 4.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Per-row / per-col size jitter as a fraction of base (0..1). "
        "Default 0 (uniform).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for jitter, compound accretion, and name shuffling.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output file stem. Defaults to scene_{num_rooms}rooms_{shape}.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory (created if missing). Default: current dir.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build the scene in memory and validate it, then exit without "
        "writing files.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.jitter <= 1.0:
        parser.error("--jitter must be in [0, 1]")
    if args.room_width <= 0 or args.room_depth <= 0:
        parser.error("--room-width and --room-depth must be positive")

    scene = build_scene(
        shape=args.shape,
        num_rooms=args.num_rooms,
        room_width=args.room_width,
        room_depth=args.room_depth,
        jitter=args.jitter,
        seed=args.seed,
    )

    violations = validate_scene(scene)
    if violations:
        print("Generated scene failed validation:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    if args.validate_only:
        print(f"OK: {args.shape} with {args.num_rooms} rooms is valid.")
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or f"scene_{args.num_rooms}rooms_{args.shape}"
    json_path = os.path.join(args.output_dir, f"{name}.json")
    png_path = os.path.join(args.output_dir, f"{name}.png")
    png_ok = write_scene(scene, json_path, png_path)
    print(f"Wrote {json_path}")
    if png_ok:
        print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
