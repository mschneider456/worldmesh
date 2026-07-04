"""
Validator for scene layout dicts (scenes/*.json format).

Enforces the rules documented in SCENE_CONSTRAINTS.md. All checks assume the
scene is a pure dict (as produced by json.load or our generator). No mesh
backend is touched — this module is pure geometry + graph checks.

Public API:
    validate_scene(scene) -> list[Violation]
    assert_valid(scene)   -> None  (raises ValueError listing every violation)

Design notes:
- Every room polygon is assumed to be an axis-aligned rectangle (4 vertices),
  so all geometric checks boil down to coordinate comparisons — no shapely
  dependency needed.
- Floating-point comparisons use an absolute tolerance of 1e-6 m.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable

EPS = 1e-6

EXPECTED_METADATA = {
    "units": "meters",
    "coordinate_system": "z_up",
    "wall_thickness": 0.15,
    "default_ceiling_height": 2.8,
}

DOOR_WIDTH = 0.9
DOOR_HEIGHT = 2.1
WINDOW_HEIGHT = 1.2
WINDOW_SILL = 0.9
WINDOW_WIDTHS = {1.2, 1.5}


@dataclass
class Violation:
    rule: str
    message: str
    room_id: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.rule}]"
        if self.room_id:
            prefix += f" ({self.room_id})"
        return f"{prefix} {self.message}"


Point = tuple[float, float]
Edge = tuple[Point, Point]


def _approx_eq(a: float, b: float) -> bool:
    return abs(a - b) <= EPS


def _point_eq(a: Point, b: Point) -> bool:
    return _approx_eq(a[0], b[0]) and _approx_eq(a[1], b[1])


def _edge_eq(a: Edge, b: Edge) -> bool:
    """Edges equal ignoring direction."""
    return (_point_eq(a[0], b[0]) and _point_eq(a[1], b[1])) or (
        _point_eq(a[0], b[1]) and _point_eq(a[1], b[0])
    )


def _edge_key(edge: Edge) -> tuple[Point, Point]:
    """Direction-independent key for an edge."""
    p, q = edge
    if (p[0], p[1]) <= (q[0], q[1]):
        return (p, q)
    return (q, p)


def _polygon_edges(polygon: list[list[float]]) -> list[Edge]:
    """Return CCW-oriented edges of a polygon."""
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def _signed_area(polygon: list[list[float]]) -> float:
    pts = polygon
    n = len(pts)
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += (x2 - x1) * (y2 + y1)
    # Negative signed shoelace is CCW for y-up; our shoelace returns -2*area
    # for CCW, so signed area = -s/2.
    return -s / 2.0


def _bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _rect_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """Strict interior overlap (touching does not count)."""
    return not (
        a[2] - EPS <= b[0] or b[2] - EPS <= a[0] or a[3] - EPS <= b[1] or b[3] - EPS <= a[1]
    )


def _edge_length(edge: Edge) -> float:
    dx = edge[1][0] - edge[0][0]
    dy = edge[1][1] - edge[0][1]
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------


def _check_metadata(scene: dict, violations: list[Violation]) -> None:
    md = scene.get("metadata")
    if not isinstance(md, dict):
        violations.append(Violation("metadata_fields", "missing or non-dict metadata"))
        return
    for key, expected in EXPECTED_METADATA.items():
        actual = md.get(key)
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            if not _approx_eq(float(actual), expected):
                violations.append(
                    Violation(
                        "metadata_fields",
                        f"metadata.{key} = {actual!r}, expected {expected!r}",
                    )
                )
        elif actual != expected:
            violations.append(
                Violation(
                    "metadata_fields",
                    f"metadata.{key} = {actual!r}, expected {expected!r}",
                )
            )


def _check_room_geometry(scene: dict, violations: list[Violation]) -> None:
    md = scene.get("metadata", {}) or {}
    default_ch = md.get("default_ceiling_height", 2.8)
    rooms = scene.get("rooms", []) or []
    for room in rooms:
        rid = room.get("id", "<unknown>")
        polygon = room.get("floor_polygon")

        if not isinstance(polygon, list) or len(polygon) != 4:
            violations.append(
                Violation(
                    "polygon_is_rectangle",
                    f"expected 4 vertices, got {0 if polygon is None else len(polygon)}",
                    rid,
                )
            )
            continue

        # Right angles & axis-aligned edges
        axis_ok = True
        for edge in _polygon_edges(polygon):
            dx = edge[1][0] - edge[0][0]
            dy = edge[1][1] - edge[0][1]
            if not (_approx_eq(dx, 0.0) or _approx_eq(dy, 0.0)):
                axis_ok = False
                break
        if not axis_ok:
            violations.append(
                Violation(
                    "polygon_is_rectangle",
                    f"polygon is not axis-aligned: {polygon}",
                    rid,
                )
            )
            continue

        # Check rectangle shape: 2 unique x values, 2 unique y values
        xs = sorted({round(p[0], 6) for p in polygon})
        ys = sorted({round(p[1], 6) for p in polygon})
        if len(xs) != 2 or len(ys) != 2:
            violations.append(
                Violation(
                    "polygon_is_rectangle",
                    f"polygon is not a rectangle: {polygon}",
                    rid,
                )
            )
            continue

        # CCW winding
        if _signed_area(polygon) <= 0:
            violations.append(
                Violation(
                    "polygon_is_ccw",
                    f"polygon is not counter-clockwise: {polygon}",
                    rid,
                )
            )

        # Ceiling height
        ch = room.get("ceiling_height")
        if ch is None or not _approx_eq(float(ch), float(default_ch)):
            violations.append(
                Violation(
                    "ceiling_height_matches",
                    f"ceiling_height = {ch!r}, expected {default_ch!r}",
                    rid,
                )
            )


def _check_disjoint_and_shared_walls(
    scene: dict, violations: list[Violation]
) -> dict[tuple[int, int], Edge]:
    """
    Returns a dict mapping sorted (room_i_idx, room_j_idx) pairs to the
    shared wall edge (undirected, using _edge_key form).
    """
    rooms = scene.get("rooms", []) or []
    shared: dict[tuple[int, int], Edge] = {}
    bboxes = []
    for room in rooms:
        polygon = room.get("floor_polygon")
        if not isinstance(polygon, list) or len(polygon) != 4:
            bboxes.append(None)
            continue
        bboxes.append(_bbox(polygon))

    n = len(rooms)
    for i in range(n):
        for j in range(i + 1, n):
            if bboxes[i] is None or bboxes[j] is None:
                continue
            if _rect_overlap(bboxes[i], bboxes[j]):
                violations.append(
                    Violation(
                        "rooms_disjoint",
                        f"rooms {rooms[i].get('id')!r} and {rooms[j].get('id')!r} overlap",
                    )
                )
                continue

            edges_i = _polygon_edges(rooms[i]["floor_polygon"])
            edges_j = _polygon_edges(rooms[j]["floor_polygon"])

            # Find full-edge matches (shared wall must be an entire edge of each).
            match = None
            for ei in edges_i:
                for ej in edges_j:
                    if _edge_eq(ei, ej):
                        match = (ei, ej)
                        break
                if match:
                    break

            if match:
                ei, ej = match
                # Verify CCW traversal gives opposite directions (each room's
                # edge goes start->end in its own CCW order). For two rooms
                # sharing a wall, the directions must be reversed.
                if not (_point_eq(ei[0], ej[1]) and _point_eq(ei[1], ej[0])):
                    violations.append(
                        Violation(
                            "shared_walls_exact",
                            f"rooms {rooms[i].get('id')!r} and {rooms[j].get('id')!r} "
                            f"share a wall but CCW directions are not reversed: "
                            f"{ei} vs {ej}",
                        )
                    )
                shared[(i, j)] = _edge_key(ei)
                continue

            # No full edge match, but rooms may still touch partially. Detect
            # that: if bounding boxes are adjacent (share an axis-aligned line
            # segment with non-zero length), but no full edge matches, that's
            # a violation of shared_walls_exact.
            a, b = bboxes[i], bboxes[j]
            # Vertical shared line: a.x_max == b.x_min or vice versa.
            touches = False
            if _approx_eq(a[2], b[0]) or _approx_eq(b[2], a[0]):
                y_overlap = min(a[3], b[3]) - max(a[1], b[1])
                if y_overlap > EPS:
                    touches = True
            if _approx_eq(a[3], b[1]) or _approx_eq(b[3], a[1]):
                x_overlap = min(a[2], b[2]) - max(a[0], b[0])
                if x_overlap > EPS:
                    touches = True
            if touches:
                violations.append(
                    Violation(
                        "shared_walls_exact",
                        f"rooms {rooms[i].get('id')!r} and {rooms[j].get('id')!r} "
                        f"touch but the shared segment is not a full edge of both",
                    )
                )

    return shared


def _classify_walls(
    scene: dict, shared: dict[tuple[int, int], Edge]
) -> tuple[dict[tuple[int, tuple[Point, Point]], int], set[tuple[int, tuple[Point, Point]]]]:
    """
    For each room index, collect:
    - interior_edge_to_neighbor: {(room_idx, edge_key): neighbor_room_idx}
    - exterior_edges: set of (room_idx, edge_key)
    """
    rooms = scene.get("rooms", []) or []
    shared_per_pair = shared

    # Map each (room_idx, edge_key) to neighbor room_idx (if any).
    interior: dict[tuple[int, tuple[Point, Point]], int] = {}
    for (i, j), ekey in shared_per_pair.items():
        interior[(i, ekey)] = j
        interior[(j, ekey)] = i

    exterior: set[tuple[int, tuple[Point, Point]]] = set()
    for idx, room in enumerate(rooms):
        polygon = room.get("floor_polygon")
        if not isinstance(polygon, list) or len(polygon) != 4:
            continue
        for edge in _polygon_edges(polygon):
            ekey = _edge_key(edge)
            key = (idx, ekey)
            if key not in interior:
                exterior.add(key)
    return interior, exterior


def _check_openings(
    scene: dict,
    interior: dict[tuple[int, tuple[Point, Point]], int],
    exterior: set[tuple[int, tuple[Point, Point]]],
    violations: list[Violation],
) -> list[tuple[int, int]]:
    """
    Return list of door edges as (room_i_idx, room_j_idx) pairs so the caller
    can do connectivity checks.
    """
    rooms = scene.get("rooms", []) or []
    door_edges: list[tuple[int, int]] = []
    # Track door entries per shared wall; multiple entries are only valid if
    # they describe the same physical door (identical width/height/position).
    door_entries_per_shared: dict[
        tuple[int, tuple[Point, Point]], list[tuple[float, float, float]]
    ] = defaultdict(list)

    for idx, room in enumerate(rooms):
        rid = room.get("id")
        polygon = room.get("floor_polygon")
        if not isinstance(polygon, list) or len(polygon) != 4:
            continue
        room_ccw_edges = _polygon_edges(polygon)
        for opening in room.get("openings", []) or []:
            otype = opening.get("type")
            wall_segment = opening.get("wall_segment")
            if not isinstance(wall_segment, list) or len(wall_segment) != 2:
                violations.append(
                    Violation(
                        "opening_fits_wall",
                        f"wall_segment malformed: {wall_segment!r}",
                        rid,
                    )
                )
                continue
            seg_edge: Edge = (
                (float(wall_segment[0][0]), float(wall_segment[0][1])),
                (float(wall_segment[1][0]), float(wall_segment[1][1])),
            )
            ekey = _edge_key(seg_edge)

            # Check orientation matches CCW traversal of this room.
            ccw_match = any(
                _point_eq(e[0], seg_edge[0]) and _point_eq(e[1], seg_edge[1])
                for e in room_ccw_edges
            )
            if not ccw_match:
                violations.append(
                    Violation(
                        "opening_fits_wall",
                        f"opening wall_segment {seg_edge} is not a CCW edge of room",
                        rid,
                    )
                )
                continue

            wall_len = _edge_length(seg_edge)
            width = float(opening.get("width", 0))
            position = float(opening.get("position", 0))
            if position < -EPS or position + width > wall_len + EPS:
                violations.append(
                    Violation(
                        "opening_fits_wall",
                        f"opening does not fit wall: position={position}, "
                        f"width={width}, wall_len={wall_len}",
                        rid,
                    )
                )

            if otype == "door":
                if (idx, ekey) not in interior:
                    violations.append(
                        Violation(
                            "door_on_shared_wall",
                            f"door on exterior wall {seg_edge}",
                            rid,
                        )
                    )
                    continue
                # Dim check
                if not _approx_eq(width, DOOR_WIDTH):
                    violations.append(
                        Violation(
                            "opening_dims_valid",
                            f"door width {width} != {DOOR_WIDTH}",
                            rid,
                        )
                    )
                if not _approx_eq(float(opening.get("height", 0)), DOOR_HEIGHT):
                    violations.append(
                        Violation(
                            "opening_dims_valid",
                            f"door height {opening.get('height')} != {DOOR_HEIGHT}",
                            rid,
                        )
                    )
                neighbor_idx = interior[(idx, ekey)]
                door_edges.append(tuple(sorted((idx, neighbor_idx))))
                door_entries_per_shared[(min(idx, neighbor_idx), ekey)].append(
                    (
                        round(width, 6),
                        round(float(opening.get("height", 0)), 6),
                        round(position, 6),
                    )
                )

            elif otype == "window":
                if (idx, ekey) not in exterior:
                    violations.append(
                        Violation(
                            "window_on_exterior_wall",
                            f"window on interior (shared) wall {seg_edge}",
                            rid,
                        )
                    )
                    continue
                if width not in {round(w, 6) for w in WINDOW_WIDTHS} and not any(
                    _approx_eq(width, w) for w in WINDOW_WIDTHS
                ):
                    violations.append(
                        Violation(
                            "opening_dims_valid",
                            f"window width {width} not in {sorted(WINDOW_WIDTHS)}",
                            rid,
                        )
                    )
                if not _approx_eq(float(opening.get("height", 0)), WINDOW_HEIGHT):
                    violations.append(
                        Violation(
                            "opening_dims_valid",
                            f"window height {opening.get('height')} != {WINDOW_HEIGHT}",
                            rid,
                        )
                    )
                if not _approx_eq(float(opening.get("sill_height", 0)), WINDOW_SILL):
                    violations.append(
                        Violation(
                            "opening_dims_valid",
                            f"window sill_height {opening.get('sill_height')} != {WINDOW_SILL}",
                            rid,
                        )
                    )
            else:
                violations.append(
                    Violation(
                        "opening_dims_valid",
                        f"unknown opening type {otype!r}",
                        rid,
                    )
                )

    # Dedup check: a shared wall may have multiple door entries only if they
    # describe the same physical door (same width/height/position).
    for key, entries in door_entries_per_shared.items():
        distinct = set(entries)
        if len(distinct) > 1:
            violations.append(
                Violation(
                    "door_not_duplicated",
                    f"shared wall {key[1]} has {len(entries)} conflicting "
                    f"door entries: {sorted(distinct)}",
                )
            )

    # Collapse duplicate door_edges so connectivity counts each physical
    # shared-wall door once.
    door_edges = list(set(door_edges))
    return door_edges


def _check_connectivity(
    scene: dict, door_edges: list[tuple[int, int]], violations: list[Violation]
) -> None:
    rooms = scene.get("rooms", []) or []
    n = len(rooms)
    if n <= 1:
        return
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in door_edges:
        adj[i].add(j)
        adj[j].add(i)
    # BFS from room 0
    seen = {0}
    q = deque([0])
    while q:
        cur = q.popleft()
        for nb in adj[cur]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)
    if len(seen) != n:
        missing = [rooms[i].get("id") for i in range(n) if i not in seen]
        violations.append(
            Violation(
                "graph_connected",
                f"rooms not reachable from room 0 via doors: {missing}",
            )
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_scene(scene: dict) -> list[Violation]:
    """Run every rule. Returns list of violations (empty if valid)."""
    violations: list[Violation] = []
    _check_metadata(scene, violations)
    _check_room_geometry(scene, violations)
    shared = _check_disjoint_and_shared_walls(scene, violations)
    interior, exterior = _classify_walls(scene, shared)
    door_edges = _check_openings(scene, interior, exterior, violations)
    _check_connectivity(scene, door_edges, violations)
    return violations


def assert_valid(scene: dict) -> None:
    """Raise ValueError listing every violation if the scene is invalid."""
    violations = validate_scene(scene)
    if violations:
        msg = "scene failed validation:\n" + "\n".join(f"  - {v}" for v in violations)
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Validate scene layout JSON against scenes rules."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON files or glob patterns to validate.",
    )
    args = parser.parse_args()

    files: list[str] = []
    for p in args.paths:
        matches = sorted(glob.glob(p))
        files.extend(matches if matches else [p])

    failed = 0
    for f in files:
        try:
            with open(f, "r") as fh:
                scene = json.load(fh)
        except Exception as e:
            print(f"{f}: ERROR loading: {e}")
            failed += 1
            continue
        violations = validate_scene(scene)
        if violations:
            failed += 1
            print(f"{f}: {len(violations)} violation(s)")
            for v in violations:
                print(f"  - {v}")
        else:
            print(f"{f}: OK")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
