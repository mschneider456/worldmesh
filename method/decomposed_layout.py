"""
Decomposed scene-layout generation: the LLM only emits room polygons + door
adjacency pairs; Python deterministically generates all openings (doors and
windows) with correct CCW wall_segment direction, centered positions, and
the exact dimensions required by scene_constraints.

Public API:
    DECOMPOSED_SCHEMA              -- JSON Schema for vLLM guided_json
    build_decomposed_system_prompt(repo_root) -> str
    decomposed_user_message(prompt) -> str
    decomposed_correction_message(prompt, issues) -> str
    parse_decomposed(raw_text) -> tuple[dict | None, list[str], str]
    compose_full_scene(decomposed) -> tuple[dict, list[str]]
    to_decomposed(full_scene) -> dict     -- (used to build few-shot examples)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from scene_constraints import (
    DOOR_HEIGHT,
    DOOR_WIDTH,
    EPS,
    EXPECTED_METADATA,
    WINDOW_HEIGHT,
    WINDOW_SILL,
    _edge_key,
    _edge_length,
    _point_eq,
    _polygon_edges,
)


DECOMPOSED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rooms", "doors"],
    "properties": {
        "rooms": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "name", "floor_polygon"],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                    "name": {"type": "string"},
                    "floor_polygon": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                        },
                    },
                },
            },
        },
        "doors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["between"],
                "properties": {
                    "between": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "string"},
                    }
                },
            },
        },
    },
}


_PROMPT_HEAD = """You design simple house layouts.

Output exactly one JSON object and nothing else — no Markdown, no comments, no \
explanations. Schema:

{
  "rooms": [
    {"id": "snake_case_id", "name": "Display Name", "floor_polygon": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]}
  ],
  "doors": [
    {"between": ["room_a_id", "room_b_id"]}
  ]
}

Rules:
- All coordinates are in meters; the frame is z-up; the floor is z=0.
- Each `floor_polygon` is a 4-vertex axis-aligned RECTANGLE in counter-clockwise order: \
[SW, SE, NE, NW] = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]] with x1 > x0 and y1 > y0. \
Never list vertices clockwise or in any other order.
  CORRECT (CCW) for a 6×4 m room rooted at the origin: [[0,0],[6,0],[6,4],[0,4]].
  WRONG (CW): [[0,0],[0,4],[6,4],[6,0]] — same shape but clockwise.
- Rooms must NOT overlap. When two rooms touch, they must share a COMPLETE wall edge \
(the shared segment is one full side of each rectangle). Partial overlaps are forbidden.
  Example of "share a complete wall": rooms `[[0,0],[6,0],[6,4],[0,4]]` and \
`[[6,0],[12,0],[12,4],[6,4]]` share the full edge x=6 from y=0 to y=4.
- Each door connects two rooms that share a wall. Express as \
`{"between":["a","b"]}`. Do not include doors between non-adjacent rooms.
- The door graph must be CONNECTED — every room reachable from every other via doors.
- Use snake_case ids: foyer, living_room, dining_room, kitchen, master_bedroom, \
bedroom, bathroom, hallway, study, office, etc.
- Do NOT include openings, ceiling_height, metadata, or `objects` fields. They are \
generated automatically downstream from your polygons + doors.
- Choose room sizes that make physical sense: typically 3–6 m per side.

Self-check before responding:
- Every polygon is a 4-vertex CCW axis-aligned rectangle.
- No two rooms overlap; touching rooms share full-edge walls.
- Every door references two rooms that actually share a wall.
- The door graph is connected.
"""


_FEW_SHOT_PATHS = [
    "method/scenes/scene_3rooms_linear.json",
    "method/scenes/scene_3rooms_lshape.json",
    "method/scenes/scene_4rooms_ushape.json",
]


def to_decomposed(full_scene: dict) -> dict:
    """Convert a full scenes-style dict into the decomposed (polygons + doors) form."""
    rooms_in = full_scene.get("rooms", []) or []
    rooms_out = [
        {
            "id": r["id"],
            "name": r["name"],
            "floor_polygon": r["floor_polygon"],
        }
        for r in rooms_in
        if isinstance(r, dict) and "id" in r and "floor_polygon" in r
    ]

    rooms_by_id = {r["id"]: r for r in rooms_in}
    door_pairs: set[tuple[str, str]] = set()

    for r in rooms_in:
        for opening in r.get("openings", []) or []:
            if opening.get("type") != "door":
                continue
            wall = opening.get("wall_segment")
            if not (isinstance(wall, list) and len(wall) == 2):
                continue
            try:
                p1 = (float(wall[0][0]), float(wall[0][1]))
                p2 = (float(wall[1][0]), float(wall[1][1]))
            except (TypeError, ValueError, IndexError):
                continue
            wall_rev = (p2, p1)
            for other_id, other_room in rooms_by_id.items():
                if other_id == r["id"]:
                    continue
                other_polygon = other_room.get("floor_polygon", [])
                if not (isinstance(other_polygon, list) and len(other_polygon) == 4):
                    continue
                try:
                    other_edges = _polygon_edges(other_polygon)
                except Exception:
                    continue
                if any(
                    _point_eq(e[0], wall_rev[0]) and _point_eq(e[1], wall_rev[1])
                    for e in other_edges
                ):
                    door_pairs.add(tuple(sorted([r["id"], other_id])))
                    break

    doors_out = [{"between": list(pair)} for pair in sorted(door_pairs)]
    return {"rooms": rooms_out, "doors": doors_out}


def _format_example(label: str, decomposed: dict) -> str:
    return f"Example ({label}):\n{json.dumps(decomposed, indent=2)}"


def build_decomposed_system_prompt(repo_root: Path) -> str:
    examples_blocks: list[str] = []
    for rel in _FEW_SHOT_PATHS:
        path = repo_root / rel
        if not path.exists():
            continue
        try:
            full = json.loads(path.read_text())
        except Exception:
            continue
        decomposed = to_decomposed(full)
        examples_blocks.append(_format_example(path.stem, decomposed))

    if not examples_blocks:
        return _PROMPT_HEAD

    return _PROMPT_HEAD + "\n\nExamples follow.\n\n" + "\n\n".join(examples_blocks)


def decomposed_user_message(prompt: str) -> str:
    return (
        "Design a house layout for this request:\n"
        f"{prompt.strip()}\n\n"
        "Output only the decomposed JSON object (rooms + doors) per the schema."
    )


def decomposed_correction_message(prompt: str, issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Your previous layout has the following problems:\n"
        f"{bullets}\n\n"
        f"Original request: {prompt.strip()}\n\n"
        "Output a corrected decomposed JSON object. Pay special attention to: "
        "axis-aligned 4-vertex CCW rectangles ([SW, SE, NE, NW]), rooms not "
        "overlapping, touching rooms sharing complete walls, doors only between "
        "actually-adjacent rooms, and the door graph being connected."
    )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_decomposed(raw_text: str) -> tuple[dict | None, list[str], str]:
    """Parse a model response into a decomposed dict.

    Returns: (decomposed_or_None, parse_issues, normalized_text)
    """
    cleaned = _strip_code_fences(raw_text).lstrip()
    if cleaned and not cleaned.startswith("{"):
        cleaned = "{" + cleaned

    issues: list[str] = []
    if not cleaned:
        return None, ["[response_empty] empty response"], cleaned

    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        return (
            None,
            [
                "[response_json] not valid JSON: "
                f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            ],
            cleaned,
        )

    if not isinstance(parsed, dict):
        return None, ["[response_top_level] top-level value must be an object"], cleaned[:end]

    if cleaned[end:].strip():
        issues.append("[response_trailing_text] extra text after the JSON object")

    if "rooms" not in parsed or "doors" not in parsed:
        issues.append(
            f"[schema_top_level] response must have keys 'rooms' and 'doors'; got {sorted(parsed.keys())}"
        )

    return parsed, issues, cleaned[:end]


def _adjacent_rooms_for(room_id: str, edge_map: dict[tuple, list[tuple[str, tuple]]]) -> set[str]:
    """Return the set of room ids that share at least one full-edge wall with `room_id`."""
    neighbors: set[str] = set()
    for occurrences in edge_map.values():
        ids = {o[0] for o in occurrences}
        if room_id in ids:
            neighbors.update(other for other in ids if other != room_id)
    return neighbors


def _format_polygon(poly: list) -> str:
    if not (isinstance(poly, list) and poly):
        return "<missing>"
    parts: list[str] = []
    for p in poly:
        try:
            parts.append(f"[{p[0]},{p[1]}]")
        except (IndexError, TypeError):
            return "<malformed>"
    return "[" + ", ".join(parts) + "]"


def _emit_window_for_edge(edge, openings: list[dict], diagnostics: list[str], room_id: str) -> None:
    wl = _edge_length(edge)
    if wl < 1.2 - EPS:
        # too short to fit even a 1.2 m window; just leave the wall blank.
        return
    width = 1.5 if wl >= 5.0 - EPS else 1.2
    if wl + EPS < width:
        diagnostics.append(
            f"[compose_warn] room '{room_id}': exterior wall length {wl:.3f} too short for window width {width}"
        )
        return
    position = (wl - width) / 2.0
    openings.append(
        {
            "type": "window",
            "wall_segment": [list(edge[0]), list(edge[1])],
            "position": position,
            "width": width,
            "height": WINDOW_HEIGHT,
            "sill_height": WINDOW_SILL,
        }
    )


def compose_full_scene(decomposed: dict) -> tuple[dict, list[str]]:
    """Take a decomposed scene and produce a full scenes-style scene.

    Auto-generates door entries (on shared walls listed in `doors`, with correct
    CCW wall_segment direction, centered position, exact dimensions) and window
    entries (on every exterior wall, centered, width = 1.5 if wall ≥ 5 m else 1.2).

    Returns: (full_scene_dict, compose_diagnostics)

    The returned dict is ALWAYS a complete scene — even if the input is partly
    malformed. Validation is left to scene_constraints.validate_scene; this
    function only adds informational diagnostics for issues it detects locally.
    """
    diagnostics: list[str] = []
    rooms_in = decomposed.get("rooms", []) or []
    doors_in = decomposed.get("doors", []) or []

    rooms_by_id: dict[str, dict] = {}
    for r in rooms_in:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if isinstance(rid, str):
            rooms_by_id[rid] = r

    # edge_map[edge_key] -> [(room_id, ccw_edge), ...]
    edge_map: dict[tuple, list[tuple[str, tuple]]] = {}
    for r in rooms_in:
        if not isinstance(r, dict):
            continue
        polygon = r.get("floor_polygon", [])
        if not (isinstance(polygon, list) and len(polygon) == 4):
            continue
        try:
            edges = _polygon_edges(polygon)
        except Exception as exc:
            diagnostics.append(f"[compose_warn] room '{r.get('id')}': polygon edges failed: {exc}")
            continue
        for edge in edges:
            edge_map.setdefault(_edge_key(edge), []).append((r.get("id"), edge))

    # Resolve door pairs
    door_pairs: set[tuple[str, str]] = set()
    for d in doors_in:
        if not isinstance(d, dict):
            diagnostics.append(f"[compose_warn] non-object door entry: {d!r}")
            continue
        between = d.get("between")
        if not (
            isinstance(between, list)
            and len(between) == 2
            and all(isinstance(x, str) for x in between)
        ):
            diagnostics.append(f"[compose_warn] malformed door entry: {d!r}")
            continue
        a, b = between
        if a == b:
            diagnostics.append(f"[compose_warn] door connects room to itself: {a}")
            continue
        if a not in rooms_by_id or b not in rooms_by_id:
            missing = [x for x in between if x not in rooms_by_id]
            diagnostics.append(f"[compose_warn] door references unknown room(s): {missing}")
            continue
        door_pairs.add(tuple(sorted([a, b])))

    # Compose rooms with openings
    rendered_door_pairs: set[tuple[str, str]] = set()
    full_rooms: list[dict] = []
    for r in rooms_in:
        if not isinstance(r, dict):
            continue
        rid = r.get("id", "")
        polygon = r.get("floor_polygon", [])
        room_dict: dict[str, Any] = {
            "id": rid,
            "name": r.get("name", rid),
            "floor_polygon": polygon,
            "ceiling_height": 2.8,
            "openings": [],
        }
        if not (isinstance(polygon, list) and len(polygon) == 4):
            full_rooms.append(room_dict)
            continue
        try:
            edges = _polygon_edges(polygon)
        except Exception:
            full_rooms.append(room_dict)
            continue

        openings: list[dict] = []
        for edge in edges:
            occurrences = edge_map.get(_edge_key(edge), [])
            other_ids = [o[0] for o in occurrences if o[0] != rid]
            if other_ids:
                # Shared wall — emit a door if there's a door pair AND this room owns it.
                door_partner = None
                for oid in other_ids:
                    pair = tuple(sorted([rid, oid]))
                    if pair in door_pairs:
                        door_partner = oid
                        rendered_door_pairs.add(pair)
                        break
                if door_partner is not None and rid < door_partner:
                    wl = _edge_length(edge)
                    position = (wl - DOOR_WIDTH) / 2.0
                    openings.append(
                        {
                            "type": "door",
                            "wall_segment": [list(edge[0]), list(edge[1])],
                            "position": position,
                            "width": DOOR_WIDTH,
                            "height": DOOR_HEIGHT,
                        }
                    )
                # else: shared wall without a door (just a wall).
            else:
                _emit_window_for_edge(edge, openings, diagnostics, rid)

        room_dict["openings"] = openings
        full_rooms.append(room_dict)

    for pair in sorted(door_pairs - rendered_door_pairs):
        a_id, b_id = pair
        poly_a = rooms_by_id.get(a_id, {}).get("floor_polygon", [])
        poly_b = rooms_by_id.get(b_id, {}).get("floor_polygon", [])
        a_neighbors = sorted(_adjacent_rooms_for(a_id, edge_map))
        b_neighbors = sorted(_adjacent_rooms_for(b_id, edge_map))
        diagnostics.append(
            f"[compose_warn] door {a_id} <-> {b_id} not emitted: rooms don't share a complete wall edge. "
            f"{a_id} polygon = {_format_polygon(poly_a)}, actually shares walls with: "
            f"{a_neighbors if a_neighbors else '(none — isolated)'}. "
            f"{b_id} polygon = {_format_polygon(poly_b)}, actually shares walls with: "
            f"{b_neighbors if b_neighbors else '(none — isolated)'}. "
            f"To fix, EITHER replace this door with one between rooms that already share a wall "
            f"(see the lists above), OR reposition {a_id} and {b_id} so they touch on a full edge."
        )

    full_scene = {
        "metadata": copy.deepcopy(EXPECTED_METADATA),
        "rooms": full_rooms,
        "objects": [],
    }
    return full_scene, diagnostics
