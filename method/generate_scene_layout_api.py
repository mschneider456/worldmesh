#!/usr/bin/env python3
"""
Generate a scenes-style layout from a natural-language prompt using an LLM.

The script iteratively asks an LLM (Claude Opus 4.6 by default, or local
gpt-oss-20b via HuggingFace transformers) for a scene description, validates
it with scene_constraints.validate_scene(), and feeds violations back to the
LLM until the scene passes or the iteration limit is reached.

The default --mode decomposed asks the LLM only for room polygons + door
adjacency pairs; openings are composed deterministically in Python from those
inputs, which dramatically improves constraint-passing reliability. The legacy
--mode standard preserves the original full-scene-JSON prompting approach.

Example:
    python generate_scene_layout_api.py \
        --prompt "A 6-room U-shaped house with a central foyer" \
        --output-dir out \
        --max-iterations 5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from checkpoint_requirements import (
    anthropic_api_key_requirement,
    format_missing_checkpoints_error,
)
from decomposed_layout import (
    build_decomposed_system_prompt,
    compose_full_scene,
    decomposed_correction_message,
    decomposed_user_message,
    parse_decomposed,
)
from generate_scene_layout import write_scene
from scene_constraints import validate_scene

LAYOUT_MODEL_CLAUDE_OPUS_4_6 = "claude-opus-4-6"
LAYOUT_MODEL_GPT_OSS_20B = "gpt-oss-20b"
LAYOUT_MODELS = [LAYOUT_MODEL_CLAUDE_OPUS_4_6, LAYOUT_MODEL_GPT_OSS_20B]

MODE_DECOMPOSED = "decomposed"
MODE_STANDARD = "standard"
MODES = [MODE_DECOMPOSED, MODE_STANDARD]

# Anthropic model string that maps to the LAYOUT_MODEL_CLAUDE_OPUS_4_6 alias.
ANTHROPIC_MODEL_FOR_OPUS_4_6 = "claude-opus-4-6"
# Local HuggingFace checkpoint directory for gpt-oss-20b (relative to repo root).
GPT_OSS_20B_LOCAL_DIR = Path("comfyui") / "models" / "llm" / "gpt-oss-20b"

DEFAULT_LAYOUT_MODEL = LAYOUT_MODEL_CLAUDE_OPUS_4_6
DEFAULT_MODE = MODE_DECOMPOSED
DEFAULT_MODEL = ANTHROPIC_MODEL_FOR_OPUS_4_6  # backward-compat alias for --model
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_MAX_NEW_TOKENS = 4096
EXAMPLE_SCENE_PATH = Path("scenes/scene_3rooms_linear.json")
CONSTRAINTS_PATH = Path("SCENE_CONSTRAINTS.md")
ALLOWED_TOP_LEVEL_KEYS = {"metadata", "rooms", "objects"}
REQUIRED_TOP_LEVEL_KEYS = {"metadata", "rooms", "objects"}
ALLOWED_ROOM_KEYS = {"id", "name", "floor_polygon", "ceiling_height", "openings"}
REQUIRED_ROOM_KEYS = {"id", "name", "floor_polygon", "ceiling_height", "openings"}
ALLOWED_DOOR_KEYS = {"type", "wall_segment", "position", "width", "height"}
ALLOWED_WINDOW_KEYS = {
    "type",
    "wall_segment",
    "position",
    "width",
    "height",
    "sill_height",
}


def build_system_prompt(
    constraints_text: str,
    example_scene_json: str,
) -> list[dict[str, Any]]:
    """Construct the static Anthropic system prompt content blocks."""
    return [
        {
            "type": "text",
            "text": (
                "You generate scene layout JSON for this repository. "
                "Return exactly one JSON object and nothing else. "
                "Do not use Markdown, code fences, comments, or explanations. "
                "Do not return arbitrary code. "
                "Always output a full corrected scene object, never a patch or diff. "
                "Before responding, self-check the scene against the constraints: "
                "metadata exact, axis-aligned CCW rectangles, no room overlaps, "
                "doors only on shared interior walls, windows only on exterior walls, "
                "openings fit their walls, the door graph is connected, and objects is []."
            ),
        },
        {
            "type": "text",
            "text": (
                "Scene schema:\n"
                "- Top level keys must be exactly: metadata, rooms, objects.\n"
                "- metadata must be: "
                '{"units": "meters", "coordinate_system": "z_up", '
                '"wall_thickness": 0.15, "default_ceiling_height": 2.8}.\n'
                "- rooms must be a list of room objects.\n"
                "- Each room must contain only: id, name, floor_polygon, "
                "ceiling_height, openings.\n"
                "- floor_polygon must be a 4-vertex axis-aligned rectangle in "
                "counter-clockwise order [SW, SE, NE, NW].\n"
                "- openings is a list of door/window objects following the rules.\n"
                "- objects must always be [].\n"
                "- Use concise snake_case ids like living_room, foyer, hallway.\n"
                "- Every response must be a complete scene dict matching the "
                "scenes JSON format exactly."
            ),
        },
        {
            "type": "text",
            "text": (
                "Constraint rules that must be satisfied:\n\n"
                f"{constraints_text.strip()}"
            ),
        },
        {
            "type": "text",
            "text": (
                "Example of a valid scene JSON object in this repository:\n\n"
                f"{example_scene_json.strip()}"
            ),
            "cache_control": {"type": "ephemeral"},
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a scenes-style scene layout from a natural-language prompt "
            "using Claude plus iterative constraint validation."
        )
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Natural-language description of the desired scene layout.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the final {name}.json and {name}.png outputs.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum correction attempts before failing. Default {DEFAULT_MAX_ITERATIONS}.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output file stem. Defaults to scene_{num_rooms}rooms_llm.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Anthropic model id passthrough (used when --layout-model claude-opus-4-6). "
            f"Default {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--layout-model",
        default=DEFAULT_LAYOUT_MODEL,
        choices=LAYOUT_MODELS,
        help=(
            f"LLM to use. '{LAYOUT_MODEL_CLAUDE_OPUS_4_6}' (default) calls the Anthropic API. "
            f"'{LAYOUT_MODEL_GPT_OSS_20B}' loads the local HuggingFace model in-process; "
            "requires the worldmesh-llm conda env."
        ),
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=MODES,
        help=(
            f"Generation mode. '{MODE_DECOMPOSED}' (default) asks the LLM only for room "
            "polygons and door adjacency pairs; openings are composed deterministically. "
            f"'{MODE_STANDARD}' asks the LLM for the full scene JSON (legacy)."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key. Defaults to ANTHROPIC_API_KEY from the environment.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Maximum new tokens generated per LLM call. Default "
            f"{DEFAULT_MAX_NEW_TOKENS}."
        ),
    )
    args = parser.parse_args()
    if args.max_iterations < 1:
        parser.error("--max-iterations must be >= 1")
    return args


def load_prompt_context(repo_root: Path) -> tuple[str, str]:
    constraints_text = (repo_root / CONSTRAINTS_PATH).read_text()
    example_scene_json = (repo_root / EXAMPLE_SCENE_PATH).read_text()
    return constraints_text, example_scene_json


def get_api_key(args: argparse.Namespace) -> str | None:
    return args.api_key or os.environ.get("ANTHROPIC_API_KEY")


def import_anthropic():
    try:
        import anthropic
        from anthropic import Anthropic
    except ImportError:
        print(
            "Anthropic SDK is not installed. Install it with "
            "`pip install anthropic` or update the `scene3d` environment.",
            file=sys.stderr,
        )
        return None, None
    return anthropic, Anthropic


def prompt_user_message(prompt: str) -> str:
    return (
        "Generate a scenes-style house layout JSON for this request:\n"
        f"{prompt.strip()}\n\n"
        "Return only the complete JSON object."
    )


def correction_user_message(prompt: str, issues: list[str]) -> str:
    bullet_lines = "\n".join(f"- {issue}" for issue in issues)
    return (
        "Correct the previous scene JSON for this original request:\n"
        f"{prompt.strip()}\n\n"
        "Your previous scene has the following problems:\n"
        f"{bullet_lines}\n\n"
        "You may rewrite any room polygons and openings as needed; do not try to "
        "preserve invalid doors or windows. Recompute openings from scratch if that "
        "is simpler. Pay special attention to shared-wall adjacency and ensure that "
        "windows are only on exterior walls and doors are only on shared interior walls.\n\n"
        "Return a complete corrected scene JSON object only."
    )


def prepare_system_prompt(
    system_blocks: list[dict[str, Any]],
    use_prompt_cache: bool,
) -> list[dict[str, Any]]:
    if use_prompt_cache:
        return copy.deepcopy(system_blocks)

    stripped: list[dict[str, Any]] = []
    for block in system_blocks:
        new_block = dict(block)
        new_block.pop("cache_control", None)
        stripped.append(new_block)
    return stripped


def extract_response_text(response: Any) -> str:
    chunks: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            chunks.append(block.text)
    return "".join(chunks)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_scene_response(raw_text: str) -> tuple[dict[str, Any] | None, list[str], str]:
    cleaned = strip_code_fences(raw_text)
    candidate = cleaned.lstrip()
    if candidate and not candidate.startswith("{"):
        candidate = "{" + candidate

    issues: list[str] = []
    if not candidate:
        return None, ["[response_empty] Claude returned an empty response"], candidate

    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        return (
            None,
            [
                "[response_json] Claude did not return valid JSON: "
                f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            ],
            candidate,
        )

    trailing = candidate[end:].strip()
    if trailing:
        issues.append(
            "[response_trailing_text] Claude returned extra text after the JSON object"
        )
    if not isinstance(parsed, dict):
        issues.append("[response_top_level] Top-level JSON value must be an object")
        return None, issues, candidate

    return parsed, issues, candidate[:end]


def validate_scene_format(scene: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    scene_keys = set(scene.keys())
    missing_top = sorted(REQUIRED_TOP_LEVEL_KEYS - scene_keys)
    extra_top = sorted(scene_keys - ALLOWED_TOP_LEVEL_KEYS)
    if missing_top:
        issues.append(f"[schema_top_level] Missing top-level keys: {missing_top}")
    if extra_top:
        issues.append(f"[schema_top_level] Unexpected top-level keys: {extra_top}")

    rooms = scene.get("rooms")
    if not isinstance(rooms, list):
        issues.append("[schema_rooms] `rooms` must be a list")
        rooms = []
    elif not rooms:
        issues.append("[schema_rooms] `rooms` must contain at least one room")

    objects = scene.get("objects")
    if objects != []:
        issues.append("[schema_objects] `objects` must be exactly []")

    metadata = scene.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("[schema_metadata] `metadata` must be an object")

    for idx, room in enumerate(rooms):
        label = f"rooms[{idx}]"
        if not isinstance(room, dict):
            issues.append(f"[schema_room] {label} must be an object")
            continue

        room_keys = set(room.keys())
        missing_room = sorted(REQUIRED_ROOM_KEYS - room_keys)
        extra_room = sorted(room_keys - ALLOWED_ROOM_KEYS)
        if missing_room:
            issues.append(f"[schema_room] {label} missing keys: {missing_room}")
        if extra_room:
            issues.append(f"[schema_room] {label} has unexpected keys: {extra_room}")

        openings = room.get("openings")
        if not isinstance(openings, list):
            issues.append(f"[schema_openings] {label}.openings must be a list")
            continue

        for opening_idx, opening in enumerate(openings):
            opening_label = f"{label}.openings[{opening_idx}]"
            if not isinstance(opening, dict):
                issues.append(f"[schema_opening] {opening_label} must be an object")
                continue

            opening_type = opening.get("type")
            if opening_type == "door":
                extra = sorted(set(opening.keys()) - ALLOWED_DOOR_KEYS)
                if extra:
                    issues.append(
                        f"[schema_opening] {opening_label} has unexpected door keys: {extra}"
                    )
            elif opening_type == "window":
                extra = sorted(set(opening.keys()) - ALLOWED_WINDOW_KEYS)
                if extra:
                    issues.append(
                        f"[schema_opening] {opening_label} has unexpected window keys: {extra}"
                    )
            else:
                issues.append(
                    f"[schema_opening] {opening_label} type must be 'door' or 'window'"
                )

    return issues


def format_constraint_violations(violations: list[Any]) -> list[str]:
    return [str(violation) for violation in violations]


def try_validate_scene(scene: dict[str, Any]) -> list[str]:
    try:
        violations = validate_scene(scene)
    except Exception as exc:
        return [f"[validator_error] scene_constraints.validate_scene() failed: {exc}"]
    return format_constraint_violations(violations)


def is_cache_control_error(error: Exception) -> bool:
    text = str(error).lower()
    return "cache_control" in text or "prompt caching" in text or "cache control" in text


def get_request_id(error: Exception) -> str | None:
    for attr in ("request_id", "_request_id"):
        value = getattr(error, attr, None)
        if value:
            return str(value)

    response = getattr(error, "response", None)
    for attr in ("request_id", "_request_id"):
        value = getattr(response, attr, None)
        if value:
            return str(value)
    return None


def request_scene_json(
    client: Any,
    model: str,
    system_blocks: list[dict[str, Any]],
    messages: list[dict[str, str]],
    use_prompt_cache: bool,
    anthropic_module: Any,
    max_tokens: int = 4096,
) -> tuple[str, bool]:
    request_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": prepare_system_prompt(system_blocks, use_prompt_cache),
        "messages": list(messages),
    }

    try:
        response = client.messages.create(**request_kwargs)
        return extract_response_text(response), use_prompt_cache
    except anthropic_module.APIStatusError as exc:
        if use_prompt_cache and is_cache_control_error(exc):
            response = client.messages.create(
                **{
                    **request_kwargs,
                    "system": prepare_system_prompt(system_blocks, False),
                }
            )
            return extract_response_text(response), False
        raise


def call_anthropic(
    client: Any,
    model: str,
    system_input: Any,
    messages: list[dict[str, str]],
    use_prompt_cache: bool,
    anthropic_module: Any,
    max_tokens: int,
) -> tuple[str, bool]:
    """Provider-agnostic wrapper around request_scene_json.

    system_input may be either a list of Anthropic content blocks (standard
    mode, supports prompt caching) or a plain string (decomposed mode; wrapped
    as a single cached block).
    """
    if isinstance(system_input, str):
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": system_input,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_blocks = system_input
    return request_scene_json(
        client=client,
        model=model,
        system_blocks=system_blocks,
        messages=messages,
        use_prompt_cache=use_prompt_cache,
        anthropic_module=anthropic_module,
        max_tokens=max_tokens,
    )


# Module-level singleton: loaded once per process, reused across iterations.
_GPT_OSS_PIPE: Any = None
_GPT_OSS_TOKENIZER: Any = None

import re

# Harmony-template tokens used by gpt-oss-20b. The model emits an
# analysis/commentary preamble on its own channel before the actual response
# on the `final` channel. The default chat template renders these as
# <|channel|>NAME<|message|>BODY<|end|> (or <|return|> at the very end).
_HARMONY_NOISE_CHANNELS_RE = re.compile(
    r"<\|channel\|>(?:analysis|commentary)<\|message\|>.*?<\|end\|>",
    flags=re.DOTALL,
)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    flags=re.DOTALL,
)
_HARMONY_STRAY_TOKENS = (
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|start|>",
    "<|im_end|>",
    "<|im_start|>",
)


def strip_template_artifacts(text: str) -> str:
    """Strip Harmony template tokens from gpt-oss-20b output.

    Drops analysis/commentary channels entirely; for the `final` channel,
    extracts the body and keeps it. Sanitizes any stray template tokens that
    survive after the structural strip.
    """
    if not text:
        return text
    # 1. Drop analysis / commentary channels entirely.
    text = _HARMONY_NOISE_CHANNELS_RE.sub("", text)
    # 2. Extract the final-channel body if present.
    match = _HARMONY_FINAL_RE.search(text)
    if match:
        text = match.group(1)
    # 3. Strip any stray template tokens that survived.
    for tok in _HARMONY_STRAY_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def _system_blocks_to_str(system_input: Any) -> str:
    """Flatten a list of Anthropic content blocks into a single string."""
    if isinstance(system_input, str):
        return system_input
    parts: list[str] = []
    for block in system_input:
        text = block.get("text") if isinstance(block, dict) else None
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _load_gpt_oss(model_dir: Path) -> tuple[Any, Any]:
    """Lazily load gpt-oss-20b and cache as a singleton.

    gpt-oss-20b ships with MXFP4 quantization baked in (~13 GB on disk,
    fits in ~16 GB VRAM at runtime), so no bitsandbytes / re-quantization
    is needed — transformers reads the quantization_config from the model
    files directly.
    """
    global _GPT_OSS_PIPE, _GPT_OSS_TOKENIZER
    if _GPT_OSS_PIPE is not None and _GPT_OSS_TOKENIZER is not None:
        return _GPT_OSS_PIPE, _GPT_OSS_TOKENIZER

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "gpt-oss-20b path requires 'transformers', 'accelerate', and "
            "'torch'. Activate the 'worldmesh-llm' environment or run "
            f"./setup.sh to create it. Import error: {exc}"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # Pin everything to GPU 0; "auto" can silently spill layers to CPU which
    # then have to be paged in during forward and OOMs the activation budget.
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype="auto",
        device_map={"": 0},
    )
    model.eval()
    _GPT_OSS_PIPE = model
    _GPT_OSS_TOKENIZER = tokenizer
    return _GPT_OSS_PIPE, _GPT_OSS_TOKENIZER


def call_gpt_oss(
    model_dir: Path,
    system_input: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Run one gpt-oss-20b completion. Lazy-loads the model on first call."""
    model, tokenizer = _load_gpt_oss(model_dir)
    system_str = _system_blocks_to_str(system_input)

    chat: list[dict[str, str]] = [{"role": "system", "content": system_str}]
    chat.extend(messages)

    # apply_chat_template in transformers >=5 returns a BatchEncoding (dict-like)
    # with input_ids + attention_mask when return_tensors='pt'. Pass both to
    # generate() via ** so attention masking is correct.
    # reasoning_effort="low" caps the model's chain-of-thought preamble — the
    # decomposed-mode task is small (just polygons + door pairs) so we don't
    # need medium/high reasoning, and medium reasoning routinely overflows
    # max_new_tokens before reaching the `final` channel.
    encoded = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort="low",
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[-1]

    import torch
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, prompt_len:]
    # Decode WITHOUT skip_special_tokens so we can see the Harmony channel markers
    # and strip them in strip_template_artifacts; otherwise final-channel
    # extraction would lose its anchors.
    raw = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
    return strip_template_artifacts(raw)


def print_final_issues(iteration_count: int, issues: list[str]) -> None:
    print(
        f"Failed to produce a valid scene after {iteration_count} iteration(s).",
        file=sys.stderr,
    )
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)


def _evaluate_iteration(
    raw_response_text: str,
    mode: str,
) -> tuple[dict[str, Any] | None, list[str], str]:
    """Parse + validate an LLM response based on the chosen mode.

    Returns (final_scene_or_None, issues, normalized_response_text).
    """
    if mode == MODE_DECOMPOSED:
        decomposed, parse_issues, normalized = parse_decomposed(raw_response_text)
        issues: list[str] = list(parse_issues)
        scene: dict[str, Any] | None = None
        if (
            decomposed is not None
            and isinstance(decomposed.get("rooms"), list)
            and isinstance(decomposed.get("doors"), list)
        ):
            composed_scene, compose_diag = compose_full_scene(decomposed)
            issues.extend(compose_diag)
            issues.extend(str(v) for v in validate_scene(composed_scene))
            scene = composed_scene
        return scene, issues, normalized

    # standard mode
    parsed_scene, response_issues, normalized = parse_scene_response(raw_response_text)
    issues = list(response_issues)
    if parsed_scene is not None:
        issues.extend(validate_scene_format(parsed_scene))
        issues.extend(try_validate_scene(parsed_scene))
    return parsed_scene, issues, normalized


def _build_system_input(mode: str, repo_root: Path) -> Any:
    if mode == MODE_DECOMPOSED:
        return build_decomposed_system_prompt(repo_root)
    constraints_text, example_scene_json = load_prompt_context(repo_root)
    return build_system_prompt(constraints_text, example_scene_json)


def _initial_user_message(mode: str, prompt: str) -> str:
    if mode == MODE_DECOMPOSED:
        return decomposed_user_message(prompt)
    return prompt_user_message(prompt)


def _correction_user_message(mode: str, prompt: str, issues: list[str]) -> str:
    if mode == MODE_DECOMPOSED:
        return decomposed_correction_message(prompt, issues)
    return correction_user_message(prompt, issues)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # Resolve LLM provider from --layout-model
    layout_model = args.layout_model
    use_anthropic = layout_model == LAYOUT_MODEL_CLAUDE_OPUS_4_6

    anthropic = None
    client: Any = None
    gpt_oss_dir: Path | None = None
    if use_anthropic:
        api_key = get_api_key(args)
        if not api_key:
            print(
                format_missing_checkpoints_error([
                    anthropic_api_key_requirement(
                        None,
                        stage="Anthropic layout generation",
                    )
                ]),
                file=sys.stderr,
            )
            return 1
        anthropic, Anthropic = import_anthropic()
        if anthropic is None or Anthropic is None:
            return 1
        client = Anthropic(api_key=api_key)
    else:  # gpt-oss-20b
        gpt_oss_dir = repo_root / GPT_OSS_20B_LOCAL_DIR
        if not (gpt_oss_dir / "config.json").exists():
            print(
                f"gpt-oss-20b checkpoint not found at {gpt_oss_dir}.\n"
                "Run ./setup.sh to download it, or download manually from "
                "huggingface.co/openai/gpt-oss-20b to that directory.",
                file=sys.stderr,
            )
            return 1

    try:
        system_input = _build_system_input(args.mode, repo_root)
    except OSError as exc:
        print(f"Failed to load prompt context files: {exc}", file=sys.stderr)
        return 1

    conversation: list[dict[str, str]] = [
        {"role": "user", "content": _initial_user_message(args.mode, args.prompt)}
    ]
    use_prompt_cache = True
    last_issues: list[str] = []
    last_scene: dict[str, Any] | None = None
    last_response_text = ""

    try:
        for iteration in range(1, args.max_iterations + 1):
            if use_anthropic:
                raw_response_text, use_prompt_cache = call_anthropic(
                    client=client,
                    model=args.model,
                    system_input=system_input,
                    messages=conversation,
                    use_prompt_cache=use_prompt_cache,
                    anthropic_module=anthropic,
                    max_tokens=args.max_new_tokens,
                )
            else:
                raw_response_text = call_gpt_oss(
                    model_dir=gpt_oss_dir,
                    system_input=system_input,
                    messages=conversation,
                    max_new_tokens=args.max_new_tokens,
                )

            scene, issues, normalized_response_text = _evaluate_iteration(
                raw_response_text, args.mode
            )
            last_response_text = normalized_response_text or raw_response_text
            if scene is not None:
                last_scene = scene

            if not issues and scene is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                name = args.name or f"scene_{len(scene['rooms'])}rooms_llm"
                json_path = os.path.join(args.output_dir, f"{name}.json")
                png_path = os.path.join(args.output_dir, f"{name}.png")
                png_ok = write_scene(scene, json_path, png_path)
                print(f"Wrote {json_path}")
                if not png_ok:
                    print(
                        f"JSON was written, but PNG rendering failed for {png_path}.",
                        file=sys.stderr,
                    )
                    return 1
                print(f"Wrote {png_path}")
                print(f"Scene validated successfully in {iteration} iteration(s).")
                return 0

            last_issues = issues
            if iteration == args.max_iterations:
                break

            assistant_turn = last_response_text.strip() or "{}"
            correction = _correction_user_message(args.mode, args.prompt, issues)
            if use_anthropic:
                # Anthropic accumulates the full conversation — prompt caching
                # makes the long context cheap and the model benefits from
                # seeing past attempts.
                conversation.append({"role": "assistant", "content": assistant_turn})
                conversation.append({"role": "user", "content": correction})
            else:
                # gpt-oss-20b runs locally with eager attention, which has
                # O(seq_len^2) memory; accumulating full history OOMs ~24 GB
                # VRAM after a few iterations. The correction message already
                # embeds the original prompt and the violation list, so a
                # single (assistant, correction) pair is self-sufficient.
                conversation = [
                    {"role": "assistant", "content": assistant_turn},
                    {"role": "user", "content": correction},
                ]

    except Exception as exc:
        # Anthropic-specific exceptions only fire when use_anthropic is True
        if use_anthropic and anthropic is not None:
            if isinstance(exc, anthropic.AuthenticationError):
                print(
                    "Anthropic authentication failed. Check `--api-key` or "
                    "`ANTHROPIC_API_KEY` and confirm the workspace has access to the model.",
                    file=sys.stderr,
                )
                return 1
            if isinstance(exc, anthropic.RateLimitError):
                print(f"Anthropic rate limit error: {exc}", file=sys.stderr)
                return 1
            if isinstance(exc, anthropic.APIConnectionError):
                print(
                    f"Anthropic connection error: {exc}. Check network access and try again.",
                    file=sys.stderr,
                )
                return 1
            if isinstance(exc, anthropic.APIStatusError):
                request_id = get_request_id(exc)
                if request_id:
                    print(
                        f"Anthropic API error ({exc.status_code}, request {request_id}): {exc}",
                        file=sys.stderr,
                    )
                else:
                    print(f"Anthropic API error ({exc.status_code}): {exc}", file=sys.stderr)
                return 1
        raise

    if last_scene is None and last_response_text and not last_issues:
        last_issues = [
            "[response_json] LLM did not return a usable scene JSON object"
        ]

    print_final_issues(args.max_iterations, last_issues)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
