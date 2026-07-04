#!/usr/bin/env python3
"""
Unified Scene Generation Pipeline

Takes a scene JSON and produces photorealistic multi-view images suitable for
nerfstudio's splatfacto training.

Pipeline stages:
1. Generate structure mesh (no objects)
2. Render multi-view (structure only)
3. Initial Flux generation (per room, first camera only)
4. Extract & merge objects (per room, sequential)
5. Re-render with objects
6. Final Flux generation (per room, all cameras)
7. Export splatfacto COLMAP data for nerfstudio training

Usage:
    python run_full_pipeline.py \
        --scene-json scene_layout_large.json \
        --output-dir output/my_scene \
        --num-cameras 16 \
        --verbose
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from checkpoint_requirements import (
    comfy_api_key_requirement,
    default_flux_workflow_paths,
    depth_pro_requirement,
    find_missing_requirements,
    format_missing_checkpoints_error,
    sam3_requirement,
    sam3d_requirement,
    workflow_model_requirements,
)
from opening_visibility import get_visible_openings_description, generate_prompts_for_room
from flux_generation.camera_poses import get_poses_for_room, order_by_rotational_proximity
from flux_generation.config import MODEL_REGISTRY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Room-type-specific prompts for SAM3 object segmentation
# Comprehensive prompts include items typically on top (e.g., "bed with bedding and pillows")
# When prompts overlap (e.g., "sofa" vs "sofa with pillows and blanket"), SAM3 returns
# the more comprehensive segmentation that captures the full object with accessories
ROOM_PROMPTS = {
    "living_room": [
        # Seating - Primary
        "sofa with pillows and blanket",
        "sectional sofa",
        "loveseat",
        "armchair with cushion",
        "accent chair",
        "recliner",
        "chaise lounge",
        "ottoman",
        "pouf",
        "bean bag chair",
        "rocking chair",
        "papasan chair",
        # Tables
        "coffee table with books and decor",
        "side table with lamp",
        "end table",
        "console table",
        "sofa table",
        "nesting tables",
        "accent table",
        "tray table",
        # Storage & Display
        "bookshelf with books",
        "built-in shelving",
        "floating shelves",
        "tv stand with electronics",
        "entertainment center",
        "media console",
        "storage ottoman",
        "storage bench",
        "credenza",
        # Electronics
        "television",
        "speaker system",
        "soundbar",
        "gaming console",
        "record player",
        # Lighting
        "floor lamp",
        "arc floor lamp",
        "table lamp",
        "desk lamp",
        "reading lamp",
        "pendant light",
        "chandelier",
        "track lighting",
        "string lights",
        # Textiles & Rugs
        "area rug",
        "accent rug",
        "throw blanket",
        "decorative pillows",
        "curtains",
        "drapes",
        "blinds",
        "sheer curtains",
        # Decor & Art
        "framed picture",
        "large painting",
        "sculpture",
        "decorative objects",
        "candles",
        "candle holder",
        "vase with flowers",
        "decorative vase",
        "decorative bowl",
        "tray with items",
        # Plants
        "plant with pot",
        "large indoor plant",
        "small potted plant",
        "hanging plant",
        "plant stand with plants",
        "succulent arrangement",
        "faux plant",
        # Fireplace
        "fireplace",
        "fireplace mantel with decor",
        "fireplace screen",
        "fireplace tools",
        # Other
        "magazine rack",
        "basket",
        "decorative ladder",
        "room divider",
        "coat rack",
        "umbrella stand",
        "piano",
        "guitar on stand",
        "pet bed",
    ],
    "kitchen": [
        # Major Appliances
        "refrigerator",
        "french door refrigerator",
        "side-by-side refrigerator",
        "stove with oven",
        "gas range",
        "electric range",
        "oven",
        "microwave",
        "over-range microwave",
        "dishwasher",
        "freezer",
        "wine fridge",
        # Small Appliances - Countertop
        "coffee maker",
        "espresso machine",
        "coffee grinder",
        "toaster",
        "toaster oven",
        "air fryer",
        "blender",
        "food processor",
        "stand mixer",
        "instant pot",
        "slow cooker",
        "rice cooker",
        "kettle",
        "electric kettle",
        "juicer",
        "waffle maker",
        # Furniture & Seating
        "kitchen island with stools",
        "kitchen island",
        "bar stool",
        "counter stool",
        "kitchen table with chairs",
        "breakfast nook",
        "kitchen chair",
        "step stool",
        # Storage (freestanding only)
        "open shelving",
        "floating shelves",
        "pantry shelving",
        "spice rack",
        "pot rack",
        "wine rack",
        "cookbook shelf",
        # Countertop Items
        "cutting board",
        "knife block",
        "utensil holder",
        "paper towel holder",
        "fruit bowl with fruit",
        "bread box",
        "cookie jar",
        "canisters",
        "salt and pepper shakers",
        "oil and vinegar bottles",
        "dish rack",
        "drying mat",
        # Cookware & Bakeware (visible)
        "pots and pans",
        "cast iron skillet",
        "dutch oven",
        "baking dishes",
        # Sink Area
        "kitchen sink",
        "faucet",
        "soap dispenser",
        "sponge holder",
        # Fixtures & Lighting
        "pendant light",
        "kitchen pendant lights",
        "range hood",
        "vent hood",
        "ceiling fan",
        # Decor
        "plant with pot",
        "herb garden",
        "kitchen clock",
        "chalkboard",
        "fruit basket",
        "decorative bowls",
        # Window
        "kitchen curtains",
        "cafe curtains",
        "window valance",
        "blinds",
        # Organization
        "magnetic knife strip",
        "pegboard with tools",
        "hanging basket",
        "produce basket",
        "trash can",
        "recycling bin",
    ],
    "bedroom": [
        # Bed & Bedding
        "bed with bedding and pillows",
        "bed frame with headboard",
        "upholstered headboard",
        "wooden headboard",
        "canopy bed",
        "platform bed",
        "sleigh bed",
        "daybed",
        "trundle bed",
        "bunk bed",
        "mattress",
        "comforter",
        "duvet",
        "bed skirt",
        "decorative pillows on bed",
        "throw blanket on bed",
        # Nightstands & Side Tables
        "nightstand with lamp and items",
        "bedside table",
        "nightstand",
        "floating nightstand",
        # Seating
        "chair with cushion",
        "armchair",
        "accent chair",
        "reading chair",
        "bench at foot of bed",
        "upholstered bench",
        "storage bench",
        "vanity stool",
        "pouf",
        # Storage - Dressers
        "dresser",
        "tall dresser",
        "wide dresser",
        "chest of drawers",
        "lingerie chest",
        # Storage - Closets & Wardrobes
        "wardrobe",
        "armoire",
        "closet organizer",
        "closet shelving",
        "clothing rack",
        "garment rack",
        # Vanity & Grooming
        "makeup vanity",
        "jewelry box",
        "jewelry stand",
        "perfume bottles",
        # Desk & Work Area
        "desk with items",
        "writing desk",
        "secretary desk",
        "desk chair",
        "office chair",
        "bookshelf with books",
        "floating shelves",
        "bulletin board",
        # Lighting
        "floor lamp",
        "table lamp",
        "bedside lamp",
        "reading lamp",
        "pendant light",
        "chandelier",
        "string lights",
        "fairy lights",
        # Textiles & Rugs
        "area rug",
        "bedside rug",
        "throw blanket",
        "curtains",
        "drapes",
        "blackout curtains",
        "sheer curtains",
        "blinds",
        "roman shades",
        # Decor & Art
        "framed photos",
        "sculpture",
        "decorative objects",
        "candles",
        "vase with flowers",
        "decorative tray",
        # Plants
        "plant with pot",
        "small potted plant",
        "hanging plant",
        "succulent",
        # Electronics
        "television",
        "alarm clock",
        "charging station",
        "speaker",
        # Other
        "laundry hamper",
        "laundry basket",
        "blanket ladder",
        "coat rack",
        "pet bed",
        "baby crib",
        "bassinet",
        "rocking horse",
    ],
    "dining_room": [
        # Main Dining Furniture
        "dining table with centerpiece",
        "dining table",
        "round dining table",
        "rectangular dining table",
        "extendable dining table",
        "farmhouse table",
        "glass dining table",
        "dining chair with cushion",
        "dining chair",
        "upholstered dining chair",
        "wooden dining chair",
        "parsons chair",
        "dining bench",
        "upholstered bench",
        "banquette seating",
        # Storage & Display
        "buffet table",
        "buffet",
        "sideboard",
        "credenza",
        "hutch",
        "bar cart with bottles",
        "wine rack",
        # Tabletop Items
        "centerpiece",
        "floral centerpiece",
        "candle centerpiece",
        "fruit bowl",
        "decorative bowl",
        "vase with flowers",
        "candelabra",
        "candlesticks",
        "table runner",
        "placemats",
        "napkin holder",
        "salt and pepper shakers",
        # Lighting
        "chandelier",
        "pendant light",
        "hanging light fixture",
        "floor lamp",
        "table lamp",
        "candles",
        "lantern",
        # Textiles & Rugs
        "area rug",
        "dining room rug",
        "curtains",
        "drapes",
        "blinds",
        "window valance",
        # Decor & Art
        "large painting",
        "framed artwork",
        "sculpture",
        # Plants
        "plant with pot",
        "large indoor plant",
        "small potted plant",
        "hanging plant",
        "plant stand with plants",
        # Serving Items (visible on display)
        "serving tray",
        "pitcher",
        "decanter",
        "cake stand",
        "tiered tray",
        # Other
        "room divider",
        "folding screen",
        "coat rack",
        "umbrella stand",
        "fireplace",
        "fireplace mantel",
    ],
    "home_gym": [
        # Cardio equipment
        "treadmill",
        "exercise bike",
        "stationary bike",
        "elliptical machine",
        "rowing machine",
        # Strength equipment
        "weight bench",
        "dumbbell rack",
        "dumbbells",
        "barbell",
        "weight plates",
        "squat rack",
        "power rack",
        "cable machine",
        "pull-up bar",
        "kettlebells",
        # Flexibility & recovery
        "yoga mat",
        "exercise mat",
        "foam roller",
        "stretching area",
        "resistance bands",
        "exercise ball",
        "stability ball",
        # Storage & organization
        "equipment storage rack",
        "towel rack",
        "water bottle",
        "gym bag",
        # Mirrors & walls
        "wall mirror",
        "full-length mirror",
        # Flooring
        "rubber floor mat",
        "gym flooring",
        # Electronics
        "wall-mounted TV",
        "speaker",
        "sound system",
        # Other
        "jump rope",
        "medicine ball",
        "punching bag",
        "boxing bag",
        "fan",
        "air purifier",
        "clock",
        "timer",
    ],
    # Fallback for unknown room types
    "_default": [
        # Seating
        "sofa with pillows",
        "armchair",
        "chair with cushion",
        "bench",
        "ottoman",
        # Tables
        "table with items",
        "side table",
        "coffee table",
        "console table",
        # Storage
        "shelf with items",
        "bookshelf",
        "storage unit",
        "dresser",
        # Lighting
        "lamp",
        "floor lamp",
        "table lamp",
        "pendant light",
        # Textiles
        "rug",
        "area rug",
        "curtains",
        "blanket",
        "pillows",
        # Decor
        "plant with pot",
        "vase with flowers",
        "clock",
        "candles",
        "decorative objects",
        # Electronics
        "television",
        "speaker",
    ],
}


def get_prompts_for_room(room_id: str) -> List[str]:
    """
    Get appropriate SAM3 segmentation prompts for a room type.

    Args:
        room_id: Room identifier (e.g., "master_bedroom", "living_room")

    Returns:
        List of prompt strings for SAM3 segmentation
    """
    # Direct match
    if room_id in ROOM_PROMPTS:
        return ROOM_PROMPTS[room_id]

    # Fuzzy match for bedroom variants (master_bedroom, guest_bedroom, etc.)
    room_lower = room_id.lower()
    if "bedroom" in room_lower:
        return ROOM_PROMPTS["bedroom"]
    if "living" in room_lower:
        return ROOM_PROMPTS["living_room"]
    if "kitchen" in room_lower:
        return ROOM_PROMPTS["kitchen"]
    if "dining" in room_lower:
        return ROOM_PROMPTS["dining_room"]
    if "gym" in room_lower:
        return ROOM_PROMPTS["home_gym"]

    # Fallback
    return ROOM_PROMPTS["_default"]


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    scene_json: Path
    output_dir: Path
    num_cameras: int = 16
    num_overhead: int = 8
    image_width: int = 1360
    image_height: int = 768
    fov: int = 90
    fov_final: int = 60
    wall_offset: float = 0.4  # Distance from wall for perimeter/overhead cameras (meters)
    rooms_filter: Optional[List[str]] = None
    skip_stages: List[int] = field(default_factory=list)
    verbose: bool = False
    debug: bool = False

    # Object extraction prompts (optional override; if None, uses room-specific prompts)
    prompts: Optional[List[str]] = None

    doorway_overlap_threshold: float = 0.5
    min_area: float = 0.003  # Minimum mask area ratio for SAM3 segmentation
    scale_boost: float = 2.3  # Scale multiplier for reconstructed objects (fixed default replaces wall calibration)
    wall_calibration: bool = False  # Enable wall-based scale calibration (off by default)
    placement_mode: str = "smart"  # Object placement mode: 'smart' (full heuristics) or 'simple' (SAM3D-direct)

    # Manual mask creation mode (uses Gradio UI instead of automatic segmentation)
    manual_masks: bool = True
    manual_masks_port: int = 7860

    # Flux generation validation
    use_edge_validation: bool = False  # Edge validation off by default
    iou_threshold: float = 0.3
    max_retries: int = 2  # Max attempts per iterative image
    bootstrap_max_retries: int = 30  # More retries for bootstrap
    min_bootstrap_attempts: int = 3  # Minimum attempts before selecting best bootstrap
    use_depth_validation: bool = True
    depth_threshold: float = 0.5  # Min depth Edge IoU for validation pass (higher is better)
    depth_conda_env: str = "worldmesh-depth-pro"
    depth_canny_low: float = 0.1  # Lower threshold for depth edge detection
    depth_canny_high: float = 0.3  # Upper threshold for depth edge detection
    depth_dilation_pixels: int = 10  # Pixel tolerance for depth edge alignment
    depth_sharpen: float = 4.0  # Sharpening for estimated depth (1.0=none)
    depth_sharpen_gt: float = 3.0  # Sharpening for GT depth (1.0=none)
    depth_min_gradient_percentile: float = 25.0  # Filter edges below this GT gradient percentile (0=disabled)

    # Skip all validation (accept first generation attempt)
    no_validation: bool = False

    # API mode: enables 1376w resolution.
    # When api=True and model="api" (default), Nano Banana Pro is used (paid).
    # When api=True and model="flux2-klein-9b-distilled", a local workflow is used (free).
    api: bool = True

    # Iterative-camera model name (key into flux_generation.config.MODEL_REGISTRY).
    model: str = "api"

    # Parallel room processing in stage 6 (requires --api mode)
    parallel_rooms: bool = False

    # ComfyOrg API key for API nodes (Gemini, etc.)
    comfy_api_key: Optional[str] = None

    # Theme for bootstrap prompt (e.g. "a red and grim vampire castle")
    theme: Optional[str] = None

    # Wall texture projection (project bootstrap image onto walls for iterative consistency)
    wall_texture_alpha: float = 1.0  # Blend factor for wall textures (0=disabled, 1=full texture)
    occlusion_tolerance: float = 0.10  # Depth tolerance for object occlusion masking in texture projection (meters)
    uv_margin: float = 0.05  # UV margin beyond [0,1] to include corner faces in wall texture projection

    # Ablation study mode (replaces stages 5-7 with 3 conditioning variants)
    ablation: bool = False

    # Splatfacto COLMAP export (Stage 7)
    skip_splatfacto: bool = False
    splatfacto_subsample: str = "auto"
    splatfacto_max_points: int = 5_000_000
    splatfacto_export_depths: bool = True

    # Nerfstudio training (Stage 8)
    skip_nerfstudio_training: bool = False
    nerfstudio_conda_env: str = "worldmesh-nerfstudio"


@dataclass
class PipelineState:
    """Serializable state for resume capability."""
    current_stage: int = 1
    completed_rooms: Dict[str, List[int]] = field(default_factory=dict)
    current_mesh_path: Optional[str] = None
    started_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'current_stage': self.current_stage,
            'completed_rooms': self.completed_rooms,
            'current_mesh_path': self.current_mesh_path,
            'started_at': self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineState':
        return cls(
            current_stage=data.get('current_stage', 1),
            completed_rooms=data.get('completed_rooms', {}),
            current_mesh_path=data.get('current_mesh_path'),
            started_at=data.get('started_at'),
        )


@dataclass
class PipelinePaths:
    """Helper for managing output paths."""
    base: Path

    @property
    def structure_mesh(self) -> Path:
        return self.base / "structure_only.glb"

    @property
    def renders_structure(self) -> Path:
        return self.base / "renders_structure"

    @property
    def initial_flux(self) -> Path:
        return self.base / "initial_flux"

    @property
    def extracted(self) -> Path:
        return self.base / "extracted_objects"

    @property
    def final_mesh(self) -> Path:
        return self.base / "scene_with_all_objects.glb"

    @property
    def renders_final(self) -> Path:
        return self.base / "renders_final"

    @property
    def flux_final(self) -> Path:
        return self.base / "flux_final"

    def textured_mesh_v1(self, room_id: str) -> Path:
        return self.base / f"textured_scene_{room_id}_v1.glb"

    def textured_mesh(self, room_id: str) -> Path:
        return self.base / f"textured_scene_{room_id}.glb"

    @property
    def shared_textured_mesh_v1(self) -> Path:
        return self.base / "shared_textured_v1.glb"

    @property
    def shared_textured_mesh(self) -> Path:
        return self.base / "shared_textured_final.glb"

    @property
    def splatfacto_colmap(self) -> Path:
        return self.base / "splatfacto_colmap_all"

    @property
    def state_file(self) -> Path:
        return self.base / "pipeline_state.json"

    @property
    def log_file(self) -> Path:
        return self.base / "pipeline.log"


@dataclass
class AblationPipelinePaths(PipelinePaths):
    """Pipeline paths for ablation variants.

    Overrides structure_mesh to point to the shared parent directory,
    while all other paths (renders_final, flux_final, etc.) derive from
    self.base which is set to the ablation subdirectory.
    """
    shared_base: Path = None

    def __post_init__(self):
        if self.shared_base is None:
            self.shared_base = self.base

    @property
    def structure_mesh(self) -> Path:
        return self.shared_base / "structure_only.glb"


def _find_generated_image(generated_dir: Path, cam_idx: int) -> Optional[Path]:
    """Find generated image for camera index, regardless of ref suffix."""
    matches = sorted(generated_dir.glob(f"generated_{cam_idx:04d}*.png"))
    return matches[0] if matches else None


def cleanup_gpu():
    """Force GPU memory cleanup between stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass  # torch not available in this env


_subprocess_log_dir: Optional[Path] = None


def set_subprocess_log_dir(path: Optional[Path]) -> None:
    """Set the directory where failed-subprocess logs should be written."""
    global _subprocess_log_dir
    _subprocess_log_dir = path


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "subprocess"


def _dump_subprocess_log(description: str, cmd: List[str],
                         stdout: Optional[str], stderr: Optional[str]) -> Optional[Path]:
    """Write full captured subprocess output to a file; return the path or None."""
    if _subprocess_log_dir is None:
        return None
    try:
        _subprocess_log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = _subprocess_log_dir / f"{ts}_{_slugify(description)}.log"
        with open(log_path, "w") as f:
            f.write(f"# {description}\n")
            f.write(f"# cmd: {' '.join(str(c) for c in cmd)}\n")
            f.write("\n===== STDOUT =====\n")
            f.write(stdout or "")
            f.write("\n===== STDERR =====\n")
            f.write(stderr or "")
        return log_path
    except OSError:
        return None


def run_command(description: str, cmd: List[str], cwd: Path = None,
                env: dict = None, verbose: bool = False,
                timeout: int = None) -> bool:
    """
    Run subprocess with logging and error handling.

    Args:
        description: Human-readable description of the command
        cmd: Command and arguments as list
        cwd: Working directory (optional)
        env: Environment variables (optional)
        verbose: Show command output in real-time
        timeout: Timeout in seconds (optional)

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running: {description}")
    if verbose:
        logger.debug(f"Command: {' '.join(str(c) for c in cmd)}")
        if cwd:
            logger.debug(f"Working dir: {cwd}")

    try:
        if verbose:
            # Stream output in real-time
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                check=True,
                timeout=timeout
            )
        else:
            # Capture output
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )

        logger.info(f"Completed: {description}")
        return True

    except subprocess.TimeoutExpired as e:
        logger.error(f"Timed out after {timeout}s: {description}")
        stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr
        log_path = _dump_subprocess_log(description, cmd, stdout, stderr)
        if log_path:
            logger.error(f"Full subprocess log: {log_path}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        log_path = _dump_subprocess_log(description, cmd, e.stdout, e.stderr)
        if log_path:
            logger.error(f"Full subprocess log: {log_path}")
        if e.stderr:
            logger.error(f"Error output (tail):\n{e.stderr[-5000:]}")
        elif e.stdout:
            logger.error(f"Output (tail):\n{e.stdout[-5000:]}")
        return False
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        return False


def get_rooms_from_json(scene_json: Path) -> List[str]:
    """Extract room IDs from scene JSON."""
    with open(scene_json) as f:
        data = json.load(f)
    return [room['id'] for room in data['rooms']]


def format_command(cmd: List[str]) -> str:
    """Format a command for shell reuse."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def write_resume_file(resume_path: Path, resume_command: str):
    """Persist the fallback resume command for the current pipeline run."""
    resume_path.write_text(
        "If the pipeline does not continue automatically after mask confirmation, run:\n\n"
        f"{resume_command}\n",
        encoding="utf-8",
    )


class FullPipeline:
    """Orchestrates the complete scene generation pipeline."""

    # Base prompt for Flux generation
    BASE_PROMPT = (
        "Professional interior photography of a modern room with natural lighting, "
        "high-end finishes, and contemporary furniture. Photorealistic, architectural digest style."
    )

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = PipelinePaths(config.output_dir)
        self.state = PipelineState()
        self.rooms: List[str] = []
        self.project_dir = Path(__file__).parent.resolve()
        self.repo_root = self.project_dir.parent  # project root (parent of method/)
        self.scene_json: Optional[Dict] = None  # Cached scene JSON
        self._state_lock = threading.Lock()  # Protects self.state in parallel mode
        self._gpu_lock = threading.Lock()  # Serializes local GPU subprocesses during parallel room processing

        # Setup file logging
        self._setup_logging()

        # Direct failed-subprocess logs into <output_dir>/logs/
        set_subprocess_log_dir(self.paths.base / "logs")

    def _setup_logging(self):
        """Configure file logging."""
        self.paths.base.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(self.paths.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

        if self.config.verbose:
            logger.setLevel(logging.DEBUG)

    def _load_state(self) -> bool:
        """Load state from file if exists. Returns True if loaded."""
        if self.paths.state_file.exists():
            try:
                with open(self.paths.state_file) as f:
                    data = json.load(f)
                self.state = PipelineState.from_dict(data)
                logger.info(f"Resumed from stage {self.state.current_stage}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return False

    def _save_state(self):
        """Save current state to file."""
        with open(self.paths.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _load_scene(self):
        """Load scene data and extract room IDs."""
        # Load and cache the full scene JSON
        with open(self.config.scene_json) as f:
            self.scene_json = json.load(f)

        self.rooms = [room['id'] for room in self.scene_json.get('rooms', [])]

        # Apply room filter if specified
        if self.config.rooms_filter:
            self.rooms = [r for r in self.rooms if r in self.config.rooms_filter]

        logger.info(f"Processing {len(self.rooms)} rooms: {self.rooms}")

    def _get_room_name(self, room_id: str) -> str:
        """Get the human-readable name for a room."""
        if not self.scene_json:
            return room_id
        for room in self.scene_json.get('rooms', []):
            if room['id'] == room_id:
                return room.get('name', room_id)
        return room_id

    def _get_initial_flux_prompt(self, room_id: str, camera_name: str) -> str:
        """Generate a prompt for initial Flux generation (stage 3) with theme-aware style."""
        images_txt_path = self.paths.renders_structure / "images.txt"

        opening_desc = get_visible_openings_description(
            self.scene_json,
            room_id,
            camera_name,
            images_txt_path,
            include_destinations=False,
        )

        room_name = self._get_room_name(room_id)
        if self.config.theme:
            base = f"{room_name} with a lot of furniture. {self.config.theme.title()}. Photorealistic."
        else:
            base = f"{room_name} with a lot of furniture. Photorealistic."

        if opening_desc:
            return f"{base} {opening_desc}."
        return base

    def _should_skip_stage(self, stage: int) -> bool:
        """Check if a stage should be skipped."""
        if stage in self.config.skip_stages:
            return True
        if self.state.current_stage > stage:
            return True
        return False

    def _build_resume_command(self) -> str:
        """Build the exact command used to resume this pipeline invocation."""
        cmd = [
            "python",
            "run_full_pipeline.py",
            "--scene-json",
            str(self.config.scene_json),
            "--output-dir",
            str(self.paths.base),
            "--num-cameras",
            str(self.config.num_cameras),
            "--num-overhead",
            str(self.config.num_overhead),
            "--width",
            str(self.config.image_width),
            "--height",
            str(self.config.image_height),
            "--fov",
            str(self.config.fov),
            "--fov-final",
            str(self.config.fov_final),
            "--wall-offset",
            str(self.config.wall_offset),
            "--min-bootstrap-attempts",
            str(self.config.min_bootstrap_attempts),
            "--depth-threshold",
            str(self.config.depth_threshold),
            "--placement-mode",
            self.config.placement_mode,
            "--manual-masks-port",
            str(self.config.manual_masks_port),
        ]

        if self.config.theme:
            cmd.extend(["--theme", self.config.theme])

        if self.config.api:
            cmd.append("--api")
            cmd.extend(["--model", self.config.model])
        else:
            cmd.append("--no-api")

        if self.config.comfy_api_key:
            cmd.extend(["--comfy-api-key", self.config.comfy_api_key])

        if self.config.verbose:
            cmd.append("--verbose")

        if self.config.debug:
            cmd.append("--debug")

        return format_command(cmd)

    def _write_resume_file(self) -> Path:
        """Write the resume command to the output directory."""
        resume_path = self.paths.base / "RESUME.txt"
        write_resume_file(resume_path, self._build_resume_command())
        return resume_path

    def _preflight_required_checkpoints(self) -> bool:
        """Fail fast when required checkpoints for active stages are missing."""
        active_stages = {stage for stage in range(1, 9) if not self._should_skip_stage(stage)}
        requirements = []

        if 4 in active_stages:
            requirements.append(sam3d_requirement(self.repo_root, stage="Stage 4: Extract Objects"))
            if not self.config.manual_masks:
                requirements.append(sam3_requirement(self.repo_root, stage="Stage 4: Automatic SAM3 segmentation"))

        needs_flux = 3 in active_stages or 6 in active_stages
        if needs_flux:
            workflow_paths = default_flux_workflow_paths(
                self.repo_root,
                api=self.config.api,
                include_initial=(3 in active_stages) or (6 in active_stages),
                include_final=6 in active_stages,
                model=self.config.model,
            )
            requirements.extend(
                workflow_model_requirements(
                    self.repo_root,
                    workflow_paths,
                    stage="Flux generation via ComfyUI",
                )
            )
            if self.config.use_depth_validation:
                requirements.append(depth_pro_requirement(self.repo_root, stage="Flux depth validation"))
            if 6 in active_stages and self.config.api and self.config.model == "api":
                requirements.append(
                    comfy_api_key_requirement(
                        self.config.comfy_api_key,
                        stage="Stage 6: Final Flux generation via ComfyOrg API",
                    )
                )

        missing = find_missing_requirements(requirements)
        if missing:
            logger.error("\n" + format_missing_checkpoints_error(missing))
            return False
        return True

    def run(self) -> bool:
        """Run the complete pipeline."""
        logger.info("=" * 60)
        logger.info("UNIFIED SCENE GENERATION PIPELINE")
        logger.info("=" * 60)

        # Initialize
        self._load_state()
        if not self.state.started_at:
            self.state.started_at = datetime.now().isoformat()

        self._load_scene()
        if not self._preflight_required_checkpoints():
            self._save_state()
            return False

        # Run stages
        stages = [
            (1, "Generate Structure Mesh", self._stage1_generate_structure),
            (2, "Render Structure", self._stage2_render_structure),
            (3, "Initial Flux Generation", self._stage3_initial_flux),
        ]

        stages.append((4, "Extract Objects", self._stage4_extract_objects))

        if not self.config.ablation:
            stages.extend([
                (5, "Render Final", self._stage5_render_final),
                (6, "Final Flux Generation", self._stage6_final_flux),
                (7, "Export Splatfacto COLMAP", self._stage7_splatfacto_export),
                (8, "Nerfstudio Training", self._stage8_nerfstudio_training),
            ])

        for stage_num, stage_name, stage_func in stages:
            if self._should_skip_stage(stage_num):
                logger.info(f"[SKIP] Stage {stage_num}: {stage_name}")
                continue

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"STAGE {stage_num}: {stage_name}")
            logger.info("=" * 60)

            success = stage_func()

            if not success:
                logger.error(f"Pipeline failed at Stage {stage_num}: {stage_name}")
                self._save_state()
                return False

            # Update state and save
            self.state.current_stage = stage_num + 1
            self._save_state()

            # Cleanup GPU memory between stages
            cleanup_gpu()

        # Run ablation variants if enabled (replaces stages 5-7)
        if self.config.ablation:
            logger.info("")
            logger.info("=" * 60)
            logger.info("RUNNING ABLATION STUDY")
            logger.info("=" * 60)
            if not self._run_ablations():
                logger.error("Ablation study had failures")
                self._save_state()
                return False

        # Pipeline complete
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        self._print_summary()

        return True

    def _stage1_generate_structure(self) -> bool:
        """Stage 1: Generate structure mesh (no objects)."""
        cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(self.project_dir / "generate_scene.py"),
            "--input", str(self.config.scene_json),
            "--output", str(self.paths.structure_mesh),
            "--no-objects"
        ]

        success = run_command(
            "Generate structure mesh",
            cmd,
            cwd=self.project_dir,
            verbose=self.config.verbose
        )

        if success and self.paths.structure_mesh.exists():
            self.state.current_mesh_path = str(self.paths.structure_mesh)
            logger.info(f"Created: {self.paths.structure_mesh}")
            return True
        return False

    def _stage2_render_structure(self) -> bool:
        """Stage 2: Render multi-view (structure only)."""
        cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(self.project_dir / "render_multiview.py"),
            "--scene-json", str(self.config.scene_json),
            "--scene-mesh", str(self.paths.structure_mesh),
            "--output-dir", str(self.paths.renders_structure),
            "--num-cameras", str(self.config.num_cameras),
            "--num-overhead", str(self.config.num_overhead),
            "--width", str(self.config.image_width),
            "--height", str(self.config.image_height),
            "--flat-lighting",
            "--structure-mesh", str(self.paths.structure_mesh),
            "--fov", str(self.config.fov),
            "--wall-offset", str(self.config.wall_offset),
        ]

        success = run_command(
            "Render structure multi-view",
            cmd,
            cwd=self.project_dir,
            verbose=self.config.verbose
        )

        if success:
            # Verify renders exist
            images_dir = self.paths.renders_structure / "images"
            if images_dir.exists():
                room_dirs = list(images_dir.iterdir())
                logger.info(f"Rendered {len(room_dirs)} room(s)")
                return True
        return False

    def _stage3_initial_flux(self) -> bool:
        """Stage 3: Initial Flux generation (first camera per room)."""
        all_success = True

        # Create consolidated prompts file for all rooms
        self.paths.initial_flux.mkdir(parents=True, exist_ok=True)
        all_prompts_file = self.paths.initial_flux / "all_prompts.txt"
        with open(all_prompts_file, 'w') as f_all:
            f_all.write("=== Stage 3: Initial Flux Prompts ===\n")
            f_all.write(f"Generated: {datetime.now().isoformat()}\n")
            f_all.write(f"Rooms: {len(self.rooms)}\n\n")

        for room_id in self.rooms:
            # Check if already completed
            if room_id in self.state.completed_rooms.get('stage3', []):
                logger.info(f"[SKIP] {room_id}: already completed")
                continue

            # Find the first depth map image for this room
            room_depth_dir = self.paths.renders_structure / "depth" / room_id
            depth_images = sorted(room_depth_dir.glob("*_depth.png"))

            if not depth_images:
                logger.error(f"No depth images found for {room_id}")
                all_success = False
                continue

            input_image = depth_images[0]  # First camera (0000)
            output_dir = self.paths.initial_flux / room_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_image = output_dir / "generated.png"

            # Copy input depth image to output folder for comparison
            shutil.copy(input_image, output_dir / "input_depth.png")

            # Generate random seed for variation between rooms
            seed = random.randint(0, 2**32 - 1)

            # Generate initial flux prompt for first camera (with detailed style description)
            camera_name = f"{room_id}_0000"
            prompt = self._get_initial_flux_prompt(room_id, camera_name)
            logger.info(f"Prompt for {room_id}: {prompt[:80]}...")

            # Write per-room prompt file
            prompt_file = output_dir / "prompt.txt"
            with open(prompt_file, 'w') as f:
                f.write(f"Stage: 3 (Initial Flux)\n")
                f.write(f"Room: {room_id}\n")
                f.write(f"Camera: {camera_name}\n")
                f.write(f"Seed: {seed}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"\n--- Prompt ---\n")
                f.write(f"{prompt}\n")

            # Append to consolidated prompts file
            with open(all_prompts_file, 'a') as f_all:
                f_all.write(f"--- {room_id} ({camera_name}) ---\n")
                f_all.write(f"Seed: {seed}\n")
                f_all.write(f"{prompt}\n\n")

            cmd = [
                "conda", "run", "-n", "worldmesh-comfy", "--no-capture-output",
                "python", str(self.project_dir / "run_initial_flux.py"),
                "--input", str(input_image),
                "--output", str(output_image),
                "--seed", str(seed),
                "--prompt", prompt,
            ]

            success = run_command(
                f"Initial Flux for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            if success and output_image.exists():
                # Track completion
                if 'stage3' not in self.state.completed_rooms:
                    self.state.completed_rooms['stage3'] = []
                self.state.completed_rooms['stage3'].append(room_id)
                self._save_state()
                logger.info(f"Generated: {output_image}")
            else:
                logger.error(f"Failed to generate Flux image for {room_id}")
                all_success = False

            # GPU cleanup between rooms
            cleanup_gpu()

        return all_success

    def _stage4_extract_objects(self) -> bool:
        """Stage 4: Extract & merge objects (per room, sequential)."""
        # Use manual mask creation if configured
        if self.config.manual_masks:
            return self._stage4_manual_masks()

        # Start with structure mesh
        current_mesh = self.paths.structure_mesh

        for room_id in self.rooms:
            # Check if already completed
            if room_id in self.state.completed_rooms.get('stage4', []):
                # Use the output mesh from this room for the next
                room_output_dir = self.paths.extracted / room_id
                potential_mesh = room_output_dir / "scene_with_objects.glb"
                if potential_mesh.exists():
                    current_mesh = potential_mesh
                    logger.info(f"[SKIP] {room_id}: already completed, using mesh")
                    continue

            input_image = self.paths.initial_flux / room_id / "generated.png"
            if not input_image.exists():
                logger.error(f"Initial flux image not found for {room_id}: {input_image}")
                return False

            room_output_dir = self.paths.extracted / room_id
            room_output_dir.mkdir(parents=True, exist_ok=True)

            # Build camera name from room ID (first camera)
            camera_name = f"{room_id}_0000"

            # Get prompts: override or default
            if self.config.prompts:
                prompts = self.config.prompts
                prompt_source = "override"
                logger.info(f"Using {len(prompts)} override prompts for {room_id}")
            else:
                prompts = get_prompts_for_room(room_id)
                prompt_source = "default"
                logger.info(f"Using {len(prompts)} default prompts for {room_id}")

            # Always add "doorway" for adjacent room exclusion
            if "doorway" not in [p.lower() for p in prompts]:
                prompts = list(prompts) + ["doorway"]
                logger.info("Added 'doorway' prompt for adjacent room exclusion")

            # Save final prompts used to file
            prompts_file = room_output_dir / "prompts_used.json"
            with open(prompts_file, 'w') as f:
                json.dump({
                    'room_id': room_id,
                    'source': prompt_source,
                    'prompts': prompts,
                    'doorway_overlap_threshold': self.config.doorway_overlap_threshold,
                }, f, indent=2)
            logger.info(f"Saved {len(prompts)} prompts to {prompts_file}")

            cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "extract_objects" / "run_pipeline.py"),
                "--input-image", str(input_image),
                "--room-mesh", str(current_mesh),
                "--camera-name", camera_name,
                "--images-txt", str(self.paths.renders_structure / "images.txt"),
                "--cameras-txt", str(self.paths.renders_structure / "cameras.txt"),
                "--output-dir", str(room_output_dir),
                "--min-area", str(self.config.min_area),
                "--doorway-overlap-threshold", str(self.config.doorway_overlap_threshold),
                "--scale-boost", str(self.config.scale_boost),
            ]

            # Only pass wall calibration args when enabled
            if self.config.wall_calibration:
                cmd.extend([
                    "--wall-calibration",
                    "--segmentation-map", str(self.paths.renders_structure / "segmentation" / room_id / f"{camera_name}_seg.png"),
                    "--segmentation-metadata", str(self.paths.renders_structure / "segmentation_metadata.json"),
                    "--scene-json", str(self.config.scene_json),
                    "--room-id", room_id,
                ])
            else:
                # Still pass scene-json and room-id for step3 wall-aware conflict resolution
                cmd.extend([
                    "--scene-json", str(self.config.scene_json),
                    "--room-id", room_id,
                ])

            cmd.extend(["--prompts"] + prompts)
            cmd.extend(["--placement-mode", self.config.placement_mode])

            if self.config.debug:
                cmd.append("--debug")

            success = run_command(
                f"Extract objects for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            # Find the output mesh (may be in timestamped subdirectory)
            output_mesh = self._find_latest_mesh(room_output_dir)

            if success and output_mesh and output_mesh.exists():
                # Update current mesh for next room
                current_mesh = output_mesh
                self.state.current_mesh_path = str(current_mesh)

                # Track completion
                if 'stage4' not in self.state.completed_rooms:
                    self.state.completed_rooms['stage4'] = []
                self.state.completed_rooms['stage4'].append(room_id)
                self._save_state()

                logger.info(f"Updated mesh: {current_mesh}")
            else:
                if not success:
                    logger.error(f"Object extraction failed for {room_id}, stopping pipeline")
                    return False
                logger.warning(f"No objects extracted for {room_id}, continuing with current mesh")

            # GPU cleanup between rooms
            cleanup_gpu()

        # Copy final mesh to top-level
        if current_mesh != self.paths.structure_mesh:
            shutil.copy(current_mesh, self.paths.final_mesh)
            logger.info(f"Final mesh: {self.paths.final_mesh}")
        else:
            # No objects extracted, use structure mesh
            shutil.copy(self.paths.structure_mesh, self.paths.final_mesh)
            logger.info("No objects extracted, using structure-only mesh")

        # Fill any unfilled window openings by cloning existing windows
        self._fill_missing_windows()

        # Create consolidated prompts summary for all rooms
        all_prompts_summary = {
            'stage': 4,
            'description': 'Object extraction prompts used per room',
            'doorway_overlap_threshold': self.config.doorway_overlap_threshold,
            'rooms': {}
        }
        for room_id in self.rooms:
            room_prompts_file = self.paths.extracted / room_id / "prompts_used.json"
            if room_prompts_file.exists():
                with open(room_prompts_file) as f:
                    all_prompts_summary['rooms'][room_id] = json.load(f)

        summary_file = self.paths.extracted / "all_prompts_used.json"
        with open(summary_file, 'w') as f:
            json.dump(all_prompts_summary, f, indent=2)
        logger.info(f"Saved consolidated prompts summary to {summary_file}")

        return True

    def _stage4_manual_masks(self) -> bool:
        """Stage 4 (Manual Mode): Create masks via Gradio UI, then run reconstruction."""
        logger.info("Manual mask creation mode enabled")

        # Check if masks already exist for all rooms
        rooms_with_masks = []
        rooms_without_masks = []
        for room_id in self.rooms:
            room_masks_dir = self.paths.extracted / room_id / "masks"
            metadata_path = room_masks_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if metadata.get("num_masks", 0) > 0:
                    rooms_with_masks.append(room_id)
                    continue
            rooms_without_masks.append(room_id)

        if not rooms_without_masks:
            logger.info(f"Masks already exist for all {len(rooms_with_masks)} rooms, skipping Gradio UI")
        else:
            missing = find_missing_requirements([
                sam3_requirement(self.repo_root, stage="Manual mask creation with SAM3")
            ])
            if missing:
                logger.error("\n" + format_missing_checkpoints_error(missing))
                return False

            if rooms_with_masks:
                logger.info(f"Masks already exist for: {', '.join(rooms_with_masks)}")
                logger.info(f"Need masks for: {', '.join(rooms_without_masks)}")

            logger.info("Launching Gradio UI for interactive mask creation...")
            resume_path = self._write_resume_file()
            resume_command = self._build_resume_command()

            # Step 1: Launch manual mask creation UI
            # This will block until user completes all rooms
            # Use --live-stream instead of --no-capture-output to allow Gradio's
            # websocket/SSE connections to work properly through conda run
            cmd = [
                "conda", "run", "-n", "worldmesh-sam3", "--live-stream",
                "python", str(self.project_dir / "manual_mask_gradio.py"),
                "--input-dir", str(self.paths.initial_flux),
                "--output-dir", str(self.paths.extracted),
                "--port", str(self.config.manual_masks_port),
                "--resume-command", resume_command,
                "--resume-file", str(resume_path),
                "--rooms",
            ] + rooms_without_masks  # Only show rooms that need masks

            logger.info(f"Open http://localhost:{self.config.manual_masks_port} to create masks")
            logger.info("After the UI shows 'All done!', confirm the masks to close Gradio and continue automatically")
            logger.info(f"If the pipeline does not continue automatically, run: {resume_command}")
            logger.info(f"Saved fallback command to: {resume_path}")

            # Run Gradio UI - this will return False when user presses Ctrl+C,
            # which is the expected way to exit. We check for saved masks below.
            run_command(
                "Manual mask creation (Gradio UI)",
                cmd,
                cwd=self.project_dir,
                verbose=True,  # Always show Gradio output
            )

            # Re-check if any masks were saved (Ctrl+C is expected exit method)
            masks_found = False
            for room_id in self.rooms:
                room_masks_dir = self.paths.extracted / room_id / "masks"
                if room_masks_dir.exists() and (room_masks_dir / "metadata.json").exists():
                    masks_found = True
                    break

            if not masks_found:
                logger.error("No masks were saved. Please run again and save masks before closing.")
                return False

        # Step 2: Run reconstruction and positioning for each room with masks
        logger.info("Masks found. Running reconstruction and positioning...")

        current_mesh = self.paths.structure_mesh

        for room_id in self.rooms:
            # Check if already completed (resume support)
            if room_id in self.state.completed_rooms.get('stage4', []):
                room_output_dir = self.paths.extracted / room_id
                potential_mesh = self._find_latest_mesh(room_output_dir)
                if potential_mesh and potential_mesh.exists():
                    current_mesh = potential_mesh
                    logger.info(f"[SKIP] {room_id}: already completed, using mesh")
                    continue

            room_masks_dir = self.paths.extracted / room_id / "masks"

            if not room_masks_dir.exists():
                logger.warning(f"No masks found for {room_id}, skipping reconstruction")
                continue

            metadata_path = room_masks_dir / "metadata.json"
            if not metadata_path.exists():
                logger.warning(f"No metadata.json for {room_id}, skipping reconstruction")
                continue

            with open(metadata_path) as f:
                metadata = json.load(f)

            if metadata.get("num_masks", 0) == 0:
                logger.info(f"No masks for {room_id}, skipping")
                continue

            logger.info(f"Processing {room_id} with {metadata['num_masks']} masks")

            input_image = self.paths.initial_flux / room_id / "generated.png"
            if not input_image.exists():
                logger.error(f"Input image not found for {room_id}: {input_image}")
                continue

            room_output_dir = self.paths.extracted / room_id
            camera_name = f"{room_id}_0000"

            # Run step2 (reconstruction) and step3 (positioning) via run_pipeline.py
            # with --skip-step1 since masks are already created
            cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "extract_objects" / "run_pipeline.py"),
                "--input-image", str(input_image),
                "--room-mesh", str(current_mesh),
                "--camera-name", camera_name,
                "--images-txt", str(self.paths.renders_structure / "images.txt"),
                "--cameras-txt", str(self.paths.renders_structure / "cameras.txt"),
                "--output-dir", str(room_output_dir),
                "--doorway-overlap-threshold", str(self.config.doorway_overlap_threshold),
                "--scale-boost", str(self.config.scale_boost),
                "--skip-step1",  # Skip segmentation, use manual masks
            ]

            # Only pass wall calibration args when enabled
            if self.config.wall_calibration:
                cmd.extend([
                    "--wall-calibration",
                    "--segmentation-map", str(self.paths.renders_structure / "segmentation" / room_id / f"{camera_name}_seg.png"),
                    "--segmentation-metadata", str(self.paths.renders_structure / "segmentation_metadata.json"),
                    "--scene-json", str(self.config.scene_json),
                    "--room-id", room_id,
                ])
            else:
                # Still pass scene-json and room-id for step3 wall-aware conflict resolution
                cmd.extend([
                    "--scene-json", str(self.config.scene_json),
                    "--room-id", room_id,
                ])

            cmd.extend(["--placement-mode", self.config.placement_mode])

            if self.config.debug:
                cmd.append("--debug")

            success = run_command(
                f"Reconstruct objects for {room_id} (this step can take a while)",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            # Find output mesh
            output_mesh = self._find_latest_mesh(room_output_dir)

            if success and output_mesh and output_mesh.exists():
                current_mesh = output_mesh
                self.state.current_mesh_path = str(current_mesh)

                if 'stage4' not in self.state.completed_rooms:
                    self.state.completed_rooms['stage4'] = []
                self.state.completed_rooms['stage4'].append(room_id)
                self._save_state()

                logger.info(f"Updated mesh: {current_mesh}")
            else:
                if not success:
                    logger.error(f"Object reconstruction failed for {room_id}, stopping pipeline")
                    return False
                logger.warning(f"No objects reconstructed for {room_id}")

            cleanup_gpu()

        # Copy final mesh to top-level
        if current_mesh != self.paths.structure_mesh:
            shutil.copy(current_mesh, self.paths.final_mesh)
            logger.info(f"Final mesh: {self.paths.final_mesh}")
        else:
            shutil.copy(self.paths.structure_mesh, self.paths.final_mesh)
            logger.info("No objects extracted, using structure-only mesh")

        # Fill any unfilled window openings by cloning existing windows
        self._fill_missing_windows()

        return True

    def _fill_missing_windows(self):
        """Fill unfilled window openings by cloning existing windows.

        Non-fatal: logs warning on failure, pipeline continues.
        """
        logger.info("Filling unfilled window openings...")
        cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(self.project_dir / "fill_missing_windows.py"),
            "--mesh", str(self.paths.final_mesh),
            "--scene-json", str(self.config.scene_json),
            "--verbose",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.project_dir, timeout=120,
            )
            if result.returncode != 0:
                logger.warning(f"fill_missing_windows failed: {result.stderr}")
            else:
                if result.stdout:
                    logger.info(result.stdout.strip())
                if result.stderr:
                    logger.info(result.stderr.strip())
        except Exception as e:
            logger.warning(f"fill_missing_windows error: {e}")

    def _find_latest_mesh(self, output_dir: Path) -> Optional[Path]:
        """Find the latest scene_with_objects.glb in output directory."""
        # Check direct path first
        direct = output_dir / "scene_with_objects.glb"
        if direct.exists():
            return direct

        # Check timestamped subdirectories
        subdirs = sorted(output_dir.iterdir(), reverse=True)
        for subdir in subdirs:
            if subdir.is_dir():
                mesh = subdir / "scene_with_objects.glb"
                if mesh.exists():
                    return mesh

        return None

    def _stage5_render_final(self) -> bool:
        """Stage 5: Re-render with objects."""
        # Check which mesh to use
        mesh_to_render = self.paths.final_mesh
        if not mesh_to_render.exists():
            mesh_to_render = self.paths.structure_mesh
            logger.debug("Final mesh not found, using structure mesh")

        cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(self.project_dir / "render_multiview.py"),
            "--scene-json", str(self.config.scene_json),
            "--scene-mesh", str(mesh_to_render),
            "--output-dir", str(self.paths.renders_final),
            "--num-cameras", str(self.config.num_cameras),
            "--num-overhead", str(self.config.num_overhead),
            "--width", str(self.config.image_width),
            "--height", str(self.config.image_height),
            "--flat-lighting",
            "--structure-mesh", str(self.paths.structure_mesh),
            "--with-objects",
            "--fov", str(self.config.fov_final),
            "--wall-offset", str(self.config.wall_offset),
        ]

        success = run_command(
            "Render final multi-view",
            cmd,
            cwd=self.project_dir,
            verbose=self.config.verbose
        )

        if success:
            images_dir = self.paths.renders_final / "images"
            if images_dir.exists():
                logger.info(f"Final renders saved to {images_dir}")

                # Generate depth+objects conditioning images for Flux
                logger.info("Generating depth+objects conditioning images...")
                depth_cmd = [
                    "conda", "run", "-n", "worldmesh", "--no-capture-output",
                    "python", str(self.project_dir / "generate_depth_object_conditioning.py"),
                    "--renders-dir", str(self.paths.renders_final),
                    "--structure-mesh", str(self.paths.structure_mesh),
                    "--scene-json", str(self.config.scene_json),
                ]

                if self.config.verbose:
                    depth_cmd.append("--verbose")

                depth_success = run_command(
                    "Generate depth+objects conditioning",
                    depth_cmd,
                    cwd=self.project_dir,
                    verbose=self.config.verbose
                )

                if not depth_success:
                    logger.warning("Depth+objects conditioning generation failed, will fall back to edge-based conditioning")

                return True
        return False

    def _build_flux_cmd_base(self, input_dir, output_dir, prompts_file, base_prompt, room_id, comfyui_port=None):
        """Build common flux CLI command arguments."""
        cmd = [
            "conda", "run", "-n", "worldmesh-comfy", "--no-capture-output",
            "python", "-m", "flux_generation.cli",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--prompts-file", str(prompts_file),
            "--prompt", base_prompt,
            "--scene-json", str(self.config.scene_json),
            "--room-id", room_id,
            "--iou-threshold", str(self.config.iou_threshold),
            "--max-retries", str(self.config.max_retries),
            "--bootstrap-max-retries", str(self.config.bootstrap_max_retries),
            "--min-bootstrap-attempts", str(self.config.min_bootstrap_attempts),
            "--depth-threshold", str(self.config.depth_threshold),
            "--depth-conda-env", self.config.depth_conda_env,
            "--depth-canny-low", str(self.config.depth_canny_low),
            "--depth-canny-high", str(self.config.depth_canny_high),
            "--depth-dilation-pixels", str(self.config.depth_dilation_pixels),
            "--depth-sharpen", str(self.config.depth_sharpen),
            "--depth-sharpen-gt", str(self.config.depth_sharpen_gt),
            "--depth-min-gradient-percentile", str(self.config.depth_min_gradient_percentile),
        ]

        if self.config.theme:
            cmd.extend(["--theme", self.config.theme])
        if self.config.api:
            cmd.append("--api")
            cmd.extend(["--model", self.config.model])
        if self.config.comfy_api_key:
            cmd.extend(["--comfy-api-key", self.config.comfy_api_key])
        if self.config.use_edge_validation:
            cmd.append("--use-edge-validation")
        if self.config.no_validation:
            cmd.append("--no-validation")
        if not self.config.use_depth_validation:
            cmd.append("--no-depth-validation")
        if self.config.verbose:
            cmd.append("--verbose")

        if comfyui_port is not None:
            cmd.extend(["--comfyui-port", str(comfyui_port)])
            cmd.append("--no-auto-start")

        return cmd

    def _write_prompts_for_room(self, room_id, output_dir, opening_descriptions):
        """Write human-readable prompts.txt for a room."""
        base_prompt = "Photorealistic."
        prompts_txt = output_dir / "prompts.txt"

        with open(self.config.scene_json) as sf:
            scene_json = json.load(sf)

        with open(prompts_txt, 'w') as f:
            f.write(f"Stage: 6 (Final Flux)\n")
            f.write(f"Room: {room_id}\n")
            f.write(f"Base prompt: {base_prompt}\n")
            f.write(f"Cameras: {len(opening_descriptions)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("Flow:\n")
            f.write("  6a. Bootstrap camera 0: Flux2-Klein 9B\n")
            f.write("  6b. Project wall texture from camera 0 -> textured_v1.glb\n")
            f.write("  6c. Regenerate conditioning with camera 0 textures\n")
            f.write("  6d. Bootstrap camera 1 (opposite wall): iterative with camera 0 as ref\n")
            f.write("  6e. Project wall texture from camera 1 onto textured_v1 -> textured_final.glb\n")
            f.write("  6f. Regenerate conditioning with both cameras' textures\n")
            f.write("  6g. Iterative cameras 2-N: style ref = rotationally-closest of {0, 1}\n\n")

            for i, (cam_name, opening_desc) in enumerate(sorted(opening_descriptions.items())):
                f.write(f"--- {cam_name} ---\n")

                if i == 0:
                    room_name = self._get_room_name(room_id)
                    if self.config.theme:
                        flux_prompt = f"{room_name}. {self.config.theme.title()}. Photorealistic."
                    else:
                        flux_prompt = f"{room_name}. Photorealistic."
                    f.write(f"[BOOTSTRAP 0 - Flux2-Klein 9B] {flux_prompt}\n\n")
                elif i == 1:
                    iterative_prompt = "Generate a photorealistic version from camera pose of reference_image 1 which shows the scene in reference_image 2."
                    if self.config.theme:
                        iterative_prompt = f"{iterative_prompt} {self.config.theme}"
                    f.write(f"[BOOTSTRAP 1 - iterative, style ref = camera 0] {iterative_prompt}\n\n")
                else:
                    iterative_prompt = "Generate a photorealistic version from camera pose of reference_image 1 which shows the scene in reference_image 2."
                    if self.config.theme:
                        iterative_prompt = f"{iterative_prompt} {self.config.theme}"
                    f.write(f"[Flux2-Klein iterative, style ref = closest of {{0, 1}}] {iterative_prompt}\n\n")

        return scene_json

    def _start_comfyui_instances(self, room_contexts, base_port=8189):
        """Start one ComfyUI instance per room on sequential ports for parallel execution."""
        import urllib.request

        processes = []
        log_files = []
        for i, ctx in enumerate(room_contexts):
            port = base_port + i
            ctx['comfyui_port'] = port
            logger.info(f"Starting ComfyUI instance for {ctx['room_id']} on port {port}")
            log_path = self.paths.base / f"comfyui_{ctx['room_id']}.log"
            log_file = open(log_path, 'w')
            log_files.append(log_file)
            logger.info(f"ComfyUI logs for {ctx['room_id']}: {log_path}")
            proc = subprocess.Popen(
                ["conda", "run", "-n", "worldmesh-comfy", "--no-capture-output",
                 "python", "main.py", "--listen", "127.0.0.1", "--port", str(port)],
                cwd=str(self.repo_root / "comfyui"),
                stdout=log_file, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            processes.append(proc)

        # Wait for all instances to be ready
        for ctx in room_contexts:
            port = ctx['comfyui_port']
            for attempt in range(120):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
                    logger.info(f"ComfyUI on port {port} is ready ({ctx['room_id']})")
                    break
                except Exception:
                    time.sleep(1)
            else:
                logger.warning(f"ComfyUI on port {port} failed to start within 120s ({ctx['room_id']})")

        self._comfyui_log_files = log_files
        return processes

    def _stop_comfyui_instances(self, processes):
        """Terminate all pre-started ComfyUI instances and their child processes."""
        for proc in processes:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                proc.wait(timeout=5)
            except Exception:
                pass
        # Close log file handles
        for log_file in getattr(self, '_comfyui_log_files', []):
            try:
                log_file.close()
            except Exception:
                pass

    def _stage6_shared_texture_pass(self, room_contexts, cam_idx, input_mesh, output_mesh):
        """Project one camera's texture from all rooms onto a shared mesh.

        Iterates through all rooms sequentially, each projecting its bootstrap
        image onto the accumulating shared mesh. This ensures cross-room texture
        visibility (e.g., walls seen through doorways get textured).

        Args:
            room_contexts: List of room context dicts from _stage6_setup_room
            cam_idx: Camera index to project (0 or 1)
            input_mesh: Starting mesh path (structure_only.glb or previous shared mesh)
            output_mesh: Output path for accumulated shared mesh
        """
        current_mesh = input_mesh
        for ctx in room_contexts:
            room_id = ctx['room_id']
            output_dir = ctx['output_dir']
            bootstrap_image = _find_generated_image(output_dir / "generated", cam_idx)
            if bootstrap_image is None:
                logger.warning(f"No camera {cam_idx} image for {room_id}, skipping in shared texture pass")
                continue

            camera_name = f"{room_id}_{cam_idx:04d}"
            cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "project_wall_texture.py"),
                "--generated-image", str(bootstrap_image),
                "--structure-mesh", str(current_mesh),
                "--cameras-txt", str(self.paths.renders_final / "cameras.txt"),
                "--images-txt", str(self.paths.renders_final / "images.txt"),
                "--camera-name", camera_name,
                "--room-id", room_id,
                "--output-mesh", str(output_mesh),
                "--uv-margin", str(self.config.uv_margin),
            ]

            scene_depth = self.paths.renders_final / "depth" / room_id / f"{room_id}_{cam_idx:04d}_depth.npy"
            if scene_depth.exists():
                cmd.extend(["--scene-depth", str(scene_depth),
                            "--occlusion-tolerance", str(self.config.occlusion_tolerance)])
            if self.config.verbose:
                cmd.append("--verbose")

            success = run_command(
                f"Project camera {cam_idx} texture for {room_id} (shared)",
                cmd, cwd=self.project_dir, verbose=self.config.verbose
            )
            if success:
                current_mesh = output_mesh  # Next room projects onto accumulated result
            else:
                logger.warning(f"Shared texture projection failed for {room_id} camera {cam_idx}")

    def _stage6_shared_conditioning(self, shared_mesh):
        """Regenerate conditioning for ALL rooms using a shared textured mesh.

        This ensures every room's conditioning images include textures from
        all rooms' bootstrap cameras, so walls visible through doorways appear
        textured.

        Args:
            shared_mesh: Path to the shared textured mesh
        """
        depth_cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(self.project_dir / "generate_depth_object_conditioning.py"),
            "--renders-dir", str(self.paths.renders_final),
            "--structure-mesh", str(self.paths.structure_mesh),
            "--scene-json", str(self.config.scene_json),
            "--textured-mesh", str(shared_mesh),
            "--wall-texture-alpha", str(self.config.wall_texture_alpha),
            # No --rooms flag = process all rooms
        ]
        if self.config.verbose:
            depth_cmd.append("--verbose")

        return run_command(
            "Regen conditioning for all rooms (shared textures)",
            depth_cmd, cwd=self.project_dir, verbose=self.config.verbose
        )

    def _stage6_setup_room(self, room_id, all_prompts_file):
        """Stage 6 per-room setup: skip check, directory creation, prompt generation.

        Returns a context dict for subsequent stages, or None if room should be skipped.
        """
        # Check if already completed
        if room_id in self.state.completed_rooms.get('stage6', []):
            logger.info(f"[SKIP] {room_id}: already completed")
            return None

        input_dir = self.paths.renders_final / "images" / room_id
        output_dir = self.paths.flux_final / room_id

        if not input_dir.exists():
            logger.error(f"Final renders not found for {room_id}")
            return None

        # Generate per-camera prompts file (contains only opening descriptions)
        output_dir.mkdir(parents=True, exist_ok=True)
        prompts_file = output_dir / "camera_prompts.json"

        base_prompt = "Photorealistic."

        # Get opening descriptions for each camera (without door destination names)
        # Total cameras = 2 coverage + num_cameras regular + num_overhead
        total_cameras = 2 + self.config.num_cameras + self.config.num_overhead
        opening_descriptions = generate_prompts_for_room(
            self.scene_json,
            room_id,
            self.paths.renders_final / "images.txt",
            total_cameras,
            base_prompt,
            include_destinations=False,
        )

        with open(prompts_file, 'w') as f:
            json.dump(opening_descriptions, f, indent=2)

        # Write human-readable prompts
        self._write_prompts_for_room(room_id, output_dir, opening_descriptions)

        # Append to consolidated prompts file
        with open(all_prompts_file, 'a') as f_all:
            f_all.write(f"=== {room_id} ({len(opening_descriptions)} cameras) ===\n")
            f_all.write(f"Base prompt: {base_prompt}\n")
            f_all.write("Flow: Bootstrap 0 -> Texture v1 -> Bootstrap 1 -> Texture final -> Iterative 2-N\n\n")

        logger.info(f"Generated {len(opening_descriptions)} per-camera prompts for {room_id}")

        return {
            'room_id': room_id,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'prompts_file': prompts_file,
            'base_prompt': base_prompt,
        }

    def _stage6a_bootstrap(self, ctx):
        """Stage 6a: Bootstrap camera 0 for a room. Returns True on success."""
        room_id = ctx['room_id']
        input_dir = ctx['input_dir']
        output_dir = ctx['output_dir']
        prompts_file = ctx['prompts_file']
        base_prompt = ctx['base_prompt']

        bootstrap_image_0 = output_dir / "generated" / "generated_0000.png"

        if not bootstrap_image_0.exists():
            logger.info(f"[6a] Bootstrap camera 0 for {room_id}")
            cmd = self._build_flux_cmd_base(input_dir, output_dir, prompts_file, base_prompt, room_id)
            cmd.append("--only-bootstrap")

            success = run_command(
                f"Bootstrap Flux camera 0 for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            if not success or not bootstrap_image_0.exists():
                logger.error(f"Bootstrap camera 0 failed for {room_id}")
                cleanup_gpu()
                return False

            cleanup_gpu()
        else:
            logger.info(f"[6a] Bootstrap camera 0 already exists for {room_id}")

        return True

    def _stage6_room_post_bootstrap(self, ctx):
        """Stages 6b-6g for one room (post-bootstrap). Returns True on success.

        Used in sequential (non-parallel) mode only. In parallel mode,
        shared texture passes + _stage6d_camera1_bootstrap + _stage6g_iterative
        are used instead.
        """
        room_id = ctx['room_id']
        input_dir = ctx['input_dir']
        output_dir = ctx['output_dir']
        prompts_file = ctx['prompts_file']
        base_prompt = ctx['base_prompt']

        bootstrap_image_0 = output_dir / "generated" / "generated_0000.png"
        use_wall_texture = self.config.wall_texture_alpha > 0

        # === Stage 6b: Project wall texture from camera 0 -> textured_v1 ===
        textured_v1_path = self.paths.textured_mesh_v1(room_id)

        if use_wall_texture and not textured_v1_path.exists():
            logger.info(f"[6b] Projecting camera 0 wall textures for {room_id}")

            camera_name = f"{room_id}_0000"
            cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "project_wall_texture.py"),
                "--generated-image", str(bootstrap_image_0),
                "--structure-mesh", str(self.paths.structure_mesh),
                "--cameras-txt", str(self.paths.renders_final / "cameras.txt"),
                "--images-txt", str(self.paths.renders_final / "images.txt"),
                "--camera-name", camera_name,
                "--room-id", room_id,
                "--output-mesh", str(textured_v1_path),
                "--uv-margin", str(self.config.uv_margin),
            ]

            # Occlusion detection: use scene depth (with objects) to mask object regions
            scene_depth_0 = self.paths.renders_final / "depth" / room_id / f"{room_id}_0000_depth.npy"
            if scene_depth_0.exists():
                cmd.extend(["--scene-depth", str(scene_depth_0), "--occlusion-tolerance", str(self.config.occlusion_tolerance)])

            if self.config.verbose:
                cmd.append("--verbose")

            success = run_command(
                f"Project camera 0 wall texture for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            if not success:
                logger.warning(f"Camera 0 wall texture projection failed for {room_id}, continuing without textures")
                use_wall_texture = False
        elif use_wall_texture:
            logger.info(f"[6b] Textured v1 mesh already exists for {room_id}")

        # === Stage 6c: Regenerate conditioning with camera 0 textures ===
        if use_wall_texture and textured_v1_path.exists():
            logger.info(f"[6c] Regenerating conditioning with camera 0 textures for {room_id}")

            depth_cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "generate_depth_object_conditioning.py"),
                "--renders-dir", str(self.paths.renders_final),
                "--structure-mesh", str(self.paths.structure_mesh),
                "--scene-json", str(self.config.scene_json),
                "--textured-mesh", str(textured_v1_path),
                "--wall-texture-alpha", str(self.config.wall_texture_alpha),
                "--rooms", room_id,
            ]

            if self.config.verbose:
                depth_cmd.append("--verbose")

            depth_success = run_command(
                f"Regenerate conditioning with camera 0 textures for {room_id}",
                depth_cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            if not depth_success:
                logger.warning(f"Camera 0 texture conditioning failed for {room_id}")

        # === Stage 6d: Generate camera 1 (opposite bootstrap) ===
        bootstrap_image_1 = _find_generated_image(output_dir / "generated", 1)
        bootstrap_1_success = False

        if bootstrap_image_1 is None:
            logger.info(f"[6d] Generating camera 1 (opposite bootstrap) for {room_id}")
            cmd = self._build_flux_cmd_base(input_dir, output_dir, prompts_file, base_prompt, room_id,
                                            comfyui_port=ctx.get('comfyui_port'))
            cmd.extend(["--bootstrap-cameras", "0", "1", "--only-bootstrap"])

            success = run_command(
                f"Bootstrap Flux camera 1 for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            bootstrap_image_1 = _find_generated_image(output_dir / "generated", 1)
            if success and bootstrap_image_1 is not None:
                bootstrap_1_success = True
                cleanup_gpu()
            else:
                logger.warning(f"Camera 1 generation failed for {room_id}, continuing with camera 0 only")
                cleanup_gpu()
        else:
            logger.info(f"[6d] Bootstrap camera 1 already exists for {room_id}")
            bootstrap_1_success = True

        # === Stage 6e: Project wall texture from camera 1 onto textured_v1 -> textured_final ===
        textured_final_path = self.paths.textured_mesh(room_id)

        if use_wall_texture and bootstrap_1_success and not textured_final_path.exists():
            logger.info(f"[6e] Projecting camera 1 wall textures for {room_id}")

            camera_name = f"{room_id}_0001"
            cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "project_wall_texture.py"),
                "--generated-image", str(bootstrap_image_1),
                "--structure-mesh", str(textured_v1_path),  # Use already-textured mesh
                "--cameras-txt", str(self.paths.renders_final / "cameras.txt"),
                "--images-txt", str(self.paths.renders_final / "images.txt"),
                "--camera-name", camera_name,
                "--room-id", room_id,
                "--output-mesh", str(textured_final_path),
                "--uv-margin", str(self.config.uv_margin),
            ]

            # Occlusion detection: use scene depth (with objects) to mask object regions
            scene_depth_1 = self.paths.renders_final / "depth" / room_id / f"{room_id}_0001_depth.npy"
            if scene_depth_1.exists():
                cmd.extend(["--scene-depth", str(scene_depth_1), "--occlusion-tolerance", str(self.config.occlusion_tolerance)])

            if self.config.verbose:
                cmd.append("--verbose")

            success = run_command(
                f"Project camera 1 wall texture for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            if not success:
                logger.warning(f"Camera 1 wall texture projection failed for {room_id}, using v1 textures only")
                # Fallback: copy textured_v1 as textured_final
                shutil.copy(str(textured_v1_path), str(textured_final_path))
        elif use_wall_texture and not bootstrap_1_success and not textured_final_path.exists():
            # No camera 1 available, copy v1 as final
            if textured_v1_path.exists():
                logger.info(f"[6e] No camera 1, copying v1 textures as final for {room_id}")
                shutil.copy(str(textured_v1_path), str(textured_final_path))
        elif use_wall_texture:
            logger.info(f"[6e] Textured final mesh already exists for {room_id}")

        # === Stage 6f: Regenerate ALL conditioning with final textures ===
        if use_wall_texture and textured_final_path.exists():
            logger.info(f"[6f] Regenerating conditioning with final textures for {room_id}")

            depth_cmd = [
                "conda", "run", "-n", "worldmesh", "--no-capture-output",
                "python", str(self.project_dir / "generate_depth_object_conditioning.py"),
                "--renders-dir", str(self.paths.renders_final),
                "--structure-mesh", str(self.paths.structure_mesh),
                "--scene-json", str(self.config.scene_json),
                "--textured-mesh", str(textured_final_path),
                "--wall-texture-alpha", str(self.config.wall_texture_alpha),
                "--rooms", room_id,
            ]

            if self.config.verbose:
                depth_cmd.append("--verbose")

            depth_success = run_command(
                f"Regenerate conditioning with final textures for {room_id}",
                depth_cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            if not depth_success:
                logger.warning(f"Final texture conditioning failed for {room_id}, iterative cameras will use previous conditioning")

        # === Stage 6g: Iterative generation ===
        return self._stage6g_iterative(ctx)

    def _stage6d_camera1_bootstrap(self, ctx):
        """Stage 6d: Generate camera 1 bootstrap for a room. Returns True on success.

        Used in parallel mode where camera 1 generation is decoupled from
        texture projection (which happens in shared passes instead).
        """
        room_id = ctx['room_id']
        input_dir = ctx['input_dir']
        output_dir = ctx['output_dir']
        prompts_file = ctx['prompts_file']
        base_prompt = ctx['base_prompt']

        bootstrap_image_1 = _find_generated_image(output_dir / "generated", 1)

        if bootstrap_image_1 is None:
            logger.info(f"[6d] Generating camera 1 (opposite bootstrap) for {room_id}")
            cmd = self._build_flux_cmd_base(input_dir, output_dir, prompts_file, base_prompt, room_id,
                                            comfyui_port=ctx.get('comfyui_port'))
            cmd.extend(["--bootstrap-cameras", "0", "1", "--only-bootstrap"])

            success = run_command(
                f"Bootstrap Flux camera 1 for {room_id}",
                cmd,
                cwd=self.project_dir,
                verbose=self.config.verbose
            )

            bootstrap_image_1 = _find_generated_image(output_dir / "generated", 1)
            if success and bootstrap_image_1 is not None:
                cleanup_gpu()
                return True
            else:
                logger.warning(f"Camera 1 generation failed for {room_id}")
                cleanup_gpu()
                return False
        else:
            logger.info(f"[6d] Bootstrap camera 1 already exists for {room_id}")
            return True

    def _stage6g_iterative(self, ctx):
        """Stage 6g: Iterative Flux generation with per-camera texture accumulation.

        Generates cameras 2-N, projecting each camera's texture onto the room's
        textured mesh and regenerating conditioning after each projection.

        In parallel mode, the room's textured mesh starts as a copy of the shared
        textured mesh (containing all rooms' bootstrap textures). In sequential mode,
        it starts from the per-room textured_final mesh.

        Returns True on success.
        """
        room_id = ctx['room_id']
        input_dir = ctx['input_dir']
        output_dir = ctx['output_dir']
        prompts_file = ctx['prompts_file']
        base_prompt = ctx['base_prompt']
        use_wall_texture = self.config.wall_texture_alpha > 0

        textured_final_path = self.paths.textured_mesh(room_id)

        logger.info(f"[6g] Iterative Flux generation with texture accumulation for {room_id}")

        # Determine camera list from conditioning images
        all_conditioning = sorted(input_dir.glob("*_depth_objects.png"))
        camera_indices_raw = []
        for cond_path in all_conditioning:
            stem = cond_path.stem.replace("_depth_objects", "")
            idx_str = stem.split("_")[-1]
            try:
                idx = int(idx_str)
                if idx >= 2:  # Skip bootstrap cameras 0 and 1
                    camera_indices_raw.append(idx)
            except ValueError:
                continue

        # Sort by rotational proximity: greedy nearest-neighbor from bootstraps,
        # with overhead cameras deferred to the end
        images_txt = self.paths.renders_final / "images.txt"
        poses = get_poses_for_room(images_txt, room_id) if images_txt.exists() else {}
        overhead_start = 2 + self.config.num_cameras
        non_overhead = [c for c in camera_indices_raw if c < overhead_start]
        overhead = [c for c in camera_indices_raw if c >= overhead_start]
        camera_indices = order_by_rotational_proximity([0, 1], non_overhead, poses) + overhead
        if poses:
            logger.info(f"  Camera order (rotational proximity): {camera_indices}")

        for cam_idx in camera_indices:
            camera_name = f"{room_id}_{cam_idx:04d}"
            generated_image = _find_generated_image(output_dir / "generated", cam_idx)

            # 6g.i.1: Generate this camera
            if generated_image is None:
                logger.info(f"[6g] Generating camera {cam_idx} for {room_id}")
                cmd = self._build_flux_cmd_base(input_dir, output_dir, prompts_file, base_prompt, room_id,
                                                comfyui_port=ctx.get('comfyui_port'))
                cmd.extend(["--only-camera", str(cam_idx), "--skip-bootstrap"])

                success = run_command(
                    f"Flux camera {cam_idx} for {room_id}",
                    cmd,
                    cwd=self.project_dir,
                    verbose=self.config.verbose
                )
                cleanup_gpu()

                generated_image = _find_generated_image(output_dir / "generated", cam_idx)
                if not success or generated_image is None:
                    logger.warning(f"Camera {cam_idx} generation failed for {room_id}, skipping")
                    continue

            # 6g.i.2: Project texture from this camera onto the mesh
            current_textured = textured_final_path if textured_final_path.exists() else self.paths.structure_mesh

            if use_wall_texture and generated_image is not None:
                logger.info(f"[6g] Projecting camera {cam_idx} texture for {room_id}")
                proj_cmd = [
                    "conda", "run", "-n", "worldmesh", "--no-capture-output",
                    "python", str(self.project_dir / "project_wall_texture.py"),
                    "--generated-image", str(generated_image),
                    "--structure-mesh", str(current_textured),
                    "--cameras-txt", str(self.paths.renders_final / "cameras.txt"),
                    "--images-txt", str(self.paths.renders_final / "images.txt"),
                    "--camera-name", camera_name,
                    "--room-id", room_id,
                    "--output-mesh", str(textured_final_path),
                    "--uv-margin", str(self.config.uv_margin),
                ]

                # Occlusion detection
                scene_depth_path = self.paths.renders_final / "depth" / room_id / f"{camera_name}_depth.npy"
                if scene_depth_path.exists():
                    proj_cmd.extend(["--scene-depth", str(scene_depth_path), "--occlusion-tolerance", str(self.config.occlusion_tolerance)])

                if self.config.verbose:
                    proj_cmd.append("--verbose")

                with self._gpu_lock:
                    proj_success = run_command(
                        f"Project camera {cam_idx} texture for {room_id}",
                        proj_cmd,
                        cwd=self.project_dir,
                        verbose=self.config.verbose,
                        timeout=300,
                    )

                # 6g.i.3: Regenerate conditioning with updated textures
                if proj_success and textured_final_path.exists():
                    depth_cmd = [
                        "conda", "run", "-n", "worldmesh", "--no-capture-output",
                        "python", str(self.project_dir / "generate_depth_object_conditioning.py"),
                        "--renders-dir", str(self.paths.renders_final),
                        "--structure-mesh", str(self.paths.structure_mesh),
                        "--scene-json", str(self.config.scene_json),
                        "--textured-mesh", str(textured_final_path),
                        "--wall-texture-alpha", str(self.config.wall_texture_alpha),
                        "--rooms", room_id,
                    ]
                    if self.config.verbose:
                        depth_cmd.append("--verbose")

                    with self._gpu_lock:
                        run_command(
                            f"Regen conditioning after camera {cam_idx} for {room_id}",
                            depth_cmd,
                            cwd=self.project_dir,
                            verbose=self.config.verbose,
                            timeout=300,
                        )

        # Track completion (thread-safe)
        generated_dir = output_dir / "generated"
        if generated_dir.exists():
            count = len(list(generated_dir.glob("*.png")))
            logger.info(f"Generated {count} images for {room_id}")
            self._mark_room_completed_stage6(room_id)
        else:
            logger.error(f"No generated images found for {room_id}")
            return False

        # GPU cleanup
        cleanup_gpu()
        return True

    def _mark_room_completed_stage6(self, room_id):
        """Thread-safe helper to mark a room as completed in stage 6."""
        with self._state_lock:
            if 'stage6' not in self.state.completed_rooms:
                self.state.completed_rooms['stage6'] = []
            self.state.completed_rooms['stage6'].append(room_id)
            self._save_state()

    def _stage6_parallel_post_bootstrap(self, room_contexts):
        """Run shared texture projection + parallel generation for all rooms.

        New flow for parallel mode:
          Phase 1.5: Shared cam 0 texture pass (sequential, all rooms)
          Phase 2: Camera 1 bootstrap (parallel, all rooms)
          Phase 2.5: Shared cam 1 texture pass (sequential, all rooms)
          Phase 3: Iterative cameras 2-N (parallel, per-room with shared mesh as base)

        Returns True if all rooms succeeded.
        """
        logger.info(f"[Stage 6] Running parallel generation for {len(room_contexts)} rooms")
        all_success = True
        use_wall_texture = self.config.wall_texture_alpha > 0

        # === Phase 1.5: Shared cam 0 texture pass ===
        if use_wall_texture:
            logger.info("[Phase 1.5] Shared camera 0 texture pass (all rooms)")
            if not self.paths.shared_textured_mesh_v1.exists():
                self._stage6_shared_texture_pass(
                    room_contexts, 0,
                    self.paths.structure_mesh,
                    self.paths.shared_textured_mesh_v1
                )
                if self.paths.shared_textured_mesh_v1.exists():
                    self._stage6_shared_conditioning(self.paths.shared_textured_mesh_v1)
                else:
                    logger.warning("Shared cam 0 texture pass produced no output")
            else:
                logger.info("[Phase 1.5] Shared textured v1 mesh already exists")

        # === Phase 2: Camera 1 bootstrap (parallel) ===
        logger.info("[Phase 2] Camera 1 bootstrap for all rooms (parallel)")
        comfyui_processes = self._start_comfyui_instances(room_contexts)
        try:
            with ThreadPoolExecutor(max_workers=len(room_contexts)) as executor:
                future_to_room = {
                    executor.submit(self._stage6d_camera1_bootstrap, ctx): ctx['room_id']
                    for ctx in room_contexts
                }
                for future in as_completed(future_to_room):
                    room_id = future_to_room[future]
                    try:
                        success = future.result()
                        if success:
                            logger.info(f"[Phase 2] {room_id}: camera 1 bootstrap completed")
                        else:
                            logger.warning(f"[Phase 2] {room_id}: camera 1 bootstrap failed")
                    except Exception as e:
                        logger.error(f"[Phase 2] {room_id}: camera 1 bootstrap raised exception: {e}")
        finally:
            self._stop_comfyui_instances(comfyui_processes)

        # === Phase 2.5: Shared cam 1 texture pass ===
        if use_wall_texture:
            logger.info("[Phase 2.5] Shared camera 1 texture pass (all rooms)")
            input_mesh = self.paths.shared_textured_mesh_v1 if self.paths.shared_textured_mesh_v1.exists() else self.paths.structure_mesh
            if not self.paths.shared_textured_mesh.exists():
                self._stage6_shared_texture_pass(
                    room_contexts, 1,
                    input_mesh,
                    self.paths.shared_textured_mesh
                )
                if self.paths.shared_textured_mesh.exists():
                    self._stage6_shared_conditioning(self.paths.shared_textured_mesh)
                else:
                    logger.warning("Shared cam 1 texture pass produced no output, falling back to v1")
                    # If cam 1 pass failed but v1 exists, use v1 as the shared final
                    if self.paths.shared_textured_mesh_v1.exists():
                        shutil.copy(str(self.paths.shared_textured_mesh_v1), str(self.paths.shared_textured_mesh))
            else:
                logger.info("[Phase 2.5] Shared textured final mesh already exists")

        # === Phase 3: Iterative cameras (parallel, per-room) ===
        # Copy shared mesh as each room's starting point for iterative accumulation
        shared_mesh = self.paths.shared_textured_mesh if self.paths.shared_textured_mesh.exists() else (
            self.paths.shared_textured_mesh_v1 if self.paths.shared_textured_mesh_v1.exists() else None
        )
        if use_wall_texture and shared_mesh:
            for ctx in room_contexts:
                room_textured = self.paths.textured_mesh(ctx['room_id'])
                if not room_textured.exists():
                    logger.info(f"[Phase 3] Copying shared mesh as starting point for {ctx['room_id']}")
                    shutil.copy(str(shared_mesh), str(room_textured))

        logger.info(f"[Phase 3] Iterative cameras 2-N for {len(room_contexts)} rooms (parallel)")
        comfyui_processes = self._start_comfyui_instances(room_contexts)
        try:
            with ThreadPoolExecutor(max_workers=len(room_contexts)) as executor:
                future_to_room = {
                    executor.submit(self._stage6g_iterative, ctx): ctx['room_id']
                    for ctx in room_contexts
                }
                for future in as_completed(future_to_room):
                    room_id = future_to_room[future]
                    try:
                        success = future.result()
                        if success:
                            logger.info(f"[Phase 3] {room_id}: iterative generation completed successfully")
                        else:
                            logger.error(f"[Phase 3] {room_id}: iterative generation failed")
                            all_success = False
                    except Exception as e:
                        logger.error(f"[Phase 3] {room_id}: iterative generation raised exception: {e}")
                        all_success = False
        finally:
            self._stop_comfyui_instances(comfyui_processes)

        return all_success

    def _stage6_final_flux(self) -> bool:
        """Stage 6: Final Flux generation with dual-bootstrap wall texture projection.

        Sequential mode (per room):
          6a: Bootstrap camera 0 (Flux2-Klein 9B)
          6b: Project wall texture from camera 0 -> textured_v1.glb
          6c: Regenerate conditioning images with camera 0 textures
          6d: Generate camera 1 (opposite bootstrap, iterative with camera 0 as ref)
          6e: Project wall texture from camera 1 onto textured_v1 -> textured_final.glb
          6f: Regenerate conditioning images with both cameras' textures
          6g: Generate cameras 2-N (style ref = rotationally-closest of {0, 1})

        Parallel mode (cross-room shared textures):
          Phase 1:   Bootstrap camera 0 for all rooms (sequential)
          Phase 1.5: Shared cam 0 texture pass — project all rooms' cam 0 textures
                     onto one shared mesh, regen conditioning for ALL rooms
          Phase 2:   Camera 1 bootstrap for all rooms (parallel)
          Phase 2.5: Shared cam 1 texture pass — project all rooms' cam 1 textures
                     onto shared mesh, regen conditioning for ALL rooms
          Phase 3:   Iterative cameras 2-N (parallel, per-room starting from shared mesh)

        The shared texture passes ensure walls visible through doorways appear
        textured in conditioning images, even when those walls belong to other rooms.

        When wall_texture_alpha == 0: skip texture passes, still generate both
        bootstraps for reference diversity.
        """
        all_success = True

        # Create consolidated prompts file for all rooms
        self.paths.flux_final.mkdir(parents=True, exist_ok=True)
        all_prompts_file = self.paths.flux_final / "all_prompts.txt"
        is_parallel = self.config.parallel_rooms and len(self.rooms) > 1
        with open(all_prompts_file, 'w') as f_all:
            f_all.write("=== Stage 6: Final Flux Prompts (Dual Bootstrap) ===\n")
            f_all.write(f"Generated: {datetime.now().isoformat()}\n")
            f_all.write(f"Rooms: {len(self.rooms)}\n")
            f_all.write(f"Cameras per room: {self.config.num_cameras}\n")
            f_all.write(f"Mode: {'parallel (shared textures)' if is_parallel else 'sequential'}\n\n")
            if is_parallel:
                f_all.write("Flow (parallel mode with shared textures):\n")
                f_all.write("  Phase 1:   Bootstrap camera 0 for all rooms (sequential)\n")
                f_all.write("  Phase 1.5: Shared cam 0 texture pass (all rooms -> shared_textured_v1.glb)\n")
                f_all.write("  Phase 2:   Camera 1 bootstrap for all rooms (parallel)\n")
                f_all.write("  Phase 2.5: Shared cam 1 texture pass (all rooms -> shared_textured_final.glb)\n")
                f_all.write("  Phase 3:   Iterative cameras 2-N (parallel, per-room from shared mesh)\n\n")
            else:
                f_all.write("Flow per room:\n")
                f_all.write("  6a. Bootstrap camera 0: Flux2-Klein 9B\n")
                f_all.write("  6b. Project wall texture from camera 0 -> textured_v1\n")
                f_all.write("  6c. Regenerate conditioning with camera 0 textures\n")
                f_all.write("  6d. Bootstrap camera 1 (opposite wall): iterative with camera 0 as ref\n")
                f_all.write("  6e. Project wall texture from camera 1 onto textured_v1 -> textured_final\n")
                f_all.write("  6f. Regenerate conditioning with both cameras' textures\n")
                f_all.write("  6g. Iterative cameras 2-N: style ref = rotationally-closest of {0, 1}\n\n")

        # Phase 1: Setup + bootstrap camera 0 for all rooms (sequential)
        room_contexts = []
        for room_id in self.rooms:
            ctx = self._stage6_setup_room(room_id, all_prompts_file)
            if ctx is None:
                # Room was skipped (already completed) or had an error (missing renders)
                input_dir = self.paths.renders_final / "images" / room_id
                if room_id not in self.state.completed_rooms.get('stage6', []) and not input_dir.exists():
                    all_success = False
                continue

            if not self._stage6a_bootstrap(ctx):
                all_success = False
                continue

            room_contexts.append(ctx)

        # Phase 2+: Post-bootstrap — parallel (shared textures) or sequential
        if self.config.parallel_rooms and len(room_contexts) > 1:
            if not self._stage6_parallel_post_bootstrap(room_contexts):
                all_success = False
        else:
            for ctx in room_contexts:
                if not self._stage6_room_post_bootstrap(ctx):
                    all_success = False

        return all_success

    def _stage7_splatfacto_export(self) -> bool:
        """Stage 7: Export splatfacto COLMAP data for nerfstudio training."""
        cmd = [
            "conda", "run", "-n", "worldmesh", "--no-capture-output",
            "python", str(self.project_dir / "create_splatfacto_colmap.py"),
            "--pipeline-output", str(self.paths.base),
            "--rooms",
        ] + self.rooms + [
            "--output-dir", str(self.paths.splatfacto_colmap),
            "--subsample", str(self.config.splatfacto_subsample),
            "--max-points", str(self.config.splatfacto_max_points),
        ]

        if not self.config.splatfacto_export_depths:
            cmd.append("--no-export-depths")

        success = run_command(
            "Export splatfacto COLMAP",
            cmd,
            cwd=self.project_dir,
            verbose=self.config.verbose
        )

        if success and self.paths.splatfacto_colmap.exists():
            logger.info(f"Splatfacto COLMAP data exported to {self.paths.splatfacto_colmap}")

            # Log nerfstudio training commands
            output_dir = self.paths.splatfacto_colmap
            logger.info("")
            logger.info("Nerfstudio training commands:")
            logger.info(f"  ns-train splatfacto colmap --data {output_dir} --eval-mode all --pipeline.model.camera-optimizer.mode off")
            if self.config.splatfacto_export_depths:
                logger.info(f"  ns-train depth-splatfacto colmap --data {output_dir} \\")
                logger.info(f"    --pipeline.datamanager.dataparser.depth-unit-scale-factor 1e-3 \\")
                logger.info(f"    --pipeline.model.depth-loss-mult 0.1 \\")
                logger.info(f"    --eval-mode all --pipeline.model.camera-optimizer.mode off")
            return True

        logger.error("Splatfacto COLMAP export failed")
        return False

    def _stage8_nerfstudio_training(self) -> bool:
        """Stage 8: Run nerfstudio depth-splatfacto training."""
        colmap_data = self.paths.splatfacto_colmap
        if not colmap_data.exists():
            logger.error(f"Splatfacto COLMAP data not found at {colmap_data}")
            return False

        nerfstudio_output = self.paths.base / "nerfstudio_output"
        nerfstudio_output.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", self.config.nerfstudio_conda_env, "--no-capture-output",
            "ns-train", "depth-splatfacto",
            "--output-dir", str(nerfstudio_output),
            "--pipeline.model.depth-loss-mult", "0.7",
            "--pipeline.model.camera-optimizer.mode", "off",
            "--viewer.quit-on-train-completion", "True",
            "colmap",
            "--data", str(colmap_data),
            "--depth-unit-scale-factor", "1e-3",
            "--eval-mode", "all",
        ]

        # Always verbose for long-running training
        success = run_command(
            "Nerfstudio depth-splatfacto training",
            cmd,
            cwd=self.project_dir,
            verbose=True,
        )

        if not success:
            logger.error("Nerfstudio training failed")
            return False

        # Find the config.yml output by nerfstudio
        # nerfstudio writes to: <output-dir>/<data-name>/depth-splatfacto/<timestamp>/config.yml
        config_files = sorted(nerfstudio_output.rglob("config.yml"))
        if not config_files:
            logger.warning("Training completed but could not find config.yml in nerfstudio output")
            return True

        config_path = config_files[-1]  # most recent
        viewer_cmd = f"ns-viewer --load-config {config_path}"

        # Write view_scene.txt
        view_file = self.paths.base / "view_scene.txt"
        view_file.write_text(viewer_cmd + "\n")
        logger.info(f"Wrote viewer command to {view_file}")
        logger.info(f"To view the scene: {viewer_cmd}")

        return True

    # Ablation study variants
    ABLATION_VARIANTS = [
        {'name': 'depth_only',           'wall_texture_alpha': 0.0,  'needs_final_mesh': False, 'needs_bbox': False},
        {'name': 'depth_objects',        'wall_texture_alpha': 0.0,  'needs_final_mesh': True,  'needs_bbox': False},
        {'name': 'depth_textures_bbox',  'wall_texture_alpha': None, 'needs_final_mesh': False, 'needs_bbox': True},
    ]

    def _run_ablations(self) -> bool:
        """Run ablation study variants (replaces stages 5-7).

        For each variant:
        1. Create ablation subdirectory
        2. Set up final_mesh per variant strategy
        3. Swap paths/state/config
        4. Run stage 5 (render) + stage 6 (flux generation)
        5. Restore original paths/state/config

        Returns True if all variants succeed.
        """
        original_paths = self.paths
        original_state = self.state
        original_wall_texture_alpha = self.config.wall_texture_alpha
        all_success = True

        for variant in self.ABLATION_VARIANTS:
            variant_name = variant['name']
            ablation_dir = original_paths.base / f"ablation_{variant_name}"
            ablation_dir.mkdir(parents=True, exist_ok=True)

            logger.info("")
            logger.info("=" * 60)
            logger.info(f"ABLATION: {variant_name}")
            logger.info("=" * 60)

            # Check if this variant already completed (has flux_final output for all rooms)
            ablation_flux = ablation_dir / "flux_final"
            if ablation_flux.exists():
                rooms_done = sum(
                    1 for r in self.rooms
                    if (ablation_flux / r / "generated").exists()
                    and len(list((ablation_flux / r / "generated").glob("*.png"))) > 0
                )
                if rooms_done == len(self.rooms):
                    logger.info(f"[SKIP] {variant_name}: all {rooms_done} rooms already have generated images")
                    continue

            # Set up final_mesh per variant strategy
            ablation_final_mesh = ablation_dir / "scene_with_all_objects.glb"

            if variant['needs_final_mesh']:
                # depth_objects: copy shared scene_with_all_objects.glb
                shared_final = original_paths.final_mesh
                if shared_final.exists():
                    shutil.copy(str(shared_final), str(ablation_final_mesh))
                    logger.info(f"Copied final mesh to {ablation_final_mesh}")
                else:
                    logger.warning("No shared final mesh found, depth_objects will use structure mesh")

            elif variant['needs_bbox']:
                # depth_textures_bbox: run create_bbox_scene.py
                shared_final = original_paths.final_mesh
                if shared_final.exists():
                    logger.info(f"Creating AABB bbox scene for {variant_name}")
                    cmd = [
                        "conda", "run", "-n", "worldmesh", "--no-capture-output",
                        "python", str(self.project_dir / "create_bbox_scene.py"),
                        "--scene-mesh", str(shared_final),
                        "--structure-mesh", str(original_paths.structure_mesh),
                        "--output", str(ablation_final_mesh),
                    ]
                    success = run_command(
                        f"Create bbox scene for {variant_name}",
                        cmd,
                        cwd=self.project_dir,
                        verbose=self.config.verbose
                    )
                    if not success:
                        logger.error(f"Failed to create bbox scene for {variant_name}")
                        all_success = False
                        continue
                else:
                    logger.warning("No shared final mesh found, depth_textures_bbox will use structure mesh")

            # else: depth_only — no final_mesh needed, stage 5 falls back to structure_mesh

            # Swap paths to ablation directory
            self.paths = AblationPipelinePaths(base=ablation_dir, shared_base=original_paths.base)

            # Fresh state per variant (prevents room completion leaking between variants)
            self.state = PipelineState()
            self.state.started_at = original_state.started_at

            # Override wall_texture_alpha
            if variant['wall_texture_alpha'] is not None:
                self.config.wall_texture_alpha = variant['wall_texture_alpha']
            else:
                self.config.wall_texture_alpha = original_wall_texture_alpha

            # Run stage 5 (render) + stage 6 (flux generation)
            try:
                logger.info(f"[{variant_name}] Running Stage 5: Render Final")
                if not self._stage5_render_final():
                    logger.error(f"[{variant_name}] Stage 5 failed")
                    all_success = False
                    continue

                cleanup_gpu()

                logger.info(f"[{variant_name}] Running Stage 6: Final Flux Generation")
                if not self._stage6_final_flux():
                    logger.error(f"[{variant_name}] Stage 6 failed")
                    all_success = False
                    continue

                cleanup_gpu()

                if not self.config.skip_splatfacto:
                    logger.info(f"[{variant_name}] Running Stage 7: Export Splatfacto COLMAP")
                    if not self._stage7_splatfacto_export():
                        logger.warning(f"[{variant_name}] Stage 7 failed")
                    cleanup_gpu()

                if not self.config.skip_nerfstudio_training:
                    logger.info(f"[{variant_name}] Running Stage 8: Nerfstudio Training")
                    if not self._stage8_nerfstudio_training():
                        logger.warning(f"[{variant_name}] Stage 8 failed")
                    cleanup_gpu()

                # Save ablation variant state
                self._save_state()
                logger.info(f"[{variant_name}] Completed successfully")

            except Exception as e:
                logger.error(f"[{variant_name}] Error: {e}")
                all_success = False
            finally:
                # Restore original paths/state/config
                self.paths = original_paths
                self.state = original_state
                self.config.wall_texture_alpha = original_wall_texture_alpha

        return all_success

    def _print_summary(self):
        """Print pipeline summary."""
        logger.info("")
        logger.info("Output directory structure:")
        logger.info(f"  {self.paths.base}/")
        logger.info(f"    structure_only.glb        - Stage 1: Structure mesh")
        logger.info(f"    renders_structure/        - Stage 2: Initial renders")
        logger.info(f"    initial_flux/             - Stage 3: Initial Flux outputs")
        logger.info(f"    extracted_objects/        - Stage 4: Extracted objects")
        logger.info(f"    scene_with_all_objects.glb - Stage 4: Final mesh")
        logger.info(f"    renders_final/            - Stage 5: Final renders")
        logger.info(f"    flux_final/               - Stage 6: Final Flux outputs")
        logger.info(f"    splatfacto_colmap_all/    - Stage 7: Splatfacto COLMAP data")
        logger.info(f"    nerfstudio_output/       - Stage 8: Nerfstudio training output")
        logger.info(f"    view_scene.txt           - Stage 8: ns-viewer command")
        logger.info("")

        # Count final images
        total_images = 0
        for room_id in self.rooms:
            generated_dir = self.paths.flux_final / room_id / "generated"
            if generated_dir.exists():
                count = len(list(generated_dir.glob("*.png")))
                total_images += count
                logger.info(f"  {room_id}: {count} images")

        logger.info("")
        logger.info(f"Total generated images: {total_images}")

        # Print viewer command if Stage 8 completed, otherwise print training commands
        view_file = self.paths.base / "view_scene.txt"
        if view_file.exists():
            logger.info("")
            logger.info(f"Nerfstudio training complete! Viewer command in: {view_file}")
            logger.info(f"  {view_file.read_text().strip()}")
        elif self.paths.splatfacto_colmap.exists():
            output_dir = self.paths.splatfacto_colmap
            logger.info("")
            logger.info("Nerfstudio training commands:")
            logger.info(f"  ns-train splatfacto colmap --data {output_dir} --eval-mode all --pipeline.model.camera-optimizer.mode off")
            if self.config.splatfacto_export_depths:
                logger.info(f"  ns-train depth-splatfacto colmap --data {output_dir} --pipeline.datamanager.dataparser.depth-unit-scale-factor 1e-3 --pipeline.model.depth-loss-mult 0.1 --eval-mode all --pipeline.model.camera-optimizer.mode off")

        logger.info("")
        logger.info(f"Pipeline complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Scene Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_full_pipeline.py \\
      --scene-json scene_layout_large.json \\
      --output-dir output/my_scene \\
      --num-cameras 16 \\
      --verbose

  # Skip stages (for debugging/resume)
  python run_full_pipeline.py \\
      --scene-json scene_layout_large.json \\
      --output-dir output/my_scene \\
      --skip-stage 1 2  # Skip mesh generation and initial render

  # Room filtering
  python run_full_pipeline.py \\
      --scene-json scene_layout_large.json \\
      --output-dir output/my_scene \\
      --rooms living_room kitchen  # Only process specific rooms
"""
    )

    # Required arguments
    parser.add_argument(
        "--scene-json",
        type=Path,
        required=True,
        help="Path to scene JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for all pipeline outputs"
    )

    # Optional arguments
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=16,
        help="Number of camera views per room (default: 16)"
    )
    parser.add_argument(
        "--num-overhead",
        type=int,
        default=8,
        help="Number of overhead (elevated) camera views per room (default: 8)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1360,
        help="Image width in pixels (default: 1360)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height in pixels (default: 768)"
    )
    parser.add_argument(
        "--fov",
        type=int,
        default=90,
        help="Camera field of view in degrees for initial Flux generation (default: 90)"
    )
    parser.add_argument(
        "--fov-final",
        type=int,
        default=60,
        help="Camera field of view in degrees for final Flux generation (default: 60)"
    )
    parser.add_argument(
        "--wall-offset",
        type=float,
        default=0.4,
        help="Distance from wall for perimeter/overhead cameras in meters (default: 0.4)"
    )
    parser.add_argument(
        "--rooms",
        nargs="+",
        default=None,
        help="Only process specific rooms (default: all rooms)"
    )
    parser.add_argument(
        "--skip-stage",
        type=int,
        nargs="+",
        default=[],
        help="Skip specific stages (1-7)"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Text prompts for object segmentation"
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        help='Theme applied to bootstrap prompt (e.g. "a red and grim vampire castle")'
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Enable API mode (Nano Banana Pro / Gemini). This is already on by default."
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable API mode (Nano Banana Pro / Gemini). API is on by default."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="api",
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            "Iterative-camera model. 'api' (default) = Nano Banana Pro (paid). "
            "'flux2-klein-9b-distilled' = fully local ComfyUI workflow (free). "
            "Only used when --api is on (default)."
        ),
    )
    parser.add_argument(
        "--no-parallel-rooms",
        action="store_true",
        help="Disable parallel room processing in stage 6 (parallel is default for --model api)"
    )
    parser.add_argument(
        "--parallel-rooms",
        action="store_true",
        help=(
            "Force-enable parallel room processing in stage 6 even for local models "
            "(--model flux2-klein-9b-distilled). Only safe if you have enough VRAM "
            "for num_rooms simultaneous ComfyUI instances (each ~10 GB for flux2-klein-9b)."
        ),
    )
    parser.add_argument(
        "--comfy-api-key",
        type=str,
        default=None,
        help="ComfyOrg API key for API nodes (Gemini, etc.). Falls back to COMFY_API_KEY env var."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include AABB debug boxes in object placement output"
    )
    parser.add_argument(
        "--doorway-overlap-threshold",
        type=float,
        default=0.5,
        help="Fraction of object overlapping doorway to trigger exclusion (default: 0.5)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.003,
        help="Minimum mask area ratio for SAM3 segmentation (default: 0.003)"
    )
    parser.add_argument(
        "--scale-boost",
        type=float,
        default=2.3,
        help="Scale multiplier applied to all reconstructed objects (default: 2.3)"
    )
    parser.add_argument(
        "--wall-calibration",
        action="store_true",
        help="Enable wall-based scale calibration (default: off, uses fixed scale-boost instead)"
    )
    parser.add_argument(
        "--placement-mode",
        choices=["smart", "simple"],
        default="smart",
        help="Object placement mode: 'smart' (full heuristics) or 'simple' (minimal SAM3D-direct)"
    )
    parser.add_argument(
        "--no-manual-masks",
        action="store_false",
        dest="manual_masks",
        help="Use automatic SAM3 segmentation instead of Gradio UI for manual mask creation"
    )
    parser.add_argument(
        "--manual-masks-port",
        type=int,
        default=7860,
        help="Port for Gradio server for manual mask creation (default: 7860)"
    )
    parser.add_argument(
        "--use-edge-validation",
        action="store_true",
        help="Enable edge IoU validation (disabled by default)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="Edge IoU threshold for Flux generation validation (default: 0.3)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum attempts per iterative image before marking failed (default: 2)"
    )
    parser.add_argument(
        "--bootstrap-max-retries",
        type=int,
        default=30,
        help="Maximum attempts for bootstrap image before abandoning room (default: 30)"
    )
    parser.add_argument(
        "--min-bootstrap-attempts",
        type=int,
        default=3,
        help="Minimum bootstrap attempts before selecting best passing one (default: 3)"
    )
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.5,
        help="Minimum depth Edge IoU for validation pass (default: 0.5, higher is better)"
    )
    parser.add_argument(
        "--no-depth-validation",
        action="store_true",
        help="Disable depth-based validation, use edge-only"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip all validation, accept first generation attempt"
    )
    parser.add_argument(
        "--depth-conda-env",
        type=str,
        default="worldmesh-depth-pro",
        help="Conda environment with Depth Pro installed (default: worldmesh-depth-pro)"
    )
    parser.add_argument(
        "--depth-canny-low",
        type=float,
        default=0.1,
        help="Lower threshold for depth edge detection (default: 0.1)"
    )
    parser.add_argument(
        "--depth-canny-high",
        type=float,
        default=0.3,
        help="Upper threshold for depth edge detection (default: 0.3)"
    )
    parser.add_argument(
        "--depth-dilation-pixels",
        type=int,
        default=10,
        help="Pixel tolerance for depth edge alignment (default: 10)"
    )
    parser.add_argument(
        "--depth-sharpen",
        type=float,
        default=4.0,
        help="Sharpening strength for estimated depth edges (default: 4.0, 1.0=none)"
    )
    parser.add_argument(
        "--depth-sharpen-gt",
        type=float,
        default=3.0,
        help="Sharpening strength for GT depth edges (default: 3.0, 1.0=none)"
    )
    parser.add_argument(
        "--depth-min-gradient-percentile",
        type=float,
        default=25.0,
        help="Filter depth edges below this GT gradient percentile (default: 25.0, 0=disabled)"
    )
    parser.add_argument(
        "--wall-texture-alpha",
        type=float,
        default=1.0,
        help="Blend factor for wall textures in conditioning images (0=disabled, 1=full texture, default: 1.0)"
    )
    parser.add_argument(
        "--occlusion-tolerance",
        type=float,
        default=0.10,
        help="Depth tolerance in meters for object occlusion masking in wall texture projection (default: 0.10)"
    )
    parser.add_argument(
        "--uv-margin",
        type=float,
        default=0.05,
        help="UV margin beyond [0,1] to include corner faces in wall texture projection (default: 0.05)"
    )
    parser.add_argument(
        "--skip-splatfacto",
        action="store_true",
        help="Skip Stage 7 (splatfacto COLMAP export)"
    )
    parser.add_argument(
        "--splatfacto-subsample",
        type=str,
        default="auto",
        help="Point cloud subsampling factor for splatfacto export. Use 'auto' (default) to compute dynamically based on --splatfacto-max-points, or an integer for a fixed factor."
    )
    parser.add_argument(
        "--splatfacto-max-points",
        type=int,
        default=5_000_000,
        help="Target maximum number of points when --splatfacto-subsample auto (default: 5000000)"
    )
    parser.add_argument(
        "--no-export-depths",
        action="store_true",
        help="Disable depth map export for splatfacto training"
    )
    parser.add_argument(
        "--skip-nerfstudio-training",
        action="store_true",
        help="Skip Stage 8 (nerfstudio depth-splatfacto training)"
    )
    parser.add_argument(
        "--nerfstudio-conda-env",
        type=str,
        default="worldmesh-nerfstudio",
        help="Conda environment for nerfstudio training (default: worldmesh-nerfstudio)"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study: stages 1-4 shared, then 3 conditioning variants (depth_only, depth_objects, depth_textures_bbox) each with stages 5-6. Skips stage 7."
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.scene_json.exists():
        print(f"Error: Scene JSON not found: {args.scene_json}", file=sys.stderr)
        return 1

    # Build skip stages list
    skip_stages = list(args.skip_stage)
    if args.skip_splatfacto and 7 not in skip_stages:
        skip_stages.append(7)
    if args.skip_nerfstudio_training and 8 not in skip_stages:
        skip_stages.append(8)

    # Build config
    config = PipelineConfig(
        scene_json=args.scene_json.resolve(),
        output_dir=args.output_dir.resolve(),
        num_cameras=args.num_cameras,
        num_overhead=args.num_overhead,
        image_width=args.width,
        image_height=args.height,
        fov=args.fov,
        fov_final=args.fov_final,
        wall_offset=args.wall_offset,
        rooms_filter=args.rooms,
        skip_stages=skip_stages,
        verbose=args.verbose,
        debug=args.debug,
        doorway_overlap_threshold=args.doorway_overlap_threshold,
        min_area=args.min_area,
        scale_boost=args.scale_boost,
        wall_calibration=args.wall_calibration,
        placement_mode=args.placement_mode,
        manual_masks=args.manual_masks,
        manual_masks_port=args.manual_masks_port,
        use_edge_validation=args.use_edge_validation,
        iou_threshold=args.iou_threshold,
        max_retries=args.max_retries,
        bootstrap_max_retries=args.bootstrap_max_retries,
        min_bootstrap_attempts=args.min_bootstrap_attempts,
        no_validation=args.no_validation,
        use_depth_validation=not args.no_depth_validation,
        depth_threshold=args.depth_threshold,
        depth_conda_env=args.depth_conda_env,
        depth_canny_low=args.depth_canny_low,
        depth_canny_high=args.depth_canny_high,
        depth_dilation_pixels=args.depth_dilation_pixels,
        depth_sharpen=args.depth_sharpen,
        depth_sharpen_gt=args.depth_sharpen_gt,
        depth_min_gradient_percentile=args.depth_min_gradient_percentile,
        splatfacto_subsample=args.splatfacto_subsample,
        splatfacto_max_points=args.splatfacto_max_points,
        splatfacto_export_depths=not args.no_export_depths,
        skip_nerfstudio_training=args.skip_nerfstudio_training,
        nerfstudio_conda_env=args.nerfstudio_conda_env,
        theme=args.theme,
        api=args.api or (not args.no_api),
        model=args.model,
        # parallel_rooms is safe only when each ComfyUI instance is cheap.
        # - --model api (Nano Banana Pro): ComfyUI just makes HTTP API calls; cheap; safe to parallelize.
        # - --model flux2-klein-9b-distilled (or any future local model): each instance loads the
        #   full 9B UNet into VRAM (~10 GB on Ampere), so N rooms in parallel needs N × ~10 GB and
        #   would OOM on a 24 GB GPU for N > 2. Default to sequential for local models; the user
        #   can override with --parallel-rooms if they have enough VRAM.
        parallel_rooms=(
            (args.api or (not args.no_api))
            and not args.no_parallel_rooms
            and (args.model == "api" or args.parallel_rooms)
        ),
        comfy_api_key=args.comfy_api_key or os.environ.get("COMFY_API_KEY"),
        wall_texture_alpha=args.wall_texture_alpha,
        occlusion_tolerance=args.occlusion_tolerance,
        uv_margin=args.uv_margin,
        ablation=args.ablation,
    )

    # Override resolution for api mode (1376x768)
    if args.api or (not args.no_api):
        config.image_width = 1376

    if args.prompts:
        config.prompts = args.prompts

    # Run pipeline
    pipeline = FullPipeline(config)

    try:
        success = pipeline.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
