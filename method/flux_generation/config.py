"""Configuration dataclass for Flux generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .workflow_manager import (
    prepare_flux2_klein_base_iterative_workflow,
    prepare_flux2_klein_distilled_iterative_workflow,
    prepare_nano_workflow_style_first,
)


@dataclass(frozen=True)
class ModelSpec:
    """Per-model configuration for the iterative-camera workflow."""

    workflow_filename: str
    prepare_fn: Callable[[Dict[str, Any], str, str, str, Optional[int]], Dict[str, Any]]


# Registry of available models for the --model flag.
# Default "api" reproduces the Nano Banana Pro (paid ComfyOrg API) path.
# "flux2-klein-9b-distilled" routes through a local ComfyUI workflow with no API calls.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "api": ModelSpec(
        workflow_filename="api_nano_banana_pro_api.json",
        prepare_fn=prepare_nano_workflow_style_first,
    ),
    "flux2-klein-9b-distilled": ModelSpec(
        workflow_filename="image_flux2_klein_image_edit_9b_distilled_iterative_api.json",
        prepare_fn=prepare_flux2_klein_distilled_iterative_workflow,
    ),
    "flux2-klein-9b-base": ModelSpec(
        workflow_filename="image_flux2_klein_image_edit_9b_base_iterative.json",
        prepare_fn=prepare_flux2_klein_base_iterative_workflow,
    ),
}


@dataclass
class Config:
    """Configuration for Flux2-Klein image generation pipeline."""

    # Input/Output paths
    input_dir: Path
    output_dir: Path

    # ComfyUI settings
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188

    # Server management
    auto_start_server: bool = True
    comfyui_path: Optional[Path] = None
    startup_timeout: int = 120  # seconds

    # Workflow paths (auto-detected from ComfyUI)
    initial_workflow_path: Optional[Path] = None
    iterative_workflow_path: Optional[Path] = None
    qwen_workflow_path: Optional[Path] = None
    bootstrap_workflow_path: Optional[Path] = None

    # Validation settings (Edge IoU)
    use_edge_validation: bool = False  # Edge validation off by default
    iou_threshold: float = 0.3  # Edge IoU threshold (0.3 = 30% overlap after dilation)
    max_retries: int = 2
    bootstrap_max_retries: int = 30  # More retries for bootstrap (establishes style)
    min_bootstrap_attempts: int = 3  # Minimum attempts before selecting best bootstrap
    canny_low: float = 0.1  # Lower threshold for photorealistic images
    canny_high: float = 0.3
    dilation_pixels: int = 3  # Pixel tolerance for edge alignment

    # Depth validation settings (Edge IoU on depth maps)
    use_depth_validation: bool = True  # Enable depth-based validation
    depth_threshold: float = 0.78  # Min depth Edge IoU for validation pass (higher is better)
    depth_conda_env: str = "worldmesh-depth-pro"  # Conda environment for Depth Pro
    depth_canny_low: float = 0.1  # Lower threshold for depth edge detection
    depth_canny_high: float = 0.3  # Upper threshold for depth edge detection
    depth_dilation_pixels: int = 21  # Pixel tolerance for depth edge alignment
    depth_sharpen: float = 4.0  # Sharpening for estimated depth (1.0=none)
    depth_sharpen_gt: float = 3.0  # Sharpening for GT depth (1.0=none)
    depth_min_gradient_percentile: float = 25.0  # Filter edges below this GT gradient percentile (0=disabled)

    # Skip all validation (accept first generation attempt)
    no_validation: bool = False

    # Theme for bootstrap prompt (e.g. "a red and grim vampire castle")
    theme: Optional[str] = None

    # Optional prompt override (if None, use workflow's embedded prompt)
    prompt: Optional[str] = None

    # Per-camera prompts file (JSON: {camera_name: prompt})
    # Takes precedence over prompt if specified
    prompts_file: Optional[Path] = None

    # Scene JSON path (for opening detection in prompts)
    scene_json_path: Optional[Path] = None

    # Room ID (for opening detection, extracted from input_dir if not specified)
    room_id: Optional[str] = None

    # Verbose output
    verbose: bool = False

    # Bootstrap control flags
    only_bootstrap: bool = False  # Generate camera 0 only, then exit
    skip_bootstrap: bool = False  # Skip camera 0, start from camera 1
    only_camera: Optional[int] = None  # Generate only this camera using iterative workflow
    bootstrap_cameras: Optional[List[int]] = None  # Bootstrap camera IDs for style ref selection

    # API mode: enable 1376w resolution + paid Nano Banana Pro by default.
    # When api=True and model="api" (default), Nano Banana Pro is used (paid).
    # When api=True and model="flux2-klein-9b-distilled", a local workflow is used (free).
    api: bool = False

    # Iterative-camera model name (key into MODEL_REGISTRY).
    # Default "api" reproduces the legacy Nano Banana Pro path.
    model: str = "api"

    # ComfyOrg API key for API nodes (Gemini, etc.)
    # Get one at https://platform.comfy.org/login
    # Can also be set via COMFY_API_KEY environment variable
    comfy_api_key: Optional[str] = None

    # Nano workflow path (auto-detected)
    nano_workflow_path: Optional[Path] = None

    # Model workflow path (auto-detected from MODEL_REGISTRY[model].workflow_filename)
    model_workflow_path: Optional[Path] = None

    # Resume from progress file
    resume: bool = True

    def __post_init__(self):
        """Post-initialization: convert paths and auto-detect workflows."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

        # Convert comfyui_path if specified
        if self.comfyui_path is not None:
            self.comfyui_path = Path(self.comfyui_path)

        # Convert prompts_file if specified
        if self.prompts_file is not None:
            self.prompts_file = Path(self.prompts_file)

        # Convert scene_json_path if specified
        if self.scene_json_path is not None:
            self.scene_json_path = Path(self.scene_json_path)

        # Extract room_id from input_dir if not specified
        if self.room_id is None:
            self.room_id = self.input_dir.name

        # Auto-detect workflow paths if not specified (prefer API format)
        if self.initial_workflow_path is None:
            self.initial_workflow_path = self._find_initial_depth_qwen_workflow()
        else:
            self.initial_workflow_path = Path(self.initial_workflow_path)

        if self.iterative_workflow_path is None:
            self.iterative_workflow_path = self._find_api_workflow("iterative_cameras")
        else:
            self.iterative_workflow_path = Path(self.iterative_workflow_path)

        if self.qwen_workflow_path is None:
            self.qwen_workflow_path = self._find_qwen_workflow()
        else:
            self.qwen_workflow_path = Path(self.qwen_workflow_path)

        if self.bootstrap_workflow_path is None:
            self.bootstrap_workflow_path = self._find_bootstrap_workflow()
        else:
            self.bootstrap_workflow_path = Path(self.bootstrap_workflow_path)

        # Auto-detect nano workflow
        if self.nano_workflow_path is None:
            self.nano_workflow_path = self._find_nano_workflow()
        else:
            self.nano_workflow_path = Path(self.nano_workflow_path)

        # Auto-detect model workflow from registry
        if self.model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model}'. Available: {sorted(MODEL_REGISTRY.keys())}"
            )
        if self.model_workflow_path is None:
            self.model_workflow_path = self._find_model_workflow(self.model)
        else:
            self.model_workflow_path = Path(self.model_workflow_path)

        # When api mode is enabled, override bootstrap and qwen workflows to 1376w variants
        if self.api:
            bootstrap_1376w = self._find_bootstrap_1376w_workflow()
            if bootstrap_1376w is not None:
                self.bootstrap_workflow_path = bootstrap_1376w
            qwen_1376w = self._find_qwen_1376w_workflow()
            if qwen_1376w is not None:
                self.qwen_workflow_path = qwen_1376w

    def _find_workflow(self, filename: str) -> Path:
        """Find workflow file in common locations."""
        # Check relative to this package
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "user" / "default" / "workflows" / filename,
            package_dir / "workflows" / filename,
            Path.home() / ".comfyui" / "workflows" / filename,
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find workflow file: {filename}. "
            f"Searched in: {[str(p) for p in search_paths]}"
        )

    def _find_api_workflow(self, base_name: str) -> Path:
        """Find API-format workflow file (preferred) or fall back to regular workflow."""
        api_filename = f"{base_name}_api.json"
        regular_filename = f"{base_name}.json"

        # Try API format first
        try:
            return self._find_workflow(api_filename)
        except FileNotFoundError:
            # Fall back to regular format
            return self._find_workflow(regular_filename)

    def _find_qwen_workflow(self) -> Optional[Path]:
        """Find Flux2-Klein Base iterative workflow file in scenes_workflows directory."""
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / "image_flux2_klein_image_edit_9b_base_1302_api.json",
            package_dir / "comfyui" / "user" / "default" / "workflows" / "image_flux2_klein_image_edit_9b_base_1302_api.json",
            package_dir / "workflows" / "image_flux2_klein_image_edit_9b_base_1302_api.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _find_bootstrap_workflow(self) -> Optional[Path]:
        """Find bootstrap Flux2-Klein 9B workflow file in scenes_workflows directory."""
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / "bootstrap_flux2_klein_9b_api.json",
            package_dir / "comfyui" / "user" / "default" / "workflows" / "bootstrap_flux2_klein_9b_api.json",
            package_dir / "workflows" / "bootstrap_flux2_klein_9b_api.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _find_initial_depth_qwen_workflow(self) -> Path:
        """Find initial workflow file in scenes_workflows directory."""
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / "image_flux2_klein_image_edit_9b_distilled_api.json",
            package_dir / "comfyui" / "user" / "default" / "workflows" / "image_flux2_klein_image_edit_9b_distilled_api.json",
            package_dir / "workflows" / "image_flux2_klein_image_edit_9b_distilled_api.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Fall back to legacy initial_camera workflow
        return self._find_api_workflow("initial_camera")

    def _find_nano_workflow(self) -> Optional[Path]:
        """Find Nano Banana Pro workflow file in scenes_workflows directory."""
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / "api_nano_banana_pro_api.json",
            package_dir / "comfyui" / "user" / "default" / "workflows" / "api_nano_banana_pro_api.json",
            package_dir / "workflows" / "api_nano_banana_pro_api.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _find_model_workflow(self, model_name: str) -> Optional[Path]:
        """Find the iterative-camera workflow JSON for the named model."""
        filename = MODEL_REGISTRY[model_name].workflow_filename
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / filename,
            package_dir / "comfyui" / "user" / "default" / "workflows" / filename,
            package_dir / "workflows" / filename,
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _find_bootstrap_1376w_workflow(self) -> Optional[Path]:
        """Find 1376w bootstrap workflow for api mode."""
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / "bootstrap_flux2_klein_9b_api_1376w.json",
            package_dir / "comfyui" / "user" / "default" / "workflows" / "bootstrap_flux2_klein_9b_api_1376w.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _find_qwen_1376w_workflow(self) -> Optional[Path]:
        """Find 1376w qwen iterative workflow for api mode."""
        package_dir = Path(__file__).parent.parent.parent

        search_paths = [
            package_dir / "comfyui" / "scenes_workflows" / "image_flux2_klein_image_edit_9b_base_1376w_api.json",
            package_dir / "comfyui" / "user" / "default" / "workflows" / "image_flux2_klein_image_edit_9b_base_1376w_api.json",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    @property
    def edges_dir(self) -> Path:
        """Get edges directory (sibling to images in input)."""
        # Input dir is like .../images/master_bedroom
        # Edges dir is like .../edges/master_bedroom
        room_name = self.input_dir.name
        parent = self.input_dir.parent.parent  # Go up from images/room to parent
        return parent / "edges" / room_name

    @property
    def images_txt_path(self) -> Path:
        """Get path to images.txt (COLMAP format)."""
        # images.txt is at the root level alongside images/ and edges/
        return self.input_dir.parent.parent / "images.txt"

    @property
    def generated_dir(self) -> Path:
        """Get directory for successful generations."""
        return self.output_dir / "generated"

    @property
    def failed_dir(self) -> Path:
        """Get directory for failed attempts."""
        return self.output_dir / "failed"

    @property
    def comparison_dir(self) -> Path:
        """Get directory for edge comparisons."""
        return self.output_dir / "edges_comparison"

    @property
    def depth_comparison_dir(self) -> Path:
        """Get directory for depth comparisons."""
        return self.output_dir / "depth_comparison"

    @property
    def progress_path(self) -> Path:
        """Get path to progress JSON file."""
        return self.output_dir / "progress.json"

    @property
    def failures_path(self) -> Path:
        """Get path to failures JSON file."""
        return self.output_dir / "failures.json"

    def ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        if self.use_depth_validation:
            self.depth_comparison_dir.mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate configuration."""
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")

        if not self.edges_dir.exists():
            import warnings
            warnings.warn(f"Edges directory does not exist: {self.edges_dir} (not required for depth-based workflows)")

        if not self.images_txt_path.exists():
            raise ValueError(f"images.txt not found at: {self.images_txt_path}")

        if not self.initial_workflow_path.exists():
            raise ValueError(
                f"Initial workflow not found: {self.initial_workflow_path}"
            )

        if not self.iterative_workflow_path.exists():
            raise ValueError(
                f"Iterative workflow not found: {self.iterative_workflow_path}"
            )

        if not 0 <= self.iou_threshold <= 1:
            raise ValueError(f"IoU threshold must be between 0 and 1: {self.iou_threshold}")

        if self.max_retries < 1:
            raise ValueError(f"Max retries must be at least 1: {self.max_retries}")

        if self.api and (self.model_workflow_path is None or not self.model_workflow_path.exists()):
            raise ValueError(
                f"API mode enabled but model workflow not found for --model {self.model!r}: {self.model_workflow_path}"
            )

        if not 0 <= self.depth_threshold <= 1:
            raise ValueError(f"Depth threshold must be between 0 and 1 (IoU): {self.depth_threshold}")
