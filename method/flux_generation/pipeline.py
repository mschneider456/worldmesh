"""Main orchestration for Flux2-Klein image generation pipeline."""

import asyncio
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image

import sys
# Allow importing from parent directory (for opening_visibility)
sys.path.insert(0, str(Path(__file__).parent.parent))

from .camera_poses import CameraPose, get_poses_for_room, find_best_reference, order_by_rotational_proximity
from .comfyui_client import ComfyUIClient
from .comfyui_server import ComfyUIServer
from .config import Config, MODEL_REGISTRY
from .edge_validator import validate_generation, create_comparison_image
from .depth_validator import validate_depth

from .workflow_manager import (
    load_workflow,
    get_prompt_from_workflow,
    prepare_initial_workflow,
    prepare_iterative_workflow,
    prepare_nano_workflow,
    prepare_qwen_iterative_workflow,
)


class ComfyUIUnrecoverableError(Exception):
    """Raised when ComfyUI is unrecoverable and the pipeline must abort."""
    pass


@dataclass
class AttemptRecord:
    """Record of a single generation attempt."""

    attempt: int
    seed: int
    ssim: float  # Actually stores edge IoU, not SSIM (legacy name)
    image_path: str
    success: bool
    depth_edge_recall: float = 0.0  # Edge recall from depth validation (higher is better)


@dataclass
class Progress:
    """Progress tracking for the pipeline."""

    completed_ids: List[int] = field(default_factory=list)
    failed_ids: Dict[int, List[AttemptRecord]] = field(default_factory=dict)
    successful_generations: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "completed_ids": self.completed_ids,
            "failed_ids": {
                str(k): [
                    {
                        "attempt": r.attempt,
                        "seed": r.seed,
                        "ssim": r.ssim,
                        "image_path": r.image_path,
                        "success": r.success,
                        "depth_edge_recall": r.depth_edge_recall,
                    }
                    for r in v
                ]
                for k, v in self.failed_ids.items()
            },
            "successful_generations": {
                str(k): v for k, v in self.successful_generations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Progress":
        """Load from JSON dict."""
        progress = cls()
        progress.completed_ids = data.get("completed_ids", [])
        progress.successful_generations = {
            int(k): v for k, v in data.get("successful_generations", {}).items()
        }

        for k, attempts in data.get("failed_ids", {}).items():
            progress.failed_ids[int(k)] = [
                AttemptRecord(
                    attempt=a["attempt"],
                    seed=a["seed"],
                    ssim=a["ssim"],
                    image_path=a["image_path"],
                    success=a.get("success", False),
                    # Support old field names: depth_edge_iou, depth_si_rmse
                    depth_edge_recall=a.get("depth_edge_recall", a.get("depth_edge_iou", a.get("depth_si_rmse", 0.0))),
                )
                for a in attempts
            ]

        return progress

    def save(self, path: Path) -> None:
        """Save progress to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Progress":
        """Load progress from JSON file."""
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class FluxGenerationPipeline:
    """Main pipeline for generating images with Flux2-Klein."""

    # Max consecutive recovery failures before forcing server restart
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, config: Config):
        self.config = config
        self.progress = Progress()
        self.client: Optional[ComfyUIClient] = None
        self.server: Optional[ComfyUIServer] = None
        self.initial_workflow: Optional[Dict[str, Any]] = None
        self.iterative_workflow: Optional[Dict[str, Any]] = None
        self.qwen_workflow: Optional[Dict[str, Any]] = None
        self.bootstrap_workflow: Optional[Dict[str, Any]] = None
        self.nano_workflow: Optional[Dict[str, Any]] = None
        self.model_workflow: Optional[Dict[str, Any]] = None
        self.prompt: Optional[str] = None
        self.poses: Dict[int, CameraPose] = {}
        self.camera_prompts: Dict[str, str] = {}  # Per-camera prompts (camera_name -> prompt)
        self.scene_json: Optional[Dict[str, Any]] = None  # Loaded scene JSON for window detection
        self._consecutive_recovery_failures = 0

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    async def _recover_comfyui(self) -> None:
        """
        Attempt to recover ComfyUI after a timeout.

        First tries soft recovery (interrupt/clear_queue).
        After MAX_CONSECUTIVE_FAILURES, forces a server restart.

        Raises ComfyUIUnrecoverableError if recovery is impossible.
        """
        self.log("    Attempting to recover ComfyUI...")

        # Try soft recovery first — scope to the last queued prompt if possible
        try:
            prompt_id = getattr(self.client, 'last_prompt_id', None)
            if prompt_id:
                await self.client.interrupt(prompt_id)
                await self.client.delete_from_queue(prompt_id)
                self.log(f"    Recovery complete (per-prompt: {prompt_id[:8]})")
            else:
                await self.client.interrupt()
                await self.client.clear_queue()
                self.log("    Recovery complete (global)")
            await asyncio.sleep(2.0)
            self._consecutive_recovery_failures = 0
            return
        except Exception as e:
            self._consecutive_recovery_failures += 1
            self.log(f"    Soft recovery failed ({self._consecutive_recovery_failures}/{self.MAX_CONSECUTIVE_FAILURES}): {e}")

        # Check if we need to force restart
        if self._consecutive_recovery_failures >= self.MAX_CONSECUTIVE_FAILURES:
            if self.server is None:
                self.log("    Cannot restart server (not managed by pipeline)")
                self.log("    FATAL: ComfyUI is unresponsive. Please restart it manually.")
                raise ComfyUIUnrecoverableError(
                    "ComfyUI is unresponsive and cannot be restarted (not managed by pipeline)"
                )

            self.log("    Forcing ComfyUI server restart...")
            try:
                await self.server.restart(startup_timeout=self.config.startup_timeout)
                # Reconnect client
                if not await self.client.check_connection():
                    raise RuntimeError("Failed to reconnect after restart")
                self._consecutive_recovery_failures = 0
                self.log("    Recovery complete (server restart)")
                return
            except Exception as restart_error:
                self.log(f"    Server restart failed: {restart_error}")
                self.log("    FATAL: ComfyUI is unrecoverable. Aborting.")
                raise ComfyUIUnrecoverableError(
                    f"ComfyUI server restart failed: {restart_error}"
                )

        # Soft recovery failed but we haven't hit the limit yet - continue trying

    def _get_input_images(self) -> List[Path]:
        """Get sorted list of input images (prefer depth+objects, fallback to edges)."""
        # Try depth+objects conditioning first (new format)
        pattern = "*_depth_objects.png"
        images = sorted(self.config.input_dir.glob(pattern))
        if images:
            self.log(f"Using depth+objects conditioning ({len(images)} images)")
            return images

        # Fall back to edges (legacy format)
        pattern = "*_with_edges.png"
        images = sorted(self.config.input_dir.glob(pattern))
        if images:
            self.log(f"Using edge-based conditioning ({len(images)} images)")
        return images

    def _get_corner_camera_ids(self) -> set:
        """Read camera_info.json to determine corner camera IDs.

        Layout: [coverage, regular, central_corner, overhead(last)]
        Corner cameras start after coverage + regular.
        """
        camera_info_path = self.config.input_dir.parent.parent / "camera_info.json"
        if not camera_info_path.exists():
            return set()
        try:
            with open(camera_info_path) as f:
                info = json.load(f)
            cpr = info.get("cameras_per_room", {})
            num_coverage = cpr.get("room_coverage", 1)
            num_regular = cpr.get("regular", 0)
            num_corner = cpr.get("central_corner", 0)
            if num_corner > 0:
                start = num_coverage + num_regular
                return set(range(start, start + num_corner))
        except (json.JSONDecodeError, KeyError):
            pass
        return set()

    def _get_overhead_camera_ids(self) -> set:
        """Read camera_info.json to determine overhead camera IDs.

        Layout: [coverage, regular, central_corner, overhead(last)]
        Overhead cameras are the last block.
        """
        camera_info_path = self.config.input_dir.parent.parent / "camera_info.json"
        if not camera_info_path.exists():
            return set()
        try:
            with open(camera_info_path) as f:
                info = json.load(f)
            cpr = info.get("cameras_per_room", {})
            num_overhead = cpr.get("overhead", 0)
            total = cpr.get("total", 0)
            if num_overhead > 0 and total > 0:
                return set(range(total - num_overhead, total))
        except (json.JSONDecodeError, KeyError):
            pass
        return set()

    def _select_style_reference(self, image_id: int) -> tuple:
        """Pick rotationally-closest successful generation as style reference.

        Searches ALL successful generations (not just bootstrap cameras).
        Returns (path, ref_id) or (None, None).
        """
        available = list(self.progress.successful_generations.keys())
        if not available:
            return None, None
        if len(available) == 1:
            bid = available[0]
            return Path(self.progress.successful_generations[bid]), bid
        best_id = find_best_reference(image_id, available, self.poses)
        if best_id is None:
            best_id = available[0]
        return Path(self.progress.successful_generations[best_id]), best_id

    def _extract_image_id(self, image_path: Path) -> int:
        """Extract numeric ID from image filename."""
        stem = image_path.stem
        # Handle both "_depth_objects" and "_with_edges" suffixes
        for suffix in ("_depth_objects", "_with_edges"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        match = re.search(r"_(\d+)$", stem)
        if match:
            return int(match.group(1))
        return -1

    def _extract_camera_name(self, image_path: Path) -> str:
        """Extract camera name from image filename."""
        # Filename like "master_bedroom_0000_with_edges.png" or "master_bedroom_0000_depth_objects.png"
        # Camera name is "master_bedroom_0000"
        stem = image_path.stem
        for suffix in ("_depth_objects", "_with_edges"):
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem

    def _get_prompt_for_image(self, image_path: Path) -> Optional[str]:
        """Get prompt for a specific image, checking per-camera prompts first."""
        camera_name = self._extract_camera_name(image_path)
        if camera_name in self.camera_prompts:
            return self.camera_prompts[camera_name]
        return self.prompt

    def _get_generated_path(self, image_id: int, ref_id: int = None) -> Path:
        """Get path where successful generation will be saved.

        If ref_id is provided, the filename includes the reference camera index
        (e.g. generated_0007_ref0003.png).
        """
        if ref_id is not None:
            return self.config.generated_dir / f"generated_{image_id:04d}_ref{ref_id:04d}.png"
        return self.config.generated_dir / f"generated_{image_id:04d}.png"

    def _find_generated(self, image_id: int) -> Optional[Path]:
        """Find generated image for camera index, regardless of ref suffix."""
        matches = sorted(self.config.generated_dir.glob(f"generated_{image_id:04d}*.png"))
        return matches[0] if matches else None

    def _get_failed_path(self, image_id: int, attempt: int, seed: int, ref_id: Optional[int] = None) -> Path:
        """Get path for a failed attempt."""
        ref_suffix = f"_ref{ref_id}" if ref_id is not None else ""
        return self.config.failed_dir / f"{image_id}_attempt_{attempt}_seed_{seed}{ref_suffix}.png"

    def _get_comparison_path(self, image_id: int, attempt: int, ref_id: Optional[int] = None) -> Path:
        """Get path for edge comparison image."""
        ref_suffix = f"_ref{ref_id}" if ref_id is not None else ""
        return self.config.comparison_dir / f"{image_id}_attempt_{attempt}{ref_suffix}_comparison.png"

    def _get_depth_comparison_path(self, image_id: int, attempt: int, ref_id: Optional[int] = None) -> Path:
        """Get path for depth comparison image."""
        ref_suffix = f"_ref{ref_id}" if ref_id is not None else ""
        return self.config.depth_comparison_dir / f"{image_id}_attempt_{attempt}{ref_suffix}_depth_comparison.png"

    async def _upload_image(self, image_path: Path) -> str:
        """Upload an image to ComfyUI and return the filename."""
        return await self.client.upload_image(image_path)

    async def _generate_initial(
        self, input_filename: str, seed: int, prompt: Optional[str] = None
    ) -> List[Image.Image]:
        """Generate using initial Flux2-Klein workflow (single depth map input).

        Args:
            input_filename: Uploaded depth map image filename
            seed: Random seed for generation
            prompt: Full prompt for generation (already constructed by caller)
        """
        effective_prompt = prompt or self.prompt or ""

        workflow = prepare_initial_workflow(
            self.initial_workflow,
            image_filename=input_filename,
            prompt=effective_prompt,
            seed=seed,
        )
        return await self.client.generate(workflow)

    async def _generate_iterative(
        self,
        current_filename: str,
        reference_filename: str,
        seed: int,
        prompt: Optional[str] = None,
    ) -> List[Image.Image]:
        """Generate using iterative (dual-image) workflow.

        Args:
            current_filename: Uploaded current input image filename
            reference_filename: Uploaded reference (previously generated) image filename
            seed: Random seed for generation
            prompt: Opening description (not the full prompt)
        """
        iterative_prompt = "Generate scene in reference_image2 from the perspective of reference_image1."

        workflow = prepare_iterative_workflow(
            self.iterative_workflow,
            current_filename,
            reference_filename,
            prompt=iterative_prompt,
            seed=seed,
        )
        return await self.client.generate(workflow)

    def _build_qwen_prompt(self, camera_name: str) -> str:
        """Build prompt for Flux2-Klein Base iterative workflow.

        Args:
            camera_name: Camera name (e.g., "master_bedroom_0000")

        Returns:
            Complete prompt for iterative workflow
        """
        base = "Generate a photorealistic version from camera pose of reference_image 1 which shows the scene in reference_image 2."
        if self.config.theme:
            return f"{base} {self.config.theme}"
        return base

    async def _generate_qwen_iterative(
        self,
        style_image_path: Path,
        structure_image_path: Path,
        prompt: str,
        seed: int,
    ) -> List[Image.Image]:
        """Generate image using Flux2-Klein Base iterative workflow.

        Args:
            style_image_path: Path to the style reference image
            structure_image_path: Path to the structure image (depth+objects)
            prompt: The generation prompt
            seed: Random seed for generation

        Returns:
            List of generated images
        """
        # Upload both images
        style_filename = await self._upload_image(style_image_path)
        structure_filename = await self._upload_image(structure_image_path)

        if self.config.verbose:
            self.log(f"    [DEBUG] reference_image1 (structure/camera pose) [node 76]: {structure_image_path.name} -> {structure_filename}")
            self.log(f"    [DEBUG] reference_image2 (style/scene reference) [node 81]: {style_image_path.name} -> {style_filename}")

        # Prepare and queue workflow
        workflow = prepare_qwen_iterative_workflow(
            self.qwen_workflow,
            style_filename,
            structure_filename,
            prompt,
            seed,
        )

        if self.config.verbose:
            self.log(f"    [DEBUG] Node 76 (structure/ref_image1): {workflow.get('76', {}).get('inputs', {}).get('image', 'NOT SET')}")
            self.log(f"    [DEBUG] Node 81 (style/ref_image2): {workflow.get('81', {}).get('inputs', {}).get('image', 'NOT SET')}")

        return await self.client.generate(workflow, verbose=self.config.verbose)

    def _build_nano_prompt(self, camera_name: str) -> str:
        """Build prompt for Nano Banana Pro iterative workflow.

        Args:
            camera_name: Camera name (e.g., "master_bedroom_0000")

        Returns:
            Complete prompt for nano iterative workflow
        """
        theme = self.config.theme or "a photorealistic interior scene"
        return (
            f'The scene depicts a "{theme}"\n\n'
            "The first image is a 3D rendering showing the correct camera angle, composition, and object layout for the desired output. "
            "However, it is not photorealistic — the objects are placeholders that show correct position, scale, and shape but not correct appearance. "
            "The rendering contains texture seams and stretching artifacts from steep-angle projection that must be removed. "
            "Additionally, wall and surface textures in the rendering are flat 2D projections — the output should depict these surfaces with proper 3D depth, volume, and parallax-correct detail.\n\n"
            "The second image is a photorealistic photograph of the same scene from a very different camera angle. "
            "It defines how the scene should actually look.\n\n"
            "The camera positions differ significantly, so not all objects are visible in both images. "
            "Before generating, establish correspondences between objects in the rendering and the photograph based on shape, position, and spatial context.\n\n"
            "This image is part of an iterative generation pipeline, so the input rendering may contain accumulated visual artifacts such as blur, color degradation, loss of detail, or repeated compression artifacts from previous iterations. Actively counteract these by producing a clean, sharp, and visually pleasing output — treat the generation as an opportunity to restore and enhance visual quality rather than propagate existing degradation.\n\n"
            "Generate a photorealistic image that:\n"
            "1. Preserves the exact camera angle, composition, and object layout from the rendering — do not add, remove, or rearrange any objects\n"
            "2. Transfers photorealistic appearance, materials, lighting, and textures from corresponding objects in the photograph — not from the rendering\n"
            "3. Does not introduce objects from the photograph that are absent in the rendering\n"
            "4. Removes all texture seams, stretching artifacts, flat 2D texturing, placeholder appearances, and any accumulated pipeline artifacts — surfaces should have realistic 3D depth and detail\n"
            "5. Is multi-view consistent with the photograph, as if both were real photos of the same physical scene\n"
            "6. For objects in the rendering with no match in the photograph, infers plausible photorealistic appearance consistent with the scene\n\n"
            "Use the rendering for geometry and camera only. Use the photograph for appearance only."
        )

    async def _generate_nano_iterative(
        self,
        style_image_path: Path,
        structure_image_path: Path,
        prompt: str,
        seed: int,
    ) -> List[Image.Image]:
        """Generate image using Nano Banana Pro (GeminiImage2Node) iterative workflow.

        Args:
            style_image_path: Path to the style reference image (photorealistic)
            structure_image_path: Path to the structure image (3D rendering)
            prompt: The generation prompt
            seed: Random seed for generation

        Returns:
            List of generated images
        """
        # Upload both images
        structure_filename = await self._upload_image(structure_image_path)
        style_filename = await self._upload_image(style_image_path)

        if self.config.verbose:
            self.log(f"    [DEBUG] Node 11 (structure/3D rendering): {structure_image_path.name} -> {structure_filename}")
            self.log(f"    [DEBUG] Node 12 (style/photo reference): {style_image_path.name} -> {style_filename}")

        # Prepare and queue workflow
        workflow = prepare_nano_workflow(
            self.nano_workflow,
            structure_filename,
            style_filename,
            prompt,
            seed,
        )

        return await self.client.generate(workflow, verbose=self.config.verbose)

    async def _generate_iterative_via_spec(
        self,
        spec,
        style_image_path: Path,
        structure_image_path: Path,
        prompt: str,
        seed: int,
    ) -> List[Image.Image]:
        """Generate image using a model spec from MODEL_REGISTRY.

        Mirrors _generate_nano_iterative: same upload order (structure, then style).
        The spec's prepare_fn injects filenames into the workflow's model-specific node IDs.
        """
        structure_filename = await self._upload_image(structure_image_path)
        style_filename = await self._upload_image(style_image_path)

        if self.config.verbose:
            self.log(f"    [DEBUG] structure (3D rendering): {structure_image_path.name} -> {structure_filename}")
            self.log(f"    [DEBUG] style (photo reference):  {style_image_path.name} -> {style_filename}")

        workflow = spec.prepare_fn(
            self.model_workflow,
            style_filename,
            structure_filename,
            prompt,
            seed,
        )

        return await self.client.generate(workflow, verbose=self.config.verbose)

    async def _process_image_qwen(
        self,
        image_id: int,
        input_path: Path,
        style_reference_path: Path,
        extra_prompt: Optional[str] = None,
        use_nano: bool = True,
        is_bootstrap: bool = False,
    ) -> bool:
        """
        Process a single image using Flux2-Klein Base iterative workflow with retry logic.

        Args:
            image_id: The image ID (e.g., 0, 1, 2, ...)
            input_path: Path to the structure image (depth+objects)
            style_reference_path: Path to the style reference image (generated_0 or bootstrap)
            extra_prompt: Optional extra instruction appended to the prompt
            use_nano: When True and api mode is active, use Nano Banana Pro workflow instead of qwen
            is_bootstrap: When True, use bootstrap_max_retries and pick best attempt on failure

        Returns:
            True if successful, False if all attempts failed.
        """
        # Determine whether to use a model-registry workflow for this call.
        # When --api is on, dispatch via MODEL_REGISTRY[config.model] (covers nano/api, flux2-klein-9b-distilled, etc.).
        model_spec = MODEL_REGISTRY[self.config.model]
        use_model_workflow = use_nano and self.config.api and self.model_workflow is not None

        # Track attempts for this image
        attempts: List[AttemptRecord] = []

        # Extract reference image ID for visualization filenames
        ref_id = self._extract_image_id(style_reference_path)
        if ref_id < 0:
            ref_id = None  # Bootstrap or unknown — omit from filename

        # Build iterative prompt
        camera_name = self._extract_camera_name(input_path)
        if use_model_workflow:
            iterative_prompt = self._build_nano_prompt(camera_name)
            self.log(f"  Using model={self.config.model!r} workflow ({model_spec.workflow_filename})")
        else:
            iterative_prompt = self._build_qwen_prompt(camera_name)
        if extra_prompt:
            iterative_prompt = f"{iterative_prompt} {extra_prompt}"
        self.log(f"  Structure input (ref_image1): {input_path.name}")
        self.log(f"  Style reference (ref_image2): {style_reference_path.name}")
        self.log(f"  Iterative prompt: {iterative_prompt}")

        max_attempts = self.config.bootstrap_max_retries if is_bootstrap else self.config.max_retries
        for attempt in range(1, max_attempts + 1):
            seed = random.randint(0, 2**31 - 1)
            self.log(f"  Attempt {attempt}/{max_attempts} (seed={seed})")

            try:
                # Generate using appropriate workflow
                if use_model_workflow:
                    images = await self._generate_iterative_via_spec(
                        model_spec,
                        style_reference_path,
                        input_path,
                        iterative_prompt,
                        seed,
                    )
                else:
                    images = await self._generate_qwen_iterative(
                        style_reference_path,
                        input_path,
                        iterative_prompt,
                        seed,
                    )

                if not images:
                    self.log("    No images generated")
                    continue

                generated_image = images[0]

                # Validation
                if self.config.no_validation:
                    # Skip all validation, accept first generation
                    edge_passed = True
                    iou_score = 0.0
                    depth_passed = True
                    depth_edge_recall = 0.0
                    depth_comparison = None
                    self.log("    Validation skipped (--no-validation)")
                else:
                    # Edge validation (if enabled)
                    edge_passed = True
                    iou_score = 0.0
                    if self.config.use_edge_validation:
                        edge_passed, iou_score = validate_generation(
                            generated_image,
                            input_path,
                            threshold=self.config.iou_threshold,
                            canny_low=self.config.canny_low,
                            canny_high=self.config.canny_high,
                            dilation_pixels=self.config.dilation_pixels,
                            exclude_floor=False,
                        )
                        status = "PASS" if edge_passed else "FAIL"
                        self.log(f"    Edge IoU: {iou_score:.4f} (threshold: {self.config.iou_threshold}) - {status}")

                    # Depth validation (if enabled)
                    depth_passed = True
                    depth_edge_recall = 0.0
                    depth_comparison = None
                    if self.config.use_depth_validation:
                        depth_passed, depth_edge_recall, depth_comparison = validate_depth(
                            generated_image,
                            input_path,
                            threshold=self.config.depth_threshold,
                            conda_env=self.config.depth_conda_env,
                            verbose=self.config.verbose,
                            canny_low=self.config.depth_canny_low,
                            canny_high=self.config.depth_canny_high,
                            dilation_pixels=self.config.depth_dilation_pixels,
                            sharpen_estimated=self.config.depth_sharpen,
                            sharpen_gt=self.config.depth_sharpen_gt,
                            min_gradient_percentile=self.config.depth_min_gradient_percentile,
                        )
                        status = "PASS" if depth_passed else "FAIL"
                        self.log(f"    Depth Edge Recall: {depth_edge_recall:.4f} (threshold: {self.config.depth_threshold}) - {status}")

                # Overall pass requires enabled validations to pass
                passed = edge_passed and depth_passed

                # Save the attempt (all attempts are saved to failed/ initially)
                failed_path = self._get_failed_path(image_id, attempt, seed, ref_id=ref_id)
                generated_image.save(failed_path)

                # Create edge comparison image (always, for debugging)
                comparison = create_comparison_image(
                    generated_image,
                    input_path,
                    self.config.canny_low,
                    self.config.canny_high,
                )
                comparison_path = self._get_comparison_path(image_id, attempt, ref_id=ref_id)
                comparison.save(comparison_path)

                # Save depth comparison image if available
                if depth_comparison is not None:
                    depth_comparison_path = self._get_depth_comparison_path(image_id, attempt, ref_id=ref_id)
                    depth_comparison.save(depth_comparison_path)

                record = AttemptRecord(
                    attempt=attempt,
                    seed=seed,
                    ssim=iou_score,
                    image_path=str(failed_path),
                    success=passed,
                    depth_edge_recall=depth_edge_recall,
                )
                attempts.append(record)

                if passed:
                    if is_bootstrap:
                        # Bootstrap: collect min_bootstrap_attempts before selecting best
                        min_attempts = 1 if self.config.no_validation else self.config.min_bootstrap_attempts
                        if attempt >= min_attempts:
                            passing = [a for a in attempts if a.success]
                            if passing:
                                best = max(passing, key=lambda a: a.depth_edge_recall)
                                self.log(f"    Selected best passing attempt: attempt {best.attempt} (Edge Recall: {best.depth_edge_recall:.4f})")
                                gen_path = self._get_generated_path(image_id, ref_id=ref_id)
                                shutil.copy(best.image_path, gen_path)
                                self.progress.completed_ids.append(image_id)
                                self.progress.successful_generations[image_id] = str(gen_path)
                                if image_id not in self.progress.failed_ids:
                                    self.progress.failed_ids[image_id] = []
                                self.progress.failed_ids[image_id].extend(attempts)
                                self.progress.save(self.config.progress_path)
                                return True
                        else:
                            self.log(f"    PASS (collecting min {min_attempts} bootstrap attempts, have {attempt})")
                    else:
                        # Non-bootstrap: accept first passing attempt
                        gen_path = self._get_generated_path(image_id, ref_id=ref_id)
                        generated_image.save(gen_path)

                        self.progress.completed_ids.append(image_id)
                        self.progress.successful_generations[image_id] = str(gen_path)
                        self.log(f"    SUCCESS - saved to {gen_path}")

                        # Save all attempts to failed_ids for tracking
                        if image_id not in self.progress.failed_ids:
                            self.progress.failed_ids[image_id] = []
                        self.progress.failed_ids[image_id].extend(attempts)

                        self.progress.save(self.config.progress_path)
                        return True

            except TimeoutError as e:
                print(f"    [TIMEOUT] {e}")
                self.log(f"    Timeout: {e}")
                await self._recover_comfyui()  # Raises ComfyUIUnrecoverableError if fatal
                record = AttemptRecord(
                    attempt=attempt,
                    seed=seed,
                    ssim=0.0,
                    image_path="",
                    success=False,
                )
                attempts.append(record)
                backoff = min(2 ** attempt, 60) + random.uniform(0, 2)
                self.log(f"    Waiting {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

            except Exception as e:
                print(f"    [ERROR] {e}")
                self.log(f"    Error: {e}")
                record = AttemptRecord(
                    attempt=attempt,
                    seed=seed,
                    ssim=0.0,
                    image_path="",
                    success=False,
                )
                attempts.append(record)
                backoff = min(2 ** attempt, 60) + random.uniform(0, 2)
                self.log(f"    Waiting {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

        # All attempts failed
        self.progress.failed_ids[image_id] = attempts
        self.progress.save(self.config.progress_path)

        # Bootstrap cameras: pick best attempt instead of giving up
        if is_bootstrap:
            saved_attempts = [a for a in attempts if a.image_path]
            if saved_attempts:
                best = max(saved_attempts, key=lambda a: a.depth_edge_recall)
                self.log(f"    No passing attempt found. Using best attempt {best.attempt} (Edge Recall: {best.depth_edge_recall:.4f})")
                gen_path = self._get_generated_path(image_id, ref_id=ref_id)
                shutil.copy(best.image_path, gen_path)
                self.progress.completed_ids.append(image_id)
                self.progress.successful_generations[image_id] = str(gen_path)
                self.progress.save(self.config.progress_path)
                return True

        return False

    async def _generate_bootstrap(
        self,
        input_path: Path,
        image_id: int,
    ) -> Optional[Path]:
        """
        Generate a bootstrap camera image using Flux 2 Klein.

        Bootstrap cameras use the single-image workflow (no reference image).
        Their output is used as style reference for subsequent iterative generations.

        Strategy:
        1. Generate at least min_bootstrap_attempts images
        2. Track all attempts with their depth scores
        3. After min attempts, if any pass threshold -> pick the best passing one
        4. If none pass -> keep trying until bootstrap_max_retries
        5. If still none pass -> pick the best one anyway

        Args:
            input_path: Path to the input image (depth+objects for camera 0)
            image_id: The image ID (should be 0)

        Returns:
            Path to saved generated image if successful, None otherwise.
        """
        self.log(f"Generating camera {image_id} (Flux bootstrap)")

        # Build simplified Flux prompt (room name + theme only, no openings)
        room_name = self.config.room_id.replace('_', ' ').title()
        if self.config.theme:
            flux_prompt = f"{room_name}. {self.config.theme.title()}. Photorealistic."
        else:
            flux_prompt = f"{room_name}. Photorealistic."
        self.log(f"  Flux prompt: {flux_prompt[:80]}...")

        # Track all attempts: (attempt, seed, depth_edge_recall, passed, image_path)
        bootstrap_attempts: List[tuple] = []
        min_bootstrap_attempts = 1 if self.config.no_validation else self.config.min_bootstrap_attempts

        for attempt in range(1, self.config.bootstrap_max_retries + 1):
            flux_seed = random.randint(0, 2**31 - 1)
            self.log(f"  Attempt {attempt}/{self.config.bootstrap_max_retries} (seed={flux_seed})")

            try:
                # --- Step 1: Flux 2 Klein (rough style/color baseline) ---
                input_filename = await self._upload_image(input_path)
                # Use dedicated bootstrap workflow if available, otherwise fall back to initial workflow
                bootstrap_wf = self.bootstrap_workflow or self.initial_workflow
                workflow = prepare_initial_workflow(
                    bootstrap_wf,
                    image_filename=input_filename,
                    prompt=flux_prompt,
                    seed=flux_seed,
                )
                flux_images = await self.client.generate(workflow)

                if not flux_images:
                    self.log("    Step 1 (Flux): No images generated")
                    continue

                flux_image = flux_images[0]
                generated_image = flux_image

                # Validation
                if self.config.no_validation:
                    edge_passed = True
                    iou_score = 0.0
                    depth_passed = True
                    depth_edge_recall = 0.0
                    depth_comparison = None
                    self.log("    Validation skipped (--no-validation)")
                else:
                    # Edge validation (if enabled)
                    edge_passed = True
                    iou_score = 0.0
                    if self.config.use_edge_validation:
                        edge_passed, iou_score = validate_generation(
                            generated_image,
                            input_path,
                            threshold=self.config.iou_threshold,
                            canny_low=self.config.canny_low,
                            canny_high=self.config.canny_high,
                            dilation_pixels=self.config.dilation_pixels,
                            exclude_floor=False,
                        )
                        status = "PASS" if edge_passed else "FAIL"
                        self.log(f"    Edge IoU: {iou_score:.4f} - {status}")

                    # Depth validation (if enabled)
                    depth_passed = True
                    depth_edge_recall = 0.0
                    depth_comparison = None
                    if self.config.use_depth_validation:
                        depth_passed, depth_edge_recall, depth_comparison = validate_depth(
                            generated_image,
                            input_path,
                            threshold=self.config.depth_threshold,
                            conda_env=self.config.depth_conda_env,
                            verbose=self.config.verbose,
                            canny_low=self.config.depth_canny_low,
                            canny_high=self.config.depth_canny_high,
                            dilation_pixels=self.config.depth_dilation_pixels,
                            sharpen_estimated=self.config.depth_sharpen,
                            sharpen_gt=self.config.depth_sharpen_gt,
                            min_gradient_percentile=self.config.depth_min_gradient_percentile,
                        )
                        status = "PASS" if depth_passed else "FAIL"
                        self.log(f"    Depth Edge Recall: {depth_edge_recall:.4f} - {status}")

                # Overall pass requires enabled validations to pass
                passed = edge_passed and depth_passed

                # Save attempt to failed/ for debugging
                failed_path = self.config.failed_dir / f"initial_{image_id:04d}_attempt_{attempt}_seed_{flux_seed}.png"
                generated_image.save(failed_path)

                # Create edge comparison image (always, for debugging)
                comparison = create_comparison_image(
                    generated_image,
                    input_path,
                    self.config.canny_low,
                    self.config.canny_high,
                )
                comparison_path = self.config.comparison_dir / f"initial_{image_id:04d}_attempt_{attempt}_comparison.png"
                comparison.save(comparison_path)

                # Save depth comparison image if available
                if depth_comparison is not None:
                    depth_comparison_path = self.config.depth_comparison_dir / f"initial_{image_id:04d}_attempt_{attempt}_depth_comparison.png"
                    depth_comparison.save(depth_comparison_path)

                # Track this attempt
                bootstrap_attempts.append((attempt, flux_seed, depth_edge_recall, passed, str(failed_path)))

                # Check if we should stop and pick the best
                if attempt >= min_bootstrap_attempts:
                    # Check if any attempt passed
                    passing_attempts = [a for a in bootstrap_attempts if a[3]]  # a[3] = passed
                    if passing_attempts:
                        # Pick the best passing attempt (highest depth edge recall)
                        best = max(passing_attempts, key=lambda a: a[2])  # a[2] = depth_edge_recall
                        best_attempt, best_seed, best_score, _, best_path = best
                        self.log(f"    Selected best passing attempt: attempt {best_attempt} (Edge Recall: {best_score:.4f})")

                        # Save as generated_0000.png (final output for camera 0)
                        gen_path = self._get_generated_path(image_id)
                        shutil.copy(best_path, gen_path)
                        self.progress.completed_ids.append(image_id)
                        self.progress.successful_generations[image_id] = str(gen_path)
                        self.progress.save(self.config.progress_path)
                        self.log(f"    Saved to {gen_path}")
                        return gen_path

            except TimeoutError as e:
                self.log(f"    Timeout: {e}")
                await self._recover_comfyui()  # Raises ComfyUIUnrecoverableError if fatal
                backoff = min(2 ** attempt, 60) + random.uniform(0, 2)
                self.log(f"    Waiting {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

            except Exception as e:
                self.log(f"    Error: {e}")
                backoff = min(2 ** attempt, 60) + random.uniform(0, 2)
                self.log(f"    Waiting {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

        # No passing attempts after all retries - pick the best one anyway
        if bootstrap_attempts:
            # Pick attempt with highest depth edge recall (best depth match)
            best = max(bootstrap_attempts, key=lambda a: a[2])  # a[2] = depth_edge_recall
            best_attempt, best_seed, best_score, _, best_path = best
            self.log(f"    No passing attempt found. Using best attempt {best_attempt} (Edge Recall: {best_score:.4f})")

            gen_path = self._get_generated_path(image_id)
            shutil.copy(best_path, gen_path)
            self.progress.completed_ids.append(image_id)
            self.progress.successful_generations[image_id] = str(gen_path)
            self.progress.save(self.config.progress_path)
            self.log(f"    Saved to {gen_path}")
            return gen_path

        self.log("    Camera 0 generation failed after all attempts")
        return None

    async def _process_image(
        self,
        image_id: int,
        input_path: Path,
        use_iterative: bool,
        reference_id: Optional[int] = None,
    ) -> bool:
        """
        Process a single image with retry logic.

        Returns True if successful, False if all attempts failed.
        """
        # Track attempts for this image
        attempts: List[AttemptRecord] = []

        # Get per-camera prompt if available
        image_prompt = self._get_prompt_for_image(input_path)

        for attempt in range(1, self.config.max_retries + 1):
            seed = random.randint(0, 2**31 - 1)
            self.log(f"  Attempt {attempt}/{self.config.max_retries} (seed={seed})")

            try:
                # Upload input image
                input_filename = await self._upload_image(input_path)

                # Generate
                if use_iterative and reference_id is not None:
                    # Upload reference image (previously generated)
                    ref_path = self.progress.successful_generations[reference_id]
                    ref_filename = await self._upload_image(Path(ref_path))
                    images = await self._generate_iterative(
                        input_filename, ref_filename, seed, prompt=image_prompt
                    )
                else:
                    images = await self._generate_initial(input_filename, seed, prompt=image_prompt)

                if not images:
                    self.log("    No images generated")
                    continue

                generated_image = images[0]

                # Validation
                if self.config.no_validation:
                    edge_passed = True
                    iou_score = 0.0
                    depth_passed = True
                    depth_edge_recall = 0.0
                    depth_comparison = None
                    self.log("    Validation skipped (--no-validation)")
                else:
                    # Edge validation (if enabled)
                    edge_passed = True
                    iou_score = 0.0
                    if self.config.use_edge_validation:
                        edge_passed, iou_score = validate_generation(
                            generated_image,
                            input_path,  # Compare to input image, not pre-computed edges
                            threshold=self.config.iou_threshold,
                            canny_low=self.config.canny_low,
                            canny_high=self.config.canny_high,
                            dilation_pixels=self.config.dilation_pixels,
                            exclude_floor=False,
                        )
                        status = "PASS" if edge_passed else "FAIL"
                        self.log(f"    Edge IoU: {iou_score:.4f} (threshold: {self.config.iou_threshold}) - {status}")

                    # Depth validation (if enabled)
                    depth_passed = True
                    depth_edge_recall = 0.0
                    depth_comparison = None
                    if self.config.use_depth_validation:
                        depth_passed, depth_edge_recall, depth_comparison = validate_depth(
                            generated_image,
                            input_path,
                            threshold=self.config.depth_threshold,
                            conda_env=self.config.depth_conda_env,
                            verbose=self.config.verbose,
                            canny_low=self.config.depth_canny_low,
                            canny_high=self.config.depth_canny_high,
                            dilation_pixels=self.config.depth_dilation_pixels,
                            sharpen_estimated=self.config.depth_sharpen,
                            sharpen_gt=self.config.depth_sharpen_gt,
                            min_gradient_percentile=self.config.depth_min_gradient_percentile,
                        )
                        status = "PASS" if depth_passed else "FAIL"
                        self.log(f"    Depth Edge Recall: {depth_edge_recall:.4f} (threshold: {self.config.depth_threshold}) - {status}")

                # Overall pass requires enabled validations to pass
                passed = edge_passed and depth_passed

                # Save the attempt (all attempts are saved to failed/ initially)
                failed_path = self._get_failed_path(image_id, attempt, seed)
                generated_image.save(failed_path)

                # Create edge comparison image (always, for debugging)
                comparison = create_comparison_image(
                    generated_image,
                    input_path,  # Compare to input image
                    self.config.canny_low,
                    self.config.canny_high,
                )
                comparison_path = self._get_comparison_path(image_id, attempt)
                comparison.save(comparison_path)

                # Save depth comparison image if available
                if depth_comparison is not None:
                    depth_comparison_path = self._get_depth_comparison_path(image_id, attempt)
                    depth_comparison.save(depth_comparison_path)

                record = AttemptRecord(
                    attempt=attempt,
                    seed=seed,
                    ssim=iou_score,  # Now stores IoU instead of SSIM
                    image_path=str(failed_path),
                    success=passed,
                    depth_edge_recall=depth_edge_recall,
                )
                attempts.append(record)

                if passed:
                    # Success! Move to generated folder
                    gen_path = self._get_generated_path(image_id)
                    generated_image.save(gen_path)

                    self.progress.completed_ids.append(image_id)
                    self.progress.successful_generations[image_id] = str(gen_path)
                    self.log(f"    SUCCESS - saved to {gen_path}")

                    # Save all attempts (including successful ones) to failed_ids for tracking
                    if image_id not in self.progress.failed_ids:
                        self.progress.failed_ids[image_id] = []
                    self.progress.failed_ids[image_id].extend(attempts)

                    self.progress.save(self.config.progress_path)
                    return True

            except TimeoutError as e:
                self.log(f"    Timeout: {e}")
                await self._recover_comfyui()  # Raises ComfyUIUnrecoverableError if fatal
                record = AttemptRecord(
                    attempt=attempt,
                    seed=seed,
                    ssim=0.0,
                    image_path="",
                    success=False,
                )
                attempts.append(record)
                backoff = min(2 ** attempt, 60) + random.uniform(0, 2)
                self.log(f"    Waiting {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

            except Exception as e:
                self.log(f"    Error: {e}")
                record = AttemptRecord(
                    attempt=attempt,
                    seed=seed,
                    ssim=0.0,
                    image_path="",
                    success=False,
                )
                attempts.append(record)
                backoff = min(2 ** attempt, 60) + random.uniform(0, 2)
                self.log(f"    Waiting {backoff:.1f}s before retry")
                await asyncio.sleep(backoff)

        # All attempts failed
        self.progress.failed_ids[image_id] = attempts
        self.progress.save(self.config.progress_path)
        return False

    async def run(self) -> None:
        """Run the full generation pipeline."""
        # Validate config
        self.config.validate()
        self.config.ensure_output_dirs()

        # Load progress if resuming
        if self.config.resume and self.config.progress_path.exists():
            self.progress = Progress.load(self.config.progress_path)
            self.log(f"Resumed from progress: {len(self.progress.completed_ids)} completed")
        else:
            self.progress = Progress()

        # Load workflows
        self.initial_workflow = load_workflow(self.config.initial_workflow_path)
        self.iterative_workflow = load_workflow(self.config.iterative_workflow_path)

        # Load Qwen workflow if available
        if self.config.qwen_workflow_path and self.config.qwen_workflow_path.exists():
            self.qwen_workflow = load_workflow(self.config.qwen_workflow_path)
            self.log(f"Loaded iterative workflow from {self.config.qwen_workflow_path}")
        else:
            self.qwen_workflow = None

        # Load Nano Banana Pro workflow (kept for legacy callers / debug; the
        # model-registry workflow below is what _process_image_qwen actually uses).
        if self.config.api and self.config.nano_workflow_path and self.config.nano_workflow_path.exists():
            self.nano_workflow = load_workflow(self.config.nano_workflow_path)
            self.log(f"Loaded nano workflow from {self.config.nano_workflow_path}")
        else:
            self.nano_workflow = None

        # Load model workflow (selected via --model). For model="api" this points to
        # the same file as nano_workflow_path; for other models it points to that
        # model's JSON.
        if self.config.api and self.config.model_workflow_path and self.config.model_workflow_path.exists():
            self.model_workflow = load_workflow(self.config.model_workflow_path)
            self.log(f"Loaded model={self.config.model!r} workflow from {self.config.model_workflow_path}")
        else:
            self.model_workflow = None

        if not self.model_workflow:
            self.log("No API iterative workflow found, using legacy flow")

        # Load bootstrap workflow if available
        if self.config.bootstrap_workflow_path and self.config.bootstrap_workflow_path.exists():
            self.bootstrap_workflow = load_workflow(self.config.bootstrap_workflow_path)
            self.log(f"Loaded bootstrap workflow from {self.config.bootstrap_workflow_path}")
        else:
            self.bootstrap_workflow = None
            self.log("No bootstrap workflow found, will use initial workflow for bootstrap")

        # Load scene JSON if available
        if self.config.scene_json_path and self.config.scene_json_path.exists():
            with open(self.config.scene_json_path) as f:
                self.scene_json = json.load(f)
            self.log(f"Loaded scene JSON from {self.config.scene_json_path}")
        else:
            self.scene_json = None
            self.log("No scene JSON provided, window descriptions will be empty")

        # Get prompt (from config or workflow)
        self.prompt = self.config.prompt
        if self.prompt is None:
            self.prompt = get_prompt_from_workflow(self.initial_workflow)
        self.log(f"Using prompt: {self.prompt[:100]}..." if self.prompt else "No prompt")

        # Load per-camera prompts if specified
        if self.config.prompts_file and self.config.prompts_file.exists():
            with open(self.config.prompts_file) as f:
                self.camera_prompts = json.load(f)
            self.log(f"Loaded {len(self.camera_prompts)} per-camera prompts")

        # Load camera poses
        room_name = self.config.input_dir.name
        self.poses = get_poses_for_room(self.config.images_txt_path, room_name)
        self.log(f"Loaded {len(self.poses)} camera poses for {room_name}")

        # Get input images
        input_images = self._get_input_images()
        self.log(f"Found {len(input_images)} input images")

        if not input_images:
            self.log("No input images found!")
            return

        # Start ComfyUI server if auto-start is enabled
        if self.config.auto_start_server:
            self.server = ComfyUIServer(
                host=self.config.comfyui_host,
                port=self.config.comfyui_port,
                comfyui_path=self.config.comfyui_path,
            )
            self.log("Starting ComfyUI server...")
            await self.server.start(self.config.startup_timeout)
            self.log("ComfyUI server ready")

        try:
            # Connect to ComfyUI
            async with ComfyUIClient(
                self.config.comfyui_host, self.config.comfyui_port,
                api_key=self.config.comfy_api_key,
            ) as client:
                self.client = client

                # Check connection
                if not await client.check_connection():
                    raise RuntimeError(
                        f"Could not connect to ComfyUI at "
                        f"{self.config.comfyui_host}:{self.config.comfyui_port}"
                    )
                self.log("Connected to ComfyUI")

                # Process images using appropriate flow
                if self.qwen_workflow:
                    # === QWEN FLOW ===
                    # Step 1: Generate bootstrap cameras with initial workflow (no reference image)
                    # Step 2: Generate all other cameras with Qwen iterative workflow (bootstrap as style reference)

                    self.log("Using Flux2-Klein iterative flow")

                    # Find inputs for all bootstrap cameras
                    bootstrap_camera_ids = self.config.bootstrap_cameras or [0]
                    bootstrap_inputs = {}
                    for bid in bootstrap_camera_ids:
                        for input_path in input_images:
                            if self._extract_image_id(input_path) == bid:
                                bootstrap_inputs[bid] = input_path
                                break
                        if bid not in bootstrap_inputs:
                            self.log(f"ERROR: No input found for bootstrap camera {bid}")
                            return

                    # Handle --only-camera: generate a single non-bootstrap camera using iterative workflow
                    if self.config.only_camera is not None:
                        target_id = self.config.only_camera
                        # Ensure bootstrap cameras are in progress
                        for bid in bootstrap_camera_ids:
                            gen_path = self._find_generated(bid)
                            if gen_path is not None and bid not in self.progress.completed_ids:
                                self.progress.completed_ids.append(bid)
                                self.progress.successful_generations[bid] = str(gen_path)
                        # Find target input
                        target_input = None
                        for p in input_images:
                            if self._extract_image_id(p) == target_id:
                                target_input = p
                                break
                        if target_input is None:
                            self.log(f"ERROR: No input for camera {target_id}")
                            return
                        style_ref, ref_id = self._select_style_reference(target_id)
                        if style_ref is None:
                            self.log("ERROR: No bootstrap reference available")
                            return
                        self.log(f"Only-camera mode: generating camera {target_id} (ref: {ref_id})")
                        if target_id not in self.progress.completed_ids:
                            await self._process_image_qwen(target_id, target_input, style_ref,
                                                           extra_prompt="The camera faces the opposite direction from reference_image 2. Do not add objects not present in reference_image 1. Do not copy furniture from reference_image 2.")
                        self.progress.save(self.config.progress_path)
                        return

                    # Handle --skip-bootstrap: assume bootstrap cameras already exist
                    if self.config.skip_bootstrap:
                        for bid in bootstrap_camera_ids:
                            gen_path = self._find_generated(bid)
                            if gen_path is None:
                                self.log(f"ERROR: --skip-bootstrap but no generated image for camera {bid}")
                                return
                            if bid not in self.progress.completed_ids:
                                self.progress.completed_ids.append(bid)
                                self.progress.successful_generations[bid] = str(gen_path)
                                self.progress.save(self.config.progress_path)
                            self.log(f"Skipping bootstrap camera {bid}")
                    else:
                        # Generate all bootstrap cameras
                        # First bootstrap: single-image workflow (no reference)
                        # Subsequent bootstraps: iterative workflow using first bootstrap as style ref
                        first_bootstrap_path = None
                        for bid in bootstrap_camera_ids:
                            if bid in self.progress.completed_ids:
                                self.log(f"Skipping bootstrap camera {bid} (already completed)")
                                if first_bootstrap_path is None:
                                    first_bootstrap_path = Path(self.progress.successful_generations[bid])
                                continue

                            if first_bootstrap_path is None:
                                # First bootstrap: single-image workflow (no reference)
                                self.log(f"Generating bootstrap camera {bid} (Flux2-Klein 9B, initial)")
                                gen_path = await self._generate_bootstrap(bootstrap_inputs[bid], bid)
                                if gen_path is None:
                                    self.log(f"ERROR: Bootstrap camera {bid} generation failed")
                                    return
                                first_bootstrap_path = gen_path
                            else:
                                # Subsequent bootstraps: iterative workflow with first bootstrap as style ref
                                self.log(f"Generating bootstrap camera {bid} (iterative, style ref from camera {bootstrap_camera_ids[0]})")
                                success = await self._process_image_qwen(
                                    image_id=bid,
                                    input_path=bootstrap_inputs[bid],
                                    style_reference_path=first_bootstrap_path,
                                    extra_prompt="Do not add furniture not present in the rendering image!",
                                    is_bootstrap=True,
                                )
                                if not success:
                                    self.log(f"ERROR: Bootstrap camera {bid} generation failed")
                                    return

                    # If --only-bootstrap, exit after all bootstraps
                    if self.config.only_bootstrap:
                        self.log("Bootstrap complete (--only-bootstrap), exiting")
                        return

                    # Build set of bootstrap camera IDs to skip in iterative loop
                    bootstrap_set = set(self.config.bootstrap_cameras or [0])

                    # Step 2: Generate remaining cameras, ordered by rotational proximity
                    # Overhead cameras are deferred to the end (fundamentally different viewpoint)
                    overhead_ids = self._get_overhead_camera_ids()

                    # Collect non-bootstrap, non-overhead candidate IDs
                    candidate_ids = []
                    overhead_list = []
                    for p in input_images:
                        cid = self._extract_image_id(p)
                        if cid < 0 or cid in bootstrap_set:
                            continue
                        if cid in overhead_ids:
                            overhead_list.append(cid)
                        else:
                            candidate_ids.append(cid)

                    # Sort by greedy rotational proximity from bootstrap cameras
                    sorted_ids = order_by_rotational_proximity(
                        list(bootstrap_set), candidate_ids, self.poses
                    ) + overhead_list

                    # Build ordered input path list
                    id_to_path = {self._extract_image_id(p): p for p in input_images}
                    ordered_images = [id_to_path[cid] for cid in sorted_ids if cid in id_to_path]

                    if overhead_ids:
                        self.log(f"Step 2: Generating remaining cameras by rotational proximity (overhead {sorted(overhead_ids)} deferred to end)")
                    else:
                        self.log(f"Step 2: Generating remaining cameras by rotational proximity")
                    for input_path in ordered_images:
                        image_id = self._extract_image_id(input_path)
                        if image_id < 0:
                            self.log(f"Could not extract ID from: {input_path.name}")
                            continue

                        if image_id in bootstrap_set:
                            continue  # Already generated as bootstrap

                        # Skip already completed
                        if image_id in self.progress.completed_ids:
                            self.log(f"Skipping {image_id:04d} (already completed)")
                            continue

                        # Skip if already exhausted max retries (2x for initial + fallback rounds)
                        previous_attempts = len(self.progress.failed_ids.get(image_id, []))
                        if previous_attempts >= 2 * self.config.max_retries:
                            self.log(f"Skipping {image_id:04d} (already failed {previous_attempts} times)")
                            continue

                        # Select rotationally-closest bootstrap as style reference
                        style_ref, ref_id = self._select_style_reference(image_id)
                        if style_ref is None:
                            self.log(f"ERROR: No bootstrap reference available for camera {image_id:04d}")
                            continue

                        self.log(f"Processing camera {image_id:04d} (style ref: camera {ref_id:04d})")
                        success = await self._process_image_qwen(
                            image_id=image_id,
                            input_path=input_path,
                            style_reference_path=style_ref,
                        )

                        if not success:
                            # Fallback: retry with the most rotationally-similar successful generation
                            # Exclude the bootstrap ref we already tried
                            successful_ids = [
                                sid for sid in self.progress.successful_generations.keys()
                                if sid != ref_id
                            ]
                            fallback_id = find_best_reference(image_id, successful_ids, self.poses)

                            if fallback_id is not None:
                                fallback_path = Path(self.progress.successful_generations[fallback_id])
                                self.log(f"  Retrying with rotationally-similar reference {fallback_id:04d}")
                                # Save first-round attempts before _process_image_qwen overwrites them
                                first_round_attempts = list(self.progress.failed_ids.get(image_id, []))

                                success = await self._process_image_qwen(
                                    image_id=image_id,
                                    input_path=input_path,
                                    style_reference_path=fallback_path,
                                )

                                # Merge first-round attempts back (line 587 replaces, not appends)
                                if image_id in self.progress.failed_ids:
                                    self.progress.failed_ids[image_id] = first_round_attempts + self.progress.failed_ids[image_id]
                                self.progress.save(self.config.progress_path)

                            if not success:
                                self.log(f"  FAILED after all attempts (including fallback)")

                else:
                    # === LEGACY FLOW ===
                    # Use initial workflow for camera 0, iterative for others
                    self.log("Using legacy workflow flow")

                    for input_path in input_images:
                        image_id = self._extract_image_id(input_path)
                        if image_id < 0:
                            self.log(f"Could not extract ID from: {input_path.name}")
                            continue

                        # Skip already completed
                        if image_id in self.progress.completed_ids:
                            self.log(f"Skipping {image_id:04d} (already completed)")
                            continue

                        # Skip if already exhausted max retries in previous runs
                        previous_attempts = len(self.progress.failed_ids.get(image_id, []))
                        if previous_attempts >= self.config.max_retries:
                            self.log(f"Skipping {image_id:04d} (already failed {previous_attempts} times)")
                            continue

                        self.log(f"Processing image {image_id:04d}: {input_path.name}")

                        # Determine if we should use initial or iterative workflow
                        successful_ids = list(self.progress.successful_generations.keys())

                        if image_id == 0 or not successful_ids:
                            # Use initial workflow (single reference)
                            self.log("  Using initial workflow (single-image)")
                            use_iterative = False
                            reference_id = None
                        else:
                            # Find best reference and use iterative workflow
                            reference_id = find_best_reference(
                                image_id, successful_ids, self.poses
                            )
                            if reference_id is not None:
                                self.log(f"  Using iterative workflow (reference: {reference_id:04d})")
                                use_iterative = True
                            else:
                                self.log("  Using initial workflow (no suitable reference)")
                                use_iterative = False

                        success = await self._process_image(
                            image_id,
                            input_path,
                            use_iterative,
                            reference_id,
                        )

                        if not success:
                            self.log(f"  FAILED after {self.config.max_retries} attempts")
        finally:
            if self.server:
                self.log("Shutting down ComfyUI server...")
                self.server.shutdown()

        # Final summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print final summary of results."""
        total = len(self._get_input_images())
        completed = len(self.progress.completed_ids)
        failed = len([
            k for k, v in self.progress.failed_ids.items()
            if k not in self.progress.completed_ids
        ])

        print("\n" + "=" * 50)
        print("Generation Complete")
        print("=" * 50)
        print(f"Total images: {total}")
        print(f"Successful:   {completed}")
        print(f"Failed:       {failed}")
        print(f"Output dir:   {self.config.output_dir}")
        print("=" * 50)


async def run_pipeline(config: Config) -> None:
    """Run the Flux generation pipeline with the given config."""
    pipeline = FluxGenerationPipeline(config)
    await pipeline.run()
