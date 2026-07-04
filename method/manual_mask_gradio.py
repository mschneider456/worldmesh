#!/usr/bin/env python3
"""
Manual Mask Creation with Gradio UI

Interactive point-based mask creation using SAM3's interactive predictor.
Users can click on images to add positive/negative points and create precise object masks.

Usage:
    # Standalone usage
    conda activate worldmesh-sam3
    python manual_mask_gradio.py \
        --input-dir output/my_scene/initial_flux \
        --output-dir output/my_scene/extracted_objects \
        --rooms living_room kitchen \
        --port 7860

    # The UI will open in browser at http://localhost:7860
"""

import argparse
import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from checkpoint_requirements import (
    find_missing_requirements,
    format_missing_checkpoints_error,
    sam3_requirement,
)

# Gradio import (must be installed: pip install gradio)
try:
    import gradio as gr
except ImportError:
    print("Error: gradio not installed. Run: pip install gradio")
    sys.exit(1)


@dataclass
class RoomEntry:
    """A room with its per-scene input/output directories."""
    room_id: str
    input_dir: Path   # scene-specific initial_flux dir
    output_dir: Path  # scene-specific extracted_objects dir
    scene_name: str = ""


@dataclass
class Point:
    """A point with coordinates and label (positive=1, negative=0)."""
    x: int
    y: int
    label: int  # 1 = positive (foreground), 0 = negative (background)


@dataclass
class MaskData:
    """Data for a confirmed mask."""
    mask: np.ndarray
    label: str
    score: float
    points: List[Point]


@dataclass
class SessionState:
    """State for the current UI session."""
    rooms: List[str] = field(default_factory=list)
    current_room_idx: int = 0
    current_image: Optional[np.ndarray] = None
    current_image_path: Optional[Path] = None

    # Current mask being created
    current_points: List[Point] = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    current_logits: Optional[np.ndarray] = None

    # Confirmed masks for current room
    confirmed_masks: List[MaskData] = field(default_factory=list)

    def reset_current_mask(self):
        """Reset the current mask being created."""
        self.current_points = []
        self.current_mask = None
        self.current_logits = None

    def reset_room(self):
        """Reset all masks for current room."""
        self.confirmed_masks = []
        self.reset_current_mask()


class SAM3PointPredictor:
    """Wrapper for SAM3 interactive point-based prediction."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._inference_state = None

    def load_model(self):
        """Load SAM3 model with instance interactivity enabled."""
        if self.model is not None:
            return

        repo_root = Path(__file__).resolve().parents[1]
        checkpoint_requirement = sam3_requirement(
            repo_root,
            stage="Manual mask creation with SAM3",
        )
        missing = find_missing_requirements([checkpoint_requirement])
        if missing:
            raise FileNotFoundError(format_missing_checkpoints_error(missing))

        print("Loading SAM3 model with instance interactivity...")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Find BPE path explicitly (pkg_resources fails in editable installs)
        import sam3.model_builder as mb
        sam3_dir = Path(mb.__file__).parent
        bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if not bpe_path.exists():
            raise FileNotFoundError(f"BPE tokenizer not found at {bpe_path}")

        print(f"Using BPE tokenizer: {bpe_path}")
        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_requirement.candidate_paths[0]),
            bpe_path=str(bpe_path),
            device=self.device,
            eval_mode=True,
            enable_inst_interactivity=True,
        )
        self.processor = Sam3Processor(self.model, device=self.device)
        print("SAM3 model loaded successfully.")

    def set_image(self, image: np.ndarray):
        """Set the image and compute embeddings."""
        if self.processor is None:
            self.load_model()

        # IMPORTANT: Sam3Processor.set_image() expects PIL Image or CHW tensor.
        # For numpy HWC arrays, it incorrectly extracts dimensions as shape[-2:]
        # which gives (W, C) instead of (H, W). Convert to PIL to fix this.
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        # Use processor to compute backbone features including SAM2 features
        self._inference_state = self.processor.set_image(pil_image)
        print(f"Image set: {pil_image.size[0]}x{pil_image.size[1]} (WxH)")

    def predict(
        self,
        points: List[Point],
        prev_logits: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mask from points.

        Args:
            points: List of Point objects with x, y, and label
            prev_logits: Previous low-res logits for iterative refinement
            multimask_output: Return multiple mask candidates

        Returns:
            Tuple of (masks, scores, low_res_logits)
            - masks: (num_masks, H, W) boolean array
            - scores: (num_masks,) float array
            - low_res_logits: (num_masks, 256, 256) float array for refinement
        """
        if self._inference_state is None:
            raise RuntimeError("Image must be set before prediction")

        if len(points) == 0:
            raise ValueError("At least one point is required")

        # Convert points to numpy arrays
        point_coords = np.array([[p.x, p.y] for p in points])
        point_labels = np.array([p.label for p in points])

        # Use previous logits for refinement if available
        mask_input = None
        if prev_logits is not None:
            # Select best mask from previous prediction
            mask_input = prev_logits[0:1]  # Shape: (1, H, W)

        # Use model.predict_inst which properly sets up features for the interactive predictor
        masks, scores, low_res_logits = self.model.predict_inst(
            self._inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=False,
        )

        # Debug: print actual shapes to diagnose issues
        print(f"DEBUG predict_inst output: masks.shape={masks.shape}, scores.shape={scores.shape}")

        # Handle different possible output formats from predict_inst
        if len(masks.shape) == 2:
            # Single mask (H, W) - wrap in extra dimension
            masks = masks[np.newaxis, ...]
            if np.isscalar(scores):
                scores = np.array([scores])
        elif len(masks.shape) == 3:
            # Check if it's (num_masks, H, W) or (H, W, num_masks)
            if masks.shape[0] not in [1, 3, 4]:  # num_masks is usually 1, 3, or 4
                # Might be (H, W, num_masks) - transpose to (num_masks, H, W)
                masks = np.transpose(masks, (2, 0, 1))
                print(f"DEBUG transposed masks to: {masks.shape}")

        return masks, scores, low_res_logits

    def reset(self):
        """Reset predictor state."""
        self._inference_state = None


class ManualMaskUI:
    """Gradio-based UI for manual mask creation."""

    # Colors for visualization
    POSITIVE_COLOR = (0, 255, 0)  # Green for positive points
    NEGATIVE_COLOR = (255, 0, 0)  # Red for negative points
    MASK_COLOR = (0, 100, 255)    # Blue overlay for masks
    MASK_ALPHA = 0.4

    CONFIRMED_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    CLOSE_TAB_JS = """
    () => {
        setTimeout(() => {
            try {
                window.open("", "_self");
                window.close();
            } catch (e) {
                console.warn("window.close() failed", e);
            }
            setTimeout(() => {
                try {
                    window.location.replace("about:blank");
                    window.close();
                } catch (e) {
                    console.warn("about:blank close fallback failed", e);
                }
            }, 250);
        }, 1200);
    }
    """

    def __init__(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        rooms: Optional[List[str]] = None,
        port: int = 7860,
        batch_config: Optional[List[dict]] = None,
        resume_command: Optional[str] = None,
        resume_file: Optional[Path] = None,
    ):
        self.port = port
        self.example_mask_path = Path(__file__).with_name("example_masks.png")
        self.resume_command = (resume_command or "").strip()
        self.resume_file = Path(resume_file) if resume_file else None
        self.demo = None

        # Build room_entries from batch_config or single input/output dirs
        if batch_config:
            self.room_entries = self._build_entries_from_batch(batch_config)
        else:
            self.input_dir = Path(input_dir)
            self.output_dir = Path(output_dir)
            if rooms:
                discovered = rooms
            else:
                discovered = self._discover_rooms()
            self.room_entries = [
                RoomEntry(room_id=r, input_dir=self.input_dir, output_dir=self.output_dir)
                for r in discovered
            ]

        if not self.room_entries:
            raise ValueError("No rooms found")

        # Build flat room list for SessionState compatibility
        self.rooms = [e.room_id for e in self.room_entries]

        print(f"Found {len(self.room_entries)} rooms:")
        for e in self.room_entries:
            prefix = f"[{e.scene_name}] " if e.scene_name else ""
            print(f"  {prefix}{e.room_id}")

        # Initialize state and predictor
        self.state = SessionState(rooms=self.rooms)
        self.predictor = SAM3PointPredictor()

        # Point mode: True = positive, False = negative
        self.point_mode_positive = True

    def _get_room_entry(self, room_index: int) -> RoomEntry:
        """Get the RoomEntry for a specific room index."""
        return self.room_entries[room_index]

    def _get_instruction_html(self) -> str:
        """Return the prominent manual masking guidance block."""
        return """
        <div style="
            background: #f5f7fb;
            border: 2px solid #d7deea;
            border-radius: 12px;
            padding: 18px 20px;
            color: #1f2937;
            line-height: 1.5;
        ">
            <h3 style="margin: 0 0 10px 0;">Manual Masking Guidelines</h3>
            <p style="margin: 0 0 10px 0;">
                Create each mask as one object group that belongs together physically.
                Do <strong>not</strong> split objects that stand on top of each other
                into separate masks.
            </p>
            <p style="margin: 0 0 8px 0;"><strong>Examples</strong></p>
            <ul style="margin: 0 0 12px 18px; padding: 0;">
                <li style="margin-bottom: 6px;">
                    If a lamp stands on a desk, include the <strong>desk and lamp in the same mask</strong>.
                </li>
                <li>
                    If chairs are tucked under a table, include the <strong>table and chairs in the same mask</strong>.
                </li>
            </ul>
            <p style="margin: 0 0 8px 0;"><strong>Use these label conventions</strong></p>
            <ul style="margin: 0 0 12px 18px; padding: 0;">
                <li style="margin-bottom: 6px;">
                    Use a plain name for normal floor objects:
                    <code>sofa</code>, <code>desk</code>, <code>chair</code>
                </li>
                <li style="margin-bottom: 6px;">
                    Append <code>_w</code> for wall-mounted objects:
                    <code>painting_w</code>, <code>tv_w</code>, <code>mirror_w</code>
                </li>
                <li style="margin-bottom: 6px;">
                    Append <code>_c</code> for ceiling-mounted or hanging objects:
                    <code>chandelier_c</code>, <code>pendant_light_c</code>, <code>ceiling_fan_c</code>
                </li>
                <li>
                    Include the word <code>window</code> for windows:
                    <code>window</code>, <code>arched_window</code>, <code>bay_window</code>
                </li>
            </ul>
            <p style="margin: 0 0 8px 0;"><strong>Window note</strong></p>
            <ul style="margin: 0 0 0 18px; padding: 0;">
                <li style="margin-bottom: 6px;">
                    Window masks are automatically fitted to the correct wall opening.
                </li>
                <li style="margin-bottom: 6px;">
                    If there are fewer window masks than openings, existing windows will be duplicated
                    to fill the remaining openings.
                </li>
                <li>
                    Avoid segmenting <strong>more window objects than there are real openings</strong>,
                    because extra window masks do not have matching openings and may be skipped.
                </li>
            </ul>
        </div>
        """

    def _build_entries_from_batch(self, batch_config: List[dict]) -> List[RoomEntry]:
        """Build RoomEntry list from batch_mask_config.json entries."""
        entries = []
        for scene_entry in batch_config:
            scene_name = scene_entry["scene_name"]
            input_dir = Path(scene_entry["input_dir"])
            output_dir = Path(scene_entry["output_dir"])
            for room_id in scene_entry["rooms"]:
                entries.append(RoomEntry(
                    room_id=room_id,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    scene_name=scene_name,
                ))
        return entries

    def _discover_rooms(self) -> List[str]:
        """Discover room directories from input."""
        rooms = []
        for subdir in sorted(self.input_dir.iterdir()):
            if subdir.is_dir():
                # Check for generated.png or any image
                if (subdir / "generated.png").exists():
                    rooms.append(subdir.name)
                elif list(subdir.glob("*.png")):
                    rooms.append(subdir.name)
        return rooms

    def _get_current_entry(self) -> RoomEntry:
        """Get the RoomEntry for the current room."""
        return self._get_room_entry(self.state.current_room_idx)

    def _get_room_masks_dir(self, room_index: int) -> Path:
        """Get the masks directory for a room index."""
        entry = self._get_room_entry(room_index)
        return entry.output_dir / entry.room_id / "masks"

    def _get_room_display_name(self, room_index: int) -> str:
        """Get display name including optional scene prefix."""
        entry = self._get_room_entry(room_index)
        scene_prefix = f"[{entry.scene_name}] " if entry.scene_name else ""
        return f"{scene_prefix}{entry.room_id}"

    def _get_progress_text(self, room_index: Optional[int] = None) -> str:
        """Get progress text for a room index."""
        if room_index is None:
            room_index = self.state.current_room_idx
        return f"Room {room_index + 1}/{len(self.rooms)}: {self._get_room_display_name(room_index)}"

    def _get_confirmation_popup_html(self) -> str:
        """Build the final confirmation popup message."""
        file_note = f"`{self.resume_file}`" if self.resume_file else "`RESUME.txt`"
        fallback = self.resume_command or "resume command unavailable"
        return f"""
        <div style="
            background: white;
            border: 2px solid #1f2937;
            border-radius: 16px;
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.35);
            max-width: 760px;
            margin: 0 auto;
            padding: 24px 28px;
            color: #111827;
        ">
            <h2 style="margin: 0 0 12px 0;">All done!</h2>
            <p style="margin: 0 0 12px 0;">
                Review the masks one more time. If you are happy with them, click
                <strong>Confirm Masks and Continue</strong>. The browser tab should close
                automatically and the pipeline should continue automatically.
            </p>
            <p style="margin: 0 0 12px 0;">
                If the pipeline does not continue automatically, run this command:
            </p>
            <pre style="
                background: #111827;
                color: #f9fafb;
                border-radius: 10px;
                padding: 14px;
                overflow-x: auto;
                margin: 0 0 12px 0;
            ">{fallback}</pre>
            <p style="margin: 0;">
                The same command is saved in {file_note}.
            </p>
        </div>
        """

    def _get_room_image_path(self, room_id: str) -> Optional[Path]:
        """Get the image path for a room."""
        entry = self._get_current_entry()
        room_dir = entry.input_dir / room_id

        # Try common names
        for name in ["generated.png", "flux_output.png"]:
            path = room_dir / name
            if path.exists():
                return path

        # Try generated_0000 with optional ref suffix
        matches = sorted(room_dir.glob("generated_0000*.png"))
        if matches:
            return matches[0]

        # Fall back to first PNG
        pngs = list(room_dir.glob("*.png"))
        if pngs:
            return sorted(pngs)[0]

        return None

    def _load_room_image(self, room_id: str) -> Tuple[np.ndarray, Path]:
        """Load image for a room."""
        path = self._get_room_image_path(room_id)
        if path is None:
            raise FileNotFoundError(f"No image found for room {room_id}")

        image = np.array(Image.open(path).convert("RGB"))
        return image, path

    def _draw_overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw current mask and points overlay on image."""
        overlay = image.copy().astype(np.float32)

        # Draw confirmed masks first (with transparency)
        for i, mask_data in enumerate(self.state.confirmed_masks):
            color = self.CONFIRMED_COLORS[i % len(self.CONFIRMED_COLORS)]
            mask_region = mask_data.mask.astype(bool)
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_region,
                    overlay[:, :, c] * (1 - self.MASK_ALPHA * 0.5) + color[c] * self.MASK_ALPHA * 0.5,
                    overlay[:, :, c]
                )

        # Draw current mask preview
        if self.state.current_mask is not None:
            mask_region = self.state.current_mask.astype(bool)
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_region,
                    overlay[:, :, c] * (1 - self.MASK_ALPHA) + self.MASK_COLOR[c] * self.MASK_ALPHA,
                    overlay[:, :, c]
                )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Draw points
        from PIL import ImageDraw
        pil_img = Image.fromarray(overlay)
        draw = ImageDraw.Draw(pil_img)

        # Draw current points
        for point in self.state.current_points:
            color = self.POSITIVE_COLOR if point.label == 1 else self.NEGATIVE_COLOR
            r = 8
            draw.ellipse(
                [point.x - r, point.y - r, point.x + r, point.y + r],
                fill=color,
                outline=(255, 255, 255),
                width=2
            )

        return np.array(pil_img)

    def _update_mask_preview(self):
        """Update mask preview from current points."""
        if len(self.state.current_points) == 0:
            self.state.current_mask = None
            self.state.current_logits = None
            return

        try:
            masks, scores, logits = self.predictor.predict(
                self.state.current_points,
                prev_logits=self.state.current_logits,
                multimask_output=len(self.state.current_points) == 1,
            )

            # Select best mask based on score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]

            # Ensure mask is 2D (H, W)
            if len(best_mask.shape) != 2:
                print(f"WARNING: Unexpected mask shape {best_mask.shape}, squeezing")
                best_mask = best_mask.squeeze()

            self.state.current_mask = best_mask
            self.state.current_logits = logits

        except Exception as e:
            print(f"Error predicting mask: {e}")
            self.state.current_mask = None

    def _get_mask_list_html(self) -> str:
        """Generate HTML for mask list with color indicators."""
        print(f"DEBUG _get_mask_list_html called, confirmed_masks count: {len(self.state.confirmed_masks)}")
        if not self.state.confirmed_masks:
            return "<div style='color: gray; padding: 10px;'>No masks yet. Click on the image to start.</div>"

        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for i, mask_data in enumerate(self.state.confirmed_masks):
            color = self.CONFIRMED_COLORS[i % len(self.CONFIRMED_COLORS)]
            color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            area_pct = mask_data.score * 100  # score is area_ratio

            html += f"""
            <div style='display: flex; align-items: center; padding: 8px;
                        border-bottom: 1px solid #444; gap: 10px;'>
                <div style='width: 20px; height: 20px; background: {color_hex};
                            border-radius: 3px; flex-shrink: 0;'></div>
                <div style='flex-grow: 1;'>
                    <strong>{i}: {mask_data.label}</strong><br>
                    <small style='color: gray;'>{area_pct:.1f}% area</small>
                </div>
            </div>
            """
        html += "</div>"
        return html

    def _get_mask_dropdown_choices(self) -> List[str]:
        """Get dropdown choices for mask selection."""
        return [f"{i}: {m.label}" for i, m in enumerate(self.state.confirmed_masks)]

    def _load_saved_masks_for_room(self, room_index: int) -> List[MaskData]:
        """Load previously saved masks for a room."""
        room_masks_dir = self._get_room_masks_dir(room_index)
        metadata_path = room_masks_dir / "metadata.json"
        if not metadata_path.exists():
            return []

        with open(metadata_path) as f:
            metadata = json.load(f)

        loaded_masks: List[MaskData] = []
        for item in metadata.get("masks", []):
            mask_path = room_masks_dir / item["filename"]
            if not mask_path.exists():
                print(f"WARNING: Saved mask missing: {mask_path}")
                continue

            rgba = np.array(Image.open(mask_path).convert("RGBA"))
            mask = rgba[..., 3] > 0
            loaded_masks.append(
                MaskData(
                    mask=mask,
                    label=item["label"],
                    score=float(item.get("score", item.get("area_ratio", 0.0))),
                    points=[],
                )
            )

        return loaded_masks

    def _has_pending_preview(self) -> bool:
        """Check if there is an unfinished preview mask."""
        return bool(self.state.current_points) or self.state.current_mask is not None

    def _refresh_room_ui(
        self,
        status: str,
        label_value: str = "",
        progress_value: Optional[str] = None,
        popup_visible: bool = False,
    ):
        """Return a consistent set of UI updates for room-level actions."""
        image = self._draw_overlay(self.state.current_image) if self.state.current_image is not None else None
        progress = progress_value if progress_value is not None else self._get_progress_text()
        return (
            image,
            status,
            progress,
            label_value,
            gr.update(value=self._get_mask_list_html()),
            gr.update(choices=self._get_mask_dropdown_choices(), value=None),
            gr.update(value="All done!" if popup_visible else ""),
            gr.update(visible=popup_visible),
            gr.update(value=self._get_confirmation_popup_html()),
        )

    def handle_click(self, image: np.ndarray, evt: gr.SelectData) -> Tuple[np.ndarray, str]:
        """Handle click on image to add point."""
        if self.state.current_image is None:
            return image, "No image loaded"

        x, y = evt.index
        label = 1 if self.point_mode_positive else 0
        point = Point(x=x, y=y, label=label)
        self.state.current_points.append(point)

        # Update mask preview
        self._update_mask_preview()

        # Draw overlay
        overlay = self._draw_overlay(self.state.current_image)

        mode_str = "positive" if label == 1 else "negative"
        status = f"Added {mode_str} point at ({x}, {y}). Total points: {len(self.state.current_points)}"

        return overlay, status

    def undo_point(self) -> Tuple[np.ndarray, str]:
        """Remove the last point."""
        if self.state.current_image is None:
            return None, "No image loaded"

        if len(self.state.current_points) > 0:
            removed = self.state.current_points.pop()

            # Reset logits to force fresh prediction
            self.state.current_logits = None
            self._update_mask_preview()

        overlay = self._draw_overlay(self.state.current_image)
        return overlay, f"Points: {len(self.state.current_points)}"

    def clear_current(self) -> Tuple[np.ndarray, str]:
        """Clear current mask and points."""
        if self.state.current_image is None:
            return None, "No image loaded"

        self.state.reset_current_mask()
        overlay = self._draw_overlay(self.state.current_image)
        return overlay, "Cleared current mask"

    def confirm_mask(self, label: str):
        """Confirm current mask with label."""
        print(f"DEBUG confirm_mask called with label='{label}'")
        print(f"DEBUG current_image is None: {self.state.current_image is None}")
        print(f"DEBUG current_mask is None: {self.state.current_mask is None}")

        if self.state.current_image is None:
            return None, "No image loaded", "", gr.update(value=self._get_mask_list_html()), gr.update(choices=[])

        if self.state.current_mask is None:
            overlay = self._draw_overlay(self.state.current_image)
            return overlay, "No mask to confirm", label, gr.update(value=self._get_mask_list_html()), gr.update(choices=self._get_mask_dropdown_choices())

        if not label.strip():
            overlay = self._draw_overlay(self.state.current_image)
            return overlay, "Please enter a label", label, gr.update(value=self._get_mask_list_html()), gr.update(choices=self._get_mask_dropdown_choices())

        # Calculate score (placeholder - could use IoU with previous or area ratio)
        mask_area = self.state.current_mask.sum()
        total_area = self.state.current_mask.shape[0] * self.state.current_mask.shape[1]
        area_ratio = mask_area / total_area

        mask_data = MaskData(
            mask=self.state.current_mask.copy(),
            label=label.strip(),
            score=float(area_ratio),  # Use area ratio as proxy score
            points=self.state.current_points.copy(),
        )
        self.state.confirmed_masks.append(mask_data)
        print(f"DEBUG mask added, total confirmed_masks: {len(self.state.confirmed_masks)}")

        # Reset current mask state
        self.state.reset_current_mask()

        overlay = self._draw_overlay(self.state.current_image)
        status = f"Confirmed mask '{label}'. Total masks: {len(self.state.confirmed_masks)}"

        mask_html = self._get_mask_list_html()
        print(f"DEBUG mask_html length: {len(mask_html)}")
        print(f"DEBUG mask_html preview: {mask_html[:200]}...")

        return overlay, status, "", gr.update(value=mask_html), gr.update(choices=self._get_mask_dropdown_choices())

    def delete_mask(self, selection: str):
        """Delete a specific mask by selection from dropdown."""
        if self.state.current_image is None:
            return None, "No image loaded", gr.update(value=self._get_mask_list_html()), gr.update(choices=[])

        if not selection or len(self.state.confirmed_masks) == 0:
            overlay = self._draw_overlay(self.state.current_image)
            return overlay, "No mask selected or no masks to delete", gr.update(value=self._get_mask_list_html()), gr.update(choices=self._get_mask_dropdown_choices())

        # Parse index from selection (format: "0: label")
        try:
            mask_index = int(selection.split(":")[0])
        except (ValueError, IndexError):
            overlay = self._draw_overlay(self.state.current_image)
            return overlay, "Invalid selection", gr.update(value=self._get_mask_list_html()), gr.update(choices=self._get_mask_dropdown_choices())

        if mask_index < 0 or mask_index >= len(self.state.confirmed_masks):
            overlay = self._draw_overlay(self.state.current_image)
            return overlay, "Invalid mask index", gr.update(value=self._get_mask_list_html()), gr.update(choices=self._get_mask_dropdown_choices())

        removed = self.state.confirmed_masks.pop(mask_index)
        overlay = self._draw_overlay(self.state.current_image)
        status = f"Deleted mask '{removed.label}'. Remaining: {len(self.state.confirmed_masks)}"

        return overlay, status, gr.update(value=self._get_mask_list_html()), gr.update(choices=self._get_mask_dropdown_choices(), value=None)

    def toggle_point_mode(self) -> str:
        """Toggle between positive and negative point mode."""
        self.point_mode_positive = not self.point_mode_positive
        mode = "Positive (foreground)" if self.point_mode_positive else "Negative (background)"
        return f"Point mode: {mode}"

    def _save_masks_for_room(self, room_id: str) -> Path:
        """Save confirmed masks for a room in the expected format."""
        entry = self._get_current_entry()
        room_output = entry.output_dir / room_id / "masks"
        room_output.mkdir(parents=True, exist_ok=True)

        for old_mask in room_output.glob("*.png"):
            old_mask.unlink()

        metadata = {
            "image_shape": list(self.state.current_image.shape[:2]),
            "num_masks": len(self.state.confirmed_masks),
            "masks": []
        }

        for i, mask_data in enumerate(self.state.confirmed_masks):
            # Create RGBA image with mask as alpha
            h, w = mask_data.mask.shape
            mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            mask_rgba[..., 3] = (mask_data.mask * 255).astype(np.uint8)

            # Save mask
            filename = f"{i:02d}_{mask_data.label.replace(' ', '_')}.png"
            filepath = room_output / filename
            Image.fromarray(mask_rgba).save(filepath)

            # Check if doorway
            is_doorway = any(
                term in mask_data.label.lower()
                for term in ["doorway", "door", "doorframe"]
            )

            metadata["masks"].append({
                "index": i,
                "filename": filename,
                "label": mask_data.label,
                "score": mask_data.score,
                "area": int(mask_data.mask.sum()),
                "area_ratio": float(mask_data.mask.sum() / (h * w)),
                "is_doorway": is_doorway,
            })

            print(f"  Saved: {filename}")

        # Save metadata
        meta_path = room_output / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Metadata saved: {meta_path}")
        return room_output

    def _save_current_room_state(self) -> str:
        """Persist the currently loaded room."""
        if self.state.current_image is None:
            return "No image loaded"

        room_id = self.rooms[self.state.current_room_idx]
        output_path = self._save_masks_for_room(room_id)
        return f"Saved {len(self.state.confirmed_masks)} masks for {room_id} to {output_path}"

    def _load_room(self, room_index: int, prev_msg: str = ""):
        """Load a room image and its saved masks."""
        self.state.reset_room()
        self.state.current_room_idx = room_index
        room_id = self.rooms[room_index]

        try:
            image, path = self._load_room_image(room_id)
            self.state.current_image = image
            self.state.current_image_path = path
            self.state.confirmed_masks = self._load_saved_masks_for_room(room_index)

            self.predictor.set_image(image)

            status = f"{prev_msg}\n\nLoaded {self._get_room_display_name(room_index)}" if prev_msg else f"Loaded {self._get_room_display_name(room_index)}"
            return self._refresh_room_ui(status=status)

        except Exception as e:
            status = f"{prev_msg}\n\nError loading {room_id}: {e}"
            return (
                None,
                status,
                f"Error: {room_id}",
                "",
                gr.update(value=self._get_mask_list_html()),
                gr.update(choices=[]),
                gr.update(value=""),
                gr.update(visible=False),
                gr.update(value=self._get_confirmation_popup_html()),
            )

    def _navigate_to_room(self, target_index: int, action_label: str):
        """Save current room, then navigate to another room or finish review."""
        if self.state.current_image is None:
            return self._refresh_room_ui(status="No image loaded")

        if self._has_pending_preview():
            return self._refresh_room_ui(
                status="Finish the current preview first: confirm the mask or clear the current points before changing rooms."
            )

        if target_index < 0:
            return self._refresh_room_ui(
                status="Already at the first room."
            )

        save_msg = self._save_current_room_state()

        if target_index >= len(self.rooms):
            done_status = (
                f"{save_msg}\n\nAll rooms completed. Review any room if needed. "
                "When you are happy with the masks, confirm to continue."
            )
            return self._refresh_room_ui(
                status=done_status,
                progress_value="All done!",
                popup_visible=True,
            )

        return self._load_room(target_index, f"{save_msg}\n\n{action_label}")

    def previous_room(self):
        """Save and go back to the previous room."""
        return self._navigate_to_room(self.state.current_room_idx - 1, "Moved to previous room")

    def next_room(self):
        """Save and go forward to the next room, or enter final review."""
        return self._navigate_to_room(self.state.current_room_idx + 1, "Moved to next room")

    def skip_room(self):
        """Skip current room but keep it reviewable later."""
        if self._has_pending_preview():
            return self._refresh_room_ui(
                status="Clear the current preview before skipping this room."
            )

        self.state.confirmed_masks = []
        self.state.reset_current_mask()
        return self._navigate_to_room(self.state.current_room_idx + 1, "Skipped room")

    def keep_editing(self):
        """Dismiss the confirmation popup and keep editing."""
        return gr.update(visible=False), self._refresh_room_ui(
            status="Popup dismissed. You can keep reviewing rooms and editing masks.",
        )[1], gr.update(value="")

    def _close_demo(self):
        """Close the Gradio server after the response is sent."""
        if self.demo is None:
            return
        try:
            self.demo.close()
        except Exception as e:
            print(f"Warning: failed to close Gradio server cleanly: {e}")

    def confirm_and_continue(self):
        """Persist masks, close Gradio, and allow the parent pipeline to continue."""
        if self._has_pending_preview():
            return self._refresh_room_ui(
                status="Finish the current preview first: confirm the mask or clear the current points before continuing.",
                progress_value="All done!",
                popup_visible=True,
            )

        save_msg = self._save_current_room_state()
        threading.Timer(0.5, self._close_demo).start()
        return self._refresh_room_ui(
            status=f"{save_msg}\n\nConfirmation received. Closing the Gradio UI tab and server so the pipeline can continue.",
            progress_value="All done!",
            popup_visible=False,
        )

    def launch(self):
        """Launch the Gradio interface."""
        # Load model first
        print("Pre-loading SAM3 model...")
        self.predictor.load_model()

        # Load first room (ignore gr.update objects for initial values)
        image, status, progress, _, _, _, _, _, _ = self._load_room(0)
        # Get initial mask list HTML directly
        mask_list_html = self._get_mask_list_html()
        mask_choices = self._get_mask_dropdown_choices()

        # Build Gradio interface
        with gr.Blocks(title="Manual Mask Creation") as demo:
            self.demo = demo
            gr.Markdown("# Manual Mask Creation with SAM3")
            done_text = gr.Textbox(
                value="",
                label="Completion",
                interactive=False,
                visible=False,
            )
            gr.Markdown(
                "Click on objects to add points. Left-click adds positive points (foreground). "
                "Use the toggle to switch to negative points (background). "
                "The mask preview updates in real-time."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(self._get_instruction_html())
                with gr.Column(scale=2, min_width=280):
                    if self.example_mask_path.exists():
                        gr.Image(
                            value=str(self.example_mask_path),
                            label="Example: correct mask grouping",
                            interactive=False,
                            height=260,
                        )
                    else:
                        gr.Markdown(
                            "*Example mask image not found at* "
                            f"`{self.example_mask_path}`"
                        )

            with gr.Row():
                # Image display column (large)
                with gr.Column(scale=4):
                    image_display = gr.Image(
                        value=image,
                        label="Click to add points",
                        type="numpy",
                        interactive=True,
                        height=720,
                    )

                # Mask gallery column
                with gr.Column(scale=1, min_width=240):
                    gr.Markdown("### Confirmed Masks")
                    mask_list_display = gr.HTML(
                        value=mask_list_html,
                    )

                    # Mask deletion controls
                    gr.Markdown("#### Delete Mask")
                    mask_selector = gr.Dropdown(
                        choices=mask_choices,
                        label="Select mask to delete",
                        interactive=True,
                    )
                    delete_btn = gr.Button("Delete Selected", variant="stop")

                # Controls column (1/4 width)
                with gr.Column(scale=1):
                    # Status and progress
                    progress_text = gr.Textbox(
                        value=progress,
                        label="Progress",
                        interactive=False,
                    )
                    status_text = gr.Textbox(
                        value=status,
                        label="Status",
                        lines=5,
                        interactive=False,
                    )

                    # Point mode toggle
                    mode_text = gr.Textbox(
                        value="Point mode: Positive (foreground)",
                        label="Current Mode",
                        interactive=False,
                    )
                    toggle_btn = gr.Button("Toggle Point Mode")

                    # Label input
                    label_input = gr.Textbox(
                        label="Object Label",
                        placeholder="e.g., sofa, painting_w, chandelier_c, arched_window",
                    )

                    # Mask actions
                    with gr.Row():
                        confirm_btn = gr.Button("Confirm Mask", variant="primary")
                        undo_btn = gr.Button("Undo Point")

                    clear_btn = gr.Button("Clear Current Points")

                    # Navigation
                    gr.Markdown("---")
                    with gr.Row():
                        prev_btn = gr.Button("Previous Room")
                        next_btn = gr.Button("Next Room", variant="primary")
                        skip_btn = gr.Button("Skip Room")

            with gr.Group(visible=False) as confirm_group:
                confirm_popup = gr.HTML(value=self._get_confirmation_popup_html())
                with gr.Row():
                    keep_editing_btn = gr.Button("Keep Editing")
                    confirm_continue_btn = gr.Button("Confirm Masks and Continue", variant="primary")

            # Event handlers
            image_display.select(
                fn=self.handle_click,
                inputs=[image_display],
                outputs=[image_display, status_text],
            )

            toggle_btn.click(
                fn=self.toggle_point_mode,
                outputs=[mode_text],
            )

            undo_btn.click(
                fn=self.undo_point,
                outputs=[image_display, status_text],
            )

            clear_btn.click(
                fn=self.clear_current,
                outputs=[image_display, status_text],
            )

            confirm_btn.click(
                fn=self.confirm_mask,
                inputs=[label_input],
                outputs=[image_display, status_text, label_input, mask_list_display, mask_selector],
            )

            delete_btn.click(
                fn=self.delete_mask,
                inputs=[mask_selector],
                outputs=[image_display, status_text, mask_list_display, mask_selector],
            )

            prev_btn.click(
                fn=self.previous_room,
                outputs=[image_display, status_text, progress_text, label_input, mask_list_display, mask_selector, done_text, confirm_group, confirm_popup],
            )

            next_btn.click(
                fn=self.next_room,
                outputs=[image_display, status_text, progress_text, label_input, mask_list_display, mask_selector, done_text, confirm_group, confirm_popup],
            )

            skip_btn.click(
                fn=self.skip_room,
                outputs=[image_display, status_text, progress_text, label_input, mask_list_display, mask_selector, done_text, confirm_group, confirm_popup],
            )

            keep_editing_btn.click(
                fn=self.keep_editing,
                outputs=[confirm_group, status_text, done_text],
            )

            confirm_continue_btn.click(
                fn=self.confirm_and_continue,
                outputs=[image_display, status_text, progress_text, label_input, mask_list_display, mask_selector, done_text, confirm_group, confirm_popup],
                js=self.CLOSE_TAB_JS,
            )

        # Launch
        print(f"\nLaunching Gradio UI on port {self.port}...")
        print(f"Open http://localhost:{self.port} in your browser\n")
        demo.launch(server_port=self.port, share=False)


def main():
    parser = argparse.ArgumentParser(
        description="Manual Mask Creation with Gradio UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single-scene mode (original)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing room subdirectories with generated images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for masks (will create room_id/masks subdirs)",
    )
    parser.add_argument(
        "--rooms",
        nargs="+",
        default=None,
        help="Specific rooms to process (default: auto-discover from input-dir)",
    )

    # Batch mode (multi-scene)
    parser.add_argument(
        "--batch-config",
        type=Path,
        default=None,
        help="Path to batch_mask_config.json for multi-scene sessions (mutually exclusive with --input-dir/--output-dir)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server (default: 7860)",
    )
    parser.add_argument(
        "--resume-command",
        type=str,
        default="",
        help="Fallback command to run if the pipeline does not continue automatically",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="Path to the file that stores the fallback resume command",
    )

    args = parser.parse_args()

    # Validate args
    if args.batch_config and (args.input_dir or args.output_dir):
        print("Error: --batch-config is mutually exclusive with --input-dir/--output-dir")
        return 1

    if not args.batch_config and not (args.input_dir and args.output_dir):
        print("Error: Either --batch-config or both --input-dir and --output-dir are required")
        return 1

    missing = find_missing_requirements([
        sam3_requirement(Path(__file__).resolve().parents[1], stage="Manual mask creation with SAM3")
    ])
    if missing:
        print("\n" + format_missing_checkpoints_error(missing))
        return 1

    try:
        if args.batch_config:
            if not args.batch_config.exists():
                print(f"Error: Batch config not found: {args.batch_config}")
                return 1
            with open(args.batch_config) as f:
                batch_config = json.load(f)
            ui = ManualMaskUI(
                batch_config=batch_config,
                port=args.port,
                resume_command=args.resume_command,
                resume_file=args.resume_file,
            )
        else:
            if not args.input_dir.exists():
                print(f"Error: Input directory not found: {args.input_dir}")
                return 1
            args.output_dir.mkdir(parents=True, exist_ok=True)
            ui = ManualMaskUI(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                rooms=args.rooms,
                port=args.port,
                resume_command=args.resume_command,
                resume_file=args.resume_file,
            )
        ui.launch()
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
