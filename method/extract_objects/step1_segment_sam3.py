#!/usr/bin/env python3
"""
Step 1: SAM3 Object Segmentation

Segments objects in an image using SAM3's text-prompt capabilities.
Runs in the 'worldmesh-sam3' conda environment.

Usage:
    conda run -n worldmesh-sam3 python step1_segment_sam3.py \
        --input-image ../generations/Flux2_00013_.png \
        --output-dir ../output/masks \
        --confidence 0.3
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint_requirements import (
    find_missing_requirements,
    format_missing_checkpoints_error,
    sam3_requirement,
)


def load_sam3_model(checkpoint_path: Path, device="cuda"):
    """Load SAM3 image model and processor."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        eval_mode=True,
    )
    return model


def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_containment(mask_small, mask_large):
    """
    Compute what fraction of mask_small is contained within mask_large.
    Returns: fraction of mask_small's area that overlaps with mask_large
    """
    small_area = mask_small.sum()
    if small_area == 0:
        return 0.0
    intersection = np.logical_and(mask_small, mask_large).sum()
    return intersection / small_area


def segment_objects(model, image, prompts, confidence_threshold=0.3, device="cuda"):
    """
    Segment objects using SAM3 text prompts.

    Args:
        model: SAM3 model
        image: PIL Image or numpy array
        prompts: List of text prompts (furniture types)
        confidence_threshold: Minimum confidence score
        device: Device to run on

    Returns:
        List of (mask, label, score) tuples
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)

    # Set image
    state = processor.set_image(image)

    all_detections = []

    for prompt in prompts:
        print(f"  Querying: '{prompt}'...", end=" ", flush=True)

        # Reset prompts for new query
        processor.reset_all_prompts(state)

        # Run text prompt
        state = processor.set_text_prompt(prompt=prompt, state=state)

        if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
            print("no detections")
            continue

        masks = state["masks"].cpu().numpy()
        scores = state["scores"].cpu().numpy()
        boxes = state["boxes"].cpu().numpy()

        print(f"{len(masks)} detections")

        for i in range(len(masks)):
            mask = masks[i, 0]  # Remove channel dimension
            score = scores[i]
            box = boxes[i]
            all_detections.append({
                'mask': mask,
                'label': prompt,
                'score': float(score),
                'box': box.tolist()
            })

    return all_detections


def segment_structure_and_invert(model, image, structure_prompts, confidence_threshold=0.3, device="cuda"):
    """
    Segment structural elements and invert to get object mask.

    Args:
        model: SAM3 model
        image: PIL Image
        structure_prompts: List like ["ceiling", "wall", "window", "bedroom floor"]
        confidence_threshold: Minimum confidence score
        device: Device to run on

    Returns:
        Single detection dict with inverted mask (objects = NOT structure)
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
    state = processor.set_image(image)

    # Get image dimensions
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Collect all structural masks
    combined_structure = np.zeros((h, w), dtype=bool)

    for prompt in structure_prompts:
        print(f"  Querying structure: '{prompt}'...", end=" ", flush=True)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
            print("no detections")
            continue

        masks = state["masks"].cpu().numpy()
        scores = state["scores"].cpu().numpy()
        print(f"{len(masks)} detections")

        # Add all detected masks for this prompt to combined structure
        for i in range(len(masks)):
            mask = masks[i, 0] > 0.5  # Binary threshold
            combined_structure = np.logical_or(combined_structure, mask)
            print(f"    Added {prompt} mask {i} (score={scores[i]:.3f})")

    # Invert to get objects (everything that is NOT structure)
    objects_mask = ~combined_structure

    # Calculate area
    object_area = objects_mask.sum()
    total_area = h * w

    print(f"\n  Structure coverage: {combined_structure.sum()/total_area*100:.1f}%")
    print(f"  Objects coverage: {object_area/total_area*100:.1f}%")

    return {
        'mask': objects_mask,
        'label': 'objects',
        'score': 1.0,  # Synthetic score
        'structure_mask': combined_structure  # Keep for reference
    }


def merge_overlapping_masks(detections, iou_threshold=0.7, containment_threshold=0.7):
    """
    Merge overlapping masks.

    Strategy:
    - If mask A is significantly contained in mask B (>containment_threshold of A's area),
      merge A into B and keep B's label (the larger/parent object)
    - This handles: pillows on beds, items on nightstands, etc.

    Args:
        detections: List of detection dicts with 'mask', 'label', 'score'
        iou_threshold: IoU threshold for considering masks as same object
        containment_threshold: Containment threshold for merging small into large

    Returns:
        Filtered list of detections after merging
    """
    if len(detections) == 0:
        return []

    # Sort by mask area (largest first)
    detections = sorted(detections, key=lambda d: d['mask'].sum(), reverse=True)

    merged = []
    used = set()

    for i, det_i in enumerate(detections):
        if i in used:
            continue

        current_mask = det_i['mask'].copy()
        current_label = det_i['label']
        current_score = det_i['score']

        # Check all smaller masks for merging
        for j, det_j in enumerate(detections):
            if j <= i or j in used:
                continue

            # Check containment (is smaller mask contained in larger?)
            containment = compute_containment(det_j['mask'], current_mask)
            iou = compute_iou(det_j['mask'], current_mask)

            if containment > containment_threshold or iou > iou_threshold:
                # Merge smaller mask into current
                current_mask = np.logical_or(current_mask, det_j['mask'])
                used.add(j)
                print(f"    Merged '{det_j['label']}' into '{current_label}' "
                      f"(containment={containment:.2f}, iou={iou:.2f})")

        merged.append({
            'mask': current_mask,
            'label': current_label,
            'score': current_score
        })
        used.add(i)

    return merged


def split_disconnected_masks(detections, min_component_ratio=0.05):
    """
    Split masks with disconnected components into separate detections.

    Args:
        detections: List of detection dicts with 'mask', 'label', 'score'
        min_component_ratio: Minimum component size as ratio of total mask area

    Returns:
        List of detections with each connected component as separate detection
    """
    from scipy import ndimage

    result = []

    for det in detections:
        mask = det['mask']
        label = det['label']
        score = det['score']

        # Find connected components
        labeled_array, num_features = ndimage.label(mask)

        if num_features <= 1:
            # Single component, keep as is
            result.append(det)
            continue

        # Multiple components - split them
        total_area = mask.sum()
        min_area = total_area * min_component_ratio

        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            component_area = component_mask.sum()

            if component_area >= min_area:
                result.append({
                    'mask': component_mask,
                    'label': label,
                    'score': score
                })
                print(f"    Split '{label}' component {i}/{num_features} "
                      f"(area={component_area:.0f})")
            else:
                print(f"    Discarded tiny '{label}' component {i}/{num_features} "
                      f"(area={component_area:.0f}, min={min_area:.0f})")

    return result


def filter_small_masks(detections, min_area_ratio=0.01):
    """
    Filter out masks that are too small.

    Args:
        detections: List of detection dicts
        min_area_ratio: Minimum mask area as fraction of image area

    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []

    # Get image dimensions from first mask
    h, w = detections[0]['mask'].shape
    image_area = h * w
    min_area = image_area * min_area_ratio

    filtered = []
    for det in detections:
        area = det['mask'].sum()
        if area >= min_area:
            filtered.append(det)
        else:
            print(f"    Filtered out small '{det['label']}' mask "
                  f"(area={area:.0f}, min={min_area:.0f})")

    return filtered


# Labels that indicate doorway/door for exclusion in step2
DOORWAY_LABELS = ["doorway", "door", "doorframe", "door frame", "open door", "door opening"]


def is_doorway_label(label: str) -> bool:
    """Check if a label indicates a doorway/door for exclusion."""
    label_lower = label.lower()
    for doorway_label in DOORWAY_LABELS:
        if doorway_label in label_lower:
            return True
    return False


def save_masks(detections, output_dir, image_shape):
    """
    Save masks as PNG files with alpha channel.
    Also saves metadata JSON with is_doorway flag.

    Args:
        detections: List of detection dicts
        output_dir: Directory to save masks
        image_shape: (height, width) of original image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for i, det in enumerate(detections):
        mask = det['mask']
        label = det['label']
        score = det['score']

        # Create RGBA image with mask as alpha
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[..., 3] = (mask * 255).astype(np.uint8)

        # Save mask
        filename = f"{i:02d}_{label.replace(' ', '_')}.png"
        filepath = output_dir / filename
        Image.fromarray(mask_rgba).save(filepath)

        # Check if this is a doorway mask (for exclusion in step2)
        is_doorway = is_doorway_label(label)

        metadata.append({
            'index': i,
            'filename': filename,
            'label': label,
            'score': score,
            'area': int(mask.sum()),
            'area_ratio': float(mask.sum() / (mask.shape[0] * mask.shape[1])),
            'is_doorway': is_doorway,
        })

        doorway_marker = " [DOORWAY]" if is_doorway else ""
        print(f"  Saved: {filename} (area={mask.sum()}, score={score:.3f}){doorway_marker}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'image_shape': list(image_shape),
            'num_masks': len(detections),
            'masks': metadata
        }, f, indent=2)

    print(f"  Metadata saved: {meta_path}")


def save_structure_mask(mask, output_dir):
    """Save the combined structure mask for debugging."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[..., 3] = (mask * 255).astype(np.uint8)
    filepath = output_dir / "structure_combined.png"
    Image.fromarray(mask_rgba).save(filepath)
    print(f"  Saved structure mask: {filepath}")


def create_overlay_visualization(image, detections, output_dir, alpha=0.5):
    """
    Create an overlay image showing all segmentation masks on the original image.

    Args:
        image: Original PIL Image or numpy array
        detections: List of detection dicts with 'mask', 'label', 'score'
        output_dir: Directory to save the overlay
        alpha: Transparency of the overlay (0-1, default 0.5)
    """
    from PIL import ImageDraw, ImageFont

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()

    # Ensure RGB
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    # Create overlay image
    overlay = image_np.copy().astype(np.float32)

    # Distinct colors for different objects (RGB)
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring green
        (255, 0, 128),    # Pink
        (128, 255, 0),    # Lime
        (0, 128, 255),    # Sky blue
        (255, 128, 128),  # Light red
        (128, 255, 128),  # Light green
        (128, 128, 255),  # Light blue
    ]

    # Apply each mask with its color
    labels_info = []
    for i, det in enumerate(detections):
        mask = det['mask']
        label = det['label']
        score = det['score']
        color = colors[i % len(colors)]

        # Find mask centroid for label placement
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            labels_info.append((centroid_x, centroid_y, label, score, color))

        # Apply colored overlay where mask is True
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

        # Draw mask boundary
        from scipy import ndimage
        # Find boundary by comparing mask with eroded mask
        eroded = ndimage.binary_erosion(mask, iterations=2)
        boundary = mask & ~eroded
        for c in range(3):
            overlay[:, :, c] = np.where(boundary, color[c], overlay[:, :, c])

    # Convert back to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Convert to PIL for text drawing
    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Draw labels
    for cx, cy, label, score, color in labels_info:
        text = f"{label} ({score:.2f})"

        # Get text bounding box
        bbox = draw.textbbox((cx, cy), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position text (centered on centroid)
        text_x = cx - text_w // 2
        text_y = cy - text_h // 2

        # Draw background rectangle for readability
        padding = 2
        draw.rectangle(
            [text_x - padding, text_y - padding, text_x + text_w + padding, text_y + text_h + padding],
            fill=(0, 0, 0, 180)
        )

        # Draw text in the mask's color
        draw.text((text_x, text_y), text, fill=color, font=font)

    # Save overlay
    filepath = output_dir / "overlay.png"
    overlay_pil.save(filepath)
    print(f"  Saved overlay visualization: {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="SAM3 Object Segmentation")
    parser.add_argument("--input-image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Output directory for masks")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--min-area", type=float, default=0.003,
                        help="Minimum mask area ratio (default: 0.003)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--prompts", nargs="+",
                        default=["bed with bedding and pillows", "nightstand", "lamp",
                                 "dresser", "chair", "desk", "wardrobe", "plant with pot",
                                 "book", "bag", "shoes", "trash can", "laundry basket"],
                        help="Text prompts for object types")
    parser.add_argument("--invert-structure", action="store_true",
                        help="Use inverted structure mask approach")
    parser.add_argument("--structure-prompts", nargs="+",
                        default=["ceiling", "wall", "window", "bedroom floor"],
                        help="Prompts for structural elements to invert")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: SAM3 Object Segmentation")
    print("=" * 60)

    repo_root = Path(__file__).resolve().parents[2]
    checkpoint_requirement = sam3_requirement(repo_root, stage="Step 1: SAM3 segmentation")
    missing = find_missing_requirements([checkpoint_requirement])
    if missing:
        print("\n" + format_missing_checkpoints_error(missing), file=sys.stderr)
        return 1

    # Load image
    print(f"\nLoading image: {args.input_image}")
    image = Image.open(args.input_image)
    image_np = np.array(image)
    print(f"  Image size: {image.size}")

    # Load model
    print("\n[1/3] Loading SAM3 model...")
    model = load_sam3_model(checkpoint_requirement.candidate_paths[0], device=args.device)

    if args.invert_structure:
        # Inverted structure approach: segment structure, invert to get objects
        print(f"\n[2/3] Segmenting structure to invert...")
        print(f"  Structure prompts: {args.structure_prompts}")
        detection = segment_structure_and_invert(
            model, image, args.structure_prompts,
            confidence_threshold=args.confidence,
            device=args.device
        )
        detections = [detection]

        # Also save the structure mask for debugging
        print(f"\n[3/3] Saving structure mask for debugging...")
        save_structure_mask(detection['structure_mask'], args.output_dir)
    else:
        # Original object-based segmentation
        print(f"\n[2/5] Segmenting objects with confidence threshold {args.confidence}...")
        print(f"  Prompts: {args.prompts}")
        detections = segment_objects(
            model, image, args.prompts,
            confidence_threshold=args.confidence,
            device=args.device
        )
        print(f"  Total detections: {len(detections)}")

        # Filter small masks
        print(f"\n[3/5] Filtering small masks (min area ratio: {args.min_area})...")
        detections = filter_small_masks(detections, min_area_ratio=args.min_area)
        print(f"  After filtering: {len(detections)}")

        # Merge overlapping masks
        print("\n[4/5] Merging overlapping masks...")
        detections = merge_overlapping_masks(detections)
        print(f"  After merging: {len(detections)}")

        # Split disconnected components
        print("\n[5/5] Splitting disconnected mask components...")
        detections = split_disconnected_masks(detections)
        print(f"  After splitting: {len(detections)}")

    # Save masks
    print(f"\nSaving masks to: {args.output_dir}")
    save_masks(detections, args.output_dir, image_np.shape[:2])

    # Create overlay visualization
    print(f"\nCreating overlay visualization...")
    create_overlay_visualization(image, detections, args.output_dir)

    print("\n" + "=" * 60)
    print(f"Segmentation complete! {len(detections)} objects found.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
