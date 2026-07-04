"""Edge extraction and IoU comparison for validation."""

from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image


def load_segmentation_mask(seg_path: Path) -> Optional[np.ndarray]:
    """
    Load segmentation map and return it as numpy array.

    Args:
        seg_path: Path to the segmentation PNG file

    Returns:
        RGB segmentation map as numpy array, or None if file doesn't exist
    """
    if not seg_path.exists():
        return None

    img = Image.open(seg_path)
    return np.array(img.convert("RGB"))


def detect_floor_color_from_bottom(segmentation: np.ndarray) -> np.ndarray:
    """
    Detect floor color by sampling the bottom-center pixel.

    In interior renders, the floor is always visible at the bottom of the frame.
    The bottom-center pixel reliably captures the floor color.

    Args:
        segmentation: RGB segmentation map

    Returns:
        Floor color as RGB array
    """
    h, w = segmentation.shape[:2]
    # Sample from bottom row, center column
    bottom_center_y = h - 1
    center_x = w // 2
    return segmentation[bottom_center_y, center_x].copy()


def detect_floor_color(segmentation: np.ndarray, bottom_fraction: float = 0.3) -> Optional[np.ndarray]:
    """
    Detect the floor color by finding the most common color in the bottom portion of the image.

    In typical interior renders, the floor is visible in the lower part of the frame.
    This is more robust than hardcoding a specific color, since segmentation colors
    depend on geometry names in the GLB file which may not be semantic.

    Note: For per-room floor color detection, prefer detect_floor_color_from_bottom()
    which uses the first image's bottom-center pixel for consistent results.

    Args:
        segmentation: RGB segmentation map
        bottom_fraction: Fraction of image height to sample from the bottom (default 0.3 = 30%)

    Returns:
        Floor color as RGB array, or None if cannot determine
    """
    h, w = segmentation.shape[:2]
    bottom_start = int(h * (1 - bottom_fraction))

    # Get the bottom region
    bottom_region = segmentation[bottom_start:, :]

    # Reshape to list of pixels and find most common color
    pixels = bottom_region.reshape(-1, 3)

    # Count unique colors
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    if len(counts) == 0:
        return None

    # Return the most common color in the bottom region
    most_common_idx = np.argmax(counts)
    return unique_colors[most_common_idx]


def create_floor_mask(segmentation: np.ndarray, floor_color: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a binary mask identifying floor pixels from segmentation.

    Args:
        segmentation: RGB segmentation map
        floor_color: RGB color of floor (if None, auto-detect from bottom of image)

    Returns:
        Binary mask where True = floor pixel
    """
    if floor_color is None:
        floor_color = detect_floor_color(segmentation)
        if floor_color is None:
            # Cannot detect floor, return all-False mask
            return np.zeros(segmentation.shape[:2], dtype=bool)

    # Exact color match (segmentation maps have discrete colors)
    is_floor = np.all(segmentation == floor_color, axis=2)

    return is_floor


def create_non_floor_mask(segmentation: np.ndarray, floor_color: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a binary mask for non-floor regions (walls, objects, etc).

    Args:
        segmentation: RGB segmentation map
        floor_color: RGB color of floor (if None, auto-detect from bottom of image)

    Returns:
        Binary mask where True = non-floor pixel (include in comparison)
    """
    return ~create_floor_mask(segmentation, floor_color)


def get_segmentation_path(input_image_path: Path) -> Path:
    """
    Derive segmentation path from input image path.

    Input: .../images/room_name/room_name_0000_with_edges.png
    Output: .../segmentation/room_name/room_name_0000_seg.png
    """
    # Get room name and image index
    filename = input_image_path.stem  # e.g., "master_bedroom_0000_with_edges"
    base_name = filename.replace("_with_edges", "")  # e.g., "master_bedroom_0000"

    # Navigate to segmentation folder (sibling of images/)
    images_dir = input_image_path.parent  # .../images/room_name/
    room_name = images_dir.name
    base_dir = images_dir.parent.parent  # .../

    seg_path = base_dir / "segmentation" / room_name / f"{base_name}_seg.png"
    return seg_path


def get_room_floor_color(input_image_path: Path) -> Optional[np.ndarray]:
    """
    Get the floor color for a room by checking the first image's segmentation.

    This function derives the path to the first segmentation image (index 0000)
    from any input image path in the same room, then samples the bottom-center
    pixel to determine the floor color.

    Args:
        input_image_path: Path to any image in the room

    Returns:
        Floor color RGB as numpy array, or None if segmentation not found
    """
    # Derive path to first image's segmentation
    images_dir = input_image_path.parent
    room_name = images_dir.name
    base_dir = images_dir.parent.parent

    # First image is always index 0000
    first_seg_path = base_dir / "segmentation" / room_name / f"{room_name}_0000_seg.png"

    if not first_seg_path.exists():
        return None

    first_seg = load_segmentation_mask(first_seg_path)
    if first_seg is None:
        return None

    return detect_floor_color_from_bottom(first_seg)


def extract_canny_edges(
    image: np.ndarray,
    low_threshold: float = 0.4,
    high_threshold: float = 0.8,
) -> np.ndarray:
    """
    Extract Canny edges from an image.

    Args:
        image: RGB or grayscale image as numpy array
        low_threshold: Lower threshold for Canny (normalized 0-1)
        high_threshold: Upper threshold for Canny (normalized 0-1)

    Returns:
        Binary edge map (0 or 255)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Convert normalized thresholds to 0-255 range
    low = int(low_threshold * 255)
    high = int(high_threshold * 255)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low, high)

    return edges


def dilate_edges(edges: np.ndarray, dilation_pixels: int = 3) -> np.ndarray:
    """
    Dilate edge map to be tolerant of small shifts.

    Args:
        edges: Binary edge map
        dilation_pixels: Number of pixels to dilate

    Returns:
        Dilated edge map
    """
    if dilation_pixels <= 0:
        return edges

    kernel_size = dilation_pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    return dilated


def compute_edge_iou(
    edges1: np.ndarray,
    edges2: np.ndarray,
    dilation_pixels: int = 3,
    include_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Edge IoU between two edge maps with tolerance.

    Args:
        edges1: First edge map
        edges2: Second edge map
        dilation_pixels: Dilation for shift tolerance
        include_mask: Optional binary mask - only compare edges where mask is True
                      (e.g., exclude floor regions)

    Returns:
        IoU score in [0, 1]
    """
    # Resize if shapes don't match
    if edges1.shape != edges2.shape:
        edges1 = cv2.resize(
            edges1,
            (edges2.shape[1], edges2.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Resize include_mask if provided and shapes don't match
    if include_mask is not None and include_mask.shape != edges1.shape:
        include_mask = cv2.resize(
            include_mask.astype(np.uint8),
            (edges1.shape[1], edges1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Dilate both edge maps for tolerance
    edges1_dilated = dilate_edges(edges1, dilation_pixels)
    edges2_dilated = dilate_edges(edges2, dilation_pixels)

    # Convert to binary masks
    mask1 = edges1_dilated > 127
    mask2 = edges2_dilated > 127

    # Apply include mask if provided (only consider edges in included regions)
    if include_mask is not None:
        mask1 = mask1 & include_mask
        mask2 = mask2 & include_mask

    # Compute IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0  # Both empty = perfect match

    return float(intersection) / float(union)


def compute_edge_recall(
    input_edges: np.ndarray,
    generated_edges: np.ndarray,
    dilation_pixels: int = 3,
    include_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute what fraction of input edges are captured in generated edges.

    This measures: "How many of the expected edges appear in the generated image?"

    Args:
        input_edges: Edge map from input image (ground truth structure)
        generated_edges: Edge map from generated image
        dilation_pixels: Dilation for shift tolerance
        include_mask: Optional binary mask - only check recall for input edges where
                      mask is True (e.g., exclude windows/sky regions)

    Returns:
        Recall score in [0, 1]
    """
    # Resize if shapes don't match
    if input_edges.shape != generated_edges.shape:
        input_edges = cv2.resize(
            input_edges,
            (generated_edges.shape[1], generated_edges.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Resize include_mask if provided and shapes don't match
    if include_mask is not None and include_mask.shape != input_edges.shape:
        include_mask = cv2.resize(
            include_mask.astype(np.uint8),
            (input_edges.shape[1], input_edges.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Dilate generated edges for tolerance (we check if input edges fall within)
    generated_dilated = dilate_edges(generated_edges, dilation_pixels)

    # Convert to binary masks
    input_mask = input_edges > 127
    generated_mask = generated_dilated > 127

    # Apply include mask to input edges (only check recall in valid regions)
    if include_mask is not None:
        input_mask = input_mask & include_mask

    # Compute recall: what fraction of input edges are covered?
    input_edge_count = input_mask.sum()

    if input_edge_count == 0:
        return 1.0  # No input edges = trivially satisfied

    covered = np.logical_and(input_mask, generated_mask).sum()

    return float(covered) / float(input_edge_count)


def validate_generation(
    generated_image: Image.Image,
    input_image_path: Path,
    threshold: float = 0.5,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    dilation_pixels: int = 3,
    exclude_floor: bool = True,
    floor_color: Optional[np.ndarray] = None,
) -> Tuple[bool, float]:
    """
    Validate a generated image by comparing edges to the input image.

    Extracts Canny edges from both input and generated images,
    then computes Edge IoU with dilation tolerance.

    Optionally excludes floor regions from comparison (since generated
    images often have floor textures like parquet that create extra edges).

    Args:
        generated_image: PIL Image of the generated output
        input_image_path: Path to the input image (*_with_edges.png)
        threshold: Minimum Edge IoU score for validation pass
        canny_low: Canny low threshold (normalized, lower for photorealistic)
        canny_high: Canny high threshold (normalized)
        dilation_pixels: Pixel tolerance for edge alignment
        exclude_floor: If True, exclude floor regions from edge comparison
        floor_color: Pre-computed floor color RGB (if None and exclude_floor=True,
                     will auto-detect from current image's segmentation)

    Returns:
        Tuple of (passed, iou_score)
    """
    # Load input image
    input_image = Image.open(input_image_path)
    input_array = np.array(input_image.convert("RGB"))

    # Convert generated image to numpy array
    gen_array = np.array(generated_image.convert("RGB"))

    # Extract edges from both images using same parameters
    input_edges = extract_canny_edges(input_array, canny_low, canny_high)
    generated_edges = extract_canny_edges(gen_array, canny_low, canny_high)

    # Load segmentation mask to exclude floor if requested
    include_mask = None
    if exclude_floor:
        seg_path = get_segmentation_path(input_image_path)
        segmentation = load_segmentation_mask(seg_path)
        if segmentation is not None:
            # Use provided floor_color or auto-detect from segmentation
            include_mask = create_non_floor_mask(segmentation, floor_color)

    # Compute Edge IoU (optionally excluding floor regions)
    iou = compute_edge_iou(input_edges, generated_edges, dilation_pixels, include_mask)

    return iou >= threshold, iou


def validate_generation_recall(
    generated_image: Image.Image,
    input_image_path: Path,
    threshold: float = 0.6,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    dilation_pixels: int = 3,
) -> Tuple[bool, float]:
    """
    Validate using edge recall (what fraction of input edges are captured).

    This is more lenient than IoU - it only checks if the expected structure
    appears in the generated image, not if extra edges are added.

    Args:
        generated_image: PIL Image of the generated output
        input_image_path: Path to the input image (*_with_edges.png)
        threshold: Minimum recall score for validation pass
        canny_low: Canny low threshold (normalized)
        canny_high: Canny high threshold (normalized)
        dilation_pixels: Pixel tolerance for edge alignment

    Returns:
        Tuple of (passed, recall_score)
    """
    # Load input image
    input_image = Image.open(input_image_path)
    input_array = np.array(input_image.convert("RGB"))

    # Convert generated image to numpy array
    gen_array = np.array(generated_image.convert("RGB"))

    # Extract edges from both images
    input_edges = extract_canny_edges(input_array, canny_low, canny_high)
    generated_edges = extract_canny_edges(gen_array, canny_low, canny_high)

    # Compute recall
    recall = compute_edge_recall(input_edges, generated_edges, dilation_pixels)

    return recall >= threshold, recall


def create_comparison_image(
    generated_image: Image.Image,
    input_image_path: Path,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    floor_color: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Create a side-by-side comparison image.

    Layout: [Input Edges | Generated Edges | Difference]
    Colors in difference: Green=input only, Red=generated only, White=both
    Floor regions (excluded from IoU) are shown with cyan tint

    Args:
        generated_image: PIL Image of the generated output
        input_image_path: Path to the input image
        canny_low: Canny low threshold (normalized)
        canny_high: Canny high threshold (normalized)
        floor_color: Pre-computed floor color RGB for visualization (optional)

    Returns:
        PIL Image with comparison layout
    """
    # Load input image
    input_image = Image.open(input_image_path)
    input_array = np.array(input_image.convert("RGB"))

    # Convert generated image to numpy array
    gen_array = np.array(generated_image.convert("RGB"))

    # Extract edges from both images
    input_edges = extract_canny_edges(input_array, canny_low, canny_high)
    generated_edges = extract_canny_edges(gen_array, canny_low, canny_high)

    # Resize generated edges to match input if needed
    if generated_edges.shape != input_edges.shape:
        generated_edges = cv2.resize(
            generated_edges,
            (input_edges.shape[1], input_edges.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Load floor mask for visualization
    floor_mask = None
    seg_path = get_segmentation_path(input_image_path)
    segmentation = load_segmentation_mask(seg_path)
    if segmentation is not None:
        floor_mask = create_floor_mask(segmentation, floor_color)
        # Resize floor mask if needed
        if floor_mask.shape != input_edges.shape:
            floor_mask = cv2.resize(
                floor_mask.astype(np.uint8),
                (input_edges.shape[1], input_edges.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    # Create difference image
    # Green = input only, Red = generated only, White = both
    diff = np.zeros((*input_edges.shape, 3), dtype=np.uint8)

    input_mask = input_edges > 127
    generated_mask = generated_edges > 127

    # Both have edges -> White
    both = input_mask & generated_mask
    diff[both] = [255, 255, 255]

    # Input only -> Green (missing in generated)
    input_only = input_mask & ~generated_mask
    diff[input_only] = [0, 255, 0]

    # Generated only -> Red (extra edges)
    generated_only = ~input_mask & generated_mask
    diff[generated_only] = [255, 0, 0]

    # Convert grayscale edges to RGB for concatenation
    input_rgb = cv2.cvtColor(input_edges, cv2.COLOR_GRAY2RGB)
    generated_rgb = cv2.cvtColor(generated_edges, cv2.COLOR_GRAY2RGB)

    # Apply cyan tint to floor regions to show excluded areas
    if floor_mask is not None:
        # Cyan color for floor overlay: RGB(0, 180, 180)
        floor_tint = np.array([0, 180, 180], dtype=np.uint8)
        alpha = 0.4  # Transparency of the floor overlay

        # Apply tint to input edges panel
        input_rgb[floor_mask] = (
            input_rgb[floor_mask].astype(float) * (1 - alpha) +
            floor_tint.astype(float) * alpha
        ).astype(np.uint8)

        # Apply tint to generated edges panel
        generated_rgb[floor_mask] = (
            generated_rgb[floor_mask].astype(float) * (1 - alpha) +
            floor_tint.astype(float) * alpha
        ).astype(np.uint8)

        # Apply tint to difference panel (floor edges shown as excluded)
        diff[floor_mask] = (
            diff[floor_mask].astype(float) * (1 - alpha) +
            floor_tint.astype(float) * alpha
        ).astype(np.uint8)

    # Concatenate horizontally
    comparison = np.hstack([input_rgb, generated_rgb, diff])

    return Image.fromarray(comparison)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (RGB)."""
    return np.array(image.convert("RGB"))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy array (RGB) to PIL Image."""
    return Image.fromarray(array)
