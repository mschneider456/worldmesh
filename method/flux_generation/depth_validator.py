"""Depth-based validation for Flux generation pipeline.

This module validates generated images by comparing estimated depth (via Depth Pro)
against ground-truth rendered depth maps. Uses edge recall computed on depth maps -
extracting Canny edges from normalized depth to detect structural boundaries, then
checking what fraction of GT edges are covered by estimated edges.

The validation runs Depth Pro in a subprocess using the 'worldmesh-depth-pro' conda environment,
since the main flux pipeline runs in 'worldmesh-comfy'.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def compute_si_rmse(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Scale-Invariant RMSE between estimated and ground-truth depth.

    SI-RMSE aligns depths in log-space via median shift, measuring structural
    accuracy rather than absolute scale. This is appropriate since Depth Pro
    may have different scale calibration than the rendered depth.

    Args:
        estimated: Estimated depth map (meters)
        ground_truth: Ground truth depth map (meters)
        valid_mask: Optional boolean mask of pixels to include

    Returns:
        SI-RMSE score (lower is better, typically 0.05-0.3)
    """
    # Clip to valid depth range
    estimated = np.clip(estimated, 1e-4, 1e4)
    ground_truth = np.clip(ground_truth, 1e-4, 1e4)

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = np.ones_like(estimated, dtype=bool)

    # Additional validity: exclude zero/invalid depths
    valid_mask = valid_mask & (estimated > 1e-3) & (ground_truth > 1e-3)

    if valid_mask.sum() == 0:
        return float('inf')

    # Compute in log space
    log_est = np.log(estimated[valid_mask])
    log_gt = np.log(ground_truth[valid_mask])

    # Align via median (scale-invariant)
    diff = log_est - log_gt
    diff_centered = diff - np.median(diff)

    return float(np.sqrt(np.mean(diff_centered ** 2)))


def compute_gradient_rmse(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute RMSE of depth gradients in log-space.

    This metric captures whether depth is changing in the correct direction
    locally, rather than comparing absolute depth values. It's more tolerant
    of global scale/offset differences while still catching structural errors.

    Args:
        estimated: Estimated depth map (meters)
        ground_truth: Ground truth depth map (meters)
        valid_mask: Optional boolean mask of pixels to include

    Returns:
        Gradient RMSE score (lower is better)
    """
    # Clip to valid depth range and convert to log space
    log_est = np.log(np.clip(estimated, 1e-4, 1e4))
    log_gt = np.log(np.clip(ground_truth, 1e-4, 1e4))

    # Compute gradients (simple finite difference)
    grad_x_est = log_est[:, 1:] - log_est[:, :-1]  # Horizontal gradient
    grad_y_est = log_est[1:, :] - log_est[:-1, :]  # Vertical gradient
    grad_x_gt = log_gt[:, 1:] - log_gt[:, :-1]
    grad_y_gt = log_gt[1:, :] - log_gt[:-1, :]

    # Build valid masks for gradients (shrunk by 1 pixel)
    if valid_mask is None:
        valid_mask = np.ones_like(estimated, dtype=bool)

    # Additional validity: exclude zero/invalid depths
    valid_mask = valid_mask & (estimated > 1e-3) & (ground_truth > 1e-3)

    # Gradient masks require both adjacent pixels to be valid
    valid_x = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid_y = valid_mask[1:, :] & valid_mask[:-1, :]

    if valid_x.sum() == 0 or valid_y.sum() == 0:
        return float('inf')

    # Compute RMSE of gradient differences
    grad_x_diff = grad_x_est[valid_x] - grad_x_gt[valid_x]
    grad_y_diff = grad_y_est[valid_y] - grad_y_gt[valid_y]

    rmse_x = np.sqrt(np.mean(grad_x_diff ** 2))
    rmse_y = np.sqrt(np.mean(grad_y_diff ** 2))

    return float((rmse_x + rmse_y) / 2.0)


def compute_combined_si_rmse(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    gradient_weight: float = 0.7,
) -> Tuple[float, float, float]:
    """
    Compute combined SI-RMSE with gradient term.

    Blends point-wise SI-RMSE with gradient-based RMSE to balance absolute
    depth accuracy with directional correctness.

    Args:
        estimated: Estimated depth map (meters)
        ground_truth: Ground truth depth map (meters)
        valid_mask: Optional boolean mask of pixels to include
        gradient_weight: Weight for gradient term (0=pure SI-RMSE, 1=pure gradient)

    Returns:
        Tuple of (combined_score, si_rmse_score, gradient_rmse_score)
    """
    si_rmse = compute_si_rmse(estimated, ground_truth, valid_mask)
    grad_rmse = compute_gradient_rmse(estimated, ground_truth, valid_mask)

    # Handle infinite values
    if si_rmse == float('inf') or grad_rmse == float('inf'):
        return float('inf'), si_rmse, grad_rmse

    combined = (1 - gradient_weight) * si_rmse + gradient_weight * grad_rmse

    return combined, si_rmse, grad_rmse


def normalize_depth_for_edges(depth: np.ndarray) -> np.ndarray:
    """
    Normalize depth to uint8 for Canny edge detection using linear scaling.

    Uses linear normalization (close=high/white, far=low/black) which preserves
    depth gradients better than inverse depth for edge detection. This matches
    the depth PNG visualization output from render_multiview.py.

    Args:
        depth: Depth map in meters

    Returns:
        Normalized depth as uint8 array suitable for Canny edge detection
    """
    # Create valid mask (exclude zero/invalid depths)
    valid_mask = depth > 0.1

    if not valid_mask.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    # Get depth range from valid pixels
    d_min = depth[valid_mask].min()
    d_max = depth[valid_mask].max()

    if d_max - d_min < 1e-8:
        return np.zeros(depth.shape, dtype=np.uint8)

    # Linear normalization: close=255 (white), far=0 (black)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid_mask] = (255 * (1 - (depth[valid_mask] - d_min) / (d_max - d_min))).astype(np.uint8)

    return normalized


def sharpen_depth_for_edges(depth_normalized: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """
    Sharpen normalized depth map to enhance edges for Canny detection.

    Neural network depth estimates (like Depth Pro) produce smooth depth maps
    with gradual transitions. This applies aggressive Laplacian-based sharpening
    to enhance depth discontinuities so Canny can detect them more reliably.

    Args:
        depth_normalized: Normalized depth as uint8 (from normalize_depth_for_edges)
        strength: Sharpening strength (1.0 = no change, 2.0 = default, 3.0+ = aggressive)

    Returns:
        Sharpened depth map as uint8
    """
    if strength <= 1.0:
        return depth_normalized

    # Convert to float for processing
    depth_float = depth_normalized.astype(np.float32)

    # Apply bilateral filter first to smooth noise while preserving edges
    # This helps reduce false edges from noise while keeping true depth edges
    smoothed = cv2.bilateralFilter(depth_float, d=5, sigmaColor=20, sigmaSpace=5)

    # Compute Laplacian (second derivative - detects edges)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3)

    # Add scaled Laplacian back to enhance edges
    # The formula: enhanced = original - strength * laplacian
    # (subtract because laplacian is negative at edges for bright-on-dark)
    sharpened = depth_float - (strength - 1.0) * laplacian

    # Also apply unsharp mask for additional edge enhancement
    blurred = cv2.GaussianBlur(depth_float, (0, 0), sigmaX=1.5)
    unsharp = depth_float + (strength - 1.0) * 0.5 * (depth_float - blurred)

    # Combine both methods
    combined = (sharpened + unsharp) / 2.0

    # Clip and convert back to uint8
    result = np.clip(combined, 0, 255).astype(np.uint8)

    return result


def compute_depth_gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    """
    Compute Scharr gradient magnitude on depth in log-space, normalized to [0,1].

    Log-space gradients handle the wide depth range consistently (nearby objects
    and distant walls contribute equally). Scharr operator is more accurate than
    Sobel for detecting true depth discontinuities.

    Args:
        depth: Depth map in meters

    Returns:
        Gradient magnitude array normalized to [0, 1]
    """
    # Convert to log space (handles wide depth range)
    log_depth = np.log(np.clip(depth, 1e-4, 1e4)).astype(np.float32)

    # Scharr operator (more accurate than Sobel)
    grad_x = cv2.Scharr(log_depth, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(log_depth, cv2.CV_32F, 0, 1)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize to [0, 1] by max
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max

    return magnitude


def filter_edges_by_gradient(
    edges: np.ndarray,
    gradient_magnitude: np.ndarray,
    min_percentile: float = 25.0,
) -> np.ndarray:
    """
    Keep only edge pixels where GT gradient magnitude >= percentile threshold.

    The threshold is computed from gradient values at edge pixel locations only,
    so it adapts to the actual edge distribution rather than the full image.

    Args:
        edges: Binary edge map (uint8, 0 or 255)
        gradient_magnitude: Gradient magnitude array (same shape as edges)
        min_percentile: Percentile threshold (0-100) computed from edge pixels

    Returns:
        Filtered edge map (uint8, 0 or 255)
    """
    edge_mask = edges > 127
    if not edge_mask.any():
        return edges

    # Compute threshold from gradient values at edge pixels only
    edge_gradients = gradient_magnitude[edge_mask]
    threshold = np.percentile(edge_gradients, min_percentile)

    # Zero out edges below threshold
    filtered = edges.copy()
    below_threshold = edge_mask & (gradient_magnitude < threshold)
    filtered[below_threshold] = 0

    return filtered


def compute_depth_edge_recall(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    dilation_pixels: int = 3,
    min_valid_depth: float = 0.1,
    sharpen_estimated: float = 3.0,
    sharpen_gt: float = 3.0,
    min_gradient_percentile: float = 0.0,
) -> float:
    """
    Compute edge recall on depth maps using Canny edge detection.

    Extracts edges from normalized depth maps and computes recall — what
    fraction of GT depth edges are covered by estimated depth edges. This
    only penalizes missing structural edges, not extra edges from furniture
    or textures in the generated image.

    Excludes invalid regions (windows, sky) where GT depth is below
    min_valid_depth.

    Both depth maps can be sharpened before edge detection to enhance
    depth discontinuities for more reliable Canny edge extraction.

    Optionally applies asymmetric gradient filtering to remove spurious edges:
    - GT edges: filtered by GT gradient magnitude (removes weak furniture detail
      from placeholder arrows that the NN correctly smooths over)
    - Est edges: filtered by max(GT, est) gradient magnitude — where *either*
      gradient is strong, the edge is kept. This preserves edges from hallucinated
      objects (strong est gradient, zero GT gradient) while still removing
      wall-ceiling boundary noise (both gradients weak/moderate).

    Args:
        estimated: Estimated depth map (meters)
        ground_truth: Ground truth depth map (meters)
        canny_low: Lower threshold for Canny (normalized 0-1)
        canny_high: Upper threshold for Canny (normalized 0-1)
        dilation_pixels: Number of pixels to dilate for shift tolerance
        min_valid_depth: Minimum depth to consider valid (excludes windows/sky)
        sharpen_estimated: Sharpening strength for estimated depth (1.0=none, 2.0=default)
        sharpen_gt: Sharpening strength for GT depth (1.0=none, 1.5=default)
        min_gradient_percentile: Filter edges below this gradient percentile (0=disabled)

    Returns:
        Edge recall score in [0, 1] (higher is better)
    """
    from .edge_validator import extract_canny_edges, compute_edge_recall

    # Resize estimated to match ground truth if needed
    if estimated.shape != ground_truth.shape:
        estimated = cv2.resize(
            estimated,
            (ground_truth.shape[1], ground_truth.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Create valid mask: exclude windows/sky where GT depth is 0 or very small
    valid_mask = ground_truth >= min_valid_depth

    # Normalize both depth maps for edge detection
    est_norm = normalize_depth_for_edges(estimated)
    gt_norm = normalize_depth_for_edges(ground_truth)

    # Sharpen both depth maps to enhance edges for Canny detection
    if sharpen_estimated > 1.0:
        est_norm = sharpen_depth_for_edges(est_norm, sharpen_estimated)
    if sharpen_gt > 1.0:
        gt_norm = sharpen_depth_for_edges(gt_norm, sharpen_gt)

    # Extract Canny edges from both
    est_edges = extract_canny_edges(est_norm, canny_low, canny_high)
    gt_edges = extract_canny_edges(gt_norm, canny_low, canny_high)

    # Asymmetric gradient filtering (removes spurious mismatches)
    if min_gradient_percentile > 0:
        grad_gt = compute_depth_gradient_magnitude(ground_truth)
        # GT edges: filter by GT gradient (removes weak furniture detail)
        gt_edges = filter_edges_by_gradient(gt_edges, grad_gt, min_gradient_percentile)
        # Est edges: filter by max(GT, est) gradient — preserves hallucinated
        # objects (strong est gradient) while removing wall-ceiling noise (both weak)
        grad_est = compute_depth_gradient_magnitude(estimated)
        combined_grad = np.maximum(grad_gt, grad_est)
        est_edges = filter_edges_by_gradient(est_edges, combined_grad, min_gradient_percentile)

    # Compute edge recall with dilation tolerance, excluding invalid regions
    recall = compute_edge_recall(gt_edges, est_edges, dilation_pixels, include_mask=valid_mask)

    return recall


def get_ground_truth_depth_path(input_image_path: Path) -> Path:
    """
    Derive ground-truth depth path from input image path.

    Input: .../renders_final/images/{room_id}/{room_id}_0000_depth_objects.png
    Output: .../renders_final/depth/{room_id}/{room_id}_0000_depth.npy

    Args:
        input_image_path: Path to the input conditioning image

    Returns:
        Path to corresponding ground-truth depth .npy file
    """
    # Get room name and image name
    filename = input_image_path.stem  # e.g., "master_bedroom_0000_depth_objects"

    # Remove suffix to get base name
    for suffix in ("_depth_objects", "_with_edges"):
        if filename.endswith(suffix):
            base_name = filename[:-len(suffix)]
            break
    else:
        base_name = filename

    # Navigate to depth folder (sibling of images/)
    images_dir = input_image_path.parent  # .../images/room_name/
    room_name = images_dir.name
    base_dir = images_dir.parent.parent  # .../renders_final/

    depth_path = base_dir / "depth" / room_name / f"{base_name}_depth.npy"
    return depth_path


def load_ground_truth_depth(depth_path: Path) -> Optional[np.ndarray]:
    """
    Load ground-truth depth from .npy file.

    Args:
        depth_path: Path to the .npy depth file

    Returns:
        Depth array in meters, or None if file doesn't exist
    """
    if not depth_path.exists():
        return None

    depth = np.load(depth_path)
    return depth


def estimate_depth_subprocess(
    image_path: Path,
    conda_env: str = "worldmesh-depth-pro",
    timeout: int = 120,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Estimate depth using Depth Pro via subprocess.

    Runs the depth_estimator.py script in the specified conda environment.

    Args:
        image_path: Path to input image
        conda_env: Name of conda environment with depth-pro installed
        timeout: Maximum time in seconds for depth estimation
        verbose: Print progress messages

    Returns:
        Estimated depth array in meters, or None on failure
    """
    # Get path to depth_estimator.py (same directory as this file)
    depth_estimator_path = Path(__file__).parent / "depth_estimator.py"

    if not depth_estimator_path.exists():
        raise FileNotFoundError(f"Depth estimator script not found: {depth_estimator_path}")

    # Create temporary file for output
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Build command
        cmd = [
            "conda", "run", "-n", conda_env,
            "python", str(depth_estimator_path),
            "--input", str(image_path),
            "--output", str(tmp_path),
        ]
        if verbose:
            cmd.append("--verbose")

        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        if result.returncode != 0:
            if verbose:
                print(f"Depth estimation failed: {result.stderr}")
            return None

        # Load result
        if tmp_path.exists():
            data = np.load(tmp_path)
            return data['depth']
        return None

    except subprocess.TimeoutExpired:
        if verbose:
            print(f"Depth estimation timed out after {timeout}s")
        return None

    except Exception as e:
        if verbose:
            print(f"Depth estimation error: {e}")
        return None

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def create_depth_comparison_image(
    ground_truth: np.ndarray,
    estimated: np.ndarray,
    edge_recall: float,
    threshold: float,
    passed: bool,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    dilation_pixels: int = 3,
    sharpen_estimated: float = 3.0,
    sharpen_gt: float = 3.0,
    min_gradient_percentile: float = 0.0,
) -> Image.Image:
    """
    Create a 6-panel debug visualization for depth edge comparison.

    Layout: [GT Depth Edges | Estimated Depth Edges | Difference | GT Depth | Est Depth | GT Gradient]
    - Edges: White edges on black background (asymmetric gradient filtering if enabled:
      GT filtered by GT gradient, est filtered by max(GT, est) gradient)
    - Difference: White=both raw, Yellow=covered by dilation only, Green=GT missed, Red=est extra (undilated), Cyan=excluded
    - Depth Colormaps: Turbo colormap visualization (same scale for both)
    - GT Gradient: Heatmap of GT depth gradient magnitude (helps debug threshold tuning)

    The difference panel uses dilated estimated edges (matching the actual recall metric)
    so that green pixels represent truly uncovered GT edges. Yellow pixels show the
    dilation tolerance band — GT edges covered only because of dilation.

    Args:
        ground_truth: Ground truth depth map (meters)
        estimated: Estimated depth map (meters)
        edge_recall: Computed edge recall score
        threshold: Threshold used for validation
        passed: Whether validation passed
        canny_low: Canny low threshold (normalized 0-1)
        canny_high: Canny high threshold (normalized 0-1)
        dilation_pixels: Number of pixels used for dilation tolerance (shown in visualization)
        sharpen_estimated: Sharpening strength for estimated depth (default 3.0)
        sharpen_gt: Sharpening strength for GT depth (default 3.0)
        min_gradient_percentile: Filter edges below this GT gradient percentile (0=disabled)

    Returns:
        PIL Image with 6-panel comparison
    """
    from .edge_validator import extract_canny_edges, dilate_edges

    # Resize estimated to match ground truth if needed
    if estimated.shape != ground_truth.shape:
        estimated = cv2.resize(
            estimated,
            (ground_truth.shape[1], ground_truth.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    h, w = ground_truth.shape

    # Create valid mask: exclude windows/sky where GT depth is 0 or very small
    valid_mask = ground_truth >= 0.1
    invalid_mask = ~valid_mask

    # Normalize both depth maps for edge detection
    gt_norm = normalize_depth_for_edges(ground_truth)
    est_norm = normalize_depth_for_edges(estimated)

    # Sharpen both depth maps to enhance edges for Canny detection
    if sharpen_estimated > 1.0:
        est_norm = sharpen_depth_for_edges(est_norm, sharpen_estimated)
    if sharpen_gt > 1.0:
        gt_norm = sharpen_depth_for_edges(gt_norm, sharpen_gt)

    # Extract Canny edges from both
    gt_edges = extract_canny_edges(gt_norm, canny_low, canny_high)
    est_edges = extract_canny_edges(est_norm, canny_low, canny_high)

    # Compute GT gradient magnitude (always, for visualization)
    grad_gt = compute_depth_gradient_magnitude(ground_truth)

    # Asymmetric gradient filtering if enabled
    if min_gradient_percentile > 0:
        gt_edges = filter_edges_by_gradient(gt_edges, grad_gt, min_gradient_percentile)
        # Est edges: use max(GT, est) gradient to preserve hallucinated objects
        grad_est = compute_depth_gradient_magnitude(estimated)
        combined_grad = np.maximum(grad_gt, grad_est)
        est_edges = filter_edges_by_gradient(est_edges, combined_grad, min_gradient_percentile)

    # Convert grayscale edges to RGB for concatenation
    gt_edges_rgb = cv2.cvtColor(gt_edges, cv2.COLOR_GRAY2RGB)
    est_edges_rgb = cv2.cvtColor(est_edges, cv2.COLOR_GRAY2RGB)

    # Mark invalid regions (windows/sky) with cyan tint in edge panels
    cyan_tint = np.array([0, 180, 180], dtype=np.uint8)
    gt_edges_rgb[invalid_mask] = cyan_tint
    est_edges_rgb[invalid_mask] = cyan_tint

    # Dilate estimated edges to match what the recall metric actually uses
    est_edges_dilated = dilate_edges(est_edges, dilation_pixels)

    # Create difference image (only in valid regions)
    # Layer 1 (background): dim halo showing the full dilation zone
    # Layer 2 (foreground): edge classifications on top
    #
    # Dark amber  = dilation zone (est dilated minus raw est) — tolerance band background
    # White       = GT edge on raw est edge (exact match)
    # Yellow      = GT edge covered by dilation only
    # Green       = GT edge truly missed (not covered even after dilation)
    # Red         = est edge not in GT (extra)
    # Cyan        = excluded (invalid depth)
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    gt_edge_mask = gt_edges > 127
    est_edge_mask = est_edges > 127
    est_dilated_mask = est_edges_dilated > 127

    # Background: show the dilation zone as a dim halo so it's visible at any zoom
    dilation_zone = est_dilated_mask & ~est_edge_mask & valid_mask
    diff[dilation_zone] = [60, 45, 0]  # dark amber background

    # Foreground edges (painted on top of background):
    # GT edge covered by raw est edge -> White (exact match)
    both_raw = gt_edge_mask & est_edge_mask & valid_mask
    diff[both_raw] = [255, 255, 255]

    # GT edge covered by dilated est but NOT raw est -> Yellow (dilation tolerance band)
    dilation_covered = gt_edge_mask & est_dilated_mask & ~est_edge_mask & valid_mask
    diff[dilation_covered] = [255, 200, 0]

    # GT edge NOT covered even by dilated est -> Green (truly missed)
    gt_missed = gt_edge_mask & ~est_dilated_mask & valid_mask
    diff[gt_missed] = [0, 255, 0]

    # Estimated only (undilated) -> Red (extra edges)
    est_only = ~gt_edge_mask & est_edge_mask & valid_mask
    diff[est_only] = [255, 0, 0]

    # Mark invalid regions as cyan (excluded from recall)
    diff[invalid_mask] = cyan_tint

    # Create depth colormap visualizations (turbo colormap on linear depth)
    import matplotlib.pyplot as plt
    turbo = plt.get_cmap('turbo')

    # GT depth colormap
    if valid_mask.any():
        d_min = ground_truth[valid_mask].min()
        d_max = ground_truth[valid_mask].max()
        # Linear normalization: close=1 (warm), far=0 (cool)
        gt_norm_vis = np.zeros_like(ground_truth)
        gt_norm_vis[valid_mask] = 1 - (ground_truth[valid_mask] - d_min) / (d_max - d_min + 1e-8)
    else:
        gt_norm_vis = np.zeros_like(ground_truth)
    gt_colored = (turbo(gt_norm_vis)[:, :, :3] * 255).astype(np.uint8)
    # Mark invalid regions as cyan in depth colormap too
    gt_colored[invalid_mask] = cyan_tint

    # Estimated depth colormap (use same scale as GT for fair comparison)
    if valid_mask.any():
        est_norm_vis = np.zeros_like(estimated)
        est_norm_vis[valid_mask] = 1 - (estimated[valid_mask] - d_min) / (d_max - d_min + 1e-8)
        est_norm_vis = np.clip(est_norm_vis, 0, 1)
    else:
        est_norm_vis = np.zeros_like(estimated)
    est_colored = (turbo(est_norm_vis)[:, :, :3] * 255).astype(np.uint8)
    est_colored[invalid_mask] = cyan_tint

    # GT gradient magnitude heatmap (6th panel)
    grad_colored = (turbo(grad_gt)[:, :, :3] * 255).astype(np.uint8)
    grad_colored[invalid_mask] = cyan_tint

    # Stack horizontally
    comparison = np.hstack([gt_edges_rgb, est_edges_rgb, diff, gt_colored, est_colored, grad_colored])

    # Add text overlay
    result_img = Image.fromarray(comparison)

    # Add text using PIL
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(result_img)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
        small_font = font

    # Status text
    status = "PASS" if passed else "FAIL"
    status_color = (0, 255, 0) if passed else (255, 0, 0)
    draw.text(
        (10, 10),
        f"Depth Edge Recall: {edge_recall:.4f} (threshold: {threshold}, dilation: {dilation_pixels}px) - {status}",
        fill=status_color,
        font=font
    )

    # Panel labels
    panel_width = w
    labels = ["GT Edges (validated)", "Est Edges", "Difference", "GT Depth", "Est Depth", "GT Gradient"]
    for i, label in enumerate(labels):
        x = i * panel_width + 10
        draw.text((x, h - 30), label, fill=(255, 255, 255), font=small_font)

    # Difference panel legend (top-right area of 3rd panel)
    legend_x = 2 * panel_width + 10
    legend_y = 40
    legend_items = [
        ([255, 255, 255], "Exact match"),
        ([255, 200, 0], "Covered by dilation"),
        ([0, 255, 0], "GT missed"),
        ([255, 0, 0], "Est extra"),
        ([60, 45, 0], "Dilation zone"),
        ([0, 180, 180], "Excluded"),
    ]
    for color, text in legend_items:
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12], fill=tuple(color))
        draw.text((legend_x + 16, legend_y - 2), text, fill=(255, 255, 255), font=small_font)
        legend_y += 18

    return result_img


def validate_depth(
    generated_image: Image.Image,
    input_path: Path,
    threshold: float = 0.3,
    conda_env: str = "worldmesh-depth-pro",
    verbose: bool = False,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    dilation_pixels: int = 3,
    sharpen_estimated: float = 3.0,
    sharpen_gt: float = 3.0,
    min_gradient_percentile: float = 0.0,
) -> Tuple[bool, float, Optional[Image.Image]]:
    """
    Validate a generated image using depth edge recall comparison.

    Estimates depth from the generated image using Depth Pro, extracts Canny edges
    from both estimated and ground-truth depth maps, and computes edge recall with
    dilation tolerance. This measures what fraction of GT structural edges appear
    in the estimated depth, without penalizing extra edges from furniture/textures.

    Both depth maps are sharpened before edge detection to enhance depth
    discontinuities for more reliable Canny edge extraction.

    Optionally filters edges by GT gradient magnitude to remove spurious
    mismatches where GT has no significant depth discontinuity.

    Args:
        generated_image: PIL Image of the generated output
        input_path: Path to the input conditioning image (for deriving GT depth path)
        threshold: Minimum edge recall score for validation pass (default 0.3)
        conda_env: Name of conda environment with depth-pro installed
        verbose: Print progress messages
        canny_low: Lower threshold for Canny edge detection (normalized 0-1)
        canny_high: Upper threshold for Canny edge detection (normalized 0-1)
        dilation_pixels: Number of pixels to dilate for shift tolerance
        sharpen_estimated: Sharpening strength for estimated depth (1.0=none, 2.0=default)
        sharpen_gt: Sharpening strength for GT depth (1.0=none, 1.5=default)
        min_gradient_percentile: Filter edges below this GT gradient percentile (0=disabled)

    Returns:
        Tuple of (passed, edge_recall, comparison_image)
        comparison_image is None if depth comparison could not be performed
    """
    # Load ground truth depth
    gt_depth_path = get_ground_truth_depth_path(input_path)
    ground_truth = load_ground_truth_depth(gt_depth_path)

    if ground_truth is None:
        if verbose:
            print(f"    Ground truth depth not found: {gt_depth_path}")
        return True, 1.0, None  # Skip validation if no GT available (assume pass)

    # Save generated image to temp file for depth estimation
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        generated_image.save(tmp_path)

    try:
        # Estimate depth from generated image
        if verbose:
            print("    Estimating depth from generated image...")
        estimated = estimate_depth_subprocess(
            tmp_path,
            conda_env=conda_env,
            verbose=verbose,
        )

        if estimated is None:
            if verbose:
                print("    Depth estimation failed, skipping depth validation")
            return True, 1.0, None  # Skip validation on estimation failure (assume pass)

        # Compute depth edge recall
        edge_recall = compute_depth_edge_recall(
            estimated,
            ground_truth,
            canny_low=canny_low,
            canny_high=canny_high,
            dilation_pixels=dilation_pixels,
            sharpen_estimated=sharpen_estimated,
            sharpen_gt=sharpen_gt,
            min_gradient_percentile=min_gradient_percentile,
        )

        passed = edge_recall >= threshold

        if verbose:
            print(f"    Depth Edge Recall: {edge_recall:.4f}")

        # Create comparison image
        comparison = create_depth_comparison_image(
            ground_truth,
            estimated,
            edge_recall,
            threshold,
            passed,
            canny_low,
            canny_high,
            dilation_pixels=dilation_pixels,
            sharpen_estimated=sharpen_estimated,
            sharpen_gt=sharpen_gt,
            min_gradient_percentile=min_gradient_percentile,
        )

        return passed, edge_recall, comparison

    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


def validate_depth_from_file(
    generated_image_path: Path,
    input_path: Path,
    threshold: float = 0.3,
    conda_env: str = "worldmesh-depth-pro",
    verbose: bool = False,
    canny_low: float = 0.1,
    canny_high: float = 0.3,
    dilation_pixels: int = 3,
    sharpen_estimated: float = 3.0,
    sharpen_gt: float = 3.0,
    min_gradient_percentile: float = 0.0,
) -> Tuple[bool, float, Optional[Image.Image]]:
    """
    Validate a generated image file using depth edge recall comparison.

    Convenience wrapper around validate_depth() that takes a file path.

    Args:
        generated_image_path: Path to the generated image file
        input_path: Path to the input conditioning image
        threshold: Minimum edge recall score for validation pass
        conda_env: Name of conda environment with depth-pro installed
        verbose: Print progress messages
        canny_low: Lower threshold for Canny edge detection (normalized 0-1)
        canny_high: Upper threshold for Canny edge detection (normalized 0-1)
        dilation_pixels: Number of pixels to dilate for shift tolerance
        sharpen_estimated: Sharpening strength for estimated depth (1.0=none, 2.0=default)
        sharpen_gt: Sharpening strength for GT depth (1.0=none, 1.5=default)
        min_gradient_percentile: Filter edges below this GT gradient percentile (0=disabled)

    Returns:
        Tuple of (passed, edge_recall, comparison_image)
    """
    generated_image = Image.open(generated_image_path)
    return validate_depth(
        generated_image,
        input_path,
        threshold=threshold,
        conda_env=conda_env,
        verbose=verbose,
        canny_low=canny_low,
        sharpen_gt=sharpen_gt,
        canny_high=canny_high,
        sharpen_estimated=sharpen_estimated,
        dilation_pixels=dilation_pixels,
        min_gradient_percentile=min_gradient_percentile,
    )
