#!/usr/bin/env python3
"""Standalone Depth Pro estimator script.

This script is designed to run in the 'worldmesh-depth-pro' conda environment.
It takes an input image and outputs metric depth to a numpy file.

Usage:
    conda run -n worldmesh-depth-pro python depth_estimator.py --input image.png --output depth.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def get_torch_device() -> torch.device:
    """Get the best available Torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def estimate_depth(image_path: Path, output_path: Path, verbose: bool = False) -> None:
    """
    Estimate metric depth from an image using Depth Pro.

    Args:
        image_path: Path to input image (PNG, JPEG, etc.)
        output_path: Path to output .npz file containing 'depth' array
        verbose: Print progress messages
    """
    # Import depth_pro here (only available in depth-pro environment)
    try:
        import depth_pro
        from depth_pro.depth_pro import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT
    except ImportError as e:
        print(f"Error: Cannot import depth_pro. Are you running in the worldmesh-depth-pro conda environment?", file=sys.stderr)
        print(f"Try: conda run -n worldmesh-depth-pro python {sys.argv[0]} ...", file=sys.stderr)
        sys.exit(1)

    device = get_torch_device()
    if verbose:
        print(f"Using device: {device}")

    # Find checkpoint path (relative to ml-depth-pro directory)
    project_root = Path(__file__).parent.parent.parent  # repo root
    checkpoint_path = project_root / "ml-depth-pro" / "checkpoints" / "depth_pro.pt"

    if not checkpoint_path.exists():
        print(f"Error: Depth Pro checkpoint not found at {checkpoint_path}", file=sys.stderr)
        print(f"Download it with: cd ml-depth-pro && source get_pretrained_models.sh", file=sys.stderr)
        sys.exit(1)

    # Create config with absolute checkpoint path
    config = DepthProConfig(
        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset,
        checkpoint_uri=str(checkpoint_path),
    )

    # Load model
    if verbose:
        print(f"Loading Depth Pro model from {checkpoint_path}...")
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=device,
        precision=torch.half,
    )
    model.eval()

    # Load image
    if verbose:
        print(f"Loading image: {image_path}")
    try:
        image, _, f_px = depth_pro.load_rgb(image_path)
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

    # Run inference
    if verbose:
        print("Running depth estimation...")
    with torch.no_grad():
        prediction = model.infer(transform(image), f_px=f_px)

    # Extract depth (metric, in meters)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    focallength_px = prediction["focallength_px"].detach().cpu().item() if prediction["focallength_px"] is not None else None

    # Save to output
    if verbose:
        print(f"Saving depth to: {output_path}")
        print(f"  Depth shape: {depth.shape}")
        print(f"  Depth range: [{depth.min():.3f}, {depth.max():.3f}] meters")
        if focallength_px:
            print(f"  Estimated focal length: {focallength_px:.2f} pixels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, depth=depth, focallength_px=np.array(focallength_px if focallength_px else 0))

    if verbose:
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate metric depth from an image using Depth Pro."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Path to output .npz file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress messages",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    estimate_depth(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
