#!/usr/bin/env python3
"""
Simple test script for SAM 3 on a local image.

Usage:
    python test_sam3.py                           # Uses default prompt "furniture"
    python test_sam3.py --prompt "chair"          # Custom prompt
    python test_sam3.py --prompt "person" --save  # Save output instead of displaying
"""

import argparse
import torch
import numpy as np
from PIL import Image

# SAM 3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    parser = argparse.ArgumentParser(description="Test SAM 3 on an image")
    parser.add_argument(
        "--image",
        type=str,
        default="/mnt/hdd/scenes/generations/Flux2_00013_.png",
        help="Path to input image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="furniture",
        help="Text prompt for segmentation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to file instead of displaying",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sam3_output.png",
        help="Output file path (when --save is used)",
    )
    args = parser.parse_args()

    print(f"Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    print(f"Setting image for inference...")
    inference_state = processor.set_image(image)

    print(f"Running segmentation with prompt: '{args.prompt}'")
    output = processor.set_text_prompt(state=inference_state, prompt=args.prompt)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print(f"\nResults:")
    print(f"  Found {len(masks)} object(s)")
    for i, score in enumerate(scores):
        print(f"  Object {i}: score={score:.3f}, box={boxes[i].tolist()}")

    # Visualize results
    if len(masks) > 0:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Image with masks overlay
        axes[1].imshow(image)

        # Overlay each mask with a different color
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(masks), 10)))
        for i, mask in enumerate(masks):
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze()
            colored_mask = np.zeros((*mask_np.shape, 4))
            colored_mask[mask_np > 0.5] = [*colors[i % 10][:3], 0.5]
            axes[1].imshow(colored_mask)

        # Draw bounding boxes
        for i, box in enumerate(boxes):
            box_np = box.cpu().numpy() if torch.is_tensor(box) else np.array(box)
            x1, y1, x2, y2 = box_np
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=colors[i % 10], linewidth=2
            )
            axes[1].add_patch(rect)
            axes[1].text(
                x1, y1 - 5, f"{scores[i]:.2f}",
                color=colors[i % 10], fontsize=10, fontweight="bold"
            )

        axes[1].set_title(f"Segmentation: '{args.prompt}'")
        axes[1].axis("off")

        plt.tight_layout()

        if args.save:
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"\nSaved output to: {args.output}")
        else:
            plt.show()
    else:
        print(f"\nNo objects found matching '{args.prompt}'")


if __name__ == "__main__":
    main()
