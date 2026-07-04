# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Depth-supervised Splatfacto model.

Extends Splatfacto with pixel-wise depth supervision for improved geometry quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils import colormaps


@dataclass
class DepthSplatfactoModelConfig(SplatfactoModelConfig):
    """Depth-supervised Splatfacto Model Config."""

    _target: Type = field(default_factory=lambda: DepthSplatfactoModel)
    depth_loss_mult: float = 0.1
    """Weight for depth loss."""
    depth_loss_type: Literal["l1", "l2", "log_l1"] = "l1"
    """Type of depth loss: l1, l2, or log_l1 (scale-invariant)."""
    depth_tolerance: float = 0.0
    """Ignore depth errors below this threshold (meters). Set to 0 for no tolerance."""

    def __post_init__(self):
        # Ensure depth is rendered during training - this is required for depth loss
        self.output_depth_during_training = True


class DepthSplatfactoModel(SplatfactoModel):
    """Depth-supervised Gaussian Splatting model.

    Extends Splatfacto with pixel-wise depth supervision using ground truth depth maps.
    This helps improve geometry quality and reduce floaters.

    Args:
        config: DepthSplatfacto configuration to instantiate model
    """

    config: DepthSplatfactoModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict including depth loss.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # Get base splatfacto losses
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Add depth loss if ground truth depth is available
        if "depth_image" in batch and "depth" in outputs and self.training:
            depth_loss = self._compute_depth_loss(outputs, batch)
            if depth_loss is not None:
                loss_dict["depth_loss"] = depth_loss

        return loss_dict

    def _compute_depth_loss(self, outputs, batch) -> torch.Tensor | None:
        """Compute pixel-wise depth loss.

        Args:
            outputs: Model outputs containing rendered depth
            batch: Batch containing ground truth depth

        Returns:
            Depth loss tensor, or None if no valid pixels
        """
        pred_depth = outputs["depth"]  # [H, W, 1]
        gt_depth = batch["depth_image"].to(self.device)  # [H, W, 1]

        # Handle shape mismatches (can occur during resolution scheduling)
        if pred_depth.shape != gt_depth.shape:
            # Resize ground truth to match prediction
            gt_depth = torch.nn.functional.interpolate(
                gt_depth.permute(2, 0, 1).unsqueeze(0),
                size=pred_depth.shape[:2],
                mode="nearest",
            ).squeeze(0).permute(1, 2, 0)

        # Create valid mask: GT depth > 0 (valid pixels) and predicted depth > 0
        valid_mask = (gt_depth > 0) & (pred_depth > 0)

        # Optionally apply tolerance threshold
        if self.config.depth_tolerance > 0:
            abs_diff = torch.abs(pred_depth - gt_depth)
            valid_mask = valid_mask & (abs_diff > self.config.depth_tolerance)

        # Check if we have valid pixels
        if valid_mask.sum() == 0:
            return None

        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]

        # Compute loss based on type
        if self.config.depth_loss_type == "l1":
            depth_loss = torch.abs(pred_valid - gt_valid).mean()
        elif self.config.depth_loss_type == "l2":
            depth_loss = ((pred_valid - gt_valid) ** 2).mean()
        elif self.config.depth_loss_type == "log_l1":
            # Scale-invariant log loss
            eps = 1e-6
            depth_loss = torch.abs(
                torch.log(pred_valid + eps) - torch.log(gt_valid + eps)
            ).mean()
        else:
            raise ValueError(f"Unknown depth loss type: {self.config.depth_loss_type}")

        return self.config.depth_loss_mult * depth_loss

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics including depth metrics.

        Args:
            outputs: the output to compute metrics for
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = super().get_metrics_dict(outputs, batch)

        # Add depth metrics if available
        if "depth_image" in batch and "depth" in outputs:
            pred_depth = outputs["depth"]
            gt_depth = batch["depth_image"].to(self.device)

            # Handle shape mismatches
            if pred_depth.shape != gt_depth.shape:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),
                    size=pred_depth.shape[:2],
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)

            # Valid mask
            valid_mask = (gt_depth > 0) & (pred_depth > 0)
            if valid_mask.sum() > 0:
                pred_valid = pred_depth[valid_mask]
                gt_valid = gt_depth[valid_mask]

                # Depth MAE
                metrics_dict["depth_mae"] = torch.abs(pred_valid - gt_valid).mean()

                # Depth RMSE
                metrics_dict["depth_rmse"] = torch.sqrt(((pred_valid - gt_valid) ** 2).mean())

        return metrics_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs including depth comparisons.

        Args:
            outputs: Outputs of the model.
            batch: Batch of data.

        Returns:
            A dictionary of metrics and a dictionary of images.
        """
        metrics, images = super().get_image_metrics_and_images(outputs, batch)

        # Add depth visualization if ground truth available
        if "depth_image" in batch and "depth" in outputs:
            gt_depth = batch["depth_image"].to(self.device)
            pred_depth = outputs["depth"]

            # Handle shape mismatches
            if pred_depth.shape != gt_depth.shape:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth.permute(2, 0, 1).unsqueeze(0),
                    size=pred_depth.shape[:2],
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)

            # Valid mask for metrics
            valid_mask = gt_depth > 0
            if valid_mask.sum() > 0:
                pred_valid = pred_depth[valid_mask]
                gt_valid = gt_depth[valid_mask]

                # Depth MSE metric
                metrics["depth_mse"] = float(
                    torch.nn.functional.mse_loss(pred_valid, gt_valid).cpu()
                )

                # Depth MAE metric
                metrics["depth_mae"] = float(torch.abs(pred_valid - gt_valid).mean().cpu())

            # Create depth visualization
            # Get depth range from valid GT pixels
            if valid_mask.sum() > 0:
                near_plane = float(gt_depth[valid_mask].min().cpu())
                far_plane = float(gt_depth[valid_mask].max().cpu())
            else:
                near_plane = 0.0
                far_plane = 10.0

            # Apply colormaps
            gt_depth_colormap = colormaps.apply_depth_colormap(
                gt_depth,
                near_plane=near_plane,
                far_plane=far_plane,
            )
            pred_depth_colormap = colormaps.apply_depth_colormap(
                pred_depth,
                accumulation=outputs.get("accumulation"),
                near_plane=near_plane,
                far_plane=far_plane,
            )

            # Side-by-side depth comparison (GT | Predicted)
            images["depth"] = torch.cat([gt_depth_colormap, pred_depth_colormap], dim=1)

        return metrics, images
