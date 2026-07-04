# Copyright (c) Meta Platforms, Inc. and affiliates.
from .base import DepthModel

class MoGe(DepthModel):
    def __call__(self, image, fov_x=None):
        output = self.model.infer(
            image.to(self.device),
            fov_x=fov_x,
            force_projection=fov_x is not None,  # Reproject if FOV provided
        )
        pointmaps = output["points"]
        output["pointmaps"] = pointmaps
        return output