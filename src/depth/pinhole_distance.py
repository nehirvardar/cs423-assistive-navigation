"""Pipeline 3 Classical Distance Estimator: Pinhole Camera Geometry.

Estimates pedestrian distance using:
    Z = (f * H) / h
where:
    f = camera vertical focal length from KITTI calibration matrix P2[1, 1] (in pixels)
    H = assumed real-world height of a pedestrian (default 1.70 meters)
    h = 2D bounding box height in pixels (y_max - y_min)
"""

import numpy as np


class PinholeDistanceEstimator:
    """Computes object distance from 2D bounding box height via Pinhole Camera Model."""

    def __init__(
        self,
        assumed_pedestrian_height_m: float = 1.70,
        min_depth: float = 0.5,
        max_depth: float = 80.0,
    ):
        self.assumed_height_m = assumed_pedestrian_height_m
        self.min_depth = min_depth
        self.max_depth = max_depth

    def estimate_box_distance(
        self,
        bbox: np.ndarray,
        focal_length_px: float,
        assumed_height_m: float | None = None,
    ) -> float:
        """Computes Z = (f * H) / h in meters for a single 2D bounding box [x1, y1, x2, y2]."""
        H = assumed_height_m if assumed_height_m is not None else self.assumed_height_m
        box_h = float(max(1.0, bbox[3] - bbox[1]))
        z_m = (float(focal_length_px) * float(H)) / box_h
        return float(np.clip(z_m, self.min_depth, self.max_depth))
