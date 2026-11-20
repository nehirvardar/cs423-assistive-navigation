"""Common Depth Estimator Interface."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseDepthEstimator(ABC):
    """Abstract base class for dense monocular depth estimators."""

    @abstractmethod
    def estimate_depth(
        self,
        image_bgr: np.ndarray,
        gt_sparse_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Returns a dense 2D metric depth map (H, W) in meters."""
        raise NotImplementedError
