"""Common Detector Interface for Unified Pipeline Evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import numpy as np


ID_TO_CLASS = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
CLASS_TO_ID = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


@dataclass
class Detection:
    """Single 2D object detection output."""
    bbox: np.ndarray   # [x1, y1, x2, y2] in pixel coordinates
    score: float       # confidence score
    class_id: int      # 0: Car, 1: Pedestrian, 2: Cyclist
    class_name: str    # 'Car', 'Pedestrian', or 'Cyclist'

    @property
    def height_px(self) -> float:
        return float(max(1.0, self.bbox[3] - self.bbox[1]))


class BaseDetector(ABC):
    """Abstract base class for all object detectors in the 3 pipelines."""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray, conf_thresh: float = 0.25) -> List[Detection]:
        """Runs object detection on a single BGR image (H, W, 3) and returns Detections."""
        raise NotImplementedError
