"""Unified Pipeline Runner for Monocular Obstacle Detection & Distance Estimation.

Executes any of the 3 project pipelines under a common interface:
- Pipeline 1: YOLOv10-Nano + Depth Anything V2 (Metric Outdoor)
- Pipeline 2: YOLOv8-Nano + MiDaS Small (Per-Image LiDAR Scale-Shift Aligned)
- Pipeline 3: Classical HOG+SVM Pedestrian Detector + Pinhole Camera Distance (Z = f*H/h)
"""

from dataclasses import dataclass
import time
from typing import List, Optional
import numpy as np

from src.depth.base_depth import BaseDepthEstimator
from src.depth.pinhole_distance import PinholeDistanceEstimator
from src.detectors.base_detector import BaseDetector, Detection
from src.pipelines.box_distance import extract_distance_from_bbox


@dataclass
class ObstaclePrediction:
    """Represents a detected obstacle along with its estimated distance in meters."""
    bbox: np.ndarray        # [x1, y1, x2, y2]
    score: float
    class_id: int
    class_name: str
    distance_m: float       # Estimated distance in meters


@dataclass
class PipelineOutput:
    """Full output of a pipeline on a single frame."""
    obstacles: List[ObstaclePrediction]
    dense_depth_map: Optional[np.ndarray]  # (H, W) in meters for P1 & P2; None for P3
    detector_time_ms: float
    depth_time_ms: float
    total_time_ms: float


class ObstacleDistancePipeline:
    """Combines an object detector and a depth/distance estimator into a unified pipeline."""

    def __init__(
        self,
        name: str,
        detector: BaseDetector,
        depth_estimator: Optional[BaseDepthEstimator] = None,
        pinhole_estimator: Optional[PinholeDistanceEstimator] = None,
        box_strategy: str = "center_crop_median",
        center_crop_ratio: float = 0.50,
        conf_thresh: float = 0.25,
    ):
        self.name = name
        self.detector = detector
        self.depth_estimator = depth_estimator
        self.pinhole_estimator = pinhole_estimator
        self.box_strategy = box_strategy
        self.center_crop_ratio = center_crop_ratio
        self.conf_thresh = conf_thresh

    def run_frame(
        self,
        image_bgr: np.ndarray,
        focal_length_px: Optional[float] = None,
        gt_sparse_depth: Optional[np.ndarray] = None,
    ) -> PipelineOutput:
        """Runs object detection and distance estimation on a single BGR frame."""
        t0 = time.perf_counter()
        detections: List[Detection] = self.detector.predict(image_bgr, conf_thresh=self.conf_thresh)
        t1 = time.perf_counter()

        dense_depth: Optional[np.ndarray] = None
        obstacles: List[ObstaclePrediction] = []

        if self.depth_estimator is not None:
            dense_depth = self.depth_estimator.estimate_depth(
                image_bgr=image_bgr,
                gt_sparse_depth=gt_sparse_depth,
            )
            for det in detections:
                dist_m = extract_distance_from_bbox(
                    depth_map=dense_depth,
                    bbox=det.bbox,
                    strategy=self.box_strategy,
                    center_crop_ratio=self.center_crop_ratio,
                )
                obstacles.append(
                    ObstaclePrediction(
                        bbox=det.bbox,
                        score=det.score,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        distance_m=dist_m,
                    )
                )
        elif self.pinhole_estimator is not None:
            if focal_length_px is None:
                raise ValueError("focal_length_px is required for Pinhole distance estimation.")
            for det in detections:
                dist_m = self.pinhole_estimator.estimate_box_distance(
                    bbox=det.bbox,
                    focal_length_px=focal_length_px,
                )
                obstacles.append(
                    ObstaclePrediction(
                        bbox=det.bbox,
                        score=det.score,
                        class_id=det.class_id,
                        class_name=det.class_name,
                        distance_m=dist_m,
                    )
                )
        else:
            raise ValueError("Pipeline requires either a dense depth_estimator or a pinhole_estimator.")

        t2 = time.perf_counter()
        det_ms = (t1 - t0) * 1000.0
        depth_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        return PipelineOutput(
            obstacles=obstacles,
            dense_depth_map=dense_depth,
            detector_time_ms=det_ms,
            depth_time_ms=depth_ms,
            total_time_ms=total_ms,
        )
