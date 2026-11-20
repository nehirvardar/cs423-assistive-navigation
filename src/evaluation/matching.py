"""Bounding Box IoU Calculation and Prediction-to-Ground-Truth Matching.

Implements the proposal evaluation protocol:
'The distance error of the whole system is computed only on detections that
match a ground-truth box (IoU >= 0.5, same class).'
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.data_prep.kitti_calibration import KittiObject
from src.pipelines.box_distance import extract_distance_from_bbox
from src.pipelines.runner import ObstaclePrediction


@dataclass
class MatchedObstaclePair:
    """A matched pair of predicted obstacle and ground-truth KITTI object (IoU >= 0.5, same class)."""
    class_name: str
    iou: float
    pred_distance_m: float
    gt_distance_m: float
    pred_bbox: np.ndarray
    gt_bbox: np.ndarray
    occlusion: int
    truncation: float


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU matrix (N, M) between two sets of [x1, y1, x2, y2] boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    box1_area = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    box2_area = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)

    union_area = box1_area + np.transpose(box2_area) - inter_area + 1e-8
    return inter_area / union_area


def get_gt_object_distance(
    gt_obj: KittiObject,
    gt_sparse_depth: np.ndarray | None = None,
) -> float:
    """Returns ground-truth distance in meters for a KITTI object.

    Prefers the median of projected LiDAR points inside the GT bounding box
    if available (>3 valid LiDAR returns); otherwise uses the KITTI 3D annotation's
    camera-frame Z distance (`location[2]`).
    """
    if gt_sparse_depth is not None:
        lidar_med = extract_distance_from_bbox(
            depth_map=gt_sparse_depth,
            bbox=gt_obj.bbox,
            strategy="center_crop_median",
            center_crop_ratio=0.60,
        )
        if lidar_med > 0.5:
            return lidar_med

    return gt_obj.gt_distance_z


def match_predictions_to_gt(
    predictions: List[ObstaclePrediction],
    gt_objects: List[KittiObject],
    gt_sparse_depth: np.ndarray | None = None,
    iou_threshold: float = 0.50,
) -> Tuple[List[MatchedObstaclePair], int, int]:
    """Matches predicted obstacles to GT boxes of the same class with IoU >= iou_threshold.

    Predictions are sorted by confidence score descending, and each GT box can be
    matched at most once (greedy assignment standard in COCO/KITTI evaluation).
    """
    if not predictions or not gt_objects:
        return [], len(predictions), len(gt_objects)

    sorted_preds = sorted(predictions, key=lambda p: p.score, reverse=True)
    matched_gt_indices = set()
    matches: List[MatchedObstaclePair] = []

    gt_boxes = np.array([g.bbox for g in gt_objects], dtype=np.float64)

    for pred in sorted_preds:
        pred_box = pred.bbox.reshape(1, 4)
        ious = compute_iou_matrix(pred_box, gt_boxes)[0]

        best_iou = -1.0
        best_gt_idx = -1
        for gt_idx, gt_obj in enumerate(gt_objects):
            if gt_idx in matched_gt_indices:
                continue
            if gt_obj.cls_name != pred.class_name:
                continue
            if ious[gt_idx] >= iou_threshold and ious[gt_idx] > best_iou:
                best_iou = float(ious[gt_idx])
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt_indices.add(best_gt_idx)
            gt_obj = gt_objects[best_gt_idx]
            gt_dist = get_gt_object_distance(gt_obj, gt_sparse_depth=gt_sparse_depth)

            if pred.distance_m > 0.0 and gt_dist > 0.0:
                matches.append(
                    MatchedObstaclePair(
                        class_name=pred.class_name,
                        iou=best_iou,
                        pred_distance_m=pred.distance_m,
                        gt_distance_m=gt_dist,
                        pred_bbox=pred.bbox,
                        gt_bbox=gt_obj.bbox,
                        occlusion=gt_obj.occlusion,
                        truncation=gt_obj.truncation,
                    )
                )

    unmatched_preds = len(predictions) - len(matched_gt_indices)
    unmatched_gts = len(gt_objects) - len(matched_gt_indices)
    return matches, unmatched_preds, unmatched_gts
