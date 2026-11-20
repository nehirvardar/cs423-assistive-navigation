"""Depth and Object-Level Distance Metrics (AbsRel, RMSE, delta < 1.25).

Implements:
1. Pixel-level depth evaluation on valid LiDAR returns (AbsRel, RMSE, delta < 1.25).
2. System-level object distance evaluation on matched detections (IoU >= 0.5, same class),
   both overall and per-class (specifically isolating the Pedestrian class for fair
   comparison against the HOG+SVM + Pinhole classical baseline).
"""

from typing import Dict, List
import numpy as np

from src.evaluation.matching import MatchedObstaclePair


def compute_depth_errors(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
) -> Dict[str, float]:
    """Computes standard monocular depth metrics between 1D arrays of valid predictions and GTs."""
    if pred_values.size == 0 or gt_values.size == 0:
        return {"AbsRel": float("nan"), "RMSE": float("nan"), "Delta1": float("nan"), "Count": 0}

    pred = np.maximum(1e-3, pred_values.astype(np.float64))
    gt = np.maximum(1e-3, gt_values.astype(np.float64))

    abs_rel = float(np.mean(np.abs(pred - gt) / gt))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))

    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = float(np.mean(ratio < 1.25))

    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "Delta1": delta1,
        "Count": int(gt.size),
    }


def evaluate_pixel_level_depth(
    pred_depth_maps: List[np.ndarray],
    gt_sparse_depth_maps: List[np.ndarray],
    min_depth: float = 0.001,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Aggregates pixel-level depth metrics across all test images on valid LiDAR pixels."""
    all_preds = []
    all_gts = []

    for pred_map, gt_map in zip(pred_depth_maps, gt_sparse_depth_maps):
        if pred_map is None or gt_map is None:
            continue
        valid_mask = (gt_map >= min_depth) & (gt_map <= max_depth) & (pred_map >= min_depth)
        if np.any(valid_mask):
            all_preds.append(pred_map[valid_mask])
            all_gts.append(gt_map[valid_mask])

    if not all_preds:
        return {"AbsRel": float("nan"), "RMSE": float("nan"), "Delta1": float("nan"), "Count": 0}

    return compute_depth_errors(np.concatenate(all_preds), np.concatenate(all_gts))


def evaluate_object_level_distance(
    matched_pairs: List[MatchedObstaclePair],
    classes: List[str] = ["Car", "Pedestrian", "Cyclist"],
) -> Dict[str, Dict[str, float]]:
    """Computes object-distance errors per class and overall on matched boxes (IoU >= 0.5)."""
    results: Dict[str, Dict[str, float]] = {}

    for cls_name in classes:
        cls_pairs = [m for m in matched_pairs if m.class_name == cls_name]
        preds = np.array([m.pred_distance_m for m in cls_pairs], dtype=np.float64)
        gts = np.array([m.gt_distance_m for m in cls_pairs], dtype=np.float64)
        results[cls_name] = compute_depth_errors(preds, gts)

    all_preds = np.array([m.pred_distance_m for m in matched_pairs], dtype=np.float64)
    all_gts = np.array([m.gt_distance_m for m in matched_pairs], dtype=np.float64)
    results["All_Classes"] = compute_depth_errors(all_preds, all_gts)

    return results
