"""Object Detection mAP@0.5 and mAP@0.5:0.95 Computation per Class.

Computes 101-point interpolated Average Precision (COCO standard) across
IoU thresholds [0.50, 0.55, ..., 0.95] for each class ('Car', 'Pedestrian', 'Cyclist').
"""

from typing import Dict, List
import numpy as np

from src.data_prep.kitti_calibration import KittiObject
from src.evaluation.matching import compute_iou_matrix
from src.pipelines.runner import ObstaclePrediction


def compute_ap_at_iou(
    all_preds_per_frame: List[List[ObstaclePrediction]],
    all_gts_per_frame: List[List[KittiObject]],
    target_class: str,
    iou_thresh: float = 0.50,
) -> float:
    """Computes Average Precision (AP) for a single class at a specific IoU threshold."""
    flat_scores: List[float] = []
    flat_tp: List[int] = []
    total_gt = 0

    # Collect predictions per frame while keeping frame index for local GT matching
    scored_records = []
    gt_boxes_by_frame: Dict[int, np.ndarray] = {}
    gt_used_by_frame: Dict[int, np.ndarray] = {}

    for frame_idx, (preds, gts) in enumerate(zip(all_preds_per_frame, all_gts_per_frame)):
        cls_gts = [g for g in gts if g.cls_name == target_class]
        total_gt += len(cls_gts)
        if cls_gts:
            gt_boxes_by_frame[frame_idx] = np.array([g.bbox for g in cls_gts], dtype=np.float64)
            gt_used_by_frame[frame_idx] = np.zeros(len(cls_gts), dtype=bool)
        else:
            gt_boxes_by_frame[frame_idx] = np.zeros((0, 4), dtype=np.float64)
            gt_used_by_frame[frame_idx] = np.zeros(0, dtype=bool)

        for p in preds:
            if p.class_name == target_class:
                scored_records.append((p.score, frame_idx, p.bbox))

    if total_gt == 0:
        return 0.0
    if not scored_records:
        return 0.0

    # Sort all detections of target_class across dataset by confidence descending
    scored_records.sort(key=lambda x: x[0], reverse=True)

    for score, frame_idx, pred_box in scored_records:
        flat_scores.append(score)
        gt_boxes = gt_boxes_by_frame[frame_idx]
        if len(gt_boxes) == 0:
            flat_tp.append(0)
            continue

        ious = compute_iou_matrix(pred_box.reshape(1, 4), gt_boxes)[0]
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_thresh and not gt_used_by_frame[frame_idx][best_gt_idx]:
            gt_used_by_frame[frame_idx][best_gt_idx] = True
            flat_tp.append(1)
        else:
            flat_tp.append(0)

    tp_arr = np.array(flat_tp, dtype=np.float64)
    fp_arr = 1.0 - tp_arr

    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)

    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(1e-8, tp_cum + fp_cum)

    # 101-point COCO AP interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    recall_thresholds = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for r_thresh in recall_thresholds:
        inds = np.where(mrec >= r_thresh)[0]
        p_val = float(np.max(mpre[inds])) if inds.size > 0 else 0.0
        ap += p_val / 101.0

    return float(ap)


def evaluate_detection_metrics(
    all_preds_per_frame: List[List[ObstaclePrediction]],
    all_gts_per_frame: List[List[KittiObject]],
    classes: List[str] = ["Car", "Pedestrian", "Cyclist"],
) -> Dict[str, Dict[str, float]]:
    """Computes per-class and mean mAP@0.5 and mAP@0.5:0.95."""
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    results: Dict[str, Dict[str, float]] = {}

    ap50_list = []
    ap50_95_list = []

    for cls_name in classes:
        aps = [
            compute_ap_at_iou(all_preds_per_frame, all_gts_per_frame, cls_name, float(th))
            for th in iou_thresholds
        ]
        ap50 = aps[0]
        ap50_95 = float(np.mean(aps))
        results[cls_name] = {
            "AP@0.5": ap50,
            "AP@0.5:0.95": ap50_95,
        }
        ap50_list.append(ap50)
        ap50_95_list.append(ap50_95)

    results["mAP_All"] = {
        "AP@0.5": float(np.mean(ap50_list)) if ap50_list else 0.0,
        "AP@0.5:0.95": float(np.mean(ap50_95_list)) if ap50_95_list else 0.0,
    }
    return results
