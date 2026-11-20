"""End-to-End Unit and Integration Tests for All Project Components."""

from pathlib import Path
import sys
import tempfile
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.drive_split import create_drive_sequence_splits, parse_devkit_drive_mapping
from src.data_prep.kitti_calibration import KittiCalibration, parse_kitti_label_file
from src.data_prep.lidar_projection import load_velodyne_points, project_velodyne_to_depth_map
from src.depth.midas_aligned import align_depth_least_squares
from src.depth.pinhole_distance import PinholeDistanceEstimator
from src.detectors.hog_svm import HogSvmPedestrianDetector
from src.evaluation.depth_metrics import evaluate_object_level_distance, evaluate_pixel_level_depth
from src.evaluation.detection_metrics import evaluate_detection_metrics
from src.evaluation.matching import match_predictions_to_gt
from src.pipelines.box_distance import extract_distance_from_bbox
from src.pipelines.runner import ObstaclePrediction


def run_all_tests() -> None:
    print("[TEST 1] Verifying Drive-Sequence Split (Zero Data Leakage)...")
    available_ids = [f"{i:06d}" for i in range(40)]
    id_to_drive = {f"{i:06d}": f"drive_{i // 4:02d}" for i in range(40)}
    tr, va, te = create_drive_sequence_splits(available_ids, id_to_drive, 0.70, 0.10, seed=42)
    tr_d = {id_to_drive[i] for i in tr}
    va_d = {id_to_drive[i] for i in va}
    te_d = {id_to_drive[i] for i in te}
    assert tr_d.isdisjoint(va_d) and tr_d.isdisjoint(te_d) and va_d.isdisjoint(te_d)
    assert len(tr) + len(va) + len(te) == 40
    print(f"  -> Passed! Split counts: Train={len(tr)}, Val={len(va)}, Test={len(te)}")

    print("[TEST 2] Verifying MiDaS Per-Image Least-Squares LiDAR Scale-Shift Alignment...")
    gt_depth = np.zeros((100, 100), dtype=np.float32)
    gt_depth[20:80, 20:80] = 20.0
    gt_depth[40:60, 40:60] = 10.0
    # True inverse depth = 1 / gt_depth; simulate unaligned MiDaS output = 5.0 * (1/gt) + 2.0
    pred_rel = np.where(gt_depth > 0, 5.0 * (1.0 / np.maximum(1.0, gt_depth)) + 2.0, 2.2).astype(np.float32)
    aligned, s, t = align_depth_least_squares(pred_rel, gt_depth)
    pix_err = evaluate_pixel_level_depth([aligned], [gt_depth])
    assert pix_err["AbsRel"] < 1e-4, f"Expected near-zero AbsRel after exact alignment, got {pix_err['AbsRel']}"
    print(f"  -> Passed! Aligned AbsRel={pix_err['AbsRel']:.6f}, RMSE={pix_err['RMSE']:.6f}m")

    print("[TEST 3] Verifying Pinhole Camera Distance Formula (Z = f * H / h)...")
    pinhole = PinholeDistanceEstimator(assumed_pedestrian_height_m=1.70)
    # If f = 721.5377 px, H = 1.70m, and pedestrian is at Z = 12.266m -> h = 100 px
    bbox = np.array([100.0, 100.0, 145.0, 200.0])
    z_est = pinhole.estimate_box_distance(bbox, focal_length_px=721.5377)
    assert abs(z_est - 12.26614) < 1e-3
    print(f"  -> Passed! Estimated Z = {z_est:.3f}m")

    print("[TEST 4] Verifying Bounding Box Background Filtering Ablation Strategies...")
    # Create a bounding box where outer edges have 45m background sky/wall, center has 12m pedestrian
    dmap = np.full((100, 100), 45.0, dtype=np.float32)
    dmap[35:65, 35:65] = 12.0  # Foreground pedestrian in central region of box [20, 20, 80, 80]
    box = np.array([20.0, 20.0, 80.0, 80.0])
    d_full = extract_distance_from_bbox(dmap, box, strategy="full_median")
    d_center = extract_distance_from_bbox(dmap, box, strategy="center_crop_median", center_crop_ratio=0.50)
    d_trimmed = extract_distance_from_bbox(dmap, box, strategy="trimmed_lower_median")
    assert d_full == 45.0, f"Full median should be corrupted by background (45.0), got {d_full}"
    assert d_center == 12.0, f"Center crop median should recover foreground (12.0), got {d_center}"
    print(f"  -> Passed! Full Median={d_full:.1f}m (corrupted) vs. Center-Crop Median={d_center:.1f}m (filtered)")

    print("[TEST 5] Verifying Detection mAP & Matched Object Distance Metrics...")
    from src.data_prep.kitti_calibration import KittiObject
    gt_obj = KittiObject(
        cls_name="Pedestrian",
        truncation=0.0,
        occlusion=0,
        alpha=0.0,
        bbox=np.array([100.0, 100.0, 150.0, 200.0]),
        dimensions=np.array([1.7, 0.6, 0.6]),
        location=np.array([0.0, 1.6, 15.0]),
        rotation_y=0.0,
    )
    pred_obj = ObstaclePrediction(
        bbox=np.array([102.0, 101.0, 149.0, 199.0]),
        score=0.92,
        class_id=1,
        class_name="Pedestrian",
        distance_m=14.4,
    )
    det_res = evaluate_detection_metrics([[pred_obj]], [[gt_obj]], classes=["Pedestrian"])
    matches, _, _ = match_predictions_to_gt([pred_obj], [gt_obj], iou_threshold=0.5)
    obj_res = evaluate_object_level_distance(matches, classes=["Pedestrian"])
    assert det_res["Pedestrian"]["AP@0.5"] > 0.99
    assert abs(obj_res["Pedestrian"]["AbsRel"] - (0.6 / 15.0)) < 1e-4
    print(
        f"  -> Passed! Pedestrian AP@0.5={det_res['Pedestrian']['AP@0.5']:.4f}, "
        f"Object Distance AbsRel={obj_res['Pedestrian']['AbsRel']:.4f}"
    )

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")


if __name__ == "__main__":
    run_all_tests()
