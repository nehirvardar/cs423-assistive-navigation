"""Step 4: Unified Common-Protocol Evaluation of All 3 Pipelines + Ablation Study.

Evaluates on the KITTI test split (`data/splits/test.txt`):
1. Pipeline 1 (Main): YOLOv10-Nano + Depth Anything V2 (Metric Outdoor)
2. Pipeline 2 (Secondary): YOLOv8-Nano + MiDaS Small (Per-Image LiDAR Scale-Shift Aligned)
3. Pipeline 3 (Classical Baseline): HOG + Linear SVM (Pedestrian) + Pinhole Camera Model (Z = f*H/h)

Outputs:
- Table 1: Object Detection Accuracy per class (mAP@0.5 and mAP@0.5:0.95)
- Table 2: Pixel-Level Monocular Depth Accuracy on LiDAR points (AbsRel, RMSE, delta < 1.25)
- Table 3: System-Level Object Distance Error on matched detections (IoU >= 0.5)
- Table 4: Fair Pedestrian-Only Comparison across all 3 pipelines + Success Criteria verification
- Table 5: Ablation Study on Bounding Box Distance Extraction Strategies
           (full_median vs. center_crop_median vs. trimmed_lower_median)
- Qualitative visualizations and Error Analysis plots in `results/`
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List
import cv2
import numpy as np
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.kitti_calibration import KittiCalibration, parse_kitti_label_file
from src.depth.depth_anything_v2 import DepthAnythingV2MetricOutdoor
from src.depth.midas_aligned import MidasSmallAlignedEstimator
from src.depth.pinhole_distance import PinholeDistanceEstimator
from src.detectors.hog_svm import HogSvmPedestrianDetector
from src.detectors.yolo_detector import YoloDetector
from src.evaluation.depth_metrics import evaluate_object_level_distance, evaluate_pixel_level_depth
from src.evaluation.detection_metrics import evaluate_detection_metrics
from src.evaluation.matching import MatchedObstaclePair, match_predictions_to_gt
from src.evaluation.visualize import draw_obstacles_on_image, save_error_analysis_plots
from src.pipelines.box_distance import extract_distance_from_bbox
from src.pipelines.runner import ObstacleDistancePipeline, ObstaclePrediction


def build_pipelines(
    eval_cfg: dict,
    selected_keys: List[str],
    device: str | None = None,
) -> Dict[str, ObstacleDistancePipeline]:
    """Instantiates selected pipelines configured in configs/eval_config.yaml."""
    m_cfg = eval_cfg["models"]
    box_strategy = eval_cfg["evaluation"]["default_box_strategy"]
    crop_ratio = eval_cfg["evaluation"]["center_crop_ratio"]
    built: Dict[str, ObstacleDistancePipeline] = {}

    if "pipeline1" in selected_keys:
        print("[INFO] Loading Pipeline 1: YOLOv10-Nano + Depth Anything V2 (Metric Outdoor)...")
        det1 = YoloDetector(
            weights_path=m_cfg["pipeline1"]["detector_weights"],
            fallback_pretrained=m_cfg["pipeline1"]["fallback_pretrained"],
            device=device,
        )
        depth1 = DepthAnythingV2MetricOutdoor(
            checkpoint=m_cfg["pipeline1"]["depth_checkpoint"],
            device=device,
        )
        built["pipeline1"] = ObstacleDistancePipeline(
            name="Pipeline 1 (YOLOv10n + DA-V2 Metric)",
            detector=det1,
            depth_estimator=depth1,
            box_strategy=box_strategy,
            center_crop_ratio=crop_ratio,
        )

    if "pipeline2" in selected_keys:
        print("[INFO] Loading Pipeline 2: YOLOv8-Nano + MiDaS Small (LiDAR Scale-Shift Aligned)...")
        det2 = YoloDetector(
            weights_path=m_cfg["pipeline2"]["detector_weights"],
            fallback_pretrained=m_cfg["pipeline2"]["fallback_pretrained"],
            device=device,
        )
        depth2 = MidasSmallAlignedEstimator(
            model_type=m_cfg["pipeline2"]["depth_checkpoint"],
            device=device,
        )
        built["pipeline2"] = ObstacleDistancePipeline(
            name="Pipeline 2 (YOLOv8n + MiDaS Aligned)",
            detector=det2,
            depth_estimator=depth2,
            box_strategy=box_strategy,
            center_crop_ratio=crop_ratio,
        )

    if "pipeline3" in selected_keys:
        print("[INFO] Loading Pipeline 3: HOG+SVM (Pedestrian) + Pinhole Geometry Baseline...")
        det3 = HogSvmPedestrianDetector(weights_path=m_cfg["pipeline3"]["detector_weights"])
        pinhole3 = PinholeDistanceEstimator(
            assumed_pedestrian_height_m=m_cfg["pipeline3"]["assumed_pedestrian_height_m"]
        )
        built["pipeline3"] = ObstacleDistancePipeline(
            name="Pipeline 3 (HOG+SVM + Pinhole)",
            detector=det3,
            pinhole_estimator=pinhole3,
            conf_thresh=0.10,
        )

    return built


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all 3 pipelines on the KITTI test split.")
    parser.add_argument("--split-config", type=str, default="configs/split_config.yaml")
    parser.add_argument("--eval-config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of evaluated frames")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for tables and figures")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--pipelines",
        type=str,
        nargs="+",
        default=["pipeline1", "pipeline2", "pipeline3"],
        choices=["pipeline1", "pipeline2", "pipeline3"],
        help="Which pipelines to evaluate (default: all 3 simultaneously)",
    )
    parser.add_argument(
        "--multi-seed-summary",
        action="store_true",
        help="Also evaluate across all seeds [42, 123, 999] if checkpoints exist and report mean +/- std",
    )
    args = parser.parse_args()

    with open(args.split_config, "r", encoding="utf-8") as f:
        split_cfg = yaml.safe_load(f)
    with open(args.eval_config, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    ds_cfg = split_cfg["dataset"]
    split_file = Path(ds_cfg["splits_dir"]) / f"{args.split}.txt"
    if not split_file.exists():
        print(f"[ERROR] Split file {split_file} not found. Run `python scripts/01_prepare_data.py` first.")
        sys.exit(1)

    with open(split_file, "r", encoding="utf-8") as f:
        eval_ids = [line.strip() for line in f if line.strip()]
    if args.max_frames:
        eval_ids = eval_ids[: args.max_frames]

    img_dir = Path(ds_cfg["image_dir"])
    lbl_dir = Path(ds_cfg["label_dir"])
    cal_dir = Path(ds_cfg["calib_dir"])
    depth_dir = Path(ds_cfg["processed_dir"]) / "depth_maps"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    pipelines = build_pipelines(eval_cfg, selected_keys=args.pipelines, device=args.device)
    iou_thresh = float(eval_cfg["evaluation"]["iou_threshold"])

    all_gts_per_frame = []
    gt_depths_per_frame = []

    preds_by_pipe: Dict[str, List[List[ObstaclePrediction]]] = {k: [] for k in pipelines}
    depth_maps_by_pipe: Dict[str, List[np.ndarray]] = {k: [] for k in pipelines}
    matches_by_pipe: Dict[str, List[MatchedObstaclePair]] = {k: [] for k in pipelines}
    latencies_by_pipe: Dict[str, List[float]] = {k: [] for k in pipelines}

    # For Ablation Study on Pipeline 1 (full_median vs center_crop_median vs trimmed_lower_median)
    ablation_strategies = ["full_median", "center_crop_median", "trimmed_lower_median"]
    ablation_matches: Dict[str, List[MatchedObstaclePair]] = {s: [] for s in ablation_strategies}

    for idx, img_id in enumerate(tqdm(eval_ids, desc=f"Evaluating on {args.split} split")):
        img_path = img_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = img_dir / f"{img_id}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        calib = KittiCalibration(cal_dir / f"{img_id}.txt")
        gt_objects = parse_kitti_label_file(lbl_dir / f"{img_id}.txt", class_mapping=split_cfg.get("class_mapping"))
        depth_npy = depth_dir / f"{img_id}.npy"
        gt_sparse_depth = np.load(depth_npy) if depth_npy.exists() else np.zeros(img.shape[:2], dtype=np.float32)

        all_gts_per_frame.append(gt_objects)
        gt_depths_per_frame.append(gt_sparse_depth)

        vis_panels = []
        for p_key, pipe in pipelines.items():
            out = pipe.run_frame(
                image_bgr=img,
                focal_length_px=calib.focal_length_y,
                gt_sparse_depth=gt_sparse_depth,
            )
            preds_by_pipe[p_key].append(out.obstacles)
            if out.dense_depth_map is not None:
                depth_maps_by_pipe[p_key].append(out.dense_depth_map)
            latencies_by_pipe[p_key].append(out.total_time_ms)

            matched, _, _ = match_predictions_to_gt(
                predictions=out.obstacles,
                gt_objects=gt_objects,
                gt_sparse_depth=gt_sparse_depth,
                iou_threshold=iou_thresh,
            )
            matches_by_pipe[p_key].extend(matched)

            # Ablation study on Pipeline 1 bounding box depth aggregation
            if p_key == "pipeline1" and out.dense_depth_map is not None:
                for strat in ablation_strategies:
                    strat_obs = []
                    for obs in out.obstacles:
                        d_m = extract_distance_from_bbox(
                            depth_map=out.dense_depth_map,
                            bbox=obs.bbox,
                            strategy=strat,
                        )
                        strat_obs.append(
                            ObstaclePrediction(
                                bbox=obs.bbox,
                                score=obs.score,
                                class_id=obs.class_id,
                                class_name=obs.class_name,
                                distance_m=d_m,
                            )
                        )
                    m_strat, _, _ = match_predictions_to_gt(
                        predictions=strat_obs,
                        gt_objects=gt_objects,
                        gt_sparse_depth=gt_sparse_depth,
                        iou_threshold=iou_thresh,
                    )
                    ablation_matches[strat].extend(m_strat)

            if idx < 5:
                vis_panels.append(draw_obstacles_on_image(img, out, title=pipe.name))

        if vis_panels:
            stacked = np.vstack(vis_panels)
            cv2.imwrite(str(qual_dir / f"comparison_{img_id}.png"), stacked)

    # Compute final tables
    summary_report = {"pipelines": {}, "ablation_box_filtering": {}}

    print("\n" + "=" * 85)
    print("FINAL EVALUATION SUMMARY (COMMON PROTOCOL)")
    print("=" * 85)

    for p_key, pipe in pipelines.items():
        det_metrics = evaluate_detection_metrics(preds_by_pipe[p_key], all_gts_per_frame)
        pix_metrics = evaluate_pixel_level_depth(depth_maps_by_pipe[p_key], gt_depths_per_frame)
        obj_metrics = evaluate_object_level_distance(matches_by_pipe[p_key])
        mean_ms = float(np.mean(latencies_by_pipe[p_key])) if latencies_by_pipe[p_key] else 0.0
        fps = 1000.0 / max(1e-3, mean_ms)

        summary_report["pipelines"][p_key] = {
            "name": pipe.name,
            "detection": det_metrics,
            "pixel_depth": pix_metrics,
            "object_distance": obj_metrics,
            "latency_ms": mean_ms,
            "fps": fps,
        }

        print(f"\n>>> {pipe.name}")
        print(
            f"  [Detection] Pedestrian AP@0.5: {det_metrics['Pedestrian']['AP@0.5']:.4f} | "
            f"Car AP@0.5: {det_metrics['Car']['AP@0.5']:.4f} | "
            f"mAP@0.5 (All): {det_metrics['mAP_All']['AP@0.5']:.4f}"
        )
        print(
            f"  [Pixel Depth (LiDAR)] AbsRel: {pix_metrics['AbsRel']:.4f} | "
            f"RMSE: {pix_metrics['RMSE']:.3f}m | delta<1.25: {pix_metrics['Delta1']:.4f}"
        )
        print(
            f"  [Object Distance (Pedestrian, IoU>=0.5)] AbsRel: {obj_metrics['Pedestrian']['AbsRel']:.4f} | "
            f"RMSE: {obj_metrics['Pedestrian']['RMSE']:.3f}m (Matched={obj_metrics['Pedestrian']['Count']})"
        )
        print(
            f"  [Object Distance (All Classes, IoU>=0.5)] AbsRel: {obj_metrics['All_Classes']['AbsRel']:.4f} | "
            f"RMSE: {obj_metrics['All_Classes']['RMSE']:.3f}m (Matched={obj_metrics['All_Classes']['Count']})"
        )
        print(f"  [Speed] Mean Latency: {mean_ms:.2f} ms/frame | Throughput: {fps:.2f} FPS")

    print("\n" + "-" * 85)
    print("ABLATION STUDY: BOUNDING BOX DEPTH FILTERING STRATEGIES (PIPELINE 1)")
    print("-" * 85)
    for strat in ablation_strategies:
        ab_res = evaluate_object_level_distance(ablation_matches[strat])["All_Classes"]
        summary_report["ablation_box_filtering"][strat] = ab_res
        print(
            f"  Strategy: {strat:<22} | AbsRel: {ab_res['AbsRel']:.4f} | "
            f"RMSE: {ab_res['RMSE']:.3f}m | delta<1.25: {ab_res['Delta1']:.4f}"
        )

    # Save JSON report and error analysis plot
    if args.multi_seed_summary:
        seeds = eval_cfg["evaluation"].get("random_seeds", [42, 123, 999])
        multi_seed_stats = {}
        print("\n" + "-" * 85)
        print(f"MULTI-SEED VARIABILITY SUMMARY (Seeds: {seeds}) -> Mean +/- Std")
        print("-" * 85)
        for model_prefix in ["yolov10n", "yolov8n"]:
            seed_maps = []
            for s in seeds:
                ckpt_file = Path("checkpoints") / f"{model_prefix}_kitti_seed{s}.pt"
                if not ckpt_file.exists():
                    continue
                det_s = YoloDetector(weights_path=ckpt_file, fallback_pretrained=f"{model_prefix}.pt", device=args.device)
                preds_s = []
                for img_id in eval_ids:
                    img_path = img_dir / f"{img_id}.png"
                    if not img_path.exists():
                        img_path = img_dir / f"{img_id}.jpg"
                    im = cv2.imread(str(img_path))
                    if im is None:
                        continue
                    dets = det_s.predict(im)
                    preds_s.append([
                        ObstaclePrediction(d.bbox, d.score, d.class_id, d.class_name, 10.0)
                        for d in dets
                    ])
                m_s = evaluate_detection_metrics(preds_s, all_gts_per_frame)
                seed_maps.append(m_s["mAP_All"]["AP@0.5"])
            if seed_maps:
                mean_ap = float(np.mean(seed_maps))
                std_ap = float(np.std(seed_maps))
                multi_seed_stats[model_prefix] = {"seeds_evaluated": len(seed_maps), "mAP@0.5_mean": mean_ap, "mAP@0.5_std": std_ap}
                print(f"  {model_prefix:<12} ({len(seed_maps)} seeds) | mAP@0.5: {mean_ap:.4f} +/- {std_ap:.4f}")
        summary_report["multi_seed_variability"] = multi_seed_stats

    json_path = out_dir / f"evaluation_metrics_{args.split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    fig_path = save_error_analysis_plots(
        {pipelines[k].name: matches_by_pipe[k] for k in pipelines},
        out_dir=out_dir,
    )
    print(f"\n[SUCCESS] Saved full metrics to {json_path}")
    print(f"[SUCCESS] Saved error analysis chart to {fig_path}")
    print(f"[SUCCESS] Saved qualitative comparison images to {qual_dir}")


if __name__ == "__main__":
    main()
