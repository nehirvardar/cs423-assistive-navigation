"""Step 5: Hardware Latency (ms) and Throughput (FPS) Benchmark Script.

Measures detector time, depth estimation time, and end-to-end pipeline FPS
with proper CUDA synchronization and warm-up iterations, verifying the >15 FPS
success criterion from the project proposal.
"""

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth.depth_anything_v2 import DepthAnythingV2MetricOutdoor
from src.depth.midas_aligned import MidasSmallAlignedEstimator
from src.depth.pinhole_distance import PinholeDistanceEstimator
from src.detectors.hog_svm import HogSvmPedestrianDetector
from src.detectors.yolo_detector import YoloDetector
from src.pipelines.runner import ObstacleDistancePipeline


def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark latency (ms) and FPS for all 3 pipelines.")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    parser.add_argument("--runs", type=int, default=20, help="Timed benchmark iterations")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        eval_cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    hw_name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU"
    print(f"[INFO] Benchmarking on hardware: {hw_name} (device={device})")

    # Dummy KITTI resolution image (375 x 1242 x 3) and synthetic sparse LiDAR depth
    dummy_img = np.random.randint(0, 255, size=(375, 1242, 3), dtype=np.uint8)
    dummy_gt_depth = np.zeros((375, 1242), dtype=np.float32)
    dummy_gt_depth[180:350, 100:1100] = 15.0

    m_cfg = eval_cfg["models"]
    pipelines = [
        ObstacleDistancePipeline(
            name=m_cfg["pipeline1"]["name"],
            detector=YoloDetector(
                m_cfg["pipeline1"]["detector_weights"],
                m_cfg["pipeline1"]["fallback_pretrained"],
                device=device,
            ),
            depth_estimator=DepthAnythingV2MetricOutdoor(
                m_cfg["pipeline1"]["depth_checkpoint"], device=device
            ),
        ),
        ObstacleDistancePipeline(
            name=m_cfg["pipeline2"]["name"],
            detector=YoloDetector(
                m_cfg["pipeline2"]["detector_weights"],
                m_cfg["pipeline2"]["fallback_pretrained"],
                device=device,
            ),
            depth_estimator=MidasSmallAlignedEstimator(
                m_cfg["pipeline2"]["depth_checkpoint"], device=device
            ),
        ),
        ObstacleDistancePipeline(
            name=m_cfg["pipeline3"]["name"],
            detector=HogSvmPedestrianDetector(m_cfg["pipeline3"]["detector_weights"]),
            pinhole_estimator=PinholeDistanceEstimator(
                m_cfg["pipeline3"]["assumed_pedestrian_height_m"]
            ),
        ),
    ]

    print("\n" + "=" * 90)
    print(f"{'Pipeline':<44} | {'Det (ms)':>9} | {'Depth (ms)':>10} | {'Total (ms)':>10} | {'FPS':>7}")
    print("-" * 90)

    for pipe in pipelines:
        for _ in range(args.warmup):
            pipe.run_frame(dummy_img, focal_length_px=721.54, gt_sparse_depth=dummy_gt_depth)
        sync_device()

        det_times = []
        depth_times = []
        total_times = []

        for _ in range(args.runs):
            sync_device()
            t0 = time.perf_counter()
            out = pipe.run_frame(dummy_img, focal_length_px=721.54, gt_sparse_depth=dummy_gt_depth)
            sync_device()
            t1 = time.perf_counter()

            det_times.append(out.detector_time_ms)
            depth_times.append(out.depth_time_ms)
            total_times.append((t1 - t0) * 1000.0)

        m_det = float(np.mean(det_times))
        m_dep = float(np.mean(depth_times))
        m_tot = float(np.mean(total_times))
        fps = 1000.0 / max(1e-3, m_tot)
        print(f"{pipe.name:<44} | {m_det:>9.2f} | {m_dep:>10.2f} | {m_tot:>10.2f} | {fps:>7.2f}")

    print("=" * 90)


if __name__ == "__main__":
    main()
