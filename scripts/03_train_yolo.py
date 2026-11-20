"""Step 3: Fine-Tune YOLOv10-Nano (Pipeline 1) and YOLOv8-Nano (Pipeline 2) Across 3 Random Seeds.

Fulfills the course requirement on Variability (Section 3):
'Detector training is repeated with three random seeds, and we give mean and standard deviation.'
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.yolo_detector import YoloDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv10n or YOLOv8n on KITTI across multiple random seeds.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov10n",
        choices=["yolov10n", "yolov8n"],
        help="YOLO variant to fine-tune ('yolov10n' for Pipeline 1, 'yolov8n' for Pipeline 2)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 999],
        help="Random seeds for multi-run variability reporting",
    )
    parser.add_argument("--data-yaml", type=str, default="configs/kitti_yolo.yaml", help="Ultralytics data YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device ('0' for GPU, 'cpu' for CPU)")
    args = parser.parse_args()

    print(f"[INFO] Starting fine-tuning for {args.model} across seeds: {args.seeds}")
    saved_checkpoints = []

    for seed in args.seeds:
        print(f"\n--- Training {args.model} | Seed = {seed} ---")
        ckpt = YoloDetector.fine_tune(
            model_variant=args.model,
            data_yaml=args.data_yaml,
            seed=seed,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            output_dir="checkpoints",
            device=args.device,
        )
        saved_checkpoints.append(ckpt)

    print("\n[SUCCESS] Completed all seed runs. Saved checkpoints:")
    for ckpt in saved_checkpoints:
        print(f"  - {ckpt}")


if __name__ == "__main__":
    main()
