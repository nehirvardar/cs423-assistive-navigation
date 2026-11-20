"""Step 2: Train Classical Baseline (Dalal & Triggs HOG + Linear SVM Pedestrian Detector).

Extracts positive Pedestrian crops and negative non-pedestrian background patches
from the KITTI training split, trains a Linear SVM with Hard Negative Mining, and
saves the weights to `checkpoints/hog_svm_pedestrian.joblib`.
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.kitti_calibration import parse_kitti_label_file
from src.detectors.hog_svm import HogSvmPedestrianDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HOG + Linear SVM Pedestrian detector on KITTI train split.")
    parser.add_argument("--config", type=str, default="configs/split_config.yaml", help="Path to split_config.yaml")
    parser.add_argument("--output", type=str, default="checkpoints/hog_svm_pedestrian.joblib", help="Output model path")
    parser.add_argument("--c-param", type=float, default=0.01, help="LinearSVC regularization parameter C")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    img_dir = Path(ds_cfg["image_dir"])
    lbl_dir = Path(ds_cfg["label_dir"])
    train_split_file = Path(ds_cfg["splits_dir"]) / "train.txt"

    if not train_split_file.exists():
        print(f"[ERROR] {train_split_file} not found. Run `python scripts/01_prepare_data.py` first.")
        sys.exit(1)

    with open(train_split_file, "r", encoding="utf-8") as f:
        train_ids = [line.strip() for line in f if line.strip()]

    rng = np.random.default_rng(args.seed)
    pos_patches = []
    neg_patches = []
    hard_neg_images = []

    for img_id in tqdm(train_ids, desc="Collecting HOG training patches"):
        img_path = img_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        objects = parse_kitti_label_file(lbl_dir / f"{img_id}.txt", class_mapping=cfg.get("class_mapping"))

        for obj in objects:
            if obj.cls_name == "Pedestrian":
                x1, y1, x2, y2 = [int(round(v)) for v in obj.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if (x2 - x1) >= 12 and (y2 - y1) >= 24:
                    crop = img[y1:y2, x1:x2]
                    pos_patches.append(crop)
                    # Horizontal flip augmentation for pedestrians
                    pos_patches.append(cv2.flip(crop, 1))

        # Sample random negative patches from background regions
        for _ in range(3):
            nw = int(rng.integers(48, 96))
            nh = nw * 2
            if nh >= H or nw >= W:
                continue
            nx1 = int(rng.integers(0, W - nw))
            ny1 = int(rng.integers(0, H - nh))
            neg_patches.append(img[ny1 : ny1 + nh, nx1 : nx1 + nw])

        if len(hard_neg_images) < 80:
            hard_neg_images.append(img)

    if not pos_patches:
        print("[WARNING] No pedestrian crops found in train split; creating synthetic positive samples for test.")
        pos_patches = [np.zeros((128, 64, 3), dtype=np.uint8) for _ in range(10)]

    print(f"[INFO] Training HOG+LinearSVC on {len(pos_patches)} positive and {len(neg_patches)} negative patches...")
    detector = HogSvmPedestrianDetector()
    detector.train_on_kitti(
        pos_patches=pos_patches,
        neg_patches=neg_patches,
        hard_neg_images=hard_neg_images,
        c_param=args.c_param,
        seed=args.seed,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
