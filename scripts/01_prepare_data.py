"""Step 1: KITTI Data Preparation Script.

1. Performs Drive-Sequence based Train/Val/Test split (~70/10/20) using KITTI devkit mapping
   so that neighboring video frames never end up in different splits (no data leakage).
2. Projects 3D Velodyne LiDAR point clouds (.bin) into 2D sparse metric depth maps (.npy).
3. Converts KITTI 2D object annotations into Ultralytics YOLO format.
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import yaml

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.convert_to_yolo import export_split_to_yolo
from src.data_prep.drive_split import (
    create_drive_sequence_splits,
    parse_devkit_drive_mapping,
    save_splits,
)
from src.data_prep.kitti_calibration import KittiCalibration
from src.data_prep.lidar_projection import load_velodyne_points, project_velodyne_to_depth_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KITTI drive-sequence splits, LiDAR depth maps, and YOLO labels.")
    parser.add_argument("--config", type=str, default="configs/split_config.yaml", help="Path to split_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    img_dir = Path(ds_cfg["image_dir"])
    lbl_dir = Path(ds_cfg["label_dir"])
    cal_dir = Path(ds_cfg["calib_dir"])
    vel_dir = Path(ds_cfg["velodyne_dir"])
    splits_dir = Path(ds_cfg["splits_dir"])
    processed_dir = Path(ds_cfg["processed_dir"])
    yolo_dir = Path(ds_cfg["yolo_dataset_dir"])

    if not img_dir.exists() or not any(img_dir.iterdir()):
        print(
            f"[ERROR] No KITTI images found in {img_dir}.\n"
            "Please download the official KITTI dataset first by running:\n"
            "  python scripts/download_kitti.py"
        )
        sys.exit(1)

    available_ids = sorted(
        [p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg")]
    )
    print(f"[INFO] Found {len(available_ids)} KITTI images in {img_dir}")

    # 1. Drive-Sequence Grouping & Split
    id_to_drive = parse_devkit_drive_mapping(
        rand_file=ds_cfg["devkit_rand_file"],
        mapping_file=ds_cfg["devkit_mapping_file"],
        available_ids=available_ids,
    )
    train_ids, val_ids, test_ids = create_drive_sequence_splits(
        available_ids=available_ids,
        id_to_drive=id_to_drive,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        seed=cfg["split"]["random_seed"],
    )
    save_splits(splits_dir, train_ids, val_ids, test_ids)

    # Verify zero drive overlap across splits
    train_drives = {id_to_drive[i] for i in train_ids}
    val_drives = {id_to_drive[i] for i in val_ids}
    test_drives = {id_to_drive[i] for i in test_ids}
    assert train_drives.isdisjoint(val_drives), "Data leakage detected between train and val drives!"
    assert train_drives.isdisjoint(test_drives), "Data leakage detected between train and test drives!"
    assert val_drives.isdisjoint(test_drives), "Data leakage detected between val and test drives!"

    print(
        f"[SUCCESS] Drive-sequence split complete (0 drive leakage):\n"
        f"  Train: {len(train_ids)} frames ({len(train_drives)} drives)\n"
        f"  Val:   {len(val_ids)} frames ({len(val_drives)} drives)\n"
        f"  Test:  {len(test_ids)} frames ({len(test_drives)} drives)"
    )

    # 2. Project Velodyne LiDAR points to 2D sparse metric depth maps
    depth_out_dir = processed_dir / "depth_maps"
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    projected_count = 0

    for img_id in tqdm(available_ids, desc="Projecting LiDAR to 2D depth"):
        bin_path = vel_dir / f"{img_id}.bin"
        cal_path = cal_dir / f"{img_id}.txt"
        img_path = img_dir / f"{img_id}.png"
        if not img_path.exists():
            img_path = img_dir / f"{img_id}.jpg"

        if not (bin_path.exists() and cal_path.exists() and img_path.exists()):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        calib = KittiCalibration(cal_path)
        velo_pts = load_velodyne_points(bin_path)
        depth_map = project_velodyne_to_depth_map(velo_pts, calib, img.shape[:2])
        np.save(depth_out_dir / f"{img_id}.npy", depth_map)
        projected_count += 1

    print(f"[SUCCESS] Projected {projected_count} Velodyne LiDAR point clouds to {depth_out_dir}")

    # 3. Export splits to Ultralytics YOLO format
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        cnt = export_split_to_yolo(
            split_name=split_name,
            image_ids=split_ids,
            raw_image_dir=img_dir,
            raw_label_dir=lbl_dir,
            yolo_root_dir=yolo_dir,
            classes_dict=cfg["classes"],
            class_mapping=cfg.get("class_mapping"),
        )
        print(f"[SUCCESS] Exported {cnt} frames for YOLO '{split_name}' split to {yolo_dir}")


if __name__ == "__main__":
    main()
