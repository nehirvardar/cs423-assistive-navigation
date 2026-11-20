"""Converts KITTI 2D Bounding Box Annotations to Ultralytics YOLO Format.

Creates the directory structure expected by Ultralytics YOLO:
data/yolo_dataset/
  images/{train,val,test}/000000.png
  labels/{train,val,test}/000000.txt
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from src.data_prep.kitti_calibration import parse_kitti_label_file


def kitti_bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Converts [x1, y1, x2, y2] pixel coordinates to normalized [x_c, y_c, w, h]."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(img_w), x1))
    x2 = max(0.0, min(float(img_w), x2))
    y1 = max(0.0, min(float(img_h), y1))
    y2 = max(0.0, min(float(img_h), y2))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return (xc / img_w, yc / img_h, bw / img_w, bh / img_h)


def export_split_to_yolo(
    split_name: str,
    image_ids: List[str],
    raw_image_dir: str | Path,
    raw_label_dir: str | Path,
    yolo_root_dir: str | Path,
    classes_dict: Dict[str, int],
    class_mapping: Optional[Dict[str, Optional[str]]] = None,
    use_hardlinks_or_copy: bool = True,
) -> int:
    """Exports a single split (train, val, or test) to Ultralytics YOLO format."""
    raw_img_path = Path(raw_image_dir)
    raw_lbl_path = Path(raw_label_dir)
    yolo_root = Path(yolo_root_dir)

    out_img_dir = yolo_root / "images" / split_name
    out_lbl_dir = yolo_root / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    for img_id in image_ids:
        # Find image file (.png or .jpg)
        src_img = raw_img_path / f"{img_id}.png"
        if not src_img.exists():
            src_img = raw_img_path / f"{img_id}.jpg"
        if not src_img.exists():
            continue

        with Image.open(src_img) as im:
            img_w, img_h = im.size

        dst_img = out_img_dir / src_img.name
        if not dst_img.exists() and use_hardlinks_or_copy:
            try:
                os.link(src_img, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        # Parse KITTI labels and write YOLO normalized labels
        src_lbl = raw_lbl_path / f"{img_id}.txt"
        objects = parse_kitti_label_file(src_lbl, class_mapping=class_mapping)

        dst_lbl = out_lbl_dir / f"{img_id}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for obj in objects:
                if obj.cls_name not in classes_dict:
                    continue
                cls_id = classes_dict[obj.cls_name]
                xc, yc, bw, bh = kitti_bbox_to_yolo(
                    (obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]),
                    img_w=img_w,
                    img_h=img_h,
                )
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        exported_count += 1

    return exported_count
