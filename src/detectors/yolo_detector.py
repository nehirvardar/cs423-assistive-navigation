"""Ultralytics YOLOv8-Nano and YOLOv10-Nano Detector Wrapper.

No external repository cloning is needed. Uses the official `ultralytics` package
to fine-tune and run inference with both `yolov10n` (Pipeline 1) and `yolov8n` (Pipeline 2).
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.detectors.base_detector import BaseDetector, CLASS_TO_ID, Detection, ID_TO_CLASS


class YoloDetector(BaseDetector):
    """Wraps Ultralytics YOLO for training across multiple random seeds and inference."""

    # Map COCO class IDs to our KITTI class IDs when running un-finetuned COCO weights as fallback
    COCO_TO_KITTI: Dict[int, int] = {
        2: 0,  # COCO car -> KITTI Car (0)
        0: 1,  # COCO person -> KITTI Pedestrian (1)
        1: 2,  # COCO bicycle -> KITTI Cyclist (2)
    }

    def __init__(
        self,
        weights_path: str | Path = "yolov10n.pt",
        fallback_pretrained: str = "yolov10n.pt",
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = Path(weights_path)
        if weights.exists():
            self.model_path = str(weights)
            self.is_kitti_finetuned = True
        else:
            print(
                f"[INFO] Fine-tuned checkpoint '{weights}' not found. "
                f"Loading pretrained '{fallback_pretrained}' via Ultralytics."
            )
            self.model_path = fallback_pretrained
            self.is_kitti_finetuned = False

        self.model = YOLO(self.model_path)
        # Check if model has 3 classes (KITTI fine-tuned) or 80 classes (COCO pretrained)
        num_classes = len(getattr(self.model, "names", {}))
        if num_classes == 3:
            self.is_kitti_finetuned = True

    def predict(self, image_bgr: np.ndarray, conf_thresh: float = 0.25) -> List[Detection]:
        """Runs YOLO inference on a BGR image and returns standardized Detection objects."""
        results = self.model.predict(
            source=image_bgr,
            conf=conf_thresh,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return detections

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)

        for box, score, raw_cls in zip(xyxy, confs, cls_ids):
            if self.is_kitti_finetuned:
                if raw_cls not in ID_TO_CLASS:
                    continue
                kitti_cls_id = int(raw_cls)
            else:
                if raw_cls not in self.COCO_TO_KITTI:
                    continue
                kitti_cls_id = self.COCO_TO_KITTI[raw_cls]

            detections.append(
                Detection(
                    bbox=box.astype(np.float64),
                    score=float(score),
                    class_id=kitti_cls_id,
                    class_name=ID_TO_CLASS[kitti_cls_id],
                )
            )
        return detections

    @staticmethod
    def fine_tune(
        model_variant: str,
        data_yaml: str | Path,
        seed: int = 42,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        output_dir: str | Path = "checkpoints",
        device: Optional[str] = None,
    ) -> Path:
        """Fine-tunes YOLOv8n or YOLOv10n on KITTI with a fixed random seed for reproducibility."""
        import shutil
        import torch
        from ultralytics import YOLO

        resolved_device = device or ("0" if torch.cuda.is_available() else "cpu")
        base_weights = f"{model_variant}.pt" if not model_variant.endswith(".pt") else model_variant
        model_name = Path(base_weights).stem

        model = YOLO(base_weights)
        run_name = f"{model_name}_kitti_seed{seed}"
        results = model.train(
            data=str(Path(data_yaml).resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=seed,
            deterministic=True,
            device=resolved_device,
            project="runs/detect",
            name=run_name,
            exist_ok=True,
        )

        # Copy best checkpoint to checkpoints/<model_name>_kitti_seed<seed>.pt
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        target_ckpt = out_path / f"{model_name}_kitti_seed{seed}.pt"

        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, target_ckpt)
            print(f"[SUCCESS] Saved fine-tuned checkpoint to {target_ckpt}")
        return target_ckpt
