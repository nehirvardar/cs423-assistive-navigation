"""Classical Baseline Detector: Dalal & Triggs (2005) HOG + Linear SVM for Pedestrians.

Supports:
1. Custom training on KITTI Pedestrian crops + background negative crops with Hard Negative Mining.
2. Saving/loading trained SVM weights via joblib.
3. Fallback to OpenCV's pre-calibrated Dalal & Triggs DefaultPeopleDetector when custom weights
   have not yet been trained, ensuring the pipeline can always be tested immediately.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import joblib
import numpy as np
from sklearn.svm import LinearSVC

from src.detectors.base_detector import BaseDetector, CLASS_TO_ID, Detection


def non_max_suppression_fast(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4,
) -> List[int]:
    """Applies Non-Maximum Suppression (NMS) on [x1, y1, x2, y2] boxes."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class HogSvmPedestrianDetector(BaseDetector):
    """Dalal & Triggs HOG + Linear SVM Pedestrian Detector (64x128 canonical window)."""

    WIN_SIZE: Tuple[int, int] = (64, 128)
    BLOCK_SIZE: Tuple[int, int] = (16, 16)
    BLOCK_STRIDE: Tuple[int, int] = (8, 8)
    CELL_SIZE: Tuple[int, int] = (8, 8)
    NBINS: int = 9

    def __init__(self, weights_path: Optional[str | Path] = None):
        self.hog = cv2.HOGDescriptor(
            self.WIN_SIZE,
            self.BLOCK_SIZE,
            self.BLOCK_STRIDE,
            self.CELL_SIZE,
            self.NBINS,
        )
        self.weights_path = Path(weights_path) if weights_path else None

        if self.weights_path and self.weights_path.exists():
            saved = joblib.load(self.weights_path)
            support_vec = saved["svm_detector"].astype(np.float32)
            self.hog.setSVMDetector(support_vec)
            self.is_custom_trained = True
        else:
            # OpenCV built-in Dalal & Triggs 64x128 pedestrian detector vector
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.is_custom_trained = False

    def extract_feature(self, patch_bgr: np.ndarray) -> np.ndarray:
        """Resizes image patch to 64x128 and computes 3,780-D HOG feature vector."""
        resized = cv2.resize(patch_bgr, self.WIN_SIZE, interpolation=cv2.INTER_LINEAR)
        feat = self.hog.compute(resized)
        return np.asarray(feat, dtype=np.float32).reshape(-1)

    def predict(self, image_bgr: np.ndarray, conf_thresh: float = 0.0) -> List[Detection]:
        """Detects pedestrians using multi-scale sliding window HOG + SVM + NMS."""
        rects, weights = self.hog.detectMultiScale(
            image_bgr,
            hitThreshold=0.0,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )
        if len(rects) == 0:
            return []

        boxes = []
        scores = []
        for (x, y, w, h), weight in zip(rects, weights):
            score = float(weight[0]) if isinstance(weight, (list, np.ndarray)) else float(weight)
            # Convert SVM margin to [0, 1] pseudo-probability via sigmoid for mAP sorting
            prob = float(1.0 / (1.0 + np.exp(-score)))
            if prob < conf_thresh:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            scores.append(prob)

        if not boxes:
            return []

        boxes_np = np.array(boxes, dtype=np.float64)
        scores_np = np.array(scores, dtype=np.float64)
        keep_indices = non_max_suppression_fast(boxes_np, scores_np, iou_threshold=0.4)

        detections: List[Detection] = []
        for idx in keep_indices:
            detections.append(
                Detection(
                    bbox=boxes_np[idx],
                    score=float(scores_np[idx]),
                    class_id=CLASS_TO_ID["Pedestrian"],
                    class_name="Pedestrian",
                )
            )
        return detections

    def train_on_kitti(
        self,
        pos_patches: List[np.ndarray],
        neg_patches: List[np.ndarray],
        hard_neg_images: Optional[List[np.ndarray]] = None,
        c_param: float = 0.01,
        seed: int = 42,
        save_path: Optional[str | Path] = None,
    ) -> LinearSVC:
        """Trains Linear SVM on HOG features with 1 round of Hard Negative Mining."""
        X_pos = [self.extract_feature(p) for p in pos_patches]
        X_neg = [self.extract_feature(n) for n in neg_patches]

        X = np.array(X_pos + X_neg, dtype=np.float32)
        y = np.concatenate([
            np.ones(len(X_pos), dtype=np.int32),
            np.zeros(len(X_neg), dtype=np.int32),
        ])

        svm = LinearSVC(C=c_param, class_weight="balanced", random_state=seed, max_iter=5000)
        svm.fit(X, y)

        # Update detector vector for Hard Negative Mining
        primal_vector = np.append(svm.coef_.flatten(), svm.intercept_[0]).astype(np.float32)
        self.hog.setSVMDetector(primal_vector)

        # Hard Negative Mining on negative/background regions
        if hard_neg_images:
            hard_negs: List[np.ndarray] = []
            for img in hard_neg_images[:100]:
                rects, _ = self.hog.detectMultiScale(
                    img, hitThreshold=0.1, winStride=(8, 8), padding=(8, 8), scale=1.08
                )
                for (rx, ry, rw, rh) in rects[:5]:
                    x1, y1 = max(0, int(rx)), max(0, int(ry))
                    x2, y2 = min(img.shape[1], int(rx + rw)), min(img.shape[0], int(ry + rh))
                    if x2 - x1 >= 16 and y2 - y1 >= 32:
                        hard_negs.append(self.extract_feature(img[y1:y2, x1:x2]))

            if len(hard_negs) > 0:
                hard_negs_arr = np.array(hard_negs, dtype=np.float32)
                X = np.concatenate([X, hard_negs_arr], axis=0)
                y = np.concatenate([y, np.zeros(len(hard_negs), dtype=np.int32)], axis=0)
                svm.fit(X, y)
                primal_vector = np.append(svm.coef_.flatten(), svm.intercept_[0]).astype(np.float32)
                self.hog.setSVMDetector(primal_vector)

        self.is_custom_trained = True
        if save_path:
            out_file = Path(save_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"svm_detector": primal_vector, "C": c_param, "seed": seed}, out_file)
            print(f"[SUCCESS] Saved trained HOG+SVM Pedestrian detector to {out_file}")

        return svm
