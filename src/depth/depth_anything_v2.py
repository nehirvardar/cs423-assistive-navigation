"""Pipeline 1 Depth Model: Depth Anything V2 (Metric Outdoor Checkpoint).

Loads the official Metric Outdoor checkpoint trained on Virtual KITTI 2
(`depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`) via Hugging Face
`transformers`. Predicts dense depth directly in meters without any ground-truth
alignment at test time.
"""

from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.depth.base_depth import BaseDepthEstimator


class DepthAnythingV2MetricOutdoor(BaseDepthEstimator):
    """Zero-shot Metric Outdoor Depth Estimator using Depth Anything V2."""

    DEFAULT_CKPT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CKPT,
        device: Optional[str] = None,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.checkpoint = checkpoint

        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def estimate_depth(
        self,
        image_bgr: np.ndarray,
        gt_sparse_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predicts dense metric depth map (H, W) in meters."""
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, h', w')

        # Interpolate back to original KITTI resolution (H, W)
        pred_resized = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_m = pred_resized.cpu().numpy().astype(np.float32)
        depth_m = np.clip(depth_m, self.min_depth, self.max_depth)
        return depth_m
