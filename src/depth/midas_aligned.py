"""Pipeline 2 Depth Model: MiDaS Small with Per-Image Ground-Truth LiDAR Scale-Shift Alignment.

As stated in the project proposal:
MiDaS predicts relative inverse depth (disparity) only. For each image, we fit a
least-squares scale (s) and shift (t) against the valid ground-truth LiDAR depth pixels,
giving MiDaS an explicit oracle alignment advantage that Depth Anything V2 Metric does not get.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.depth.base_depth import BaseDepthEstimator


def align_depth_least_squares(
    pred_relative: np.ndarray,
    gt_sparse_depth: np.ndarray,
    min_depth: float = 0.001,
    max_depth: float = 80.0,
    align_in_disparity_space: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Fits per-image least-squares scale (s) and shift (t) to align MiDaS output to LiDAR depth.

    Standard Ranftl et al. (MiDaS) alignment fits s and t in inverse depth (disparity) space:
        d_gt_inv = 1.0 / d_gt
        pred_disp_aligned = s * pred_relative + t
        d_aligned = 1.0 / max(eps, pred_disp_aligned)
    Also evaluates direct linear fit in depth space and picks the lower RMSE fit so that
    whether the checkpoint outputs relative inverse depth or relative depth, alignment is optimal.
    """
    valid_mask = (gt_sparse_depth >= min_depth) & (gt_sparse_depth <= max_depth)
    if np.count_nonzero(valid_mask) < 10:
        # Fallback if no valid LiDAR points are provided
        norm = (pred_relative - pred_relative.min()) / (pred_relative.max() - pred_relative.min() + 1e-6)
        fallback_depth = np.clip(1.0 / (norm + 0.05) * 2.0, min_depth, max_depth)
        return fallback_depth.astype(np.float32), 1.0, 0.0

    p_valid = pred_relative[valid_mask].astype(np.float64)
    gt_valid = gt_sparse_depth[valid_mask].astype(np.float64)

    A = np.vstack([p_valid, np.ones_like(p_valid)]).T

    # 1. Fit in inverse depth (disparity) space (MiDaS native representation)
    gt_inv = 1.0 / gt_valid
    sol_inv, _, _, _ = np.linalg.lstsq(A, gt_inv, rcond=None)
    s_inv, t_inv = float(sol_inv[0]), float(sol_inv[1])
    aligned_inv_disp = np.maximum(1.0 / max_depth, s_inv * pred_relative + t_inv)
    depth_from_inv = np.clip(1.0 / aligned_inv_disp, min_depth, max_depth).astype(np.float32)

    # 2. Direct linear fit in metric depth space
    sol_dir, _, _, _ = np.linalg.lstsq(A, gt_valid, rcond=None)
    s_dir, t_dir = float(sol_dir[0]), float(sol_dir[1])
    depth_from_dir = np.clip(s_dir * pred_relative + t_dir, min_depth, max_depth).astype(np.float32)

    if not align_in_disparity_space:
        return depth_from_dir, s_dir, t_dir

    # Compare RMSE on valid LiDAR pixels and return the better aligned map
    rmse_inv = np.sqrt(np.mean((depth_from_inv[valid_mask] - gt_valid) ** 2))
    rmse_dir = np.sqrt(np.mean((depth_from_dir[valid_mask] - gt_valid) ** 2))

    if rmse_inv <= rmse_dir:
        return depth_from_inv, s_inv, t_inv
    return depth_from_dir, s_dir, t_dir


class MidasSmallAlignedEstimator(BaseDepthEstimator):
    """MiDaS Small depth estimator with per-image least-squares LiDAR alignment."""

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: Optional[str] = None,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.model_type = model_type

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True).to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def predict_relative(self, image_bgr: np.ndarray) -> np.ndarray:
        """Runs MiDaS Small forward pass and returns relative inverse depth at (H, W)."""
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image_rgb).to(self.device)

        prediction = self.model(input_batch)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy().astype(np.float32)

    def estimate_depth(
        self,
        image_bgr: np.ndarray,
        gt_sparse_depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predicts relative depth and aligns it per-image against GT LiDAR depth."""
        rel_pred = self.predict_relative(image_bgr)
        if gt_sparse_depth is None:
            gt_sparse_depth = np.zeros_like(rel_pred)

        aligned_depth, _, _ = align_depth_least_squares(
            pred_relative=rel_pred,
            gt_sparse_depth=gt_sparse_depth,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
        return aligned_depth
