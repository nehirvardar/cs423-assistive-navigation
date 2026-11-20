"""LiDAR 3D Point Cloud to 2D Image Plane Sparse Metric Depth Projection.

Projects KITTI Velodyne (.bin) 3D points into the Left Color Camera (image_2)
coordinate frame using KittiCalibration matrices, producing a sparse 2D metric
depth map (in meters) where 0.0 indicates no valid LiDAR return.
"""

from pathlib import Path
from typing import Tuple
import numpy as np

from src.data_prep.kitti_calibration import KittiCalibration


def load_velodyne_points(bin_path: str | Path) -> np.ndarray:
    """Loads KITTI Velodyne .bin file containing [x, y, z, reflectance] float32 points."""
    points = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    return points[:, :3]  # Keep only [x, y, z]


def project_velodyne_to_depth_map(
    velo_points_xyz: np.ndarray,
    calib: KittiCalibration,
    image_shape: Tuple[int, int],
    min_depth: float = 0.001,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Projects 3D Velodyne points onto a 2D sparse depth map (H, W) in meters.

    Args:
        velo_points_xyz: (N, 3) array of 3D points in Velodyne coordinates.
        calib: KittiCalibration instance for the frame.
        image_shape: (H, W) of the RGB image.
        min_depth: Minimum valid depth in meters.
        max_depth: Maximum valid depth in meters.

    Returns:
        depth_map: (H, W) float32 array in meters (0.0 where no LiDAR point projects).
    """
    H, W = image_shape[:2]
    depth_map = np.zeros((H, W), dtype=np.float32)

    if velo_points_xyz.shape[0] == 0:
        return depth_map

    # Filter points in front of the Velodyne sensor (x > 0)
    front_mask = velo_points_xyz[:, 0] > 0.1
    pts = velo_points_xyz[front_mask]
    if pts.shape[0] == 0:
        return depth_map

    # Convert to homogeneous coordinates (N, 4)
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])

    # Transform to rectified camera coordinates to get metric depth Z_cam
    pts_cam_rect = (calib.Velo_to_Cam_rect @ pts_hom.T).T  # (N, 4)
    z_cam = pts_cam_rect[:, 2]

    # Filter points within valid metric depth range [min_depth, max_depth]
    valid_z = (z_cam >= min_depth) & (z_cam <= max_depth)
    pts_hom = pts_hom[valid_z]
    z_cam = z_cam[valid_z]

    if pts_hom.shape[0] == 0:
        return depth_map

    # Project to 2D image coordinates: y_img = P2 * R0_rect * Tr_velo_to_cam * X
    pts_img_hom = (calib.P_velo_to_img @ pts_hom.T).T  # (N, 3)
    u = np.round(pts_img_hom[:, 0] / pts_img_hom[:, 2]).astype(np.int32)
    v = np.round(pts_img_hom[:, 1] / pts_img_hom[:, 2]).astype(np.int32)

    # Filter points inside image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds]
    v = v[in_bounds]
    z_cam = z_cam[in_bounds].astype(np.float32)

    # Sort from farthest to nearest so that nearer points overwrite farther ones on collision
    order = np.argsort(-z_cam)
    u = u[order]
    v = v[order]
    z_cam = z_cam[order]

    depth_map[v, u] = z_cam
    return depth_map
