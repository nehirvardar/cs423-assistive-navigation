"""Bounding Box Object Distance Extraction & Background Filtering Strategies.

Addresses the proposal risk:
'Inaccurate Depth Inside Bounding Boxes: Background pixels in a bounding box can lead
to inaccurate calculation of the median distance. Fallback Plan: We shall attempt to
filter out background pixels before calculating the median depth.'

Provides three strategies for ablation study:
1. `full_median`: Median of all valid depth pixels inside the full 2D bounding box.
2. `center_crop_median`: Crops the central region (e.g. central 50% width & height)
   of the box to eliminate background pixels near box corners/edges, then computes median.
3. `trimmed_lower_median`: Filters out the farthest 40% of pixels inside the box
   (which typically belong to background walls/sky behind the obstacle) and takes the
   median of the foreground cluster (15th to 60th depth percentiles).
"""

import numpy as np


def extract_distance_from_bbox(
    depth_map: np.ndarray,
    bbox: np.ndarray,
    strategy: str = "center_crop_median",
    center_crop_ratio: float = 0.50,
    min_depth: float = 0.001,
    max_depth: float = 80.0,
) -> float:
    """Computes object distance (in meters) from a 2D depth map and a bounding box [x1, y1, x2, y2]."""
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox

    x1_i = int(np.clip(np.floor(x1), 0, W - 1))
    y1_i = int(np.clip(np.floor(y1), 0, H - 1))
    x2_i = int(np.clip(np.ceil(x2), x1_i + 1, W))
    y2_i = int(np.clip(np.ceil(y2), y1_i + 1, H))

    if strategy == "center_crop_median":
        bw = x2_i - x1_i
        bh = y2_i - y1_i
        cx = x1_i + bw / 2.0
        cy = y1_i + bh / 2.0
        half_w = max(1.0, (bw * center_crop_ratio) / 2.0)
        half_h = max(1.0, (bh * center_crop_ratio) / 2.0)

        cx1 = int(np.clip(np.floor(cx - half_w), x1_i, x2_i - 1))
        cy1 = int(np.clip(np.floor(cy - half_h), y1_i, y2_i - 1))
        cx2 = int(np.clip(np.ceil(cx + half_w), cx1 + 1, x2_i))
        cy2 = int(np.clip(np.ceil(cy + half_h), cy1 + 1, y2_i))
        patch = depth_map[cy1:cy2, cx1:cx2]
    else:
        patch = depth_map[y1_i:y2_i, x1_i:x2_i]

    valid_pixels = patch[(patch >= min_depth) & (patch <= max_depth)]

    # If center crop had no valid pixels (e.g. sparse LiDAR map), fall back to full box
    if valid_pixels.size == 0 and strategy != "full_median":
        full_patch = depth_map[y1_i:y2_i, x1_i:x2_i]
        valid_pixels = full_patch[(full_patch >= min_depth) & (full_patch <= max_depth)]

    if valid_pixels.size == 0:
        return -1.0

    if strategy == "trimmed_lower_median" and valid_pixels.size >= 5:
        p15, p60 = np.percentile(valid_pixels, [15, 60])
        fg_pixels = valid_pixels[(valid_pixels >= p15) & (valid_pixels <= p60)]
        if fg_pixels.size > 0:
            return float(np.median(fg_pixels))

    return float(np.median(valid_pixels))
