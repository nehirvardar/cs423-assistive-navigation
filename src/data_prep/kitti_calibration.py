"""KITTI Calibration and Ground-Truth Label Parser.

Reads KITTI camera calibration matrices (P2, R0_rect, Tr_velo_to_cam),
extracts focal length (f_x, f_y) for the Pinhole baseline, and parses
2D/3D object annotations from KITTI label_2 files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


@dataclass
class KittiObject:
    """Represents a single annotated object in KITTI label_2 format."""
    cls_name: str
    truncation: float
    occlusion: int
    alpha: float
    bbox: np.ndarray        # [x1, y1, x2, y2] in pixels
    dimensions: np.ndarray  # [h, w, l] in meters
    location: np.ndarray    # [x, y, z] in camera coordinates (meters); z is depth
    rotation_y: float

    @property
    def gt_distance_z(self) -> float:
        """Returns the 3D camera-coordinate Z depth (in meters)."""
        return float(self.location[2])

    @property
    def box_height_px(self) -> float:
        """Returns 2D bounding box height in pixels (y2 - y1)."""
        return float(max(1.0, self.bbox[3] - self.bbox[1]))


class KittiCalibration:
    """Parses a KITTI calibration file (.txt) and provides projection utilities."""

    def __init__(self, calib_path: str | Path):
        self.calib_path = Path(calib_path)
        self.data: Dict[str, np.ndarray] = self._read_calib_file(self.calib_path)

        # P2: 3x4 projection matrix for Left Color Camera (image_2)
        self.P2 = self.data["P2"].reshape(3, 4)

        # R0_rect: 3x3 rectifying rotation matrix -> padded to 4x4
        R0 = self.data["R0_rect"].reshape(3, 3)
        self.R0_rect = np.eye(4, dtype=np.float64)
        self.R0_rect[:3, :3] = R0

        # Tr_velo_to_cam: 3x4 Velodyne to reference camera transformation -> padded to 4x4
        Tr = self.data["Tr_velo_to_cam"].reshape(3, 4)
        self.Tr_velo_to_cam = np.eye(4, dtype=np.float64)
        self.Tr_velo_to_cam[:3, :4] = Tr

        # Full 3x4 projection matrix from 3D Velodyne homogeneous coords to image plane
        self.Velo_to_Cam_rect = self.R0_rect @ self.Tr_velo_to_cam
        self.P_velo_to_img = self.P2 @ self.Velo_to_Cam_rect

    @staticmethod
    def _read_calib_file(filepath: Path) -> Dict[str, np.ndarray]:
        data: Dict[str, np.ndarray] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                vals = [float(x) for x in value.strip().split()]
                data[key.strip()] = np.array(vals, dtype=np.float64)
        return data

    @property
    def focal_length_y(self) -> float:
        """Focal length f_y from P2[1, 1] (used for vertical height pinhole formula Z = f * H / h)."""
        return float(self.P2[1, 1])

    @property
    def focal_length_x(self) -> float:
        """Focal length f_x from P2[0, 0]."""
        return float(self.P2[0, 0])


def parse_kitti_label_file(
    label_path: str | Path,
    class_mapping: Optional[Dict[str, Optional[str]]] = None,
) -> List[KittiObject]:
    """Parses a KITTI label_2 text file into a list of KittiObject instances."""
    objects: List[KittiObject] = []
    label_path = Path(label_path)
    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            raw_cls = parts[0]
            if class_mapping is not None:
                mapped_cls = class_mapping.get(raw_cls, None)
                if mapped_cls is None:
                    continue
            else:
                if raw_cls == "DontCare":
                    continue
                mapped_cls = raw_cls

            obj = KittiObject(
                cls_name=mapped_cls,
                truncation=float(parts[1]),
                occlusion=int(float(parts[2])),
                alpha=float(parts[3]),
                bbox=np.array(
                    [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    dtype=np.float64,
                ),
                dimensions=np.array(
                    [float(parts[8]), float(parts[9]), float(parts[10])],
                    dtype=np.float64,
                ),
                location=np.array(
                    [float(parts[11]), float(parts[12]), float(parts[13])],
                    dtype=np.float64,
                ),
                rotation_y=float(parts[14]),
            )
            objects.append(obj)
    return objects
