"""Drive-Sequence Based Train / Validation / Test Split Generator for KITTI.

Uses KITTI devkit mapping files (`train_rand.txt` and `train_mapping.txt`)
to group all 7,481 labeled images by their source video drive sequence
(e.g. `2011_09_26_drive_0001_sync`). Ensures all frames from the same drive
sequence are placed in the exact same split (~70% train / 10% val / 20% test)
to prevent data leakage between neighboring frames.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def parse_devkit_drive_mapping(
    rand_file: str | Path,
    mapping_file: str | Path,
    available_ids: List[str],
) -> Dict[str, str]:
    """Maps each 6-digit KITTI image ID (e.g. '000000') to its drive sequence name.

    If the devkit mapping files are not present on disk (e.g. during a dry-run),
    falls back to contiguous chunk grouping while logging a warning.
    """
    rand_path = Path(rand_file)
    map_path = Path(mapping_file)

    if rand_path.exists() and map_path.exists():
        with open(rand_path, "r", encoding="utf-8") as f:
            rand_indices = [int(x.strip()) for x in f.read().strip().split(",") if x.strip()]

        with open(map_path, "r", encoding="utf-8") as f:
            mapping_lines = [line.strip() for line in f if line.strip()]

        id_to_drive: Dict[str, str] = {}
        for img_idx, raw_line_idx in enumerate(rand_indices):
            img_id = f"{img_idx:06d}"
            # raw_line_idx is 1-based index into train_mapping.txt
            if 1 <= raw_line_idx <= len(mapping_lines):
                parts = mapping_lines[raw_line_idx - 1].split()
                # Format: "2011_09_26 2011_09_26_drive_0005_sync 0000000123"
                drive_name = parts[1] if len(parts) >= 2 else f"drive_{raw_line_idx}"
                id_to_drive[img_id] = drive_name

        # Ensure every available_id has a drive assignment
        for img_id in available_ids:
            if img_id not in id_to_drive:
                id_to_drive[img_id] = f"unmapped_{img_id}"
        return id_to_drive

    # Fallback when devkit mapping files are not yet downloaded
    print(
        "[WARNING] KITTI devkit mapping files not found at "
        f"{rand_path} and {map_path}. Grouping into synthetic contiguous sequences."
    )
    id_to_drive = {}
    sorted_ids = sorted(available_ids)
    chunk_size = max(1, len(sorted_ids) // 50) if len(sorted_ids) >= 50 else 1
    for i, img_id in enumerate(sorted_ids):
        id_to_drive[img_id] = f"synthetic_drive_{i // chunk_size:03d}"
    return id_to_drive


def create_drive_sequence_splits(
    available_ids: List[str],
    id_to_drive: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Splits image IDs into train, val, and test sets grouped strictly by drive sequence.

    Uses greedy bucket assignment over shuffled drive sequences to hit the target
    70% / 10% / 20% frame counts as closely as possible without ever splitting a drive.
    """
    drive_to_ids: Dict[str, List[str]] = defaultdict(list)
    for img_id in sorted(available_ids):
        drive_name = id_to_drive.get(img_id, f"drive_{img_id}")
        drive_to_ids[drive_name].append(img_id)

    drives = list(drive_to_ids.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(drives)

    total_frames = len(available_ids)
    target_train = int(round(total_frames * train_ratio))
    target_val = int(round(total_frames * val_ratio))

    train_drives_list: List[str] = []
    val_drives_list: List[str] = []
    test_drives_list: List[str] = []

    train_count = 0
    val_count = 0

    for drive in drives:
        n_frames = len(drive_to_ids[drive])
        if train_count < target_train:
            train_drives_list.append(drive)
            train_count += n_frames
        elif val_count < target_val:
            val_drives_list.append(drive)
            val_count += n_frames
        else:
            test_drives_list.append(drive)

    # Ensure val and test each have at least 1 whole drive when total drives >= 3
    if len(drives) >= 3:
        if not val_drives_list and len(train_drives_list) > 1:
            val_drives_list.append(train_drives_list.pop())
        if not test_drives_list and len(train_drives_list) > 1:
            test_drives_list.append(train_drives_list.pop())

    train_ids = [fid for d in train_drives_list for fid in drive_to_ids[d]]
    val_ids = [fid for d in val_drives_list for fid in drive_to_ids[d]]
    test_ids = [fid for d in test_drives_list for fid in drive_to_ids[d]]

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def save_splits(
    splits_dir: str | Path,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> None:
    """Writes train.txt, val.txt, and test.txt containing 6-digit frame IDs."""
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    for name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        file_path = splits_path / f"{name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for img_id in id_list:
                f.write(f"{img_id}\n")
