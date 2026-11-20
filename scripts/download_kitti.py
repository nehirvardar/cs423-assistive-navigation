"""Automated Official KITTI Dataset Downloader & Extractor (Ideal for Google Colab / Cloud GPUs).

Downloads the official KITTI 2D Object Detection + Velodyne LiDAR + Devkit archives
directly from the official Karlsruhe Institute of Technology (KIT) AWS S3 mirror
(`https://s3.eu-central-1.amazonaws.com/avg-kitti/`) into `data/raw/` and extracts
them into the exact directory layout expected by `01_prepare_data.py`.

Usage on Colab:
  # Download everything (Images + Labels + Calib + Devkit + Velodyne LiDAR ~29 GB):
  python scripts/download_kitti.py

  # Or download without the 26 GB Velodyne zip if you only want to train YOLO/HOG first (~12 GB):
  python scripts/download_kitti.py --skip-velodyne
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile


KITTI_URLS = {
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "devkit": "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip",
    "image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
}


def download_file(url: str, dest_zip: Path) -> None:
    """Downloads a file using `aria2c` or `wget` if available (fast on Colab), else urllib."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists():
        print(f"[INFO] Archive already exists: {dest_zip}")
        return

    print(f"[INFO] Downloading {url} -> {dest_zip} ...")
    if shutil.which("aria2c"):
        subprocess.run(
            ["aria2c", "-x", "8", "-s", "8", "-d", str(dest_zip.parent), "-o", dest_zip.name, url],
            check=True,
        )
    elif shutil.which("wget"):
        subprocess.run(["wget", "-c", "-O", str(dest_zip), url], check=True)
    else:
        urllib.request.urlretrieve(url, str(dest_zip))


def extract_zip(zip_path: Path, extract_to: Path, remove_zip_after: bool = True) -> None:
    """Extracts zip archive and optionally deletes the zip to save Colab disk space."""
    print(f"[INFO] Extracting {zip_path.name} into {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    if remove_zip_after:
        zip_path.unlink(missing_ok=True)
        print(f"[INFO] Removed temporary archive {zip_path.name} to save disk space.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract official KITTI dataset on Google Colab.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Target raw KITTI directory")
    parser.add_argument(
        "--skip-velodyne",
        action="store_true",
        help="Skip the 26 GB Velodyne LiDAR zip (useful when only training YOLO/HOG detectors first)",
    )
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep .zip files after extraction (by default deleted to save Colab disk quota)",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    raw_root.mkdir(parents=True, exist_ok=True)

    # Remove synthetic sample data if it exists (20 synthetic frames) before downloading real KITTI
    img_dir = raw_root / "training" / "image_2"
    if img_dir.exists():
        existing_pngs = list(img_dir.glob("*.png"))
        if 0 < len(existing_pngs) <= 50:
            print("[INFO] Removing synthetic sample frames from data/raw/ before downloading real KITTI...")
            shutil.rmtree(raw_root / "training", ignore_errors=True)
            shutil.rmtree(raw_root / "devkit", ignore_errors=True)

    components = ["calib", "label_2", "devkit", "image_2"]
    if not args.skip_velodyne:
        components.append("velodyne")

    for comp in components:
        url = KITTI_URLS[comp]
        zip_name = url.split("/")[-1]
        zip_path = raw_root / zip_name

        # Target directory: devkit_object.zip extracts to data/raw/devkit/
        target_dir = (raw_root / "devkit") if comp == "devkit" else raw_root
        download_file(url, zip_path)
        extract_zip(zip_path, target_dir, remove_zip_after=not args.keep_zips)

    # Note: devkit_object.zip sometimes extracts into data/raw/devkit/devkit/mapping or data/raw/devkit/mapping
    nested_mapping = raw_root / "devkit" / "devkit" / "mapping"
    target_mapping = raw_root / "devkit" / "mapping"
    if nested_mapping.exists() and not target_mapping.exists():
        shutil.move(str(nested_mapping), str(target_mapping))

    print("\n[SUCCESS] Official KITTI dataset is ready in:", raw_root.resolve())


if __name__ == "__main__":
    main()
