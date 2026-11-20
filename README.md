# Monocular Obstacle Detection and Distance Estimation for Assistive Navigation

**Course:** CS 423/523 Computer Vision (Fall 2026) — **Track A (Application)**  
**Team Members:** Esma Nehir Vardar (S040236), Ali Raza Mustafa (S026665), Emirhan Tarlabaşı (S033747)

---

## 1. Overview & Architecture

This repository implements and compares **three monocular obstacle detection and distance estimation pipelines** on the **KITTI Object Detection & LiDAR Benchmark** under a single unified evaluation protocol:

1. **Pipeline 1 (Main):** `YOLOv10-Nano` (fine-tuned across 3 random seeds) + `Depth Anything V2` (*Metric Outdoor VKITTI2 checkpoint*, predicting dense depth directly in meters without ground-truth alignment).
2. **Pipeline 2 (Secondary):** `YOLOv8-Nano` (fine-tuned across 3 random seeds) + `MiDaS Small` (*relative inverse depth aligned per-image via least-squares scale and shift against sparse LiDAR depth*).
3. **Pipeline 3 (Classical Baseline):** `Dalal & Triggs HOG + Linear SVM` (Pedestrian detector with Hard Negative Mining) + `Pinhole Camera Geometry` ($Z = \frac{f \cdot H}{h}$, with focal length $f$ from KITTI calibration and assumed pedestrian height $H = 1.70\text{ m}$).

---

## 2. Environment Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Install dependencies:
pip install -r requirements.txt
```

---

## 3. Dataset Setup (KITTI)

Place the official KITTI 2D Object Detection files (`data_object_image_2.zip`, `data_object_label_2.zip`, `data_object_calib.zip`, `data_object_velodyne.zip`, and `devkit_object.zip`) inside `data/raw/` so that the structure matches:

```text
data/raw/
├── training/
│   ├── image_2/      # 000000.png ... 007480.png
│   ├── label_2/      # 000000.txt ... 007480.txt
│   ├── calib/        # 000000.txt ... 007480.txt
│   └── velodyne/     # 000000.bin ... 007480.bin
└── devkit/
    └── mapping/
        ├── train_rand.txt
        └── train_mapping.txt
```

> **Automatic Download on Google Colab:**  
> Instead of downloading 28 GB to your local computer, run the automated downloader directly on Colab:
> ```bash
> python scripts/download_kitti.py
> ```

---

## 4. Reproducing All Main Results

### Step 1: Drive-Sequence Split (70/10/20), LiDAR Projection & YOLO Export
Prevents data leakage by grouping frames strictly by their source video drive sequence (`train_mapping.txt`), projects 3D Velodyne point clouds onto 2D sparse metric depth maps (`data/processed/depth_maps/`), and exports YOLO labels:
```bash
python scripts/01_prepare_data.py
```

### Step 2: Train Classical Baseline (HOG + Linear SVM Pedestrian Detector)
Trains the Dalal & Triggs HOG + LinearSVC detector with Hard Negative Mining on the training split:
```bash
python scripts/02_train_hog.py
```

### Step 3: Fine-Tune YOLOv10-Nano and YOLOv8-Nano Across 3 Random Seeds
Trains both detectors with seeds `42`, `123`, and `999` to report `mean ± std`:
```bash
python scripts/03_train_yolo.py --model yolov10n --seeds 42 123 999 --epochs 50
python scripts/03_train_yolo.py --model yolov8n  --seeds 42 123 999 --epochs 50
```

### Step 4: Unified Evaluation & Ablation Study on Test Split
Evaluates all 3 pipelines on `data/splits/test.txt`, computing per-class `mAP@0.5` and `mAP@0.5:0.95`, pixel-level LiDAR depth metrics (`AbsRel`, `RMSE`, $\delta < 1.25$), object-level distance errors on matched detections (`IoU >= 0.5`), and the Bounding-Box Background Filtering Ablation Study:
```bash
python scripts/04_evaluate_all.py --split test
```

### Step 5: Hardware Latency & FPS Benchmark
Measures detector latency, depth latency, and end-to-end throughput (FPS):
```bash
python scripts/05_benchmark_speed.py
```
