"""Qualitative Visualization and Failure-Case Error Analysis Plots for IEEE Final Report."""

from pathlib import Path
from typing import Dict, List
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.matching import MatchedObstaclePair
from src.pipelines.runner import PipelineOutput


CLASS_COLORS = {
    "Car": (0, 200, 0),         # Green
    "Pedestrian": (0, 140, 255), # Orange
    "Cyclist": (255, 100, 0),    # Blue
}


def draw_obstacles_on_image(
    image_bgr: np.ndarray,
    output: PipelineOutput,
    title: str = "",
) -> np.ndarray:
    """Draws 2D bounding boxes, class labels, and estimated distances (in meters) onto image."""
    vis = image_bgr.copy()
    for obs in output.obstacles:
        x1, y1, x2, y2 = [int(round(v)) for v in obs.bbox]
        color = CLASS_COLORS.get(obs.class_name, (0, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{obs.class_name} {obs.distance_m:.1f}m ({obs.score:.2f})"
        cv2.putText(
            vis,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    if title:
        cv2.putText(
            vis,
            f"{title} | {output.total_time_ms:.1f} ms ({1000.0 / max(1e-3, output.total_time_ms):.1f} FPS)",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def save_error_analysis_plots(
    matches_by_pipeline: Dict[str, List[MatchedObstaclePair]],
    out_dir: str | Path,
) -> Path:
    """Generates scatter and distance-bucket error analysis charts for the IEEE report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / "distance_error_analysis.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Subplot 1: Predicted vs Ground-Truth Distance Scatter
    ax1 = axes[0]
    ax1.plot([0, 60], [0, 60], "k--", label="Ideal (y = x)", alpha=0.6)
    for pipe_name, pairs in matches_by_pipeline.items():
        if not pairs:
            continue
        gts = [p.gt_distance_m for p in pairs]
        preds = [p.pred_distance_m for p in pairs]
        ax1.scatter(gts, preds, alpha=0.65, s=24, label=pipe_name)

    ax1.set_title("Predicted vs. Ground-Truth Object Distance")
    ax1.set_xlabel("Ground-Truth LiDAR Distance (m)")
    ax1.set_ylabel("Predicted Distance (m)")
    ax1.set_xlim(0, 65)
    ax1.set_ylim(0, 65)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8)

    # Subplot 2: Absolute Relative Error by Distance Range Bucket
    ax2 = axes[1]
    buckets = [(0, 15, "Near (0-15m)"), (15, 30, "Mid (15-30m)"), (30, 80, "Far (30-80m)")]
    x_inds = np.arange(len(buckets))
    width = 0.25

    for idx, (pipe_name, pairs) in enumerate(matches_by_pipeline.items()):
        bucket_errors = []
        for lo, hi, _ in buckets:
            sub = [abs(p.pred_distance_m - p.gt_distance_m) / p.gt_distance_m for p in pairs if lo <= p.gt_distance_m < hi]
            bucket_errors.append(float(np.mean(sub)) if sub else 0.0)
        ax2.bar(x_inds + (idx - 1) * width, bucket_errors, width=width, label=pipe_name)

    ax2.set_xticks(x_inds)
    ax2.set_xticklabels([b[2] for b in buckets])
    ax2.set_title("Object Distance AbsRel by Distance Range")
    ax2.set_ylabel("Mean AbsRel Error")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path
