"""Compare MultiBlobMatcher vs Viterbi tracker against GT annotations.

Metrics:
1. Detection recall: % of GT match_ball frames with a detection
2. Pixel accuracy: distance from detected position to GT position (<5px, <10px, <20px)
3. Jitter: mean frame-to-frame pixel displacement (measures visual stability / 抖动)

Usage:
    python -m tools.compare_multi_vs_viterbi
"""

import json
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_DIR = "uploads/cam66_20260307_173403_2min"
MAX_FRAMES = 1800


def load_gt(gt_dir: str, max_frames: int) -> dict:
    """Load GT match_ball pixel positions from LabelMe JSON files."""
    match_ball = {}
    for fi in range(max_frames):
        fp = os.path.join(gt_dir, f"{fi:05d}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            data = json.load(f)
        for s in data.get("shapes", []):
            label = s.get("label", "")
            desc = s.get("description", "").lower().replace("\uff0c", ",")
            if label != "ball":
                continue
            if "match_ball" not in desc:
                continue
            if s["shape_type"] == "point":
                px, py = s["points"][0]
            elif s["shape_type"] == "rectangle":
                pts = s["points"]
                px = (pts[0][0] + pts[2][0]) / 2
                py = (pts[0][1] + pts[2][1]) / 2
            else:
                continue
            match_ball[fi] = (px, py)
    return match_ball


def run_detection(max_frames, top_k=2):
    """Run TrackNet detection on both cameras. Returns multi-blob dicts."""
    from tools.render_tracking_video import build_detector, load_config, run_detection_multi

    cfg = load_config()
    detector, postproc = build_detector(cfg)

    logger.info("=== Detection: cam66 ===")
    multi66, det66, n66 = run_detection_multi(
        "uploads/cam66_20260307_173403_2min.mp4",
        detector, postproc, max_frames, top_k=top_k,
    )

    # Reset detector state for second video
    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("=== Detection: cam68 ===")
    multi68, det68, n68 = run_detection_multi(
        "uploads/cam68_20260307_173403_2min.mp4",
        detector, postproc, max_frames, top_k=top_k,
    )

    return cfg, multi66, multi68, det66, det68


def run_multi_blob(multi66, multi68, cfg):
    """Run MultiBlobMatcher, return chosen cam66 pixels."""
    from tools.render_tracking_video import triangulate_multi_blob

    points_3d, chosen_pixels, stats = triangulate_multi_blob(multi66, multi68, cfg)
    # Extract cam66 pixel positions
    cam66_px = {}
    for fi, cp in chosen_pixels.items():
        cam66_px[fi] = cp["cam66"]  # (px, py)
    return cam66_px, points_3d, stats


def run_viterbi(multi66, multi68, cfg):
    """Run Viterbi tracker, return chosen cam66 pixels."""
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.viterbi_tracker import ViterbiTracker

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")
    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    tracker = ViterbiTracker(
        cam1_pos=cam66_pos, cam2_pos=cam68_pos,
        max_ray_distance=2.5, valid_z_range=(0.0, 8.0),
        fps=25.0, gap_threshold=5,
    )
    points_3d, chosen_pixels, stats = tracker.track(
        multi66, multi68, homo66, homo68,
    )
    cam66_px = {}
    for fi, cp in chosen_pixels.items():
        cam66_px[fi] = cp["cam66"]
    return cam66_px, points_3d, stats


def compute_metrics(name, cam66_px, gt_ball, det66):
    """Compute all metrics for a given method.

    Args:
        name: method name for logging
        cam66_px: {frame: (px, py)} matched pixel positions from the method
        gt_ball: {frame: (px, py)} GT match_ball positions
        det66: {frame: (px, py, conf)} raw single-best detections (for raw recall)
    """
    gt_frames = set(gt_ball.keys())
    method_frames = set(cam66_px.keys())
    raw_frames = set(det66.keys())

    # --- Recall ---
    raw_hit = gt_frames & raw_frames
    method_hit = gt_frames & method_frames
    raw_recall = len(raw_hit) / len(gt_frames) if gt_frames else 0
    method_recall = len(method_hit) / len(gt_frames) if gt_frames else 0

    # --- Pixel accuracy (only on frames where both GT and method have detections) ---
    errors = []
    for fi in sorted(method_hit):
        gx, gy = gt_ball[fi]
        px, py = cam66_px[fi]
        errors.append(math.hypot(px - gx, py - gy))

    errors = np.array(errors) if errors else np.array([])

    # --- Jitter: frame-to-frame displacement over ALL method frames ---
    sorted_frames = sorted(cam66_px.keys())
    displacements_all = []
    for i in range(1, len(sorted_frames)):
        f_prev, f_curr = sorted_frames[i - 1], sorted_frames[i]
        if f_curr - f_prev > 3:
            continue  # skip gaps (different rally segments)
        px0, py0 = cam66_px[f_prev]
        px1, py1 = cam66_px[f_curr]
        disp = math.hypot(px1 - px0, py1 - py0)
        displacements_all.append(disp)

    disps_all = np.array(displacements_all) if displacements_all else np.array([])

    # --- Jitter on GT frames only (frame-to-frame displacement within GT segments) ---
    gt_sorted = sorted(method_hit)
    displacements_gt = []
    for i in range(1, len(gt_sorted)):
        f_prev, f_curr = gt_sorted[i - 1], gt_sorted[i]
        if f_curr - f_prev > 3:
            continue
        px0, py0 = cam66_px[f_prev]
        px1, py1 = cam66_px[f_curr]
        disp = math.hypot(px1 - px0, py1 - py0)
        displacements_gt.append(disp)

    disps_gt = np.array(displacements_gt) if displacements_gt else np.array([])

    # --- GT displacement (ground truth jitter baseline) ---
    gt_displacements = []
    gt_sorted_all = sorted(gt_ball.keys())
    for i in range(1, len(gt_sorted_all)):
        f_prev, f_curr = gt_sorted_all[i - 1], gt_sorted_all[i]
        if f_curr - f_prev > 3:
            continue
        gx0, gy0 = gt_ball[f_prev]
        gx1, gy1 = gt_ball[f_curr]
        gt_displacements.append(math.hypot(gx1 - gx0, gy1 - gy0))

    gt_disps = np.array(gt_displacements) if gt_displacements else np.array([])

    return {
        "name": name,
        "gt_frames": len(gt_frames),
        "method_frames": len(method_frames),
        "raw_recall": raw_recall,
        "method_recall": method_recall,
        "errors": errors,
        "disps_all": disps_all,
        "disps_gt": disps_gt,
        "gt_disps": gt_disps,
    }


def print_report(results_list):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON: MultiBlobMatcher vs Viterbi Tracker")
    print("=" * 80)

    # GT baseline
    gt_disps = results_list[0]["gt_disps"]
    print(f"\nGT match_ball frames: {results_list[0]['gt_frames']}")
    if len(gt_disps) > 0:
        print(f"GT baseline jitter (frame-to-frame displacement):")
        print(f"  mean={gt_disps.mean():.1f}px  median={np.median(gt_disps):.1f}px  "
              f"p90={np.percentile(gt_disps, 90):.1f}px")

    header = f"{'Metric':<40} {'MultiBlobMatcher':>18} {'Viterbi':>18}"
    print(f"\n{header}")
    print("-" * 80)

    for key, label, fmt in [
        ("method_frames", "Matched frames (cam66)", "d"),
        ("raw_recall", "Raw detection recall", ".1%"),
        ("method_recall", "Pipeline recall (vs GT)", ".1%"),
    ]:
        vals = []
        for r in results_list:
            v = r[key]
            vals.append(f"{v:{fmt}}")
        print(f"  {label:<38} {vals[0]:>18} {vals[1]:>18}")

    # Pixel accuracy
    print(f"\n{'Pixel Accuracy (on matched GT frames)':<40}")
    print("-" * 80)
    for threshold in [5, 10, 20]:
        vals = []
        for r in results_list:
            e = r["errors"]
            if len(e) > 0:
                pct = 100 * np.mean(e < threshold)
                vals.append(f"{pct:.1f}%")
            else:
                vals.append("N/A")
        print(f"  <{threshold}px{'':<35} {vals[0]:>18} {vals[1]:>18}")

    for stat_name, stat_fn in [("mean", np.mean), ("median", np.median),
                                ("p90", lambda a: np.percentile(a, 90)),
                                ("max", np.max)]:
        vals = []
        for r in results_list:
            e = r["errors"]
            if len(e) > 0:
                vals.append(f"{stat_fn(e):.1f}px")
            else:
                vals.append("N/A")
        print(f"  {stat_name:<38} {vals[0]:>18} {vals[1]:>18}")

    # Jitter
    print(f"\n{'Jitter (frame-to-frame displacement)':<40}")
    print("-" * 80)

    for stat_name, stat_fn in [("mean", np.mean), ("median", np.median),
                                ("p90", lambda a: np.percentile(a, 90)),
                                ("p99", lambda a: np.percentile(a, 99))]:
        # All frames jitter
        vals_all = []
        vals_gt = []
        for r in results_list:
            da = r["disps_all"]
            dg = r["disps_gt"]
            if len(da) > 0:
                vals_all.append(f"{stat_fn(da):.1f}px")
            else:
                vals_all.append("N/A")
            if len(dg) > 0:
                vals_gt.append(f"{stat_fn(dg):.1f}px")
            else:
                vals_gt.append("N/A")
        print(f"  {stat_name} (all frames){'':<22} {vals_all[0]:>18} {vals_all[1]:>18}")
        print(f"  {stat_name} (GT frames only){'':<17} {vals_gt[0]:>18} {vals_gt[1]:>18}")

    # Large jumps (>50px) count
    print(f"\n{'Large jumps (>50px between consecutive frames)':<40}")
    print("-" * 80)
    for r in results_list:
        da = r["disps_all"]
        if len(da) > 0:
            big = np.sum(da > 50)
            print(f"  {r['name']:<38} {big:>5} / {len(da)} ({100*big/len(da):.1f}%)")

    print("\n" + "=" * 80)


def main():
    logger.info("Loading GT annotations from %s ...", GT_DIR)
    gt_ball = load_gt(GT_DIR, MAX_FRAMES)
    logger.info("GT match_ball: %d frames", len(gt_ball))

    logger.info("\n" + "=" * 60)
    logger.info("Running detection on both cameras (top-2 blobs)...")
    logger.info("=" * 60)
    cfg, multi66, multi68, det66, det68 = run_detection(MAX_FRAMES, top_k=2)

    logger.info("\n" + "=" * 60)
    logger.info("Running MultiBlobMatcher...")
    logger.info("=" * 60)
    multi_px, multi_3d, multi_stats = run_multi_blob(multi66, multi68, cfg)

    logger.info("\n" + "=" * 60)
    logger.info("Running Viterbi tracker...")
    logger.info("=" * 60)
    viterbi_px, viterbi_3d, viterbi_stats = run_viterbi(multi66, multi68, cfg)

    logger.info("\n" + "=" * 60)
    logger.info("Computing metrics...")
    logger.info("=" * 60)

    results_multi = compute_metrics("MultiBlobMatcher", multi_px, gt_ball, det66)
    results_viterbi = compute_metrics("Viterbi", viterbi_px, gt_ball, det66)

    print_report([results_multi, results_viterbi])


if __name__ == "__main__":
    main()
