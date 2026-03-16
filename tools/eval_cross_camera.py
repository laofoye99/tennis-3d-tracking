"""Cross-camera validation evaluation.

Evaluates whether two cameras can cross-validate each other's detections
using 3D triangulation + ray_distance as a consistency metric.

When one camera drifts (detects dead ball / false positive), the other
camera's detection may disagree — producing a large ray_distance.
This script measures:

1. ray_distance distribution across all common frames
2. How often cameras disagree (ray_distance > threshold)
3. Whether disagreement correlates with detection errors (vs GT)
4. Whether trusting the "better" camera improves recall

Usage:
    python -m tools.eval_cross_camera
"""

import json
import logging
import math
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Camera positions from config.yaml
CAM66_POS = [3.7657, -15.7273, 7.0]
CAM68_POS = [5.8058, 45.0351, 7.0]

# Thresholds
RAY_DIST_AGREE = 1.0       # meters — cameras agree
RAY_DIST_DISAGREE = 2.0    # meters — cameras clearly disagree
GT_MATCH_RADIUS = 15.0     # pixels — detection matches GT


def load_gt_annotations(cam_dir: Path) -> dict[int, tuple[float, float]]:
    """Load GT ball positions per frame (first GT annotation per frame)."""
    gt = {}
    for jf in sorted(cam_dir.glob("*.json")):
        try:
            fi = int(jf.stem)
        except ValueError:
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            if shape.get("score") is not None:
                continue
            pts = shape.get("points", [])
            if not pts:
                continue
            shape_type = shape.get("shape_type", "point")
            if shape_type == "rectangle" and len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[2] if len(pts) >= 3 else pts[1]
                gt[fi] = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                gt[fi] = (pts[0][0], pts[0][1])
            break  # first GT per frame
    return gt


def _triangulate_with_distance(
    world_2d_cam1: tuple[float, float],
    world_2d_cam2: tuple[float, float],
    cam_pos_1: list[float],
    cam_pos_2: list[float],
) -> tuple[float, float, float, float]:
    """Triangulate and return (x, y, z, ray_distance)."""
    cam1 = np.asarray(cam_pos_1, dtype=np.float64)
    cam2 = np.asarray(cam_pos_2, dtype=np.float64)
    ground1 = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    ground2 = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])

    d1 = ground1 - cam1
    d2 = ground2 - cam2

    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        mid = (ground1 + ground2) / 2.0
        return float(mid[0]), float(mid[1]), 0.0, 999.0

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    p1_fixed = cam1 + s * d1
    t = float(np.dot(p1_fixed - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)

    p2_fixed = cam2 + t * d2
    s = float(np.dot(p2_fixed - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0
    ray_dist = float(np.linalg.norm(p1 - p2))

    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2]), ray_dist


def pixel_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def run_evaluation():
    """Main evaluation logic."""
    export_dir = Path("exports/eval_motion_yolo")

    # Load detections
    with open(export_dir / "cam66_detections.json") as f:
        det66 = json.load(f)
    with open(export_dir / "cam68_detections.json") as f:
        det68 = json.load(f)

    # Load GT
    gt66 = load_gt_annotations(Path("uploads/cam66_20260307_173403_2min"))
    gt68 = load_gt_annotations(Path("uploads/cam68_20260307_173403_2min"))

    logger.info("=== Cross-Camera Validation Evaluation ===\n")
    logger.info("Detections: cam66=%d frames, cam68=%d frames", len(det66), len(det68))
    logger.info("GT: cam66=%d frames, cam68=%d frames", len(gt66), len(gt68))

    # Find common frames (both cameras have detections)
    common_frames = sorted(set(det66.keys()) & set(det68.keys()), key=int)
    logger.info("Common detection frames: %d\n", len(common_frames))

    # === Test 1: ray_distance distribution ===
    logger.info("=" * 60)
    logger.info("Test 1: Ray Distance Distribution")
    logger.info("=" * 60)

    ray_distances = []
    frame_data = []  # store per-frame analysis

    for fi_str in common_frames:
        d66 = det66[fi_str]
        d68 = det68[fi_str]

        x, y, z, ray_dist = _triangulate_with_distance(
            (d66["world_x"], d66["world_y"]),
            (d68["world_x"], d68["world_y"]),
            CAM66_POS, CAM68_POS,
        )
        ray_distances.append(ray_dist)

        fi = int(fi_str)
        frame_data.append({
            "frame": fi,
            "ray_distance": ray_dist,
            "pos_3d": (x, y, z),
            "cam66_pixel": (d66["pixel_x"], d66["pixel_y"]),
            "cam68_pixel": (d68["pixel_x"], d68["pixel_y"]),
            "cam66_world": (d66["world_x"], d66["world_y"]),
            "cam68_world": (d68["world_x"], d68["world_y"]),
            "cam66_yolo": d66.get("yolo_conf", -1),
            "cam68_yolo": d68.get("yolo_conf", -1),
            "cam66_conf": d66.get("confidence", 0),
            "cam68_conf": d68.get("confidence", 0),
        })

    rd = np.array(ray_distances)
    logger.info("  Total frames: %d", len(rd))
    logger.info("  Mean ray_distance: %.2f m", np.mean(rd))
    logger.info("  Median ray_distance: %.2f m", np.median(rd))
    logger.info("  Std: %.2f m", np.std(rd))
    logger.info("  Min: %.2f m, Max: %.2f m", np.min(rd), np.max(rd))

    # Distribution buckets
    buckets = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    for threshold in buckets:
        count = np.sum(rd < threshold)
        logger.info("  < %.1f m: %d (%.1f%%)", threshold, count, 100 * count / len(rd))

    agree_count = np.sum(rd < RAY_DIST_AGREE)
    disagree_count = np.sum(rd >= RAY_DIST_DISAGREE)
    middle_count = len(rd) - agree_count - disagree_count
    logger.info("\n  Agreement (< %.1f m): %d (%.1f%%)", RAY_DIST_AGREE, agree_count, 100 * agree_count / len(rd))
    logger.info("  Uncertain (%.1f-%.1f m): %d (%.1f%%)", RAY_DIST_AGREE, RAY_DIST_DISAGREE, middle_count, 100 * middle_count / len(rd))
    logger.info("  Disagreement (>= %.1f m): %d (%.1f%%)\n", RAY_DIST_DISAGREE, disagree_count, 100 * disagree_count / len(rd))

    # === Test 2: Disagreement vs GT errors ===
    logger.info("=" * 60)
    logger.info("Test 2: Disagreement Correlation with GT Errors")
    logger.info("=" * 60)

    # For frames with GT in both cameras, check if disagreement = error
    gt_both = {fi for fi in common_frames if int(fi) in gt66 and int(fi) in gt68}
    gt_66_only = {fi for fi in common_frames if int(fi) in gt66 and int(fi) not in gt68}
    gt_68_only = {fi for fi in common_frames if int(fi) not in gt66 and int(fi) in gt68}
    gt_any = gt_both | gt_66_only | gt_68_only

    logger.info("  Common frames with GT in both cameras: %d", len(gt_both))
    logger.info("  Common frames with GT in cam66 only: %d", len(gt_66_only))
    logger.info("  Common frames with GT in cam68 only: %d", len(gt_68_only))

    # Analyze: when cameras agree, is detection correct?
    # When cameras disagree, which one is wrong?
    agree_correct_66 = 0
    agree_wrong_66 = 0
    disagree_66_correct = 0
    disagree_66_wrong = 0
    disagree_68_correct = 0
    disagree_68_wrong = 0

    agree_correct_68 = 0
    agree_wrong_68 = 0

    cross_val_results = []

    for fd in frame_data:
        fi = fd["frame"]
        fi_str = str(fi)
        rd_val = fd["ray_distance"]

        has_gt66 = fi in gt66
        has_gt68 = fi in gt68

        err66 = pixel_dist(fd["cam66_pixel"], gt66[fi]) if has_gt66 else None
        err68 = pixel_dist(fd["cam68_pixel"], gt68[fi]) if has_gt68 else None

        correct66 = err66 is not None and err66 < GT_MATCH_RADIUS
        correct68 = err68 is not None and err68 < GT_MATCH_RADIUS

        is_agree = rd_val < RAY_DIST_AGREE
        is_disagree = rd_val >= RAY_DIST_DISAGREE

        if has_gt66:
            if is_agree:
                if correct66:
                    agree_correct_66 += 1
                else:
                    agree_wrong_66 += 1
            elif is_disagree:
                if correct66:
                    disagree_66_correct += 1
                else:
                    disagree_66_wrong += 1

        if has_gt68:
            if is_agree:
                if correct68:
                    agree_correct_68 += 1
                else:
                    agree_wrong_68 += 1
            elif is_disagree:
                if correct68:
                    disagree_68_correct += 1
                else:
                    disagree_68_wrong += 1

        cross_val_results.append({
            "frame": fi,
            "ray_distance": rd_val,
            "err66": err66,
            "err68": err68,
            "correct66": correct66 if has_gt66 else None,
            "correct68": correct68 if has_gt68 else None,
            "agree": is_agree,
            "disagree": is_disagree,
        })

    logger.info("\n  cam66 — Agreement frames:")
    logger.info("    Correct: %d, Wrong: %d", agree_correct_66, agree_wrong_66)
    if agree_correct_66 + agree_wrong_66 > 0:
        logger.info("    Accuracy: %.1f%%", 100 * agree_correct_66 / (agree_correct_66 + agree_wrong_66))

    logger.info("  cam66 — Disagreement frames:")
    logger.info("    Correct: %d, Wrong: %d", disagree_66_correct, disagree_66_wrong)
    if disagree_66_correct + disagree_66_wrong > 0:
        logger.info("    Accuracy: %.1f%%", 100 * disagree_66_correct / (disagree_66_correct + disagree_66_wrong))

    logger.info("\n  cam68 — Agreement frames:")
    logger.info("    Correct: %d, Wrong: %d", agree_correct_68, agree_wrong_68)
    if agree_correct_68 + agree_wrong_68 > 0:
        logger.info("    Accuracy: %.1f%%", 100 * agree_correct_68 / (agree_correct_68 + agree_wrong_68))

    logger.info("  cam68 — Disagreement frames:")
    logger.info("    Correct: %d, Wrong: %d", disagree_68_correct, disagree_68_wrong)
    if disagree_68_correct + disagree_68_wrong > 0:
        logger.info("    Accuracy: %.1f%%", 100 * disagree_68_correct / (disagree_68_correct + disagree_68_wrong))

    # === Test 3: Cross-validation rescue potential ===
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Cross-Validation Rescue Potential")
    logger.info("=" * 60)
    logger.info("When cameras disagree and one is wrong, could the other rescue it?\n")

    # For frames with GT in at least one camera where cameras disagree:
    # Case A: cam66 wrong, cam68 correct → cam68 could rescue cam66
    # Case B: cam68 wrong, cam66 correct → cam66 could rescue cam68
    # Case C: both wrong → cross-validation can't help
    # Case D: both correct but disagree → false alarm (ball high in air?)

    rescue_66_by_68 = 0  # cam66 wrong, cam68 correct
    rescue_68_by_66 = 0  # cam68 wrong, cam66 correct
    both_wrong_disagree = 0
    both_correct_disagree = 0
    one_correct_no_gt_other = 0

    # Also track: when cameras agree but one is wrong
    agree_one_wrong = 0

    for r in cross_val_results:
        if not r["disagree"]:
            # Agreement — check if both correct
            if r["correct66"] is not None and r["correct68"] is not None:
                if r["correct66"] and not r["correct68"]:
                    agree_one_wrong += 1
                elif not r["correct66"] and r["correct68"]:
                    agree_one_wrong += 1
            continue

        c66 = r["correct66"]
        c68 = r["correct68"]

        if c66 is not None and c68 is not None:
            if not c66 and c68:
                rescue_66_by_68 += 1
            elif c66 and not c68:
                rescue_68_by_66 += 1
            elif not c66 and not c68:
                both_wrong_disagree += 1
            else:
                both_correct_disagree += 1
        elif c66 is not None and c68 is None:
            if not c66:
                one_correct_no_gt_other += 1
        elif c68 is not None and c66 is None:
            if not c68:
                one_correct_no_gt_other += 1

    total_rescue = rescue_66_by_68 + rescue_68_by_66
    logger.info("  Disagreement frames with GT in both cameras:")
    logger.info("    cam66 wrong, cam68 correct (rescue cam66): %d", rescue_66_by_68)
    logger.info("    cam68 wrong, cam66 correct (rescue cam68): %d", rescue_68_by_66)
    logger.info("    Both wrong (can't rescue): %d", both_wrong_disagree)
    logger.info("    Both correct (false alarm): %d", both_correct_disagree)
    logger.info("    Total rescuable: %d", total_rescue)
    logger.info("    Agreement frames where one is wrong: %d", agree_one_wrong)

    # === Test 4: Trust scoring strategies ===
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Trust Scoring Strategies")
    logger.info("=" * 60)
    logger.info("When cameras disagree, which signal best identifies the correct one?\n")

    # For disagreement frames with GT in both cameras, test different trust signals
    strategies = {
        "yolo_conf": {"correct": 0, "wrong": 0, "tie": 0},
        "blob_sum": {"correct": 0, "wrong": 0, "tie": 0},
        "motion_continuity": {"correct": 0, "wrong": 0, "tie": 0},
    }

    # Sort frame_data by frame index for motion continuity
    frame_data_sorted = sorted(frame_data, key=lambda x: x["frame"])
    frame_data_map = {fd["frame"]: fd for fd in frame_data_sorted}

    for r in cross_val_results:
        if not r["disagree"]:
            continue
        c66 = r["correct66"]
        c68 = r["correct68"]
        if c66 is None or c68 is None:
            continue
        if c66 == c68:
            continue  # both correct or both wrong — no decision to make

        fi = r["frame"]
        fd = frame_data_map[fi]

        # Strategy 1: Trust higher YOLO confidence
        y66 = fd["cam66_yolo"]
        y68 = fd["cam68_yolo"]
        if y66 > y68:
            if c66:
                strategies["yolo_conf"]["correct"] += 1
            else:
                strategies["yolo_conf"]["wrong"] += 1
        elif y68 > y66:
            if c68:
                strategies["yolo_conf"]["correct"] += 1
            else:
                strategies["yolo_conf"]["wrong"] += 1
        else:
            strategies["yolo_conf"]["tie"] += 1

        # Strategy 2: Trust higher blob_sum (TrackNet confidence)
        bs66 = fd["cam66_conf"]
        bs68 = fd["cam68_conf"]
        if bs66 > bs68:
            if c66:
                strategies["blob_sum"]["correct"] += 1
            else:
                strategies["blob_sum"]["wrong"] += 1
        elif bs68 > bs66:
            if c68:
                strategies["blob_sum"]["correct"] += 1
            else:
                strategies["blob_sum"]["wrong"] += 1
        else:
            strategies["blob_sum"]["tie"] += 1

        # Strategy 3: Trust camera with smaller displacement from previous frame
        prev_fi = fi - 1
        if prev_fi in frame_data_map:
            prev = frame_data_map[prev_fi]
            disp66 = pixel_dist(fd["cam66_pixel"], prev["cam66_pixel"])
            disp68 = pixel_dist(fd["cam68_pixel"], prev["cam68_pixel"])
            # Smaller displacement = more continuous = more trustworthy
            if disp66 < disp68:
                if c66:
                    strategies["motion_continuity"]["correct"] += 1
                else:
                    strategies["motion_continuity"]["wrong"] += 1
            elif disp68 < disp66:
                if c68:
                    strategies["motion_continuity"]["correct"] += 1
                else:
                    strategies["motion_continuity"]["wrong"] += 1
            else:
                strategies["motion_continuity"]["tie"] += 1
        else:
            strategies["motion_continuity"]["tie"] += 1

    for name, s in strategies.items():
        total = s["correct"] + s["wrong"]
        if total > 0:
            acc = 100 * s["correct"] / total
            logger.info("  %s: correct=%d, wrong=%d, tie=%d → accuracy=%.1f%%",
                        name, s["correct"], s["wrong"], s["tie"], acc)
        else:
            logger.info("  %s: no testable frames", name)

    # === Test 5: Overall improvement potential ===
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Overall Recall Impact")
    logger.info("=" * 60)

    # Baseline: each camera's recall on its own (common frames only)
    baseline_66_tp = 0
    baseline_68_tp = 0
    cross_val_tp = 0  # using best strategy from Test 4
    total_gt_frames = 0

    for r in cross_val_results:
        fi = r["frame"]
        has_gt66 = r["correct66"] is not None
        has_gt68 = r["correct68"] is not None

        if not has_gt66 and not has_gt68:
            continue

        # We only count frames where GT exists in at least one camera
        # For simplicity, evaluate per-camera

    # Per-camera recall on common frames
    common_gt66 = [r for r in cross_val_results if r["correct66"] is not None]
    common_gt68 = [r for r in cross_val_results if r["correct68"] is not None]

    baseline_recall_66 = sum(1 for r in common_gt66 if r["correct66"]) / len(common_gt66) if common_gt66 else 0
    baseline_recall_68 = sum(1 for r in common_gt68 if r["correct68"]) / len(common_gt68) if common_gt68 else 0

    # Cross-validated recall: when cameras disagree, trust the more continuous one
    cross_val_recall_66 = 0
    cross_val_recall_68 = 0
    rescued_frames_66 = []
    rescued_frames_68 = []

    for r in cross_val_results:
        fi = r["frame"]
        fd = frame_data_map.get(fi)
        if fd is None:
            continue

        # For cam66
        if r["correct66"] is not None:
            if r["agree"]:
                # Cameras agree — keep original detection
                if r["correct66"]:
                    cross_val_recall_66 += 1
            elif r["disagree"]:
                # Cameras disagree — could we rescue?
                # If cam66 is wrong AND cam68 exists, flag it
                if r["correct66"]:
                    cross_val_recall_66 += 1
                elif r["correct68"]:
                    # cam66 wrong, cam68 correct → rescue
                    cross_val_recall_66 += 1
                    rescued_frames_66.append(fi)
                # else both wrong, no rescue
            else:
                # Uncertain zone — keep original
                if r["correct66"]:
                    cross_val_recall_66 += 1

        # For cam68
        if r["correct68"] is not None:
            if r["agree"]:
                if r["correct68"]:
                    cross_val_recall_68 += 1
            elif r["disagree"]:
                if r["correct68"]:
                    cross_val_recall_68 += 1
                elif r["correct66"]:
                    cross_val_recall_68 += 1
                    rescued_frames_68.append(fi)
            else:
                if r["correct68"]:
                    cross_val_recall_68 += 1

    cv_recall_66 = cross_val_recall_66 / len(common_gt66) if common_gt66 else 0
    cv_recall_68 = cross_val_recall_68 / len(common_gt68) if common_gt68 else 0

    logger.info("  cam66: baseline recall=%.1f%% → cross-validated=%.1f%% (rescued %d frames)",
                100 * baseline_recall_66, 100 * cv_recall_66, len(rescued_frames_66))
    logger.info("  cam68: baseline recall=%.1f%% → cross-validated=%.1f%% (rescued %d frames)",
                100 * baseline_recall_68, 100 * cv_recall_68, len(rescued_frames_68))

    if rescued_frames_66:
        logger.info("  cam66 rescued frames: %s", rescued_frames_66[:20])
    if rescued_frames_68:
        logger.info("  cam68 rescued frames: %s", rescued_frames_68[:20])

    # === Save results ===
    out_dir = Path("exports/eval_cross_camera")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "common_frames": len(common_frames),
        "ray_distance": {
            "mean": float(np.mean(rd)),
            "median": float(np.median(rd)),
            "std": float(np.std(rd)),
            "min": float(np.min(rd)),
            "max": float(np.max(rd)),
            "agree_count": int(agree_count),
            "disagree_count": int(disagree_count),
        },
        "gt_correlation": {
            "agree_correct_66": agree_correct_66,
            "agree_wrong_66": agree_wrong_66,
            "disagree_correct_66": disagree_66_correct,
            "disagree_wrong_66": disagree_66_wrong,
            "agree_correct_68": agree_correct_68,
            "agree_wrong_68": agree_wrong_68,
            "disagree_correct_68": disagree_68_correct,
            "disagree_wrong_68": disagree_68_wrong,
        },
        "rescue_potential": {
            "rescue_66_by_68": rescue_66_by_68,
            "rescue_68_by_66": rescue_68_by_66,
            "both_wrong_disagree": both_wrong_disagree,
            "both_correct_disagree": both_correct_disagree,
        },
        "trust_strategies": strategies,
        "recall_impact": {
            "baseline_66": baseline_recall_66,
            "cross_val_66": cv_recall_66,
            "rescued_66": len(rescued_frames_66),
            "baseline_68": baseline_recall_68,
            "cross_val_68": cv_recall_68,
            "rescued_68": len(rescued_frames_68),
        },
    }

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-frame data for visualization
    per_frame = []
    for fd, r in zip(frame_data, cross_val_results):
        per_frame.append({
            "frame": fd["frame"],
            "ray_distance": fd["ray_distance"],
            "pos_3d": fd["pos_3d"],
            "cam66_pixel": fd["cam66_pixel"],
            "cam68_pixel": fd["cam68_pixel"],
            "cam66_world": fd["cam66_world"],
            "cam68_world": fd["cam68_world"],
            "err66": r["err66"],
            "err68": r["err68"],
            "correct66": r["correct66"],
            "correct68": r["correct68"],
        })

    with open(out_dir / "per_frame_data.json", "w") as f:
        json.dump(per_frame, f)

    logger.info("\nResults saved to %s/", out_dir)

    # === Generate visualization ===
    _plot_results(frame_data, cross_val_results, rd, out_dir)


def _plot_results(
    frame_data: list[dict],
    cross_val_results: list[dict],
    ray_distances: np.ndarray,
    out_dir: Path,
):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Ray distance histogram
    ax = axes[0, 0]
    ax.hist(ray_distances, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(RAY_DIST_AGREE, color="green", linestyle="--", label=f"Agree < {RAY_DIST_AGREE}m")
    ax.axvline(RAY_DIST_DISAGREE, color="red", linestyle="--", label=f"Disagree >= {RAY_DIST_DISAGREE}m")
    ax.set_xlabel("Ray Distance (m)")
    ax.set_ylabel("Frame Count")
    ax.set_title("Ray Distance Distribution")
    ax.legend()

    # Plot 2: Ray distance timeline
    ax = axes[0, 1]
    frames = [fd["frame"] for fd in frame_data]
    ax.plot(frames, ray_distances, "b.", alpha=0.3, markersize=1)
    ax.axhline(RAY_DIST_AGREE, color="green", linestyle="--", alpha=0.5)
    ax.axhline(RAY_DIST_DISAGREE, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Ray Distance (m)")
    ax.set_title("Ray Distance Over Time")
    ax.set_ylim(0, min(20, np.max(ray_distances) * 1.1))

    # Plot 3: Disagreement vs GT error correlation
    ax = axes[1, 0]
    gt_err_66 = []
    gt_rd_66 = []
    for r in cross_val_results:
        if r["err66"] is not None:
            gt_err_66.append(r["err66"])
            gt_rd_66.append(r["ray_distance"])

    if gt_err_66:
        colors = ["green" if e < GT_MATCH_RADIUS else "red" for e in gt_err_66]
        ax.scatter(gt_rd_66, gt_err_66, c=colors, alpha=0.3, s=5)
        ax.axvline(RAY_DIST_DISAGREE, color="red", linestyle="--", alpha=0.5)
        ax.axhline(GT_MATCH_RADIUS, color="blue", linestyle="--", alpha=0.5, label=f"GT match {GT_MATCH_RADIUS}px")
        ax.set_xlabel("Ray Distance (m)")
        ax.set_ylabel("cam66 Pixel Error vs GT")
        ax.set_title("cam66: Ray Distance vs Detection Error")
        ax.set_ylim(0, min(500, max(gt_err_66) * 1.1))
        ax.legend()

    # Plot 4: Same for cam68
    ax = axes[1, 1]
    gt_err_68 = []
    gt_rd_68 = []
    for r in cross_val_results:
        if r["err68"] is not None:
            gt_err_68.append(r["err68"])
            gt_rd_68.append(r["ray_distance"])

    if gt_err_68:
        colors = ["green" if e < GT_MATCH_RADIUS else "red" for e in gt_err_68]
        ax.scatter(gt_rd_68, gt_err_68, c=colors, alpha=0.3, s=5)
        ax.axvline(RAY_DIST_DISAGREE, color="red", linestyle="--", alpha=0.5)
        ax.axhline(GT_MATCH_RADIUS, color="blue", linestyle="--", alpha=0.5, label=f"GT match {GT_MATCH_RADIUS}px")
        ax.set_xlabel("Ray Distance (m)")
        ax.set_ylabel("cam68 Pixel Error vs GT")
        ax.set_title("cam68: Ray Distance vs Detection Error")
        ax.set_ylim(0, min(500, max(gt_err_68) * 1.1))
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "cross_camera_analysis.png", dpi=150)
    plt.close()
    logger.info("Visualization saved to %s/cross_camera_analysis.png", out_dir)


if __name__ == "__main__":
    run_evaluation()
