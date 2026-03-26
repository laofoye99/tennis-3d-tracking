"""Evaluate tracking pipeline output against cam66 GT annotations.

Compares:
1. Detection recall: How many GT match_ball frames did we detect?
2. Bounce accuracy: Do detected bounces match GT bounce frames?
3. Pixel accuracy: How close are detected positions to GT positions?

Usage:
    python -m tools.eval_vs_gt
"""

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_DIR = "uploads/cam66_20260307_173403_2min"
MAX_FRAMES = 1800


def load_gt(gt_dir: str, max_frames: int) -> dict:
    """Load GT annotations, return structured data.

    Returns:
        {
            'match_ball': {frame: (px, py)},       # ball position (point shape)
            'bounces': [(frame, px, py), ...],
            'serves': [(frame, px, py), ...],
            'shots': [(frame, px, py), ...],
            'all_ball': {frame: (px, py)},          # any annotated ball position
        }
    """
    match_ball = {}
    bounces = []
    serves = []
    shots = []
    all_ball = {}

    for fi in range(max_frames):
        fp = os.path.join(gt_dir, f"{fi:05d}.json")
        if not os.path.exists(fp):
            continue

        with open(fp) as f:
            data = json.load(f)

        for s in data.get("shapes", []):
            desc = s.get("description", "")
            label = s.get("label", "")
            if label != "ball":
                continue

            # Get pixel position
            if s["shape_type"] == "point":
                px, py = s["points"][0]
            elif s["shape_type"] == "rectangle":
                # Rectangle: use center
                pts = s["points"]
                px = (pts[0][0] + pts[2][0]) / 2
                py = (pts[0][1] + pts[2][1]) / 2
            else:
                continue

            # Classify by description
            desc_lower = desc.lower().replace("，", ",")  # handle Chinese comma

            if "match_ball" in desc_lower:
                match_ball[fi] = (px, py)
                all_ball[fi] = (px, py)

                if "bounce" in desc_lower or "boounce" in desc_lower:
                    bounces.append((fi, px, py))
                if "serve" in desc_lower:
                    serves.append((fi, px, py))
                if "shot" in desc_lower:
                    shots.append((fi, px, py))
            elif "bounce" in desc_lower:
                bounces.append((fi, px, py))
                all_ball[fi] = (px, py)
            elif "shot" in desc_lower:
                shots.append((fi, px, py))
                all_ball[fi] = (px, py)

            # Also check for auto-detected ball (blob metadata in description)
            if "blob_sum" in desc and fi not in all_ball:
                all_ball[fi] = (px, py)

    return {
        "match_ball": match_ball,
        "bounces": bounces,
        "serves": serves,
        "shots": shots,
        "all_ball": all_ball,
    }


def run_pipeline(max_frames: int, mode: str = "multi", top_k: int = 2):
    """Run the tracking pipeline and return detections + bounces.

    Returns:
        det66: {frame: (px, py, conf)}
        points_3d: {frame: (x, y, z, ray_dist)}
        bounces: list of bounce dicts
    """
    from tools.render_tracking_video import (
        build_detector,
        build_flight_mask,
        detect_bounces,
        detect_bounces_2d,
        refine_bounces_with_2d,
        merge_3d_and_2d_bounces,
        load_config,
        run_detection_multi,
        smooth_trajectory_sg,
        triangulate_detections,
        triangulate_multi_blob,
    )

    cfg = load_config()
    detector, postproc = build_detector(cfg)

    logger.info("--- cam66 ---")
    multi66, det66, n66 = run_detection_multi(
        "uploads/cam66_20260307_173403_2min.mp4",
        detector, postproc, max_frames, top_k=top_k,
    )

    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("--- cam68 ---")
    multi68, det68, n68 = run_detection_multi(
        "uploads/cam68_20260307_173403_2min.mp4",
        detector, postproc, max_frames, top_k=top_k,
    )

    if mode == "viterbi":
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
        matched_frames = set(points_3d.keys())
        render_det66 = {fi: det66[fi] for fi in matched_frames if fi in det66}
    elif mode == "multi":
        points_3d, chosen_pixels, stats = triangulate_multi_blob(multi66, multi68, cfg)
        matched_frames = set(points_3d.keys())
        render_det66 = {fi: det66[fi] for fi in matched_frames if fi in det66}
    else:
        points_3d = triangulate_detections(det66, det68, cfg)
        _, render_det66, _ = build_flight_mask(points_3d, det66, det68)

    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)
    bounces = detect_bounces(smoothed_3d)

    # Also run 2D pixel bounce detection (with 3D confirmation for rally filtering)
    bounces_2d = detect_bounces_2d(det66, det68, cfg, smoothed_3d=smoothed_3d)

    # Also run 2D-refined 3D bounces
    bounces_refined = refine_bounces_with_2d(bounces, det66, det68, cfg)

    # Merged: 3D bounces + non-overlapping 2D bounces
    bounces_merged = merge_3d_and_2d_bounces(bounces, bounces_2d)

    # Apply same net-crossing anchor filter to 2D bounces
    from tools.render_tracking_video import detect_net_crossings
    net_crossings = detect_net_crossings(smoothed_3d, fps=25.0)
    nc_frames = [nc["frame"] for nc in net_crossings]
    NC_ANCHOR_WINDOW = 150
    if nc_frames:
        pre_2d = len(bounces_2d)
        bounces_2d = [
            b for b in bounces_2d
            if any(abs(b["frame"] - ncf) <= NC_ANCHOR_WINDOW for ncf in nc_frames)
        ]
        removed_2d = pre_2d - len(bounces_2d)
        if removed_2d:
            logger.info("2D net-crossing anchor filter removed %d bounce(s)", removed_2d)

    return det66, det68, render_det66, points_3d, smoothed_3d, bounces, bounces_2d, bounces_refined, bounces_merged, cfg


def eval_detection_recall(gt, det66, render_det66):
    """Evaluate: how many GT match_ball frames did we detect?"""
    gt_frames = set(gt["match_ball"].keys())
    det_frames = set(det66.keys())
    render_frames = set(render_det66.keys())

    raw_hit = gt_frames & det_frames
    render_hit = gt_frames & render_frames

    logger.info("=== Detection Recall (cam66) ===")
    logger.info("  GT match_ball frames: %d", len(gt_frames))
    logger.info("  Raw TrackNet detections: %d", len(det_frames))
    logger.info("  After pipeline filter: %d", len(render_frames))
    logger.info("  Raw recall: %d/%d = %.1f%%",
                len(raw_hit), len(gt_frames), 100.0 * len(raw_hit) / len(gt_frames))
    logger.info("  Pipeline recall: %d/%d = %.1f%%",
                len(render_hit), len(gt_frames), 100.0 * len(render_hit) / len(gt_frames))

    # False positive rate (detection on non-match_ball frames)
    non_gt = det_frames - gt_frames
    logger.info("  Extra detections (not in GT match_ball): %d", len(non_gt))

    return {
        "gt_frames": len(gt_frames),
        "raw_recall": len(raw_hit) / len(gt_frames) if gt_frames else 0,
        "pipeline_recall": len(render_hit) / len(gt_frames) if gt_frames else 0,
    }


def eval_pixel_accuracy(gt, det66, render_det66):
    """Evaluate pixel-level accuracy on GT match_ball frames."""
    gt_mb = gt["match_ball"]
    errors_raw = []
    errors_render = []

    for fi, (gx, gy) in gt_mb.items():
        if fi in det66:
            px, py, _ = det66[fi]
            errors_raw.append(math.hypot(px - gx, py - gy))
        if fi in render_det66:
            px, py, _ = render_det66[fi]
            errors_render.append(math.hypot(px - gx, py - gy))

    logger.info("=== Pixel Accuracy (cam66 vs GT) ===")
    if errors_raw:
        arr = np.array(errors_raw)
        logger.info("  Raw detections (%d frames):", len(arr))
        logger.info("    mean=%.1fpx  median=%.1fpx  p90=%.1fpx  max=%.1fpx",
                    arr.mean(), np.median(arr), np.percentile(arr, 90), arr.max())
        logger.info("    <5px: %.1f%%  <10px: %.1f%%  <20px: %.1f%%",
                    100 * np.mean(arr < 5), 100 * np.mean(arr < 10), 100 * np.mean(arr < 20))

    if errors_render:
        arr = np.array(errors_render)
        logger.info("  Pipeline output (%d frames):", len(arr))
        logger.info("    mean=%.1fpx  median=%.1fpx  p90=%.1fpx  max=%.1fpx",
                    arr.mean(), np.median(arr), np.percentile(arr, 90), arr.max())
        logger.info("    <5px: %.1f%%  <10px: %.1f%%  <20px: %.1f%%",
                    100 * np.mean(arr < 5), 100 * np.mean(arr < 10), 100 * np.mean(arr < 20))

    return {
        "raw_mean_px": float(np.mean(errors_raw)) if errors_raw else None,
        "render_mean_px": float(np.mean(errors_render)) if errors_render else None,
    }


def eval_bounces(gt, detected_bounces, tolerance_frames=5):
    """Evaluate bounce detection against GT.

    A detected bounce is "matched" if within tolerance_frames of a GT bounce.
    """
    gt_bounces = gt["bounces"]

    logger.info("=== Bounce Detection ===")
    logger.info("  GT bounces: %d", len(gt_bounces))
    logger.info("  Detected bounces: %d", len(detected_bounces))

    # Match detected → GT
    gt_matched = set()
    det_matched = set()
    matches = []

    for di, db in enumerate(detected_bounces):
        df = db["frame"]
        best_dist = float("inf")
        best_gi = -1
        for gi, (gf, gpx, gpy) in enumerate(gt_bounces):
            dist = abs(df - gf)
            if dist < best_dist:
                best_dist = dist
                best_gi = gi

        if best_dist <= tolerance_frames and best_gi not in gt_matched:
            gt_matched.add(best_gi)
            det_matched.add(di)
            gf, gpx, gpy = gt_bounces[best_gi]
            matches.append({
                "gt_frame": gf, "det_frame": df,
                "frame_error": df - gf,
                "gt_px": (gpx, gpy),
                "det_3d": (db["x"], db["y"], db["z"]),
            })

    # Precision / Recall
    precision = len(det_matched) / len(detected_bounces) if detected_bounces else 0
    recall = len(gt_matched) / len(gt_bounces) if gt_bounces else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    logger.info("  Tolerance: ±%d frames", tolerance_frames)
    logger.info("  True positives: %d", len(matches))
    logger.info("  False positives: %d (detected but no GT nearby)", len(detected_bounces) - len(det_matched))
    logger.info("  False negatives: %d (GT missed)", len(gt_bounces) - len(gt_matched))
    logger.info("  Precision: %.1f%%", 100 * precision)
    logger.info("  Recall: %.1f%%", 100 * recall)
    logger.info("  F1: %.1f%%", 100 * f1)

    logger.info("\n  Matched bounces:")
    for m in matches:
        logger.info("    GT frame=%d → Det frame=%d (Δ=%+d)  3D=(%.1f, %.1f, z=%.2f)",
                    m["gt_frame"], m["det_frame"], m["frame_error"],
                    m["det_3d"][0], m["det_3d"][1], m["det_3d"][2])

    # Unmatched GT
    unmatched_gt = [gt_bounces[gi] for gi in range(len(gt_bounces)) if gi not in gt_matched]
    if unmatched_gt:
        logger.info("\n  Missed GT bounces:")
        for gf, gpx, gpy in unmatched_gt:
            logger.info("    GT frame=%d  px=(%.1f, %.1f)", gf, gpx, gpy)

    # Unmatched detections (false positives)
    unmatched_det = [detected_bounces[di] for di in range(len(detected_bounces)) if di not in det_matched]
    if unmatched_det:
        logger.info("\n  False positive bounces:")
        for db in unmatched_det:
            logger.info("    Det frame=%d  3D=(%.1f, %.1f, z=%.2f)",
                        db["frame"], db["x"], db["y"], db["z"])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": len(matches),
        "fp": len(detected_bounces) - len(det_matched),
        "fn": len(gt_bounces) - len(gt_matched),
    }


def eval_rally_coverage(gt, points_3d):
    """Analyze coverage: which GT match_ball ranges have 3D tracking?"""
    gt_frames = sorted(gt["match_ball"].keys())
    if not gt_frames:
        return

    # Find contiguous GT ranges
    ranges = []
    start = gt_frames[0]
    prev = start
    for f in gt_frames[1:]:
        if f - prev > 5:
            ranges.append((start, prev))
            start = f
        prev = f
    ranges.append((start, prev))

    p3d_set = set(points_3d.keys())

    logger.info("=== Rally Coverage ===")
    logger.info("  GT match_ball rally segments:")
    total_gt = 0
    total_covered = 0
    for s, e in ranges:
        gt_in_range = [f for f in gt_frames if s <= f <= e]
        covered = [f for f in gt_in_range if f in p3d_set]
        total_gt += len(gt_in_range)
        total_covered += len(covered)
        pct = 100 * len(covered) / len(gt_in_range) if gt_in_range else 0
        logger.info("    frames %d-%d (%d GT frames, %d tracked = %.0f%%)",
                    s, e, len(gt_in_range), len(covered), pct)

    logger.info("  Overall: %d/%d GT frames tracked (%.1f%%)",
                total_covered, total_gt, 100 * total_covered / total_gt if total_gt else 0)


def main():
    logger.info("Loading GT annotations...")
    gt = load_gt(GT_DIR, MAX_FRAMES)

    logger.info(
        "GT: %d match_ball, %d bounces, %d serves, %d shots",
        len(gt["match_ball"]), len(gt["bounces"]),
        len(gt["serves"]), len(gt["shots"]),
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["multi", "viterbi"], default="multi")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("Running pipeline (%s mode)...", args.mode)
    logger.info("=" * 60)
    det66, det68, render_det66, points_3d, smoothed_3d, bounces, bounces_2d, bounces_refined, bounces_merged, cfg = run_pipeline(
        MAX_FRAMES, mode=args.mode, top_k=args.top_k,
    )

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    eval_detection_recall(gt, det66, render_det66)
    logger.info("")
    eval_pixel_accuracy(gt, det66, render_det66)
    logger.info("")
    logger.info("--- 3D V-shape Bounce Detection ---")
    result_3d = eval_bounces(gt, bounces, tolerance_frames=5)
    logger.info("")
    logger.info("--- 2D Pixel Bounce Detection (standalone) ---")
    result_2d = eval_bounces(gt, bounces_2d, tolerance_frames=5)
    logger.info("")
    logger.info("--- 2D-Refined 3D Bounces ---")
    result_refined = eval_bounces(gt, bounces_refined, tolerance_frames=5)
    logger.info("")
    logger.info("--- 3D + 2D Merged ---")
    result_merged = eval_bounces(gt, bounces_merged, tolerance_frames=5)
    logger.info("")

    # ── Summary comparison table ──────────────────────────────────
    logger.info("=" * 70)
    logger.info("BOUNCE DETECTION COMPARISON")
    logger.info("=" * 70)
    logger.info("%-20s | %3s | %3s | %3s | %9s | %6s | %5s",
                "Method", "TP", "FP", "FN", "Precision", "Recall", "F1")
    logger.info("-" * 70)
    for name, r in [("3D V-shape", result_3d),
                    ("2D pixel (alone)", result_2d),
                    ("3D + 2D refined", result_refined),
                    ("3D + 2D merged", result_merged)]:
        logger.info("%-20s | %3d | %3d | %3d | %8.1f%% | %5.1f%% | %4.1f%%",
                    name, r["tp"], r["fp"], r["fn"],
                    100 * r["precision"], 100 * r["recall"], 100 * r["f1"])
    logger.info("=" * 70)

    logger.info("")
    eval_rally_coverage(gt, points_3d)


if __name__ == "__main__":
    main()
