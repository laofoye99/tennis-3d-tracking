"""Evaluate detection pipeline against match_ball GT annotations.

Loads match_ball labels (with bounce/serve/shot events) from LabelMe annotations,
runs motion+YOLO pipeline, and evaluates:
  1. Match ball recall / precision / position error
  2. Bounce recall (GT bounce frames detected correctly)
  3. Per-segment analysis (which rally segments are weakest)

Usage:
    python -m tools.eval_match_ball
    python -m tools.eval_match_ball --max-frames 900
"""

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

CAM66_VIDEO = "uploads/cam66_20260307_173403_2min.mp4"
CAM66_LABELS = "uploads/cam66_20260307_173403_2min"
VERIFIER_PATH = "model_weight/blob_verifier_yolo.pt"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
OUT_DIR = Path("exports/eval_match_ball")
DIST_THRESHOLD = 15.0
OSD_MASK = (0, 0, 620, 41)


# ── GT Loading (match_ball specific) ────────────────────────────────────────


def load_match_ball_gt(label_dir: str, max_frame: int = 9999) -> dict:
    """Load match_ball GT with event tags from LabelMe description field.

    Returns:
        {
            "positions": {frame_index: (px, py)},   # match_ball pixel positions
            "bounces": [(frame_index, px, py), ...],
            "serves": [(frame_index, px, py), ...],
            "shots": [(frame_index, px, py), ...],
        }
    """
    label_path = Path(label_dir)
    positions = {}
    bounces = []
    serves = []
    shots = []

    for fi in range(max_frame + 1):
        fn = label_path / f"{fi:05d}.json"
        if not fn.exists():
            continue
        with open(fn) as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            desc = (shape.get("description") or "").lower()
            pts = shape.get("points", [])
            if not pts or "match_ball" not in desc:
                continue

            # Use first point (rectangle top-left or point annotation)
            if shape.get("shape_type") == "rectangle" and len(pts) >= 2:
                px = (pts[0][0] + pts[1][0]) / 2
                py = (pts[0][1] + pts[1][1]) / 2
            else:
                px, py = pts[0][0], pts[0][1]

            positions[fi] = (px, py)

            if "bounce" in desc:
                bounces.append((fi, px, py))
            if "serve" in desc:
                serves.append((fi, px, py))
            if "shot" in desc:
                shots.append((fi, px, py))
            break  # one match_ball per frame

    return {
        "positions": positions,
        "bounces": bounces,
        "serves": serves,
        "shots": shots,
    }


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_match_ball(
    detections: dict[int, dict],
    gt: dict,
    dist_threshold: float = DIST_THRESHOLD,
) -> dict:
    """Evaluate detections against match_ball GT.

    Returns detailed metrics including per-event recall.
    """
    positions = gt["positions"]

    tp = 0
    fn = 0
    fp = 0
    errors = []
    tp_frames = []
    fn_frames = []
    fp_frames = []

    for fi, (gx, gy) in positions.items():
        det = detections.get(fi)
        if det is None:
            fn += 1
            fn_frames.append(fi)
            continue

        dist = float(np.hypot(det["pixel_x"] - gx, det["pixel_y"] - gy))
        if dist <= dist_threshold:
            tp += 1
            errors.append(dist)
            tp_frames.append(fi)
        else:
            fp += 1
            fp_frames.append((fi, dist))

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    # Event recall
    def event_recall(events):
        if not events:
            return 1.0, 0, []
        hit = 0
        missed = []
        for fi, gx, gy in events:
            det = detections.get(fi)
            if det is not None:
                dist = float(np.hypot(det["pixel_x"] - gx, det["pixel_y"] - gy))
                if dist <= dist_threshold:
                    hit += 1
                    continue
            missed.append(fi)
        return hit / len(events), hit, missed

    bounce_recall, bounce_hit, bounce_missed = event_recall(gt["bounces"])
    serve_recall, serve_hit, serve_missed = event_recall(gt["serves"])
    shot_recall, shot_hit, shot_missed = event_recall(gt["shots"])

    return {
        "gt_frames": len(positions),
        "detected_frames": len(detections),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "recall": recall,
        "precision": precision,
        "mean_error_px": float(np.mean(errors)) if errors else 0.0,
        "median_error_px": float(np.median(errors)) if errors else 0.0,
        "bounce_recall": bounce_recall,
        "bounce_total": len(gt["bounces"]),
        "bounce_hit": bounce_hit,
        "bounce_missed": bounce_missed,
        "serve_recall": serve_recall,
        "serve_total": len(gt["serves"]),
        "serve_hit": serve_hit,
        "serve_missed": serve_missed,
        "shot_recall": shot_recall,
        "shot_total": len(gt["shots"]),
        "shot_hit": shot_hit,
        "shot_missed": shot_missed,
        "fn_frames": fn_frames,
        "fp_frames": fp_frames,
    }


def analyze_gaps(fn_frames: list[int], gt_positions: dict) -> list[dict]:
    """Analyze consecutive miss gaps to find weak segments."""
    if not fn_frames:
        return []

    gaps = []
    current = [fn_frames[0]]
    for fi in fn_frames[1:]:
        if fi <= current[-1] + 2:  # allow 1-frame tolerance
            current.append(fi)
        else:
            gaps.append(current)
            current = [fi]
    gaps.append(current)

    # Sort by gap length (worst first)
    gap_info = []
    for g in gaps:
        gap_info.append({
            "start": g[0],
            "end": g[-1],
            "length": len(g),
            "frames": g[:10],  # first 10 for display
        })
    gap_info.sort(key=lambda x: x["length"], reverse=True)
    return gap_info


# ── Print Results ───────────────────────────────────────────────────────────


def print_results(metrics: dict, gt: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 65)
    print("  MATCH BALL EVALUATION (cam66)")
    print("=" * 65)

    print(f"\n  GT match_ball frames:  {metrics['gt_frames']}")
    print(f"  Total detections:      {metrics['detected_frames']}")
    print(f"  Threshold:             {DIST_THRESHOLD}px")

    print(f"\n  ── Detection Metrics ──")
    print(f"  Recall:     {100 * metrics['recall']:.1f}%  (TP={metrics['tp']}, FN={metrics['fn']})")
    print(f"  Precision:  {100 * metrics['precision']:.1f}%  (TP={metrics['tp']}, FP={metrics['fp']})")
    print(f"  Mean err:   {metrics['mean_error_px']:.1f}px")
    print(f"  Median err: {metrics['median_error_px']:.1f}px")

    print(f"\n  ── Event Recall ──")
    print(f"  Bounce:  {metrics['bounce_hit']}/{metrics['bounce_total']}  "
          f"({100 * metrics['bounce_recall']:.0f}%)  "
          f"missed: {metrics['bounce_missed']}")
    print(f"  Serve:   {metrics['serve_hit']}/{metrics['serve_total']}  "
          f"({100 * metrics['serve_recall']:.0f}%)  "
          f"missed: {metrics['serve_missed']}")
    print(f"  Shot:    {metrics['shot_hit']}/{metrics['shot_total']}  "
          f"({100 * metrics['shot_recall']:.0f}%)  "
          f"missed: {metrics['shot_missed']}")

    # Gap analysis
    gaps = analyze_gaps(metrics["fn_frames"], gt["positions"])
    if gaps:
        print(f"\n  ── Worst Miss Gaps ──")
        for g in gaps[:10]:
            print(f"  frames {g['start']}-{g['end']} ({g['length']} missed)")

    # Wrong position analysis
    if metrics["fp_frames"]:
        print(f"\n  ── Wrong Detections (>15px from GT) ──")
        for fi, dist in sorted(metrics["fp_frames"], key=lambda x: -x[1])[:10]:
            print(f"  frame {fi}: {dist:.0f}px off")


# ── Post-Processing: Trajectory-Aware Selection ─────────────────────────────


def build_dead_ball_registry(
    all_candidates: dict[int, list[dict]],
    cluster_radius: float = 25.0,
    min_persistence: int = 30,
) -> list[tuple[float, float, int]]:
    """Identify persistent static detections (dead balls).

    Scans all candidates across all frames and finds positions that appear
    repeatedly at nearly the same location (within cluster_radius).

    Returns list of (x, y, count) for each dead ball cluster.
    """
    # Collect all candidate positions
    all_positions = []
    for fi, cands in all_candidates.items():
        for c in cands:
            all_positions.append((c["pixel_x"], c["pixel_y"], fi))

    if not all_positions:
        return []

    # Simple grid-based clustering
    clusters: list[list[tuple[float, float, int]]] = []

    for px, py, fi in all_positions:
        merged = False
        for cluster in clusters:
            cx = np.mean([p[0] for p in cluster])
            cy = np.mean([p[1] for p in cluster])
            if np.hypot(px - cx, py - cy) <= cluster_radius:
                cluster.append((px, py, fi))
                merged = True
                break
        if not merged:
            clusters.append([(px, py, fi)])

    # Find clusters with high persistence
    dead_balls = []
    for cluster in clusters:
        unique_frames = len(set(p[2] for p in cluster))
        if unique_frames >= min_persistence:
            cx = float(np.mean([p[0] for p in cluster]))
            cy = float(np.mean([p[1] for p in cluster]))
            dead_balls.append((cx, cy, unique_frames))
            log.info("Dead ball at (%.0f, %.0f) in %d frames", cx, cy, unique_frames)

    return dead_balls


def smart_top1_selection(
    raw_top1: dict[int, dict],
    all_candidates: dict[int, list[dict]],
    dead_balls: list[tuple[float, float, int]],
    dead_ball_radius: float = 30.0,
    dead_ball_penalty: float = 0.5,
) -> dict[int, dict]:
    """Re-select top-1 using dead ball penalty.

    For each frame, score candidates by:
      score = yolo_conf - penalty (if near dead ball)
    Then pick highest scoring candidate.
    """
    result = {}
    reselected = 0

    for fi, det in raw_top1.items():
        cands = all_candidates.get(fi, [])
        if not cands:
            result[fi] = det
            continue

        # Score each candidate
        best_score = -1
        best_c = None

        for c in cands:
            score = c.get("yolo_conf", 0.3)

            # Apply dead ball penalty
            for dbx, dby, _ in dead_balls:
                if np.hypot(c["pixel_x"] - dbx, c["pixel_y"] - dby) <= dead_ball_radius:
                    score -= dead_ball_penalty
                    break

            if score > best_score:
                best_score = score
                best_c = c

        if best_c is not None:
            old_px = det["pixel_x"]
            old_py = det["pixel_y"]
            if np.hypot(best_c["pixel_x"] - old_px, best_c["pixel_y"] - old_py) > 5:
                reselected += 1
            result[fi] = best_c
        else:
            result[fi] = det

    log.info("Smart top-1: %d frames reselected (dead ball penalty)", reselected)
    return result


def median_trajectory_filter(
    raw_top1: dict[int, dict],
    all_candidates: dict[int, list[dict]],
    max_frames: int,
    window_half: int = 4,
    outlier_threshold_px: float = 150.0,
) -> dict[int, dict]:
    """Clean detections using sliding window median + candidate reselection.

    For each frame:
      1. Compute median position from surrounding frames (±window_half)
      2. If top-1 is far from median → it's an outlier (wrong ball)
      3. For outliers: search all candidates for one closer to median
      4. If no good candidate → drop frame

    This handles both jump-to-dead-ball and oscillation cases.
    """
    sorted_frames = sorted(raw_top1.keys())
    if not sorted_frames:
        return {}

    positions = {fi: (raw_top1[fi]["pixel_x"], raw_top1[fi]["pixel_y"]) for fi in sorted_frames}

    # ── Compute median reference for each frame ──
    median_ref: dict[int, tuple[float, float]] = {}
    for idx, fi in enumerate(sorted_frames):
        # Gather neighbors in window
        neighbors_x = []
        neighbors_y = []
        for j in range(max(0, idx - window_half), min(len(sorted_frames), idx + window_half + 1)):
            nfi = sorted_frames[j]
            if abs(nfi - fi) <= window_half * 2:  # within temporal window
                nx, ny = positions[nfi]
                neighbors_x.append(nx)
                neighbors_y.append(ny)

        if len(neighbors_x) >= 3:
            median_ref[fi] = (float(np.median(neighbors_x)), float(np.median(neighbors_y)))

    # ── Filter: keep consistent, reselect or drop outliers ──
    result: dict[int, dict] = {}
    kept = 0
    reselected = 0
    dropped = 0

    for fi in sorted_frames:
        det = raw_top1[fi]
        px, py = det["pixel_x"], det["pixel_y"]

        if fi not in median_ref:
            # Not enough neighbors → keep as-is
            result[fi] = det
            kept += 1
            continue

        med_x, med_y = median_ref[fi]
        dist_to_median = float(np.hypot(px - med_x, py - med_y))

        if dist_to_median <= outlier_threshold_px:
            # Consistent with median → keep
            result[fi] = det
            kept += 1
            continue

        # Outlier: try to find a better candidate
        candidates = all_candidates.get(fi, [])
        best_c = None
        best_d = float("inf")

        for c in candidates:
            cx, cy = c["pixel_x"], c["pixel_y"]
            d = float(np.hypot(cx - med_x, cy - med_y))
            if d < best_d:
                best_d = d
                best_c = c

        if best_c is not None and best_d <= outlier_threshold_px:
            result[fi] = best_c
            reselected += 1
        else:
            # No good candidate → drop (FN > FP)
            dropped += 1

    log.info(
        "Median filter: %d raw → %d output (kept=%d, reselected=%d, dropped=%d)",
        len(raw_top1), len(result), kept, reselected, dropped,
    )
    return result


def cross_camera_validate(
    detections: dict[int, dict],
    other_cam_detections: dict[int, dict],
    max_world_dist: float = 5.0,
) -> dict[int, dict]:
    """Remove cam66 detections that disagree with cam68 in world space.

    If both cameras detect something, and their world coordinates differ by
    more than max_world_dist meters, cam66 is likely tracking a wrong ball.
    Only remove if cam68 has a consistent trajectory (not itself wrong).
    """
    removed = 0
    result = {}

    for fi, det in detections.items():
        other = other_cam_detections.get(fi)

        if other is None or other.get("world_x") is None or det.get("world_x") is None:
            # No cam68 to compare → keep cam66
            result[fi] = det
            continue

        world_dist = float(np.hypot(
            det["world_x"] - other["world_x"],
            det["world_y"] - other["world_y"],
        ))

        if world_dist <= max_world_dist:
            result[fi] = det
        else:
            # Disagreement → remove cam66 detection
            removed += 1

    log.info("Cross-camera validation: %d frames removed (world dist > %.1fm)", removed, max_world_dist)
    return result


def cross_camera_rescue(
    detections: dict[int, dict],
    other_cam_detections: dict[int, dict],
    other_cam_candidates: dict[int, list[dict]],
    homography_self,
    homography_other,
    max_frames: int,
    max_dist_px: float = 100.0,
) -> dict[int, dict]:
    """Rescue missing frames using the other camera's detection.

    Two modes:
      1. Consistency mode: if self has neighbors within 10 frames,
         only rescue if consistent with them
      2. Trust mode: if self has a gap >5 frames (median filter removed a segment),
         trust the other camera directly (use cam68 trajectory consistency instead)
    """
    rescued = dict(detections)
    sorted_frames = sorted(detections.keys())
    rescue_count = 0

    # Build cam68 trajectory for self-consistency check
    other_sorted = sorted(other_cam_detections.keys())

    for fi in range(max_frames):
        if fi in rescued:
            continue

        other = other_cam_detections.get(fi)
        if other is None or other.get("world_x") is None:
            continue

        # Project other camera's world coords to self's pixel space
        wx, wy = other["world_x"], other["world_y"]
        px, py = homography_self.world_to_pixel(wx, wy)

        # Sanity check: pixel should be within frame
        if px < 0 or px > 1920 or py < 0 or py > 1080:
            continue

        # Find nearest self-detections
        prev_fi = None
        next_fi = None
        for sf in sorted_frames:
            if sf < fi:
                prev_fi = sf
            elif sf > fi:
                next_fi = sf
                break

        # Determine gap size to nearest self-detection
        gap_prev = (fi - prev_fi) if prev_fi is not None else 999
        gap_next = (next_fi - fi) if next_fi is not None else 999
        min_gap = min(gap_prev, gap_next)

        if min_gap <= 5:
            # Consistency mode: check against self neighbors
            consistent = False
            if prev_fi is not None and gap_prev <= 10:
                d_prev = rescued[prev_fi]
                dist = float(np.hypot(px - d_prev["pixel_x"], py - d_prev["pixel_y"]))
                if dist <= max_dist_px * gap_prev:
                    consistent = True
            if next_fi is not None and gap_next <= 10:
                d_next = rescued[next_fi]
                dist = float(np.hypot(px - d_next["pixel_x"], py - d_next["pixel_y"]))
                if dist <= max_dist_px * gap_next:
                    consistent = True
            if not consistent:
                continue
        else:
            # Trust mode: self has a large gap, check cam68's own consistency
            # Verify cam68 detection is consistent with its own neighbors
            cam68_consistent = False
            for ofi in other_sorted:
                if ofi == fi:
                    continue
                if abs(ofi - fi) <= 3 and ofi in other_cam_detections:
                    o_other = other_cam_detections[ofi]
                    if o_other.get("world_x") is not None:
                        dist = float(np.hypot(
                            other["world_x"] - o_other["world_x"],
                            other["world_y"] - o_other["world_y"],
                        ))
                        if dist < 3.0:  # 3m in world space
                            cam68_consistent = True
                            break
            if not cam68_consistent:
                continue

        rescued[fi] = {
            "pixel_x": px,
            "pixel_y": py,
            "world_x": wx,
            "world_y": wy,
            "yolo_conf": other.get("yolo_conf", 0.3),
            "confidence": 0.0,
            "rescued_from": "cross_camera",
        }
        rescue_count += 1

    log.info("Cross-camera rescue: %d frames added", rescue_count)
    return rescued


def interpolate_gaps(
    detections: dict[int, dict],
    max_frames: int,
    max_gap: int = 3,
) -> dict[int, dict]:
    """Fill small gaps by quadratic interpolation using 2 neighbors on each side.

    Uses up to 2 points before and after the gap to fit a polynomial,
    giving better position estimates than simple linear interpolation.
    Falls back to linear when not enough context points exist.
    """
    result = dict(detections)
    sorted_frames = sorted(detections.keys())
    interp_count = 0

    for i in range(len(sorted_frames) - 1):
        f1 = sorted_frames[i]
        f2 = sorted_frames[i + 1]
        gap = f2 - f1

        if gap <= 1 or gap > max_gap + 1:
            continue

        # Gather context points (up to 2 before gap, up to 2 after gap)
        ctx_frames = []
        ctx_x = []
        ctx_y = []

        # Before gap
        for j in range(max(0, i - 1), i + 1):
            fj = sorted_frames[j]
            if abs(fj - f1) <= 10:  # within temporal range
                d = detections[fj]
                ctx_frames.append(fj)
                ctx_x.append(d["pixel_x"])
                ctx_y.append(d["pixel_y"])

        # After gap
        for j in range(i + 1, min(len(sorted_frames), i + 3)):
            fj = sorted_frames[j]
            if abs(fj - f2) <= 10:
                d = detections[fj]
                ctx_frames.append(fj)
                ctx_x.append(d["pixel_x"])
                ctx_y.append(d["pixel_y"])

        if len(ctx_frames) < 2:
            continue

        # Fit polynomial (degree = min(2, n-1))
        degree = min(2, len(ctx_frames) - 1)
        try:
            poly_x = np.polyfit(ctx_frames, ctx_x, degree)
            poly_y = np.polyfit(ctx_frames, ctx_y, degree)
        except (np.linalg.LinAlgError, ValueError):
            # Fall back to linear
            d1 = detections[f1]
            d2 = detections[f2]
            poly_x = None

        for fi in range(f1 + 1, f2):
            if poly_x is not None:
                px = float(np.polyval(poly_x, fi))
                py = float(np.polyval(poly_y, fi))
            else:
                t = (fi - f1) / (f2 - f1)
                d1 = detections[f1]
                d2 = detections[f2]
                px = d1["pixel_x"] * (1 - t) + d2["pixel_x"] * t
                py = d1["pixel_y"] * (1 - t) + d2["pixel_y"] * t

            # Clamp to frame bounds
            px = max(0, min(1920, px))
            py = max(0, min(1080, py))

            result[fi] = {
                "pixel_x": px,
                "pixel_y": py,
                "world_x": 0.0,
                "world_y": 0.0,
                "yolo_conf": 0.0,
                "confidence": 0.0,
                "rescued_from": "interpolation",
            }
            interp_count += 1

    log.info("Interpolation: %d frames filled (max_gap=%d)", interp_count, max_gap)
    return result


def trajectory_extrapolation_reselection(
    detections: dict[int, dict],
    all_candidates: dict[int, list[dict]],
    max_frames: int,
    consistency_radius: float = 30.0,
    reselect_if_closer_by: float = 15.0,
) -> dict[int, dict]:
    """Reselect candidates using linear extrapolation from consistent neighbors.

    For each frame, find the nearest frames where detection is "self-consistent"
    (i.e., it's the closest candidate to itself — meaning no better option existed).
    Use these anchors to extrapolate, then check if any candidate is closer to
    the extrapolated position than the current detection.

    This targets frames where median filter picked the wrong candidate because
    the local window was contaminated by dead ball detections.
    """
    sorted_frames = sorted(detections.keys())
    if not sorted_frames:
        return {}

    # Identify "anchor" frames where detection is likely correct:
    # detection is the closest candidate (or within 5px of closest)
    anchors = set()
    for fi in sorted_frames:
        det = detections[fi]
        cands = all_candidates.get(fi, [])
        if not cands:
            continue
        # Check if detection matches its closest candidate
        det_px, det_py = det["pixel_x"], det["pixel_y"]
        min_d = float("inf")
        for c in cands:
            d = float(np.hypot(c["pixel_x"] - det_px, c["pixel_y"] - det_py))
            if d < min_d:
                min_d = d
        if min_d < 5.0:  # detection is close to a real candidate
            anchors.add(fi)

    result = dict(detections)
    reselected = 0

    for idx, fi in enumerate(sorted_frames):
        det = result[fi]
        px, py = det["pixel_x"], det["pixel_y"]

        # Find nearest anchors before and after
        prev_anchor = None
        next_anchor = None
        for j in range(idx - 1, max(0, idx - 15) - 1, -1):
            nfi = sorted_frames[j]
            if nfi in anchors:
                prev_anchor = nfi
                break
        for j in range(idx + 1, min(len(sorted_frames), idx + 15)):
            nfi = sorted_frames[j]
            if nfi in anchors:
                next_anchor = nfi
                break

        if prev_anchor is None and next_anchor is None:
            continue

        # Compute expected position by linear interpolation/extrapolation
        if prev_anchor is not None and next_anchor is not None:
            dp = result[prev_anchor]
            dn = result[next_anchor]
            t = (fi - prev_anchor) / max(next_anchor - prev_anchor, 1)
            ex = dp["pixel_x"] * (1 - t) + dn["pixel_x"] * t
            ey = dp["pixel_y"] * (1 - t) + dn["pixel_y"] * t
        elif prev_anchor is not None:
            # Only past anchor — extrapolate using velocity from before
            dp = result[prev_anchor]
            # Find another anchor before prev_anchor for velocity
            prev2 = None
            for j2 in range(sorted_frames.index(prev_anchor) - 1, max(0, sorted_frames.index(prev_anchor) - 10) - 1, -1):
                nfi2 = sorted_frames[j2]
                if nfi2 in anchors:
                    prev2 = nfi2
                    break
            if prev2 is not None:
                dp2 = result[prev2]
                vx = (dp["pixel_x"] - dp2["pixel_x"]) / max(prev_anchor - prev2, 1)
                vy = (dp["pixel_y"] - dp2["pixel_y"]) / max(prev_anchor - prev2, 1)
                dt = fi - prev_anchor
                ex = dp["pixel_x"] + vx * dt
                ey = dp["pixel_y"] + vy * dt
            else:
                ex, ey = dp["pixel_x"], dp["pixel_y"]
        else:
            # Only future anchor — extrapolate backward
            dn = result[next_anchor]
            next2 = None
            for j2 in range(sorted_frames.index(next_anchor) + 1, min(len(sorted_frames), sorted_frames.index(next_anchor) + 10)):
                nfi2 = sorted_frames[j2]
                if nfi2 in anchors:
                    next2 = nfi2
                    break
            if next2 is not None:
                dn2 = result[next2]
                vx = (dn2["pixel_x"] - dn["pixel_x"]) / max(next2 - next_anchor, 1)
                vy = (dn2["pixel_y"] - dn["pixel_y"]) / max(next2 - next_anchor, 1)
                dt = fi - next_anchor
                ex = dn["pixel_x"] + vx * dt
                ey = dn["pixel_y"] + vy * dt
            else:
                ex, ey = dn["pixel_x"], dn["pixel_y"]

        # Check if any candidate is closer to extrapolated position
        dist_current = float(np.hypot(px - ex, py - ey))
        if dist_current < reselect_if_closer_by:
            continue  # current detection is already close enough

        candidates = all_candidates.get(fi, [])
        best_c = None
        best_d = dist_current - reselect_if_closer_by

        for c in candidates:
            d = float(np.hypot(c["pixel_x"] - ex, c["pixel_y"] - ey))
            if d < best_d:
                best_d = d
                best_c = c

        if best_c is not None:
            result[fi] = best_c
            reselected += 1

    log.info("Trajectory extrapolation: %d anchors, %d reselected", len(anchors), reselected)
    return result


def final_candidate_reselection(
    detections: dict[int, dict],
    all_candidates: dict[int, list[dict]],
    max_frames: int,
    window_half: int = 5,
    improvement_threshold_px: float = 20.0,
) -> dict[int, dict]:
    """Final pass: reselect candidates using asymmetric (forward/backward) median.

    For each frame, compute median from:
      - backward (future) neighbors only
      - forward (past) neighbors only
      - full window
    Pick the median with lowest variance (most consistent direction).
    If a candidate is closer to this median than current detection, switch.

    This fixes segment boundaries where one direction is contaminated.
    """
    sorted_frames = sorted(detections.keys())
    if not sorted_frames:
        return {}

    result = dict(detections)
    reselected = 0

    for idx, fi in enumerate(sorted_frames):
        det = result[fi]
        px, py = det["pixel_x"], det["pixel_y"]

        # Gather forward (past) and backward (future) neighbors
        fwd_x, fwd_y = [], []
        bwd_x, bwd_y = [], []

        for j in range(max(0, idx - window_half), idx):
            nfi = sorted_frames[j]
            if abs(nfi - fi) <= window_half * 3:
                d = result[nfi]
                fwd_x.append(d["pixel_x"])
                fwd_y.append(d["pixel_y"])

        for j in range(idx + 1, min(len(sorted_frames), idx + window_half + 1)):
            nfi = sorted_frames[j]
            if abs(nfi - fi) <= window_half * 3:
                d = result[nfi]
                bwd_x.append(d["pixel_x"])
                bwd_y.append(d["pixel_y"])

        # Choose best reference: pick direction with lowest variance
        best_ref = None
        best_var = float("inf")

        for label, xs, ys in [("fwd", fwd_x, fwd_y), ("bwd", bwd_x, bwd_y), ("all", fwd_x + bwd_x, fwd_y + bwd_y)]:
            if len(xs) < 2:
                continue
            var = float(np.var(xs) + np.var(ys))
            if var < best_var:
                best_var = var
                best_ref = (float(np.median(xs)), float(np.median(ys)))

        if best_ref is None:
            continue

        med_x, med_y = best_ref
        dist_current = float(np.hypot(px - med_x, py - med_y))

        # Check candidates for a better match
        candidates = all_candidates.get(fi, [])
        best_c = None
        best_d = dist_current

        for c in candidates:
            cx, cy = c["pixel_x"], c["pixel_y"]
            d = float(np.hypot(cx - med_x, cy - med_y))
            if d < best_d - improvement_threshold_px:
                best_d = d
                best_c = c

        if best_c is not None:
            result[fi] = best_c
            reselected += 1

    log.info("Final reselection: %d frames reselected", reselected)
    return result


def velocity_outlier_filter(
    detections: dict[int, dict],
    all_candidates: dict[int, list[dict]],
    max_speed_px: float = 200.0,
) -> dict[int, dict]:
    """Remove detections that create impossible velocity jumps.

    If a detection jumps far from both neighbors (prev and next), it's likely
    a wrong ball. Try to reselect from candidates, or drop if no good option.
    """
    sorted_frames = sorted(detections.keys())
    if len(sorted_frames) < 3:
        return dict(detections)

    result = dict(detections)
    removed = 0
    reselected = 0

    for idx in range(1, len(sorted_frames) - 1):
        fi = sorted_frames[idx]
        f_prev = sorted_frames[idx - 1]
        f_next = sorted_frames[idx + 1]

        det = result[fi]
        d_prev = result[f_prev]
        d_next = result[f_next]

        px, py = det["pixel_x"], det["pixel_y"]
        dt_prev = fi - f_prev
        dt_next = f_next - fi

        # Speed from previous and to next
        speed_prev = float(np.hypot(px - d_prev["pixel_x"], py - d_prev["pixel_y"])) / max(dt_prev, 1)
        speed_next = float(np.hypot(px - d_next["pixel_x"], py - d_next["pixel_y"])) / max(dt_next, 1)

        # Both speeds high → likely a jump to wrong ball and back
        if speed_prev > max_speed_px and speed_next > max_speed_px:
            # Check neighbors are consistent with each other
            neighbor_speed = float(np.hypot(
                d_prev["pixel_x"] - d_next["pixel_x"],
                d_prev["pixel_y"] - d_next["pixel_y"],
            )) / max(f_next - f_prev, 1)

            if neighbor_speed < max_speed_px:
                # Neighbors consistent, this frame is the outlier
                # Try to find a candidate between neighbors
                mid_x = (d_prev["pixel_x"] + d_next["pixel_x"]) / 2
                mid_y = (d_prev["pixel_y"] + d_next["pixel_y"]) / 2

                candidates = all_candidates.get(fi, [])
                best_c = None
                best_d = float("inf")
                for c in candidates:
                    d = float(np.hypot(c["pixel_x"] - mid_x, c["pixel_y"] - mid_y))
                    if d < best_d:
                        best_d = d
                        best_c = c

                if best_c is not None and best_d < max_speed_px:
                    result[fi] = best_c
                    reselected += 1
                else:
                    del result[fi]
                    removed += 1

    log.info("Velocity filter: %d removed, %d reselected", removed, reselected)
    return result


# ── Video Rendering ─────────────────────────────────────────────────────────


def render_video(
    video_path: str,
    detections: dict[int, dict],
    gt: dict,
    output_path: Path,
    max_frames: int,
    dist_threshold: float,
) -> None:
    """Render evaluation video showing detections vs GT with trajectory."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    positions = gt["positions"]
    bounce_set = {fi for fi, _, _ in gt["bounces"]}
    serve_set = {fi for fi, _, _ in gt["serves"]}
    shot_set = {fi for fi, _, _ in gt["shots"]}

    # Trajectory buffer
    trail_len = 15
    det_trail: list[tuple[int, int]] = []
    gt_trail: list[tuple[int, int]] = []

    for fi in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        det = detections.get(fi)
        gt_pos = positions.get(fi)

        # Classify detection
        status = ""
        if gt_pos is not None and det is not None:
            dist = float(np.hypot(det["pixel_x"] - gt_pos[0], det["pixel_y"] - gt_pos[1]))
            if dist <= dist_threshold:
                status = "TP"
            else:
                status = "FP"
        elif gt_pos is not None and det is None:
            status = "FN"
        elif gt_pos is None and det is not None:
            status = "DET"  # detection without GT (may or may not be correct)

        # Draw GT trail (white)
        if gt_pos is not None:
            gt_trail.append((int(gt_pos[0]), int(gt_pos[1])))
        if len(gt_trail) > trail_len:
            gt_trail = gt_trail[-trail_len:]
        for i in range(1, len(gt_trail)):
            alpha = i / len(gt_trail)
            cv2.line(frame, gt_trail[i - 1], gt_trail[i], (255, 255, 255), max(1, int(alpha * 2)))

        # Draw detection trail (colored by status)
        if det is not None:
            det_trail.append((int(det["pixel_x"]), int(det["pixel_y"])))
        if len(det_trail) > trail_len:
            det_trail = det_trail[-trail_len:]
        for i in range(1, len(det_trail)):
            alpha = i / len(det_trail)
            cv2.line(frame, det_trail[i - 1], det_trail[i], (0, 200, 255), max(1, int(alpha * 2)))

        # Draw GT position (diamond, white)
        if gt_pos is not None:
            gx, gy = int(gt_pos[0]), int(gt_pos[1])
            sz = 8
            pts_diamond = np.array([[gx, gy - sz], [gx + sz, gy], [gx, gy + sz], [gx - sz, gy]], np.int32)
            cv2.polylines(frame, [pts_diamond], True, (255, 255, 255), 2)

            # Event markers
            if fi in bounce_set:
                cv2.putText(frame, "BOUNCE", (gx + 12, gy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if fi in serve_set:
                cv2.putText(frame, "SERVE", (gx + 12, gy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if fi in shot_set:
                cv2.putText(frame, "SHOT", (gx + 12, gy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw detection (circle)
        if det is not None:
            dx, dy = int(det["pixel_x"]), int(det["pixel_y"])
            if status == "TP":
                color = (0, 255, 0)  # green = correct
            elif status == "FP":
                color = (0, 0, 255)  # red = wrong position
            else:
                color = (0, 200, 255)  # yellow = no GT to compare
            cv2.circle(frame, (dx, dy), 8, color, 2)
            conf_text = f"{det.get('yolo_conf', 0):.2f}"
            cv2.putText(frame, conf_text, (dx + 10, dy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Status bar
        bar_color = (0, 0, 0)
        cv2.rectangle(frame, (0, h - 35), (w, h), bar_color, -1)

        # Frame info
        info = f"Frame {fi}"
        if status:
            status_colors = {"TP": (0, 255, 0), "FP": (0, 0, 255), "FN": (0, 100, 255), "DET": (0, 200, 255)}
            cv2.putText(frame, f"{info}  {status}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_colors.get(status, (255, 255, 255)), 2)
        else:
            cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Legend
        cv2.putText(frame, "GT", (w - 280, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame, (w - 220, h - 15), 5, (0, 255, 0), -1)
        cv2.putText(frame, "TP", (w - 210, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (w - 170, h - 15), 5, (0, 0, 255), -1)
        cv2.putText(frame, "FP", (w - 160, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(frame, (w - 120, h - 15), 5, (0, 100, 255), -1)
        cv2.putText(frame, "FN", (w - 110, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        writer.write(frame)

        if (fi + 1) % 200 == 0:
            log.info("  Rendered %d/%d", fi + 1, max_frames)

    cap.release()
    writer.release()
    log.info("Video saved: %s", output_path)


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate match_ball detection")
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--dist-threshold", type=float, default=DIST_THRESHOLD)
    args = parser.parse_args()

    dist_threshold = args.dist_threshold

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load GT
    log.info("Loading match_ball GT from %s ...", CAM66_LABELS)
    gt = load_match_ball_gt(CAM66_LABELS, max_frame=args.max_frames)
    log.info(
        "GT: %d match_ball frames, %d bounces, %d serves, %d shots",
        len(gt["positions"]), len(gt["bounces"]), len(gt["serves"]), len(gt["shots"]),
    )

    from tools.eval_motion_yolo import (
        MotionParams,
        run_motion_yolo_pipeline,
        CAMERAS,
    )
    from app.pipeline.homography import HomographyTransformer

    params = MotionParams()

    # ── Stage 0: Raw detection (both cameras) ──
    log.info("=== Stage 0: Raw motion+YOLO detection ===")
    raw_top1_66, all_cands_66 = run_motion_yolo_pipeline(
        "cam66", CAMERAS["cam66"], params,
        max_frames=args.max_frames, return_all_candidates=True,
    )
    raw_top1_68, all_cands_68 = run_motion_yolo_pipeline(
        "cam68", CAMERAS["cam68"], params,
        max_frames=args.max_frames, return_all_candidates=True,
    )

    m0 = evaluate_match_ball(raw_top1_66, gt, dist_threshold=dist_threshold)
    print(f"\n  [Stage 0] Raw top-1: recall={100*m0['recall']:.1f}% prec={100*m0['precision']:.1f}% "
          f"FN={m0['fn']} FP={m0['fp']}")

    # ── Stage 1: Median trajectory filter ──
    log.info("=== Stage 1: Median trajectory filter ===")
    detections = median_trajectory_filter(raw_top1_66, all_cands_66, args.max_frames)

    m1 = evaluate_match_ball(detections, gt, dist_threshold=dist_threshold)
    print(f"  [Stage 1] Trajectory: recall={100*m1['recall']:.1f}% prec={100*m1['precision']:.1f}% "
          f"FN={m1['fn']} FP={m1['fp']}")

    # ── Stage 1b: Second pass median with tighter threshold ──
    log.info("=== Stage 1b: Tight median pass ===")
    # Rebuild candidates from current result for reselection
    detections = median_trajectory_filter(
        detections, all_cands_66, args.max_frames,
        window_half=6, outlier_threshold_px=100.0,
    )

    m1b = evaluate_match_ball(detections, gt, dist_threshold=dist_threshold)
    print(f"  [Stage 1b] Tight:   recall={100*m1b['recall']:.1f}% prec={100*m1b['precision']:.1f}% "
          f"FN={m1b['fn']} FP={m1b['fp']}")

    # ── Stage 2: Interpolate small gaps ──
    log.info("=== Stage 2: Gap interpolation ===")
    detections = interpolate_gaps(detections, args.max_frames, max_gap=5)

    m2 = evaluate_match_ball(detections, gt, dist_threshold=dist_threshold)
    print(f"  [Stage 2] +Interp:  recall={100*m2['recall']:.1f}% prec={100*m2['precision']:.1f}% "
          f"FN={m2['fn']} FP={m2['fp']}")

    # Show metrics at different thresholds
    for t in [25, 50]:
        mt = evaluate_match_ball(detections, gt, dist_threshold=float(t))
        print(f"    @{t}px: recall={100*mt['recall']:.1f}% prec={100*mt['precision']:.1f}% "
              f"TP={mt['tp']} FP={mt['fp']}")

    # Final evaluation
    metrics = m2
    print_results(metrics, gt)

    # Save results
    save_metrics = {k: v for k, v in metrics.items()}
    with open(OUT_DIR / "match_ball_metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)
    log.info("Metrics saved to %s", OUT_DIR / "match_ball_metrics.json")

    # Render video
    video_path = OUT_DIR / "match_ball_eval.mp4"
    log.info("Rendering evaluation video...")
    render_video(CAM66_VIDEO, detections, gt, video_path, args.max_frames, dist_threshold)


if __name__ == "__main__":
    main()
