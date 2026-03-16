"""Evaluate bounce/serve/shot/rally detection using GT match_ball trajectory.

Uses GT pixel positions → homography → world coords, then:
  1. pixel_y find_peaks → bounce detection
  2. world_y net crossing → rally state machine
  3. Compare detected events against GT labels

This validates the analysis pipeline INDEPENDENT of detection quality.

Usage:
    python -m tools.eval_rally_gt
    python -m tools.eval_rally_gt --render
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

CAM66_VIDEO = "uploads/cam66_20260307_173403_2min.mp4"
CAM68_VIDEO = "uploads/cam68_20260307_173403_2min.mp4"
CAM66_LABELS = "uploads/cam66_20260307_173403_2min"
CAM68_DETECTIONS = "exports/eval_motion_yolo/cam68_detections.json"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
OUT_DIR = Path("exports/eval_rally_gt")
FPS = 25.0

# Court constants (from app/analytics.py)
NET_Y = 11.885
COURT_X = 8.23
COURT_Y = 23.77
BASELINE_NEAR_MAX = 5.0
BASELINE_FAR_MIN = 18.77


# ── GT Loading ───────────────────────────────────────────────────────────────


def load_match_ball_gt(label_dir: str, max_frame: int = 9999) -> dict:
    """Load match_ball GT with event tags from LabelMe description field."""
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
            break

    return {
        "positions": positions,
        "bounces": bounces,
        "serves": serves,
        "shots": shots,
    }


# ── Bounce Detection (pixel_y find_peaks) ───────────────────────────────────


@dataclass
class DetectedBounce:
    frame_index: int
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    side: str  # "near" or "far"
    prominence: float


def detect_bounces_combined(
    trajectory: dict[int, dict],
    wy_prominence: float = 0.5,
    py_prominence: float = 8.0,
    distance: int = 5,
    cooldown: int = 8,
    court_y_max: float = 32.0,
) -> list[DetectedBounce]:
    """Detect bounces using BOTH world_y and pixel_y extrema (union).

    Strategy:
    - world_y extrema: catch direction-changing bounces in both halves
    - pixel_y maxima: catch near-end bounces (large pixel_y = close to camera)
    - Merge with dedup (cooldown prevents double-counting)

    Limitation: bounces where ball continues same world_y direction (e.g., serve
    bounce traveling away) are undetectable without height (Z) information.
    Dual-camera triangulation is needed for those cases.
    """
    sorted_frames = sorted(trajectory.keys())
    if len(sorted_frames) < 7:
        return []

    # Build contiguous segments (break at gaps > 3 frames)
    segments = []
    current_seg = [sorted_frames[0]]
    for i in range(1, len(sorted_frames)):
        if sorted_frames[i] - sorted_frames[i - 1] <= 3:
            current_seg.append(sorted_frames[i])
        else:
            if len(current_seg) >= 5:
                segments.append(current_seg)
            current_seg = [sorted_frames[i]]
    if len(current_seg) >= 5:
        segments.append(current_seg)

    all_candidates = []  # (frame, entry, prominence, side, source)

    for seg in segments:
        world_ys = np.array([trajectory[fi]["world_y"] for fi in seg])
        pixel_ys = np.array([trajectory[fi]["pixel_y"] for fi in seg])

        # ── world_y maxima (far-end bounces) ──
        peaks_max, props_max = find_peaks(world_ys, prominence=wy_prominence, distance=distance)
        for j, idx in enumerate(peaks_max):
            fi = seg[idx]
            entry = trajectory[fi]
            wy = entry["world_y"]
            if 2.0 < wy < court_y_max:
                all_candidates.append((fi, entry, float(props_max["prominences"][j]), "far", "wy_max"))

        # ── world_y minima (near-end bounces) ──
        peaks_min, props_min = find_peaks(-world_ys, prominence=wy_prominence, distance=distance)
        for j, idx in enumerate(peaks_min):
            fi = seg[idx]
            entry = trajectory[fi]
            wy = entry["world_y"]
            if 2.0 < wy < court_y_max:
                all_candidates.append((fi, entry, float(props_min["prominences"][j]), "near", "wy_min"))

        # ── pixel_y maxima (near-end bounces — ball close to camera) ──
        peaks_py, props_py = find_peaks(pixel_ys, prominence=py_prominence, distance=distance)
        for j, idx in enumerate(peaks_py):
            fi = seg[idx]
            entry = trajectory[fi]
            if entry["pixel_y"] > 100:  # Only near-end (large pixel_y = close)
                all_candidates.append((fi, entry, float(props_py["prominences"][j]), "near", "py_max"))

    # Sort by frame, dedup with cooldown
    all_candidates.sort(key=lambda x: x[0])

    bounces = []
    last_bounce_fi = -999

    for fi, entry, prom, side, source in all_candidates:
        if fi - last_bounce_fi < cooldown:
            continue

        bounces.append(DetectedBounce(
            frame_index=fi,
            pixel_x=entry["pixel_x"],
            pixel_y=entry["pixel_y"],
            world_x=entry["world_x"],
            world_y=entry["world_y"],
            side=side,
            prominence=prom,
        ))
        last_bounce_fi = fi
        log.debug("Bounce f%d: world_y=%.1f pixel_y=%.0f side=%s source=%s prom=%.2f",
                   fi, entry["world_y"], entry["pixel_y"], side, source, prom)

    return bounces


def detect_bounces_dual_camera(
    traj_cam66: dict[int, dict],
    cam68_detections: dict[str, dict],
    h68: "HomographyTransformer",
    dist_threshold: float = 4.0,
    cooldown: int = 8,
) -> list[DetectedBounce]:
    """Detect bounces using cross-camera world coordinate agreement.

    At Z≈0 (bounce), both cameras' homographies project accurately to the same
    world point, so world coordinate distance is small (~1-3m).
    When airborne, parallax causes projections to diverge (5-50m).

    We find local minima in the cross-camera world distance signal.
    """
    # Compute world distance for all overlapping frames
    frame_dist = {}
    for fi in sorted(traj_cam66.keys()):
        k = str(fi)
        if k not in cam68_detections:
            continue
        e66 = traj_cam66[fi]
        d68 = cam68_detections[k]
        wx68, wy68 = d68["world_x"], d68["world_y"]
        dist = float(np.hypot(e66["world_x"] - wx68, e66["world_y"] - wy68))
        frame_dist[fi] = dist

    if len(frame_dist) < 10:
        log.warning("Only %d overlapping frames, skipping dual-camera bounce detection", len(frame_dist))
        return []

    log.info("Cross-camera overlap: %d frames, dist range=[%.1f, %.1f], median=%.1f",
             len(frame_dist), min(frame_dist.values()), max(frame_dist.values()),
             float(np.median(list(frame_dist.values()))))

    # Build contiguous segments
    sorted_frames = sorted(frame_dist.keys())
    segments = []
    current_seg = [sorted_frames[0]]
    for i in range(1, len(sorted_frames)):
        if sorted_frames[i] - sorted_frames[i - 1] <= 3:
            current_seg.append(sorted_frames[i])
        else:
            if len(current_seg) >= 5:
                segments.append(current_seg)
            current_seg = [sorted_frames[i]]
    if len(current_seg) >= 5:
        segments.append(current_seg)

    bounces = []
    last_bounce_fi = -999

    for seg in segments:
        dists = np.array([frame_dist[fi] for fi in seg])

        # Find local minima in distance signal
        # Invert to find minima as peaks
        peaks, props = find_peaks(-dists, prominence=1.0, distance=5)

        for j, idx in enumerate(peaks):
            fi = seg[idx]
            d = frame_dist[fi]

            if d > dist_threshold:
                continue
            if fi - last_bounce_fi < cooldown:
                continue

            entry = traj_cam66[fi]
            side = "near" if entry["world_y"] < NET_Y else "far"

            bounces.append(DetectedBounce(
                frame_index=fi,
                pixel_x=entry["pixel_x"],
                pixel_y=entry["pixel_y"],
                world_x=entry["world_x"],
                world_y=entry["world_y"],
                side=side,
                prominence=float(props["prominences"][j]),
            ))
            last_bounce_fi = fi
            log.info("Dual-cam bounce f%d: dist=%.2fm world=(%.1f,%.1f) side=%s",
                     fi, d, entry["world_x"], entry["world_y"], side)

    return bounces


# ── Serve Detection ──────────────────────────────────────────────────────────


@dataclass
class DetectedServe:
    frame_index: int
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    side: str


def detect_serves(
    trajectory: dict[int, dict],
    gap_threshold: int = 15,
) -> list[DetectedServe]:
    """Detect serves as the impact point after a gap.

    Strategy: find significant gaps (>gap_threshold frames), then scan forward
    from the gap to find the serve contact point = the last frame in the
    baseline zone before the ball leaves toward the net.

    This distinguishes the serve toss (ball going up) from the actual contact.
    """
    sorted_frames = sorted(trajectory.keys())
    if not sorted_frames:
        return []

    serves = []
    prev_fi = -999

    for idx, fi in enumerate(sorted_frames):
        gap = fi - prev_fi
        if gap > gap_threshold:
            entry = trajectory[fi]
            wy = entry["world_y"]

            # First frame must be in baseline zone
            if wy < BASELINE_NEAR_MAX or wy > BASELINE_FAR_MIN:
                side = "near" if wy < NET_Y else "far"

                # Scan forward to find serve impact:
                # last frame in baseline zone before ball moves toward net
                impact_fi = fi
                impact_entry = entry
                for j in range(idx, min(idx + 40, len(sorted_frames))):
                    fj = sorted_frames[j]
                    if fj - fi > 40:  # Don't look too far
                        break
                    ej = trajectory[fj]
                    wyj = ej["world_y"]

                    if side == "near" and wyj < BASELINE_NEAR_MAX:
                        impact_fi = fj
                        impact_entry = ej
                    elif side == "far" and wyj > BASELINE_FAR_MIN:
                        impact_fi = fj
                        impact_entry = ej
                    elif side == "near" and wyj > BASELINE_NEAR_MAX + 2:
                        break  # Ball has left baseline zone
                    elif side == "far" and wyj < BASELINE_FAR_MIN - 2:
                        break

                serves.append(DetectedServe(
                    frame_index=impact_fi,
                    pixel_x=impact_entry["pixel_x"],
                    pixel_y=impact_entry["pixel_y"],
                    world_x=impact_entry["world_x"],
                    world_y=impact_entry["world_y"],
                    side=side,
                ))
        prev_fi = fi

    return serves


# ── Shot Detection (velocity peaks) ─────────────────────────────────────────


@dataclass
class DetectedShot:
    frame_index: int
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    velocity: float  # px/frame


def detect_shots(
    trajectory: dict[int, dict],
    bounces: list[DetectedBounce] = None,
    serves: list[DetectedServe] = None,
    min_wy_change: float = 0.4,
    cooldown: int = 10,
    bounce_exclusion: int = 5,
    serve_exclusion: int = 8,
    vel_ratio_threshold: float = 2.0,
    min_speed: float = 0.3,
) -> list[DetectedShot]:
    """Detect shots via two complementary methods:

    1. **Direction reversal**: world_y velocity sign flips (ball changes Y direction).
       Catches most baseline rallies.
    2. **Velocity discontinuity**: speed magnitude ratio > threshold (ball accelerates
       or decelerates sharply without reversing direction). Catches volleys, drop shots,
       and returns where ball continues same direction.

    Both methods feed into a unified NMS to pick the strongest candidate per window.
    Excludes frames near detected bounces and serves.
    """
    sorted_frames = sorted(trajectory.keys())
    if len(sorted_frames) < 3:
        return []

    # Build exclusion sets — tighter for velocity method (only exclude exact bounce/serve)
    bounce_excl_reversal = set()
    bounce_excl_velocity = set()
    if bounces:
        for b in bounces:
            for offset in range(-bounce_exclusion, bounce_exclusion + 1):
                bounce_excl_reversal.add(b.frame_index + offset)
            # Velocity method: only ±2 frames from bounce (shots often near bounces)
            for offset in range(-2, 3):
                bounce_excl_velocity.add(b.frame_index + offset)
    if serves:
        for s in serves:
            for offset in range(-serve_exclusion, serve_exclusion + 1):
                bounce_excl_reversal.add(s.frame_index + offset)
                bounce_excl_velocity.add(s.frame_index + offset)

    # Collect all candidates first (from both methods)
    candidates = []  # (frame, entry, pixel_vel, score, method)

    for i in range(1, len(sorted_frames) - 1):
        fi = sorted_frames[i]
        f_prev = sorted_frames[i - 1]
        f_next = sorted_frames[i + 1]

        if fi - f_prev > 5 or f_next - fi > 5:
            continue

        e_prev = trajectory[f_prev]
        e_curr = trajectory[fi]
        e_next = trajectory[f_next]

        # World velocity before and after
        dt_prev = max(fi - f_prev, 1)
        dt_next = max(f_next - fi, 1)
        vy_before = (e_curr["world_y"] - e_prev["world_y"]) / dt_prev
        vy_after = (e_next["world_y"] - e_curr["world_y"]) / dt_next
        vx_before = (e_curr["world_x"] - e_prev["world_x"]) / dt_prev
        vx_after = (e_next["world_x"] - e_curr["world_x"]) / dt_next

        vel = float(np.hypot(
            e_curr["pixel_x"] - e_prev["pixel_x"],
            e_curr["pixel_y"] - e_prev["pixel_y"],
        )) / dt_prev

        # Method 1: Direction reversal in world_y (uses wider exclusion zone)
        if vy_before * vy_after < 0:
            if fi not in bounce_excl_reversal:
                wy_change = abs(vy_after - vy_before)
                if wy_change > min_wy_change:
                    candidates.append((fi, e_curr, vel, wy_change, "reversal"))

        # Method 2: Velocity magnitude discontinuity (tighter exclusion, catches shots near bounces)
        elif fi not in bounce_excl_velocity:
            speed_before = np.hypot(vx_before, vy_before)
            speed_after = np.hypot(vx_after, vy_after)
            min_spd = min(speed_before, speed_after)

            if min_spd >= min_speed:
                ratio = max(speed_before, speed_after) / min_spd
                if ratio >= vel_ratio_threshold:
                    speed_change = abs(speed_after - speed_before)
                    candidates.append((fi, e_curr, vel, speed_change, "velocity"))

    # NMS: within each cooldown window, keep the strongest score
    candidates.sort(key=lambda c: -c[3])  # Sort by score descending
    shots = []
    used_frames = set()

    for fi, e_curr, vel, score, method in candidates:
        if any(abs(fi - uf) < cooldown for uf in used_frames):
            continue
        shots.append(DetectedShot(
            frame_index=fi,
            pixel_x=e_curr["pixel_x"],
            pixel_y=e_curr["pixel_y"],
            world_x=e_curr["world_x"],
            world_y=e_curr["world_y"],
            velocity=vel,
        ))
        used_frames.add(fi)
        log.info("  Shot f%d: method=%s, score=%.2f", fi, method, score)

    shots.sort(key=lambda s: s.frame_index)
    return shots


# ── Rally State Machine (simplified for single-cam) ─────────────────────────


@dataclass
class RallyResult:
    rally_id: int
    start_frame: int
    end_frame: int
    start_type: str  # "serve" or "resume"
    end_reason: str  # "timeout", "out", "double_bounce", "ongoing"
    serve_side: str
    bounces: list
    strokes: int


def detect_rallies(
    trajectory: dict[int, dict],
    bounces: list[DetectedBounce],
    serves: list[DetectedServe],
    timeout_frames: int = 75,  # 3 seconds at 25fps
) -> list[RallyResult]:
    """Simple rally state machine using world coordinates.

    States: IDLE → SERVING → RALLY → IDLE
    """
    sorted_frames = sorted(trajectory.keys())
    if not sorted_frames:
        return []

    # Pre-index bounces and serves by frame
    bounce_by_frame = {b.frame_index: b for b in bounces}
    serve_frames = {s.frame_index for s in serves}

    rallies = []
    state = "idle"
    rally_id = 0
    rally_start = 0
    serve_side = ""
    rally_bounces = []
    last_bounce_side = ""
    bounce_count_since_cross = 0
    prev_y = None
    last_fi = -999
    stroke_count = 0

    for fi in sorted_frames:
        entry = trajectory[fi]
        wy = entry["world_y"]
        side = "near" if wy < NET_Y else "far"

        # Check timeout
        if state in ("serving", "rally", "pre_serve") and fi - last_fi > timeout_frames:
            if state == "rally":
                rallies.append(RallyResult(
                    rally_id=rally_id,
                    start_frame=rally_start,
                    end_frame=last_fi,
                    start_type="serve" if serve_side else "resume",
                    end_reason="timeout",
                    serve_side=serve_side,
                    bounces=rally_bounces,
                    strokes=stroke_count,
                ))
            state = "idle"

        # State transitions
        if state == "idle":
            if fi in serve_frames:
                rally_id += 1
                state = "serving"
                rally_start = fi
                s = next(s for s in serves if s.frame_index == fi)
                serve_side = s.side
                rally_bounces = []
                last_bounce_side = ""
                bounce_count_since_cross = 0
                stroke_count = 0
            elif fi - last_fi > 15:
                # Activity after gap — check if there's a serve nearby (within 40 frames)
                nearby_serve = None
                for s in serves:
                    if fi <= s.frame_index <= fi + 40:
                        nearby_serve = s
                        break
                if nearby_serve:
                    # Start as pre-serve (toss phase), will transition to serving
                    rally_id += 1
                    state = "pre_serve"
                    rally_start = fi
                    serve_side = nearby_serve.side
                    rally_bounces = []
                    last_bounce_side = ""
                    bounce_count_since_cross = 0
                    stroke_count = 0
                else:
                    rally_id += 1
                    state = "rally"
                    rally_start = fi
                    serve_side = ""
                    rally_bounces = []
                    last_bounce_side = ""
                    bounce_count_since_cross = 0
                    stroke_count = 0

        if state == "pre_serve" and fi in serve_frames:
            state = "serving"
            rally_start = fi  # Update start to actual serve impact

        # Net crossing (skip if gap too large — ball didn't really cross)
        if prev_y is not None and fi - last_fi <= 5:
            crossed = (prev_y < NET_Y <= wy) or (prev_y >= NET_Y > wy)
            if crossed:
                if state in ("serving", "pre_serve"):
                    state = "rally"
                if state == "rally":
                    stroke_count += 1
                    bounce_count_since_cross = 0

        # Handle bounce
        if fi in bounce_by_frame and state == "rally":
            b = bounce_by_frame[fi]
            rally_bounces.append(b)

            # Check out of court
            in_court = 0 <= b.world_x <= COURT_X and 0 <= b.world_y <= COURT_Y
            if not in_court:
                rallies.append(RallyResult(
                    rally_id=rally_id,
                    start_frame=rally_start,
                    end_frame=fi,
                    start_type="serve" if serve_side else "resume",
                    end_reason="out",
                    serve_side=serve_side,
                    bounces=rally_bounces,
                    strokes=stroke_count,
                ))
                state = "idle"

            # Check double bounce
            elif b.side == last_bounce_side:
                bounce_count_since_cross += 1
                if bounce_count_since_cross >= 2:
                    rallies.append(RallyResult(
                        rally_id=rally_id,
                        start_frame=rally_start,
                        end_frame=fi,
                        start_type="serve" if serve_side else "resume",
                        end_reason="double_bounce",
                        serve_side=serve_side,
                        bounces=rally_bounces,
                        strokes=stroke_count,
                    ))
                    state = "idle"
            else:
                bounce_count_since_cross = 1

            last_bounce_side = b.side

        prev_y = wy
        last_fi = fi

    # Close any ongoing rally
    if state == "rally":
        rallies.append(RallyResult(
            rally_id=rally_id,
            start_frame=rally_start,
            end_frame=last_fi,
            start_type="serve" if serve_side else "resume",
            end_reason="ongoing",
            serve_side=serve_side,
            bounces=rally_bounces,
            strokes=stroke_count,
        ))

    return rallies


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_events(
    detected_bounces: list[DetectedBounce],
    detected_serves: list[DetectedServe],
    detected_shots: list[DetectedShot],
    gt: dict,
    frame_tolerance: int = 3,
) -> dict:
    """Compare detected events against GT labels.

    Matching: detected event within ±frame_tolerance of GT event.
    """
    results = {}

    # Bounce evaluation
    gt_bounces = gt["bounces"]
    det_bounce_frames = [b.frame_index for b in detected_bounces]
    bounce_tp = 0
    bounce_matched_gt = []
    bounce_matched_det = []
    for gt_fi, gx, gy in gt_bounces:
        matched = False
        for di, dfi in enumerate(det_bounce_frames):
            if abs(dfi - gt_fi) <= frame_tolerance and di not in bounce_matched_det:
                bounce_tp += 1
                bounce_matched_gt.append(gt_fi)
                bounce_matched_det.append(di)
                matched = True
                break
        if not matched:
            bounce_matched_gt.append(None)

    results["bounce"] = {
        "gt_total": len(gt_bounces),
        "detected_total": len(detected_bounces),
        "tp": bounce_tp,
        "fn": len(gt_bounces) - bounce_tp,
        "fp": len(detected_bounces) - bounce_tp,
        "recall": bounce_tp / max(len(gt_bounces), 1),
        "gt_frames": [fi for fi, _, _ in gt_bounces],
        "det_frames": det_bounce_frames,
        "missed_gt": [fi for fi, _, _ in gt_bounces if fi not in [
            gt_bounces[i][0] for i in range(len(gt_bounces)) if bounce_matched_gt[i] is not None
        ]],
    }

    # Serve evaluation
    gt_serves = gt["serves"]
    det_serve_frames = [s.frame_index for s in detected_serves]
    serve_tp = 0
    for gt_fi, _, _ in gt_serves:
        for dfi in det_serve_frames:
            if abs(dfi - gt_fi) <= frame_tolerance:
                serve_tp += 1
                break

    results["serve"] = {
        "gt_total": len(gt_serves),
        "detected_total": len(detected_serves),
        "tp": serve_tp,
        "fn": len(gt_serves) - serve_tp,
        "fp": len(detected_serves) - serve_tp,
        "recall": serve_tp / max(len(gt_serves), 1),
        "gt_frames": [fi for fi, _, _ in gt_serves],
        "det_frames": det_serve_frames,
    }

    # Shot evaluation
    gt_shots = gt["shots"]
    det_shot_frames = [s.frame_index for s in detected_shots]
    shot_tp = 0
    for gt_fi, _, _ in gt_shots:
        for dfi in det_shot_frames:
            if abs(dfi - gt_fi) <= frame_tolerance:
                shot_tp += 1
                break

    results["shot"] = {
        "gt_total": len(gt_shots),
        "detected_total": len(detected_shots),
        "tp": shot_tp,
        "fn": len(gt_shots) - shot_tp,
        "fp": len(detected_shots) - shot_tp,
        "recall": shot_tp / max(len(gt_shots), 1),
        "gt_frames": [fi for fi, _, _ in gt_shots],
        "det_frames": det_shot_frames,
    }

    return results


# ── Video Rendering ──────────────────────────────────────────────────────────


def draw_minimap(
    frame: np.ndarray,
    fi: int,
    trajectory: dict[int, dict],
    cam68_dets: Optional[dict],
    detected_bounces: list[DetectedBounce],
    detected_serves: list[DetectedServe],
    detected_shots: list[DetectedShot],
    rally_state: dict,
    cross_cam_dist: Optional[dict],
    map_x: int, map_y: int, map_w: int, map_h: int,
) -> None:
    """Draw a 2D court mini-map overlay on the frame.

    Court coords: x=[0, 8.23], y=[0, 23.77].
    We extend slightly to show out-of-court balls.
    """
    # Court extents with margin
    cx_min, cx_max = -1.0, 9.23
    cy_min, cy_max = -2.0, 26.0
    cx_range = cx_max - cx_min
    cy_range = cy_max - cy_min

    def world_to_map(wx, wy):
        mx = int(map_x + (wx - cx_min) / cx_range * map_w)
        my = int(map_y + map_h - (wy - cy_min) / cy_range * map_h)  # flip Y
        return mx, my

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (map_x, map_y), (map_x + map_w, map_y + map_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Draw court lines (green)
    court_color = (0, 120, 0)
    # Outer boundary
    pts = [world_to_map(0, 0), world_to_map(COURT_X, 0),
           world_to_map(COURT_X, COURT_Y), world_to_map(0, COURT_Y)]
    for i in range(4):
        cv2.line(frame, pts[i], pts[(i + 1) % 4], court_color, 1)

    # Net line
    n1 = world_to_map(0, NET_Y)
    n2 = world_to_map(COURT_X, NET_Y)
    cv2.line(frame, n1, n2, (255, 255, 255), 2)

    # Service lines (y=5.485 and y=18.285 approx)
    for sy in [5.485, 18.285]:
        s1 = world_to_map(0, sy)
        s2 = world_to_map(COURT_X, sy)
        cv2.line(frame, s1, s2, court_color, 1)

    # Center service line
    cs1 = world_to_map(COURT_X / 2, 5.485)
    cs2 = world_to_map(COURT_X / 2, 18.285)
    cv2.line(frame, cs1, cs2, court_color, 1)

    # Doubles sidelines (1.37m from each side)
    for sx in [1.37, COURT_X - 1.37]:
        s1 = world_to_map(sx, 0)
        s2 = world_to_map(sx, COURT_Y)
        cv2.line(frame, s1, s2, (0, 80, 0), 1)

    # Bounce markers (persist for 30 frames)
    for b in detected_bounces:
        if 0 <= fi - b.frame_index < 30:
            bx, by = world_to_map(b.world_x, b.world_y)
            fade = max(0.3, 1.0 - (fi - b.frame_index) / 30.0)
            color = (0, int(255 * fade), int(255 * fade))  # cyan fade
            cv2.drawMarker(frame, (bx, by), color, cv2.MARKER_TRIANGLE_DOWN, 8, 2)
            if fi - b.frame_index < 10:
                cv2.putText(frame, "B", (bx + 5, by - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Serve markers (persist for 20 frames)
    for s in detected_serves:
        if 0 <= fi - s.frame_index < 20:
            sx, sy = world_to_map(s.world_x, s.world_y)
            cv2.drawMarker(frame, (sx, sy), (0, 255, 255), cv2.MARKER_STAR, 10, 1)
            cv2.putText(frame, "S", (sx + 5, sy - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Shot markers (persist for 20 frames)
    for sh in detected_shots:
        if 0 <= fi - sh.frame_index < 20:
            sx, sy = world_to_map(sh.world_x, sh.world_y)
            cv2.drawMarker(frame, (sx, sy), (0, 255, 0), cv2.MARKER_DIAMOND, 8, 2)

    # Trail (last 30 frames)
    trail_frames = []
    for offset in range(30, 0, -1):
        prev_fi = fi - offset
        if prev_fi in trajectory:
            e = trajectory[prev_fi]
            trail_frames.append((world_to_map(e["world_x"], e["world_y"]), offset))

    for i in range(1, len(trail_frames)):
        p1, off1 = trail_frames[i - 1]
        p2, off2 = trail_frames[i]
        alpha = 1.0 - off2 / 30.0
        color = (0, int(200 * alpha), int(255 * alpha))
        cv2.line(frame, p1, p2, color, 1)

    # Current ball position
    entry = trajectory.get(fi)
    if entry:
        bx, by = world_to_map(entry["world_x"], entry["world_y"])
        cv2.circle(frame, (bx, by), 5, (0, 255, 0), -1)
        cv2.circle(frame, (bx, by), 5, (255, 255, 255), 1)

    # cam68 position (if available)
    if cam68_dets and str(fi) in cam68_dets:
        d68 = cam68_dets[str(fi)]
        cx, cy_pt = world_to_map(d68["world_x"], d68["world_y"])
        cv2.circle(frame, (cx, cy_pt), 4, (255, 128, 0), -1)  # orange
        # Draw line between cam66 and cam68 world positions
        if entry:
            bx, by = world_to_map(entry["world_x"], entry["world_y"])
            cv2.line(frame, (bx, by), (cx, cy_pt), (100, 100, 100), 1)

    # Cross-camera distance bar (bottom of minimap)
    if cross_cam_dist is not None and fi in cross_cam_dist:
        dist = cross_cam_dist[fi]
        bar_y = map_y + map_h + 5
        bar_max = 15.0  # max distance for full bar
        bar_len = min(int(dist / bar_max * map_w), map_w)
        bar_color = (0, 255, 0) if dist < 4.0 else (0, 200, 255) if dist < 8.0 else (0, 0, 255)
        cv2.rectangle(frame, (map_x, bar_y), (map_x + bar_len, bar_y + 12), bar_color, -1)
        cv2.putText(frame, f"cam dist: {dist:.1f}m", (map_x + 2, bar_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        # Threshold line
        thresh_x = map_x + int(4.0 / bar_max * map_w)
        cv2.line(frame, (thresh_x, bar_y), (thresh_x, bar_y + 12), (255, 255, 255), 1)

    # Rally state label on minimap
    rally = rally_state.get(fi)
    if rally:
        state_color = (0, 200, 0)
        label = f"R{rally.rally_id}"
        if rally.end_frame == fi:
            label += f" {rally.end_reason}"
            state_color = (0, 0, 255)
        cv2.putText(frame, label, (map_x + 2, map_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1)


def _annotate_camera_frame(
    frame: np.ndarray,
    fi: int,
    trajectory: dict[int, dict],
    gt: dict,
    gt_bounce_set: set,
    gt_serve_set: set,
    gt_shot_set: set,
    det_bounce_set: set,
    det_serve_set: set,
    det_shot_set: set,
    trail: list,
    trail_len: int,
    cam_label: str = "cam66",
) -> list:
    """Draw annotations (trail, markers, events) on a single camera frame."""
    entry = trajectory.get(fi)

    # Trail
    if entry:
        trail.append((int(entry["pixel_x"]), int(entry["pixel_y"])))
    if len(trail) > trail_len:
        trail[:] = trail[-trail_len:]
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        cv2.line(frame, trail[i - 1], trail[i], (0, 200, 255), max(1, int(alpha * 2)))

    # GT position
    gt_pos = gt["positions"].get(fi)
    if gt_pos:
        gx, gy = int(gt_pos[0]), int(gt_pos[1])
        cv2.circle(frame, (gx, gy), 6, (255, 255, 255), 2)

    # Detected position
    if entry:
        ex, ey = int(entry["pixel_x"]), int(entry["pixel_y"])
        cv2.circle(frame, (ex, ey), 6, (0, 255, 0), -1)

    # Event markers - GT (white)
    if fi in gt_bounce_set and gt_pos:
        gx, gy = int(gt_pos[0]), int(gt_pos[1])
        cv2.putText(frame, "GT:BOUNCE", (gx + 12, gy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if fi in gt_serve_set and gt_pos:
        gx, gy = int(gt_pos[0]), int(gt_pos[1])
        cv2.putText(frame, "GT:SERVE", (gx + 12, gy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if fi in gt_shot_set and gt_pos:
        gx, gy = int(gt_pos[0]), int(gt_pos[1])
        cv2.putText(frame, "GT:SHOT", (gx + 12, gy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Event markers - Detected (color-coded)
    if fi in det_bounce_set and entry:
        cv2.putText(frame, "BOUNCE", (int(entry["pixel_x"]) - 40, int(entry["pixel_y"]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.drawMarker(frame, (int(entry["pixel_x"]), int(entry["pixel_y"])),
                       (0, 255, 255), cv2.MARKER_TRIANGLE_DOWN, 15, 2)
    if fi in det_serve_set and entry:
        cv2.putText(frame, "SERVE", (int(entry["pixel_x"]) - 30, int(entry["pixel_y"]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if fi in det_shot_set and entry:
        cv2.putText(frame, "SHOT", (int(entry["pixel_x"]) - 25, int(entry["pixel_y"]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Camera label
    cv2.putText(frame, cam_label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    return trail


def render_video(
    video_path: str,
    trajectory: dict[int, dict],
    gt: dict,
    detected_bounces: list[DetectedBounce],
    detected_serves: list[DetectedServe],
    detected_shots: list[DetectedShot],
    rallies: list[RallyResult],
    output_path: Path,
    max_frames: int,
    cam68_dets: Optional[dict] = None,
    cam68_video_path: Optional[str] = None,
) -> None:
    """Render dual-camera evaluation video with trajectory, events, rally state, and 2D mini-map.

    Layout (1920x1080):
        ┌──────────────┬──────────────┐
        │  cam66 (960×540)  │  cam68 (960×540)  │
        ├──────────────┴──────────────┤
        │  minimap (center)  │  info + rally bar  │
        └─────────────────────────────┘
    """
    cap66 = cv2.VideoCapture(video_path)
    fps = cap66.get(cv2.CAP_PROP_FPS) or 25.0

    # Dual-camera mode
    dual_mode = cam68_video_path is not None
    cap68 = cv2.VideoCapture(cam68_video_path) if dual_mode else None

    # Output canvas size
    out_w, out_h = 1920, 1080
    half_w, half_h = out_w // 2, out_h // 2  # 960×540

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    gt_bounce_set = {fi for fi, _, _ in gt["bounces"]}
    gt_serve_set = {fi for fi, _, _ in gt["serves"]}
    gt_shot_set = {fi for fi, _, _ in gt["shots"]}

    det_bounce_set = {b.frame_index for b in detected_bounces}
    det_serve_set = {s.frame_index for s in detected_serves}
    det_shot_set = {s.frame_index for s in detected_shots}

    # Build rally state lookup
    rally_state = {}
    for r in rallies:
        for fi in range(r.start_frame, r.end_frame + 1):
            rally_state[fi] = r

    # Precompute cross-camera distances
    cross_cam_dist = None
    if cam68_dets:
        cross_cam_dist = {}
        for fi in trajectory:
            k = str(fi)
            if k in cam68_dets:
                e66 = trajectory[fi]
                d68 = cam68_dets[k]
                cross_cam_dist[fi] = float(np.hypot(
                    e66["world_x"] - d68["world_x"],
                    e66["world_y"] - d68["world_y"],
                ))

    # Build cam68 pixel trajectory for annotations
    cam68_trajectory = {}
    if cam68_dets:
        for k, d in cam68_dets.items():
            fi_k = int(k)
            if fi_k < max_frames:
                cam68_trajectory[fi_k] = d

    trail66 = []
    trail68 = []
    trail_len = 20

    # Mini-map position: center-bottom area
    map_w, map_h = 200, 440
    map_x = (out_w - map_w) // 2
    map_y = half_h + 10

    for fi in range(max_frames):
        ret66, frame66 = cap66.read()
        if not ret66:
            break

        # Read cam68 frame
        frame68 = None
        if cap68:
            ret68, frame68 = cap68.read()
            if not ret68:
                frame68 = None

        # Annotate cam66
        trail66 = _annotate_camera_frame(
            frame66, fi, trajectory, gt,
            gt_bounce_set, gt_serve_set, gt_shot_set,
            det_bounce_set, det_serve_set, det_shot_set,
            trail66, trail_len, "cam66 (GT)",
        )

        # Annotate cam68 with detection markers
        if frame68 is not None and cam68_dets:
            d68 = cam68_dets.get(str(fi))
            if d68:
                px68, py68 = int(d68["pixel_x"]), int(d68["pixel_y"])
                cv2.circle(frame68, (px68, py68), 6, (0, 165, 255), -1)
                # Trail for cam68
                trail68.append((px68, py68))
                if len(trail68) > trail_len:
                    trail68[:] = trail68[-trail_len:]
                for i in range(1, len(trail68)):
                    alpha = i / len(trail68)
                    cv2.line(frame68, trail68[i - 1], trail68[i],
                             (0, 165, 255), max(1, int(alpha * 2)))
            cv2.putText(frame68, "cam68 (Motion+YOLO)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Build output canvas
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # Top-left: cam66 scaled to half
        small66 = cv2.resize(frame66, (half_w, half_h))
        canvas[0:half_h, 0:half_w] = small66

        # Top-right: cam68 scaled to half (or black if no cam68)
        if frame68 is not None:
            small68 = cv2.resize(frame68, (half_w, half_h))
            canvas[0:half_h, half_w:out_w] = small68
        else:
            cv2.putText(canvas, "cam68: no video", (half_w + 100, half_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

        # Divider lines
        cv2.line(canvas, (half_w, 0), (half_w, half_h), (80, 80, 80), 2)
        cv2.line(canvas, (0, half_h), (out_w, half_h), (80, 80, 80), 2)

        # ── 2D Mini-map (center bottom) ──
        draw_minimap(
            canvas, fi, trajectory, cam68_dets,
            detected_bounces, detected_serves, detected_shots,
            rally_state, cross_cam_dist,
            map_x, map_y, map_w, map_h,
        )

        # ── Info panel (left of minimap) ──
        info_x = 30
        info_y = half_h + 30

        # Frame counter
        cv2.putText(canvas, f"Frame: {fi}", (info_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # World coords
        entry = trajectory.get(fi)
        if entry:
            cv2.putText(canvas, f"cam66 world: ({entry['world_x']:.1f}, {entry['world_y']:.1f})",
                        (info_x, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        d68_entry = cam68_dets.get(str(fi)) if cam68_dets else None
        if d68_entry:
            cv2.putText(canvas, f"cam68 world: ({d68_entry['world_x']:.1f}, {d68_entry['world_y']:.1f})",
                        (info_x, info_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        # Cross-camera distance
        if cross_cam_dist and fi in cross_cam_dist:
            dist = cross_cam_dist[fi]
            dist_color = (0, 255, 0) if dist < 4.0 else (0, 255, 255) if dist < 8.0 else (0, 0, 255)
            cv2.putText(canvas, f"Cross-cam dist: {dist:.1f}m",
                        (info_x, info_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dist_color, 1)

        # Detection counts (left bottom)
        cv2.putText(canvas, f"Bounces: {len(detected_bounces)}  Serves: {len(detected_serves)}  Shots: {len(detected_shots)}",
                    (info_x, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── Rally state bar (right of minimap) ──
        rally_x = map_x + map_w + 40
        rally_y = half_h + 30

        rally = rally_state.get(fi)
        if rally:
            state_color = (0, 200, 0)
            cv2.putText(canvas, f"Rally {rally.rally_id}", (rally_x, rally_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(canvas, f"Type: {rally.start_type}", (rally_x, rally_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1)
            cv2.putText(canvas, f"Strokes: {rally.strokes}", (rally_x, rally_y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1)
            cv2.putText(canvas, f"Bounces: {len(rally.bounces)}", (rally_x, rally_y + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 1)
            if rally.end_frame == fi:
                cv2.putText(canvas, f"END: {rally.end_reason}", (rally_x, rally_y + 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(canvas, "IDLE", (rally_x, rally_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # ── Event timeline bar (very bottom) ──
        bar_y = out_h - 30
        bar_h = 20
        cv2.rectangle(canvas, (0, bar_y), (out_w, out_h), (30, 30, 30), -1)
        # Progress
        progress = fi / max(max_frames - 1, 1)
        cv2.rectangle(canvas, (0, bar_y), (int(out_w * progress), out_h), (60, 60, 60), -1)
        # Mark events on timeline
        for b in detected_bounces:
            bx = int(b.frame_index / max_frames * out_w)
            cv2.line(canvas, (bx, bar_y), (bx, out_h), (0, 255, 255), 2)
        for s in detected_serves:
            sx = int(s.frame_index / max_frames * out_w)
            cv2.line(canvas, (sx, bar_y), (sx, out_h), (0, 255, 255), 2)
        for s in detected_shots:
            sx = int(s.frame_index / max_frames * out_w)
            cv2.line(canvas, (sx, bar_y), (sx, out_h), (0, 255, 0), 2)
        # Playhead
        px = int(fi / max_frames * out_w)
        cv2.line(canvas, (px, bar_y), (px, out_h), (255, 255, 255), 2)

        writer.write(canvas)

    cap66.release()
    if cap68:
        cap68.release()
    writer.release()
    log.info("Video saved: %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate rally detection with GT trajectory")
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--render", action="store_true", help="Render evaluation video")
    parser.add_argument("--bounce-prominence", type=float, default=0.8)
    parser.add_argument("--bounce-distance", type=int, default=5)
    parser.add_argument("--bounce-cooldown", type=int, default=8)
    parser.add_argument("--shot-velocity", type=float, default=0.4,
                        help="Min world_y velocity change for shot detection (m/frame)")
    parser.add_argument("--vel-ratio", type=float, default=2.0,
                        help="Min velocity ratio for discontinuity-based shot detection")
    parser.add_argument("--frame-tolerance", type=int, default=5)
    parser.add_argument("--dual-cam", action="store_true", default=True,
                        help="Use dual-camera bounce detection (default: True)")
    parser.add_argument("--single-cam", dest="dual_cam", action="store_false",
                        help="Single-camera only")
    parser.add_argument("--cam-dist-threshold", type=float, default=4.0,
                        help="Cross-camera world distance threshold for bounce")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load GT
    log.info("Loading GT from %s ...", CAM66_LABELS)
    gt = load_match_ball_gt(CAM66_LABELS, max_frame=args.max_frames)
    log.info("GT: %d match_ball, %d bounces, %d serves, %d shots",
             len(gt["positions"]), len(gt["bounces"]), len(gt["serves"]), len(gt["shots"]))

    # Convert GT pixel positions to world coords via homography
    from app.pipeline.homography import HomographyTransformer
    h66 = HomographyTransformer(HOMOGRAPHY_PATH, "cam66")

    trajectory: dict[int, dict] = {}
    for fi, (px, py) in gt["positions"].items():
        wx, wy = h66.pixel_to_world(px, py)
        trajectory[fi] = {
            "pixel_x": px,
            "pixel_y": py,
            "world_x": wx,
            "world_y": wy,
        }

    log.info("Trajectory: %d frames, world_y range=[%.1f, %.1f]",
             len(trajectory),
             min(e["world_y"] for e in trajectory.values()),
             max(e["world_y"] for e in trajectory.values()))

    # ── Load cam68 detections (for dual-camera) ──
    cam68_dets = None
    h68 = None
    if args.dual_cam:
        cam68_path = Path(CAM68_DETECTIONS)
        if cam68_path.exists():
            with open(cam68_path) as f:
                cam68_dets = json.load(f)
            h68 = HomographyTransformer(HOMOGRAPHY_PATH, "cam68")
            cam68_in_range = sum(1 for k in cam68_dets if int(k) <= args.max_frames)
            log.info("Loaded cam68 detections: %d total, %d in range", len(cam68_dets), cam68_in_range)
        else:
            log.warning("cam68 detections not found at %s, falling back to single-cam", cam68_path)

    # ── Detect Events ──
    if cam68_dets is not None:
        log.info("=== Bounce Detection (dual-camera cross-distance) ===")
        detected_bounces = detect_bounces_dual_camera(
            trajectory,
            cam68_dets,
            h68,
            dist_threshold=args.cam_dist_threshold,
            cooldown=args.bounce_cooldown,
        )
        log.info("Dual-cam detected %d bounces", len(detected_bounces))
    else:
        log.info("=== Bounce Detection (single-cam combined) ===")
        detected_bounces = detect_bounces_combined(
            trajectory,
            wy_prominence=args.bounce_prominence,
            distance=args.bounce_distance,
            cooldown=args.bounce_cooldown,
        )
    log.info("Detected %d bounces total", len(detected_bounces))

    log.info("=== Serve Detection ===")
    detected_serves = detect_serves(trajectory, gap_threshold=15)
    log.info("Detected %d serves", len(detected_serves))

    log.info("=== Shot Detection (reversal + velocity discontinuity) ===")
    detected_shots = detect_shots(trajectory, bounces=detected_bounces,
                                   serves=detected_serves,
                                   min_wy_change=args.shot_velocity, cooldown=15,
                                   bounce_exclusion=8, serve_exclusion=15,
                                   vel_ratio_threshold=args.vel_ratio, min_speed=0.3)
    log.info("Detected %d shots", len(detected_shots))

    log.info("=== Rally Detection ===")
    rallies = detect_rallies(trajectory, detected_bounces, detected_serves)
    log.info("Detected %d rallies", len(rallies))

    # ── Evaluate ──
    eval_results = evaluate_events(
        detected_bounces, detected_serves, detected_shots, gt,
        frame_tolerance=args.frame_tolerance,
    )

    # ── Print Results ──
    print("\n" + "=" * 65)
    print("  RALLY ANALYSIS EVALUATION (GT trajectory, cam66)")
    print("=" * 65)

    print(f"\n  GT trajectory: {len(gt['positions'])} frames")
    print(f"  Frame tolerance: ±{args.frame_tolerance} frames")

    print(f"\n  ── Bounce Detection ──")
    b = eval_results["bounce"]
    print(f"  GT:       {b['gt_total']} bounces  frames: {b['gt_frames']}")
    print(f"  Detected: {b['detected_total']} bounces  frames: {b['det_frames']}")
    print(f"  Recall:   {b['tp']}/{b['gt_total']} ({100*b['recall']:.0f}%)")
    print(f"  FP:       {b['fp']}")
    if b['missed_gt']:
        print(f"  Missed:   {b['missed_gt']}")

    print(f"\n  ── Serve Detection ──")
    s = eval_results["serve"]
    print(f"  GT:       {s['gt_total']} serves  frames: {s['gt_frames']}")
    print(f"  Detected: {s['detected_total']} serves  frames: {s['det_frames']}")
    print(f"  Recall:   {s['tp']}/{s['gt_total']} ({100*s['recall']:.0f}%)")
    print(f"  FP:       {s['fp']}")

    print(f"\n  ── Shot Detection ──")
    sh = eval_results["shot"]
    print(f"  GT:       {sh['gt_total']} shots  frames: {sh['gt_frames']}")
    print(f"  Detected: {sh['detected_total']} shots  frames: {sh['det_frames']}")
    print(f"  Recall:   {sh['tp']}/{sh['gt_total']} ({100*sh['recall']:.0f}%)")
    print(f"  FP:       {sh['fp']}")

    print(f"\n  ── Rallies ──")
    for r in rallies:
        bounce_frames = [b.frame_index for b in r.bounces]
        print(f"  Rally {r.rally_id}: f{r.start_frame}-{r.end_frame} "
              f"({r.start_type}, {r.serve_side}) "
              f"strokes={r.strokes} bounces={len(r.bounces)}{bounce_frames} "
              f"end={r.end_reason}")

    # ── Trajectory Details (for debugging) ──
    print(f"\n  ── Trajectory world_y at GT events ──")
    for fi, px, py in gt["bounces"]:
        e = trajectory.get(fi)
        if e:
            print(f"  Bounce f{fi}: pixel_y={py:.0f} world=({e['world_x']:.1f}, {e['world_y']:.1f})")
    for fi, px, py in gt["serves"]:
        e = trajectory.get(fi)
        if e:
            print(f"  Serve  f{fi}: pixel_y={py:.0f} world=({e['world_x']:.1f}, {e['world_y']:.1f})")
    for fi, px, py in gt["shots"]:
        e = trajectory.get(fi)
        if e:
            print(f"  Shot   f{fi}: pixel_y={py:.0f} world=({e['world_x']:.1f}, {e['world_y']:.1f})")

    # Save results
    with open(OUT_DIR / "rally_eval_results.json", "w") as f:
        json.dump({
            "events": eval_results,
            "rallies": [
                {
                    "rally_id": r.rally_id,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "start_type": r.start_type,
                    "end_reason": r.end_reason,
                    "serve_side": r.serve_side,
                    "strokes": r.strokes,
                    "bounce_frames": [b.frame_index for b in r.bounces],
                }
                for r in rallies
            ],
            "params": {
                "bounce_prominence": args.bounce_prominence,
                "bounce_distance": args.bounce_distance,
                "bounce_cooldown": args.bounce_cooldown,
                "shot_velocity": args.shot_velocity,
                "frame_tolerance": args.frame_tolerance,
            },
        }, f, indent=2, default=str)
    log.info("Results saved to %s", OUT_DIR / "rally_eval_results.json")

    # Render video
    if args.render:
        video_path = OUT_DIR / "rally_gt_eval.mp4"
        log.info("Rendering video...")
        render_video(
            CAM66_VIDEO, trajectory, gt,
            detected_bounces, detected_serves, detected_shots, rallies,
            video_path, args.max_frames,
            cam68_dets=cam68_dets,
            cam68_video_path=CAM68_VIDEO if args.dual_cam else None,
        )


if __name__ == "__main__":
    main()
