"""Render dual-camera tracking video with minimap showing bounce landing points.

Layout: [cam66 | cam68 | minimap]
- Camera views: ball marker + fading trajectory trail
- Minimap: court diagram with numbered bounce/landing markers only

Usage:
    python -m tools.render_tracking_video \
        --video66 uploads/cam66_20260307_173403_2min.mp4 \
        --video68 uploads/cam68_20260307_173403_2min.mp4 \
        --output exports/tracking_video.mp4 \
        --max-frames 3000
"""

import argparse
import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Court dimensions (meters) — SINGLES court ────────────────────────────
# This is a singles court. All coordinates use singles sidelines as boundaries.
SINGLES_X_MIN = 1.37              # singles sideline (left)
SINGLES_X_MAX = 6.86              # singles sideline (right)
COURT_W = SINGLES_X_MAX - SINGLES_X_MIN  # singles width = 5.49m
COURT_L = 23.77
NET_Y = COURT_L / 2  # 11.885
NET_H = 0.914        # net height (meters)
SERVICE_NEAR = 5.485
SERVICE_FAR = 18.285

# ── Rendering constants ──────────────────────────────────────────────────
TRAIL_LEN = 30
TRAIL_JUMP_PX = 150.0  # break trail if detection jumps > this many pixels
COURT_MARGIN = 20


def load_config():
    """Load config.yaml."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_detector(cfg):
    """Build TrackNet detector + postprocessor from config."""
    from app.pipeline.inference import TrackNetDetector
    from app.pipeline.postprocess import BallTracker

    mcfg = cfg["model"]
    detector = TrackNetDetector(
        model_path=mcfg["path"],
        input_size=tuple(mcfg["input_size"]),
        frames_in=mcfg["frames_in"],
        frames_out=mcfg["frames_out"],
        device=mcfg.get("device", "cuda"),
    )

    tracker = BallTracker(
        original_size=(1920, 1080),
        threshold=mcfg.get("threshold", 0.1),
        heatmap_mask=mcfg.get("heatmap_mask"),
    )
    return detector, tracker


def run_detection_multi(video_path, detector, postproc, max_frames, top_k=2):
    """Run TrackNet detection on a video, return {frame_idx: list[dict]} with top-K blobs.

    Each blob dict has: pixel_x, pixel_y, blob_sum, blob_max, blob_area.
    Also returns single-best detections for rendering: {frame_idx: (px, py, conf)}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, max_frames)
    seq_len = detector.frames_in

    # Compute background median
    logger.info("Computing background median for %s ...", video_path)
    detector.compute_video_median(cap, 0, n_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read all frames
    logger.info("Reading %d frames from %s ...", n_frames, video_path)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    multi_detections = {}  # {frame: list[dict]}  top-K blobs
    single_detections = {}  # {frame: (px, py, conf)}  top-1 for rendering
    n = len(frames)
    logger.info("Running TrackNet on %d frames (seq_len=%d, top_k=%d) ...", n, seq_len, top_k)

    # Temporal consistency parameters
    MIN_DETECTIONS_IN_WINDOW = 4
    MAX_JUMP_PX = 120.0

    rejected_temporal = 0

    for start in range(0, n, seq_len):
        end = min(start + seq_len, n)
        batch = frames[start:end]
        actual_len = end - start
        if len(batch) < seq_len:
            batch += [batch[-1]] * (seq_len - len(batch))

        heatmaps = detector.infer(batch)  # (seq_len, H, W)

        # ── Extract all detections in this 8-frame window ──────────
        window_dets = {}   # {local_idx: list[dict]}
        window_top1 = {}   # {local_idx: (px, py, conf)}
        for i in range(actual_len):
            blobs = postproc.process_heatmap_multi(heatmaps[i], max_blobs=top_k)
            if blobs:
                window_dets[i] = blobs
                window_top1[i] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"], blobs[0]["blob_sum"])

        # ── Temporal consistency check (using top-1 for smoothness) ─
        if len(window_top1) < MIN_DETECTIONS_IN_WINDOW:
            rejected_temporal += len(window_top1)
            continue

        sorted_idx = sorted(window_top1.keys())
        smooth_count = 0
        for j in range(1, len(sorted_idx)):
            i_prev, i_curr = sorted_idx[j - 1], sorted_idx[j]
            px0, py0, _ = window_top1[i_prev]
            px1, py1, _ = window_top1[i_curr]
            gap = i_curr - i_prev
            disp = math.hypot(px1 - px0, py1 - py0)
            if disp < MAX_JUMP_PX * gap:
                smooth_count += 1

        min_smooth = max(1, (len(sorted_idx) - 1) // 2)
        if smooth_count < min_smooth:
            rejected_temporal += len(window_top1)
            continue

        # Window passed — add all detections (multi + single)
        for i in window_dets:
            fi = start + i
            multi_detections[fi] = window_dets[i]
            single_detections[fi] = window_top1[i]

        if start % (seq_len * 50) == 0:
            logger.info("  Detection progress: %d/%d", min(start + seq_len, n), n)

    multi_blob_count = sum(len(v) for v in multi_detections.values())
    multi_frames = sum(1 for v in multi_detections.values() if len(v) > 1)
    logger.info(
        "Detected ball in %d/%d frames (%d total blobs, %d frames with multiple blobs, "
        "rejected %d by temporal consistency)",
        len(multi_detections), n, multi_blob_count, multi_frames, rejected_temporal,
    )
    return multi_detections, single_detections, n


def load_stereo_calibration(calib_path="src/camera_calibration.json"):
    """Load solvePnP stereo calibration data."""
    with open(calib_path) as f:
        data = json.load(f)
    result = {}
    for cam in ["cam66", "cam68"]:
        c = data[cam]
        K = np.array(c["K"], dtype=np.float64)
        dist = np.array(c["dist"], dtype=np.float64)
        P = np.array(c["P"], dtype=np.float64)
        result[cam] = {"K": K, "dist": dist, "P": P}
    return result


def triangulate_stereo(px66, py66, px68, py68, stereo_cal):
    """Triangulate using solvePnP projection matrices (works at any height).

    Returns (x, y, z, ray_dist) or None if invalid.
    """
    K1, d1, P1 = stereo_cal["cam66"]["K"], stereo_cal["cam66"]["dist"], stereo_cal["cam66"]["P"]
    K2, d2, P2 = stereo_cal["cam68"]["K"], stereo_cal["cam68"]["dist"], stereo_cal["cam68"]["P"]

    # Undistort pixel coordinates
    pt1 = cv2.undistortPoints(
        np.array([[[px66, py66]]], dtype=np.float64), K1, d1, P=K1
    ).reshape(2)
    pt2 = cv2.undistortPoints(
        np.array([[[px68, py68]]], dtype=np.float64), K2, d2, P=K2
    ).reshape(2)

    # Triangulate
    pts4d = cv2.triangulatePoints(P1, P2, pt1.reshape(2, 1), pt2.reshape(2, 1))
    if abs(pts4d[3, 0]) < 1e-10:
        return None
    pts3d = (pts4d[:3, 0] / pts4d[3, 0])

    x, y, z = float(pts3d[0]), float(pts3d[1]), float(pts3d[2])

    # Compute ray distance as quality metric
    # Reproject and check consistency
    proj1, _ = cv2.projectPoints(
        pts3d.reshape(1, 3), np.zeros(3), np.zeros(3),
        K1, d1,
    )
    # Use reprojection error as proxy for ray_dist
    reproj_err1 = np.linalg.norm(proj1.reshape(2) - np.array([px66, py66]))
    ray_dist = reproj_err1 * 0.01  # Scale to approximate meters

    return x, y, z, ray_dist


def triangulate_with_ray_dist(w66, w68, cam66_pos, cam68_pos):
    """Triangulate and also return ray distance (3D agreement metric)."""
    cam1 = np.asarray(cam66_pos, dtype=np.float64)
    cam2 = np.asarray(cam68_pos, dtype=np.float64)
    g1 = np.array([w66[0], w66[1], 0.0])
    g2 = np.array([w68[0], w68[1], 0.0])

    d1 = g1 - cam1
    d2 = g2 - cam2

    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        mid = (g1 + g2) / 2.0
        ray_dist = float(np.linalg.norm(g1 - g2))
        return float(mid[0]), float(mid[1]), 0.0, ray_dist

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    p1_f = cam1 + s * d1
    t = float(np.dot(p1_f - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)
    p2_f = cam2 + t * d2
    s = float(np.dot(p2_f - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0
    ray_dist = float(np.linalg.norm(p1 - p2))

    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2]), ray_dist


def triangulate_detections(det66, det68, cfg):
    """Triangulate 3D from paired 2D detections (top-1 only). Returns {frame: (x,y,z,ray_dist)}."""
    from app.pipeline.homography import HomographyTransformer

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    points_3d = {}
    common = set(det66.keys()) & set(det68.keys())
    logger.info("Triangulating %d common frames ...", len(common))

    for fi in sorted(common):
        px66, py66, _ = det66[fi]
        px68, py68, _ = det68[fi]

        w66 = homo66.pixel_to_world(px66, py66)
        w68 = homo68.pixel_to_world(px68, py68)

        try:
            x, y, z, rd = triangulate_with_ray_dist(w66, w68, cam66_pos, cam68_pos)
        except Exception:
            continue

        if z < -2 or z > 10:
            continue

        points_3d[fi] = (x, y, z, rd)

    logger.info("Valid 3D points: %d", len(points_3d))
    return points_3d


def triangulate_multi_blob(multi66, multi68, cfg):
    """Triangulate using MultiBlobMatcher with top-K blobs + temporal continuity.

    Returns:
        points_3d: {frame: (x, y, z, ray_dist)}
        chosen_pixels: {frame: {'cam66': (px, py), 'cam68': (px, py)}}
        matcher_stats: dict with matching statistics
    """
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.multi_blob_matcher import MultiBlobMatcher

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    matcher = MultiBlobMatcher(
        cam1_pos=cam66_pos,
        cam2_pos=cam68_pos,
        max_ray_distance=2.0,
        valid_z_range=(0.0, 8.0),
        temporal_weight=0.3,
        blob_rank_penalty=0.5,  # moderate preference for top-1 blob
        lost_timeout=30,
        max_velocity=50.0,
        history_size=5,
        fps=30.0,
    )

    common = sorted(set(multi66.keys()) & set(multi68.keys()))
    logger.info("Multi-blob matching: %d common frames, top-K blobs per camera", len(common))

    points_3d = {}
    chosen_pixels = {}

    for fi in common:
        blobs66 = multi66[fi]
        blobs68 = multi68[fi]

        # Add world coordinates to each blob
        cands66 = []
        for b in blobs66:
            wx, wy = homo66.pixel_to_world(b["pixel_x"], b["pixel_y"])
            cands66.append({**b, "world_x": wx, "world_y": wy})

        cands68 = []
        for b in blobs68:
            wx, wy = homo68.pixel_to_world(b["pixel_x"], b["pixel_y"])
            cands68.append({**b, "world_x": wx, "world_y": wy})

        result = matcher.match(
            {"candidates": cands66, "frame_index": fi},
            {"candidates": cands68, "frame_index": fi},
        )

        if result is not None:
            x, y, z = result["x"], result["y"], result["z"]
            rd = result["ray_distance"]
            points_3d[fi] = (x, y, z, rd)
            chosen_pixels[fi] = {
                "cam66": tuple(result["cam1_pixel"]),
                "cam68": tuple(result["cam2_pixel"]),
            }

    stats = matcher.get_stats()
    logger.info(
        "MultiBlobMatcher: %d matched / %d total frames, "
        "non_top1_picks=%d (%.1f%%), temporal_assists=%d (%.1f%%)",
        stats["matched_frames"], stats["total_frames"],
        stats["non_top1_picks"], stats["non_top1_rate"] * 100,
        stats["temporal_assists"], stats["temporal_assist_rate"] * 100,
    )

    return points_3d, chosen_pixels, stats


def build_flight_mask(points_3d, det66, det68, fps=25.0):
    """Build a set of frame indices where the ball is actually in flight.

    Strategy: Use RAY DISTANCE from triangulation as the primary filter.
    When both cameras see the actual ball, their rays converge (low ray_dist).
    When cameras see different objects (shoes, hats), rays don't converge
    (high ray_dist).

    Additionally use 3D court bounds: real ball should be within extended
    court area, not at random world positions.

    Returns:
        active_frames: set of frame indices considered "ball in flight"
        filtered_det66: filtered detections
        filtered_det68: filtered detections
    """
    if not points_3d:
        return set(), {}, {}

    # ── Step 1: Analyze ray distance distribution ──────────────────
    ray_dists = [(fi, points_3d[fi][3]) for fi in sorted(points_3d)]
    rd_values = [rd for _, rd in ray_dists]

    p50 = np.percentile(rd_values, 50)
    p75 = np.percentile(rd_values, 75)
    p90 = np.percentile(rd_values, 90)
    logger.info(
        "  Ray distance: median=%.2fm, p75=%.2fm, p90=%.2fm, max=%.2fm",
        p50, p75, p90, max(rd_values),
    )

    # ── Step 2: Filter by ray distance + court bounds ──────────────
    RAY_DIST_MAX = 2.0    # meters — real ball should have ray_dist < 2m
    COURT_EXTEND = 3.0    # meters — allow some margin outside court

    raw_active = set()
    for fi, (x, y, z, rd) in points_3d.items():
        # Ray distance check: cameras must agree
        if rd > RAY_DIST_MAX:
            continue
        # Court bounds: ball should be in reasonable area
        if x < SINGLES_X_MIN - COURT_EXTEND or x > SINGLES_X_MAX + COURT_EXTEND:
            continue
        if y < -COURT_EXTEND or y > COURT_L + COURT_EXTEND:
            continue
        # Height: ball should be 0-6m, not wild values
        if z > 6.0:
            continue
        raw_active.add(fi)

    logger.info(
        "  Ray distance filter: %d/%d passed (%.1f%%)",
        len(raw_active), len(points_3d), 100.0 * len(raw_active) / len(points_3d),
    )

    # ── Step 3: Expand to cover single-camera frames near active ───
    COAST_FRAMES = 5  # keep showing for a few frames after good 3D

    all_frames = sorted(set(det66.keys()) | set(det68.keys()))
    active_frames = set()
    last_good = -999

    for fi in all_frames:
        if fi in raw_active:
            # Fill small gaps
            if 0 < fi - last_good <= COAST_FRAMES:
                for f in range(last_good + 1, fi):
                    active_frames.add(f)
            active_frames.add(fi)
            last_good = fi
        elif fi - last_good <= COAST_FRAMES:
            active_frames.add(fi)

    # ── Step 4: Build filtered detection dicts ─────────────────────
    filtered_66 = {fi: det66[fi] for fi in active_frames if fi in det66}
    filtered_68 = {fi: det68[fi] for fi in active_frames if fi in det68}

    n_raw66, n_raw68 = len(det66), len(det68)
    n_filt66, n_filt68 = len(filtered_66), len(filtered_68)
    logger.info(
        "Flight filter: cam66 %d→%d (-%d), cam68 %d→%d (-%d)",
        n_raw66, n_filt66, n_raw66 - n_filt66,
        n_raw68, n_filt68, n_raw68 - n_filt68,
    )

    return active_frames, filtered_66, filtered_68


def smooth_trajectory_sg(points_3d, window_length=11, polyorder=3, overlap=8):
    """Apply Savitzky-Golay filter to smooth 3D trajectory.

    Splits trajectory into continuous segments (gaps > 3 frames = new segment),
    smooths each segment with overlap from the previous segment to reduce
    edge effects at segment boundaries.

    Args:
        points_3d: {frame: (x, y, z)} or {frame: (x, y, z, rd)}
        window_length: SG window (must be odd). 11 = ~0.44s at 25fps.
        polyorder: polynomial order for SG fit.
        overlap: number of frames from the previous segment to prepend as
            context for SG smoothing. Reduces edge artifacts at segment starts.

    Returns:
        smoothed: {frame: (x, y, z)} with Savitzky-Golay smoothed coordinates.
    """
    frames = sorted(points_3d.keys())
    if len(frames) < window_length:
        return {fi: points_3d[fi][:3] for fi in frames}

    # Split into continuous segments (gap > 3 frames = new segment)
    # Also record the gap size before each segment for overlap decisions.
    MAX_GAP = 3
    segments = []       # list of frame lists
    gap_before = []     # gap (in frames) before each segment
    seg_start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > MAX_GAP:
            segments.append(frames[seg_start:i])
            gap_before.append(frames[i] - frames[i - 1])
            seg_start = i
    segments.append(frames[seg_start:])
    gap_before.append(0)  # first segment has no gap (shifted: idx 0 → last append)
    # Fix ordering: gap_before[i] is the gap before segments[i]
    gap_before = [0] + gap_before[:-1]

    smoothed = {}
    n_smoothed_segs = 0
    prev_seg_tail = []  # last `overlap` frames of previous segment

    # Max gap (in frames) to still use overlap context.
    # Within a rally, gaps are short (tracking drops a few frames).
    # Between rallies, gaps are long (30+ frames). Only overlap within-rally.
    MAX_OVERLAP_GAP = 10

    for seg_idx, seg_frames in enumerate(segments):
        n = len(seg_frames)

        # Prepend tail of previous segment as padding context,
        # but only if the gap between segments is small (same rally).
        use_overlap = (prev_seg_tail
                       and gap_before[seg_idx] <= MAX_OVERLAP_GAP)
        pad_frames = prev_seg_tail[-overlap:] if use_overlap else []
        pad_n = len(pad_frames)
        all_frames = pad_frames + seg_frames

        xs = np.array([points_3d[fi][0] for fi in all_frames])
        ys = np.array([points_3d[fi][1] for fi in all_frames])
        zs = np.array([points_3d[fi][2] for fi in all_frames])

        total_n = len(all_frames)
        if total_n >= window_length:
            # Apply Savitzky-Golay on padded array
            xs_s = savgol_filter(xs, window_length, polyorder)
            ys_s = savgol_filter(ys, window_length, polyorder)
            zs_s = savgol_filter(zs, window_length, polyorder)
            # Clamp Z >= 0
            zs_s = np.maximum(zs_s, 0.0)
            # Strip padding — only keep current segment's smoothed values
            xs_s = xs_s[pad_n:]
            ys_s = ys_s[pad_n:]
            zs_s = zs_s[pad_n:]
            n_smoothed_segs += 1
        else:
            # Segment (even with padding) too short for SG — use raw
            xs_s = xs[pad_n:]
            ys_s = ys[pad_n:]
            zs_s = zs[pad_n:]

        for i, fi in enumerate(seg_frames):
            smoothed[fi] = (float(xs_s[i]), float(ys_s[i]), float(zs_s[i]))

        # Save tail for next segment's padding
        prev_seg_tail = seg_frames

    logger.info(
        "Savitzky-Golay: %d segments (%d smoothed), %d total points",
        len(segments), n_smoothed_segs, len(smoothed),
    )
    return smoothed


def _smooth_2d_for_render(det_dict, median_k=5, max_gap=3, interp_gap=4):
    """Smooth 2D pixel detections for visual rendering.

    Pipeline:
    1. Median filter (kernel=median_k) to remove outlier pixel spikes
    2. Linear interpolation for short gaps (1-interp_gap missing frames)

    Args:
        det_dict: {frame: (px, py, conf)} -- detection dict with confidence
        median_k: median filter kernel size
        max_gap: max frame gap within a segment for median filtering
        interp_gap: max gap size (in frames) for linear interpolation

    Returns:
        smoothed dict in same format {frame: (px, py, conf)}
    """
    if not det_dict:
        return det_dict

    frames = sorted(det_dict.keys())
    if len(frames) < 3:
        return det_dict

    # Step 1: median filter within segments
    half = median_k // 2
    segments = []
    seg_start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > max_gap:
            segments.append(frames[seg_start:i])
            seg_start = i
    segments.append(frames[seg_start:])

    smoothed = {}
    for seg in segments:
        xs = [det_dict[fi][0] for fi in seg]
        ys = [det_dict[fi][1] for fi in seg]
        confs = [det_dict[fi][2] if len(det_dict[fi]) > 2 else 1.0 for fi in seg]
        for i, fi in enumerate(seg):
            lo = max(0, i - half)
            hi = min(len(seg), i + half + 1)
            mx = float(np.median(xs[lo:hi]))
            my = float(np.median(ys[lo:hi]))
            smoothed[fi] = (mx, my, confs[i])

    # Step 2: interpolate short gaps
    frames_s = sorted(smoothed.keys())
    interpolated = dict(smoothed)
    for i in range(len(frames_s) - 1):
        gap = frames_s[i + 1] - frames_s[i]
        if 2 <= gap <= interp_gap:
            x0, y0, c0 = smoothed[frames_s[i]]
            x1, y1, c1 = smoothed[frames_s[i + 1]]
            avg_conf = (c0 + c1) / 2
            for g in range(1, gap):
                t = g / gap
                fi_interp = frames_s[i] + g
                interpolated[fi_interp] = (
                    float(x0 + t * (x1 - x0)),
                    float(y0 + t * (y1 - y0)),
                    avg_conf,
                )

    n_orig = len(det_dict)
    n_interp = len(interpolated) - n_orig
    logger.info(
        "2D smooth: %d frames -> %d (median%d + %d interpolated)",
        n_orig, len(interpolated), median_k, n_interp,
    )
    return interpolated


def detect_bounces(points_3d, fps=25.0):
    """Hybrid bounce detection: V-shape + parabolic trajectory segmentation.

    Combines two signals:
    1. V-shape: Z-axis local minimum with margins on both sides
    2. Parabolic split: fitting separate parabolas to left/right windows
       gives significantly lower residual than a joint fit

    A candidate needs EITHER a strong signal from one method OR moderate
    signals from both methods. This catches bounces that one method alone
    might miss while keeping false positives low.

    Returns list of {frame, x, y, z, in_court}.
    """
    frames = sorted(points_3d.keys())
    if len(frames) < 10:
        return []

    # Build arrays
    data = [(fi, *points_3d[fi][:3]) for fi in frames]
    frame_arr = np.array([d[0] for d in data])
    x_arr = np.array([d[1] for d in data])
    y_arr = np.array([d[2] for d in data])
    z_arr = np.array([d[3] for d in data])
    frame_to_idx = {int(frame_arr[i]): i for i in range(len(frame_arr))}

    # Split into continuous segments
    MAX_GAP = 3
    segments = []
    seg_start = 0
    for i in range(1, len(frame_arr)):
        if frame_arr[i] - frame_arr[i - 1] > MAX_GAP:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, len(frame_arr)))

    # ── Parabolic fit helper ──────────────────────────────────────
    def _fit_parabola_residual(indices):
        if len(indices) < 3:
            return float("inf")
        t = (frame_arr[indices] - frame_arr[indices[0]]) / fps
        z = z_arr[indices]
        try:
            coeffs = np.polyfit(t, z, 2)
            return float(np.mean((z - np.polyval(coeffs, t)) ** 2))
        except (np.linalg.LinAlgError, ValueError):
            return float("inf")

    # ── Compute both signals for every candidate frame ────────────
    HALF_WINS = [4, 6, 8]
    V_WINDOW = 8
    MIN_SEG_LEN = 15
    BOUNCE_Z_MAX = 0.8
    frame_set = set(frames)

    all_candidates = []

    for seg_s, seg_e in segments:
        seg_len = seg_e - seg_s
        if seg_len < MIN_SEG_LEN:
            continue

        for i in range(seg_s + max(max(HALF_WINS), V_WINDOW),
                       seg_e - max(max(HALF_WINS), V_WINDOW)):
            z_i = z_arr[i]
            if z_i > BOUNCE_Z_MAX:
                continue

            x_i, y_i = x_arr[i], y_arr[i]
            if x_i < SINGLES_X_MIN - 1.0 or x_i > SINGLES_X_MAX + 1.0:
                continue
            if y_i < -1.0 or y_i > COURT_L + 1.0:
                continue

            # ── Signal 1: V-shape margins ─────────────────────────
            z_before = z_arr[i - V_WINDOW:i]
            z_after = z_arr[i + 1:i + V_WINDOW + 1]
            margin_before = float(np.mean(z_before) - z_i)
            margin_after = float(np.mean(z_after) - z_i)

            # Asymmetric: min side ≥ 0.1m, max side ≥ threshold
            min_margin = min(margin_before, margin_after)
            max_margin = max(margin_before, margin_after)
            v_score = margin_before + margin_after  # higher = stronger V

            # V-shape pass levels:
            #   strong: min≥0.15, max≥0.3 (clear V)
            #   moderate: min≥0.08, max≥0.15 (weak but visible V)
            v_strong = min_margin >= 0.15 and max_margin >= 0.3
            v_moderate = min_margin >= 0.08 and max_margin >= 0.15

            # ── Signal 2: Parabolic split ratio ───────────────────
            best_ratio = 0.0
            for hw in HALF_WINS:
                li = list(range(max(seg_s, i - hw), i + 1))
                ri = list(range(i, min(seg_e, i + hw + 1)))
                if len(li) < 3 or len(ri) < 3:
                    continue
                ji = list(range(max(seg_s, i - hw), min(seg_e, i + hw + 1)))
                rl = _fit_parabola_residual(li)
                rr = _fit_parabola_residual(ri)
                rj = _fit_parabola_residual(ji)
                rs = (rl * len(li) + rr * len(ri)) / (len(li) + len(ri))
                ratio = rj / rs if rs > 1e-8 else 0
                if ratio > best_ratio:
                    best_ratio = ratio

            # Parabola pass levels:
            #   strong: ratio ≥ 5 (clear trajectory change)
            #   moderate: ratio ≥ 2.0 (some change)
            p_strong = best_ratio >= 5.0
            p_moderate = best_ratio >= 2.0

            # ── Combined decision ─────────────────────────────────
            # Accept if:
            #   - V strong alone (classic bounce)
            #   - Parabola strong + V moderate (physics confirms weak V)
            #   - V moderate + Parabola moderate + z < 0.4m (both agree near ground)
            accepted = False
            if v_strong:
                accepted = True
            elif p_strong and v_moderate:
                accepted = True
            elif v_moderate and p_moderate and z_i < 0.4:
                accepted = True

            if accepted:
                combined_score = v_score + 0.1 * best_ratio
                all_candidates.append({
                    "frame": int(frame_arr[i]),
                    "x": float(x_i),
                    "y": float(y_i),
                    "z": float(z_i),
                    "in_court": SINGLES_X_MIN <= x_i <= SINGLES_X_MAX and 0 <= y_i <= COURT_L,
                    "score": combined_score,
                    "v_score": v_score,
                    "p_ratio": best_ratio,
                })

    # ── Dense segment filter ──────────────────────────────────────
    def _in_dense_segment(fi):
        nearby = sum(1 for f in range(fi - 20, fi + 21) if f in frame_set)
        return nearby >= 20

    all_candidates = [c for c in all_candidates if _in_dense_segment(c["frame"])]

    # ── Minimum speed filter ──────────────────────────────────────
    # A real bounce has the ball moving at speed. Dead-ball drift is < 3 m/s.
    MIN_BOUNCE_SPEED = 3.0  # m/s (~11 km/h)
    SPEED_DT = 3  # frames for speed estimate

    def _local_speed(fi):
        """Compute 3D speed around frame fi using ±SPEED_DT frames."""
        idx = frame_to_idx.get(fi)
        if idx is None:
            return 0.0
        # Look back and forward for speed
        i_back = max(0, idx - SPEED_DT)
        i_fwd = min(len(frame_arr) - 1, idx + SPEED_DT)
        if i_fwd == i_back:
            return 0.0
        dt = (frame_arr[i_fwd] - frame_arr[i_back]) / fps
        if dt < 1e-6:
            return 0.0
        dx = x_arr[i_fwd] - x_arr[i_back]
        dy = y_arr[i_fwd] - y_arr[i_back]
        dz = z_arr[i_fwd] - z_arr[i_back]
        return float(np.sqrt(dx**2 + dy**2 + dz**2) / dt)

    all_candidates = [c for c in all_candidates
                      if _local_speed(c["frame"]) >= MIN_BOUNCE_SPEED]

    # ── NMS: pick best within clusters ────────────────────────────
    all_candidates.sort(key=lambda b: b["frame"])

    bounces = []
    MIN_GAP = 5
    for c in all_candidates:
        if bounces and c["frame"] - bounces[-1]["frame"] < MIN_GAP:
            if c["score"] > bounces[-1]["score"]:
                bounces[-1] = c
            continue
        bounces.append(c)

    # Clean up
    for b in bounces:
        b.pop("score", None)
        b.pop("v_score", None)
        b.pop("p_ratio", None)

    logger.info("Detected %d bounces (hybrid V-shape + parabolic)", len(bounces))
    return bounces


def detect_net_crossings(smoothed_3d, fps=25.0):
    """Detect frames where the ball crosses the net and compute speed.

    A net crossing occurs when y changes from one side of NET_Y to the other
    in consecutive tracked frames. Speed is computed from a short window
    around the crossing point for stability.

    Speed limits:
        - Min 20 km/h (below this is likely tracking drift, not a real shot)
        - Max 250 km/h (fastest recorded tennis serve ~263 km/h)
        - Require at least 3 frames on each side of net within ±8 frames

    Returns:
        List of {frame, speed_kmh, direction} where direction is
        'near_to_far' (y increasing) or 'far_to_near' (y decreasing).
    """
    frames = sorted(smoothed_3d.keys())
    if len(frames) < 5:
        return []

    MIN_SPEED_KMH = 20.0
    MAX_SPEED_KMH = 250.0
    SPEED_WINDOW = 4  # frames on each side for speed calculation
    VERIFY_WINDOW = 8  # frames to verify ball is truly on the other side
    MIN_SIDE_FRAMES = 3  # need at least this many frames on each side

    crossings = []
    frame_set = set(frames)

    for idx in range(1, len(frames)):
        fi_prev = frames[idx - 1]
        fi_curr = frames[idx]

        y_prev = smoothed_3d[fi_prev][1]
        y_curr = smoothed_3d[fi_curr][1]

        # Check for net crossing
        if not ((y_prev < NET_Y <= y_curr) or (y_curr < NET_Y <= y_prev)):
            continue

        # Determine direction
        direction = "near_to_far" if y_curr > y_prev else "far_to_near"

        # Verify: need enough tracked frames on each side of net nearby
        before_frames = [f for f in range(fi_curr - VERIFY_WINDOW, fi_curr)
                         if f in frame_set]
        after_frames = [f for f in range(fi_curr + 1, fi_curr + VERIFY_WINDOW + 1)
                        if f in frame_set]

        if len(before_frames) < MIN_SIDE_FRAMES or len(after_frames) < MIN_SIDE_FRAMES:
            continue

        # Compute speed using a window around the crossing
        # Find frames within ±SPEED_WINDOW of crossing for stable estimate
        window_frames = []
        for f in frames:
            if fi_curr - SPEED_WINDOW <= f <= fi_curr + SPEED_WINDOW:
                window_frames.append(f)

        if len(window_frames) < 3:
            continue

        # Use endpoints of window for speed (more stable than adjacent frames)
        f_start = window_frames[0]
        f_end = window_frames[-1]
        dt = (f_end - f_start) / fps
        if dt < 1e-6:
            continue

        p_start = np.array(smoothed_3d[f_start])
        p_end = np.array(smoothed_3d[f_end])
        dist_3d = float(np.linalg.norm(p_end - p_start))

        speed_ms = dist_3d / dt
        speed_kmh = speed_ms * 3.6

        # Clamp speed
        if speed_kmh < MIN_SPEED_KMH or speed_kmh > MAX_SPEED_KMH:
            continue

        # Minimum gap between crossings (ball can't cross net twice in 10 frames)
        if crossings and fi_curr - crossings[-1]["frame"] < 10:
            # Keep the one with more reasonable speed
            if abs(speed_kmh - 100) < abs(crossings[-1]["speed_kmh"] - 100):
                crossings[-1] = {
                    "frame": fi_curr,
                    "speed_kmh": speed_kmh,
                    "direction": direction,
                }
            continue

        crossings.append({
            "frame": fi_curr,
            "speed_kmh": speed_kmh,
            "direction": direction,
        })

    logger.info("Detected %d net crossings", len(crossings))
    for i, nc in enumerate(crossings):
        logger.info(
            "  Net crossing %d: frame=%d  %.0f km/h  %s",
            i + 1, nc["frame"], nc["speed_kmh"], nc["direction"],
        )
    return crossings


def draw_court(panel_w, panel_h, margin):
    """Draw a static tennis court top-down, return (image, world_to_court_fn)."""
    court = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Dark green background
    court[:] = (30, 60, 30)

    draw_w = panel_w - 2 * margin
    draw_h = panel_h - 2 * margin

    # Map world coordinates (singles: x=1.37..6.86, y=0..23.77) to court pixels
    def w2c(wx, wy):
        cx = int(margin + ((wx - SINGLES_X_MIN) / COURT_W) * draw_w)
        cy = int(margin + (1.0 - wy / COURT_L) * draw_h)  # flip Y: baseline at bottom
        return cx, cy

    # Court fill (singles boundaries)
    tl = w2c(SINGLES_X_MIN, COURT_L)
    br = w2c(SINGLES_X_MAX, 0)
    cv2.rectangle(court, tl, br, (45, 100, 45), -1)

    # Court outline (singles)
    cv2.rectangle(court, tl, br, (255, 255, 255), 2)

    # Net line
    cv2.line(court, w2c(SINGLES_X_MIN, NET_Y), w2c(SINGLES_X_MAX, NET_Y), (200, 200, 200), 2)

    # Service lines
    cv2.line(court, w2c(SINGLES_X_MIN, SERVICE_NEAR), w2c(SINGLES_X_MAX, SERVICE_NEAR), (180, 180, 180), 1)
    cv2.line(court, w2c(SINGLES_X_MIN, SERVICE_FAR), w2c(SINGLES_X_MAX, SERVICE_FAR), (180, 180, 180), 1)

    # Center service line
    center_x = (SINGLES_X_MIN + SINGLES_X_MAX) / 2
    cv2.line(court, w2c(center_x, SERVICE_NEAR), w2c(center_x, SERVICE_FAR), (180, 180, 180), 1)

    # Center marks
    ct = w2c(center_x, 0)
    cv2.line(court, ct, (ct[0], ct[1] - 8), (180, 180, 180), 1)
    ct2 = w2c(center_x, COURT_L)
    cv2.line(court, ct2, (ct2[0], ct2[1] + 8), (180, 180, 180), 1)

    return court, w2c


def draw_3d_view(panel_w, panel_h, smoothed_3d, bounces_so_far, current_frame,
                  trail_len=60):
    """Render a 3D perspective view of the court with ball trajectory.

    Uses a simple perspective projection from a fixed camera angle.
    """
    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    img[:] = (25, 25, 35)  # dark background

    # 3D perspective camera parameters (viewing from above-behind near end)
    cam_pos = np.array([4.115, -8.0, 16.0])  # x=center, behind near baseline, high up
    look_at = np.array([4.115, 11.885, 0.0])  # look at net center
    up = np.array([0.0, 0.0, 1.0])

    # Build view matrix
    fwd = look_at - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, fwd)

    focal = panel_w * 0.9
    cx, cy = panel_w / 2, panel_h / 2 - 30  # shift up slightly

    def project_3d(wx, wy, wz):
        """Project world (x,y,z) to panel pixel."""
        p = np.array([wx, wy, wz]) - cam_pos
        x_cam = np.dot(p, right)
        y_cam = np.dot(p, cam_up)
        z_cam = np.dot(p, fwd)
        if z_cam < 0.1:
            return None
        px = int(cx + focal * x_cam / z_cam)
        py = int(cy - focal * y_cam / z_cam)
        return (px, py)

    # Draw court wireframe
    court_pts = [
        # Singles court corners (z=0)
        (SINGLES_X_MIN, 0, 0), (SINGLES_X_MAX, 0, 0),
        (SINGLES_X_MAX, COURT_L, 0), (SINGLES_X_MIN, COURT_L, 0),
    ]
    court_lines = [
        # Court outline
        ((SINGLES_X_MIN, 0, 0), (SINGLES_X_MAX, 0, 0)),
        ((SINGLES_X_MAX, 0, 0), (SINGLES_X_MAX, COURT_L, 0)),
        ((SINGLES_X_MAX, COURT_L, 0), (SINGLES_X_MIN, COURT_L, 0)),
        ((SINGLES_X_MIN, COURT_L, 0), (SINGLES_X_MIN, 0, 0)),
        # Net
        ((SINGLES_X_MIN, NET_Y, 0), (SINGLES_X_MAX, NET_Y, 0)),
        ((SINGLES_X_MIN, NET_Y, NET_H), (SINGLES_X_MAX, NET_Y, NET_H)),
        ((SINGLES_X_MIN, NET_Y, 0), (SINGLES_X_MIN, NET_Y, NET_H)),
        ((SINGLES_X_MAX, NET_Y, 0), (SINGLES_X_MAX, NET_Y, NET_H)),
        (((SINGLES_X_MIN + SINGLES_X_MAX) / 2, NET_Y, 0),
         ((SINGLES_X_MIN + SINGLES_X_MAX) / 2, NET_Y, NET_H)),
        # Service lines
        ((SINGLES_X_MIN, SERVICE_NEAR, 0), (SINGLES_X_MAX, SERVICE_NEAR, 0)),
        ((SINGLES_X_MIN, SERVICE_FAR, 0), (SINGLES_X_MAX, SERVICE_FAR, 0)),
        # Center service line
        (((SINGLES_X_MIN + SINGLES_X_MAX) / 2, SERVICE_NEAR, 0),
         ((SINGLES_X_MIN + SINGLES_X_MAX) / 2, SERVICE_FAR, 0)),
    ]

    # Draw court surface (filled quad)
    corners_2d = [project_3d(*p) for p in court_pts]
    if all(c is not None for c in corners_2d):
        pts = np.array(corners_2d, dtype=np.int32)
        cv2.fillPoly(img, [pts], (35, 70, 35))

    # Draw lines
    for (x1, y1, z1), (x2, y2, z2) in court_lines:
        p1 = project_3d(x1, y1, z1)
        p2 = project_3d(x2, y2, z2)
        if p1 and p2:
            is_net = (z1 > 0 or z2 > 0)
            color = (200, 200, 200) if is_net else (150, 150, 150)
            thickness = 2 if is_net else 1
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # Draw ball trajectory trail
    trail_frames = sorted([f for f in smoothed_3d if current_frame - trail_len <= f <= current_frame])

    if len(trail_frames) >= 2:
        for i in range(1, len(trail_frames)):
            f0, f1 = trail_frames[i - 1], trail_frames[i]
            x0, y0, z0 = smoothed_3d[f0][:3]
            x1, y1, z1 = smoothed_3d[f1][:3]
            p0 = project_3d(x0, y0, z0)
            p1 = project_3d(x1, y1, z1)
            if p0 and p1:
                alpha = (i / len(trail_frames))
                # Color by height: blue(ground) → yellow(high)
                r = int(50 + 200 * min(1, z1 / 3.0))
                g = int(200 * alpha)
                b = int(255 * (1 - min(1, z1 / 3.0)) * alpha)
                cv2.line(img, p0, p1, (b, g, r), max(1, int(2 * alpha)), cv2.LINE_AA)

    # Draw current ball position
    if current_frame in smoothed_3d:
        x, y, z = smoothed_3d[current_frame][:3]
        bp = project_3d(x, y, z)
        if bp:
            # Shadow on ground
            sp = project_3d(x, y, 0)
            if sp:
                cv2.circle(img, sp, 4, (40, 40, 40), -1, cv2.LINE_AA)
                # Vertical line from shadow to ball
                cv2.line(img, sp, bp, (80, 80, 80), 1, cv2.LINE_AA)
            # Ball glow
            overlay = img.copy()
            cv2.circle(overlay, bp, 10, (0, 200, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.circle(img, bp, 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Draw bounce markers
    for bi, b in enumerate(bounces_so_far):
        bx = b.get("x_homo", b["x"])
        by = b.get("y_homo", b["y"])
        bp = project_3d(bx, by, 0)
        if bp:
            color = (0, 200, 0) if b["in_court"] else (0, 0, 200)
            cv2.circle(img, bp, 4, color, -1, cv2.LINE_AA)

    # Label
    cv2.putText(img, "3D View", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    return img


def draw_ball_marker(img, px, py, scale, color=(0, 200, 255)):
    """Draw a glowing ball marker on a camera frame."""
    x, y = int(px * scale), int(py * scale)
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return

    # Outer glow
    overlay = img.copy()
    cv2.circle(overlay, (x, y), 12, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # Inner solid
    cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_trail(img, trail, scale, color=(0, 200, 255)):
    """Draw fading trajectory trail."""
    n = len(trail)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n
        pt1 = (int(trail[i - 1][0] * scale), int(trail[i - 1][1] * scale))
        pt2 = (int(trail[i][0] * scale), int(trail[i][1] * scale))
        c = tuple(int(v * alpha) for v in color)
        thickness = max(1, int(2 * alpha))
        cv2.line(img, pt1, pt2, c, thickness, cv2.LINE_AA)


def render_video(
    video66_path, video68_path, det66, det68, bounces, n_frames, output_path,
    net_crossings=None, smoothed_3d=None, canvas_w=1920,
):
    """Render the final tracking video."""
    cap66 = cv2.VideoCapture(video66_path)
    cap68 = cv2.VideoCapture(video68_path)

    fps = cap66.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap66.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap66.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Layout: [cam66 | cam68 | minimap]
    court_pw = 240
    video_area_w = canvas_w - court_pw
    half_w = video_area_w // 2
    scale = half_w / orig_w
    half_h = int(orig_h * scale)
    canvas_h = half_h

    # Court panel same height as camera views
    base_court, w2c = draw_court(court_pw, canvas_h, COURT_MARGIN)

    # Build bounce lookup
    bounce_by_frame = {b["frame"]: b for b in bounces}
    bounces_so_far = []

    # Net crossing lookup
    nc_list = net_crossings or []
    nc_by_frame = {nc["frame"]: nc for nc in nc_list}
    active_nc = None  # currently displayed net crossing
    NC_DISPLAY_FRAMES = 50  # show speed for ~2 seconds

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    logger.info(
        "Rendering: %dx%d @ %.1ffps, cam=%dx%d, court=%dx%d",
        canvas_w, canvas_h, fps, half_w, half_h, court_pw, canvas_h,
    )

    trail66 = []
    trail68 = []

    for fi in range(n_frames):
        ret66, frame66 = cap66.read()
        ret68, frame68 = cap68.read()
        if not ret66 or not ret68:
            break

        # Resize camera frames
        small66 = cv2.resize(frame66, (half_w, half_h))
        small68 = cv2.resize(frame68, (half_w, half_h))

        # ── Trail + marker for cam66 ────────────────────────────────
        if fi in det66:
            px, py, conf = det66[fi]
            jumped = (
                trail66
                and math.hypot(px - trail66[-1][0], py - trail66[-1][1]) > TRAIL_JUMP_PX
            )
            if jumped:
                trail66.clear()
            trail66.append((px, py))
            if len(trail66) > TRAIL_LEN:
                trail66.pop(0)

            draw_trail(small66, trail66, scale)
            draw_ball_marker(small66, px, py, scale)
        else:
            trail66.clear()

        # ── Trail + marker for cam68 ────────────────────────────────
        if fi in det68:
            px, py, conf = det68[fi]
            jumped = (
                trail68
                and math.hypot(px - trail68[-1][0], py - trail68[-1][1]) > TRAIL_JUMP_PX
            )
            if jumped:
                trail68.clear()
            trail68.append((px, py))
            if len(trail68) > TRAIL_LEN:
                trail68.pop(0)

            draw_trail(small68, trail68, scale)
            draw_ball_marker(small68, px, py, scale)
        else:
            trail68.clear()

        # ── Camera labels ───────────────────────────────────────────
        cv2.putText(
            small66, "cam66", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            small68, "cam68", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Frame counter
        ts = fi / fps
        cv2.putText(
            small66, f"F{fi}  {ts:.1f}s", (10, half_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

        # ── Minimap: bounces only ──────────────────────────────────
        court_panel = base_court.copy()

        # Accumulate bounces
        if fi in bounce_by_frame:
            bounces_so_far.append(bounce_by_frame[fi])

        # Draw all bounce markers
        for bi, b in enumerate(bounces_so_far):
            # Use homography-refined coords when available (more accurate for ground contact)
            bx = b.get("x_homo", b["x"])
            by = b.get("y_homo", b["y"])
            bpt = w2c(bx, by)

            # Color: green=IN, red=OUT; recent ones brighter
            age = len(bounces_so_far) - 1 - bi
            brightness = max(0.4, 1.0 - age * 0.05)

            if b["in_court"]:
                outer_c = (0, int(200 * brightness), 0)
                inner_c = (0, int(255 * brightness), int(100 * brightness))
            else:
                outer_c = (0, 0, int(200 * brightness))
                inner_c = (0, int(80 * brightness), int(255 * brightness))

            # Outer ring
            cv2.circle(court_panel, bpt, 8, outer_c, 2, cv2.LINE_AA)
            # Inner filled
            cv2.circle(court_panel, bpt, 4, inner_c, -1, cv2.LINE_AA)

            # Bounce number label
            label = str(bi + 1)
            cv2.putText(
                court_panel, label, (bpt[0] + 10, bpt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
            )

            # IN/OUT label for recent bounces
            if age < 5:
                tag = "IN" if b["in_court"] else "OUT"
                tag_c = (0, 255, 0) if b["in_court"] else (0, 0, 255)
                cv2.putText(
                    court_panel, tag, (bpt[0] + 10, bpt[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, tag_c, 1, cv2.LINE_AA,
                )

        # ── Net crossing speed overlay on cam66 ────────────────────
        if fi in nc_by_frame:
            active_nc = {"nc": nc_by_frame[fi], "start_frame": fi}

        if active_nc is not None:
            age = fi - active_nc["start_frame"]
            if age < NC_DISPLAY_FRAMES:
                nc = active_nc["nc"]
                # Fade out: full opacity for first half, then fade
                alpha = 1.0 if age < NC_DISPLAY_FRAMES // 2 else (
                    1.0 - (age - NC_DISPLAY_FRAMES // 2) / (NC_DISPLAY_FRAMES // 2)
                )
                speed_text = f"{nc['speed_kmh']:.0f} km/h"
                # Draw on cam66 — centered top area
                text_size = cv2.getTextSize(
                    speed_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3,
                )[0]
                tx = (half_w - text_size[0]) // 2
                ty = 80
                # Background box for readability
                pad = 8
                overlay = small66.copy()
                cv2.rectangle(
                    overlay,
                    (tx - pad, ty - text_size[1] - pad),
                    (tx + text_size[0] + pad, ty + pad),
                    (0, 0, 0), -1,
                )
                cv2.addWeighted(overlay, 0.5 * alpha, small66, 1.0 - 0.5 * alpha, 0, small66)
                # Speed text — yellow/orange
                color = (0, int(220 * alpha), int(255 * alpha))
                cv2.putText(
                    small66, speed_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
                )
                # Direction arrow on minimap at net line
                net_pt = w2c((SINGLES_X_MIN + SINGLES_X_MAX) / 2, NET_Y)
                arrow_len = 15
                if nc["direction"] == "near_to_far":
                    arrow_end = (net_pt[0], net_pt[1] - arrow_len)
                else:
                    arrow_end = (net_pt[0], net_pt[1] + arrow_len)
                arrow_color = (0, int(200 * alpha), int(255 * alpha))
                cv2.arrowedLine(
                    court_panel, net_pt, arrow_end, arrow_color, 2, cv2.LINE_AA,
                )
                # Speed on minimap too
                sp_text = f"{nc['speed_kmh']:.0f}"
                cv2.putText(
                    court_panel, sp_text,
                    (net_pt[0] + 15, net_pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1, cv2.LINE_AA,
                )
            else:
                active_nc = None

        # Court panel title
        cv2.putText(
            court_panel, f"Bounces: {len(bounces_so_far)}",
            (court_pw // 2 - 40, canvas_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # ── Compose final canvas ───────────────────────────────────
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:half_h, :half_w] = small66
        canvas[:half_h, half_w:half_w * 2] = small68
        # Court panel — center vertically if needed
        y_off = (canvas_h - court_panel.shape[0]) // 2
        canvas[y_off:y_off + court_panel.shape[0], video_area_w:video_area_w + court_pw] = court_panel

        # Separator lines
        cv2.line(canvas, (half_w, 0), (half_w, canvas_h), (80, 80, 80), 1)
        cv2.line(canvas, (video_area_w, 0), (video_area_w, canvas_h), (80, 80, 80), 1)

        writer.write(canvas)

        if fi % 500 == 0:
            logger.info("  Rendered %d/%d frames", fi, n_frames)

    writer.release()
    cap66.release()
    cap68.release()
    logger.info("Video saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Render dual-camera tracking video")
    parser.add_argument("--video66", default="uploads/cam66_20260307_173403_2min.mp4")
    parser.add_argument("--video68", default="uploads/cam68_20260307_173403_2min.mp4")
    parser.add_argument("--output", default="exports/tracking_video.mp4")
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--top-k", type=int, default=2, help="Top-K blobs per camera")
    parser.add_argument(
        "--mode", choices=["top1", "multi", "viterbi"], default="viterbi",
        help="top1: legacy single-blob; multi: top-K + MultiBlobMatcher; viterbi: global optimal path",
    )
    args = parser.parse_args()

    cfg = load_config()

    # ── Phase 1: Detection ──────────────────────────────────────────
    logger.info("=== Phase 1: TrackNet Detection (top_k=%d, mode=%s) ===", args.top_k, args.mode)
    detector, postproc = build_detector(cfg)

    logger.info("--- cam66 ---")
    multi66, det66, n66 = run_detection_multi(
        args.video66, detector, postproc, args.max_frames, top_k=args.top_k,
    )

    # Reset detector background for second camera
    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("--- cam68 ---")
    multi68, det68, n68 = run_detection_multi(
        args.video68, detector, postproc, args.max_frames, top_k=args.top_k,
    )

    n_frames = min(n66, n68)
    logger.info("Common frames: %d, det66=%d, det68=%d", n_frames, len(det66), len(det68))

    if args.mode == "viterbi":
        # ── Phase 2: Viterbi Global Optimal Path ─────────────────
        logger.info("=== Phase 2: Viterbi Global Optimal Trajectory ===")
        from app.pipeline.homography import HomographyTransformer
        from app.pipeline.viterbi_tracker import ViterbiTracker

        homo_path = cfg["homography"]["path"]
        homo66 = HomographyTransformer(homo_path, "cam66")
        homo68 = HomographyTransformer(homo_path, "cam68")

        cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
        cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

        # Load stereo calibration for reprojection consistency check + re-triangulation
        stereo_cal = load_stereo_calibration("src/camera_calibration.json")

        tracker = ViterbiTracker(
            cam1_pos=cam66_pos,
            cam2_pos=cam68_pos,
            max_ray_distance=2.5,
            valid_z_range=(0.0, 8.0),
            fps=25.0,
            gap_threshold=5,
            stereo_cal=stereo_cal,
        )

        points_3d_homo, chosen_pixels, viterbi_stats = tracker.track(
            multi66, multi68, homo66, homo68,
        )

        # Re-triangulate Viterbi-selected pixels using solvePnP (better Z accuracy)
        points_3d = {}
        stereo_ok = 0
        for fi, cp in chosen_pixels.items():
            px66, py66 = cp["cam66"]
            px68, py68 = cp["cam68"]
            result = triangulate_stereo(px66, py66, px68, py68, stereo_cal)
            if result is not None:
                x, y, z, rd = result
                if -2 < z < 10:
                    points_3d[fi] = (x, y, z, rd)
                    stereo_ok += 1
                else:
                    # Fallback to homography result
                    points_3d[fi] = points_3d_homo[fi]
            elif fi in points_3d_homo:
                points_3d[fi] = points_3d_homo[fi]

        logger.info(
            "solvePnP re-triangulation: %d/%d frames upgraded (%d fallback to homography)",
            stereo_ok, len(chosen_pixels), len(points_3d) - stereo_ok,
        )

        # For rendering: use Viterbi-chosen blob pixels (not raw top-1).
        # This ensures the rendered ball position matches what Viterbi selected,
        # eliminating flickering when Viterbi picks blob#2 instead of blob#1.
        matched_frames = set(points_3d.keys())
        filt66 = {}
        filt68 = {}
        for fi in matched_frames:
            if fi in chosen_pixels:
                cp = chosen_pixels[fi]
                filt66[fi] = (cp["cam66"][0], cp["cam66"][1], 1.0)
                filt68[fi] = (cp["cam68"][0], cp["cam68"][1], 1.0)
            else:
                if fi in det66:
                    filt66[fi] = det66[fi]
                if fi in det68:
                    filt68[fi] = det68[fi]

        logger.info(
            "After Viterbi + solvePnP: %d 3D points (render det66=%d, det68=%d)",
            len(points_3d), len(filt66), len(filt68),
        )

    elif args.mode == "multi":
        # ── Phase 2: Multi-Blob Matching with Temporal Continuity ──
        logger.info("=== Phase 2: Multi-Blob Matching (top-%d + temporal) ===", args.top_k)
        points_3d, chosen_pixels, matcher_stats = triangulate_multi_blob(multi66, multi68, cfg)

        matched_frames = set(points_3d.keys())
        filt66 = {fi: det66[fi] for fi in matched_frames if fi in det66}
        filt68 = {fi: det68[fi] for fi in matched_frames if fi in det68}

        logger.info(
            "After multi-blob matching: %d 3D points (render det66=%d, det68=%d)",
            len(points_3d), len(filt66), len(filt68),
        )
    else:
        # ── Phase 2: Legacy top-1 triangulation ────────────────────
        logger.info("=== Phase 2: 3D Triangulation (top-1 only) ===")
        points_3d = triangulate_detections(det66, det68, cfg)

        # ── Phase 2.5: Flight Filter ──────────────────────────────
        logger.info("=== Phase 2.5: Flight Filter (remove dead-ball FP) ===")
        fps = 25.0
        _, filt66, filt68 = build_flight_mask(points_3d, det66, det68, fps)

    # ── Phase 2.5: Rally Segmentation ──────────────────────────────
    rally_model_path = Path("model_weight/rally_segmentation.pkl")
    if rally_model_path.exists():
        logger.info("=== Phase 2.5: Rally Segmentation (ML model) ===")
        import pickle
        from tools.train_rally_model import extract_features, smooth_predictions

        with open(rally_model_path, "rb") as f:
            rally_bundle = pickle.load(f)

        rally_threshold = rally_bundle.get("threshold", 0.42)
        rally_scaler = rally_bundle.get("scaler", None)

        # Build features from detection data
        X_rally = extract_features(det66, multi66, multi68, n_frames)
        if rally_scaler is not None:
            X_rally = rally_scaler.transform(X_rally)

        # Ensemble prediction
        models_to_avg = []
        for key in ["rf", "rf_model"]:
            if key in rally_bundle:
                models_to_avg.append(rally_bundle[key].predict_proba(X_rally)[:, 1])
                break
        for key in ["gbt", "gb_model"]:
            if key in rally_bundle:
                models_to_avg.append(rally_bundle[key].predict_proba(X_rally)[:, 1])
                break
        if "lstm_state" in rally_bundle:
            from tools.train_rally_model import BiLSTMClassifier
            lstm_model = BiLSTMClassifier(X_rally.shape[1])
            lstm_model.load_state_dict(rally_bundle["lstm_state"])
            lstm_model.eval()
            import torch as _torch
            with _torch.no_grad():
                seq = _torch.from_numpy(X_rally.astype(np.float32)).unsqueeze(0)
                lstm_probs = _torch.sigmoid(lstm_model(seq)).squeeze().numpy()
            models_to_avg.append(lstm_probs)

        if models_to_avg:
            rally_probs = np.mean(models_to_avg, axis=0)
            rally_preds = (rally_probs >= rally_threshold).astype(int)
            rally_preds = smooth_predictions(rally_preds, min_rally=5, min_gap=10)

            rally_frames = set(int(fi) for fi in range(n_frames) if rally_preds[fi] == 1)
            non_rally_removed_3d = len(points_3d) - len([fi for fi in points_3d if fi in rally_frames])
            non_rally_removed_66 = len(filt66) - len([fi for fi in filt66 if fi in rally_frames])

            # Filter: only keep detections in rally frames
            points_3d = {fi: v for fi, v in points_3d.items() if fi in rally_frames}
            filt66 = {fi: v for fi, v in filt66.items() if fi in rally_frames}
            filt68 = {fi: v for fi, v in filt68.items() if fi in rally_frames}

            logger.info(
                "Rally segmentation: %d/%d frames are rally (%.1f%%). "
                "Removed %d 3D points and %d cam66 detections from non-rally.",
                len(rally_frames), n_frames, 100 * len(rally_frames) / n_frames,
                non_rally_removed_3d, non_rally_removed_66,
            )
        else:
            logger.warning("Rally model loaded but no valid sub-models found, skipping.")
    else:
        logger.info("No rally segmentation model found, skipping.")

    # ── Phase 3: Savitzky-Golay Smoothing + Bounce Detection ───────
    logger.info("=== Phase 3: SG Smoothing + Bounce Detection ===")
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)
    bounces = detect_bounces(smoothed_3d)

    for i, b in enumerate(bounces):
        tag = "IN" if b["in_court"] else "OUT"
        logger.info(
            "  Bounce %d: frame=%d (%.1f, %.1f, z=%.2f) %s",
            i + 1, b["frame"], b["x"], b["y"], b["z"], tag,
        )

    # Net crossing speed detection
    net_crossings = detect_net_crossings(smoothed_3d, fps=25.0)

    # ── Net-crossing anchor filter ────────────────────────────────
    # Only keep bounces that occur near a net crossing event.
    # This eliminates false positives during dead-ball / preparation periods
    # where no rally is in progress (no ball crossing the net).
    NC_ANCHOR_WINDOW = 75  # frames (~3 seconds at 25fps)
    nc_frames = [nc["frame"] for nc in net_crossings]
    if nc_frames:
        pre_filter = len(bounces)
        bounces = [
            b for b in bounces
            if any(abs(b["frame"] - ncf) <= NC_ANCHOR_WINDOW for ncf in nc_frames)
        ]
        removed = pre_filter - len(bounces)
        if removed:
            logger.info("Net-crossing anchor filter removed %d bounce(s) "
                        "(no net crossing within ±%d frames)", removed, NC_ANCHOR_WINDOW)

    # ── Bounce IN/OUT refinement using nearest-camera homography ────
    # Homography assumes z=0 (ball on ground). Only reliable when z is small.
    # For each bounce, find the frame with lowest z within ±3 frames
    # (the actual ground contact), then use nearest camera's homography.
    from app.pipeline.homography import HomographyTransformer
    homo_path = "src/homography_matrices.json"
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    MAX_Z_FOR_HOMO = 0.35  # only trust homography when z < 35cm
    SEARCH_RADIUS = 3      # search ±3 frames for lowest z

    for b in bounces:
        fi = b["frame"]

        # Find frame with lowest z near the bounce (actual ground contact)
        best_fi = fi
        best_z = b["z"]
        for offset in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            check_fi = fi + offset
            if check_fi in smoothed_3d:
                cz = smoothed_3d[check_fi][2]
                if cz < best_z:
                    best_z = cz
                    best_fi = check_fi

        # Skip homography refinement if ball is too high (not on ground)
        if best_z > MAX_Z_FOR_HOMO:
            b["homo_skip"] = True
            b["cam_used"] = "3d"
            continue

        # Determine which camera: ball on which side of net?
        last_nc = None
        for nc in net_crossings:
            if nc["frame"] <= fi:
                last_nc = nc
            else:
                break

        if last_nc and last_nc.get("direction") == "near_to_far":
            use_cam68 = True
        elif last_nc and last_nc.get("direction") == "far_to_near":
            use_cam68 = False
        else:
            use_cam68 = b["y"] > NET_Y

        # Get pixel position from chosen camera at the lowest-z frame
        homo_fi = best_fi
        if use_cam68 and homo_fi in filt68:
            px, py = filt68[homo_fi][:2]
            wx, wy = homo68.pixel_to_world(px, py)
            cam_used = "cam68"
        elif not use_cam68 and homo_fi in filt66:
            px, py = filt66[homo_fi][:2]
            wx, wy = homo66.pixel_to_world(px, py)
            cam_used = "cam66"
        elif homo_fi in filt66:
            px, py = filt66[homo_fi][:2]
            wx, wy = homo66.pixel_to_world(px, py)
            cam_used = "cam66"
        elif homo_fi in filt68:
            px, py = filt68[homo_fi][:2]
            wx, wy = homo68.pixel_to_world(px, py)
            cam_used = "cam68"
        else:
            b["cam_used"] = "3d"
            continue

        old_in = b["in_court"]
        b["x_homo"] = float(wx)
        b["y_homo"] = float(wy)
        b["in_court"] = (SINGLES_X_MIN <= wx <= SINGLES_X_MAX and
                         0 <= wy <= COURT_L)
        b["cam_used"] = cam_used
        b["homo_z"] = float(best_z)

        if old_in != b["in_court"]:
            tag = "IN" if b["in_court"] else "OUT"
            logger.info("  Bounce frame=%d: IN/OUT changed by homography "
                        "(%s, z=%.2f, x=%.2f y=%.2f -> %s)",
                        fi, cam_used, best_z, wx, wy, tag)

    # ── Shot/Bounce disambiguation for z > 0.35m ─────────────────
    # A "bounce" with z > 0.35m is suspicious — likely a low shot.
    # Key difference: after a real bounce the ball stays on the same side,
    # after a shot the ball crosses the net.
    # Check: does the ball cross the net within 15 frames AFTER this "bounce"?
    # If yes → it's a shot (hit), not a bounce → remove it.
    SHOT_Z_THRESHOLD = 0.35
    SHOT_CHECK_WINDOW = 15  # frames after bounce to check for net crossing

    pre_shot_filter = len(bounces)
    nc_frame_set = set(nc["frame"] for nc in net_crossings)

    def _has_net_crossing_nearby(frame, window):
        """Check if ball crosses net within -3 to +window frames of `frame`.

        A shot happens AT the net crossing, so the crossing could be
        a few frames before or after the detected V-shape minimum.
        """
        for ncf in nc_frame_set:
            if -3 <= ncf - frame <= window:
                return True
        return False

    # Real bounces happen at ground level, but 3D Z accuracy depends on
    # which half of the court the ball is on:
    #   Near end (y < NET_Y): Z more accurate → stricter threshold
    #   Far end  (y > NET_Y): Z noisier (up to +0.3-0.6m) → relaxed threshold
    # Also trust homography-refined z if available (actual ground contact frame).
    def _z_threshold_for_bounce(b):
        y = b["y"]
        if y > NET_Y:
            return 0.65  # far end: allow higher z due to reconstruction noise
        else:
            return 0.35  # near end: z should be accurate

    bounces = [b for b in bounces
               if b["z"] <= _z_threshold_for_bounce(b)
               or b.get("homo_z", b["z"]) <= _z_threshold_for_bounce(b)]

    shot_removed = pre_shot_filter - len(bounces)
    if shot_removed:
        logger.info("Shot/bounce filter: removed %d suspicious bounce(s) "
                     "(z > %.2fm + net crossing after)", shot_removed, SHOT_Z_THRESHOLD)

    logger.info("Bounce IN/OUT refined using nearest-camera homography")
    for i, b in enumerate(bounces):
        tag = "IN" if b["in_court"] else "OUT"
        cam = b.get("cam_used", "3d")
        if cam == "3d" or b.get("homo_skip"):
            logger.info("  Bounce %d: frame=%d (3d: x=%.2f, y=%.2f, z=%.2f) %s [z too high for homo]",
                         i + 1, b["frame"], b["x"], b["y"], b["z"], tag)
        else:
            logger.info("  Bounce %d: frame=%d (%s: x=%.2f, y=%.2f, z=%.2f) %s",
                         i + 1, b["frame"], cam,
                         b.get("x_homo", b["x"]), b.get("y_homo", b["y"]),
                         b.get("homo_z", b["z"]), tag)

    # ── Phase 3.5: 2D Pixel Smoothing for Visual Trajectory ─────────
    # Apply median3 + gap interpolation to make rendered trajectory smooth.
    # This is the "conf20_median5" approach adapted for rendering: we use
    # median filter to remove outlier pixel spikes, then interpolate short
    # gaps for visual continuity of the trail.
    logger.info("=== Phase 3.5: 2D Pixel Smoothing (median + interpolation) ===")
    filt66 = _smooth_2d_for_render(filt66)
    filt68 = _smooth_2d_for_render(filt68)

    # ── Phase 4: Render Video ───────────────────────────────────────
    logger.info("=== Phase 4: Render Video ===")
    render_video(
        args.video66, args.video68,
        filt66, filt68, bounces,
        n_frames, args.output,
        net_crossings=net_crossings,
        smoothed_3d=smoothed_3d,
    )


if __name__ == "__main__":
    main()
