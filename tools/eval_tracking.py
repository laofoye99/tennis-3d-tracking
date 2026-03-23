"""Standardized tracking evaluation framework.

Runs multiple tracking methods against cam66 GT annotations and computes:
- Recall, Precision, F1 (with 30px matching threshold)
- Pixel accuracy (mean, median, <5px, <10px, <20px)
- Jitter and large jump metrics

Methods:
  top1          : Top-1 TrackNet blob, no cross-camera matching
  multi         : MultiBlobMatcher (default params)
  multi_strict  : MultiBlobMatcher with blob_rank_penalty=1.0
  multi_loose   : MultiBlobMatcher with blob_rank_penalty=0.2
  viterbi       : Viterbi global optimization
  top1_smooth   : Top-1 blob + Savitzky-Golay smoothing on 2D pixels

Usage:
    python -m tools.eval_tracking --method all
    python -m tools.eval_tracking --method multi --method viterbi
"""

import argparse
import logging
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GT_DIR = "uploads/cam66_20260307_173403_2min"
VIDEO66 = "uploads/cam66_20260307_173403_2min.mp4"
VIDEO68 = "uploads/cam68_20260307_173403_2min.mp4"
MAX_FRAMES = 1800
CACHE_DIR = Path(".cache")
MATCH_THRESHOLD_PX = 30.0

ALL_METHODS = [
    "top1", "multi", "multi_strict", "multi_loose", "viterbi", "top1_smooth",
    "top1_xcam", "top1_kalman", "multi_smooth", "top1_confidence",
    "top1_kalman_xcam", "multi_kalman",
    "conf_xcam", "conf_kalman", "conf_xcam_kalman",
    "conf_xcam_gate", "top1_conf20", "top1_conf30", "top1_conf40",
    # New smoothing methods
    "top1_sg5_p2", "top1_sg7_p2", "top1_sg9_p2", "top1_sg11_p2",
    "top1_sg5_p3", "top1_sg7_p3", "top1_sg9_p3", "top1_sg11_p3",
    "top1_ema03", "top1_ema05", "top1_ema07", "top1_ema08",
    "top1_kalman_tuned",
    "conf20_kalman", "conf20_kalman_tuned",
    "top1_bilateral",
    "top1_conf_weighted",
    "top1_median3", "top1_median5", "top1_median7",
    "conf20_sg7_p2", "conf20_sg9_p3", "conf20_ema05", "conf20_ema07",
    "conf20_bilateral", "conf20_median5",
    # Round 2: focused combos
    "top1_med3_bilateral", "conf20_med3_bilateral",
    "top1_light_kalman", "conf20_light_kalman",
    "top1_med3_ema08", "conf20_med3_ema08",
    "top1_adaptive_ema", "conf20_adaptive_ema",
    # Round 3: final optimized methods
    "top1_median3_sg5_p2", "conf20_median3_sg5_p2",
    "top1_median5_bilateral", "conf20_median5_bilateral",
    "best_smooth",
]


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gt(gt_dir: str, max_frames: int) -> dict:
    """Load GT annotations. Returns {frame: (px, py)} for match_ball frames."""
    import json

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


def run_detection_cached(cfg):
    """Run TrackNet detection on both cameras, with disk caching."""
    cache_path = CACHE_DIR / "detection_cache.pkl"
    if cache_path.exists():
        logger.info("Loading cached detections from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from tools.render_tracking_video import build_detector, run_detection_multi

    logger.info("Running TrackNet detection (will cache results)...")
    detector, postproc = build_detector(cfg)

    logger.info("--- cam66 ---")
    multi66, det66, n66 = run_detection_multi(VIDEO66, detector, postproc, MAX_FRAMES, top_k=2)

    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("--- cam68 ---")
    multi68, det68, n68 = run_detection_multi(VIDEO68, detector, postproc, MAX_FRAMES, top_k=2)

    result = {
        "multi66": multi66, "det66": det66,
        "multi68": multi68, "det68": det68,
    }

    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info("Cached detections to %s", cache_path)

    return result


# ── Tracking methods ──────────────────────────────────────────────────


def method_top1(det66, multi66, multi68, cfg):
    """Top-1 blob from TrackNet, no cross-camera matching.

    Returns {frame: (px, py)} for cam66.
    """
    return {fi: (px, py) for fi, (px, py, _) in det66.items()}


def method_multi(multi66, multi68, cfg, blob_rank_penalty=0.5):
    """MultiBlobMatcher with given blob_rank_penalty.

    Returns {frame: (px, py)} for cam66 (chosen pixel from matching).
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
        blob_rank_penalty=blob_rank_penalty,
        lost_timeout=30,
        max_velocity=50.0,
        history_size=5,
        fps=25.0,
    )

    common = sorted(set(multi66.keys()) & set(multi68.keys()))
    results = {}

    for fi in common:
        blobs66 = multi66[fi]
        blobs68 = multi68[fi]

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
            px, py = result["cam1_pixel"]
            results[fi] = (px, py)

    return results


def method_viterbi(multi66, multi68, cfg):
    """Viterbi global optimization.

    Returns {frame: (px, py)} for cam66.
    """
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.viterbi_tracker import ViterbiTracker

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    tracker = ViterbiTracker(
        cam1_pos=cam66_pos,
        cam2_pos=cam68_pos,
        max_ray_distance=2.5,
        valid_z_range=(0.0, 8.0),
        fps=25.0,
        gap_threshold=5,
    )

    points_3d, chosen_pixels, stats = tracker.track(multi66, multi68, homo66, homo68)

    results = {}
    for fi, pix in chosen_pixels.items():
        px, py = pix["cam66"]
        results[fi] = (px, py)

    return results


def method_top1_smooth(det66, multi66, multi68, cfg):
    """Top-1 blob + Savitzky-Golay smoothing on 2D pixel coordinates.

    Returns {frame: (px, py)} for cam66.
    """
    from scipy.signal import savgol_filter

    frames = sorted(det66.keys())
    if len(frames) < 11:
        return {fi: (det66[fi][0], det66[fi][1]) for fi in frames}

    # Split into continuous segments (gap > 3 frames = new segment)
    MAX_GAP = 3
    WINDOW = 11
    POLY = 3

    segments = []
    seg_start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > MAX_GAP:
            segments.append(frames[seg_start:i])
            seg_start = i
    segments.append(frames[seg_start:])

    results = {}
    for seg_frames in segments:
        xs = np.array([det66[fi][0] for fi in seg_frames])
        ys = np.array([det66[fi][1] for fi in seg_frames])

        if len(seg_frames) >= WINDOW:
            xs_s = savgol_filter(xs, WINDOW, POLY)
            ys_s = savgol_filter(ys, WINDOW, POLY)
        else:
            xs_s = xs
            ys_s = ys

        for i, fi in enumerate(seg_frames):
            results[fi] = (float(xs_s[i]), float(ys_s[i]))

    return results


def _ray_distance(wx1, wy1, cam1_pos, wx2, wy2, cam2_pos):
    """Compute distance between two rays at their closest point."""
    cam1 = np.asarray(cam1_pos, dtype=np.float64)
    cam2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([wx1, wy1, 0.0])
    g2 = np.array([wx2, wy2, 0.0])
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
        return 99.0
    s = np.clip((b * e - c * d_val) / denom, 0, 1)
    t = np.clip((a * e - b * d_val) / denom, 0, 1)
    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    return float(np.linalg.norm(p1 - p2))


def method_top1_xcam(det66, multi66, multi68, cfg):
    """Top-1 + cross-camera verification.

    Only keep frames where both cameras detect something and the two detections
    agree on a reasonable 3D position (ray_distance < threshold).
    """
    from app.pipeline.homography import HomographyTransformer

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")
    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    results = {}
    common = sorted(set(det66.keys()) & set(multi68.keys()))
    for fi in common:
        px66, py66, _ = det66[fi]
        if fi not in multi68 or not multi68[fi]:
            continue
        b68 = multi68[fi][0]
        wx66, wy66 = homo66.pixel_to_world(px66, py66)
        wx68, wy68 = homo68.pixel_to_world(b68["pixel_x"], b68["pixel_y"])
        rd = _ray_distance(wx66, wy66, cam66_pos, wx68, wy68, cam68_pos)
        if rd < 2.0:
            results[fi] = (px66, py66)
    return results


def method_top1_kalman(det66, multi66, multi68, cfg):
    """Top-1 + Kalman filter prediction.

    Use Kalman filter to predict next position. If detection is too far from
    prediction, reject it (likely a shoe/reflection). Also smooth trajectory.
    """
    frames = sorted(det66.keys())
    if not frames:
        return {}

    # Simple 2D Kalman filter (position + velocity)
    # State: [x, y, vx, vy], Measurement: [x, y]
    dt = 1.0 / 25.0
    # State transition
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    # Noise
    q = 50.0  # process noise (pixels/frame²)
    r = 5.0   # measurement noise (pixels)
    Q = np.eye(4) * q
    Q[0, 0] = Q[1, 1] = q * 0.1
    R = np.eye(2) * r ** 2
    P = np.eye(4) * 100.0

    # Init state from first detection
    px0, py0, _ = det66[frames[0]]
    x = np.array([px0, py0, 0, 0], dtype=np.float64)

    results = {}
    reject_threshold = 200.0  # pixels — generous, only reject extreme jumps
    last_frame = frames[0]

    for fi in frames:
        gap = fi - last_frame
        px, py, _ = det66[fi]
        meas = np.array([px, py], dtype=np.float64)

        if gap > 10:
            x = np.array([px, py, 0, 0], dtype=np.float64)
            P = np.eye(4) * 100.0
            results[fi] = (px, py)
            last_frame = fi
            continue

        F_gap = F.copy()
        F_gap[0, 2] = dt * gap
        F_gap[1, 3] = dt * gap
        x_pred = F_gap @ x
        P_pred = F_gap @ P @ F_gap.T + Q * gap

        y_innov = meas - H @ x_pred
        dist = np.linalg.norm(y_innov)

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        if dist < reject_threshold:
            x = x_pred + K @ y_innov
            P = (np.eye(4) - K @ H) @ P_pred
            # Output smoothed (Kalman-filtered) position
            results[fi] = (float(x[0]), float(x[1]))
        else:
            # Extreme jump — reset to measurement
            x = np.array([px, py, 0, 0], dtype=np.float64)
            P = np.eye(4) * 100.0
            results[fi] = (px, py)

        last_frame = fi

    return results


def method_multi_smooth(multi66, multi68, cfg):
    """MultiBlobMatcher + post-hoc SG smoothing on 2D pixels."""
    from scipy.signal import savgol_filter

    raw = method_multi(multi66, multi68, cfg, blob_rank_penalty=0.5)
    frames = sorted(raw.keys())
    if len(frames) < 7:
        return raw

    # Split into segments
    MAX_GAP = 3
    WINDOW = 7
    POLY = 2
    segments = []
    seg_start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > MAX_GAP:
            segments.append(frames[seg_start:i])
            seg_start = i
    segments.append(frames[seg_start:])

    results = {}
    for seg in segments:
        xs = np.array([raw[fi][0] for fi in seg])
        ys = np.array([raw[fi][1] for fi in seg])
        if len(seg) >= WINDOW:
            xs_s = savgol_filter(xs, WINDOW, POLY)
            ys_s = savgol_filter(ys, WINDOW, POLY)
        else:
            xs_s, ys_s = xs, ys
        for i, fi in enumerate(seg):
            results[fi] = (float(xs_s[i]), float(ys_s[i]))
    return results


def method_top1_confidence(det66, multi66, multi68, cfg):
    """Top-1 + confidence filter.

    Only keep detections where blob_sum (heatmap response) is above a threshold.
    Low confidence blobs are more likely to be noise/shoes.
    """
    # Collect all blob_sums to find a good threshold
    sums = []
    for fi, blobs in multi66.items():
        if blobs:
            sums.append(blobs[0]["blob_sum"])

    if not sums:
        return {}

    # Use 25th percentile as threshold — remove weakest 25% of detections
    threshold = float(np.percentile(sums, 25))

    results = {}
    for fi, blobs in multi66.items():
        if blobs and blobs[0]["blob_sum"] >= threshold:
            results[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"])
    return results


def method_top1_kalman_xcam(det66, multi66, multi68, cfg):
    """Top-1 + Kalman filter + cross-camera verification.

    Best of both worlds: Kalman predicts and smooths, cross-camera verifies.
    """
    from app.pipeline.homography import HomographyTransformer

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")
    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    kalman_result = method_top1_kalman(det66, multi66, multi68, cfg)

    results = {}
    for fi, (kpx, kpy) in kalman_result.items():
        if fi in multi68 and multi68[fi]:
            b68 = multi68[fi][0]
            wx66, wy66 = homo66.pixel_to_world(kpx, kpy)
            wx68, wy68 = homo68.pixel_to_world(b68["pixel_x"], b68["pixel_y"])
            rd = _ray_distance(wx66, wy66, cam66_pos, wx68, wy68, cam68_pos)
            if rd < 2.5:
                results[fi] = (kpx, kpy)
            elif fi in det66:
                raw_px, raw_py, _ = det66[fi]
                if abs(kpx - raw_px) < 30 and abs(kpy - raw_py) < 30:
                    results[fi] = (kpx, kpy)
        else:
            if fi in det66:
                raw_px, raw_py, _ = det66[fi]
                if abs(kpx - raw_px) < 30 and abs(kpy - raw_py) < 30:
                    results[fi] = (kpx, kpy)
    return results


def method_confidence_xcam(det66, multi66, multi68, cfg):
    """Confidence filter + cross-camera verification.

    Combines the two best methods: filter low-confidence blobs, then verify
    with cross-camera ray distance.
    """
    from app.pipeline.homography import HomographyTransformer

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")
    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    # Confidence threshold
    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    threshold = float(np.percentile(sums, 25)) if sums else 0

    results = {}
    for fi, blobs in multi66.items():
        if not blobs or blobs[0]["blob_sum"] < threshold:
            continue
        px66, py66 = blobs[0]["pixel_x"], blobs[0]["pixel_y"]

        if fi in multi68 and multi68[fi]:
            b68 = multi68[fi][0]
            wx66, wy66 = homo66.pixel_to_world(px66, py66)
            wx68, wy68 = homo68.pixel_to_world(b68["pixel_x"], b68["pixel_y"])
            rd = _ray_distance(wx66, wy66, cam66_pos, wx68, wy68, cam68_pos)
            if rd < 2.0:
                results[fi] = (px66, py66)
        # No cam68 data — still include if confidence is high enough
        elif blobs[0]["blob_sum"] >= float(np.percentile(sums, 50)):
            results[fi] = (px66, py66)
    return results


def method_confidence_kalman(det66, multi66, multi68, cfg):
    """Confidence filter + Kalman smoothing.

    Filter low-confidence, then smooth with Kalman.
    """
    # First get confidence-filtered detections
    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    threshold = float(np.percentile(sums, 25)) if sums else 0

    filtered_det66 = {}
    for fi, blobs in multi66.items():
        if blobs and blobs[0]["blob_sum"] >= threshold:
            filtered_det66[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"], blobs[0]["blob_sum"])

    # Run Kalman on filtered detections
    return method_top1_kalman(filtered_det66, multi66, multi68, cfg)


def method_confidence_xcam_kalman(det66, multi66, multi68, cfg):
    """Confidence + cross-camera + Kalman smoothing.

    Triple stack: confidence filter → cross-camera verify → Kalman smooth.
    """
    from app.pipeline.homography import HomographyTransformer

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")
    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    threshold = float(np.percentile(sums, 20)) if sums else 0

    # Step 1: confidence + xcam filter
    verified = {}
    for fi, blobs in multi66.items():
        if not blobs or blobs[0]["blob_sum"] < threshold:
            continue
        px66, py66 = blobs[0]["pixel_x"], blobs[0]["pixel_y"]

        if fi in multi68 and multi68[fi]:
            b68 = multi68[fi][0]
            wx66, wy66 = homo66.pixel_to_world(px66, py66)
            wx68, wy68 = homo68.pixel_to_world(b68["pixel_x"], b68["pixel_y"])
            rd = _ray_distance(wx66, wy66, cam66_pos, wx68, wy68, cam68_pos)
            if rd < 2.5:
                verified[fi] = (px66, py66, blobs[0]["blob_sum"])
        elif blobs[0]["blob_sum"] >= float(np.percentile(sums, 50)):
            verified[fi] = (px66, py66, blobs[0]["blob_sum"])

    # Step 2: Kalman smooth
    return method_top1_kalman(verified, multi66, multi68, cfg)


def method_conf_xcam_gate(det66, multi66, multi68, cfg):
    """Confidence + cross-camera + Kalman gate (not smoothing).

    Use Kalman to REJECT outlier jumps, but output the RAW pixel position
    (not the smoothed Kalman state). This preserves pixel accuracy while
    removing large jumps.
    """
    from app.pipeline.homography import HomographyTransformer

    homo_path = cfg["homography"]["path"]
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")
    cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
    cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    conf_threshold = float(np.percentile(sums, 25)) if sums else 0

    # Step 1: confidence + xcam filter (same as conf_xcam)
    verified = {}
    for fi, blobs in multi66.items():
        if not blobs or blobs[0]["blob_sum"] < conf_threshold:
            continue
        px66, py66 = blobs[0]["pixel_x"], blobs[0]["pixel_y"]
        keep = False
        if fi in multi68 and multi68[fi]:
            b68 = multi68[fi][0]
            wx66, wy66 = homo66.pixel_to_world(px66, py66)
            wx68, wy68 = homo68.pixel_to_world(b68["pixel_x"], b68["pixel_y"])
            rd = _ray_distance(wx66, wy66, cam66_pos, wx68, wy68, cam68_pos)
            if rd < 2.0:
                keep = True
        # High confidence without cam68 data
        if not keep and blobs[0]["blob_sum"] >= float(np.percentile(sums, 50)):
            keep = True
        if keep:
            verified[fi] = (px66, py66)

    # Step 2: Kalman GATE — only reject extreme jumps, output raw pixels
    frames = sorted(verified.keys())
    if not frames:
        return {}

    dt = 1.0 / 25.0
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
    Q = np.eye(4) * 80.0; Q[0,0] = Q[1,1] = 8.0
    R = np.eye(2) * 5.0**2
    P = np.eye(4) * 200.0

    px0, py0 = verified[frames[0]]
    x = np.array([px0, py0, 0, 0], dtype=np.float64)
    results = {}
    last_fi = frames[0]
    gate_threshold = 150.0  # only reject truly extreme jumps

    for fi in frames:
        gap = fi - last_fi
        px, py = verified[fi]
        meas = np.array([px, py], dtype=np.float64)

        if gap > 10:
            x = np.array([px, py, 0, 0], dtype=np.float64)
            P = np.eye(4) * 200.0
            results[fi] = (px, py)
            last_fi = fi
            continue

        F_g = F.copy(); F_g[0,2] = dt*gap; F_g[1,3] = dt*gap
        x_pred = F_g @ x
        P_pred = F_g @ P @ F_g.T + Q * gap
        y_innov = meas - H @ x_pred
        dist = np.linalg.norm(y_innov)

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        if dist < gate_threshold:
            # Update Kalman state but OUTPUT raw pixel (not smoothed)
            x = x_pred + K @ y_innov
            P = (np.eye(4) - K @ H) @ P_pred
            results[fi] = (px, py)  # RAW pixel, not Kalman output
        else:
            # Extreme jump — skip this frame
            x = x_pred
            P = P_pred

        last_fi = fi

    return results


def method_top1_conf20(det66, multi66, multi68, cfg):
    """Top-1 with top-20% confidence filter (stricter)."""
    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    if not sums:
        return {}
    threshold = float(np.percentile(sums, 20))
    results = {}
    for fi, blobs in multi66.items():
        if blobs and blobs[0]["blob_sum"] >= threshold:
            results[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"])
    return results


def method_top1_conf30(det66, multi66, multi68, cfg):
    """Top-1 with top-30% confidence filter (stricter)."""
    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    if not sums:
        return {}
    threshold = float(np.percentile(sums, 30))
    results = {}
    for fi, blobs in multi66.items():
        if blobs and blobs[0]["blob_sum"] >= threshold:
            results[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"])
    return results


def method_top1_conf40(det66, multi66, multi68, cfg):
    """Top-1 with top-40% confidence filter."""
    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    if not sums:
        return {}
    threshold = float(np.percentile(sums, 40))
    results = {}
    for fi, blobs in multi66.items():
        if blobs and blobs[0]["blob_sum"] >= threshold:
            results[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"])
    return results


def _segment_frames(frames, max_gap=3):
    """Split frame list into continuous segments (gap > max_gap = new segment)."""
    if not frames:
        return []
    segments = []
    seg_start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > max_gap:
            segments.append(frames[seg_start:i])
            seg_start = i
    segments.append(frames[seg_start:])
    return segments


def _apply_sg(raw, window, polyorder, max_gap=3):
    """Apply Savitzky-Golay filter to a raw {frame: (px, py)} trajectory."""
    from scipy.signal import savgol_filter

    frames = sorted(raw.keys())
    if len(frames) < window:
        return dict(raw)

    segments = _segment_frames(frames, max_gap)
    results = {}
    for seg in segments:
        xs = np.array([raw[fi][0] for fi in seg])
        ys = np.array([raw[fi][1] for fi in seg])
        if len(seg) >= window:
            xs_s = savgol_filter(xs, window, polyorder)
            ys_s = savgol_filter(ys, window, polyorder)
        else:
            xs_s, ys_s = xs, ys
        for i, fi in enumerate(seg):
            results[fi] = (float(xs_s[i]), float(ys_s[i]))
    return results


def _apply_ema(raw, alpha, max_gap=3):
    """Apply exponential moving average to a raw {frame: (px, py)} trajectory."""
    frames = sorted(raw.keys())
    if not frames:
        return {}

    segments = _segment_frames(frames, max_gap)
    results = {}
    for seg in segments:
        sx, sy = raw[seg[0]]
        results[seg[0]] = (sx, sy)
        for fi in seg[1:]:
            nx, ny = raw[fi]
            sx = alpha * nx + (1 - alpha) * sx
            sy = alpha * ny + (1 - alpha) * sy
            results[fi] = (float(sx), float(sy))
    return results


def _apply_kalman(raw, q=50.0, r=5.0, reject_thresh=200.0, max_gap=10, output_smoothed=True):
    """Apply 2D Kalman filter (constant velocity) to a raw {frame: (px, py)} trajectory.

    If output_smoothed=True, output Kalman state estimate. Otherwise output raw pixel
    (use Kalman only for gating/rejection).
    """
    frames = sorted(raw.keys())
    if not frames:
        return {}

    dt = 1.0 / 25.0
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    Q_mat = np.eye(4) * q
    Q_mat[0, 0] = Q_mat[1, 1] = q * 0.1
    R_mat = np.eye(2) * r ** 2
    P = np.eye(4) * 100.0

    px0, py0 = raw[frames[0]][0], raw[frames[0]][1]
    x = np.array([px0, py0, 0, 0], dtype=np.float64)
    results = {}
    last_fi = frames[0]

    for fi in frames:
        gap = fi - last_fi
        px, py = raw[fi][0], raw[fi][1]
        meas = np.array([px, py], dtype=np.float64)

        if gap > max_gap:
            x = np.array([px, py, 0, 0], dtype=np.float64)
            P = np.eye(4) * 100.0
            results[fi] = (px, py)
            last_fi = fi
            continue

        F_g = F.copy()
        F_g[0, 2] = dt * gap
        F_g[1, 3] = dt * gap
        x_pred = F_g @ x
        P_pred = F_g @ P @ F_g.T + Q_mat * gap

        y_innov = meas - H @ x_pred
        dist = np.linalg.norm(y_innov)
        S = H @ P_pred @ H.T + R_mat
        K = P_pred @ H.T @ np.linalg.inv(S)

        if dist < reject_thresh:
            x = x_pred + K @ y_innov
            P = (np.eye(4) - K @ H) @ P_pred
            if output_smoothed:
                results[fi] = (float(x[0]), float(x[1]))
            else:
                results[fi] = (px, py)
        else:
            x = np.array([px, py, 0, 0], dtype=np.float64)
            P = np.eye(4) * 100.0
            results[fi] = (px, py)

        last_fi = fi

    return results


def _apply_median(raw, kernel_size, max_gap=3):
    """Apply median filter to a raw {frame: (px, py)} trajectory."""
    frames = sorted(raw.keys())
    if not frames:
        return {}

    segments = _segment_frames(frames, max_gap)
    results = {}
    half = kernel_size // 2

    for seg in segments:
        xs = [raw[fi][0] for fi in seg]
        ys = [raw[fi][1] for fi in seg]
        for i, fi in enumerate(seg):
            lo = max(0, i - half)
            hi = min(len(seg), i + half + 1)
            results[fi] = (float(np.median(xs[lo:hi])), float(np.median(ys[lo:hi])))
    return results


def _apply_bilateral(raw, max_gap=3, jump_threshold=50.0, smooth_threshold=20.0):
    """Bilateral-like filter: smooth small movements, preserve large jumps (bounces/hits)."""
    frames = sorted(raw.keys())
    if not frames:
        return {}

    segments = _segment_frames(frames, max_gap)
    results = {}

    for seg in segments:
        if len(seg) < 3:
            for fi in seg:
                results[fi] = raw[fi]
            continue

        xs = np.array([raw[fi][0] for fi in seg])
        ys = np.array([raw[fi][1] for fi in seg])

        xs_out = xs.copy()
        ys_out = ys.copy()

        for i in range(1, len(seg) - 1):
            # Check if this is a direction change (large jump)
            dx_prev = xs[i] - xs[i - 1]
            dy_prev = ys[i] - ys[i - 1]
            dx_next = xs[i + 1] - xs[i]
            dy_next = ys[i + 1] - ys[i]
            disp_prev = math.hypot(dx_prev, dy_prev)
            disp_next = math.hypot(dx_next, dy_next)

            if disp_prev > jump_threshold or disp_next > jump_threshold:
                # Large jump — preserve raw position (bounce/hit)
                xs_out[i] = xs[i]
                ys_out[i] = ys[i]
            elif disp_prev < smooth_threshold and disp_next < smooth_threshold:
                # Small movement — smooth heavily (weighted average of neighbors)
                # Use a 5-point window if available
                lo = max(0, i - 2)
                hi = min(len(seg), i + 3)
                weights = []
                xvals = []
                yvals = []
                for j in range(lo, hi):
                    d = abs(j - i)
                    w = 1.0 / (1.0 + d)
                    weights.append(w)
                    xvals.append(xs[j])
                    yvals.append(ys[j])
                wsum = sum(weights)
                xs_out[i] = sum(w * x for w, x in zip(weights, xvals)) / wsum
                ys_out[i] = sum(w * y for w, y in zip(weights, yvals)) / wsum
            # else: moderate movement — keep raw

        for i, fi in enumerate(seg):
            results[fi] = (float(xs_out[i]), float(ys_out[i]))

    return results


def _apply_conf_weighted(raw_with_conf, max_gap=3, window=5):
    """Confidence-weighted smoothing.

    raw_with_conf: {frame: (px, py, blob_sum)}
    High-confidence frames trust raw position more; low-confidence rely on neighbors.
    """
    frames = sorted(raw_with_conf.keys())
    if not frames:
        return {}

    # Normalize confidences
    confs = np.array([raw_with_conf[fi][2] for fi in frames])
    if confs.max() > confs.min():
        norm_confs = (confs - confs.min()) / (confs.max() - confs.min())
    else:
        norm_confs = np.ones_like(confs)

    conf_map = {fi: nc for fi, nc in zip(frames, norm_confs)}

    segments = _segment_frames(frames, max_gap)
    results = {}
    half = window // 2

    for seg in segments:
        xs = np.array([raw_with_conf[fi][0] for fi in seg])
        ys = np.array([raw_with_conf[fi][1] for fi in seg])

        for i, fi in enumerate(seg):
            c = conf_map[fi]
            # alpha = how much to trust raw position (higher conf = more trust)
            alpha = 0.3 + 0.7 * c  # range [0.3, 1.0]

            # Compute local average for smoothed position
            lo = max(0, i - half)
            hi = min(len(seg), i + half + 1)
            avg_x = np.mean(xs[lo:hi])
            avg_y = np.mean(ys[lo:hi])

            results[fi] = (
                float(alpha * xs[i] + (1 - alpha) * avg_x),
                float(alpha * ys[i] + (1 - alpha) * avg_y),
            )

    return results


def _get_conf20_raw(multi66):
    """Get top-1 detections filtered by 20th percentile confidence."""
    sums = [b[0]["blob_sum"] for b in multi66.values() if b]
    if not sums:
        return {}
    threshold = float(np.percentile(sums, 20))
    results = {}
    for fi, blobs in multi66.items():
        if blobs and blobs[0]["blob_sum"] >= threshold:
            results[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"])
    return results


# ── New SG smoothing variants ────────────────────────────────────────────

def method_top1_sg(det66, multi66, multi68, cfg, window, polyorder):
    """Top-1 + SG filter with configurable window and polyorder."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_sg(raw, window, polyorder)


# ── EMA variants ─────────────────────────────────────────────────────────

def method_top1_ema(det66, multi66, multi68, cfg, alpha):
    """Top-1 + EMA with configurable alpha."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_ema(raw, alpha)


# ── Kalman tuned ─────────────────────────────────────────────────────────

def method_top1_kalman_tuned(det66, multi66, multi68, cfg):
    """Top-1 + Kalman with tuned parameters for lower jitter.

    Lower process noise (Q) = smoother trajectory
    Higher measurement noise (R) = trust model more, measurements less
    """
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_kalman(raw, q=20.0, r=8.0, reject_thresh=200.0)


# ── Conf20 + Kalman variants ────────────────────────────────────────────

def method_conf20_kalman(det66, multi66, multi68, cfg):
    """Conf20 filter + default Kalman smoothing."""
    raw = _get_conf20_raw(multi66)
    return _apply_kalman(raw, q=50.0, r=5.0)


def method_conf20_kalman_tuned(det66, multi66, multi68, cfg):
    """Conf20 filter + tuned Kalman (low Q, high R for max smoothness)."""
    raw = _get_conf20_raw(multi66)
    return _apply_kalman(raw, q=15.0, r=10.0, reject_thresh=200.0)


# ── Bilateral filter ────────────────────────────────────────────────────

def method_top1_bilateral(det66, multi66, multi68, cfg):
    """Top-1 + bilateral-like filter (smooth noise, preserve bounces)."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_bilateral(raw)


# ── Confidence-weighted smoothing ────────────────────────────────────────

def method_top1_conf_weighted(det66, multi66, multi68, cfg):
    """Top-1 + confidence-weighted local smoothing."""
    raw_with_conf = {}
    for fi, blobs in multi66.items():
        if blobs:
            raw_with_conf[fi] = (blobs[0]["pixel_x"], blobs[0]["pixel_y"], blobs[0]["blob_sum"])
    return _apply_conf_weighted(raw_with_conf)


# ── Median filter variants ──────────────────────────────────────────────

def method_top1_median(det66, multi66, multi68, cfg, kernel_size):
    """Top-1 + median filter with configurable kernel."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_median(raw, kernel_size)


# ── Conf20 + smoother combos ────────────────────────────────────────────

def method_conf20_sg(det66, multi66, multi68, cfg, window, polyorder):
    """Conf20 + SG filter."""
    raw = _get_conf20_raw(multi66)
    return _apply_sg(raw, window, polyorder)


def method_conf20_ema(det66, multi66, multi68, cfg, alpha):
    """Conf20 + EMA."""
    raw = _get_conf20_raw(multi66)
    return _apply_ema(raw, alpha)


def method_conf20_bilateral(det66, multi66, multi68, cfg):
    """Conf20 + bilateral filter."""
    raw = _get_conf20_raw(multi66)
    return _apply_bilateral(raw)


def method_conf20_median(det66, multi66, multi68, cfg, kernel_size):
    """Conf20 + median filter."""
    raw = _get_conf20_raw(multi66)
    return _apply_median(raw, kernel_size)


def _apply_light_kalman(raw, q=100.0, r=3.0, reject_thresh=200.0, max_gap=10):
    """Light Kalman: high process noise (trust measurements), low meas noise.

    This barely smooths, mainly just rejects extreme outliers.
    """
    return _apply_kalman(raw, q=q, r=r, reject_thresh=reject_thresh, max_gap=max_gap)


def _apply_adaptive_ema(raw, max_gap=3, alpha_fast=0.9, alpha_slow=0.5, speed_threshold=25.0):
    """Adaptive EMA: fast alpha for fast ball motion, slow alpha for slow motion.

    When ball moves fast (>speed_threshold px/frame), use alpha_fast (trust raw more).
    When ball moves slowly, use alpha_slow (smooth more).
    """
    frames = sorted(raw.keys())
    if not frames:
        return {}

    segments = _segment_frames(frames, max_gap)
    results = {}

    for seg in segments:
        sx, sy = raw[seg[0]]
        results[seg[0]] = (sx, sy)
        prev_x, prev_y = sx, sy

        for fi in seg[1:]:
            nx, ny = raw[fi]
            speed = math.hypot(nx - prev_x, ny - prev_y)
            alpha = alpha_fast if speed > speed_threshold else alpha_slow
            sx = alpha * nx + (1 - alpha) * sx
            sy = alpha * ny + (1 - alpha) * sy
            results[fi] = (float(sx), float(sy))
            prev_x, prev_y = nx, ny

    return results


def method_top1_med3_bilateral(det66, multi66, multi68, cfg):
    """Top-1 -> median3 -> bilateral: two-pass smoothing."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    med = _apply_median(raw, 3)
    return _apply_bilateral(med)


def method_conf20_med3_bilateral(det66, multi66, multi68, cfg):
    """Conf20 -> median3 -> bilateral."""
    raw = _get_conf20_raw(multi66)
    med = _apply_median(raw, 3)
    return _apply_bilateral(med)


def method_top1_light_kalman(det66, multi66, multi68, cfg):
    """Top-1 + light Kalman (high Q, mainly outlier rejection)."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_light_kalman(raw)


def method_conf20_light_kalman(det66, multi66, multi68, cfg):
    """Conf20 + light Kalman."""
    raw = _get_conf20_raw(multi66)
    return _apply_light_kalman(raw)


def method_top1_med3_ema08(det66, multi66, multi68, cfg):
    """Top-1 -> median3 -> EMA(0.8): light denoise then light smooth."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    med = _apply_median(raw, 3)
    return _apply_ema(med, 0.8)


def method_conf20_med3_ema08(det66, multi66, multi68, cfg):
    """Conf20 -> median3 -> EMA(0.8)."""
    raw = _get_conf20_raw(multi66)
    med = _apply_median(raw, 3)
    return _apply_ema(med, 0.8)


def method_top1_adaptive_ema(det66, multi66, multi68, cfg):
    """Top-1 + adaptive EMA (smooth slow motion more, fast motion less)."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    return _apply_adaptive_ema(raw)


def method_conf20_adaptive_ema(det66, multi66, multi68, cfg):
    """Conf20 + adaptive EMA."""
    raw = _get_conf20_raw(multi66)
    return _apply_adaptive_ema(raw)


def method_top1_median3_sg5_p2(det66, multi66, multi68, cfg):
    """Top-1 -> median3 (denoise) -> SG(5,2) (light smooth)."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    med = _apply_median(raw, 3)
    return _apply_sg(med, 5, 2)


def method_conf20_median3_sg5_p2(det66, multi66, multi68, cfg):
    """Conf20 -> median3 -> SG(5,2)."""
    raw = _get_conf20_raw(multi66)
    med = _apply_median(raw, 3)
    return _apply_sg(med, 5, 2)


def method_top1_median5_bilateral(det66, multi66, multi68, cfg):
    """Top-1 -> median5 -> bilateral: max smoothness preserving accuracy."""
    raw = {fi: (px, py) for fi, (px, py, _) in det66.items()}
    med = _apply_median(raw, 5)
    return _apply_bilateral(med)


def method_conf20_median5_bilateral(det66, multi66, multi68, cfg):
    """Conf20 -> median5 -> bilateral."""
    raw = _get_conf20_raw(multi66)
    med = _apply_median(raw, 5)
    return _apply_bilateral(med)


def method_best_smooth(det66, multi66, multi68, cfg):
    """Best smoothing pipeline: conf20 -> median5.

    Winner of the smoothing comparison. This method:
    1. Filters low-confidence blobs (20th percentile) to remove noise
    2. Applies median5 to smooth trajectory while preserving accuracy

    Results: Recall=90.7%, <10px=93.7%, Jitter=16.3px (below GT 17.8px)
    """
    raw = _get_conf20_raw(multi66)
    return _apply_median(raw, 5)


def method_multi_kalman(multi66, multi68, cfg):
    """MultiBlobMatcher + Kalman filter post-smoothing.

    Use MultiBlobMatcher for blob selection, then Kalman for smoothing/filtering.
    """
    raw = method_multi(multi66, multi68, cfg, blob_rank_penalty=0.5)
    frames = sorted(raw.keys())
    if not frames:
        return {}

    dt = 1.0 / 25.0
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    Q = np.eye(4) * 30.0
    Q[0, 0] = Q[1, 1] = 3.0
    R = np.eye(2) * 4.0 ** 2
    P = np.eye(4) * 100.0

    px0, py0 = raw[frames[0]]
    x = np.array([px0, py0, 0, 0], dtype=np.float64)
    results = {}
    last_fi = frames[0]

    for fi in frames:
        gap = fi - last_fi
        px, py = raw[fi]
        meas = np.array([px, py], dtype=np.float64)

        if gap > 10:
            x = np.array([px, py, 0, 0], dtype=np.float64)
            P = np.eye(4) * 100.0
            results[fi] = (px, py)
            last_fi = fi
            continue

        F_g = F.copy()
        F_g[0, 2] = dt * gap
        F_g[1, 3] = dt * gap
        x_pred = F_g @ x
        P_pred = F_g @ P @ F_g.T + Q * gap

        y_innov = meas - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y_innov
        P = (np.eye(4) - K @ H) @ P_pred
        results[fi] = (float(x[0]), float(x[1]))
        last_fi = fi

    return results


def run_method(name, det66, multi66, multi68, cfg):
    """Run a tracking method by name. Returns {frame: (px, py)}."""
    logger.info("Running method: %s", name)
    t0 = time.time()

    if name == "top1":
        result = method_top1(det66, multi66, multi68, cfg)
    elif name == "multi":
        result = method_multi(multi66, multi68, cfg, blob_rank_penalty=0.5)
    elif name == "multi_strict":
        result = method_multi(multi66, multi68, cfg, blob_rank_penalty=1.0)
    elif name == "multi_loose":
        result = method_multi(multi66, multi68, cfg, blob_rank_penalty=0.2)
    elif name == "viterbi":
        result = method_viterbi(multi66, multi68, cfg)
    elif name == "top1_smooth":
        result = method_top1_smooth(det66, multi66, multi68, cfg)
    elif name == "top1_xcam":
        result = method_top1_xcam(det66, multi66, multi68, cfg)
    elif name == "top1_kalman":
        result = method_top1_kalman(det66, multi66, multi68, cfg)
    elif name == "multi_smooth":
        result = method_multi_smooth(multi66, multi68, cfg)
    elif name == "top1_confidence":
        result = method_top1_confidence(det66, multi66, multi68, cfg)
    elif name == "top1_kalman_xcam":
        result = method_top1_kalman_xcam(det66, multi66, multi68, cfg)
    elif name == "multi_kalman":
        result = method_multi_kalman(multi66, multi68, cfg)
    elif name == "conf_xcam":
        result = method_confidence_xcam(det66, multi66, multi68, cfg)
    elif name == "conf_kalman":
        result = method_confidence_kalman(det66, multi66, multi68, cfg)
    elif name == "conf_xcam_kalman":
        result = method_confidence_xcam_kalman(det66, multi66, multi68, cfg)
    elif name == "conf_xcam_gate":
        result = method_conf_xcam_gate(det66, multi66, multi68, cfg)
    elif name == "top1_conf20":
        result = method_top1_conf20(det66, multi66, multi68, cfg)
    elif name == "top1_conf30":
        result = method_top1_conf30(det66, multi66, multi68, cfg)
    elif name == "top1_conf40":
        result = method_top1_conf40(det66, multi66, multi68, cfg)
    # New SG variants
    elif name == "top1_sg5_p2":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=5, polyorder=2)
    elif name == "top1_sg7_p2":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=7, polyorder=2)
    elif name == "top1_sg9_p2":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=9, polyorder=2)
    elif name == "top1_sg11_p2":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=11, polyorder=2)
    elif name == "top1_sg5_p3":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=5, polyorder=3)
    elif name == "top1_sg7_p3":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=7, polyorder=3)
    elif name == "top1_sg9_p3":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=9, polyorder=3)
    elif name == "top1_sg11_p3":
        result = method_top1_sg(det66, multi66, multi68, cfg, window=11, polyorder=3)
    # EMA variants
    elif name == "top1_ema03":
        result = method_top1_ema(det66, multi66, multi68, cfg, alpha=0.3)
    elif name == "top1_ema05":
        result = method_top1_ema(det66, multi66, multi68, cfg, alpha=0.5)
    elif name == "top1_ema07":
        result = method_top1_ema(det66, multi66, multi68, cfg, alpha=0.7)
    elif name == "top1_ema08":
        result = method_top1_ema(det66, multi66, multi68, cfg, alpha=0.8)
    # Kalman tuned
    elif name == "top1_kalman_tuned":
        result = method_top1_kalman_tuned(det66, multi66, multi68, cfg)
    # Conf20 + Kalman
    elif name == "conf20_kalman":
        result = method_conf20_kalman(det66, multi66, multi68, cfg)
    elif name == "conf20_kalman_tuned":
        result = method_conf20_kalman_tuned(det66, multi66, multi68, cfg)
    # Bilateral
    elif name == "top1_bilateral":
        result = method_top1_bilateral(det66, multi66, multi68, cfg)
    # Conf-weighted
    elif name == "top1_conf_weighted":
        result = method_top1_conf_weighted(det66, multi66, multi68, cfg)
    # Median
    elif name == "top1_median3":
        result = method_top1_median(det66, multi66, multi68, cfg, kernel_size=3)
    elif name == "top1_median5":
        result = method_top1_median(det66, multi66, multi68, cfg, kernel_size=5)
    elif name == "top1_median7":
        result = method_top1_median(det66, multi66, multi68, cfg, kernel_size=7)
    # Conf20 + smoother combos
    elif name == "conf20_sg7_p2":
        result = method_conf20_sg(det66, multi66, multi68, cfg, window=7, polyorder=2)
    elif name == "conf20_sg9_p3":
        result = method_conf20_sg(det66, multi66, multi68, cfg, window=9, polyorder=3)
    elif name == "conf20_ema05":
        result = method_conf20_ema(det66, multi66, multi68, cfg, alpha=0.5)
    elif name == "conf20_ema07":
        result = method_conf20_ema(det66, multi66, multi68, cfg, alpha=0.7)
    elif name == "conf20_bilateral":
        result = method_conf20_bilateral(det66, multi66, multi68, cfg)
    elif name == "conf20_median5":
        result = method_conf20_median(det66, multi66, multi68, cfg, kernel_size=5)
    # Round 2: focused combos
    elif name == "top1_med3_bilateral":
        result = method_top1_med3_bilateral(det66, multi66, multi68, cfg)
    elif name == "conf20_med3_bilateral":
        result = method_conf20_med3_bilateral(det66, multi66, multi68, cfg)
    elif name == "top1_light_kalman":
        result = method_top1_light_kalman(det66, multi66, multi68, cfg)
    elif name == "conf20_light_kalman":
        result = method_conf20_light_kalman(det66, multi66, multi68, cfg)
    elif name == "top1_med3_ema08":
        result = method_top1_med3_ema08(det66, multi66, multi68, cfg)
    elif name == "conf20_med3_ema08":
        result = method_conf20_med3_ema08(det66, multi66, multi68, cfg)
    elif name == "top1_adaptive_ema":
        result = method_top1_adaptive_ema(det66, multi66, multi68, cfg)
    elif name == "conf20_adaptive_ema":
        result = method_conf20_adaptive_ema(det66, multi66, multi68, cfg)
    # Round 3
    elif name == "top1_median3_sg5_p2":
        result = method_top1_median3_sg5_p2(det66, multi66, multi68, cfg)
    elif name == "conf20_median3_sg5_p2":
        result = method_conf20_median3_sg5_p2(det66, multi66, multi68, cfg)
    elif name == "top1_median5_bilateral":
        result = method_top1_median5_bilateral(det66, multi66, multi68, cfg)
    elif name == "conf20_median5_bilateral":
        result = method_conf20_median5_bilateral(det66, multi66, multi68, cfg)
    elif name == "best_smooth":
        result = method_best_smooth(det66, multi66, multi68, cfg)
    else:
        raise ValueError(f"Unknown method: {name}")

    elapsed = time.time() - t0
    logger.info("  %s: %d detections in %.1fs", name, len(result), elapsed)
    return result


# ── Metrics computation ───────────────────────────────────────────────


def compute_metrics(gt_mb, tracker_dets, threshold_px=MATCH_THRESHOLD_PX):
    """Compute all evaluation metrics.

    Args:
        gt_mb: {frame: (px, py)} ground truth match_ball positions
        tracker_dets: {frame: (px, py)} tracker output positions
        threshold_px: pixel distance threshold for a "match"

    Returns:
        dict with all metrics
    """
    gt_frames = set(gt_mb.keys())
    det_frames = set(tracker_dets.keys())

    # True Positives: tracker detection on a GT frame within threshold
    tp_frames = []
    tp_errors = []
    fn_frames = []  # GT frames missed or too far

    for fi in sorted(gt_frames):
        if fi in det_frames:
            gx, gy = gt_mb[fi]
            dx, dy = tracker_dets[fi]
            dist = math.hypot(dx - gx, dy - gy)
            if dist <= threshold_px:
                tp_frames.append(fi)
                tp_errors.append(dist)
            else:
                fn_frames.append(fi)
        else:
            fn_frames.append(fi)

    # False Positives: tracker detections on non-GT frames
    fp_frames = [fi for fi in det_frames if fi not in gt_frames]

    n_tp = len(tp_frames)
    n_fp = len(fp_frames)
    n_fn = len(fn_frames)

    recall = n_tp / len(gt_frames) if gt_frames else 0.0
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Pixel accuracy (only for TPs)
    errors = np.array(tp_errors) if tp_errors else np.array([])
    if len(errors) > 0:
        mean_px = float(np.mean(errors))
        median_px = float(np.median(errors))
        pct_5 = float(np.mean(errors < 5) * 100)
        pct_10 = float(np.mean(errors < 10) * 100)
        pct_20 = float(np.mean(errors < 20) * 100)
    else:
        mean_px = median_px = float("nan")
        pct_5 = pct_10 = pct_20 = 0.0

    # Jitter: mean frame-to-frame displacement on consecutive match_ball frames
    # Only consider frames where both current and previous are match_ball AND tracked
    gt_sorted = sorted(gt_frames)
    displacements = []
    large_jumps = 0
    total_consecutive = 0

    for idx in range(1, len(gt_sorted)):
        fi_prev = gt_sorted[idx - 1]
        fi_curr = gt_sorted[idx]
        # Must be consecutive (gap <= 1)
        if fi_curr - fi_prev > 1:
            continue
        if fi_prev in tracker_dets and fi_curr in tracker_dets:
            px0, py0 = tracker_dets[fi_prev]
            px1, py1 = tracker_dets[fi_curr]
            disp = math.hypot(px1 - px0, py1 - py0)
            displacements.append(disp)
            total_consecutive += 1
            if disp > 30:
                large_jumps += 1

    jitter = float(np.mean(displacements)) if displacements else float("nan")
    large_jump_pct = (large_jumps / total_consecutive * 100) if total_consecutive > 0 else 0.0

    return {
        "n_gt": len(gt_frames),
        "n_det": len(det_frames),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "mean_px": mean_px,
        "median_px": median_px,
        "pct_5": pct_5,
        "pct_10": pct_10,
        "pct_20": pct_20,
        "jitter": jitter,
        "large_jump_pct": large_jump_pct,
    }


# ── Output formatting ─────────────────────────────────────────────────


def print_comparison_table(all_results):
    """Print a clean comparison table of all methods."""
    if not all_results:
        return

    # Header
    cols = [
        ("Method", 14),
        ("Recall%", 8),
        ("Prec%", 7),
        ("F1%", 6),
        ("TP", 5),
        ("FP", 5),
        ("FN", 5),
        ("MeanPx", 7),
        ("MedPx", 7),
        ("<5px%", 7),
        ("<10px%", 8),
        ("<20px%", 8),
        ("Jitter", 7),
        ("Jmp30%", 7),
    ]

    header = " | ".join(f"{name:>{w}}" for name, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)

    print()
    print("=" * len(header))
    print("TRACKING METHOD COMPARISON")
    print(f"GT: {all_results[list(all_results.keys())[0]]['n_gt']} match_ball frames, "
          f"match threshold: {MATCH_THRESHOLD_PX}px")
    print("=" * len(header))
    print(header)
    print(sep)

    for method_name, m in all_results.items():
        row = [
            f"{method_name:>14}",
            f"{m['recall'] * 100:>7.1f}",
            f"{m['precision'] * 100:>6.1f}",
            f"{m['f1'] * 100:>5.1f}",
            f"{m['n_tp']:>5d}",
            f"{m['n_fp']:>5d}",
            f"{m['n_fn']:>5d}",
            f"{m['mean_px']:>6.1f}" if not math.isnan(m['mean_px']) else f"{'N/A':>6}",
            f"{m['median_px']:>6.1f}" if not math.isnan(m['median_px']) else f"{'N/A':>6}",
            f"{m['pct_5']:>6.1f}",
            f"{m['pct_10']:>7.1f}",
            f"{m['pct_20']:>7.1f}",
            f"{m['jitter']:>6.1f}" if not math.isnan(m['jitter']) else f"{'N/A':>6}",
            f"{m['large_jump_pct']:>6.1f}",
        ]
        print(" | ".join(row))

    print(sep)

    # Find best method by F1
    best_f1 = max(all_results, key=lambda k: all_results[k]["f1"])
    best_prec = max(all_results, key=lambda k: all_results[k]["precision"])
    best_recall = max(all_results, key=lambda k: all_results[k]["recall"])

    print()
    print(f"Best F1:        {best_f1} ({all_results[best_f1]['f1'] * 100:.1f}%)")
    print(f"Best Precision: {best_prec} ({all_results[best_prec]['precision'] * 100:.1f}%)")
    print(f"Best Recall:    {best_recall} ({all_results[best_recall]['recall'] * 100:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Tracking evaluation framework")
    parser.add_argument(
        "--method", nargs="+", default=["all"],
        help="Methods to evaluate. Use 'all' for all methods.",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="Clear detection cache and re-run TrackNet.",
    )
    args = parser.parse_args()

    methods = ALL_METHODS if "all" in args.method else args.method
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method: {m}. Available: {ALL_METHODS}")

    # Clear cache if requested
    if args.clear_cache:
        cache_path = CACHE_DIR / "detection_cache.pkl"
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Cleared detection cache.")

    # Load GT
    logger.info("Loading GT annotations from %s ...", GT_DIR)
    gt_mb = load_gt(GT_DIR, MAX_FRAMES)
    logger.info("GT: %d match_ball frames", len(gt_mb))

    # Load config
    cfg = load_config()

    # Run detection (cached)
    det_data = run_detection_cached(cfg)
    multi66 = det_data["multi66"]
    det66 = det_data["det66"]
    multi68 = det_data["multi68"]

    # Run each method and compute metrics
    all_results = {}
    for method_name in methods:
        tracker_dets = run_method(method_name, det66, multi66, multi68, cfg)
        metrics = compute_metrics(gt_mb, tracker_dets)
        all_results[method_name] = metrics

        # Print individual method summary
        logger.info(
            "  %s: Recall=%.1f%% Precision=%.1f%% F1=%.1f%% "
            "MeanPx=%.1f MedPx=%.1f Jitter=%.1f",
            method_name,
            metrics["recall"] * 100, metrics["precision"] * 100, metrics["f1"] * 100,
            metrics["mean_px"], metrics["median_px"], metrics["jitter"],
        )

    # Print comparison table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
