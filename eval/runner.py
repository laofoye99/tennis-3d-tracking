"""Run detection + triangulation + bounce detection for each method combination.

Method combinations:
    1. TrackNet + per-frame pairing
    2. TrackNet + track-first
    3. MedianBG + per-frame pairing
    4. MedianBG + track-first
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.config import EvalConfig
from eval.metrics import BounceMetrics, compute_metrics

logger = logging.getLogger(__name__)


def _load_gt(config: EvalConfig) -> tuple[list[dict], list[dict]]:
    """Load GT trajectory and bounces from bounce_results.json.

    Returns:
        (gt_trajectory, gt_bounces) where each bounce has frame, x, y, z.
    """
    with open(config.gt_bounce_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_trajectory = data.get("trajectory", [])
    gt_bounces = data.get("bounces", [])
    # Normalize bounce format
    normalized = []
    for b in gt_bounces:
        normalized.append({
            "frame": b["frame"],
            "x": b["x"],
            "y": b["y"],
            "z": b["z"],
        })
    return gt_trajectory, normalized


def _load_frames(cam_dir: Path, max_frames: int) -> list[np.ndarray]:
    """Load JPEG frames from a camera directory, sorted by filename."""
    jpg_files = sorted(cam_dir.glob("*.jpg"))[:max_frames]
    frames = []
    for p in jpg_files:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def _load_homography(config: EvalConfig):
    """Load homography matrices from JSON file.

    Returns:
        (H_cam66_img2world, H_cam68_img2world) as numpy arrays
    """
    with open(config.homography_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    H66 = np.array(data["cam66"]["H_image_to_world"], dtype=np.float64)
    H68 = np.array(data["cam68"]["H_image_to_world"], dtype=np.float64)
    return H66, H68


def _pixel_to_world(px, py, H):
    """Apply homography: pixel -> world (x, y)."""
    pt = np.array([px, py, 1.0])
    r = H @ pt
    return float(r[0] / r[2]), float(r[1] / r[2])


def _triangulate(w1, w2, cam1_pos, cam2_pos):
    """Ray-based triangulation returning (x, y, z)."""
    cam1 = np.asarray(cam1_pos, dtype=np.float64)
    cam2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
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
        return float(mid[0]), float(mid[1]), 0.0
    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom
    s = max(s, 0.0)
    t = max(t, 0.0)
    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0
    if mid[2] < 0:
        mid[2] = 0.0
    return float(mid[0]), float(mid[1]), float(mid[2])


# ----------------------------------------------------------------
# Detection backends
# ----------------------------------------------------------------

def _detect_tracknet(frames: list[np.ndarray], config: EvalConfig) -> dict[int, list[tuple]]:
    """Run TrackNet detection on frames. Returns {frame_idx: [(px, py, conf), ...]}."""
    from app.pipeline.inference import TrackNetDetector
    from app.pipeline.postprocess import BallTracker

    model_path = str(config.project_root / config.tracknet_model)
    detector = TrackNetDetector(
        model_path=model_path,
        input_size=config.input_size,
        frames_in=config.tracknet_seq_len,
        frames_out=config.tracknet_seq_len,
        device=config.device,
    )
    tracker = BallTracker(
        original_size=(1920, 1080),
        threshold=config.heatmap_threshold,
    )

    detections: dict[int, list[tuple]] = {}
    seq_len = config.tracknet_seq_len
    n = len(frames)

    # Process in sliding windows of seq_len
    for start in range(0, n, seq_len):
        end = min(start + seq_len, n)
        batch = frames[start:end]
        # Pad if needed
        while len(batch) < seq_len:
            batch.append(batch[-1])

        heatmaps = detector.infer(batch)  # (seq_len, H, W)
        actual_count = end - start
        for i in range(actual_count):
            result = tracker.process_heatmap(heatmaps[i])
            frame_idx = start + i
            if result is not None:
                px, py, conf = result
                detections[frame_idx] = [(px, py, conf)]
            else:
                detections[frame_idx] = []

    return detections


def _detect_medianbg(frames: list[np.ndarray], config: EvalConfig) -> dict[int, list[tuple]]:
    """Run MedianBG detection on frames. Returns {frame_idx: [(px, py), ...]}."""
    from app.pipeline.blob_detector import BallBlobDetector

    detector = BallBlobDetector(
        thresh=config.median_bg_thresh,
        min_area=2,
        max_area=600,
    )

    detections: dict[int, list[tuple]] = {}
    block_size = config.median_bg_block
    n = len(frames)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        batch = frames[start:end]
        gray_batch = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in batch]

        if len(gray_batch) < 2:
            for i in range(len(gray_batch)):
                detections[start + i] = []
            continue

        block_dets = detector.detect_block(gray_batch)
        for local_idx, blobs in block_dets.items():
            detections[start + local_idx] = blobs

    return detections


# ----------------------------------------------------------------
# Triangulation modes
# ----------------------------------------------------------------

def _per_frame_pairing(
    det66: dict[int, list[tuple]],
    det68: dict[int, list[tuple]],
    H66, H68,
    cam66_pos, cam68_pos,
) -> list[tuple]:
    """Per-frame pairing: match best detections by timestamp, triangulate.

    Returns list of (frame, x, y, z) tuples.
    """
    trajectory = []
    common_frames = sorted(set(det66.keys()) & set(det68.keys()))

    for fi in common_frames:
        blobs66 = det66[fi]
        blobs68 = det68[fi]
        if not blobs66 or not blobs68:
            continue

        # Use top-1 detection from each camera
        p66 = blobs66[0][:2]  # (px, py)
        p68 = blobs68[0][:2]

        w66 = _pixel_to_world(p66[0], p66[1], H66)
        w68 = _pixel_to_world(p68[0], p68[1], H68)

        x, y, z = _triangulate(w66, w68, cam66_pos, cam68_pos)

        # Basic sanity: skip wild outliers
        if abs(x) > 20 or abs(y) > 30 or z > 10:
            continue

        trajectory.append((fi, x, y, z))

    return trajectory


def _track_first(
    det66: dict[int, list[tuple]],
    det68: dict[int, list[tuple]],
    H66, H68,
    cam66_pos, cam68_pos,
    config: EvalConfig,
) -> list[tuple]:
    """Track-first: single-camera tracking then cross-camera matching.

    Returns list of (frame, x, y, z) tuples from the best matched track.
    """
    from app.pipeline.tracker import track_single_camera, match_and_triangulate

    # Normalize to (cx, cy) tuples — strip confidence if present
    def _normalize(dets):
        return {fi: [(b[0], b[1]) for b in blobs] for fi, blobs in dets.items()}

    tracks66 = track_single_camera(
        _normalize(det66),
        max_pixel_dist=config.tracker_max_pixel_dist,
        max_gap=config.tracker_max_gap,
        min_len=config.tracker_min_len,
    )
    tracks68 = track_single_camera(
        _normalize(det68),
        max_pixel_dist=config.tracker_max_pixel_dist,
        max_gap=config.tracker_max_gap,
        min_len=config.tracker_min_len,
    )

    logger.info("Tracks: cam66=%d, cam68=%d", len(tracks66), len(tracks68))

    if not tracks66 or not tracks68:
        return []

    results = match_and_triangulate(
        tracks66, tracks68,
        H66, H68,
        cam66_pos, cam68_pos,
        max_ray_dist=config.max_ray_dist,
        min_overlap=config.min_overlap,
    )

    if not results:
        return []

    # Merge all matched trajectories (sorted by frame)
    all_points = {}
    for r in results:
        for pt in r["trajectory"]:
            fi = pt[0]
            if fi not in all_points:
                all_points[fi] = (fi, pt[1], pt[2], pt[3])

    trajectory = sorted(all_points.values(), key=lambda t: t[0])
    return trajectory


# ----------------------------------------------------------------
# Run a single combination
# ----------------------------------------------------------------

def run_combination(
    method_name: str,
    detector_type: str,
    triangulation_type: str,
    frames66: list[np.ndarray],
    frames68: list[np.ndarray],
    H66, H68,
    config: EvalConfig,
    gt_bounces: list[dict],
) -> BounceMetrics:
    """Run one detector+triangulation combination and evaluate.

    Args:
        method_name: human-readable label
        detector_type: "tracknet" or "medianbg"
        triangulation_type: "per_frame" or "track_first"
        frames66, frames68: loaded camera frames
        H66, H68: homography matrices (image-to-world)
        config: EvalConfig
        gt_bounces: ground truth bounces

    Returns:
        BounceMetrics for this combination
    """
    from app.pipeline.bounce_detect import detect_bounces

    logger.info("=== Running: %s ===", method_name)

    # Step 1: Detection
    logger.info("Step 1: Detection (%s)...", detector_type)
    if detector_type == "tracknet":
        det66 = _detect_tracknet(frames66, config)
        det68 = _detect_tracknet(frames68, config)
    elif detector_type == "medianbg":
        det66 = _detect_medianbg(frames66, config)
        det68 = _detect_medianbg(frames68, config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

    n66 = sum(1 for v in det66.values() if v)
    n68 = sum(1 for v in det68.values() if v)
    logger.info("Detections: cam66=%d frames with detections, cam68=%d", n66, n68)

    # Step 2: Triangulation
    logger.info("Step 2: Triangulation (%s)...", triangulation_type)
    cam66_pos = config.cam66_pos
    cam68_pos = config.cam68_pos

    if triangulation_type == "per_frame":
        trajectory = _per_frame_pairing(det66, det68, H66, H68, cam66_pos, cam68_pos)
    elif triangulation_type == "track_first":
        trajectory = _track_first(det66, det68, H66, H68, cam66_pos, cam68_pos, config)
    else:
        raise ValueError(f"Unknown triangulation type: {triangulation_type}")

    logger.info("3D trajectory points: %d", len(trajectory))

    # Step 3: Bounce detection on 3D trajectory
    logger.info("Step 3: Bounce detection...")
    detected_bounces_raw = detect_bounces(
        trajectory,
        z_max=config.bounce_z_max,
        prominence=config.bounce_prominence,
        min_distance=config.bounce_min_distance,
        smooth=config.bounce_smooth,
    )
    logger.info("Detected bounces: %d", len(detected_bounces_raw))

    # Step 4: Evaluate
    metrics = compute_metrics(
        method_name,
        gt_bounces,
        detected_bounces_raw,
        frame_tolerance=config.frame_tolerance,
        position_tolerance=config.position_tolerance,
    )

    logger.info(
        "Results: recall=%.3f, precision=%.3f, f1=%.3f, matched=%d/%d, FP=%d",
        metrics.recall, metrics.precision, metrics.f1,
        metrics.matched, metrics.gt_count, metrics.false_positives,
    )

    return metrics


def run_all_combinations(config: EvalConfig) -> list[BounceMetrics]:
    """Run all 4 method combinations and return metrics.

    Combinations:
        1. TrackNet + per-frame pairing
        2. TrackNet + track-first
        3. MedianBG + per-frame pairing
        4. MedianBG + track-first
    """
    errors = config.validate()
    # Allow missing TrackNet model (skip those combos)
    tracknet_missing = any("TrackNet model" in e for e in errors)
    critical_errors = [e for e in errors if "TrackNet model" not in e]
    if critical_errors:
        raise RuntimeError("Config validation failed:\n" + "\n".join(critical_errors))

    # Load shared data
    logger.info("Loading GT data...")
    _gt_trajectory, gt_bounces = _load_gt(config)
    logger.info("GT bounces: %d", len(gt_bounces))

    logger.info("Loading frames (cam66)...")
    frames66 = _load_frames(config.cam66_dir, config.max_frames)
    logger.info("Loaded %d frames from cam66", len(frames66))

    logger.info("Loading frames (cam68)...")
    frames68 = _load_frames(config.cam68_dir, config.max_frames)
    logger.info("Loaded %d frames from cam68", len(frames68))

    logger.info("Loading homography matrices...")
    H66, H68 = _load_homography(config)

    combinations = [
        ("TrackNet + PerFrame", "tracknet", "per_frame"),
        ("TrackNet + TrackFirst", "tracknet", "track_first"),
        ("MedianBG + PerFrame", "medianbg", "per_frame"),
        ("MedianBG + TrackFirst", "medianbg", "track_first"),
    ]

    results = []
    for method_name, det_type, tri_type in combinations:
        if det_type == "tracknet" and tracknet_missing:
            logger.warning("Skipping %s (TrackNet model not found)", method_name)
            # Create empty metrics for skipped combo
            m = BounceMetrics(method=method_name + " [SKIPPED]")
            m.gt_count = len(gt_bounces)
            m.missed = len(gt_bounces)
            m.missed_frames = [b["frame"] for b in gt_bounces]
            results.append(m)
            continue

        try:
            metrics = run_combination(
                method_name, det_type, tri_type,
                frames66, frames68, H66, H68,
                config, gt_bounces,
            )
            results.append(metrics)
        except Exception as e:
            logger.error("Failed to run %s: %s", method_name, e, exc_info=True)
            m = BounceMetrics(method=method_name + " [ERROR]")
            m.gt_count = len(gt_bounces)
            m.missed = len(gt_bounces)
            m.missed_frames = [b["frame"] for b in gt_bounces]
            results.append(m)

    return results
