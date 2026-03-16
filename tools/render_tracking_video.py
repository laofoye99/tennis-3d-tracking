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

# ── Court dimensions (meters) ─────────────────────────────────────────────
COURT_W = 8.23
COURT_L = 23.77
NET_Y = COURT_L / 2  # 11.885
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


def run_detection(video_path, detector, postproc, max_frames):
    """Run TrackNet detection on a video, return {frame_idx: (px, py, conf)}."""
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

    detections = {}
    n = len(frames)
    logger.info("Running TrackNet on %d frames (seq_len=%d) ...", n, seq_len)

    # Temporal consistency parameters
    MIN_DETECTIONS_IN_WINDOW = 4   # need at least 4/8 detections in a window
    MAX_JUMP_PX = 120.0            # max pixel jump between consecutive frames

    rejected_temporal = 0

    for start in range(0, n, seq_len):
        end = min(start + seq_len, n)
        batch = frames[start:end]
        actual_len = end - start
        if len(batch) < seq_len:
            batch += [batch[-1]] * (seq_len - len(batch))

        heatmaps = detector.infer(batch)  # (seq_len, H, W)

        # ── Extract all detections in this 8-frame window ──────────
        window_dets = {}
        for i in range(actual_len):
            result = postproc.process_heatmap(heatmaps[i])
            if result is not None:
                window_dets[i] = result  # (px, py, conf)

        # ── Temporal consistency check within the 8-frame window ───
        # Build the longest smooth chain: consecutive detections with
        # small pixel jumps form a coherent trajectory
        if len(window_dets) < MIN_DETECTIONS_IN_WINDOW:
            # Too few detections in window → reject all (likely noise)
            rejected_temporal += len(window_dets)
            continue

        # Check trajectory smoothness: count how many consecutive pairs
        # have reasonable displacement
        sorted_idx = sorted(window_dets.keys())
        smooth_count = 0
        for j in range(1, len(sorted_idx)):
            i_prev, i_curr = sorted_idx[j - 1], sorted_idx[j]
            px0, py0, _ = window_dets[i_prev]
            px1, py1, _ = window_dets[i_curr]
            gap = i_curr - i_prev  # frames apart
            disp = math.hypot(px1 - px0, py1 - py0)
            # Allow larger jump if frames are further apart
            if disp < MAX_JUMP_PX * gap:
                smooth_count += 1

        # Need at least half the transitions to be smooth
        min_smooth = max(1, (len(sorted_idx) - 1) // 2)
        if smooth_count < min_smooth:
            rejected_temporal += len(window_dets)
            continue

        # Window passed — add all detections
        for i, (px, py, conf) in window_dets.items():
            fi = start + i
            detections[fi] = (px, py, conf)

        if start % (seq_len * 50) == 0:
            logger.info("  Detection progress: %d/%d", min(start + seq_len, n), n)

    logger.info(
        "Detected ball in %d/%d frames (rejected %d by temporal consistency)",
        len(detections), n, rejected_temporal,
    )
    return detections, n


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
    """Triangulate 3D from paired 2D detections. Returns {frame: (x,y,z,ray_dist)}."""
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
        if x < -COURT_EXTEND or x > COURT_W + COURT_EXTEND:
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


def smooth_trajectory_sg(points_3d, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter to smooth 3D trajectory.

    Splits trajectory into continuous segments (gaps > 3 frames = new segment),
    smooths each segment independently, returns smoothed points_3d dict.

    Args:
        points_3d: {frame: (x, y, z)} or {frame: (x, y, z, rd)}
        window_length: SG window (must be odd). 11 = ~0.44s at 25fps.
        polyorder: polynomial order for SG fit.

    Returns:
        smoothed: {frame: (x, y, z)} with Savitzky-Golay smoothed coordinates.
    """
    frames = sorted(points_3d.keys())
    if len(frames) < window_length:
        return {fi: points_3d[fi][:3] for fi in frames}

    # Split into continuous segments (gap > 3 frames = new segment)
    MAX_GAP = 3
    segments = []
    seg_start = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > MAX_GAP:
            segments.append(frames[seg_start:i])
            seg_start = i
    segments.append(frames[seg_start:])

    smoothed = {}
    n_smoothed_segs = 0

    for seg_frames in segments:
        n = len(seg_frames)
        xs = np.array([points_3d[fi][0] for fi in seg_frames])
        ys = np.array([points_3d[fi][1] for fi in seg_frames])
        zs = np.array([points_3d[fi][2] for fi in seg_frames])

        if n >= window_length:
            # Apply Savitzky-Golay
            xs_s = savgol_filter(xs, window_length, polyorder)
            ys_s = savgol_filter(ys, window_length, polyorder)
            zs_s = savgol_filter(zs, window_length, polyorder)
            # Clamp Z >= 0
            zs_s = np.maximum(zs_s, 0.0)
            n_smoothed_segs += 1
        else:
            # Segment too short for SG — use raw
            xs_s, ys_s, zs_s = xs, ys, zs

        for i, fi in enumerate(seg_frames):
            smoothed[fi] = (float(xs_s[i]), float(ys_s[i]), float(zs_s[i]))

    logger.info(
        "Savitzky-Golay: %d segments (%d smoothed), %d total points",
        len(segments), n_smoothed_segs, len(smoothed),
    )
    return smoothed


def detect_bounces(points_3d):
    """Simple V-shape bounce detection on Z axis.

    Returns list of {frame, x, y, z, in_court}.
    """
    frames = sorted(points_3d.keys())
    if len(frames) < 10:
        return []

    zvals = [(fi, points_3d[fi][:3]) for fi in frames]  # strip extra fields if present
    bounces = []
    window = 8

    for i in range(window, len(zvals) - window):
        fi, (x, y, z) = zvals[i]

        # Must be near ground
        if z > 0.8:
            continue

        # Check V-shape: z should be local minimum
        z_before = [zvals[i - j][1][2] for j in range(1, window + 1)]
        z_after = [zvals[i + j][1][2] for j in range(1, window + 1)]

        avg_before = np.mean(z_before)
        avg_after = np.mean(z_after)

        # V-shape: surrounding points higher than center
        if avg_before > z + 0.3 and avg_after > z + 0.3:
            # Check not too close to last bounce
            if bounces and fi - bounces[-1]["frame"] < 15:
                continue

            in_court = 0 <= x <= COURT_W and 0 <= y <= COURT_L
            bounces.append({
                "frame": fi,
                "x": x, "y": y, "z": z,
                "in_court": in_court,
            })

    logger.info("Detected %d bounces", len(bounces))
    return bounces


def draw_court(panel_w, panel_h, margin):
    """Draw a static tennis court top-down, return (image, world_to_court_fn)."""
    court = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    # Dark green background
    court[:] = (30, 60, 30)

    draw_w = panel_w - 2 * margin
    draw_h = panel_h - 2 * margin

    def w2c(wx, wy):
        cx = int(margin + (wx / COURT_W) * draw_w)
        cy = int(margin + (1.0 - wy / COURT_L) * draw_h)  # flip Y: baseline at bottom
        return cx, cy

    # Court fill
    tl = w2c(0, COURT_L)
    br = w2c(COURT_W, 0)
    cv2.rectangle(court, tl, br, (45, 100, 45), -1)

    # Court outline
    cv2.rectangle(court, tl, br, (255, 255, 255), 2)

    # Net line
    cv2.line(court, w2c(0, NET_Y), w2c(COURT_W, NET_Y), (200, 200, 200), 2)

    # Service lines
    cv2.line(court, w2c(0, SERVICE_NEAR), w2c(COURT_W, SERVICE_NEAR), (180, 180, 180), 1)
    cv2.line(court, w2c(0, SERVICE_FAR), w2c(COURT_W, SERVICE_FAR), (180, 180, 180), 1)

    # Center service line
    cv2.line(court, w2c(COURT_W / 2, SERVICE_NEAR), w2c(COURT_W / 2, SERVICE_FAR), (180, 180, 180), 1)

    # Center marks
    ct = w2c(COURT_W / 2, 0)
    cv2.line(court, ct, (ct[0], ct[1] - 8), (180, 180, 180), 1)
    ct2 = w2c(COURT_W / 2, COURT_L)
    cv2.line(court, ct2, (ct2[0], ct2[1] + 8), (180, 180, 180), 1)

    return court, w2c


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
    video66_path, video68_path, det66, det68, bounces, n_frames, output_path, canvas_w=1920
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
            bpt = w2c(b["x"], b["y"])

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
    args = parser.parse_args()

    cfg = load_config()

    # ── Phase 1: Detection ──────────────────────────────────────────
    logger.info("=== Phase 1: TrackNet Detection ===")
    detector, postproc = build_detector(cfg)

    logger.info("--- cam66 ---")
    det66, n66 = run_detection(args.video66, detector, postproc, args.max_frames)

    # Reset detector background for second camera
    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("--- cam68 ---")
    det68, n68 = run_detection(args.video68, detector, postproc, args.max_frames)

    n_frames = min(n66, n68)
    logger.info("Common frames: %d, det66=%d, det68=%d", n_frames, len(det66), len(det68))

    # ── Phase 2: Triangulation ──────────────────────────────────────
    logger.info("=== Phase 2: 3D Triangulation ===")
    points_3d = triangulate_detections(det66, det68, cfg)

    # ── Phase 2.5: Flight Filter ────────────────────────────────────
    logger.info("=== Phase 2.5: Flight Filter (remove dead-ball FP) ===")
    fps = 25.0
    active_frames, filt66, filt68 = build_flight_mask(points_3d, det66, det68, fps)

    # ── Phase 3: Savitzky-Golay Smoothing + Bounce Detection ───────
    logger.info("=== Phase 3: SG Smoothing + Bounce Detection ===")
    # Only keep 3D points that are in active flight
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d if fi in active_frames}
    # Smooth trajectory with Savitzky-Golay filter
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)
    bounces = detect_bounces(smoothed_3d)

    for i, b in enumerate(bounces):
        tag = "IN" if b["in_court"] else "OUT"
        logger.info(
            "  Bounce %d: frame=%d (%.1f, %.1f, z=%.2f) %s",
            i + 1, b["frame"], b["x"], b["y"], b["z"], tag,
        )

    # ── Phase 4: Render Video ───────────────────────────────────────
    logger.info("=== Phase 4: Render Video ===")
    render_video(
        args.video66, args.video68,
        filt66, filt68, bounces,
        n_frames, args.output,
    )


if __name__ == "__main__":
    main()
