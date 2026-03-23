"""Render cross-camera validation video with rally analysis overlay.

Shows both camera views side-by-side with:
- Detection markers (green=correct, red=wrong, yellow=no GT)
- Ray distance bar (green→red gradient)
- Agreement/disagreement status
- Rescued frame indicators
- Bounce markers (landing positions in 2D on both views)
- Rally state banner (SERVING / RALLY / IDLE)
- Rally end reason labels
- Mini court diagram showing real-time trajectory and bounce positions

Usage:
    python -m tools.render_cross_camera_video
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from app.pipeline.homography import HomographyTransformer

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_66 = "uploads/cam66_20260307_173403_2min.mp4"
VIDEO_68 = "uploads/cam68_20260307_173403_2min.mp4"
DETECTIONS_66 = "exports/eval_motion_yolo/cam66_detections.json"
DETECTIONS_68 = "exports/eval_motion_yolo/cam68_detections.json"
CROSS_DATA = "exports/eval_cross_camera/per_frame_data.json"
CROSS_RESULTS = "exports/eval_cross_camera/eval_results.json"
GT_DIR_66 = "uploads/cam66_20260307_173403_2min"
GT_DIR_68 = "uploads/cam68_20260307_173403_2min"
OUT_PATH = "exports/eval_cross_camera/cross_camera_video.mp4"

MAX_FRAMES = 3000
RAY_DIST_AGREE = 1.0
RAY_DIST_DISAGREE = 2.0
GT_MATCH_RADIUS = 15.0
CANVAS_W = 1920
CANVAS_H = 600  # two half-height views stacked or side-by-side
TRAIL_LENGTH = 30  # number of past frames to draw as trajectory trail
TRAIL_JUMP_THRESHOLD = 150.0  # pixels — break trail if detection jumps too far

# Court dimensions (meters) — singles court
SINGLES_X_MIN = 1.37
SINGLES_X_MAX = 6.86
COURT_W = SINGLES_X_MAX - SINGLES_X_MIN  # 5.49m singles width
COURT_L = 23.77
NET_Y = COURT_L / 2  # 11.885m
SERVICE_LINE_NEAR = 5.485
SERVICE_LINE_FAR = 18.285

# Mini court diagram dimensions (pixels)
COURT_PANEL_W = 200
COURT_PANEL_H = 400  # aspect ratio ~2:1 for tennis court
COURT_MARGIN = 20  # margin around court lines inside the panel
BOUNCE_MARKER_FRAMES = 60  # how long to show bounce marker on court


def load_gt(gt_dir: str) -> dict[int, tuple[float, float]]:
    """Load GT positions per frame."""
    gt = {}
    for jf in sorted(Path(gt_dir).glob("*.json")):
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
            st = shape.get("shape_type", "point")
            if st == "rectangle" and len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[2] if len(pts) >= 3 else pts[1]
                gt[fi] = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                gt[fi] = (pts[0][0], pts[0][1])
            break
    return gt


def ray_dist_color(rd: float) -> tuple[int, int, int]:
    """Map ray distance to BGR color: green (agree) → red (disagree)."""
    if rd < RAY_DIST_AGREE:
        return (0, 220, 0)  # green
    elif rd < RAY_DIST_DISAGREE:
        t = (rd - RAY_DIST_AGREE) / (RAY_DIST_DISAGREE - RAY_DIST_AGREE)
        return (0, int(220 * (1 - t)), int(220 * t))  # green→red
    else:
        return (0, 0, 220)  # red


def draw_detection(img, px, py, is_correct, has_gt, label="", scale=1.0):
    """Draw detection marker on image."""
    x, y = int(px * scale), int(py * scale)
    if not has_gt:
        color = (0, 220, 220)  # yellow — no GT
        cv2.circle(img, (x, y), 8, color, 2)
    elif is_correct:
        color = (0, 220, 0)  # green — correct
        cv2.circle(img, (x, y), 8, color, 2)
    else:
        color = (0, 0, 220)  # red — wrong
        cv2.circle(img, (x, y), 8, color, 2)
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 16, 2)

    if label:
        cv2.putText(img, label, (x + 12, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_gt(img, px, py, scale=1.0):
    """Draw GT marker on image."""
    x, y = int(px * scale), int(py * scale)
    cv2.circle(img, (x, y), 10, (220, 180, 0), 1)  # cyan circle for GT
    cv2.drawMarker(img, (x, y), (220, 180, 0), cv2.MARKER_DIAMOND, 12, 1)


def draw_trail(img, trail: list[tuple[int, int]], color=(0, 200, 255), x_offset=0, scale=1.0):
    """Draw trajectory trail with fading opacity on the image.

    trail: list of (pixel_x, pixel_y) in original coords, oldest first.
    """
    n = len(trail)
    if n < 2:
        return
    for i in range(1, n):
        alpha = i / n  # 0→1, newer points are brighter
        pt1 = (int(trail[i - 1][0] * scale) + x_offset, int(trail[i - 1][1] * scale))
        pt2 = (int(trail[i][0] * scale) + x_offset, int(trail[i][1] * scale))
        c = tuple(int(v * alpha) for v in color)
        thickness = max(1, int(2 * alpha))
        cv2.line(img, pt1, pt2, c, thickness, cv2.LINE_AA)
    # Draw a filled circle at the latest point
    last = (int(trail[-1][0] * scale) + x_offset, int(trail[-1][1] * scale))
    cv2.circle(img, last, 4, color, -1, cv2.LINE_AA)


def draw_mini_court(panel_w, panel_h, margin):
    """Draw a static tennis court top-down diagram, returning BGR image."""
    court = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    court[:] = (40, 40, 40)  # dark background

    # Playable area inside margin
    draw_w = panel_w - 2 * margin
    draw_h = panel_h - 2 * margin

    def world_to_court(wx, wy):
        """Convert world meters to court panel pixel coords."""
        cx = int(margin + (wx / COURT_W) * draw_w)
        cy = int(margin + (wy / COURT_L) * draw_h)
        return cx, cy

    # Court outline
    tl = world_to_court(0, 0)
    br = world_to_court(COURT_W, COURT_L)
    cv2.rectangle(court, tl, br, (255, 255, 255), 1)

    # Net line
    nl = world_to_court(0, NET_Y)
    nr = world_to_court(COURT_W, NET_Y)
    cv2.line(court, nl, nr, (200, 200, 200), 2)

    # Service lines
    sl_near = world_to_court(0, SERVICE_LINE_NEAR)
    sr_near = world_to_court(COURT_W, SERVICE_LINE_NEAR)
    cv2.line(court, sl_near, sr_near, (150, 150, 150), 1)

    sl_far = world_to_court(0, SERVICE_LINE_FAR)
    sr_far = world_to_court(COURT_W, SERVICE_LINE_FAR)
    cv2.line(court, sl_far, sr_far, (150, 150, 150), 1)

    # Center service line
    cm_near = world_to_court(COURT_W / 2, SERVICE_LINE_NEAR)
    cm_far = world_to_court(COURT_W / 2, SERVICE_LINE_FAR)
    cv2.line(court, cm_near, cm_far, (150, 150, 150), 1)

    # Center mark at baselines
    ct = world_to_court(COURT_W / 2, 0)
    cb = world_to_court(COURT_W / 2, COURT_L)
    cv2.line(court, ct, (ct[0], ct[1] + 5), (150, 150, 150), 1)
    cv2.line(court, cb, (cb[0], cb[1] - 5), (150, 150, 150), 1)

    return court, world_to_court


def draw_bounce_on_camera(img, bounce_world_x, bounce_world_y, homo, scale, in_court):
    """Project a bounce world position back to camera pixel and draw marker."""
    try:
        px, py = homo.world_to_pixel(bounce_world_x, bounce_world_y)
        x, y = int(px * scale), int(py * scale)
        h, w = img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            color = (0, 255, 255) if in_court else (0, 0, 255)  # yellow=in, red=out
            # Pulsing target marker
            cv2.circle(img, (x, y), 14, color, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
    except Exception:
        pass


def compute_rally_analytics(det66, det68):
    """Pre-compute bounce and rally events using FusionCoordinator.

    Processes the UNION of all frames (not just the intersection), so
    single-camera detections can also produce bounce events via find_peaks.
    """
    from app.analytics import FusionCoordinator

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cam_positions = {
        "cam66": cfg["cameras"]["cam66"]["position_3d"],
        "cam68": cfg["cameras"]["cam68"]["position_3d"],
    }

    coordinator = FusionCoordinator(cam_positions=cam_positions, fps=25.0)

    # Process ALL frames (union, not intersection)
    all_frames: set[int] = set()
    all_frames.update(int(k) for k in det66.keys())
    all_frames.update(int(k) for k in det68.keys())

    bounce_frames: dict[int, dict] = {}
    rally_state_frames: dict[int, dict] = {}
    rally_events: dict[int, dict] = {}

    for fi in sorted(all_frames):
        fi_str = str(fi)
        d66 = det66.get(fi_str)
        d68 = det68.get(fi_str)

        # Get rally state BEFORE processing (bounce may end rally)
        pre_state = coordinator.get_rally_state()["state"]

        point_3d, bounce, rally_result = coordinator.process_frame(fi, d66, d68)

        if point_3d is not None:
            rally_state_frames[fi] = coordinator.get_rally_state()

        if rally_result is not None:
            rally_events[fi] = rally_result.to_dict()

        # Only keep bounces during active rallies
        if bounce is not None and pre_state in ("rally", "serving"):
            bounce_frames[fi] = bounce.to_dict()

    modes = coordinator.get_mode_counts()
    print(f"  FusionCoordinator modes: 3d={modes.get('3d',0)}, "
          f"single_cam={modes.get('single_cam',0)}, gap={modes.get('gap',0)}")

    return bounce_frames, rally_state_frames, rally_events


def render_video():
    """Render the cross-camera validation video."""
    # Load data
    with open(DETECTIONS_66) as f:
        det66 = json.load(f)
    with open(DETECTIONS_68) as f:
        det68 = json.load(f)
    with open(CROSS_DATA) as f:
        cross_data = json.load(f)
    with open(CROSS_RESULTS) as f:
        cross_results = json.load(f)

    # Pre-compute rally analytics
    print("Computing rally analytics...")
    bounce_frames, rally_state_frames, rally_events = compute_rally_analytics(det66, det68)
    print(f"  Bounces: {len(bounce_frames)}, Rally events: {len(rally_events)}")

    gt66 = load_gt(GT_DIR_66)
    gt68 = load_gt(GT_DIR_68)

    # Index cross data by frame
    cross_map = {d["frame"]: d for d in cross_data}

    # Compute rescued frames from per-frame data
    # cam66 rescued = disagree + cam66 wrong + cam68 correct
    # cam68 rescued = disagree + cam68 wrong + cam66 correct
    rescued_66 = set()
    rescued_68 = set()
    for d in cross_data:
        fi = d["frame"]
        rd_val = d["ray_distance"]
        if rd_val < RAY_DIST_DISAGREE:
            continue
        c66 = d.get("correct66")
        c68 = d.get("correct68")
        if c66 is not None and c68 is not None:
            if not c66 and c68:
                rescued_66.add(fi)
            elif not c68 and c66:
                rescued_68.add(fi)

    # Open videos
    cap66 = cv2.VideoCapture(VIDEO_66)
    cap68 = cv2.VideoCapture(VIDEO_68)

    if not cap66.isOpened() or not cap68.isOpened():
        print("ERROR: Cannot open video files")
        return

    fps = cap66.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap66.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap66.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Layout: two views side by side + mini court panel on right
    # [cam66 | cam68 | mini court]
    #        info bar
    court_pw = COURT_PANEL_W
    video_area_w = CANVAS_W - court_pw  # remaining width for two camera views
    half_w = video_area_w // 2
    scale = half_w / orig_w
    half_h = int(orig_h * scale)
    canvas_h = half_h + 80  # extra space for info bar at bottom

    # Build static mini court (reused every frame)
    court_ph = half_h  # same height as camera views
    base_court, world_to_court = draw_mini_court(court_pw, court_ph, COURT_MARGIN)

    # Load homography transformers for bounce 2D projection
    homo_path = "src/homography_matrices.json"
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (CANVAS_W, canvas_h))

    print(f"Rendering: {CANVAS_W}x{canvas_h} @ {fps}fps, scale={scale:.3f}")
    print(f"  Video area: {video_area_w}px, court panel: {court_pw}px")
    print(f"Rescued frames — cam66: {len(rescued_66)}, cam68: {len(rescued_68)}")

    # Trajectory trail buffers (store recent detection pixel positions)
    trail_66: list[tuple[float, float]] = []
    trail_68: list[tuple[float, float]] = []

    # Rally overlay state
    last_rally_state = None
    last_rally_event = None
    last_bounce = None
    rally_end_display = 0
    bounce_display = 0

    # Court diagram state
    court_trail: list[tuple[float, float]] = []  # recent 3D positions (world_x, world_y)
    court_trail_max = 30
    all_bounces: list[dict] = []  # accumulated bounce events for court diagram
    bounce_2d_display = 0  # countdown for showing bounce on camera views

    for fi in range(MAX_FRAMES):
        ret66, frame66 = cap66.read()
        ret68, frame68 = cap68.read()
        if not ret66 or not ret68:
            break

        fi_str = str(fi)

        # Resize frames
        small66 = cv2.resize(frame66, (half_w, half_h))
        small68 = cv2.resize(frame68, (half_w, half_h))

        # Get cross-validation data for this frame
        cd = cross_map.get(fi)
        has_det66 = fi_str in det66
        has_det68 = fi_str in det68

        # Determine if cameras disagree this frame
        is_disagree = cd is not None and cd["ray_distance"] >= RAY_DIST_DISAGREE

        # Update trajectory trails
        # Skip: disagreement frames, or large jumps (detection switched to different object)
        if has_det66:
            px66, py66 = det66[fi_str]["pixel_x"], det66[fi_str]["pixel_y"]
            jumped_66 = (trail_66 and
                         math.sqrt((px66 - trail_66[-1][0])**2 + (py66 - trail_66[-1][1])**2) > TRAIL_JUMP_THRESHOLD)
            if is_disagree or jumped_66:
                trail_66.clear()  # break trail on disagree or jump
            else:
                trail_66.append((px66, py66))
                if len(trail_66) > TRAIL_LENGTH:
                    trail_66.pop(0)
        else:
            trail_66.clear()

        if has_det68:
            px68, py68 = det68[fi_str]["pixel_x"], det68[fi_str]["pixel_y"]
            jumped_68 = (trail_68 and
                         math.sqrt((px68 - trail_68[-1][0])**2 + (py68 - trail_68[-1][1])**2) > TRAIL_JUMP_THRESHOLD)
            if is_disagree or jumped_68:
                trail_68.clear()
            else:
                trail_68.append((px68, py68))
                if len(trail_68) > TRAIL_LENGTH:
                    trail_68.pop(0)
        else:
            trail_68.clear()

        # Draw trajectory trails
        draw_trail(small66, trail_66, color=(0, 200, 255), scale=scale)
        draw_trail(small68, trail_68, color=(0, 200, 255), scale=scale)

        # Draw detections on cam66
        if has_det66:
            d = det66[fi_str]
            has_gt = fi in gt66
            if has_gt:
                gt_pos = gt66[fi]
                err = math.sqrt((d["pixel_x"] - gt_pos[0])**2 + (d["pixel_y"] - gt_pos[1])**2)
                is_correct = err < GT_MATCH_RADIUS
                draw_gt(small66, gt_pos[0], gt_pos[1], scale)
            else:
                is_correct = False
            draw_detection(small66, d["pixel_x"], d["pixel_y"], is_correct, has_gt, scale=scale)

        # Draw detections on cam68
        if has_det68:
            d = det68[fi_str]
            has_gt = fi in gt68
            if has_gt:
                gt_pos = gt68[fi]
                err = math.sqrt((d["pixel_x"] - gt_pos[0])**2 + (d["pixel_y"] - gt_pos[1])**2)
                is_correct = err < GT_MATCH_RADIUS
                draw_gt(small68, gt_pos[0], gt_pos[1], scale)
            else:
                is_correct = False
            draw_detection(small68, d["pixel_x"], d["pixel_y"], is_correct, has_gt, scale=scale)

        # ── Update court diagram state ────────────────────────────
        # Add current 3D position to court trail (cd already set above)
        if cd is not None and cd.get("pos_3d"):
            pos = cd["pos_3d"]
            wx, wy = pos[0], pos[1]
            # Check for jumps in world coords
            if court_trail:
                last_wx, last_wy = court_trail[-1]
                world_jump = math.sqrt((wx - last_wx)**2 + (wy - last_wy)**2)
                if world_jump > 5.0:  # 5m jump = break trail
                    court_trail.clear()
            court_trail.append((wx, wy))
            if len(court_trail) > court_trail_max:
                court_trail.pop(0)
        elif is_disagree:
            court_trail.clear()

        # Track bounce for 2D camera overlay
        if fi in bounce_frames:
            all_bounces.append(bounce_frames[fi])
            bounce_2d_display = BOUNCE_MARKER_FRAMES
            last_bounce = bounce_frames[fi]

        # Draw bounce markers on camera views (2D projection)
        if bounce_2d_display > 0 and last_bounce:
            alpha_b = bounce_2d_display / BOUNCE_MARKER_FRAMES
            if alpha_b > 0.3:  # only draw when still visible
                draw_bounce_on_camera(small66, last_bounce["x"], last_bounce["y"],
                                      homo66, scale, last_bounce["in_court"])
                draw_bounce_on_camera(small68, last_bounce["x"], last_bounce["y"],
                                      homo68, scale, last_bounce["in_court"])
            bounce_2d_display -= 1

        # ── Build mini court panel for this frame ─────────────────
        court_panel = base_court.copy()

        # Draw trajectory trail on court
        if len(court_trail) >= 2:
            for i in range(1, len(court_trail)):
                a = i / len(court_trail)
                c = (0, int(200 * a), int(255 * a))  # fading orange trail
                pt1 = world_to_court(*court_trail[i - 1])
                pt2 = world_to_court(*court_trail[i])
                cv2.line(court_panel, pt1, pt2, c, max(1, int(2 * a)), cv2.LINE_AA)
            # Current position dot
            cur = world_to_court(*court_trail[-1])
            cv2.circle(court_panel, cur, 4, (0, 200, 255), -1, cv2.LINE_AA)

        # Draw all accumulated bounces on court
        for bi, b in enumerate(all_bounces):
            bpt = world_to_court(b["x"], b["y"])
            # Recent bounces brighter
            age = len(all_bounces) - 1 - bi
            brightness = max(0.3, 1.0 - age * 0.1)
            if b["in_court"]:
                bc = (0, int(255 * brightness), int(255 * brightness))  # yellow
            else:
                bc = (0, 0, int(255 * brightness))  # red
            cv2.circle(court_panel, bpt, 5, bc, -1, cv2.LINE_AA)
            cv2.circle(court_panel, bpt, 5, (255, 255, 255), 1, cv2.LINE_AA)
            # Label bounce number
            cv2.putText(court_panel, str(bi + 1), (bpt[0] + 6, bpt[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Court panel title
        cv2.putText(court_panel, "Court View", (court_pw // 2 - 35, court_ph - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # Compose canvas
        canvas = np.zeros((canvas_h, CANVAS_W, 3), dtype=np.uint8)
        canvas[:half_h, :half_w] = small66
        canvas[:half_h, half_w:half_w * 2] = small68
        canvas[:court_ph, video_area_w:] = court_panel

        # Draw separator lines
        cv2.line(canvas, (half_w, 0), (half_w, half_h), (255, 255, 255), 1)
        cv2.line(canvas, (video_area_w, 0), (video_area_w, half_h), (255, 255, 255), 1)

        # Camera labels
        cv2.putText(canvas, "cam66", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "cam68", (half_w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Info bar at bottom
        bar_y = half_h
        info_bg = canvas[bar_y:, :]
        info_bg[:] = (30, 30, 30)

        # Frame number
        cv2.putText(canvas, f"Frame {fi}", (10, bar_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if cd is not None:
            rd_val = cd["ray_distance"]
            rd_color = ray_dist_color(rd_val)

            # Ray distance value
            cv2.putText(canvas, f"Ray Dist: {rd_val:.2f}m", (150, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rd_color, 2, cv2.LINE_AA)

            # Ray distance bar
            bar_x0 = 380
            bar_x1 = 580
            bar_fill = min(1.0, rd_val / 10.0)
            cv2.rectangle(canvas, (bar_x0, bar_y + 10), (bar_x1, bar_y + 30), (80, 80, 80), -1)
            cv2.rectangle(canvas, (bar_x0, bar_y + 10), (bar_x0 + int(200 * bar_fill), bar_y + 30), rd_color, -1)
            cv2.rectangle(canvas, (bar_x0, bar_y + 10), (bar_x1, bar_y + 30), (150, 150, 150), 1)

            # Agreement status
            if rd_val < RAY_DIST_AGREE:
                status = "AGREE"
                status_color = (0, 220, 0)
            elif rd_val >= RAY_DIST_DISAGREE:
                status = "DISAGREE"
                status_color = (0, 0, 220)
            else:
                status = "UNCERTAIN"
                status_color = (0, 180, 220)

            cv2.putText(canvas, status, (600, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

            # Errors vs GT
            err66_str = f"err66={cd['err66']:.1f}px" if cd.get("err66") is not None else "err66=N/A"
            err68_str = f"err68={cd['err68']:.1f}px" if cd.get("err68") is not None else "err68=N/A"
            cv2.putText(canvas, err66_str, (780, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, err68_str, (950, bar_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # 3D position
            pos = cd.get("pos_3d")
            if pos:
                cv2.putText(canvas, f"3D: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                            (1100, bar_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 220), 1, cv2.LINE_AA)

        # Rescued frame indicator
        if fi in rescued_66:
            cv2.putText(canvas, "RESCUED", (half_w - 120, bar_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)
            # Flash border on cam66
            cv2.rectangle(canvas, (0, 0), (half_w - 1, half_h - 1), (0, 255, 128), 3)

        if fi in rescued_68:
            cv2.putText(canvas, "RESCUED", (CANVAS_W - 120, bar_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (half_w, 0), (CANVAS_W - 1, half_h - 1), (0, 255, 128), 3)

        # Second row info
        if has_det66 and has_det68:
            d66 = det66[fi_str]
            d68 = det68[fi_str]
            y66 = d66.get("yolo_conf", -1)
            y68 = d68.get("yolo_conf", -1)
            cv2.putText(canvas, f"YOLO: {y66:.2f}" if y66 >= 0 else "YOLO: N/A",
                        (10, bar_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"YOLO: {y68:.2f}" if y68 >= 0 else "YOLO: N/A",
                        (half_w + 10, bar_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Rally state banner (top center) ─────────────────────────
        rally_st = rally_state_frames.get(fi) or last_rally_state
        if rally_st:
            last_rally_state = rally_st
            state_str = rally_st["state"].upper()
            if state_str == "RALLY":
                banner_color = (0, 180, 0)
                banner_text = f"RALLY #{rally_st['rally_id']}  strokes:{rally_st['stroke_count']}  bounces:{rally_st['bounce_count']}"
            elif state_str == "SERVING":
                banner_color = (0, 180, 220)
                banner_text = f"SERVING #{rally_st['rally_id']}  ({rally_st['server_side']})"
            else:
                banner_color = (100, 100, 100)
                banner_text = "IDLE"

            # Draw banner background
            bx = CANVAS_W // 2 - 200
            cv2.rectangle(canvas, (bx, 2), (bx + 400, 28), banner_color, -1)
            cv2.putText(canvas, banner_text, (bx + 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ── Rally end event ─────────────────────────────────────────
        if fi in rally_events:
            rev = rally_events[fi]
            reason_text = rev["end_reason"].upper()
            reason_colors = {
                "out": (0, 0, 255), "net": (0, 100, 255),
                "double_bounce": (0, 180, 255), "timeout": (128, 128, 128),
            }
            rc = reason_colors.get(rev["end_reason"], (200, 200, 200))
            cv2.putText(canvas, f"POINT END: {reason_text}", (CANVAS_W // 2 - 120, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc, 2, cv2.LINE_AA)
            # Show rally end for a few frames after the event
            rally_end_display = 30  # show for 30 frames
        elif rally_end_display > 0:
            rally_end_display -= 1
            if last_rally_event:
                rev = last_rally_event
                reason_text = rev["end_reason"].upper()
                reason_colors = {
                    "out": (0, 0, 255), "net": (0, 100, 255),
                    "double_bounce": (0, 180, 255), "timeout": (128, 128, 128),
                }
                rc = reason_colors.get(rev["end_reason"], (200, 200, 200))
                alpha = rally_end_display / 30.0
                rc_faded = tuple(int(v * alpha) for v in rc)
                cv2.putText(canvas, f"POINT END: {reason_text}", (CANVAS_W // 2 - 120, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, rc_faded, 2, cv2.LINE_AA)

        if fi in rally_events:
            last_rally_event = rally_events[fi]

        # ── Bounce marker (flash on both views) ─────────────────────
        if fi in bounce_frames:
            b = bounce_frames[fi]
            bounce_display = 20  # show bounce marker for 20 frames
            last_bounce = b

        if bounce_display > 0 and last_bounce:
            bounce_display -= 1
            b = last_bounce
            alpha = bounce_display / 20.0
            bc = (0, int(255 * alpha), int(255 * alpha))  # fading yellow

            # Draw "BOUNCE" text between the two views
            btext = f"BOUNCE ({b['x']:.1f}, {b['y']:.1f}) {'IN' if b['in_court'] else 'OUT'}"
            btx = CANVAS_W // 2 - 150
            cv2.putText(canvas, btext, (btx, half_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2, cv2.LINE_AA)

            # Draw bounce circle on the info bar
            side_label = b.get("side", "")
            src_label = b.get("source_camera", "")
            cv2.putText(canvas, f"src={src_label} side={side_label}",
                        (btx, half_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, bc, 1, cv2.LINE_AA)

        # Legend (bottom-right)
        legend_x = CANVAS_W - 350
        legend_y = bar_y + 45
        cv2.circle(canvas, (legend_x, legend_y), 5, (0, 220, 0), -1)
        cv2.putText(canvas, "Correct", (legend_x + 10, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.circle(canvas, (legend_x + 80, legend_y), 5, (0, 0, 220), -1)
        cv2.putText(canvas, "Wrong", (legend_x + 90, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.circle(canvas, (legend_x + 150, legend_y), 5, (0, 220, 220), -1)
        cv2.putText(canvas, "No GT", (legend_x + 160, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.drawMarker(canvas, (legend_x + 220, legend_y), (220, 180, 0), cv2.MARKER_DIAMOND, 10, 1)
        cv2.putText(canvas, "GT", (legend_x + 230, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        writer.write(canvas)

        if fi % 500 == 0:
            print(f"  Frame {fi}/{MAX_FRAMES}")

    writer.release()
    cap66.release()
    cap68.release()
    print(f"\nVideo saved to {OUT_PATH}")


if __name__ == "__main__":
    render_video()
