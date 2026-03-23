"""Stream ball tracking data to 3D display via WebSocket.

Runs the full detection → matching → smoothing pipeline, then streams
ball positions frame-by-frame at video FPS to a WebSocket server.

Coordinate mapping (world → display):
    display_x = world_x / COURT_W                      → [0, 1], doubles lines
    display_y = 1.0 - (world_y / COURT_L)             → [0, 1], far=0, net=0.5, near=1

Usage:
    python -m tools.stream_to_3d_display
    python -m tools.stream_to_3d_display --video66 path1.mp4 --video68 path2.mp4
    python -m tools.stream_to_3d_display --speed 2.0   # 2x playback speed
"""

import argparse
import asyncio
import json
import logging
import ssl
import time
from pathlib import Path

import numpy as np
import websockets
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Court dimensions (singles — matches camera calibration)
SINGLES_LEFT = 1.37    # left singles sideline (m)
SINGLES_RIGHT = 6.86   # right singles sideline (m)
SINGLES_W = SINGLES_RIGHT - SINGLES_LEFT  # 5.49m
COURT_L = 23.77
NET_Y = COURT_L / 2  # 11.885

WS_URL = "wss://tennisserver.motionrivalry.com:8086/general"
ROOM = "general"


def world_to_display(x, y, z):
    """Convert world coordinates (meters) to display normalized coords.

    Display coordinate system (singles court):
        x: [0, 1]    singles width, 0 = left sideline, 1 = right sideline
        y: [0, 1]    court length, 0 = far baseline, 0.5 = net, 1 = near baseline
    """
    dx = (x - SINGLES_LEFT) / SINGLES_W  # [0, 1] within singles lines
    dy = 1.0 - (y / COURT_L)             # [0, 1], far=0, net=0.5, near=1
    return dx, dy


def run_pipeline(args):
    """Run detection + matching + smoothing, return per-frame ball data."""
    # Import here to avoid slow torch import if just checking --help
    from tools.render_tracking_video import (
        build_detector,
        detect_bounces,
        detect_net_crossings,
        load_config,
        run_detection_multi,
        smooth_trajectory_sg,
        triangulate_multi_blob,
    )

    cfg = load_config()

    logger.info("=== Phase 1: Detection ===")
    detector, postproc = build_detector(cfg)

    logger.info("--- cam66 ---")
    multi66, det66, n66 = run_detection_multi(
        args.video66, detector, postproc, args.max_frames, top_k=2,
    )

    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("--- cam68 ---")
    multi68, det68, n68 = run_detection_multi(
        args.video68, detector, postproc, args.max_frames, top_k=2,
    )

    n_frames = min(n66, n68)
    logger.info("Common frames: %d", n_frames)

    logger.info("=== Phase 2: Multi-Blob Matching ===")
    points_3d, chosen_pixels, stats = triangulate_multi_blob(multi66, multi68, cfg)

    logger.info("=== Phase 3: Smoothing + Events ===")
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)
    bounces = detect_bounces(smoothed_3d)
    net_crossings = detect_net_crossings(smoothed_3d, fps=25.0)

    # Build bounce/net-crossing frame lookup
    bounce_frames = {b["frame"]: b for b in bounces}
    crossing_frames = {nc["frame"]: nc for nc in net_crossings}

    return smoothed_3d, bounce_frames, crossing_frames, n_frames


def build_frame_data(smoothed_3d, bounce_frames, crossing_frames, n_frames):
    """Build per-frame data list for streaming."""
    frame_data = []
    for fi in range(n_frames):
        if fi in smoothed_3d:
            x, y, z = smoothed_3d[fi]
            dx, dy = world_to_display(x, y, z)

            # Determine ball type
            if fi in bounce_frames:
                ball_type = "bounce"
            elif fi in crossing_frames:
                ball_type = "net_crossing"
            else:
                ball_type = "rally"

            entry = {
                "frame": fi,
                "detected": 1,
                "x": round(dx, 6),
                "y": round(dy, 6),
                "z": round(z, 4),
                "type": ball_type,
            }
            # Attach speed from nearest preceding net crossing
            if fi in bounce_frames and crossing_frames:
                prev_frames = [nf for nf in crossing_frames if nf < fi]
                if prev_frames:
                    nearest_f = max(prev_frames)
                    entry["speed_kmh"] = round(crossing_frames[nearest_f]["speed_kmh"], 1)
                else:
                    entry["speed_kmh"] = 0

            frame_data.append(entry)
        else:
            frame_data.append({
                "frame": fi,
                "detected": 0,
                "x": 0,
                "y": 0,
                "z": 0,
                "type": "none",
            })
    return frame_data


async def stream_to_display(frame_data, fps, speed, batch_size,
                            ws_url=WS_URL, room=ROOM):
    """Connect to WebSocket and stream ball data at video FPS."""
    interval = 1.0 / fps / speed

    # Only use SSL context for wss:// URLs
    connect_kwargs = {}
    if ws_url.startswith("wss://"):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        connect_kwargs["ssl"] = ssl_ctx

    logger.info("Connecting to %s ...", ws_url)
    async with websockets.connect(ws_url, **connect_kwargs) as ws:
        logger.info("Connected! Streaming %d frames at %.1fx speed (%.1f fps effective)",
                     len(frame_data), speed, fps * speed)

        # Filter to bounce frames only and send in the new format
        bounce_data = [fd for fd in frame_data if fd["type"] == "bounce"]
        logger.info("Sending %d bounce events", len(bounce_data))

        for i, bd in enumerate(bounce_data):
            t0 = time.perf_counter()

            msg = json.dumps({
                "room": room,
                "msg": {
                    "message": "bounce_data",
                    "data": {
                        "bounce": {
                            "timeStamp": int(time.time() * 1000),
                            "x": bd["x"],
                            "y": bd["y"],
                            "speed": bd.get("speed_kmh", 0),
                        },
                    },
                },
            })

            await ws.send(msg)
            logger.info("  Bounce %d/%d: frame=%d  x=%.3f y=%.3f speed=%.0f km/h",
                         i + 1, len(bounce_data), bd["frame"],
                         bd["x"], bd["y"], bd.get("speed_kmh", 0))

            # Space out bounce sends at video timing
            if i + 1 < len(bounce_data):
                next_frame = bounce_data[i + 1]["frame"]
                dt = (next_frame - bd["frame"]) / fps / speed
                if dt > 0:
                    await asyncio.sleep(dt)

        logger.info("Streaming complete! %d frames sent.", len(frame_data))


def main():
    parser = argparse.ArgumentParser(description="Stream ball tracking to 3D display")
    parser.add_argument("--video66", default="uploads/cam66_20260307_173403_2min.mp4")
    parser.add_argument("--video68", default="uploads/cam68_20260307_173403_2min.mp4")
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (2.0 = 2x)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Frames per WebSocket message (1 = real-time, 10 = batch)")
    parser.add_argument("--ws-url", default=WS_URL,
                        help="WebSocket server URL")
    parser.add_argument("--room", default=ROOM,
                        help="Room name")
    parser.add_argument("--cache", default="exports/tracking_cache.json",
                        help="Cache file to skip re-running pipeline")
    args = parser.parse_args()

    ws_url = args.ws_url
    room = args.room

    cache_path = Path(args.cache)

    # Try to load cached pipeline results
    if cache_path.exists():
        logger.info("Loading cached pipeline results from %s", cache_path)
        with open(cache_path, "r") as f:
            cached = json.load(f)
        frame_data = cached["frame_data"]
        fps = cached["fps"]
        logger.info("Loaded %d frames from cache", len(frame_data))
    else:
        # Run full pipeline
        smoothed_3d, bounce_frames, crossing_frames, n_frames = run_pipeline(args)
        fps = 25.0
        frame_data = build_frame_data(smoothed_3d, bounce_frames, crossing_frames, n_frames)

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"fps": fps, "frame_data": frame_data}, f)
        logger.info("Cached pipeline results to %s", cache_path)

    # Stream to 3D display
    asyncio.run(stream_to_display(frame_data, fps, args.speed, args.batch,
                                  ws_url=ws_url, room=room))


if __name__ == "__main__":
    main()
