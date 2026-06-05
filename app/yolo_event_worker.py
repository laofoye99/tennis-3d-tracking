"""Background worker for realtime YOLO hit/bounce event analysis."""

from __future__ import annotations

import queue
import time
from typing import Any


def _nearest_capture_ts(detections: list[dict], event_frame: int, fallback: float) -> float:
    if not detections:
        return fallback
    try:
        nearest = min(
            detections,
            key=lambda d: abs(int(d.get("frame_index", event_frame)) - event_frame),
        )
    except Exception:
        return fallback
    ts = nearest.get("capture_ts", nearest.get("timestamp", fallback))
    try:
        return float(ts)
    except Exception:
        return fallback


def _event_frame(event: dict) -> int | None:
    frame = event.get("frame_index", event.get("frame"))
    if frame is None:
        return None
    try:
        return int(frame)
    except Exception:
        return None


def _attach_event_capture_ts(result: dict[str, Any], detections: list[dict]) -> None:
    now = time.time()
    for key in ("bounces", "hits", "speed_events"):
        for event in result.get(key, []) or []:
            frame = _event_frame(event)
            if frame is None:
                continue
            event.setdefault("capture_ts", _nearest_capture_ts(detections, frame, now))
            event.setdefault("timestamp", event.get("capture_ts"))


def run_yolo_event_worker(
    *,
    camera_name: str,
    homography_path: str,
    homography_key: str,
    hit_bounce_config: dict[str, Any],
    task_queue,
    result_queue,
    stop_event,
) -> None:
    """Run expensive YOLO event analysis outside the dashboard HTTP process."""
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.yolo_bounce_filter import detect_single_camera_events

    homography = None
    try:
        homography = HomographyTransformer(homography_path, homography_key)
    except Exception:
        homography = None

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if task is None:
            break
        if not isinstance(task, dict):
            continue

        detections = list(task.get("detections") or [])
        player_poses = list(task.get("player_poses") or [])
        latest_frame = task.get("latest_frame")
        task_id = task.get("task_id")

        analysis_t0 = time.perf_counter()
        try:
            result = detect_single_camera_events(
                detections,
                camera_name=camera_name,
                player_pose_messages=player_poses,
                homography=homography,
                hit_angle_thresh=float(hit_bounce_config.get("hit_angle_thresh", 45.0)),
                hit_dist_px_net=float(hit_bounce_config.get("bottom_hit_dist_px_net", 100.0)),
                hit_dist_px_base=float(hit_bounce_config.get("bottom_hit_dist_px_base", 250.0)),
                lookback_frames=int(hit_bounce_config.get("lookback_frames", 50) or 50),
                hit_suppress_frames=int(hit_bounce_config.get("hit_suppression_frames", 3) or 3),
                clean_time_frames=int(hit_bounce_config.get("clean_time_frames", 25) or 25),
                clean_space_meters=float(hit_bounce_config.get("clean_space_meters", 1.5)),
            )
            _attach_event_capture_ts(result, detections)
            payload = {
                "type": "yolo_event_result",
                "camera_name": camera_name,
                "task_id": task_id,
                "latest_frame": latest_frame,
                "result": result,
                "analysis_ms": round((time.perf_counter() - analysis_t0) * 1000.0, 2),
                "error": None,
            }
        except Exception as exc:
            payload = {
                "type": "yolo_event_result",
                "camera_name": camera_name,
                "task_id": task_id,
                "latest_frame": latest_frame,
                "result": None,
                "analysis_ms": round((time.perf_counter() - analysis_t0) * 1000.0, 2),
                "error": str(exc),
            }

        try:
            result_queue.put_nowait(payload)
        except Exception:
            pass
