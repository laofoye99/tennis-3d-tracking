"""Replay saved frames through the dashboard YOLO event chain.

This is a deployment smoke harness, not a separate offline detector. It reuses
the dashboard detector output shape and publishes events through
Orchestrator._run_yolo_fuzzy_single_cam_locked().
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.orchestrator import Orchestrator
from app.pipeline.camera_pipeline import YOLO_META_KEYS
from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import create_detector
from app.pipeline.player_detector import PlayerPoseDetector
from app.pipeline.yolo_bounce_filter import (
    detect_single_camera_events,
    filter_dashboard_yolo_publishable_bounces,
)


def _frame_index(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def _iter_frame_paths(frames_dir: Path, max_frames: int | None) -> list[Path]:
    paths = sorted(
        [
            *frames_dir.glob("*.jpg"),
            *frames_dir.glob("*.jpeg"),
            *frames_dir.glob("*.png"),
        ],
        key=_frame_index,
    )
    if max_frames is not None and max_frames > 0:
        paths = paths[:max_frames]
    return paths


def _make_detection(
    *,
    camera_name: str,
    blobs: list[dict],
    homography: HomographyTransformer,
    frame_index: int,
    capture_ts: float,
) -> dict | None:
    if not blobs:
        return None

    top = blobs[0]
    px = float(top["pixel_x"])
    py = float(top["pixel_y"])
    conf = float(top.get("blob_sum", top.get("yolo_conf", 0.0)) or 0.0)
    wx, wy = homography.pixel_to_world(px, py)

    candidates = []
    for blob in blobs:
        bpx = float(blob["pixel_x"])
        bpy = float(blob["pixel_y"])
        bwx, bwy = homography.pixel_to_world(bpx, bpy)
        candidate = {
            "x": bwx,
            "y": bwy,
            "world_x": bwx,
            "world_y": bwy,
            "pixel_x": bpx,
            "pixel_y": bpy,
            "blob_sum": float(blob.get("blob_sum", blob.get("yolo_conf", 0.0)) or 0.0),
        }
        for key in YOLO_META_KEYS:
            if blob.get(key) is not None:
                candidate[key] = blob[key]
        candidates.append(candidate)

    detection = {
        "camera_name": camera_name,
        "x": wx,
        "y": wy,
        "pixel_x": px,
        "pixel_y": py,
        "confidence": conf,
        "blob_sum": conf,
        "timestamp": capture_ts,
        "capture_ts": capture_ts,
        "frame_index": frame_index,
        "candidates": candidates,
    }
    for key in (
        "yolo_conf",
        "source",
        "static_count",
        "static_status",
        "static_zone_id",
        "raw_candidates",
        "event_only_raw_candidates",
    ):
        if top.get(key) is not None:
            detection[key] = top[key]
    return detection


def _event_frame(event: dict) -> int | None:
    for key in ("frame_index", "frame"):
        value = event.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _world_distance_m(a: dict, b: dict) -> float | None:
    try:
        return float(
            ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2)
            ** 0.5
        )
    except Exception:
        return None


def _match_events_frame_space(
    reference: list[dict],
    candidate: list[dict],
    *,
    frame_tolerance: int = 10,
    space_tolerance_m: float = 3.0,
) -> dict:
    """Match two event lists by frame and court distance for replay audits."""
    unmatched = set(range(len(candidate)))
    matches: list[dict] = []
    misses: list[dict] = []
    for ref in reference:
        ref_frame = _event_frame(ref)
        if ref_frame is None:
            misses.append(ref)
            continue
        best_idx = None
        best_key = None
        for idx in sorted(unmatched):
            cand = candidate[idx]
            cand_frame = _event_frame(cand)
            if cand_frame is None:
                continue
            frame_delta = cand_frame - ref_frame
            if abs(frame_delta) > frame_tolerance:
                continue
            dist = _world_distance_m(ref, cand)
            if dist is None or dist > space_tolerance_m:
                continue
            key = (abs(frame_delta), dist)
            if best_key is None or key < best_key:
                best_idx = idx
                best_key = key
        if best_idx is None:
            misses.append(ref)
            continue
        cand = candidate[best_idx]
        unmatched.remove(best_idx)
        matches.append(
            {
                "reference_frame": ref_frame,
                "candidate_frame": _event_frame(cand),
                "frame_delta": _event_frame(cand) - ref_frame,
                "distance_m": round(_world_distance_m(ref, cand) or 0.0, 4),
            }
        )
    false_positives = [candidate[idx] for idx in sorted(unmatched)]
    return {
        "frame_tolerance": frame_tolerance,
        "space_tolerance_m": space_tolerance_m,
        "matches": matches,
        "misses": misses,
        "false_positives": false_positives,
        "match_count": len(matches),
        "miss_count": len(misses),
        "false_positive_count": len(false_positives),
    }


def run_replay(args: argparse.Namespace) -> dict:
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames directory not found: {frames_dir}")

    frame_paths = _iter_frame_paths(frames_dir, args.max_frames)
    if not frame_paths:
        raise FileNotFoundError(f"no image frames found in {frames_dir}")

    config = load_config(args.config)
    orch = Orchestrator(config)
    try:
        orch.switch_model("yolo_roadmap")
        orch._ws_enabled = bool(args.enable_ws_queue)
        cam_cfg = config.cameras[args.camera]
        homography = HomographyTransformer(config.homography.path, cam_cfg.homography_key)
        detector = create_detector(
            config.model.path,
            input_size=tuple(config.model.input_size),
            frames_in=config.model.frames_in,
            frames_out=config.model.frames_out,
            device=args.device or config.model.device,
            detector_type=config.model.detector_type,
        )

        player_detector = None
        player_settings = orch._player_detection_settings_for_model(config.model)
        if not args.skip_player and player_settings["enabled"]:
            player_detector = PlayerPoseDetector(
                player_settings["model_path"],
                device=args.device or player_settings["device"],
                conf=player_settings["conf"],
                imgsz=player_settings["imgsz"],
                use_tracking=player_settings["use_tracking"],
                run_every_n=player_settings["run_every_n_frames"],
            )

        cam_positions = orch._get_camera_positions()
        started = time.perf_counter()
        raw_frames = 0
        yolo_frames = 0
        player_pose_messages = 0
        offline_detections: list[dict] = []
        offline_player_poses: list[dict] = []

        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            raw_frames += 1
            frame_index = _frame_index(path)
            dataset_time_s = frame_index / float(args.fps)
            capture_ts = time.time()

            if player_detector is not None:
                player_dets = player_detector.detect(frame)
                if player_dets:
                    pose_msg = {
                        "type": "player_pose",
                        "camera_name": args.camera,
                        "frame_id": frame_index,
                        "timestamp": capture_ts,
                        "capture_ts": capture_ts,
                        "dataset_time_s": dataset_time_s,
                        "detections": player_dets,
                    }
                    offline_player_poses.append(pose_msg)
                    orch._handle_player_pose(
                        pose_msg,
                        cam_positions,
                    )
                    player_pose_messages += 1

            infer_frame = frame.copy()
            h, w = infer_frame.shape[:2]
            infer_frame[0 : min(41, h), 0 : min(603, w)] = 0
            blobs = detector.infer([infer_frame])[0]
            if not blobs:
                continue
            yolo_frames += 1

            detection = _make_detection(
                camera_name=args.camera,
                blobs=blobs,
                homography=homography,
                frame_index=frame_index,
                capture_ts=capture_ts,
            )
            if detection is None:
                continue
            detection["dataset_time_s"] = dataset_time_s
            offline_detections.append(dict(detection))
            with orch._analytics_lock:
                orch._run_yolo_fuzzy_single_cam_locked(args.camera, detection)

        analytics = orch.get_live_analytics()
        hb_cfg = config.hit_bounce_refiner
        offline_batch = detect_single_camera_events(
            offline_detections,
            camera_name=args.camera,
            player_pose_messages=offline_player_poses,
            homography=homography,
            fps=float(args.fps),
            hit_angle_thresh=float(getattr(hb_cfg, "hit_angle_thresh", 45.0)),
            hit_dist_px_net=float(getattr(hb_cfg, "bottom_hit_dist_px_net", 100.0)),
            hit_dist_px_base=float(getattr(hb_cfg, "bottom_hit_dist_px_base", 250.0)),
            lookback_frames=int(getattr(hb_cfg, "lookback_frames", 50) or 50),
            hit_suppress_frames=int(getattr(hb_cfg, "hit_suppression_frames", 3) or 3),
            clean_time_frames=int(getattr(hb_cfg, "clean_time_frames", 25) or 25),
            clean_space_meters=float(getattr(hb_cfg, "clean_space_meters", 1.5)),
        )
        streamed_bounces = analytics.get("recent_bounces", [])
        offline_bounces = offline_batch.get("bounces", [])
        dashboard_publish_filter = filter_dashboard_yolo_publishable_bounces(
            offline_bounces,
            hit_events=analytics.get("recent_hits", offline_batch.get("hits", [])),
            latest_frame=max(0, raw_frames - 1),
            hit_suppress_frames=int(getattr(hb_cfg, "hit_suppression_frames", 3) or 3),
            clean_time_frames=int(getattr(hb_cfg, "clean_time_frames", 25) or 25),
            clean_space_meters=float(getattr(hb_cfg, "clean_space_meters", 1.5)),
            release_delay_frames=int(getattr(hb_cfg, "release_delay_frames", 50) or 0),
        )
        offline_dashboard_bounces = dashboard_publish_filter.get("bounces", [])
        stream_vs_offline = _match_events_frame_space(
            streamed_bounces,
            offline_bounces,
            frame_tolerance=10,
            space_tolerance_m=3.0,
        )
        stream_vs_offline_dashboard = _match_events_frame_space(
            streamed_bounces,
            offline_dashboard_bounces,
            frame_tolerance=10,
            space_tolerance_m=3.0,
        )
        detector_stats = detector.get_runtime_stats() if hasattr(detector, "get_runtime_stats") else {}
        summary = {
            "source": "dashboard_yolo_single_cam_replay",
            "camera": args.camera,
            "frames_dir": str(frames_dir),
            "frames_requested": args.max_frames,
            "frames_read": raw_frames,
            "yolo_detection_frames": yolo_frames,
            "player_pose_messages": player_pose_messages,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model": orch.get_current_model(),
            "detector_stats": detector_stats,
            "offline_batch": {
                "source": "detect_single_camera_events_full_batch_same_yolo_detections",
                "detections": len(offline_detections),
                "player_pose_messages": len(offline_player_poses),
                "total_bounces": len(offline_bounces),
                "total_hits": len(offline_batch.get("hits", [])),
                "total_speed_events": len(offline_batch.get("speed_events", [])),
                "bounce_frames": [_event_frame(event) for event in offline_bounces],
                "hit_frames": [_event_frame(event) for event in offline_batch.get("hits", [])],
                "speed_frames": [_event_frame(event) for event in offline_batch.get("speed_events", [])],
                "bounces": offline_bounces,
                "hits": offline_batch.get("hits", []),
                "speed_events": offline_batch.get("speed_events", []),
                "raw_bounce_candidate_count": offline_batch.get("raw_bounce_candidate_count", 0),
                "suppressed_bounces_by_hit_window": offline_batch.get(
                    "suppressed_bounces_by_hit_window",
                    0,
                ),
                "deduped_bounces_after_hit": offline_batch.get(
                    "deduped_bounces_after_hit",
                    0,
                ),
                "gate_only_bounce_count": offline_batch.get("gate_only_bounce_count", 0),
                "out_rally_suppressed_bounce_count": offline_batch.get(
                    "out_rally_suppressed_bounce_count",
                    0,
                ),
            },
            "stream_vs_offline_batch": stream_vs_offline,
            "offline_dashboard_publishable": {
                "source": "offline_batch_plus_dashboard_publish_gate",
                "hit_source": "dashboard_stream_hits",
                "total_bounces": len(offline_dashboard_bounces),
                "bounce_frames": [_event_frame(event) for event in offline_dashboard_bounces],
                "bounces": offline_dashboard_bounces,
                "suppressed_bounces": dashboard_publish_filter.get("suppressed_bounces", []),
                "suppressed_frames": [
                    _event_frame(event)
                    for event in dashboard_publish_filter.get("suppressed_bounces", [])
                ],
                "suppression_counts": dashboard_publish_filter.get("suppression_counts", {}),
                "params": dashboard_publish_filter.get("params", {}),
            },
            "stream_vs_offline_dashboard_publishable": stream_vs_offline_dashboard,
            "analytics": {
                "total_bounces": analytics.get("total_bounces", 0),
                "total_hits": analytics.get("total_hits", 0),
                "total_speed_events": analytics.get("total_speed_events", 0),
                "ws_queue_enabled": bool(args.enable_ws_queue),
                "ws_pending_bounces": analytics.get("ws_pending_bounces", 0),
                "recent_bounces": analytics.get("recent_bounces", []),
                "recent_hits": analytics.get("recent_hits", []),
                "recent_speed_events": analytics.get("recent_speed_events", []),
                "single_cam_bounce_stats": analytics.get("single_cam_bounce_stats", {}),
                "post_filter_stats": analytics.get("post_filter_stats", {}),
            },
        }
        return summary
    finally:
        try:
            orch._manager.shutdown()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--camera", default="cam68")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-player", action="store_true")
    parser.add_argument("--enable-ws-queue", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary = run_replay(args)
    out_path = Path(args.out) if args.out else None
    if out_path is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("reports") / f"dashboard_yolo_replay_{ts}" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(out_path),
                **summary["analytics"],
                "offline_batch_total_bounces": summary["offline_batch"]["total_bounces"],
                "offline_batch_bounce_frames": summary["offline_batch"]["bounce_frames"],
                "offline_dashboard_publishable_total_bounces": summary[
                    "offline_dashboard_publishable"
                ]["total_bounces"],
                "offline_dashboard_publishable_bounce_frames": summary[
                    "offline_dashboard_publishable"
                ]["bounce_frames"],
                "offline_dashboard_publish_suppression_counts": summary[
                    "offline_dashboard_publishable"
                ]["suppression_counts"],
                "stream_vs_offline_batch": {
                    "match_count": summary["stream_vs_offline_batch"]["match_count"],
                    "miss_count": summary["stream_vs_offline_batch"]["miss_count"],
                    "false_positive_count": summary["stream_vs_offline_batch"][
                        "false_positive_count"
                    ],
                },
                "stream_vs_offline_dashboard_publishable": {
                    "match_count": summary[
                        "stream_vs_offline_dashboard_publishable"
                    ]["match_count"],
                    "miss_count": summary[
                        "stream_vs_offline_dashboard_publishable"
                    ]["miss_count"],
                    "false_positive_count": summary[
                        "stream_vs_offline_dashboard_publishable"
                    ]["false_positive_count"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
