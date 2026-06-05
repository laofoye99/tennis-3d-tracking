"""Compare cam68 clip11 GT match-ball bounces with offline/dashboard events.

The GT LabelMe files store events in each ball shape ``description`` field,
not in the top-level ``events`` array. Only shapes with
``is_match_ball=true`` are used as GT.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.pipeline.homography import HomographyTransformer
from app.pipeline.yolo_bounce_filter import detect_single_camera_events


DEFAULT_GT_DIR = Path(
    r"D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min"
)


def _parse_description(desc: str | None) -> dict[str, str]:
    meta: dict[str, str] = {}
    for part in str(desc or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def _shape_center(points: list[list[float]]) -> tuple[float, float, list[float]]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return sum(xs) / len(xs), sum(ys) / len(ys), bbox


def load_gt(gt_dir: Path, *, max_frames: int | None = None) -> dict[str, Any]:
    files = sorted(gt_dir.glob("*.json"))
    if max_frames is not None and max_frames > 0:
        files = files[:max_frames]
    match_points: list[dict[str, Any]] = []
    match_bounces: list[dict[str, Any]] = []
    match_hits: list[dict[str, Any]] = []
    all_ball_event_counts: dict[str, int] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        frame = int(path.stem)
        for shape in data.get("shapes") or []:
            if shape.get("label") != "ball":
                continue
            meta = _parse_description(shape.get("description"))
            event = meta.get("ball_event")
            if event:
                all_ball_event_counts[event] = all_ball_event_counts.get(event, 0) + 1
            if str(meta.get("is_match_ball", "")).lower() != "true":
                continue
            cx, cy, bbox = _shape_center(shape.get("points") or [])
            item = {
                "frame": frame,
                "frame_index": frame,
                "pixel_x": round(cx, 3),
                "pixel_y": round(cy, 3),
                "bbox": [round(v, 3) for v in bbox],
                "event": event,
                "description": shape.get("description") or "",
            }
            match_points.append(item)
            if event == "bounce":
                match_bounces.append(item)
            elif event == "hit":
                match_hits.append(item)
    return {
        "frames_scanned": len(files),
        "match_points": match_points,
        "match_bounces": match_bounces,
        "match_hits": match_hits,
        "all_ball_event_counts": all_ball_event_counts,
    }


def _in_court(wx: float, wy: float) -> bool:
    return abs(wx) <= 4.115 and abs(wy) <= 11.885


def add_world_coords(items: list[dict[str, Any]], homography: HomographyTransformer) -> None:
    for item in items:
        wx, wy = homography.pixel_to_world(item["pixel_x"], item["pixel_y"])
        item["x"] = round(wx, 4)
        item["y"] = round(wy, 4)
        item["in_court"] = _in_court(wx, wy)


def gt_points_to_detections(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for point in points:
        det = {
            "camera_name": "cam68",
            "frame_index": int(point["frame"]),
            "pixel_x": float(point["pixel_x"]),
            "pixel_y": float(point["pixel_y"]),
            "x": float(point["x"]),
            "y": float(point["y"]),
            "world_x": float(point["x"]),
            "world_y": float(point["y"]),
            "confidence": 1.0,
            "blob_sum": 1.0,
            "yolo_conf": 1.0,
            "source": "gt_match_ball",
            "candidates": [
                {
                    "pixel_x": float(point["pixel_x"]),
                    "pixel_y": float(point["pixel_y"]),
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "world_x": float(point["x"]),
                    "world_y": float(point["y"]),
                    "blob_sum": 1.0,
                    "yolo_conf": 1.0,
                    "source": "gt_match_ball",
                }
            ],
        }
        detections.append(det)
    return detections


def _event_frame(event: dict[str, Any]) -> int | None:
    value = event.get("frame_index", event.get("frame"))
    try:
        return int(value)
    except Exception:
        return None


def match_by_frame(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    *,
    tolerance: int,
) -> dict[str, Any]:
    unmatched_pred = list(pred_events)
    matches: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for gt in gt_events:
        gt_frame = _event_frame(gt)
        if gt_frame is None:
            continue
        best_idx = None
        best_delta = tolerance + 1
        for idx, pred in enumerate(unmatched_pred):
            pred_frame = _event_frame(pred)
            if pred_frame is None:
                continue
            delta = abs(pred_frame - gt_frame)
            if delta <= tolerance and delta < best_delta:
                best_idx = idx
                best_delta = delta
        if best_idx is None:
            misses.append(gt)
            continue
        pred = unmatched_pred.pop(best_idx)
        matches.append(
            {
                "gt_frame": gt_frame,
                "pred_frame": _event_frame(pred),
                "frame_delta": _event_frame(pred) - gt_frame,
                "gt": gt,
                "pred": pred,
            }
        )
    return {
        "tolerance_frames": tolerance,
        "matches": matches,
        "misses": misses,
        "false_positives": unmatched_pred,
        "match_count": len(matches),
        "miss_count": len(misses),
        "false_positive_count": len(unmatched_pred),
    }


def _world_distance_m(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    try:
        return ((float(a["x"]) - float(b["x"])) ** 2 + (float(a["y"]) - float(b["y"])) ** 2) ** 0.5
    except Exception:
        return None


def match_by_frame_and_space(
    gt_events: list[dict[str, Any]],
    pred_events: list[dict[str, Any]],
    *,
    frame_tolerance: int,
    space_tolerance_m: float,
) -> dict[str, Any]:
    unmatched_pred = list(pred_events)
    matches: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for gt in gt_events:
        gt_frame = _event_frame(gt)
        if gt_frame is None:
            continue
        best_idx = None
        best_key: tuple[int, float] | None = None
        best_dist = None
        for idx, pred in enumerate(unmatched_pred):
            pred_frame = _event_frame(pred)
            if pred_frame is None:
                continue
            frame_delta_abs = abs(pred_frame - gt_frame)
            if frame_delta_abs > frame_tolerance:
                continue
            distance_m = _world_distance_m(gt, pred)
            if distance_m is None or distance_m > space_tolerance_m:
                continue
            key = (frame_delta_abs, distance_m)
            if best_key is None or key < best_key:
                best_idx = idx
                best_key = key
                best_dist = distance_m
        if best_idx is None:
            misses.append(gt)
            continue
        pred = unmatched_pred.pop(best_idx)
        matches.append(
            {
                "gt_frame": gt_frame,
                "pred_frame": _event_frame(pred),
                "frame_delta": _event_frame(pred) - gt_frame,
                "distance_m": round(float(best_dist), 4) if best_dist is not None else None,
                "gt": gt,
                "pred": pred,
            }
        )
    return {
        "tolerance_frames": frame_tolerance,
        "space_tolerance_m": space_tolerance_m,
        "matches": matches,
        "misses": misses,
        "false_positives": unmatched_pred,
        "match_count": len(matches),
        "miss_count": len(misses),
        "false_positive_count": len(unmatched_pred),
    }


def load_dashboard_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    analytics = data.get("analytics", data)
    offline_dashboard = data.get("offline_dashboard_publishable") or {}
    return {
        "path": str(path),
        "recent_bounces": analytics.get("recent_bounces", []),
        "recent_hits": analytics.get("recent_hits", []),
        "total_bounces": analytics.get("total_bounces"),
        "total_hits": analytics.get("total_hits"),
        "single_cam_bounce_stats": analytics.get("single_cam_bounce_stats", {}),
        "offline_dashboard_publishable": offline_dashboard,
        "stream_vs_offline_dashboard_publishable": data.get(
            "stream_vs_offline_dashboard_publishable"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--camera", default="cam68")
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--frame-tolerance", type=int, default=8)
    parser.add_argument("--frame-space-tolerance", type=int, default=10)
    parser.add_argument("--space-tolerance-meters", type=float, default=3.0)
    parser.add_argument("--dashboard-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cam_cfg = config.cameras[args.camera]
    homography = HomographyTransformer(config.homography.path, cam_cfg.homography_key)

    gt = load_gt(args.gt_dir, max_frames=args.max_frames)
    add_world_coords(gt["match_points"], homography)
    add_world_coords(gt["match_bounces"], homography)
    add_world_coords(gt["match_hits"], homography)

    gt_detections = gt_points_to_detections(gt["match_points"])
    hb_cfg = config.hit_bounce_refiner
    offline = detect_single_camera_events(
        gt_detections,
        camera_name=args.camera,
        homography=homography,
        fps=25.0,
        max_gap=3,
        smooth_window=3,
        filter_window=3,
        angle_thresh=10,
        momentum_thresh=15,
        tolerance=2,
        hit_angle_thresh=float(getattr(hb_cfg, "hit_angle_thresh", 45.0)),
        hit_dist_px_net=float(getattr(hb_cfg, "bottom_hit_dist_px_net", 100.0)),
        hit_dist_px_base=float(getattr(hb_cfg, "bottom_hit_dist_px_base", 250.0)),
        lookback_frames=int(getattr(hb_cfg, "lookback_frames", 50) or 50),
        hit_suppress_frames=int(getattr(hb_cfg, "hit_suppression_frames", 3) or 3),
        clean_time_frames=int(getattr(hb_cfg, "clean_time_frames", 25) or 25),
        clean_space_meters=float(getattr(hb_cfg, "clean_space_meters", 1.5)),
    )
    offline_bounces = offline.get("bounces", [])
    offline_hits = offline.get("hits", [])
    dashboard = load_dashboard_summary(args.dashboard_summary)

    gt_driven_baseline = {
        "note": (
            "GT match-ball positions fed through detect_single_camera_events(); "
            "this is a detector-rule baseline, not a YOLO offline inference run."
        ),
        "bounce_count": len(offline_bounces),
        "hit_count": len(offline_hits),
        "bounce_frames": [_event_frame(event) for event in offline_bounces],
        "hit_frames": [_event_frame(event) for event in offline_hits],
        "stats": offline.get("stats", {}),
        "match_to_gt": match_by_frame(
            gt["match_bounces"],
            offline_bounces,
            tolerance=args.frame_tolerance,
        ),
        "match_to_gt_frame_space": match_by_frame_and_space(
            gt["match_bounces"],
            offline_bounces,
            frame_tolerance=args.frame_space_tolerance,
            space_tolerance_m=args.space_tolerance_meters,
        ),
    }

    report = {
        "camera": args.camera,
        "gt_dir": str(args.gt_dir),
        "max_frames": args.max_frames,
        "frame_tolerance": args.frame_tolerance,
        "gt": {
            "frames_scanned": gt["frames_scanned"],
            "match_points": len(gt["match_points"]),
            "match_bounce_count": len(gt["match_bounces"]),
            "match_hit_count": len(gt["match_hits"]),
            "match_bounce_frames": [item["frame"] for item in gt["match_bounces"]],
            "match_hit_frames": [item["frame"] for item in gt["match_hits"]],
            "all_ball_event_counts": gt["all_ball_event_counts"],
        },
        "gt_driven_detector_baseline": gt_driven_baseline,
        "offline_from_gt_match_ball": gt_driven_baseline,
        "dashboard_replay": None,
    }
    if dashboard is not None:
        report["dashboard_replay"] = {
            "summary_path": dashboard["path"],
            "total_bounces": dashboard.get("total_bounces"),
            "total_hits": dashboard.get("total_hits"),
            "recent_bounce_count": len(dashboard["recent_bounces"]),
            "recent_hit_count": len(dashboard["recent_hits"]),
            "recent_bounce_frames": [
                _event_frame(event) for event in dashboard["recent_bounces"]
            ],
            "match_to_gt": match_by_frame(
                gt["match_bounces"],
                dashboard["recent_bounces"],
                tolerance=args.frame_tolerance,
            ),
            "match_to_gt_frame_space": match_by_frame_and_space(
                gt["match_bounces"],
                dashboard["recent_bounces"],
                frame_tolerance=args.frame_space_tolerance,
                space_tolerance_m=args.space_tolerance_meters,
            ),
            "single_cam_bounce_stats": dashboard.get("single_cam_bounce_stats", {}),
        }
        offline_dashboard = dashboard.get("offline_dashboard_publishable") or {}
        offline_dashboard_bounces = offline_dashboard.get("bounces", [])
        if offline_dashboard_bounces:
            report["dashboard_replay"]["offline_dashboard_publishable"] = {
                "total_bounces": len(offline_dashboard_bounces),
                "bounce_frames": [
                    _event_frame(event) for event in offline_dashboard_bounces
                ],
                "suppression_counts": offline_dashboard.get(
                    "suppression_counts",
                    {},
                ),
                "match_to_gt": match_by_frame(
                    gt["match_bounces"],
                    offline_dashboard_bounces,
                    tolerance=args.frame_tolerance,
                ),
                "match_to_gt_frame_space": match_by_frame_and_space(
                    gt["match_bounces"],
                    offline_dashboard_bounces,
                    frame_tolerance=args.frame_space_tolerance,
                    space_tolerance_m=args.space_tolerance_meters,
                ),
            }
        if dashboard.get("stream_vs_offline_dashboard_publishable"):
            report["dashboard_replay"][
                "stream_vs_offline_dashboard_publishable"
            ] = dashboard["stream_vs_offline_dashboard_publishable"]

    out_dir = args.out_dir
    if out_dir is None:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports") / f"cam68_clip11_gt_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printed = {
        "summary": str(out_path),
        "gt_bounces": report["gt"]["match_bounce_frames"],
        "gt_driven_baseline_note": gt_driven_baseline["note"],
        "gt_driven_baseline_bounces": gt_driven_baseline["bounce_frames"],
        "gt_driven_baseline_match_count": gt_driven_baseline["match_to_gt"][
            "match_count"
        ],
        "gt_driven_baseline_miss_count": gt_driven_baseline["match_to_gt"][
            "miss_count"
        ],
        "gt_driven_baseline_false_positive_count": gt_driven_baseline[
            "match_to_gt"
        ]["false_positive_count"],
        "gt_driven_baseline_frame_space_match_count": gt_driven_baseline[
            "match_to_gt_frame_space"
        ]["match_count"],
        "gt_driven_baseline_frame_space_miss_count": gt_driven_baseline[
            "match_to_gt_frame_space"
        ]["miss_count"],
        "gt_driven_baseline_frame_space_false_positive_count": gt_driven_baseline[
            "match_to_gt_frame_space"
        ]["false_positive_count"],
    }
    if report.get("dashboard_replay"):
        dash = report["dashboard_replay"]
        dash_space = dash["match_to_gt_frame_space"]
        printed.update(
            {
                "dashboard_summary": dash["summary_path"],
                "dashboard_bounces": dash["recent_bounce_frames"],
                "dashboard_frame_space_match_count": dash_space["match_count"],
                "dashboard_frame_space_miss_count": dash_space["miss_count"],
                "dashboard_frame_space_false_positive_count": dash_space[
                    "false_positive_count"
                ],
            }
        )
        if dash.get("offline_dashboard_publishable"):
            off_dash = dash["offline_dashboard_publishable"]
            off_space = off_dash["match_to_gt_frame_space"]
            printed.update(
                {
                    "offline_dashboard_publishable_bounces": off_dash[
                        "bounce_frames"
                    ],
                    "offline_dashboard_publish_suppression_counts": off_dash[
                        "suppression_counts"
                    ],
                    "offline_dashboard_frame_space_match_count": off_space[
                        "match_count"
                    ],
                    "offline_dashboard_frame_space_miss_count": off_space[
                        "miss_count"
                    ],
                    "offline_dashboard_frame_space_false_positive_count": off_space[
                        "false_positive_count"
                    ],
                }
            )
        if dash.get("stream_vs_offline_dashboard_publishable"):
            stream_vs_off = dash["stream_vs_offline_dashboard_publishable"]
            printed.update(
                {
                    "stream_vs_offline_dashboard_match_count": stream_vs_off[
                        "match_count"
                    ],
                    "stream_vs_offline_dashboard_miss_count": stream_vs_off[
                        "miss_count"
                    ],
                    "stream_vs_offline_dashboard_false_positive_count": stream_vs_off[
                        "false_positive_count"
                    ],
                }
            )
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
