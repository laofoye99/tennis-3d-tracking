"""Single-camera YOLO trajectory bounce filter.

This is the dashboard-friendly version of the fuzzy 2D bounce logic from
``yolo_roadmap/main.py``. It operates on per-frame YOLO detections and emits
court-coordinate bounce events without requiring stereo triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SINGLES_X_MIN = -8.23 / 2.0
SINGLES_X_MAX = 8.23 / 2.0
COURT_Y_MIN = -23.77 / 2.0
COURT_Y_MAX = 23.77 / 2.0


@dataclass
class _Point:
    frame_index: int
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    confidence: float


def _is_in_court(x: float, y: float) -> bool:
    return (
        SINGLES_X_MIN <= x <= SINGLES_X_MAX
        and COURT_Y_MIN <= y <= COURT_Y_MAX
    )


def _dedupe_detections(detections: list[dict]) -> list[_Point]:
    """Keep the highest-confidence detection per frame."""
    by_frame: dict[int, _Point] = {}
    for det in detections:
        if det.get("frame_index") is None:
            continue
        px = det.get("pixel_x")
        py = det.get("pixel_y")
        wx = det.get("world_x", det.get("x"))
        wy = det.get("world_y", det.get("y"))
        if px is None or py is None or wx is None or wy is None:
            continue
        conf = float(det.get("yolo_conf", det.get("confidence", det.get("blob_sum", 0.0))) or 0.0)
        point = _Point(
            frame_index=int(det["frame_index"]),
            pixel_x=float(px),
            pixel_y=float(py),
            world_x=float(wx),
            world_y=float(wy),
            confidence=conf,
        )
        prev = by_frame.get(point.frame_index)
        if prev is None or point.confidence >= prev.confidence:
            by_frame[point.frame_index] = point
    return [by_frame[k] for k in sorted(by_frame)]


def _fill_small_gaps(points: list[_Point], max_gap: int) -> list[_Point]:
    if len(points) < 2:
        return points
    filled: list[_Point] = []
    for a, b in zip(points, points[1:]):
        filled.append(a)
        gap = b.frame_index - a.frame_index
        if 1 < gap <= max_gap + 1:
            for step in range(1, gap):
                t = step / gap
                filled.append(
                    _Point(
                        frame_index=a.frame_index + step,
                        pixel_x=(1 - t) * a.pixel_x + t * b.pixel_x,
                        pixel_y=(1 - t) * a.pixel_y + t * b.pixel_y,
                        world_x=(1 - t) * a.world_x + t * b.world_x,
                        world_y=(1 - t) * a.world_y + t * b.world_y,
                        confidence=(1 - t) * a.confidence + t * b.confidence,
                    )
                )
    filled.append(points[-1])
    return filled


def _smooth_points(points: list[_Point], window: int) -> list[_Point]:
    if window <= 1 or len(points) <= 2:
        return points
    half = window // 2
    smoothed: list[_Point] = []
    for i, point in enumerate(points):
        lo = max(0, i - half)
        hi = min(len(points), i + half + 1)
        chunk = points[lo:hi]
        smoothed.append(
            _Point(
                frame_index=point.frame_index,
                pixel_x=float(np.mean([p.pixel_x for p in chunk])),
                pixel_y=float(np.mean([p.pixel_y for p in chunk])),
                world_x=float(np.mean([p.world_x for p in chunk])),
                world_y=float(np.mean([p.world_y for p in chunk])),
                confidence=float(np.mean([p.confidence for p in chunk])),
            )
        )
    return smoothed


def _build_lookup(points: list[_Point]) -> tuple[list[tuple[float, float] | None], dict[int, _Point]]:
    if not points:
        return [], {}
    max_frame = max(p.frame_index for p in points)
    lookup: list[tuple[float, float] | None] = [None] * (max_frame + 1)
    point_by_frame: dict[int, _Point] = {}
    for p in points:
        lookup[p.frame_index] = (p.pixel_x, p.pixel_y)
        point_by_frame[p.frame_index] = p
    return lookup, point_by_frame


def evaluate_bounces_fuzzy(
    lookup_table: list[tuple[float, float] | None],
    window: int,
    angle_thresh: float,
    momentum_thresh: float,
    tolerance: int = 2,
) -> tuple[dict[int, tuple[float, float]], dict[int, dict[str, Any]]]:
    """Detect 2D trajectory bounces using angle + nearby Y/momentum evidence."""
    candidate_bounces: set[int] = set()
    frame_stats: dict[int, dict[str, Any]] = {}

    valid_frames = [i for i, pt in enumerate(lookup_table) if pt is not None]
    for i in range(len(valid_frames)):
        curr_idx = valid_frames[i]
        if i < window or i >= len(valid_frames) - window:
            continue

        prev_idx = valid_frames[i - window]
        next_idx = valid_frames[i + window]
        if (curr_idx - prev_idx) > window * 3 or (next_idx - curr_idx) > window * 3:
            continue

        p_prev = lookup_table[prev_idx]
        p_curr = lookup_table[curr_idx]
        p_next = lookup_table[next_idx]
        if p_prev is None or p_curr is None or p_next is None:
            continue

        v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])

        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)
        angle = 0.0
        y_reversal = False
        delta_v = 0.0

        if norm_in > 1e-5 and norm_out > 1e-5:
            cos_theta = np.clip(np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_theta)))
            y_reversal = bool(v_in[1] > 0 and v_out[1] < 0)
            speed_in = norm_in / max(1, curr_idx - prev_idx)
            speed_out = norm_out / max(1, next_idx - curr_idx)
            delta_v = float(abs(speed_in - speed_out))

        frame_stats[curr_idx] = {
            "angle": angle,
            "y_reversal": y_reversal,
            "delta_v": delta_v,
            "angle_ok": angle >= angle_thresh,
            "y_ok": y_reversal,
            "mom_ok": delta_v >= momentum_thresh,
        }

    for curr_idx, stats in frame_stats.items():
        if not stats["angle_ok"]:
            continue
        local_y_ok = False
        local_mom_ok = False
        for j in range(curr_idx - tolerance, curr_idx + tolerance + 1):
            if j in frame_stats:
                local_y_ok = local_y_ok or bool(frame_stats[j]["y_ok"])
                local_mom_ok = local_mom_ok or bool(frame_stats[j]["mom_ok"])
        if local_y_ok or local_mom_ok:
            candidate_bounces.add(curr_idx)

    bounces: dict[int, tuple[float, float]] = {}
    sorted_bounces = sorted(candidate_bounces)
    if not sorted_bounces:
        return bounces, frame_stats

    cluster = [sorted_bounces[0]]
    for idx in sorted_bounces[1:]:
        if idx - cluster[-1] <= window * 2 + tolerance:
            cluster.append(idx)
        else:
            best_idx = max(cluster, key=lambda k: lookup_table[k][1] if lookup_table[k] is not None else -1e9)
            pt = lookup_table[best_idx]
            if pt is not None:
                bounces[best_idx] = pt
            cluster = [idx]

    best_idx = max(cluster, key=lambda k: lookup_table[k][1] if lookup_table[k] is not None else -1e9)
    pt = lookup_table[best_idx]
    if pt is not None:
        bounces[best_idx] = pt
    return bounces, frame_stats


def detect_single_camera_bounces(
    detections: list[dict],
    *,
    camera_name: str = "",
    max_gap: int = 3,
    smooth_window: int = 3,
    filter_window: int = 3,
    angle_thresh: float = 10.0,
    momentum_thresh: float = 15.0,
    tolerance: int = 2,
) -> dict[str, Any]:
    """Run the YOLO roadmap fuzzy bounce filter on one camera's detections."""
    points = _dedupe_detections(detections)
    raw_count = len(points)
    points = _fill_small_gaps(points, max_gap=max_gap)
    points = _smooth_points(points, window=smooth_window)
    lookup, point_by_frame = _build_lookup(points)
    bounces, frame_stats = evaluate_bounces_fuzzy(
        lookup,
        window=filter_window,
        angle_thresh=angle_thresh,
        momentum_thresh=momentum_thresh,
        tolerance=tolerance,
    )

    events = []
    for seq, frame_index in enumerate(sorted(bounces), start=1):
        point = point_by_frame.get(frame_index)
        if point is None:
            continue
        stats = frame_stats.get(frame_index, {})
        in_court = _is_in_court(point.world_x, point.world_y)
        events.append({
            "sequence": seq,
            "frame": frame_index,
            "frame_index": frame_index,
            "camera": camera_name,
            "x": round(point.world_x, 4),
            "y": round(point.world_y, 4),
            "pixel_x": round(point.pixel_x, 2),
            "pixel_y": round(point.pixel_y, 2),
            "type": "IN" if in_court else "OUT",
            "in_court": in_court,
            "confidence": round(float(point.confidence), 4),
            "angle": round(float(stats.get("angle", 0.0)), 2),
            "delta_v": round(float(stats.get("delta_v", 0.0)), 2),
            "y_reversal": bool(stats.get("y_reversal", False)),
            "source": "yolo_fuzzy_single_cam",
        })

    return {
        "camera": camera_name,
        "detections": raw_count,
        "filtered_points": len(point_by_frame),
        "bounces": events,
        "count": len(events),
        "params": {
            "max_gap": max_gap,
            "smooth_window": smooth_window,
            "filter_window": filter_window,
            "angle_thresh": angle_thresh,
            "momentum_thresh": momentum_thresh,
            "tolerance": tolerance,
        },
    }


def _frame_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _project_player_poses(
    player_pose_messages: list[dict],
    homography: Any,
) -> dict[int, list[dict[str, Any]]]:
    """Index player detections by frame with court and hit-anchor coordinates."""
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for msg in player_pose_messages or []:
        frame = _frame_id(msg.get("frame_index", msg.get("frame_id")))
        if frame is None:
            continue
        projected = by_frame.setdefault(frame, [])
        for det in msg.get("detections", []) or []:
            bbox = det.get("bbox") or []
            if len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                continue

            foot_px = det.get("foot_px") or [(x1 + x2) / 2.0, y2]
            try:
                court_x, court_y = homography.pixel_to_world(float(foot_px[0]), float(foot_px[1]))
            except Exception:
                continue

            projected.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(det.get("conf", 0.0) or 0.0),
                "foot_px": [float(foot_px[0]), float(foot_px[1])],
                "court_x": float(court_x),
                "court_y": float(court_y),
                # Racket/contact proxy from the prototype: upper-body center.
                "hit_anchor_px": [(x1 + x2) / 2.0, y1 + (y2 - y1) * 0.3],
            })
    return by_frame


def _nearest_player_hit_match(
    event: dict[str, Any],
    players_by_frame: dict[int, list[dict[str, Any]]],
    *,
    roi_net_margin_m: float,
    hit_dist_px_net: float,
    hit_dist_px_base: float,
    hit_search_radius_frames: int,
) -> dict[str, Any] | None:
    frame = _frame_id(event.get("frame_index", event.get("frame")))
    if frame is None:
        return None
    ex = float(event.get("pixel_x", 0.0) or 0.0)
    ey = float(event.get("pixel_y", 0.0) or 0.0)
    world_y = float(event.get("y", 0.0) or 0.0)
    side_sign = 1.0 if world_y >= 0.0 else -1.0

    best: dict[str, Any] | None = None
    offsets = [0]
    for delta in range(1, hit_search_radius_frames + 1):
        offsets.extend([-delta, delta])

    for offset in offsets:
        pf = frame + offset
        for player in players_by_frame.get(pf, []):
            player_y = float(player["court_y"])
            # Keep the player on the same side of the net, with a small slack.
            if side_sign > 0 and player_y < -roi_net_margin_m:
                continue
            if side_sign < 0 and player_y > roi_net_margin_m:
                continue

            anchor_x, anchor_y = player["hit_anchor_px"]
            dist_px = float(np.hypot(ex - anchor_x, ey - anchor_y))
            depth = min(abs(player_y), abs(COURT_Y_MAX)) / max(abs(COURT_Y_MAX), 1e-6)
            dynamic_thr = hit_dist_px_net + (hit_dist_px_base - hit_dist_px_net) * depth
            score = dist_px / max(dynamic_thr, 1.0) + abs(offset) * 0.03
            if dist_px <= dynamic_thr and (best is None or score < best["score"]):
                best = {
                    "score": score,
                    "frame": pf,
                    "distance_px": round(dist_px, 2),
                    "threshold_px": round(dynamic_thr, 2),
                    "player": player,
                }
    return best


def _detect_single_cam_speed_crossings(
    points: list[_Point],
    *,
    fps: float,
    min_kmh: float,
    max_kmh: float,
    cooldown_frames: int,
) -> list[dict[str, Any]]:
    if len(points) < 2 or fps <= 0:
        return []

    events: list[dict[str, Any]] = []
    last_emit = -10**9
    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]
        if curr.frame_index - prev.frame_index <= 0:
            continue
        crossed = (
            (prev.world_y < 0.0 <= curr.world_y)
            or (prev.world_y > 0.0 >= curr.world_y)
        )
        if not crossed:
            continue
        if curr.frame_index - last_emit <= cooldown_frames:
            continue

        lo = max(0, i - 2)
        hi = min(len(points), i + 3)
        chunk = points[lo:hi]
        if len(chunk) < 2:
            continue
        first = chunk[0]
        last = chunk[-1]
        frame_span = last.frame_index - first.frame_index
        if frame_span <= 0:
            continue
        dt = frame_span / fps
        dist_m = float(np.hypot(last.world_x - first.world_x, last.world_y - first.world_y))
        speed_kmh = dist_m / dt * 3.6
        if not (min_kmh <= speed_kmh <= max_kmh):
            continue

        # Linear interpolation to y=0 for a more stable map point.
        denom = curr.world_y - prev.world_y
        t = 0.0 if abs(denom) < 1e-6 else float(np.clip((0.0 - prev.world_y) / denom, 0.0, 1.0))
        x_cross = (1.0 - t) * prev.world_x + t * curr.world_x
        px_cross = (1.0 - t) * prev.pixel_x + t * curr.pixel_x
        py_cross = (1.0 - t) * prev.pixel_y + t * curr.pixel_y
        events.append({
            "frame": curr.frame_index,
            "frame_index": curr.frame_index,
            "x": round(float(x_cross), 4),
            "y": 0.0,
            "pixel_x": round(float(px_cross), 2),
            "pixel_y": round(float(py_cross), 2),
            "speed_kmh": int(round(speed_kmh)),
            "direction": "near_to_far" if curr.world_y > prev.world_y else "far_to_near",
            "source": "single_cam_speed_crossing",
        })
        last_emit = curr.frame_index
    return events


def detect_single_camera_events(
    detections: list[dict],
    *,
    camera_name: str = "",
    player_pose_messages: list[dict] | None = None,
    homography: Any | None = None,
    fps: float = 25.0,
    max_gap: int = 3,
    smooth_window: int = 3,
    filter_window: int = 3,
    angle_thresh: float = 10.0,
    momentum_thresh: float = 15.0,
    tolerance: int = 2,
    hit_angle_thresh: float = 45.0,
    hit_dist_px_net: float = 100.0,
    hit_dist_px_base: float = 250.0,
    hit_search_radius_frames: int = 2,
    roi_net_margin_m: float = 1.5,
    speed_min_kmh: float = 30.0,
    speed_max_kmh: float = 220.0,
    speed_cooldown_frames: int = 12,
) -> dict[str, Any]:
    """Detect bounce, hit and speed events from a single-camera YOLO track.

    Bounce candidates still come from the original fuzzy angle/Y/momentum rule.
    Player proximity is used only to re-label a trajectory reversal as a hit.
    """
    points = _dedupe_detections(detections)
    raw_count = len(points)
    filled = _fill_small_gaps(points, max_gap=max_gap)
    smoothed = _smooth_points(filled, window=smooth_window)
    lookup, point_by_frame = _build_lookup(smoothed)
    bounces, frame_stats = evaluate_bounces_fuzzy(
        lookup,
        window=filter_window,
        angle_thresh=angle_thresh,
        momentum_thresh=momentum_thresh,
        tolerance=tolerance,
    )

    players_by_frame = (
        _project_player_poses(player_pose_messages or [], homography)
        if homography is not None and player_pose_messages
        else {}
    )

    landing_events: list[dict[str, Any]] = []
    hit_events: list[dict[str, Any]] = []
    for frame_index in sorted(bounces):
        point = point_by_frame.get(frame_index)
        if point is None:
            continue
        stats = frame_stats.get(frame_index, {})
        base_event = {
            "sequence": 0,
            "frame": frame_index,
            "frame_index": frame_index,
            "camera": camera_name,
            "x": round(point.world_x, 4),
            "y": round(point.world_y, 4),
            "pixel_x": round(point.pixel_x, 2),
            "pixel_y": round(point.pixel_y, 2),
            "in_court": _is_in_court(point.world_x, point.world_y),
            "confidence": round(float(point.confidence), 4),
            "angle": round(float(stats.get("angle", 0.0)), 2),
            "delta_v": round(float(stats.get("delta_v", 0.0)), 2),
            "y_reversal": bool(stats.get("y_reversal", False)),
        }

        player_match = None
        if float(base_event["angle"]) >= hit_angle_thresh:
            player_match = _nearest_player_hit_match(
                base_event,
                players_by_frame,
                roi_net_margin_m=roi_net_margin_m,
                hit_dist_px_net=hit_dist_px_net,
                hit_dist_px_base=hit_dist_px_base,
                hit_search_radius_frames=hit_search_radius_frames,
            )

        if player_match is not None:
            player = player_match["player"]
            hit_event = {
                **base_event,
                "type": "HIT",
                "kind": "hit",
                "source": "yolo_fuzzy_player_hit",
                "player_frame": player_match["frame"],
                "player_distance_px": player_match["distance_px"],
                "player_threshold_px": player_match["threshold_px"],
                "player_court_x": round(float(player["court_x"]), 4),
                "player_court_y": round(float(player["court_y"]), 4),
                "player_conf": round(float(player.get("conf", 0.0)), 4),
            }
            hit_events.append(hit_event)
        else:
            landing_events.append({
                **base_event,
                "type": "IN" if base_event["in_court"] else "OUT",
                "kind": "bounce",
                "source": "yolo_fuzzy_single_cam",
            })

    for seq, event in enumerate(landing_events, start=1):
        event["sequence"] = seq
    for seq, event in enumerate(hit_events, start=1):
        event["sequence"] = seq

    speed_events = _detect_single_cam_speed_crossings(
        smoothed,
        fps=fps,
        min_kmh=speed_min_kmh,
        max_kmh=speed_max_kmh,
        cooldown_frames=speed_cooldown_frames,
    )

    return {
        "camera": camera_name,
        "detections": raw_count,
        "filtered_points": len(point_by_frame),
        "bounces": landing_events,
        "hits": hit_events,
        "speed_events": speed_events,
        "count": len(landing_events),
        "hit_count": len(hit_events),
        "speed_count": len(speed_events),
        "params": {
            "max_gap": max_gap,
            "smooth_window": smooth_window,
            "filter_window": filter_window,
            "angle_thresh": angle_thresh,
            "momentum_thresh": momentum_thresh,
            "tolerance": tolerance,
            "hit_angle_thresh": hit_angle_thresh,
            "hit_dist_px_net": hit_dist_px_net,
            "hit_dist_px_base": hit_dist_px_base,
            "hit_search_radius_frames": hit_search_radius_frames,
            "roi_net_margin_m": roi_net_margin_m,
            "speed_min_kmh": speed_min_kmh,
            "speed_max_kmh": speed_max_kmh,
        },
    }
