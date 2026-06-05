"""Single-camera YOLO trajectory bounce filter.

This is the dashboard-friendly version of the fuzzy 2D bounce logic from
``yolo_roadmap/main.py``. It operates on per-frame YOLO detections and emits
court-coordinate bounce events without requiring stereo triangulation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


SINGLES_X_MIN = -8.23 / 2.0
SINGLES_X_MAX = 8.23 / 2.0
COURT_Y_MIN = -23.77 / 2.0
COURT_Y_MAX = 23.77 / 2.0
DOUBLES_X_MIN = -10.97 / 2.0
DOUBLES_X_MAX = 10.97 / 2.0

DEFAULT_LOOKBACK_FRAMES = 50
DEFAULT_HIT_SUPPRESS_FRAMES = 3
DEFAULT_CLEAN_TIME_FRAMES = 25
DEFAULT_CLEAN_SPACE_METERS = 1.5
DEFAULT_HIT_CLEAN_SPACE_METERS = 1.8
DEFAULT_NET_OFFSET_PX = 50.0
DEFAULT_SPEED_LINE_OFFSET_PX = 150.0
DEFAULT_SPEED_COEF_UP = 1.5
DEFAULT_SPEED_COEF_DOWN = 1.5
DEFAULT_SPEED_SEARCH_FRAMES = 25
DEFAULT_OUT_RESTART_HIT_GAP_FRAMES = 100
DEFAULT_OUT_RESTART_SPEED_KMH = 20.0
DEFAULT_GATE_ONLY_OUT_MARGIN_M = 1.0
DEFAULT_GATE_ONLY_OUT_MIN_SCORE = 160.0
DEFAULT_GATE_ONLY_OUT_MIN_DELTA_V = 3.0
DEFAULT_DASHBOARD_MIN_BOUNCE_FRAME = 50
DEFAULT_DASHBOARD_MIN_BOUNCE_HISTORY = 8
DEFAULT_DASHBOARD_WEAK_NON_REVERSAL_MAX_ANGLE = 45.0
DEFAULT_DASHBOARD_WEAK_NON_REVERSAL_MIN_SCORE = 90.0
DEFAULT_DASHBOARD_LIVE_DUPLICATE_SPACE_METERS = 2.5
DEFAULT_DASHBOARD_RELEASE_DELAY_FRAMES = 12
DEFAULT_DASHBOARD_MIN_CONFIDENCE = 0.16
DEFAULT_BOUNCE_SPEED_BONUS_CAP_PX = 8.0
DEFAULT_BOUNCE_SPEED_BONUS_WEIGHT = 10.0
VERIFY_TRK_SEARCH_WINDOW = 10
VERIFY_TRK_CREATE_THR = 70.0
VERIFY_TRK_MAX_HISTORY = 60
VERIFY_TRK_EVAL_WINDOW = 7
VERIFY_TRK_STATIC_THR = 1.0
VERIFY_TRK_STATIC_RADIUS = 20.0
VERIFY_TRK_ZONE_PERSIST = 50
VERIFY_STITCH_MAX_GAP = 35
VERIFY_STITCH_MAX_ANGLE = 40.0
VERIFY_STITCH_MIN_SPEED = 2.0


@dataclass
class _Point:
    frame_index: int
    pixel_x: float
    pixel_y: float
    world_x: float
    world_y: float
    confidence: float


class _VerifyTrajectoryStitcher:
    def __init__(
        self,
        max_time_gap: int = VERIFY_STITCH_MAX_GAP,
        max_angle_deg: float = VERIFY_STITCH_MAX_ANGLE,
        min_speed: float = VERIFY_STITCH_MIN_SPEED,
    ):
        self.max_time_gap = max_time_gap
        self.max_angle_deg = max_angle_deg
        self.min_speed = min_speed

    def _get_velocity_vector(
        self,
        history_list: list[tuple[int, list[float], bool]],
        *,
        mode: str,
        sample_frames: int = 5,
    ) -> tuple[np.ndarray | None, float | None]:
        if len(history_list) < 2:
            return None, None
        pts = history_list[-sample_frames:] if mode == "out" else history_list[:sample_frames]
        if len(pts) < 2:
            return None, None
        start_frame, start_det = pts[0][:2]
        end_frame, end_det = pts[-1][:2]
        frame_span = end_frame - start_frame
        if frame_span <= 0:
            return None, None
        vec = np.array(end_det[0:2]) - np.array(start_det[0:2])
        speed = float(np.hypot(vec[0], vec[1]) / frame_span)
        return vec, speed

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = float(np.hypot(v1[0], v1[1]))
        norm2 = float(np.hypot(v2[0], v2[1]))
        if norm1 == 0.0 or norm2 == 0.0:
            return 180.0
        cos_theta = np.clip(float(np.dot(v1, v2)) / (norm1 * norm2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def stitch_queues(self, queues: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stitched_queues: list[dict[str, Any]] = []
        skip_ids: set[int] = set()
        active_queues = [q for q in queues if not q["is_static"]]
        match_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []

        for q_a in active_queues:
            for q_b in active_queues:
                if q_a["id"] == q_b["id"]:
                    continue
                hist_a = list(q_a["history"])
                hist_b = list(q_b["history"])
                time_gap = hist_b[0][0] - hist_a[-1][0]
                if not (0 < time_gap <= self.max_time_gap):
                    continue
                v_out, speed_a = self._get_velocity_vector(hist_a, mode="out")
                v_in, speed_b = self._get_velocity_vector(hist_b, mode="in")
                if v_out is None or v_in is None or speed_a is None or speed_b is None:
                    continue
                if speed_a < self.min_speed or speed_b < self.min_speed:
                    continue
                vec_displacement = np.array(hist_b[0][1][0:2]) - np.array(hist_a[-1][1][0:2])
                angle_a_to_d = self._angle_between(v_out, vec_displacement)
                angle_b_to_d = self._angle_between(v_in, vec_displacement)
                angle_a_to_b = self._angle_between(v_out, v_in)
                if (
                    angle_a_to_d <= self.max_angle_deg
                    and angle_b_to_d <= self.max_angle_deg
                    and angle_a_to_b <= self.max_angle_deg
                ):
                    match_candidates.append((angle_a_to_d + angle_b_to_d + angle_a_to_b, q_a, q_b))

        match_candidates.sort(key=lambda item: item[0])
        for _cost, q_a, q_b in match_candidates:
            if q_a["id"] in skip_ids or q_b["id"] in skip_ids:
                continue
            hist_a = list(q_a["history"])
            hist_b = list(q_b["history"])
            frame_a_end, det_a = hist_a[-1][:2]
            frame_b_start, det_b = hist_b[0][:2]
            gap = frame_b_start - frame_a_end
            interpolated_pts: list[tuple[int, list[float], bool]] = []
            if gap > 1:
                for i in range(1, gap):
                    alpha = i / gap
                    interp_frame = frame_a_end + i
                    interp_det = [det_a[j] + alpha * (det_b[j] - det_a[j]) for j in range(6)]
                    interpolated_pts.append((interp_frame, interp_det, True))
            q_a["history"] = deque(hist_a + interpolated_pts + hist_b, maxlen=q_a["history"].maxlen)
            skip_ids.add(q_a["id"])
            skip_ids.add(q_b["id"])

        consumed_sources = {c[2]["id"] for c in match_candidates if c[2]["id"] in skip_ids}
        for q in queues:
            if q["id"] not in skip_ids or q["id"] not in consumed_sources:
                stitched_queues.append(q)
        return stitched_queues


class _VerifyTrajectoryAnalyzer:
    def __init__(
        self,
        *,
        window: int,
        angle_thresh: float,
        momentum_thresh: float,
        tolerance: int,
    ):
        self.window = window
        self.angle_thresh = angle_thresh
        self.momentum_thresh = momentum_thresh
        self.tolerance = tolerance

    def smooth(self, pts: list[tuple[int, list[float], bool]]) -> list[tuple[int, list[float], bool]]:
        smoothed: list[tuple[int, list[float], bool]] = []
        half_w = 1
        for i, point in enumerate(pts):
            start = max(0, i - half_w)
            end = min(len(pts), i + half_w + 1)
            subset = pts[start:end]
            avg_cx = sum(p[1][0] for p in subset) / len(subset)
            avg_cy = sum(p[1][1] for p in subset) / len(subset)
            new_det = list(point[1])
            new_det[0] = avg_cx
            new_det[1] = avg_cy
            smoothed.append((point[0], new_det, point[2]))
        return smoothed

    def detect_bounces(
        self,
        pts: list[tuple[int, list[float], bool]],
    ) -> tuple[list[tuple[int, float, float, float]], dict[int, dict[str, Any]]]:
        if len(pts) < self.window * 2 + 1:
            return [], {}
        lookup = {p[0]: (p[1][0], p[1][1]) for p in pts}
        valid_frames = [p[0] for p in pts]
        candidate_bounces: set[int] = set()
        frame_stats: dict[int, dict[str, Any]] = {}

        for i, curr_idx in enumerate(valid_frames):
            if i < self.window or i >= len(valid_frames) - self.window:
                continue
            prev_idx = valid_frames[i - self.window]
            next_idx = valid_frames[i + self.window]
            if (curr_idx - prev_idx) > self.window * 3 or (next_idx - curr_idx) > self.window * 3:
                continue

            p_prev = lookup[prev_idx]
            p_curr = lookup[curr_idx]
            p_next = lookup[next_idx]
            v_in = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
            v_out = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])
            norm_in = float(np.hypot(v_in[0], v_in[1]))
            norm_out = float(np.hypot(v_out[0], v_out[1]))
            angle = 0.0
            delta_v = 0.0
            y_reversal = False
            if norm_in > 1e-5 and norm_out > 1e-5:
                cos_theta = np.clip(float(np.dot(v_in, v_out)) / (norm_in * norm_out), -1.0, 1.0)
                angle = float(np.degrees(np.arccos(cos_theta)))
                y_reversal = bool(v_in[1] > 0 and v_out[1] < 0)
                speed_in = norm_in / (curr_idx - prev_idx)
                speed_out = norm_out / (next_idx - curr_idx)
                delta_v = abs(speed_in - speed_out)
            frame_stats[curr_idx] = {
                "angle": angle,
                "angle_ok": angle >= self.angle_thresh,
                "y_ok": y_reversal,
                "mom_ok": delta_v >= self.momentum_thresh,
                "delta_v": delta_v,
                "y_reversal": y_reversal,
            }

        for curr_idx, stats in frame_stats.items():
            if not stats["angle_ok"]:
                continue
            local_y_ok = False
            local_mom_ok = False
            for j in range(curr_idx - self.tolerance, curr_idx + self.tolerance + 1):
                if j in frame_stats:
                    local_y_ok = local_y_ok or bool(frame_stats[j]["y_ok"])
                    local_mom_ok = local_mom_ok or bool(frame_stats[j]["mom_ok"])
            if local_y_ok or local_mom_ok:
                candidate_bounces.add(curr_idx)

        bounces_raw: list[tuple[int, float, float, float]] = []
        sorted_bounces = sorted(candidate_bounces)
        if not sorted_bounces:
            return bounces_raw, frame_stats
        cluster = [sorted_bounces[0]]
        for idx in sorted_bounces[1:]:
            if idx - cluster[-1] <= self.window * 2 + self.tolerance:
                cluster.append(idx)
            else:
                best_idx = max(cluster, key=lambda frame: lookup[frame][1])
                bounces_raw.append((best_idx, lookup[best_idx][0], lookup[best_idx][1], frame_stats[best_idx]["angle"]))
                cluster = [idx]
        if cluster:
            best_idx = max(cluster, key=lambda frame: lookup[frame][1])
            bounces_raw.append((best_idx, lookup[best_idx][0], lookup[best_idx][1], frame_stats[best_idx]["angle"]))
        return bounces_raw, frame_stats

    def detect_net_crossings(
        self,
        pts: list[tuple[int, list[float], bool]],
        *,
        net_line: np.ndarray,
        speed_line: np.ndarray,
        search_frames: int,
    ) -> list[dict[str, Any]]:
        crossings: list[dict[str, Any]] = []
        if len(pts) < 2:
            return crossings

        def line_value(item: tuple[int, list[float], bool], line: np.ndarray) -> float:
            return float(line[0] * item[1][0] + line[1] * item[1][1] + line[2])

        def check_crossing(p1: tuple[int, list[float], bool], p2: tuple[int, list[float], bool], line: np.ndarray) -> bool:
            val1 = line_value(p1, line)
            val2 = line_value(p2, line)
            return val1 * val2 <= 0.0 and val1 != val2

        for i in range(1, len(pts)):
            p_prev = pts[i - 1]
            p_curr = pts[i]
            if not check_crossing(p_prev, p_curr, net_line):
                continue
            direction = "bottom_up" if p_prev[1][1] > p_curr[1][1] else "top_down"
            speed_cross_idx = -1
            for j in range(i, max(0, i - search_frames), -1):
                if j - 1 >= 0 and check_crossing(pts[j - 1], pts[j], speed_line):
                    speed_cross_idx = j
                    break
            if speed_cross_idx == -1:
                for j in range(i, min(len(pts) - 1, i + search_frames)):
                    if check_crossing(pts[j], pts[j + 1], speed_line):
                        speed_cross_idx = j
                        break
            if speed_cross_idx == -1:
                continue
            p_speed = pts[speed_cross_idx]
            frame_diff = abs(p_curr[0] - p_speed[0])
            if frame_diff <= 0:
                continue
            pixel_dist = float(np.hypot(p_curr[1][0] - p_speed[1][0], p_curr[1][1] - p_speed[1][1]))
            crossings.append({
                "frame": int(p_curr[0]),
                "frame_index": int(p_curr[0]),
                "direction": direction,
                "speed_px": pixel_dist / frame_diff,
                "pixel_x": float(p_curr[1][0]),
                "pixel_y": float(p_curr[1][1]),
                "speed_ref_frame": int(p_speed[0]),
            })
        return crossings


class _VerifyQueueTracker:
    def __init__(self):
        self.base_search_window = VERIFY_TRK_SEARCH_WINDOW
        self.base_create_thr = VERIFY_TRK_CREATE_THR
        self.eval_window = VERIFY_TRK_EVAL_WINDOW
        self.static_thr = VERIFY_TRK_STATIC_THR
        self.static_lock_radius = VERIFY_TRK_STATIC_RADIUS
        self.zone_persistence = VERIFY_TRK_ZONE_PERSIST
        self.max_search_window = 10
        self.max_create_thr = 100.0
        self.queues: list[dict[str, Any]] = []
        self.queue_id_counter = 0
        self.static_zones: dict[int, dict[str, Any]] = {}

    def process_frame(self, frame_idx: int, detections: list[list[float]]) -> None:
        matched_detects: set[int] = set()
        self.static_zones = {
            qid: zone
            for qid, zone in self.static_zones.items()
            if frame_idx - int(zone["last_seen"]) <= self.zone_persistence
        }

        for q in self.queues:
            age = frame_idx - int(q["history"][0][0])
            if q["is_static"] and age >= self.eval_window:
                self.static_zones[q["id"]] = {
                    "pos": np.array(q["history"][-1][1][0:2]),
                    "last_seen": frame_idx,
                    "q_ref": q,
                }

        for d_idx, det in enumerate(detections):
            det_pos = np.array(det[0:2])
            for zone in self.static_zones.values():
                if np.hypot(det_pos[0] - zone["pos"][0], det_pos[1] - zone["pos"][1]) <= self.static_lock_radius:
                    zone["last_seen"] = frame_idx
                    zone["pos"] = det_pos
                    zone["q_ref"]["history"].append((frame_idx, det, False))
                    matched_detects.add(d_idx)
                    break

        matched_queues = {zone["q_ref"]["id"] for zone in self.static_zones.values() if zone["last_seen"] == frame_idx}
        match_candidates: list[tuple[float, float, int, int]] = []
        for d_idx, det in enumerate(detections):
            if d_idx in matched_detects:
                continue
            for q_idx, q in enumerate(self.queues):
                if q["id"] in matched_queues:
                    continue
                speed = float(q.get("speed", 0.0))
                dynamic_window = min(self.max_search_window, self.base_search_window + int(speed))
                dynamic_thr = min(self.max_create_thr, self.base_create_thr + (speed * dynamic_window * 0.8))
                min_dist = float("inf")
                for past_frame, past_det, _is_stitched in reversed(q["history"]):
                    if frame_idx - past_frame <= dynamic_window:
                        dist = float(np.hypot(det[0] - past_det[0], det[1] - past_det[1]))
                        min_dist = min(min_dist, dist)
                    else:
                        break
                if min_dist != float("inf"):
                    match_candidates.append((min_dist, dynamic_thr, q_idx, d_idx))

        match_candidates.sort(key=lambda item: item[0])
        queue_allocated = set(matched_queues)
        for dist, dynamic_thr, q_idx, d_idx in match_candidates:
            if self.queues[q_idx]["id"] in queue_allocated or d_idx in matched_detects:
                continue
            if dist <= dynamic_thr:
                self.queues[q_idx]["history"].append((frame_idx, detections[d_idx], False))
                queue_allocated.add(self.queues[q_idx]["id"])
                matched_detects.add(d_idx)

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detects:
                self.queues.append({
                    "id": self.queue_id_counter,
                    "history": deque([(frame_idx, det, False)], maxlen=VERIFY_TRK_MAX_HISTORY),
                    "is_static": True,
                    "speed": 0.0,
                })
                self.queue_id_counter += 1

        for q in self.queues:
            history_list = list(q["history"])
            eval_items = [item for item in history_list if frame_idx - item[0] <= self.eval_window]
            if len(eval_items) >= 2:
                oldest_det = eval_items[0][1]
                newest_det = eval_items[-1][1]
                displacement = float(np.hypot(newest_det[0] - oldest_det[0], newest_det[1] - oldest_det[1]))
                frame_span = eval_items[-1][0] - eval_items[0][0]
                if frame_span > 0:
                    avg_dist = displacement / frame_span
                    q["speed"] = avg_dist
                    q["is_static"] = avg_dist < self.static_thr

        surviving = []
        for q in self.queues:
            time_since_last_seen = frame_idx - q["history"][-1][0]
            speed = float(q.get("speed", 0.0))
            dynamic_window = min(self.max_search_window, self.base_search_window + int(speed))
            max_allowed_gap = self.zone_persistence if q["is_static"] else dynamic_window
            if time_since_last_seen <= max_allowed_gap:
                surviving.append(q)
        self.queues = surviving

    def get_events(
        self,
        frame_idx: int,
        analyzer: _VerifyTrajectoryAnalyzer,
        *,
        net_line: np.ndarray,
        speed_line: np.ndarray,
        speed_search_frames: int,
    ) -> tuple[list[list[float]], list[list[float]], list[tuple[int, float, float, float, dict[str, Any]]], list[dict[str, Any]]]:
        moving_dets: list[list[float]] = []
        static_dets: list[list[float]] = []
        bounces: list[tuple[int, float, float, float, dict[str, Any]]] = []
        crossings: list[dict[str, Any]] = []
        for q in self.queues:
            if not q["is_static"] and len(q["history"]) > 1:
                pts = [item for item in q["history"] if frame_idx - item[0] <= VERIFY_TRK_MAX_HISTORY]
                smoothed_pts = analyzer.smooth(pts)
                raw_bounces, frame_stats = analyzer.detect_bounces(smoothed_pts)
                for b_frame, bx, by, angle in raw_bounces:
                    stats = dict(frame_stats.get(b_frame, {}))
                    stats["queue_id"] = int(q["id"])
                    stats["queue_history_len"] = len(q["history"])
                    stats["queue_speed_px"] = round(float(q.get("speed", 0.0) or 0.0), 4)
                    history_items = list(q["history"])
                    track_ids = [
                        _candidate_det_track_id(item[1])
                        for item in history_items
                        if _candidate_det_track_id(item[1]) is not None
                    ]
                    if track_ids:
                        stats["queue_track_id"] = track_ids[-1]
                        stats["queue_track_id_unique"] = len(set(track_ids))
                    confidences = [
                        _candidate_det_conf(item[1])
                        for item in history_items
                    ]
                    if confidences:
                        stats["queue_conf_last"] = round(float(confidences[-1]), 4)
                        stats["queue_conf_max"] = round(float(max(confidences)), 4)
                        stats["queue_conf_avg"] = round(float(np.mean(confidences)), 4)
                    ranks = [
                        rank
                        for rank in (_candidate_det_rank(item[1]) for item in history_items)
                        if rank is not None
                    ]
                    if ranks:
                        stats["queue_candidate_rank_last"] = int(ranks[-1])
                        stats["queue_candidate_rank_min"] = int(min(ranks))
                        stats["queue_candidate_rank_max"] = int(max(ranks))
                        stats["queue_candidate_rank_avg"] = round(float(np.mean(ranks)), 2)
                    if history_items:
                        nearest_item = min(
                            history_items,
                            key=lambda item: abs(int(item[0]) - int(b_frame)),
                        )
                        stats["queue_event_frame_gap"] = int(abs(int(nearest_item[0]) - int(b_frame)))
                        stats["queue_conf_at_event"] = round(
                            float(_candidate_det_conf(nearest_item[1])),
                            4,
                        )
                        event_rank = _candidate_det_rank(nearest_item[1])
                        if event_rank is not None:
                            stats["queue_candidate_rank_event"] = int(event_rank)
                    static_blocked_seen = sum(
                        1 for item in history_items if _candidate_det_static_blocked(item[1])
                    )
                    if static_blocked_seen:
                        stats["queue_static_blocked_history"] = int(static_blocked_seen)
                    bounces.append((b_frame, bx, by, angle, stats))
                crossings.extend(
                    analyzer.detect_net_crossings(
                        smoothed_pts,
                        net_line=net_line,
                        speed_line=speed_line,
                        search_frames=speed_search_frames,
                    )
                )
            if q["history"][-1][0] == frame_idx:
                det = q["history"][-1][1]
                if q["is_static"]:
                    static_dets.append(det)
                else:
                    moving_dets.append(det)
        return moving_dets, static_dets, bounces, crossings


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

            hit_anchor_px = det.get("hit_anchor_px")
            if not hit_anchor_px and len(bbox) >= 4:
                hit_anchor_px = [(x1 + x2) / 2.0, y1 + (y2 - y1) * 0.3]

            projected.append({
                "bbox": [x1, y1, x2, y2],
                "conf": float(det.get("conf", 0.0) or 0.0),
                "foot_px": [float(foot_px[0]), float(foot_px[1])],
                "court_x": float(court_x),
                "court_y": float(court_y),
                # Racket/contact proxy from the prototype: upper-body center.
                "hit_anchor_px": [float(hit_anchor_px[0]), float(hit_anchor_px[1])]
                if hit_anchor_px and len(hit_anchor_px) >= 2
                else [(x1 + x2) / 2.0, y1 + (y2 - y1) * 0.3],
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


def _bottom_player_distance_threshold_px(
    player: dict[str, Any],
    *,
    hit_dist_px_net: float,
    hit_dist_px_base: float,
) -> float:
    ry_ratio = min(max(float(player.get("court_y", 0.0) or 0.0), 0.0), COURT_Y_MAX) / max(COURT_Y_MAX, 1e-6)
    return hit_dist_px_net + (hit_dist_px_base - hit_dist_px_net) * ry_ratio


def _player_in_half(player: dict[str, Any], half: str, *, roi_net_margin_m: float) -> bool:
    x = float(player.get("court_x", 0.0) or 0.0)
    y = float(player.get("court_y", 0.0) or 0.0)
    if x < DOUBLES_X_MIN - 0.5 or x > DOUBLES_X_MAX + 0.5:
        return False
    if half == "top":
        return y <= -roi_net_margin_m
    if half == "bottom":
        return y >= roi_net_margin_m
    return False


def _event_float(event: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(event.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _bounce_signal_score(event: dict[str, Any]) -> float:
    """Rank duplicate bounce candidates by trajectory evidence strength."""
    angle = max(0.0, _event_float(event, "angle"))
    delta_v = max(0.0, _event_float(event, "delta_v"))
    confidence = max(0.0, _event_float(event, "confidence"))
    speed_px = max(0.0, _event_float(event, "queue_speed_px"))
    speed_bonus = min(speed_px, DEFAULT_BOUNCE_SPEED_BONUS_CAP_PX) * DEFAULT_BOUNCE_SPEED_BONUS_WEIGHT
    y_bonus = 25.0 if event.get("y_reversal") else 0.0
    return round(angle + delta_v * 2.0 + confidence * 10.0 + y_bonus + speed_bonus, 4)


def _bounce_shape_score(event: dict[str, Any]) -> float:
    """Score trajectory shape without speed bonus for weak-candidate gating."""
    angle = max(0.0, _event_float(event, "angle"))
    delta_v = max(0.0, _event_float(event, "delta_v"))
    confidence = max(0.0, _event_float(event, "confidence"))
    y_bonus = 25.0 if event.get("y_reversal") else 0.0
    return round(angle + delta_v * 2.0 + confidence * 10.0 + y_bonus, 4)


def _out_distance_from_court_m(event: dict[str, Any]) -> float:
    x = _event_float(event, "x")
    y = _event_float(event, "y")
    dx = max(SINGLES_X_MIN - x, 0.0, x - SINGLES_X_MAX)
    dy = max(COURT_Y_MIN - y, 0.0, y - COURT_Y_MAX)
    return float(np.hypot(dx, dy))


def _is_gate_only_weak_out_bounce(event: dict[str, Any]) -> bool:
    """Use weak near-line OUTs to close a rally without publishing them."""
    if bool(event.get("in_court", True)):
        return False
    out_distance_m = _out_distance_from_court_m(event)
    if out_distance_m > DEFAULT_GATE_ONLY_OUT_MARGIN_M:
        return False
    if "delta_v" not in event and "confidence" not in event:
        score = _event_float(
            event,
            "bounce_signal_score",
            _bounce_signal_score(event),
        )
    else:
        score = _bounce_shape_score(event)
    delta_v = _event_float(event, "delta_v")
    return (
        score < DEFAULT_GATE_ONLY_OUT_MIN_SCORE
        and delta_v < DEFAULT_GATE_ONLY_OUT_MIN_DELTA_V
    )


def _split_gate_only_bounces(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    publishable: list[dict[str, Any]] = []
    gate_only: list[dict[str, Any]] = []
    for event in events:
        if _is_gate_only_weak_out_bounce(event):
            gate_event = {
                **event,
                "publishable": False,
                "gate_only": True,
                "gate_only_reason": "near_boundary_weak_out",
                "out_distance_m": round(_out_distance_from_court_m(event), 4),
            }
            gate_only.append(gate_event)
        else:
            publishable.append(event)
    return publishable, gate_only


def _same_bounce_window(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    clean_time_frames: int,
    clean_space_meters: float,
) -> bool:
    af = int(a.get("frame_index", a.get("frame", -10**9)))
    bf = int(b.get("frame_index", b.get("frame", -10**9)))
    if abs(af - bf) > clean_time_frames:
        return False
    return (
        np.hypot(
            _event_float(a, "x") - _event_float(b, "x"),
            _event_float(a, "y") - _event_float(b, "y"),
        )
        <= clean_space_meters
    )


def _select_strongest_bounces(
    events: list[dict[str, Any]],
    *,
    clean_time_frames: int,
    clean_space_meters: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster duplicate bounce candidates and keep the strongest per cluster."""
    clusters: list[list[dict[str, Any]]] = []
    for event in sorted(events, key=lambda e: int(e.get("frame_index", e.get("frame", 0)))):
        enriched = {
            **event,
            "bounce_signal_score": _event_float(
                event,
                "bounce_signal_score",
                _bounce_signal_score(event),
            ),
        }
        target = None
        for cluster in clusters:
            if any(
                _same_bounce_window(
                    enriched,
                    member,
                    clean_time_frames=clean_time_frames,
                    clean_space_meters=clean_space_meters,
                )
                for member in cluster
            ):
                target = cluster
                break
        if target is None:
            clusters.append([enriched])
        else:
            target.append(enriched)

    selected: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for cluster in clusters:
        best = max(
            cluster,
            key=lambda e: (
                _event_float(e, "bounce_signal_score"),
                _event_float(e, "delta_v"),
                _event_float(e, "angle"),
                _event_float(e, "confidence"),
                -int(e.get("frame_index", e.get("frame", 0))),
            ),
        )
        selected.append({**best, "dedupe_cluster_size": len(cluster)})
        for event in cluster:
            if event is best:
                continue
            dropped.append({
                **event,
                "deduped_by_frame": best.get("frame_index", best.get("frame")),
                "dedupe_reason": "weaker_bounce_signal_same_window",
            })

    return (
        sorted(selected, key=lambda e: int(e.get("frame_index", e.get("frame", 0)))),
        sorted(dropped, key=lambda e: int(e.get("frame_index", e.get("frame", 0)))),
    )


def _event_frame_value(event: dict[str, Any]) -> int | None:
    return _frame_id(event.get("frame_index", event.get("frame")))


def _is_bottom_hit_restart(event: dict[str, Any]) -> bool:
    source = str(event.get("source", ""))
    return source.startswith("bottom_reversal") or source.startswith("bottom_up")


def _out_gate_has_restart(
    *,
    out_frame: int,
    candidate_frame: int,
    hit_events: list[dict[str, Any]],
    speed_events: list[dict[str, Any]],
    restart_hit_gap_frames: int,
    restart_speed_kmh: float,
) -> bool:
    for event in speed_events:
        frame = _event_frame_value(event)
        if frame is None or not (out_frame < frame <= candidate_frame):
            continue
        if _event_float(event, "speed_kmh") >= restart_speed_kmh:
            return True
    for event in hit_events:
        frame = _event_frame_value(event)
        if frame is None or not (out_frame < frame <= candidate_frame):
            continue
        if frame - out_frame < restart_hit_gap_frames:
            continue
        if _is_bottom_hit_restart(event):
            return True
    return False


def _apply_out_rally_gate(
    events: list[dict[str, Any]],
    *,
    hit_events: list[dict[str, Any]],
    speed_events: list[dict[str, Any]],
    restart_hit_gap_frames: int = DEFAULT_OUT_RESTART_HIT_GAP_FRAMES,
    restart_speed_kmh: float = DEFAULT_OUT_RESTART_SPEED_KMH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """After an OUT bounce, ignore later bounces until a new-play cue appears.

    The YOLO detector can correctly see non-match balls after a rally-ending
    OUT. A bottom-half player HIT or a real net-crossing speed event is the
    lightweight signal that the next play has begun.
    """
    passthrough: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    blocked_after_out_frame: int | None = None

    for event in sorted(events, key=lambda e: int(e.get("frame_index", e.get("frame", 0)))):
        frame = _event_frame_value(event)
        if frame is None:
            kept.append(event)
            continue

        if blocked_after_out_frame is not None:
            if _out_gate_has_restart(
                out_frame=blocked_after_out_frame,
                candidate_frame=frame,
                hit_events=hit_events,
                speed_events=speed_events,
                restart_hit_gap_frames=restart_hit_gap_frames,
                restart_speed_kmh=restart_speed_kmh,
            ):
                blocked_after_out_frame = None

        if blocked_after_out_frame is not None and frame > blocked_after_out_frame:
            suppressed.append({
                **event,
                "suppressed_by_out_frame": blocked_after_out_frame,
                "suppression_reason": "out_rally_gate",
            })
            continue

        publishable = bool(event.get("publishable", True)) and not bool(event.get("gate_only"))
        if publishable:
            kept.append(event)
        if not bool(event.get("in_court", True)):
            blocked_after_out_frame = frame

    return kept, suppressed


def is_yolo_queue_event(event: dict[str, Any]) -> bool:
    source = str(event.get("source", "") or "")
    return (
        source.startswith("yolo_")
        or event.get("queue_id") is not None
        or event.get("bounce_signal_score") is not None
    )


def dashboard_yolo_quality_reject_reason(
    event: dict[str, Any],
    *,
    event_frame: int | None = None,
    min_bounce_frame: int = DEFAULT_DASHBOARD_MIN_BOUNCE_FRAME,
    min_history: int = DEFAULT_DASHBOARD_MIN_BOUNCE_HISTORY,
    min_confidence: float = DEFAULT_DASHBOARD_MIN_CONFIDENCE,
    weak_non_reversal_max_angle: float = DEFAULT_DASHBOARD_WEAK_NON_REVERSAL_MAX_ANGLE,
    weak_non_reversal_min_score: float = DEFAULT_DASHBOARD_WEAK_NON_REVERSAL_MIN_SCORE,
) -> str | None:
    """Return the dashboard publish-layer quality reject reason, if any."""
    if not is_yolo_queue_event(event):
        return None

    frame = event_frame if event_frame is not None else _event_frame_value(event)
    if frame is not None and frame < int(min_bounce_frame):
        return "quality_warmup"

    history_len_raw = event.get("queue_history_len")
    if history_len_raw is not None:
        history_len = _event_float(event, "queue_history_len")
        if history_len < float(min_history):
            return "quality_short_track"

    confidence_raw = event.get("confidence")
    if confidence_raw is not None and _event_float(event, "confidence") < float(min_confidence):
        return "quality_low_confidence"

    angle_raw = event.get("angle")
    score_raw = event.get("bounce_signal_score")
    if angle_raw is None or score_raw is None:
        return None
    angle = _event_float(event, "angle")
    if "delta_v" not in event and "confidence" not in event:
        score = _event_float(event, "bounce_signal_score")
    else:
        score = _bounce_shape_score(event)
    y_reversal = bool(event.get("y_reversal"))
    if (
        not y_reversal
        and angle < float(weak_non_reversal_max_angle)
        and score < float(weak_non_reversal_min_score)
    ):
        return "quality_weak_non_reversal"
    return None


def filter_dashboard_yolo_publishable_bounces(
    bounces: list[dict[str, Any]],
    *,
    hit_events: list[dict[str, Any]] | None = None,
    latest_frame: int | None = None,
    hit_suppress_frames: int = DEFAULT_HIT_SUPPRESS_FRAMES,
    clean_time_frames: int = DEFAULT_CLEAN_TIME_FRAMES,
    clean_space_meters: float = DEFAULT_CLEAN_SPACE_METERS,
    duplicate_space_meters: float = DEFAULT_DASHBOARD_LIVE_DUPLICATE_SPACE_METERS,
    release_delay_frames: int = DEFAULT_DASHBOARD_RELEASE_DELAY_FRAMES,
) -> dict[str, Any]:
    """Apply the dashboard publish gate to already-final YOLO bounce candidates.

    ``detect_single_camera_events()`` returns the verified offline event-rule
    output. The live dashboard applies a second publish layer before minimap,
    reports and 3D push. This helper makes that layer reusable by offline
    replay/video tools, while preserving raw candidates for diagnostics.
    """
    hit_events = hit_events or []
    passthrough: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    hit_suppressed_candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    def _suppress(event: dict[str, Any], reason: str, **extra: Any) -> None:
        suppressed_event = {
            **event,
            "publish_suppression_reason": reason,
            **extra,
        }
        suppressed.append(suppressed_event)
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

    live_duplicate_space = max(float(clean_space_meters), float(duplicate_space_meters))
    for event in sorted(bounces or [], key=lambda e: int(e.get("frame_index", e.get("frame", 0)))):
        event_frame = _event_frame_value(event)
        if event_frame is None:
            passthrough.append(dict(event))
            continue

        quality_reason = dashboard_yolo_quality_reject_reason(
            event,
            event_frame=event_frame,
        )
        if quality_reason:
            _suppress(event, quality_reason)
            continue

        suppressing_hit_frame = None
        if hit_suppress_frames > 0:
            for hit in hit_events:
                hit_frame = _event_frame_value(hit)
                if hit_frame is not None and abs(event_frame - hit_frame) <= hit_suppress_frames:
                    suppressing_hit_frame = hit_frame
                    break
        if suppressing_hit_frame is not None:
            hit_suppressed = {
                **event,
                "suppressed_by_hit_frame": suppressing_hit_frame,
            }
            hit_suppressed_candidates.append(hit_suppressed)
            _suppress(hit_suppressed, "hit_window", suppressed_by_hit_frame=suppressing_hit_frame)
            continue

        hit_shadow: dict[str, Any] | None = None
        if hit_suppress_frames > 0 and live_duplicate_space > 0:
            for prev in hit_suppressed_candidates:
                if _same_bounce_window(
                    event,
                    prev,
                    clean_time_frames=hit_suppress_frames,
                    clean_space_meters=live_duplicate_space,
                ):
                    hit_shadow = prev
                    break
        if hit_shadow is not None:
            _suppress(
                event,
                "hit_window",
                suppressed_by_hit_frame=hit_shadow.get("suppressed_by_hit_frame"),
                hit_window_shadow_frame=_event_frame_value(hit_shadow),
            )
            continue

        if (
            latest_frame is not None
            and release_delay_frames > 0
            and int(latest_frame) - event_frame + 1 < int(release_delay_frames)
        ):
            _suppress(
                event,
                "release_delay",
                latest_frame=int(latest_frame),
                release_delay_frames=int(release_delay_frames),
            )
            continue

        candidates.append(dict(event))

    if clean_time_frames > 0 and live_duplicate_space > 0:
        kept, duplicate_suppressed = _select_strongest_bounces(
            candidates,
            clean_time_frames=clean_time_frames,
            clean_space_meters=live_duplicate_space,
        )
        for event in duplicate_suppressed:
            _suppress(
                event,
                "duplicate_live_bounce",
                deduped_by_frame=event.get("deduped_by_frame"),
                duplicate_space_meters=round(live_duplicate_space, 4),
                dedupe_reason=event.get("dedupe_reason"),
            )
    else:
        kept = [dict(event) for event in candidates]
    kept, gate_only_bounces = _split_gate_only_bounces(kept)
    if passthrough:
        kept = passthrough + kept

    for seq, event in enumerate(kept, start=1):
        event["sequence"] = seq

    return {
        "bounces": kept,
        "suppressed_bounces": suppressed,
        "gate_only_bounces": gate_only_bounces,
        "count": len(kept),
        "suppressed_count": len(suppressed),
        "gate_only_count": len(gate_only_bounces),
        "suppression_counts": reason_counts,
        "params": {
            "hit_suppress_frames": hit_suppress_frames,
            "clean_time_frames": clean_time_frames,
            "clean_space_meters": clean_space_meters,
            "duplicate_space_meters": duplicate_space_meters,
            "effective_duplicate_space_meters": live_duplicate_space,
            "release_delay_frames": release_delay_frames,
            "min_confidence": DEFAULT_DASHBOARD_MIN_CONFIDENCE,
        },
    }


def _make_hit_event(
    *,
    base_event: dict[str, Any],
    player: dict[str, Any],
    source: str,
    distance_px: float,
    threshold_px: float,
    angle: float | None = None,
    crossing_frame: int | None = None,
) -> dict[str, Any]:
    """Build HIT using the offline visual rule: ball_x + player foot_y."""
    event = {
        **base_event,
        "type": "HIT",
        "kind": "hit",
        "source": source,
        "x": round(float(base_event["x"]), 4),
        "y": round(float(player["court_y"]), 4),
        "ball_x": round(float(base_event["x"]), 4),
        "ball_y": round(float(base_event["y"]), 4),
        "player_court_x": round(float(player["court_x"]), 4),
        "player_court_y": round(float(player["court_y"]), 4),
        "player_distance_px": round(float(distance_px), 2),
        "player_threshold_px": round(float(threshold_px), 2),
        "player_conf": round(float(player.get("conf", 0.0) or 0.0), 4),
    }
    if angle is not None:
        event["angle"] = round(float(angle), 2)
    if crossing_frame is not None:
        event["crossing_frame"] = int(crossing_frame)
    return event


def _dedupe_hit(
    hits: dict[int, dict[str, Any]],
    event: dict[str, Any],
    *,
    clean_time_frames: int,
    clean_space_meters: float,
) -> bool:
    frame = int(event["frame_index"])
    duplicate_frames: list[int] = []
    hit_clean_space_meters = max(float(clean_space_meters), DEFAULT_HIT_CLEAN_SPACE_METERS)
    for prev_frame_key, prev in hits.items():
        prev_frame = int(prev.get("frame_index", -10**9))
        if abs(frame - prev_frame) > clean_time_frames:
            continue
        if np.hypot(float(event["x"]) - float(prev["x"]), float(event["y"]) - float(prev["y"])) <= hit_clean_space_meters:
            duplicate_frames.append(prev_frame_key)
    for prev_frame_key in duplicate_frames:
        hits.pop(prev_frame_key, None)
    hits[frame] = event
    return True


def _find_lookback_hit(
    *,
    crossing: dict[str, Any],
    half: str,
    source: str,
    point_by_frame: dict[int, _Point],
    players_by_frame: dict[int, list[dict[str, Any]]],
    lookback_frames: int,
    roi_net_margin_m: float,
    top_hit_dist_px: float,
    hit_dist_px_net: float,
    hit_dist_px_base: float,
) -> dict[str, Any] | None:
    crossing_frame = _frame_id(crossing.get("frame_index", crossing.get("frame")))
    if crossing_frame is None:
        return None

    best: tuple[float, float, dict[str, Any], _Point] | None = None
    start = max(0, crossing_frame - lookback_frames)
    for frame in range(crossing_frame, start - 1, -1):
        point = point_by_frame.get(frame)
        if point is None:
            continue
        for player in players_by_frame.get(frame, []):
            if not _player_in_half(player, half, roi_net_margin_m=roi_net_margin_m):
                continue
            anchor_x, anchor_y = player["hit_anchor_px"]
            dist_px = float(np.hypot(point.pixel_x - anchor_x, point.pixel_y - anchor_y))
            threshold = (
                top_hit_dist_px
                if half == "top"
                else _bottom_player_distance_threshold_px(
                    player,
                    hit_dist_px_net=hit_dist_px_net,
                    hit_dist_px_base=hit_dist_px_base,
                )
            )
            score = dist_px / max(threshold, 1.0)
            if best is None or score < best[0]:
                best = (score, dist_px, player, point)

    if best is None or best[0] > 1.0:
        return None
    _score, dist_px, player, point = best
    base_event = {
        "sequence": 0,
        "frame": point.frame_index,
        "frame_index": point.frame_index,
        "camera": crossing.get("camera", ""),
        "x": round(point.world_x, 4),
        "y": round(point.world_y, 4),
        "pixel_x": round(point.pixel_x, 2),
        "pixel_y": round(point.pixel_y, 2),
        "in_court": _is_in_court(point.world_x, point.world_y),
        "confidence": round(float(point.confidence), 4),
    }
    threshold_px = (
        top_hit_dist_px
        if half == "top"
        else _bottom_player_distance_threshold_px(
            player,
            hit_dist_px_net=hit_dist_px_net,
            hit_dist_px_base=hit_dist_px_base,
        )
    )
    return _make_hit_event(
        base_event=base_event,
        player=player,
        source=source,
        distance_px=dist_px,
        threshold_px=threshold_px,
        crossing_frame=crossing_frame,
    )


def _line_from_points(
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> np.ndarray | None:
    line = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-9:
        return None
    return line / norm


def _line_value(line: np.ndarray, point: _Point) -> float:
    return float(line[0] * point.pixel_x + line[1] * point.pixel_y + line[2])


def _crosses_pixel_line(prev: _Point, curr: _Point, line: np.ndarray) -> bool:
    val1 = _line_value(line, prev)
    val2 = _line_value(line, curr)
    return val1 * val2 <= 0.0 and val1 != val2


def _speed_trap_lines_from_homography(
    homography: Any | None,
    *,
    net_offset_px: float,
    speed_line_offset_px: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if homography is None or not hasattr(homography, "world_to_pixel"):
        return None
    try:
        p1_raw = homography.world_to_pixel(-5.485, 0.0)
        p2_raw = homography.world_to_pixel(5.485, 0.0)
    except Exception:
        return None

    p1_net = (float(p1_raw[0]), float(p1_raw[1]) - float(net_offset_px))
    p2_net = (float(p2_raw[0]), float(p2_raw[1]) - float(net_offset_px))
    net_line = _line_from_points(p1_net, p2_net)
    if net_line is None:
        return None

    p1_speed = (p1_net[0], p1_net[1] + float(speed_line_offset_px))
    p2_speed = (p2_net[0], p2_net[1] + float(speed_line_offset_px))
    speed_line = _line_from_points(p1_speed, p2_speed)
    if speed_line is None:
        return None
    return net_line, speed_line


def _detect_single_cam_pixel_speed_trap_crossings(
    points: list[_Point],
    *,
    homography: Any | None,
    min_kmh: float,
    max_kmh: float,
    cooldown_frames: int,
    net_offset_px: float,
    speed_line_offset_px: float,
    speed_coef_up: float,
    speed_coef_down: float,
    speed_search_frames: int,
) -> list[dict[str, Any]] | None:
    if len(points) < 2:
        return []
    lines = _speed_trap_lines_from_homography(
        homography,
        net_offset_px=net_offset_px,
        speed_line_offset_px=speed_line_offset_px,
    )
    if lines is None:
        return None
    net_line, speed_line = lines

    events: list[dict[str, Any]] = []
    last_emit = -10**9
    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]
        if curr.frame_index - prev.frame_index <= 0:
            continue
        if not _crosses_pixel_line(prev, curr, net_line):
            continue
        if curr.frame_index - last_emit <= cooldown_frames:
            continue

        speed_idx = -1
        for j in range(i, max(0, i - speed_search_frames), -1):
            if j - 1 >= 0 and _crosses_pixel_line(points[j - 1], points[j], speed_line):
                speed_idx = j
                break
        if speed_idx == -1:
            for j in range(i, min(len(points) - 1, i + speed_search_frames)):
                if _crosses_pixel_line(points[j], points[j + 1], speed_line):
                    speed_idx = j
                    break
        if speed_idx == -1:
            continue

        speed_ref = points[speed_idx]
        frame_diff = abs(curr.frame_index - speed_ref.frame_index)
        if frame_diff <= 0:
            continue
        pixel_dist = float(np.hypot(curr.pixel_x - speed_ref.pixel_x, curr.pixel_y - speed_ref.pixel_y))
        direction = "bottom_up" if prev.pixel_y > curr.pixel_y else "top_down"
        coef = float(speed_coef_up) if direction == "bottom_up" else float(speed_coef_down)
        speed_px = pixel_dist / frame_diff
        speed_kmh = speed_px * coef
        if not (min_kmh <= speed_kmh <= max_kmh):
            continue

        denom = curr.world_y - prev.world_y
        t = 0.0 if abs(denom) < 1e-6 else float(np.clip((0.0 - prev.world_y) / denom, 0.0, 1.0))
        x_cross = (1.0 - t) * prev.world_x + t * curr.world_x
        px_cross = (1.0 - t) * prev.pixel_x + t * curr.pixel_x
        py_cross = (1.0 - t) * prev.pixel_y + t * curr.pixel_y
        legacy_direction = "near_to_far" if direction == "top_down" else "far_to_near"
        events.append({
            "frame": curr.frame_index,
            "frame_index": curr.frame_index,
            "x": round(float(x_cross), 4),
            "y": 0.0,
            "pixel_x": round(float(px_cross), 2),
            "pixel_y": round(float(py_cross), 2),
            "px": round(float(curr.pixel_x), 2),
            "py": round(float(curr.pixel_y), 2),
            "speed_px": round(float(speed_px), 4),
            "speed_kmh": round(float(speed_kmh), 1),
            "speed_ref_frame": speed_ref.frame_index,
            "direction": direction,
            "legacy_direction": legacy_direction,
            "source": "single_cam_pixel_speed_trap",
            "net_offset_px": float(net_offset_px),
            "speed_line_offset_px": float(speed_line_offset_px),
        })
        last_emit = curr.frame_index
    return events


def _detect_single_cam_speed_crossings(
    points: list[_Point],
    *,
    homography: Any | None = None,
    fps: float,
    min_kmh: float,
    max_kmh: float,
    cooldown_frames: int,
    net_offset_px: float = DEFAULT_NET_OFFSET_PX,
    speed_line_offset_px: float = DEFAULT_SPEED_LINE_OFFSET_PX,
    speed_coef_up: float = DEFAULT_SPEED_COEF_UP,
    speed_coef_down: float = DEFAULT_SPEED_COEF_DOWN,
    speed_search_frames: int = DEFAULT_SPEED_SEARCH_FRAMES,
) -> list[dict[str, Any]]:
    if len(points) < 2 or fps <= 0:
        return []

    pixel_trap_events = _detect_single_cam_pixel_speed_trap_crossings(
        points,
        homography=homography,
        min_kmh=min_kmh,
        max_kmh=max_kmh,
        cooldown_frames=cooldown_frames,
        net_offset_px=net_offset_px,
        speed_line_offset_px=speed_line_offset_px,
        speed_coef_up=speed_coef_up,
        speed_coef_down=speed_coef_down,
        speed_search_frames=speed_search_frames,
    )
    if pixel_trap_events is not None:
        return pixel_trap_events

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
        direction = "top_down" if curr.world_y > prev.world_y else "bottom_up"
        legacy_direction = "near_to_far" if direction == "top_down" else "far_to_near"
        events.append({
            "frame": curr.frame_index,
            "frame_index": curr.frame_index,
            "x": round(float(x_cross), 4),
            "y": 0.0,
            "pixel_x": round(float(px_cross), 2),
            "pixel_y": round(float(py_cross), 2),
            "speed_kmh": int(round(speed_kmh)),
            "direction": direction,
            "legacy_direction": legacy_direction,
            "source": "single_cam_speed_crossing",
        })
        last_emit = curr.frame_index
    return events


def _candidate_det_conf(det: list[float]) -> float:
    return float(det[6]) if len(det) > 6 else 0.0


def _candidate_det_track_id(det: list[Any]) -> str | None:
    if len(det) <= 7:
        return None
    value = det[7]
    if value is None:
        return None
    value_str = str(value)
    return value_str if value_str and value_str != "-1" else None


def _candidate_det_rank(det: list[Any]) -> int | None:
    if len(det) <= 8:
        return None
    try:
        return int(det[8])
    except (TypeError, ValueError):
        return None


def _candidate_det_static_blocked(det: list[Any]) -> bool:
    if len(det) <= 9:
        return False
    return bool(det[9])


def _candidate_detections_by_frame(detections: list[dict]) -> dict[int, list[list[float]]] | None:
    by_frame: dict[int, list[list[float]]] = {}
    saw_candidates = False
    for det in detections:
        frame = _frame_id(det.get("frame_index", det.get("frame")))
        if frame is None:
            continue
        candidates = det.get("raw_candidates") or det.get("candidates")
        if not candidates:
            continue
        saw_candidates = True
        frame_dets = by_frame.setdefault(frame, [])
        for rank, cand in enumerate(candidates):
            try:
                cx = float(cand.get("pixel_x"))
                cy = float(cand.get("pixel_y"))
            except (TypeError, ValueError):
                continue
            bbox = cand.get("bbox") or [cx, cy, cx, cy]
            if len(bbox) < 4:
                bbox = [cx, cy, cx, cy]
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                x1 = x2 = cx
                y1 = y2 = cy
            conf = float(cand.get("yolo_conf", cand.get("confidence", cand.get("blob_sum", 0.0))) or 0.0)
            track_id = cand.get("track_id")
            if track_id is None:
                track_id = cand.get("pseudo_track_id")
            track_id_value = str(track_id) if track_id is not None else "-1"
            static_blocked = bool(cand.get("static_blocked", False))
            frame_dets.append([cx, cy, x1, y1, x2, y2, conf, track_id_value, rank, static_blocked])
    return by_frame if saw_candidates else None


def _point_from_queue_det(
    *,
    frame: int,
    det: list[float],
    homography: Any,
) -> _Point | None:
    try:
        wx, wy = homography.pixel_to_world(float(det[0]), float(det[1]))
    except Exception:
        return None
    return _Point(
        frame_index=int(frame),
        pixel_x=float(det[0]),
        pixel_y=float(det[1]),
        world_x=float(wx),
        world_y=float(wy),
        confidence=_candidate_det_conf(det),
    )


def _event_from_queue_bounce(
    *,
    frame: int,
    pixel_x: float,
    pixel_y: float,
    angle: float,
    stats: dict[str, Any],
    confidence: float,
    camera_name: str,
    homography: Any,
) -> dict[str, Any] | None:
    try:
        wx, wy = homography.pixel_to_world(float(pixel_x), float(pixel_y))
    except Exception:
        return None
    in_court = _is_in_court(float(wx), float(wy))
    return {
        "sequence": 0,
        "frame": int(frame),
        "frame_index": int(frame),
        "camera": camera_name,
        "x": round(float(wx), 4),
        "y": round(float(wy), 4),
        "pixel_x": round(float(pixel_x), 2),
        "pixel_y": round(float(pixel_y), 2),
        "in_court": bool(in_court),
        "confidence": round(float(confidence), 4),
        "angle": round(float(angle), 2),
        "delta_v": round(float(stats.get("delta_v", 0.0)), 2),
        "y_reversal": bool(stats.get("y_reversal", False)),
        "queue_id": stats.get("queue_id"),
        "queue_history_len": stats.get("queue_history_len"),
        "queue_speed_px": stats.get("queue_speed_px"),
        "queue_track_id": stats.get("queue_track_id"),
        "queue_track_id_unique": stats.get("queue_track_id_unique"),
        "queue_conf_at_event": stats.get("queue_conf_at_event"),
        "queue_conf_last": stats.get("queue_conf_last"),
        "queue_conf_max": stats.get("queue_conf_max"),
        "queue_conf_avg": stats.get("queue_conf_avg"),
        "queue_candidate_rank_event": stats.get("queue_candidate_rank_event"),
        "queue_candidate_rank_last": stats.get("queue_candidate_rank_last"),
        "queue_candidate_rank_min": stats.get("queue_candidate_rank_min"),
        "queue_candidate_rank_max": stats.get("queue_candidate_rank_max"),
        "queue_candidate_rank_avg": stats.get("queue_candidate_rank_avg"),
        "queue_event_frame_gap": stats.get("queue_event_frame_gap"),
        "queue_static_blocked_history": stats.get("queue_static_blocked_history"),
        "type": "IN" if in_court else "OUT",
        "kind": "raw_bounce_candidate",
        "source": "yolo_verify_queue_raw_candidate",
    }


def _detect_candidate_queue_events(
    detections: list[dict],
    *,
    camera_name: str,
    players_by_frame: dict[int, list[dict[str, Any]]],
    homography: Any | None,
    filter_window: int,
    angle_thresh: float,
    momentum_thresh: float,
    tolerance: int,
    hit_angle_thresh: float,
    hit_dist_px_net: float,
    hit_dist_px_base: float,
    hit_search_radius_frames: int,
    roi_net_margin_m: float,
    speed_min_kmh: float,
    speed_max_kmh: float,
    speed_cooldown_frames: int,
    net_offset_px: float,
    speed_line_offset_px: float,
    speed_coef_up: float,
    speed_coef_down: float,
    speed_search_frames: int,
    lookback_frames: int,
    hit_suppress_frames: int,
    clean_time_frames: int,
    clean_space_meters: float,
) -> dict[str, Any] | None:
    if homography is None:
        return None
    candidates_by_frame = _candidate_detections_by_frame(detections)
    if not candidates_by_frame:
        return None
    lines = _speed_trap_lines_from_homography(
        homography,
        net_offset_px=net_offset_px,
        speed_line_offset_px=speed_line_offset_px,
    )
    if lines is None:
        return None
    net_line, speed_line = lines

    tracker = _VerifyQueueTracker()
    stitcher = _VerifyTrajectoryStitcher()
    analyzer = _VerifyTrajectoryAnalyzer(
        window=filter_window,
        angle_thresh=angle_thresh,
        momentum_thresh=momentum_thresh,
        tolerance=tolerance,
    )
    point_by_frame: dict[int, _Point] = {}
    raw_bounce_candidates: dict[int, dict[str, Any]] = {}
    hit_by_frame: dict[int, dict[str, Any]] = {}
    hit_suppression_frames: set[int] = set()
    suppressed_bounces: list[dict[str, Any]] = []
    speed_events: list[dict[str, Any]] = []
    deduped_speed_keys: set[tuple[int, str]] = set()
    processed_bounce_frames: set[int] = set()
    moving_boxes = 0
    static_boxes = 0
    stitched_points_max = 0
    last_speed_emit = -10**9

    for frame in sorted(candidates_by_frame):
        tracker.process_frame(frame, candidates_by_frame.get(frame, []))
        before_stitched = sum(1 for q in tracker.queues for item in q["history"] if len(item) > 2 and item[2])
        tracker.queues = stitcher.stitch_queues(tracker.queues)
        after_stitched = sum(1 for q in tracker.queues for item in q["history"] if len(item) > 2 and item[2])
        stitched_points_max = max(stitched_points_max, before_stitched, after_stitched)

        for q in tracker.queues:
            if not q["is_static"]:
                for hist_frame, hist_det, _is_stitched in q["history"]:
                    point = _point_from_queue_det(frame=int(hist_frame), det=hist_det, homography=homography)
                    if point is not None:
                        point_by_frame[int(hist_frame)] = point

        moving_dets, static_dets, frame_bounces, frame_crossings = tracker.get_events(
            frame,
            analyzer,
            net_line=net_line,
            speed_line=speed_line,
            speed_search_frames=speed_search_frames,
        )
        moving_boxes += len(moving_dets)
        static_boxes += len(static_dets)

        for crossing in frame_crossings:
            c_frame = int(crossing.get("frame_index", crossing.get("frame", -1)))
            direction = str(crossing.get("direction", ""))
            if c_frame < 0 or not direction:
                continue
            if (c_frame, direction) in deduped_speed_keys:
                continue
            if c_frame - last_speed_emit <= speed_cooldown_frames:
                continue
            speed_px = float(crossing.get("speed_px", 0.0) or 0.0)
            coef = float(speed_coef_up) if direction == "bottom_up" else float(speed_coef_down)
            speed_kmh = speed_px * coef
            if not (speed_min_kmh <= speed_kmh <= speed_max_kmh):
                continue
            px = float(crossing.get("pixel_x", 0.0) or 0.0)
            py = float(crossing.get("pixel_y", 0.0) or 0.0)
            try:
                wx, _wy = homography.pixel_to_world(px, py)
            except Exception:
                wx = 0.0
            legacy_direction = "near_to_far" if direction == "top_down" else "far_to_near"
            speed_events.append({
                "frame": c_frame,
                "frame_index": c_frame,
                "camera": camera_name,
                "x": round(float(wx), 4),
                "y": 0.0,
                "pixel_x": round(px, 2),
                "pixel_y": round(py, 2),
                "speed_px": round(speed_px, 4),
                "speed_kmh": round(float(speed_kmh), 1),
                "speed_ref_frame": crossing.get("speed_ref_frame"),
                "direction": direction,
                "legacy_direction": legacy_direction,
                "source": "single_cam_verify_queue_speed_trap",
                "net_offset_px": float(net_offset_px),
                "speed_line_offset_px": float(speed_line_offset_px),
            })
            deduped_speed_keys.add((c_frame, direction))
            last_speed_emit = c_frame

        for b_frame, bx, by, angle, stats in frame_bounces:
            b_frame = int(b_frame)
            if b_frame in processed_bounce_frames:
                continue
            processed_bounce_frames.add(b_frame)
            point = _point_from_queue_det(frame=b_frame, det=[float(bx), float(by), 0, 0, 0, 0, 0], homography=homography)
            confidence = float(
                stats.get("queue_conf_at_event")
                or stats.get("queue_conf_last")
                or (point.confidence if point is not None else 0.0)
                or 0.0
            )
            event = _event_from_queue_bounce(
                frame=b_frame,
                pixel_x=float(bx),
                pixel_y=float(by),
                angle=float(angle),
                stats=stats,
                confidence=confidence,
                camera_name=camera_name,
                homography=homography,
            )
            if event is not None:
                raw_bounce_candidates[b_frame] = event

    for crossing in speed_events:
        direction = str(crossing.get("direction", ""))
        if direction == "top_down":
            hit = _find_lookback_hit(
                crossing=crossing,
                half="top",
                source="top_down_lookback",
                point_by_frame=point_by_frame,
                players_by_frame=players_by_frame,
                lookback_frames=lookback_frames,
                roi_net_margin_m=roi_net_margin_m,
                top_hit_dist_px=50.0,
                hit_dist_px_net=hit_dist_px_net,
                hit_dist_px_base=hit_dist_px_base,
            )
            if hit is not None:
                hit["camera"] = camera_name
                hit_suppression_frames.add(int(hit["frame_index"]))
                _dedupe_hit(
                    hit_by_frame,
                    hit,
                    clean_time_frames=clean_time_frames,
                    clean_space_meters=clean_space_meters,
                )

    for frame_index in sorted(raw_bounce_candidates):
        base_event = raw_bounce_candidates[frame_index]
        player_match = None
        if float(base_event["y"]) > 0.0 and float(base_event["angle"]) >= hit_angle_thresh:
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
            hit_event = _make_hit_event(
                base_event=base_event,
                player=player,
                source="bottom_reversal_player_anchor",
                distance_px=float(player_match["distance_px"]),
                threshold_px=float(player_match["threshold_px"]),
                angle=float(base_event["angle"]),
            )
            hit_event["player_frame"] = player_match["frame"]
            hit_suppression_frames.add(int(hit_event["frame_index"]))
            _dedupe_hit(
                hit_by_frame,
                hit_event,
                clean_time_frames=clean_time_frames,
                clean_space_meters=clean_space_meters,
            )

    for crossing in speed_events:
        if crossing.get("direction") != "bottom_up":
            continue
        c_frame = _frame_id(crossing.get("frame_index", crossing.get("frame")))
        if c_frame is None:
            continue
        start = c_frame - lookback_frames
        has_bottom_hit = any(
            start <= int(hit.get("frame_index", -10**9)) <= c_frame
            and str(hit.get("source", "")).startswith("bottom")
            for hit in hit_by_frame.values()
        )
        if has_bottom_hit:
            continue
        hit = _find_lookback_hit(
            crossing=crossing,
            half="bottom",
            source="bottom_up_lookback",
            point_by_frame=point_by_frame,
            players_by_frame=players_by_frame,
            lookback_frames=lookback_frames,
            roi_net_margin_m=roi_net_margin_m,
            top_hit_dist_px=50.0,
            hit_dist_px_net=hit_dist_px_net,
            hit_dist_px_base=hit_dist_px_base,
        )
        if hit is not None:
            hit["camera"] = camera_name
            hit_suppression_frames.add(int(hit["frame_index"]))
            _dedupe_hit(
                hit_by_frame,
                hit,
                clean_time_frames=clean_time_frames,
                clean_space_meters=clean_space_meters,
            )

    bounce_candidates_after_hit: list[dict[str, Any]] = []
    for frame_index in sorted(raw_bounce_candidates):
        event = raw_bounce_candidates[frame_index]
        suppressing_hit = next(
            (hit_frame for hit_frame in sorted(hit_suppression_frames) if abs(frame_index - hit_frame) <= hit_suppress_frames),
            None,
        )
        if suppressing_hit is not None:
            suppressed_bounces.append({
                **event,
                "suppressed_by_hit_frame": suppressing_hit,
                "suppression_reason": "hit_window",
            })
            continue
        bounce_candidates_after_hit.append({
            **event,
            "kind": "bounce",
            "type": "IN" if event["in_court"] else "OUT",
            "source": "yolo_verify_queue_single_cam",
        })

    final_bounces, deduped_bounces = _select_strongest_bounces(
        bounce_candidates_after_hit,
        clean_time_frames=clean_time_frames,
        clean_space_meters=clean_space_meters,
    )
    hit_events = [hit_by_frame[k] for k in sorted(hit_by_frame)]
    publishable_bounces, gate_only_bounces = _split_gate_only_bounces(final_bounces)
    final_bounces, out_rally_suppressed = _apply_out_rally_gate(
        [*publishable_bounces, *gate_only_bounces],
        hit_events=hit_events,
        speed_events=speed_events,
    )
    for seq, event in enumerate(final_bounces, start=1):
        event["sequence"] = seq
    for seq, event in enumerate(hit_events, start=1):
        event["sequence"] = seq

    raw_ball_boxes = sum(len(v) for v in candidates_by_frame.values())
    return {
        "camera": camera_name,
        "detections": raw_ball_boxes,
        "filtered_points": len(point_by_frame),
        "bounces": final_bounces,
        "hits": hit_events,
        "speed_events": speed_events,
        "raw_bounce_candidates": list(raw_bounce_candidates.values()),
        "suppressed_bounces": suppressed_bounces,
        "hit_suppression_frames": sorted(hit_suppression_frames),
        "deduped_bounce_candidates": deduped_bounces,
        "gate_only_bounces": gate_only_bounces,
        "out_rally_suppressed_bounces": out_rally_suppressed,
        "count": len(final_bounces),
        "hit_count": len(hit_events),
        "speed_count": len(speed_events),
        "raw_bounce_candidate_count": len(raw_bounce_candidates),
        "suppressed_bounces_by_hit_window": len(suppressed_bounces),
        "deduped_bounces_after_hit": len(deduped_bounces),
        "gate_only_bounce_count": len(gate_only_bounces),
        "out_rally_suppressed_bounce_count": len(out_rally_suppressed),
        "queue_tracker_stats": {
            "raw_ball_boxes": raw_ball_boxes,
            "moving_boxes": moving_boxes,
            "static_boxes": static_boxes,
            "trajectory_points": len(point_by_frame),
            "stitched_points": stitched_points_max,
        },
        "params": {
            "event_chain": "verify_queue_candidates",
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
            "net_offset_px": net_offset_px,
            "speed_line_offset_px": speed_line_offset_px,
            "speed_coef_up": speed_coef_up,
            "speed_coef_down": speed_coef_down,
            "speed_search_frames": speed_search_frames,
            "lookback_frames": lookback_frames,
            "hit_suppress_frames": hit_suppress_frames,
            "clean_time_frames": clean_time_frames,
            "clean_space_meters": clean_space_meters,
        },
    }


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
    speed_min_kmh: float = 0.0,
    speed_max_kmh: float = 10_000.0,
    speed_cooldown_frames: int = 12,
    net_offset_px: float = DEFAULT_NET_OFFSET_PX,
    speed_line_offset_px: float = DEFAULT_SPEED_LINE_OFFSET_PX,
    speed_coef_up: float = DEFAULT_SPEED_COEF_UP,
    speed_coef_down: float = DEFAULT_SPEED_COEF_DOWN,
    speed_search_frames: int = DEFAULT_SPEED_SEARCH_FRAMES,
    lookback_frames: int = DEFAULT_LOOKBACK_FRAMES,
    hit_suppress_frames: int = DEFAULT_HIT_SUPPRESS_FRAMES,
    clean_time_frames: int = DEFAULT_CLEAN_TIME_FRAMES,
    clean_space_meters: float = DEFAULT_CLEAN_SPACE_METERS,
) -> dict[str, Any]:
    """Detect bounce, hit and speed events from a single-camera YOLO track.

    This follows the verified ``verify_tennis.py`` order:

    1. Keep raw bounce candidates.
    2. Detect top_down lookback HIT, bottom reversal HIT, and bottom_up fallback HIT.
    3. Suppress candidates within the HIT +/- frame window.
    4. Deduplicate remaining bounces by time + court distance.
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
    queue_result = _detect_candidate_queue_events(
        detections,
        camera_name=camera_name,
        players_by_frame=players_by_frame,
        homography=homography,
        filter_window=filter_window,
        angle_thresh=angle_thresh,
        momentum_thresh=momentum_thresh,
        tolerance=tolerance,
        hit_angle_thresh=hit_angle_thresh,
        hit_dist_px_net=hit_dist_px_net,
        hit_dist_px_base=hit_dist_px_base,
        hit_search_radius_frames=hit_search_radius_frames,
        roi_net_margin_m=roi_net_margin_m,
        speed_min_kmh=speed_min_kmh,
        speed_max_kmh=speed_max_kmh,
        speed_cooldown_frames=speed_cooldown_frames,
        net_offset_px=net_offset_px,
        speed_line_offset_px=speed_line_offset_px,
        speed_coef_up=speed_coef_up,
        speed_coef_down=speed_coef_down,
        speed_search_frames=speed_search_frames,
        lookback_frames=lookback_frames,
        hit_suppress_frames=hit_suppress_frames,
        clean_time_frames=clean_time_frames,
        clean_space_meters=clean_space_meters,
    )
    if queue_result is not None:
        return queue_result

    speed_events = _detect_single_cam_speed_crossings(
        smoothed,
        homography=homography,
        fps=fps,
        min_kmh=speed_min_kmh,
        max_kmh=speed_max_kmh,
        cooldown_frames=speed_cooldown_frames,
        net_offset_px=net_offset_px,
        speed_line_offset_px=speed_line_offset_px,
        speed_coef_up=speed_coef_up,
        speed_coef_down=speed_coef_down,
        speed_search_frames=speed_search_frames,
    )
    for event in speed_events:
        event["camera"] = camera_name

    raw_bounce_candidates: dict[int, dict[str, Any]] = {}
    hit_by_frame: dict[int, dict[str, Any]] = {}
    hit_suppression_frames: set[int] = set()
    suppressed_bounces: list[dict[str, Any]] = []
    cleaned_duplicate_bounces = 0

    # Net-crossing lookback HITs first. This mirrors the offline prototype
    # where a top_down crossing can retroactively explain an upper-half hit.
    for crossing in speed_events:
        direction = str(crossing.get("direction", ""))
        if direction == "top_down":
            hit = _find_lookback_hit(
                crossing=crossing,
                half="top",
                source="top_down_lookback",
                point_by_frame=point_by_frame,
                players_by_frame=players_by_frame,
                lookback_frames=lookback_frames,
                roi_net_margin_m=roi_net_margin_m,
                top_hit_dist_px=50.0,
                hit_dist_px_net=hit_dist_px_net,
                hit_dist_px_base=hit_dist_px_base,
            )
            if hit is not None:
                hit["camera"] = camera_name
                hit_suppression_frames.add(int(hit["frame_index"]))
                _dedupe_hit(
                    hit_by_frame,
                    hit,
                    clean_time_frames=clean_time_frames,
                    clean_space_meters=clean_space_meters,
                )

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
        raw_bounce_candidates[frame_index] = {
            **base_event,
            "type": "IN" if base_event["in_court"] else "OUT",
            "kind": "raw_bounce_candidate",
            "source": "yolo_fuzzy_raw_candidate",
        }

        player_match = None
        if float(base_event["y"]) > 0.0 and float(base_event["angle"]) >= hit_angle_thresh:
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
            hit_event = _make_hit_event(
                base_event=base_event,
                player=player,
                source="bottom_reversal_player_anchor",
                distance_px=float(player_match["distance_px"]),
                threshold_px=float(player_match["threshold_px"]),
                angle=float(base_event["angle"]),
            )
            hit_event["player_frame"] = player_match["frame"]
            hit_suppression_frames.add(int(hit_event["frame_index"]))
            _dedupe_hit(
                hit_by_frame,
                hit_event,
                clean_time_frames=clean_time_frames,
                clean_space_meters=clean_space_meters,
            )

    # Bottom-up crossing fallback: if no direct lower-half HIT was seen before
    # the crossing, use the same 50-frame player-anchor lookback as offline.
    for crossing in speed_events:
        if crossing.get("direction") != "bottom_up":
            continue
        c_frame = _frame_id(crossing.get("frame_index", crossing.get("frame")))
        if c_frame is None:
            continue
        start = c_frame - lookback_frames
        has_bottom_hit = any(
            start <= int(hit.get("frame_index", -10**9)) <= c_frame
            and str(hit.get("source", "")).startswith("bottom")
            for hit in hit_by_frame.values()
        )
        if has_bottom_hit:
            continue
        hit = _find_lookback_hit(
            crossing=crossing,
            half="bottom",
            source="bottom_up_lookback",
            point_by_frame=point_by_frame,
            players_by_frame=players_by_frame,
            lookback_frames=lookback_frames,
            roi_net_margin_m=roi_net_margin_m,
            top_hit_dist_px=50.0,
            hit_dist_px_net=hit_dist_px_net,
            hit_dist_px_base=hit_dist_px_base,
        )
        if hit is not None:
            hit["camera"] = camera_name
            hit_suppression_frames.add(int(hit["frame_index"]))
            _dedupe_hit(
                hit_by_frame,
                hit,
                clean_time_frames=clean_time_frames,
                clean_space_meters=clean_space_meters,
            )

    bounce_candidates_after_hit: list[dict[str, Any]] = []
    for frame_index in sorted(raw_bounce_candidates):
        event = raw_bounce_candidates[frame_index]
        suppressing_hit = next(
            (hit_frame for hit_frame in sorted(hit_suppression_frames) if abs(frame_index - hit_frame) <= hit_suppress_frames),
            None,
        )
        if suppressing_hit is not None:
            suppressed_bounces.append({
                **event,
                "suppressed_by_hit_frame": suppressing_hit,
                "suppression_reason": "hit_window",
            })
            continue

        bounce_candidates_after_hit.append({
            **event,
            "kind": "bounce",
            "type": "IN" if event["in_court"] else "OUT",
            "source": "yolo_fuzzy_single_cam",
        })

    final_bounces, deduped_bounces = _select_strongest_bounces(
        bounce_candidates_after_hit,
        clean_time_frames=clean_time_frames,
        clean_space_meters=clean_space_meters,
    )
    cleaned_duplicate_bounces = len(deduped_bounces)

    hit_events = [hit_by_frame[k] for k in sorted(hit_by_frame)]
    publishable_bounces, gate_only_bounces = _split_gate_only_bounces(final_bounces)
    final_bounces, out_rally_suppressed = _apply_out_rally_gate(
        [*publishable_bounces, *gate_only_bounces],
        hit_events=hit_events,
        speed_events=speed_events,
    )
    for seq, event in enumerate(final_bounces, start=1):
        event["sequence"] = seq
    for seq, event in enumerate(hit_events, start=1):
        event["sequence"] = seq

    return {
        "camera": camera_name,
        "detections": raw_count,
        "filtered_points": len(point_by_frame),
        "bounces": final_bounces,
        "hits": hit_events,
        "speed_events": speed_events,
        "raw_bounce_candidates": list(raw_bounce_candidates.values()),
        "suppressed_bounces": suppressed_bounces,
        "hit_suppression_frames": sorted(hit_suppression_frames),
        "deduped_bounce_candidates": deduped_bounces,
        "gate_only_bounces": gate_only_bounces,
        "out_rally_suppressed_bounces": out_rally_suppressed,
        "count": len(final_bounces),
        "hit_count": len(hit_events),
        "speed_count": len(speed_events),
        "raw_bounce_candidate_count": len(raw_bounce_candidates),
        "suppressed_bounces_by_hit_window": len(suppressed_bounces),
        "deduped_bounces_after_hit": cleaned_duplicate_bounces,
        "gate_only_bounce_count": len(gate_only_bounces),
        "out_rally_suppressed_bounce_count": len(out_rally_suppressed),
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
            "net_offset_px": net_offset_px,
            "speed_line_offset_px": speed_line_offset_px,
            "speed_coef_up": speed_coef_up,
            "speed_coef_down": speed_coef_down,
            "speed_search_frames": speed_search_frames,
            "lookback_frames": lookback_frames,
            "hit_suppress_frames": hit_suppress_frames,
            "clean_time_frames": clean_time_frames,
            "clean_space_meters": clean_space_meters,
        },
    }
