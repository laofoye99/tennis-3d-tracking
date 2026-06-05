"""Realtime HIT-first, BOUNCE-after-cleaning event refiner.

This module ports the event order that was validated in
``yolo_roadmap/verify_tennis.py`` into a streaming form:

1. Keep raw bounce candidates intact.
2. Detect HIT events from net-crossing lookback and lower-half reversals.
3. Suppress bounce candidates around HIT frames.
4. Deduplicate the remaining bounce candidates before publishing them.
"""

from __future__ import annotations

from collections import Counter, deque
import math
from typing import Any

from app.analytics import COURT_Y_MAX, COURT_Y_MIN, SINGLES_X_MAX, SINGLES_X_MIN


class HitBounceRefiner:
    """Streaming event refiner used before realtime bounce fanout.

    The refiner intentionally delays bounce publication by the lookback window.
    That gives late net-crossing lookback HITs a chance to suppress nearby raw
    bounce candidates before minimap, report, and 3D push see them.
    """

    def __init__(self, config: Any | None = None):
        self.enabled = bool(getattr(config, "enabled", True))
        self.show_hits_on_minimap = bool(getattr(config, "show_hits_on_minimap", True))
        self.lookback_frames = int(getattr(config, "lookback_frames", 50))
        self.release_delay_frames = int(
            getattr(config, "release_delay_frames", self.lookback_frames)
        )
        self.hit_suppression_frames = int(getattr(config, "hit_suppression_frames", 3))
        self.hit_angle_thresh = float(getattr(config, "hit_angle_thresh", 45.0))
        self.top_hit_dist_px = float(getattr(config, "top_hit_dist_px", 50.0))
        self.bottom_hit_dist_px_net = float(getattr(config, "bottom_hit_dist_px_net", 100.0))
        self.bottom_hit_dist_px_base = float(getattr(config, "bottom_hit_dist_px_base", 250.0))
        self.top_hit_dist_m = float(getattr(config, "top_hit_dist_m", 1.2))
        self.bottom_hit_dist_m_net = float(getattr(config, "bottom_hit_dist_m_net", 1.2))
        self.bottom_hit_dist_m_base = float(getattr(config, "bottom_hit_dist_m_base", 2.5))
        self.clean_time_frames = int(getattr(config, "clean_time_frames", 25))
        self.clean_space_meters = float(getattr(config, "clean_space_meters", 1.5))
        history_frames = int(getattr(config, "history_frames", 150))
        self._history = deque(maxlen=max(history_frames, self.lookback_frames + 20))
        self._raw_bounces: dict[int, dict] = {}
        self._pending_bounces: dict[int, dict] = {}
        self._final_bounces: list[dict] = []
        self._hits: dict[int, dict] = {}
        self._processed_crossings: set[tuple] = set()
        self._processed_reversal_frames: set[int] = set()
        self._released_bounce_frames: set[int] = set()
        self._stats: Counter = Counter()
        self._synthetic_frame = 0

    def reset(self) -> None:
        self._history.clear()
        self._raw_bounces.clear()
        self._pending_bounces.clear()
        self._final_bounces.clear()
        self._hits.clear()
        self._processed_crossings.clear()
        self._processed_reversal_frames.clear()
        self._released_bounce_frames.clear()
        self._stats.clear()
        self._synthetic_frame = 0

    def update(
        self,
        point: dict | None,
        *,
        raw_bounce: dict | None = None,
        players: list[dict] | None = None,
        net_crossing: dict | None = None,
        cam_dets: dict | None = None,
        now: float | None = None,
    ) -> dict:
        """Process one realtime point and return newly emitted events."""
        result = {
            "new_hits": [],
            "new_final_bounces": [],
            "suppressed_bounces": [],
            "stats": self.get_stats(),
        }

        if not self.enabled:
            if raw_bounce is not None:
                result["new_final_bounces"].append(self._normalize_bounce(raw_bounce, now=now))
            result["stats"] = self.get_stats()
            return result

        latest_frame = None
        if point is not None:
            sample = self._make_sample(point, players or [], cam_dets or {}, now=now)
            latest_frame = sample["frame_index"]
            self._history.append(sample)
            direct_hit = self._detect_bottom_reversal_hit()
            if direct_hit is not None:
                result["new_hits"].append(direct_hit)

        if raw_bounce is not None:
            bd = self._normalize_bounce(raw_bounce, now=now)
            b_frame = self._event_frame(bd)
            if b_frame not in self._raw_bounces:
                self._raw_bounces[b_frame] = bd
                self._pending_bounces[b_frame] = bd
                self._stats["raw_bounce_candidates"] += 1

        if net_crossing is not None:
            crossing_hits = self._process_crossing(net_crossing)
            result["new_hits"].extend(crossing_hits)

        if latest_frame is None and point is None:
            latest_frame = self._latest_frame()
        if latest_frame is not None:
            final_bounces, suppressed = self._release_ready_bounces(latest_frame)
            result["new_final_bounces"].extend(final_bounces)
            result["suppressed_bounces"].extend(suppressed)

        result["stats"] = self.get_stats()
        return result

    def get_recent_hits(self, limit: int = 100) -> list[dict]:
        return sorted(self._hits.values(), key=lambda h: h.get("frame_index", 0))[-limit:]

    def get_stats(self) -> dict:
        stats = dict(self._stats)
        stats.update({
            "raw_bounce_candidate_count": self._stats.get("raw_bounce_candidates", 0),
            "pending_bounce_count": len(self._pending_bounces),
            "recent_hit_count": len(self._hits),
            "final_bounce_count": len(self._final_bounces),
            "show_hits_on_minimap": self.show_hits_on_minimap,
        })
        return stats

    def _make_sample(
        self,
        point: dict,
        players: list[dict],
        cam_dets: dict,
        *,
        now: float | None,
    ) -> dict:
        frame = self._event_frame(point)
        timestamp = float(point.get("timestamp", now or 0.0) or 0.0)
        capture_ts = float(point.get("capture_ts", timestamp) or timestamp)
        return {
            "frame_index": frame,
            "timestamp": timestamp,
            "capture_ts": capture_ts,
            "ball": {
                "x": float(point.get("x", 0.0) or 0.0),
                "y": float(point.get("y", 0.0) or 0.0),
                "z": float(point.get("z", 0.0) or 0.0),
            },
            "players": [self._normalize_player(p) for p in players],
            "cam_dets": cam_dets or {},
        }

    def _normalize_player(self, player: dict) -> dict:
        foot = player.get("foot_court") or player.get("foot") or [None, None]
        foot_x = self._safe_float(foot[0]) if len(foot) > 0 else None
        foot_y = self._safe_float(foot[1]) if len(foot) > 1 else None
        side = player.get("side")
        if side not in ("near", "far"):
            side = "near" if foot_y is not None and foot_y < 0 else "far"
        return {
            **player,
            "side": side,
            "foot_court": [foot_x, foot_y] if foot_x is not None and foot_y is not None else None,
            "hit_anchor_court": player.get("hit_anchor_court") or player.get("foot_court"),
            "hit_anchor_px": player.get("hit_anchor_px"),
        }

    def _normalize_bounce(self, bounce: dict, *, now: float | None) -> dict:
        bd = dict(bounce)
        frame = self._event_frame(bd)
        bd["frame_index"] = frame
        if bd.get("timestamp") is None:
            bd["timestamp"] = float(bd.get("capture_ts", now or 0.0) or 0.0)
        if bd.get("capture_ts") is None:
            bd["capture_ts"] = float(bd.get("timestamp", now or 0.0) or 0.0)
        if bd.get("in_court") is None:
            x = self._safe_float(bd.get("x"))
            y = self._safe_float(bd.get("y"))
            bd["in_court"] = (
                x is not None and y is not None
                and SINGLES_X_MIN <= x <= SINGLES_X_MAX
                and COURT_Y_MIN <= y <= COURT_Y_MAX
            )
        bd.setdefault("event_type", "bounce")
        bd.setdefault("refiner_source", "raw_hybrid")
        return bd

    def _process_crossing(self, crossing: dict) -> list[dict]:
        direction = crossing.get("direction") or ""
        c_frame = self._event_frame(crossing)
        key = (
            c_frame,
            direction,
            round(float(crossing.get("timestamp", 0.0) or 0.0), 3),
        )
        if key in self._processed_crossings:
            return []
        self._processed_crossings.add(key)
        self._stats["net_crossings"] += 1

        # The hit happens on the side the ball came from.
        hit_side = "far" if direction == "far_to_near" else "near"
        source = "bottom_up_lookback" if hit_side == "far" else "top_down_lookback"
        hit = self._find_lookback_hit(c_frame, hit_side, source, crossing)
        if hit is None:
            return []
        return [hit]

    def _find_lookback_hit(
        self,
        crossing_frame: int,
        side: str,
        source: str,
        crossing: dict | None = None,
    ) -> dict | None:
        best = None
        best_score = float("inf")
        start = crossing_frame - self.lookback_frames
        for sample in reversed(self._history):
            f = sample["frame_index"]
            if f < start:
                break
            if f > crossing_frame:
                continue
            for player in sample["players"]:
                if not self._player_on_side(player, side):
                    continue
                dist = self._distance_to_player(sample, player, bottom=(side == "far"))
                if dist is None:
                    continue
                score = dist["distance"] / max(dist["threshold"], 1e-6)
                if score < best_score:
                    best_score = score
                    best = (sample, player, dist)
        if best is None or best_score > 1.0:
            return None
        sample, player, dist = best
        hit = self._make_hit_event(
            sample,
            player,
            source=source,
            distance=dist,
            crossing=crossing,
        )
        return self._store_hit(hit)

    def _detect_bottom_reversal_hit(self) -> dict | None:
        if len(self._history) < 5:
            return None
        prev = self._history[-5]
        cand = self._history[-3]
        nxt = self._history[-1]
        frame = cand["frame_index"]
        if frame in self._processed_reversal_frames:
            return None
        self._processed_reversal_frames.add(frame)
        if cand["ball"]["y"] <= 0:
            return None

        vin = (
            cand["ball"]["x"] - prev["ball"]["x"],
            cand["ball"]["y"] - prev["ball"]["y"],
        )
        vout = (
            nxt["ball"]["x"] - cand["ball"]["x"],
            nxt["ball"]["y"] - cand["ball"]["y"],
        )
        if vin[1] <= 0 or vout[1] >= 0:
            return None
        angle = self._angle_between(vin, vout)
        self._stats["bottom_reversal_candidates"] += 1
        if angle < self.hit_angle_thresh:
            return None

        best = None
        best_score = float("inf")
        for sample in self._nearby_samples(frame, radius=2):
            for player in sample["players"]:
                if not self._player_on_side(player, "far"):
                    continue
                dist = self._distance_to_player(cand, player, bottom=True)
                if dist is None:
                    continue
                score = dist["distance"] / max(dist["threshold"], 1e-6)
                if score < best_score:
                    best_score = score
                    best = (player, dist)

        if best is None or best_score > 1.0:
            return None
        player, dist = best
        hit = self._make_hit_event(
            cand,
            player,
            source="bottom_reversal_player_anchor",
            distance=dist,
            angle=angle,
        )
        return self._store_hit(hit)

    def _release_ready_bounces(self, latest_frame: int) -> tuple[list[dict], list[dict]]:
        ready = [
            frame for frame in sorted(self._pending_bounces)
            if frame <= latest_frame - self.release_delay_frames
        ]
        suppressed = []
        candidates = []
        for frame in ready:
            bd = self._pending_bounces.pop(frame)
            if frame in self._released_bounce_frames:
                continue
            if self._is_suppressed_by_hit(bd):
                suppressed.append(bd)
                self._stats["suppressed_bounces_by_hit"] += 1
                continue
            candidates.append(bd)

        selected, deduped = self._select_strongest_bounces(candidates)
        self._stats["deduped_bounces_after_hit"] += len(deduped)

        final_bounces = []
        for bd in selected:
            if self._is_duplicate_final_bounce(bd):
                self._stats["deduped_bounces_after_hit"] += 1
                continue
            bd = dict(bd)
            bd["event_type"] = "bounce"
            bd["refiner_source"] = "hit_first_final"
            self._released_bounce_frames.add(self._event_frame(bd))
            self._final_bounces.append(bd)
            if len(self._final_bounces) > 500:
                self._final_bounces = self._final_bounces[-500:]
            final_bounces.append(bd)
        return final_bounces, suppressed

    def _is_suppressed_by_hit(self, bounce: dict) -> bool:
        b_frame = self._event_frame(bounce)
        for hit in self._hits.values():
            if abs(b_frame - int(hit.get("frame_index", -999999))) <= self.hit_suppression_frames:
                return True
        return False

    def _is_duplicate_final_bounce(self, bounce: dict) -> bool:
        bx = self._safe_float(bounce.get("x"))
        by = self._safe_float(bounce.get("y"))
        b_frame = self._event_frame(bounce)
        if bx is None or by is None:
            return False
        for prev in reversed(self._final_bounces[-20:]):
            p_frame = self._event_frame(prev)
            if b_frame - p_frame > self.clean_time_frames:
                break
            px = self._safe_float(prev.get("x"))
            py = self._safe_float(prev.get("y"))
            if px is None or py is None:
                continue
            if math.hypot(bx - px, by - py) <= self.clean_space_meters:
                return True
        return False

    def _select_strongest_bounces(self, bounces: list[dict]) -> tuple[list[dict], list[dict]]:
        clusters: list[list[dict]] = []
        for bounce in sorted(bounces, key=self._event_frame):
            enriched = dict(bounce)
            enriched["bounce_signal_score"] = self._bounce_signal_score(enriched)
            target = None
            for cluster in clusters:
                if any(self._same_bounce_window(enriched, member) for member in cluster):
                    target = cluster
                    break
            if target is None:
                clusters.append([enriched])
            else:
                target.append(enriched)

        selected = []
        dropped = []
        for cluster in clusters:
            best = max(
                cluster,
                key=lambda e: (
                    self._safe_float(e.get("bounce_signal_score")) or 0.0,
                    self._safe_float(e.get("delta_v")) or 0.0,
                    self._safe_float(e.get("angle")) or 0.0,
                    self._safe_float(e.get("confidence")) or 0.0,
                    -self._event_frame(e),
                ),
            )
            selected.append({**best, "dedupe_cluster_size": len(cluster)})
            for bounce in cluster:
                if bounce is best:
                    continue
                dropped.append({
                    **bounce,
                    "deduped_by_frame": self._event_frame(best),
                    "dedupe_reason": "weaker_bounce_signal_same_window",
                })
        return selected, dropped

    def _same_bounce_window(self, a: dict, b: dict) -> bool:
        if abs(self._event_frame(a) - self._event_frame(b)) > self.clean_time_frames:
            return False
        ax = self._safe_float(a.get("x"))
        ay = self._safe_float(a.get("y"))
        bx = self._safe_float(b.get("x"))
        by = self._safe_float(b.get("y"))
        if None in (ax, ay, bx, by):
            return False
        return math.hypot(ax - bx, ay - by) <= self.clean_space_meters

    def _bounce_signal_score(self, bounce: dict) -> float:
        angle = max(0.0, self._safe_float(bounce.get("angle")) or 0.0)
        delta_v = max(0.0, self._safe_float(bounce.get("delta_v")) or 0.0)
        confidence = max(0.0, self._safe_float(bounce.get("confidence")) or 0.0)
        y_bonus = 25.0 if bounce.get("y_reversal") else 0.0
        return round(angle + delta_v * 2.0 + confidence * 10.0 + y_bonus, 4)

    def _store_hit(self, hit: dict | None) -> dict | None:
        if hit is None:
            return None
        frame = self._event_frame(hit)
        hx = self._safe_float(hit.get("x"))
        hy = self._safe_float(hit.get("y"))
        for prev in self._hits.values():
            p_frame = self._event_frame(prev)
            if abs(frame - p_frame) > self.clean_time_frames:
                continue
            px = self._safe_float(prev.get("x"))
            py = self._safe_float(prev.get("y"))
            if None not in (hx, hy, px, py) and math.hypot(hx - px, hy - py) <= self.clean_space_meters:
                return None

        self._hits[frame] = hit
        source = str(hit.get("source", ""))
        if source == "top_down_lookback":
            self._stats["top_down_lookback_hits"] += 1
        elif source == "bottom_up_lookback":
            self._stats["bottom_up_lookback_hits"] += 1
        elif source == "bottom_reversal_player_anchor":
            self._stats["bottom_direct_hits"] += 1
        self._stats["hit_events"] += 1
        return hit

    def _make_hit_event(
        self,
        sample: dict,
        player: dict,
        *,
        source: str,
        distance: dict,
        angle: float | None = None,
        crossing: dict | None = None,
    ) -> dict:
        ball = sample["ball"]
        foot = player.get("foot_court") or [ball["x"], ball["y"]]
        foot_x, foot_y = foot[0], foot[1]
        hit_y = foot_y if foot_y is not None else ball["y"]
        hit = {
            "event_type": "hit",
            "frame_index": sample["frame_index"],
            "timestamp": sample["timestamp"],
            "capture_ts": sample["capture_ts"],
            "x": round(float(ball["x"]), 4),
            "y": round(float(hit_y), 4),
            "z": round(float(ball["z"]), 4),
            "source": source,
            "side": player.get("side"),
            "ball_x": round(float(ball["x"]), 4),
            "ball_y": round(float(ball["y"]), 4),
            "player_x": round(float(foot_x), 4) if foot_x is not None else None,
            "player_y": round(float(hit_y), 4),
            "distance": round(float(distance["distance"]), 4),
            "distance_unit": distance["unit"],
            "threshold": round(float(distance["threshold"]), 4),
        }
        if angle is not None:
            hit["angle"] = round(float(angle), 2)
        if crossing is not None:
            hit["crossing_frame"] = self._event_frame(crossing)
            hit["crossing_direction"] = crossing.get("direction")
        return hit

    def _distance_to_player(self, sample: dict, player: dict, *, bottom: bool) -> dict | None:
        cam_name = player.get("camera_name")
        anchor_px = player.get("hit_anchor_px")
        if cam_name and anchor_px and cam_name in sample["cam_dets"]:
            det = sample["cam_dets"].get(cam_name) or {}
            px = self._safe_float(det.get("pixel_x"))
            py = self._safe_float(det.get("pixel_y"))
            ax = self._safe_float(anchor_px[0]) if len(anchor_px) > 0 else None
            ay = self._safe_float(anchor_px[1]) if len(anchor_px) > 1 else None
            if px is not None and py is not None and ax is not None and ay is not None:
                dist = math.hypot(px - ax, py - ay)
                return {
                    "distance": dist,
                    "threshold": self._bottom_px_threshold(player) if bottom else self.top_hit_dist_px,
                    "unit": "px",
                }

        anchor = player.get("hit_anchor_court") or player.get("foot_court")
        if not anchor or len(anchor) < 2:
            return None
        ax = self._safe_float(anchor[0])
        ay = self._safe_float(anchor[1])
        if ax is None or ay is None:
            return None
        ball = sample["ball"]
        dist = math.hypot(ball["x"] - ax, ball["y"] - ay)
        return {
            "distance": dist,
            "threshold": self._bottom_m_threshold(player) if bottom else self.top_hit_dist_m,
            "unit": "m",
        }

    def _bottom_px_threshold(self, player: dict) -> float:
        foot = player.get("foot_court") or [0.0, 0.0]
        ry = abs(float(foot[1] or 0.0)) if len(foot) > 1 else 0.0
        ratio = min(max(ry, 0.0), COURT_Y_MAX) / max(COURT_Y_MAX, 1e-6)
        return self.bottom_hit_dist_px_net + (
            self.bottom_hit_dist_px_base - self.bottom_hit_dist_px_net
        ) * ratio

    def _bottom_m_threshold(self, player: dict) -> float:
        foot = player.get("foot_court") or [0.0, 0.0]
        ry = abs(float(foot[1] or 0.0)) if len(foot) > 1 else 0.0
        ratio = min(max(ry, 0.0), COURT_Y_MAX) / max(COURT_Y_MAX, 1e-6)
        return self.bottom_hit_dist_m_net + (
            self.bottom_hit_dist_m_base - self.bottom_hit_dist_m_net
        ) * ratio

    @staticmethod
    def _player_on_side(player: dict, side: str) -> bool:
        foot = player.get("foot_court")
        if not foot or len(foot) < 2 or foot[1] is None:
            return False
        if abs(float(foot[0])) > SINGLES_X_MAX + 2.0:
            return False
        return float(foot[1]) < 0 if side == "near" else float(foot[1]) > 0

    def _nearby_samples(self, frame: int, *, radius: int) -> list[dict]:
        return [
            sample for sample in self._history
            if abs(sample["frame_index"] - frame) <= radius
        ]

    def _latest_frame(self) -> int | None:
        if self._history:
            return self._history[-1]["frame_index"]
        return None

    def _event_frame(self, event: dict) -> int:
        frame = event.get("frame_index", event.get("frame"))
        if frame is None:
            self._synthetic_frame += 1
            return self._synthetic_frame
        return int(frame)

    @staticmethod
    def _angle_between(v1: tuple[float, float], v2: tuple[float, float]) -> float:
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        if n1 <= 1e-9 or n2 <= 1e-9:
            return 0.0
        cos_theta = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def _safe_float(value) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
