"""Cross-camera blob pairing via 3D triangulation with temporal tracking.

Given multi-blob candidates from two cameras, finds the pair whose
triangulated 3D point has the lowest composite score combining ray_distance
(geometric agreement) and temporal 3D distance (continuity with previous
detections).

When tracking is lost (no detection for >lost_timeout frames), falls back
to ray_distance-only scoring, allowing new rally starts to be accepted.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _triangulate_with_distance(
    world_2d_cam1: tuple[float, float],
    world_2d_cam2: tuple[float, float],
    cam_pos_1: list[float],
    cam_pos_2: list[float],
) -> tuple[float, float, float, float]:
    """Triangulate and return (x, y, z, ray_distance).

    ray_distance is ||p1 - p2|| where p1, p2 are the closest points
    on the two camera rays. Smaller = more consistent observation.
    """
    cam1 = np.asarray(cam_pos_1, dtype=np.float64)
    cam2 = np.asarray(cam_pos_2, dtype=np.float64)
    ground1 = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    ground2 = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])

    d1 = ground1 - cam1
    d2 = ground2 - cam2

    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        mid_ground = (ground1 + ground2) / 2.0
        return float(mid_ground[0]), float(mid_ground[1]), 0.0, 999.0

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    p1_fixed = cam1 + s * d1
    t = float(np.dot(p1_fixed - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)

    p2_fixed = cam2 + t * d2
    s = float(np.dot(p2_fixed - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0

    ray_dist = float(np.linalg.norm(p1 - p2))

    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2]), ray_dist


class MultiBlobMatcher:
    """Match blob candidates across two cameras using 3D triangulation.

    For each frame, tries all (cam1_blob, cam2_blob) pairs, triangulates,
    and picks the pair with the lowest composite score:

        score = ray_distance + temporal_weight * dist_to_predicted_3d

    When no recent tracking history exists (LOST state), temporal_weight
    is effectively 0, so only ray_distance matters. This allows new rallies
    (serves) to be detected even though the ball is far from the last
    tracked position.

    State machine:
        TRACKING: Have recent 3D detection (gap < lost_timeout)
            → Use composite score with velocity-based prediction
        LOST: No detection for >= lost_timeout frames
            → Use ray_distance only (current behavior)
            → First accepted detection transitions back to TRACKING
    """

    def __init__(
        self,
        cam1_pos: list[float],
        cam2_pos: list[float],
        max_ray_distance: float = 2.0,
        valid_z_range: tuple[float, float] = (0.0, 6.0),
        temporal_weight: float = 0.3,
        lost_timeout: int = 30,
        max_velocity: float = 50.0,
        history_size: int = 5,
        fps: float = 30.0,
    ):
        self.cam1_pos = cam1_pos
        self.cam2_pos = cam2_pos
        self.max_ray_distance = max_ray_distance
        self.z_min, self.z_max = valid_z_range

        # Temporal tracking parameters
        self.temporal_weight = temporal_weight
        self.lost_timeout = lost_timeout
        self.max_velocity = max_velocity
        self.history_size = history_size
        self.fps = fps

        # Temporal state
        self._history: list[dict] = []  # [{pos: np.array, frame_index: int}]
        self._velocity: Optional[np.ndarray] = None

        # Stats
        self.total_frames = 0
        self.matched_frames = 0
        self.non_top1_picks = 0
        self.temporal_assists = 0  # times temporal score changed the pick

    def _predict(self, frame_index: int) -> Optional[np.ndarray]:
        """Predict 3D position for the given frame based on history.

        Returns None if in LOST state (no recent history).
        """
        if not self._history:
            return None

        last = self._history[-1]
        gap = frame_index - last["frame_index"]

        # LOST state: gap too large, no prediction
        if gap > self.lost_timeout:
            return None

        # Use velocity-based prediction if available
        if self._velocity is not None and gap > 0:
            dt = gap / self.fps
            predicted = last["pos"] + self._velocity * dt
            return predicted

        # No velocity yet, use last known position
        return last["pos"].copy()

    def _update_history(self, x: float, y: float, z: float, frame_index: int) -> None:
        """Update tracking history with new 3D detection."""
        pos = np.array([x, y, z], dtype=np.float64)

        # Compute velocity from last two points
        if self._history:
            last = self._history[-1]
            dt = (frame_index - last["frame_index"]) / self.fps
            if dt > 0:
                vel = (pos - last["pos"]) / dt
                speed = float(np.linalg.norm(vel))
                if speed <= self.max_velocity:
                    self._velocity = vel
                # If speed exceeds max, keep old velocity (outlier protection)
            # If same frame, don't update velocity

        self._history.append({"pos": pos, "frame_index": frame_index})

        # Trim history
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]

    def reset(self) -> None:
        """Reset temporal state (e.g. at start of new video)."""
        self._history.clear()
        self._velocity = None

    def match(
        self,
        det1: dict,
        det2: dict,
    ) -> Optional[dict]:
        """Find the best blob pair across two cameras for one frame.

        Uses composite scoring: ray_distance + temporal_weight * 3d_distance.
        When in LOST state (no recent tracking), temporal term is zero.

        Args:
            det1: Detection dict from camera 1, must have 'candidates' list.
            det2: Detection dict from camera 2, same format.

        Returns:
            Dict with 3D point and matched blob info, or None if no valid pair.
        """
        cands1 = det1.get("candidates", [])
        cands2 = det2.get("candidates", [])
        if not cands1 or not cands2:
            return None

        self.total_frames += 1
        frame_index = det1.get("frame_index", 0)

        # Get predicted position (None if LOST)
        predicted = self._predict(frame_index) if self.temporal_weight > 0 else None

        best = None
        best_score = float("inf")
        best_ray_only = None  # track what ray-only would have picked
        best_ray_dist_only = float("inf")

        for i, c1 in enumerate(cands1):
            for j, c2 in enumerate(cands2):
                x, y, z, ray_dist = _triangulate_with_distance(
                    (c1["world_x"], c1["world_y"]),
                    (c2["world_x"], c2["world_y"]),
                    self.cam1_pos,
                    self.cam2_pos,
                )

                # Hard filters: ray_distance and z must be reasonable
                if ray_dist > self.max_ray_distance:
                    continue
                if z < self.z_min or z > self.z_max:
                    continue

                # Composite score
                score = ray_dist
                if predicted is not None:
                    d3d = float(np.linalg.norm(
                        np.array([x, y, z]) - predicted
                    ))
                    score += self.temporal_weight * d3d

                # Track ray-only best (for stats)
                if ray_dist < best_ray_dist_only:
                    best_ray_dist_only = ray_dist
                    best_ray_only = (i, j)

                if score < best_score:
                    best_score = score
                    best = {
                        "x": x,
                        "y": y,
                        "z": z,
                        "ray_distance": ray_dist,
                        "score": score,
                        "frame_index": frame_index,
                        "cam1_idx": i,
                        "cam2_idx": j,
                        "cam1_pixel": [c1["pixel_x"], c1["pixel_y"]],
                        "cam2_pixel": [c2["pixel_x"], c2["pixel_y"]],
                        "cam1_world": [c1["world_x"], c1["world_y"]],
                        "cam2_world": [c2["world_x"], c2["world_y"]],
                        "cam1_blob_sum": c1["blob_sum"],
                        "cam2_blob_sum": c2["blob_sum"],
                        "tracking_state": "tracking" if predicted is not None else "lost",
                    }

        if best is not None:
            self.matched_frames += 1
            if best["cam1_idx"] != 0 or best["cam2_idx"] != 0:
                self.non_top1_picks += 1
            # Check if temporal scoring changed the pick
            if best_ray_only and (best["cam1_idx"], best["cam2_idx"]) != best_ray_only:
                self.temporal_assists += 1
            # Update tracking history
            self._update_history(best["x"], best["y"], best["z"], frame_index)

        return best

    def get_stats(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "matched_frames": self.matched_frames,
            "non_top1_picks": self.non_top1_picks,
            "non_top1_rate": (
                self.non_top1_picks / self.matched_frames
                if self.matched_frames > 0
                else 0.0
            ),
            "temporal_assists": self.temporal_assists,
            "temporal_assist_rate": (
                self.temporal_assists / self.matched_frames
                if self.matched_frames > 0
                else 0.0
            ),
        }
