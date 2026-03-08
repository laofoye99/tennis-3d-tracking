"""Real-time bounce detection and rally tracking for tennis 3D ball tracking.

Provides streaming analytics that operate on individual 3D points as they arrive:

    - BounceDetector: sliding-window V-shape trajectory fitting
    - RallyTracker:   net-crossing state machine (idle ↔ rally)
    - run_batch_analytics(): convenience wrapper for video-test batch mode

Both classes are designed for <500ms detection latency in live streaming mode
and also work in batch mode by feeding points sequentially.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Court dimensions (meters)
COURT_X = 8.23
COURT_Y = 23.77
NET_Y = 11.885
NET_HEIGHT = 0.914


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BounceEvent:
    """A detected ball bounce."""

    x: float
    y: float
    z: float
    timestamp: float
    in_court: bool
    frame_index: Optional[int] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "timestamp": round(self.timestamp, 4),
            "in_court": self.in_court,
            "frame_index": self.frame_index,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class RallyState:
    """Current rally tracking state."""

    state: str = "idle"  # idle | rally
    rally_id: int = 0
    stroke_count: int = 0
    last_side: Optional[str] = None  # "near" (y < NET_Y) or "far"
    last_crossing_time: float = 0.0
    rally_start_time: float = 0.0
    bounces: list = field(default_factory=list)

    def to_dict(self) -> dict:
        now = time.time()
        return {
            "state": self.state,
            "rally_id": self.rally_id,
            "stroke_count": self.stroke_count,
            "last_side": self.last_side,
            "rally_duration": (
                round(now - self.rally_start_time, 1) if self.state == "rally" else 0
            ),
            "bounces": [b.to_dict() if hasattr(b, "to_dict") else b for b in self.bounces],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_in_court(x: float, y: float) -> bool:
    """Check if (x, y) falls within the singles court boundaries."""
    return 0 <= x <= COURT_X and 0 <= y <= COURT_Y


# ---------------------------------------------------------------------------
# BounceDetector
# ---------------------------------------------------------------------------


class BounceDetector:
    """Streaming bounce detector using V-shape trajectory fitting.

    Maintains a sliding window of recent 3D points.  A bounce is detected
    by fitting a V-shape (two line segments meeting at a vertex) to the
    Z vs time profile and comparing against a single-line null model.

    The V-shape must show:
      - Significant improvement over a single line (improvement ratio > threshold)
      - Vertex near ground level (Z < threshold)
      - Left slope negative (descending), right slope positive (ascending)

    Usage::

        detector = BounceDetector()
        for point in stream_of_3d_points:
            bounce = detector.update(point)
            if bounce is not None:
                print(f"Bounce at ({bounce.x}, {bounce.y}), "
                      f"confidence={bounce.confidence:.2f}")
    """

    def __init__(
        self,
        window_size: int = 10,
        z_ground_threshold: float = 0.5,
        min_descent_speed: float = 0.5,
        cooldown_seconds: float = 0.3,
        min_improvement_ratio: float = 0.3,
    ):
        self.window_size = window_size
        self.z_ground_threshold = z_ground_threshold
        self.min_descent_speed = min_descent_speed
        self.cooldown_seconds = cooldown_seconds
        self.min_improvement_ratio = min_improvement_ratio

        self._window: deque = deque(maxlen=window_size)
        self._last_bounce_time: float = 0.0
        self._all_bounces: list[BounceEvent] = []

    # ---- public API ----

    def update(self, point: dict) -> Optional[BounceEvent]:
        """Process a new 3D point and return BounceEvent if bounce detected.

        Fits a V-shape to the Z-time profile in the sliding window and
        compares against a single-line fit.  Returns a BounceEvent when the
        V-shape is a significantly better fit with the vertex near ground.

        Args:
            point: dict with keys ``x``, ``y``, ``z`` and optionally
                   ``timestamp``, ``frame_index`` (or ``frame_a``).
        """
        self._window.append(point)

        # Need at least 6 points for meaningful V-shape (2 left + 1 vertex + 2 right + margin)
        if len(self._window) < 6:
            return None

        now = point.get("timestamp", time.time())
        if now - self._last_bounce_time < self.cooldown_seconds:
            return None

        pts = list(self._window)
        n = len(pts)

        # Extract time and Z arrays
        times = np.array(
            [p.get("timestamp", i * 0.04) for i, p in enumerate(pts)],
            dtype=np.float64,
        )
        zs = np.array([p["z"] for p in pts], dtype=np.float64)

        # Fit V-shape vs null model
        best_k, left_slope, right_slope, v_ssr, null_ssr = self._fit_v_shape(
            times, zs
        )

        if best_k < 0:
            return None

        # Avoid division by zero for perfectly linear data
        if null_ssr < 1e-10:
            return None

        # Compute improvement ratio
        improvement = 1.0 - v_ssr / null_ssr

        # Decision criteria
        vertex_z = pts[best_k]["z"]
        is_bounce = (
            improvement > self.min_improvement_ratio
            and vertex_z < self.z_ground_threshold
            and left_slope < 0   # was descending
            and right_slope > 0  # now ascending
        )

        if not is_bounce:
            return None

        ground_pt = pts[best_k]
        bounce = BounceEvent(
            x=ground_pt["x"],
            y=ground_pt["y"],
            z=ground_pt["z"],
            timestamp=now,
            in_court=_is_in_court(ground_pt["x"], ground_pt["y"]),
            frame_index=ground_pt.get("frame_index") or ground_pt.get("frame_a"),
            confidence=min(1.0, improvement),
        )
        self._last_bounce_time = now
        self._all_bounces.append(bounce)
        # Clear window after detection to prevent re-detecting the same V-shape
        # as the window slides.  Keep only points after the vertex so the next
        # arc can start building immediately.
        remaining = pts[best_k + 1 :]
        self._window.clear()
        for p in remaining:
            self._window.append(p)
        return bounce

    def get_all_bounces(self) -> list[BounceEvent]:
        return list(self._all_bounces)

    def reset(self) -> None:
        self._window.clear()
        self._last_bounce_time = 0.0
        self._all_bounces.clear()

    # ---- internals ----

    @staticmethod
    def _fit_line(
        t_arr: np.ndarray, z_arr: np.ndarray
    ) -> tuple[float, float, float]:
        """Fit z = slope * t + intercept via least squares.

        Returns:
            (slope, intercept, sum_of_squared_residuals)
        """
        n = len(t_arr)
        if n < 2:
            return 0.0, float(z_arr[0]) if n == 1 else 0.0, 0.0
        A = np.column_stack([t_arr, np.ones(n)])
        sol, _, _, _ = np.linalg.lstsq(A, z_arr, rcond=None)
        slope, intercept = sol
        fitted = A @ sol
        ssr = float(np.sum((z_arr - fitted) ** 2))
        return float(slope), float(intercept), ssr

    @staticmethod
    def _fit_v_shape(
        t_arr: np.ndarray, z_arr: np.ndarray
    ) -> tuple[int, float, float, float, float]:
        """Find best V-shape fit by trying each interior point as vertex.

        For each candidate vertex k (index 2 to n-3), fits two separate
        line segments (left: points[0..k], right: points[k..n-1]) and
        computes total sum of squared residuals.  The best vertex is the
        one minimising total residual.

        Returns:
            (best_vertex_idx, left_slope, right_slope, v_ssr, null_ssr)
            Returns (-1, ...) if no valid vertex found.
        """
        n = len(t_arr)

        # Null model: single line through all points
        _, _, null_ssr = BounceDetector._fit_line(t_arr, z_arr)

        best_k = -1
        best_ssr = float("inf")
        best_left_slope = 0.0
        best_right_slope = 0.0

        # Need at least 2 points on each side of vertex
        for k in range(2, n - 2):
            # Left segment: points[0..k] inclusive
            sl, _, ssr_l = BounceDetector._fit_line(t_arr[: k + 1], z_arr[: k + 1])
            # Right segment: points[k..n-1] inclusive (vertex shared)
            sr, _, ssr_r = BounceDetector._fit_line(t_arr[k:], z_arr[k:])

            total_ssr = ssr_l + ssr_r
            if total_ssr < best_ssr:
                best_ssr = total_ssr
                best_k = k
                best_left_slope = sl
                best_right_slope = sr

        return best_k, best_left_slope, best_right_slope, best_ssr, null_ssr

    @staticmethod
    def _z_velocity(segment: list[dict]) -> float:
        """Average Z-velocity over a segment of points (legacy helper)."""
        if len(segment) < 2:
            return 0.0
        dt = segment[-1].get("timestamp", 0) - segment[0].get("timestamp", 0)
        if dt <= 0:
            dt = max(1, len(segment) - 1) * 0.04  # ~25 fps fallback
        return (segment[-1]["z"] - segment[0]["z"]) / dt


# ---------------------------------------------------------------------------
# RallyTracker
# ---------------------------------------------------------------------------


class RallyTracker:
    """State machine for rally tracking based on net crossings.

    Tracks when the ball crosses the net line (``Y = 11.885 m``) and
    maintains rally state:

        ``idle`` → ``rally`` (on first net crossing) → ``idle`` (on timeout)

    Usage::

        tracker = RallyTracker()
        for point in stream_of_3d_points:
            tracker.update(point)
        state = tracker.get_state()
    """

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout_seconds = timeout_seconds

        self._state = RallyState()
        self._last_point_time: float = 0.0
        self._prev_y: Optional[float] = None
        self._completed_rallies: list[dict] = []

    # ---- public API ----

    def update(self, point: dict, bounce: Optional[BounceEvent] = None) -> None:
        """Process a new 3D point and optional bounce event."""
        now = point.get("timestamp", time.time())
        y = point["y"]

        # Check for rally timeout
        if (
            self._state.state == "rally"
            and self._last_point_time > 0
            and now - self._last_point_time > self.timeout_seconds
        ):
            self._end_rally(self._last_point_time)

        self._last_point_time = now
        current_side = "near" if y < NET_Y else "far"

        # Detect net crossing
        if self._prev_y is not None:
            crossed = (self._prev_y < NET_Y <= y) or (self._prev_y >= NET_Y > y)
            if crossed:
                if self._state.state == "idle":
                    self._start_rally(now)
                self._state.stroke_count += 1
                self._state.last_side = current_side
                self._state.last_crossing_time = now

        self._prev_y = y

        # Register bounce
        if bounce is not None:
            self._state.bounces.append(bounce)

    def get_state(self) -> RallyState:
        """Return current rally state (auto-checks timeout)."""
        if (
            self._state.state == "rally"
            and self._last_point_time > 0
            and time.time() - self._last_point_time > self.timeout_seconds
        ):
            self._end_rally(self._last_point_time)
        return self._state

    def get_completed_rallies(self) -> list[dict]:
        return list(self._completed_rallies)

    def reset(self) -> None:
        self._state = RallyState()
        self._prev_y = None
        self._last_point_time = 0.0
        self._completed_rallies.clear()

    # ---- internals ----

    def _start_rally(self, timestamp: float) -> None:
        self._state.state = "rally"
        self._state.rally_id += 1
        self._state.stroke_count = 0
        self._state.rally_start_time = timestamp
        self._state.bounces = []
        logger.info("Rally %d started (t=%.2f)", self._state.rally_id, timestamp)

    def _end_rally(self, timestamp: float) -> None:
        duration = timestamp - self._state.rally_start_time
        summary = {
            "rally_id": self._state.rally_id,
            "stroke_count": self._state.stroke_count,
            "bounce_count": len(self._state.bounces),
            "duration": round(duration, 1),
            "bounces": [
                b.to_dict() if hasattr(b, "to_dict") else b
                for b in self._state.bounces
            ],
        }
        self._completed_rallies.append(summary)
        logger.info(
            "Rally %d ended: %d strokes, %d bounces, %.1fs",
            self._state.rally_id,
            self._state.stroke_count,
            len(self._state.bounces),
            duration,
        )
        self._state.state = "idle"
        self._state.stroke_count = 0
        self._state.bounces = []


# ---------------------------------------------------------------------------
# Batch helper (for video-test mode)
# ---------------------------------------------------------------------------


def run_batch_analytics(points_3d: list[dict]) -> dict:
    """Run bounce detection and rally tracking on a batch of 3D points.

    Used by the video-test path after ``compute_3d_trajectory()`` to add
    streaming-style analytics to the trajectory result.

    Args:
        points_3d: sorted list of 3D point dicts with keys
                   ``x``, ``y``, ``z``, ``t`` (relative time).

    Returns:
        Dict with ``bounces``, ``rally_state``, and ``completed_rallies``.
    """
    detector = BounceDetector(
        window_size=10,
        z_ground_threshold=0.5,
        cooldown_seconds=0.2,
        min_improvement_ratio=0.3,
    )
    tracker = RallyTracker(timeout_seconds=2.0)

    bounces: list[dict] = []
    for pt in points_3d:
        # Ensure timestamp exists (use t field in batch mode)
        if "timestamp" not in pt and "t" in pt:
            pt = {**pt, "timestamp": pt["t"]}

        bounce = detector.update(pt)
        tracker.update(pt, bounce)
        if bounce is not None:
            bounces.append(bounce.to_dict())

    # Force-end any open rally
    final_state = tracker.get_state()

    return {
        "bounces": bounces,
        "rally_state": final_state.to_dict(),
        "completed_rallies": tracker.get_completed_rallies(),
    }
