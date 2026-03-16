"""Real-time bounce detection and rally tracking for tennis 3D ball tracking.

Provides streaming analytics that operate on individual 3D points as they arrive:

    - BounceDetector: sliding-window V-shape trajectory fitting (legacy)
    - RallyTracker:   net-crossing state machine (legacy)
    - EnhancedBounceDetector: Z-inflection + single-camera homography landing
    - RallyStateMachine: event-driven state machine with full end-reason detection
    - FusionCoordinator: adaptive 3D / single-camera middleware with find_peaks
    - run_batch_analytics(): convenience wrapper for video-test batch mode

All classes are designed for <500ms detection latency in live streaming mode
and also work in batch mode by feeding points sequentially.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Court dimensions (meters)
COURT_X = 8.23
COURT_Y = 23.77
NET_Y = 11.885
NET_HEIGHT = 0.914

# Court boundaries with margin for in/out calls
COURT_X_MIN = 0.0
COURT_X_MAX = COURT_X
COURT_Y_MIN = 0.0
COURT_Y_MAX = COURT_Y

# Service box boundaries
SERVICE_LINE_NEAR = 5.485
SERVICE_LINE_FAR = 18.285

# Baseline zones for serve detection
BASELINE_NEAR_MAX = 5.0   # y < 5m = near baseline zone
BASELINE_FAR_MIN = 18.77  # y > 18.77m = far baseline zone


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
    source_camera: str = "3d"  # "cam66" | "cam68" | "interpolated" | "3d"
    side: str = ""  # "near" | "far"

    def __post_init__(self):
        if not self.side:
            self.side = "near" if self.y < NET_Y else "far"

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "timestamp": round(self.timestamp, 4),
            "in_court": self.in_court,
            "frame_index": self.frame_index,
            "confidence": round(self.confidence, 3),
            "source_camera": self.source_camera,
            "side": self.side,
        }


class RallyEndReason(str, Enum):
    """Reason a rally ended."""
    OUT = "out"
    NET = "net"
    DOUBLE_BOUNCE = "double_bounce"
    TIMEOUT = "timeout"


@dataclass
class RallyResult:
    """Complete result of a finished rally."""
    rally_id: int = 0
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    stroke_count: int = 0
    bounces: list = field(default_factory=list)
    end_reason: str = "timeout"
    end_side: str = ""      # side where point was won ("near" | "far")
    server_side: str = ""   # side that served ("near" | "far")

    def to_dict(self) -> dict:
        return {
            "rally_id": self.rally_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_seconds": round(self.duration_seconds, 2),
            "stroke_count": self.stroke_count,
            "bounces": [
                b.to_dict() if hasattr(b, "to_dict") else b
                for b in self.bounces
            ],
            "end_reason": self.end_reason,
            "end_side": self.end_side,
            "server_side": self.server_side,
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
# EnhancedBounceDetector
# ---------------------------------------------------------------------------


class EnhancedBounceDetector:
    """Enhanced bounce detector: Z-inflection + single-camera homography landing.

    Detects bounces by finding Z descend→ascend inflection points.
    For landing coordinates, prefers single-camera homography (accurate on
    ground plane) over 3D triangulation.

    When the bounce frame itself is missing, interpolates landing position
    from surrounding frames.

    Usage::

        detector = EnhancedBounceDetector()
        for point_3d, cam_dets in data_stream:
            bounce = detector.update(point_3d, cam_dets)
            if bounce:
                print(f"Bounce at ({bounce.x:.2f}, {bounce.y:.2f})")
    """

    def __init__(
        self,
        window_size: int = 8,
        z_ground_threshold: float = 0.8,
        cooldown_frames: int = 5,
        min_descent_points: int = 2,
        min_ascent_points: int = 1,
    ):
        self.window_size = window_size
        self.z_ground_threshold = z_ground_threshold
        self.cooldown_frames = cooldown_frames
        self.min_descent_points = min_descent_points
        self.min_ascent_points = min_ascent_points

        self._window: deque = deque(maxlen=window_size)
        self._cam_window: deque = deque(maxlen=window_size)
        self._last_bounce_frame: int = -100
        self._all_bounces: list[BounceEvent] = []

    def update(
        self,
        point_3d: dict,
        cam_detections: Optional[dict] = None,
    ) -> Optional[BounceEvent]:
        """Process a new 3D point and optional per-camera detections.

        Args:
            point_3d: dict with x, y, z, frame_index, timestamp, ray_dist.
            cam_detections: optional dict mapping camera name to detection dict
                with keys pixel_x, pixel_y, world_x, world_y, yolo_conf.

        Returns:
            BounceEvent if a bounce was detected, else None.
        """
        self._window.append(point_3d)
        self._cam_window.append(cam_detections or {})

        fi = point_3d.get("frame_index", 0)
        if fi - self._last_bounce_frame < self.cooldown_frames:
            return None

        if len(self._window) < 4:
            return None

        pts = list(self._window)
        n = len(pts)

        # Look for Z inflection: descending then ascending
        # Check if there's a local minimum in Z within the window
        zs = [p["z"] for p in pts]

        best_k = -1
        best_z = float("inf")

        for k in range(self.min_descent_points, n - self.min_ascent_points):
            # Count descending points before k
            descent_ok = all(
                zs[i] >= zs[i + 1] - 0.15  # allow small noise tolerance
                for i in range(max(0, k - self.min_descent_points), k)
            )
            # Count ascending points after k
            ascent_ok = all(
                zs[i] <= zs[i + 1] + 0.15
                for i in range(k, min(n - 1, k + self.min_ascent_points))
            )

            if descent_ok and ascent_ok and zs[k] < best_z:
                best_z = zs[k]
                best_k = k

        if best_k < 0 or best_z > self.z_ground_threshold:
            return None

        # Additional check: the minimum should be notably lower than ends
        z_left = zs[max(0, best_k - 2)]
        z_right = zs[min(n - 1, best_k + 1)]
        if best_z > min(z_left, z_right) - 0.1:
            return None

        # Found a bounce! Determine landing coordinates
        vertex_pt = pts[best_k]
        cam_dets = list(self._cam_window)[best_k]

        landing_x, landing_y, source = self._get_landing_coords(
            vertex_pt, cam_dets, pts, best_k
        )

        bounce = BounceEvent(
            x=landing_x,
            y=landing_y,
            z=best_z,
            timestamp=vertex_pt.get("timestamp", 0.0),
            in_court=_is_in_court(landing_x, landing_y),
            frame_index=vertex_pt.get("frame_index"),
            confidence=self._compute_confidence(pts, best_k, zs),
            source_camera=source,
        )

        self._last_bounce_frame = fi
        self._all_bounces.append(bounce)

        # Keep only post-vertex points
        remaining = pts[best_k + 1:]
        cam_remaining = list(self._cam_window)[best_k + 1:]
        self._window.clear()
        self._cam_window.clear()
        for p, c in zip(remaining, cam_remaining):
            self._window.append(p)
            self._cam_window.append(c)

        return bounce

    def get_all_bounces(self) -> list[BounceEvent]:
        return list(self._all_bounces)

    def reset(self) -> None:
        self._window.clear()
        self._cam_window.clear()
        self._last_bounce_frame = -100
        self._all_bounces.clear()

    def _get_landing_coords(
        self,
        vertex_pt: dict,
        cam_dets: dict,
        pts: list[dict],
        k: int,
    ) -> tuple[float, float, str]:
        """Determine landing coordinates with priority:
        1. Single-camera homography (ground plane accurate)
        2. Interpolation from surrounding frames
        3. 3D triangulation point (fallback)
        """
        # Priority 1: single camera homography at vertex frame
        if cam_dets:
            best_cam = None
            best_conf = -1.0
            for cam_name, det in cam_dets.items():
                if det and det.get("world_x") is not None:
                    conf = det.get("yolo_conf", 0.5)
                    if conf > best_conf:
                        best_conf = conf
                        best_cam = cam_name

            if best_cam is not None:
                det = cam_dets[best_cam]
                return det["world_x"], det["world_y"], best_cam

        # Priority 2: interpolation from neighboring frames' camera data
        cam_list = list(self._cam_window)
        nearby_wx, nearby_wy = [], []
        for i in range(max(0, k - 2), min(len(cam_list), k + 3)):
            if i == k:
                continue
            for det in cam_list[i].values():
                if det and det.get("world_x") is not None:
                    nearby_wx.append(det["world_x"])
                    nearby_wy.append(det["world_y"])

        if nearby_wx:
            return float(np.mean(nearby_wx)), float(np.mean(nearby_wy)), "interpolated"

        # Priority 3: 3D point (fallback)
        return vertex_pt["x"], vertex_pt["y"], "3d"

    @staticmethod
    def _compute_confidence(pts: list[dict], k: int, zs: list[float]) -> float:
        """Bounce confidence based on Z-dip depth and surrounding density."""
        n = len(pts)
        # Depth of dip relative to neighbors
        z_before = zs[max(0, k - 2)]
        z_after = zs[min(n - 1, k + 2)]
        dip = min(z_before, z_after) - zs[k]
        depth_score = min(1.0, dip / 1.0)  # 1m dip = full confidence

        # Density: how many points are near the vertex
        density_score = min(1.0, n / 6.0)

        return round(min(1.0, depth_score * 0.7 + density_score * 0.3), 3)


# ---------------------------------------------------------------------------
# RallyStateMachine
# ---------------------------------------------------------------------------


class RallyStateMachine:
    """Event-driven state machine for rally analysis.

    Detects rally start (serve), net crossings (strokes), bounces (landing
    points), and rally end with full reason classification:
    out, net, double_bounce, timeout.

    State transitions::

        IDLE ──[detection after gap + baseline zone]──→ SERVING
        IDLE ──[detection after gap + non-baseline]──→ RALLY

        SERVING ──[net crossing]──→ RALLY
        SERVING ──[timeout 3s]──→ IDLE

        RALLY ──[net crossing]──→ RALLY (stroke_count++)
        RALLY ──[bounce out of court]──→ POINT_END (out)
        RALLY ──[double bounce same side]──→ POINT_END (double_bounce)
        RALLY ──[net hit: Y≈NET, Z drops]──→ POINT_END (net)
        RALLY ──[timeout 3s]──→ POINT_END (timeout)

        POINT_END ──[auto]──→ IDLE
    """

    class State(str, Enum):
        IDLE = "idle"
        SERVING = "serving"
        RALLY = "rally"

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        gap_seconds: float = 2.0,
        fps: float = 25.0,
        net_hit_z_drop: float = 0.3,
        net_y_tolerance: float = 1.5,
    ):
        self.timeout_seconds = timeout_seconds
        self.gap_seconds = gap_seconds
        self.fps = fps
        self.net_hit_z_drop = net_hit_z_drop
        self.net_y_tolerance = net_y_tolerance

        self._state = self.State.IDLE
        self._rally_id = 0
        self._stroke_count = 0
        self._server_side = ""
        self._last_side = ""
        self._bounces: list[BounceEvent] = []
        self._last_bounce_side = ""
        self._bounce_count_since_cross = 0

        self._prev_y: Optional[float] = None
        self._prev_z: Optional[float] = None
        self._last_point_time: float = 0.0
        self._last_frame: int = 0
        self._rally_start_time: float = 0.0
        self._rally_start_frame: Optional[int] = None

        self._completed: list[RallyResult] = []

    def update(
        self,
        point_3d: dict,
        bounce: Optional[BounceEvent] = None,
    ) -> Optional[RallyResult]:
        """Process a 3D point and optional bounce. Returns RallyResult on rally end.

        Args:
            point_3d: dict with x, y, z, timestamp, frame_index.
            bounce: BounceEvent from EnhancedBounceDetector if detected this frame.

        Returns:
            RallyResult when a rally ends, else None.
        """
        now = point_3d.get("timestamp", time.time())
        y = point_3d["y"]
        z = point_3d["z"]
        fi = point_3d.get("frame_index", 0)
        current_side = "near" if y < NET_Y else "far"

        result = None

        # Check timeout
        time_gap = now - self._last_point_time if self._last_point_time > 0 else 0
        if self._state == self.State.RALLY and time_gap > self.timeout_seconds:
            result = self._end_rally(fi, now, RallyEndReason.TIMEOUT)
        elif self._state == self.State.SERVING and time_gap > self.timeout_seconds:
            self._state = self.State.IDLE

        # Handle state transitions
        if self._state == self.State.IDLE:
            if time_gap > self.gap_seconds or self._last_point_time == 0:
                # New activity after gap
                is_baseline = y < BASELINE_NEAR_MAX or y > BASELINE_FAR_MIN
                if is_baseline:
                    self._start_serving(now, fi, current_side)
                else:
                    self._start_rally(now, fi, "")

        # Detect net crossing
        if self._prev_y is not None:
            crossed = (self._prev_y < NET_Y <= y) or (self._prev_y >= NET_Y > y)
            if crossed:
                if self._state == self.State.SERVING:
                    self._state = self.State.RALLY
                    logger.info(
                        "Rally %d: serve crossed net (frame %d)", self._rally_id, fi
                    )
                if self._state == self.State.RALLY:
                    self._stroke_count += 1
                    self._bounce_count_since_cross = 0
                self._last_side = current_side

        # Detect net hit: Y near net and Z drops sharply
        # Require low ray_distance to avoid false triggers from misdetection
        ray_dist = point_3d.get("ray_dist", 0.0)
        if (
            self._state == self.State.RALLY
            and self._prev_z is not None
            and abs(y - NET_Y) < self.net_y_tolerance
            and self._prev_z - z > self.net_hit_z_drop
            and z < NET_HEIGHT
            and ray_dist < 1.5
        ):
            result = self._end_rally(fi, now, RallyEndReason.NET)

        # Handle bounce events
        if bounce is not None and self._state == self.State.RALLY:
            self._bounces.append(bounce)

            # Check out of court
            if not bounce.in_court:
                result = self._end_rally(fi, now, RallyEndReason.OUT)

            # Check double bounce (same side, no net crossing between)
            elif self._last_bounce_side == bounce.side:
                self._bounce_count_since_cross += 1
                if self._bounce_count_since_cross >= 2:
                    result = self._end_rally(fi, now, RallyEndReason.DOUBLE_BOUNCE)
            else:
                self._bounce_count_since_cross = 1

            self._last_bounce_side = bounce.side

        self._prev_y = y
        self._prev_z = z
        self._last_point_time = now
        self._last_frame = fi

        return result

    def get_state_dict(self) -> dict:
        """Return current state as a dict."""
        return {
            "state": self._state.value,
            "rally_id": self._rally_id,
            "stroke_count": self._stroke_count,
            "server_side": self._server_side,
            "bounce_count": len(self._bounces),
            "last_side": self._last_side,
        }

    def get_completed_rallies(self) -> list[RallyResult]:
        return list(self._completed)

    def reset(self) -> None:
        self._state = self.State.IDLE
        self._rally_id = 0
        self._stroke_count = 0
        self._bounces.clear()
        self._prev_y = None
        self._prev_z = None
        self._last_point_time = 0.0
        self._completed.clear()

    # ---- internals ----

    def _start_serving(self, now: float, fi: int, side: str) -> None:
        self._rally_id += 1
        self._state = self.State.SERVING
        self._stroke_count = 0
        self._server_side = side
        self._bounces = []
        self._bounce_count_since_cross = 0
        self._last_bounce_side = ""
        self._rally_start_time = now
        self._rally_start_frame = fi
        logger.info("Rally %d: serving from %s (frame %d)", self._rally_id, side, fi)

    def _start_rally(self, now: float, fi: int, server_side: str) -> None:
        self._rally_id += 1
        self._state = self.State.RALLY
        self._stroke_count = 0
        self._server_side = server_side
        self._bounces = []
        self._bounce_count_since_cross = 0
        self._last_bounce_side = ""
        self._rally_start_time = now
        self._rally_start_frame = fi
        logger.info("Rally %d: started (frame %d)", self._rally_id, fi)

    def _end_rally(
        self, fi: int, now: float, reason: RallyEndReason
    ) -> RallyResult:
        # Determine which side won the point
        if reason == RallyEndReason.OUT:
            # Ball went out: the side that hit it out loses
            end_side = self._last_side  # winner is the other side
        elif reason == RallyEndReason.NET:
            end_side = self._last_side
        elif reason == RallyEndReason.DOUBLE_BOUNCE:
            end_side = self._last_bounce_side  # side where double bounce = loser
        else:
            end_side = ""

        result = RallyResult(
            rally_id=self._rally_id,
            start_frame=self._rally_start_frame,
            end_frame=fi,
            start_time=self._rally_start_time,
            end_time=now,
            duration_seconds=now - self._rally_start_time,
            stroke_count=self._stroke_count,
            bounces=list(self._bounces),
            end_reason=reason.value,
            end_side=end_side,
            server_side=self._server_side,
        )
        self._completed.append(result)
        logger.info(
            "Rally %d ended: reason=%s, strokes=%d, bounces=%d, %.1fs",
            self._rally_id,
            reason.value,
            self._stroke_count,
            len(self._bounces),
            result.duration_seconds,
        )

        # Reset to idle
        self._state = self.State.IDLE
        self._stroke_count = 0
        self._bounces = []
        self._bounce_count_since_cross = 0
        self._last_bounce_side = ""

        return result


# ---------------------------------------------------------------------------
# FusionCoordinator — adaptive 3D / single-camera middleware
# ---------------------------------------------------------------------------


class FusionCoordinator:
    """Adaptive middleware that fuses 3D triangulation and single-camera detection.

    Dynamically selects the best detection mode per frame:
    - **3D mode**: both cameras detect + ray_distance below threshold →
      delegates to ``EnhancedBounceDetector`` for Z-inflection bounce detection.
    - **Single-camera mode**: only one camera available, or cameras disagree →
      uses ``scipy.signal.find_peaks`` on ``world_y`` to detect bounces.
    - **Gap**: no detection from either camera → skip frame.

    The coordinator owns the ``EnhancedBounceDetector`` and ``RallyStateMachine``
    instances internally.  Consumers call ``process_frame`` and receive a
    unified ``(point_3d, BounceEvent, RallyResult)`` tuple per frame.
    """

    def __init__(
        self,
        cam_positions: dict[str, list[float]],
        fps: float = 25.0,
        ray_dist_threshold: float = 2.0,
        single_cam_buffer_size: int = 20,
        single_cam_confidence: float = 0.6,
        gap_reset_frames: int = 10,
        bounce_cooldown_frames: int = 8,
        peak_prominence: float = 15.0,  # pixel_y prominence (pixels)
        peak_distance: int = 5,
    ):
        self._cam_positions = cam_positions
        self._fps = fps
        self._ray_dist_threshold = ray_dist_threshold
        self._single_cam_confidence = single_cam_confidence
        self._gap_reset_frames = gap_reset_frames
        self._bounce_cooldown_frames = bounce_cooldown_frames
        self._peak_prominence = peak_prominence
        self._peak_distance = peak_distance

        # Sub-components (owned, not shared)
        self._bounce_det = EnhancedBounceDetector(
            window_size=8,
            z_ground_threshold=0.25,
            cooldown_frames=5,
        )
        self._rally_sm = RallyStateMachine(
            timeout_seconds=3.0,
            gap_seconds=2.0,
            fps=fps,
        )

        # Per-camera world_y buffers for single-cam bounce detection
        self._cam_buffers: dict[str, deque] = {
            cam: deque(maxlen=single_cam_buffer_size)
            for cam in cam_positions
        }

        # Global bounce state (shared across 3D and single-cam)
        self._last_bounce_frame: int = -100
        self._single_cam_bounces: list[BounceEvent] = []

        # Mode tracking
        self._prev_mode: Optional[str] = None
        self._last_frame_index: int = -1
        self._mode_counts: dict[str, int] = {"3d": 0, "single_cam": 0, "gap": 0}

    # -- public API ----------------------------------------------------------

    def process_frame(
        self,
        frame_index: int,
        det66: Optional[dict],
        det68: Optional[dict],
    ) -> tuple[Optional[dict], Optional[BounceEvent], Optional[RallyResult]]:
        """Process one frame.  Returns ``(point_3d, bounce, rally_result)``."""
        from app.pipeline.multi_blob_matcher import _triangulate_with_distance

        # 1. Gap check — reset buffers on large gaps
        if self._last_frame_index >= 0:
            gap = frame_index - self._last_frame_index
            if gap > self._gap_reset_frames:
                self._reset_buffers()

        # 2. Always update per-camera buffers (pixel_y for bounce, world for position)
        #    Filter out large pixel_y jumps (misdetections / tracknet glitches)
        for cam_name, det in [("cam66", det66), ("cam68", det68)]:
            if det is not None and det.get("world_x") is not None:
                py = det.get("pixel_y", 0.0)
                buf = self._cam_buffers[cam_name]
                if len(buf) > 0:
                    prev_py = buf[-1]["pixel_y"]
                    prev_fi = buf[-1]["frame_index"]
                    # Allow larger jumps over larger frame gaps
                    max_jump = 150.0 * max(1, frame_index - prev_fi)
                    if abs(py - prev_py) > max_jump:
                        # Likely misdetection — skip this point
                        continue
                buf.append({
                    "frame_index": frame_index,
                    "pixel_y": py,
                    "world_x": det["world_x"],
                    "world_y": det["world_y"],
                })

        # 3. Mode selection
        has66 = det66 is not None and det66.get("world_x") is not None
        has68 = det68 is not None and det68.get("world_x") is not None

        point_3d: Optional[dict] = None
        bounce: Optional[BounceEvent] = None
        mode = "gap"

        if has66 and has68:
            x, y, z, rd = _triangulate_with_distance(
                (det66["world_x"], det66["world_y"]),
                (det68["world_x"], det68["world_y"]),
                self._cam_positions["cam66"],
                self._cam_positions["cam68"],
            )
            if rd < self._ray_dist_threshold:
                # === 3D MODE ===
                mode = "3d"
                point_3d = {
                    "x": x, "y": y, "z": z,
                    "frame_index": frame_index,
                    "timestamp": frame_index / self._fps,
                    "ray_dist": rd,
                }
                cam_dets = {"cam66": det66, "cam68": det68}

                # Mode transition guard: reset 3D detector when coming from
                # single-cam to avoid re-detecting the same bounce
                if self._prev_mode == "single_cam":
                    self._bounce_det.reset()

                bounce = self._bounce_det.update(point_3d, cam_dets)
            else:
                # Cameras disagree — pick higher yolo_conf, single-cam mode
                mode = "single_cam"
                cam_name, det = self._pick_better_camera(det66, det68)
                point_3d = self._make_single_cam_point(frame_index, det)
                if self._rally_sm.get_state_dict()["state"] != "idle":
                    bounce = self._detect_single_cam_bounce(frame_index, cam_name)

        elif has66:
            mode = "single_cam"
            point_3d = self._make_single_cam_point(frame_index, det66)
            if self._rally_sm.get_state_dict()["state"] != "idle":
                bounce = self._detect_single_cam_bounce(frame_index, "cam66")

        elif has68:
            mode = "single_cam"
            point_3d = self._make_single_cam_point(frame_index, det68)
            if self._rally_sm.get_state_dict()["state"] != "idle":
                bounce = self._detect_single_cam_bounce(frame_index, "cam68")

        # Update global bounce tracking
        if bounce is not None:
            self._last_bounce_frame = bounce.frame_index or frame_index
            if mode == "single_cam":
                self._single_cam_bounces.append(bounce)

        self._prev_mode = mode
        self._last_frame_index = frame_index
        self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

        # 4. Feed RallyStateMachine
        rally_result: Optional[RallyResult] = None
        if point_3d is not None:
            rally_result = self._rally_sm.update(point_3d, bounce)

        return point_3d, bounce, rally_result

    def get_rally_state(self) -> dict:
        """Current rally state dict (proxy to RallyStateMachine)."""
        return self._rally_sm.get_state_dict()

    def get_completed_rallies(self) -> list[RallyResult]:
        """All completed rally results."""
        return self._rally_sm.get_completed_rallies()

    def get_all_bounces(self) -> list[BounceEvent]:
        """All bounces from both 3D and single-cam detection."""
        return self._bounce_det.get_all_bounces() + self._single_cam_bounces

    def get_mode_counts(self) -> dict[str, int]:
        """Frame count per mode: 3d / single_cam / gap."""
        return dict(self._mode_counts)

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _pick_better_camera(
        det66: dict, det68: dict,
    ) -> tuple[str, dict]:
        """When cameras disagree, pick the one with higher YOLO confidence."""
        conf66 = det66.get("yolo_conf", 0.0) if det66 else 0.0
        conf68 = det68.get("yolo_conf", 0.0) if det68 else 0.0
        if conf66 >= conf68:
            return "cam66", det66
        return "cam68", det68

    def _make_single_cam_point(self, frame_index: int, det: dict) -> dict:
        """Build a point_3d from single-camera world coords.

        z=0 (ground-plane assumption); ray_dist=inf signals single-cam mode
        so that downstream checks (e.g. net-hit) are automatically skipped.
        """
        return {
            "x": det["world_x"],
            "y": det["world_y"],
            "z": 0.0,
            "frame_index": frame_index,
            "timestamp": frame_index / self._fps,
            "ray_dist": float("inf"),
        }

    def _detect_single_cam_bounce(
        self, frame_index: int, cam_name: str,
    ) -> Optional[BounceEvent]:
        """Detect bounce from single-camera pixel_y trajectory via find_peaks.

        Ball bounce = V-shape in pixel_y (descend → touch ground → rise).
        In image coordinates (y=0 at top), bounce = **local maximum** in
        pixel_y because the ball is at its lowest apparent position.

        Uses the world coordinates at the peak frame for landing position
        (homography is most accurate when ball is on the ground at bounce).

        Returns ``BounceEvent`` if a bounce was confirmed, else ``None``.
        """
        # Global cooldown
        if frame_index - self._last_bounce_frame < self._bounce_cooldown_frames:
            return None

        buf = self._get_contiguous_tail(cam_name)
        if len(buf) < 7:
            return None

        from scipy.signal import find_peaks  # lazy import

        pixel_ys = np.array([e["pixel_y"] for e in buf])
        n = len(buf)

        # Bounce = local maximum in pixel_y (ball at lowest point in image)
        peaks, props = find_peaks(
            pixel_ys,
            prominence=self._peak_prominence,
            distance=self._peak_distance,
        )

        # Accept peaks that have at least 1 point after for confirmation.
        # Skip peaks whose frame_index is within cooldown of last bounce.
        candidates: list[int] = []
        for idx in peaks:
            if idx <= n - 2:  # at least 1 point after for confirmation
                peak_fi = buf[idx]["frame_index"]
                if peak_fi - self._last_bounce_frame >= self._bounce_cooldown_frames:
                    candidates.append(idx)

        if not candidates:
            return None

        # Pick the most recent candidate
        idx = max(candidates)
        entry = buf[idx]
        side = "near" if entry["world_y"] < NET_Y else "far"

        bounce = BounceEvent(
            x=entry["world_x"],
            y=entry["world_y"],
            z=0.0,
            timestamp=entry["frame_index"] / self._fps,
            in_court=_is_in_court(entry["world_x"], entry["world_y"]),
            frame_index=entry["frame_index"],
            confidence=self._single_cam_confidence,
            source_camera=cam_name,
            side=side,
        )
        logger.info(
            "Single-cam bounce (pixel_y peak): frame=%d cam=%s side=%s "
            "pos=(%.2f,%.2f) pixel_y=%.0f conf=%.2f",
            entry["frame_index"], cam_name, side,
            entry["world_x"], entry["world_y"],
            entry["pixel_y"], self._single_cam_confidence,
        )
        return bounce

    def _get_contiguous_tail(
        self, cam_name: str, max_gap: int = 3,
    ) -> list[dict]:
        """Return the longest contiguous tail of a camera buffer (no gaps > max_gap)."""
        items = list(self._cam_buffers[cam_name])
        if len(items) < 2:
            return items
        # Walk backwards to find where a gap exceeds threshold
        cut = 0
        for i in range(len(items) - 1, 0, -1):
            if items[i]["frame_index"] - items[i - 1]["frame_index"] > max_gap:
                cut = i
                break
        return items[cut:]

    def _reset_buffers(self) -> None:
        """Clear all internal buffers on large frame gaps."""
        for buf in self._cam_buffers.values():
            buf.clear()
        self._bounce_det.reset()
        logger.debug("FusionCoordinator: buffers reset (large gap)")


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


def run_enhanced_batch_analytics(
    points_3d: list[dict],
    cam_detections_per_frame: Optional[dict] = None,
) -> dict:
    """Run enhanced bounce detection and rally analysis on a batch of 3D points.

    Args:
        points_3d: sorted list of 3D point dicts with keys
                   x, y, z, timestamp (or t), frame_index.
        cam_detections_per_frame: optional dict mapping frame_index to
                   per-camera detection dicts:
                   {frame_idx: {"cam66": {world_x, world_y, yolo_conf}, ...}}

    Returns:
        Dict with bounces, rallies, and state.
    """
    bounce_det = EnhancedBounceDetector(
        window_size=8,
        z_ground_threshold=0.8,
        cooldown_frames=5,
    )
    rally_sm = RallyStateMachine(
        timeout_seconds=3.0,
        gap_seconds=2.0,
    )

    bounces: list[dict] = []
    rallies: list[dict] = []

    for pt in points_3d:
        if "timestamp" not in pt and "t" in pt:
            pt = {**pt, "timestamp": pt["t"]}

        fi = pt.get("frame_index", 0)
        cam_dets = (
            cam_detections_per_frame.get(fi) if cam_detections_per_frame else None
        )

        bounce = bounce_det.update(pt, cam_dets)
        if bounce is not None:
            bounces.append(bounce.to_dict())

        rally_result = rally_sm.update(pt, bounce)
        if rally_result is not None:
            rallies.append(rally_result.to_dict())

    return {
        "bounces": bounces,
        "rallies": rallies,
        "rally_state": rally_sm.get_state_dict(),
        "completed_rallies": [r.to_dict() for r in rally_sm.get_completed_rallies()],
    }
