"""Ball detector (best.pt) + online bounce detector (angle + y-reversal + momentum)."""

import math
from collections import deque
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# BallDetector
# ---------------------------------------------------------------------------

class BallDetector:
    """YOLO-based tennis ball detector with static-object filtering."""

    def __init__(self, model_path: str, conf: float = 0.25,
                 device: str = "cuda", img_size: int = 960):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._model.to(device)
        self._conf = conf
        self._img_size = img_size
        self._track_history: dict[int, list] = {}
        self._static_threshold: float = 3.0   # px movement to consider "moving"
        self._static_limit: int = 3            # consecutive static frames → filter out

    def detect(self, frame: np.ndarray) -> Optional[tuple[float, float]]:
        results = self._model.track(
            frame, persist=True, conf=self._conf,
            imgsz=self._img_size, device=self._model.device, verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None or len(boxes) == 0:
            return None

        xywh = boxes.xywh.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()

        best = None
        for i in range(len(xywh)):
            tid = int(ids[i])
            cx, cy = float(xywh[i][0]), float(xywh[i][1])
            history = self._track_history.get(tid, [cx, cy, 0])
            dist = math.hypot(cx - history[0], cy - history[1])
            static_count = history[2] + 1 if dist < self._static_threshold else 0
            self._track_history[tid] = [cx, cy, static_count]
            if static_count >= self._static_limit:
                continue
            if best is None or dist > best[1]:
                best = ((cx, cy), dist)
        return best[0] if best else None


# ---------------------------------------------------------------------------
# BounceDetector (online port of main.py evaluate_bounces_fuzzy)
# ---------------------------------------------------------------------------

class BounceDetector:
    """Online sliding-window bounce detection in image pixel space.

    Algorithm (per frame, with lookahead delay):
      1. For frame (i - delay), compute angle between v_in and v_out,
         y-direction reversal, and momentum change.
      2. Candidate if: angle >= angle_thresh AND (y_reversal OR delta_v >= momentum_thresh).
      3. Cluster nearby candidates; emit strongest per cluster after NMS.
    """

    def __init__(self, window: int = 3, angle_thresh: float = 30.0,
                 momentum_thresh: float = 15.0, delay_frames: int = 6,
                 cluster_window: int = 8):
        self.window = window
        self.angle_thresh = angle_thresh
        self.momentum_thresh = momentum_thresh
        self.delay_frames = delay_frames
        self.cluster_window = cluster_window

        self._traj: list[Optional[tuple[float, float]]] = []  # [(x, y) or None]
        self._pending: dict[int, dict] = {}   # frame_idx → stats
        self._emitted: set[int] = set()

    def reset(self) -> None:
        self._traj.clear()
        self._pending.clear()
        self._emitted.clear()

    def push(self, frame_idx: int, px: Optional[tuple[float, float]]) -> list[dict]:
        """Push one frame's ball pixel position.  Returns list of confirmed bounce dicts.

        Each bounce dict: {'frame': int, 'x': float, 'y': float, 'source': 'bounce_det'}
        """
        self._traj.append((float(px[0]), float(px[1])) if px else None)

        emitted: list[dict] = []

        # Evaluate frame (current - delay) with full window context
        eval_idx = len(self._traj) - 1 - self.delay_frames
        w = self.window
        if eval_idx < w or eval_idx + w >= len(self._traj):
            return emitted

        pt = self._traj[eval_idx]
        if pt is None:
            return emitted
        prev_i = eval_idx - w
        next_i = eval_idx + w
        while prev_i > 0 and self._traj[prev_i] is None:
            prev_i -= 1
        while next_i < len(self._traj) - 1 and self._traj[next_i] is None:
            next_i += 1
        if prev_i < 0 or next_i >= len(self._traj):
            return emitted
        p_prev = self._traj[prev_i]
        p_next = self._traj[next_i]
        if p_prev is None or p_next is None:
            return emitted

        v_in = np.array([pt[0] - p_prev[0], pt[1] - p_prev[1]])
        v_out = np.array([p_next[0] - pt[0], p_next[1] - pt[1]])
        n_in = float(np.linalg.norm(v_in))
        n_out = float(np.linalg.norm(v_out))
        if n_in < 1e-5 or n_out < 1e-5:
            return emitted

        cos_t = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cos_t)))
        y_reversal = bool(v_in[1] > 0 and v_out[1] < 0)
        sp_in = n_in / (eval_idx - prev_i)
        sp_out = n_out / (next_i - eval_idx)
        delta_v = abs(sp_in - sp_out)

        if angle < self.angle_thresh:
            return emitted
        if not (y_reversal or delta_v >= self.momentum_thresh):
            return emitted

        self._pending[eval_idx] = {
            "frame": frame_idx - self.delay_frames,
            "x": pt[0], "y": pt[1],
            "angle": angle, "y_reversal": y_reversal, "delta_v": delta_v,
        }

        # NMS cluster and emit oldest clusters
        sorted_keys = sorted(self._pending)
        if len(sorted_keys) < 2:
            return emitted

        i = 0
        while i < len(sorted_keys) - 1:
            if sorted_keys[i + 1] - sorted_keys[i] > self.cluster_window:
                cluster = [sorted_keys[i]]
                # collect forward
                j = i
                while j < len(sorted_keys) and sorted_keys[j] - cluster[0] <= self.cluster_window:
                    if sorted_keys[j] not in cluster:
                        cluster.append(sorted_keys[j])
                    j += 1
                # emit strongest in cluster
                best_k = max(cluster, key=lambda k: self._pending[k]["y"])
                bd = self._pending.pop(best_k)
                if best_k not in self._emitted:
                    self._emitted.add(best_k)
                    emitted.append({
                        "frame": bd["frame"],
                        "x": bd["x"],
                        "y": bd["y"],
                        "source": "bounce_det",
                    })
                # remove all in cluster
                for k in cluster:
                    self._pending.pop(k, None)
                # restart
                sorted_keys = sorted(self._pending)
                i = 0
            else:
                i += 1

        return emitted
