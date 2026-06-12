"""SingleCameraProcessor — one camera's full near-field detection + export pipeline."""

import logging
import math
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from app.hit_detector import detect_hits_v2
from app.mini_program.detectors import BallDetector, BounceDetector
from app.pipeline.homography import HomographyTransformer
from app.pipeline.player_detector import PlayerPoseDetector

logger = logging.getLogger(__name__)

MAX_BUFFER_FRAMES = 3000
RALLY_TIMEOUT_S = 4.0     # seconds of no bounce/hit → rally ends


class SingleCameraProcessor:
    """Per-camera pipeline: RTSP → ball + player detection → bounce + hit → export."""

    def __init__(self, name: str, rtsp_url: str, homography_path: str,
                 homography_key: str, ball_model_path: str,
                 player_model_path: str, player_device: str, player_conf: float,
                 serial_number: str, endpoint: str, dry_run: bool,
                 side: str       # "near" for cam66 (world_y < 0), "far" for cam68 (world_y > 0)
                 ):
        self.name = name
        self.side = side
        self.serial_number = serial_number
        self.endpoint = endpoint
        self.dry_run = dry_run
        self._rtsp_url = rtsp_url

        # --- models ---
        self.ball_detector = BallDetector(ball_model_path)
        self.player_detector = PlayerPoseDetector(
            player_model_path, device=player_device, conf=player_conf,
        )
        self.homography = HomographyTransformer(homography_path, homography_key)
        self.bounce_detector = BounceDetector()

        # --- state ---
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # rally tracking
        self._buffer: list[dict] = []
        self._rally_state = "idle"       # "idle" | "rally"
        self._rally_id = 0
        self._rally_start_time = 0.0
        self._last_event_time = 0.0      # last bounce or hit timestamp

        # hit detection — accumulate trajectory then call detect_hits_v2 at rally end
        self._traj_frame_nos: list[int] = []
        self._traj_px_x: list[float] = []
        self._traj_px_y: list[float] = []

        # speed tracking
        self._last_ball_world: Optional[tuple[float, float]] = None
        self._last_ball_ts: float = 0.0

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"near-field-{self.name}")
        self._thread.start()
        logger.info("[%s] Near-field pipeline started", self.name)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._cap:
            self._cap.release()
        logger.info("[%s] Near-field pipeline stopped", self.name)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._cap = cv2.VideoCapture(self._rtsp_url or "")
        if not self._cap.isOpened():
            logger.error("[%s] Cannot open RTSP: %s", self.name, self._rtsp_url)
            return

        frame_idx = 0

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.5)
                self._cap.release()
                self._cap = cv2.VideoCapture(self._rtsp_url or "")
                continue

            now = time.time()
            frame_idx += 1
            self._process_frame(frame, frame_idx, now)

    def _process_frame(self, frame: np.ndarray, frame_idx: int, now: float) -> None:
        # --- ball detection ---
        ball_px = self.ball_detector.detect(frame)
        ball_world = None
        if ball_px is not None:
            wx, wy = self.homography.pixel_to_world(ball_px[0], ball_px[1])
            # filter: only accept balls in our near field
            side_ok = (self.side == "near" and wy < 0) or (self.side == "far" and wy > 0)
            if side_ok:
                ball_world = (wx, wy)

        # --- speed ---
        speed_kmh = 0.0
        if ball_world is not None and self._last_ball_world is not None:
            dx = ball_world[0] - self._last_ball_world[0]
            dy = ball_world[1] - self._last_ball_world[1]
            dt = now - self._last_ball_ts
            if dt > 0.001:
                speed_kmh = math.hypot(dx, dy) / dt * 3.6
        if ball_world is not None:
            self._last_ball_world = ball_world
            self._last_ball_ts = now

        # --- player detection ---
        detections = self.player_detector.detect(frame)
        player = self._select_player(detections)

        # --- bounce detection ---
        bounces = self.bounce_detector.push(frame_idx, ball_px)

        # --- trajectory for hit detection ---
        if ball_px is not None:
            self._traj_frame_nos.append(frame_idx)
            self._traj_px_x.append(ball_px[0])
            self._traj_px_y.append(ball_px[1])
            if len(self._traj_frame_nos) > MAX_BUFFER_FRAMES:
                self._traj_frame_nos.pop(0)
                self._traj_px_x.pop(0)
                self._traj_px_y.pop(0)

        # --- build frame dict ---
        player_key = "near_player" if self.side == "near" else "far_player"
        other_key = "far_player" if self.side == "near" else "near_player"
        fr = {
            "ts": now,
            "frame_index": frame_idx,
            "capture_ts": now,
            "ball": {"x": ball_world[0], "y": ball_world[1], "z": 0.0} if ball_world else None,
            player_key: player,
            other_key: None,
            "speed_kmh": speed_kmh,
            "is_bounce": False,
            "is_hit": False,
        }

        # --- rally state ---
        has_event = len(bounces) > 0
        if has_event:
            self._last_event_time = now

        if self._rally_state == "idle":
            if has_event:
                self._rally_state = "rally"
                self._rally_id += 1
                self._rally_start_time = now
                self._buffer.clear()
                logger.info("[%s] Rally %d started", self.name, self._rally_id)
                for bd in bounces:
                    self._mark_bounce(bd)
        else:
            if now - self._last_event_time > RALLY_TIMEOUT_S:
                self._end_rally(now)
                self._rally_state = "idle"
            else:
                for bd in bounces:
                    self._mark_bounce(bd)

        # always buffer in rally
        if self._rally_state == "rally":
            self._buffer.append(fr)
            if len(self._buffer) > MAX_BUFFER_FRAMES:
                self._buffer.pop(0)

        # check rally timeout when no frames arriving (e.g. end of video)
        if self._rally_state == "rally" and now - self._last_event_time > RALLY_TIMEOUT_S:
            self._end_rally(now)
            self._rally_state = "idle"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _select_player(self, detections: list[dict]) -> Optional[dict]:
        """Pick the player on our near side of the court."""
        best = None
        best_score = float("inf")
        for det in detections:
            foot_px = det.get("foot_px")
            if not foot_px or len(foot_px) < 2:
                continue
            try:
                wx, wy = self.homography.pixel_to_world(foot_px[0], foot_px[1])
            except Exception:
                continue
            side_ok = (self.side == "near" and wy < 0) or (self.side == "far" and wy > 0)
            if not side_ok:
                continue
            # prefer closer to baseline (larger |y|)
            score = -abs(wy)
            if score < best_score:
                best_score = score
                best = {
                    "foot_court": [round(wx, 3), round(wy, 3)],
                    "keypoints_px": det.get("keypoints", []),
                    "conf": det.get("conf", 0.0),
                }
        return best

    def _mark_bounce(self, bd: dict) -> None:
        """Mark matching frame in buffer as bounce."""
        target_frame = bd["frame"]
        for fr in reversed(self._buffer):
            if fr["frame_index"] == target_frame:
                fr["is_bounce"] = True
                if fr["ball"] is None:
                    wx, wy = self.homography.pixel_to_world(bd["x"], bd["y"])
                    fr["ball"] = {"x": wx, "y": wy, "z": 0.0}
                fr["event_ball"] = {"x": float(bd["x"]), "y": float(bd["y"])}
                break

    def _end_rally(self, now: float) -> None:
        if not self._buffer:
            logger.info("[%s] Rally %d ended — no frames", self.name, self._rally_id)
            return

        logger.info("[%s] Rally %d ended (%d frames)", self.name, self._rally_id, len(self._buffer))

        # --- hit detection ---
        self._detect_and_mark_hits()

        # --- export ---
        from app.mini_program.result_exporter import format_rally

        class _RallyProxy:
            pass
        proxy = _RallyProxy()
        proxy.rally_id = self._rally_id
        proxy.start_time = self._rally_start_time
        proxy.end_time = now

        frames_snapshot = list(self._buffer)
        result = format_rally(
            proxy, frames_snapshot,
            serial_number=self.serial_number,
            endpoint=self.endpoint,
            dry_run=self.dry_run,
        )
        if result:
            logger.info("[%s] Rally %d exported", self.name, self._rally_id)
        else:
            logger.warning("[%s] Rally %d export skipped (empty resultmatrix)", self.name, self._rally_id)

        self._buffer.clear()
        self._traj_frame_nos.clear()
        self._traj_px_x.clear()
        self._traj_px_y.clear()

    def _detect_and_mark_hits(self) -> None:
        """Run hit_detector v2 on accumulated trajectory, mark frames."""
        n = len(self._traj_frame_nos)
        if n < 5:
            return

        # get homography matrix for hit_detector v2
        H = self.homography.H_img2world.tolist()

        hits = detect_hits_v2(
            self._traj_frame_nos,
            self._traj_px_x,
            self._traj_px_y,
            H,
            detect_side=self.side,
            min_interval=5,
            velocity_change_threshold=0.5,
        )
        hit_frames = {h[0] for h in hits}

        for fr in self._buffer:
            if fr["frame_index"] in hit_frames:
                fr["is_hit"] = True
                if fr["ball"] is None:
                    fr["ball"] = {"x": 0.0, "y": 0.0, "z": 0.0}
