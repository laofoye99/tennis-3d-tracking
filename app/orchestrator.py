"""Main process orchestrator: manages camera pipeline subprocesses and triangulation."""

import datetime
import json
import logging
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from collections import deque
from typing import Any, Optional

import cv2
import numpy as np

from app.config import AppConfig
from app.pipeline.camera_pipeline import run_pipeline
from app.pipeline.video_pipeline import run_video_pipeline
from app.schemas import BallPosition3D, PipelineStatus, SystemStatus, WorldPoint2D
from app.analytics import (
    BounceEvent,
    BounceDetector,
    HybridBounceDetector,
    PeakBounceDetector,
    RallyTracker,
    run_batch_analytics,
)
from app.realtime_hit_bounce import HitBounceRefiner
from app.trajectory import clean_detections, find_offset_and_triangulate, fit_trajectory, segment_rallies
from app.pipeline.multi_blob_matcher import MultiBlobMatcher
from app.triangulation import triangulate

logger = logging.getLogger(__name__)

# Maximum age (seconds) for pairing detections from two cameras.
# `capture_ts` is the wall-clock at frame arrival in Python (camera_pipeline.py),
# NOT a hardware-synced PTS — RTSP jitter, decoder buffering and GIL scheduling
# make cross-camera capture_ts drift by 100-300ms in practice. A window wider
# than the worst expected drift keeps pairing alive; too tight (e.g. 0.1s)
# starves the whole pipeline. Tighten further only after measuring the live
# best_dt distribution and confirming the p95 sits under the chosen value.
_MATCH_WINDOW = 0.3


class _PipelineHandle:
    """Holds references to a single camera pipeline subprocess."""

    def __init__(self, name: str):
        self.name = name
        self.process: Optional[mp.Process] = None
        self.result_queue: Optional[mp.Queue] = None
        self.frame_queue: Optional[mp.Queue] = None
        self.stop_event: Optional[mp.Event] = None
        self.status_dict: Optional[dict] = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()


class Orchestrator:
    """Manages camera pipelines, triangulation, and exposes state for the API."""

    _LIVE_BOUNCE_HISTORY_LIMIT = 500

    def __init__(self, config: AppConfig):
        self.config = config
        self._handles: dict[str, _PipelineHandle] = {}
        self._manager = mp.Manager()

        for cam_name in config.cameras:
            self._handles[cam_name] = _PipelineHandle(cam_name)

        self._latest_detections: dict[str, dict] = {}
        self._latest_frames: dict[str, bytes] = {}
        self._latest_3d: Optional[BallPosition3D] = None
        self._triangulation_active = False
        self._last_tri_pair: tuple = (None, None)  # (d1.capture_ts, d2.capture_ts) to dedup
        self._det_queues: dict[str, list] = {}  # per-camera detection queues
        self._live_matcher: Optional[MultiBlobMatcher] = None
        self._candidate_continuity: dict[str, dict] = {}
        self._LIVE_MATCHER_CANDIDATES = 2
        self._CANDIDATE_MAX_JUMP_PX = 120.0
        self._CANDIDATE_CONF_RATIO = 0.35
        self._CANDIDATE_RANK_PENALTY_PX = 8.0
        self._consumer_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._inference_enabled: bool = True  # 全局推理开关

        # Ball 3D position queue (most recent 500 points, ~16s at 30fps)
        from collections import deque as _deque
        self._ball_3d_queue: _deque = _deque(maxlen=500)

        # Latest player pose per camera (nearest player to ball)
        self._latest_player_pose: dict[str, dict] = {}

        # 录像
        self._recording: bool = False
        self._recording_writers: dict[str, Any] = {}  # name -> {"writer": VideoWriter|None, "path": str}
        self._recording_info: dict = {}
        self._recording_lock = threading.Lock()
        self._recordings_dir = Path("recordings")
        self._ffmpeg_lock = threading.Lock()

        # Tracking JSONL (always writes when pipeline runs, independent of recording)
        self._tracking_file: Any = None
        self._tracking_file_path: str | None = None
        self._data_frame_counter: int = 0
        self._jsonl_lock = threading.Lock()
        self._jsonl_last_fsync_ts: float = 0.0
        self._recording_tracking_path: str | None = None
        self._last_completed_tracking_path: str | None = None
        self._ffmpeg_processes: dict[str, dict[str, Any]] = {}
        self._ffmpeg_start_time: float = 0.0
        self._ffmpeg_session_dir: str | None = None
        self._ffmpeg_stop_reason: str | None = None
        self._ffmpeg_monitor_thread: Optional[threading.Thread] = None
        self._ffmpeg_stopping: bool = False
        self._ffmpeg_stop_thread: Optional[threading.Thread] = None

        # Video test
        self._video_test_handle: Optional[_PipelineHandle] = None
        self._video_test_handles: dict[str, _PipelineHandle] = {}  # parallel handles
        self._video_test_detections: dict[str, list[dict]] = {}  # camera_name -> detections

        # Debug output — records all pipeline stages for GT comparison
        self._debug_dir = Path("debug_output")
        self._debug_data = self._new_debug_data()

        # Live analytics — Hybrid is the sole production source.
        # Peak runs alongside for evaluation only; its output is walled off in
        # _peak_bounces_eval and never touches _live_bounces, rally_sm,
        # _rally_tracker, JSONL, or WebSocket.
        # Params relaxed vs. the offline defaults — live 3D is sparser
        # (capture_ts drift + missed pairs break continuity). The key lever
        # is max_gap_s: offline default 0.2s (~5 frames at 25fps) cuts the
        # segment on any burst of missed matches; live needs ~0.6s to absorb
        # typical drops. Sizes also relaxed so the segment-length &
        # density gates don't swallow the remaining segments.
        bounce_cfg = self.config.bounce_detection
        hybrid_cfg = bounce_cfg.hybrid
        self._hybrid_bounce = HybridBounceDetector(
            z_max=hybrid_cfg.z_max,
            min_seg_len=hybrid_cfg.min_seg_len,
            min_dense=hybrid_cfg.min_dense,
            dense_range=hybrid_cfg.dense_range,
            min_speed=hybrid_cfg.min_speed,
            max_gap_s=hybrid_cfg.max_gap_s,
            v_window=hybrid_cfg.v_window,
            half_wins=tuple(hybrid_cfg.half_wins),
            cooldown_frames=hybrid_cfg.cooldown_frames,
        )
        self._bounce_detector = PeakBounceDetector(batch_size=10)   # eval only
        # RallyStateMachine (serve rules, PENDING, DOUBLE_FAULT, LET, ...) was
        # removed from the realtime path — the rule-based state machine never
        # had GT validation, it was producing confusing transitions under the
        # noisy realtime 3D stream, and every consumer it used to drive
        # (rally_end markers, F1 post-filter, _export_rally, auto-report)
        # was better off re-sourced from the simple RallyTracker or dropped.
        # The class stays defined in analytics.py for offline tools /
        # batch analytics (run_batch_analytics, FusionCoordinator).
        self._rally_tracker = RallyTracker()
        self._live_bounces: list[dict] = []   # real Hybrid bounces only
        self._live_hits: list[dict] = []
        self._total_live_bounces: int = 0
        self._peak_bounces_eval: list[dict] = []   # sidecar, cap 100
        self._hit_bounce_refiner = HitBounceRefiner(self.config.hit_bounce_refiner)
        self._last_smoothed_cam_dets: dict = {}
        self._last_refiner_result: dict = {}
        # Post-filter telemetry — counts per reason (incl. "accepted") so we
        # can tune thresholds from live data without re-running eval.
        from collections import Counter as _Counter
        self._post_filter_stats: _Counter = _Counter()

        # Auto report: generate after every N completed rallies
        self._rally_report_interval: int = 10  # 0 = disabled
        self._rally_completed_count: int = 0
        self._last_report_rally_count: int = 0

        # MedianBG tracking state (track-first-triangulate-later pipeline)
        self._is_median_bg = self.config.model.detector_type == "median_bg"
        self._blob_buffers: dict[str, dict[int, list]] = {}  # cam -> {frame_id: [(cx,cy)]}
        self._blob_capture_ts_by_frame: dict[str, dict[int, float]] = {}  # cam -> {frame_id: capture_ts}
        self._blob_homographies: dict = {}  # cam -> H matrix (lazy init)
        self._tracker_process_interval = 30  # run tracker every N blob_blocks
        self._tracker_block_count = 0
        self._trajectory_3d: list = []  # accumulated 3D trajectory
        self._emitted_3d_frames: set = set()  # frames already emitted
        self._emitted_bounce_frames: set = set()


        # Savitzky-Golay smoothing buffer (matches offline smooth_trajectory_sg)
        self._sg_buffer: list[dict] = []  # [{"pt": raw_3d_point, "cam": cam_dets}]
        smooth_cfg = bounce_cfg.smoothing
        self._sg_window = smooth_cfg.sg_window
        self._sg_poly = smooth_cfg.sg_poly
        self._sg_max_gap = smooth_cfg.max_frame_gap  # frames gap to split segments
        self._sg_max_gap_s = smooth_cfg.max_gap_s  # live stream is sparse; don't over-split SG segments
        self._sg_midpoint_mode = False
        self._sg_switched_to_midpoint = False
        self._live_rallies: list[dict] = []
        self._analytics_lock = threading.Lock()

        # Confidence filtering (top1_conf20)
        self._conf_percentile = 20  # reject bottom 20% by blob_sum
        self._conf_history: list[float] = []  # rolling blob_sum values
        self._conf_threshold = 0.0  # dynamic, updated from history

        # Net crossing speed detection
        self._prev_3d: Optional[dict] = None
        self._speed_points: deque = deque(maxlen=12)  # recent 3D points for local-fit speed
        self._speed_buffer: deque = deque(maxlen=5)  # recent per-frame speeds for smoothing
        self._latest_net_crossing: Optional[dict] = None
        self._net_crossings: list[dict] = []
        self._NET_Y = 0.0  # V2: net at y=0
        self._SPEED_MIN = 30   # km/h — consumer minimum
        self._SPEED_MAX = 150  # km/h — consumer maximum (net-crossing gate)
        self._MAX_PHYSICAL_SPEED = 280  # km/h — physics gate (world record ~263 km/h)
        self._SPEED_FIT_WINDOW = 7
        self._SPEED_MIN_POINTS = 4
        self._SPEED_MAX_GAP_S = 0.20
        self._SPEED_MAX_FRAME_GAP = 3
        self._SPEED_MAX_RESIDUAL_M = 0.35

        # 3D display WebSocket push
        self._ws_bounce_queue: deque[dict[str, Any]] = deque(maxlen=100)
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_url = "wss://tennisserver.motionrivalry.com:8086/general"
        self._ws_enabled = False

        # Latency instrumentation
        self._latency_buffer: deque = deque(maxlen=1000)
        self._latency_max: float = 0.0

        # ML Rally segmentation filter
        self._ml_rally_enabled = False
        self._ml_rally_model = None
        self._ml_rally_features_buffer: list[dict] = []  # rolling buffer for feature extraction

        # Feature toggles (bounce detection, net crossing, OCR align)
        self._bounce_detection_enabled: bool = True
        self._net_crossing_enabled: bool = True
        self._ocr_align_enabled: bool = False

        # Rally raw buffer for result export (120s @ 25fps ≈ 3000 frames)
        self._rally_raw_buffer: deque = deque(maxlen=3000)
        self._last_frame_speed_kmh: float = 0.0
        self._last_bounce_ts: float = 0.0  # timestamp of most recent bounce

    def _get_camera_positions(self) -> dict[str, list[float]]:
        """Get camera 3D positions, optionally overriding with calibrated values.

        When ``config.calibration.use_calibrated_positions`` is True,
        loads camera positions from the calibration JSON file.  Falls
        back to ``config.cameras[name].position_3d`` otherwise.
        """
        positions = {
            n: self.config.cameras[n].position_3d
            for n in self.config.cameras
        }
        if self.config.calibration.use_calibrated_positions:
            cal_path = Path(self.config.calibration.path)
            if cal_path.exists():
                try:
                    with open(cal_path, "r", encoding="utf-8") as f:
                        cal_data = json.load(f)
                    for n in positions:
                        if n in cal_data and "camera_position_3d" in cal_data[n]:
                            positions[n] = cal_data[n]["camera_position_3d"]
                            logger.debug("[%s] Using calibrated position: %s", n, positions[n])
                except Exception as e:
                    logger.warning("Failed to load calibration positions from %s: %s", cal_path, e)
            else:
                logger.warning("Calibration file not found: %s, using config positions", cal_path)
        return positions

    @staticmethod
    def _candidate_confidence(candidate: dict) -> float:
        try:
            return float(candidate.get("blob_sum", candidate.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _apply_live_candidate_continuity(
        self,
        cam_name: str,
        det: dict,
        max_candidates: int | None = None,
    ) -> dict:
        """Prefer the most temporally plausible 2D blob before 3D matching.

        TrackNet sometimes ranks a static distractor or player-edge blob above
        the real ball. We keep the detector's top-1 unless it makes an
        implausible pixel jump and another top-k candidate follows the recent
        per-camera motion well enough. The 3D matcher still receives only the
        top candidates, so this improves ordering without widening the noisy
        cross-camera search space.
        """
        candidates_in = det.get("candidates") or []
        if not candidates_in:
            return det

        max_candidates = max_candidates or self._LIVE_MATCHER_CANDIDATES
        candidates = [
            dict(c)
            for c in candidates_in
            if c.get("pixel_x") is not None and c.get("pixel_y") is not None
        ]
        if not candidates:
            return det

        state = self._candidate_continuity.get(cam_name)
        chosen_idx = 0

        try:
            frame_index = int(det.get("frame_index")) if det.get("frame_index") is not None else None
        except (TypeError, ValueError):
            frame_index = None

        if state is not None and frame_index is not None and state.get("frame_index") is not None:
            gap = frame_index - int(state["frame_index"])
            if 0 < gap <= 10:
                last_px = np.array([state["pixel_x"], state["pixel_y"]], dtype=np.float64)
                velocity = state.get("velocity_px_per_frame")
                if velocity is not None:
                    pred_px = last_px + np.array(velocity, dtype=np.float64) * gap
                else:
                    pred_px = last_px

                dists = []
                for idx, cand in enumerate(candidates):
                    px = np.array([cand["pixel_x"], cand["pixel_y"]], dtype=np.float64)
                    dist = float(np.linalg.norm(px - pred_px))
                    score = dist + self._CANDIDATE_RANK_PENALTY_PX * idx
                    dists.append((score, dist, idx))

                top_score, top_dist, _ = dists[0]
                best_score, best_dist, best_idx = min(dists, key=lambda item: item[0])
                max_jump = self._CANDIDATE_MAX_JUMP_PX * gap
                if best_idx != 0 and top_dist > max_jump and best_dist <= max_jump:
                    top_conf = self._candidate_confidence(candidates[0])
                    best_conf = self._candidate_confidence(candidates[best_idx])
                    if best_score < top_score and best_conf >= top_conf * self._CANDIDATE_CONF_RATIO:
                        chosen_idx = best_idx

        if chosen_idx:
            chosen = candidates.pop(chosen_idx)
            candidates.insert(0, chosen)

        candidates = candidates[:max_candidates]
        selected = candidates[0]
        new_det = dict(det)
        new_det["candidates"] = candidates
        new_det["pixel_x"] = selected["pixel_x"]
        new_det["pixel_y"] = selected["pixel_y"]
        if selected.get("world_x") is not None:
            new_det["x"] = selected["world_x"]
            new_det["world_x"] = selected["world_x"]
        elif selected.get("x") is not None:
            new_det["x"] = selected["x"]
        if selected.get("world_y") is not None:
            new_det["y"] = selected["world_y"]
            new_det["world_y"] = selected["world_y"]
        elif selected.get("y") is not None:
            new_det["y"] = selected["y"]
        conf = self._candidate_confidence(selected)
        new_det["confidence"] = conf
        new_det["blob_sum"] = conf

        px_now = np.array([selected["pixel_x"], selected["pixel_y"]], dtype=np.float64)
        velocity = None
        if state is not None and frame_index is not None and state.get("frame_index") is not None:
            gap = frame_index - int(state["frame_index"])
            if 0 < gap <= 10:
                last_px = np.array([state["pixel_x"], state["pixel_y"]], dtype=np.float64)
                displacement = px_now - last_px
                if float(np.linalg.norm(displacement)) <= self._CANDIDATE_MAX_JUMP_PX * gap * 1.5:
                    velocity = (displacement / gap).tolist()

        self._candidate_continuity[cam_name] = {
            "pixel_x": float(selected["pixel_x"]),
            "pixel_y": float(selected["pixel_y"]),
            "frame_index": frame_index,
            "capture_ts": new_det.get("capture_ts", new_det.get("timestamp")),
            "velocity_px_per_frame": velocity,
        }
        return new_det

    def start_pipeline(self, name: str) -> None:
        if name not in self._handles:
            raise ValueError(f"Unknown pipeline: {name}")

        handle = self._handles[name]
        if handle.is_alive():
            logger.warning("[%s] Pipeline already running", name)
            return

        cam_cfg = self.config.cameras[name]
        model_cfg = self.config.model

        handle.result_queue = mp.Queue(maxsize=64)
        handle.frame_queue = mp.Queue(maxsize=128)
        handle.stop_event = mp.Event()
        handle.status_dict = self._manager.dict(
            {
                "state": "stopped",
                "fps": 0.0,
                "last_detection_time": None,
                "error_msg": "",
                "inference_enabled": self._inference_enabled,
            }
        )

        player_cfg = self.config.player_detection
        handle.process = mp.Process(
            target=run_pipeline,
            kwargs={
                "name": name,
                "rtsp_url": cam_cfg.rtsp_url,
                "model_path": model_cfg.path,
                "input_size": tuple(model_cfg.input_size),
                "frames_in": model_cfg.frames_in,
                "frames_out": model_cfg.frames_out,
                "threshold": model_cfg.threshold,
                "device": model_cfg.device,
                "homography_path": self.config.homography.path,
                "homography_key": cam_cfg.homography_key,
                "result_queue": handle.result_queue,
                "frame_queue": handle.frame_queue,
                "stop_event": handle.stop_event,
                "status_dict": handle.status_dict,
                "detector_type": model_cfg.detector_type,
                "player_model_path": player_cfg.model_path if player_cfg.enabled else "",
                "player_device": player_cfg.device,
                "player_conf": player_cfg.conf,
                "player_run_every_n": player_cfg.run_every_n_frames,
            },
            daemon=True,
        )
        handle.process.start()
        logger.info("[%s] Pipeline process started (pid=%d)", name, handle.process.pid)
        self._candidate_continuity.pop(name, None)

        # Ensure consumer thread is running.
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._stopped.clear()
            self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._consumer_thread.start()

    def stop_pipeline(self, name: str) -> None:
        if name not in self._handles:
            raise ValueError(f"Unknown pipeline: {name}")
        handle = self._handles[name]
        if handle.stop_event is not None:
            handle.stop_event.set()
        if handle.process is not None:
            handle.process.join(timeout=10.0)
            if handle.process.is_alive():
                logger.warning("[%s] Force terminating pipeline", name)
                handle.process.terminate()
                handle.process.join(timeout=5.0)
        if handle.status_dict is not None:
            handle.status_dict["state"] = "stopped"
        self._latest_frames.pop(name, None)
        self._candidate_continuity.pop(name, None)
        logger.info("[%s] Pipeline stopped", name)

        # Auto-save debug output if there's data
        if self._debug_data["trajectory"]:
            try:
                self.save_debug_output()
            except Exception as e:
                logger.warning("Failed to auto-save debug output: %s", e)

    def shutdown(self) -> None:
        self._stopped.set()
        for name in list(self._handles):
            self.stop_pipeline(name)
        self._manager.shutdown()

    # ------------------------------------------------------------------
    # Consumer loop: reads detection results from all pipeline queues
    # ------------------------------------------------------------------
    def _consume_loop(self) -> None:
        logger.info("Consumer thread started")
        self._triangulation_active = True

        # Always keep one active tracking JSONL open while the consumer runs.
        with self._jsonl_lock:
            if self._tracking_file is None:
                tracking_path = self._make_tracking_jsonl_path(label="bg")
                try:
                    self._open_tracking_file_locked(tracking_path, reset_counter=True)
                    logger.info("Tracking JSONL started: %s", tracking_path)
                except Exception as e:
                    logger.warning("Failed to open tracking file: %s", e)

        # Only cameras with position_3d can do triangulation
        cam_positions = self._get_camera_positions()
        cam_names = [n for n in self.config.cameras if n in cam_positions]
        tri_cams = cam_names[:2]  # first 2 positioned cameras for triangulation

        # Initialize multi-blob matcher for live mode
        self._live_matcher = None
        if len(cam_names) == 2:
            pos1 = cam_positions.get(tri_cams[0])
            pos2 = cam_positions.get(tri_cams[1])
            if pos1 and pos2:
                self._live_matcher = MultiBlobMatcher(
                    pos1,
                    pos2,
                    valid_z_range=(0.0, 8.0),
                    fps=25.0,
                )

        while not self._stopped.is_set():
            got_any = False
            for name, handle in list(self._handles.items()):
                # 消费检测结果 — 每个检测都保存，不丢弃
                if handle.result_queue is not None:
                    try:
                        while not handle.result_queue.empty():
                            det = handle.result_queue.get_nowait()

                            # Player pose detection result
                            if det.get("type") == "player_pose":
                                self._handle_player_pose(det, cam_positions)
                                got_any = True
                                continue

                            # MedianBG: blob_block → accumulate for tracker
                            if det.get("type") == "blob_block":
                                cam = det["camera_name"]
                                self._blob_buffers.setdefault(cam, {}).update(det["blobs"])
                                self._blob_capture_ts_by_frame.setdefault(cam, {}).update(
                                    det.get("capture_ts_by_frame", {})
                                )
                                self._tracker_block_count += 1
                                # Build a minimal latest_detection for dashboard overlay
                                # (pick first blob of last frame as rough position)
                                last_fi = max(det["blobs"].keys()) if det["blobs"] else None
                                if last_fi is not None and det["blobs"][last_fi]:
                                    cx, cy = det["blobs"][last_fi][0]
                                    self._latest_detections[cam] = {
                                        "camera_name": cam, "pixel_x": cx, "pixel_y": cy,
                                        "timestamp": det["timestamp"],
                                        "capture_ts": det["capture_ts"],
                                    }
                                got_any = True
                                continue

                            # Queue all detections for triangulation (not just latest)
                            if name in tri_cams:
                                det = self._apply_live_candidate_continuity(
                                    name,
                                    det,
                                    max_candidates=self._LIVE_MATCHER_CANDIDATES,
                                )
                                self._det_queues.setdefault(name, []).append(det)
                            self._latest_detections[name] = det
                            if name.startswith("_video_test"):
                                cam = det.get("camera_name", "unknown")
                                self._video_test_detections.setdefault(cam, []).append(det)
                            got_any = True
                    except Exception:
                        pass
                # 消费最新预览/录像帧
                if handle.frame_queue is not None:
                    try:
                        new_payload = None
                        while not handle.frame_queue.empty():
                            new_payload = handle.frame_queue.get_nowait()
                        if new_payload is not None:
                            if isinstance(new_payload, dict):
                                preview_jpeg = new_payload.get("preview")
                                recording_jpeg = new_payload.get("recording")
                            else:
                                preview_jpeg = new_payload
                                recording_jpeg = new_payload if self._recording else None
                            if preview_jpeg is not None:
                                self._latest_frames[name] = preview_jpeg
                            with self._recording_lock:
                                if self._recording and recording_jpeg is not None:
                                    self._write_recording_frame(name, recording_jpeg)
                    except Exception:
                        pass

            # ---- MedianBG: track → match → triangulate → events ----
            if self._is_median_bg and self._tracker_block_count > 0 and len(tri_cams) == 2:
                # Run every time both cameras have new blocks
                cam1, cam2 = tri_cams
                buf1 = self._blob_buffers.get(cam1, {})
                buf2 = self._blob_buffers.get(cam2, {})
                if buf1 and buf2:
                    self._run_tracker_pipeline(cam1, cam2, cam_positions)
                    self._tracker_block_count = 0

            # ---- TrackNet: pair detections by capture_ts ----
            q1 = self._det_queues.get(tri_cams[0], []) if len(tri_cams) == 2 else []
            q2 = self._det_queues.get(tri_cams[1], []) if len(tri_cams) == 2 else []

            # Only match when both queues have data
            pairs = []
            if q1 and q2:
                used_i, used_j = set(), set()
                for i, d1 in enumerate(q1):
                    t1 = d1.get("capture_ts", d1["timestamp"])
                    best_j, best_dt = -1, _MATCH_WINDOW
                    for j, d2 in enumerate(q2):
                        if j in used_j:
                            continue
                        t2 = d2.get("capture_ts", d2["timestamp"])
                        dt = abs(t1 - t2)
                        if dt < best_dt:
                            best_dt = dt
                            best_j = j
                    if best_j >= 0:
                        pairs.append((d1, q2[best_j]))
                        used_i.add(i)
                        used_j.add(best_j)
                # Keep unmatched detections for next round
                self._det_queues[tri_cams[0]] = [d for i, d in enumerate(q1) if i not in used_i]
                self._det_queues[tri_cams[1]] = [d for j, d in enumerate(q2) if j not in used_j]
                # Cap queue size to prevent memory leak
                for c in tri_cams:
                    if len(self._det_queues[c]) > 32:
                        self._det_queues[c] = self._det_queues[c][-16:]

            for d1, d2 in pairs:
                # Dedup exact re-processing of the same cross-camera pair by
                # capture time, not by pixel coords. A stationary ball can
                # legitimately occupy the same pixels across consecutive frames.
                pair_id = (
                    d1.get("capture_ts", d1["timestamp"]),
                    d2.get("capture_ts", d2["timestamp"]),
                )
                if pair_id == self._last_tri_pair:
                    continue
                # --- Confidence filtering (top1_conf20) ---
                blob_sum1 = d1.get("blob_sum", d1.get("confidence", 1.0))
                blob_sum2 = d2.get("blob_sum", d2.get("confidence", 1.0))
                avg_conf = (blob_sum1 + blob_sum2) / 2
                self._conf_history.append(avg_conf)
                if len(self._conf_history) > 500:
                    self._conf_history = self._conf_history[-500:]
                # Update threshold every 50 new pairs (not every pair)
                if len(self._conf_history) >= 50 and len(self._conf_history) % 50 == 0:
                    sorted_h = sorted(self._conf_history)
                    self._conf_threshold = sorted_h[int(len(sorted_h) * self._conf_percentile / 100)]
                if avg_conf < self._conf_threshold:
                    continue

                try:
                    x, y, z = None, None, None
                    _tri_smoothed = None
                    _tri_bounce = None
                    match = None


                    if (self._live_matcher
                            and "candidates" in d1
                            and "candidates" in d2):
                        match = self._live_matcher.match(d1, d2)
                        if match is not None:
                            x, y, z = match["x"], match["y"], match["z"]

                    if x is None:
                        x, y, z = triangulate(
                            (d1["x"], d1["y"]),
                            (d2["x"], d2["y"]),
                            cam_positions[tri_cams[0]],
                            cam_positions[tri_cams[1]],
                        )

                    self._latest_3d = BallPosition3D(
                        x=x, y=y, z=z,
                        cam66_world=WorldPoint2D(**d1),
                        cam68_world=WorldPoint2D(**d2),
                    )
                    self._ball_3d_queue.append({
                        "x": x, "y": y, "z": z,
                        "timestamp": time.time(),
                        "capture_ts": min(
                            d1.get("capture_ts", d1["timestamp"]),
                            d2.get("capture_ts", d2["timestamp"]),
                        ),
                    })
                    self._last_tri_pair = pair_id

                    # Debug recording
                    self._debug_data["frame_counter"] += 1
                    fi = self._debug_data["frame_counter"]
                    self._debug_record_detection(tri_cams[0], d1, fi)
                    self._debug_record_detection(tri_cams[1], d2, fi)
                    rd = match.get("ray_distance", 0) if match else 0
                    self._debug_record_3d(fi, x, y, z, rd, d1, d2, tri_cams[0], tri_cams[1])


                    cap_ts1 = d1.get("capture_ts", d1["timestamp"])
                    cap_ts2 = d2.get("capture_ts", d2["timestamp"])
                    latency_ms = (time.time() - min(cap_ts1, cap_ts2)) * 1000
                    self._latency_buffer.append(latency_ms)
                    if latency_ms > self._latency_max:
                        self._latency_max = latency_ms

                    now = time.time()
                    capture_ts = min(
                        d1.get("capture_ts", d1["timestamp"]),
                        d2.get("capture_ts", d2["timestamp"]),
                    )
                    # Use max of the two cameras' frame_index (they may differ slightly)
                    fi = max(d1.get("frame_index", 0), d2.get("frame_index", 0))
                    pt = {"x": x, "y": y, "z": z, "timestamp": now,
                          "capture_ts": capture_ts, "frame_index": fi}
                    if self._prev_3d is not None:
                        gap_s = capture_ts - self._prev_3d.get("capture_ts", capture_ts)
                        frame_gap = fi - self._prev_3d.get("frame_index", fi)
                        if gap_s <= 0 or gap_s > self._SPEED_MAX_GAP_S or frame_gap > self._SPEED_MAX_FRAME_GAP:
                            self._speed_points.clear()
                            self._speed_buffer.clear()
                    self._speed_points.append(pt)
                    if self._prev_3d is not None:
                        # Estimate speed from a short local fit on recent
                        # capture_ts-aligned points. This is more robust than
                        # consecutive-frame differencing when tracking noise
                        # perturbs one or two frames.
                        speed_kmh_fit = self._estimate_speed_kmh_from_window()
                        if speed_kmh_fit is not None:
                            self._speed_buffer.append(speed_kmh_fit)

                        # Net crossing detection
                        prev_y = self._prev_3d["y"]
                        curr_y = y
                        if self._net_crossing_enabled and (
                            (prev_y < self._NET_Y and curr_y >= self._NET_Y) or
                            (prev_y > self._NET_Y and curr_y <= self._NET_Y)
                        ):
                            # Use median of recent speeds (smoothed, robust to outliers)
                            if len(self._speed_buffer) >= 1:
                                speed_kmh = float(np.median(list(self._speed_buffer)))
                                if self._SPEED_MIN <= speed_kmh <= self._SPEED_MAX:
                                    direction = "near_to_far" if curr_y > prev_y else "far_to_near"
                                    # Speed surfaced externally is always an
                                    # integer km/h — dashboard / WS / report
                                    # never need 0.1 precision and float noise
                                    # across the net-crossing → bounce → WS
                                    # pipeline caused 'fractional' values to
                                    # leak into reports.
                                    speed_kmh_int = int(round(speed_kmh))
                                    crossing = {
                                        "speed_kmh": speed_kmh_int,
                                        "direction": direction,
                                        "timestamp": now,
                                        "capture_ts": capture_ts,
                                        "frame_index": fi,
                                        "x": x, "y": y, "z": z,
                                    }
                                    self._latest_net_crossing = crossing
                                    self._net_crossings.append(crossing)
                                    self._debug_data.setdefault("net_crossings", []).append({
                                        "frame": fi, "speed_kmh": speed_kmh_int,
                                        "direction": direction, "x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                                    })
                                    if len(self._net_crossings) > 100:
                                        self._net_crossings = self._net_crossings[-100:]
                    self._prev_3d = pt

                    # Track per-frame speed for raw buffer — use median of recent
                    # samples so a single noisy detection doesn't spike the value
                    # surfaced to the dashboard / rally_raw_buffer.
                    if self._speed_buffer:
                        # Surface as integer km/h for consistency with bounces.
                        self._last_frame_speed_kmh = float(
                            int(round(float(np.median(list(self._speed_buffer)))))
                        )
                    else:
                        self._last_frame_speed_kmh = 0.0


                    # cam_dets feeds HybridBounceDetector's landing-coord selector.
                    # Must reflect the blob live_matcher actually chose (may be
                    # non-top1 ~15% of frames per MultiBlobMatcher stats), not
                    # whatever top-1 blob happens to be in d1/d2.
                    cam_dets = {}
                    if match is not None:
                        c1w = match.get("cam1_world") or [d1.get("x"), d1.get("y")]
                        c2w = match.get("cam2_world") or [d2.get("x"), d2.get("y")]
                        c1p = match.get("cam1_pixel") or [d1.get("pixel_x"), d1.get("pixel_y")]
                        c2p = match.get("cam2_pixel") or [d2.get("pixel_x"), d2.get("pixel_y")]
                        cam_dets[tri_cams[0]] = {
                            "world_x": c1w[0], "world_y": c1w[1],
                            "pixel_x": c1p[0], "pixel_y": c1p[1],
                            "yolo_conf": d1.get("yolo_conf"),   # None if not provided
                            "blob_sum": match.get("cam1_blob_sum", d1.get("blob_sum", 0.0)),
                        }
                        cam_dets[tri_cams[1]] = {
                            "world_x": c2w[0], "world_y": c2w[1],
                            "pixel_x": c2p[0], "pixel_y": c2p[1],
                            "yolo_conf": d2.get("yolo_conf"),
                            "blob_sum": match.get("cam2_blob_sum", d2.get("blob_sum", 0.0)),
                        }
                    else:
                        # Fallback: no matcher selection available, use top-1 blob
                        for cname, det in [(tri_cams[0], d1), (tri_cams[1], d2)]:
                            cam_dets[cname] = {
                                "world_x": det.get("x"),
                                "world_y": det.get("y"),
                                "pixel_x": det.get("pixel_x"),
                                "pixel_y": det.get("pixel_y"),
                                "yolo_conf": det.get("yolo_conf"),
                                "blob_sum": det.get("blob_sum", 0.0),
                            }
                    with self._analytics_lock:
                        # --- Peak eval sidecar (isolated — no production impact) ---
                        _tri_smoothed, hbounce = self._run_live_bounce_detectors_locked(
                            pt,
                            cam_dets,
                        )

                        raw_bd = None
                        if hbounce is not None:
                            raw_bd = hbounce.to_dict()
                            event_capture_ts = raw_bd.get("capture_ts")
                            if event_capture_ts is None:
                                event_capture_ts = getattr(hbounce, "capture_ts", None)
                            if event_capture_ts is None:
                                event_capture_ts = capture_ts
                            raw_bd["capture_ts"] = float(event_capture_ts)
                            raw_bd["detect_delay"] = round(now - float(event_capture_ts), 2)

                        accepted_bounces: list[BounceEvent] = []
                        accepted_bds: list[dict] = []
                        new_hits: list[dict] = []
                        refiner_result = {
                            "new_hits": [],
                            "new_final_bounces": [],
                            "suppressed_bounces": [],
                            "stats": {},
                        }
                        if _tri_smoothed is not None:
                            refiner_result = self._hit_bounce_refiner.update(
                                _tri_smoothed,
                                raw_bounce=raw_bd,
                                players=self._build_hit_bounce_player_snapshot(now),
                                net_crossing=dict(self._latest_net_crossing) if self._latest_net_crossing else None,
                                cam_dets=self._last_smoothed_cam_dets,
                                now=now,
                            )
                            self._last_refiner_result = refiner_result
                            for hit in refiner_result.get("new_hits", []):
                                self._record_live_hit_locked(hit)
                                new_hits.append(hit)

                            for final_bd in refiner_result.get("new_final_bounces", []):
                                event_capture_ts = float(
                                    final_bd.get("capture_ts")
                                    or final_bd.get("timestamp")
                                    or capture_ts
                                )
                                final_bd["capture_ts"] = event_capture_ts
                                final_bd["detect_delay"] = round(now - event_capture_ts, 2)
                                gated_bd = self._gate_live_bounce_candidate_locked(
                                    final_bd,
                                    now=now,
                                    match_speed=True,
                                )
                                if gated_bd is None:
                                    continue
                                gated_bd = self._normalize_live_bounce_dict(
                                    gated_bd,
                                    fallback_ts=now,
                                    fallback_speed_kmh=0,
                                )
                                accepted_bds.append(gated_bd)
                                accepted_bounces.append(self._bounce_dict_to_event(gated_bd))

                        # --- Fan out ---
                        # Legacy RallyTracker gets the bounce (drives the
                        # simple idle/rally counter exposed via the API).
                        # rally_result (boundary signal) is no longer produced —
                        # end-markers, auto-report, and rally export all were
                        # tied to the removed RallyStateMachine.
                        _prev_state_str = self._rally_tracker._state.state
                        _prev_completed_count = len(self._rally_tracker.get_completed_rallies())
                        accepted_bounce = accepted_bounces[-1] if accepted_bounces else None
                        accepted_bd = accepted_bds[-1] if accepted_bds else None
                        self._rally_tracker.update(pt, accepted_bounce)
                        _curr_state_str = self._rally_tracker._state.state
                        if _prev_state_str == "idle" and _curr_state_str == "rally":
                            # Fresh rally boundary: drop any stale idle frames.
                            # The current frame gets appended below, so the
                            # first crossing frame is retained.
                            self._rally_raw_buffer.clear()
                        if _prev_state_str == "rally" and _curr_state_str == "idle":
                            _completed = self._rally_tracker.get_completed_rallies()
                            if len(_completed) > _prev_completed_count:
                                _frames_snapshot = list(self._rally_raw_buffer)
                                self._rally_raw_buffer.clear()
                                if _frames_snapshot:
                                    _last_dict = _completed[-1]

                                    class _RallyProxy:
                                        pass

                                    _proxy = _RallyProxy()
                                    _proxy.rally_id = _last_dict.get(
                                        "rally_id", self._rally_completed_count + 1
                                    )
                                    _proxy.start_time = now - _last_dict.get("duration", 0.0)
                                    _proxy.end_time = now
                                    threading.Thread(
                                        target=self._export_rally,
                                        args=(_proxy, _frames_snapshot),
                                        daemon=True,
                                        name=f"rally-export-{_proxy.rally_id}",
                                    ).start()
                                self._rally_completed_count += 1
                                if (
                                    self._rally_report_interval > 0
                                    and self._rally_completed_count - self._last_report_rally_count
                                    >= self._rally_report_interval
                                ):
                                    threading.Thread(
                                        target=self._auto_generate_report,
                                        daemon=True,
                                        name="auto-generate-report",
                                    ).start()
                        _tri_bounce = accepted_bounce

                        for idx, bd in enumerate(accepted_bds):
                            self._record_live_bounce_locked(
                                bd,
                                debug_source=accepted_bounces[idx],
                            )

                        # Rally-raw buffer still accumulates ball+player per
                        # frame so offline /api/report/generate still has
                        # data, but there's no longer an automatic rally-end
                        # trigger that flushes it to the export endpoint.
                        is_bounce_frame = bool(
                            accepted_bd is not None
                            and accepted_bd.get("frame_index") == fi
                        )
                        if accepted_bd is not None:
                            self._last_bounce_ts = float(accepted_bd.get("timestamp", capture_ts))
                        is_hit_frame = any(h.get("frame_index") == fi for h in new_hits)
                        _near_pose = self._latest_player_pose.get(tri_cams[0])
                        _far_pose = self._latest_player_pose.get(tri_cams[1])
                        near_player = _near_pose["player"] if _near_pose else None
                        far_player = _far_pose["player"] if _far_pose else None
                        self._rally_raw_buffer.append({
                            "ts": now,
                            "frame_index": fi,
                            "capture_ts": capture_ts,
                            "ball": {"x": x, "y": y, "z": z},
                            "near_player": near_player,
                            "far_player": far_player,
                            "speed_kmh": self._last_frame_speed_kmh,
                            "is_bounce": is_bounce_frame,
                            "is_hit": is_hit_frame,
                        })
                        for hit in new_hits:
                            self._mark_rally_buffer_event_locked(hit, "hit")
                        for bd in accepted_bds:
                            self._mark_rally_buffer_event_locked(bd, "bounce")

                    # Write per-frame tracking data (JSONL — always, not just during recording)
                    if self._tracking_file is not None:
                        self._write_tracking_frame(
                            d1, d2, tri_cams, x, y, z,
                            _tri_smoothed, _tri_bounce, now, capture_ts,
                        )
                except Exception as e:
                    logger.error("Triangulation/analytics error: %s", e, exc_info=True)

            if not got_any:
                time.sleep(0.001)  # 1ms — fast response to new detections

                # Timeout-triggered rally end: when no new points arrive,
                # RallyTracker only transitions inside get_state(). Poll here
                # so mini-program export still fires even if the rally ends on
                # silence rather than on a new frame.
                with self._analytics_lock:
                    if self._rally_tracker._state.state == "rally":
                        _after = self._rally_tracker.get_state().state
                        if _after == "idle":
                            _frames_snapshot = list(self._rally_raw_buffer)
                            self._rally_raw_buffer.clear()
                            _completed = self._rally_tracker.get_completed_rallies()
                            if _completed and _frames_snapshot:
                                _last_dict = _completed[-1]

                                class _RallyProxy:
                                    pass

                                _proxy = _RallyProxy()
                                _proxy.rally_id = _last_dict.get(
                                    "rally_id", self._rally_completed_count + 1
                                )
                                _proxy.end_time = time.time()
                                _proxy.start_time = _proxy.end_time - _last_dict.get("duration", 0.0)
                                threading.Thread(
                                    target=self._export_rally,
                                    args=(_proxy, _frames_snapshot),
                                    daemon=True,
                                    name=f"rally-export-timeout-{_proxy.rally_id}",
                                ).start()
                            self._rally_completed_count += 1
                            if (
                                self._rally_report_interval > 0
                                and self._rally_completed_count - self._last_report_rally_count
                                >= self._rally_report_interval
                            ):
                                threading.Thread(
                                    target=self._auto_generate_report,
                                    daemon=True,
                                    name="auto-generate-report-timeout",
                                ).start()

        self._triangulation_active = False
        # Close tracking JSONL
        with self._jsonl_lock:
            if self._tracking_file is not None:
                current_path = self._tracking_file_path
                self._close_tracking_file_locked(force_fsync=True)
                logger.info("Tracking JSONL closed: %s (%d frames)",
                            current_path, self._data_frame_counter)
        logger.info("Consumer thread stopped")

    # ------------------------------------------------------------------
    # Recording (called from FastAPI)
    # ------------------------------------------------------------------
    def start_recording(self) -> dict:
        """为所有正在运行的摄像头开始录像。"""
        with self._recording_lock:
            if self._recording:
                return {"status": "already_recording", "files": self._recording_info.get("files", {})}
            self._recordings_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            files = {}
            rec_start = time.time()
            for name, handle in self._handles.items():
                if handle.is_alive():
                    fname = str(self._recordings_dir / f"{name}_{ts}.mp4")
                    self._recording_writers[name] = {
                        "writer": None,
                        "path": fname,
                        "frame_count": 0,
                        "last_img": None,
                        "start_time": rec_start,
                    }
                    files[name] = fname
            if not files:
                return {"status": "no_cameras_running", "files": {}}
            # Signal pipelines to send every frame
            for handle in self._handles.values():
                if handle.status_dict is not None:
                    handle.status_dict["recording_enabled"] = True
            self._recording = True
            # Rotate the single active tracking JSONL onto a recording-scoped
            # file so the dashboard path matches the file that is actually
            # receiving writes.
            data_path = self._make_tracking_jsonl_path(ts=ts, label="rec")
            with self._jsonl_lock:
                try:
                    self._rotate_tracking_file_locked(data_path, reset_counter=True)
                    self._recording_tracking_path = data_path
                    files["tracking_data"] = data_path
                except Exception as e:
                    logger.warning("Failed to rotate tracking data file: %s", e)
                    files["tracking_data"] = self._tracking_file_path
            self._recording_info = {"start_time": rec_start, "files": files}
            logger.info("Recording started: %s", files)
            return {"status": "recording", "files": files}

    def stop_recording(self) -> dict:
        """停止录像并写入文件，补帧对齐两路视频时长。"""
        with self._recording_lock:
            if not self._recording:
                return {"status": "not_recording", "files": {}}
            self._recording = False
            # Signal pipelines to revert to preview rate
            for handle in self._handles.values():
                if handle.status_dict is not None:
                    handle.status_dict["recording_enabled"] = False
            elapsed = time.time() - self._recording_info.get("start_time", time.time())
            target_frames = int(elapsed * 25.0)  # 目标帧数 = 时长 × 25fps
            # 对齐：补帧到相同目标帧数
            for name, wr_info in self._recording_writers.items():
                writer = wr_info.get("writer")
                last_img = wr_info.get("last_img")
                count = wr_info.get("frame_count", 0)
                if writer is not None and last_img is not None and count < target_frames:
                    pad = target_frames - count
                    for _ in range(pad):
                        writer.write(last_img)
                    logger.info("[%s] Padded %d frames (had %d, target %d)", name, pad, count, target_frames)
            files = {}
            for name, wr_info in self._recording_writers.items():
                writer = wr_info.get("writer")
                if writer is not None:
                    writer.release()
                files[name] = wr_info["path"]
            self._recording_writers.clear()
            # Seal the recording JSONL, then rotate back to a fresh
            # background file so later live traffic does not append into the
            # finished recording session.
            with self._jsonl_lock:
                finished_tracking = self._recording_tracking_path or self._tracking_file_path
                self._flush_tracking_file_locked(force_fsync=True)
                if finished_tracking:
                    self._last_completed_tracking_path = finished_tracking
                    files["tracking_data"] = finished_tracking
                keep_tracking = (
                    self._consumer_thread is not None
                    and self._consumer_thread.is_alive()
                    and not self._stopped.is_set()
                )
                if keep_tracking:
                    bg_path = self._make_tracking_jsonl_path(label="bg")
                    self._rotate_tracking_file_locked(bg_path, reset_counter=True)
                else:
                    self._close_tracking_file_locked(force_fsync=True)
                self._recording_tracking_path = None
            logger.info("Recording stopped (%.1fs, target %d frames), files: %s", elapsed, target_frames, files)
            result = {"status": "stopped", "files": files, "duration_s": round(elapsed, 1)}
            self._recording_info = {}
            return result

    def set_rally_report_interval(self, interval: int) -> dict:
        """Set how many rallies trigger an auto report. 0 = disabled."""
        self._rally_report_interval = max(0, interval)
        return {"interval": self._rally_report_interval}

    def _export_rally(self, rally_result, frames: list) -> None:
        """Export a completed rally to the configured API endpoint (runs in background thread)."""
        try:
            from app.result_exporter import format_rally

            if not frames:
                logger.warning("Rally %d: no frames in snapshot, skipping export", rally_result.rally_id)
                print(f"[DEBUG] Rally {rally_result.rally_id} export 跳过: frames 为空")
                return

            serial_numbers = self.config.serial_numbers
            serial = serial_numbers.get("cam66") or next(iter(serial_numbers.values()), "UNKNOWN")

            endpoint = self.config.export.endpoint
            print(f"[DEBUG] Rally {rally_result.rally_id} 正在 POST → {endpoint}")
            format_rally(rally_result, frames, serial, endpoint)
        except Exception as e:
            logger.warning("Rally export error: %s", e)
            print(f"[DEBUG] Rally export 异常: {e}")

    def _auto_generate_report(self):
        """Auto-generate report from current tracking JSONL."""
        self._last_report_rally_count = self._rally_completed_count
        self.flush_data_file()
        path = self.get_current_jsonl_path()
        if not path:
            return
        try:
            from app.report import generate_report
            result = generate_report(path)
            logger.info("Auto report generated: %s (%d rallies)",
                        result.get("report_name"), self._rally_completed_count)
        except Exception as e:
            logger.warning("Auto report failed: %s", e)

    def flush_data_file(self) -> None:
        """Flush the JSONL tracking data file so report module can read it."""
        with self._jsonl_lock:
            self._flush_tracking_file_locked(force_fsync=True)

    def get_current_jsonl_path(self) -> str | None:
        """Return path to the current tracking JSONL, or most recent."""
        with self._jsonl_lock:
            if self._recording and self._tracking_file_path:
                return self._tracking_file_path
            if self._last_completed_tracking_path and Path(self._last_completed_tracking_path).exists():
                return self._last_completed_tracking_path
            if self._tracking_file_path:
                return self._tracking_file_path
        jsonls = sorted(Path("recordings").glob("tracking_*.jsonl"), reverse=True)
        return str(jsonls[0]) if jsonls else None

    def get_recording_status(self) -> dict:
        import shutil

        with self._ffmpeg_lock:
            if self._ffmpeg_processes or self._ffmpeg_stopping:
                files, segments = self._collect_ffmpeg_segment_snapshot_locked()
                elapsed = time.time() - self._ffmpeg_start_time if self._ffmpeg_start_time else 0.0
                try:
                    free_disk_gb = round(
                        shutil.disk_usage(self._recordings_dir).free / (1024 ** 3), 2
                    )
                except Exception:
                    free_disk_gb = None
                return {
                    "recording": True,
                    "mode": "ffmpeg",
                    "duration_s": round(elapsed, 1),
                    "files": files,
                    "camera_names": list(self._ffmpeg_processes.keys()),
                    "session_dir": self._ffmpeg_session_dir,
                    "segment_seconds": self._FFMPEG_SEGMENT_SECONDS,
                    "segment_counts": {name: len(paths) for name, paths in segments.items()},
                    "free_disk_gb": free_disk_gb,
                    "stopping": self._ffmpeg_stopping,
                    "stop_reason": self._ffmpeg_stop_reason,
                }
        if not self._recording:
            return {"recording": False}
        elapsed = time.time() - self._recording_info.get("start_time", time.time())
        return {
            "recording": True,
            "mode": "opencv",
            "duration_s": round(elapsed, 1),
            "files": self._recording_info.get("files", {}),
        }

    # ------------------------------------------------------------------
    # FFmpeg segmented recording with audio (safer for long sessions)
    # ------------------------------------------------------------------
    _FFMPEG_SEGMENT_SECONDS = 600
    _FFMPEG_MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
    _FFMPEG_MONITOR_INTERVAL_S = 10.0

    def _collect_ffmpeg_segment_snapshot_locked(self) -> tuple[dict[str, str], dict[str, list[str]]]:
        files: dict[str, str] = {}
        segments: dict[str, list[str]] = {}
        for name, info in self._ffmpeg_processes.items():
            session_dir = Path(info["session_dir"])
            paths = sorted(session_dir.glob(f"{name}_*.mkv"))
            str_paths = [str(p) for p in paths]
            segments[name] = str_paths
            files[name] = str_paths[-1] if str_paths else info["pattern"]
        return files, segments

    @staticmethod
    def _resolve_ffmpeg_bin() -> str | None:
        """Find ffmpeg even when the running process has a stale Windows PATH."""
        import shutil

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            return ffmpeg_bin

        path_parts: list[str] = []
        if os.name == "nt":
            try:
                import winreg

                registry_keys = (
                    (winreg.HKEY_CURRENT_USER, r"Environment"),
                    (
                        winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                    ),
                )
                for root, key_path in registry_keys:
                    try:
                        with winreg.OpenKey(root, key_path) as key:
                            value, _value_type = winreg.QueryValueEx(key, "Path")
                            path_parts.extend(str(value).split(os.pathsep))
                    except OSError:
                        continue
            except Exception:
                pass

        registry_path = os.pathsep.join(
            os.path.expandvars(p) for p in path_parts if p
        )
        ffmpeg_bin = shutil.which("ffmpeg", path=registry_path)
        if ffmpeg_bin:
            logger.info("Resolved ffmpeg from Windows environment registry: %s", ffmpeg_bin)
            return ffmpeg_bin

        for candidate in (
            Path(r"D:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"),
            Path(r"C:\msys64\ucrt64\bin\ffmpeg.exe"),
            Path(r"D:\tennis-workspace\ffmpeg\bin\ffmpeg.exe"),
        ):
            if candidate.exists():
                logger.info("Resolved ffmpeg from known local path: %s", candidate)
                return str(candidate)

        return None

    def _monitor_ffmpeg_recording(self) -> None:
        import shutil

        while not self._stopped.is_set():
            with self._ffmpeg_lock:
                if not self._ffmpeg_processes:
                    return
                if self._ffmpeg_stopping:
                    return
                procs = [(name, info["proc"]) for name, info in self._ffmpeg_processes.items()]
            try:
                free_bytes = shutil.disk_usage(self._recordings_dir).free
            except Exception:
                free_bytes = None
            if free_bytes is not None and free_bytes < self._FFMPEG_MIN_FREE_BYTES:
                free_gb = free_bytes / (1024 ** 3)
                logger.warning(
                    "Stopping ffmpeg recording due to low disk space: %.2f GB free",
                    free_gb,
                )
                self.stop_recording_ffmpeg(reason="low_disk")
                return
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    logger.warning("[%s] ffmpeg exited unexpectedly with code %s", name, ret)
                    self.stop_recording_ffmpeg(reason=f"ffmpeg_exit:{name}:{ret}")
                    return
            time.sleep(self._FFMPEG_MONITOR_INTERVAL_S)

    def _stop_recording_ffmpeg_worker(self, reason: str) -> None:
        with self._ffmpeg_lock:
            process_items = list(self._ffmpeg_processes.items())
            elapsed = time.time() - self._ffmpeg_start_time if self._ffmpeg_start_time else 0.0

        for name, info in process_items:
            proc = info["proc"]
            try:
                if proc.stdin is not None:
                    proc.stdin.write(b"q\n")
                    proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            finally:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except Exception:
                    pass
                log_handle = info.get("log_handle")
                if log_handle is not None:
                    try:
                        log_handle.flush()
                        log_handle.close()
                    except Exception:
                        pass

        with self._ffmpeg_lock:
            files, _segments = self._collect_ffmpeg_segment_snapshot_locked()
            self._ffmpeg_processes.clear()
            self._ffmpeg_start_time = 0.0
            self._ffmpeg_session_dir = None
            self._ffmpeg_stopping = False
            self._ffmpeg_stop_thread = None

        for name, latest_path in files.items():
            logger.info("[%s] ffmpeg recording stopped: %s", name, latest_path)
        logger.info("FFmpeg recording stop complete (reason=%s, duration=%.1fs)", reason, elapsed)

    def start_recording_ffmpeg(self, camera_names: list[str] | None = None) -> dict:
        """Start segmented ffmpeg recording (video + optional audio) from RTSP.

        This path is intended for long sessions: ffmpeg writes directly to disk,
        rotates every N minutes, and each completed segment is independently
        playable. That makes it much safer than a single giant MP4 if the
        process crashes or the machine runs out of disk.
        """
        import subprocess

        with self._ffmpeg_lock:
            if self._ffmpeg_processes:
                files, _segments = self._collect_ffmpeg_segment_snapshot_locked()
                return {
                    "status": "already_recording_ffmpeg",
                    "files": files,
                    "camera_names": list(self._ffmpeg_processes.keys()),
                }
            if self._ffmpeg_stopping:
                return {"status": "stopping_ffmpeg"}

            ffmpeg_bin = self._resolve_ffmpeg_bin()
            if not ffmpeg_bin:
                return {"status": "error", "message": "ffmpeg not found in PATH"}

            requested_names: list[str] | None = None
            if camera_names is not None:
                seen: set[str] = set()
                requested_names = []
                for raw_name in camera_names:
                    name = str(raw_name).strip()
                    if name and name not in seen:
                        seen.add(name)
                        requested_names.append(name)
                if not requested_names:
                    return {"status": "error", "message": "no cameras selected"}

            missing_names = [
                name for name in (requested_names or []) if name not in self.config.cameras
            ]
            if missing_names:
                return {
                    "status": "error",
                    "message": "unknown cameras: " + ", ".join(missing_names),
                }

            camera_items = (
                [(name, self.config.cameras[name]) for name in requested_names]
                if requested_names is not None
                else list(self.config.cameras.items())
            )
            camera_items = [
                (name, cam_cfg) for name, cam_cfg in camera_items
                if getattr(cam_cfg, "rtsp_url", None)
            ]
            if not camera_items:
                return {"status": "error", "message": "no selected cameras with rtsp"}

            self._recordings_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = self._recordings_dir / f"ffmpeg_{ts}"
            session_dir.mkdir(parents=True, exist_ok=True)
            files: dict[str, str] = {}
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

            for name, cam_cfg in camera_items:
                rtsp_url = cam_cfg.rtsp_url

                out_pattern = str(session_dir / f"{name}_%Y%m%d_%H%M%S.mkv")
                log_path = session_dir / f"{name}_ffmpeg.log"
                try:
                    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
                except Exception as e:
                    logger.error("[%s] Failed to open ffmpeg log file: %s", name, e)
                    continue

                cmd = [
                    ffmpeg_bin,
                    "-hide_banner",
                    "-loglevel", "warning",
                    "-rtsp_transport", "tcp",
                    "-i", rtsp_url,
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-f", "segment",
                    "-segment_time", str(self._FFMPEG_SEGMENT_SECONDS),
                    "-reset_timestamps", "1",
                    "-strftime", "1",
                    "-segment_format", "matroska",
                    out_pattern,
                ]

                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=log_handle,
                        creationflags=creationflags,
                    )
                    self._ffmpeg_processes[name] = {
                        "proc": proc,
                        "pattern": out_pattern,
                        "session_dir": str(session_dir),
                        "log_path": str(log_path),
                        "log_handle": log_handle,
                    }
                    files[name] = out_pattern
                    logger.info("[%s] ffmpeg segmented recording started: %s", name, out_pattern)
                except Exception as e:
                    logger.error("[%s] ffmpeg failed to start: %s", name, e)
                    try:
                        log_handle.close()
                    except Exception:
                        pass

            if not files:
                return {"status": "error", "message": "no cameras started"}

            self._ffmpeg_start_time = time.time()
            self._ffmpeg_session_dir = str(session_dir)
            self._ffmpeg_stop_reason = None
            self._ffmpeg_stopping = False
            if self._ffmpeg_monitor_thread is None or not self._ffmpeg_monitor_thread.is_alive():
                self._ffmpeg_monitor_thread = threading.Thread(
                    target=self._monitor_ffmpeg_recording,
                    daemon=True,
                    name="ffmpeg-recording-monitor",
                )
                self._ffmpeg_monitor_thread.start()
            return {
                "status": "recording_ffmpeg",
                "files": files,
                "camera_names": list(files.keys()),
                "session_dir": str(session_dir),
                "segment_seconds": self._FFMPEG_SEGMENT_SECONDS,
            }

    def stop_recording_ffmpeg(self, reason: str = "manual") -> dict:
        """Stop segmented ffmpeg recording and finalize current segment."""
        with self._ffmpeg_lock:
            if not self._ffmpeg_processes:
                return {"status": "not_recording_ffmpeg"}
            if self._ffmpeg_stopping:
                files, _segments = self._collect_ffmpeg_segment_snapshot_locked()
                elapsed = time.time() - self._ffmpeg_start_time if self._ffmpeg_start_time else 0.0
                return {
                    "status": "stopping_ffmpeg",
                    "files": files,
                    "duration_s": round(elapsed, 1),
                    "stop_reason": self._ffmpeg_stop_reason or reason,
                }

            self._ffmpeg_stop_reason = reason
            self._ffmpeg_stopping = True
            elapsed = time.time() - self._ffmpeg_start_time if self._ffmpeg_start_time else 0.0
            files, segments = self._collect_ffmpeg_segment_snapshot_locked()
            session_dir = self._ffmpeg_session_dir
            self._ffmpeg_stop_thread = threading.Thread(
                target=self._stop_recording_ffmpeg_worker,
                args=(reason,),
                daemon=True,
                name="ffmpeg-recording-stop",
            )
            self._ffmpeg_stop_thread.start()

            return {
                "status": "stopping_ffmpeg",
                "files": files,
                "segments": segments,
                "session_dir": session_dir,
                "duration_s": round(elapsed, 1),
                "stop_reason": reason,
            }

    def _write_recording_frame(self, name: str, jpeg: bytes) -> None:
        """解码 JPEG 并写入对应 VideoWriter，基于时间戳补帧保证 25fps。"""
        if name not in self._recording_writers:
            return
        wr_info = self._recording_writers[name]
        try:
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            h, w = img.shape[:2]
            if wr_info["writer"] is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                wr_info["writer"] = cv2.VideoWriter(wr_info["path"], fourcc, 25.0, (w, h))
            # 基于时间戳计算应写入的帧数，自动补帧填充间隙
            elapsed = time.time() - wr_info["start_time"]
            expected_frames = int(elapsed * 25.0)
            current_count = wr_info["frame_count"]
            # 如果有间隙，用上一帧填充
            if current_count < expected_frames - 1 and wr_info["last_img"] is not None:
                gap = expected_frames - 1 - current_count
                for _ in range(min(gap, 10)):  # 最多补 10 帧避免卡顿
                    wr_info["writer"].write(wr_info["last_img"])
                    wr_info["frame_count"] += 1
            wr_info["writer"].write(img)
            wr_info["frame_count"] += 1
            wr_info["last_img"] = img
        except Exception as e:
            logger.error("[%s] Recording write error: %s", name, e)

    # ------------------------------------------------------------------
    # Status queries (called from FastAPI)
    def _handle_player_pose(self, msg: dict, cam_positions: dict) -> None:
        """Process a player_pose message from a camera pipeline subprocess.

        Steps:
          1. Project each detected player's foot pixel to court (x, y) via homography.
          2. Filter to this camera's half of the court (derived from camera y position).
          3. Pick the player nearest to the latest 3D ball position.
          4. Store in _latest_player_pose and write a JSONL line.
        """
        cam_name: str = msg["camera_name"]
        detections: list[dict] = msg.get("detections", [])
        if not detections:
            return

        # Retrieve homography transformer (lazy init, one per camera)
        if not hasattr(self, "_player_homographies"):
            self._player_homographies: dict = {}
        if cam_name not in self._player_homographies:
            try:
                cam_cfg = self.config.cameras.get(cam_name)
                if cam_cfg:
                    from app.pipeline.homography import HomographyTransformer
                    self._player_homographies[cam_name] = HomographyTransformer(
                        self.config.homography.path, cam_cfg.homography_key
                    )
            except Exception as e:
                logger.warning("[%s] Failed to init player homography: %s", cam_name, e)
                return
        hom = self._player_homographies.get(cam_name)
        if hom is None:
            return

        # Determine which half this camera covers (sign of camera y position)
        cam_pos = cam_positions.get(cam_name)
        if cam_pos is None:
            return
        cam_y_sign = 1 if cam_pos[1] >= 0 else -1  # +1 → far half (y>0), -1 → near half (y<0)
        half_slack = 0.5  # metres of tolerance around the net

        # Project each player foot to court and filter by half
        candidates = []
        for det in detections:
            foot_px = det.get("foot_px")
            if not foot_px:
                continue
            bbox = det.get("bbox") or []
            hit_anchor_px = None
            hit_anchor_court = None
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                hit_anchor_px = [
                    (x1 + x2) / 2.0,
                    y1 + (y2 - y1) * 0.3,
                ]
            try:
                court_x, court_y = hom.pixel_to_world(foot_px[0], foot_px[1])
            except Exception:
                continue
            if hit_anchor_px is not None:
                try:
                    ax, ay = hom.pixel_to_world(hit_anchor_px[0], hit_anchor_px[1])
                    hit_anchor_court = [round(ax, 3), round(ay, 3)]
                except Exception:
                    hit_anchor_court = None
            # Keep players in this camera's half (+ slack toward net)
            if cam_y_sign < 0 and court_y > half_slack:
                continue
            if cam_y_sign > 0 and court_y < -half_slack:
                continue
            candidates.append({
                **det,
                "court_x": court_x,
                "court_y": court_y,
                "hit_anchor_px": hit_anchor_px,
                "hit_anchor_court": hit_anchor_court,
            })

        if not candidates:
            return

        # Pick player nearest to latest 3D ball position (2D court distance)
        ball = self._latest_3d
        if ball is not None and len(candidates) > 1:
            def _dist(c):
                return (c["court_x"] - ball.x) ** 2 + (c["court_y"] - ball.y) ** 2
            nearest = min(candidates, key=_dist)
            dist_2d = float(_dist(nearest) ** 0.5)
        else:
            nearest = candidates[0]
            dist_2d = float(
                ((nearest["court_x"] - ball.x) ** 2 + (nearest["court_y"] - ball.y) ** 2) ** 0.5
            ) if ball is not None else -1.0

        player_record = {
            "bbox": nearest["bbox"],
            "conf": nearest["conf"],
            "foot_px": nearest["foot_px"],
            "hit_anchor_px": nearest.get("hit_anchor_px"),
            "hit_anchor_court": nearest.get("hit_anchor_court"),
            "foot_court": [round(nearest["court_x"], 3), round(nearest["court_y"], 3)],
            "dist_to_ball_2d": round(dist_2d, 3),
            "keypoints_px": nearest.get("keypoints", []),
        }
        self._latest_player_pose[cam_name] = {
            "timestamp": msg["timestamp"],
            "capture_ts": msg["capture_ts"],
            "frame_id": msg["frame_id"],
            "player": player_record,
        }

        # Write JSONL
        ball_snap = (
            {"x": round(ball.x, 3), "y": round(ball.y, 3), "z": round(ball.z, 3),
             "ts": round(ball.timestamp, 4)}
            if ball is not None else None
        )
        row = {
            "type": "player_pose",
            "camera": cam_name,
            "timestamp": round(msg["timestamp"], 4),
            "capture_ts": round(msg["capture_ts"], 4),
            "ball_3d": ball_snap,
            "player": {
                "bbox": [round(v, 1) for v in player_record["bbox"]],
                "conf": round(player_record["conf"], 3),
                "foot_court": player_record["foot_court"],
                "hit_anchor_px": [
                    round(v, 1) for v in player_record["hit_anchor_px"]
                ] if player_record.get("hit_anchor_px") else None,
                "hit_anchor_court": player_record.get("hit_anchor_court"),
                "dist_to_ball_2d": player_record["dist_to_ball_2d"],
                "keypoints_px": [
                    [round(kp[0], 1), round(kp[1], 1), round(kp[2], 3)]
                    for kp in player_record["keypoints_px"]
                ],
            },
        }
        self._append_tracking_jsonl_row(row, bump_frame_counter=False)

    def _write_tracking_frame(
        self, d1, d2, cam_names, x, y, z, smoothed_pt, bounce, now, capture_ts
    ):
        """Write one JSONL line with per-frame tracking data."""
        def _cam_dict(d):
            if d is None:
                return None
            return {
                "px": round(d.get("pixel_x", 0), 1),
                "py": round(d.get("pixel_y", 0), 1),
                "conf": round(d.get("blob_sum", d.get("confidence", 0)), 2),
                "wx": round(d.get("x", 0), 3),
                "wy": round(d.get("y", 0), 3),
            }

        state = "tracking"
        row = {
            "ts": round(now, 4),
            "capture_ts": round(capture_ts, 4),
            cam_names[0]: _cam_dict(d1),
            cam_names[1]: _cam_dict(d2),
        }

        if x is not None:
            row["3d"] = {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}
        else:
            row["3d"] = None
            state = "no_match"

        if smoothed_pt is not None:
            row["smoothed"] = {
                "x": round(smoothed_pt["x"], 3),
                "y": round(smoothed_pt["y"], 3),
                "z": round(smoothed_pt["z"], 3),
            }

        if bounce is not None:
            state = "bounce"
            row["bounce"] = bounce.to_dict() if hasattr(bounce, "to_dict") else dict(bounce)

        row["state"] = state

        self._append_tracking_jsonl_row(row, bump_frame_counter=True)

    # ------------------------------------------------------------------
    def get_pipeline_status(self, name: str) -> PipelineStatus:
        handle = self._handles[name]
        if handle.status_dict is not None:
            return PipelineStatus(
                name=name,
                state=handle.status_dict.get("state", "stopped"),
                fps=handle.status_dict.get("fps", 0.0),
                last_detection_time=handle.status_dict.get("last_detection_time"),
                error_msg=handle.status_dict.get("error_msg") or None,
            )
        return PipelineStatus(name=name, state="stopped")

    def get_system_status(self) -> SystemStatus:
        pipelines = {n: self.get_pipeline_status(n) for n in self._handles}
        # Extract candidates from latest detections for minimap visualization
        det_summary = {}
        for cam_name, det in self._latest_detections.items():
            if det is None:
                continue
            candidates = det.get("candidates", [])
            det_summary[cam_name] = {
                "x": det.get("x"),
                "y": det.get("y"),
                "pixel_x": det.get("pixel_x"),
                "pixel_y": det.get("pixel_y"),
                "timestamp": det.get("timestamp"),
                "capture_ts": det.get("capture_ts"),
                "frame_index": det.get("frame_index"),
                "confidence": det.get("confidence", det.get("blob_sum")),
                "candidates": [
                    {"x": float(c["x"]), "y": float(c["y"]),
                     "pixel_x": float(c.get("pixel_x", 0)),
                     "pixel_y": float(c.get("pixel_y", 0)),
                     "blob_sum": float(c.get("blob_sum", 1.0))}
                    for c in candidates
                ],
            }
        return SystemStatus(
            pipelines=pipelines,
            triangulation_active=self._triangulation_active,
            latest_ball_3d=self._latest_3d,
            analytics=self.get_live_analytics(),
            latest_detections=det_summary or None,
        )

    def get_latest_3d(self) -> Optional[BallPosition3D]:
        return self._latest_3d

    def get_latest_detection(self, name: str) -> Optional[dict]:
        return self._latest_detections.get(name)

    def _run_live_bounce_detectors_locked(
        self,
        pt: dict,
        cam_dets: dict,
    ) -> tuple[dict | None, Any]:
        """Run smoothing + bounce detectors for one realtime 3D point.

        Must be called with ``self._analytics_lock`` held.
        Returns ``(smoothed_pt, hbounce)`` where ``hbounce`` is already
        gated by ``_bounce_detection_enabled``.
        """
        if self._bounce_detection_enabled:
            self._bounce_detector.update(pt)
        if self._bounce_detection_enabled and hasattr(self._bounce_detector, "pop_pending"):
            for pb in self._bounce_detector.pop_pending():
                self._peak_bounces_eval.append(pb.to_dict())
                self._debug_record_peak_bounce(pb)
            if len(self._peak_bounces_eval) > 100:
                self._peak_bounces_eval = self._peak_bounces_eval[-100:]

        smoothed_pt, smoothed_cam_dets = self._smooth_latest(pt, cam_dets)
        if self._sg_switched_to_midpoint:
            self._hybrid_bounce.reset()
            self._sg_switched_to_midpoint = False
        hbounce = (
            self._hybrid_bounce.update(smoothed_pt, smoothed_cam_dets or {})
            if self._bounce_detection_enabled and smoothed_pt is not None
            else None
        )
        self._last_smoothed_cam_dets = smoothed_cam_dets or {}
        return smoothed_pt, hbounce

    def get_live_analytics(self) -> dict:
        """Return current live bounce/rally state for the dashboard.

        Rally tracking is now done by the simple ``RallyTracker`` only —
        net crossings + timeout, no serve rules, no end-reason classification.
        RallyStateMachine was removed: the rule-based machine wasn't GT-
        validated, and its complex output (PENDING / SERVING / DOUBLE_FAULT /
        LET ...) was noisy on realtime data.
        """
        with self._analytics_lock:
            return {
                "rally_state": self._rally_tracker.get_state().to_dict(),
                "completed_rallies": self._rally_tracker.get_completed_rallies(),
                "recent_bounces": list(self._live_bounces),
                "total_bounces": self._total_live_bounces,
                "ws_pending_bounces": len(self._ws_bounce_queue),
                "last_frame_speed_kmh": int(round(float(self._last_frame_speed_kmh or 0.0))),
                "latest_net_crossing": dict(self._latest_net_crossing) if self._latest_net_crossing else None,
                "recent_hits": list(self._live_hits),
                "raw_bounce_candidate_count": self._hit_bounce_refiner.get_stats().get(
                    "raw_bounce_candidate_count", 0
                ),
                "suppressed_bounces_by_hit": self._hit_bounce_refiner.get_stats().get(
                    "suppressed_bounces_by_hit", 0
                ),
                "hit_bounce_refiner_stats": self._hit_bounce_refiner.get_stats(),
                # Peak sidecar (eval only)
                "peak_bounces_eval": list(self._peak_bounces_eval[-10:]),
                # Post-filter telemetry — count per rejection reason plus "accepted".
                "post_filter_stats": dict(self._post_filter_stats),
            }

    @staticmethod
    def _normalize_live_bounce_dict(
        bounce,
        *,
        fallback_ts: float | None = None,
        fallback_speed_kmh: float = 0.0,
    ) -> dict:
        """Return a bounce dict with stable fields for minimap, API and 3D push."""
        if isinstance(bounce, dict):
            bd = dict(bounce)
        else:
            bd = bounce.to_dict()
        ts = bd.get("timestamp")
        if ts is None:
            bd["timestamp"] = float(fallback_ts if fallback_ts is not None else time.time())
        speed = bd.get("speed_kmh")
        if speed is None:
            bd["speed_kmh"] = int(round(float(fallback_speed_kmh)))
        else:
            try:
                speed_val = float(speed)
            except Exception:
                speed_val = 0.0
            bd["speed_kmh"] = int(round(speed_val))
        return bd

    def _enqueue_ws_bounce_locked(self, bd: dict) -> None:
        """Queue a bounce for 3D push from the same live event source.

        Note: the remote 3D receiver expects decimeter-like court units
        (`x * 10`, `y * 10`). Minimap/API keep raw court coordinates in
        meters; only the WebSocket egress applies this protocol transform.
        """
        if not self._ws_enabled:
            return
        bx, by = bd.get("x"), bd.get("y")
        if bx is None or by is None:
            return
        ts = bd.get("timestamp")
        if ts is None:
            ts = time.time()
        speed = bd.get("speed_kmh", 0)
        try:
            speed_val = int(round(float(speed or 0)))
        except Exception:
            speed_val = 0
        ws_x = round(float(bx) * 10.0, 4)
        ws_y = round(float(by) * 10.0, 4)
        self._ws_bounce_queue.append({
            "x": ws_x,
            "y": ws_y,
            "raw_x": round(float(bx), 4),
            "raw_y": round(float(by), 4),
            "speed": speed_val,
            "timestamp": int(round(float(ts) * 1000)),
        })

    def _build_hit_bounce_player_snapshot(self, now: float) -> list[dict]:
        """Return current player anchors in the shape expected by the refiner."""
        players = []
        for cam_name, pose in self._latest_player_pose.items():
            player = dict(pose.get("player") or {})
            foot = player.get("foot_court")
            if not foot or len(foot) < 2:
                continue
            try:
                foot_y = float(foot[1])
            except (TypeError, ValueError):
                continue
            timestamp = float(pose.get("timestamp", now) or now)
            if now - timestamp > 2.0:
                continue
            players.append({
                **player,
                "camera_name": cam_name,
                "timestamp": timestamp,
                "capture_ts": pose.get("capture_ts"),
                "frame_index": pose.get("frame_id"),
                "side": "near" if foot_y < 0 else "far",
            })
        return players

    @staticmethod
    def _bounce_dict_to_event(bd: dict) -> BounceEvent:
        return BounceEvent(
            x=float(bd.get("x", 0.0) or 0.0),
            y=float(bd.get("y", 0.0) or 0.0),
            z=float(bd.get("z", 0.0) or 0.0),
            timestamp=float(bd.get("timestamp", time.time()) or time.time()),
            capture_ts=float(bd.get("capture_ts", bd.get("timestamp", time.time())) or time.time()),
            in_court=bool(bd.get("in_court", False)),
            frame_index=bd.get("frame_index") or bd.get("frame"),
            confidence=float(bd.get("confidence", 1.0) or 1.0),
            source_camera=str(bd.get("source_camera", bd.get("refiner_source", "3d"))),
            side=str(bd.get("side", "")),
            cam_pixels=dict(bd.get("cam_pixels") or {}),
        )

    def _record_live_hit_locked(self, hit: dict) -> None:
        """Publish one HIT to realtime analytics/debug/report buffers only."""
        self._live_hits.append(dict(hit))
        if len(self._live_hits) > self._LIVE_BOUNCE_HISTORY_LIMIT:
            self._live_hits = self._live_hits[-self._LIVE_BOUNCE_HISTORY_LIMIT:]
        self._debug_record_hit(hit)

    def _mark_rally_buffer_event_locked(self, event: dict, event_type: str) -> None:
        """Backfill delayed refiner events onto their original raw-buffer frame."""
        frame = event.get("frame_index", event.get("frame"))
        if frame is None:
            return
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            return
        target = None
        for row in reversed(self._rally_raw_buffer):
            row_frame = row.get("frame_index")
            if row_frame is None:
                continue
            if int(row_frame) == frame:
                target = row
                break
            if frame - int(row_frame) > 3:
                break
        if target is None:
            return
        event_ball = {
            "x": event.get("x"),
            "y": event.get("y"),
            "z": event.get("z", target.get("ball", {}).get("z", 0.0)),
        }
        if event_type == "hit":
            target["is_hit"] = True
            target["hit_event"] = dict(event)
        else:
            target["is_bounce"] = True
            target["bounce_event"] = dict(event)
        target["event_ball"] = event_ball

    def _record_live_bounce_locked(self, bd: dict, *, debug_source=None) -> None:
        """Publish one accepted bounce to every realtime consumer from one source dict."""
        self._total_live_bounces += 1
        bd["sequence"] = self._total_live_bounces
        self._live_bounces.append(bd)
        if len(self._live_bounces) > self._LIVE_BOUNCE_HISTORY_LIMIT:
            self._live_bounces = self._live_bounces[-self._LIVE_BOUNCE_HISTORY_LIMIT:]
        self._debug_record_bounce(debug_source if debug_source is not None else bd)
        self._enqueue_ws_bounce_locked(bd)

    def _gate_live_bounce_candidate_locked(
        self,
        bd: dict,
        *,
        now: float,
        match_speed: bool,
    ) -> dict | None:
        """Apply the shared realtime bounce gate to one candidate event."""
        consumed_nc = None
        if match_speed:
            event_ts = float(bd.get("timestamp", now))
            for nc in reversed(self._net_crossings):
                if nc.get("_used"):
                    continue
                age = event_ts - nc["timestamp"]
                if age < 0:
                    continue
                if age < 3.0:
                    bd["speed_kmh"] = nc["speed_kmh"]
                    bd["speed_direction"] = nc["direction"]
                    nc["_used"] = True
                    consumed_nc = nc
                    break
                break

        if self._is_duplicate_bounce(bd):
            self._post_filter_stats["duplicate"] += 1
            if consumed_nc is not None:
                consumed_nc["_used"] = False
            return None

        ok, reason = self._post_filter_bounce(bd)
        self._post_filter_stats[reason] += 1
        if not ok:
            if consumed_nc is not None:
                consumed_nc["_used"] = False
            return None
        return bd

    def _estimate_speed_kmh_from_window(self) -> float | None:
        """Estimate horizontal ball speed from a short recent 3D window.

        Uses capture timestamps instead of consumer wall-clock time and fits
        x(t), y(t) with local linear regression on the latest continuous
        segment. This is more stable than two-point finite differences when
        tracker noise or queue jitter perturb individual frames.
        """
        pts = list(self._speed_points)
        if len(pts) < self._SPEED_MIN_POINTS:
            return None

        # Keep only the latest continuous tail. Large timestamp/frame gaps
        # usually mean missed matches or queue jitter across separate shots.
        tail = [pts[-1]]
        for prev in reversed(pts[:-1]):
            dt = tail[0]["capture_ts"] - prev["capture_ts"]
            frame_gap = tail[0].get("frame_index", 0) - prev.get("frame_index", 0)
            if dt <= 0 or dt > self._SPEED_MAX_GAP_S or frame_gap > self._SPEED_MAX_FRAME_GAP:
                break
            tail.insert(0, prev)
            if len(tail) >= self._SPEED_FIT_WINDOW:
                break

        if len(tail) < self._SPEED_MIN_POINTS:
            return None

        # Deduplicate / enforce increasing capture_ts.
        filtered = []
        last_ts = None
        for p in tail:
            ts = float(p["capture_ts"])
            if last_ts is not None and ts - last_ts <= 1e-4:
                continue
            filtered.append(p)
            last_ts = ts
        if len(filtered) < self._SPEED_MIN_POINTS:
            return None

        ts = np.array([float(p["capture_ts"]) for p in filtered], dtype=float)
        xs = np.array([float(p["x"]) for p in filtered], dtype=float)
        ys = np.array([float(p["y"]) for p in filtered], dtype=float)

        span = float(ts[-1] - ts[0])
        if span <= 1e-3:
            return None

        t = ts - ts.mean()

        def _fit_speed(t_arr, x_arr, y_arr):
            cx = np.polyfit(t_arr, x_arr, 1)
            cy = np.polyfit(t_arr, y_arr, 1)
            pred_x = np.polyval(cx, t_arr)
            pred_y = np.polyval(cy, t_arr)
            residual = np.hypot(x_arr - pred_x, y_arr - pred_y)
            speed = (float(cx[0]) ** 2 + float(cy[0]) ** 2) ** 0.5 * 3.6
            return speed, residual

        speed_kmh, residual = _fit_speed(t, xs, ys)

        # If the newest point is far off the local trend, treat this estimate
        # as untrustworthy rather than emitting a spike.
        if residual[-1] > self._SPEED_MAX_RESIDUAL_M:
            return None

        # One-pass interior outlier rejection for tracker jumps.
        if len(filtered) > self._SPEED_MIN_POINTS:
            worst_idx = int(np.argmax(residual))
            if 0 < worst_idx < len(filtered) - 1 and residual[worst_idx] > self._SPEED_MAX_RESIDUAL_M:
                keep = np.ones(len(filtered), dtype=bool)
                keep[worst_idx] = False
                speed_kmh, residual = _fit_speed(t[keep], xs[keep], ys[keep])
                if residual[-1] > self._SPEED_MAX_RESIDUAL_M:
                    return None

        if not np.isfinite(speed_kmh):
            return None
        if speed_kmh < 0 or speed_kmh > self._MAX_PHYSICAL_SPEED:
            return None
        return float(speed_kmh)

    # ------------------------------------------------------------------
    # Bounce deduplication (PeakBounceDetector and Hybrid/Enhanced fire
    # independently; filter near-duplicate reports of the same physical event).
    # ------------------------------------------------------------------
    _BOUNCE_DEDUP_DT = 0.7    # seconds: detectors can lag each other ~0.5s
    _BOUNCE_DEDUP_DIST = 1.0  # meters: same physical bounce should be < 1m apart

    def _is_duplicate_bounce(self, bd: dict) -> bool:
        """True if `bd` looks like a near-duplicate of a recently-recorded bounce."""
        ts = bd.get("timestamp")
        if ts is None or not self._live_bounces:
            return False
        bx, by = bd.get("x"), bd.get("y")
        if bx is None or by is None:
            return False
        for prev in reversed(self._live_bounces[-5:]):
            dt = ts - prev.get("timestamp", 0)
            if dt > self._BOUNCE_DEDUP_DT:
                break  # older than window; reverse order → no earlier ones matter
            if dt < 0:
                continue
            dx = bx - prev.get("x", 0)
            dy = by - prev.get("y", 0)
            if (dx * dx + dy * dy) ** 0.5 < self._BOUNCE_DEDUP_DIST:
                return True
        return False

    # ------------------------------------------------------------------
    # Bounce precision post-filter (applied AFTER Hybrid + dedup, BEFORE
    # any production consumer sees the event). See
    # memory/project_bounce_architecture.md "Next-step roadmap A" for the
    # 4 candidate filters — v1 ships F1 + F2. F3 (speed/height continuity)
    # and F4 (net-crossing context) are parameterized but disabled by
    # default; flip the toggles below after side-by-side eval.
    # ------------------------------------------------------------------
    _POSTFILT_MIN_INTERVAL_S = 0.6      # F2: above Hybrid's ~0.48s internal cooldown
    _POSTFILT_MIN_INTERVAL_DIST_M = 2.0  # only near-space repeats count as spam
    _POSTFILT_NET_CTX_WINDOW_S = 3.0    # F4: net crossing must be within this
    # F1 (rally-state gating) removed with RallyStateMachine — no rally-sm,
    # no state to gate on. F3 / F4 remain parameterized but off by default.
    _POSTFILT_F2_ENABLED = True
    _POSTFILT_F3_ENABLED = False
    _POSTFILT_F4_ENABLED = False

    def _post_filter_bounce(self, bd: dict) -> tuple[bool, str]:
        """Run enabled precision filters on an already-Hybrid-approved,
        non-duplicate bounce. Return (accepted, reason). Reason names are
        stable strings so _post_filter_stats can aggregate them.
        """
        # F1 removed with RallyStateMachine (rally-state gating no longer
        # available; dead-ball bounces are allowed through and filtered on
        # the dashboard side if needed).

        # F2: minimum interval since last accepted bounce. Tighter than the
        # dedup gate (which only catches near-simultaneous dual reports of
        # the *same* physical bounce). Real rally bounces are usually > 0.4s
        # apart; anything tighter is likely a second spurious report.
        #
        # Do not require same_side here: if landing refinement / homography
        # jitter nudges one report across NET_Y, the side label flips and a
        # true duplicate would slip through.
        if self._POSTFILT_F2_ENABLED and self._live_bounces:
            ts = bd.get("timestamp")
            if ts is not None:
                last = self._live_bounces[-1]
                dt = ts - last.get("timestamp", 0)
                if 0 <= dt < self._POSTFILT_MIN_INTERVAL_S:
                    bx, by = bd.get("x"), bd.get("y")
                    lx, ly = last.get("x"), last.get("y")
                    if None not in (bx, by, lx, ly):
                        dx = bx - lx
                        dy = by - ly
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist < self._POSTFILT_MIN_INTERVAL_DIST_M:
                            return False, "f2_min_interval"

        # F3: speed/height continuity — placeholder. Hybrid's internal
        # v_window + min_speed already cover the primary signal; a more
        # sophisticated check (e.g. require z to rise in the next ~5
        # frames) would need read-ahead which the streaming detector
        # doesn't provide. Revisit if F1+F2 aren't enough.
        # if self._POSTFILT_F3_ENABLED:
        #     pass

        # F4: net-crossing context — require a net crossing within the
        # last N seconds. Disabled by default because slow serves or
        # drop-shots may not produce a _net_crossings entry (speed gate
        # SPEED_MIN filters out < 30 km/h crossings), which would drop
        # legitimate slow-play bounces.
        if self._POSTFILT_F4_ENABLED:
            ts = bd.get("timestamp")
            if ts is None or not self._net_crossings:
                return False, "f4_no_net_crossing"
            recent = self._net_crossings[-1].get("timestamp", 0)
            if ts - recent > self._POSTFILT_NET_CTX_WINDOW_S:
                return False, "f4_stale_net_crossing"

        return True, "accepted"

    # ------------------------------------------------------------------
    # Debug output for GT comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _new_debug_data():
        return {
            "detections": {},       # cam_name -> [{frame, pixel_x, pixel_y, confidence, world_x, world_y, n_candidates}]
            "trajectory": [],       # [{frame, x, y, z, ray_dist, px66, py66, px68, py68, world66, world68}]
            "bounces": [],          # Hybrid (production): [{frame, z, x, y, in_court}]
            "hits": [],             # Refiner HIT events
            "peak_bounces": [],     # Peak (eval-only sidecar): same schema as bounces
            "frame_counter": 0,
        }

    def _debug_record_detection(self, cam: str, det: dict, frame_idx: int = None):
        """Record a per-camera detection for debug output."""
        fi = frame_idx if frame_idx is not None else self._debug_data["frame_counter"]
        self._debug_data["detections"].setdefault(cam, []).append({
            "frame": fi,
            "pixel_x": round(det.get("pixel_x", 0), 1),
            "pixel_y": round(det.get("pixel_y", 0), 1),
            "confidence": round(det.get("confidence", det.get("blob_sum", 0)), 2),
            "world_x": round(det.get("x", 0), 4),
            "world_y": round(det.get("y", 0), 4),
            "n_candidates": len(det.get("candidates", [])),
        })

    def _debug_record_3d(self, frame_idx, x, y, z, ray_dist,
                         d1: dict, d2: dict, cam1: str, cam2: str):
        """Record a triangulated 3D point for debug output."""
        self._debug_data["trajectory"].append({
            "frame": frame_idx,
            "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
            "ray_dist": round(ray_dist, 4) if ray_dist else 0,
            f"px{cam1[-2:]}": round(d1.get("pixel_x", 0), 1),
            f"py{cam1[-2:]}": round(d1.get("pixel_y", 0), 1),
            f"px{cam2[-2:]}": round(d2.get("pixel_x", 0), 1),
            f"py{cam2[-2:]}": round(d2.get("pixel_y", 0), 1),
            f"world{cam1[-2:]}": [round(d1.get("x", 0), 4), round(d1.get("y", 0), 4)],
            f"world{cam2[-2:]}": [round(d2.get("x", 0), 4), round(d2.get("y", 0), 4)],
        })

    @staticmethod
    def _bounce_to_debug_row(bounce) -> dict:
        """Convert a BounceEvent or bounce-dict to the shared debug row schema."""
        if isinstance(bounce, dict):
            return {
                "frame": bounce.get("frame") or bounce.get("frame_index"),
                "z": round(bounce.get("z", 0), 4),
                "x": round(bounce.get("x", 0), 4),
                "y": round(bounce.get("y", 0), 4),
                "in_court": bool(bounce.get("in_court", False)),
            }
        return {
            "frame": bounce.frame_index,
            "z": round(float(bounce.z), 4),
            "x": round(float(bounce.x), 4),
            "y": round(float(bounce.y), 4),
            "in_court": bool(bounce.in_court),
        }

    def _debug_record_bounce(self, bounce):
        """Record a Hybrid (production) bounce for debug output."""
        self._debug_data["bounces"].append(self._bounce_to_debug_row(bounce))

    def _debug_record_hit(self, hit: dict):
        """Record one refiner HIT event for debug output."""
        self._debug_data.setdefault("hits", []).append({
            "frame": hit.get("frame_index") or hit.get("frame"),
            "x": round(float(hit.get("x", 0.0) or 0.0), 4),
            "y": round(float(hit.get("y", 0.0) or 0.0), 4),
            "source": hit.get("source"),
            "side": hit.get("side"),
            "distance": hit.get("distance"),
            "distance_unit": hit.get("distance_unit"),
        })

    def _debug_record_peak_bounce(self, bounce):
        """Record a Peak (eval-only) bounce in a separate bucket so it
        never mixes with the production ``bounces`` list when diffing
        debug_output against GT or Hybrid."""
        self._debug_data.setdefault("peak_bounces", []).append(
            self._bounce_to_debug_row(bounce)
        )

    def save_debug_output(self) -> str:
        """Save accumulated debug data to timestamped directory. Returns path."""
        import datetime as dt
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self._debug_dir / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        # Config
        cfg = {
            "model": self.config.model.model_dump(),
            "cameras": {n: c.model_dump() for n, c in self.config.cameras.items()},
            "camera_positions": self._get_camera_positions(),
            "homography_path": self.config.homography.path,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

        # Per-camera detections
        for cam, dets in self._debug_data["detections"].items():
            with open(out_dir / f"detections_{cam}.json", "w") as f:
                json.dump(dets, f, indent=2)

        # 3D trajectory
        with open(out_dir / "trajectory_3d.json", "w") as f:
            json.dump(self._debug_data["trajectory"], f, indent=2)

        # Bounces — production (Hybrid) and eval sidecar (Peak) kept separate
        with open(out_dir / "bounces.json", "w") as f:
            json.dump(self._debug_data["bounces"], f, indent=2)
        peak_bounces = self._debug_data.get("peak_bounces", [])
        with open(out_dir / "peak_bounces.json", "w") as f:
            json.dump(peak_bounces, f, indent=2)
        hits = self._debug_data.get("hits", [])
        with open(out_dir / "hits.json", "w") as f:
            json.dump(hits, f, indent=2)

        # Summary
        summary = {
            "total_detections": {cam: len(d) for cam, d in self._debug_data["detections"].items()},
            "trajectory_points": len(self._debug_data["trajectory"]),
            "bounces": len(self._debug_data["bounces"]),
            "hits": len(hits),
            "bounce_in": sum(1 for b in self._debug_data["bounces"] if b.get("in_court")),
            "bounce_out": sum(1 for b in self._debug_data["bounces"] if not b.get("in_court")),
            "peak_bounces_eval": len(peak_bounces),
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Debug output saved to %s (%d traj points, %d bounces)",
                     out_dir, summary["trajectory_points"], summary["bounces"])

        # Reset for next session
        self._debug_data = self._new_debug_data()
        return str(out_dir)

    def _init_homographies(self):
        """Lazy-init homography matrices from homography_matrices.json."""
        if self._blob_homographies:
            return
        import json as _json
        hpath = self.config.homography.path
        try:
            with open(hpath, "r", encoding="utf-8") as f:
                hdata = _json.load(f)
            for cam_name in self.config.cameras:
                key = self.config.cameras[cam_name].homography_key
                if key and key in hdata:
                    H = np.array(hdata[key], dtype=np.float64)
                    self._blob_homographies[cam_name] = H
            logger.info("Loaded homographies for %s", list(self._blob_homographies.keys()))
        except Exception as e:
            logger.error("Failed to load homographies: %s", e)

    def _run_tracker_pipeline(self, cam1: str, cam2: str, cam_positions: dict):
        """Run the full track-first-triangulate-later pipeline on accumulated blobs.

        Steps from research (tracker_3d.py + bounce_detector.py):
        1. track_single_camera() per camera
        2. match_and_triangulate() cross-camera
        3. detect_bounces() / detect_events() on 3D trajectory
        """
        from app.pipeline.tracker import track_single_camera, match_and_triangulate
        from app.pipeline.bounce_detect import detect_bounces

        self._init_homographies()
        H1 = self._blob_homographies.get(cam1)
        H2 = self._blob_homographies.get(cam2)
        if H1 is None or H2 is None:
            return

        buf1 = self._blob_buffers.get(cam1, {})
        buf2 = self._blob_buffers.get(cam2, {})
        ts_map1 = self._blob_capture_ts_by_frame.get(cam1, {})
        ts_map2 = self._blob_capture_ts_by_frame.get(cam2, {})
        if not buf1 or not buf2:
            return

        now = time.time()

        # Step 1: Single-camera tracking
        tracks1 = track_single_camera(buf1, max_pixel_dist=80, max_gap=3, min_len=10)
        tracks2 = track_single_camera(buf2, max_pixel_dist=80, max_gap=3, min_len=10)

        if not tracks1 or not tracks2:
            return

        # Step 2: Cross-camera matching + 3D triangulation
        pos1 = cam_positions.get(cam1)
        pos2 = cam_positions.get(cam2)
        if not pos1 or not pos2:
            return

        matched = match_and_triangulate(
            tracks1, tracks2, H1, H2, pos1, pos2,
            max_ray_dist=1.0, min_overlap=10, max_tracks=50,
        )

        if not matched:
            return

        # Best trajectory
        best = matched[0]["trajectory"]
        # (frame, x, y, z, px1, py1, px2, py2, ray_dist)

        frame_capture_ts: dict[int, float] = {}
        for pt in best:
            fi = int(pt[0])
            ts_candidates = []
            if fi in ts_map1:
                ts_candidates.append(float(ts_map1[fi]))
            if fi in ts_map2:
                ts_candidates.append(float(ts_map2[fi]))
            if ts_candidates:
                frame_capture_ts[fi] = float(sum(ts_candidates) / len(ts_candidates))

        # Emit new 3D points
        new_refiner_points = []
        for pt in best:
            fi = pt[0]
            if fi in self._emitted_3d_frames:
                continue
            self._emitted_3d_frames.add(fi)

            x, y, z = pt[1], pt[2], pt[3]
            self._latest_3d = BallPosition3D(x=x, y=y, z=z)

            d1 = {"pixel_x": pt[4], "pixel_y": pt[5], "x": x, "y": y}
            d2 = {"pixel_x": pt[6], "pixel_y": pt[7], "x": x, "y": y}
            self._debug_record_3d(fi, x, y, z, pt[8], d1, d2, cam1, cam2)

            self._latest_detections[cam1] = {
                "camera_name": cam1, "pixel_x": pt[4], "pixel_y": pt[5],
                "x": x, "y": y, "timestamp": now,
                "capture_ts": frame_capture_ts.get(int(fi), now),
            }
            self._latest_detections[cam2] = {
                "camera_name": cam2, "pixel_x": pt[6], "pixel_y": pt[7],
                "x": x, "y": y, "timestamp": now,
                "capture_ts": frame_capture_ts.get(int(fi), now),
            }
            point_capture_ts = frame_capture_ts.get(int(fi), now)
            new_refiner_points.append((
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "timestamp": point_capture_ts,
                    "capture_ts": point_capture_ts,
                    "frame_index": int(fi),
                },
                {
                    cam1: {"pixel_x": pt[4], "pixel_y": pt[5], "world_x": x, "world_y": y},
                    cam2: {"pixel_x": pt[6], "pixel_y": pt[7], "world_x": x, "world_y": y},
                },
            ))

        # Step 3: Bounce detection on trajectory
        # Convert to tuple format: (frame, x, y, z, ray_dist)
        traj_tuples = [(pt[0], pt[1], pt[2], pt[3], pt[8]) for pt in best]
        bounces = detect_bounces(traj_tuples)

        with self._analytics_lock:
            for point, cam_dets in new_refiner_points:
                refiner_result = self._hit_bounce_refiner.update(
                    point,
                    players=self._build_hit_bounce_player_snapshot(now),
                    net_crossing=dict(self._latest_net_crossing) if self._latest_net_crossing else None,
                    cam_dets=cam_dets,
                    now=now,
                )
                for hit in refiner_result.get("new_hits", []):
                    self._record_live_hit_locked(hit)
                for final_bd in refiner_result.get("new_final_bounces", []):
                    gated_bd = self._gate_live_bounce_candidate_locked(
                        final_bd,
                        now=now,
                        match_speed=False,
                    )
                    if gated_bd is None:
                        continue
                    gated_bd = self._normalize_live_bounce_dict(
                        gated_bd,
                        fallback_ts=now,
                        fallback_speed_kmh=0,
                    )
                    self._record_live_bounce_locked(
                        gated_bd,
                        debug_source=self._bounce_dict_to_event(gated_bd),
                    )

            for b in bounces:
                if b["frame"] in self._emitted_bounce_frames:
                    continue
                self._emitted_bounce_frames.add(b["frame"])
                bounce_capture_ts = frame_capture_ts.get(int(b["frame"]), now)
                bd = self._normalize_live_bounce_dict(
                    b,
                    fallback_ts=bounce_capture_ts,
                    fallback_speed_kmh=0,
                )
                bd["capture_ts"] = float(
                    bd["capture_ts"] if bd.get("capture_ts") is not None
                    else bd["timestamp"] if bd.get("timestamp") is not None
                    else bounce_capture_ts
                )
                bd["detect_delay"] = round(now - float(bd["capture_ts"]), 2)
                refiner_result = self._hit_bounce_refiner.update(
                    None,
                    raw_bounce=bd,
                    now=now,
                )
                for hit in refiner_result.get("new_hits", []):
                    self._record_live_hit_locked(hit)
                for final_bd in refiner_result.get("new_final_bounces", []):
                    accepted_bd = self._gate_live_bounce_candidate_locked(
                        final_bd,
                        now=now,
                        match_speed=False,
                    )
                    if accepted_bd is None:
                        continue
                    accepted_bd = self._normalize_live_bounce_dict(
                        accepted_bd,
                        fallback_ts=now,
                        fallback_speed_kmh=0,
                    )
                    self._record_live_bounce_locked(
                        accepted_bd,
                        debug_source=self._bounce_dict_to_event(accepted_bd),
                    )
                    logger.info(
                        "Bounce: frame=%d z=%.3f (%.2f, %.2f) %s",
                        accepted_bd.get("frame_index", b["frame"]),
                        accepted_bd.get("z", b["z"]),
                        accepted_bd.get("x", b["x"]),
                        accepted_bd.get("y", b["y"]),
                        "IN" if accepted_bd.get("in_court") else "OUT",
                    )

    def _smooth_latest(self, pt: dict, cam_dets: dict | None = None) -> tuple[dict | None, dict | None]:
        """Add a raw 3D point to the SG buffer and return a smoothed point.

        Matches offline ``smooth_trajectory_sg()`` logic: applies Savitzky-Golay
        filter to recent continuous points and returns the smoothed value at the
        midpoint of the window (best smoothing quality).

        Returns ``(smoothed_pt, aligned_cam_dets)`` where ``aligned_cam_dets``
        comes from the same source frame as the emitted smoothed point.

        During warm-up / short-segment periods we fall back to the raw point.
        When SG first becomes stable enough to emit midpoint frames, the
        caller resets Hybrid once so we don't mix earlier raw timestamps with
        later midpoint timestamps inside the same Hybrid buffer.
        """
        from scipy.signal import savgol_filter

        self._sg_buffer.append({"pt": pt, "cam": cam_dets or {}})
        if len(self._sg_buffer) > 60:
            self._sg_buffer = self._sg_buffer[-60:]

        buf = self._sg_buffer
        n = len(buf)
        if n < self._sg_window:
            self._sg_midpoint_mode = False
            return pt, cam_dets

        # Find the latest continuous segment using capture time plus frame gap.
        seg_start = n - 1
        for i in range(n - 2, -1, -1):
            t_prev = float(buf[i]["pt"].get("capture_ts", buf[i]["pt"]["timestamp"]))
            t_next = float(buf[i + 1]["pt"].get("capture_ts", buf[i + 1]["pt"]["timestamp"]))
            dt = t_next - t_prev
            fi_prev = buf[i]["pt"].get("frame_index")
            fi_next = buf[i + 1]["pt"].get("frame_index")
            frame_gap_bad = (
                fi_prev is not None
                and fi_next is not None
                and (fi_next - fi_prev < 0 or fi_next - fi_prev > self._sg_max_gap)
            )
            if dt < 0 or dt > self._sg_max_gap_s or frame_gap_bad:
                break
            seg_start = i

        seg = buf[seg_start:]
        if len(seg) < self._sg_window:
            self._sg_midpoint_mode = False
            return pt, cam_dets

        xs = np.array([e["pt"]["x"] for e in seg])
        ys = np.array([e["pt"]["y"] for e in seg])
        zs = np.array([e["pt"]["z"] for e in seg])

        xs_s = savgol_filter(xs, self._sg_window, self._sg_poly)
        ys_s = savgol_filter(ys, self._sg_window, self._sg_poly)
        zs_s = savgol_filter(zs, self._sg_window, self._sg_poly)
        zs_s = np.maximum(zs_s, 0.0)

        # Return the smoothed point at the midpoint of the window end
        # (the latest point with full context on both sides)
        mid = len(seg) - 1 - self._sg_window // 2
        if mid < 0:
            mid = len(seg) - 1

        src = seg[mid]["pt"]
        if not self._sg_midpoint_mode:
            self._sg_midpoint_mode = True
            self._sg_switched_to_midpoint = True
            return None, None
        return (
            {
                "x": float(xs_s[mid]),
                "y": float(ys_s[mid]),
                "z": float(zs_s[mid]),
                "timestamp": src.get("capture_ts", src["timestamp"]),
                "capture_ts": src.get("capture_ts", src["timestamp"]),
                "frame_index": src.get("frame_index"),
            },
            seg[mid].get("cam", {}),
        )

    def reset_live_analytics(self) -> None:
        """Reset every piece of per-session analytics state so the next
        session starts clean. After this returns, the first frames of the
        new session produce no rally/bounce events until Hybrid's buffer
        (60 frames) and Peak's batch (10 frames) refill — this is expected.
        """
        with self._analytics_lock:
            # Detectors
            self._hybrid_bounce.reset()
            self._bounce_detector.reset()
            self._rally_tracker.reset()

            # Production + sidecar bounce buffers
            self._live_bounces.clear()
            self._live_hits.clear()
            self._total_live_bounces = 0
            self._peak_bounces_eval.clear()
            self._post_filter_stats.clear()
            self._hit_bounce_refiner.reset()
            self._last_refiner_result = {}
            self._last_smoothed_cam_dets = {}

            # Rally buffers (kept for offline report/export code paths)
            self._live_rallies.clear()
            self._rally_raw_buffer.clear()

            # Speed / motion state so the next session doesn't see stale prev_3d
            self._speed_points.clear()
            self._speed_buffer.clear()
            self._sg_buffer.clear()
            self._sg_midpoint_mode = False
            self._sg_switched_to_midpoint = False
            self._blob_buffers.clear()
            self._blob_capture_ts_by_frame.clear()
            self._tracker_block_count = 0
            self._emitted_3d_frames.clear()
            self._emitted_bounce_frames.clear()
            self._last_bounce_ts = 0.0
            self._prev_3d = None
            self._last_tri_pair = None
            self._candidate_continuity.clear()
            self._conf_history.clear()
            if self._live_matcher is not None:
                self._live_matcher.reset()

            # Net crossing state (clearing both the history and the "latest")
            self._net_crossings.clear()
            self._latest_net_crossing = None

            # WS push queue (stale bounces from prior session would confuse client)
            self._ws_bounce_queue.clear()

            # Debug recording buckets (keep shape, wipe contents)
            if isinstance(self._debug_data, dict):
                self._debug_data["bounces"] = []
                self._debug_data["hits"] = []
                self._debug_data["peak_bounces"] = []

    _TRACKING_FSYNC_INTERVAL_S = 1.0

    def _make_tracking_jsonl_path(self, ts: str | None = None, label: str | None = None) -> str:
        """Return a unique tracking_*.jsonl path under recordings/."""
        if ts is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"tracking_{ts}"
        if label:
            base += f"_{label}"
        candidate = self._recordings_dir / f"{base}.jsonl"
        idx = 1
        while candidate.exists():
            candidate = self._recordings_dir / f"{base}_{idx}.jsonl"
            idx += 1
        return str(candidate)

    def _open_tracking_file_locked(self, path: str, reset_counter: bool) -> None:
        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        self._tracking_file = open(path, "w", encoding="utf-8", buffering=1)
        self._tracking_file_path = path
        if reset_counter:
            self._data_frame_counter = 0
        self._jsonl_last_fsync_ts = time.time()

    def _flush_tracking_file_locked(self, force_fsync: bool = False) -> None:
        if self._tracking_file is None:
            return
        try:
            self._tracking_file.flush()
        except Exception:
            return
        now = time.time()
        if force_fsync or now - self._jsonl_last_fsync_ts >= self._TRACKING_FSYNC_INTERVAL_S:
            try:
                os.fsync(self._tracking_file.fileno())
                self._jsonl_last_fsync_ts = now
            except Exception:
                pass

    def _close_tracking_file_locked(self, force_fsync: bool = True) -> None:
        if self._tracking_file is None:
            return
        self._flush_tracking_file_locked(force_fsync=force_fsync)
        try:
            self._tracking_file.close()
        except Exception:
            pass
        self._tracking_file = None

    def _rotate_tracking_file_locked(self, path: str, reset_counter: bool) -> None:
        self._close_tracking_file_locked(force_fsync=True)
        self._open_tracking_file_locked(path, reset_counter=reset_counter)
        logger.info("Tracking JSONL started: %s", path)

    def _append_tracking_jsonl_row(self, row: dict, *, bump_frame_counter: bool) -> None:
        with self._jsonl_lock:
            if self._tracking_file is None:
                return
            if bump_frame_counter:
                row = {"frame": self._data_frame_counter, **row}
            try:
                self._tracking_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                # Keep the JSONL durable for long sessions without fsyncing
                # every single line: flush on each row, fsync at bounded
                # intervals so a crash loses at most a small tail window.
                self._flush_tracking_file_locked(force_fsync=False)
                if bump_frame_counter:
                    self._data_frame_counter += 1
            except Exception:
                pass

    def get_latest_net_crossing(self) -> Optional[dict]:
        """Return the most recent net crossing event with speed."""
        return self._latest_net_crossing

    def get_net_crossings(self) -> list[dict]:
        """Return recent net crossing events."""
        return list(self._net_crossings[-20:])

    def get_latency_stats(self) -> dict:
        """Return end-to-end latency statistics (capture → 3D output)."""
        buf = list(self._latency_buffer)
        if not buf:
            return {"count": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0}
        buf_sorted = sorted(buf)
        n = len(buf_sorted)
        return {
            "count": n,
            "p50_ms": round(buf_sorted[n // 2], 1),
            "p95_ms": round(buf_sorted[min(int(n * 0.95), n - 1)], 1),
            "max_ms": round(self._latency_max, 1),
        }

    def enable_3d_display(self, url: str = None) -> dict:
        """Enable WebSocket push to 3D display."""
        if url:
            self._ws_url = url
        self._ws_enabled = True
        if self._ws_thread is None or not self._ws_thread.is_alive():
            self._ws_thread = threading.Thread(
                target=self._ws_push_loop, daemon=True, name="ws-3d-push"
            )
            self._ws_thread.start()
        return {"enabled": True, "url": self._ws_url}

    def disable_3d_display(self) -> dict:
        """Disable WebSocket push."""
        self._ws_enabled = False
        return {"enabled": False}

    def enable_ml_rally(self) -> dict:
        """Enable ML-based rally segmentation filter."""
        if self._ml_rally_model is None:
            model_path = Path("model_weight/rally_segmentation.pkl")
            if model_path.exists():
                import pickle
                with open(model_path, "rb") as f:
                    self._ml_rally_model = pickle.load(f)
                logger.info("ML Rally model loaded from %s", model_path)
            else:
                logger.warning("ML Rally model not found at %s", model_path)
                return {"enabled": False, "error": "model not found"}
        self._ml_rally_enabled = True
        return {"enabled": True}

    def disable_ml_rally(self) -> dict:
        """Disable ML rally filter (pass all detections through)."""
        self._ml_rally_enabled = False
        return {"enabled": False}

    def get_ml_rally_status(self) -> dict:
        """Return ML rally filter status."""
        return {
            "enabled": self._ml_rally_enabled,
            "model_loaded": self._ml_rally_model is not None,
        }

    # ------------------------------------------------------------------
    # Feature toggles: bounce detection, net crossing, OCR align
    # ------------------------------------------------------------------
    def set_bounce_detection_enabled(self, enabled: bool) -> dict:
        enabled = bool(enabled)
        if self._bounce_detection_enabled != enabled:
            with self._analytics_lock:
                # Switching bounce detection on/off should not reuse stale
                # SG or detector state from the previous mode.
                self._hybrid_bounce.reset()
                self._bounce_detector.reset()
                self._sg_buffer.clear()
        self._bounce_detection_enabled = enabled
        return {"enabled": self._bounce_detection_enabled}

    def set_net_crossing_enabled(self, enabled: bool) -> dict:
        self._net_crossing_enabled = enabled
        return {"enabled": self._net_crossing_enabled}

    def set_ocr_align_enabled(self, enabled: bool) -> dict:
        self._ocr_align_enabled = enabled
        return {"enabled": self._ocr_align_enabled}

    def get_feature_toggles(self) -> dict:
        return {
            "bounce_detection": self._bounce_detection_enabled,
            "net_crossing": self._net_crossing_enabled,
            "ocr_align": self._ocr_align_enabled,
            "ws_3d_display": self._ws_enabled,
            "ml_rally": self._ml_rally_enabled,
            "inference": self._inference_enabled,
        }

    def _ws_push_loop(self) -> None:
        """Background thread: push bounce events to 3D display via WebSocket."""
        import asyncio
        import ssl

        async def _run():
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

            try:
                import websockets
            except ImportError:
                logger.warning("websockets not installed, 3D display push disabled")
                return

            while self._ws_enabled and not self._stopped.is_set():
                try:
                    connect_kwargs = {"ssl": ssl_ctx} if self._ws_url.startswith("wss://") else {}
                    async with websockets.connect(self._ws_url, **connect_kwargs) as ws:
                        logger.info("3D display connected: %s", self._ws_url)
                        while self._ws_enabled and not self._stopped.is_set():
                            if self._ws_bounce_queue:
                                bd = self._ws_bounce_queue[0]
                                msg = json.dumps({
                                    "msg": {
                                        "message": "bounce_data",
                                        "data": {
                                            "bounce": {
                                                "timeStamp": bd["timestamp"],
                                                "x": round(bd["x"], 4),
                                                "y": round(bd["y"], 4),
                                                "speed": int(round(bd["speed"])),
                                            }
                                        }
                                    }
                                })
                                await ws.send(msg)
                                if self._ws_bounce_queue and self._ws_bounce_queue[0] is bd:
                                    self._ws_bounce_queue.popleft()
                                logger.info("3D display: sent bounce x=%.3f y=%.3f speed=%.0f",
                                           bd["x"], bd["y"], bd["speed"])
                            else:
                                await asyncio.sleep(0.1)
                except Exception as e:
                    logger.warning("3D display WebSocket error: %s, reconnecting in 5s", e)
                    await asyncio.sleep(5)

        asyncio.run(_run())

    def get_latest_frame(self, name: str) -> Optional[bytes]:
        """返回指定摄像头的最新 JPEG 帧字节（用于 MJPEG 流）。"""
        return self._latest_frames.get(name)

    def set_inference_enabled(self, enabled: bool) -> None:
        """全局开关：启用/禁用所有摄像头的 GPU 推理（track ball）。"""
        self._inference_enabled = enabled
        for handle in self._handles.values():
            if handle.status_dict is not None:
                handle.status_dict["inference_enabled"] = enabled
        logger.info("Inference %s", "enabled" if enabled else "disabled")

    @property
    def inference_enabled(self) -> bool:
        return self._inference_enabled

    def switch_model(self, model_name: str) -> dict:
        """Switch between HRNet and TrackNet models at runtime.

        Args:
            model_name: "hrnet" or "tracknet"

        Returns:
            Dict with new model config info.
        """
        model_name = model_name.lower().strip()
        configs = {
            "hrnet": {
                "path": "model_weight/hrnet_tennis.onnx",
                "frames_in": 3,
                "frames_out": 3,
                "detector_type": "auto",
            },
            "tracknet": {
                "path": "model_weight/TrackNet_finetuned.onnx",
                "frames_in": 8,
                "frames_out": 8,
                "detector_type": "tracknet",
            },
        }
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}. Use 'hrnet' or 'tracknet'")

        running_live_cameras = [
            name
            for name, handle in self._handles.items()
            if name in self.config.cameras and handle.is_alive()
        ]
        for name in running_live_cameras:
            self.stop_pipeline(name)

        selected = configs[model_name]
        self.config.model.path = selected["path"]
        self.config.model.frames_in = selected["frames_in"]
        self.config.model.frames_out = selected["frames_out"]
        self.config.model.detector_type = selected["detector_type"]
        self._is_median_bg = self.config.model.detector_type == "median_bg"

        for name in running_live_cameras:
            self.start_pipeline(name)

        logger.info(
            "Model switched to %s: %s (detector=%s, frames=%d, restarted=%s)",
            model_name,
            self.config.model.path,
            self.config.model.detector_type,
            self.config.model.frames_in,
            running_live_cameras,
        )
        return {
            "model": model_name,
            "path": self.config.model.path,
            "frames_in": self.config.model.frames_in,
            "frames_out": self.config.model.frames_out,
            "detector_type": self.config.model.detector_type,
            "restarted_live_cameras": running_live_cameras,
        }

    def get_current_model(self) -> dict:
        """Return current model info."""
        path = self.config.model.path
        path_name = Path(path).name.lower()
        if self.config.model.detector_type == "tracknet" or "tracknet" in path_name:
            name = "tracknet"
        elif "hrnet" in path_name:
            name = "hrnet"
        else:
            name = self.config.model.detector_type or "auto"
        return {
            "model": name,
            "path": path,
            "frames_in": self.config.model.frames_in,
            "frames_out": self.config.model.frames_out,
            "detector_type": self.config.model.detector_type,
        }

    # ------------------------------------------------------------------
    # Video test (called from FastAPI)
    # ------------------------------------------------------------------
    def start_video_test(
        self, video_path: str, start_time: float, end_time: float, camera_name: str
    ) -> dict:
        """Start processing a video file segment."""
        if self._video_test_handle is not None and self._video_test_handle.is_alive():
            self.stop_video_test()

        self._video_test_detections.pop(camera_name, None)

        cam_cfg = self.config.cameras.get(camera_name)
        if cam_cfg is None:
            raise ValueError(f"Unknown camera: {camera_name}")

        handle = _PipelineHandle("_video_test")
        handle.result_queue = mp.Queue(maxsize=256)
        handle.frame_queue = mp.Queue(maxsize=32)
        handle.stop_event = mp.Event()
        handle.status_dict = self._manager.dict(
            {
                "state": "starting",
                "fps": 0.0,
                "total_frames": 0,
                "processed_frames": 0,
                "error_msg": "",
            }
        )

        # Build ensemble config dict if enabled
        ens_cfg = self.config.ensemble
        ensemble_dict = None
        if ens_cfg.enabled:
            ensemble_dict = {
                "enabled": True,
                "hrnet_path": ens_cfg.hrnet_path,
                "agree_distance": ens_cfg.agree_distance,
                "boost_factor": ens_cfg.boost_factor,
                "penalty_factor": ens_cfg.penalty_factor,
                "single_factor": ens_cfg.single_factor,
            }

        handle.process = mp.Process(
            target=run_video_pipeline,
            kwargs={
                "video_path": video_path,
                "start_time": start_time,
                "end_time": end_time,
                "camera_name": camera_name,
                "model_path": self.config.model.path,
                "input_size": tuple(self.config.model.input_size),
                "frames_in": self.config.model.frames_in,
                "frames_out": self.config.model.frames_out,
                "threshold": self.config.model.threshold,
                "device": self.config.model.device,
                "homography_path": self.config.homography.path,
                "homography_key": cam_cfg.homography_key,
                "result_queue": handle.result_queue,
                "frame_queue": handle.frame_queue,
                "stop_event": handle.stop_event,
                "status_dict": handle.status_dict,
                "ensemble_config": ensemble_dict,
                "heatmap_mask": [tuple(r) for r in self.config.model.heatmap_mask],
                "blob_verifier_config": self.config.blob_verifier.model_dump()
                    if self.config.blob_verifier.enabled else None,
                "detector_type": self.config.model.detector_type,
            },
            daemon=True,
        )
        handle.process.start()
        logger.info(
            "[video-test] Started: %s [%.1f-%.1f] cam=%s pid=%d ensemble=%s",
            video_path, start_time, end_time, camera_name, handle.process.pid,
            "ON" if ensemble_dict else "OFF",
        )

        self._video_test_handle = handle
        self._handles["_video_test"] = handle

        # Ensure consumer thread is running
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._stopped.clear()
            self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._consumer_thread.start()

        return {"status": "started"}

    def start_video_test_parallel(self, cameras: list[dict]) -> dict:
        """Start processing multiple camera videos in parallel.

        Args:
            cameras: List of dicts with keys: camera_name, video_path, start_time, end_time.

        Returns:
            Status dict with started camera names.
        """
        # Stop any existing video tests (both single and parallel)
        self.stop_video_test()

        # Build ensemble config dict if enabled
        ens_cfg = self.config.ensemble
        ensemble_dict = None
        if ens_cfg.enabled:
            ensemble_dict = {
                "enabled": True,
                "hrnet_path": ens_cfg.hrnet_path,
                "agree_distance": ens_cfg.agree_distance,
                "boost_factor": ens_cfg.boost_factor,
                "penalty_factor": ens_cfg.penalty_factor,
                "single_factor": ens_cfg.single_factor,
            }

        started = []
        for cam_info in cameras:
            camera_name = cam_info["camera_name"]
            video_path = cam_info["video_path"]
            start_time = cam_info["start_time"]
            end_time = cam_info["end_time"]

            self._video_test_detections.pop(camera_name, None)

            cam_cfg = self.config.cameras.get(camera_name)
            if cam_cfg is None:
                raise ValueError(f"Unknown camera: {camera_name}")

            handle_name = f"_video_test_{camera_name}"
            handle = _PipelineHandle(handle_name)
            handle.result_queue = mp.Queue(maxsize=256)
            handle.frame_queue = mp.Queue(maxsize=32)
            handle.stop_event = mp.Event()
            handle.status_dict = self._manager.dict(
                {
                    "state": "starting",
                    "fps": 0.0,
                    "total_frames": 0,
                    "processed_frames": 0,
                    "error_msg": "",
                }
            )

            handle.process = mp.Process(
                target=run_video_pipeline,
                kwargs={
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "camera_name": camera_name,
                    "model_path": self.config.model.path,
                    "input_size": tuple(self.config.model.input_size),
                    "frames_in": self.config.model.frames_in,
                    "frames_out": self.config.model.frames_out,
                    "threshold": self.config.model.threshold,
                    "device": self.config.model.device,
                    "homography_path": self.config.homography.path,
                    "homography_key": cam_cfg.homography_key,
                    "result_queue": handle.result_queue,
                    "frame_queue": handle.frame_queue,
                    "stop_event": handle.stop_event,
                    "status_dict": handle.status_dict,
                    "ensemble_config": ensemble_dict,
                    "heatmap_mask": [tuple(r) for r in self.config.model.heatmap_mask],
                    "blob_verifier_config": self.config.blob_verifier.model_dump()
                        if self.config.blob_verifier.enabled else None,
                    "detector_type": self.config.model.detector_type,
                },
                daemon=True,
            )
            handle.process.start()
            logger.info(
                "[video-test-parallel] Started: %s [%.1f-%.1f] cam=%s pid=%d ensemble=%s",
                video_path, start_time, end_time, camera_name, handle.process.pid,
                "ON" if ensemble_dict else "OFF",
            )

            self._video_test_handles[camera_name] = handle
            self._handles[handle_name] = handle
            started.append(camera_name)

        # Ensure consumer thread is running
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._stopped.clear()
            self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._consumer_thread.start()

        return {"status": "started", "cameras": started}

    def stop_video_test(self) -> dict:
        """Stop video test pipeline (both single and parallel handles)."""
        had_any = bool(self._video_test_handles) or self._video_test_handle is not None

        # Stop parallel handles
        for cam_name, handle in list(self._video_test_handles.items()):
            if handle.stop_event is not None:
                handle.stop_event.set()
            if handle.process is not None:
                handle.process.join(timeout=10.0)
                if handle.process.is_alive():
                    handle.process.terminate()
                    handle.process.join(timeout=5.0)
            handle_name = f"_video_test_{cam_name}"
            self._handles.pop(handle_name, None)
            self._latest_frames.pop(handle_name, None)
            self._latest_detections.pop(handle_name, None)
        if self._video_test_handles:
            logger.info("[video-test-parallel] Stopped %d cameras", len(self._video_test_handles))
        self._video_test_handles.clear()

        # Stop legacy single handle
        handle = self._video_test_handle
        if not had_any:
            return {"status": "not_running"}
        if handle is not None:
            if handle.stop_event is not None:
                handle.stop_event.set()
            if handle.process is not None:
                handle.process.join(timeout=10.0)
                if handle.process.is_alive():
                    handle.process.terminate()
                    handle.process.join(timeout=5.0)
            self._handles.pop("_video_test", None)
            self._latest_frames.pop("_video_test", None)
            self._latest_detections.pop("_video_test", None)
            self._video_test_handle = None
            logger.info("[video-test] Stopped")
        return {"status": "stopped"}

    def get_video_test_detections(self, camera_name: str | None = None) -> list[dict]:
        """Return accumulated video test detections, optionally filtered by camera."""
        if camera_name:
            return list(self._video_test_detections.get(camera_name, []))
        all_dets: list[dict] = []
        for cam_dets in self._video_test_detections.values():
            all_dets.extend(cam_dets)
        return sorted(all_dets, key=lambda d: d.get("frame_index", 0))

    def clear_video_test_detections(self, camera_name: str | None = None) -> None:
        """Clear stored video test detections."""
        if camera_name:
            self._video_test_detections.pop(camera_name, None)
        else:
            self._video_test_detections.clear()

    def get_video_test_detections_since(self, cursors: dict[str, int]) -> dict[str, list[dict]]:
        """Return detections newer than cursor index per camera.

        Args:
            cursors: Maps camera name to last-seen index, e.g. {"cam66": 42, "cam68": 38}.

        Returns:
            Dict of camera_name -> list of new detections since cursor.
        """
        result: dict[str, list[dict]] = {}
        for cam, dets in self._video_test_detections.items():
            start = cursors.get(cam, 0)
            if start < len(dets):
                result[cam] = dets[start:]
        return result

    def export_cvat_xml(self, camera_name: str, video_path: str) -> str:
        """Export detections for a single camera as CVAT for Video 1.1 XML.

        Automatically splits detections into multiple tracks when there are
        frame gaps >= ``_TRACK_SPLIT_GAP``, so the user can directly review
        and adjust in CVAT without having to manually split one giant track.

        Label attributes:
            state  – visible / occluded
            main   – yes / no  (whether this is the active game ball)

        Args:
            camera_name: Camera to export detections for.
            video_path: Original video file path (used for metadata).

        Returns:
            XML string in CVAT annotation format.
        """
        _TRACK_SPLIT_GAP = 5  # frame gap threshold to start a new track

        dets = self._video_test_detections.get(camera_name, [])
        if not dets:
            raise ValueError(f"No detections for camera: {camera_name}")

        # Sort by frame_index
        dets = sorted(dets, key=lambda d: d.get("frame_index", 0))

        # --- Split detections into track segments ----
        segments: list[list[dict]] = []
        current_seg: list[dict] = [dets[0]]
        for i in range(1, len(dets)):
            gap = dets[i].get("frame_index", 0) - dets[i - 1].get("frame_index", 0)
            if gap >= _TRACK_SPLIT_GAP:
                segments.append(current_seg)
                current_seg = []
            current_seg.append(dets[i])
        segments.append(current_seg)

        logger.info(
            "CVAT export %s: %d detections → %d tracks (gap threshold=%d)",
            camera_name, len(dets), len(segments), _TRACK_SPLIT_GAP,
        )

        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

        lines = [
            '<?xml version="1.0" encoding="utf-8"?>',
            "<annotations>",
            "  <version>1.1</version>",
            "  <meta>",
            "    <job>",
            f"      <id>0</id>",
            f"      <size>{total_frames}</size>",
            "      <mode>interpolation</mode>",
            "      <overlap>0</overlap>",
            "      <bugtracker></bugtracker>",
            f"      <created>{now}</created>",
            f"      <updated>{now}</updated>",
            "      <subset>default</subset>",
            "      <start_frame>0</start_frame>",
            f"      <stop_frame>{total_frames - 1}</stop_frame>",
            "      <frame_filter></frame_filter>",
            "      <segments>",
            "        <segment>",
            "          <id>0</id>",
            "          <start>0</start>",
            f"          <stop>{total_frames - 1}</stop>",
            "          <url></url>",
            "        </segment>",
            "      </segments>",
            "      <owner>",
            "        <username>auto-detect</username>",
            "        <email></email>",
            "      </owner>",
            "      <labels>",
            "        <label>",
            '          <name>ball</name>',
            '          <color>#804080</color>',
            '          <type>any</type>',
            "          <attributes>",
            "            <attribute>",
            '              <name>state</name>',
            '              <mutable>True</mutable>',
            '              <input_type>select</input_type>',
            '              <default_value>visible</default_value>',
            "              <values>visible\noccluded</values>",
            "            </attribute>",
            "            <attribute>",
            '              <name>main</name>',
            '              <mutable>True</mutable>',
            '              <input_type>select</input_type>',
            '              <default_value>yes</default_value>',
            "              <values>yes\nno</values>",
            "            </attribute>",
            "          </attributes>",
            "        </label>",
            "      </labels>",
            "    </job>",
            f"    <dumped>{now}</dumped>",
            "    <original_size>",
            f"      <width>{vid_w}</width>",
            f"      <height>{vid_h}</height>",
            "    </original_size>",
            "  </meta>",
        ]

        # Build one <track> per segment
        for track_id, seg in enumerate(segments):
            lines.append(f'  <track id="{track_id}" label="ball" source="auto">')

            for i, det in enumerate(seg):
                frame = det.get("frame_index", 0)
                px = round(det.get("pixel_x", 0), 2)
                py = round(det.get("pixel_y", 0), 2)

                # Keyframe at segment boundaries and around internal gaps
                is_first = i == 0
                is_last = i == len(seg) - 1
                prev_gap = (frame - seg[i - 1].get("frame_index", 0)) > 1 if i > 0 else True
                next_gap = (seg[i + 1].get("frame_index", 0) - frame) > 1 if i < len(seg) - 1 else True
                keyframe = 1 if (is_first or is_last or prev_gap or next_gap) else 0

                lines.append(
                    f'    <points frame="{frame}" keyframe="{keyframe}" '
                    f'outside="0" occluded="0" points="{px},{py}" z_order="0">'
                )
                lines.append(f'      <attribute name="state">visible</attribute>')
                lines.append(f'      <attribute name="main">yes</attribute>')
                lines.append("    </points>")

            # Close track with outside=1 on the frame after the last detection
            last_frame = seg[-1].get("frame_index", 0)
            close_frame = last_frame + 1
            if close_frame < total_frames:
                last_px = round(seg[-1].get("pixel_x", 0), 2)
                last_py = round(seg[-1].get("pixel_y", 0), 2)
                lines.append(
                    f'    <points frame="{close_frame}" keyframe="1" '
                    f'outside="1" occluded="0" points="{last_px},{last_py}" z_order="0">'
                )
                lines.append(f'      <attribute name="state">visible</attribute>')
                lines.append(f'      <attribute name="main">yes</attribute>')
                lines.append("    </points>")

            lines.append("  </track>")

        lines.append("</annotations>")

        return "\n".join(lines)

    def compute_3d_from_detections(self) -> dict:
        """Match detections from two cameras by frame_index and compute 3D positions.

        When detections contain 'candidates' (multi-blob), uses MultiBlobMatcher
        to pick the best blob pair per frame via ray_distance minimization.
        Falls back to single-blob triangulation for legacy detections.

        Returns dict with 'points', 'stats', 'cam_order', and 'matcher_stats'.
        """
        cam_names = list(self._video_test_detections.keys())
        if len(cam_names) < 2:
            return {"points": [], "stats": {}, "cam_order": cam_names}

        cam1_name, cam2_name = cam_names[0], cam_names[1]
        cam1_dets = {d["frame_index"]: d for d in self._video_test_detections[cam1_name]}
        cam2_dets = {d["frame_index"]: d for d in self._video_test_detections[cam2_name]}

        common_frames = sorted(set(cam1_dets.keys()) & set(cam2_dets.keys()))

        stats = {
            cam1_name: {
                "total_detections": len(cam1_dets),
                "frame_range": [min(cam1_dets.keys()), max(cam1_dets.keys())] if cam1_dets else [],
            },
            cam2_name: {
                "total_detections": len(cam2_dets),
                "frame_range": [min(cam2_dets.keys()), max(cam2_dets.keys())] if cam2_dets else [],
            },
            "common_frames": len(common_frames),
        }
        logger.info(
            "3D compute: %s has %d dets, %s has %d dets, %d common frames",
            cam1_name, len(cam1_dets), cam2_name, len(cam2_dets), len(common_frames),
        )

        cam1_cfg = self.config.cameras.get(cam1_name)
        cam2_cfg = self.config.cameras.get(cam2_name)
        if not cam1_cfg or not cam2_cfg:
            logger.error("Camera config not found for %s or %s", cam1_name, cam2_name)
            return {"points": [], "stats": stats, "cam_order": cam_names}

        cam_pos = self._get_camera_positions()
        pos1 = cam_pos.get(cam1_name, cam1_cfg.position_3d)
        pos2 = cam_pos.get(cam2_name, cam2_cfg.position_3d)

        # Check if detections have candidates (multi-blob mode)
        has_candidates = any(
            "candidates" in d for d in self._video_test_detections[cam1_name][:10]
        )

        matcher = MultiBlobMatcher(pos1, pos2) if has_candidates else None
        results = []

        for frame_idx in common_frames:
            d1 = cam1_dets[frame_idx]
            d2 = cam2_dets[frame_idx]
            try:
                if matcher and "candidates" in d1 and "candidates" in d2:
                    # Multi-blob matching: try all pairs, pick lowest ray_distance
                    match = matcher.match(d1, d2)
                    if match is not None:
                        results.append({
                            "frame_index": frame_idx,
                            "x": round(match["x"], 4),
                            "y": round(match["y"], 4),
                            "z": round(match["z"], 4),
                            "ray_distance": round(match["ray_distance"], 4),
                            "cam1_pixel": [round(match["cam1_pixel"][0], 1),
                                           round(match["cam1_pixel"][1], 1)],
                            "cam2_pixel": [round(match["cam2_pixel"][0], 1),
                                           round(match["cam2_pixel"][1], 1)],
                            "cam1_world": [round(match["cam1_world"][0], 4),
                                           round(match["cam1_world"][1], 4)],
                            "cam2_world": [round(match["cam2_world"][0], 4),
                                           round(match["cam2_world"][1], 4)],
                            "cam1_blob_idx": match["cam1_idx"],
                            "cam2_blob_idx": match["cam2_idx"],
                        })
                else:
                    # Legacy single-blob fallback
                    x, y, z = triangulate(
                        (d1["x"], d1["y"]),
                        (d2["x"], d2["y"]),
                        pos1, pos2,
                    )
                    results.append({
                        "frame_index": frame_idx,
                        "x": round(x, 4),
                        "y": round(y, 4),
                        "z": round(z, 4),
                        "cam1_pixel": [round(d1["pixel_x"], 1), round(d1["pixel_y"], 1)],
                        "cam2_pixel": [round(d2["pixel_x"], 1), round(d2["pixel_y"], 1)],
                        "cam1_world": [round(d1["x"], 4), round(d1["y"], 4)],
                        "cam2_world": [round(d2["x"], 4), round(d2["y"], 4)],
                    })
            except Exception as e:
                logger.warning("3D computation failed for frame %d: %s", frame_idx, e)

        result = {"points": results, "stats": stats, "cam_order": cam_names}

        if matcher:
            m_stats = matcher.get_stats()
            result["matcher_stats"] = m_stats
            logger.info(
                "MultiBlobMatcher: %d/%d frames matched, %d non-top1 picks (%.1f%%)",
                m_stats["matched_frames"], m_stats["total_frames"],
                m_stats["non_top1_picks"], m_stats["non_top1_rate"] * 100,
            )

        return result

    # ------------------------------------------------------------------
    # Import LabelImg annotations → 3D
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_labelimg_folder(folder: Path) -> dict[int, dict]:
        """Parse a folder of LabelImg JSON annotation files.

        Returns {frame_number: {"pixel_x": float, "pixel_y": float}}.
        """
        result = {}
        for jf in sorted(folder.glob("*.json")):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data.get("shapes", [])
                if not shapes:
                    continue
                pts = shapes[0].get("points", [])
                if not pts or len(pts[0]) < 2:
                    continue
                frame_num = int(jf.stem)
                result[frame_num] = {"pixel_x": float(pts[0][0]), "pixel_y": float(pts[0][1])}
            except Exception:
                continue
        return result

    def import_labelimg_annotations(self, cam1_folder: str, cam2_folder: str) -> dict:
        """Import LabelImg annotations from two camera folders and triangulate to 3D.

        Args:
            cam1_folder: Subfolder name under uploads/ for camera 1 (e.g. "cam66_clip").
            cam2_folder: Subfolder name under uploads/ for camera 2 (e.g. "cam68_clip").

        Returns:
            dict with 'points' (list of 3D point dicts), 'stats', and 'bounces'.
        """
        from app.pipeline.homography import HomographyTransformer

        uploads_dir = Path("uploads")
        folder1 = uploads_dir / cam1_folder
        folder2 = uploads_dir / cam2_folder

        if not folder1.is_dir():
            return {"error": f"Folder not found: {cam1_folder}", "points": []}
        if not folder2.is_dir():
            return {"error": f"Folder not found: {cam2_folder}", "points": []}

        # Parse annotation files
        ann1 = self._parse_labelimg_folder(folder1)
        ann2 = self._parse_labelimg_folder(folder2)

        if not ann1 or not ann2:
            return {"error": "No valid annotations found", "points": []}

        common_frames = sorted(set(ann1.keys()) & set(ann2.keys()))
        if not common_frames:
            return {"error": "No common frames between cameras", "points": []}

        # Determine camera names from folder names
        cam_names = list(self.config.cameras.keys())
        if len(cam_names) < 2:
            return {"error": "Need at least 2 cameras in config", "points": []}

        cam1_name = cam_names[0]  # cam66
        cam2_name = cam_names[1]  # cam68

        # Load homography transformers
        h_path = self.config.homography.path
        cam1_cfg = self.config.cameras[cam1_name]
        cam2_cfg = self.config.cameras[cam2_name]
        h1 = HomographyTransformer(h_path, cam1_cfg.homography_key)
        h2 = HomographyTransformer(h_path, cam2_cfg.homography_key)

        # Get camera 3D positions
        cam_pos = self._get_camera_positions()
        pos1 = cam_pos.get(cam1_name, cam1_cfg.position_3d)
        pos2 = cam_pos.get(cam2_name, cam2_cfg.position_3d)

        # Triangulate each common frame
        points = []
        for fi in common_frames:
            a1 = ann1[fi]
            a2 = ann2[fi]
            try:
                wx1, wy1 = h1.pixel_to_world(a1["pixel_x"], a1["pixel_y"])
                wx2, wy2 = h2.pixel_to_world(a2["pixel_x"], a2["pixel_y"])
                x, y, z = triangulate((wx1, wy1), (wx2, wy2), pos1, pos2)
                points.append({
                    "frame_index": fi,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": round(z, 4),
                    "cam1_pixel": [round(a1["pixel_x"], 1), round(a1["pixel_y"], 1)],
                    "cam2_pixel": [round(a2["pixel_x"], 1), round(a2["pixel_y"], 1)],
                    "cam1_world": [round(wx1, 4), round(wy1, 4)],
                    "cam2_world": [round(wx2, 4), round(wy2, 4)],
                })
            except Exception as e:
                logger.warning("Triangulation failed for frame %d: %s", fi, e)

        # Detect bounces (Z-axis V-shape)
        bounces = []
        for i in range(1, len(points) - 1):
            prev_z = points[i - 1]["z"]
            curr_z = points[i]["z"]
            next_z = points[i + 1]["z"]
            if curr_z < prev_z and curr_z < next_z and curr_z < 0.3:
                bx, by = points[i]["x"], points[i]["y"]
                in_court = 1.37 <= bx <= 6.86 and 0 <= by <= 23.77
                bounces.append({
                    "frame": points[i]["frame_index"],
                    "x": bx, "y": by, "z": curr_z,
                    "type": "IN" if in_court else "OUT",
                })

        stats = {
            "cam1_annotations": len(ann1),
            "cam2_annotations": len(ann2),
            "common_frames": len(common_frames),
            "triangulated_points": len(points),
            "bounces_detected": len(bounces),
        }

        logger.info(
            "Imported annotations: %d cam1, %d cam2, %d common → %d 3D points, %d bounces",
            len(ann1), len(ann2), len(common_frames), len(points), len(bounces),
        )

        return {"points": points, "bounces": bounces, "stats": stats, "cam_order": [cam1_name, cam2_name]}

    def compute_3d_trajectory(self) -> dict:
        """Compute 3D trajectory using auto time-offset and spatial parabola fitting.

        Unlike compute_3d_from_detections (which requires frame_index matching),
        this method works WITHOUT frame synchronization between cameras.

        Steps:
            1. Extract pixel-level detections from both cameras
            2. Auto-find optimal time offset via interpolated triangulation
            3. Triangulate 3D points at the optimal offset
            4. Fit piecewise spatial parabolas (frame-rate independent)

        Returns dict with raw 3D points, trajectory fit, and smooth curve.
        """
        import json

        cam_names = list(self._video_test_detections.keys())
        if len(cam_names) < 2:
            return {"error": "Need detections from 2 cameras", "cameras": cam_names}

        cam1_name, cam2_name = cam_names[0], cam_names[1]
        cam1_cfg = self.config.cameras.get(cam1_name)
        cam2_cfg = self.config.cameras.get(cam2_name)
        if not cam1_cfg or not cam2_cfg:
            return {"error": f"Camera config not found: {cam1_name} or {cam2_name}"}

        # Load homography matrices
        try:
            with open(self.config.homography.path) as f:
                hdata = json.load(f)
            H1 = np.array(
                hdata[cam1_cfg.homography_key]["H_image_to_world"], dtype=np.float64
            )
            H2 = np.array(
                hdata[cam2_cfg.homography_key]["H_image_to_world"], dtype=np.float64
            )
        except Exception as e:
            return {"error": f"Failed to load homography: {e}"}

        # Extract pixel detections with confidence: (frame_index, pixel_x, pixel_y, confidence)
        raw_dets1 = sorted(
            [
                (d["frame_index"], d["pixel_x"], d["pixel_y"], d.get("confidence", 999.0))
                for d in self._video_test_detections[cam1_name]
            ]
        )
        raw_dets2 = sorted(
            [
                (d["frame_index"], d["pixel_x"], d["pixel_y"], d.get("confidence", 999.0))
                for d in self._video_test_detections[cam2_name]
            ]
        )

        if not raw_dets1 or not raw_dets2:
            return {"error": "No detections from one or both cameras"}

        # Diagnostic: log per-camera pixel and world coord samples
        def _pixel_to_world(H, px, py):
            pt = np.array([px, py, 1.0])
            r = H @ pt
            return float(r[0] / r[2]), float(r[1] / r[2])

        for cam_label, dets, H in [
            (cam1_name, raw_dets1, H1),
            (cam2_name, raw_dets2, H2),
        ]:
            sample = dets[:5]
            world_xs = []
            for d in dets:
                wx, _ = _pixel_to_world(H, d[1], d[2])
                world_xs.append(wx)
            mean_x = np.mean(world_xs) if world_xs else 0
            logger.info(
                "[3d-diag] %s: %d dets, mean_world_x=%.2f, sample pixels: %s",
                cam_label,
                len(dets),
                mean_x,
                [(round(d[1], 0), round(d[2], 0)) for d in sample],
            )
            logger.info(
                "[3d-diag] %s: sample world coords: %s",
                cam_label,
                [
                    (round(_pixel_to_world(H, d[1], d[2])[0], 2),
                     round(_pixel_to_world(H, d[1], d[2])[1], 2))
                    for d in sample
                ],
            )

        # Use 25fps as nominal (actual timing reconstructed by offset search)
        fps = 25.0

        # Stage 0: Per-camera detection cleaning
        dets1, clean_stats1 = clean_detections(raw_dets1, fps, H1)
        dets2, clean_stats2 = clean_detections(raw_dets2, fps, H2)
        logger.info(
            "[3d-traj] Cleaning: %s %d->%d, %s %d->%d",
            cam1_name, len(raw_dets1), len(dets1),
            cam2_name, len(raw_dets2), len(dets2),
        )

        if not dets1 or not dets2:
            return {"error": "No detections remaining after cleaning"}

        # Stage 1: Auto offset + interpolated triangulation
        cam_pos = self._get_camera_positions()
        best_dt, points_3d = find_offset_and_triangulate(
            dets1, dets2, fps, fps, H1, H2,
            cam_pos.get(cam1_name, cam1_cfg.position_3d),
            cam_pos.get(cam2_name, cam2_cfg.position_3d),
        )

        if not points_3d:
            return {"error": "No matched points after offset search"}

        # Diagnostic: log first 10 triangulated 3D points with per-camera world coords
        for i, p in enumerate(points_3d[:10]):
            caw = p.get("cam_a_world", [0, 0])
            cbw = p.get("cam_b_world", [0, 0])
            logger.info(
                "[3d-diag] point[%d] 3D=(%.2f, %.2f, %.2f) ray=%.3f "
                "camA_world=(%.2f, %.2f) camB_world=(%.2f, %.2f)",
                i, p["x"], p["y"], p["z"], p["ray_dist"],
                caw[0], caw[1], cbw[0], cbw[1],
            )

        # Stage 2: Rally segmentation — split by time gaps / spatial jumps
        rallies = segment_rallies(points_3d, fps=fps, max_gap_seconds=1.0, min_rally_points=5)
        logger.info(
            "[3d-traj] Rally segmentation: %d points -> %d rallies (%s)",
            len(points_3d),
            len(rallies),
            [len(r) for r in rallies],
        )

        # Stage 3: RANSAC spatial parabolic fit per rally
        rally_results = []
        for ri, rally_pts in enumerate(rallies):
            traj_fit = fit_trajectory(rally_pts)
            # Round point coordinates for JSON
            for p in rally_pts:
                p["x"] = round(p["x"], 4)
                p["y"] = round(p["y"], 4)
                p["z"] = round(p["z"], 4)
                p["ray_dist"] = round(p["ray_dist"], 4)
                p["t"] = round(p["t"], 4)
            rally_results.append({
                "rally_index": ri,
                "points": rally_pts,
                "trajectory": traj_fit,
            })

        # Use the largest rally as the primary result for backward compat
        primary_rally = max(rally_results, key=lambda r: len(r["points"])) if rally_results else None
        primary_points = primary_rally["points"] if primary_rally else []
        primary_traj = primary_rally["trajectory"] if primary_rally else {"type": "insufficient_data"}

        # Collect all points across all rallies for the full point cloud
        all_points = []
        for r in rally_results:
            all_points.extend(r["points"])

        # Run batch bounce detection & rally tracking on all 3D points
        batch_analytics = run_batch_analytics(all_points)

        # Compute stats
        ray_dists = [p["ray_dist"] for p in points_3d]
        stats = {
            cam1_name: {
                "raw_detections": len(raw_dets1),
                "cleaned_detections": len(dets1),
                "cleaning": clean_stats1,
            },
            cam2_name: {
                "raw_detections": len(raw_dets2),
                "cleaned_detections": len(dets2),
                "cleaning": clean_stats2,
            },
            "matched_points": len(points_3d),
            "n_rallies": len(rallies),
            "rally_sizes": [len(r) for r in rallies],
            "time_offset_s": round(best_dt, 4),
            "time_offset_frames": round(best_dt * fps, 1),
            "mean_ray_dist": round(float(np.mean(ray_dists)), 4),
            "max_ray_dist": round(float(np.max(ray_dists)), 4),
            "n_inliers": primary_traj.get("n_inliers", len(primary_points)),
            "n_outliers": primary_traj.get("n_outliers", 0),
        }

        return {
            "points": primary_points,
            "trajectory": primary_traj,
            "rallies": rally_results,
            "stats": stats,
            "cam_order": cam_names,
            "analytics": batch_analytics,
        }

    def get_video_test_status(self) -> dict:
        """Get video test pipeline status (supports both single and parallel)."""
        # Parallel mode: combine status from all handles
        if self._video_test_handles:
            total_frames = 0
            processed_frames = 0
            fps_sum = 0.0
            cameras_done = 0
            error_msg = ""
            per_camera = {}

            for cam_name, handle in self._video_test_handles.items():
                sd = handle.status_dict
                t = sd.get("total_frames", 0)
                p = sd.get("processed_frames", 0)
                total_frames += t
                processed_frames += p
                fps_sum += sd.get("fps", 0.0)
                state = sd.get("state", "idle")
                if state in ("completed", "error"):
                    cameras_done += 1
                if sd.get("error_msg"):
                    error_msg += f"{cam_name}: {sd['error_msg']}; "
                per_camera[cam_name] = {
                    "state": state,
                    "total_frames": t,
                    "processed_frames": p,
                }

            any_error = any(pc["state"] == "error" for pc in per_camera.values())
            all_done = cameras_done == len(self._video_test_handles)

            if all_done and any_error:
                combined_state = "error"
            elif all_done:
                combined_state = "completed"
            elif any(pc["state"] == "running" for pc in per_camera.values()):
                combined_state = "running"
            else:
                combined_state = "starting"

            # Collect ensemble stats from completed cameras
            ensemble_stats = {}
            for cam_name, handle in self._video_test_handles.items():
                sd = handle.status_dict
                es = sd.get("ensemble_stats")
                if es:
                    ensemble_stats[cam_name] = es

            result = {
                "state": combined_state,
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "fps": round(fps_sum, 1),
                "error_msg": error_msg,
                "per_camera": per_camera,
            }
            if ensemble_stats:
                result["ensemble_stats"] = ensemble_stats
            return result

        # Legacy single handle
        handle = self._video_test_handle
        if handle is None or handle.status_dict is None:
            return {"state": "idle"}
        sd = handle.status_dict
        return {
            "state": sd.get("state", "idle"),
            "total_frames": sd.get("total_frames", 0),
            "processed_frames": sd.get("processed_frames", 0),
            "fps": round(sd.get("fps", 0.0), 1),
            "error_msg": sd.get("error_msg", ""),
        }
