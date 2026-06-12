"""Main process orchestrator: manages camera pipeline subprocesses and triangulation."""

import datetime
import json
import logging
import multiprocessing as mp
import os
import queue
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
    _YOLO_FUZZY_BUFFER_SIZE = 600
    _YOLO_FUZZY_EMIT_DELAY_FRAMES = 5
    _YOLO_FUZZY_COOLDOWN_FRAMES = 10
    _YOLO_FUZZY_ANALYSIS_STRIDE = 5
    _CONSUMER_MAX_RESULTS_PER_HANDLE_TICK = 8
    _LIVE_ANALYTICS_EVENT_LIMIT = 120
    _LIVE_ANALYTICS_SPEED_LIMIT = 80
    _DASHBOARD_ANALYTICS_EVENT_LIMIT = 24
    _DASHBOARD_ANALYTICS_SPEED_LIMIT = 12
    _YOLO_OUT_RESTART_HIT_GAP_FRAMES = 100
    _YOLO_OUT_RESTART_SPEED_KMH = 20.0
    _YOLO_OUT_GATE_PENDING_LIMIT = 100
    _YOLO_LIVE_MIN_BOUNCE_FRAME = 50
    _YOLO_LIVE_MIN_BOUNCE_HISTORY = 8
    _YOLO_LIVE_WEAK_NON_REVERSAL_MAX_ANGLE = 45.0
    _YOLO_LIVE_WEAK_NON_REVERSAL_MIN_SCORE = 90.0
    _YOLO_LIVE_DUPLICATE_SPACE_METERS = 2.5
    _YOLO_LIVE_MAX_FRAME_REGRESSION = 25
    _YOLO_LIVE_LATE_RECOVERY_MAX_REGRESSION = 800
    _YOLO_LIVE_BASELINE_STALE_SPEED_FRAMES = 80
    _YOLO_LIVE_BASELINE_STALE_SPEED_MARGIN_M = 0.25
    _YOLO_LIVE_FUTURE_SPEED_MAX_GAP_FRAMES = 120
    _YOLO_LIVE_LOW_SPEED_STALE_CONTEXT_FRAMES = 240
    _YOLO_LIVE_LOW_SPEED_STALE_CONTEXT_KMH = 10.0
    _YOLO_LIVE_FUTURE_LOW_SPEED_MAX_GAP_FRAMES = 300
    _YOLO_LIVE_FUTURE_LOW_SPEED_KMH = 35.0
    _YOLO_LIVE_SLOW_QUEUE_STALE_SPEED_FRAMES = 80
    _YOLO_GAP_FILL_MIN_FRAMES = 225
    _YOLO_GAP_FILL_SHORT_TRACK_MIN_SCORE = 120.0
    _YOLO_GAP_FILL_WEAK_MIN_SCORE = 50.0
    _YOLO_GAP_FILL_WEAK_MIN_DELTA_V = 8.0
    _YOLO_GAP_FILL_MIN_CONFIDENCE = 0.14
    _COURT_HALF_WIDTH_M = 4.115
    _COURT_HALF_LENGTH_M = 11.89
    _SINGLE_CAM_CLAMP_MARGIN_M = 12.0

    def __init__(self, config: AppConfig):
        self.config = config
        self._handles: dict[str, _PipelineHandle] = {}
        self._manager = mp.Manager()

        for cam_name in config.cameras:
            self._handles[cam_name] = _PipelineHandle(cam_name)

        self._latest_detections: dict[str, dict] = {}
        self._latest_frames: dict[str, bytes] = {}
        self._latest_frame_meta: dict[str, dict[str, Any]] = {}
        self._latest_frame_seq: dict[str, int] = {}
        self._latest_frame_condition = threading.Condition()
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
        self._preview_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._inference_enabled: bool = True  # 全局推理开关

        # Ball 3D position queue (most recent 500 points, ~16s at 30fps)
        from collections import deque as _deque
        self._ball_3d_queue: _deque = _deque(maxlen=500)

        # Latest player pose per camera (nearest player to ball)
        self._latest_player_pose: dict[str, dict] = {}
        self._live_player_poses: dict[str, deque[dict]] = {
            cam: deque(maxlen=400) for cam in config.cameras
        }
        self._event_homography_cache: dict[str, Any] = {}

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
        self._total_retracted_live_bounces: int = 0
        self._total_live_hits: int = 0
        self._live_speed_events: list[dict] = []
        self._total_live_speed_events: int = 0
        self._yolo_fuzzy_live_detections: dict[str, deque[dict]] = {
            "cam68": deque(maxlen=self._YOLO_FUZZY_BUFFER_SIZE),
            "cam66": deque(maxlen=self._YOLO_FUZZY_BUFFER_SIZE),
        }
        self._yolo_fuzzy_emitted_frames: dict[str, deque[int]] = {
            "cam68": deque(maxlen=50),
            "cam66": deque(maxlen=50),
        }
        self._yolo_fuzzy_emitted_hit_frames: dict[str, deque[int]] = {
            "cam68": deque(maxlen=50),
            "cam66": deque(maxlen=50),
        }
        self._yolo_fuzzy_hit_suppression_frames: dict[str, deque[int]] = {
            "cam68": deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
            "cam66": deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
        }
        self._yolo_fuzzy_hit_suppressed_bounces: dict[str, deque[dict]] = {
            "cam68": deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
            "cam66": deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
        }
        self._yolo_out_gate_state: dict[str, dict[str, Any]] = {
            "cam68": {},
            "cam66": {},
        }
        self._yolo_out_gate_pending_bounces: dict[str, dict[int, dict[str, Any]]] = {
            "cam68": {},
            "cam66": {},
        }
        self._yolo_fuzzy_emitted_speed_frames: dict[str, deque[int]] = {
            "cam68": deque(maxlen=50),
            "cam66": deque(maxlen=50),
        }
        self._yolo_fuzzy_last_emitted_frame: dict[str, int | None] = {
            "cam68": None,
            "cam66": None,
        }
        self._yolo_fuzzy_last_emitted_hit_frame: dict[str, int | None] = {
            "cam68": None,
            "cam66": None,
        }
        self._yolo_fuzzy_last_emitted_speed_frame: dict[str, int | None] = {
            "cam68": None,
            "cam66": None,
        }
        self._yolo_fuzzy_live_stats: dict[str, dict[str, Any]] = {
            "cam68": {},
            "cam66": {},
        }
        self._yolo_event_task_queues: dict[str, mp.Queue] = {}
        self._yolo_event_result_queues: dict[str, mp.Queue] = {}
        self._yolo_event_stop_events: dict[str, mp.Event] = {}
        self._yolo_event_processes: dict[str, mp.Process] = {}
        self._yolo_event_task_ids: dict[str, int] = {}
        self._yolo_event_last_applied_task_ids: dict[str, int] = {}
        self._reset_yolo_fuzzy_live_locked()
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
        self._dashboard_analytics_cache: dict = {}
        self._pipeline_status_cache: dict[str, dict] = {}
        self._pipeline_status_cache_ts: float = 0.0
        self._pipeline_status_cache_lock = threading.Lock()

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
        self._ws_zero_speed_grace_seconds = 24.0
        self._ws_min_send_interval_seconds = 1.0
        self._ws_last_send_monotonic = 0.0
        self._ws_generation = 0

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
                "inference_ready": False,
                "inference_error": "",
                "detector_stats": None,
                "preview_fps": 0.0,
                "preview_frame_id": None,
            }
        )

        player_cfg = self._player_detection_settings_for_model(model_cfg)
        preview_stride = 1 if self._is_yolo_model_config(model_cfg) else 2
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
                "player_model_path": player_cfg["model_path"] if player_cfg["enabled"] else "",
                "player_device": player_cfg["device"],
                "player_conf": player_cfg["conf"],
                "player_imgsz": player_cfg["imgsz"],
                "player_use_tracking": player_cfg["use_tracking"],
                "player_run_every_n": player_cfg["run_every_n_frames"],
                "preview_stride": preview_stride,
            },
            daemon=True,
        )
        handle.process.start()
        logger.info("[%s] Pipeline process started (pid=%d)", name, handle.process.pid)
        self._candidate_continuity.pop(name, None)
        if self._is_yolo_model_config(model_cfg):
            self._ensure_yolo_event_worker(name)

        self._ensure_worker_threads()

    def _ensure_worker_threads(self) -> None:
        """Start background consumers for preview frames and detection results."""
        self._stopped.clear()
        if self._preview_thread is None or not self._preview_thread.is_alive():
            self._preview_thread = threading.Thread(
                target=self._preview_loop,
                name="preview-consumer",
                daemon=True,
            )
            self._preview_thread.start()
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._consumer_thread = threading.Thread(
                target=self._consume_loop,
                name="result-consumer",
                daemon=True,
            )
            self._consumer_thread.start()
        if self._status_thread is None or not self._status_thread.is_alive():
            self._status_thread = threading.Thread(
                target=self._status_snapshot_loop,
                name="status-snapshot",
                daemon=True,
            )
            self._status_thread.start()

    def _read_pipeline_status_snapshot(self, name: str, handle: _PipelineHandle) -> dict:
        """Read a small pipeline status dict for dashboard/API snapshots."""
        preview_info = self.get_latest_frame_meta(name)
        preview_age_ms = None
        if preview_info is not None and preview_info.get("age_ms") is not None:
            preview_age_ms = round(float(preview_info["age_ms"]), 1)

        data: dict[str, Any] = {}
        if handle.status_dict is not None:
            try:
                # One proxy copy is much cheaper than many manager `.get()` IPC
                # calls, and the HTTP route never touches the proxy directly.
                data = dict(handle.status_dict)
            except Exception:
                logger.debug("Failed to read status snapshot for %s", name, exc_info=True)

        state = str(data.get("state", "stopped") or "stopped")
        is_running = state == "running" and handle.is_alive()
        if not is_running:
            state = "stopped"

        return {
            "name": name,
            "state": state,
            "fps": data.get("fps", 0.0) if is_running else 0.0,
            "last_detection_time": data.get("last_detection_time") if is_running else None,
            "error_msg": data.get("error_msg") or None,
            "inference_enabled": bool(data.get("inference_enabled", True)),
            "inference_ready": bool(data.get("inference_ready", True)),
            "inference_error": data.get("inference_error") or None,
            "detector_stats": data.get("detector_stats") if is_running else None,
            "preview_fps": data.get("preview_fps", 0.0) if is_running else 0.0,
            "preview_frame_id": data.get("preview_frame_id") if is_running else None,
            "latest_preview_seq": preview_info.get("seq") if preview_info else None,
            "latest_preview_frame_id": preview_info.get("frame_id") if preview_info else None,
            "latest_preview_capture_ts": preview_info.get("capture_ts") if preview_info else None,
            "latest_preview_age_ms": preview_age_ms,
        }

    def _refresh_pipeline_status_cache(self) -> None:
        snapshot = {
            name: self._read_pipeline_status_snapshot(name, handle)
            for name, handle in list(self._handles.items())
        }
        with self._pipeline_status_cache_lock:
            self._pipeline_status_cache = snapshot
            self._pipeline_status_cache_ts = time.time()

    def _status_snapshot_loop(self) -> None:
        logger.info("Status snapshot thread started")
        while not self._stopped.is_set():
            try:
                self._refresh_pipeline_status_cache()
            except Exception:
                logger.debug("Status snapshot refresh failed", exc_info=True)
            time.sleep(0.5)

    def _get_pipeline_status_cache(self) -> dict[str, dict]:
        with self._pipeline_status_cache_lock:
            return {name: dict(status) for name, status in self._pipeline_status_cache.items()}

    def _player_detection_settings_for_model(self, model_cfg=None) -> dict[str, Any]:
        player_cfg = self.config.player_detection
        model_cfg = model_cfg or self.config.model
        use_offline_yolo_person = self._is_yolo_model_config(model_cfg)
        if use_offline_yolo_person:
            return {
                "enabled": bool(player_cfg.enabled),
                "model_path": player_cfg.yolo_model_path or player_cfg.model_path,
                "device": player_cfg.device,
                "conf": float(player_cfg.yolo_conf),
                "imgsz": int(player_cfg.yolo_imgsz),
                "use_tracking": bool(player_cfg.yolo_use_tracking),
                "run_every_n_frames": int(player_cfg.yolo_run_every_n_frames),
            }
        return {
            "enabled": bool(player_cfg.enabled),
            "model_path": player_cfg.model_path,
            "device": player_cfg.device,
            "conf": float(player_cfg.conf),
            "imgsz": int(player_cfg.imgsz),
            "use_tracking": bool(player_cfg.use_tracking),
            "run_every_n_frames": int(player_cfg.run_every_n_frames),
        }

    @staticmethod
    def _is_yolo_model_config(model_cfg) -> bool:
        detector_type = (model_cfg.detector_type or "").lower()
        model_path = (model_cfg.path or "").replace("\\", "/").lower()
        return detector_type in {"yolo", "yolo_roadmap"} or "yolo_roadmap/" in model_path

    def _hit_bounce_config_dict(self) -> dict[str, Any]:
        cfg = self.config.hit_bounce_refiner
        return {
            "hit_angle_thresh": float(getattr(cfg, "hit_angle_thresh", 45.0)),
            "bottom_hit_dist_px_net": float(getattr(cfg, "bottom_hit_dist_px_net", 100.0)),
            "bottom_hit_dist_px_base": float(getattr(cfg, "bottom_hit_dist_px_base", 250.0)),
            "lookback_frames": int(getattr(cfg, "lookback_frames", 50) or 50),
            "hit_suppression_frames": int(getattr(cfg, "hit_suppression_frames", 3) or 3),
            "clean_time_frames": int(getattr(cfg, "clean_time_frames", 25) or 25),
            "clean_space_meters": float(getattr(cfg, "clean_space_meters", 1.5)),
        }

    def _ensure_yolo_event_worker(self, cam_name: str) -> bool:
        proc = self._yolo_event_processes.get(cam_name)
        if proc is not None and proc.is_alive():
            return True
        cam_cfg = self.config.cameras.get(cam_name)
        if cam_cfg is None:
            return False

        self._stop_yolo_event_worker(cam_name)
        task_queue: mp.Queue = mp.Queue(maxsize=1)
        result_queue: mp.Queue = mp.Queue(maxsize=8)
        stop_event = mp.Event()
        from app.yolo_event_worker import run_yolo_event_worker

        proc = mp.Process(
            target=run_yolo_event_worker,
            kwargs={
                "camera_name": cam_name,
                "homography_path": self.config.homography.path,
                "homography_key": cam_cfg.homography_key,
                "hit_bounce_config": self._hit_bounce_config_dict(),
                "task_queue": task_queue,
                "result_queue": result_queue,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        proc.start()
        self._yolo_event_task_queues[cam_name] = task_queue
        self._yolo_event_result_queues[cam_name] = result_queue
        self._yolo_event_stop_events[cam_name] = stop_event
        self._yolo_event_processes[cam_name] = proc
        self._yolo_event_task_ids[cam_name] = 0
        self._yolo_event_last_applied_task_ids[cam_name] = 0
        logger.info("[%s] YOLO event worker started (pid=%d)", cam_name, proc.pid)
        return True

    def _stop_yolo_event_worker(self, cam_name: str) -> None:
        stop_event = self._yolo_event_stop_events.pop(cam_name, None)
        task_queue = self._yolo_event_task_queues.pop(cam_name, None)
        result_queue = self._yolo_event_result_queues.pop(cam_name, None)
        proc = self._yolo_event_processes.pop(cam_name, None)
        self._yolo_event_task_ids.pop(cam_name, None)
        self._yolo_event_last_applied_task_ids.pop(cam_name, None)
        if stop_event is not None:
            stop_event.set()
        if task_queue is not None:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
        for q in (task_queue, result_queue):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass

    def _submit_yolo_event_task_locked(
        self,
        cam_name: str,
        *,
        latest_frame: int,
        detections: list[dict],
        player_poses: list[dict],
    ) -> bool:
        if not self._ensure_yolo_event_worker(cam_name):
            return False
        task_queue = self._yolo_event_task_queues.get(cam_name)
        if task_queue is None:
            return False
        task_id = int(self._yolo_event_task_ids.get(cam_name, 0)) + 1
        self._yolo_event_task_ids[cam_name] = task_id
        task = {
            "task_id": task_id,
            "latest_frame": latest_frame,
            "detections": detections,
            "player_poses": player_poses,
        }
        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        stats["worker_enabled"] = True
        try:
            task_queue.put_nowait(task)
            stats["worker_last_submitted_task_id"] = task_id
            return True
        except queue.Full:
            try:
                task_queue.get_nowait()
                stats["worker_submit_replaced_busy"] = int(
                    stats.get("worker_submit_replaced_busy", 0)
                ) + 1
                task_queue.put_nowait(task)
                stats["worker_last_submitted_task_id"] = task_id
                return True
            except Exception:
                stats["worker_submit_skipped_busy"] = int(
                    stats.get("worker_submit_skipped_busy", 0)
                ) + 1
                return True
        except Exception:
            stats["worker_submit_dropped"] = int(stats.get("worker_submit_dropped", 0)) + 1
            return False

    def stop_pipeline(self, name: str) -> None:
        if name not in self._handles:
            raise ValueError(f"Unknown pipeline: {name}")
        self._stop_yolo_event_worker(name)
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
        self._clear_latest_frame(name)
        self._candidate_continuity.pop(name, None)
        logger.info("[%s] Pipeline stopped", name)

        # Auto-save debug output if there's data
        if self._debug_data["trajectory"]:
            try:
                self.save_debug_output()
            except Exception as e:
                logger.warning("Failed to auto-save debug output: %s", e)

    def reload_homography(self, name: str, *, restart_pipeline: bool = False) -> dict:
        """Clear cached homography state after the matrix file is updated.

        Running pipeline processes load homography at process start, so a full
        camera restart is needed for pixel->court conversion to change there.
        """
        if name not in self._handles:
            raise ValueError(f"Unknown pipeline: {name}")

        handle = self._handles[name]
        was_running = bool(handle.is_alive())

        self._blob_homographies.pop(name, None)
        if hasattr(self, "_player_homographies"):
            try:
                self._player_homographies.pop(name, None)
            except Exception:
                pass
        self._candidate_continuity.pop(name, None)
        self._stop_yolo_event_worker(name)

        restarted = False
        if restart_pipeline and was_running:
            self.stop_pipeline(name)
            self.start_pipeline(name)
            restarted = True

        return {
            "camera": name,
            "was_running": was_running,
            "restart_requested": bool(restart_pipeline),
            "restarted": restarted,
            "note": "Restart the camera pipeline for running processes to use the new homography."
            if not restarted
            else "Camera pipeline restarted with the new homography.",
        }

    def shutdown(self) -> None:
        self._stopped.set()
        for cam_name in list(self._yolo_event_processes):
            self._stop_yolo_event_worker(cam_name)
        for name in list(self._handles):
            self.stop_pipeline(name)
        self._manager.shutdown()

    def _store_latest_frame(
        self,
        name: str,
        jpeg: bytes,
        *,
        frame_id: int | None = None,
        capture_ts: float | None = None,
        source_width: int | None = None,
        source_height: int | None = None,
        preview_width: int | None = None,
        preview_height: int | None = None,
    ) -> None:
        """Publish a fresh preview frame and wake MJPEG consumers."""
        now = time.time()
        with self._latest_frame_condition:
            seq = self._latest_frame_seq.get(name, 0) + 1
            self._latest_frames[name] = jpeg
            self._latest_frame_seq[name] = seq
            self._latest_frame_meta[name] = {
                "seq": seq,
                "frame_id": frame_id,
                "capture_ts": capture_ts,
                "updated_ts": now,
                "source_width": source_width,
                "source_height": source_height,
                "preview_width": preview_width,
                "preview_height": preview_height,
            }
            self._latest_frame_condition.notify_all()

    def _clear_latest_frame(self, name: str) -> None:
        with self._latest_frame_condition:
            self._latest_frames.pop(name, None)
            self._latest_frame_meta.pop(name, None)
            self._latest_frame_seq.pop(name, None)
            self._latest_frame_condition.notify_all()

    def _copy_latest_frame_info_locked(
        self,
        name: str,
        *,
        include_jpeg: bool = True,
    ) -> dict | None:
        jpeg = self._latest_frames.get(name)
        if jpeg is None:
            return None
        meta = dict(self._latest_frame_meta.get(name, {}))
        meta.setdefault("seq", self._latest_frame_seq.get(name, 0))
        if include_jpeg:
            meta["jpeg"] = jpeg
        updated_ts = meta.get("updated_ts")
        if updated_ts is not None:
            meta["age_ms"] = max(0.0, (time.time() - float(updated_ts)) * 1000.0)
        return meta

    def get_latest_frame_info(self, name: str) -> dict | None:
        with self._latest_frame_condition:
            return self._copy_latest_frame_info_locked(name)

    def get_latest_frame_meta(self, name: str) -> dict | None:
        with self._latest_frame_condition:
            return self._copy_latest_frame_info_locked(name, include_jpeg=False)

    def wait_for_latest_frame(
        self,
        name: str,
        *,
        after_seq: int | None = None,
        timeout: float = 1.0,
    ) -> dict | None:
        """Wait until a preview frame newer than ``after_seq`` is available."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._latest_frame_condition:
            while True:
                info = self._copy_latest_frame_info_locked(name)
                if info is not None:
                    seq = info.get("seq")
                    if after_seq is None or seq != after_seq:
                        return info
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._latest_frame_condition.wait(remaining)

    def _handle_preview_payload(self, name: str, payload: Any) -> None:
        frame_id = None
        capture_ts = None
        if isinstance(payload, dict):
            preview_jpeg = payload.get("preview")
            recording_jpeg = payload.get("recording")
            frame_id = payload.get("frame_id")
            capture_ts = payload.get("capture_ts")
            source_width = payload.get("source_width")
            source_height = payload.get("source_height")
            preview_width = payload.get("preview_width")
            preview_height = payload.get("preview_height")
        else:
            preview_jpeg = payload
            recording_jpeg = payload if self._recording else None
            source_width = None
            source_height = None
            preview_width = None
            preview_height = None
        if preview_jpeg is not None:
            self._store_latest_frame(
                name,
                preview_jpeg,
                frame_id=frame_id,
                capture_ts=capture_ts,
                source_width=source_width,
                source_height=source_height,
                preview_width=preview_width,
                preview_height=preview_height,
            )
        with self._recording_lock:
            if self._recording and recording_jpeg is not None:
                self._write_recording_frame(name, recording_jpeg)

    def _preview_loop(self) -> None:
        logger.info("Preview consumer thread started")
        while not self._stopped.is_set():
            got_any = False
            for name, handle in list(self._handles.items()):
                if handle.frame_queue is None:
                    continue
                try:
                    new_payload = None
                    while True:
                        try:
                            new_payload = handle.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    if new_payload is not None:
                        self._handle_preview_payload(name, new_payload)
                        got_any = True
                except Exception:
                    logger.debug("Preview consumer error for %s", name, exc_info=True)
            if not got_any:
                time.sleep(0.005)

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
                # Preview frames drive the visible monitoring UI. Drain them
                # before heavier detection/event work so long YOLO refiner
                # passes cannot freeze the main playback image.
                if False and handle.frame_queue is not None:
                    try:
                        new_payload = None
                        while True:
                            try:
                                new_payload = handle.frame_queue.get_nowait()
                            except queue.Empty:
                                break
                        if new_payload is not None:
                            frame_id = None
                            capture_ts = None
                            if isinstance(new_payload, dict):
                                preview_jpeg = new_payload.get("preview")
                                recording_jpeg = new_payload.get("recording")
                                frame_id = new_payload.get("frame_id")
                                capture_ts = new_payload.get("capture_ts")
                            else:
                                preview_jpeg = new_payload
                                recording_jpeg = new_payload if self._recording else None
                            if preview_jpeg is not None:
                                self._store_latest_frame(
                                    name,
                                    preview_jpeg,
                                    frame_id=frame_id,
                                    capture_ts=capture_ts,
                                )
                            with self._recording_lock:
                                if self._recording and recording_jpeg is not None:
                                    self._write_recording_frame(name, recording_jpeg)
                            got_any = True
                    except Exception:
                        pass

                # 消费检测结果 — 每个检测都保存，不丢弃
                if handle.result_queue is not None:
                    try:
                        processed_results = 0
                        while processed_results < self._CONSUMER_MAX_RESULTS_PER_HANDLE_TICK:
                            try:
                                det = handle.result_queue.get_nowait()
                            except queue.Empty:
                                break
                            processed_results += 1

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
                            if det.get("event_only_raw_candidates"):
                                if self._is_yolo_roadmap_active() and name == "cam68":
                                    with self._analytics_lock:
                                        self._run_yolo_fuzzy_single_cam_locked(name, det)
                                    got_any = True
                                continue

                            if name in tri_cams:
                                det = self._apply_live_candidate_continuity(
                                    name,
                                    det,
                                    max_candidates=self._LIVE_MATCHER_CANDIDATES,
                                )
                                self._det_queues.setdefault(name, []).append(det)
                            self._latest_detections[name] = det
                            if self._is_yolo_roadmap_active() and name == "cam68":
                                with self._analytics_lock:
                                    self._run_yolo_fuzzy_single_cam_locked(name, det)
                            if name.startswith("_video_test"):
                                cam = det.get("camera_name", "unknown")
                                self._video_test_detections.setdefault(cam, []).append(det)
                            got_any = True
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
        self._live_player_poses.setdefault(cam_name, deque(maxlen=400)).append(dict(msg))

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
        return PipelineStatus(**self._read_pipeline_status_snapshot(name, handle))

    def _latest_detection_summary(self, *, max_candidates: int = 4) -> dict | None:
        det_summary = {}
        for cam_name, det in self._latest_detections.items():
            if det is None:
                continue
            candidates = list(det.get("candidates", []) or [])[:max(0, max_candidates)]
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
        return det_summary or None

    @staticmethod
    def _compact_detector_stats(stats: dict | None) -> dict | None:
        if not stats:
            return None
        keep_keys = (
            "type",
            "track_available",
            "frame",
            "active_static_zones",
            "static_starvation",
            "static_fail_open_remaining",
            "raw_detections",
            "kept_detections",
            "static_blocked",
            "static_zones_created",
            "static_zones_expired",
            "motion_released",
            "fail_open_kept",
            "untracked_kept",
            "pseudo_tracked",
        )
        return {key: stats.get(key) for key in keep_keys if key in stats}

    def get_system_status(self) -> SystemStatus:
        pipelines = {n: self.get_pipeline_status(n) for n in self._handles}
        return SystemStatus(
            pipelines=pipelines,
            triangulation_active=self._triangulation_active,
            latest_ball_3d=self._latest_3d,
            analytics=self.get_live_analytics(),
            latest_detections=self._latest_detection_summary(max_candidates=4),
        )

    def get_dashboard_status(self) -> dict:
        """Return the compact payload used by the live dashboard poll loop."""
        pipelines: dict[str, dict] = {}
        cached = self._get_pipeline_status_cache()
        if not cached:
            self._refresh_pipeline_status_cache()
            cached = self._get_pipeline_status_cache()
        for name in self._handles:
            status = dict(cached.get(name) or {"name": name, "state": "stopped"})
            status["detector_stats"] = self._compact_detector_stats(status.get("detector_stats"))
            pipelines[name] = status
        return {
            "pipelines": pipelines,
            "triangulation_active": self._triangulation_active,
        }

    def get_dashboard_live_payload(self) -> dict:
        """Return the lightweight live overlay/minimap payload."""
        return {
            "latest_ball_3d": self._latest_3d.model_dump() if self._latest_3d is not None else None,
            "analytics": self.get_live_analytics(compact=True),
            "latest_detections": self._latest_detection_summary(max_candidates=2),
            "server_ts": time.time(),
        }

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

    def _is_yolo_roadmap_active(self) -> bool:
        detector_type = (self.config.model.detector_type or "").lower()
        model_path = (self.config.model.path or "").replace("\\", "/").lower()
        return detector_type in {"yolo", "yolo_roadmap"} or "yolo_roadmap/" in model_path

    def _reset_yolo_fuzzy_live_locked(self) -> None:
        for cam, buf in self._yolo_fuzzy_live_detections.items():
            buf.clear()
            self._yolo_fuzzy_emitted_frames.setdefault(cam, deque(maxlen=50)).clear()
            self._yolo_fuzzy_emitted_hit_frames.setdefault(cam, deque(maxlen=50)).clear()
            self._yolo_fuzzy_hit_suppression_frames.setdefault(
                cam,
                deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
            ).clear()
            self._yolo_fuzzy_hit_suppressed_bounces.setdefault(
                cam,
                deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
            ).clear()
            self._yolo_fuzzy_emitted_speed_frames.setdefault(cam, deque(maxlen=50)).clear()
            self._yolo_fuzzy_last_emitted_frame[cam] = None
            self._yolo_fuzzy_last_emitted_hit_frame[cam] = None
            self._yolo_fuzzy_last_emitted_speed_frame[cam] = None
            self._yolo_out_gate_state[cam] = {}
            self._yolo_out_gate_pending_bounces[cam] = {}
            self._yolo_fuzzy_live_stats[cam] = {
                "detector": "yolo_events_single_cam",
                "detections": 0,
                "buffered": 0,
                "candidate_bounces": 0,
                "candidate_hits": 0,
                "candidate_speed_events": 0,
                "accepted": 0,
                "accepted_hits": 0,
                "accepted_speed_events": 0,
                "last_frame": None,
                "last_candidate_frame": None,
                "last_hit_frame": None,
                "last_speed_frame": None,
                "last_reject_reason": "",
                "player_pose_buffered": 0,
                "analysis_stride": self._YOLO_FUZZY_ANALYSIS_STRIDE,
                "analysis_calls": 0,
                "skipped_analysis_stride": 0,
                "last_analysis_ms": 0.0,
            }

    @staticmethod
    def _nearest_detection_for_frame(detections: list[dict], frame_index: int) -> dict | None:
        if not detections:
            return None
        return min(
            detections,
            key=lambda d: abs(int(d.get("frame_index", frame_index)) - frame_index),
        )

    def _event_homography_for_camera(self, cam_name: str):
        if cam_name in self._event_homography_cache:
            return self._event_homography_cache[cam_name]
        cam_cfg = self.config.cameras.get(cam_name)
        if cam_cfg is None:
            self._event_homography_cache[cam_name] = None
            return None
        try:
            from app.pipeline.homography import HomographyTransformer
            homography = HomographyTransformer(
                self.config.homography.path,
                cam_cfg.homography_key,
            )
        except Exception as e:
            logger.warning("Live event homography unavailable for %s: %s", cam_name, e)
            homography = None
        self._event_homography_cache[cam_name] = homography
        return homography

    def _can_emit_yolo_event(
        self,
        *,
        event_frame: int,
        latest_frame: int,
        last_emitted_frame: int | None,
        seen_frames: deque[int],
        cooldown_frames: int | None = None,
    ) -> bool:
        if latest_frame - event_frame < self._YOLO_FUZZY_EMIT_DELAY_FRAMES:
            return False
        cooldown = (
            self._YOLO_FUZZY_COOLDOWN_FRAMES
            if cooldown_frames is None
            else max(0, int(cooldown_frames))
        )
        if cooldown == 0:
            return event_frame not in seen_frames
        if (
            last_emitted_frame is not None
            and event_frame <= last_emitted_frame + cooldown
        ):
            return False
        return not any(
            abs(event_frame - old_frame) <= cooldown
            for old_frame in seen_frames
        )

    def _reject_stale_yolo_bounce_frame_locked(
        self,
        cam_name: str,
        event_frame: int,
        stats: dict,
        event: dict | None = None,
    ) -> bool:
        """Reject YOLO bounces that arrive far behind the already published stream."""
        last_emitted_frame = self._yolo_fuzzy_last_emitted_frame.get(cam_name)
        if last_emitted_frame is None:
            return False
        max_regression = int(self._YOLO_LIVE_MAX_FRAME_REGRESSION)
        if event_frame >= int(last_emitted_frame) - max_regression:
            return False
        if event is not None and self._allow_late_yolo_bounce_recovery_locked(
            cam_name,
            event,
            event_frame=event_frame,
            last_emitted_frame=int(last_emitted_frame),
            stats=stats,
        ):
            return False
        stats["skipped_stale_yolo_bounces"] = int(
            stats.get("skipped_stale_yolo_bounces", 0)
        ) + 1
        stats["last_reject_reason"] = (
            f"stale_yolo_frame:{event_frame}<last:{last_emitted_frame}"
        )
        stats["last_stale_yolo_frame"] = event_frame
        stats["last_stale_yolo_last_emitted_frame"] = int(last_emitted_frame)
        self._remember_yolo_bounce_reject_locked(
            stats,
            event or {"frame_index": event_frame},
            f"stale_yolo_frame:{event_frame}<last:{last_emitted_frame}",
        )
        return True

    def _allow_late_yolo_bounce_recovery_locked(
        self,
        cam_name: str,
        event: dict,
        *,
        event_frame: int,
        last_emitted_frame: int,
        stats: dict,
    ) -> bool:
        """Allow select high-quality YOLO bounces that arrive out of order."""
        if not self._is_yolo_queue_event(event):
            return False
        regression = int(last_emitted_frame) - int(event_frame)
        if regression <= 0 or regression > int(self._YOLO_LIVE_LATE_RECOVERY_MAX_REGRESSION):
            return False
        if self._is_duplicate_yolo_live_bounce_locked(cam_name, event):
            return False

        confidence = self._event_float(event, "confidence") or 0.0
        score = self._yolo_bounce_signal_score(event)
        delta_v = self._event_float(event, "delta_v") or 0.0
        queue_speed = self._event_float(event, "queue_speed_px") or 0.0
        try:
            history = int(event.get("queue_history_len") or 0)
        except Exception:
            history = 0

        strong_shape = (
            confidence >= 0.30
            and score >= 300.0
            and delta_v >= 10.0
            and queue_speed >= 7.0
            and history >= 8
        )
        steady_queue_shape = (
            confidence >= 0.30
            and score >= 145.0
            and delta_v >= 2.5
            and queue_speed >= 8.0
            and history >= 12
        )
        if not (strong_shape or steady_queue_shape):
            return False

        event["late_yolo_recovery"] = True
        event["late_yolo_recovery_regression_frames"] = int(regression)
        stats["allowed_late_yolo_recovery_bounces"] = int(
            stats.get("allowed_late_yolo_recovery_bounces", 0)
        ) + 1
        stats["last_late_yolo_recovery_frame"] = int(event_frame)
        stats["last_late_yolo_recovery_regression_frames"] = int(regression)
        return True

    def _remember_yolo_bounce_reject_locked(
        self,
        stats: dict,
        event: dict,
        reason: str,
    ) -> None:
        entry = {
            "frame": self._event_frame(event),
            "reason": reason,
            "x": event.get("raw_x", event.get("x")),
            "y": event.get("raw_y", event.get("y")),
            "source": event.get("source"),
            "queue_id": event.get("queue_id"),
            "queue_history_len": event.get("queue_history_len"),
            "queue_speed_px": event.get("queue_speed_px"),
            "confidence": event.get("confidence"),
            "angle": event.get("angle"),
            "delta_v": event.get("delta_v"),
            "bounce_signal_score": event.get("bounce_signal_score"),
            "slow_queue_shape_override": event.get("slow_queue_shape_override"),
            "speed_frame_gap": event.get("speed_frame_gap"),
            "speed_source": event.get("speed_source"),
        }
        recent = stats.get("recent_rejected_bounces")
        if not isinstance(recent, list):
            recent = []
        recent.append(entry)
        stats["recent_rejected_bounces"] = recent[-80:]

    def _set_yolo_bounce_debug_list_locked(
        self,
        stats: dict,
        key: str,
        events: list[dict] | tuple[dict, ...] | None,
        *,
        latest_frame: int | None = None,
        limit: int = 80,
    ) -> None:
        rows: list[dict] = []
        for event in list(events or [])[-limit:]:
            if not isinstance(event, dict):
                continue
            rows.append(
                {
                    "frame": self._event_frame(event),
                    "x": event.get("raw_x", event.get("x")),
                    "y": event.get("raw_y", event.get("y")),
                    "source": event.get("source"),
                    "type": event.get("type"),
                    "in_court": event.get("in_court"),
                    "publish_suppression_reason": event.get("publish_suppression_reason"),
                    "deduped_by_frame": event.get("deduped_by_frame"),
                    "queue_id": event.get("queue_id"),
                    "queue_history_len": event.get("queue_history_len"),
                    "queue_speed_px": event.get("queue_speed_px"),
                    "confidence": event.get("confidence"),
                    "angle": event.get("angle"),
                    "delta_v": event.get("delta_v"),
                    "bounce_signal_score": event.get("bounce_signal_score"),
                    "slow_queue_shape_override": event.get("slow_queue_shape_override"),
                    "near_side_low_delta_override": event.get("near_side_low_delta_override"),
                    "late_yolo_recovery": event.get("late_yolo_recovery"),
                    "late_yolo_recovery_regression_frames": event.get("late_yolo_recovery_regression_frames"),
                }
            )
        stats[key] = rows
        if key.startswith("last_"):
            history_key = "recent_" + key[len("last_"):]
            history = stats.get(history_key)
            if not isinstance(history, list):
                history = []
            for row in rows:
                if history and all(
                    history[-1].get(field) == row.get(field)
                    for field in ("frame", "source", "publish_suppression_reason")
                ):
                    continue
                history.append(row)
            stats[history_key] = history[-300:]
        if latest_frame is not None:
            stats[f"{key}_latest_frame"] = int(latest_frame)

    @staticmethod
    def _event_frame(event: dict) -> int | None:
        frame = event.get("frame_index", event.get("frame"))
        if frame is None:
            return None
        try:
            return int(frame)
        except Exception:
            return None

    def _event_capture_ts(self, buffered: list[dict], event_frame: int, now: float) -> float:
        src_det = self._nearest_detection_for_frame(buffered, event_frame)
        if src_det is None:
            return now
        event_capture_ts = src_det.get("capture_ts", src_det.get("timestamp", now))
        return float(event_capture_ts if event_capture_ts is not None else now)

    def _run_yolo_fuzzy_single_cam_locked(self, cam_name: str, det: dict | None) -> dict | None:
        """Run the cam68/cam66 YOLO roadmap single-camera event chain."""
        if not self._bounce_detection_enabled or det is None:
            return None
        frame_index = det.get("frame_index")
        if frame_index is None:
            return None
        try:
            frame_index = int(frame_index)
        except Exception:
            return None

        buf = self._yolo_fuzzy_live_detections.setdefault(
            cam_name,
            deque(maxlen=self._YOLO_FUZZY_BUFFER_SIZE),
        )
        seen = self._yolo_fuzzy_emitted_frames.setdefault(cam_name, deque(maxlen=50))
        seen_hits = self._yolo_fuzzy_emitted_hit_frames.setdefault(cam_name, deque(maxlen=50))
        seen_speeds = self._yolo_fuzzy_emitted_speed_frames.setdefault(cam_name, deque(maxlen=50))
        last_emitted_frame = self._yolo_fuzzy_last_emitted_frame.get(cam_name)
        last_hit_frame = self._yolo_fuzzy_last_emitted_hit_frame.get(cam_name)
        last_speed_frame = self._yolo_fuzzy_last_emitted_speed_frame.get(cam_name)
        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        buf.append(det)
        stats["detector"] = "yolo_events_single_cam"
        stats["detections"] = int(stats.get("detections", 0)) + 1
        stats["buffered"] = len(buf)
        stats["last_frame"] = frame_index
        stats["analysis_stride"] = self._YOLO_FUZZY_ANALYSIS_STRIDE
        emitted_from_worker = self._drain_yolo_event_results_locked(cam_name)

        last_analyzed_frame = stats.get("last_analyzed_frame")
        if (
            last_analyzed_frame is not None
            and frame_index - int(last_analyzed_frame) < self._YOLO_FUZZY_ANALYSIS_STRIDE
        ):
            stats["skipped_analysis_stride"] = int(stats.get("skipped_analysis_stride", 0)) + 1
            return None
        stats["last_analyzed_frame"] = frame_index
        stats["analysis_calls"] = int(stats.get("analysis_calls", 0)) + 1

        from app.pipeline.yolo_bounce_filter import (
            detect_single_camera_events,
            filter_dashboard_yolo_publishable_bounces,
        )

        buffered = list(buf)
        lookback_frames = int(
            getattr(self.config.hit_bounce_refiner, "lookback_frames", 50) or 50
        )
        buffered_frames = [
            int(d.get("frame_index"))
            for d in buffered
            if d.get("frame_index") is not None
        ]
        min_buffer_frame = min(buffered_frames) if buffered_frames else frame_index
        max_buffer_frame = max(buffered_frames) if buffered_frames else frame_index
        player_poses = []
        for pose in self._live_player_poses.get(cam_name, []):
            pose_frame = pose.get("frame_index", pose.get("frame_id"))
            try:
                pose_frame = int(pose_frame)
            except Exception:
                continue
            if min_buffer_frame - lookback_frames <= pose_frame <= max_buffer_frame + 3:
                player_poses.append(pose)

        handle = self._handles.get(cam_name)
        use_worker = (
            handle is not None
            and handle.is_alive()
            and self._is_yolo_roadmap_active()
        )
        if use_worker and self._submit_yolo_event_task_locked(
            cam_name,
            latest_frame=frame_index,
            detections=buffered,
            player_poses=player_poses,
        ):
            stats["player_pose_buffered"] = len(player_poses)
            return emitted_from_worker

        analysis_t0 = time.perf_counter()
        hb_cfg = self.config.hit_bounce_refiner
        result = detect_single_camera_events(
            buffered,
            camera_name=cam_name,
            player_pose_messages=player_poses,
            homography=self._event_homography_for_camera(cam_name),
            hit_angle_thresh=float(getattr(hb_cfg, "hit_angle_thresh", 45.0)),
            hit_dist_px_net=float(getattr(hb_cfg, "bottom_hit_dist_px_net", 100.0)),
            hit_dist_px_base=float(getattr(hb_cfg, "bottom_hit_dist_px_base", 250.0)),
            lookback_frames=int(getattr(hb_cfg, "lookback_frames", 50) or 50),
            hit_suppress_frames=int(getattr(hb_cfg, "hit_suppression_frames", 3) or 3),
            clean_time_frames=int(getattr(hb_cfg, "clean_time_frames", 25) or 25),
            clean_space_meters=float(getattr(hb_cfg, "clean_space_meters", 1.5)),
        )
        stats["last_analysis_ms"] = round((time.perf_counter() - analysis_t0) * 1000.0, 2)
        bounce_events = result.get("bounces", [])
        hit_events = result.get("hits", [])
        speed_events = result.get("speed_events", [])
        gate_only_bounce_events = result.get("gate_only_bounces", []) or []
        self._set_yolo_bounce_debug_list_locked(
            stats,
            "last_raw_yolo_bounce_candidates",
            bounce_events,
            latest_frame=frame_index,
        )
        stats["candidate_bounces"] = int(result.get("count", len(bounce_events)) or 0)
        stats["raw_bounce_candidates"] = int(result.get("raw_bounce_candidate_count", 0) or 0)
        stats["suppressed_bounces_by_hit_window"] = int(result.get("suppressed_bounces_by_hit_window", 0) or 0)
        stats["deduped_bounces_after_hit"] = int(result.get("deduped_bounces_after_hit", 0) or 0)
        stats["gate_only_bounces"] = int(result.get("gate_only_bounce_count", len(gate_only_bounce_events)) or 0)
        stats["out_rally_suppressed_bounces"] = int(result.get("out_rally_suppressed_bounce_count", 0) or 0)
        stats["candidate_hits"] = int(result.get("hit_count", len(hit_events)) or 0)
        stats["candidate_speed_events"] = int(result.get("speed_count", len(speed_events)) or 0)
        stats["auxiliary_fallback_bounce_count"] = int(result.get("auxiliary_fallback_bounce_count", 0) or 0)
        stats["auxiliary_fallback_speed_count"] = int(result.get("auxiliary_fallback_speed_count", 0) or 0)
        stats["auxiliary_fallback_speed_ignored_count"] = int(
            result.get("auxiliary_fallback_speed_ignored_count", 0) or 0
        )
        stats["player_pose_buffered"] = len(player_poses)
        if result.get("queue_tracker_stats"):
            stats["queue_tracker_stats"] = dict(result.get("queue_tracker_stats") or {})
        self._remember_yolo_hit_suppressed_result_bounces_locked(cam_name, result, stats)
        self._retract_yolo_live_bounces_shadowing_hit_suppressed_locked(cam_name, stats)

        suppression_frames = self._yolo_fuzzy_hit_suppression_frames.setdefault(
            cam_name,
            deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
        )
        known_suppression_frames = set(int(f) for f in suppression_frames)
        for hit_frame in result.get("hit_suppression_frames", []) or []:
            try:
                hit_frame_int = int(hit_frame)
            except Exception:
                continue
            if hit_frame_int in known_suppression_frames:
                continue
            suppression_frames.append(hit_frame_int)
            known_suppression_frames.add(hit_frame_int)
        hit_suppress_frames = int(
            getattr(self.config.hit_bounce_refiner, "hit_suppression_frames", 3) or 0
        )
        stats["hit_suppression_frames"] = len(suppression_frames)
        latest_frame = frame_index
        release_delay = int(getattr(self.config.hit_bounce_refiner, "release_delay_frames", 50) or 0)
        yolo_bounce_events = [event for event in bounce_events if self._is_yolo_queue_event(event)]
        passthrough_bounce_events = [event for event in bounce_events if not self._is_yolo_queue_event(event)]
        publish_suppressed_events = []
        if yolo_bounce_events:
            publish_filter = filter_dashboard_yolo_publishable_bounces(
                yolo_bounce_events,
                hit_events=hit_events,
                latest_frame=latest_frame,
                hit_suppress_frames=hit_suppress_frames,
                clean_time_frames=int(getattr(self.config.hit_bounce_refiner, "clean_time_frames", 25) or 25),
                clean_space_meters=float(getattr(self.config.hit_bounce_refiner, "clean_space_meters", 1.5)),
                release_delay_frames=release_delay,
            )
            publish_suppressed_events = (
                publish_filter.get("suppressed_bounces")
                or publish_filter.get("suppressed")
                or []
            )
            bounce_events = passthrough_bounce_events + (publish_filter.get("bounces") or [])
            stats["publish_suppression_counts"] = publish_filter.get("suppression_counts", {})
            stats["publish_suppressed_bounces"] = len(publish_suppressed_events)
        self._set_yolo_bounce_debug_list_locked(
            stats,
            "last_publishable_yolo_bounces",
            bounce_events,
            latest_frame=frame_index,
        )
        self._set_yolo_bounce_debug_list_locked(
            stats,
            "last_suppressed_yolo_bounces",
            publish_suppressed_events,
            latest_frame=frame_index,
        )

        emitted = None
        last_speed_frame = self._record_yolo_speed_events_locked(
            cam_name,
            speed_events,
            latest_frame=latest_frame,
            last_speed_frame=last_speed_frame,
            seen_speeds=seen_speeds,
            stats=stats,
            buffered=buffered,
        )
        self._record_yolo_gate_only_bounces_locked(
            cam_name,
            gate_only_bounce_events,
            latest_frame=latest_frame,
            hit_events=hit_events,
            speed_events=speed_events,
            stats=stats,
        )
        for event in bounce_events:
            event_frame = self._event_frame(event)
            if event_frame is None:
                continue
            if self._reject_stale_yolo_bounce_frame_locked(cam_name, event_frame, stats, event):
                continue
            quality_reject_reason = self._reject_yolo_live_bounce_quality_locked(
                event,
                event_frame=event_frame,
                stats=stats,
            )
            if quality_reject_reason:
                if not self._allow_yolo_gap_fill_quality_bounce_locked(
                    cam_name,
                    event,
                    event_frame=event_frame,
                    quality_reject_reason=quality_reject_reason,
                    stats=stats,
                ):
                    stats["last_reject_reason"] = quality_reject_reason
                    self._remember_yolo_bounce_reject_locked(
                        stats,
                        event,
                        quality_reject_reason,
                    )
                    continue
            suppressing_hit_frame = None
            if hit_suppress_frames > 0:
                for hit_frame in list(suppression_frames):
                    if abs(event_frame - int(hit_frame)) <= hit_suppress_frames:
                        suppressing_hit_frame = int(hit_frame)
                        break
                if suppressing_hit_frame is None:
                    for hit in self._live_hits:
                        hit_frame = self._event_frame(hit)
                        if hit_frame is not None and abs(event_frame - hit_frame) <= hit_suppress_frames:
                            suppressing_hit_frame = hit_frame
                            break
            if suppressing_hit_frame is not None:
                if self._has_live_yolo_hit_frame_locked(cam_name, suppressing_hit_frame):
                    self._remember_yolo_hit_suppressed_bounce_locked(
                        cam_name,
                        event,
                        suppressing_hit_frame=suppressing_hit_frame,
                    )
                stats["skipped_persistent_hit_suppressed_bounces"] = int(
                    stats.get("skipped_persistent_hit_suppressed_bounces", 0)
                ) + 1
                stats["last_reject_reason"] = f"hit_window:{suppressing_hit_frame}"
                continue
            shadow_frame = self._yolo_hit_suppressed_duplicate_frame_locked(cam_name, event)
            if shadow_frame is not None:
                stats["skipped_hit_suppressed_duplicate_bounces"] = int(
                    stats.get("skipped_hit_suppressed_duplicate_bounces", 0)
                ) + 1
                stats["last_reject_reason"] = f"hit_window_shadow:{shadow_frame}"
                continue
            if release_delay > 0 and latest_frame - event_frame + 1 < release_delay:
                stats["pending_release_delay_frames"] = release_delay
                stats["pending_release_bounces"] = int(stats.get("pending_release_bounces", 0)) + 1
                continue
            if not self._can_emit_yolo_event(
                event_frame=event_frame,
                latest_frame=latest_frame,
                last_emitted_frame=last_emitted_frame,
                seen_frames=seen,
                cooldown_frames=0,
            ):
                continue
            self._prime_yolo_out_gate_restarts_locked(
                cam_name,
                hit_events=hit_events,
                speed_events=speed_events,
                candidate_frame=event_frame,
                latest_frame=latest_frame,
            )

            now = time.time()
            event_capture_ts = self._event_capture_ts(buffered, event_frame, now)
            bd = {
                "frame": event_frame,
                "frame_index": event_frame,
                "x": event.get("x"),
                "y": event.get("y"),
                "z": 0.0,
                "pixel_x": event.get("pixel_x"),
                "pixel_y": event.get("pixel_y"),
                "camera": cam_name,
                "camera_name": cam_name,
                "type": event.get("type", "IN"),
                "in_court": bool(event.get("in_court", True)),
                "confidence": event.get("confidence", 0.0),
                "timestamp": event_capture_ts,
                "capture_ts": event_capture_ts,
                "detect_delay": round(now - event_capture_ts, 2),
                "bounce_mode": f"mono_{cam_name}",
                "source": event.get("source", "yolo_fuzzy_single_cam"),
                "angle": event.get("angle"),
                "delta_v": event.get("delta_v"),
                "y_reversal": event.get("y_reversal"),
                "queue_id": event.get("queue_id"),
                "queue_history_len": event.get("queue_history_len"),
                "queue_speed_px": event.get("queue_speed_px"),
                "queue_track_id": event.get("queue_track_id"),
                "queue_track_id_unique": event.get("queue_track_id_unique"),
                "queue_conf_at_event": event.get("queue_conf_at_event"),
                "queue_conf_last": event.get("queue_conf_last"),
                "queue_conf_max": event.get("queue_conf_max"),
                "queue_conf_avg": event.get("queue_conf_avg"),
                "queue_candidate_rank_event": event.get("queue_candidate_rank_event"),
                "queue_candidate_rank_last": event.get("queue_candidate_rank_last"),
                "queue_candidate_rank_min": event.get("queue_candidate_rank_min"),
                "queue_candidate_rank_max": event.get("queue_candidate_rank_max"),
                "queue_candidate_rank_avg": event.get("queue_candidate_rank_avg"),
                "queue_event_frame_gap": event.get("queue_event_frame_gap"),
                "queue_static_blocked_history": event.get("queue_static_blocked_history"),
                "bounce_signal_score": event.get("bounce_signal_score"),
                "dedupe_cluster_size": event.get("dedupe_cluster_size"),
                "publish_quality_override": event.get("publish_quality_override"),
                "slow_queue_shape_override": event.get("slow_queue_shape_override"),
                "near_side_low_delta_override": event.get("near_side_low_delta_override"),
                "stale_speed_context_override": event.get("stale_speed_context_override"),
                "late_yolo_recovery": event.get("late_yolo_recovery"),
                "late_yolo_recovery_regression_frames": event.get("late_yolo_recovery_regression_frames"),
                "gap_fill_reason": event.get("gap_fill_reason"),
                "gap_fill_from_frame": event.get("gap_fill_from_frame"),
                "gap_fill_frames": event.get("gap_fill_frames"),
            }
            # detect_single_camera_events() already applies the verified
            # HIT-first suppression and 25f/1.5m final-bounce cleanup. Do not
            # run the older realtime post-filter here or the dashboard/3D push
            # can diverge from the offline YOLO validation chain.
            accepted_bd = self._normalize_live_bounce_dict(
                bd,
                fallback_ts=now,
                fallback_speed_kmh=0,
            )
            accepted_bd["refiner_source"] = accepted_bd.get("refiner_source", "yolo_hit_first_final")
            if self._reject_yolo_live_bounce_speed_context_locked(accepted_bd, stats=stats):
                continue
            duplicate_action = self._replace_weaker_yolo_duplicate_bounce_locked(
                cam_name,
                accepted_bd,
                stats=stats,
                seen_frames=seen,
            )
            if duplicate_action == "skip":
                continue
            if duplicate_action != "replace" and not self._yolo_out_gate_allows_bounce_locked(cam_name, accepted_bd):
                self._stash_yolo_out_gate_pending_bounce_locked(cam_name, accepted_bd)
                stats["last_reject_reason"] = "out_rally_gate"
                continue
            if not self._record_live_bounce_locked(accepted_bd, debug_source=accepted_bd):
                continue
            seen.append(event_frame)
            self._yolo_fuzzy_last_emitted_frame[cam_name] = event_frame
            last_emitted_frame = event_frame
            stats["accepted"] = int(stats.get("accepted", 0)) + 1
            stats["last_candidate_frame"] = event_frame
            stats["last_reject_reason"] = ""
            emitted = accepted_bd

        for event in hit_events:
            event_frame = self._event_frame(event)
            if event_frame is None:
                continue
            hit_release_delay = int(getattr(self.config.hit_bounce_refiner, "release_delay_frames", 50) or 0)
            if hit_release_delay > 0 and latest_frame - event_frame + 1 < hit_release_delay:
                stats["pending_hit_release_delay_frames"] = hit_release_delay
                stats["pending_release_hits"] = int(stats.get("pending_release_hits", 0)) + 1
                continue
            if not self._can_emit_yolo_event(
                event_frame=event_frame,
                latest_frame=latest_frame,
                last_emitted_frame=last_hit_frame,
                seen_frames=seen_hits,
                cooldown_frames=0,
            ):
                continue
            now = time.time()
            event_capture_ts = self._event_capture_ts(buffered, event_frame, now)
            hit = {
                "frame": event_frame,
                "frame_index": event_frame,
                "x": event.get("x"),
                "y": event.get("y"),
                "pixel_x": event.get("pixel_x"),
                "pixel_y": event.get("pixel_y"),
                "camera": cam_name,
                "camera_name": cam_name,
                "type": "HIT",
                "kind": "hit",
                "confidence": event.get("confidence", 0.0),
                "timestamp": event_capture_ts,
                "capture_ts": event_capture_ts,
                "detect_delay": round(now - event_capture_ts, 2),
                "bounce_mode": f"mono_{cam_name}",
                "source": event.get("source", "yolo_fuzzy_player_hit"),
                "angle": event.get("angle"),
                "delta_v": event.get("delta_v"),
                "y_reversal": event.get("y_reversal"),
                "player_frame": event.get("player_frame"),
                "player_distance_px": event.get("player_distance_px"),
                "player_threshold_px": event.get("player_threshold_px"),
                "player_court_x": event.get("player_court_x"),
                "player_court_y": event.get("player_court_y"),
                "player_conf": event.get("player_conf"),
            }
            self._record_live_hit_locked(hit)
            seen_hits.append(event_frame)
            self._yolo_fuzzy_last_emitted_hit_frame[cam_name] = event_frame
            last_hit_frame = event_frame
            stats["accepted_hits"] = int(stats.get("accepted_hits", 0)) + 1
            stats["last_hit_frame"] = event_frame

        self._remember_yolo_hit_suppressed_result_bounces_locked(cam_name, result, stats)
        self._retract_yolo_live_bounces_shadowing_hit_suppressed_locked(cam_name, stats)

        return emitted

    @staticmethod
    def _event_float(event: dict, key: str) -> float | None:
        try:
            value = event.get(key)
            if value is None:
                return None
            value = float(value)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    @staticmethod
    def _is_yolo_queue_event(event: dict) -> bool:
        source = str(event.get("source", "") or "")
        return (
            source.startswith("yolo_")
            or event.get("queue_id") is not None
            or event.get("bounce_signal_score") is not None
        )

    def _reject_yolo_live_bounce_quality_locked(
        self,
        event: dict,
        *,
        event_frame: int,
        stats: dict,
    ) -> str | None:
        """Reject YOLO single-cam bounce events that are too weak to publish."""
        from app.pipeline.yolo_bounce_filter import dashboard_yolo_quality_reject_reason

        reason = dashboard_yolo_quality_reject_reason(
            event,
            event_frame=event_frame,
            min_bounce_frame=self._YOLO_LIVE_MIN_BOUNCE_FRAME,
            min_history=self._YOLO_LIVE_MIN_BOUNCE_HISTORY,
            weak_non_reversal_max_angle=self._YOLO_LIVE_WEAK_NON_REVERSAL_MAX_ANGLE,
            weak_non_reversal_min_score=self._YOLO_LIVE_WEAK_NON_REVERSAL_MIN_SCORE,
        )
        if reason:
            key = f"skipped_{reason}_bounces"
            stats[key] = int(stats.get(key, 0)) + 1
        return reason

    def _reject_yolo_live_bounce_speed_context_locked(
        self,
        event: dict,
        *,
        stats: dict,
    ) -> str | None:
        """Reject YOLO bounces whose attached speed context is clearly stale."""
        if not self._is_yolo_queue_event(event):
            return None
        try:
            speed_frame_gap = int(event.get("speed_frame_gap"))
        except (TypeError, ValueError):
            return None

        speed_source = str(event.get("speed_source", "") or "")
        raw_y = self._event_float(event, "raw_y")
        if raw_y is None:
            raw_y = self._event_float(event, "y")
        raw_x = self._event_float(event, "raw_x")
        if raw_x is None:
            raw_x = self._event_float(event, "x")
        near_baseline = (
            raw_y is not None
            and abs(raw_y) >= float(self._COURT_HALF_LENGTH_M) - float(
                self._YOLO_LIVE_BASELINE_STALE_SPEED_MARGIN_M
            )
        )
        speed_kmh = self._event_float(event, "speed_kmh")
        if speed_kmh is None:
            speed_kmh = self._event_float(event, "speed")

        reason = None
        if (
            near_baseline
            and abs(speed_frame_gap) > int(self._YOLO_LIVE_BASELINE_STALE_SPEED_FRAMES)
        ):
            reason = "quality_stale_baseline_speed"
        elif (
            bool(event.get("slow_queue_shape_override"))
            and abs(speed_frame_gap)
            > int(self._YOLO_LIVE_SLOW_QUEUE_STALE_SPEED_FRAMES)
        ):
            reason = "quality_stale_slow_queue_speed"
        elif (
            speed_kmh is not None
            and speed_kmh <= float(self._YOLO_LIVE_LOW_SPEED_STALE_CONTEXT_KMH)
            and abs(speed_frame_gap)
            > int(self._YOLO_LIVE_LOW_SPEED_STALE_CONTEXT_FRAMES)
        ):
            reason = "quality_stale_low_speed_context"
        elif (
            speed_source == "future_single_cam_speed_backfill"
            and speed_kmh is not None
            and speed_kmh < float(self._YOLO_LIVE_FUTURE_LOW_SPEED_KMH)
            and abs(speed_frame_gap)
            > int(self._YOLO_LIVE_FUTURE_LOW_SPEED_MAX_GAP_FRAMES)
        ):
            reason = "quality_distant_future_speed"
        elif (
            raw_y is not None
            and raw_x is not None
            and abs(raw_x) <= 1.5
            and raw_y >= 5.0
            and str(event.get("speed_direction", "") or "") == "top_down"
            and speed_source == "future_single_cam_speed_backfill"
            and abs(speed_frame_gap) > 60
        ):
            reason = "quality_distant_top_down_positive_future_speed"
        elif (
            raw_y is not None
            and raw_x is not None
            and abs(raw_x) <= 1.5
            and raw_y >= 5.0
            and str(event.get("speed_direction", "") or "") == "top_down"
            and speed_frame_gap >= 45
        ):
            reason = "quality_stale_top_down_positive_speed"

        if reason == "quality_stale_slow_queue_speed":
            try:
                confidence = float(event.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            try:
                score = float(event.get("bounce_signal_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            try:
                delta_v = float(event.get("delta_v", 0.0) or 0.0)
            except Exception:
                delta_v = 0.0
            try:
                queue_speed = float(event.get("queue_speed_px", 0.0) or 0.0)
            except Exception:
                queue_speed = 0.0
            try:
                unique_tracks = int(event.get("queue_track_id_unique") or 0)
            except Exception:
                unique_tracks = 0
            if (
                raw_y is not None
                and raw_y <= -3.0
                and confidence >= 0.25
                and score >= 180.0
                and delta_v >= 2.0
                and queue_speed <= 3.0
                and unique_tracks <= 1
            ):
                event["stale_speed_context_override"] = "near_side_slow_queue"
                stats["allowed_stale_slow_queue_speed_bounces"] = int(
                    stats.get("allowed_stale_slow_queue_speed_bounces", 0)
                ) + 1
                return None

        if reason == "quality_stale_baseline_speed":
            try:
                confidence = float(event.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            try:
                score = float(event.get("bounce_signal_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            try:
                delta_v = float(event.get("delta_v", 0.0) or 0.0)
            except Exception:
                delta_v = 0.0
            try:
                queue_speed = float(event.get("queue_speed_px", 0.0) or 0.0)
            except Exception:
                queue_speed = 0.0
            try:
                history = int(event.get("queue_history_len") or 0)
            except Exception:
                history = 0
            if (
                confidence >= 0.25
                and score >= 300.0
                and delta_v >= 10.0
                and queue_speed >= 8.0
                and history >= 20
            ):
                event["stale_speed_context_override"] = "baseline_high_quality"
                stats["allowed_stale_baseline_speed_bounces"] = int(
                    stats.get("allowed_stale_baseline_speed_bounces", 0)
                ) + 1
                return None

        if reason is None:
            return None
        stats[f"skipped_{reason}_bounces"] = int(
            stats.get(f"skipped_{reason}_bounces", 0)
        ) + 1
        stats["last_reject_reason"] = reason
        stats["last_reject_speed_frame_gap"] = speed_frame_gap
        stats["last_reject_speed_source"] = speed_source
        self._remember_yolo_bounce_reject_locked(stats, event, reason)
        return reason

    def _previous_live_bounce_frame_locked(
        self,
        cam_name: str,
        event_frame: int,
    ) -> int | None:
        previous_frame = None
        for prev in self._live_bounces:
            if not self._event_matches_camera(prev, cam_name):
                continue
            prev_frame = self._event_frame(prev)
            if prev_frame is None or prev_frame >= event_frame:
                continue
            if previous_frame is None or prev_frame > previous_frame:
                previous_frame = prev_frame
        return previous_frame

    def _allow_yolo_gap_fill_quality_bounce_locked(
        self,
        cam_name: str,
        event: dict,
        *,
        event_frame: int,
        quality_reject_reason: str,
        stats: dict,
    ) -> bool:
        """Allow a borderline YOLO bounce only when it fills a long live gap."""
        if quality_reject_reason not in {
            "quality_short_track",
            "quality_weak_non_reversal",
            "quality_low_delta_v",
        }:
            return False
        if not bool(event.get("in_court", True)):
            return False

        previous_frame = self._previous_live_bounce_frame_locked(cam_name, event_frame)
        if previous_frame is None:
            return False
        gap_frames = event_frame - previous_frame
        if gap_frames < int(self._YOLO_GAP_FILL_MIN_FRAMES):
            return False

        try:
            confidence = float(event.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < float(self._YOLO_GAP_FILL_MIN_CONFIDENCE):
            return False

        try:
            score = float(event.get("bounce_signal_score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        try:
            delta_v = float(event.get("delta_v", 0.0) or 0.0)
        except Exception:
            delta_v = 0.0
        try:
            history_len = int(float(event.get("queue_history_len", 0) or 0))
        except Exception:
            history_len = 0
        try:
            queue_speed = float(event.get("queue_speed_px", 0.0) or 0.0)
        except Exception:
            queue_speed = 0.0

        allowed = False
        if quality_reject_reason == "quality_short_track":
            allowed = history_len >= 5 and score >= float(
                self._YOLO_GAP_FILL_SHORT_TRACK_MIN_SCORE
            )
        elif quality_reject_reason == "quality_weak_non_reversal":
            allowed = (
                score >= float(self._YOLO_GAP_FILL_WEAK_MIN_SCORE)
                or delta_v >= float(self._YOLO_GAP_FILL_WEAK_MIN_DELTA_V)
            )
        elif quality_reject_reason == "quality_low_delta_v":
            allowed = (
                history_len >= 20
                and score >= 240.0
                and delta_v >= 0.8
                and queue_speed >= 8.0
            )
        if not allowed:
            return False

        event["publish_quality_override"] = quality_reject_reason
        event["gap_fill_from_frame"] = previous_frame
        event["gap_fill_frames"] = gap_frames
        event["gap_fill_reason"] = "long_bounce_gap"
        stats["gap_filled_quality_bounces"] = int(
            stats.get("gap_filled_quality_bounces", 0)
        ) + 1
        stats[f"gap_filled_{quality_reject_reason}_bounces"] = int(
            stats.get(f"gap_filled_{quality_reject_reason}_bounces", 0)
        ) + 1
        stats["last_gap_fill_frame"] = event_frame
        stats["last_gap_fill_previous_frame"] = previous_frame
        stats["last_gap_fill_frames"] = gap_frames
        return True

    def _event_capture_ts_from_event(self, event: dict, now: float) -> float:
        event_capture_ts = event.get("capture_ts", event.get("timestamp", now))
        try:
            return float(event_capture_ts)
        except Exception:
            return now

    def _record_yolo_speed_events_locked(
        self,
        cam_name: str,
        speed_events: list[dict],
        *,
        latest_frame: int,
        last_speed_frame: int | None,
        seen_speeds,
        stats: dict,
        buffered: list[dict] | None = None,
    ) -> int | None:
        """Publish YOLO single-camera speed events before related bounces.

        Some YOLO result batches contain both the net/speed event and the
        following bounce. Recording speed first lets bounce egress attach the
        speed in the same consumer tick instead of emitting 000.
        """
        for event in speed_events:
            event_frame = self._event_frame(event)
            if event_frame is None:
                continue
            if not self._can_emit_yolo_event(
                event_frame=event_frame,
                latest_frame=latest_frame,
                last_emitted_frame=last_speed_frame,
                seen_frames=seen_speeds,
                cooldown_frames=0,
            ):
                continue
            now = time.time()
            if buffered is not None:
                event_capture_ts = self._event_capture_ts(buffered, event_frame, now)
            else:
                event_capture_ts = self._event_capture_ts_from_event(event, now)
            speed_event = {
                "frame": event_frame,
                "frame_index": event_frame,
                "x": event.get("x"),
                "y": event.get("y"),
                "pixel_x": event.get("pixel_x"),
                "pixel_y": event.get("pixel_y"),
                "camera": cam_name,
                "camera_name": cam_name,
                "type": "SPEED",
                "kind": "speed",
                "speed_kmh": int(round(float(event.get("speed_kmh", 0) or 0))),
                "direction": event.get("direction"),
                "timestamp": event_capture_ts,
                "capture_ts": event_capture_ts,
                "detect_delay": round(now - event_capture_ts, 2),
                "bounce_mode": f"mono_{cam_name}",
                "source": event.get("source", "single_cam_speed_crossing"),
            }
            self._record_live_speed_event_locked(speed_event)
            seen_speeds.append(event_frame)
            self._yolo_fuzzy_last_emitted_speed_frame[cam_name] = event_frame
            last_speed_frame = event_frame
            stats["accepted_speed_events"] = int(stats.get("accepted_speed_events", 0)) + 1
            stats["last_speed_frame"] = event_frame
        return last_speed_frame

    def _attach_recent_single_cam_speed_locked(self, bd: dict) -> dict:
        """Attach the latest preceding single-camera speed to a bounce event."""
        try:
            current_speed = int(round(float(bd.get("speed_kmh", 0) or 0)))
        except Exception:
            current_speed = 0
        if current_speed > 0 or not self._live_speed_events:
            return bd

        event_camera = bd.get("camera_name", bd.get("camera"))
        if not event_camera:
            return bd
        event_frame = self._event_frame(bd)
        try:
            event_ts = float(bd.get("timestamp", bd.get("capture_ts", 0.0)) or 0.0)
        except (TypeError, ValueError):
            event_ts = 0.0

        max_frame_gap = 600
        max_age_seconds = 24.0
        fresh_frame_gap = 320
        fresh_age_seconds = 12.0
        best = None
        best_frame_gap = None
        best_age_seconds = None
        for speed_event in reversed(self._live_speed_events):
            speed_camera = speed_event.get("camera_name", speed_event.get("camera"))
            if speed_camera and str(speed_camera) != str(event_camera):
                continue

            speed_frame = self._event_frame(speed_event)
            if event_frame is not None and speed_frame is not None:
                frame_gap = event_frame - speed_frame
                if frame_gap < -max_frame_gap:
                    continue
                if frame_gap > max_frame_gap:
                    break
            else:
                frame_gap = None

            try:
                speed_ts = float(
                    speed_event.get("timestamp", speed_event.get("capture_ts", event_ts))
                )
            except (TypeError, ValueError):
                speed_ts = event_ts
            age = None
            if event_ts and speed_ts:
                age = event_ts - speed_ts
                if age < -max_age_seconds:
                    continue
                if age > max_age_seconds:
                    break

            try:
                speed_kmh = int(round(float(speed_event.get("speed_kmh", 0) or 0)))
            except Exception:
                speed_kmh = 0
            if speed_kmh <= 0:
                continue
            best = speed_event
            best_frame_gap = frame_gap
            best_age_seconds = age
            break

        if best is None:
            return bd

        bd["speed_kmh"] = int(round(float(best.get("speed_kmh", 0) or 0)))
        bd["speed_direction"] = best.get("direction")
        is_fresh = True
        if best_frame_gap is not None and best_frame_gap > fresh_frame_gap:
            is_fresh = False
        if best_age_seconds is not None and best_age_seconds > fresh_age_seconds:
            is_fresh = False
        if best_frame_gap is not None and best_frame_gap < 0:
            bd["speed_source"] = "future_single_cam_speed_backfill"
        else:
            bd["speed_source"] = (
                "nearest_single_cam_speed" if is_fresh else "stale_single_cam_speed"
            )
        bd["speed_frame"] = best.get("frame_index", best.get("frame"))
        if best_frame_gap is not None:
            bd["speed_frame_gap"] = int(best_frame_gap)
        if best_age_seconds is not None:
            bd["speed_age_s"] = round(float(best_age_seconds), 3)
        return bd

    def _drain_yolo_event_results_locked(self, cam_name: str) -> dict | None:
        result_queue = self._yolo_event_result_queues.get(cam_name)
        if result_queue is None:
            return None
        emitted = None
        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        while True:
            try:
                msg = result_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if not isinstance(msg, dict):
                continue
            task_id = int(msg.get("task_id") or 0)
            last_applied = int(self._yolo_event_last_applied_task_ids.get(cam_name, 0) or 0)
            if task_id and task_id <= last_applied:
                continue
            if task_id:
                self._yolo_event_last_applied_task_ids[cam_name] = task_id
            stats["worker_last_result_task_id"] = task_id
            stats["last_analysis_ms"] = float(msg.get("analysis_ms") or 0.0)
            if msg.get("error"):
                stats["worker_last_error"] = msg.get("error")
                continue
            result = msg.get("result")
            if not isinstance(result, dict):
                continue
            try:
                latest_frame = int(msg.get("latest_frame") or stats.get("last_frame") or 0)
            except Exception:
                latest_frame = int(stats.get("last_frame") or 0)
            emitted = self._apply_yolo_fuzzy_result_locked(
                cam_name,
                result,
                latest_frame=latest_frame,
                analysis_ms=stats["last_analysis_ms"],
            ) or emitted
        return emitted

    def _apply_yolo_fuzzy_result_locked(
        self,
        cam_name: str,
        result: dict,
        *,
        latest_frame: int,
        analysis_ms: float | None = None,
    ) -> dict | None:
        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        if analysis_ms is not None:
            stats["last_analysis_ms"] = round(float(analysis_ms), 2)
        bounce_events = result.get("bounces", [])
        hit_events = result.get("hits", [])
        speed_events = result.get("speed_events", [])
        gate_only_bounce_events = result.get("gate_only_bounces", []) or []
        self._set_yolo_bounce_debug_list_locked(
            stats,
            "last_raw_yolo_bounce_candidates",
            bounce_events,
            latest_frame=latest_frame,
        )
        stats["candidate_bounces"] = int(result.get("count", len(bounce_events)) or 0)
        stats["raw_bounce_candidates"] = int(result.get("raw_bounce_candidate_count", 0) or 0)
        stats["suppressed_bounces_by_hit_window"] = int(result.get("suppressed_bounces_by_hit_window", 0) or 0)
        stats["deduped_bounces_after_hit"] = int(result.get("deduped_bounces_after_hit", 0) or 0)
        stats["gate_only_bounces"] = int(result.get("gate_only_bounce_count", len(gate_only_bounce_events)) or 0)
        stats["out_rally_suppressed_bounces"] = int(result.get("out_rally_suppressed_bounce_count", 0) or 0)
        stats["candidate_hits"] = int(result.get("hit_count", len(hit_events)) or 0)
        stats["candidate_speed_events"] = int(result.get("speed_count", len(speed_events)) or 0)
        stats["auxiliary_fallback_bounce_count"] = int(result.get("auxiliary_fallback_bounce_count", 0) or 0)
        stats["auxiliary_fallback_speed_count"] = int(result.get("auxiliary_fallback_speed_count", 0) or 0)
        stats["auxiliary_fallback_speed_ignored_count"] = int(
            result.get("auxiliary_fallback_speed_ignored_count", 0) or 0
        )
        if result.get("queue_tracker_stats"):
            stats["queue_tracker_stats"] = dict(result.get("queue_tracker_stats") or {})
        self._remember_yolo_hit_suppressed_result_bounces_locked(cam_name, result, stats)
        self._retract_yolo_live_bounces_shadowing_hit_suppressed_locked(cam_name, stats)

        seen = self._yolo_fuzzy_emitted_frames.setdefault(cam_name, deque(maxlen=50))
        seen_hits = self._yolo_fuzzy_emitted_hit_frames.setdefault(cam_name, deque(maxlen=50))
        seen_speeds = self._yolo_fuzzy_emitted_speed_frames.setdefault(cam_name, deque(maxlen=50))
        last_emitted_frame = self._yolo_fuzzy_last_emitted_frame.get(cam_name)
        last_hit_frame = self._yolo_fuzzy_last_emitted_hit_frame.get(cam_name)
        last_speed_frame = self._yolo_fuzzy_last_emitted_speed_frame.get(cam_name)

        suppression_frames = self._yolo_fuzzy_hit_suppression_frames.setdefault(
            cam_name,
            deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
        )
        known_suppression_frames = set(int(f) for f in suppression_frames)
        for hit_frame in result.get("hit_suppression_frames", []) or []:
            try:
                hit_frame_int = int(hit_frame)
            except Exception:
                continue
            if hit_frame_int in known_suppression_frames:
                continue
            suppression_frames.append(hit_frame_int)
            known_suppression_frames.add(hit_frame_int)
        hit_suppress_frames = int(
            getattr(self.config.hit_bounce_refiner, "hit_suppression_frames", 3) or 0
        )
        stats["hit_suppression_frames"] = len(suppression_frames)
        release_delay = int(getattr(self.config.hit_bounce_refiner, "release_delay_frames", 50) or 0)
        from app.pipeline.yolo_bounce_filter import filter_dashboard_yolo_publishable_bounces
        yolo_bounce_events = [event for event in bounce_events if self._is_yolo_queue_event(event)]
        passthrough_bounce_events = [event for event in bounce_events if not self._is_yolo_queue_event(event)]
        publish_suppressed_events = []
        if yolo_bounce_events:
            publish_filter = filter_dashboard_yolo_publishable_bounces(
                yolo_bounce_events,
                hit_events=hit_events,
                latest_frame=latest_frame,
                hit_suppress_frames=hit_suppress_frames,
                clean_time_frames=int(getattr(self.config.hit_bounce_refiner, "clean_time_frames", 25) or 25),
                clean_space_meters=float(getattr(self.config.hit_bounce_refiner, "clean_space_meters", 1.5)),
                release_delay_frames=release_delay,
            )
            publish_suppressed_events = (
                publish_filter.get("suppressed_bounces")
                or publish_filter.get("suppressed")
                or []
            )
            bounce_events = passthrough_bounce_events + (publish_filter.get("bounces") or [])
            stats["publish_suppression_counts"] = publish_filter.get("suppression_counts", {})
            stats["publish_suppressed_bounces"] = len(publish_suppressed_events)
        self._set_yolo_bounce_debug_list_locked(
            stats,
            "last_publishable_yolo_bounces",
            bounce_events,
            latest_frame=latest_frame,
        )
        self._set_yolo_bounce_debug_list_locked(
            stats,
            "last_suppressed_yolo_bounces",
            publish_suppressed_events,
            latest_frame=latest_frame,
        )

        emitted = None
        last_speed_frame = self._record_yolo_speed_events_locked(
            cam_name,
            speed_events,
            latest_frame=latest_frame,
            last_speed_frame=last_speed_frame,
            seen_speeds=seen_speeds,
            stats=stats,
        )
        self._record_yolo_gate_only_bounces_locked(
            cam_name,
            gate_only_bounce_events,
            latest_frame=latest_frame,
            hit_events=hit_events,
            speed_events=speed_events,
            stats=stats,
        )
        for event in bounce_events:
            event_frame = self._event_frame(event)
            if event_frame is None:
                continue
            if self._reject_stale_yolo_bounce_frame_locked(cam_name, event_frame, stats, event):
                continue
            quality_reject_reason = self._reject_yolo_live_bounce_quality_locked(
                event,
                event_frame=event_frame,
                stats=stats,
            )
            if quality_reject_reason:
                if not self._allow_yolo_gap_fill_quality_bounce_locked(
                    cam_name,
                    event,
                    event_frame=event_frame,
                    quality_reject_reason=quality_reject_reason,
                    stats=stats,
                ):
                    stats["last_reject_reason"] = quality_reject_reason
                    self._remember_yolo_bounce_reject_locked(
                        stats,
                        event,
                        quality_reject_reason,
                    )
                    continue
            suppressing_hit_frame = None
            if hit_suppress_frames > 0:
                for hit_frame in list(suppression_frames):
                    if abs(event_frame - int(hit_frame)) <= hit_suppress_frames:
                        suppressing_hit_frame = int(hit_frame)
                        break
                if suppressing_hit_frame is None:
                    for hit in self._live_hits:
                        hit_frame = self._event_frame(hit)
                        if hit_frame is not None and abs(event_frame - hit_frame) <= hit_suppress_frames:
                            suppressing_hit_frame = hit_frame
                            break
            if suppressing_hit_frame is not None:
                if self._has_live_yolo_hit_frame_locked(cam_name, suppressing_hit_frame):
                    self._remember_yolo_hit_suppressed_bounce_locked(
                        cam_name,
                        event,
                        suppressing_hit_frame=suppressing_hit_frame,
                    )
                stats["skipped_persistent_hit_suppressed_bounces"] = int(
                    stats.get("skipped_persistent_hit_suppressed_bounces", 0)
                ) + 1
                stats["last_reject_reason"] = f"hit_window:{suppressing_hit_frame}"
                continue
            shadow_frame = self._yolo_hit_suppressed_duplicate_frame_locked(cam_name, event)
            if shadow_frame is not None:
                stats["skipped_hit_suppressed_duplicate_bounces"] = int(
                    stats.get("skipped_hit_suppressed_duplicate_bounces", 0)
                ) + 1
                stats["last_reject_reason"] = f"hit_window_shadow:{shadow_frame}"
                continue
            if release_delay > 0 and latest_frame - event_frame + 1 < release_delay:
                stats["pending_release_delay_frames"] = release_delay
                stats["pending_release_bounces"] = int(stats.get("pending_release_bounces", 0)) + 1
                continue
            if not self._can_emit_yolo_event(
                event_frame=event_frame,
                latest_frame=latest_frame,
                last_emitted_frame=last_emitted_frame,
                seen_frames=seen,
                cooldown_frames=0,
            ):
                continue
            self._prime_yolo_out_gate_restarts_locked(
                cam_name,
                hit_events=hit_events,
                speed_events=speed_events,
                candidate_frame=event_frame,
                latest_frame=latest_frame,
            )

            now = time.time()
            event_capture_ts = self._event_capture_ts_from_event(event, now)
            bd = {
                "frame": event_frame,
                "frame_index": event_frame,
                "x": event.get("x"),
                "y": event.get("y"),
                "z": 0.0,
                "pixel_x": event.get("pixel_x"),
                "pixel_y": event.get("pixel_y"),
                "camera": cam_name,
                "camera_name": cam_name,
                "type": event.get("type", "IN"),
                "in_court": bool(event.get("in_court", True)),
                "confidence": event.get("confidence", 0.0),
                "timestamp": event_capture_ts,
                "capture_ts": event_capture_ts,
                "detect_delay": round(now - event_capture_ts, 2),
                "bounce_mode": f"mono_{cam_name}",
                "source": event.get("source", "yolo_fuzzy_single_cam"),
                "angle": event.get("angle"),
                "delta_v": event.get("delta_v"),
                "y_reversal": event.get("y_reversal"),
                "queue_id": event.get("queue_id"),
                "queue_history_len": event.get("queue_history_len"),
                "queue_speed_px": event.get("queue_speed_px"),
                "queue_track_id": event.get("queue_track_id"),
                "queue_track_id_unique": event.get("queue_track_id_unique"),
                "queue_conf_at_event": event.get("queue_conf_at_event"),
                "queue_conf_last": event.get("queue_conf_last"),
                "queue_conf_max": event.get("queue_conf_max"),
                "queue_conf_avg": event.get("queue_conf_avg"),
                "queue_candidate_rank_event": event.get("queue_candidate_rank_event"),
                "queue_candidate_rank_last": event.get("queue_candidate_rank_last"),
                "queue_candidate_rank_min": event.get("queue_candidate_rank_min"),
                "queue_candidate_rank_max": event.get("queue_candidate_rank_max"),
                "queue_candidate_rank_avg": event.get("queue_candidate_rank_avg"),
                "queue_event_frame_gap": event.get("queue_event_frame_gap"),
                "queue_static_blocked_history": event.get("queue_static_blocked_history"),
                "bounce_signal_score": event.get("bounce_signal_score"),
                "dedupe_cluster_size": event.get("dedupe_cluster_size"),
                "publish_quality_override": event.get("publish_quality_override"),
                "slow_queue_shape_override": event.get("slow_queue_shape_override"),
                "near_side_low_delta_override": event.get("near_side_low_delta_override"),
                "stale_speed_context_override": event.get("stale_speed_context_override"),
                "late_yolo_recovery": event.get("late_yolo_recovery"),
                "late_yolo_recovery_regression_frames": event.get("late_yolo_recovery_regression_frames"),
                "gap_fill_reason": event.get("gap_fill_reason"),
                "gap_fill_from_frame": event.get("gap_fill_from_frame"),
                "gap_fill_frames": event.get("gap_fill_frames"),
            }
            accepted_bd = self._normalize_live_bounce_dict(
                bd,
                fallback_ts=now,
                fallback_speed_kmh=0,
            )
            accepted_bd["refiner_source"] = accepted_bd.get("refiner_source", "yolo_hit_first_final")
            if self._reject_yolo_live_bounce_speed_context_locked(accepted_bd, stats=stats):
                continue
            duplicate_action = self._replace_weaker_yolo_duplicate_bounce_locked(
                cam_name,
                accepted_bd,
                stats=stats,
                seen_frames=seen,
            )
            if duplicate_action == "skip":
                continue
            if duplicate_action != "replace" and not self._yolo_out_gate_allows_bounce_locked(cam_name, accepted_bd):
                self._stash_yolo_out_gate_pending_bounce_locked(cam_name, accepted_bd)
                stats["last_reject_reason"] = "out_rally_gate"
                continue
            if not self._record_live_bounce_locked(accepted_bd, debug_source=accepted_bd):
                continue
            seen.append(event_frame)
            self._yolo_fuzzy_last_emitted_frame[cam_name] = event_frame
            last_emitted_frame = event_frame
            stats["accepted"] = int(stats.get("accepted", 0)) + 1
            stats["last_candidate_frame"] = event_frame
            stats["last_reject_reason"] = ""
            emitted = accepted_bd

        for event in hit_events:
            event_frame = self._event_frame(event)
            if event_frame is None:
                continue
            hit_release_delay = int(getattr(self.config.hit_bounce_refiner, "release_delay_frames", 50) or 0)
            if hit_release_delay > 0 and latest_frame - event_frame + 1 < hit_release_delay:
                stats["pending_hit_release_delay_frames"] = hit_release_delay
                stats["pending_release_hits"] = int(stats.get("pending_release_hits", 0)) + 1
                continue
            if not self._can_emit_yolo_event(
                event_frame=event_frame,
                latest_frame=latest_frame,
                last_emitted_frame=last_hit_frame,
                seen_frames=seen_hits,
                cooldown_frames=0,
            ):
                continue
            now = time.time()
            event_capture_ts = self._event_capture_ts_from_event(event, now)
            hit = {
                "frame": event_frame,
                "frame_index": event_frame,
                "x": event.get("x"),
                "y": event.get("y"),
                "pixel_x": event.get("pixel_x"),
                "pixel_y": event.get("pixel_y"),
                "camera": cam_name,
                "camera_name": cam_name,
                "type": "HIT",
                "kind": "hit",
                "confidence": event.get("confidence", 0.0),
                "timestamp": event_capture_ts,
                "capture_ts": event_capture_ts,
                "detect_delay": round(now - event_capture_ts, 2),
                "bounce_mode": f"mono_{cam_name}",
                "source": event.get("source", "yolo_fuzzy_player_hit"),
                "angle": event.get("angle"),
                "delta_v": event.get("delta_v"),
                "y_reversal": event.get("y_reversal"),
                "player_frame": event.get("player_frame"),
                "player_distance_px": event.get("player_distance_px"),
                "player_threshold_px": event.get("player_threshold_px"),
                "player_court_x": event.get("player_court_x"),
                "player_court_y": event.get("player_court_y"),
                "player_conf": event.get("player_conf"),
            }
            self._record_live_hit_locked(hit)
            seen_hits.append(event_frame)
            self._yolo_fuzzy_last_emitted_hit_frame[cam_name] = event_frame
            last_hit_frame = event_frame
            stats["accepted_hits"] = int(stats.get("accepted_hits", 0)) + 1
            stats["last_hit_frame"] = event_frame

        self._remember_yolo_hit_suppressed_result_bounces_locked(cam_name, result, stats)
        self._retract_yolo_live_bounces_shadowing_hit_suppressed_locked(cam_name, stats)

        return emitted

    @staticmethod
    def _compact_yolo_live_stats(stats_by_cam: dict) -> dict:
        keep_keys = (
            "detector",
            "detections",
            "buffered",
            "candidate_bounces",
            "candidate_hits",
            "candidate_speed_events",
            "auxiliary_fallback_bounce_count",
            "auxiliary_fallback_speed_count",
            "auxiliary_fallback_speed_ignored_count",
            "accepted",
            "accepted_hits",
            "accepted_speed_events",
            "last_frame",
            "last_candidate_frame",
            "last_hit_frame",
            "last_speed_frame",
            "last_reject_reason",
            "recent_rejected_bounces",
            "last_raw_yolo_bounce_candidates",
            "last_raw_yolo_bounce_candidates_latest_frame",
            "last_publishable_yolo_bounces",
            "last_publishable_yolo_bounces_latest_frame",
            "last_suppressed_yolo_bounces",
            "last_suppressed_yolo_bounces_latest_frame",
            "recent_raw_yolo_bounce_candidates",
            "recent_publishable_yolo_bounces",
            "recent_suppressed_yolo_bounces",
            "player_pose_buffered",
            "analysis_stride",
            "analysis_calls",
            "skipped_analysis_stride",
            "last_analysis_ms",
            "last_analyzed_frame",
            "skipped_quality_warmup_bounces",
            "skipped_quality_short_track_bounces",
            "skipped_quality_weak_non_reversal_bounces",
            "skipped_quality_no_y_reversal_bounces",
            "skipped_quality_auxiliary_fallback_bounces",
            "skipped_quality_weak_out_bounces",
            "raw_bounce_candidates",
            "suppressed_bounces_by_hit_window",
            "deduped_bounces_after_hit",
            "gate_only_bounces",
            "pending_gate_only_bounces",
            "gate_only_out_gate_bounces",
            "out_rally_suppressed_bounces",
            "out_gate_suppressed_bounces",
            "out_gate_last_suppressed_frame",
            "out_gate_last_suppressed_interval",
            "out_gate_blocked_after_out_frame",
            "out_gate_last_out_frame",
            "out_gate_pending_bounces",
            "out_gate_stashed_pending_bounces",
            "out_gate_released_pending_bounces",
            "out_gate_dropped_pending_closed_interval",
            "out_gate_dropped_pending_hit_suppressed",
            "out_gate_dropped_pending_already_seen",
            "out_gate_dropped_pending_duplicate_live",
            "out_gate_last_dropped_pending_interval",
            "pending_release_hits",
            "pending_release_bounces",
            "worker_enabled",
            "worker_last_submitted_task_id",
            "worker_last_result_task_id",
            "worker_submit_skipped_busy",
            "worker_submit_replaced_busy",
            "worker_submit_dropped",
            "worker_last_error",
            "gap_filled_quality_bounces",
            "gap_filled_quality_short_track_bounces",
            "gap_filled_quality_weak_non_reversal_bounces",
            "last_gap_fill_frame",
            "last_gap_fill_previous_frame",
            "last_gap_fill_frames",
            "retro_suppressed_bounces_by_hit",
            "retro_suppressed_ws_bounces_by_hit",
            "last_retro_suppress_hit_frame",
            "last_retro_suppressed_bounce_frames",
        )
        compact: dict[str, dict] = {}
        for cam, stats in (stats_by_cam or {}).items():
            if not isinstance(stats, dict):
                continue
            item = {key: stats.get(key) for key in keep_keys if key in stats}
            queue_stats = stats.get("queue_tracker_stats")
            if isinstance(queue_stats, dict):
                item["queue_tracker_stats"] = {
                    key: queue_stats.get(key)
                    for key in (
                        "raw_ball_boxes",
                        "moving_boxes",
                        "static_boxes",
                        "trajectory_points",
                        "stitched_points",
                    )
                    if key in queue_stats
                }
            compact[cam] = item
        return compact

    @staticmethod
    def _compact_refiner_stats(stats: dict) -> dict:
        keep_keys = (
            "raw_bounce_candidate_count",
            "pending_bounce_count",
            "recent_hit_count",
            "final_bounce_count",
            "show_hits_on_minimap",
            "suppressed_bounces_by_hit",
            "deduped_bounces_after_hit",
        )
        return {key: stats.get(key) for key in keep_keys if key in stats}

    @staticmethod
    def _compact_event(event: dict) -> dict:
        keep_keys = (
            "frame",
            "frame_index",
            "x",
            "y",
            "z",
            "pixel_x",
            "pixel_y",
            "camera",
            "camera_name",
            "type",
            "kind",
            "in_court",
            "timestamp",
            "capture_ts",
            "time_ms",
            "speed_kmh",
            "speed",
            "speed_source",
            "speed_frame",
            "speed_frame_gap",
            "speed_age_s",
            "direction",
            "speed_direction",
            "legacy_direction",
            "sequence",
            "raw_x",
            "raw_y",
            "projected_x",
            "projected_y",
            "ws_x",
            "ws_y",
            "court_correction",
            "publish_quality_override",
            "gap_fill_reason",
            "gap_fill_from_frame",
            "gap_fill_frames",
            "event_kind",
            "protocol_version",
        )
        return {key: event.get(key) for key in keep_keys if key in event}

    @staticmethod
    def _event_has_display_speed(event: dict) -> bool:
        try:
            return int(round(float(event.get("speed_kmh", event.get("speed", 0)) or 0))) > 0
        except Exception:
            return False

    def _compact_ready_bounces_locked(self, event_limit: int) -> list[dict]:
        ready_bounces = [
            event for event in self._live_bounces if self._event_has_display_speed(event)
        ]
        return [self._compact_event(e) for e in ready_bounces[-event_limit:]]

    def _effective_total_live_bounces(self) -> int:
        return max(0, int(self._total_live_bounces) - int(self._total_retracted_live_bounces))

    def _build_compact_live_analytics_locked(self) -> dict:
        self._refresh_zero_speed_live_bounces_locked()
        event_limit = max(1, int(self._DASHBOARD_ANALYTICS_EVENT_LIMIT))
        speed_limit = max(1, int(self._DASHBOARD_ANALYTICS_SPEED_LIMIT))
        refiner_stats = self._hit_bounce_refiner.get_stats()
        return {
            "rally_state": self._rally_tracker.get_state().to_dict(),
            "recent_bounces": self._compact_ready_bounces_locked(event_limit),
            "total_bounces": self._effective_total_live_bounces(),
            "last_frame_speed_kmh": int(round(float(self._last_frame_speed_kmh or 0.0))),
            "latest_net_crossing": (
                self._compact_event(self._latest_net_crossing)
                if self._latest_net_crossing
                else None
            ),
            "recent_hits": [
                self._compact_event(e) for e in self._live_hits[-event_limit:]
            ],
            "total_hits": self._total_live_hits,
            "recent_speed_events": [
                self._compact_event(e) for e in self._live_speed_events[-speed_limit:]
            ],
            "total_speed_events": self._total_live_speed_events,
            "recent_event_limit": event_limit,
            "recent_speed_event_limit": speed_limit,
            "latest_single_cam_speed_event": (
                self._compact_event(self._live_speed_events[-1])
                if self._live_speed_events
                else None
            ),
            "single_cam_bounce_stats": self._compact_yolo_live_stats(
                self._yolo_fuzzy_live_stats
            ),
            "raw_bounce_candidate_count": refiner_stats.get(
                "raw_bounce_candidate_count", 0
            ),
            "suppressed_bounces_by_hit": refiner_stats.get(
                "suppressed_bounces_by_hit", 0
            ),
            "hit_bounce_refiner_stats": self._compact_refiner_stats(refiner_stats),
        }

    def get_live_analytics(self, *, compact: bool = False) -> dict:
        """Return current live bounce/rally state for the dashboard.

        Rally tracking is now done by the simple ``RallyTracker`` only —
        net crossings + timeout, no serve rules, no end-reason classification.
        RallyStateMachine was removed: the rule-based machine wasn't GT-
        validated, and its complex output (PENDING / SERVING / DOUBLE_FAULT /
        LET ...) was noisy on realtime data.
        """
        if compact:
            if not self._analytics_lock.acquire(blocking=False):
                if self._dashboard_analytics_cache:
                    return dict(self._dashboard_analytics_cache)
                return {
                    "rally_state": {},
                    "recent_bounces": [],
                    "total_bounces": self._effective_total_live_bounces(),
                    "recent_hits": [],
                    "total_hits": self._total_live_hits,
                    "recent_speed_events": [],
                    "total_speed_events": self._total_live_speed_events,
                    "recent_event_limit": self._DASHBOARD_ANALYTICS_EVENT_LIMIT,
                    "recent_speed_event_limit": self._DASHBOARD_ANALYTICS_SPEED_LIMIT,
                    "last_frame_speed_kmh": 0,
                    "latest_net_crossing": None,
                    "latest_single_cam_speed_event": None,
                    "single_cam_bounce_stats": {},
                    "raw_bounce_candidate_count": 0,
                    "suppressed_bounces_by_hit": 0,
                    "hit_bounce_refiner_stats": {},
                }
            try:
                payload = self._build_compact_live_analytics_locked()
                self._dashboard_analytics_cache = payload
                return payload
            finally:
                self._analytics_lock.release()
        with self._analytics_lock:
            self._refresh_zero_speed_live_bounces_locked()
            if compact:
                event_limit = max(1, int(self._DASHBOARD_ANALYTICS_EVENT_LIMIT))
                speed_limit = max(1, int(self._DASHBOARD_ANALYTICS_SPEED_LIMIT))
                refiner_stats = self._hit_bounce_refiner.get_stats()
                recent_bounces = self._compact_ready_bounces_locked(event_limit)
                recent_hits = [
                    self._compact_event(e) for e in self._live_hits[-event_limit:]
                ]
                recent_speed_events = [
                    self._compact_event(e) for e in self._live_speed_events[-speed_limit:]
                ]
                return {
                    "rally_state": self._rally_tracker.get_state().to_dict(),
                    "recent_bounces": recent_bounces,
                    "total_bounces": self._effective_total_live_bounces(),
                    "last_frame_speed_kmh": int(round(float(self._last_frame_speed_kmh or 0.0))),
                    "latest_net_crossing": (
                        self._compact_event(self._latest_net_crossing)
                        if self._latest_net_crossing
                        else None
                    ),
                    "recent_hits": recent_hits,
                    "total_hits": self._total_live_hits,
                    "recent_speed_events": recent_speed_events,
                    "total_speed_events": self._total_live_speed_events,
                    "recent_event_limit": event_limit,
                    "recent_speed_event_limit": speed_limit,
                    "latest_single_cam_speed_event": (
                        self._compact_event(self._live_speed_events[-1])
                        if self._live_speed_events
                        else None
                    ),
                    "single_cam_bounce_stats": self._compact_yolo_live_stats(
                        self._yolo_fuzzy_live_stats
                    ),
                    "raw_bounce_candidate_count": refiner_stats.get(
                        "raw_bounce_candidate_count", 0
                    ),
                    "suppressed_bounces_by_hit": refiner_stats.get(
                        "suppressed_bounces_by_hit", 0
                    ),
                    "hit_bounce_refiner_stats": self._compact_refiner_stats(refiner_stats),
                }
            event_limit = max(1, int(self._LIVE_ANALYTICS_EVENT_LIMIT))
            speed_limit = max(1, int(self._LIVE_ANALYTICS_SPEED_LIMIT))
            return {
                "rally_state": self._rally_tracker.get_state().to_dict(),
                "completed_rallies": self._rally_tracker.get_completed_rallies(),
                "recent_bounces": list(self._live_bounces[-event_limit:]),
                "total_bounces": self._effective_total_live_bounces(),
                "ws_pending_bounces": len(self._ws_bounce_queue),
                "last_frame_speed_kmh": int(round(float(self._last_frame_speed_kmh or 0.0))),
                "latest_net_crossing": dict(self._latest_net_crossing) if self._latest_net_crossing else None,
                "recent_hits": list(self._live_hits[-event_limit:]),
                "total_hits": self._total_live_hits,
                "recent_speed_events": list(self._live_speed_events[-speed_limit:]),
                "total_speed_events": self._total_live_speed_events,
                "recent_event_limit": event_limit,
                "recent_speed_event_limit": speed_limit,
                "latest_single_cam_speed_event": (
                    dict(self._live_speed_events[-1]) if self._live_speed_events else None
                ),
                "single_cam_bounce_stats": dict(self._yolo_fuzzy_live_stats),
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
        ts = bd.get("timestamp")
        try:
            ts_float = float(ts)
        except (TypeError, ValueError):
            ts_float = float(fallback_ts if fallback_ts is not None else time.time())
            bd["timestamp"] = ts_float
        bd["time_ms"] = int(round(ts_float * 1000))
        bd["speed"] = int(round(float(bd.get("speed_kmh", 0) or 0)))
        bd["event_kind"] = "bounce"
        bd["protocol_version"] = 1
        try:
            raw_x = float(bd.get("x"))
            raw_y = float(bd.get("y"))
        except (TypeError, ValueError):
            return bd
        is_single_cam = str(bd.get("bounce_mode", "")).startswith("mono_") or str(
            bd.get("source", "")
        ).startswith("yolo_")
        if is_single_cam:
            half_w = Orchestrator._COURT_HALF_WIDTH_M
            half_l = Orchestrator._COURT_HALF_LENGTH_M
            margin = Orchestrator._SINGLE_CAM_CLAMP_MARGIN_M
            if (
                -half_w - margin <= raw_x <= half_w + margin
                and -half_l - margin <= raw_y <= half_l + margin
                and not (-half_w <= raw_x <= half_w and -half_l <= raw_y <= half_l)
            ):
                bd["projected_x"] = round(raw_x, 4)
                bd["projected_y"] = round(raw_y, 4)
                raw_x = min(max(raw_x, -half_w), half_w)
                raw_y = min(max(raw_y, -half_l), half_l)
                bd["x"] = round(raw_x, 4)
                bd["y"] = round(raw_y, 4)
                bd["in_court"] = True
                bd["type"] = "IN"
                bd["court_correction"] = "clamped_single_cam_projection"
        bd["raw_x"] = round(raw_x, 4)
        bd["raw_y"] = round(raw_y, 4)
        bd["ws_x"] = round(raw_x * 10.0, 4)
        bd["ws_y"] = round(raw_y * 10.0, 4)
        return bd

    def _enqueue_ws_bounce_locked(self, bd: dict) -> None:
        """Queue a bounce for 3D push from the same live event source.

        Note: the remote 3D receiver expects decimeter-like court units
        (`x * 10`, `y * 10`). Minimap/API keep raw court coordinates in
        meters; only the WebSocket egress applies this protocol transform.
        """
        if not self._ws_enabled:
            return
        event = self._normalize_live_bounce_dict(bd)
        if event.get("raw_x") is None or event.get("raw_y") is None:
            return
        payload = {
            "event_kind": event["event_kind"],
            "protocol_version": event["protocol_version"],
            "x": event["ws_x"],
            "y": event["ws_y"],
            "ws_x": event["ws_x"],
            "ws_y": event["ws_y"],
            "raw_x": event["raw_x"],
            "raw_y": event["raw_y"],
            "projected_x": event.get("projected_x"),
            "projected_y": event.get("projected_y"),
            "speed": event["speed"],
            "speed_kmh": event["speed_kmh"],
            "speed_source": event.get("speed_source"),
            "speed_frame": event.get("speed_frame"),
            "speed_frame_gap": event.get("speed_frame_gap"),
            "speed_age_s": event.get("speed_age_s"),
            "timestamp": event["time_ms"],
            "timeStamp": event["time_ms"],
            "time_ms": event["time_ms"],
            "_queued_at": time.time(),
            "capture_ts": event.get("capture_ts"),
            "frame": event.get("frame"),
            "frame_index": event.get("frame_index"),
            "camera": event.get("camera"),
            "camera_name": event.get("camera_name"),
            "type": event.get("type"),
            "in_court": event.get("in_court"),
            "court_correction": event.get("court_correction"),
            "direction": event.get("direction"),
            "speed_direction": event.get("speed_direction"),
            "sequence": event.get("sequence"),
        }
        for key in (
            "source",
            "refiner_source",
            "bounce_mode",
            "confidence",
            "pixel_x",
            "pixel_y",
            "angle",
            "delta_v",
            "y_reversal",
            "queue_id",
            "queue_history_len",
            "queue_speed_px",
            "queue_track_id",
            "queue_track_id_unique",
            "queue_conf_at_event",
            "queue_conf_last",
            "queue_conf_max",
            "queue_conf_avg",
            "queue_candidate_rank_event",
            "queue_candidate_rank_last",
            "queue_candidate_rank_min",
            "queue_candidate_rank_max",
            "queue_candidate_rank_avg",
            "queue_event_frame_gap",
            "queue_static_blocked_history",
            "bounce_signal_score",
            "dedupe_cluster_size",
            "publish_quality_override",
            "slow_queue_shape_override",
            "near_side_low_delta_override",
            "stale_speed_context_override",
            "late_yolo_recovery",
            "late_yolo_recovery_regression_frames",
            "gap_fill_reason",
            "gap_fill_from_frame",
            "gap_fill_frames",
        ):
            if key in event:
                payload[key] = event.get(key)
        self._ws_bounce_queue.append(payload)

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

    def _yolo_out_gate_cam(self, event: dict) -> str | None:
        camera_name = event.get("camera_name", event.get("camera"))
        if camera_name is None:
            return None
        return str(camera_name)

    def _yolo_out_gate_state_for_cam(self, cam_name: str) -> dict[str, Any]:
        return self._yolo_out_gate_state.setdefault(cam_name, {})

    @staticmethod
    def _is_yolo_bottom_restart_hit(event: dict) -> bool:
        source = str(event.get("source", ""))
        return source.startswith("bottom_reversal") or source.startswith("bottom_up")

    def _record_yolo_out_gate_restart_locked(self, event: dict, *, kind: str) -> bool:
        cam_name = self._yolo_out_gate_cam(event)
        if not cam_name:
            return False
        state = self._yolo_out_gate_state_for_cam(cam_name)
        out_frame = self._event_frame({"frame_index": state.get("blocked_after_out_frame")})
        event_frame = self._event_frame(event)
        if out_frame is None or event_frame is None or event_frame <= out_frame:
            return False

        should_restart = False
        if kind == "speed":
            try:
                speed_kmh = float(event.get("speed_kmh", 0.0) or 0.0)
            except Exception:
                speed_kmh = 0.0
            should_restart = speed_kmh >= self._YOLO_OUT_RESTART_SPEED_KMH
        elif kind == "hit":
            should_restart = (
                event_frame - out_frame >= self._YOLO_OUT_RESTART_HIT_GAP_FRAMES
                and self._is_yolo_bottom_restart_hit(event)
            )

        if not should_restart:
            return False
        intervals = state.setdefault("blocked_intervals", [])
        intervals.append({
            "out_frame": out_frame,
            "restart_frame": event_frame,
            "restart_kind": kind,
            "restart_source": event.get("source"),
        })
        if len(intervals) > 20:
            del intervals[:-20]
        state.pop("blocked_after_out_frame", None)
        state["last_restart_frame"] = event_frame
        state["last_restart_kind"] = kind
        state["last_restart_source"] = event.get("source")
        return True

    def _record_yolo_out_gate_bounce_locked(self, event: dict) -> None:
        if bool(event.get("in_court", True)):
            return
        cam_name = self._yolo_out_gate_cam(event)
        if not cam_name:
            return
        event_frame = self._event_frame(event)
        if event_frame is None:
            return
        state = self._yolo_out_gate_state_for_cam(cam_name)
        state["blocked_after_out_frame"] = event_frame
        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        stats["out_gate_last_out_frame"] = event_frame
        self._prime_yolo_out_gate_restarts_from_recorded_events_locked(cam_name, event_frame)

    def _record_yolo_gate_only_bounces_locked(
        self,
        cam_name: str,
        events: list[dict],
        *,
        latest_frame: int,
        hit_events: list[dict],
        speed_events: list[dict],
        stats: dict,
    ) -> None:
        if not events:
            return
        release_delay = int(getattr(self.config.hit_bounce_refiner, "release_delay_frames", 50) or 0)
        state = self._yolo_out_gate_state_for_cam(cam_name)
        seen_gate_only = state.setdefault("gate_only_out_frames", set())
        if not isinstance(seen_gate_only, set):
            seen_gate_only = set(seen_gate_only or [])
            state["gate_only_out_frames"] = seen_gate_only

        for event in events:
            if bool(event.get("in_court", True)):
                continue
            event_frame = self._event_frame(event)
            if event_frame is None:
                continue
            if release_delay > 0 and latest_frame - event_frame + 1 < release_delay:
                stats["pending_gate_only_bounces"] = int(
                    stats.get("pending_gate_only_bounces", 0)
                ) + 1
                continue
            if event_frame in seen_gate_only:
                continue

            self._prime_yolo_out_gate_restarts_locked(
                cam_name,
                hit_events=hit_events,
                speed_events=speed_events,
                candidate_frame=event_frame,
                latest_frame=latest_frame,
            )
            gate_event = {
                **event,
                "frame": event_frame,
                "frame_index": event_frame,
                "camera": event.get("camera", cam_name),
                "camera_name": event.get("camera_name", cam_name),
                "type": event.get("type", "OUT"),
                "in_court": False,
                "gate_only": True,
            }
            self._record_yolo_out_gate_bounce_locked(gate_event)
            seen_gate_only.add(event_frame)
            if len(seen_gate_only) > self._LIVE_BOUNCE_HISTORY_LIMIT:
                keep = set(sorted(seen_gate_only)[-self._LIVE_BOUNCE_HISTORY_LIMIT:])
                state["gate_only_out_frames"] = keep
                seen_gate_only = keep
            stats["gate_only_out_gate_bounces"] = int(
                stats.get("gate_only_out_gate_bounces", 0)
            ) + 1

    def _yolo_out_gate_allows_bounce_locked(self, cam_name: str, event: dict) -> bool:
        state = self._yolo_out_gate_state_for_cam(cam_name)
        out_frame = self._event_frame({"frame_index": state.get("blocked_after_out_frame")})
        event_frame = self._event_frame(event)
        if event_frame is None:
            return True

        for interval in state.get("blocked_intervals", []) or []:
            start = self._event_frame({"frame_index": interval.get("out_frame")})
            end = self._event_frame({"frame_index": interval.get("restart_frame")})
            if start is None or end is None:
                continue
            if start < event_frame < end:
                stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
                stats["out_gate_suppressed_bounces"] = int(
                    stats.get("out_gate_suppressed_bounces", 0)
                ) + 1
                stats["out_gate_last_suppressed_frame"] = event_frame
                stats["out_gate_last_suppressed_interval"] = [start, end]
                return False

        if out_frame is None or event_frame <= out_frame:
            return True

        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        stats["out_gate_suppressed_bounces"] = int(
            stats.get("out_gate_suppressed_bounces", 0)
        ) + 1
        stats["out_gate_last_suppressed_frame"] = event_frame
        stats["out_gate_blocked_after_out_frame"] = out_frame
        return False

    def _yolo_out_gate_closed_interval_for_frame_locked(
        self,
        cam_name: str,
        event_frame: int,
    ) -> list[int] | None:
        state = self._yolo_out_gate_state_for_cam(cam_name)
        for interval in state.get("blocked_intervals", []) or []:
            start = self._event_frame({"frame_index": interval.get("out_frame")})
            end = self._event_frame({"frame_index": interval.get("restart_frame")})
            if start is None or end is None:
                continue
            if start < event_frame < end:
                return [start, end]
        return None

    def _yolo_live_hit_suppresses_frame_locked(
        self,
        cam_name: str,
        event_frame: int,
    ) -> int | None:
        suppress_frames = int(
            getattr(self.config.hit_bounce_refiner, "hit_suppression_frames", 3) or 0
        )
        if suppress_frames <= 0:
            return None
        for hit in self._live_hits:
            if not self._event_matches_camera(hit, cam_name):
                continue
            hit_frame = self._event_frame(hit)
            if hit_frame is not None and abs(event_frame - hit_frame) <= suppress_frames:
                return hit_frame
        return None

    def _is_duplicate_yolo_live_bounce_locked(self, cam_name: str, event: dict) -> bool:
        event_frame = self._event_frame(event)
        if event_frame is None:
            return False
        clean_time = int(getattr(self.config.hit_bounce_refiner, "clean_time_frames", 25) or 0)
        clean_space = float(getattr(self.config.hit_bounce_refiner, "clean_space_meters", 1.5) or 0.0)
        if clean_time <= 0 or clean_space <= 0:
            return False
        for prev in self._live_bounces:
            if not self._event_matches_camera(prev, cam_name):
                continue
            prev_frame = self._event_frame(prev)
            if prev_frame is None or abs(event_frame - prev_frame) > clean_time:
                continue
            try:
                dist = float(np.hypot(float(event.get("x")) - float(prev.get("x")), float(event.get("y")) - float(prev.get("y"))))
            except Exception:
                continue
            if dist <= clean_space:
                return True
        return False

    def _remember_yolo_hit_suppressed_bounce_locked(
        self,
        cam_name: str,
        event: dict,
        *,
        suppressing_hit_frame: int,
    ) -> None:
        event_frame = self._event_frame(event)
        if event_frame is None:
            return
        suppressed = self._yolo_fuzzy_hit_suppressed_bounces.setdefault(
            cam_name,
            deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
        )
        for prev in reversed(suppressed):
            if self._event_frame(prev) != event_frame:
                continue
            if self._event_frame({"frame_index": prev.get("suppressed_by_hit_frame")}) != int(suppressing_hit_frame):
                continue
            return
        item = {
            **event,
            "frame": event_frame,
            "frame_index": event_frame,
            "camera": event.get("camera", cam_name),
            "camera_name": event.get("camera_name", cam_name),
            "suppressed_by_hit_frame": int(suppressing_hit_frame),
        }
        suppressed.append(item)

    def _has_live_yolo_hit_frame_locked(self, cam_name: str, hit_frame: int) -> bool:
        for hit in reversed(self._live_hits):
            if not self._event_matches_camera(hit, cam_name):
                continue
            if self._event_frame(hit) == int(hit_frame):
                return True
        return False

    def _remember_yolo_hit_suppressed_result_bounces_locked(
        self,
        cam_name: str,
        result: dict,
        stats: dict,
    ) -> None:
        remembered = 0
        for event in result.get("suppressed_bounces", []) or []:
            if not isinstance(event, dict):
                continue
            reason = str(event.get("suppression_reason", "") or "")
            if reason != "hit_window" and event.get("suppressed_by_hit_frame") is None:
                continue
            hit_frame = self._event_frame({"frame_index": event.get("suppressed_by_hit_frame")})
            if hit_frame is None:
                continue
            if not self._has_live_yolo_hit_frame_locked(cam_name, hit_frame):
                continue
            self._remember_yolo_hit_suppressed_bounce_locked(
                cam_name,
                event,
                suppressing_hit_frame=hit_frame,
            )
            remembered += 1
        if remembered:
            stats["remembered_hit_suppressed_bounces"] = int(
                stats.get("remembered_hit_suppressed_bounces", 0)
            ) + remembered

    def _retract_yolo_live_bounces_shadowing_hit_suppressed_locked(
        self,
        cam_name: str,
        stats: dict,
    ) -> list[dict]:
        if not self._live_bounces:
            return []
        kept: list[dict] = []
        retracted: list[dict] = []
        retracted_shadow_frames: list[int] = []
        for bd in self._live_bounces:
            if not self._event_matches_camera(bd, cam_name):
                kept.append(bd)
                continue
            shadow_frame = self._yolo_hit_suppressed_duplicate_frame_locked(cam_name, bd)
            if shadow_frame is None:
                kept.append(bd)
                continue
            retracted.append(bd)
            retracted_shadow_frames.append(shadow_frame)

        if not retracted:
            return []

        self._live_bounces = kept
        self._total_retracted_live_bounces += len(retracted)
        self._drop_ws_payloads_for_bounces_locked(retracted)
        seen = self._yolo_fuzzy_emitted_frames.setdefault(cam_name, deque(maxlen=50))
        state = self._yolo_out_gate_state_for_cam(cam_name)
        for bd in retracted:
            self._clear_rally_buffer_bounce_locked(bd)
            frame = self._event_frame(bd)
            if frame is None:
                continue
            try:
                seen.remove(frame)
            except ValueError:
                pass
            if self._event_frame({"frame_index": state.get("blocked_after_out_frame")}) == frame:
                state.pop("blocked_after_out_frame", None)

        stats["retro_suppressed_hit_shadow_live_bounces"] = int(
            stats.get("retro_suppressed_hit_shadow_live_bounces", 0)
        ) + len(retracted)
        stats["last_retro_hit_shadow_live_bounce_frames"] = [
            self._event_frame(bd) for bd in retracted
        ]
        stats["last_retro_hit_shadow_seed_frames"] = retracted_shadow_frames
        history = stats.setdefault("retro_hit_shadow_history", [])
        if isinstance(history, list):
            history.append({
                "bounce_frames": [self._event_frame(bd) for bd in retracted],
                "seed_frames": list(retracted_shadow_frames),
            })
            del history[:-20]
        return retracted

    def _yolo_hit_suppressed_duplicate_frame_locked(
        self,
        cam_name: str,
        event: dict,
    ) -> int | None:
        """Return the earlier HIT-suppressed bounce frame this candidate shadows."""
        event_frame = self._event_frame(event)
        if event_frame is None:
            return None
        x = self._event_float(event, "x")
        y = self._event_float(event, "y")
        if x is None or y is None:
            return None
        shadow_time = int(getattr(self.config.hit_bounce_refiner, "hit_suppression_frames", 3) or 0)
        clean_space = float(getattr(self.config.hit_bounce_refiner, "clean_space_meters", 1.5) or 0.0)
        duplicate_space = max(clean_space, self._YOLO_LIVE_DUPLICATE_SPACE_METERS)
        if shadow_time <= 0 or duplicate_space <= 0:
            return None

        for prev in reversed(
            self._yolo_fuzzy_hit_suppressed_bounces.setdefault(
                cam_name,
                deque(maxlen=self._LIVE_BOUNCE_HISTORY_LIMIT),
            )
        ):
            if not self._event_matches_camera(prev, cam_name):
                continue
            prev_frame = self._event_frame(prev)
            if prev_frame is None or abs(event_frame - prev_frame) > shadow_time:
                continue
            prev_x = self._event_float(prev, "x")
            prev_y = self._event_float(prev, "y")
            if prev_x is None or prev_y is None:
                continue
            if float(np.hypot(x - prev_x, y - prev_y)) <= duplicate_space:
                return prev_frame
        return None

    @staticmethod
    def _yolo_bounce_signal_score(event: dict) -> float:
        try:
            score = event.get("bounce_signal_score")
            if score is not None:
                return float(score)
        except Exception:
            pass
        angle = Orchestrator._event_float(event, "angle") or 0.0
        delta_v = Orchestrator._event_float(event, "delta_v") or 0.0
        confidence = Orchestrator._event_float(event, "confidence") or 0.0
        speed_px = Orchestrator._event_float(event, "queue_speed_px") or 0.0
        speed_bonus = min(max(0.0, speed_px), 8.0) * 10.0
        y_bonus = 25.0 if event.get("y_reversal") else 0.0
        return (
            max(0.0, angle)
            + max(0.0, delta_v) * 2.0
            + max(0.0, confidence) * 10.0
            + y_bonus
            + speed_bonus
        )

    def _replace_weaker_yolo_duplicate_bounce_locked(
        self,
        cam_name: str,
        event: dict,
        *,
        stats: dict,
        seen_frames: deque[int],
    ) -> str:
        """Return ``none``, ``skip`` or ``replace`` for a YOLO live duplicate."""
        if not self._is_yolo_queue_event(event):
            return "none"
        event_frame = self._event_frame(event)
        if event_frame is None:
            return "none"
        clean_time = int(getattr(self.config.hit_bounce_refiner, "clean_time_frames", 25) or 0)
        clean_space = float(getattr(self.config.hit_bounce_refiner, "clean_space_meters", 1.5) or 0.0)
        duplicate_space = max(clean_space, self._YOLO_LIVE_DUPLICATE_SPACE_METERS)
        if clean_time <= 0 or duplicate_space <= 0:
            return "none"

        best_idx = None
        best_prev = None
        best_dist = None
        for idx, prev in enumerate(self._live_bounces):
            if not self._event_matches_camera(prev, cam_name):
                continue
            prev_frame = self._event_frame(prev)
            if prev_frame is None or abs(event_frame - prev_frame) > clean_time:
                continue
            try:
                dist = float(
                    np.hypot(
                        float(event.get("x")) - float(prev.get("x")),
                        float(event.get("y")) - float(prev.get("y")),
                    )
                )
            except Exception:
                continue
            if dist > duplicate_space:
                continue
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_prev = prev
                best_dist = dist

        if best_idx is None or best_prev is None:
            return "none"

        prev_frame = self._event_frame(best_prev)
        suppressing_hit_frame = (
            self._yolo_live_hit_suppresses_frame_locked(cam_name, prev_frame)
            if prev_frame is not None
            else None
        )
        if suppressing_hit_frame is not None:
            previous = self._live_bounces.pop(best_idx)
            self._total_retracted_live_bounces += 1
            self._drop_ws_payloads_for_bounces_locked([previous])
            self._clear_rally_buffer_bounce_locked(previous)
            self._remember_yolo_hit_suppressed_bounce_locked(
                cam_name,
                previous,
                suppressing_hit_frame=suppressing_hit_frame,
            )
            if prev_frame is not None:
                try:
                    seen_frames.remove(prev_frame)
                except ValueError:
                    pass
                state = self._yolo_out_gate_state_for_cam(cam_name)
                if self._event_frame({"frame_index": state.get("blocked_after_out_frame")}) == prev_frame:
                    state.pop("blocked_after_out_frame", None)
            stats["retro_suppressed_duplicate_live_bounces_by_hit"] = int(
                stats.get("retro_suppressed_duplicate_live_bounces_by_hit", 0)
            ) + 1
            stats["last_retro_duplicate_hit_frame"] = suppressing_hit_frame
            stats["last_retro_duplicate_bounce_frame"] = prev_frame
            stats["last_duplicate_live_frame"] = event_frame
            stats["last_duplicate_live_kept_frame"] = None
            return "skip"

        new_score = self._yolo_bounce_signal_score(event)
        prev_score = self._yolo_bounce_signal_score(best_prev)
        if new_score <= prev_score:
            stats["skipped_duplicate_live_bounces"] = int(
                stats.get("skipped_duplicate_live_bounces", 0)
            ) + 1
            stats["last_duplicate_live_frame"] = event_frame
            stats["last_duplicate_live_kept_frame"] = self._event_frame(best_prev)
            return "skip"

        previous_still_queued = any(
            self._ws_payload_matches_bounce(payload, best_prev)
            for payload in self._ws_bounce_queue
        )
        if self._ws_enabled and not previous_still_queued:
            stats["skipped_late_duplicate_live_bounces"] = int(
                stats.get("skipped_late_duplicate_live_bounces", 0)
            ) + 1
            stats["last_late_duplicate_live_frame"] = event_frame
            stats["last_late_duplicate_live_kept_frame"] = self._event_frame(best_prev)
            stats["last_late_duplicate_live_scores"] = {
                "old": round(float(prev_score), 3),
                "new": round(float(new_score), 3),
            }
            return "skip"

        previous = self._live_bounces.pop(best_idx)
        self._total_retracted_live_bounces += 1
        self._drop_ws_payloads_for_bounces_locked([previous])
        self._clear_rally_buffer_bounce_locked(previous)
        prev_frame = self._event_frame(previous)
        if prev_frame is not None:
            try:
                seen_frames.remove(prev_frame)
            except ValueError:
                pass
            state = self._yolo_out_gate_state_for_cam(cam_name)
            if self._event_frame({"frame_index": state.get("blocked_after_out_frame")}) == prev_frame:
                state.pop("blocked_after_out_frame", None)

        stats["replaced_duplicate_live_bounces"] = int(
            stats.get("replaced_duplicate_live_bounces", 0)
        ) + 1
        stats["last_replaced_duplicate_live_frame"] = prev_frame
        stats["last_replaced_by_frame"] = event_frame
        stats["last_replaced_duplicate_scores"] = {
            "old": round(float(prev_score), 3),
            "new": round(float(new_score), 3),
        }
        return "replace"

    def _stash_yolo_out_gate_pending_bounce_locked(self, cam_name: str, bd: dict) -> None:
        event_frame = self._event_frame(bd)
        if event_frame is None:
            return
        pending = self._yolo_out_gate_pending_bounces.setdefault(cam_name, {})
        is_new = event_frame not in pending
        pending[event_frame] = dict(bd)
        while len(pending) > self._YOLO_OUT_GATE_PENDING_LIMIT:
            oldest = min(pending)
            pending.pop(oldest, None)
        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        if is_new:
            stats["out_gate_stashed_pending_bounces"] = int(
                stats.get("out_gate_stashed_pending_bounces", 0)
            ) + 1
        stats["out_gate_pending_bounces"] = len(pending)

    def _release_yolo_out_gate_pending_bounces_locked(self, cam_name: str) -> list[dict]:
        pending = self._yolo_out_gate_pending_bounces.setdefault(cam_name, {})
        if not pending:
            return []

        stats = self._yolo_fuzzy_live_stats.setdefault(cam_name, {})
        seen = self._yolo_fuzzy_emitted_frames.setdefault(cam_name, deque(maxlen=50))
        released: list[dict] = []

        for event_frame in sorted(list(pending)):
            bd = pending.get(event_frame)
            if not bd:
                continue
            if self._reject_stale_yolo_bounce_frame_locked(cam_name, event_frame, stats, bd):
                pending.pop(event_frame, None)
                stats["out_gate_dropped_pending_stale_frame"] = int(
                    stats.get("out_gate_dropped_pending_stale_frame", 0)
                ) + 1
                continue

            closed_interval = self._yolo_out_gate_closed_interval_for_frame_locked(cam_name, event_frame)
            if closed_interval is not None:
                pending.pop(event_frame, None)
                stats["out_gate_dropped_pending_closed_interval"] = int(
                    stats.get("out_gate_dropped_pending_closed_interval", 0)
                ) + 1
                stats["out_gate_last_dropped_pending_interval"] = closed_interval
                continue

            suppressing_hit_frame = self._yolo_live_hit_suppresses_frame_locked(cam_name, event_frame)
            if suppressing_hit_frame is not None:
                pending.pop(event_frame, None)
                stats["out_gate_dropped_pending_hit_suppressed"] = int(
                    stats.get("out_gate_dropped_pending_hit_suppressed", 0)
                ) + 1
                stats["last_reject_reason"] = f"pending_hit_window:{suppressing_hit_frame}"
                continue

            if not self._yolo_out_gate_allows_bounce_locked(cam_name, bd):
                continue

            if event_frame in set(int(f) for f in seen):
                pending.pop(event_frame, None)
                stats["out_gate_dropped_pending_already_seen"] = int(
                    stats.get("out_gate_dropped_pending_already_seen", 0)
                ) + 1
                continue

            if self._is_duplicate_yolo_live_bounce_locked(cam_name, bd):
                pending.pop(event_frame, None)
                stats["out_gate_dropped_pending_duplicate_live"] = int(
                    stats.get("out_gate_dropped_pending_duplicate_live", 0)
                ) + 1
                continue

            accepted_bd = dict(bd)
            if not self._record_live_bounce_locked(accepted_bd, debug_source=accepted_bd):
                continue
            seen.append(event_frame)
            last_emitted = self._yolo_fuzzy_last_emitted_frame.get(cam_name)
            if last_emitted is None or event_frame > int(last_emitted):
                self._yolo_fuzzy_last_emitted_frame[cam_name] = event_frame
            stats["accepted"] = int(stats.get("accepted", 0)) + 1
            stats["out_gate_released_pending_bounces"] = int(
                stats.get("out_gate_released_pending_bounces", 0)
            ) + 1
            stats["last_candidate_frame"] = event_frame
            stats["last_reject_reason"] = ""
            pending.pop(event_frame, None)
            released.append(accepted_bd)

        stats["out_gate_pending_bounces"] = len(pending)
        return released

    def _prime_yolo_out_gate_restarts_locked(
        self,
        cam_name: str,
        *,
        hit_events: list[dict],
        speed_events: list[dict],
        candidate_frame: int,
        latest_frame: int,
    ) -> None:
        for event in speed_events:
            event_frame = self._event_frame(event)
            if event_frame is None or event_frame > candidate_frame:
                continue
            self._record_yolo_out_gate_restart_locked(
                {
                    **event,
                    "camera": event.get("camera", cam_name),
                    "camera_name": event.get("camera_name", cam_name),
                },
                kind="speed",
            )

        for event in hit_events:
            event_frame = self._event_frame(event)
            if event_frame is None or event_frame > candidate_frame:
                continue
            self._record_yolo_out_gate_restart_locked(
                {
                    **event,
                    "camera": event.get("camera", cam_name),
                    "camera_name": event.get("camera_name", cam_name),
                },
                kind="hit",
            )

    def _prime_yolo_out_gate_restarts_from_recorded_events_locked(
        self,
        cam_name: str,
        out_frame: int,
    ) -> bool:
        restart_events: list[tuple[int, str, dict]] = []
        for event in self._live_speed_events:
            if not self._event_matches_camera(event, cam_name):
                continue
            event_frame = self._event_frame(event)
            if event_frame is not None and event_frame > out_frame:
                restart_events.append((event_frame, "speed", event))
        for event in self._live_hits:
            if not self._event_matches_camera(event, cam_name):
                continue
            event_frame = self._event_frame(event)
            if event_frame is not None and event_frame > out_frame:
                restart_events.append((event_frame, "hit", event))

        restarted = False
        for _event_frame, kind, event in sorted(restart_events, key=lambda item: item[0]):
            if self._record_yolo_out_gate_restart_locked(event, kind=kind):
                restarted = True
                self._release_yolo_out_gate_pending_bounces_locked(cam_name)
                break
        return restarted

    def _record_live_hit_locked(self, hit: dict) -> None:
        """Publish one HIT to realtime analytics/debug/report buffers only."""
        self._dashboard_analytics_cache = {}
        clean_time = int(getattr(self.config.hit_bounce_refiner, "clean_time_frames", 25) or 0)
        clean_space = max(float(getattr(self.config.hit_bounce_refiner, "clean_space_meters", 1.5) or 0.0), 1.8)
        hit_frame = self._event_frame(hit)
        camera_name = hit.get("camera_name", hit.get("camera"))
        self._retract_live_bounces_suppressed_by_hit_locked(hit)
        replacement_indexes: list[int] = []
        replacement_sequence = None
        if hit_frame is not None and clean_time > 0 and clean_space > 0:
            for idx, prev in enumerate(self._live_hits):
                prev_frame = self._event_frame(prev)
                if prev_frame is None or abs(hit_frame - prev_frame) > clean_time:
                    continue
                try:
                    dist = float(np.hypot(float(hit.get("x")) - float(prev.get("x")), float(hit.get("y")) - float(prev.get("y"))))
                except Exception:
                    continue
                if dist <= clean_space:
                    replacement_indexes.append(idx)
                    prev_seq = prev.get("sequence")
                    if prev_seq is not None:
                        replacement_sequence = int(prev_seq) if replacement_sequence is None else min(replacement_sequence, int(prev_seq))

        for idx in reversed(replacement_indexes):
            self._live_hits.pop(idx)
        if replacement_sequence is None:
            self._total_live_hits += 1
            hit["sequence"] = self._total_live_hits
        else:
            hit["sequence"] = replacement_sequence
        self._live_hits.append(dict(hit))
        self._live_hits.sort(key=lambda item: int(item.get("frame_index", item.get("frame", 0)) or 0))
        if len(self._live_hits) > self._LIVE_BOUNCE_HISTORY_LIMIT:
            self._live_hits = self._live_hits[-self._LIVE_BOUNCE_HISTORY_LIMIT:]
        self._debug_record_hit(hit)
        if self._record_yolo_out_gate_restart_locked(hit, kind="hit"):
            self._release_yolo_out_gate_pending_bounces_locked(str(camera_name))

    def _event_matches_camera(self, event: dict, camera_name: str | None) -> bool:
        if not camera_name:
            return True
        event_camera = event.get("camera_name", event.get("camera"))
        return not event_camera or str(event_camera) == str(camera_name)

    @staticmethod
    def _ws_payload_matches_bounce(payload: dict, bd: dict) -> bool:
        try:
            raw_x = round(float(bd.get("x")), 4)
            raw_y = round(float(bd.get("y")), 4)
            payload_x = round(float(payload.get("raw_x")), 4)
            payload_y = round(float(payload.get("raw_y")), 4)
        except Exception:
            return False
        if raw_x != payload_x or raw_y != payload_y:
            return False
        ts = bd.get("timestamp")
        if ts is None:
            return True
        try:
            return int(round(float(ts) * 1000)) == int(payload.get("timestamp"))
        except Exception:
            return True

    def _drop_ws_payloads_for_bounces_locked(self, bounces: list[dict]) -> int:
        if not bounces or not self._ws_bounce_queue:
            return 0
        kept = deque(maxlen=self._ws_bounce_queue.maxlen)
        dropped = 0
        for payload in self._ws_bounce_queue:
            if any(self._ws_payload_matches_bounce(payload, bd) for bd in bounces):
                dropped += 1
                continue
            kept.append(payload)
        self._ws_bounce_queue = kept
        return dropped

    def _retract_live_bounces_locked(
        self,
        bounces: list[dict],
        *,
        stats: dict | None = None,
        stat_key: str | None = None,
    ) -> int:
        if not bounces or not self._live_bounces:
            return 0
        drop_ids = {id(bd) for bd in bounces}
        kept = [bd for bd in self._live_bounces if id(bd) not in drop_ids]
        dropped = len(self._live_bounces) - len(kept)
        if dropped <= 0:
            return 0
        self._live_bounces = kept
        self._total_retracted_live_bounces += dropped
        self._drop_ws_payloads_for_bounces_locked(bounces)
        for bd in bounces:
            self._clear_rally_buffer_bounce_locked(bd)
        self._dashboard_analytics_cache = {}
        if stats is not None and stat_key:
            stats[stat_key] = int(stats.get(stat_key, 0)) + dropped
        return dropped

    def _drop_live_bounces_for_ws_payload_locked(
        self,
        payload: dict,
        *,
        stats: dict | None = None,
    ) -> int:
        if not self._live_bounces:
            return 0
        sequence = payload.get("sequence")
        matches: list[dict] = []
        for bd in self._live_bounces:
            if sequence is not None and bd.get("sequence") == sequence:
                matches.append(bd)
                continue
            if self._ws_payload_matches_bounce(payload, bd):
                matches.append(bd)
        return self._retract_live_bounces_locked(
            matches,
            stats=stats,
            stat_key="speed_context_retracted_live_bounces",
        )

    def _clear_rally_buffer_bounce_locked(self, bd: dict) -> None:
        frame = self._event_frame(bd)
        if frame is None:
            return
        for row in reversed(self._rally_raw_buffer):
            row_frame = row.get("frame_index")
            if row_frame is None:
                continue
            try:
                row_frame_int = int(row_frame)
            except (TypeError, ValueError):
                continue
            if row_frame_int == frame:
                row["is_bounce"] = False
                row.pop("bounce_event", None)
                if not row.get("is_hit"):
                    row.pop("event_ball", None)
                return
            if row_frame_int < frame - 3:
                return

    def _retract_live_bounces_suppressed_by_hit_locked(self, hit: dict) -> list[dict]:
        hit_frame = self._event_frame(hit)
        suppress_frames = int(
            getattr(self.config.hit_bounce_refiner, "hit_suppression_frames", 3) or 0
        )
        if hit_frame is None or suppress_frames <= 0 or not self._live_bounces:
            return []

        camera_name = hit.get("camera_name", hit.get("camera"))
        if not camera_name:
            return []
        kept: list[dict] = []
        retracted: list[dict] = []
        for bd in self._live_bounces:
            bounce_frame = self._event_frame(bd)
            if (
                bounce_frame is not None
                and abs(bounce_frame - hit_frame) <= suppress_frames
                and self._event_matches_camera(bd, camera_name)
            ):
                retracted.append(bd)
            else:
                kept.append(bd)

        if not retracted:
            return []

        self._live_bounces = kept
        self._total_retracted_live_bounces += len(retracted)
        dropped_ws = self._drop_ws_payloads_for_bounces_locked(retracted)
        for bd in retracted:
            self._clear_rally_buffer_bounce_locked(bd)

        stats = self._yolo_fuzzy_live_stats.setdefault(str(camera_name or "unknown"), {})
        stats["retro_suppressed_bounces_by_hit"] = int(
            stats.get("retro_suppressed_bounces_by_hit", 0)
        ) + len(retracted)
        if dropped_ws:
            stats["retro_suppressed_ws_bounces_by_hit"] = int(
                stats.get("retro_suppressed_ws_bounces_by_hit", 0)
            ) + dropped_ws
        stats["last_retro_suppress_hit_frame"] = hit_frame
        stats["last_retro_suppressed_bounce_frames"] = [
            self._event_frame(bd) for bd in retracted
        ]
        return retracted

    def _update_ws_payload_speed_for_bounce_locked(self, bd: dict) -> int:
        updated = 0
        sequence = bd.get("sequence")
        for payload in self._ws_bounce_queue:
            if sequence is not None:
                if payload.get("sequence") != sequence:
                    continue
            elif not self._ws_payload_matches_bounce(payload, bd):
                continue

            speed = int(round(float(bd.get("speed_kmh", 0) or 0)))
            payload["speed"] = speed
            payload["speed_kmh"] = speed
            for key in (
                "speed_source",
                "speed_frame",
                "speed_frame_gap",
                "speed_age_s",
                "direction",
                "speed_direction",
            ):
                payload[key] = bd.get(key)
            updated += 1
        return updated

    def _refresh_zero_speed_live_bounces_locked(self) -> int:
        if not self._live_bounces or not self._live_speed_events:
            return 0
        refreshed = 0
        retracted: list[dict] = []
        for bd in self._live_bounces:
            try:
                current_speed = int(round(float(bd.get("speed_kmh", 0) or 0)))
            except Exception:
                current_speed = 0
            if current_speed > 0:
                continue
            updated = self._attach_recent_single_cam_speed_locked(dict(bd))
            try:
                updated_speed = int(round(float(updated.get("speed_kmh", 0) or 0)))
            except Exception:
                updated_speed = 0
            if updated_speed <= 0:
                continue
            camera_name = updated.get("camera_name", updated.get("camera", "unknown"))
            stats = self._yolo_fuzzy_live_stats.setdefault(str(camera_name or "unknown"), {})
            if self._reject_yolo_live_bounce_speed_context_locked(updated, stats=stats):
                retracted.append(bd)
                continue
            bd.update(updated)
            bd.update(self._normalize_live_bounce_dict(bd))
            self._update_ws_payload_speed_for_bounce_locked(bd)
            refreshed += 1
        if retracted:
            camera_name = retracted[-1].get("camera_name", retracted[-1].get("camera", "unknown"))
            stats = self._yolo_fuzzy_live_stats.setdefault(str(camera_name or "unknown"), {})
            self._retract_live_bounces_locked(
                retracted,
                stats=stats,
                stat_key="speed_context_retracted_live_bounces",
            )
        if refreshed:
            self._dashboard_analytics_cache = {}
            stats = self._yolo_fuzzy_live_stats.setdefault("cam68", {})
            stats["speed_refreshed_live_bounces"] = int(
                stats.get("speed_refreshed_live_bounces", 0)
            ) + refreshed
        return refreshed

    def _pop_ready_ws_bounce_locked(self) -> dict | None:
        """Return the next non-zero-speed WS bounce without blocking behind zeros."""
        if not self._ws_bounce_queue:
            return None
        self._refresh_zero_speed_live_bounces_locked()

        idx = 0
        while idx < len(self._ws_bounce_queue):
            payload = self._ws_bounce_queue[idx]
            try:
                speed_kmh = int(round(float(payload.get("speed_kmh", 0) or 0)))
            except Exception:
                speed_kmh = 0
            if speed_kmh > 0:
                camera_name = payload.get("camera_name", payload.get("camera", "unknown"))
                stats = self._yolo_fuzzy_live_stats.setdefault(
                    str(camera_name or "unknown"), {}
                )
                if self._reject_yolo_live_bounce_speed_context_locked(payload, stats=stats):
                    dropped_payload = dict(payload)
                    del self._ws_bounce_queue[idx]
                    self._drop_live_bounces_for_ws_payload_locked(
                        dropped_payload,
                        stats=stats,
                    )
                    continue
                del self._ws_bounce_queue[idx]
                return payload
            idx += 1

        now = time.time()
        for idx, payload in enumerate(list(self._ws_bounce_queue)):
            queued_at = float(payload.get("_queued_at", now) or now)
            if now - queued_at >= self._ws_zero_speed_grace_seconds:
                del self._ws_bounce_queue[idx]
                camera_name = payload.get("camera_name", payload.get("camera", "unknown"))
                stats = self._yolo_fuzzy_live_stats.setdefault(
                    str(camera_name or "unknown"), {}
                )
                stats["dropped_zero_speed_ws_bounces"] = int(
                    stats.get("dropped_zero_speed_ws_bounces", 0)
                ) + 1
                logger.info(
                    "3D display: dropped zero-speed bounce frame=%s sequence=%s",
                    payload.get("frame_index", payload.get("frame")),
                    payload.get("sequence"),
                )
                break
        return None

    def _backfill_zero_speed_bounces_from_speed_locked(self, speed_event: dict) -> int:
        try:
            speed_kmh = int(round(float(speed_event.get("speed_kmh", 0) or 0)))
        except Exception:
            speed_kmh = 0
        if speed_kmh <= 0 or not self._live_bounces:
            return 0

        speed_camera = speed_event.get("camera_name", speed_event.get("camera"))
        speed_frame = self._event_frame(speed_event)
        try:
            speed_ts = float(speed_event.get("timestamp", speed_event.get("capture_ts", 0.0)) or 0.0)
        except (TypeError, ValueError):
            speed_ts = 0.0

        updated = 0
        retracted: list[dict] = []
        max_frame_gap = 600
        max_age_seconds = 24.0
        fresh_frame_gap = 320
        fresh_age_seconds = 12.0
        for bd in reversed(self._live_bounces[-self._LIVE_BOUNCE_HISTORY_LIMIT:]):
            try:
                current_speed = int(round(float(bd.get("speed_kmh", 0) or 0)))
            except Exception:
                current_speed = 0
            if current_speed > 0:
                continue

            bounce_camera = bd.get("camera_name", bd.get("camera"))
            if speed_camera and bounce_camera and str(speed_camera) != str(bounce_camera):
                continue

            signed_frame_gap = None
            bounce_frame = self._event_frame(bd)
            if bounce_frame is not None and speed_frame is not None:
                signed_frame_gap = bounce_frame - speed_frame
                if signed_frame_gap > max_frame_gap:
                    continue
                if signed_frame_gap < -max_frame_gap:
                    break

            signed_age = None
            try:
                bounce_ts = float(bd.get("timestamp", bd.get("capture_ts", 0.0)) or 0.0)
            except (TypeError, ValueError):
                bounce_ts = 0.0
            if bounce_ts and speed_ts:
                signed_age = bounce_ts - speed_ts
                if signed_age > max_age_seconds:
                    continue
                if signed_age < -max_age_seconds:
                    break

            bd["speed_kmh"] = speed_kmh
            bd["speed"] = speed_kmh
            bd["speed_direction"] = speed_event.get("direction")
            if signed_frame_gap is not None and signed_frame_gap < 0:
                bd["speed_source"] = "future_single_cam_speed_backfill"
            else:
                is_fresh = True
                if signed_frame_gap is not None and signed_frame_gap > fresh_frame_gap:
                    is_fresh = False
                if signed_age is not None and signed_age > fresh_age_seconds:
                    is_fresh = False
                bd["speed_source"] = (
                    "nearest_single_cam_speed" if is_fresh else "stale_single_cam_speed"
                )
            bd["speed_frame"] = speed_event.get("frame_index", speed_event.get("frame"))
            if signed_frame_gap is not None:
                bd["speed_frame_gap"] = int(signed_frame_gap)
            if signed_age is not None:
                bd["speed_age_s"] = round(float(signed_age), 3)
            bounce_camera = bd.get("camera_name", bd.get("camera", "unknown"))
            stats = self._yolo_fuzzy_live_stats.setdefault(str(bounce_camera or "unknown"), {})
            if self._reject_yolo_live_bounce_speed_context_locked(bd, stats=stats):
                retracted.append(bd)
                continue
            self._update_ws_payload_speed_for_bounce_locked(bd)
            updated += 1
        if retracted:
            camera_name = retracted[-1].get("camera_name", retracted[-1].get("camera", "unknown"))
            stats = self._yolo_fuzzy_live_stats.setdefault(str(camera_name or "unknown"), {})
            self._retract_live_bounces_locked(
                retracted,
                stats=stats,
                stat_key="speed_context_retracted_live_bounces",
            )
        return updated

    def _record_live_speed_event_locked(self, event: dict) -> None:
        self._dashboard_analytics_cache = {}
        self._total_live_speed_events += 1
        event["sequence"] = self._total_live_speed_events
        self._live_speed_events.append(dict(event))
        if len(self._live_speed_events) > self._LIVE_BOUNCE_HISTORY_LIMIT:
            self._live_speed_events = self._live_speed_events[-self._LIVE_BOUNCE_HISTORY_LIMIT:]
        backfilled = self._backfill_zero_speed_bounces_from_speed_locked(event)
        if backfilled:
            cam_name = event.get("camera_name", event.get("camera", "unknown"))
            stats = self._yolo_fuzzy_live_stats.setdefault(str(cam_name or "unknown"), {})
            stats["speed_backfilled_bounces"] = int(
                stats.get("speed_backfilled_bounces", 0)
            ) + backfilled
        try:
            self._last_frame_speed_kmh = float(event.get("speed_kmh", 0.0) or 0.0)
        except Exception:
            pass
        if self._record_yolo_out_gate_restart_locked(event, kind="speed"):
            cam_name = self._yolo_out_gate_cam(event)
            if cam_name:
                self._release_yolo_out_gate_pending_bounces_locked(cam_name)

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

    def _record_live_bounce_locked(self, bd: dict, *, debug_source=None) -> bool:
        """Publish one accepted bounce to every realtime consumer from one source dict."""
        self._dashboard_analytics_cache = {}
        bd = self._attach_recent_single_cam_speed_locked(dict(bd))
        bd = self._normalize_live_bounce_dict(bd)
        camera_name = bd.get("camera_name", bd.get("camera", "unknown"))
        stats = self._yolo_fuzzy_live_stats.setdefault(str(camera_name or "unknown"), {})
        if self._reject_yolo_live_bounce_speed_context_locked(bd, stats=stats):
            return False
        self._total_live_bounces += 1
        bd["sequence"] = self._total_live_bounces
        self._live_bounces.append(bd)
        if len(self._live_bounces) > self._LIVE_BOUNCE_HISTORY_LIMIT:
            self._live_bounces = self._live_bounces[-self._LIVE_BOUNCE_HISTORY_LIMIT:]
        self._debug_record_bounce(debug_source if debug_source is not None else bd)
        self._enqueue_ws_bounce_locked(bd)
        self._record_yolo_out_gate_bounce_locked(bd)
        return True

    def _gate_live_bounce_candidate_locked(
        self,
        bd: dict,
        *,
        now: float,
        match_speed: bool,
    ) -> dict | None:
        """Apply the shared realtime bounce gate to one candidate event."""
        consumed_nc = None
        consumed_speed_event = None
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
            if consumed_nc is None:
                event_camera = bd.get("camera_name", bd.get("camera"))
                for speed_event in reversed(self._live_speed_events):
                    if speed_event.get("_used_for_bounce"):
                        continue
                    speed_camera = speed_event.get("camera_name", speed_event.get("camera"))
                    if event_camera and speed_camera and str(event_camera) != str(speed_camera):
                        continue
                    try:
                        speed_ts = float(
                            speed_event.get(
                                "timestamp",
                                speed_event.get("capture_ts", event_ts),
                            )
                        )
                    except (TypeError, ValueError):
                        continue
                    age = event_ts - speed_ts
                    if age < 0:
                        continue
                    if age < 3.0:
                        bd["speed_kmh"] = speed_event.get("speed_kmh", 0)
                        bd["speed_direction"] = speed_event.get("direction")
                        speed_event["_used_for_bounce"] = True
                        consumed_speed_event = speed_event
                        break
                    break

        if self._is_duplicate_bounce(bd):
            self._post_filter_stats["duplicate"] += 1
            if consumed_nc is not None:
                consumed_nc["_used"] = False
            if consumed_speed_event is not None:
                consumed_speed_event["_used_for_bounce"] = False
            return None

        ok, reason = self._post_filter_bounce(bd)
        self._post_filter_stats[reason] += 1
        if not ok:
            if consumed_nc is not None:
                consumed_nc["_used"] = False
            if consumed_speed_event is not None:
                consumed_speed_event["_used_for_bounce"] = False
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
            self._live_speed_events.clear()
            self._total_live_bounces = 0
            self._total_retracted_live_bounces = 0
            self._total_live_hits = 0
            self._total_live_speed_events = 0
            self._peak_bounces_eval.clear()
            self._post_filter_stats.clear()
            self._hit_bounce_refiner.reset()
            self._reset_yolo_fuzzy_live_locked()
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
        with self._analytics_lock:
            self._ws_bounce_queue.clear()
            self._ws_last_send_monotonic = 0.0
            self._ws_generation += 1
            generation = self._ws_generation
        self._ws_enabled = True
        self._ws_thread = threading.Thread(
            target=self._ws_push_loop, args=(generation,), daemon=True, name="ws-3d-push"
        )
        self._ws_thread.start()
        return {"enabled": True, "url": self._ws_url}

    def disable_3d_display(self) -> dict:
        """Disable WebSocket push."""
        self._ws_enabled = False
        with self._analytics_lock:
            self._ws_bounce_queue.clear()
            self._ws_last_send_monotonic = 0.0
            self._ws_generation += 1
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

    def _ws_seconds_until_next_send(self, *, now: float | None = None) -> float:
        min_interval = max(0.0, float(self._ws_min_send_interval_seconds or 0.0))
        if min_interval <= 0.0 or self._ws_last_send_monotonic <= 0.0:
            return 0.0
        now_val = float(now if now is not None else time.monotonic())
        return max(0.0, self._ws_last_send_monotonic + min_interval - now_val)

    def _mark_ws_bounce_sent(self, *, now: float | None = None) -> None:
        self._ws_last_send_monotonic = float(now if now is not None else time.monotonic())

    def _ws_generation_active(self, generation: int) -> bool:
        return (
            self._ws_enabled
            and int(generation) == int(self._ws_generation)
            and not self._stopped.is_set()
        )

    def _ws_push_loop(self, generation: int) -> None:
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

            while self._ws_generation_active(generation):
                try:
                    ws_url = self._ws_url
                    connect_kwargs = {"ping_interval": None}
                    if ws_url.startswith("wss://"):
                        connect_kwargs["ssl"] = ssl_ctx
                    if "tennisserver.motionrivalry.com" in ws_url:
                        connect_kwargs["origin"] = "https://tennis.motionrivalry.com"
                    async with websockets.connect(ws_url, **connect_kwargs) as ws:
                        logger.info("3D display connected: %s", ws_url)
                        while self._ws_generation_active(generation):
                            wait_s = self._ws_seconds_until_next_send()
                            if wait_s > 0:
                                await asyncio.sleep(min(wait_s, 0.25))
                                continue
                            with self._analytics_lock:
                                bd = self._pop_ready_ws_bounce_locked()
                            if bd:
                                if not self._ws_generation_active(generation):
                                    break
                                bounce_payload = {
                                    key: value
                                    for key, value in bd.items()
                                    if not str(key).startswith("_")
                                }
                                bounce_payload.update({
                                    "timeStamp": bd["timeStamp"],
                                    "timestamp": bd["timestamp"],
                                    "time_ms": bd["time_ms"],
                                    "capture_ts": bd.get("capture_ts"),
                                    "x": round(bd["x"], 4),
                                    "y": round(bd["y"], 4),
                                    "ws_x": round(bd["ws_x"], 4),
                                    "ws_y": round(bd["ws_y"], 4),
                                    "raw_x": round(bd["raw_x"], 4),
                                    "raw_y": round(bd["raw_y"], 4),
                                    "projected_x": bd.get("projected_x"),
                                    "projected_y": bd.get("projected_y"),
                                    "speed": int(round(bd["speed"])),
                                    "speed_kmh": int(round(bd["speed_kmh"])),
                                    "frame": bd.get("frame"),
                                    "frame_index": bd.get("frame_index"),
                                    "event_kind": bd.get("event_kind", "bounce"),
                                    "protocol_version": bd.get("protocol_version", 1),
                                })
                                msg = json.dumps({
                                    "room": "general",
                                    "msg": {
                                        "message": "bounce_data",
                                        "data": {
                                            "bounce": bounce_payload,
                                        }
                                    }
                                })
                                await ws.send(msg)
                                self._mark_ws_bounce_sent()
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
        info = self.get_latest_frame_info(name)
        return info.get("jpeg") if info else None

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
        """Switch between TrackNet and YOLO roadmap models at runtime.

        Args:
            model_name: "tracknet" or "yolo_roadmap"

        Returns:
            Dict with new model config info.
        """
        model_name = model_name.lower().strip()
        configs = {
            "tracknet": {
                "path": self.config.model.tracknet_path or "model_weight/TrackNet_finetuned.onnx",
                "frames_in": 8,
                "frames_out": 8,
                "detector_type": "tracknet",
            },
            "yolo": {
                "path": self.config.model.yolo_roadmap_path or "model_weight/best.pt",
                "frames_in": 1,
                "frames_out": 1,
                "detector_type": "yolo_roadmap",
            },
            "yolo_roadmap": {
                "path": self.config.model.yolo_roadmap_path or "model_weight/best.pt",
                "frames_in": 1,
                "frames_out": 1,
                "detector_type": "yolo_roadmap",
            },
        }
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}. Use 'tracknet' or 'yolo_roadmap'")

        running_live_cameras = [
            name
            for name, handle in self._handles.items()
            if name in self.config.cameras and handle.is_alive()
        ]
        for name in running_live_cameras:
            self.stop_pipeline(name)

        selected = configs[model_name]
        public_model_name = "yolo_roadmap" if selected["detector_type"] == "yolo_roadmap" else model_name
        self.config.model.path = selected["path"]
        self.config.model.frames_in = selected["frames_in"]
        self.config.model.frames_out = selected["frames_out"]
        self.config.model.detector_type = selected["detector_type"]
        self._is_median_bg = self.config.model.detector_type == "median_bg"
        self.reset_live_analytics()
        self._det_queues.clear()
        for name in running_live_cameras:
            self._latest_detections.pop(name, None)

        for name in running_live_cameras:
            self.start_pipeline(name)

        logger.info(
            "Model switched to %s: %s (detector=%s, frames=%d, restarted=%s)",
            public_model_name,
            self.config.model.path,
            self.config.model.detector_type,
            self.config.model.frames_in,
            running_live_cameras,
        )
        return {
            "model": public_model_name,
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
        detector_type = (self.config.model.detector_type or "").lower()
        normalized_path = path.replace("\\", "/").lower()
        if detector_type in {"yolo", "yolo_roadmap"} or "yolo_roadmap/" in normalized_path:
            name = "yolo_roadmap"
        elif detector_type == "tracknet" or "tracknet" in path_name:
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

        self._ensure_worker_threads()

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

        self._ensure_worker_threads()

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
            self._clear_latest_frame(handle_name)
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
            self._clear_latest_frame("_video_test")
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
