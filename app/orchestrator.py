"""Main process orchestrator: manages camera pipeline subprocesses and triangulation."""

import datetime
import json
import logging
import multiprocessing as mp
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
    BounceDetector,
    EnhancedBounceDetector,
    RallyStateMachine,
    RallyTracker,
    run_batch_analytics,
)
from app.trajectory import clean_detections, find_offset_and_triangulate, fit_trajectory, segment_rallies
from app.pipeline.multi_blob_matcher import MultiBlobMatcher
from app.triangulation import triangulate

logger = logging.getLogger(__name__)

# Maximum age (seconds) for pairing detections from two cameras.
_MATCH_WINDOW = 0.1


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
        self._last_tri_pair: tuple = (None, None)  # (d1.ts, d2.ts) to dedup
        self._consumer_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._inference_enabled: bool = True  # 全局推理开关

        # 录像
        self._recording: bool = False
        self._recording_writers: dict[str, Any] = {}  # name -> {"writer": VideoWriter|None, "path": str}
        self._recording_info: dict = {}
        self._recording_lock = threading.Lock()
        self._recordings_dir = Path("recordings")

        # Video test
        self._video_test_handle: Optional[_PipelineHandle] = None
        self._video_test_handles: dict[str, _PipelineHandle] = {}  # parallel handles
        self._video_test_detections: dict[str, list[dict]] = {}  # camera_name -> detections

        # Live analytics (bounce detection + rally tracking)
        self._bounce_detector = BounceDetector()
        self._rally_tracker = RallyTracker()
        self._live_bounces: list[dict] = []
        # Enhanced analytics (event-driven state machine)
        self._enhanced_bounce = EnhancedBounceDetector()
        self._rally_sm = RallyStateMachine()
        self._live_rallies: list[dict] = []
        self._analytics_lock = threading.Lock()

        # Confidence filtering (top1_conf20)
        self._conf_percentile = 20  # reject bottom 20% by blob_sum
        self._conf_history: list[float] = []  # rolling blob_sum values
        self._conf_threshold = 0.0  # dynamic, updated from history

        # Net crossing speed detection
        self._prev_3d: Optional[dict] = None  # previous 3D point for speed calc
        self._latest_net_crossing: Optional[dict] = None
        self._net_crossings: list[dict] = []
        NET_Y = 11.885  # net position
        self._NET_Y = NET_Y

        # 3D display WebSocket push
        self._ws_bounce_queue: list[dict] = []  # bounces to push
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
            },
            daemon=True,
        )
        handle.process.start()
        logger.info("[%s] Pipeline process started (pid=%d)", name, handle.process.pid)

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
        logger.info("[%s] Pipeline stopped", name)

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

        cam_names = list(self.config.cameras.keys())
        cam_positions = self._get_camera_positions()

        # Initialize multi-blob matcher for live mode
        live_matcher = None
        if len(cam_names) == 2:
            pos1 = cam_positions.get(cam_names[0])
            pos2 = cam_positions.get(cam_names[1])
            if pos1 and pos2:
                live_matcher = MultiBlobMatcher(pos1, pos2)

        while not self._stopped.is_set():
            got_any = False
            for name, handle in list(self._handles.items()):
                # 消费检测结果
                if handle.result_queue is not None:
                    try:
                        while not handle.result_queue.empty():
                            det = handle.result_queue.get_nowait()
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
                        new_jpeg: bytes | None = None
                        while not handle.frame_queue.empty():
                            new_jpeg = handle.frame_queue.get_nowait()
                            self._latest_frames[name] = new_jpeg
                        if new_jpeg is not None and self._recording:
                            self._write_recording_frame(name, new_jpeg)
                    except Exception:
                        pass

            # Attempt triangulation when both cameras have recent data.
            if len(cam_names) == 2 and all(c in self._latest_detections for c in cam_names):
                d1 = self._latest_detections[cam_names[0]]
                d2 = self._latest_detections[cam_names[1]]
                # Skip if we already triangulated this exact pair (by pixel coords)
                pair_id = (d1.get("pixel_x"), d1.get("pixel_y"),
                           d2.get("pixel_x"), d2.get("pixel_y"))
                if pair_id == self._last_tri_pair:
                    pass  # already processed
                elif abs(d1["timestamp"] - d2["timestamp"]) >= _MATCH_WINDOW:
                    pass  # too far apart
                else:
                    # --- Confidence filtering (top1_conf20) ---
                    blob_sum1 = d1.get("blob_sum", d1.get("confidence", 1.0))
                    blob_sum2 = d2.get("blob_sum", d2.get("confidence", 1.0))
                    avg_conf = (blob_sum1 + blob_sum2) / 2
                    self._conf_history.append(avg_conf)
                    if len(self._conf_history) > 500:
                        self._conf_history = self._conf_history[-500:]
                    if len(self._conf_history) >= 50:
                        sorted_h = sorted(self._conf_history)
                        idx = int(len(sorted_h) * self._conf_percentile / 100)
                        self._conf_threshold = sorted_h[idx]
                    if avg_conf < self._conf_threshold:
                        continue  # skip low-confidence detection

                    try:
                        x, y, z = None, None, None

                        # Try multi-blob matching first
                        if (live_matcher
                                and "candidates" in d1
                                and "candidates" in d2):
                            match = live_matcher.match(d1, d2)
                            if match is not None:
                                x, y, z = match["x"], match["y"], match["z"]

                        # Fallback to single-blob triangulation
                        if x is None:
                            x, y, z = triangulate(
                                (d1["x"], d1["y"]),
                                (d2["x"], d2["y"]),
                                cam_positions[cam_names[0]],
                                cam_positions[cam_names[1]],
                            )

                        self._latest_3d = BallPosition3D(
                            x=x, y=y, z=z,
                            cam66_world=WorldPoint2D(**d1),
                            cam68_world=WorldPoint2D(**d2),
                        )
                        self._last_tri_pair = pair_id

                        # --- Latency measurement ---
                        cap_ts1 = d1.get("capture_ts", d1["timestamp"])
                        cap_ts2 = d2.get("capture_ts", d2["timestamp"])
                        latency_ms = (time.time() - min(cap_ts1, cap_ts2)) * 1000
                        self._latency_buffer.append(latency_ms)
                        if latency_ms > self._latency_max:
                            self._latency_max = latency_ms

                        # --- Net crossing speed detection ---
                        now = time.time()
                        capture_ts = min(
                            d1.get("capture_ts", d1["timestamp"]),
                            d2.get("capture_ts", d2["timestamp"]),
                        )
                        pt = {"x": x, "y": y, "z": z, "timestamp": now,
                              "capture_ts": capture_ts}
                        if self._prev_3d is not None:
                            prev_y = self._prev_3d["y"]
                            curr_y = y
                            # Check if ball crossed the net
                            if (prev_y < self._NET_Y and curr_y >= self._NET_Y) or \
                               (prev_y > self._NET_Y and curr_y <= self._NET_Y):
                                t_delta = now - self._prev_3d["timestamp"]
                                if t_delta > 0.001:
                                    dx = x - self._prev_3d["x"]
                                    dy = y - self._prev_3d["y"]
                                    dz = z - self._prev_3d["z"]
                                    dist = (dx**2 + dy**2 + dz**2) ** 0.5
                                    speed_ms = dist / t_delta
                                    speed_kmh = speed_ms * 3.6
                                    if 20 <= speed_kmh <= 250:  # sane range
                                        direction = "near_to_far" if curr_y > prev_y else "far_to_near"
                                        crossing = {
                                            "speed_kmh": round(speed_kmh, 1),
                                            "direction": direction,
                                            "timestamp": now,
                                            "x": x, "y": y, "z": z,
                                        }
                                        self._latest_net_crossing = crossing
                                        self._net_crossings.append(crossing)
                                        if len(self._net_crossings) > 100:
                                            self._net_crossings = self._net_crossings[-100:]
                        self._prev_3d = pt

                        # Feed live analytics
                        cam_dets = {}
                        for cname, det in [(cam_names[0], d1), (cam_names[1], d2)]:
                            cam_dets[cname] = {
                                "world_x": det.get("x"),
                                "world_y": det.get("y"),
                                "pixel_x": det.get("pixel_x"),
                                "pixel_y": det.get("pixel_y"),
                                "yolo_conf": det.get("yolo_conf", 0.5),
                            }
                        with self._analytics_lock:
                            # Legacy analytics
                            bounce = self._bounce_detector.update(pt)
                            self._rally_tracker.update(pt, bounce)
                            if bounce is not None:
                                self._live_bounces.append(bounce.to_dict())
                                if len(self._live_bounces) > 50:
                                    self._live_bounces = self._live_bounces[-50:]
                                # --- Push bounce to 3D display queue ---
                                if self._ws_enabled:
                                    bx, by = bounce.x, bounce.y
                                    self._ws_bounce_queue.append({
                                        "x": (bx - 1.37) / 5.49,  # singles normalized
                                        "y": 1.0 - (by / 23.77),  # 0=far, 1=near
                                        "speed": self._latest_net_crossing["speed_kmh"] if self._latest_net_crossing else 0,
                                        "timestamp": int(now * 1000),
                                    })
                            # Enhanced analytics
                            ebounce = self._enhanced_bounce.update(pt, cam_dets)
                            rally_result = self._rally_sm.update(pt, ebounce)
                            if ebounce is not None:
                                bd = ebounce.to_dict()
                                bd["capture_ts"] = capture_ts  # when the frame was actually captured
                                bd["detect_delay"] = round(now - capture_ts, 2)  # detection pipeline delay
                                # Attach most recent net crossing speed (within 3s)
                                if self._latest_net_crossing and (now - self._latest_net_crossing["timestamp"]) < 3.0:
                                    bd["speed_kmh"] = self._latest_net_crossing["speed_kmh"]
                                    bd["speed_direction"] = self._latest_net_crossing["direction"]
                                self._live_bounces.append(bd)
                            if rally_result is not None:
                                self._live_rallies.append(rally_result.to_dict())
                                if len(self._live_rallies) > 20:
                                    self._live_rallies = self._live_rallies[-20:]
                    except Exception as e:
                        logger.error("Triangulation error: %s", e)

            if not got_any:
                time.sleep(0.005)

        self._triangulation_active = False
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
            logger.info("Recording stopped (%.1fs, target %d frames), files: %s", elapsed, target_frames, files)
            result = {"status": "stopped", "files": files, "duration_s": round(elapsed, 1)}
            self._recording_info = {}
            return result

    def get_recording_status(self) -> dict:
        if not self._recording:
            ffmpeg_active = bool(self._ffmpeg_processes)
            if ffmpeg_active:
                elapsed = time.time() - self._ffmpeg_start_time
                return {
                    "recording": True,
                    "mode": "ffmpeg",
                    "duration_s": round(elapsed, 1),
                    "files": {n: p["path"] for n, p in self._ffmpeg_processes.items()},
                }
            return {"recording": False}
        elapsed = time.time() - self._recording_info.get("start_time", time.time())
        return {
            "recording": True,
            "mode": "opencv",
            "duration_s": round(elapsed, 1),
            "files": self._recording_info.get("files", {}),
        }

    # ------------------------------------------------------------------
    # FFmpeg Recording with Audio (alternative to OpenCV recording)
    # ------------------------------------------------------------------
    _ffmpeg_processes: dict = {}
    _ffmpeg_start_time: float = 0

    def start_recording_ffmpeg(self) -> dict:
        """Start recording with ffmpeg (video + audio from RTSP).

        Uses ffmpeg subprocess to directly capture RTSP streams including
        audio tracks. This preserves the original stream quality and
        includes microphone audio for frame alignment.

        Does NOT interfere with the existing OpenCV recording method.
        """
        import subprocess, shutil

        if self._ffmpeg_processes:
            return {"status": "already_recording_ffmpeg",
                    "files": {n: p["path"] for n, p in self._ffmpeg_processes.items()}}

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return {"status": "error", "message": "ffmpeg not found in PATH"}

        self._recordings_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        for name, cam_cfg in self._config.cameras.items():
            rtsp_url = cam_cfg.rtsp_url
            if not rtsp_url:
                continue

            out_path = str(self._recordings_dir / f"{name}_{ts}_av.mp4")

            # ffmpeg command:
            # -rtsp_transport tcp: use TCP for reliable RTSP
            # -i: input RTSP stream
            # -c:v copy: copy video stream without re-encoding (fast, lossless)
            # -c:a aac: encode audio to AAC (RTSP usually sends PCM/G711)
            # -y: overwrite output
            cmd = [
                ffmpeg_bin,
                "-rtsp_transport", "tcp",
                "-i", rtsp_url,
                "-c:v", "copy",
                "-c:a", "aac",
                "-y",
                out_path,
            ]

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                self._ffmpeg_processes[name] = {"proc": proc, "path": out_path}
                files[name] = out_path
                logger.info("[%s] ffmpeg recording started: %s", name, out_path)
            except Exception as e:
                logger.error("[%s] ffmpeg failed to start: %s", name, e)

        if not files:
            return {"status": "error", "message": "no cameras started"}

        self._ffmpeg_start_time = time.time()
        return {"status": "recording_ffmpeg", "files": files}

    def stop_recording_ffmpeg(self) -> dict:
        """Stop ffmpeg recording by sending 'q' to stdin or SIGINT."""
        import signal

        if not self._ffmpeg_processes:
            return {"status": "not_recording_ffmpeg"}

        files = {}
        elapsed = time.time() - self._ffmpeg_start_time

        for name, info in self._ffmpeg_processes.items():
            proc = info["proc"]
            try:
                # Send SIGINT (graceful stop, ffmpeg finalizes the file)
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            files[name] = info["path"]
            logger.info("[%s] ffmpeg recording stopped: %s", name, info["path"])

        self._ffmpeg_processes.clear()
        return {
            "status": "stopped_ffmpeg",
            "files": files,
            "duration_s": round(elapsed, 1),
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
        return SystemStatus(
            pipelines=pipelines,
            triangulation_active=self._triangulation_active,
            latest_ball_3d=self._latest_3d,
            analytics=self.get_live_analytics(),
        )

    def get_latest_3d(self) -> Optional[BallPosition3D]:
        return self._latest_3d

    def get_latest_detection(self, name: str) -> Optional[dict]:
        return self._latest_detections.get(name)

    def get_live_analytics(self) -> dict:
        """Return current live bounce/rally state for the dashboard."""
        with self._analytics_lock:
            return {
                "rally_state": self._rally_tracker.get_state().to_dict(),
                "recent_bounces": list(self._live_bounces[-10:]),
                "completed_rallies": self._rally_tracker.get_completed_rallies(),
                # Enhanced analytics
                "enhanced_state": self._rally_sm.get_state_dict(),
                "enhanced_rallies": self._enrich_rallies(),
            }

    def _enrich_rallies(self) -> list[dict]:
        """Enrich rally results with net crossing speeds for each bounce."""
        rallies = []
        for r in self._rally_sm.get_completed_rallies():
            rd = r.to_dict()
            # Attach speed_kmh to each bounce by finding the nearest net crossing
            for b in rd.get("bounces", []):
                bt = b.get("timestamp", 0)
                best = None
                best_dt = 3.0  # max 3 seconds lookback
                for nc in self._net_crossings:
                    dt = bt - nc["timestamp"]
                    if 0 < dt < best_dt:
                        best_dt = dt
                        best = nc
                if best:
                    b["speed_kmh"] = best["speed_kmh"]
                    b["speed_direction"] = best["direction"]
            # Also add rally-level summary
            rd["duration"] = rd.get("duration_seconds", 0)
            rallies.append(rd)
        return rallies

    def reset_live_analytics(self) -> None:
        """Reset analytics state (e.g. when starting a new session)."""
        with self._analytics_lock:
            self._bounce_detector.reset()
            self._rally_tracker.reset()
            self._live_bounces.clear()
            self._enhanced_bounce.reset()
            self._rally_sm.reset()
            self._live_rallies.clear()

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
                                bd = self._ws_bounce_queue.pop(0)
                                msg = json.dumps({
                                    "room": "general",
                                    "msg": {
                                        "message": "bounce_data",
                                        "data": {
                                            "bounce": {
                                                "timeStamp": bd["timestamp"],
                                                "x": round(bd["x"], 4),
                                                "y": round(bd["y"], 4),
                                                "speed": round(bd["speed"], 1),
                                            }
                                        }
                                    }
                                })
                                await ws.send(msg)
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
        if model_name == "hrnet":
            self.config.model.path = "model_weight/hrnet_tennis.onnx"
            self.config.model.frames_in = 3
            self.config.model.frames_out = 3
        elif model_name == "tracknet":
            self.config.model.path = "model_weight/TrackNet_best.pt"
            self.config.model.frames_in = 8
            self.config.model.frames_out = 8
        else:
            raise ValueError(f"Unknown model: {model_name}. Use 'hrnet' or 'tracknet'")

        logger.info("Model switched to %s: %s (frames=%d)",
                     model_name, self.config.model.path, self.config.model.frames_in)
        return {
            "model": model_name,
            "path": self.config.model.path,
            "frames_in": self.config.model.frames_in,
            "frames_out": self.config.model.frames_out,
        }

    def get_current_model(self) -> dict:
        """Return current model info."""
        path = self.config.model.path
        name = "hrnet" if path.endswith(".onnx") else "tracknet"
        return {
            "model": name,
            "path": path,
            "frames_in": self.config.model.frames_in,
            "frames_out": self.config.model.frames_out,
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
