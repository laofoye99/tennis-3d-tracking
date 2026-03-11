"""Main process orchestrator: manages camera pipeline subprocesses and triangulation."""

import datetime
import json
import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from app.config import AppConfig
from app.pipeline.camera_pipeline import run_pipeline
from app.pipeline.video_pipeline import run_video_pipeline
from app.schemas import BallPosition3D, PipelineStatus, SystemStatus, WorldPoint2D
from app.analytics import BounceDetector, RallyTracker, run_batch_analytics
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
        self._analytics_lock = threading.Lock()

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
                dt = abs(d1["timestamp"] - d2["timestamp"])
                if dt < _MATCH_WINDOW:
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
                        # Feed live analytics
                        pt = {"x": x, "y": y, "z": z, "timestamp": time.time()}
                        with self._analytics_lock:
                            bounce = self._bounce_detector.update(pt)
                            self._rally_tracker.update(pt, bounce)
                            if bounce is not None:
                                self._live_bounces.append(bounce.to_dict())
                                if len(self._live_bounces) > 50:
                                    self._live_bounces = self._live_bounces[-50:]
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
            return {"recording": False}
        elapsed = time.time() - self._recording_info.get("start_time", time.time())
        return {
            "recording": True,
            "duration_s": round(elapsed, 1),
            "files": self._recording_info.get("files", {}),
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
            }

    def reset_live_analytics(self) -> None:
        """Reset analytics state (e.g. when starting a new session)."""
        with self._analytics_lock:
            self._bounce_detector.reset()
            self._rally_tracker.reset()
            self._live_bounces.clear()

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
