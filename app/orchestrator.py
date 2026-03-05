"""Main process orchestrator: manages camera pipeline subprocesses and triangulation."""

import datetime
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
        self._video_test_detections: dict[str, list[dict]] = {}  # camera_name -> detections

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
        handle.frame_queue = mp.Queue(maxsize=32)
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
        cam_positions = {
            n: self.config.cameras[n].position_3d for n in cam_names
        }

        while not self._stopped.is_set():
            got_any = False
            for name, handle in list(self._handles.items()):
                # 消费检测结果
                if handle.result_queue is not None:
                    try:
                        while not handle.result_queue.empty():
                            det = handle.result_queue.get_nowait()
                            self._latest_detections[name] = det
                            if name == "_video_test":
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
            for name, handle in self._handles.items():
                if handle.is_alive():
                    fname = str(self._recordings_dir / f"{name}_{ts}.mp4")
                    self._recording_writers[name] = {"writer": None, "path": fname}
                    files[name] = fname
            if not files:
                return {"status": "no_cameras_running", "files": {}}
            # Signal pipelines to send every frame
            for handle in self._handles.values():
                if handle.status_dict is not None:
                    handle.status_dict["recording_enabled"] = True
            self._recording = True
            self._recording_info = {"start_time": time.time(), "files": files}
            logger.info("Recording started: %s", files)
            return {"status": "recording", "files": files}

    def stop_recording(self) -> dict:
        """停止录像并写入文件。"""
        with self._recording_lock:
            if not self._recording:
                return {"status": "not_recording", "files": {}}
            self._recording = False
            # Signal pipelines to revert to preview rate
            for handle in self._handles.values():
                if handle.status_dict is not None:
                    handle.status_dict["recording_enabled"] = False
            files = {}
            for name, wr_info in self._recording_writers.items():
                writer = wr_info.get("writer")
                if writer is not None:
                    writer.release()
                files[name] = wr_info["path"]
            self._recording_writers.clear()
            elapsed = time.time() - self._recording_info.get("start_time", time.time())
            logger.info("Recording stopped (%.1fs), files: %s", elapsed, files)
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
        """解码 JPEG 并写入对应 VideoWriter（在 _recording_lock 外调用）。"""
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
            wr_info["writer"].write(img)
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
        )

    def get_latest_3d(self) -> Optional[BallPosition3D]:
        return self._latest_3d

    def get_latest_detection(self, name: str) -> Optional[dict]:
        return self._latest_detections.get(name)

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
            },
            daemon=True,
        )
        handle.process.start()
        logger.info(
            "[video-test] Started: %s [%.1f-%.1f] cam=%s pid=%d",
            video_path, start_time, end_time, camera_name, handle.process.pid,
        )

        self._video_test_handle = handle
        self._handles["_video_test"] = handle

        # Ensure consumer thread is running
        if self._consumer_thread is None or not self._consumer_thread.is_alive():
            self._stopped.clear()
            self._consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._consumer_thread.start()

        return {"status": "started"}

    def stop_video_test(self) -> dict:
        """Stop video test pipeline."""
        handle = self._video_test_handle
        if handle is None:
            return {"status": "not_running"}
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

    def compute_3d_from_detections(self) -> list[dict]:
        """Match detections from two cameras by frame_index and compute 3D positions."""
        cam_names = list(self._video_test_detections.keys())
        if len(cam_names) < 2:
            return []

        cam1_name, cam2_name = cam_names[0], cam_names[1]
        cam1_dets = {d["frame_index"]: d for d in self._video_test_detections[cam1_name]}
        cam2_dets = {d["frame_index"]: d for d in self._video_test_detections[cam2_name]}

        common_frames = sorted(set(cam1_dets.keys()) & set(cam2_dets.keys()))

        cam1_cfg = self.config.cameras.get(cam1_name)
        cam2_cfg = self.config.cameras.get(cam2_name)
        if not cam1_cfg or not cam2_cfg:
            logger.error("Camera config not found for %s or %s", cam1_name, cam2_name)
            return []

        results = []
        for frame_idx in common_frames:
            d1 = cam1_dets[frame_idx]
            d2 = cam2_dets[frame_idx]
            try:
                x, y, z = triangulate(
                    (d1["x"], d1["y"]),
                    (d2["x"], d2["y"]),
                    cam1_cfg.position_3d,
                    cam2_cfg.position_3d,
                )
                results.append({
                    "frame_index": frame_idx,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": round(z, 4),
                    "cam1_pixel": [round(d1["pixel_x"], 1), round(d1["pixel_y"], 1)],
                    "cam2_pixel": [round(d2["pixel_x"], 1), round(d2["pixel_y"], 1)],
                })
            except Exception as e:
                logger.warning("3D computation failed for frame %d: %s", frame_idx, e)

        return results

    def get_video_test_status(self) -> dict:
        """Get video test pipeline status."""
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
