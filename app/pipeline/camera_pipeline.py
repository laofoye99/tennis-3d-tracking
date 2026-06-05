"""Camera pipeline subprocess: stream → inference → postprocess → homography → output queue."""

import logging
import multiprocessing as mp
import time
from typing import Any, Optional

import cv2
from app.pipeline.camera_stream import CameraStream
from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import create_detector
from app.pipeline.postprocess import BallTracker

logger = logging.getLogger(__name__)

YOLO_META_KEYS = (
    "yolo_conf",
    "bbox",
    "track_id",
    "pseudo_track_id",
    "static_count",
    "static_status",
    "static_zone_id",
    "raw_candidates",
    "event_only_raw_candidates",
    "source",
)


def run_pipeline(
    name: str,
    rtsp_url: str,
    model_path: str,
    input_size: tuple[int, int],
    frames_in: int,
    frames_out: int,
    threshold: float,
    device: str,
    homography_path: str,
    homography_key: str,
    result_queue: mp.Queue,
    stop_event: mp.Event,
    status_dict: dict[str, Any],
    frame_queue: Optional[mp.Queue] = None,
    detector_type: str = "auto",
    player_model_path: str = "",
    player_device: str = "cuda",
    player_conf: float = 0.4,
    player_imgsz: int = 960,
    player_use_tracking: bool = False,
    player_run_every_n: int = 5,
    preview_stride: int = 2,
) -> None:
    """Entry point for a camera pipeline subprocess.

    Runs a continuous loop: read frames → detect ball → transform to world coords → send result.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{name}] %(levelname)s %(message)s",
    )
    log = logging.getLogger(name)
    log.info("Pipeline starting...")
    preview_stride = max(1, int(preview_stride or 2))

    status_dict["state"] = "starting"
    status_dict["error_msg"] = ""
    status_dict["inference_ready"] = False
    status_dict["inference_error"] = ""

    stream: CameraStream | None = None
    _jpeg_q = None
    _jpeg_thread = None
    _preview_thread = None
    try:
        # Initialize components
        stream = CameraStream(rtsp_url, name)
        stream.start()

        # Model loading is optional – if it fails, video stream and recording
        # continue to work; only inference is disabled.
        detector = None
        tracker = None
        homography = None
        player_detector = None
        try:
            detector = create_detector(
                model_path, input_size, frames_in, frames_out, device,
                detector_type=detector_type,
            )
            # Load per-camera static median background if available
            if hasattr(detector, "load_static_median"):
                detector.load_static_median(name)
            # MedianBGDetector returns blobs directly; no BallTracker needed.
            tracker = None
            if not getattr(detector, "returns_blobs", False):
                tracker = BallTracker(original_size=(1920, 1080), threshold=threshold)
            homography = HomographyTransformer(homography_path, homography_key)
            status_dict["inference_ready"] = True
            status_dict["inference_error"] = ""
        except Exception as e:
            log.warning("Inference components failed to load, inference disabled: %s", e)
            status_dict["inference_enabled"] = False
            status_dict["inference_ready"] = False
            status_dict["inference_error"] = str(e)
            status_dict["error_msg"] = f"Inference disabled: {e}"

        if player_model_path:
            try:
                from app.pipeline.player_detector import PlayerPoseDetector
                player_detector = PlayerPoseDetector(
                    model_path=player_model_path,
                    device=player_device,
                    conf=player_conf,
                    imgsz=player_imgsz,
                    use_tracking=player_use_tracking,
                    run_every_n=player_run_every_n,
                )
            except Exception as e:
                log.warning("Player detector failed to load, disabled: %s", e)

        status_dict["state"] = "running"
        log.info("Pipeline running")

        frame_buffer: list = []
        raw_frame_buffer: list = []
        frame_id_buffer: list[int] = []
        capture_ts_buffer: list[float] = []  # wall-clock per frame
        last_frame_id = -1
        fps_counter = 0
        fps_time = time.time()

        # JPEG encoding and preview publishing run outside the inference loop.
        # YOLO can take longer than a frame budget; tying preview publication to
        # detector throughput makes the dashboard look frozen even while RTSP is
        # still healthy.
        import threading, queue as _queue
        _jpeg_q: _queue.Queue = _queue.Queue(maxsize=4)
        _is_yolo_preview = (
            str(detector_type or "").lower() in {"yolo", "yolo_roadmap"}
            or "yolo_roadmap" in str(model_path or "").replace("\\", "/").lower()
        )
        _preview_max_width = 640 if _is_yolo_preview else 720
        _preview_jpeg_quality = 55 if _is_yolo_preview else 60
        _preview_last_frame_id = -1
        _preview_counter = 0
        _preview_fps_time = time.time()

        def _push_latest_frame_payload(payload: dict) -> None:
            """Keep only the freshest preview/recording payload in frame_queue."""
            try:
                while True:
                    try:
                        frame_queue.get_nowait()
                    except _queue.Empty:
                        break
                    except Exception:
                        break
                frame_queue.put_nowait(payload)
            except Exception:
                pass

        def _put_latest_jpeg_job(
            raw_frame,
            is_rec: bool,
            payload_frame_id: int,
            payload_capture_ts: float,
        ) -> None:
            try:
                _jpeg_q.put_nowait((raw_frame, is_rec, payload_frame_id, payload_capture_ts))
            except _queue.Full:
                try:
                    _jpeg_q.get_nowait()
                except _queue.Empty:
                    pass
                try:
                    _jpeg_q.put_nowait((raw_frame, is_rec, payload_frame_id, payload_capture_ts))
                except _queue.Full:
                    pass

        def _jpeg_worker():
            while not stop_event.is_set():
                try:
                    item = _jpeg_q.get(timeout=1.0)
                except _queue.Empty:
                    continue
                if item is None:
                    break
                raw_frame, is_rec, payload_frame_id, payload_capture_ts = item
                try:
                    h, w = raw_frame.shape[:2]
                    preview = (
                        cv2.resize(
                            raw_frame,
                            (_preview_max_width, int(h * _preview_max_width / w)),
                            interpolation=cv2.INTER_AREA,
                        )
                        if w > _preview_max_width
                        else raw_frame
                    )
                    _, preview_jpeg = cv2.imencode(
                        ".jpg",
                        preview,
                        [cv2.IMWRITE_JPEG_QUALITY, _preview_jpeg_quality],
                    )
                    if is_rec:
                        _, recording_jpeg = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        _push_latest_frame_payload({
                            "preview": preview_jpeg.tobytes(),
                            "recording": recording_jpeg.tobytes(),
                            "frame_id": payload_frame_id,
                            "capture_ts": payload_capture_ts,
                        })
                    else:
                        _push_latest_frame_payload({
                            "preview": preview_jpeg.tobytes(),
                            "recording": None,
                            "frame_id": payload_frame_id,
                            "capture_ts": payload_capture_ts,
                        })
                except Exception:
                    pass

        def _preview_feeder():
            nonlocal _preview_last_frame_id, _preview_counter, _preview_fps_time
            while not stop_event.is_set():
                preview_frame, preview_frame_id, preview_ts = stream.read()
                if preview_frame is None or preview_frame_id == _preview_last_frame_id:
                    time.sleep(0.002)
                    continue
                _preview_last_frame_id = preview_frame_id

                is_recording = bool(status_dict.get("recording_enabled", False))
                if not is_recording and preview_stride > 1 and preview_frame_id % preview_stride != 0:
                    continue

                _put_latest_jpeg_job(
                    preview_frame.copy(),
                    is_recording,
                    preview_frame_id,
                    preview_ts or time.time(),
                )
                _preview_counter += 1
                now_preview = time.time()
                if now_preview - _preview_fps_time >= 1.0:
                    status_dict["preview_fps"] = _preview_counter / (now_preview - _preview_fps_time)
                    status_dict["preview_frame_id"] = preview_frame_id
                    _preview_counter = 0
                    _preview_fps_time = now_preview

        if frame_queue is not None:
            _jpeg_thread = threading.Thread(target=_jpeg_worker, daemon=True)
            _jpeg_thread.start()
            _preview_thread = threading.Thread(target=_preview_feeder, daemon=True)
            _preview_thread.start()

        while not stop_event.is_set():
            frame, frame_id, ts = stream.read()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)
                continue
            last_frame_id = frame_id
            capture_ts = time.time()  # wall-clock at frame arrival
            raw_frame = frame.copy()

            # Player pose detection (clean frame, before OSD mask)
            if player_detector is not None:
                try:
                    player_dets = player_detector.detect(raw_frame)
                    if player_dets and getattr(player_detector, "last_inference_ran", True):
                        player_msg = {
                            "type": "player_pose",
                            "camera_name": name,
                            "frame_id": frame_id,
                            "timestamp": time.time(),
                            "capture_ts": capture_ts,
                            "detections": player_dets,
                        }
                        try:
                            result_queue.put_nowait(player_msg)
                        except Exception:
                            pass
                except Exception as e:
                    log.debug("Player detection error: %s", e)

            # Mask OSD for inference only (after JPEG thread got clean ref)
            frame = raw_frame.copy()
            frame[0:41, 0:603] = 0

            frame_buffer.append(frame)
            raw_frame_buffer.append(raw_frame)
            frame_id_buffer.append(frame_id)
            capture_ts_buffer.append(capture_ts)

            if len(frame_buffer) < frames_in:
                continue

            # 推理开关关闭或模型未加载时直接跳过 GPU 调用
            if (
                detector is None
                or not status_dict.get("inference_enabled", True)
                or not status_dict.get("inference_ready", detector is not None)
            ):
                frame_buffer.clear()
                raw_frame_buffer.clear()
                frame_id_buffer.clear()
                capture_ts_buffer.clear()
                continue

            # Inference on the buffer
            try:
                heatmaps = detector.infer(frame_buffer)
                if hasattr(detector, "get_runtime_stats"):
                    status_dict["detector_stats"] = detector.get_runtime_stats()
            except Exception as e:
                log.error("Inference error: %s", e)
                frame_buffer.clear()
                raw_frame_buffer.clear()
                frame_id_buffer.clear()
                capture_ts_buffer.clear()
                continue

            if isinstance(heatmaps, dict):
                # ---- MedianBG path: send raw blob_block ----
                blob_block = {}
                capture_ts_by_frame = {}
                for local_i, blobs in heatmaps.items():
                    if local_i < len(frame_id_buffer):
                        frame_key = frame_id_buffer[local_i]
                        blob_block[frame_key] = blobs
                        capture_ts_by_frame[frame_key] = capture_ts_buffer[local_i]
                msg = {
                    "camera_name": name,
                    "type": "blob_block",
                    "blobs": blob_block,
                    "capture_ts_by_frame": capture_ts_by_frame,
                    "capture_ts": capture_ts_buffer[0],
                    "timestamp": time.time(),
                }
                try:
                    result_queue.put_nowait(msg)
                except Exception:
                    pass
                status_dict["last_detection_time"] = time.time()
            elif isinstance(heatmaps, list) and heatmaps and isinstance(heatmaps[0], list):
                # ---- BallSelector path: list of blob lists ----
                for i in range(min(frames_out, len(heatmaps))):
                    blobs = heatmaps[i]
                    if not blobs:
                        continue
                    frame_capture_ts = capture_ts_buffer[i] if i < len(capture_ts_buffer) else capture_ts_buffer[0]

                    top = blobs[0]
                    px, py, conf = top["pixel_x"], top["pixel_y"], top["blob_sum"]
                    wx, wy = homography.pixel_to_world(px, py)

                    candidates = []
                    for b in blobs:
                        bwx, bwy = homography.pixel_to_world(b["pixel_x"], b["pixel_y"])
                        candidate = {
                            "x": bwx, "y": bwy,
                            "world_x": bwx, "world_y": bwy,
                            "pixel_x": b["pixel_x"], "pixel_y": b["pixel_y"],
                            "blob_sum": b["blob_sum"],
                        }
                        for key in YOLO_META_KEYS:
                            if b.get(key) is not None:
                                candidate[key] = b[key]
                        candidates.append(candidate)

                    detection = {
                        "camera_name": name,
                        "x": wx, "y": wy,
                        "pixel_x": px, "pixel_y": py,
                        "confidence": conf, "blob_sum": conf,
                        "timestamp": time.time(),
                        "capture_ts": frame_capture_ts,
                        "frame_index": frame_id_buffer[i] if i < len(frame_id_buffer) else 0,
                        "candidates": candidates,
                    }
                    if top.get("yolo_conf") is not None:
                        detection["yolo_conf"] = top["yolo_conf"]
                    if top.get("source") is not None:
                        detection["source"] = top["source"]
                    for key in ("static_count", "static_status", "static_zone_id", "raw_candidates", "event_only_raw_candidates"):
                        if top.get(key) is not None:
                            detection[key] = top[key]
                    try:
                        result_queue.put_nowait(detection)
                    except Exception:
                        pass
                    status_dict["last_detection_time"] = time.time()
            else:
                # ---- TrackNet / HRNet path: heatmaps → BallTracker ----
                n_out = min(frames_out, len(heatmaps))
                blobs_by_frame = [
                    tracker.process_heatmap_multi(heatmaps[i], max_blobs=4)
                    for i in range(n_out)
                ]
                for i, blobs in enumerate(blobs_by_frame):
                    if not blobs:
                        continue
                    frame_capture_ts = capture_ts_buffer[i] if i < len(capture_ts_buffer) else capture_ts_buffer[0]

                    top = blobs[0]
                    px, py, conf = top["pixel_x"], top["pixel_y"], top["blob_sum"]
                    wx, wy = homography.pixel_to_world(px, py)

                    candidates = []
                    for b in blobs:
                        bwx, bwy = homography.pixel_to_world(b["pixel_x"], b["pixel_y"])
                        candidate = {
                            "x": bwx, "y": bwy,
                            "world_x": bwx, "world_y": bwy,
                            "pixel_x": b["pixel_x"], "pixel_y": b["pixel_y"],
                            "blob_sum": b["blob_sum"],
                        }
                        for key in YOLO_META_KEYS:
                            if b.get(key) is not None:
                                candidate[key] = b[key]
                        candidates.append(candidate)

                    detection = {
                        "camera_name": name,
                        "x": wx, "y": wy,
                        "pixel_x": px, "pixel_y": py,
                        "confidence": conf, "blob_sum": conf,
                        "timestamp": time.time(),
                        "capture_ts": frame_capture_ts,
                        "frame_index": frame_id_buffer[i] if i < len(frame_id_buffer) else 0,
                        "candidates": candidates,
                    }
                    if top.get("yolo_conf") is not None:
                        detection["yolo_conf"] = top["yolo_conf"]
                    if top.get("source") is not None:
                        detection["source"] = top["source"]
                    for key in ("static_count", "static_status", "static_zone_id", "raw_candidates", "event_only_raw_candidates"):
                        if top.get(key) is not None:
                            detection[key] = top[key]
                    try:
                        result_queue.put_nowait(detection)
                    except Exception:
                        pass
                    status_dict["last_detection_time"] = time.time()

            fps_counter += frames_out
            now = time.time()
            if now - fps_time >= 1.0:
                status_dict["fps"] = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            frame_buffer.clear()
            raw_frame_buffer.clear()
            frame_id_buffer.clear()
            capture_ts_buffer.clear()

    except Exception as e:
        log.exception("Pipeline crashed: %s", e)
        status_dict["state"] = "error"
        status_dict["error_msg"] = str(e)
    finally:
        stop_event.set()
        if _jpeg_q is not None:
            try:
                _jpeg_q.put_nowait(None)
            except Exception:
                pass
        if _preview_thread is not None:
            try:
                _preview_thread.join(timeout=2.0)
            except Exception:
                pass
        if _jpeg_thread is not None:
            try:
                _jpeg_thread.join(timeout=2.0)
            except Exception:
                pass
        if stream is not None:
            stream.stop()
        status_dict["state"] = status_dict.get("state", "stopped")
        if status_dict["state"] == "running":
            status_dict["state"] = "stopped"
        log.info("Pipeline exited (state=%s)", status_dict["state"])
