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

    status_dict["state"] = "starting"
    status_dict["error_msg"] = ""

    stream: CameraStream | None = None
    try:
        # Initialize components
        stream = CameraStream(rtsp_url, name)
        stream.start()

        # Model loading is optional – if it fails, video stream and recording
        # continue to work; only inference is disabled.
        detector = None
        tracker = None
        homography = None
        try:
            detector = create_detector(model_path, input_size, frames_in, frames_out, device)
            tracker = BallTracker(original_size=(1920, 1080), threshold=threshold)
            homography = HomographyTransformer(homography_path, homography_key)
        except Exception as e:
            log.warning("Inference components failed to load, inference disabled: %s", e)
            status_dict["inference_enabled"] = False

        status_dict["state"] = "running"
        log.info("Pipeline running")

        frame_buffer: list = []
        frame_id_buffer: list[int] = []
        capture_ts_buffer: list[float] = []  # wall-clock per frame
        last_frame_id = -1
        fps_counter = 0
        fps_time = time.time()

        while not stop_event.is_set():
            frame, frame_id, ts = stream.read()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)
                continue
            last_frame_id = frame_id
            capture_ts = time.time()  # wall-clock at frame arrival

            # --- 向 frame_queue 放 JPEG（预览或录像）---
            if frame_queue is not None:
                is_recording = status_dict.get("recording_enabled", False)
                if is_recording or frame_id % 4 == 0:
                    try:
                        h, w = frame.shape[:2]
                        if is_recording:
                            # 录像模式：原画质，不缩放，JPEG quality 95
                            _, jpeg = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                            )
                            try:
                                frame_queue.put(jpeg.tobytes(), timeout=0.5)
                            except Exception:
                                pass  # 队列满超时丢帧
                        else:
                            # 预览模式：缩放到 960 宽，JPEG quality 75，只保留最新帧
                            preview = cv2.resize(frame, (960, int(h * 960 / w))) if w > 960 else frame
                            _, jpeg = cv2.imencode(
                                ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75]
                            )
                            while not frame_queue.empty():
                                try:
                                    frame_queue.get_nowait()
                                except Exception:
                                    break
                            frame_queue.put_nowait(jpeg.tobytes())
                    except Exception:
                        pass

            # Mask out timestamp overlay (top-left corner) to avoid interfering with detection
            frame[0:41, 0:603] = 0

            frame_buffer.append(frame)
            frame_id_buffer.append(frame_id)
            capture_ts_buffer.append(capture_ts)

            if len(frame_buffer) < frames_in:
                continue

            # 推理开关关闭或模型未加载时直接跳过 GPU 调用
            if detector is None or not status_dict.get("inference_enabled", True):
                frame_buffer.clear()
                frame_id_buffer.clear()
                capture_ts_buffer.clear()
                continue

            # Inference on the buffer
            try:
                heatmaps = detector.infer(frame_buffer)
            except Exception as e:
                log.error("Inference error: %s", e)
                frame_buffer.clear()
                frame_id_buffer.clear()
                continue

            # Post-process each output frame — extract top-K blobs (matches offline)
            for i in range(min(frames_out, len(heatmaps))):
                blobs = tracker.process_heatmap_multi(heatmaps[i], max_blobs=2)
                if not blobs:
                    continue

                # Top-1 blob as primary detection
                top = blobs[0]
                px, py, conf = top["pixel_x"], top["pixel_y"], top["blob_sum"]
                wx, wy = homography.pixel_to_world(px, py)

                if not (homography.court_x_min <= wx <= homography.court_x_max):
                    continue

                # Build candidates list with world coords for MultiBlobMatcher
                candidates = []
                for b in blobs:
                    bwx, bwy = homography.pixel_to_world(b["pixel_x"], b["pixel_y"])
                    candidates.append({
                        "x": bwx, "y": bwy,
                        "world_x": bwx, "world_y": bwy,
                        "pixel_x": b["pixel_x"], "pixel_y": b["pixel_y"],
                        "blob_sum": b["blob_sum"],
                    })

                detection = {
                    "camera_name": name,
                    "x": wx,
                    "y": wy,
                    "pixel_x": px,
                    "pixel_y": py,
                    "confidence": conf,
                    "blob_sum": conf,
                    "timestamp": time.time(),
                    "capture_ts": capture_ts_buffer[0],
                    "candidates": candidates,  # top-K for MultiBlobMatcher
                }

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
            frame_id_buffer.clear()
            capture_ts_buffer.clear()

    except Exception as e:
        log.exception("Pipeline crashed: %s", e)
        status_dict["state"] = "error"
        status_dict["error_msg"] = str(e)
    finally:
        if stream is not None:
            stream.stop()
        status_dict["state"] = status_dict.get("state", "stopped")
        if status_dict["state"] == "running":
            status_dict["state"] = "stopped"
        log.info("Pipeline exited (state=%s)", status_dict["state"])
