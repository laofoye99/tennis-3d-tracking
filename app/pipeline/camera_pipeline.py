"""Camera pipeline subprocess: stream → inference → postprocess → homography → output queue.

Architecture:
  - Main loop: reads frames from RTSP, sends JPEG to preview queue (always smooth)
  - Inference thread: consumes frames from an internal queue, runs TrackNet + postprocess
  This ensures video preview is never blocked by GPU inference.
"""

import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
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

        # Internal queue for passing frames to inference thread
        infer_q: queue.Queue = queue.Queue(maxsize=2)

        # --- Inference thread (runs in background, never blocks preview) ---
        def inference_loop():
            frame_buffer: list = []
            fps_counter = 0
            fps_time = time.time()
            infer_count = 0
            det_count = 0

            while not stop_event.is_set():
                try:
                    det_frame = infer_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_buffer.append(det_frame)
                if len(frame_buffer) < frames_in:
                    log.debug("Buffer: %d/%d frames", len(frame_buffer), frames_in)
                    continue

                # Run inference
                try:
                    heatmaps = detector.infer(frame_buffer)
                    infer_count += 1
                    if infer_count <= 3 or infer_count % 50 == 0:
                        log.info("Inference #%d: got %d heatmaps from %d frames",
                                 infer_count, len(heatmaps), len(frame_buffer))
                except Exception as e:
                    log.error("Inference error: %s", e)
                    frame_buffer.clear()
                    continue

                # Post-process
                for i in range(min(frames_out, len(heatmaps))):
                    result = tracker.process_heatmap(heatmaps[i])
                    if result is None:
                        continue

                    det_count += 1
                    px, py, conf = result
                    wx, wy = homography.pixel_to_world(px, py)
                    log.info("Detection #%d: px=(%.0f,%.0f) conf=%.1f world=(%.2f,%.2f)",
                             det_count, px, py, conf, wx, wy)

                    if not (homography.court_x_min <= wx <= homography.court_x_max):
                        continue

                    detection = {
                        "camera_name": name,
                        "x": wx,
                        "y": wy,
                        "pixel_x": px,
                        "pixel_y": py,
                        "confidence": conf,
                        "timestamp": time.time(),
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

        # Start inference thread if model is loaded
        if detector is not None:
            infer_thread = threading.Thread(target=inference_loop, daemon=True)
            infer_thread.start()
            log.info("Inference thread started")

        # --- Main loop: frame reading + preview (never blocks) ---
        last_frame_id = -1
        frame_count = 0
        infer_sent = 0

        while not stop_event.is_set():
            frame, frame_id, ts = stream.read()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)
                continue
            last_frame_id = frame_id
            frame_count += 1
            if frame_count <= 3 or frame_count % 100 == 0:
                log.info("Frame %d (id=%d), shape=%s, infer_queue=%d, sent_to_infer=%d",
                         frame_count, frame_id, frame.shape, infer_q.qsize(), infer_sent)

            # 预览/录像：始终执行，不受推理影响
            if frame_queue is not None:
                is_recording = status_dict.get("recording_enabled", False)
                if is_recording or frame_id % 4 == 0:
                    try:
                        h, w = frame.shape[:2]
                        if is_recording:
                            _, jpeg = cv2.imencode(
                                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                            )
                            try:
                                frame_queue.put(jpeg.tobytes(), timeout=0.5)
                            except Exception:
                                pass
                        else:
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

            # 推理：把帧送到推理线程（非阻塞，满了就丢）
            if detector is not None and status_dict.get("inference_enabled", True):
                det_frame = frame.copy()
                det_frame[0:41, 0:603] = 0
                try:
                    infer_q.put_nowait(det_frame)
                    infer_sent += 1
                except queue.Full:
                    pass  # 推理跟不上就丢帧，不影响预览
            elif detector is None:
                if frame_count <= 3:
                    log.warning("Detector is None, inference disabled")
            elif not status_dict.get("inference_enabled", True):
                if frame_count <= 3:
                    log.warning("Inference disabled by toggle")

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
