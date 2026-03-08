"""Video file processing pipeline: read video → inference → postprocess → output.

Uses a prefetch thread to overlap frame reading / preprocessing with GPU
inference so the GPU never waits for the CPU.
"""

import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Optional

import cv2

from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import create_detector
from app.pipeline.postprocess import BallTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame prefetch thread
# ---------------------------------------------------------------------------

def _prefetch_thread(
    cap: cv2.VideoCapture,
    total_frames: int,
    frames_in: int,
    detector,
    stop_event: mp.Event,
    batch_queue: queue.Queue,
    status_dict: dict[str, Any],
) -> None:
    """Read frames, preprocess, and push ready batches into batch_queue.

    Each item placed on the queue is:
        (batch_frame_count, preprocessed_list, preview_frame_or_None)
    where batch_frame_count is the cumulative processed_count after this batch.
    """
    frame_buffer: list = []
    raw_buffer: list = []  # original frames for preview
    processed_count = 0

    while processed_count < total_frames and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        processed_count += 1
        status_dict["processed_frames"] = processed_count

        raw_buffer.append(frame)

        # Mask timestamp overlay
        masked = frame.copy()
        masked[0:41, 0:603] = 0

        frame_buffer.append(masked)
        if len(frame_buffer) < frames_in:
            continue

        # Push complete batch
        preview = raw_buffer[-1].copy()
        try:
            batch_queue.put((processed_count, frame_buffer.copy(), preview), timeout=5.0)
        except queue.Full:
            pass
        frame_buffer.clear()
        raw_buffer.clear()

    # Sentinel
    batch_queue.put(None)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_video_pipeline(
    video_path: str,
    start_time: float,
    end_time: float,
    camera_name: str,
    model_path: str,
    input_size: tuple[int, int],
    frames_in: int,
    frames_out: int,
    threshold: float,
    device: str,
    homography_path: str,
    homography_key: str,
    result_queue: mp.Queue,
    frame_queue: Optional[mp.Queue],
    stop_event: mp.Event,
    status_dict: dict[str, Any],
) -> None:
    """Process a video file segment through the ball detection pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [video-test] %(levelname)s %(message)s",
    )
    log = logging.getLogger("video-test")
    log.info("Video pipeline starting: %s [%.1f - %.1f]", video_path, start_time, end_time)

    status_dict["state"] = "starting"
    status_dict["error_msg"] = ""

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if end_frame <= 0 or end_frame > total_file_frames:
            end_frame = total_file_frames
        total_frames = end_frame - start_frame

        log.info(
            "Video: %dx%d @ %.1f fps, frames %d-%d (%d total)",
            vid_w, vid_h, fps, start_frame, end_frame, total_frames,
        )

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Initialize pipeline components (auto-selects ONNX or PyTorch backend)
        detector = create_detector(model_path, input_size, frames_in, frames_out, device)

        # Compute background median for TrackNet (author's approach)
        if hasattr(detector, "compute_video_median"):
            log.info("Computing video median for TrackNet background...")
            detector.compute_video_median(cap, start_frame, end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Reset to start

        tracker = BallTracker(original_size=(vid_w, vid_h), threshold=threshold)
        homography = HomographyTransformer(homography_path, homography_key)

        status_dict["state"] = "running"
        status_dict["total_frames"] = total_frames
        status_dict["processed_frames"] = 0
        status_dict["fps"] = 0.0

        fps_counter = 0
        fps_time = time.time()
        last_pixel_detection: tuple[float, float, float] | None = None
        preview_scale = 960.0 / vid_w if vid_w > 960 else 1.0

        # Start prefetch thread — reads frames while GPU runs inference
        batch_q: queue.Queue = queue.Queue(maxsize=2)
        reader = threading.Thread(
            target=_prefetch_thread,
            args=(cap, total_frames, frames_in, detector, stop_event, batch_q, status_dict),
            daemon=True,
        )
        reader.start()

        while not stop_event.is_set():
            item = batch_q.get()
            if item is None:
                break  # sentinel — reader finished

            processed_count, frame_buffer, preview_frame = item

            # Inference
            try:
                heatmaps = detector.infer(frame_buffer)
            except Exception as e:
                log.error("Inference error: %s", e)
                continue

            # Post-process each output frame
            for i in range(min(frames_out, len(heatmaps))):
                result = tracker.process_heatmap(heatmaps[i])
                if result is None:
                    continue

                px, py, conf = result
                last_pixel_detection = (px, py, conf)
                wx, wy = homography.pixel_to_world(px, py)

                # Unique frame index: each output corresponds to an input frame
                fi = processed_count - frames_out + i + 1

                detection = {
                    "camera_name": camera_name,
                    "x": wx,
                    "y": wy,
                    "pixel_x": px,
                    "pixel_y": py,
                    "confidence": conf,
                    "timestamp": time.time(),
                    "frame_index": fi,
                }

                try:
                    result_queue.put_nowait(detection)
                except Exception:
                    pass

            # Send preview AFTER inference so detection overlay matches
            if frame_queue is not None:
                try:
                    h, w = preview_frame.shape[:2]
                    if w > 960:
                        preview = cv2.resize(preview_frame, (960, int(h * preview_scale)))
                    else:
                        preview = preview_frame
                    if last_pixel_detection is not None:
                        dpx, dpy, _dconf = last_pixel_detection
                        draw_x = int(dpx * preview_scale)
                        draw_y = int(dpy * preview_scale)
                        cv2.circle(preview, (draw_x, draw_y), 14, (0, 255, 0), 2)
                        cv2.circle(preview, (draw_x, draw_y), 4, (0, 255, 0), -1)
                    _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    while not frame_queue.empty():
                        try:
                            frame_queue.get_nowait()
                        except Exception:
                            break
                    frame_queue.put_nowait(jpeg.tobytes())
                except Exception:
                    pass

            fps_counter += frames_out
            now = time.time()
            if now - fps_time >= 1.0:
                status_dict["fps"] = round(fps_counter / (now - fps_time), 1)
                fps_counter = 0
                fps_time = now

        reader.join(timeout=5.0)
        cap.release()
        status_dict["state"] = "completed"
        status_dict["processed_frames"] = status_dict.get("processed_frames", 0)
        log.info("Video pipeline completed: %d frames processed", status_dict["processed_frames"])

    except Exception as e:
        log.exception("Video pipeline crashed: %s", e)
        status_dict["state"] = "error"
        status_dict["error_msg"] = str(e)
