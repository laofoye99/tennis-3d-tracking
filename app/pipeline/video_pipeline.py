"""Video file processing pipeline: read video → inference → postprocess → homography → output queue."""

import logging
import multiprocessing as mp
import time
from typing import Any, Optional

import cv2

from app.pipeline.homography import HomographyTransformer
from app.pipeline.inference import BallDetector
from app.pipeline.postprocess import BallTracker

logger = logging.getLogger(__name__)


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

        # Initialize pipeline components
        detector = BallDetector(model_path, input_size, frames_in, frames_out, device)
        tracker = BallTracker(original_size=(vid_w, vid_h), threshold=threshold)
        homography = HomographyTransformer(homography_path, homography_key)

        status_dict["state"] = "running"
        status_dict["total_frames"] = total_frames
        status_dict["processed_frames"] = 0
        status_dict["fps"] = 0.0

        frame_buffer: list = []
        processed_count = 0
        fps_counter = 0
        fps_time = time.time()
        # Send preview frame at roughly 15 fps
        preview_interval = max(1, int(fps / 15))
        # Track latest detection for drawing on preview frames
        last_pixel_detection: tuple[float, float, float] | None = None
        preview_scale = 960.0 / vid_w if vid_w > 960 else 1.0

        while processed_count < total_frames and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            processed_count += 1
            status_dict["processed_frames"] = processed_count

            # Send preview frame periodically
            if frame_queue is not None and processed_count % preview_interval == 0:
                try:
                    h, w = frame.shape[:2]
                    if w > 960:
                        preview = cv2.resize(frame, (960, int(h * preview_scale)))
                    else:
                        preview = frame.copy()

                    # Draw ball marker from latest detection
                    if last_pixel_detection is not None:
                        dpx, dpy, _dconf = last_pixel_detection
                        draw_x = int(dpx * preview_scale)
                        draw_y = int(dpy * preview_scale)
                        cv2.circle(preview, (draw_x, draw_y), 14, (0, 255, 0), 2)
                        cv2.circle(preview, (draw_x, draw_y), 4, (0, 255, 0), -1)

                    _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    # Only keep latest frame
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
            if len(frame_buffer) < frames_in:
                continue

            # Inference
            try:
                heatmaps = detector.infer(frame_buffer)
            except Exception as e:
                log.error("Inference error: %s", e)
                frame_buffer.clear()
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
                # Batch N processes frames (N-1)*frames_in+1 .. N*frames_in
                # Output i corresponds to frame processed_count - frames_out + i + 1
                fi = processed_count - frames_out + i + 1

                detection = {
                    "camera_name": camera_name,
                    "x": wx,
                    "y": wy,
                    "pixel_x": px,
                    "pixel_y": py,
                    "confidence": conf,
                    "timestamp": time.time(),
                    "frame_index": fi,  # relative to clip start for cross-camera matching
                }

                try:
                    result_queue.put_nowait(detection)
                except Exception:
                    pass

            fps_counter += frames_out
            now = time.time()
            if now - fps_time >= 1.0:
                status_dict["fps"] = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            frame_buffer.clear()

        cap.release()
        status_dict["state"] = "completed"
        status_dict["processed_frames"] = processed_count
        log.info("Video pipeline completed: %d frames processed", processed_count)

    except Exception as e:
        log.exception("Video pipeline crashed: %s", e)
        status_dict["state"] = "error"
        status_dict["error_msg"] = str(e)
