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
    ensemble_config: Optional[dict] = None,
) -> None:
    """Process a video file segment through the ball detection pipeline.

    Args:
        ensemble_config: If provided and enabled, runs both TrackNet + HRNet
            with cross-validation. Dict with keys: enabled, hrnet_path,
            agree_distance, boost_factor, penalty_factor, single_factor.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [video-test] %(levelname)s %(message)s",
    )
    log = logging.getLogger("video-test")
    log.info("Video pipeline starting: %s [%.1f - %.1f]", video_path, start_time, end_time)

    status_dict["state"] = "starting"
    status_dict["error_msg"] = ""

    # Determine if ensemble mode is active
    use_ensemble = (
        ensemble_config is not None
        and ensemble_config.get("enabled", False)
    )

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
        ensemble_detector = None
        if use_ensemble:
            from app.pipeline.ensemble import EnsembleDetector
            log.info("Ensemble mode: TrackNet + HRNet cross-validation")
            ensemble_detector = EnsembleDetector(
                tracknet_path=model_path,
                hrnet_path=ensemble_config["hrnet_path"],
                input_size=input_size,
                device=device,
                threshold=threshold,
                original_size=(vid_w, vid_h),
                agree_distance=ensemble_config.get("agree_distance", 3.0),
                boost_factor=ensemble_config.get("boost_factor", 1.2),
                penalty_factor=ensemble_config.get("penalty_factor", 0.6),
                single_factor=ensemble_config.get("single_factor", 0.8),
            )
            detector = ensemble_detector.tracknet  # for prefetch compatibility
            actual_frames_in = 8  # Always 8 in ensemble mode (TrackNet batch size)
            actual_frames_out = 8

            # Compute background median for TrackNet
            log.info("Computing video median for TrackNet background...")
            ensemble_detector.compute_video_median(cap, start_frame, end_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            # Single model mode (auto-selects ONNX or PyTorch backend)
            detector = create_detector(model_path, input_size, frames_in, frames_out, device)
            actual_frames_in = frames_in
            actual_frames_out = frames_out

            # Compute background median for TrackNet (author's approach)
            if hasattr(detector, "compute_video_median"):
                log.info("Computing video median for TrackNet background...")
                detector.compute_video_median(cap, start_frame, end_frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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
            args=(cap, total_frames, actual_frames_in, detector, stop_event, batch_q, status_dict),
            daemon=True,
        )
        reader.start()

        while not stop_event.is_set():
            item = batch_q.get()
            if item is None:
                break  # sentinel — reader finished

            processed_count, frame_buffer, preview_frame = item

            if ensemble_detector is not None:
                # ---- Ensemble mode: both models + cross-validation ----
                try:
                    ensemble_results = ensemble_detector.infer_ensemble(frame_buffer)
                except Exception as e:
                    log.error("Ensemble inference error: %s", e)
                    continue

                for i, ens_result in enumerate(ensemble_results):
                    if ens_result is None:
                        continue
                    px, py, conf, source = ens_result
                    last_pixel_detection = (px, py, conf)
                    wx, wy = homography.pixel_to_world(px, py)
                    fi = start_frame + processed_count - actual_frames_out + i

                    detection = {
                        "camera_name": camera_name,
                        "x": wx,
                        "y": wy,
                        "pixel_x": px,
                        "pixel_y": py,
                        "confidence": conf,
                        "timestamp": time.time(),
                        "frame_index": fi,
                        "source": source,
                    }
                    try:
                        result_queue.put_nowait(detection)
                    except Exception:
                        pass
            else:
                # ---- Single model mode: multi-blob output ----
                try:
                    heatmaps = detector.infer(frame_buffer)
                except Exception as e:
                    log.error("Inference error: %s", e)
                    continue

                for i in range(min(actual_frames_out, len(heatmaps))):
                    blobs = tracker.process_heatmap_multi(heatmaps[i])
                    if not blobs:
                        continue

                    fi = start_frame + processed_count - actual_frames_out + i
                    # Build candidates with world coordinates
                    candidates = []
                    for blob in blobs:
                        wx, wy = homography.pixel_to_world(
                            blob["pixel_x"], blob["pixel_y"]
                        )
                        candidates.append({
                            "pixel_x": blob["pixel_x"],
                            "pixel_y": blob["pixel_y"],
                            "world_x": wx,
                            "world_y": wy,
                            "blob_sum": blob["blob_sum"],
                            "blob_max": blob["blob_max"],
                            "blob_area": blob["blob_area"],
                        })

                    # Top-1 for backward compatibility and preview overlay
                    top = candidates[0]
                    last_pixel_detection = (
                        top["pixel_x"], top["pixel_y"], top["blob_sum"]
                    )

                    detection = {
                        "camera_name": camera_name,
                        "x": top["world_x"],
                        "y": top["world_y"],
                        "pixel_x": top["pixel_x"],
                        "pixel_y": top["pixel_y"],
                        "confidence": top["blob_sum"],
                        "timestamp": time.time(),
                        "frame_index": fi,
                        "candidates": candidates,
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

            fps_counter += actual_frames_out
            now = time.time()
            if now - fps_time >= 1.0:
                status_dict["fps"] = round(fps_counter / (now - fps_time), 1)
                fps_counter = 0
                fps_time = now

        reader.join(timeout=5.0)
        cap.release()

        # Log ensemble statistics
        if ensemble_detector is not None:
            stats = ensemble_detector.stats.to_dict()
            log.info(
                "Ensemble stats: agree=%d (%.1f%%), disagree=%d, tn_only=%d, hr_only=%d, neither=%d",
                stats["agree"], stats["agree_rate"] * 100,
                stats["disagree"], stats["tracknet_only"],
                stats["hrnet_only"], stats["neither"],
            )
            status_dict["ensemble_stats"] = stats

        status_dict["state"] = "completed"
        status_dict["processed_frames"] = status_dict.get("processed_frames", 0)
        log.info("Video pipeline completed: %d frames processed", status_dict["processed_frames"])

    except Exception as e:
        log.exception("Video pipeline crashed: %s", e)
        status_dict["state"] = "error"
        status_dict["error_msg"] = str(e)
