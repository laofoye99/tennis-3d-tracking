"""FastAPI routes: REST API + SSE stream + Web Dashboard."""

import asyncio
import datetime
import json
import logging
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

# The orchestrator instance is injected via app.state at startup.
_orch: Orchestrator | None = None
_UPLOAD_DIR = Path("uploads")
_ANNOTATIONS_DIR = Path("annotations")
_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


def set_orchestrator(orch: Orchestrator) -> None:
    global _orch
    _orch = orch


def _get_orch() -> Orchestrator:
    if _orch is None:
        raise HTTPException(503, "System not initialized")
    return _orch


# ---- Dashboard ----

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@router.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = _TEMPLATE_DIR / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---- System status ----

@router.get("/api/status")
async def system_status():
    orch = _get_orch()
    status = orch.get_system_status()
    return status.model_dump()


# ---- Ball 3D ----

@router.get("/api/ball3d")
async def ball_3d():
    orch = _get_orch()
    pos = orch.get_latest_3d()
    if pos is None:
        return {"detected": False}
    return {"detected": True, **pos.model_dump()}


@router.get("/api/ball3d/stream")
async def ball_3d_stream():
    """Server-Sent Events stream of 3D ball positions."""
    orch = _get_orch()

    async def event_generator():
        prev = None
        while True:
            pos = orch.get_latest_3d()
            if pos is not None and (prev is None or pos.timestamp != prev.timestamp):
                data = json.dumps(pos.model_dump())
                yield f"data: {data}\n\n"
                prev = pos
            await asyncio.sleep(0.03)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---- Live Analytics ----

@router.get("/api/analytics/live")
async def live_analytics():
    """Get current live bounce detection and rally tracking state."""
    orch = _get_orch()
    return orch.get_live_analytics()


@router.post("/api/analytics/reset")
async def reset_analytics():
    """Reset live analytics state."""
    orch = _get_orch()
    orch.reset_live_analytics()
    return {"status": "ok"}


# ---- Net Crossing Speed ----

@router.get("/api/net-crossing")
async def net_crossing():
    """Get the latest net crossing event with ball speed."""
    orch = _get_orch()
    crossing = orch.get_latest_net_crossing()
    if crossing is None:
        return {"detected": False}
    return {"detected": True, **crossing}


@router.get("/api/net-crossings")
async def net_crossings():
    """Get recent net crossing events."""
    orch = _get_orch()
    return {"crossings": orch.get_net_crossings()}


# ---- 3D Display (WebSocket Push) ----

@router.post("/api/3d-display/enable")
async def enable_3d_display(url: str = None):
    """Enable real-time bounce push to 3D display via WebSocket."""
    orch = _get_orch()
    return orch.enable_3d_display(url)


@router.post("/api/3d-display/disable")
async def disable_3d_display():
    """Disable 3D display push."""
    orch = _get_orch()
    return orch.disable_3d_display()


# ---- ML Rally Segmentation ----

@router.post("/api/ml-rally/enable")
async def enable_ml_rally():
    """Enable ML-based rally segmentation filter."""
    orch = _get_orch()
    return orch.enable_ml_rally()


@router.post("/api/ml-rally/disable")
async def disable_ml_rally():
    """Disable ML rally filter."""
    orch = _get_orch()
    return orch.disable_ml_rally()


@router.get("/api/ml-rally/status")
async def ml_rally_status():
    """Get ML rally filter status."""
    orch = _get_orch()
    return orch.get_ml_rally_status()


# ---- Camera Calibration ----

@router.post("/api/calibration/run")
async def run_calibration_api():
    """Run camera calibration from court keypoints.

    Estimates intrinsic/extrinsic parameters using PnP, saves results
    to ``src/camera_calibration.json``, and updates the homography
    matrices with calibration-derived values.
    """
    try:
        from app.calibration import run_calibration

        result = run_calibration()
        return {
            "status": "ok",
            "cam66": {
                "reproj_error_px": result["cam66"]["mean_reprojection_error_px"],
                "reproj_error_m": result["cam66"]["mean_reprojection_error_m"],
                "camera_position": result["cam66"]["camera_position_3d"],
            },
            "cam68": {
                "reproj_error_px": result["cam68"]["mean_reprojection_error_px"],
                "reproj_error_m": result["cam68"]["mean_reprojection_error_m"],
                "camera_position": result["cam68"]["camera_position_3d"],
            },
            "baseline_m": result["stereo"]["baseline_m"],
        }
    except Exception as e:
        logger.exception("Calibration failed")
        raise HTTPException(500, str(e))


@router.get("/api/calibration/status")
async def calibration_status():
    """Get current calibration data if available."""
    cal_path = Path("src/camera_calibration.json")
    if not cal_path.exists():
        return {"calibrated": False}
    try:
        with open(cal_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "calibrated": True,
            "cam66": {
                "reproj_error_m": data.get("cam66", {}).get("mean_reprojection_error_m"),
                "camera_position": data.get("cam66", {}).get("camera_position_3d"),
            },
            "cam68": {
                "reproj_error_m": data.get("cam68", {}).get("mean_reprojection_error_m"),
                "camera_position": data.get("cam68", {}).get("camera_position_3d"),
            },
            "baseline_m": data.get("stereo", {}).get("baseline_m"),
        }
    except Exception as e:
        return {"calibrated": False, "error": str(e)}


@router.post("/api/calibration/run-from-points")
async def run_calibration_from_points(request: Request):
    """Run calibration using manually marked point pairs from the UI.

    Expects JSON body:
    {
        "points": [{"cam66": [px, py], "cam68": [px, py]}, ...],
        "cam66_video": "filename.mp4",
        "cam68_video": "filename.mp4",
        "frame_time": 5.0
    }

    Uses the point correspondences (assumed ground-level, z=0) to run
    PnP camera pose estimation for each camera.
    """
    try:
        body = await request.json()
        points = body.get("points", [])
        vid1 = body.get("cam66_video", "")
        vid2 = body.get("cam68_video", "")
        frame_time = float(body.get("frame_time", 1.0))

        if len(points) < 4:
            return {"error": "Need at least 4 point pairs"}

        from app.calibration import calibrate_from_point_pairs

        result = calibrate_from_point_pairs(
            point_pairs=points,
            cam66_video=str(_UPLOAD_DIR / vid1),
            cam68_video=str(_UPLOAD_DIR / vid2),
            frame_time=frame_time,
        )
        return result
    except ImportError:
        return {"error": "calibrate_from_point_pairs not available in app.calibration"}
    except Exception as e:
        logger.exception("Point-based calibration failed")
        return {"error": str(e)}


@router.post("/api/calibration/apply")
async def apply_calibration(request: Request):
    """Apply calibrated camera positions to config.yaml.

    Expects JSON body:
    {
        "cam66_position": [x, y, z],
        "cam68_position": [x, y, z]
    }

    Updates config.yaml camera positions and sets use_calibrated_positions: true.
    """
    import yaml

    try:
        body = await request.json()
        pos1 = body.get("cam66_position")
        pos2 = body.get("cam68_position")
        if not pos1 or not pos2:
            return {"error": "Both cam66_position and cam68_position required"}

        config_path = Path("config.yaml")
        if not config_path.exists():
            return {"error": "config.yaml not found"}

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Update camera positions
        if "cameras" not in config:
            config["cameras"] = {}
        if "cam66" not in config["cameras"]:
            config["cameras"]["cam66"] = {}
        if "cam68" not in config["cameras"]:
            config["cameras"]["cam68"] = {}

        config["cameras"]["cam66"]["position_3d"] = [float(v) for v in pos1]
        config["cameras"]["cam68"]["position_3d"] = [float(v) for v in pos2]

        # Enable calibrated positions
        if "calibration" not in config:
            config["calibration"] = {}
        config["calibration"]["use_calibrated_positions"] = True

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        return {"status": "ok", "message": "Config updated with calibrated positions"}
    except Exception as e:
        logger.exception("Apply calibration failed")
        return {"error": str(e)}


# ---- Pipeline control ----

@router.post("/api/pipeline/{name}/start")
async def start_pipeline(name: str):
    orch = _get_orch()
    try:
        orch.start_pipeline(name)
        return {"status": "ok", "message": f"{name} starting"}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/pipeline/{name}/stop")
async def stop_pipeline(name: str):
    orch = _get_orch()
    try:
        orch.stop_pipeline(name)
        return {"status": "ok", "message": f"{name} stopped"}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/pipeline/{name}/detection")
async def pipeline_detection(name: str):
    orch = _get_orch()
    det = orch.get_latest_detection(name)
    if det is None:
        return {"detected": False}
    return {"detected": True, **det}


# ---- Function toggles ----

@router.post("/api/function/track/{state}")
async def set_track_ball(state: str):
    """控制 track ball 推理开关：state = 'on' | 'off'"""
    orch = _get_orch()
    if state not in ("on", "off"):
        raise HTTPException(400, "state must be 'on' or 'off'")
    orch.set_inference_enabled(state == "on")
    return {"inference_enabled": orch.inference_enabled}


@router.post("/api/model/switch/{model_name}")
async def switch_model(model_name: str):
    """Switch detection model: model_name = 'hrnet' | 'tracknet'"""
    orch = _get_orch()
    try:
        result = orch.switch_model(model_name)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/api/model/current")
async def current_model():
    """Get current model info."""
    orch = _get_orch()
    return orch.get_current_model()


# ---- Recording ----

@router.post("/api/recording/start")
async def recording_start():
    """开始录像（所有已运行的摄像头）。"""
    orch = _get_orch()
    result = orch.start_recording()
    return result


@router.post("/api/recording/stop")
async def recording_stop():
    """停止录像并写入 MP4 文件。"""
    orch = _get_orch()
    result = orch.stop_recording()
    return result


@router.get("/api/recording/status")
async def recording_status():
    """查询当前录像状态。"""
    orch = _get_orch()
    return orch.get_recording_status()


# ---- Camera MJPEG stream ----

@router.get("/api/camera/{name}/stream")
async def camera_mjpeg_stream(name: str):
    """将摄像头最新帧以 MJPEG multipart 格式持续推送给浏览器。"""
    orch = _get_orch()

    async def frame_generator():
        _BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        prev: bytes | None = None
        while True:
            jpeg = orch.get_latest_frame(name)
            if jpeg is not None and jpeg is not prev:
                prev = jpeg
                yield _BOUNDARY + jpeg + b"\r\n"
            await asyncio.sleep(1 / 25)  # 最高 25 fps

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---- Video Upload / Management ----

@router.post("/api/upload-video")
async def upload_video(files: list[UploadFile] = File(...)):
    """Upload one or more video files."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if not f.filename:
            continue
        safe_name = f.filename.replace(" ", "_")
        dest = _UPLOAD_DIR / safe_name
        # Handle duplicate names
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 1
            while dest.exists():
                dest = _UPLOAD_DIR / f"{stem}_{i}{suffix}"
                i += 1
        content = await f.read()
        dest.write_bytes(content)
        saved.append(dest.name)
    return {"status": "ok", "files": saved}


@router.get("/api/uploaded-videos")
async def list_uploaded_videos():
    """List all uploaded videos with metadata."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    videos = []
    for f in sorted(_UPLOAD_DIR.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _VIDEO_EXTS:
            continue
        info: dict = {"filename": f.name, "size_mb": round(f.stat().st_size / 1024 / 1024, 1)}
        try:
            cap = cv2.VideoCapture(str(f))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if fps > 0:
                info["duration"] = round(frames / fps, 2)
                info["fps"] = round(fps, 1)
            info["width"] = w
            info["height"] = h
        except Exception:
            pass
        videos.append(info)
    return {"videos": videos}


@router.delete("/api/uploaded-video/{filename}")
async def delete_uploaded_video(filename: str):
    """Delete an uploaded video file."""
    fpath = _UPLOAD_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, f"File not found: {filename}")
    if not fpath.resolve().parent == _UPLOAD_DIR.resolve():
        raise HTTPException(400, "Invalid filename")
    fpath.unlink()
    return {"status": "ok"}


# ---- Video Frame Preview (codec-agnostic) ----

@router.get("/api/video-preview/frame")
async def video_preview_frame(
    filename: str,
    time: float = 0,
    pixel_x: float | None = None,
    pixel_y: float | None = None,
):
    """Return a JPEG frame at the given timestamp using OpenCV (supports H.265).

    Optionally draws a ball marker at (pixel_x, pixel_y) if both are provided.
    """
    fpath = _UPLOAD_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, f"File not found: {filename}")
    if not fpath.resolve().parent == _UPLOAD_DIR.resolve():
        raise HTTPException(400, "Invalid filename")

    cap = cv2.VideoCapture(str(fpath))
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(400, "Cannot read frame at given time")

    # Draw ball marker if coordinates provided
    if pixel_x is not None and pixel_y is not None:
        cx, cy = int(round(pixel_x)), int(round(pixel_y))
        cv2.circle(frame, (cx, cy), 14, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(
        iter([buf.tobytes()]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


# ---- Video Test ----

@router.post("/api/video-test/run")
async def run_video_test(request: Request):
    """Start processing a video segment through the detection pipeline."""
    orch = _get_orch()
    body = await request.json()
    filename = body.get("filename")
    start_time = float(body.get("start_time", 0))
    end_time = float(body.get("end_time", 0))
    camera = body.get("camera", "cam66")

    if not filename:
        raise HTTPException(400, "filename is required")
    video_path = _UPLOAD_DIR / filename
    if not video_path.exists():
        raise HTTPException(404, f"Video not found: {filename}")

    try:
        result = orch.start_video_test(
            video_path=str(video_path),
            start_time=start_time,
            end_time=end_time,
            camera_name=camera,
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/video-test/run-parallel")
async def run_video_test_parallel(request: Request):
    """Start processing multiple camera videos in parallel."""
    orch = _get_orch()
    body = await request.json()
    cameras_raw = body.get("cameras", [])

    cam_list = []
    for cam in cameras_raw:
        filename = cam.get("filename")
        if not filename:
            raise HTTPException(400, "Each camera entry requires a filename")
        video_path = _UPLOAD_DIR / filename
        if not video_path.exists():
            raise HTTPException(404, f"Video not found: {filename}")
        cam_list.append(
            {
                "camera_name": cam.get("camera", "cam66"),
                "video_path": str(video_path),
                "start_time": float(cam.get("start_time", 0)),
                "end_time": float(cam.get("end_time", 0)),
            }
        )

    try:
        result = orch.start_video_test_parallel(cam_list)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/video-test/stop")
async def stop_video_test():
    """Stop video test pipeline."""
    orch = _get_orch()
    return orch.stop_video_test()


@router.get("/api/video-test/status")
async def video_test_status():
    """Get video test pipeline status."""
    orch = _get_orch()
    return orch.get_video_test_status()


@router.get("/api/video-test/detections")
async def video_test_detections(camera: str | None = None):
    """Return detection results, optionally filtered by camera name."""
    orch = _get_orch()
    detections = orch.get_video_test_detections(camera)
    return {"detections": detections, "count": len(detections)}


@router.post("/api/video-test/clear-detections")
async def clear_video_test_detections(request: Request):
    """Clear stored video test detections."""
    orch = _get_orch()
    try:
        body = await request.json()
        camera = body.get("camera")
    except Exception:
        camera = None
    orch.clear_video_test_detections(camera)
    return {"status": "ok"}


@router.get("/api/video-test/detections/stream")
async def video_test_detections_stream():
    """SSE stream of new detections during video test processing.

    Sends incremental detection batches at ~10Hz until processing completes.
    Each message is JSON with camera_name -> [new detections] or {done: true}.
    """
    orch = _get_orch()

    async def generator():
        cursors: dict[str, int] = {}
        while True:
            new_dets = orch.get_video_test_detections_since(cursors)
            if new_dets:
                for cam, dets in new_dets.items():
                    cursors[cam] = cursors.get(cam, 0) + len(dets)
                yield f"data: {json.dumps(new_dets)}\n\n"
            # Check if processing is done
            status = orch.get_video_test_status()
            state = status.get("state", "idle")
            if state in ("idle", "error", "completed"):
                # Send final batch + done signal
                final_dets = orch.get_video_test_detections_since(cursors)
                if final_dets:
                    for cam, dets in final_dets.items():
                        cursors[cam] = cursors.get(cam, 0) + len(dets)
                    yield f"data: {json.dumps(final_dets)}\n\n"
                yield f"data: {json.dumps({'done': True, 'state': state})}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(generator(), media_type="text/event-stream")


@router.post("/api/video-test/compute-3d")
async def compute_3d():
    """Compute 3D positions from two cameras' detections via triangulation."""
    orch = _get_orch()
    result = orch.compute_3d_from_detections()
    points = result.get("points", [])
    return {
        "points": points,
        "count": len(points),
        "stats": result.get("stats", {}),
        "cam_order": result.get("cam_order", []),
    }


@router.post("/api/video-test/compute-trajectory")
async def compute_trajectory():
    """Compute physics-constrained 3D trajectory (no frame sync needed).

    Uses auto time-offset + spatial parabolic fit to reconstruct the ball
    trajectory without requiring frame-level synchronization between cameras.
    Returns raw triangulated points, fitted trajectory, and smooth curve.
    """
    orch = _get_orch()
    result = orch.compute_3d_trajectory()
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


# ---- CVAT XML Export ----

@router.get("/api/video-test/export-cvat")
async def export_cvat(camera: str, filename: str):
    """Export single-camera detections as CVAT for Video 1.1 XML.

    Downloads an XML file that can be imported into CVAT.

    Query params:
        camera: camera name (e.g. "cam66")
        filename: original video filename in uploads/
    """
    orch = _get_orch()
    video_path = str(_UPLOAD_DIR / filename)
    if not Path(video_path).exists():
        raise HTTPException(404, f"Video not found: {filename}")
    try:
        xml_str = orch.export_cvat_xml(camera, video_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    stem = Path(filename).stem
    out_name = f"annotations_{camera}_{stem}.xml"
    return StreamingResponse(
        iter([xml_str.encode("utf-8")]),
        media_type="application/xml",
        headers={
            "Content-Disposition": f'attachment; filename="{out_name}"',
        },
    )


# ---------------------------------------------------------------------------
# Annotation save / list / load
# ---------------------------------------------------------------------------


@router.post("/api/annotations/save")
async def save_annotation(request: Request):
    """Save edited annotation data to a JSON file in annotations/ directory."""
    body = await request.json()
    _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cam1 = body.get("metadata", {}).get("cam1_video", "unknown")
    stem = Path(cam1).stem
    filename = f"annotation_{stem}_{ts}.json"

    filepath = _ANNOTATIONS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)

    return {"status": "ok", "filename": filename, "path": str(filepath)}


@router.get("/api/annotations/list")
async def list_annotations():
    """List all saved annotation files with basic metadata."""
    _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    annotations = []
    for fpath in sorted(_ANNOTATIONS_DIR.iterdir(), reverse=True):
        if fpath.suffix != ".json":
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            meta = data.get("metadata", {})
            annotations.append({
                "filename": fpath.name,
                "cam1_video": meta.get("cam1_video", ""),
                "cam2_video": meta.get("cam2_video", ""),
                "match_count": len(data.get("matches", [])),
                "bounce_count": len(data.get("bounces", [])),
                "rally_markers": len(data.get("rally_markers", [])),
                "saved_at": meta.get("saved_at", ""),
            })
        except Exception:
            annotations.append({"filename": fpath.name, "error": "parse failed"})
    return {"annotations": annotations}


@router.get("/api/annotations/load")
async def load_annotation(filename: str):
    """Load a specific annotation file by filename."""
    filepath = _ANNOTATIONS_DIR / filename
    if not filepath.exists():
        raise HTTPException(404, f"Annotation not found: {filename}")
    # Security: ensure file is inside annotations dir
    if not filepath.resolve().parent == _ANNOTATIONS_DIR.resolve():
        raise HTTPException(400, "Invalid filename")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ------------------------------------------------------------------
# Import LabelImg annotations → 3D
# ------------------------------------------------------------------

_UPLOADS_DIR = Path("uploads")


@router.get("/api/annotation-folders")
async def list_annotation_folders():
    """List subfolders in uploads/ that contain LabelImg JSON annotation files."""
    folders = []
    if _UPLOADS_DIR.is_dir():
        for sub in sorted(_UPLOADS_DIR.iterdir()):
            if sub.is_dir() and any(sub.glob("*.json")):
                folders.append(sub.name)
    return {"folders": folders}


@router.post("/api/import-annotations")
async def import_annotations(request: Request):
    """Import LabelImg annotations from two camera folders and compute 3D trajectory.

    Body: {"cam1_folder": "cam66_clip", "cam2_folder": "cam68_clip"}
    """
    body = await request.json()
    cam1_folder = body.get("cam1_folder", "")
    cam2_folder = body.get("cam2_folder", "")
    if not cam1_folder or not cam2_folder:
        raise HTTPException(400, "Both cam1_folder and cam2_folder are required")

    orch: Orchestrator = request.app.state.orchestrator
    result = orch.import_labelimg_annotations(cam1_folder, cam2_folder)

    if "error" in result:
        raise HTTPException(400, result["error"])

    return result
