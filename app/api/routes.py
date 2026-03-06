"""FastAPI routes: REST API + SSE stream + Web Dashboard."""

import asyncio
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
    p1_x: float | None = None,
    p1_y: float | None = None,
    p2_x: float | None = None,
    p2_y: float | None = None,
):
    """Return a JPEG frame at the given timestamp using OpenCV.
    Draws ball and players if coordinates provided.
    """
    fpath = _UPLOAD_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, f"File not found: {filename}")

    cap = cv2.VideoCapture(str(fpath))
    if not cap.isOpened():
        raise HTTPException(500, "Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(400, "Cannot read frame at given time")

    # Draw player 1 (blue)
    if p1_x is not None and p1_y is not None:
        cx, cy = int(round(p1_x)), int(round(p1_y))
        cv2.drawMarker(frame, (cx, cy), (250, 136, 79), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.putText(frame, "P1", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 136, 79), 2)

    # Draw player 2 (red)
    if p2_x is not None and p2_y is not None:
        cx, cy = int(round(p2_x)), int(round(p2_y))
        cv2.drawMarker(frame, (cx, cy), (78, 78, 250), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.putText(frame, "P2", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (78, 78, 250), 2)

    # Draw ball marker (green)
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
