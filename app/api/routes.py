"""FastAPI routes: REST API + SSE stream + Web Dashboard."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

# The orchestrator instance is injected via app.state at startup.
_orch: Orchestrator | None = None


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
