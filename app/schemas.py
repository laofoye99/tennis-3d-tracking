"""Data models for the tennis ball tracking system."""

import time
from typing import Optional

from pydantic import BaseModel, Field


class BallDetection2D(BaseModel):
    """Ball detection in image pixel coordinates."""

    camera_name: str
    frame_id: int
    pixel_x: float
    pixel_y: float
    confidence: float
    timestamp: float = Field(default_factory=time.time)


class WorldPoint2D(BaseModel):
    """Ball position in world coordinates (meters) on the court plane."""

    camera_name: str
    x: float
    y: float
    pixel_x: float
    pixel_y: float
    confidence: float
    timestamp: float = Field(default_factory=time.time)


class BallPosition3D(BaseModel):
    """3D ball position from triangulation."""

    x: float
    y: float
    z: float
    timestamp: float = Field(default_factory=time.time)
    cam66_world: Optional[WorldPoint2D] = None
    cam68_world: Optional[WorldPoint2D] = None


class PipelineStatus(BaseModel):
    """Status of a single camera pipeline."""

    name: str
    state: str = "stopped"  # running / stopped / error
    fps: float = 0.0
    last_detection_time: Optional[float] = None
    error_msg: Optional[str] = None


class SystemStatus(BaseModel):
    """Overall system status."""

    pipelines: dict[str, PipelineStatus] = {}
    triangulation_active: bool = False
    latest_ball_3d: Optional[BallPosition3D] = None
