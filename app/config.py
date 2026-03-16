"""Global configuration loader."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class CameraConfig(BaseModel):
    rtsp_url: str
    position_3d: list[float]
    homography_key: str


class ModelConfig(BaseModel):
    path: str
    input_size: list[int]
    frames_in: int
    frames_out: int
    threshold: float
    device: str
    heatmap_mask: list[list[int]] = []


class ServerConfig(BaseModel):
    host: str
    port: int


class HomographyConfig(BaseModel):
    path: str


class CalibrationConfig(BaseModel):
    path: str = "src/camera_calibration.json"
    use_calibrated_positions: bool = False


class EnsembleConfig(BaseModel):
    enabled: bool = False
    hrnet_path: str = "model_weight/hrnet_tennis.onnx"
    hrnet_frames_in: int = 3
    hrnet_frames_out: int = 3
    agree_distance: float = 3.0
    boost_factor: float = 1.2
    penalty_factor: float = 0.6
    single_factor: float = 0.8


class BlobVerifierConfig(BaseModel):
    enabled: bool = False
    model_path: str = "yolo11n.pt"
    crop_size: int = 128
    conf: float = 0.25


class AppConfig(BaseModel):
    cameras: dict[str, CameraConfig]
    model: ModelConfig
    homography: HomographyConfig
    server: ServerConfig
    calibration: CalibrationConfig = CalibrationConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    blob_verifier: BlobVerifierConfig = BlobVerifierConfig()


def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig(**raw)
