"""Global configuration loader."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class CameraConfig(BaseModel):
    rtsp_url: str
    position_3d: list[float] = [0, 0, 0]
    homography_key: str = ""
    record_only: bool = False


class ModelConfig(BaseModel):
    path: str = ""  # not needed for median_bg
    input_size: list[int] = [288, 512]
    frames_in: int = 8
    frames_out: int = 8
    threshold: float = 0.3  # not needed for median_bg
    device: str = "cuda"  # not needed for median_bg
    heatmap_mask: list[list[int]] = []
    detector_type: str = "auto"  # "auto" | "tracknet" | "median_bg" | "ball_selector"


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


class PlayerDetectionConfig(BaseModel):
    enabled: bool = False
    model_path: str = "model_weight/yolo26x-pose.pt"
    device: str = "cuda"
    conf: float = 0.4
    run_every_n_frames: int = 1  # ~6 fps at 30 fps source


class ExportConfig(BaseModel):
    endpoint: str = "https://tennisync.top/api/admin/SpaceParties/reportData"


class HybridBounceConfig(BaseModel):
    z_max: float = 0.8
    min_seg_len: int = 8
    min_dense: int = 8
    dense_range: int = 12
    min_speed: float = 3.0
    max_gap_s: float = 0.6
    v_window: int = 8
    half_wins: list[int] = [4, 6, 8]
    cooldown_frames: int = 12


class BounceSmoothingConfig(BaseModel):
    sg_window: int = 11
    sg_poly: int = 3
    max_frame_gap: int = 3
    max_gap_s: float = 0.6


class BounceDetectionConfig(BaseModel):
    hybrid: HybridBounceConfig = HybridBounceConfig()
    smoothing: BounceSmoothingConfig = BounceSmoothingConfig()


class HitBounceRefinerConfig(BaseModel):
    enabled: bool = True
    show_hits_on_minimap: bool = True
    lookback_frames: int = 50
    release_delay_frames: int = 50
    hit_suppression_frames: int = 3
    hit_angle_thresh: float = 45.0
    top_hit_dist_px: float = 50.0
    bottom_hit_dist_px_net: float = 100.0
    bottom_hit_dist_px_base: float = 250.0
    top_hit_dist_m: float = 1.2
    bottom_hit_dist_m_net: float = 1.2
    bottom_hit_dist_m_base: float = 2.5
    clean_time_frames: int = 25
    clean_space_meters: float = 1.5
    history_frames: int = 150


class AppConfig(BaseModel):
    cameras: dict[str, CameraConfig]
    model: ModelConfig
    homography: HomographyConfig
    server: ServerConfig
    calibration: CalibrationConfig = CalibrationConfig()
    ensemble: EnsembleConfig = EnsembleConfig()
    blob_verifier: BlobVerifierConfig = BlobVerifierConfig()
    player_detection: PlayerDetectionConfig = PlayerDetectionConfig()
    serial_numbers: dict[str, str] = {}
    export: ExportConfig = ExportConfig()
    bounce_detection: BounceDetectionConfig = BounceDetectionConfig()
    hit_bounce_refiner: HitBounceRefinerConfig = HitBounceRefinerConfig()


def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig(**raw)
