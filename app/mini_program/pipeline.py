"""NearFieldPipeline — spawns independent near-field pipelines for cam66 + cam68."""

import logging
from typing import Optional

from app.mini_program.camera_processor import SingleCameraProcessor

logger = logging.getLogger(__name__)


class NearFieldPipeline:
    """Top-level manager for the dual-camera mini-program detection pipeline.

    Creates two SingleCameraProcessor instances (cam66 + cam68), each running
    in its own thread.  Completely independent of the main 3D-triangulation
    orchestrator — only reuses config for RTSP URLs, homography, serial numbers,
    and export endpoint.
    """

    _instance: Optional["NearFieldPipeline"] = None

    def __init__(self, config):
        self._processors: list[SingleCameraProcessor] = []
        self._started = False

        ball_model = "model_weight/best_tennis.pt"
        player_model = config.player_detection.model_path or "model_weight/yolo26x-pose.pt"
        player_device = config.player_detection.device or "cuda"
        player_conf = float(config.player_detection.conf or 0.4)
        homography_path = config.homography.path
        endpoint = config.export.endpoint
        dry_run = bool(config.export.dry_run)

        for cam_name in ["cam66", "cam68"]:
            cam_cfg = config.cameras.get(cam_name)
            if cam_cfg is None or cam_cfg.record_only:
                continue
            rtsp = cam_cfg.rtsp_url
            homography_key = cam_cfg.homography_key or cam_name
            serial = config.serial_numbers.get(cam_name, cam_name)
            side = "near" if cam_name == "cam66" else "far"

            proc = SingleCameraProcessor(
                name=cam_name,
                rtsp_url=rtsp,
                homography_path=homography_path,
                homography_key=homography_key,
                ball_model_path=ball_model,
                player_model_path=player_model,
                player_device=player_device,
                player_conf=player_conf,
                serial_number=serial,
                endpoint=endpoint,
                dry_run=dry_run,
                side=side,
            )
            self._processors.append(proc)

        NearFieldPipeline._instance = self

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for proc in self._processors:
            proc.start()
        logger.info("NearFieldPipeline started (%d cameras)", len(self._processors))

    def stop(self) -> None:
        for proc in self._processors:
            proc.stop()
        self._started = False
        logger.info("NearFieldPipeline stopped")

    @classmethod
    def get_instance(cls) -> Optional["NearFieldPipeline"]:
        return cls._instance
