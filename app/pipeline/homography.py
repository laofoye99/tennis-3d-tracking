"""Homography transformation between image pixels and world coordinates."""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)


class HomographyTransformer:
    """Applies precomputed homography matrices for a specific camera."""

    def __init__(self, matrices_path: str, camera_key: str, court_x_margin: float = 1.0):
        with open(matrices_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cam_data = data[camera_key]
        self.H_img2world = np.array(cam_data["H_image_to_world"], dtype=np.float64)
        self.H_world2img = np.array(cam_data["H_world_to_image"], dtype=np.float64)
        self.court = data.get("court_dimensions", {})

        # Court X bounds for blob filtering
        court_w = self.court.get("width_m", 8.23)
        self.court_x_min = -court_x_margin
        self.court_x_max = court_w + court_x_margin

        logger.info(
            "[%s] Homography loaded (reproj error: %.4fm, court_x: [%.1f, %.1f])",
            camera_key,
            cam_data.get("reprojection_error_m", -1),
            self.court_x_min,
            self.court_x_max,
        )

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        """Convert image pixel coordinates to world coordinates (meters)."""
        pt = np.array([px, py, 1.0])
        result = self.H_img2world @ pt
        return float(result[0] / result[2]), float(result[1] / result[2])

    def is_in_court_x(self, px: float, py: float) -> bool:
        """Check if a pixel position projects to a world X within court bounds."""
        wx, _wy = self.pixel_to_world(px, py)
        return self.court_x_min <= wx <= self.court_x_max

    def world_to_pixel(self, wx: float, wy: float) -> tuple[float, float]:
        """Convert world coordinates (meters) to image pixel coordinates."""
        pt = np.array([wx, wy, 1.0])
        result = self.H_world2img @ pt
        return float(result[0] / result[2]), float(result[1] / result[2])
