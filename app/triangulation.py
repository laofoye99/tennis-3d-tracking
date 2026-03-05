"""3D triangulation from two camera world-coordinate observations."""

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def triangulate(
    world_2d_cam1: tuple[float, float],
    world_2d_cam2: tuple[float, float],
    camera_pos_1: list[float],
    camera_pos_2: list[float],
) -> tuple[float, float, float]:
    """Compute 3D ball position from two 2D world-coordinate observations.

    Each camera projects a ray from its 3D position through the observed
    ground-plane point (x, y, 0).  We find the pair of closest points on
    the two rays and return their midpoint.

    Args:
        world_2d_cam1: (x, y) in meters from camera 1.
        world_2d_cam2: (x, y) in meters from camera 2.
        camera_pos_1: [x, y, z] camera 1 position in meters.
        camera_pos_2: [x, y, z] camera 2 position in meters.

    Returns:
        (x, y, z) 3D position in meters.
    """
    cam1 = np.asarray(camera_pos_1, dtype=np.float64)
    cam2 = np.asarray(camera_pos_2, dtype=np.float64)
    ground1 = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    ground2 = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])

    d1 = ground1 - cam1
    d2 = ground2 - cam2

    def distance(params: np.ndarray) -> float:
        s, t = params
        p1 = cam1 + s * d1
        p2 = cam2 + t * d2
        return float(np.linalg.norm(p1 - p2))

    def z_constraint(params: np.ndarray) -> float:
        s, t = params
        z1 = cam1[2] + s * d1[2]
        z2 = cam2[2] + t * d2[2]
        return float(min(z1, z2))

    result = minimize(
        distance,
        x0=[0.5, 0.5],
        constraints={"type": "ineq", "fun": z_constraint},
        bounds=[(0, 1), (0, 1)],
    )

    s_opt, t_opt = result.x
    p1 = cam1 + s_opt * d1
    p2 = cam2 + t_opt * d2
    mid = (p1 + p2) / 2.0

    return float(mid[0]), float(mid[1]), float(mid[2])
