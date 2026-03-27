"""3D triangulation from two camera world-coordinate observations.

Uses the same algorithm as wasb/notebooks/two_views_data.ipynb:
scipy.optimize.minimize with z>=0 constraint and bounds=[0,1].
"""

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

    Identical to calculate_3d_point() in wasb/notebooks/two_views_data.ipynb.

    Each camera projects a ray from its 3D position through the observed
    ground-plane point (x, y, 0). Uses scipy.optimize.minimize to find
    the closest points on the two rays with z>=0 constraint.

    Args:
        world_2d_cam1: (x, y) ground-plane projection from camera 1 homography.
        world_2d_cam2: (x, y) ground-plane projection from camera 2 homography.
        camera_pos_1: [x, y, z] camera 1 position in meters.
        camera_pos_2: [x, y, z] camera 2 position in meters.

    Returns:
        (x, y, z) 3D position in meters.
    """
    pts1_view1_w = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    pts1_view2_w = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])
    camera_1 = np.asarray(camera_pos_1, dtype=np.float64)
    camera_2 = np.asarray(camera_pos_2, dtype=np.float64)

    # Ray directions
    d1 = pts1_view1_w - camera_1
    d2 = pts1_view2_w - camera_2

    def distance(params):
        s, t = params
        P1 = camera_1 + s * d1
        P2 = camera_2 + t * d2
        return np.linalg.norm(P1 - P2)

    def constraint(params):
        s, t = params
        P1_z = camera_1[2] + s * d1[2]
        P2_z = camera_2[2] + t * d2[2]
        return min(P1_z, P2_z)

    initial_guess = [0.5, 0.5]
    constraints = ({'type': 'ineq', 'fun': constraint})

    result = minimize(
        distance, initial_guess,
        constraints=constraints,
        bounds=[(0, 1), (0, 1)],
    )

    s_opt, t_opt = result.x
    P1_opt = camera_1 + s_opt * d1
    P2_opt = camera_2 + t_opt * d2
    mid_point = (P1_opt + P2_opt) / 2.0

    return float(mid_point[0]), float(mid_point[1]), float(mid_point[2])
