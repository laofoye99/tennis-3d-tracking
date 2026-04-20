"""3D triangulation from two camera world-coordinate observations.

Uses analytical closest-point between two rays (no scipy needed).
Falls back to z=0 clamp if result is underground.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def triangulate(
    world_2d_cam1: tuple[float, float],
    world_2d_cam2: tuple[float, float],
    camera_pos_1: list[float],
    camera_pos_2: list[float],
) -> tuple[float, float, float]:
    """Compute 3D ball position from two 2D world-coordinate observations.

    Each camera projects a ray from its 3D position through the observed
    ground-plane point (x, y, 0). We find the closest points on the two
    rays analytically and return their midpoint.

    No bounds on s/t — rays extend freely beyond the ground projection.
    Only constraint: final z >= 0 (ball can't be underground).

    Args:
        world_2d_cam1: (x, y) ground-plane projection from camera 1 homography.
        world_2d_cam2: (x, y) ground-plane projection from camera 2 homography.
        camera_pos_1: [x, y, z] camera 1 position in meters.
        camera_pos_2: [x, y, z] camera 2 position in meters.

    Returns:
        (x, y, z) 3D position in meters.
    """
    cam1 = np.asarray(camera_pos_1, dtype=np.float64)
    cam2 = np.asarray(camera_pos_2, dtype=np.float64)
    ground1 = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    ground2 = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])

    d1 = ground1 - cam1  # ray 1 direction
    d2 = ground2 - cam2  # ray 2 direction

    # Analytical closest-point between two rays
    # Ray 1: P1(s) = cam1 + s * d1
    # Ray 2: P2(t) = cam2 + t * d2
    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Rays nearly parallel — fall back to midpoint of ground projections
        mid = (ground1 + ground2) / 2.0
        return float(mid[0]), float(mid[1]), 0.0

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom

    # Ensure s > 0 and t > 0 (ball in front of cameras, toward ground)
    s = max(s, 0.0)
    t = max(t, 0.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0

    # Clamp z >= 0
    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2])
