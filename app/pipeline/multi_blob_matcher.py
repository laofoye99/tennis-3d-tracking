"""Cross-camera blob pairing via 3D triangulation.

Given multi-blob candidates from two cameras, finds the pair whose
triangulated 3D point has the lowest ray_distance (i.e. the pair most
likely to be the same physical ball).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _triangulate_with_distance(
    world_2d_cam1: tuple[float, float],
    world_2d_cam2: tuple[float, float],
    cam_pos_1: list[float],
    cam_pos_2: list[float],
) -> tuple[float, float, float, float]:
    """Triangulate and return (x, y, z, ray_distance).

    ray_distance is ||p1 - p2|| where p1, p2 are the closest points
    on the two camera rays. Smaller = more consistent observation.
    """
    cam1 = np.asarray(cam_pos_1, dtype=np.float64)
    cam2 = np.asarray(cam_pos_2, dtype=np.float64)
    ground1 = np.array([world_2d_cam1[0], world_2d_cam1[1], 0.0])
    ground2 = np.array([world_2d_cam2[0], world_2d_cam2[1], 0.0])

    d1 = ground1 - cam1
    d2 = ground2 - cam2

    w = cam1 - cam2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        mid_ground = (ground1 + ground2) / 2.0
        return float(mid_ground[0]), float(mid_ground[1]), 0.0, 999.0

    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom

    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    p1_fixed = cam1 + s * d1
    t = float(np.dot(p1_fixed - cam2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0.0, 1.0)

    p2_fixed = cam2 + t * d2
    s = float(np.dot(p2_fixed - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0

    ray_dist = float(np.linalg.norm(p1 - p2))

    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2]), ray_dist


class MultiBlobMatcher:
    """Match blob candidates across two cameras using 3D triangulation.

    For each frame, tries all (cam1_blob, cam2_blob) pairs, triangulates,
    and picks the pair with the lowest ray_distance that also has a
    physically plausible z coordinate.
    """

    def __init__(
        self,
        cam1_pos: list[float],
        cam2_pos: list[float],
        max_ray_distance: float = 2.0,
        valid_z_range: tuple[float, float] = (0.0, 6.0),
    ):
        self.cam1_pos = cam1_pos
        self.cam2_pos = cam2_pos
        self.max_ray_distance = max_ray_distance
        self.z_min, self.z_max = valid_z_range

        # Stats
        self.total_frames = 0
        self.matched_frames = 0
        self.non_top1_picks = 0  # times the best pair wasn't (blob0, blob0)

    def match(
        self,
        det1: dict,
        det2: dict,
    ) -> Optional[dict]:
        """Find the best blob pair across two cameras for one frame.

        Args:
            det1: Detection dict from camera 1, must have 'candidates' list.
                  Each candidate has 'world_x', 'world_y', 'pixel_x', 'pixel_y', 'blob_sum'.
            det2: Detection dict from camera 2, same format.

        Returns:
            Dict with 3D point and matched blob info, or None if no valid pair.
        """
        cands1 = det1.get("candidates", [])
        cands2 = det2.get("candidates", [])
        if not cands1 or not cands2:
            return None

        self.total_frames += 1

        best = None
        best_ray_dist = float("inf")

        for i, c1 in enumerate(cands1):
            for j, c2 in enumerate(cands2):
                x, y, z, ray_dist = _triangulate_with_distance(
                    (c1["world_x"], c1["world_y"]),
                    (c2["world_x"], c2["world_y"]),
                    self.cam1_pos,
                    self.cam2_pos,
                )

                # Filter: ray_distance and z must be reasonable
                if ray_dist > self.max_ray_distance:
                    continue
                if z < self.z_min or z > self.z_max:
                    continue

                if ray_dist < best_ray_dist:
                    best_ray_dist = ray_dist
                    best = {
                        "x": x,
                        "y": y,
                        "z": z,
                        "ray_distance": ray_dist,
                        "frame_index": det1.get("frame_index"),
                        "cam1_idx": i,
                        "cam2_idx": j,
                        "cam1_pixel": [c1["pixel_x"], c1["pixel_y"]],
                        "cam2_pixel": [c2["pixel_x"], c2["pixel_y"]],
                        "cam1_world": [c1["world_x"], c1["world_y"]],
                        "cam2_world": [c2["world_x"], c2["world_y"]],
                        "cam1_blob_sum": c1["blob_sum"],
                        "cam2_blob_sum": c2["blob_sum"],
                    }

        if best is not None:
            self.matched_frames += 1
            if best["cam1_idx"] != 0 or best["cam2_idx"] != 0:
                self.non_top1_picks += 1

        return best

    def get_stats(self) -> dict:
        return {
            "total_frames": self.total_frames,
            "matched_frames": self.matched_frames,
            "non_top1_picks": self.non_top1_picks,
            "non_top1_rate": (
                self.non_top1_picks / self.matched_frames
                if self.matched_frames > 0
                else 0.0
            ),
        }
