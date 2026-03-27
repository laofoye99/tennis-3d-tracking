"""Compute static homography matrices for two tennis court cameras.

Reads Labelme JSON annotations for camera 66 and camera 68, maps labeled
court keypoints to real-world singles court coordinates (meters), and
computes homography matrices (image → world) using OpenCV.
"""

import json
import numpy as np
import cv2
from pathlib import Path

# ---------------------------------------------------------------------------
# Court dimensions (meters)
# Coordinate system V2: origin at court center (net center ground projection)
#   X: left-to-right (from cam66 perspective), range [-4.115, 4.115]
#   Y: cam68-end to cam66-end, range [-11.89, 11.89]
#   Z: up
#   Net at y=0
# ---------------------------------------------------------------------------
COURT_HALF_LENGTH = 11.89     # half court length (baseline to net)
COURT_HALF_WIDTH = 4.115      # half court width
SERVICE_DIST = 6.400          # service line distance from net
NET_Y = 0.0                   # net at origin

# Derived coordinates
BASELINE_NEAR = -COURT_HALF_LENGTH   # -11.89 (cam68 side)
BASELINE_FAR = COURT_HALF_LENGTH     # +11.89 (cam66 side)
SERVICE_NEAR = -SERVICE_DIST         # -6.400
SERVICE_FAR = SERVICE_DIST           # +6.400
SIDELINE_LEFT = -COURT_HALF_WIDTH    # -4.115
SIDELINE_RIGHT = COURT_HALF_WIDTH    # +4.115

# Legacy compatibility
COURT_LENGTH = 2 * COURT_HALF_LENGTH  # 23.78
DOUBLES_WIDTH = 2 * COURT_HALF_WIDTH  # 8.23

# ---------------------------------------------------------------------------
# World coordinates for 13 labeled keypoints (12 corners + net center)
# Origin at court center, net at y=0
#
# Layout (from above):
#   left_top         center_top         right_top          y = +11.89
#   left_top_serve   center_top_serve   right_top_serve    y = +6.400
#                    center (net)                          y = 0
#   left_bottom_serve center_bottom_serve right_bottom_serve y = -6.400
#   left_bottom      center_bottom      right_bottom       y = -11.89
#
# Camera 66 sits at y≈+17 (far end), looks toward y=-11.89
# Camera 68 sits at y≈-17 (near end), looks toward y=+11.89
# ---------------------------------------------------------------------------

WORLD_COORDS_CAM66 = {
    "left_top":           (SIDELINE_LEFT,   BASELINE_NEAR),     # (-4.115, -11.89)
    "left_top_serve":     (SIDELINE_LEFT,   SERVICE_NEAR),      # (-4.115, -6.400)
    "left_bottom_serve":  (SIDELINE_LEFT,   SERVICE_FAR),       # (-4.115, +6.400)
    "left_bottom":        (SIDELINE_LEFT,   BASELINE_FAR),      # (-4.115, +11.89)
    "center_top":         (0.0,             BASELINE_NEAR),     # (0, -11.89)
    "center_top_serve":   (0.0,             SERVICE_NEAR),      # (0, -6.400)
    "center":             (0.0,             NET_Y),             # (0, 0) net center
    "center_bottom_serve":(0.0,             SERVICE_FAR),       # (0, +6.400)
    "center_bottom":      (0.0,             BASELINE_FAR),      # (0, +11.89)
    "right_top":          (SIDELINE_RIGHT,  BASELINE_NEAR),     # (+4.115, -11.89)
    "right_top_serve":    (SIDELINE_RIGHT,  SERVICE_NEAR),      # (+4.115, -6.400)
    "right_bottom_serve": (SIDELINE_RIGHT,  SERVICE_FAR),       # (+4.115, +6.400)
    "right_bottom":       (SIDELINE_RIGHT,  BASELINE_FAR),      # (+4.115, +11.89)
}

# Camera 68: faces opposite direction
WORLD_COORDS_CAM68 = {
    "left_top":           (SIDELINE_RIGHT,  BASELINE_FAR),      # (+4.115, +11.89)
    "left_top_serve":     (SIDELINE_RIGHT,  SERVICE_FAR),       # (+4.115, +6.400)
    "left_bottom_serve":  (SIDELINE_RIGHT,  SERVICE_NEAR),      # (+4.115, -6.400)
    "left_bottom":        (SIDELINE_RIGHT,  BASELINE_NEAR),     # (+4.115, -11.89)
    "center_top":         (0.0,             BASELINE_FAR),      # (0, +11.89)
    "center_top_serve":   (0.0,             SERVICE_FAR),       # (0, +6.400)
    "center":             (0.0,             NET_Y),             # (0, 0) net center
    "center_bottom_serve":(0.0,             SERVICE_NEAR),      # (0, -6.400)
    "center_bottom":      (0.0,             BASELINE_NEAR),     # (0, -11.89)
    "right_top":          (SIDELINE_LEFT,   BASELINE_FAR),      # (-4.115, +11.89)
    "right_top_serve":    (SIDELINE_LEFT,   SERVICE_FAR),       # (-4.115, +6.400)
    "right_bottom_serve": (SIDELINE_LEFT,   SERVICE_NEAR),      # (-4.115, -6.400)
    "right_bottom":       (SIDELINE_LEFT,   BASELINE_NEAR),     # (-4.115, -11.89)
}


def load_labelme_points(json_path: str) -> dict[str, tuple[float, float]]:
    """Load labeled points from a Labelme JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = {}
    for shape in data["shapes"]:
        label = shape["label"]
        x, y = shape["points"][0]
        points[label] = (x, y)
    return points


def compute_homography(
    image_points: dict[str, tuple[float, float]],
    world_coords: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute homography from image pixels to world coordinates.

    Returns:
        H_img2world: 3x3 homography matrix (image → world)
        H_world2img: 3x3 homography matrix (world → image)
        used_labels: list of labels used for computation
    """
    labels = sorted(set(image_points.keys()) & set(world_coords.keys()))
    src = np.array([image_points[l] for l in labels], dtype=np.float64)
    dst = np.array([world_coords[l] for l in labels], dtype=np.float64)

    H_img2world, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    H_world2img, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

    return H_img2world, H_world2img, labels


def verify_homography(
    H: np.ndarray,
    image_points: dict[str, tuple[float, float]],
    world_coords: dict[str, tuple[float, float]],
    labels: list[str],
) -> float:
    """Compute mean reprojection error (image → world) in meters."""
    src = np.array([image_points[l] for l in labels], dtype=np.float64)
    dst = np.array([world_coords[l] for l in labels], dtype=np.float64)

    # Convert to homogeneous and project
    ones = np.ones((len(src), 1))
    src_h = np.hstack([src, ones])
    projected = (H @ src_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]

    errors = np.linalg.norm(projected - dst, axis=1)
    return float(errors.mean()), errors


def main():
    src_dir = Path(__file__).parent

    # Use 1080p annotations (must match the video resolution used by the pipeline)
    cam66_json = r"D:\tennis\tennis-3d-tracking\.claude\worktrees\elastic-goldberg\src\cam66.json"
    cam68_json = r"D:\tennis\tennis-3d-tracking\.claude\worktrees\elastic-goldberg\src\cam68.json"

    # Load image points
    pts66 = load_labelme_points(str(cam66_json))
    pts68 = load_labelme_points(str(cam68_json))

    print("=" * 60)
    print("Computing homography for Camera 66")
    print("=" * 60)
    H66_i2w, H66_w2i, labels66 = compute_homography(pts66, WORLD_COORDS_CAM66)
    mean_err66, errs66 = verify_homography(H66_i2w, pts66, WORLD_COORDS_CAM66, labels66)
    print(f"Used {len(labels66)} points: {labels66}")
    print(f"Mean reprojection error: {mean_err66:.4f} m")
    for l, e in zip(labels66, errs66):
        print(f"  {l:25s} error = {e:.4f} m")
    print(f"\nH_image_to_world (cam66):\n{H66_i2w}")
    print(f"\nH_world_to_image (cam66):\n{H66_w2i}")

    print()
    print("=" * 60)
    print("Computing homography for Camera 68")
    print("=" * 60)
    H68_i2w, H68_w2i, labels68 = compute_homography(pts68, WORLD_COORDS_CAM68)
    mean_err68, errs68 = verify_homography(H68_i2w, pts68, WORLD_COORDS_CAM68, labels68)
    print(f"Used {len(labels68)} points: {labels68}")
    print(f"Mean reprojection error: {mean_err68:.4f} m")
    for l, e in zip(labels68, errs68):
        print(f"  {l:25s} error = {e:.4f} m")
    print(f"\nH_image_to_world (cam68):\n{H68_i2w}")
    print(f"\nH_world_to_image (cam68):\n{H68_w2i}")

    # Save matrices to a single JSON file
    output = {
        "court_dimensions": {
            "coordinate_system": "V2: origin at court center, net at y=0",
            "half_length_m": COURT_HALF_LENGTH,
            "half_width_m": COURT_HALF_WIDTH,
            "net_y_m": NET_Y,
            "service_near_y_m": SERVICE_NEAR,
            "service_far_y_m": SERVICE_FAR,
            "baseline_near_y_m": BASELINE_NEAR,
            "baseline_far_y_m": BASELINE_FAR,
        },
        "cam66": {
            "H_image_to_world": H66_i2w.tolist(),
            "H_world_to_image": H66_w2i.tolist(),
            "reprojection_error_m": mean_err66,
        },
        "cam68": {
            "H_image_to_world": H68_i2w.tolist(),
            "H_world_to_image": H68_w2i.tolist(),
            "reprojection_error_m": mean_err68,
        },
    }

    out_path = src_dir / "homography_matrices.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
