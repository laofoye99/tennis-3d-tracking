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
# Standard ITF court dimensions (meters)
# ---------------------------------------------------------------------------
COURT_LENGTH = 23.77       # baseline to baseline
DOUBLES_WIDTH = 8.23       # doubles sideline to sideline
NET_Y = COURT_LENGTH / 2   # 11.885 m
SERVICE_DIST = 6.40        # service line distance from net
SERVICE_NEAR_Y = NET_Y - SERVICE_DIST   # 5.485 m
SERVICE_FAR_Y = NET_Y + SERVICE_DIST    # 18.285 m

# Singles court dimensions — the labeled keypoints are on SINGLES lines
SINGLES_LEFT = 1.37        # singles left sideline
SINGLES_RIGHT = 6.86       # singles right sideline (8.23 - 1.37)
CENTER_X = (SINGLES_LEFT + SINGLES_RIGHT) / 2   # 4.115 m

# ---------------------------------------------------------------------------
# World coordinates for the 12 labeled keypoints
# Coordinate system: x=0 is doubles left sideline, y=0 is near baseline.
# x → right, y → away from camera.
#
# IMPORTANT: The labeled points (left_top, right_bottom, etc.) correspond to
# SINGLES sidelines, NOT doubles sidelines. left = x=1.37, right = x=6.86.
#
# Camera 66 sits at y≈-5 (near baseline), looks toward y=23.77 (far baseline)
# Camera 68 sits at y≈29 (opposite end), looks toward y=0
# Since labels are *relative* to each camera's viewpoint, the same label maps
# to different physical locations for each camera.
# ---------------------------------------------------------------------------

# Camera 66: bottom = y=0 (near), top = y=23.77 (far)
# left = singles left sideline (x=1.37), right = singles right sideline (x=6.86)
WORLD_COORDS_CAM66 = {
    "left_top":           (SINGLES_LEFT,   COURT_LENGTH),
    "left_top_serve":     (SINGLES_LEFT,   SERVICE_FAR_Y),
    "left_bottom_serve":  (SINGLES_LEFT,   SERVICE_NEAR_Y),
    "left_bottom":        (SINGLES_LEFT,   0.0),
    "center_top":         (CENTER_X,       COURT_LENGTH),
    "center_top_serve":   (CENTER_X,       SERVICE_FAR_Y),
    "center_bottom_serve":(CENTER_X,       SERVICE_NEAR_Y),
    "center_bottom":      (CENTER_X,       0.0),
    "right_top":          (SINGLES_RIGHT,  COURT_LENGTH),
    "right_top_serve":    (SINGLES_RIGHT,  SERVICE_FAR_Y),
    "right_bottom_serve": (SINGLES_RIGHT,  SERVICE_NEAR_Y),
    "right_bottom":       (SINGLES_RIGHT,  0.0),
}

# Camera 68: faces the opposite direction
# Its "left" = cam66's "right" (x=6.86), its "right" = cam66's "left" (x=1.37)
# Its "bottom" (near) = cam66's "top" (y=23.77), its "top" (far) = cam66's "bottom" (y=0)
WORLD_COORDS_CAM68 = {
    "left_top":           (SINGLES_RIGHT,  0.0),
    "left_top_serve":     (SINGLES_RIGHT,  SERVICE_NEAR_Y),
    "left_bottom_serve":  (SINGLES_RIGHT,  SERVICE_FAR_Y),
    "left_bottom":        (SINGLES_RIGHT,  COURT_LENGTH),
    "center_top":         (CENTER_X,       0.0),
    "center_top_serve":   (CENTER_X,       SERVICE_NEAR_Y),
    "center_bottom_serve":(CENTER_X,       SERVICE_FAR_Y),
    "center_bottom":      (CENTER_X,       COURT_LENGTH),
    "right_top":          (SINGLES_LEFT,   0.0),
    "right_top_serve":    (SINGLES_LEFT,   SERVICE_NEAR_Y),
    "right_bottom_serve": (SINGLES_LEFT,   SERVICE_FAR_Y),
    "right_bottom":       (SINGLES_LEFT,   COURT_LENGTH),
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
            "length_m": COURT_LENGTH,
            "width_m": DOUBLES_WIDTH,
            "net_y_m": NET_Y,
            "service_near_y_m": SERVICE_NEAR_Y,
            "service_far_y_m": SERVICE_FAR_Y,
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
