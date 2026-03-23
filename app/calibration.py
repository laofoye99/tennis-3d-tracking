"""Camera calibration module: intrinsic/extrinsic estimation from court keypoints.

Uses PnP (Perspective-n-Point) solving with known 3D court coordinates and
their 2D pixel correspondences from Labelme annotations.

    CameraCalibrator:  Single camera intrinsic + extrinsic estimation
    StereoCalibrator:  Dual camera relative pose and validation
    run_calibration():  CLI entry point for full dual-camera calibration
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard ITF singles court dimensions (meters)
# Reused from src/compute_homography.py, extended to 3D (z=0 ground plane)
# ---------------------------------------------------------------------------

COURT_LENGTH = 23.77
DOUBLES_WIDTH = 8.23
NET_Y = COURT_LENGTH / 2  # 11.885
SERVICE_DIST = 6.40
SERVICE_NEAR_Y = NET_Y - SERVICE_DIST  # 5.485
SERVICE_FAR_Y = NET_Y + SERVICE_DIST  # 18.285

# Singles court lines — the labeled keypoints are on SINGLES sidelines
SINGLES_LEFT = 1.37        # singles left sideline
SINGLES_RIGHT = 6.86       # singles right sideline
CENTER_X = (SINGLES_LEFT + SINGLES_RIGHT) / 2  # 4.115

# 3D world coordinates for 12 labeled keypoints (z=0 on ground plane)
# IMPORTANT: left/right labels correspond to SINGLES sidelines, not doubles.
# Camera 66: near baseline at y=0, looking toward y=23.77
WORLD_COORDS_CAM66 = {
    "left_top": (SINGLES_LEFT, COURT_LENGTH, 0.0),
    "left_top_serve": (SINGLES_LEFT, SERVICE_FAR_Y, 0.0),
    "left_bottom_serve": (SINGLES_LEFT, SERVICE_NEAR_Y, 0.0),
    "left_bottom": (SINGLES_LEFT, 0.0, 0.0),
    "center_top": (CENTER_X, COURT_LENGTH, 0.0),
    "center_top_serve": (CENTER_X, SERVICE_FAR_Y, 0.0),
    "center_bottom_serve": (CENTER_X, SERVICE_NEAR_Y, 0.0),
    "center_bottom": (CENTER_X, 0.0, 0.0),
    "right_top": (SINGLES_RIGHT, COURT_LENGTH, 0.0),
    "right_top_serve": (SINGLES_RIGHT, SERVICE_FAR_Y, 0.0),
    "right_bottom_serve": (SINGLES_RIGHT, SERVICE_NEAR_Y, 0.0),
    "right_bottom": (SINGLES_RIGHT, 0.0, 0.0),
}

# Camera 68: opposite end, facing toward y=0
# Its "left" = cam66's "right" (x=6.86), its "right" = cam66's "left" (x=1.37)
WORLD_COORDS_CAM68 = {
    "left_top": (SINGLES_RIGHT, 0.0, 0.0),
    "left_top_serve": (SINGLES_RIGHT, SERVICE_NEAR_Y, 0.0),
    "left_bottom_serve": (SINGLES_RIGHT, SERVICE_FAR_Y, 0.0),
    "left_bottom": (SINGLES_RIGHT, COURT_LENGTH, 0.0),
    "center_top": (CENTER_X, 0.0, 0.0),
    "center_top_serve": (CENTER_X, SERVICE_NEAR_Y, 0.0),
    "center_bottom_serve": (CENTER_X, SERVICE_FAR_Y, 0.0),
    "center_bottom": (CENTER_X, COURT_LENGTH, 0.0),
    "right_top": (SINGLES_LEFT, 0.0, 0.0),
    "right_top_serve": (SINGLES_LEFT, SERVICE_NEAR_Y, 0.0),
    "right_bottom_serve": (SINGLES_LEFT, SERVICE_FAR_Y, 0.0),
    "right_bottom": (SINGLES_LEFT, COURT_LENGTH, 0.0),
}

WORLD_COORDS = {"cam66": WORLD_COORDS_CAM66, "cam68": WORLD_COORDS_CAM68}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_labelme_points(json_path: str) -> dict[str, tuple[float, float]]:
    """Load labeled keypoints from a Labelme JSON file.

    Returns:
        Dict mapping label → (pixel_x, pixel_y).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = {}
    for shape in data["shapes"]:
        label = shape["label"]
        x, y = shape["points"][0]
        points[label] = (x, y)
    return points


# ---------------------------------------------------------------------------
# CameraCalibrator
# ---------------------------------------------------------------------------


class CameraCalibrator:
    """Estimate camera intrinsic and extrinsic parameters from court keypoints.

    Uses ``cv2.calibrateCamera`` (Zhang's method for coplanar points) to
    estimate the intrinsic matrix K and distortion coefficients, then
    refines extrinsics with ``cv2.solvePnP``.

    Usage::

        cal = CameraCalibrator("cam66", "src/cam66.json")
        result = cal.calibrate()
        print(result["camera_position_3d"])
        print(result["mean_reprojection_error_px"])
    """

    def __init__(
        self,
        camera_name: str,
        labelme_json_path: str,
        image_size: tuple[int, int] = (1920, 1080),
        fix_aspect_ratio: bool = True,
        zero_distortion: bool = False,
    ):
        self.camera_name = camera_name
        self.labelme_json_path = labelme_json_path
        self.image_size = image_size  # (width, height)
        self.fix_aspect_ratio = fix_aspect_ratio
        self.zero_distortion = zero_distortion

        # Load data
        self.pixel_points = load_labelme_points(labelme_json_path)
        if camera_name not in WORLD_COORDS:
            raise ValueError(
                f"Unknown camera: {camera_name}. "
                f"Expected one of {list(WORLD_COORDS.keys())}"
            )
        self.world_coords_3d = WORLD_COORDS[camera_name]

        # Results (populated after calibrate())
        self.K: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.camera_position: Optional[np.ndarray] = None

    def calibrate(self) -> dict:
        """Run full calibration: intrinsics + extrinsics.

        Returns:
            Dict with K, dist_coeffs, rvec, tvec, R, camera_position_3d,
            reprojection errors, and derived homography matrix.
        """
        # Match labeled points between pixel annotations and world coords
        labels = sorted(
            set(self.pixel_points.keys()) & set(self.world_coords_3d.keys())
        )
        if len(labels) < 6:
            raise ValueError(
                f"Need at least 6 matched keypoints, got {len(labels)}: {labels}"
            )

        # Build arrays: 3D object points and 2D image points
        obj_points = np.array(
            [self.world_coords_3d[l] for l in labels], dtype=np.float64
        )
        img_points = np.array(
            [self.pixel_points[l] for l in labels], dtype=np.float64
        )

        logger.info(
            "[%s] Calibrating with %d keypoints: %s",
            self.camera_name,
            len(labels),
            labels,
        )

        # Stage 1: Estimate intrinsics using calibrateCamera
        # All points are coplanar (z=0) → Zhang's method.
        flags = 0
        if self.fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if self.zero_distortion:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            [obj_points.reshape(-1, 1, 3).astype(np.float32)],
            [img_points.reshape(-1, 1, 2).astype(np.float32)],
            self.image_size,
            None,
            None,
            flags=flags,
        )

        self.K = K
        self.dist_coeffs = dist
        self.rvec = rvecs[0]
        self.tvec = tvecs[0]

        logger.info(
            "[%s] calibrateCamera RMS error: %.4f px", self.camera_name, ret
        )

        # Stage 2: Refine extrinsics with solvePnP (iterative)
        success, rvec_refined, tvec_refined = cv2.solvePnP(
            obj_points.reshape(-1, 1, 3).astype(np.float64),
            img_points.reshape(-1, 1, 2).astype(np.float64),
            self.K,
            self.dist_coeffs,
            rvec=self.rvec.copy(),
            tvec=self.tvec.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success:
            self.rvec = rvec_refined
            self.tvec = tvec_refined
        else:
            logger.warning(
                "[%s] solvePnP refinement failed, using calibrateCamera result",
                self.camera_name,
            )

        # Compute rotation matrix and camera position in world
        self.R, _ = cv2.Rodrigues(self.rvec)
        # Camera position: C = -R^T · t
        self.camera_position = (-self.R.T @ self.tvec).flatten()

        # Compute per-point reprojection error (pixel space)
        projected, _ = cv2.projectPoints(
            obj_points.reshape(-1, 1, 3).astype(np.float64),
            self.rvec,
            self.tvec,
            self.K,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)

        per_point_errors = {}
        errors_px = []
        for i, label in enumerate(labels):
            err = float(np.linalg.norm(projected[i] - img_points[i]))
            per_point_errors[label] = {
                "pixel_original": [round(float(img_points[i][0]), 2), round(float(img_points[i][1]), 2)],
                "pixel_reprojected": [round(float(projected[i][0]), 2), round(float(projected[i][1]), 2)],
                "error_px": round(err, 3),
            }
            errors_px.append(err)

        mean_err_px = float(np.mean(errors_px))
        max_err_px = float(np.max(errors_px))

        # Derive ground-plane homography from K, R, t
        H_img2world = self._compute_homography_from_calibration()

        # Compute world-space reprojection error via derived homography
        per_point_world_errors = {}
        world_errors_m = []
        for i, label in enumerate(labels):
            px, py = img_points[i]
            # Undistort point
            undist = cv2.undistortPoints(
                np.array([[px, py]], dtype=np.float64).reshape(-1, 1, 2),
                self.K,
                self.dist_coeffs,
                P=self.K,
            ).reshape(2)
            # Apply derived homography
            pt_h = np.array([undist[0], undist[1], 1.0])
            world_h = H_img2world @ pt_h
            world_proj = world_h[:2] / world_h[2]
            world_true = np.array(self.world_coords_3d[label][:2])
            err_m = float(np.linalg.norm(world_proj - world_true))
            per_point_world_errors[label] = {
                "world_projected": [round(float(world_proj[0]), 4), round(float(world_proj[1]), 4)],
                "world_true": [round(float(world_true[0]), 4), round(float(world_true[1]), 4)],
                "error_m": round(err_m, 4),
            }
            world_errors_m.append(err_m)

        mean_err_m = float(np.mean(world_errors_m))

        logger.info(
            "[%s] Calibration done: mean_reproj=%.2f px (%.4f m), "
            "camera_pos=[%.3f, %.3f, %.3f]",
            self.camera_name,
            mean_err_px,
            mean_err_m,
            *self.camera_position,
        )

        # Compute H_world_to_image (inverse)
        H_world2img = np.linalg.inv(H_img2world)
        H_world2img = H_world2img / H_world2img[2, 2]

        return {
            "camera_name": self.camera_name,
            "image_size": list(self.image_size),
            "n_keypoints": len(labels),
            "labels_used": labels,
            "K": self.K.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "rvec": self.rvec.flatten().tolist(),
            "tvec": self.tvec.flatten().tolist(),
            "R": self.R.tolist(),
            "camera_position_3d": [round(float(c), 4) for c in self.camera_position],
            "mean_reprojection_error_px": round(mean_err_px, 3),
            "max_reprojection_error_px": round(max_err_px, 3),
            "mean_reprojection_error_m": round(mean_err_m, 4),
            "per_point_errors_px": per_point_errors,
            "per_point_errors_world": per_point_world_errors,
            "rms_error": round(ret, 4),
            "H_image_to_world": H_img2world.tolist(),
            "H_world_to_image": H_world2img.tolist(),
        }

    def _compute_homography_from_calibration(self) -> np.ndarray:
        """Derive ground-plane homography (image → world) from K, R, t.

        For points on the z=0 plane, the projection simplifies to a
        3×3 homography: ``[u,v,1]^T ~ K [r1 r2 t] [X,Y,1]^T``

        Returns:
            H_img2world: 3×3 matrix mapping image pixels to world (X, Y)
            on the ground plane.
        """
        r1 = self.R[:, 0]
        r2 = self.R[:, 1]
        t = self.tvec.flatten()

        H_world2img = self.K @ np.column_stack([r1, r2, t])
        H_img2world = np.linalg.inv(H_world2img)
        # Normalise so H[2,2] = 1
        H_img2world = H_img2world / H_img2world[2, 2]
        return H_img2world


# ---------------------------------------------------------------------------
# StereoCalibrator
# ---------------------------------------------------------------------------


class StereoCalibrator:
    """Validate relative pose between two calibrated cameras.

    Computes baseline distance, relative rotation, and compares
    calibrated camera positions with the config.yaml measurements.

    Usage::

        stereo = StereoCalibrator(cam66_result, cam68_result)
        validation = stereo.validate()
    """

    def __init__(self, calib1: dict, calib2: dict):
        self.calib1 = calib1
        self.calib2 = calib2
        self.cam1_name = calib1["camera_name"]
        self.cam2_name = calib2["camera_name"]

    def validate(self) -> dict:
        """Compute relative pose and validate geometry.

        Checks baseline distance, camera heights, and compares with
        the manually measured positions in config.yaml.
        """
        pos1 = np.array(self.calib1["camera_position_3d"])
        pos2 = np.array(self.calib2["camera_position_3d"])

        baseline = float(np.linalg.norm(pos2 - pos1))
        height1 = float(pos1[2])
        height2 = float(pos2[2])

        # Relative pose
        R1 = np.array(self.calib1["R"])
        R2 = np.array(self.calib2["R"])
        t1 = np.array(self.calib1["tvec"]).flatten()
        t2 = np.array(self.calib2["tvec"]).flatten()

        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1

        # Essential matrix: E = [t_rel]_x @ R_rel
        tx = np.array(
            [
                [0, -t_rel[2], t_rel[1]],
                [t_rel[2], 0, -t_rel[0]],
                [-t_rel[1], t_rel[0], 0],
            ]
        )
        E = tx @ R_rel

        # Fundamental matrix: F = K2^{-T} E K1^{-1}
        K1 = np.array(self.calib1["K"])
        K2 = np.array(self.calib2["K"])
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        # Compare calibrated positions with config positions
        comparison = {}
        try:
            import yaml

            config_path = Path("config.yaml")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                cameras = cfg.get("cameras", {})
                for cam_name, cal_pos in [
                    (self.cam1_name, pos1),
                    (self.cam2_name, pos2),
                ]:
                    if cam_name in cameras and "position_3d" in cameras[cam_name]:
                        config_pos = cameras[cam_name]["position_3d"]
                        delta = float(np.linalg.norm(cal_pos - np.array(config_pos)))
                        comparison[cam_name] = {
                            "config_position": config_pos,
                            "calibrated_position": [round(float(c), 4) for c in cal_pos],
                            "delta_m": round(delta, 3),
                        }
        except Exception as e:
            logger.warning("Could not load config.yaml for comparison: %s", e)

        logger.info(
            "Stereo validation: baseline=%.2fm, heights=(%.2f, %.2f)m",
            baseline,
            height1,
            height2,
        )

        return {
            "baseline_m": round(baseline, 3),
            "camera_heights_m": {
                self.cam1_name: round(height1, 3),
                self.cam2_name: round(height2, 3),
            },
            "camera_positions": {
                self.cam1_name: [round(float(c), 4) for c in pos1],
                self.cam2_name: [round(float(c), 4) for c in pos2],
            },
            "relative_rotation": R_rel.tolist(),
            "essential_matrix": E.tolist(),
            "fundamental_matrix": F.tolist(),
            "config_comparison": comparison,
            "mean_reprojection_errors_px": {
                self.cam1_name: self.calib1["mean_reprojection_error_px"],
                self.cam2_name: self.calib2["mean_reprojection_error_px"],
            },
            "mean_reprojection_errors_m": {
                self.cam1_name: self.calib1["mean_reprojection_error_m"],
                self.cam2_name: self.calib2["mean_reprojection_error_m"],
            },
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_calibration(
    cam66_json: str = "src/cam66.json",
    cam68_json: str = "src/cam68.json",
    output_path: str = "src/camera_calibration.json",
    image_size: tuple[int, int] = (1920, 1080),
    zero_distortion: bool = False,
) -> dict:
    """Run full dual-camera calibration and save results.

    Calibrates each camera independently using ``cv2.calibrateCamera`` +
    ``cv2.solvePnP``, validates stereo geometry, and optionally updates
    the existing homography_matrices.json with improved matrices.

    Args:
        cam66_json: Path to Labelme annotation for camera 66.
        cam68_json: Path to Labelme annotation for camera 68.
        output_path: Where to save calibration results JSON.
        image_size: Image resolution (width, height) matching the annotations.
        zero_distortion: If True, fix distortion coefficients to zero.

    Returns:
        Full calibration result dict.
    """
    results = {}

    # Calibrate each camera
    for cam_name, json_path in [("cam66", cam66_json), ("cam68", cam68_json)]:
        cal = CameraCalibrator(
            camera_name=cam_name,
            labelme_json_path=json_path,
            image_size=image_size,
            zero_distortion=zero_distortion,
        )
        results[cam_name] = cal.calibrate()

    # Stereo validation
    stereo = StereoCalibrator(results["cam66"], results["cam68"])
    stereo_result = stereo.validate()

    # Build output
    output = {
        "calibration_method": "PnP",
        "court_dimensions": {
            "length_m": COURT_LENGTH,
            "width_m": DOUBLES_WIDTH,
            "net_y_m": NET_Y,
        },
        "cam66": results["cam66"],
        "cam68": results["cam68"],
        "stereo": stereo_result,
    }

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Calibration saved to %s", output_path)

    # Update homography_matrices.json with calibration-derived homographies
    _update_homography_file(results, "src/homography_matrices.json")

    return output


def calibrate_from_point_pairs(
    point_pairs: list[dict],
    cam66_video: str,
    cam68_video: str,
    frame_time: float = 1.0,
) -> dict:
    """Calibrate cameras using manually marked point correspondences.

    Each point pair represents the same ground-level (z=0) object seen
    in both camera images. Uses solvePnP to find camera pose for each
    camera, then cross-validates the pair.

    The approach works as follows:
    1. From each pair, the user marks pixel coordinates in both camera images
    2. We need the 3D world coordinates of the same points
    3. Since the user only provides pixel pairs (no world coords), we use the
       existing homography to project cam66 pixels → world coordinates
    4. Then use those world coordinates with cam68 pixels in solvePnP

    For a cleaner approach with ≥12 points, we use the existing CameraCalibrator
    flow if the points can be matched to known court keypoints.

    Args:
        point_pairs: List of dicts with 'cam66' and 'cam68' keys,
                     each containing [pixel_x, pixel_y].
        cam66_video: Path to cam66 video file (for frame extraction).
        cam68_video: Path to cam68 video file (for frame extraction).
        frame_time: Timestamp for frame extraction.

    Returns:
        Dict with cam66_position, cam68_position, reprojection_error.
    """
    if len(point_pairs) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(point_pairs)}")

    # Extract frames to get image dimensions
    cap1 = cv2.VideoCapture(cam66_video)
    cap2 = cv2.VideoCapture(cam68_video)

    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        raise ValueError("Cannot open one or both video files")

    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
    cap1.set(cv2.CAP_PROP_POS_FRAMES, int(frame_time * fps1))
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(frame_time * fps2))
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    cap1.release()
    cap2.release()

    if not ret1 or not ret2 or frame1 is None or frame2 is None:
        raise ValueError("Cannot read frames at given time")

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    img_size_1 = (w1, h1)
    img_size_2 = (w2, h2)

    logger.info(
        "Point-based calibration: %d pairs, cam66=%dx%d, cam68=%dx%d",
        len(point_pairs), w1, h1, w2, h2,
    )

    # Load existing homography for cam66 to project pixels → world coordinates
    # This gives us 3D world coords (z=0) for each user-marked point
    homography_path = Path("src/homography_matrices.json")
    if not homography_path.exists():
        raise ValueError("src/homography_matrices.json not found — needed for pixel→world projection")

    with open(homography_path, "r", encoding="utf-8") as f:
        hom_data = json.load(f)

    H_cam66_i2w = np.array(hom_data["cam66"]["H_image_to_world"])
    H_cam68_i2w = np.array(hom_data["cam68"]["H_image_to_world"])

    # For each point pair, project cam66 pixel → world (x, y, z=0)
    # and also project cam68 pixel → world for cross-validation
    world_points_from_cam66 = []
    world_points_from_cam68 = []
    cam66_pixels = []
    cam68_pixels = []

    for pair in point_pairs:
        px66 = pair["cam66"]
        px68 = pair["cam68"]

        if isinstance(px66, dict):
            px66 = [px66["x"], px66["y"]]
        if isinstance(px68, dict):
            px68 = [px68["x"], px68["y"]]

        cam66_pixels.append(px66)
        cam68_pixels.append(px68)

        # Project cam66 pixel → world
        pt_h = np.array([px66[0], px66[1], 1.0])
        w66 = H_cam66_i2w @ pt_h
        w66 = w66[:2] / w66[2]
        world_points_from_cam66.append([w66[0], w66[1], 0.0])

        # Project cam68 pixel → world
        pt_h = np.array([px68[0], px68[1], 1.0])
        w68 = H_cam68_i2w @ pt_h
        w68 = w68[:2] / w68[2]
        world_points_from_cam68.append([w68[0], w68[1], 0.0])

    # Average the two world projections for each point (better estimate)
    world_points_avg = []
    correspondence_errors = []
    for i in range(len(point_pairs)):
        w66 = np.array(world_points_from_cam66[i][:2])
        w68 = np.array(world_points_from_cam68[i][:2])
        avg = (w66 + w68) / 2
        err = float(np.linalg.norm(w66 - w68))
        correspondence_errors.append(err)
        world_points_avg.append([avg[0], avg[1], 0.0])

    mean_correspondence_error = float(np.mean(correspondence_errors))
    logger.info(
        "Mean correspondence error (cam66 vs cam68 world projection): %.4f m",
        mean_correspondence_error,
    )

    # Now run solvePnP for each camera using averaged world coords
    obj_points = np.array(world_points_avg, dtype=np.float64).reshape(-1, 1, 3)
    cam66_img_pts = np.array(cam66_pixels, dtype=np.float64).reshape(-1, 1, 2)
    cam68_img_pts = np.array(cam68_pixels, dtype=np.float64).reshape(-1, 1, 2)

    results = {}
    for cam_name, img_pts, img_size in [
        ("cam66", cam66_img_pts, img_size_1),
        ("cam68", cam68_img_pts, img_size_2),
    ]:
        # Estimate intrinsics using camera matrix estimation
        # For ≥6 points, use calibrateCamera; otherwise use solvePnP with estimated K
        fx = img_size[0]  # rough initial focal length
        K_init = np.array([
            [fx, 0, img_size[0] / 2],
            [0, fx, img_size[1] / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_init = np.zeros(5, dtype=np.float64)

        if len(point_pairs) >= 6:
            # Use calibrateCamera for better intrinsic estimation
            flags = cv2.CALIB_FIX_ASPECT_RATIO
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                [obj_points.astype(np.float32)],
                [img_pts.astype(np.float32)],
                img_size,
                None,
                None,
                flags=flags,
            )
            K_init = K
            dist_init = dist
            rvec_init = rvecs[0]
            tvec_init = tvecs[0]

            # Refine with solvePnP
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_pts,
                K_init,
                dist_init,
                rvec=rvec_init.copy(),
                tvec=tvec_init.copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            # Fewer points: use solvePnP with estimated K
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_pts,
                K_init,
                dist_init,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success:
            raise ValueError(f"solvePnP failed for {cam_name}")

        R, _ = cv2.Rodrigues(rvec)
        camera_position = (-R.T @ tvec).flatten()

        # Reprojection error
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_init, dist_init)
        projected = projected.reshape(-1, 2)
        errors = [float(np.linalg.norm(projected[i] - img_pts[i].flatten()))
                  for i in range(len(projected))]
        mean_err = float(np.mean(errors))

        results[cam_name] = {
            "camera_position": [round(float(c), 4) for c in camera_position],
            "reprojection_error_px": round(mean_err, 3),
            "K": K_init.tolist(),
            "R": R.tolist(),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
        }

        logger.info(
            "[%s] Position: [%.3f, %.3f, %.3f], reproj: %.2f px",
            cam_name, *camera_position, mean_err,
        )

    # Compute baseline
    pos1 = np.array(results["cam66"]["camera_position"])
    pos2 = np.array(results["cam68"]["camera_position"])
    baseline = float(np.linalg.norm(pos2 - pos1))

    return {
        "cam66_position": results["cam66"]["camera_position"],
        "cam68_position": results["cam68"]["camera_position"],
        "reprojection_error": {
            "cam66": results["cam66"]["reprojection_error_px"],
            "cam68": results["cam68"]["reprojection_error_px"],
        },
        "baseline_m": round(baseline, 3),
        "mean_correspondence_error_m": round(mean_correspondence_error, 4),
        "n_point_pairs": len(point_pairs),
        "cam66_details": results["cam66"],
        "cam68_details": results["cam68"],
    }


def _update_homography_file(results: dict, homography_path: str) -> None:
    """Update homography_matrices.json with calibration-derived matrices.

    Preserves old homography values under ``H_image_to_world_homography``
    for comparison, then replaces the active matrices with the
    calibration-derived ones.
    """
    try:
        with open(homography_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}

    for cam_name, calib in results.items():
        existing.setdefault(cam_name, {})
        cam = existing[cam_name]

        # Back up old homography before overwriting
        if "H_image_to_world" in cam and "H_image_to_world_homography" not in cam:
            cam["H_image_to_world_homography"] = cam["H_image_to_world"]
        if "H_world_to_image" in cam and "H_world_to_image_homography" not in cam:
            cam["H_world_to_image_homography"] = cam["H_world_to_image"]

        # Replace with calibration-derived matrices
        cam["H_image_to_world"] = calib["H_image_to_world"]
        cam["H_world_to_image"] = calib["H_world_to_image"]
        cam["reprojection_error_m"] = calib["mean_reprojection_error_m"]
        cam["calibration_method"] = "PnP"

    with open(homography_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    logger.info("Updated %s with calibration-derived homographies", homography_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Camera calibration from court keypoints"
    )
    parser.add_argument(
        "--cam66", default="src/cam66.json", help="Labelme JSON for cam66"
    )
    parser.add_argument(
        "--cam68", default="src/cam68.json", help="Labelme JSON for cam68"
    )
    parser.add_argument(
        "--output",
        default="src/camera_calibration.json",
        help="Output calibration JSON path",
    )
    parser.add_argument(
        "--no-distortion",
        action="store_true",
        help="Fix distortion coefficients to zero",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    result = run_calibration(
        cam66_json=args.cam66,
        cam68_json=args.cam68,
        output_path=args.output,
        image_size=(args.width, args.height),
        zero_distortion=args.no_distortion,
    )

    # Print summary
    for cam in ["cam66", "cam68"]:
        r = result[cam]
        print(f"\n{cam}:")
        print(f"  Camera position:  {r['camera_position_3d']}")
        print(
            f"  Mean reproj error: {r['mean_reprojection_error_px']:.2f} px "
            f"/ {r['mean_reprojection_error_m']:.4f} m"
        )
        print(f"  Focal length: fx={r['K'][0][0]:.1f}, fy={r['K'][1][1]:.1f}")

    s = result["stereo"]
    print(f"\nStereo:")
    print(f"  Baseline: {s['baseline_m']:.2f} m")
    if s.get("config_comparison"):
        for cam, comp in s["config_comparison"].items():
            print(f"  {cam} position delta vs config: {comp['delta_m']:.3f} m")
