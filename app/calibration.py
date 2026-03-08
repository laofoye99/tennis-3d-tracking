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
COURT_WIDTH = 8.23
NET_Y = COURT_LENGTH / 2  # 11.885
SERVICE_DIST = 6.40
SERVICE_NEAR_Y = NET_Y - SERVICE_DIST  # 5.485
SERVICE_FAR_Y = NET_Y + SERVICE_DIST  # 18.285
CENTER_X = COURT_WIDTH / 2  # 4.115

# 3D world coordinates for 12 labeled keypoints (z=0 on ground plane)
# Camera 66: near baseline at y=0, looking toward y=23.77
WORLD_COORDS_CAM66 = {
    "left_top": (0.0, COURT_LENGTH, 0.0),
    "left_top_serve": (0.0, SERVICE_FAR_Y, 0.0),
    "left_bottom_serve": (0.0, SERVICE_NEAR_Y, 0.0),
    "left_bottom": (0.0, 0.0, 0.0),
    "center_top": (CENTER_X, COURT_LENGTH, 0.0),
    "center_top_serve": (CENTER_X, SERVICE_FAR_Y, 0.0),
    "center_bottom_serve": (CENTER_X, SERVICE_NEAR_Y, 0.0),
    "center_bottom": (CENTER_X, 0.0, 0.0),
    "right_top": (COURT_WIDTH, COURT_LENGTH, 0.0),
    "right_top_serve": (COURT_WIDTH, SERVICE_FAR_Y, 0.0),
    "right_bottom_serve": (COURT_WIDTH, SERVICE_NEAR_Y, 0.0),
    "right_bottom": (COURT_WIDTH, 0.0, 0.0),
}

# Camera 68: opposite end, facing toward y=0
WORLD_COORDS_CAM68 = {
    "left_top": (COURT_WIDTH, 0.0, 0.0),
    "left_top_serve": (COURT_WIDTH, SERVICE_NEAR_Y, 0.0),
    "left_bottom_serve": (COURT_WIDTH, SERVICE_FAR_Y, 0.0),
    "left_bottom": (COURT_WIDTH, COURT_LENGTH, 0.0),
    "center_top": (CENTER_X, 0.0, 0.0),
    "center_top_serve": (CENTER_X, SERVICE_NEAR_Y, 0.0),
    "center_bottom_serve": (CENTER_X, SERVICE_FAR_Y, 0.0),
    "center_bottom": (CENTER_X, COURT_LENGTH, 0.0),
    "right_top": (0.0, 0.0, 0.0),
    "right_top_serve": (0.0, SERVICE_NEAR_Y, 0.0),
    "right_bottom_serve": (0.0, SERVICE_FAR_Y, 0.0),
    "right_bottom": (0.0, COURT_LENGTH, 0.0),
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
            "width_m": COURT_WIDTH,
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
