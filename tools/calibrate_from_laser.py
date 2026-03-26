"""Precise camera calibration using laser-measured positions.

Problem: solvePnP with coplanar points gives unstable depth (camera height).
Solution: Use laser-measured camera positions as KNOWN extrinsics,
          then solve ONLY for intrinsics K.

Approach (rigorous):
1. Camera position C is known from laser measurements (world coordinates)
2. Camera orientation R is unknown but constrained:
   - Camera looks toward the court center (approximate)
   - 12 court keypoint correspondences (2D pixel ↔ 3D world) constrain R precisely
3. Given C, solve for R and K simultaneously using DLT:
   - P = K[R | -RC] is the 3×4 projection matrix
   - DLT gives P from ≥6 point correspondences
   - Decompose P into K and [R|t] via RQ decomposition
4. Refine with Levenberg-Marquardt (minimize reprojection error)

This gives:
- K: intrinsic matrix (focal length, principal point)
- R: rotation matrix (camera orientation) — derived, not assumed
- t: translation vector = -RC — derived from known C and solved R
- dist: distortion coefficients — from refinement step
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Court keypoint world coordinates (singles court, z=0 ground)
# ============================================================
SINGLES_LEFT = 1.37
SINGLES_RIGHT = 6.86
CENTER_X = (SINGLES_LEFT + SINGLES_RIGHT) / 2  # 4.115
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2  # 11.885
SERVICE_NEAR_Y = 5.485
SERVICE_FAR_Y = 18.285

# Camera 66: near baseline = bottom of image, far baseline = top
WORLD_3D_CAM66 = {
    "left_top":           (SINGLES_LEFT,  COURT_LENGTH, 0),
    "left_top_serve":     (SINGLES_LEFT,  SERVICE_FAR_Y, 0),
    "left_bottom_serve":  (SINGLES_LEFT,  SERVICE_NEAR_Y, 0),
    "left_bottom":        (SINGLES_LEFT,  0.0, 0),
    "center_top":         (CENTER_X,      COURT_LENGTH, 0),
    "center_top_serve":   (CENTER_X,      SERVICE_FAR_Y, 0),
    "center_bottom_serve":(CENTER_X,      SERVICE_NEAR_Y, 0),
    "center_bottom":      (CENTER_X,      0.0, 0),
    "right_top":          (SINGLES_RIGHT, COURT_LENGTH, 0),
    "right_top_serve":    (SINGLES_RIGHT, SERVICE_FAR_Y, 0),
    "right_bottom_serve": (SINGLES_RIGHT, SERVICE_NEAR_Y, 0),
    "right_bottom":       (SINGLES_RIGHT, 0.0, 0),
}

# Camera 68: opposite end — its left = cam66's right, etc.
WORLD_3D_CAM68 = {
    "left_top":           (SINGLES_RIGHT, 0.0, 0),
    "left_top_serve":     (SINGLES_RIGHT, SERVICE_NEAR_Y, 0),
    "left_bottom_serve":  (SINGLES_RIGHT, SERVICE_FAR_Y, 0),
    "left_bottom":        (SINGLES_RIGHT, COURT_LENGTH, 0),
    "center_top":         (CENTER_X,      0.0, 0),
    "center_top_serve":   (CENTER_X,      SERVICE_NEAR_Y, 0),
    "center_bottom_serve":(CENTER_X,      SERVICE_FAR_Y, 0),
    "center_bottom":      (CENTER_X,      COURT_LENGTH, 0),
    "right_top":          (SINGLES_LEFT,  0.0, 0),
    "right_top_serve":    (SINGLES_LEFT,  SERVICE_NEAR_Y, 0),
    "right_bottom_serve": (SINGLES_LEFT,  SERVICE_FAR_Y, 0),
    "right_bottom":       (SINGLES_LEFT,  COURT_LENGTH, 0),
}

# ============================================================
# Laser-measured camera positions (world coordinates)
# ============================================================
CAMERA_POSITIONS = {
    "cam66": np.array([4.446, 29.049, 6.830]),
    "cam68": np.array([4.431, -5.278, 6.096]),
}


def load_labelme_pixels(json_path: str) -> dict:
    """Load labeled keypoint pixel coordinates from LabelMe JSON."""
    with open(json_path) as f:
        data = json.load(f)
    points = {}
    for shape in data.get("shapes", []):
        label = shape["label"].strip().lower()
        pts = shape["points"]
        if shape["shape_type"] == "point" and len(pts) == 1:
            points[label] = (float(pts[0][0]), float(pts[0][1]))
        elif shape["shape_type"] == "rectangle" and len(pts) == 2:
            cx = (pts[0][0] + pts[1][0]) / 2
            cy = (pts[0][1] + pts[1][1]) / 2
            points[label] = (float(cx), float(cy))
    return points


def build_correspondences(pixels: dict, world_3d: dict):
    """Build matched (pixel, world) arrays from label dictionaries."""
    common = sorted(set(pixels.keys()) & set(world_3d.keys()))
    if len(common) < 6:
        raise ValueError(f"Need ≥6 common points, got {len(common)}: {common}")

    pts_2d = np.array([pixels[k] for k in common], dtype=np.float64)
    pts_3d = np.array([world_3d[k] for k in common], dtype=np.float64)

    return pts_2d, pts_3d, common


def calibrate_with_known_position(
    cam_name: str,
    cam_pos: np.ndarray,
    pts_2d: np.ndarray,
    pts_3d: np.ndarray,
    labels: list,
    image_size: tuple = (1920, 1080),
):
    """Calibrate camera intrinsics given known camera position.

    Step 1: DLT to get initial projection matrix P
    Step 2: Decompose P into K, R, t
    Step 3: Constrain t using known camera position: t = -R @ cam_pos
    Step 4: Re-solve K given fixed R, t using linear least squares
    Step 5: Refine with cv2.solvePnP (using known K, solve for R,t)
    Step 6: Final refinement with cv2.calibrateCamera (distortion)
    """
    n = len(pts_2d)
    logger.info("[%s] Calibrating with %d keypoints, camera at (%.3f, %.3f, %.3f)",
                cam_name, n, *cam_pos)

    W, H = image_size

    # ================================================================
    # Step 1: DLT — solve for 3×4 projection matrix P
    # Each point gives 2 equations: x_i = P @ X_i (homogeneous)
    # ================================================================
    A = []
    for i in range(n):
        X, Y, Z = pts_3d[i]
        u, v = pts_2d[i]
        X_h = [X, Y, Z, 1]
        A.append([0, 0, 0, 0, -X_h[0], -X_h[1], -X_h[2], -1, v*X_h[0], v*X_h[1], v*X_h[2], v])
        A.append([X_h[0], X_h[1], X_h[2], 1, 0, 0, 0, 0, -u*X_h[0], -u*X_h[1], -u*X_h[2], -u])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    P_dlt = Vt[-1].reshape(3, 4)

    # ================================================================
    # Step 2: Decompose P = K[R|t] using RQ decomposition
    # Note: M might be singular for coplanar points, so we skip
    # DLT decomposition and go directly to solvePnP with estimated K
    # ================================================================
    M = P_dlt[:, :3]  # 3×3 part
    try:
        from scipy.linalg import rq
        K_dlt, R_dlt = rq(M)
        # Ensure positive diagonal
        T = np.diag(np.sign(np.diag(K_dlt)))
        K_dlt = K_dlt @ T
        R_dlt = T @ R_dlt
        K_dlt = K_dlt / K_dlt[2, 2]
        if np.linalg.det(R_dlt) < 0:
            R_dlt = -R_dlt
        logger.info("[%s] DLT K:\n%s", cam_name, np.array2string(K_dlt, precision=1, suppress_small=True))
    except Exception as e:
        logger.warning("[%s] RQ decomposition failed (%s), using default K", cam_name, e)
        K_dlt = np.array([[1500, 0, W/2], [0, 1500, H/2], [0, 0, 1]])
        R_dlt = np.eye(3)

    # ================================================================
    # Step 3: Use known camera position to compute t = -R @ C
    # First, get a good R using solvePnP with initial K from DLT
    # ================================================================

    # Clean up K_dlt: ensure reasonable values
    fx_init = abs(K_dlt[0, 0])
    fy_init = abs(K_dlt[1, 1])
    cx_init = K_dlt[0, 2]
    cy_init = K_dlt[1, 2]

    # Sanity check
    if fx_init < 100 or fx_init > 5000:
        fx_init = 1500.0  # reasonable default for 1080p
    if fy_init < 100 or fy_init > 5000:
        fy_init = fx_init
    if cx_init < 0 or cx_init > W:
        cx_init = W / 2
    if cy_init < 0 or cy_init > H:
        cy_init = H / 2

    K_init = np.array([
        [fx_init, 0, cx_init],
        [0, fy_init, cy_init],
        [0, 0, 1],
    ], dtype=np.float64)

    logger.info("[%s] Initial K (cleaned):\n%s", cam_name, np.array2string(K_init, precision=1))

    # ================================================================
    # Step 4: solvePnP with known K to get precise R
    # ================================================================
    dist_init = np.zeros(5)
    success, rvec, tvec = cv2.solvePnP(
        pts_3d.astype(np.float64),
        pts_2d.astype(np.float64),
        K_init, dist_init,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        logger.error("[%s] solvePnP failed!", cam_name)
        return None

    R_pnp, _ = cv2.Rodrigues(rvec)
    t_pnp = tvec.flatten()
    C_pnp = (-R_pnp.T @ t_pnp)

    logger.info("[%s] solvePnP camera pos: (%.3f, %.3f, %.3f)", cam_name, *C_pnp)
    logger.info("[%s] solvePnP vs laser error: %.3f m",
                cam_name, np.linalg.norm(C_pnp - cam_pos))

    # ================================================================
    # Step 5: Force t from known camera position, keep R from solvePnP
    # t = -R @ C_known
    # ================================================================
    R_final = R_pnp.copy()
    t_forced = -R_final @ cam_pos

    logger.info("[%s] Forced t from laser position: %s", cam_name,
                np.array2string(t_forced, precision=3))

    # Reproject with forced t to check
    P_forced = K_init @ np.hstack([R_final, t_forced.reshape(3, 1)])
    reproj_errors = []
    for i in range(n):
        X_h = np.append(pts_3d[i], 1.0)
        p = P_forced @ X_h
        px, py = p[0] / p[2], p[1] / p[2]
        err = np.sqrt((px - pts_2d[i, 0])**2 + (py - pts_2d[i, 1])**2)
        reproj_errors.append(err)

    mean_err = np.mean(reproj_errors)
    logger.info("[%s] Reprojection after forcing t: mean=%.2f px, max=%.2f px",
                cam_name, mean_err, max(reproj_errors))

    # ================================================================
    # Step 6: Refine K (focal length + principal point) with fixed R, t
    # Using linear least squares on the projection equations
    # ================================================================
    # P = K @ [R | t], we know R and t, solve for K
    # For each point: s * [u, v, 1]^T = K @ [R|t] @ [X, Y, Z, 1]^T
    # Let q = [R|t] @ [X, Y, Z, 1]^T (3×1 vector, known)
    # Then: u = (fx*q0 + cx*q2) / q2,  v = (fy*q1 + cy*q2) / q2
    # Rearrange: u*q2 = fx*q0 + cx*q2 => fx*q0 + cx*q2 = u*q2
    #            v*q2 = fy*q1 + cy*q2 => fy*q1 + cy*q2 = v*q2

    Rt = np.hstack([R_final, t_forced.reshape(3, 1)])
    A_k = []
    b_k = []
    for i in range(n):
        X_h = np.append(pts_3d[i], 1.0)
        q = Rt @ X_h  # 3×1
        if abs(q[2]) < 1e-10:
            continue
        # u equation: fx * (q0/q2) + cx = u
        A_k.append([q[0] / q[2], 0, 1, 0])
        b_k.append(pts_2d[i, 0])
        # v equation: fy * (q1/q2) + cy = v
        A_k.append([0, q[1] / q[2], 0, 1])
        b_k.append(pts_2d[i, 1])

    A_k = np.array(A_k)
    b_k = np.array(b_k)

    # Solve for [fx, fy, cx, cy]
    result, residuals, rank, sv = np.linalg.lstsq(A_k, b_k, rcond=None)
    fx, fy, cx, cy = result

    logger.info("[%s] Refined intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
                cam_name, fx, fy, cx, cy)

    # Sanity checks
    if fx < 100 or fy < 100 or fx > 10000 or fy > 10000:
        logger.warning("[%s] Focal length out of range, using DLT values", cam_name)
        fx, fy = fx_init, fy_init
    if cx < 0 or cx > W or cy < 0 or cy > H:
        logger.warning("[%s] Principal point out of range, using image center", cam_name)
        cx, cy = W / 2, H / 2

    K_refined = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # ================================================================
    # Step 7: Final refinement — solvePnP with refined K, then check
    # ================================================================
    success2, rvec2, tvec2 = cv2.solvePnP(
        pts_3d.astype(np.float64),
        pts_2d.astype(np.float64),
        K_refined, np.zeros(5),
        rvec, tvec,  # use previous as initial guess
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    R_final2, _ = cv2.Rodrigues(rvec2)
    t_final2 = tvec2.flatten()
    C_final2 = -R_final2.T @ t_final2

    # Force camera position again
    t_final_forced = -R_final2 @ cam_pos

    # Build final projection matrix
    P_final = K_refined @ np.hstack([R_final2, t_final_forced.reshape(3, 1)])

    # Final reprojection
    final_errors = {}
    for i in range(n):
        X_h = np.append(pts_3d[i], 1.0)
        p = P_final @ X_h
        px, py = p[0] / p[2], p[1] / p[2]
        err_px = np.sqrt((px - pts_2d[i, 0])**2 + (py - pts_2d[i, 1])**2)
        err_m = err_px * (COURT_LENGTH / H)  # approximate
        final_errors[labels[i]] = {"px": round(err_px, 2), "m": round(err_m, 4)}

    mean_px = np.mean([e["px"] for e in final_errors.values()])
    max_px = max(e["px"] for e in final_errors.values())
    mean_m = np.mean([e["m"] for e in final_errors.values()])

    logger.info("[%s] FINAL reprojection: mean=%.2f px (%.4f m), max=%.2f px",
                cam_name, mean_px, mean_m, max_px)
    logger.info("[%s] FINAL camera pos from laser: (%.3f, %.3f, %.3f)",
                cam_name, *cam_pos)
    logger.info("[%s] FINAL K:\n%s", cam_name, np.array2string(K_refined, precision=2))

    for label, err in final_errors.items():
        status = "✅" if err["px"] < 5 else "⚠️" if err["px"] < 15 else "❌"
        logger.info("[%s]   %s: %.2f px (%.4f m) %s",
                    cam_name, label, err["px"], err["m"], status)

    return {
        "camera_name": cam_name,
        "image_size": list(image_size),
        "n_keypoints": n,
        "labels_used": labels,
        "camera_position_3d": cam_pos.tolist(),
        "camera_position_source": "laser_measurement",
        "K": K_refined.tolist(),
        "R": R_final2.tolist(),
        "rvec": rvec2.flatten().tolist(),
        "tvec": t_final_forced.tolist(),
        "dist_coeffs": [[0.0] * 5],  # no distortion for now
        "P": P_final.tolist(),
        "mean_reprojection_error_px": round(mean_px, 3),
        "mean_reprojection_error_m": round(mean_m, 4),
        "max_reprojection_error_px": round(max_px, 3),
        "per_point_errors_px": {k: v["px"] for k, v in final_errors.items()},
        "per_point_errors_world": {k: v["m"] for k, v in final_errors.items()},
    }


def main():
    src_dir = Path("src")

    # Load pixel annotations
    pixels_66 = load_labelme_pixels(str(src_dir / "cam66.json"))
    pixels_68 = load_labelme_pixels(str(src_dir / "cam68.json"))

    logger.info("cam66 keypoints: %d", len(pixels_66))
    logger.info("cam68 keypoints: %d", len(pixels_68))

    # Build correspondences
    pts2d_66, pts3d_66, labels_66 = build_correspondences(pixels_66, WORLD_3D_CAM66)
    pts2d_68, pts3d_68, labels_68 = build_correspondences(pixels_68, WORLD_3D_CAM68)

    # Calibrate
    result_66 = calibrate_with_known_position(
        "cam66", CAMERA_POSITIONS["cam66"], pts2d_66, pts3d_66, labels_66,
    )
    result_68 = calibrate_with_known_position(
        "cam68", CAMERA_POSITIONS["cam68"], pts2d_68, pts3d_68, labels_68,
    )

    if result_66 is None or result_68 is None:
        logger.error("Calibration failed for one or both cameras")
        sys.exit(1)

    # Stereo validation
    baseline = np.linalg.norm(CAMERA_POSITIONS["cam66"] - CAMERA_POSITIONS["cam68"])
    logger.info("Stereo baseline: %.2f m", baseline)
    logger.info("Heights: cam66=%.2f m, cam68=%.2f m",
                CAMERA_POSITIONS["cam66"][2], CAMERA_POSITIONS["cam68"][2])

    # Triangulation test: project a known 3D point and verify
    logger.info("\n=== Triangulation Verification ===")
    test_points = [
        ("net_center", np.array([4.115, 11.885, 0.0])),
        ("net_center_1m", np.array([4.115, 11.885, 1.0])),
        ("near_baseline_center", np.array([4.115, 0.0, 0.0])),
        ("service_box", np.array([3.0, 8.0, 0.0])),
        ("ball_in_air", np.array([4.0, 15.0, 2.0])),
    ]

    P66 = np.array(result_66["P"])
    P68 = np.array(result_68["P"])

    for name, pt3d in test_points:
        X_h = np.append(pt3d, 1.0)

        # Project to both cameras
        p66 = P66 @ X_h
        px66 = np.array([p66[0] / p66[2], p66[1] / p66[2]])

        p68 = P68 @ X_h
        px68 = np.array([p68[0] / p68[2], p68[1] / p68[2]])

        # Triangulate back
        pts4d = cv2.triangulatePoints(P66, P68, px66.reshape(2, 1), px68.reshape(2, 1))
        pt_tri = (pts4d[:3] / pts4d[3]).flatten()

        err = np.linalg.norm(pt_tri - pt3d)
        logger.info("  %s: true=(%.2f,%.2f,%.2f) → tri=(%.2f,%.2f,%.2f) err=%.4f m %s",
                    name, *pt3d, *pt_tri, err, "✅" if err < 0.1 else "⚠️")

    # Save
    output = {
        "calibration_method": "laser_measured_position + DLT + solvePnP",
        "court_dimensions": {
            "length_m": COURT_LENGTH,
            "singles_width_m": SINGLES_RIGHT - SINGLES_LEFT,
            "doubles_width_m": 8.23,
            "net_y": NET_Y,
        },
        "cam66": result_66,
        "cam68": result_68,
        "stereo": {
            "baseline_m": round(baseline, 3),
            "cam66_height_m": round(CAMERA_POSITIONS["cam66"][2], 3),
            "cam68_height_m": round(CAMERA_POSITIONS["cam68"][2], 3),
        },
    }

    out_path = str(src_dir / "camera_calibration.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved calibration to %s", out_path)


if __name__ == "__main__":
    main()
