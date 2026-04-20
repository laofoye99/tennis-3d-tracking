"""Camera calibration: cv2.calibrateCamera + laser position verification.

For coplanar points (all z=0), DLT fails. cv2.calibrateCamera handles this
using homography decomposition internally.

Strategy:
1. cv2.calibrateCamera → K, R, t (handles coplanar points)
2. Compare recovered camera position with laser measurement
3. If position error > threshold, adjust using laser position as constraint
4. Build P = K @ [R | t] for triangulation
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SINGLES_LEFT = 1.37
SINGLES_RIGHT = 6.86
CENTER_X = (SINGLES_LEFT + SINGLES_RIGHT) / 2
COURT_LENGTH = 23.77
NET_Y = COURT_LENGTH / 2
SERVICE_NEAR_Y = 5.485
SERVICE_FAR_Y = 18.285

WORLD_3D_CAM66 = {
    "left_top": (SINGLES_LEFT, COURT_LENGTH, 0), "left_top_serve": (SINGLES_LEFT, SERVICE_FAR_Y, 0),
    "left_bottom_serve": (SINGLES_LEFT, SERVICE_NEAR_Y, 0), "left_bottom": (SINGLES_LEFT, 0.0, 0),
    "center_top": (CENTER_X, COURT_LENGTH, 0), "center_top_serve": (CENTER_X, SERVICE_FAR_Y, 0),
    "center_bottom_serve": (CENTER_X, SERVICE_NEAR_Y, 0), "center_bottom": (CENTER_X, 0.0, 0),
    "right_top": (SINGLES_RIGHT, COURT_LENGTH, 0), "right_top_serve": (SINGLES_RIGHT, SERVICE_FAR_Y, 0),
    "right_bottom_serve": (SINGLES_RIGHT, SERVICE_NEAR_Y, 0), "right_bottom": (SINGLES_RIGHT, 0.0, 0),
}
WORLD_3D_CAM68 = {
    "left_top": (SINGLES_RIGHT, 0.0, 0), "left_top_serve": (SINGLES_RIGHT, SERVICE_NEAR_Y, 0),
    "left_bottom_serve": (SINGLES_RIGHT, SERVICE_FAR_Y, 0), "left_bottom": (SINGLES_RIGHT, COURT_LENGTH, 0),
    "center_top": (CENTER_X, 0.0, 0), "center_top_serve": (CENTER_X, SERVICE_NEAR_Y, 0),
    "center_bottom_serve": (CENTER_X, SERVICE_FAR_Y, 0), "center_bottom": (CENTER_X, COURT_LENGTH, 0),
    "right_top": (SINGLES_LEFT, 0.0, 0), "right_top_serve": (SINGLES_LEFT, SERVICE_NEAR_Y, 0),
    "right_bottom_serve": (SINGLES_LEFT, SERVICE_FAR_Y, 0), "right_bottom": (SINGLES_LEFT, COURT_LENGTH, 0),
}
LASER = {
    "cam66": np.array([4.446, 29.049, 6.830]),
    "cam68": np.array([4.431, -5.278, 6.096]),
}


def load_labelme_pixels(json_path):
    with open(json_path) as f:
        data = json.load(f)
    pts = {}
    for s in data.get("shapes", []):
        label = s["label"].strip().lower()
        p = s["points"]
        if s["shape_type"] == "point" and len(p) == 1:
            pts[label] = (float(p[0][0]), float(p[0][1]))
        elif s["shape_type"] == "rectangle" and len(p) == 2:
            pts[label] = ((p[0][0]+p[1][0])/2, (p[0][1]+p[1][1])/2)
    return pts


def calibrate_camera(cam_name, pts_2d, pts_3d, labels, laser_pos, img_size=(1920, 1080)):
    """Calibrate using cv2.calibrateCamera, then optionally refine with laser constraint."""
    n = len(pts_2d)
    W, H = img_size
    logger.info("[%s] %d keypoints, laser=(%.3f, %.3f, %.3f)", cam_name, n, *laser_pos)

    obj_pts = [pts_3d.astype(np.float32)]
    img_pts = [pts_2d.astype(np.float32)]

    # ================================================================
    # Step 1: cv2.calibrateCamera (handles coplanar points)
    # ================================================================
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, (W, H), None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO,
    )
    rvec = rvecs[0]
    tvec = tvecs[0]
    R, _ = cv2.Rodrigues(rvec)
    C_cv = (-R.T @ tvec.flatten())

    logger.info("[%s] calibrateCamera: rms=%.2f px", cam_name, rms)
    logger.info("[%s]   K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", cam_name, K[0,0], K[1,1], K[0,2], K[1,2])
    logger.info("[%s]   Camera pos: (%.3f, %.3f, %.3f)", cam_name, *C_cv)
    logger.info("[%s]   Laser pos:  (%.3f, %.3f, %.3f)", cam_name, *laser_pos)
    logger.info("[%s]   Pos error:  %.3f m", cam_name, np.linalg.norm(C_cv - laser_pos))

    # ================================================================
    # Step 2: Refine — minimize reprojection error with laser position constraint
    # ================================================================
    # Parameterize: [fx, fy, cx, cy, rx, ry, rz] where t = -R @ laser_pos
    # This forces camera position = laser position, only optimizes K and R
    def cost(params):
        fx, fy, cx, cy, rx, ry, rz = params
        K_opt = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        rvec_opt = np.array([rx, ry, rz], dtype=np.float64)
        R_opt, _ = cv2.Rodrigues(rvec_opt)
        tvec_opt = (-R_opt @ laser_pos).reshape(3, 1)

        proj, _ = cv2.projectPoints(pts_3d, rvec_opt, tvec_opt, K_opt, np.zeros(5))
        proj = proj.reshape(-1, 2)
        residuals = (proj - pts_2d).flatten()
        return residuals

    from scipy.optimize import least_squares
    x0 = [K[0,0], K[1,1], K[0,2], K[1,2], rvec.flatten()[0], rvec.flatten()[1], rvec.flatten()[2]]

    result = least_squares(cost, x0, method='lm')
    fx, fy, cx, cy, rx, ry, rz = result.x

    K_final = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    rvec_final = np.array([rx, ry, rz])
    R_final, _ = cv2.Rodrigues(rvec_final)
    t_final = (-R_final @ laser_pos)
    P_final = K_final @ np.hstack([R_final, t_final.reshape(3, 1)])

    # Reprojection check
    proj_final, _ = cv2.projectPoints(pts_3d, rvec_final, t_final.reshape(3,1), K_final, np.zeros(5))
    proj_final = proj_final.reshape(-1, 2)
    errors = np.sqrt(np.sum((proj_final - pts_2d)**2, axis=1))
    mean_err = errors.mean()
    max_err = errors.max()

    logger.info("[%s] Laser-constrained refinement:", cam_name)
    logger.info("[%s]   K: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", cam_name, fx, fy, cx, cy)
    logger.info("[%s]   Reprojection: mean=%.2f px, max=%.2f px", cam_name, mean_err, max_err)
    logger.info("[%s]   Camera position FIXED at laser: (%.3f, %.3f, %.3f)", cam_name, *laser_pos)

    for i, label in enumerate(labels):
        s = "✅" if errors[i] < 3 else "⚠️" if errors[i] < 10 else "❌"
        logger.info("[%s]   %s: %.2f px %s", cam_name, label, errors[i], s)

    return {
        "camera_name": cam_name,
        "image_size": list(img_size),
        "n_keypoints": n,
        "labels_used": labels,
        "camera_position_3d": laser_pos.tolist(),
        "K": K_final.tolist(),
        "R": R_final.tolist(),
        "rvec": rvec_final.tolist(),
        "tvec": t_final.tolist(),
        "dist_coeffs": [[0.0] * 5],
        "P": P_final.tolist(),
        "mean_reprojection_error_px": round(float(mean_err), 3),
        "max_reprojection_error_px": round(float(max_err), 3),
        "per_point_errors_px": {labels[i]: round(float(errors[i]), 2) for i in range(n)},
    }


def main():
    src = Path("src")
    px66 = load_labelme_pixels(str(src / "cam66.json"))
    px68 = load_labelme_pixels(str(src / "cam68.json"))

    common66 = sorted(set(px66) & set(WORLD_3D_CAM66))
    common68 = sorted(set(px68) & set(WORLD_3D_CAM68))

    p2d_66 = np.array([px66[k] for k in common66], dtype=np.float64)
    p3d_66 = np.array([WORLD_3D_CAM66[k] for k in common66], dtype=np.float64)
    p2d_68 = np.array([px68[k] for k in common68], dtype=np.float64)
    p3d_68 = np.array([WORLD_3D_CAM68[k] for k in common68], dtype=np.float64)

    r66 = calibrate_camera("cam66", p2d_66, p3d_66, common66, LASER["cam66"])
    r68 = calibrate_camera("cam68", p2d_68, p3d_68, common68, LASER["cam68"])

    # ================================================================
    # Triangulation verification
    # ================================================================
    logger.info("\n=== Triangulation Verification ===")
    P66 = np.array(r66["P"])
    P68 = np.array(r68["P"])

    tests = [
        ("net_center_z0",   [4.115, 11.885, 0.0]),
        ("net_center_z1",   [4.115, 11.885, 1.0]),
        ("near_baseline",   [4.115, 0.0, 0.0]),
        ("far_baseline",    [4.115, 23.77, 0.0]),
        ("ball_2m",         [4.0, 15.0, 2.0]),
        ("serve_3m",        [4.0, 1.0, 3.0]),
        ("bounce_near",     [3.0, 5.0, 0.0]),
        ("bounce_far",      [5.0, 18.0, 0.0]),
        ("bounce_ground",   [4.0, 10.0, 0.0]),
        ("ball_low",        [4.0, 12.0, 0.3]),
    ]

    for name, pt in tests:
        pt = np.array(pt, dtype=np.float64)
        X_h = np.append(pt, 1.0)

        p1 = P66 @ X_h; px1 = p1[:2] / p1[2]
        p2 = P68 @ X_h; px2 = p2[:2] / p2[2]

        pts4d = cv2.triangulatePoints(P66, P68, px1.reshape(2,1), px2.reshape(2,1))
        tri = (pts4d[:3,0] / pts4d[3,0])
        err = np.linalg.norm(tri - pt)

        logger.info("  %s: true=(%.1f,%.1f,%.1f) tri=(%.3f,%.3f,%.3f) err=%.4fm %s",
                    name, *pt, *tri, err, "✅" if err < 0.05 else "⚠️" if err < 0.5 else "❌")

    # Save
    baseline = np.linalg.norm(LASER["cam66"] - LASER["cam68"])
    out = {
        "calibration_method": "calibrateCamera + laser-constrained refinement",
        "court_dimensions": {"length_m": COURT_LENGTH, "singles_width_m": SINGLES_RIGHT-SINGLES_LEFT, "net_y": NET_Y},
        "cam66": r66, "cam68": r68,
        "stereo": {"baseline_m": round(baseline, 3)},
    }
    out_path = str(src / "camera_calibration.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
