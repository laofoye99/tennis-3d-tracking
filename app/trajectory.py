"""Physics-constrained 3D trajectory reconstruction with robust outlier rejection.

Three-stage approach that works without frame-level synchronization:

Stage 0: Per-camera detection cleaning
    - Confidence filtering, court bounds checking, velocity consistency,
      isolated point removal. Removes false detections before triangulation.

Stage 1: Auto time-offset via interpolated triangulation
    - Sweep dt using trimmed-mean ray distance (robust to outliers).
    - Post-triangulation filtering by ray distance and physical bounds.

Stage 2: RANSAC spatial parabolic fit (frame-rate independent)
    - Fit X(Y) = linear, Z(Y) = quadratic using RANSAC to reject outliers.
    - Piecewise fitting with robust bounce detection.
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

GRAVITY = np.array([0.0, 0.0, -9.81])

# Court dimensions (meters) — singles sidelines
_COURT_X_MIN = -4.115  # V2
_COURT_X_MAX = 4.115  # V2
_COURT_Y = 11.89  # V2: half length
_NET_Y = 0.0  # V2: net at origin


# ========== Stage 0: Per-camera detection cleaning ==========


def clean_detections(
    dets: list[tuple],
    fps: float,
    H_i2w: np.ndarray,
    min_confidence: float = 3.0,
    court_margin: float = 5.0,
    max_speed_world: float = 100.0,
    max_gap_frames: int = 10,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Clean detections from a single camera before triangulation.

    Applies filters in order: confidence, court bounds, velocity, isolation.

    Args:
        dets: Detections as (frame_idx, pixel_x, pixel_y, confidence) 4-tuples
              or (frame_idx, pixel_x, pixel_y) 3-tuples.
        fps: Frame rate of the camera.
        H_i2w: Image-to-world homography matrix (3x3).
        min_confidence: Minimum blob confidence to keep.
        court_margin: Margin around court for bounds check (meters).
        max_speed_world: Maximum plausible ball speed in world coords (m/s).
        max_gap_frames: Max frames between neighbors for isolation filter.

    Returns:
        (cleaned_dets, stats) where cleaned_dets is list of (frame, px, py)
        and stats reports how many removed at each stage.
    """
    if not dets:
        return [], {"input": 0}

    has_conf = len(dets[0]) >= 4
    stats = {"input": len(dets)}

    # Step 1: Confidence filter
    if has_conf:
        filtered = [(d[0], d[1], d[2], d[3]) for d in dets if d[3] >= min_confidence]
        stats["after_confidence"] = len(filtered)
        stats["removed_confidence"] = stats["input"] - len(filtered)
    else:
        filtered = [(d[0], d[1], d[2], 999.0) for d in dets]
        stats["after_confidence"] = len(filtered)
        stats["removed_confidence"] = 0

    if not filtered:
        return [], stats

    # Step 2: Court bounds filter (world-space)
    x_min, x_max = _COURT_X_MIN - court_margin, _COURT_X_MAX + court_margin
    y_min, y_max = -court_margin, _COURT_Y + court_margin
    bounded = []
    for d in filtered:
        wx, wy = _pixel_to_world(H_i2w, d[1], d[2])
        if x_min <= wx <= x_max and y_min <= wy <= y_max:
            bounded.append(d)
    stats["after_bounds"] = len(bounded)
    stats["removed_bounds"] = stats["after_confidence"] - len(bounded)

    if not bounded:
        return [], stats

    # Step 3: Velocity consistency (world-space)
    # Remove detections that imply impossibly fast movement
    max_disp_per_frame = max_speed_world / fps
    velocity_ok = [bounded[0]]
    for i in range(1, len(bounded)):
        prev = bounded[i - 1]
        curr = bounded[i]
        frame_gap = curr[0] - prev[0]
        if frame_gap <= 0:
            velocity_ok.append(curr)
            continue
        wx_prev, wy_prev = _pixel_to_world(H_i2w, prev[1], prev[2])
        wx_curr, wy_curr = _pixel_to_world(H_i2w, curr[1], curr[2])
        dist = np.sqrt((wx_curr - wx_prev) ** 2 + (wy_curr - wy_prev) ** 2)
        if dist / frame_gap <= max_disp_per_frame:
            velocity_ok.append(curr)
        else:
            # Keep the one with higher confidence by checking both directions
            # If the next point also agrees with curr, keep curr and drop prev
            # For simplicity, drop curr (the "jumper")
            pass
    stats["after_velocity"] = len(velocity_ok)
    stats["removed_velocity"] = stats["after_bounds"] - len(velocity_ok)

    if not velocity_ok:
        return [], stats

    # Step 4: Isolated point removal
    # A detection must have at least one neighbor within max_gap_frames
    frames = np.array([d[0] for d in velocity_ok])
    non_isolated = []
    for i, d in enumerate(velocity_ok):
        has_neighbor = False
        for j in range(max(0, i - 1), min(len(velocity_ok), i + 2)):
            if j != i and abs(frames[j] - frames[i]) <= max_gap_frames:
                has_neighbor = True
                break
        if has_neighbor:
            non_isolated.append(d)
    stats["after_isolation"] = len(non_isolated)
    stats["removed_isolation"] = stats["after_velocity"] - len(non_isolated)

    # Return as 3-tuples (frame, px, py) - drop confidence
    result = [(d[0], d[1], d[2]) for d in non_isolated]
    stats["output"] = len(result)
    return result, stats


# ========== Stage 1: Triangulation with auto time-offset ==========


def triangulate_pair(
    w1: tuple[float, float],
    w2: tuple[float, float],
    cam1_pos: list[float],
    cam2_pos: list[float],
) -> tuple[Optional[np.ndarray], float]:
    """Triangulate a 3D point from two ground-plane projections.

    Returns (midpoint_3d, ray_distance). midpoint_3d is None if rays are
    parallel. ray_distance measures triangulation quality (lower = better).
    """
    c1 = np.asarray(cam1_pos, dtype=np.float64)
    c2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
    d1, d2 = g1 - c1, g2 - c2
    w = c1 - c2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d_val = np.dot(d1, w)
    e = np.dot(d2, w)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None, 99.0
    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom
    s, t = np.clip(s, 0, 1), np.clip(t, 0, 1)
    # Re-solve iteratively
    p1f = c1 + s * d1
    t = float(np.dot(p1f - c2, d2)) / c if c > 1e-10 else t
    t = np.clip(t, 0, 1)
    p2f = c2 + t * d2
    s = float(np.dot(p2f - c1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0, 1)
    p1 = c1 + s * d1
    p2 = c2 + t * d2
    mid = (p1 + p2) / 2.0
    rd = float(np.linalg.norm(p1 - p2))
    if mid[2] < 0:
        mid[2] = 0.0
    return mid, rd


def _interpolate_detection(
    dets: list[tuple[float, float, float]],
    target_frame: float,
) -> Optional[tuple[float, float]]:
    """Linearly interpolate a pixel detection at a non-integer frame number."""
    for i in range(len(dets) - 1):
        f0, f1 = dets[i][0], dets[i + 1][0]
        if f0 <= target_frame <= f1:
            frac = (target_frame - f0) / (f1 - f0) if f1 != f0 else 0
            px = dets[i][1] + frac * (dets[i + 1][1] - dets[i][1])
            py = dets[i][2] + frac * (dets[i + 1][2] - dets[i][2])
            return px, py
    return None


def _pixel_to_world(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    """Apply homography to convert pixel to world coordinates."""
    pt = np.array([px, py, 1.0])
    r = H @ pt
    return float(r[0] / r[2]), float(r[1] / r[2])


def _eval_offset(
    dt_seconds: float,
    dets_a: list[tuple[float, float, float]],
    dets_b: list[tuple[float, float, float]],
    fps_a: float,
    fps_b: float,
    H_a: np.ndarray,
    H_b: np.ndarray,
    cam_a_pos: list[float],
    cam_b_pos: list[float],
    trim_fraction: float = 0.2,
) -> float:
    """Evaluate a time offset using trimmed-mean ray distance (robust to outliers).

    Drops the worst `trim_fraction` of ray distances before averaging.
    """
    ray_dists = []
    for fa, px_a, py_a in dets_a:
        t_a = fa / fps_a
        fb_float = (t_a - dt_seconds) * fps_b
        interp = _interpolate_detection(dets_b, fb_float)
        if interp is None:
            continue
        px_b, py_b = interp
        w_a = _pixel_to_world(H_a, px_a, py_a)
        w_b = _pixel_to_world(H_b, px_b, py_b)
        _, rd = triangulate_pair(w_a, w_b, cam_a_pos, cam_b_pos)
        ray_dists.append(rd)
    if not ray_dists:
        return 1e6
    # Trimmed mean: sort and drop top trim_fraction
    ray_dists.sort()
    n_keep = max(1, int(len(ray_dists) * (1.0 - trim_fraction)))
    return float(np.mean(ray_dists[:n_keep]))


def find_offset_and_triangulate(
    dets_a: list[tuple],
    dets_b: list[tuple],
    fps_a: float,
    fps_b: float,
    H_a: np.ndarray,
    H_b: np.ndarray,
    cam_a_pos: list[float],
    cam_b_pos: list[float],
    dt_range: float = 3.0,
    dt_steps: int = 601,
    max_ray_dist: float = 1.5,
    z_range: tuple[float, float] = (-0.5, 8.0),
) -> tuple[float, list[dict]]:
    """Find optimal time offset and triangulate all matched points.

    Args:
        dets_a: Camera A detections as (frame_idx, pixel_x, pixel_y) 3-tuples.
        dets_b: Camera B detections as (frame_idx, pixel_x, pixel_y) 3-tuples.
        fps_a: Camera A frame rate.
        fps_b: Camera B frame rate.
        H_a: Camera A image-to-world homography (3x3).
        H_b: Camera B image-to-world homography (3x3).
        cam_a_pos: Camera A 3D position [x, y, z].
        cam_b_pos: Camera B 3D position [x, y, z].
        dt_range: Search range for time offset in seconds.
        dt_steps: Number of steps in coarse grid search.
        max_ray_dist: Reject triangulated points with ray distance above this.
        z_range: Reject points with Z outside (min, max) range.

    Returns:
        (best_dt, points_3d) where points_3d is a list of dicts with keys:
        x, y, z, ray_dist, t (relative time), frame_a.
    """
    # Ensure 3-tuples for offset evaluation
    dets_a_3 = [(d[0], d[1], d[2]) for d in dets_a]
    dets_b_3 = [(d[0], d[1], d[2]) for d in dets_b]

    # Coarse sweep with trimmed-mean cost
    best_dt, best_cost = 0.0, 1e18
    for dt in np.linspace(-dt_range, dt_range, dt_steps):
        cost = _eval_offset(
            dt, dets_a_3, dets_b_3, fps_a, fps_b, H_a, H_b, cam_a_pos, cam_b_pos
        )
        if cost < best_cost:
            best_cost = cost
            best_dt = dt

    # Refine with bounded scalar optimization
    def cost_fn(dt_val):
        return _eval_offset(
            dt_val, dets_a_3, dets_b_3, fps_a, fps_b, H_a, H_b, cam_a_pos, cam_b_pos
        )

    refined = minimize_scalar(
        cost_fn, bounds=(best_dt - 0.2, best_dt + 0.2), method="bounded"
    )
    best_dt = refined.x

    # Triangulate with best offset
    t0 = dets_a_3[0][0] / fps_a if dets_a_3 else 0.0
    points_3d_raw = []
    for fa, px_a, py_a in dets_a_3:
        t_a = fa / fps_a
        fb_float = (t_a - best_dt) * fps_b
        interp = _interpolate_detection(dets_b_3, fb_float)
        if interp is None:
            continue
        px_b, py_b = interp
        w_a = _pixel_to_world(H_a, px_a, py_a)
        w_b = _pixel_to_world(H_b, px_b, py_b)
        pt, rd = triangulate_pair(w_a, w_b, cam_a_pos, cam_b_pos)
        if pt is not None:
            points_3d_raw.append({
                "t": t_a - t0,
                "x": float(pt[0]),
                "y": float(pt[1]),
                "z": float(pt[2]),
                "ray_dist": rd,
                "frame_a": int(fa),
                # Per-camera debug: world-space ground projections
                "cam_a_world": [round(w_a[0], 3), round(w_a[1], 3)],
                "cam_b_world": [round(w_b[0], 3), round(w_b[1], 3)],
                "cam_a_pixel": [round(px_a, 1), round(py_a, 1)],
                "cam_b_pixel": [round(px_b, 1), round(py_b, 1)],
            })

    # Post-triangulation filtering
    points_3d = []
    n_rejected_ray = 0
    n_rejected_physics = 0
    for p in points_3d_raw:
        if p["ray_dist"] > max_ray_dist:
            n_rejected_ray += 1
            continue
        if p["z"] < z_range[0] or p["z"] > z_range[1]:
            n_rejected_physics += 1
            continue
        points_3d.append(p)

    logger.info(
        "Auto offset: dt=%.4fs (~%.1f frames), %d raw -> %d filtered "
        "(rejected: %d ray_dist, %d physics), mean_rd=%.4fm",
        best_dt,
        best_dt * fps_a,
        len(points_3d_raw),
        len(points_3d),
        n_rejected_ray,
        n_rejected_physics,
        np.mean([p["ray_dist"] for p in points_3d]) if points_3d else 0,
    )
    return best_dt, points_3d


# ========== Stage 2: Spatial parabolic fit ==========


def fit_spatial_parabola(
    points: list[dict],
) -> Optional[dict]:
    """Fit trajectory spatially: X(Y) = linear, Z(Y) = quadratic.

    For a ballistic trajectory:
        X = ax*Y + bx
        Z = az*Y^2 + bz*Y + cz  (where az = -g/(2*vy^2))

    This is INDEPENDENT of frame rate / timing!
    Uses weighted least squares with weights based on ray_dist.

    Returns dict with coefficients, velocity estimate, residuals, etc.
    """
    n = len(points)
    if n < 3:
        return None

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    zs = np.array([p["z"] for p in points])

    # Weights: inverse of ray distance
    rds = np.array([p.get("ray_dist", 0.01) for p in points])
    weights = 1.0 / np.clip(rds, 0.01, None)
    weights = weights / weights.sum() * n
    W_sqrt = np.diag(np.sqrt(weights))

    # X(Y) = ax*Y + bx
    A_xy = np.column_stack([ys, np.ones(n)])
    sol_x, _, _, _ = np.linalg.lstsq(W_sqrt @ A_xy, W_sqrt @ xs, rcond=None)
    ax, bx = sol_x

    # Z(Y) = az*Y^2 + bz*Y + cz
    A_zy = np.column_stack([ys**2, ys, np.ones(n)])
    sol_z, _, _, _ = np.linalg.lstsq(W_sqrt @ A_zy, W_sqrt @ zs, rcond=None)
    az, bz, cz = sol_z

    # Recover velocity from spatial coefficients: az = -g / (2*vy^2)
    g = 9.81
    vy = np.sqrt(g / (-2 * az)) if az < -1e-6 else 25.0
    vy = float(np.clip(vy, 1.0, 80.0))  # Clamp to physical range
    vx = ax * vy
    y0 = ys[0]
    vz = vy * (2 * az * y0 + bz)
    speed = float(np.linalg.norm([vx, vy, vz]))

    # Compute residuals
    residuals = []
    fitted_pts = []
    for p in points:
        y = p["y"]
        x_fit = ax * y + bx
        z_fit = az * y**2 + bz * y + cz
        err = float(np.sqrt((x_fit - p["x"]) ** 2 + (z_fit - p["z"]) ** 2))
        residuals.append(err)
        fitted_pts.append({"y": float(y), "x": float(x_fit), "z": float(z_fit)})

    return {
        "ax": float(ax),
        "bx": float(bx),
        "az": float(az),
        "bz": float(bz),
        "cz": float(cz),
        "v0": [float(vx), float(vy), float(vz)],
        "speed_ms": speed,
        "speed_kmh": speed * 3.6,
        "y_min": float(np.min(ys)),
        "y_max": float(np.max(ys)),
        "mean_error": float(np.mean(residuals)),
        "max_error": float(np.max(residuals)),
        "residuals": residuals,
        "fitted_points": fitted_pts,
    }


def _compute_spatial_error(point: dict, fit: dict) -> float:
    """Compute spatial distance from a point to the fitted curve."""
    y = point["y"]
    x_fit = fit["ax"] * y + fit["bx"]
    z_fit = fit["az"] * y ** 2 + fit["bz"] * y + fit["cz"]
    return float(np.sqrt((x_fit - point["x"]) ** 2 + (z_fit - point["z"]) ** 2))


def fit_spatial_parabola_ransac(
    points: list[dict],
    n_iterations: int = 300,
    inlier_threshold: float = 0.5,
    min_inlier_ratio: float = 0.4,
    max_speed_kmh: float = 350.0,
) -> tuple[Optional[dict], list[int]]:
    """RANSAC fit of spatial parabola: X(Y) = linear, Z(Y) = quadratic.

    Robustly fits trajectory by sampling random subsets, fitting a parabola,
    validating physics, and selecting the model with the most inliers.

    Args:
        points: Triangulated 3D points with keys x, y, z, ray_dist.
        n_iterations: Number of RANSAC iterations.
        inlier_threshold: Max spatial error (meters) for a point to be an inlier.
        min_inlier_ratio: Minimum fraction of points that must be inliers.
        max_speed_kmh: Maximum plausible ball speed for physics validation.

    Returns:
        (fit, inlier_indices) where fit is from fit_spatial_parabola on inliers,
        or (None, []) if RANSAC fails.
    """
    n = len(points)
    if n < 4:
        # Not enough for RANSAC, fall back to direct fit
        fit = fit_spatial_parabola(points)
        return fit, list(range(n)) if fit else (None, [])

    min_sample = 4  # Minimum points for X(Y)=linear + Z(Y)=quadratic
    best_inliers: list[int] = []
    rng = np.random.default_rng(42)  # Deterministic for reproducibility

    for _ in range(n_iterations):
        # Sample random subset
        sample_idx = rng.choice(n, min_sample, replace=False)
        sample = [points[i] for i in sample_idx]

        # Fit on sample
        fit = fit_spatial_parabola(sample)
        if fit is None:
            continue

        # Physics validation: az must be negative (concave down for gravity)
        if fit["az"] >= 0:
            continue

        # Physics validation: speed must be plausible
        if fit["speed_kmh"] > max_speed_kmh:
            continue

        # Count inliers
        inliers = []
        for i, p in enumerate(points):
            err = _compute_spatial_error(p, fit)
            if err < inlier_threshold:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    # Check minimum inlier ratio
    if len(best_inliers) < max(3, int(n * min_inlier_ratio)):
        logger.warning(
            "RANSAC: only %d/%d inliers (%.0f%%), falling back to direct fit",
            len(best_inliers), n, 100 * len(best_inliers) / n,
        )
        fit = fit_spatial_parabola(points)
        return fit, list(range(n)) if fit else (None, [])

    # Refit with all inliers
    inlier_points = [points[i] for i in best_inliers]
    final_fit = fit_spatial_parabola(inlier_points)

    if final_fit is None:
        return None, []

    # Re-evaluate inliers with the refined fit (may capture more)
    refined_inliers = []
    for i, p in enumerate(points):
        err = _compute_spatial_error(p, final_fit)
        if err < inlier_threshold:
            refined_inliers.append(i)

    if len(refined_inliers) > len(best_inliers):
        inlier_points = [points[i] for i in refined_inliers]
        final_fit = fit_spatial_parabola(inlier_points)
        best_inliers = refined_inliers

    logger.info(
        "RANSAC: %d/%d inliers (%.0f%%), mean_err=%.4fm, speed=%.0fkm/h",
        len(best_inliers), n, 100 * len(best_inliers) / n,
        final_fit["mean_error"] if final_fit else 0,
        final_fit["speed_kmh"] if final_fit else 0,
    )
    return final_fit, best_inliers


def _detect_bounce(points: list[dict], z_threshold: float = 0.4) -> Optional[int]:
    """Find bounce point: local Z minimum with rising Z after."""
    zs = [p["z"] for p in points]
    if len(zs) < 5:
        return None
    min_idx = int(np.argmin(zs))
    if zs[min_idx] < z_threshold and 1 < min_idx < len(zs) - 2:
        if zs[min_idx + 1] > zs[min_idx] or zs[min_idx + 2] > zs[min_idx]:
            return min_idx
    return None


def _detect_bounce_robust(
    points: list[dict],
    single_fit: Optional[dict],
    z_threshold: float = 0.5,
    min_segment: int = 3,
) -> Optional[int]:
    """Detect bounce using residuals from single-parabola fit.

    A bounce manifests as a region where the single-parabola fit has high
    residuals near the ground (low Z). We find the best split point where
    two-segment fitting significantly improves over single-segment.

    Falls back to simple Z-minimum detection if residual method fails.
    """
    n = len(points)
    if n < 2 * min_segment:
        return None

    # Method 1: Find Z minimum in low-Z region (improved version)
    zs = [p["z"] for p in points]

    # Find all local minima in Z
    candidates = []
    for i in range(1, n - 1):
        if zs[i] < z_threshold and zs[i] <= zs[i - 1] and zs[i] <= zs[i + 1]:
            # Verify Z rises after (within next 3 points)
            rises_after = any(
                zs[j] > zs[i] + 0.05
                for j in range(i + 1, min(i + 4, n))
            )
            # Verify Z was higher before (within prev 3 points)
            higher_before = any(
                zs[j] > zs[i] + 0.05
                for j in range(max(0, i - 3), i)
            )
            if rises_after and higher_before:
                candidates.append(i)

    if not candidates:
        # Fallback to simple argmin
        return _detect_bounce(points, z_threshold)

    # Method 2: Among candidates, pick the one that best splits the trajectory
    # (lowest combined fitting error for two segments)
    if single_fit is None:
        return candidates[0] if candidates else None

    best_idx = None
    best_improvement = 0.0
    single_error = single_fit["mean_error"]

    for idx in candidates:
        if idx < min_segment or n - idx < min_segment:
            continue
        pre_fit = fit_spatial_parabola(points[: idx + 1])
        post_fit = fit_spatial_parabola(points[idx:])
        if pre_fit is None or post_fit is None:
            continue
        # Weighted average error of two segments
        n_pre = idx + 1
        n_post = n - idx
        two_seg_error = (
            pre_fit["mean_error"] * n_pre + post_fit["mean_error"] * n_post
        ) / (n_pre + n_post)
        improvement = single_error - two_seg_error
        if improvement > best_improvement:
            best_improvement = improvement
            best_idx = idx

    # Only accept bounce if two-segment fit is meaningfully better
    if best_idx is not None and best_improvement > 0.05:
        return best_idx

    # Fallback: use the first candidate
    return candidates[0] if candidates else None


def _detect_all_bounces(
    points: list[dict],
    z_threshold: float = 0.5,
    min_segment: int = 3,
) -> list[int]:
    """Detect ALL bounce points (local Z minima) in the trajectory.

    A bounce is a local Z minimum below z_threshold where Z rises after and
    was higher before. Returns sorted list of indices into points.
    """
    n = len(points)
    if n < 2 * min_segment:
        return []

    zs = [p["z"] for p in points]
    bounces = []

    for i in range(1, n - 1):
        if zs[i] >= z_threshold:
            continue
        if zs[i] > zs[i - 1] or zs[i] > zs[i + 1]:
            continue
        # Verify Z rises after (within next 3 points)
        rises_after = any(
            zs[j] > zs[i] + 0.05 for j in range(i + 1, min(i + 4, n))
        )
        # Verify Z was higher before (within prev 3 points)
        higher_before = any(
            zs[j] > zs[i] + 0.05 for j in range(max(0, i - 3), i)
        )
        if rises_after and higher_before:
            bounces.append(i)

    # Merge bounces that are too close (< min_segment apart) — keep the lowest Z
    if len(bounces) > 1:
        merged = [bounces[0]]
        for b in bounces[1:]:
            if b - merged[-1] < min_segment:
                # Keep the one with lower Z
                if zs[b] < zs[merged[-1]]:
                    merged[-1] = b
            else:
                merged.append(b)
        bounces = merged

    return bounces


def _segment_strokes(
    points: list[dict],
    bounce_indices: list[int],
) -> list[dict]:
    """Split trajectory into individual strokes at bounce points.

    Each stroke is a ball flight arc between two bounces (or from start to
    first bounce, or from last bounce to end).

    Returns list of stroke dicts with:
        - points: subset of trajectory points
        - start_idx, end_idx: indices into original points list
        - bounce_before: bounce position (if stroke starts at a bounce)
        - bounce_after: bounce position (if stroke ends at a bounce)
        - fit: spatial parabola fit for this stroke
        - speed_kmh: ball speed for this stroke
        - landing_point: where Z=0 for this stroke
    """
    if not bounce_indices:
        return []

    strokes = []
    n = len(points)

    # Build segment boundaries: [0, b0, b1, ..., bN, n-1]
    boundaries = [0] + bounce_indices + [n - 1]

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]

        # Include the boundary points in the segment
        seg_points = points[seg_start : seg_end + 1]
        if len(seg_points) < 3:
            continue

        # Fit parabola to this segment
        if len(seg_points) >= 4:
            fit, _ = fit_spatial_parabola_ransac(
                seg_points, n_iterations=100, min_inlier_ratio=0.5
            )
        else:
            fit = fit_spatial_parabola(seg_points)

        if fit is None:
            continue

        # Compute landing point for this stroke
        landing = None
        az, bz, cz = fit["az"], fit["bz"], fit["cz"]
        disc = bz**2 - 4 * az * cz
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            y1 = (-bz + sqrt_disc) / (2 * az) if abs(az) > 1e-9 else None
            y2 = (-bz - sqrt_disc) / (2 * az) if abs(az) > 1e-9 else None
            candidates = [
                y for y in [y1, y2]
                if y is not None and -2 <= y <= _COURT_Y + 2
            ]
            if candidates:
                # Pick root farthest from segment start
                start_y = seg_points[0]["y"]
                land_y = max(candidates, key=lambda y: abs(y - start_y))
                x_land = fit["ax"] * land_y + fit["bx"]
                landing = {
                    "x": round(float(x_land), 4),
                    "y": round(float(land_y), 4),
                    "z": 0.0,
                }

        stroke = {
            "start_idx": seg_start,
            "end_idx": seg_end,
            "n_points": len(seg_points),
            "fit": fit,
            "speed_kmh": fit["speed_kmh"],
            "landing_point": landing,
        }

        # Bounce info
        if seg_start in bounce_indices:
            bp = points[seg_start]
            stroke["bounce_before"] = {
                "x": bp["x"], "y": bp["y"], "z": bp["z"]
            }
        if seg_end in bounce_indices:
            bp = points[seg_end]
            stroke["bounce_after"] = {
                "x": bp["x"], "y": bp["y"], "z": bp["z"]
            }

        strokes.append(stroke)

    return strokes


def fit_trajectory(points: list[dict]) -> dict:
    """Fit piecewise spatial parabola with RANSAC and robust bounce detection.

    Pipeline:
        1. RANSAC fit single parabola to find inliers
        2. Detect bounce on inlier set
        3. If bounce: fit pre/post segments separately
        4. Generate smooth curve and check net crossing

    Args:
        points: List of triangulated 3D points with keys x, y, z, ray_dist.

    Returns:
        Dict with trajectory type, fit results, bounce info, smooth curve,
        and outlier information.
    """
    if len(points) < 3:
        return {"type": "insufficient_data", "n_points": len(points)}

    # Step 1: RANSAC to identify inliers
    ransac_fit, inlier_indices = fit_spatial_parabola_ransac(points)
    inlier_set = set(inlier_indices)
    outlier_indices = [i for i in range(len(points)) if i not in inlier_set]
    inlier_points = [points[i] for i in inlier_indices]

    if not inlier_points or len(inlier_points) < 3:
        # RANSAC failed completely, fall back to all points
        inlier_points = points
        inlier_indices = list(range(len(points)))
        outlier_indices = []
        ransac_fit = fit_spatial_parabola(points)

    # Step 2: Detect bounce on inlier set
    bounce_idx_in_inliers = _detect_bounce_robust(
        inlier_points, ransac_fit
    )

    if bounce_idx_in_inliers is not None:
        pre = inlier_points[: bounce_idx_in_inliers + 1]
        post = inlier_points[bounce_idx_in_inliers:]

        # Fit segments (use RANSAC for larger segments, direct for small)
        if len(pre) >= 4:
            fit_pre, _ = fit_spatial_parabola_ransac(
                pre, n_iterations=100, min_inlier_ratio=0.5
            )
        else:
            fit_pre = fit_spatial_parabola(pre)

        if len(post) >= 4:
            fit_post, _ = fit_spatial_parabola_ransac(
                post, n_iterations=100, min_inlier_ratio=0.5
            )
        else:
            fit_post = fit_spatial_parabola(post)

        bounce_point = inlier_points[bounce_idx_in_inliers]
        smooth_curve = _generate_smooth_curve(
            inlier_points, fit_pre, fit_post, bounce_point["y"]
        )

        result = {
            "type": "piecewise",
            "bounce_idx": bounce_idx_in_inliers,
            "bounce_pos": {
                "x": bounce_point["x"],
                "y": bounce_point["y"],
                "z": bounce_point["z"],
            },
            "pre_bounce": fit_pre,
            "post_bounce": fit_post,
            "smooth_curve": smooth_curve,
        }
    else:
        smooth_curve = _generate_smooth_curve(
            inlier_points, ransac_fit, None, None
        )
        result = {
            "type": "single",
            "fit": ransac_fit,
            "smooth_curve": smooth_curve,
        }

    # Add outlier info
    result["n_inliers"] = len(inlier_indices)
    result["n_outliers"] = len(outlier_indices)
    result["outlier_indices"] = outlier_indices

    # Net crossing check (includes speed at net)
    result["net_crossing"] = _check_net_crossing(inlier_points, result)

    # Landing point (where Z=0 after bounce/apex)
    result["landing_point"] = _find_landing_point(result)

    # Step 5: Detect ball cycles (hit → land → hit → land ...)
    # Find ALL bounces in the inlier set for multi-stroke analysis
    all_bounces = _detect_all_bounces(inlier_points)
    if all_bounces:
        strokes = _segment_strokes(inlier_points, all_bounces)
        if strokes:
            result["strokes"] = strokes
            result["n_bounces"] = len(all_bounces)
            result["bounce_positions"] = [
                {
                    "x": round(inlier_points[b]["x"], 4),
                    "y": round(inlier_points[b]["y"], 4),
                    "z": round(inlier_points[b]["z"], 4),
                }
                for b in all_bounces
            ]
            logger.info(
                "Ball cycles: %d bounces, %d strokes detected",
                len(all_bounces),
                len(strokes),
            )

    return result


def _generate_smooth_curve(
    points: list[dict],
    fit_pre: Optional[dict],
    fit_post: Optional[dict],
    bounce_y: Optional[float],
    n_smooth: int = 200,
) -> list[dict]:
    """Generate a smooth XYZ curve from the spatial fit for visualization."""
    ys = [p["y"] for p in points]
    y_min, y_max = min(ys), max(ys)
    y_smooth = np.linspace(y_min, y_max, n_smooth)
    curve = []

    for y in y_smooth:
        if bounce_y is not None and fit_pre and fit_post:
            if y <= bounce_y:
                x = fit_pre["ax"] * y + fit_pre["bx"]
                z = fit_pre["az"] * y**2 + fit_pre["bz"] * y + fit_pre["cz"]
            else:
                x = fit_post["ax"] * y + fit_post["bx"]
                z = fit_post["az"] * y**2 + fit_post["bz"] * y + fit_post["cz"]
        elif fit_pre:
            x = fit_pre["ax"] * y + fit_pre["bx"]
            z = fit_pre["az"] * y**2 + fit_pre["bz"] * y + fit_pre["cz"]
        else:
            continue

        # Skip unreasonable Z values
        if z < -0.3 or z > 6.0:
            continue

        curve.append({
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
        })

    return curve


def _check_net_crossing(points: list[dict], traj_result: dict) -> Optional[dict]:
    """Check ball height and speed at net position (Y=11.885m).

    Computes:
        - Position (x, z) at the net
        - Whether the ball clears the net (center height 0.914m)
        - Ball speed at net from the spatial fit velocity model

    Key: For piecewise trajectories, uses the correct segment based on
    whether the net is before or after the bounce point.
    """
    net_y = _NET_Y
    ys = [p["y"] for p in points]

    if min(ys) > net_y or max(ys) < net_y:
        return None

    # Select the correct fit segment for the net position
    if traj_result["type"] == "piecewise":
        bounce_y = traj_result["bounce_pos"]["y"]
        if net_y <= bounce_y:
            fit = traj_result.get("pre_bounce")
        else:
            fit = traj_result.get("post_bounce")
    else:
        fit = traj_result.get("fit")

    if fit is None:
        return None

    # Validate that net_y is within (or close to) the fit's valid Y range
    y_margin = 3.0  # allow up to 3m extrapolation
    fit_y_min = fit.get("y_min", min(ys))
    fit_y_max = fit.get("y_max", max(ys))
    if net_y < fit_y_min - y_margin or net_y > fit_y_max + y_margin:
        logger.warning(
            "Net Y=%.2f outside fit range [%.2f, %.2f] + margin, skipping",
            net_y, fit_y_min, fit_y_max,
        )
        return None

    z_at_net = fit["az"] * net_y**2 + fit["bz"] * net_y + fit["cz"]
    x_at_net = fit["ax"] * net_y + fit["bx"]

    # Sanity check: net height should be physically reasonable
    if z_at_net < -1.0 or z_at_net > 10.0:
        logger.warning(
            "Net crossing z=%.2fm out of range [-1, 10], likely extrapolation error",
            z_at_net,
        )
        return None

    # Compute speed at net using the spatial-fit velocity model
    speed_at_net = _compute_speed_at_y(fit, net_y)

    # Sanity check: speed should be physically plausible (< 350 km/h)
    speed_kmh = speed_at_net * 3.6
    if speed_kmh > 350.0:
        logger.warning(
            "Net speed %.0f km/h exceeds physical limit, clamping", speed_kmh
        )
        speed_at_net = 350.0 / 3.6
        speed_kmh = 350.0

    return {
        "y": net_y,
        "x": round(float(x_at_net), 4),
        "z": round(float(z_at_net), 4),
        "clears_net": float(z_at_net) > 0.914,
        "speed_ms": round(speed_at_net, 2),
        "speed_kmh": round(speed_kmh, 1),
    }


def _compute_speed_at_y(fit: dict, y: float) -> float:
    """Compute ball speed at a given Y position from spatial fit coefficients.

    From the spatial model:
        X(Y) = ax*Y + bx  →  dX/dY = ax
        Z(Y) = az*Y² + bz*Y + cz  →  dZ/dY = 2*az*Y + bz
        az = -g / (2 * vy²)  →  vy = sqrt(g / (-2*az))

    Speed = vy * sqrt(1 + (dX/dY)² + (dZ/dY)²)

    Note: vy is clamped to [1, 80] m/s to prevent blow-up when |az| is very
    small (nearly flat trajectory over a short Y range).
    """
    g = 9.81
    az, bz = fit["az"], fit["bz"]
    ax = fit["ax"]

    vy = np.sqrt(g / (-2 * az)) if az < -1e-6 else 25.0
    # Clamp vy to physically reasonable range:
    # min ~3.6 km/h (very slow), max ~288 km/h (professional serve vy component)
    vy = float(np.clip(vy, 1.0, 80.0))

    dx_dy = ax
    dz_dy = 2 * az * y + bz
    speed = vy * np.sqrt(1.0 + dx_dy**2 + dz_dy**2)
    return float(speed)


def _find_landing_point(traj_result: dict) -> Optional[dict]:
    """Find where the ball lands (Z=0) after the bounce or after the apex.

    For piecewise trajectories, uses the post-bounce fit.
    For single trajectories, finds where Z crosses zero on the descending side.

    Returns landing position {x, y, z} or None.
    """
    if traj_result["type"] == "piecewise":
        fit = traj_result.get("post_bounce")
        if fit is None:
            return None
        # Find the second Z=0 root of post-bounce parabola:
        # az*y² + bz*y + cz = 0
        az, bz, cz = fit["az"], fit["bz"], fit["cz"]
        disc = bz**2 - 4 * az * cz
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        y1 = (-bz + sqrt_disc) / (2 * az)
        y2 = (-bz - sqrt_disc) / (2 * az)
        # Pick the root that's farther along the trajectory (away from bounce)
        bounce_y = traj_result["bounce_pos"]["y"]
        candidates = [y for y in [y1, y2] if 0 <= y <= _COURT_Y]
        if not candidates:
            # Try with some margin
            candidates = [y for y in [y1, y2] if -2 <= y <= _COURT_Y + 2]
        if not candidates:
            return None
        # Pick the one farther from the bounce point
        landing_y = max(candidates, key=lambda y: abs(y - bounce_y))
        x_land = fit["ax"] * landing_y + fit["bx"]
        return {
            "x": round(float(x_land), 4),
            "y": round(float(landing_y), 4),
            "z": 0.0,
        }
    else:
        fit = traj_result.get("fit")
        if fit is None:
            return None
        az, bz, cz = fit["az"], fit["bz"], fit["cz"]
        disc = bz**2 - 4 * az * cz
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        y1 = (-bz + sqrt_disc) / (2 * az)
        y2 = (-bz - sqrt_disc) / (2 * az)
        candidates = [y for y in [y1, y2] if 0 <= y <= _COURT_Y]
        if not candidates:
            return None
        # Pick the root on the descending side (farther from start)
        ys_data = traj_result.get("smooth_curve", [])
        if ys_data:
            start_y = ys_data[0]["y"]
            landing_y = max(candidates, key=lambda y: abs(y - start_y))
        else:
            landing_y = max(candidates)
        x_land = fit["ax"] * landing_y + fit["bx"]
        return {
            "x": round(float(x_land), 4),
            "y": round(float(landing_y), 4),
            "z": 0.0,
        }


# ========== Rally segmentation ==========


def segment_rallies(
    points_3d: list[dict],
    fps: float = 25.0,
    max_gap_seconds: float = 1.0,
    min_rally_points: int = 15,
    min_rally_duration: float = 0.5,
    min_rally_displacement: float = 2.0,
) -> list[list[dict]]:
    """Split 3D points into separate rallies based on time gaps.

    A new rally starts when:
        - There is a time gap > max_gap_seconds between consecutive points
        - The ball reappears far from where it disappeared (large spatial jump)

    Segments are discarded if they fail any of these quality checks:
        - Too few points (< min_rally_points)
        - Too short in time (< min_rally_duration)
        - Too little spatial displacement (< min_rally_displacement)
          Noise clusters in a small area; a real shot travels across the court.

    Args:
        points_3d: Triangulated 3D points sorted by time, with keys x, y, z, t.
        fps: Frame rate (used to interpret frame_a as time if t is missing).
        max_gap_seconds: Maximum time gap within a rally.
        min_rally_points: Minimum points to keep a rally segment.
        min_rally_duration: Minimum time span (seconds) to keep a rally segment.
        min_rally_displacement: Minimum 3D displacement first→last point (meters).

    Returns:
        List of rally segments, each a list of 3D point dicts.
    """
    if not points_3d:
        return []

    def _get_time(p: dict) -> float:
        return p.get("t", p.get("frame_a", 0) / fps)

    rallies: list[list[dict]] = []
    current: list[dict] = [points_3d[0]]

    for i in range(1, len(points_3d)):
        prev = points_3d[i - 1]
        curr = points_3d[i]

        # Time gap
        dt = _get_time(curr) - _get_time(prev)

        # Spatial jump
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        dz = curr["z"] - prev["z"]
        spatial_dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # Break rally on large time gap OR large spatial jump relative to time
        is_gap = dt > max_gap_seconds
        is_jump = dt > 0.2 and spatial_dist / dt > 80.0  # > 80 m/s = impossible

        if is_gap or is_jump:
            _maybe_keep(current, rallies, min_rally_points,
                        min_rally_duration, min_rally_displacement, _get_time)
            current = [curr]
        else:
            current.append(curr)

    _maybe_keep(current, rallies, min_rally_points,
                min_rally_duration, min_rally_displacement, _get_time)

    return rallies


def _maybe_keep(
    segment: list[dict],
    rallies: list[list[dict]],
    min_points: int,
    min_duration: float,
    min_displacement: float,
    get_time,
) -> None:
    """Append segment to rallies if it meets all quality requirements."""
    if len(segment) < min_points:
        return

    duration = get_time(segment[-1]) - get_time(segment[0])
    if duration < min_duration:
        return

    # 3D displacement from first to last point
    p0, p1 = segment[0], segment[-1]
    disp = np.sqrt(
        (p1["x"] - p0["x"]) ** 2
        + (p1["y"] - p0["y"]) ** 2
        + (p1["z"] - p0["z"]) ** 2
    )
    if disp < min_displacement:
        return

    rallies.append(segment)
