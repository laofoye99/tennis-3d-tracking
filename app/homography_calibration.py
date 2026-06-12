"""Dashboard helpers for four-corner court homography recalibration."""

from __future__ import annotations

import datetime as _dt
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COURT_CORNERS = (
    "near_left",
    "near_right",
    "far_right",
    "far_left",
)


def _corner_world_points(
    court: dict[str, Any],
    camera_key: str | None = None,
) -> dict[str, tuple[float, float]]:
    half_width = float(court.get("half_width_m", court.get("width_m", 8.23) / 2.0))
    half_length = float(court.get("half_length_m", court.get("length_m", 23.78) / 2.0))
    key = str(camera_key or "").lower()
    if key == "cam68":
        return {
            "near_left": (half_width, -half_length),
            "near_right": (-half_width, -half_length),
            "far_right": (-half_width, half_length),
            "far_left": (half_width, half_length),
        }
    if key == "cam66":
        return {
            "near_left": (-half_width, half_length),
            "near_right": (half_width, half_length),
            "far_right": (half_width, -half_length),
            "far_left": (-half_width, -half_length),
        }
    return {
        "near_left": (-half_width, -half_length),
        "near_right": (half_width, -half_length),
        "far_right": (half_width, half_length),
        "far_left": (-half_width, half_length),
    }


def _point_xy(value: Any, *, label: str) -> tuple[float, float]:
    if isinstance(value, dict):
        raw_x = value.get("x", value.get("px"))
        raw_y = value.get("y", value.get("py"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        raw_x, raw_y = value[0], value[1]
    else:
        raise ValueError(f"{label} must be an object with x/y or a two-item list")
    try:
        x = float(raw_x)
        y = float(raw_y)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} coordinates must be numeric") from exc
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"{label} coordinates must be finite")
    return x, y


def normalize_corner_points(points: Any) -> dict[str, tuple[float, float]]:
    """Normalize corner payloads from the dashboard.

    Accepted shapes:
    - {"near_left": {"x": 1, "y": 2}, ...}
    - [{"corner": "near_left", "x": 1, "y": 2}, ...]
    """
    normalized: dict[str, tuple[float, float]] = {}
    if isinstance(points, dict):
        for corner in COURT_CORNERS:
            if corner in points:
                normalized[corner] = _point_xy(points[corner], label=corner)
    elif isinstance(points, list):
        for item in points:
            if not isinstance(item, dict):
                raise ValueError("point list items must be objects")
            corner = str(item.get("corner") or item.get("id") or "").strip()
            if corner in COURT_CORNERS:
                normalized[corner] = _point_xy(item, label=corner)
    else:
        raise ValueError("points must be an object or list")

    missing = [corner for corner in COURT_CORNERS if corner not in normalized]
    if missing:
        raise ValueError(f"missing court corners: {', '.join(missing)}")
    return normalized


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    denom = float(matrix[2, 2])
    if abs(denom) > 1e-12:
        matrix = matrix / denom
    return matrix


def _project(matrix: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    vec = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    out = matrix @ vec
    if abs(float(out[2])) <= 1e-12:
        raise ValueError("homography projection produced a point at infinity")
    return float(out[0] / out[2]), float(out[1] / out[2])


def _fit_line_from_segments(segments: list[tuple[float, ...]]) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for x1, y1, x2, y2, length, *_rest in segments:
        repeats = max(1, int(float(length) / 80.0))
        pts.extend([(x1, y1), (x2, y2)] * repeats)
    if len(pts) < 2:
        raise ValueError("not enough line points")
    arr = np.array(pts, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return np.array([-vy, vx, vy * x0 - vx * y0], dtype=np.float64)


def _intersect_lines(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a1, b1, c1 = [float(v) for v in a]
    a2, b2, c2 = [float(v) for v in b]
    denom = a1 * b2 - a2 * b1
    if abs(denom) <= 1e-9:
        raise ValueError("parallel court lines")
    return (b1 * c2 - b2 * c1) / denom, (c1 * a2 - c2 * a1) / denom


def _cluster_horizontal_segments(
    segments: list[tuple[float, ...]],
    *,
    tolerance_px: float,
) -> list[list[tuple[float, ...]]]:
    clusters: list[list[tuple[float, ...]]] = []
    for segment in sorted(segments, key=lambda item: item[7]):
        for cluster in clusters:
            weight = sum(float(item[4]) for item in cluster)
            y_mid = sum(float(item[7]) * float(item[4]) for item in cluster) / max(weight, 1.0)
            if abs(float(segment[7]) - y_mid) <= tolerance_px:
                cluster.append(segment)
                break
        else:
            clusters.append([segment])
    return clusters


def estimate_court_corners_from_jpeg(
    jpeg: bytes,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
) -> dict[str, Any]:
    """Estimate four camera-view court corners from a snapshot JPEG.

    Returned points are in source-frame coordinates, matching detector pixels.
    """
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode snapshot")

    img_h, img_w = frame.shape[:2]
    max_width = 960.0
    scale = min(1.0, max_width / float(max(img_w, 1)))
    proc = (
        cv2.resize(frame, (int(img_w * scale), int(img_h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else frame.copy()
    )
    proc_h, proc_w = proc.shape[:2]

    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 155), (179, 85, 255))
    mask[: int(proc_h * 0.08), : int(proc_w * 0.36)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        threshold=max(45, int(proc_w * 0.06)),
        minLineLength=max(50, int(proc_w * 0.10)),
        maxLineGap=max(20, int(proc_w * 0.03)),
    )
    if lines is None:
        raise ValueError("no court lines found")

    segments: list[tuple[float, ...]] = []
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = [float(v) for v in raw]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < max(50.0, proc_w * 0.08):
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle <= -90.0:
            angle += 180.0
        while angle > 90.0:
            angle -= 180.0
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        segments.append((x1, y1, x2, y2, length, angle, mx, my))

    horizontal = [s for s in segments if abs(float(s[5])) < 12.0]
    diag_pos = [s for s in segments if 35.0 < float(s[5]) < 75.0 and float(s[6]) > proc_w * 0.42]
    diag_neg = [s for s in segments if -75.0 < float(s[5]) < -35.0 and float(s[6]) < proc_w * 0.58]
    if diag_pos:
        max_len = max(float(s[4]) for s in diag_pos)
        diag_pos = [s for s in diag_pos if float(s[4]) >= max_len * 0.55]
    if diag_neg:
        max_len = max(float(s[4]) for s in diag_neg)
        diag_neg = [s for s in diag_neg if float(s[4]) >= max_len * 0.55]

    clusters = [
        cluster
        for cluster in _cluster_horizontal_segments(horizontal, tolerance_px=max(10.0, proc_h * 0.035))
        if sum(float(s[4]) for s in cluster) >= proc_w * 0.12
    ]
    if len(clusters) < 2 or not diag_pos or not diag_neg:
        raise ValueError("not enough court line candidates")

    top_cluster = clusters[0]
    bottom_cluster = clusters[-1]
    top_line = _fit_line_from_segments(top_cluster)
    bottom_line = _fit_line_from_segments(bottom_cluster)
    left_line = _fit_line_from_segments(diag_neg)
    right_line = _fit_line_from_segments(diag_pos)

    points_preview = {
        "near_left": _intersect_lines(bottom_line, left_line),
        "near_right": _intersect_lines(bottom_line, right_line),
        "far_right": _intersect_lines(top_line, right_line),
        "far_left": _intersect_lines(top_line, left_line),
    }

    source_w = float(source_width or img_w)
    source_h = float(source_height or img_h)
    x_scale = source_w / float(img_w)
    y_scale = source_h / float(img_h)
    points = {
        corner: {
            "x": round(float(pt[0]) / max(scale, 1e-9) * x_scale, 3),
            "y": round(float(pt[1]) / max(scale, 1e-9) * y_scale, 3),
        }
        for corner, pt in points_preview.items()
    }

    for corner, point in points.items():
        if not (0 <= point["x"] <= source_w and 0 <= point["y"] <= source_h):
            raise ValueError(f"auto corner outside image: {corner}")

    return {
        "status": "ok",
        "method": "white_line_hough",
        "points": points,
        "source_width": int(source_w),
        "source_height": int(source_h),
        "preview_width": int(img_w),
        "preview_height": int(img_h),
        "diagnostics": {
            "line_count": len(segments),
            "horizontal_clusters": len(clusters),
            "left_segments": len(diag_neg),
            "right_segments": len(diag_pos),
        },
    }


def compute_corner_homography(
    image_points: dict[str, tuple[float, float]],
    court: dict[str, Any] | None = None,
    camera_key: str | None = None,
) -> dict[str, Any]:
    """Compute image/world homography matrices from four camera-view corners."""
    court = court or {}
    world_points = _corner_world_points(court, camera_key)
    src = np.array([image_points[corner] for corner in COURT_CORNERS], dtype=np.float64)
    dst = np.array([world_points[corner] for corner in COURT_CORNERS], dtype=np.float64)

    if len(np.unique(src, axis=0)) != len(COURT_CORNERS):
        raise ValueError("image corner points must be unique")

    h_img2world, _mask = cv2.findHomography(src, dst, method=0)
    h_world2img, _mask_inv = cv2.findHomography(dst, src, method=0)
    if h_img2world is None or h_world2img is None:
        raise ValueError("failed to compute homography from corner points")

    h_img2world = _normalize_matrix(h_img2world)
    h_world2img = _normalize_matrix(h_world2img)

    world_errors: list[float] = []
    pixel_errors: list[float] = []
    projected_world: dict[str, list[float]] = {}
    projected_image: dict[str, list[float]] = {}
    for corner in COURT_CORNERS:
        world = world_points[corner]
        image = image_points[corner]
        wx, wy = _project(h_img2world, image)
        px, py = _project(h_world2img, world)
        projected_world[corner] = [round(wx, 6), round(wy, 6)]
        projected_image[corner] = [round(px, 3), round(py, 3)]
        world_errors.append(float(np.hypot(wx - world[0], wy - world[1])))
        pixel_errors.append(float(np.hypot(px - image[0], py - image[1])))

    return {
        "H_image_to_world": h_img2world.tolist(),
        "H_world_to_image": h_world2img.tolist(),
        "corner_points_image": {
            corner: [round(float(image_points[corner][0]), 3), round(float(image_points[corner][1]), 3)]
            for corner in COURT_CORNERS
        },
        "corner_points_world": {
            corner: [round(float(world_points[corner][0]), 6), round(float(world_points[corner][1]), 6)]
            for corner in COURT_CORNERS
        },
        "projected_corners_world": projected_world,
        "projected_corners_image": projected_image,
        "reprojection_error_m": round(float(np.mean(world_errors)), 6),
        "reprojection_error_px": round(float(np.mean(pixel_errors)), 6),
    }


def update_homography_file(
    matrices_path: str | Path,
    camera_key: str,
    points: Any,
    *,
    backup: bool = True,
) -> dict[str, Any]:
    """Update one camera entry in homography_matrices.json from four corner points."""
    path = Path(matrices_path)
    if not path.exists():
        raise FileNotFoundError(f"homography file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if camera_key not in data or not isinstance(data.get(camera_key), dict):
        raise ValueError(f"unknown homography camera key: {camera_key}")

    image_points = normalize_corner_points(points)
    computed = compute_corner_homography(
        image_points,
        data.get("court_dimensions", {}),
        camera_key=camera_key,
    )
    timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()

    backup_path: Path | None = None
    if backup:
        backup_path = path.with_suffix(
            path.suffix + "." + _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
        )
        shutil.copy2(path, backup_path)

    updated_entry = dict(data.get(camera_key) or {})
    updated_entry.update(
        {
            "H_image_to_world": computed["H_image_to_world"],
            "H_world_to_image": computed["H_world_to_image"],
            "reprojection_error_m": computed["reprojection_error_m"],
            "reprojection_error_px": computed["reprojection_error_px"],
            "calibration_method": "dashboard_four_corners",
            "updated_at": timestamp,
            "corner_points_image": computed["corner_points_image"],
            "corner_points_world": computed["corner_points_world"],
        }
    )
    data[camera_key] = updated_entry
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "status": "ok",
        "camera_key": camera_key,
        "homography_path": str(path),
        "backup_path": str(backup_path) if backup_path else None,
        "updated_at": timestamp,
        **computed,
    }


def homography_status(matrices_path: str | Path, camera_key: str | None = None) -> dict[str, Any]:
    path = Path(matrices_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    court = data.get("court_dimensions", {})
    cameras: dict[str, Any] = {}
    keys = [camera_key] if camera_key else [key for key, value in data.items() if isinstance(value, dict)]
    for key in keys:
        if key == "court_dimensions":
            continue
        cam_data = data.get(key)
        if not isinstance(cam_data, dict):
            continue
        projected = None
        try:
            h_world2img = np.array(cam_data["H_world_to_image"], dtype=np.float64)
            projected = {
                corner: [round(float(v), 3) for v in _project(h_world2img, world)]
                for corner, world in _corner_world_points(court, key).items()
            }
        except Exception:
            projected = None
        cameras[key] = {
            "reprojection_error_m": cam_data.get("reprojection_error_m"),
            "reprojection_error_px": cam_data.get("reprojection_error_px"),
            "calibration_method": cam_data.get("calibration_method"),
            "updated_at": cam_data.get("updated_at"),
            "corner_points_image": cam_data.get("corner_points_image"),
            "projected_corners_image": projected,
        }
    return {
        "homography_path": str(path),
        "court_dimensions": court,
        "corner_order": list(COURT_CORNERS),
        "corner_points_world": {
            corner: [round(float(x), 6), round(float(y), 6)]
            for corner, (x, y) in _corner_world_points(court, camera_key).items()
        },
        "cameras": cameras,
    }
