"""Dashboard helpers for four-corner court homography recalibration."""

from __future__ import annotations

import datetime as _dt
import json
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


def _corner_world_points(court: dict[str, Any]) -> dict[str, tuple[float, float]]:
    half_width = float(court.get("half_width_m", court.get("width_m", 8.23) / 2.0))
    half_length = float(court.get("half_length_m", court.get("length_m", 23.78) / 2.0))
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


def compute_corner_homography(
    image_points: dict[str, tuple[float, float]],
    court: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute image/world homography matrices from four labeled court corners."""
    court = court or {}
    world_points = _corner_world_points(court)
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
    computed = compute_corner_homography(image_points, data.get("court_dimensions", {}))
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
                for corner, world in _corner_world_points(court).items()
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
            for corner, (x, y) in _corner_world_points(court).items()
        },
        "cameras": cameras,
    }
