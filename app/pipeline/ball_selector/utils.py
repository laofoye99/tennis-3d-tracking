"""Utility functions: preprocessing, coordinate normalization, median bg, GT heatmap."""

import json
import os
from pathlib import Path

import cv2
import numpy as np

# TrackNet input resolution
INPUT_H, INPUT_W = 288, 512
# Original image resolution
ORIG_H, ORIG_W = 1080, 1920


def preprocess_frame(frame, target_w=INPUT_W, target_h=INPUT_H):
    """Resize BGR frame → RGB CHW float32 [0, 1]. Matches TrackNet preprocessing."""
    img = cv2.resize(frame, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)


def compute_median_bg(frame_dir, max_samples=200, target_w=INPUT_W, target_h=INPUT_H):
    """Compute per-pixel median background from a directory of JPG frames.

    Args:
        frame_dir: path to directory containing .jpg frames
        max_samples: max frames to sample (evenly spaced)

    Returns:
        (3, H, W) float32 array in [0, 1]
    """
    jpg_files = sorted(Path(frame_dir).glob("*.jpg"))
    if not jpg_files:
        raise ValueError(f"No JPG files in {frame_dir}")

    # Sample evenly
    n = min(max_samples, len(jpg_files))
    indices = np.linspace(0, len(jpg_files) - 1, n, dtype=int)
    sampled = [jpg_files[i] for i in indices]

    frames = []
    for p in sampled:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (target_w, target_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    if not frames:
        raise ValueError(f"Could not read any frames from {frame_dir}")

    median = np.median(frames, axis=0).astype(np.uint8)
    return median.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)


def get_or_compute_median_bg(frame_dir, cache_dir=None):
    """Get cached median bg or compute and cache it.

    Cache stored as .npy file alongside frames or in cache_dir.
    """
    if cache_dir is None:
        cache_path = Path(frame_dir) / "_median_bg.npy"
    else:
        os.makedirs(cache_dir, exist_ok=True)
        # Use directory name as cache key
        key = str(Path(frame_dir)).replace("/", "_").replace("\\", "_").replace(":", "")
        cache_path = Path(cache_dir) / f"{key}.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    bg = compute_median_bg(frame_dir)
    np.save(str(cache_path), bg)
    return bg


def generate_gaussian_heatmap(x, y, h=INPUT_H, w=INPUT_W, sigma=2.5):
    """Generate a 2D Gaussian heatmap centered at (x, y).

    Args:
        x, y: center coordinates in heatmap space (0-based)
        h, w: heatmap dimensions
        sigma: Gaussian spread

    Returns:
        (h, w) float32 array in [0, 1]
    """
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Bounds check
    if x < 0 or x >= w or y < 0 or y >= h:
        return heatmap

    radius = int(3 * sigma + 1)
    x_int, y_int = int(round(x)), int(round(y))

    y_min = max(0, y_int - radius)
    y_max = min(h, y_int + radius + 1)
    x_min = max(0, x_int - radius)
    x_max = min(w, x_int + radius + 1)

    yy, xx = np.meshgrid(
        np.arange(y_min, y_max),
        np.arange(x_min, x_max),
        indexing="ij",
    )
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap[y_min:y_max, x_min:x_max] = gaussian
    return heatmap


def generate_gt_heatmap(ball_x_orig, ball_y_orig, sigma=2.5):
    """Generate GT heatmap from original pixel coordinates.

    Converts from original (1920x1080) to heatmap (512x288) resolution.

    Args:
        ball_x_orig, ball_y_orig: ball center in original image coords
        sigma: Gaussian spread in heatmap space

    Returns:
        (288, 512) float32 heatmap, or zeros if ball not visible
    """
    if ball_x_orig is None or ball_y_orig is None:
        return np.zeros((INPUT_H, INPUT_W), dtype=np.float32)

    # Scale to heatmap resolution
    x = ball_x_orig * INPUT_W / ORIG_W
    y = ball_y_orig * INPUT_H / ORIG_H
    return generate_gaussian_heatmap(x, y)


def normalize_coords(px, py):
    """Normalize pixel coords from original resolution to [0, 1]."""
    return px / ORIG_W, py / ORIG_H


def denormalize_coords(nx, ny):
    """Denormalize coords from [0, 1] to original pixel resolution."""
    return nx * ORIG_W, ny * ORIG_H


def parse_labelme_json(json_path):
    """Parse a LabelMe annotation JSON file.

    Returns:
        dict with:
            ball_pos: (cx, cy) in original pixel coords, or None if no ball
            ball_present: bool
            events: list of str ('bounce', 'shot', 'serve', etc.)
            score: float or None (TrackNet confidence if available)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ball_pos = None
    ball_present = False
    events = []
    score = None

    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in ("ball", "ball_1"):
            continue

        desc = shape.get("description", "") or ""
        shape_type = shape.get("shape_type", "")

        # Skip non-match balls
        if "dead_bounce" in desc and "match_ball" not in desc:
            continue

        # Extract position
        pts = shape.get("points", [])
        if shape_type == "point" and len(pts) >= 1:
            cx, cy = pts[0][0], pts[0][1]
        elif shape_type == "rectangle" and len(pts) >= 2:
            # Rectangle: corners → center
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
        else:
            continue

        # Check if this is a match ball annotation
        is_match = (
            "match_ball" in desc
            or desc == ""  # some annotations have no description
            or "blob_sum" in desc  # TrackNet detection with score
        )
        if not is_match:
            continue

        ball_pos = (cx, cy)
        ball_present = True
        score = shape.get("score", None)

        # Parse events from description
        for event in ["bounce", "shot", "serve", "bounce_out"]:
            if event in desc:
                events.append(event)

    return {
        "ball_pos": ball_pos,
        "ball_present": ball_present,
        "events": events,
        "score": score,
    }
