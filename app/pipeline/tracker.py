"""
Single-camera blob tracking + cross-camera 3D matching.

From research: "Track first, triangulate later"
1. track_single_camera(): link blobs across frames using nearest-neighbor + velocity prediction
2. match_and_triangulate(): match tracks across cameras, triangulate overlapping frames

This replaces the per-frame MultiBlobMatcher approach.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ================================================================
# Single-camera blob tracking
# ================================================================

def track_single_camera(detections, max_pixel_dist=80, max_gap=3, min_len=10):
    """Track blobs in a single camera across frames.

    Uses greedy nearest-neighbor in pixel space with velocity prediction.

    Args:
        detections: dict frame_idx -> list of (cx, cy)
        max_pixel_dist: max pixel distance for blob linking
        max_gap: max frame gap to bridge
        min_len: minimum track length to keep

    Returns:
        List of tracks. Each track = list of (frame_idx, px, py).
    """
    tracks = []      # [last_frame, last_pos, points_list]
    finished = []

    for fi in sorted(detections.keys()):
        blobs = detections[fi]
        if not blobs:
            continue

        used = set()
        # Prioritize longer tracks
        tracks.sort(key=lambda t: len(t[2]), reverse=True)

        for track in tracks:
            last_f, (lx, ly), tpts = track
            gap = fi - last_f
            if gap > max_gap:
                continue

            # Velocity prediction
            if len(tpts) >= 2:
                p1 = tpts[-1]
                p2 = tpts[-2]
                dt = p1[0] - p2[0]
                if dt > 0:
                    vx = (p1[1] - p2[1]) / dt * gap
                    vy = (p1[2] - p2[2]) / dt * gap
                    pred_x = p1[1] + vx
                    pred_y = p1[2] + vy
                else:
                    pred_x, pred_y = lx, ly
            else:
                pred_x, pred_y = lx, ly

            best_d = max_pixel_dist
            best_i = -1
            for bi, (bx, by) in enumerate(blobs):
                if bi in used:
                    continue
                d = np.sqrt((bx - pred_x)**2 + (by - pred_y)**2)
                if d < best_d:
                    best_d = d
                    best_i = bi

            if best_i >= 0:
                bx, by = blobs[best_i]
                track[0] = fi
                track[1] = (bx, by)
                track[2].append((fi, bx, by))
                used.add(best_i)

        # New tracks for unmatched
        for bi, (bx, by) in enumerate(blobs):
            if bi not in used:
                tracks.append([fi, (bx, by), [(fi, bx, by)]])

        # Clean up old tracks
        new_tracks = []
        for track in tracks:
            if fi - track[0] > max_gap:
                if len(track[2]) >= min_len:
                    finished.append(track[2])
            else:
                new_tracks.append(track)
        tracks = new_tracks

    for track in tracks:
        if len(track[2]) >= min_len:
            finished.append(track[2])

    finished.sort(key=lambda t: len(t), reverse=True)
    return finished


# ================================================================
# Cross-camera trajectory matching + 3D reconstruction
# ================================================================

def triangulate_ray(w1, w2, cam1_pos, cam2_pos):
    """Ray intersection → 3D point + ray distance."""
    cam1 = np.asarray(cam1_pos, dtype=np.float64)
    cam2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
    d1 = g1 - cam1
    d2 = g2 - cam2
    w = cam1 - cam2
    a = float(d1 @ d1)
    b = float(d1 @ d2)
    c = float(d2 @ d2)
    d_v = float(d1 @ w)
    e = float(d2 @ w)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return (0, 0, 0), 999.0
    s = np.clip((b * e - c * d_v) / denom, 0, 1)
    t = np.clip((a * e - b * d_v) / denom, 0, 1)
    p1 = cam1 + s * d1
    t = float((p1 - cam2) @ d2) / c if c > 1e-10 else t
    t = np.clip(t, 0, 1)
    p2 = cam2 + t * d2
    s = float((p2 - cam1) @ d1) / a if a > 1e-10 else s
    s = np.clip(s, 0, 1)
    p1 = cam1 + s * d1
    p2 = cam2 + t * d2
    mid = (p1 + p2) / 2.0
    rd = float(np.linalg.norm(p1 - p2))
    if mid[2] < 0:
        mid[2] = 0.0
    return (float(mid[0]), float(mid[1]), float(mid[2])), rd


def pixel_to_world(px, py, H):
    """Homography: pixel (px, py) → world (wx, wy)."""
    p = np.array([px, py, 1.0])
    r = H @ p
    return float(r[0] / r[2]), float(r[1] / r[2])


def match_and_triangulate(tracks_cam1, tracks_cam2,
                          H_cam1, H_cam2,
                          cam1_pos, cam2_pos,
                          max_ray_dist=1.0, min_overlap=10,
                          max_tracks=50):
    """Match tracks across cameras, triangulate overlapping frames.

    For each pair of tracks with sufficient frame overlap:
    - Triangulate the overlapping frames
    - Score by median ray distance and z reasonableness

    Args:
        tracks_cam1/cam2: from track_single_camera()
        H_cam1/cam2: homography matrices (pixel → world)
        cam1_pos/cam2_pos: camera 3D positions [x, y, z]
        max_ray_dist: max median ray distance for valid match
        min_overlap: min common frames between tracks
        max_tracks: limit tracks per camera to check

    Returns:
        List of matched trajectories sorted by overlap length.
        Each = {"trajectory": [(frame, x, y, z, px1, py1, px2, py2, ray_dist), ...],
                "median_ray_dist": float, "overlap": int}
    """
    results = []

    for t1 in tracks_cam1[:max_tracks]:
        frames1 = {p[0]: (p[1], p[2]) for p in t1}
        f1_set = set(frames1.keys())

        for t2 in tracks_cam2[:max_tracks]:
            frames2 = {p[0]: (p[1], p[2]) for p in t2}
            f2_set = set(frames2.keys())

            overlap = sorted(f1_set & f2_set)
            if len(overlap) < min_overlap:
                continue

            traj3d = []
            ray_dists = []
            for fi in overlap:
                px1, py1 = frames1[fi]
                px2, py2 = frames2[fi]
                w1 = pixel_to_world(px1, py1, H_cam1)
                w2 = pixel_to_world(px2, py2, H_cam2)
                pt3d, rd = triangulate_ray(w1, w2, cam1_pos, cam2_pos)

                traj3d.append((fi, pt3d[0], pt3d[1], pt3d[2],
                               px1, py1, px2, py2, rd))
                ray_dists.append(rd)

            med_rd = np.median(ray_dists)
            zs = [t[3] for t in traj3d]
            z_ok = sum(1 for z in zs if 0.05 <= z <= 5.0) / len(zs)

            if med_rd < max_ray_dist and z_ok > 0.5:
                results.append({
                    "trajectory": traj3d,
                    "median_ray_dist": float(med_rd),
                    "z_ok_ratio": float(z_ok),
                    "overlap": len(overlap),
                })

    results.sort(key=lambda r: r["overlap"], reverse=True)
    return results
