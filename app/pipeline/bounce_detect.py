"""
Event detection from 3D trajectory: bounce, shot, serve.

From research: event_detector.py + bounce_detector.py
- Bounce: local minimum in z where z < z_max (find_peaks on -z)
- Shot: sudden velocity change at higher z
- Serve: first shot near baseline
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# V2 coords
COURT_HW = 4.115
COURT_HL = 11.885
COURT_MARGIN = 0.15


def is_in_court(x, y, margin=COURT_MARGIN):
    return abs(x) <= COURT_HW + margin and abs(y) <= COURT_HL + margin


def detect_bounces(trajectory, z_max=0.5, prominence=0.10, min_distance=5, smooth=3):
    """Detect bounce frames from 3D trajectory.

    Args:
        trajectory: list of (frame, x, y, z, ...) tuples
        z_max: maximum z to count as bounce
        prominence: minimum prominence for find_peaks on -z
        min_distance: minimum frame separation between bounces
        smooth: uniform_filter1d window

    Returns:
        List of {'frame', 'z', 'x', 'y', 'in_court', 'type': 'bounce'}
    """
    if len(trajectory) < 5:
        return []

    frames = np.array([t[0] for t in trajectory])
    zs = np.array([t[3] for t in trajectory])

    zs_smooth = uniform_filter1d(zs, size=smooth) if len(zs) > smooth else zs.copy()

    neg_z = -zs_smooth
    peaks, _ = find_peaks(neg_z, prominence=prominence, distance=min_distance)

    bounces = []
    for pi in peaks:
        # z must be positive (z=0 means triangulation failed, not a real bounce)
        if 0.02 < zs_smooth[pi] <= z_max:
            bx = float(trajectory[pi][1])
            by = float(trajectory[pi][2])
            bounces.append({
                'frame': int(frames[pi]),
                'type': 'bounce',
                'z': float(zs_smooth[pi]),
                'x': bx,
                'y': by,
                'in_court': is_in_court(bx, by),
            })

    return bounces


def detect_events(trajectory, fps=25,
                  bounce_z_max=0.5,
                  bounce_min_prominence=0.10,
                  speed_change_thresh=15.0):
    """Detect bounce, shot, serve events from 3D trajectory.

    Args:
        trajectory: list of dicts with 'frame', 'x', 'y', 'z'
                    OR tuples (frame, x, y, z, ...)
        fps: frame rate
        bounce_z_max: max z for bounce detection
        bounce_min_prominence: min prominence in -z for peaks
        speed_change_thresh: min speed change for shot detection (m/s)

    Returns:
        List of event dicts: {'frame', 'type', 'z', 'x', 'y', ...}
    """
    if len(trajectory) < 5:
        return []

    # Support both dict and tuple formats
    if isinstance(trajectory[0], dict):
        frames = np.array([t['frame'] for t in trajectory])
        xs = np.array([t['x'] for t in trajectory])
        ys = np.array([t['y'] for t in trajectory])
        zs = np.array([t['z'] for t in trajectory])
    else:
        frames = np.array([t[0] for t in trajectory])
        xs = np.array([t[1] for t in trajectory])
        ys = np.array([t[2] for t in trajectory])
        zs = np.array([t[3] for t in trajectory])

    zs_smooth = uniform_filter1d(zs, size=3) if len(zs) > 5 else zs.copy()

    events = []

    # === BOUNCE: local minimum in z ===
    neg_z = -zs_smooth
    peaks, _ = find_peaks(neg_z, prominence=bounce_min_prominence, distance=5)

    for pi in peaks:
        if zs_smooth[pi] <= bounce_z_max:
            events.append({
                'frame': int(frames[pi]),
                'type': 'bounce',
                'z': float(zs_smooth[pi]),
                'x': float(xs[pi]),
                'y': float(ys[pi]),
                'in_court': is_in_court(float(xs[pi]), float(ys[pi])),
            })

    # === SHOT: sudden velocity change at higher z ===
    dt = np.diff(frames) / fps
    dt[dt == 0] = 1.0 / fps

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    vz = np.diff(zs_smooth) / dt
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    if len(speed) >= 2:
        dspeed = np.abs(np.diff(speed))
        for i in range(len(dspeed)):
            if dspeed[i] > speed_change_thresh and zs_smooth[i + 1] > bounce_z_max:
                fi = int(frames[i + 1])
                near_bounce = any(abs(fi - e['frame']) < 5
                                  for e in events if e['type'] == 'bounce')
                if not near_bounce:
                    events.append({
                        'frame': fi,
                        'type': 'shot',
                        'z': float(zs_smooth[i + 1]),
                        'x': float(xs[i + 1]),
                        'y': float(ys[i + 1]),
                    })

    # === SERVE: first shot near baseline ===
    for e in sorted(events, key=lambda e: e['frame']):
        if e['type'] == 'shot' and abs(e['y']) > 8:
            e['type'] = 'serve'
            break

    events.sort(key=lambda e: e['frame'])
    return events
