"""Viterbi-based global optimal trajectory finder for tennis ball tracking.

Given multi-blob candidates from two cameras across all frames, finds the
globally optimal assignment of one blob-pair per frame that minimizes
total transition cost. This replaces the greedy frame-by-frame approach
of MultiBlobMatcher with a global optimization.

Key insight: instead of deciding which blob is the match ball at each frame
independently, we consider ALL possible paths through the candidate graph
and pick the one with the lowest total cost. This naturally handles:
- Ball vs shoe ambiguity (shoes are static, ball moves in parabolas)
- Dead-ball periods (low-speed paths get penalized)
- Rally transitions (gap segmentation + independent optimization)

Algorithm: Viterbi (dynamic programming on a trellis/HMM graph).
Complexity: O(N * K^2) where N=frames, K=max candidates per frame.
For K=4 (top-2 per cam = 4 pairs), N=1800: ~28800 operations.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Court constants (singles)
SINGLES_X_MIN = 1.37
SINGLES_X_MAX = 6.86
COURT_L = 23.77
NET_Y = COURT_L / 2  # 11.885


def _triangulate(world_2d_1, world_2d_2, cam1_pos, cam2_pos):
    """Triangulate and return (x, y, z, ray_distance)."""
    cam1 = np.asarray(cam1_pos, dtype=np.float64)
    cam2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([world_2d_1[0], world_2d_1[1], 0.0])
    g2 = np.array([world_2d_2[0], world_2d_2[1], 0.0])

    d1 = g1 - cam1
    d2 = g2 - cam2
    w = cam1 - cam2

    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d_val = float(np.dot(d1, w))
    e = float(np.dot(d2, w))

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        mid = (g1 + g2) / 2.0
        return float(mid[0]), float(mid[1]), 0.0, 999.0

    s = (b * e - c * d_val) / denom
    t_val = (a * e - b * d_val) / denom
    s = np.clip(s, 0.0, 1.0)
    t_val = np.clip(t_val, 0.0, 1.0)

    p1_f = cam1 + s * d1
    t_val = float(np.dot(p1_f - cam2, d2)) / c if c > 1e-10 else t_val
    t_val = np.clip(t_val, 0.0, 1.0)
    p2_f = cam2 + t_val * d2
    s = float(np.dot(p2_f - cam1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0.0, 1.0)

    p1 = cam1 + s * d1
    p2 = cam2 + t_val * d2
    mid = (p1 + p2) / 2.0
    ray_dist = float(np.linalg.norm(p1 - p2))

    if mid[2] < 0:
        mid[2] = 0.0

    return float(mid[0]), float(mid[1]), float(mid[2]), ray_dist


def _emission_cost(x, y, z, ray_dist):
    """Cost of a single observation (how likely this is a real ball detection).

    Lower cost = more likely to be a real ball.
    """
    cost = 0.0

    # Ray distance: cameras must agree on 3D position
    cost += ray_dist * 2.0

    # Z sanity: ball should be 0-6m
    if z < 0 or z > 8.0:
        return 999.0

    # Court proximity: ball near court is more likely match ball
    # (generous margin — ball can be a few meters outside)
    if x < SINGLES_X_MIN - 3 or x > SINGLES_X_MAX + 3 or y < -3 or y > COURT_L + 3:
        cost += 3.0

    return cost


def _transition_cost(prev, curr, dt, prev_prev=None, dt_prev=None):
    """Cost of transitioning between two 3D points.

    Encodes physical plausibility: real match balls follow smooth
    trajectories with reasonable speed, cross the net, etc.

    Args:
        prev: dict with x, y, z
        curr: dict with x, y, z
        dt: time between frames in seconds
        prev_prev: optional dict from 2 frames ago (for oscillation detection)
        dt_prev: time between prev_prev and prev
    """
    if dt <= 0:
        return 0.0

    dx = curr["x"] - prev["x"]
    dy = curr["y"] - prev["y"]
    dz = curr["z"] - prev["z"]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    speed = dist / dt

    # ── Speed cost ──────────────────────────────────────────────
    if speed > 90:
        # > 90 m/s (~324 km/h) physically impossible
        return 999.0

    if speed < 1.5:
        # Nearly static = likely shoe or dead ball.
        # Strong penalty: a static point with ray_dist=0.14 has emission=0.28,
        # so this must be high enough to overcome that advantage.
        speed_cost = 5.0
    elif 5 < speed < 70:
        # Reasonable match ball speed range
        speed_cost = 0.0
    elif speed < 5:
        # Slow but not static (1.5-5 m/s) — possible but suspicious
        speed_cost = 2.0
    else:
        # Very fast but plausible (70-90 m/s)
        speed_cost = 1.0

    # ── Oscillation penalty (requires 3 frames) ────────────────
    # If the ball "jumps back" to near where it was 2 frames ago,
    # this is physically impossible — it's switching between two
    # static detections (e.g., ball vs shoe).
    oscillation_cost = 0.0
    if prev_prev is not None:
        dx_pp = curr["x"] - prev_prev["x"]
        dy_pp = curr["y"] - prev_prev["y"]
        dz_pp = curr["z"] - prev_prev["z"]
        dist_to_pp = np.sqrt(dx_pp**2 + dy_pp**2 + dz_pp**2)

        # If current position is closer to prev_prev than to prev,
        # the trajectory is zigzagging — heavy penalty
        dist_prev_to_pp = np.sqrt(
            (prev["x"] - prev_prev["x"])**2 +
            (prev["y"] - prev_prev["y"])**2 +
            (prev["z"] - prev_prev["z"])**2
        )
        if dist_to_pp < dist_prev_to_pp * 0.5 and dist_prev_to_pp > 0.5:
            # Snapping back to 2-frames-ago position
            oscillation_cost = 10.0

    # ── Direction reversal penalty ──────────────────────────────
    # Check if Y direction reversed (ball can't go forward then backward
    # in one frame unless bounced off something — rare)
    reversal_cost = 0.0
    if prev_prev is not None and dt_prev and dt_prev > 0:
        vy_prev = (prev["y"] - prev_prev["y"]) / dt_prev
        vy_curr = dy / dt
        # Strong reversal: both speeds > 3 m/s but opposite direction
        if abs(vy_prev) > 3 and abs(vy_curr) > 3:
            if vy_prev * vy_curr < 0:  # opposite signs
                reversal_cost = 5.0

    # ── Net crossing bonus (negative cost = reward) ─────────────
    net_bonus = 0.0
    if (prev["y"] < NET_Y and curr["y"] > NET_Y) or \
       (prev["y"] > NET_Y and curr["y"] < NET_Y):
        if speed > 8.0:
            net_bonus = -8.0

    return speed_cost + oscillation_cost + reversal_cost + net_bonus


class ViterbiTracker:
    """Find globally optimal ball trajectory through multi-blob candidates.

    Segments the frame sequence by gaps, runs Viterbi independently on
    each segment, then merges results.

    Args:
        cam1_pos: 3D position of camera 1
        cam2_pos: 3D position of camera 2
        max_ray_distance: maximum ray distance to accept a triangulation
        valid_z_range: (z_min, z_max) for valid 3D points
        fps: video frame rate
        gap_threshold: frames gap to split into segments
    """

    def __init__(
        self,
        cam1_pos: list[float],
        cam2_pos: list[float],
        max_ray_distance: float = 2.5,
        valid_z_range: tuple[float, float] = (0.0, 8.0),
        fps: float = 25.0,
        gap_threshold: int = 5,
        stereo_cal: dict = None,
    ):
        self.cam1_pos = cam1_pos
        self.cam2_pos = cam2_pos
        self.max_ray_distance = max_ray_distance
        self.z_min, self.z_max = valid_z_range
        self.fps = fps
        self.gap_threshold = gap_threshold
        self.stereo_cal = stereo_cal  # solvePnP calibration for reprojection check

        # Stats
        self.total_frames = 0
        self.matched_frames = 0
        self.segments_count = 0
        self.net_crossings_in_path = 0

    def _reprojection_cost(self, x3d, y3d, z3d, px66, py66, px68, py68):
        """Compute reprojection consistency cost.

        Projects the 3D point back to both cameras and checks if the
        projected pixels match the actual detected blob positions.
        Large discrepancy = the two cameras are seeing different objects.

        Returns cost in range [0, ~20].
        """
        if self.stereo_cal is None:
            return 0.0

        import cv2 as cv

        total_err = 0.0
        pt3d = np.array([[x3d, y3d, z3d]], dtype=np.float64)

        for cam, actual_px, actual_py in [
            ("cam66", px66, py66),
            ("cam68", px68, py68),
        ]:
            cal = self.stereo_cal[cam]
            K = cal["K"]
            dist = cal["dist"]
            R = np.array(cal["P"])[:, :3]  # 3x3 from P
            # Extract rvec and tvec from P = K @ [R|t]
            # R_cam = K_inv @ P[:,:3], t_cam = K_inv @ P[:,3]
            K_inv = np.linalg.inv(K)
            R_cam = K_inv @ np.array(cal["P"])[:, :3]
            t_cam = (K_inv @ np.array(cal["P"])[:, 3]).reshape(3, 1)
            rvec, _ = cv.Rodrigues(R_cam)

            proj, _ = cv.projectPoints(pt3d, rvec, t_cam, K, dist)
            proj_px = proj.reshape(2)

            err = np.sqrt((proj_px[0] - actual_px)**2 + (proj_px[1] - actual_py)**2)
            total_err += err

        # Average pixel error across both cameras
        avg_err = total_err / 2.0

        # Map to cost: <10px = 0, 10-50px = linear, >50px = heavy penalty
        if avg_err < 10:
            return 0.0
        elif avg_err < 50:
            return (avg_err - 10) * 0.15  # 0 to 6.0
        else:
            return 6.0 + (avg_err - 50) * 0.1  # 6.0+

    def _build_candidates(self, frame_idx, blobs66, blobs68, homo66, homo68):
        """Build all valid (cam66_blob, cam68_blob) pairs for one frame.

        Returns list of dicts with 3D position, ray_dist, pixel coords, etc.
        """
        candidates = []

        for i, b66 in enumerate(blobs66):
            w66 = homo66.pixel_to_world(b66["pixel_x"], b66["pixel_y"])
            for j, b68 in enumerate(blobs68):
                w68 = homo68.pixel_to_world(b68["pixel_x"], b68["pixel_y"])

                x, y, z, rd = _triangulate(w66, w68, self.cam1_pos, self.cam2_pos)

                # Hard filter
                if rd > self.max_ray_distance:
                    continue
                if z < self.z_min or z > self.z_max:
                    continue

                # Reprojection consistency: are both cameras seeing the same object?
                reproj_cost = self._reprojection_cost(
                    x, y, z,
                    b66["pixel_x"], b66["pixel_y"],
                    b68["pixel_x"], b68["pixel_y"],
                )

                em_cost = _emission_cost(x, y, z, rd) + reproj_cost

                candidates.append({
                    "x": x, "y": y, "z": z,
                    "ray_dist": rd,
                    "cam66_pixel": (b66["pixel_x"], b66["pixel_y"]),
                    "cam68_pixel": (b68["pixel_x"], b68["pixel_y"]),
                    "blob_rank": (i, j),
                    "emission_cost": em_cost,
                    "reproj_cost": reproj_cost,
                })

        return candidates

    def _viterbi_segment(self, segment_frames, all_candidates):
        """Run Viterbi on one continuous segment.

        Args:
            segment_frames: list of frame indices (sorted, continuous)
            all_candidates: {frame_idx: list[candidate_dict]}

        Returns:
            path: {frame_idx: candidate_dict} — the optimal assignment
        """
        n = len(segment_frames)
        if n == 0:
            return {}

        # dp[i][j] = min cost to reach frame i, candidate j
        # parent[i][j] = index of best predecessor at frame i-1
        dp = [None] * n
        parent = [None] * n

        # Initialize first frame
        cands0 = all_candidates.get(segment_frames[0], [])
        if not cands0:
            return {}

        dp[0] = {j: c["emission_cost"] for j, c in enumerate(cands0)}
        parent[0] = {j: None for j in range(len(cands0))}

        # Forward pass
        for i in range(1, n):
            fi = segment_frames[i]
            fi_prev = segment_frames[i - 1]
            dt = (fi - fi_prev) / self.fps

            cands = all_candidates.get(fi, [])
            cands_prev = all_candidates.get(fi_prev, [])

            if not cands:
                # No candidates at this frame — skip
                dp[i] = {}
                parent[i] = {}
                continue

            if not cands_prev or not dp[i - 1]:
                # No previous candidates — restart
                dp[i] = {j: c["emission_cost"] for j, c in enumerate(cands)}
                parent[i] = {j: None for j in range(len(cands))}
                continue

            dp[i] = {}
            parent[i] = {}

            # Get prev_prev candidates for oscillation detection
            cands_pp = None
            dt_prev = None
            if i >= 2:
                fi_pp = segment_frames[i - 2]
                cands_pp = all_candidates.get(fi_pp, [])
                dt_prev = (fi_prev - fi_pp) / self.fps

            for j, c in enumerate(cands):
                best_cost = float("inf")
                best_prev_j = None

                for k in dp[i - 1]:
                    cp = cands_prev[k]

                    # Get prev_prev candidate for this path
                    pp = None
                    if cands_pp and parent[i - 1] and parent[i - 1].get(k) is not None:
                        pp_idx = parent[i - 1][k]
                        if pp_idx < len(cands_pp):
                            pp = cands_pp[pp_idx]

                    trans = _transition_cost(cp, c, dt, prev_prev=pp, dt_prev=dt_prev)
                    cost = dp[i - 1][k] + trans + c["emission_cost"]

                    if cost < best_cost:
                        best_cost = cost
                        best_prev_j = k

                if best_prev_j is not None:
                    dp[i][j] = best_cost
                    parent[i][j] = best_prev_j

        # Backtrack: find best ending state
        path = {}

        # Find last frame with candidates
        last_i = n - 1
        while last_i >= 0 and not dp[last_i]:
            last_i -= 1

        if last_i < 0:
            return {}

        # Best candidate at last frame
        best_j = min(dp[last_i], key=dp[last_i].get)
        fi = segment_frames[last_i]
        cands = all_candidates.get(fi, [])
        if best_j < len(cands):
            path[fi] = cands[best_j]

        # Backtrack
        for i in range(last_i, 0, -1):
            prev_j = parent[i].get(best_j)
            if prev_j is None:
                break
            fi_prev = segment_frames[i - 1]
            cands_prev = all_candidates.get(fi_prev, [])
            if prev_j < len(cands_prev):
                path[fi_prev] = cands_prev[prev_j]
            best_j = prev_j

        return path

    def track(self, multi66, multi68, homo66, homo68):
        """Run Viterbi tracking on all frames.

        Args:
            multi66: {frame_idx: list[blob_dict]} from cam66
            multi68: {frame_idx: list[blob_dict]} from cam68
            homo66: HomographyTransformer for cam66
            homo68: HomographyTransformer for cam68

        Returns:
            points_3d: {frame: (x, y, z, ray_dist)}
            chosen_pixels: {frame: {'cam66': (px,py), 'cam68': (px,py)}}
            stats: dict with tracking statistics
        """
        common = sorted(set(multi66.keys()) & set(multi68.keys()))
        self.total_frames = len(common)
        logger.info("Viterbi tracker: %d common frames", len(common))

        if not common:
            return {}, {}, self.get_stats()

        # ── Step 1: Build all candidates for all frames ──────────
        all_candidates = {}
        total_cands = 0
        for fi in common:
            cands = self._build_candidates(
                fi, multi66[fi], multi68[fi], homo66, homo68,
            )
            if cands:
                all_candidates[fi] = cands
                total_cands += len(cands)

        logger.info(
            "Built candidates: %d frames with candidates, %d total pairs (avg %.1f/frame)",
            len(all_candidates), total_cands,
            total_cands / max(1, len(all_candidates)),
        )

        # ── Step 2: Segment by gaps ──────────────────────────────
        frames_with_cands = sorted(all_candidates.keys())
        segments = []
        if frames_with_cands:
            seg_start = 0
            for i in range(1, len(frames_with_cands)):
                if frames_with_cands[i] - frames_with_cands[i - 1] > self.gap_threshold:
                    segments.append(frames_with_cands[seg_start:i])
                    seg_start = i
            segments.append(frames_with_cands[seg_start:])

        self.segments_count = len(segments)
        logger.info("Split into %d segments (gap threshold=%d frames)",
                     len(segments), self.gap_threshold)

        # ── Step 3: Run Viterbi on each segment ──────────────────
        full_path = {}
        for seg in segments:
            if len(seg) < 3:
                continue
            seg_path = self._viterbi_segment(seg, all_candidates)
            full_path.update(seg_path)

        # ── Step 4: Extract results ──────────────────────────────
        points_3d = {}
        chosen_pixels = {}
        non_top1 = 0
        net_cross = 0
        prev_y = None

        for fi in sorted(full_path.keys()):
            c = full_path[fi]
            points_3d[fi] = (c["x"], c["y"], c["z"], c["ray_dist"])
            chosen_pixels[fi] = {
                "cam66": c["cam66_pixel"],
                "cam68": c["cam68_pixel"],
            }
            if c["blob_rank"] != (0, 0):
                non_top1 += 1

            # Count net crossings
            if prev_y is not None:
                if (prev_y < NET_Y and c["y"] > NET_Y) or \
                   (prev_y > NET_Y and c["y"] < NET_Y):
                    net_cross += 1
            prev_y = c["y"]

        self.matched_frames = len(points_3d)
        self.net_crossings_in_path = net_cross

        logger.info(
            "Viterbi result: %d/%d frames matched, non_top1=%d (%.1f%%), "
            "net_crossings=%d",
            self.matched_frames, self.total_frames,
            non_top1, 100 * non_top1 / max(1, self.matched_frames),
            net_cross,
        )

        return points_3d, chosen_pixels, self.get_stats()

    def get_stats(self):
        return {
            "total_frames": self.total_frames,
            "matched_frames": self.matched_frames,
            "segments": self.segments_count,
            "net_crossings": self.net_crossings_in_path,
        }
