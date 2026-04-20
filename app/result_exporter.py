"""Rally result formatter and exporter.

Converts a completed rally's raw frame buffer into the standard result JSON
and POSTs it to the configured endpoint.
"""

import datetime
import logging
import math
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --- Court constants (V2: origin at center, net at y=0) ---
_X_MIN = -4.115   # left sideline
_X_MAX = 4.115    # right sideline
_X_RANGE = _X_MAX - _X_MIN        # 8.23 m
_Y_MIN = -11.89   # near baseline
_Y_MAX = 11.89    # far baseline
_Y_RANGE = _Y_MAX - _Y_MIN        # 23.78 m

# COCO keypoint indices
_KP_R_SHOULDER = 6
_KP_R_WRIST = 10
_KP_CONF_MIN = 0.3


def _norm_x(x: float) -> float:
    return round((x - _X_MIN) / _X_RANGE, 4)


def _norm_y(y: float) -> float:
    return round((y - _Y_MIN) / _Y_RANGE, 4)


def _infer_hand_type(keypoints_px: list) -> str:
    if len(keypoints_px) <= _KP_R_WRIST:
        return "forehand"
    r_shoulder = keypoints_px[_KP_R_SHOULDER]
    r_wrist = keypoints_px[_KP_R_WRIST]
    if r_shoulder[2] < _KP_CONF_MIN or r_wrist[2] < _KP_CONF_MIN:
        return "forehand"
    return "backhand" if r_shoulder[0] > r_wrist[0] else "forehand"


def _player_foot_norm(player: Optional[dict]) -> tuple[float, float]:
    """Return normalized (x, y) of player foot, or (0, 0) if unavailable."""
    if player is None:
        return 0.0, 0.0
    fc = player.get("foot_court")
    if not fc or len(fc) < 2:
        return 0.0, 0.0
    return _norm_x(fc[0]), _norm_y(fc[1])


def _player_dist_m(fc_a: Optional[list], fc_b: Optional[list]) -> float:
    """Euclidean distance in metres between two foot_court [x, y] positions."""
    if fc_a is None or fc_b is None:
        return 0.0
    dx = (fc_a[0] - fc_b[0])
    dy = (fc_a[1] - fc_b[1])
    return math.sqrt(dx * dx + dy * dy)


def _compute_player_stats(frames: list, side: str) -> dict:
    """Compute movement stats for one side's player across all frames.

    side: "near" (y < 0) or "far" (y >= 0)
    """
    key = "near_player" if side == "near" else "far_player"
    positions = []   # list of foot_court [x, y]
    times = []       # corresponding timestamps

    for fr in frames:
        p = fr.get(key)
        if p and p.get("foot_court"):
            positions.append(p["foot_court"])
            times.append(fr["ts"])

    if not positions:
        return {
            "totalDistance": 0,
            "avgMoveSpeed": 0,
            "maxMoveSpeed": 0,
        }

    total_dist = 0.0
    max_speed = 0.0
    for i in range(1, len(positions)):
        d = _player_dist_m(positions[i - 1], positions[i])
        total_dist += d
        dt = times[i] - times[i - 1]
        if dt > 0.001:
            spd = d / dt  # m/s
            if spd > max_speed:
                max_speed = spd

    duration = times[-1] - times[0] if len(times) > 1 else 1.0
    avg_speed = total_dist / duration if duration > 0.001 else 0.0

    return {
        "totalDistance": round(total_dist, 2),
        "avgMoveSpeed": round(avg_speed, 3),
        "maxMoveSpeed": round(max_speed, 3),
    }


def _compute_ball_speed_stats(frames: list) -> tuple[float, float]:
    """Return (avg_speed_kmh, max_speed_kmh) from frames with speed_kmh > 0."""
    speeds = [fr["speed_kmh"] for fr in frames if fr.get("speed_kmh", 0) > 0]
    if not speeds:
        return 0.0, 0.0
    return round(sum(speeds) / len(speeds), 1), round(max(speeds), 1)


def _build_result_matrix(frames: list) -> list:
    result = []
    for fr in frames:
        if not fr.get("is_bounce") and not fr.get("is_hit"):
            continue
        ball = fr.get("ball")
        if ball is None:
            continue

        entry_type = "bounce" if fr["is_bounce"] else "hit"

        # Determine hand type from the player closest to ball
        hand_type = "forehand"
        ball_world_y = ball["y"]
        player_key = "near_player" if ball_world_y < 0 else "far_player"
        player = fr.get(player_key)
        if player and player.get("keypoints_px"):
            hand_type = _infer_hand_type(player["keypoints_px"])

        result.append({
            "x": _norm_x(ball["x"]),
            "y": _norm_y(ball["y"]),
            "type": entry_type,
            "speed": round(fr.get("speed_kmh", 0), 1),
            "handType": hand_type,
        })
    return result


def _build_track_matrix(frames: list) -> list:
    track = []
    for frame_idx, fr in enumerate(frames):
        ball = fr.get("ball")
        bx = _norm_x(ball["x"]) if ball else 0.0
        by = _norm_y(ball["y"]) if ball else 0.0

        if fr.get("is_bounce"):
            state = "bounce"
        elif fr.get("is_hit"):
            state = "hit"
        else:
            state = "running"

        near_x, near_y = _player_foot_norm(fr.get("near_player"))
        far_x, far_y = _player_foot_norm(fr.get("far_player"))

        track.append({
            "x": bx,
            "y": by,
            "type": state,
            "speed": round(fr.get("speed_kmh", 0), 1),
            "timestamp": frame_idx,
            "farCountPerson_x": far_x,
            "farCountPerson_y": far_y,
            "nearCountPerson_x": near_x,
            "nearCountPerson_y": near_y,
        })
    return track


def format_rally(
    rally_result,
    frames: list,
    serial_number: str,
    endpoint: str,
    dry_run: bool = False,
) -> dict:
    """Format rally data into the standard result JSON and POST to endpoint.

    Args:
        rally_result: RallyResult instance with start_time, end_time, bounces.
        frames: list of per-frame dicts from _rally_raw_buffer.
        serial_number: device serial number string.
        endpoint: HTTP endpoint to POST to.

    Returns:
        The formatted payload dict.
    """
    start_dt = datetime.datetime.utcfromtimestamp(rally_result.start_time)
    end_dt = datetime.datetime.utcfromtimestamp(rally_result.end_time)

    avg_ball_speed, max_ball_speed = _compute_ball_speed_stats(frames)
    near_stats = _compute_player_stats(frames, "near")
    far_stats = _compute_player_stats(frames, "far")

    result_matrix = _build_result_matrix(frames)
    track_matrix = _build_track_matrix(frames)

    def _side_block(stats: dict, avg_spd: float, max_spd: float) -> dict:
        return {
            "firstServeSuccessRate": 0,
            "returnFirstSuccessRate": 0,
            "baselineShotRate": 0,
            "baselineWinRate": 0,
            "netPointRate": 0,
            "netPointWinRate": 0,
            "totalShots": 0,
            "aceCount": 0,
            "netApproaches": 0,
            "avgBallSpeed": avg_spd,
            "maxBallSpeed": max_spd,
            "totalDistance": stats["totalDistance"],
            "avgMoveSpeed": stats["avgMoveSpeed"],
            "maxMoveSpeed": stats["maxMoveSpeed"],
        }

    payload = {
        "serial_number": serial_number,
        "startTime": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content": {
            "mete": {
                "movingDistance": 0,
                "battingAttempts": 0,
                "serveSuccessRate": 0,
                "serveAceCount": 0,
                "serveSpeed": 0,
                "receiveSuccessRate": 0,
                "breakPointConversionRate": 0,
                "hitSuccessRate": 0,
                "forehandHitRate": 0,
                "backhandHitRate": 0,
                "baselineScoreRate": 0,
                "averageHitSpeed": avg_ball_speed,
                "maxHitSpeed": max_ball_speed,
                "netApproachCount": 0,
                "netScoreRate": 0,
                "volleyScoreRate": 0,
                "farCount": _side_block(far_stats, avg_ball_speed, max_ball_speed),
                "nearCount": _side_block(near_stats, avg_ball_speed, max_ball_speed),
            },
            "resultmatrix": result_matrix,
            "trackMatrix": track_matrix,
        },
    }

    if not dry_run:
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            if resp.ok:
                logger.info(
                    "Rally %d exported → %s (%d)",
                    rally_result.rally_id, endpoint, resp.status_code,
                )
                mete = payload["content"]["mete"]
                near = mete["nearCount"]
                far = mete["farCount"]
                print(
                    f"\n{'='*50}\n"
                    f"[Rally {rally_result.rally_id}] 数据已发送\n"
                    f"  时间: {payload['startTime']} → {payload['endTime']}\n"
                    f"  球速: 均值={mete['averageHitSpeed']:.1f} km/h  最大={mete['maxHitSpeed']:.1f} km/h\n"
                    f"  近端球员: 移动距离={near['totalDistance']:.1f}m  均速={near['avgMoveSpeed']:.2f}m/s  最大={near['maxMoveSpeed']:.2f}m/s\n"
                    f"  远端球员: 移动距离={far['totalDistance']:.1f}m  均速={far['avgMoveSpeed']:.2f}m/s  最大={far['maxMoveSpeed']:.2f}m/s\n"
                    f"{'='*50}\n"
                )
            else:
                logger.warning(
                    "Rally %d export failed: %s %s",
                    rally_result.rally_id, resp.status_code, resp.text[:200],
                )
        except Exception as e:
            logger.warning("Rally %d export error: %s", rally_result.rally_id, e)

    return payload
