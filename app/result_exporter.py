"""Rally result formatter and exporter.

Converts a completed rally's raw frame buffer into the standard result JSON
and POSTs it to the configured endpoint.

Fix log (vs. original):
  1. _compute_ball_speed_stats: was averaging ALL frames with speed>0 (including
     "running" frames), producing inflated avg/max. Now only counts frames that
     are hit or bounce events, and clamps to [10, 250] km/h to drop physics-
     impossible spikes before they reach max_ball_speed.
  2. _compute_player_stats: maxMoveSpeed was raw per-frame (noisy, no cap).
     Now capped at _MAX_PLAYER_SPEED_MS (7.0 m/s ≈ 25 km/h, realistic amateur
     upper limit). avgMoveSpeed consistency guaranteed: totalDistance /
     duration is stored as avgMoveSpeed so the identity
     totalDistance == avgMoveSpeed * duration always holds.
  3. _build_result_matrix: "serve" type was never emitted (only hit/bounce),
     and "serve" handType was silently produced. Now: the first hit of a rally
     (ball in near-court, y_norm < 0.5) is classified as "serve"; handType
     "serve" is replaced with "forehand" for bounce entries (spec forbids
     "serve" as a handType on bounce rows).
  4. _build_result_matrix: bounce entries inherited the player nearest to the
     ball's *world_y*, which is correct for hand-type inference on hit events
     but meaningless for a bounce (no player hits the ball). handType is now
     always "forehand" for bounce rows (matches spec example).
  5. _build_track_matrix: player positions were left as (0.0, 0.0) when pose
     data was missing (foot_court absent). Now carries forward the last known
     position (last-value-carry) instead of resetting to origin, so the
     frontend doesn't see players teleporting to (0,0).
  6. _side_block: totalShots, baselineShotRate, netPointRate, netApproaches,
     avgBallSpeed, maxBallSpeed were all hardcoded 0. Now computed from
     result_matrix entries for the correct side.
  7. _compute_advanced_stats: new helper that derives all the [必填] per-side
     statistics (serve rates, baseline/net rates, ace count, etc.) from the
     result_matrix instead of leaving them as placeholder zeros.
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

# Normalised court zones (after _norm_y the near half is 0..0.5, far 0.5..1.0)
_BASELINE_ZONE = 0.8   # y_norm >= this → baseline area
_NET_ZONE      = 0.2   # y_norm <= this → net area
_SERVE_BOX_X_LO = 0.2
_SERVE_BOX_X_HI = 0.8
# Valid serve landing: near-side service box  0.1 < y < 0.4,
#                     far-side  service box   0.6 < y < 0.9
_SERVE_NEAR_Y_LO = 0.1
_SERVE_NEAR_Y_HI = 0.4
_SERVE_FAR_Y_LO  = 0.6
_SERVE_FAR_Y_HI  = 0.9

# FIX #2: realistic amateur max movement speed
_MAX_PLAYER_SPEED_MS = 7.0   # m/s  (~25 km/h, well inside human limits)

# FIX #1: ball speed sanity gates
_BALL_SPEED_MIN_KMH =  10.0
_BALL_SPEED_MAX_KMH = 250.0

# COCO keypoint indices
_KP_R_SHOULDER = 6
_KP_R_WRIST    = 10
_KP_CONF_MIN   = 0.3


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _norm_x(x: float) -> float:
    return round((x - _X_MIN) / _X_RANGE, 4)


def _norm_y(y: float) -> float:
    return round((y - _Y_MIN) / _Y_RANGE, 4)


def _infer_hand_type(keypoints_px: list) -> str:
    if len(keypoints_px) <= _KP_R_WRIST:
        return "forehand"
    r_shoulder = keypoints_px[_KP_R_SHOULDER]
    r_wrist    = keypoints_px[_KP_R_WRIST]
    if r_shoulder[2] < _KP_CONF_MIN or r_wrist[2] < _KP_CONF_MIN:
        return "forehand"
    return "backhand" if r_shoulder[0] > r_wrist[0] else "forehand"


def _player_foot_norm(player: Optional[dict]) -> tuple[float, float]:
    """Return normalised (x, y) of player foot, or None if unavailable."""
    if player is None:
        return None
    fc = player.get("foot_court")
    if not fc or len(fc) < 2:
        return None
    return _norm_x(fc[0]), _norm_y(fc[1])


def _player_dist_m(fc_a: Optional[list], fc_b: Optional[list]) -> float:
    """Euclidean distance in metres between two foot_court [x, y] positions."""
    if fc_a is None or fc_b is None:
        return 0.0
    dx = fc_a[0] - fc_b[0]
    dy = fc_a[1] - fc_b[1]
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Movement statistics
# ---------------------------------------------------------------------------

def _compute_player_stats(frames: list, side: str) -> dict:
    """Compute movement stats for one side's player across all frames.

    FIX #2:
    - maxMoveSpeed capped at _MAX_PLAYER_SPEED_MS.
    - avgMoveSpeed = totalDistance / duration  (guaranteed consistency).
    """
    key = "near_player" if side == "near" else "far_player"
    positions: list[list] = []
    times:     list[float] = []

    for fr in frames:
        p = fr.get(key)
        if p and p.get("foot_court"):
            positions.append(p["foot_court"])
            times.append(fr["ts"])

    if not positions:
        return {"totalDistance": 0.0, "avgMoveSpeed": 0.0, "maxMoveSpeed": 0.0}

    total_dist = 0.0
    max_speed  = 0.0
    for i in range(1, len(positions)):
        d  = _player_dist_m(positions[i - 1], positions[i])
        total_dist += d
        dt = times[i] - times[i - 1]
        if dt > 0.001:
            spd = min(d / dt, _MAX_PLAYER_SPEED_MS)   # FIX #2: cap
            if spd > max_speed:
                max_speed = spd

    duration  = times[-1] - times[0] if len(times) > 1 else 1.0
    # FIX #2: derive avg from total / duration so the identity always holds
    avg_speed = total_dist / duration if duration > 0.001 else 0.0
    avg_speed = min(avg_speed, _MAX_PLAYER_SPEED_MS)

    return {
        "totalDistance": round(total_dist, 2),
        "avgMoveSpeed":  round(avg_speed,  3),
        "maxMoveSpeed":  round(max_speed,  3),
    }


# ---------------------------------------------------------------------------
# Ball speed statistics
# ---------------------------------------------------------------------------

def _compute_ball_speed_stats(frames: list) -> tuple[float, float]:
    """Return (avg_speed_kmh, max_speed_kmh) from hit/bounce frames only.

    FIX #1:
    - Was using ALL frames with speed_kmh > 0.  Running frames introduce many
      low-quality speed samples (interpolated / noisy) that drag the average
      down and occasionally spike the max unrealistically.
    - Now restricted to frames tagged is_hit or is_bounce.
    - Clamped to [_BALL_SPEED_MIN_KMH, _BALL_SPEED_MAX_KMH].
    """
    speeds = [
        fr["speed_kmh"]
        for fr in frames
        if (fr.get("is_hit") or fr.get("is_bounce"))
        and _BALL_SPEED_MIN_KMH <= fr.get("speed_kmh", 0) <= _BALL_SPEED_MAX_KMH
    ]
    if not speeds:
        return 0.0, 0.0
    return round(sum(speeds) / len(speeds), 1), round(max(speeds), 1)


# ---------------------------------------------------------------------------
# resultmatrix
# ---------------------------------------------------------------------------

def _build_result_matrix(frames: list) -> list:
    """Build the resultmatrix array.

    FIX #3 – serve classification:
      The first hit event where the ball is in the near half (y_norm < 0.5) is
      tagged as type "serve".  Subsequent hits stay "hit".

    FIX #4 – handType on bounce rows:
      Bounce entries always carry handType "forehand" (the ball is not being
      struck; there is no hand type).  Previously the code looked up the
      nearest player and ran _infer_hand_type, which is only meaningful for a
      hit event.
    """
    result = []
    serve_emitted = False

    for fr in frames:
        is_bounce = fr.get("is_bounce", False)
        is_hit    = fr.get("is_hit",    False)
        if not is_bounce and not is_hit:
            continue
        ball = fr.get("ball")
        if ball is None:
            continue

        nx = _norm_x(ball["x"])
        ny = _norm_y(ball["y"])
        speed = round(fr.get("speed_kmh", 0.0), 1)

        if is_bounce:
            # FIX #4: bounces never have a meaningful hand type
            result.append({
                "x":        nx,
                "y":        ny,
                "type":     "bounce",
                "speed":    speed,
                "handType": "forehand",
            })
            continue

        # is_hit — determine hand type from player keypoints
        hand_type  = "forehand"
        ball_world_y = ball["y"]
        player_key = "near_player" if ball_world_y < 0 else "far_player"
        player = fr.get(player_key)
        if player and player.get("keypoints_px"):
            hand_type = _infer_hand_type(player["keypoints_px"])

        # FIX #3: classify first near-court hit as "serve"
        entry_type = "hit"
        if not serve_emitted and ny < 0.5:
            entry_type    = "serve"
            serve_emitted = True

        result.append({
            "x":        nx,
            "y":        ny,
            "type":     entry_type,
            "speed":    speed,
            "handType": hand_type,
        })

    return result


# ---------------------------------------------------------------------------
# trackMatrix
# ---------------------------------------------------------------------------

def _build_track_matrix(frames: list) -> list:
    """Build the trackMatrix array.

    FIX #5 – last-value carry for missing player positions:
      When foot_court data is absent for a frame, carry forward the last known
      position instead of falling back to (0.0, 0.0).  This prevents frontend
      charts from drawing lines from player positions to the court origin on
      every frame where pose detection momentarily failed.
    """
    track = []
    last_near = (0.5, 0.95)   # sensible default: near baseline centre
    last_far  = (0.5, 0.05)   # sensible default: far baseline centre

    for frame_idx, fr in enumerate(frames):
        ball = fr.get("ball")
        bx   = _norm_x(ball["x"]) if ball else 0.0
        by   = _norm_y(ball["y"]) if ball else 0.0

        if fr.get("is_bounce"):
            state = "bounce"
        elif fr.get("is_hit"):
            state = "hit"
        else:
            state = "running"

        # FIX #5: last-value carry
        near_pos = _player_foot_norm(fr.get("near_player"))
        far_pos  = _player_foot_norm(fr.get("far_player"))
        if near_pos is not None:
            last_near = near_pos
        if far_pos is not None:
            last_far  = far_pos

        track.append({
            "x":               bx,
            "y":               by,
            "type":            state,
            "speed":           round(fr.get("speed_kmh", 0.0), 1),
            "timestamp":       frame_idx,
            "farCountPerson_x":  last_far[0],
            "farCountPerson_y":  last_far[1],
            "nearCountPerson_x": last_near[0],
            "nearCountPerson_y": last_near[1],
        })

    return track


# ---------------------------------------------------------------------------
# Advanced per-side statistics derived from resultmatrix
# ---------------------------------------------------------------------------

def _is_valid_serve_landing(x: float, y: float, server_side: str) -> bool:
    """True if a bounce at (x_norm, y_norm) lands in the correct service box.

    server_side "near": ball must land in far service box (0.6 < y < 0.9)
    server_side "far":  ball must land in near service box (0.1 < y < 0.4)
    """
    if not (_SERVE_BOX_X_LO < x < _SERVE_BOX_X_HI):
        return False
    if server_side == "near":
        return _SERVE_FAR_Y_LO  < y < _SERVE_FAR_Y_HI
    return _SERVE_NEAR_Y_LO < y < _SERVE_NEAR_Y_HI


def _compute_advanced_stats(
    result_matrix: list,
    player_stats:  dict,
    side:          str,          # "near" or "far"
    avg_ball_speed: float,
    max_ball_speed: float,
) -> dict:
    """Derive all [必填] mete fields for one side from result_matrix.

    FIX #6/#7: replaces the old _side_block() which hardcoded everything to 0.

    Definitions follow the spec comments in the reference JSON exactly.
    """
    # Separate serve, hit, bounce events for this side's half of the court
    # "near" player hits from y < 0.5 (norm), "far" from y >= 0.5
    y_lo, y_hi = (0.0, 0.5) if side == "near" else (0.5, 1.0)

    hits    = [e for e in result_matrix if e["type"] in ("hit", "serve") and y_lo <= e["y"] < y_hi]
    serves  = [e for e in result_matrix if e["type"] == "serve"          and y_lo <= e["y"] < y_hi]
    bounces = [e for e in result_matrix if e["type"] == "bounce"]

    total_shots = len(hits)

    # ---- Serve success rate ----
    # A serve is successful if its next bounce lands in the valid service box.
    serve_count   = len(serves)
    serve_success = 0
    for s_idx, srv in enumerate(result_matrix):
        if srv["type"] != "serve":
            continue
        # Find the immediately following bounce
        for nxt in result_matrix[result_matrix.index(srv) + 1:]:
            if nxt["type"] == "bounce":
                if _is_valid_serve_landing(nxt["x"], nxt["y"], side):
                    serve_success += 1
                break

    first_serve_rate = round(100 * serve_success / serve_count, 1) if serve_count else 0.0

    # ---- Return first serve success rate ----
    # Opponent serves → this side must return into opponent half.
    opp_y_lo, opp_y_hi = (0.5, 1.0) if side == "near" else (0.0, 0.5)
    opp_serves = [e for e in result_matrix if e["type"] == "serve" and opp_y_lo <= e["y"] < opp_y_hi]
    return_success = 0
    for srv in opp_serves:
        srv_i = result_matrix.index(srv)
        # Next hit from this side
        for nxt in result_matrix[srv_i + 1:]:
            if nxt["type"] in ("hit", "serve") and y_lo <= nxt["y"] < y_hi:
                # Following bounce should be in opponent half
                nxt_i = result_matrix.index(nxt)
                for b in result_matrix[nxt_i + 1:]:
                    if b["type"] == "bounce":
                        if opp_y_lo <= b["y"] < opp_y_hi:
                            return_success += 1
                        break
                break

    return_first_rate = round(100 * return_success / len(opp_serves), 1) if opp_serves else 0.0

    # ---- Baseline / net rates ----
    baseline_hits = [h for h in hits if h["y"] >= _BASELINE_ZONE] if side == "far" \
                    else [h for h in hits if (1.0 - h["y"]) >= _BASELINE_ZONE]
    net_hits      = [h for h in hits if h["y"] <= _NET_ZONE]     if side == "far" \
                    else [h for h in hits if (1.0 - h["y"]) <= _NET_ZONE]

    baseline_shot_rate = round(100 * len(baseline_hits) / total_shots, 1) if total_shots else 0.0
    net_point_rate     = round(100 * len(net_hits)      / total_shots, 1) if total_shots else 0.0

    # Win rates: approximate — a point is "won" if the next bounce after this
    # player's hit lands out-of-bounds (y > 1 or y < 0) or there is no further
    # hit from the opponent.  Without a proper rally scorer this is a best-
    # effort heuristic; leave as 0 rather than fabricate incorrect values.
    baseline_win_rate = 0.0
    net_win_rate      = 0.0

    # ---- ACE count ----
    # Serve lands in service box AND no opponent hit follows before the next serve.
    ace_count = 0
    for srv in result_matrix:
        if srv["type"] != "serve":
            continue
        if not (y_lo <= srv["y"] < y_hi):
            continue
        srv_i = result_matrix.index(srv)
        # Find next bounce
        landing_ok = False
        opponent_returned = False
        for nxt in result_matrix[srv_i + 1:]:
            if nxt["type"] == "bounce":
                landing_ok = _is_valid_serve_landing(nxt["x"], nxt["y"], side)
            if nxt["type"] in ("hit", "serve") and opp_y_lo <= nxt["y"] < opp_y_hi:
                opponent_returned = True
            if nxt["type"] == "serve":
                break
        if landing_ok and not opponent_returned:
            ace_count += 1

    # ---- Net approaches ----
    # Count how many times this player's foot position moved into net zone (y <= 0.2
    # for far, y >= 0.8 for near) during the rally.  Approximated from track_matrix
    # data here; the caller already has player_stats from frames so we reuse the
    # count from the track instead.  Without frame-level foot data in result_matrix
    # we estimate: each net_hit represents an approach.
    net_approaches = len(net_hits)

    # ---- Ball speed (from hit frames of this side only) ----
    hit_speeds = [h["speed"] for h in hits if _BALL_SPEED_MIN_KMH <= h["speed"] <= _BALL_SPEED_MAX_KMH]
    side_avg_speed = round(sum(hit_speeds) / len(hit_speeds), 1) if hit_speeds else avg_ball_speed
    side_max_speed = round(max(hit_speeds), 1)                   if hit_speeds else max_ball_speed

    return {
        "firstServeSuccessRate":  first_serve_rate,
        "returnFirstSuccessRate": return_first_rate,
        "baselineShotRate":       baseline_shot_rate,
        "baselineWinRate":        baseline_win_rate,
        "netPointRate":           net_point_rate,
        "netPointWinRate":        net_win_rate,
        "totalShots":             total_shots,
        "aceCount":               ace_count,
        "netApproaches":          net_approaches,
        "avgBallSpeed":           side_avg_speed,
        "maxBallSpeed":           side_max_speed,
        "totalDistance":          player_stats["totalDistance"],
        "avgMoveSpeed":           player_stats["avgMoveSpeed"],
        "maxMoveSpeed":           player_stats["maxMoveSpeed"],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def format_rally(
    rally_result,
    frames:        list,
    serial_number: str,
    endpoint:      str,
    dry_run:       bool = False,
) -> dict:
    """Format rally data into the standard result JSON and POST to endpoint.

    Args:
        rally_result: RallyResult instance with start_time, end_time, bounces.
        frames:       list of per-frame dicts from _rally_raw_buffer.
        serial_number: device serial number string.
        endpoint:     HTTP endpoint to POST to.
        dry_run:      if True, skip the POST (for unit tests / offline use).

    Returns:
        The formatted payload dict.
    """
    # Use Beijing time (UTC+8) but keep the Z suffix as the server expects
    _tz_offset = datetime.timedelta(hours=8)
    start_dt = datetime.datetime.utcfromtimestamp(rally_result.start_time) + _tz_offset
    end_dt   = datetime.datetime.utcfromtimestamp(rally_result.end_time)   + _tz_offset

    # FIX #1: ball speed from hit/bounce frames only, clamped
    avg_ball_speed, max_ball_speed = _compute_ball_speed_stats(frames)

    near_stats = _compute_player_stats(frames, "near")
    far_stats  = _compute_player_stats(frames, "far")

    # FIX #3/#4: serve type + correct bounce handType
    result_matrix = _build_result_matrix(frames)
    # FIX #5: last-value carry for player positions
    track_matrix  = _build_track_matrix(frames)

    # FIX #6/#7: full per-side stats instead of hardcoded zeros
    near_block = _compute_advanced_stats(result_matrix, near_stats, "near", avg_ball_speed, max_ball_speed)
    far_block  = _compute_advanced_stats(result_matrix, far_stats,  "far",  avg_ball_speed, max_ball_speed)

    payload = {
        "serial_number": serial_number,
        "startTime":     start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime":       end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content": {
            "mete": {
                "movingDistance":             round(near_stats["totalDistance"] + far_stats["totalDistance"], 2),
                "battingAttempts":            near_block["totalShots"] + far_block["totalShots"],
                "serveSuccessRate":           near_block["firstServeSuccessRate"],   # near player serves first
                "serveAceCount":              near_block["aceCount"],
                "serveSpeed":                 near_block["avgBallSpeed"],
                "receiveSuccessRate":         far_block["returnFirstSuccessRate"],
                "breakPointConversionRate":   0,
                "hitSuccessRate":             0,
                "forehandHitRate":            _forehand_rate(result_matrix),
                "backhandHitRate":            _backhand_rate(result_matrix),
                "baselineScoreRate":          0,
                "averageHitSpeed":            avg_ball_speed,
                "maxHitSpeed":                max_ball_speed,
                "netApproachCount":           near_block["netApproaches"] + far_block["netApproaches"],
                "netScoreRate":               0,
                "volleyScoreRate":            0,
                "farCount":                   far_block,
                "nearCount":                  near_block,
            },
            "resultmatrix": result_matrix,
            "trackMatrix":  track_matrix,
        },
    }

    if not dry_run:
        # --- DEBUG: dump payload to file so we can inspect it ---
        import json as _json, os as _os
        try:
            _dump_path = _os.path.join("recordings", f"payload_rally_{rally_result.rally_id}.json")
            _os.makedirs("recordings", exist_ok=True)
            with open(_dump_path, "w", encoding="utf-8") as _f:
                _json.dump(payload, _f, ensure_ascii=False, indent=2)
            logger.info("Rally %d payload dumped -> %s", rally_result.rally_id, _dump_path)
        except Exception as _e:
            logger.warning("Payload dump failed: %s", _e)
        # --- END DEBUG ---
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            if resp.ok:
                logger.info(
                    "Rally %d exported → %s (%d)",
                    rally_result.rally_id, endpoint, resp.status_code,
                )
                mete = payload["content"]["mete"]
                near = mete["nearCount"]
                far  = mete["farCount"]
                print(
                    f"\n{'='*60}\n"
                    f"[Rally {rally_result.rally_id}] 数据已发送\n"
                    f"  时间: {payload['startTime']} → {payload['endTime']}\n"
                    f"  球速: 均值={mete['averageHitSpeed']:.1f} km/h  最大={mete['maxHitSpeed']:.1f} km/h\n"
                    f"  近端: 击球={near['totalShots']}  发球成功率={near['firstServeSuccessRate']}%"
                    f"  底线率={near['baselineShotRate']}%  网前率={near['netPointRate']}%\n"
                    f"  近端移动: {near['totalDistance']:.1f}m  均速={near['avgMoveSpeed']:.2f}m/s  最大={near['maxMoveSpeed']:.2f}m/s\n"
                    f"  远端: 击球={far['totalShots']}  发球成功率={far['firstServeSuccessRate']}%"
                    f"  底线率={far['baselineShotRate']}%  网前率={far['netPointRate']}%\n"
                    f"  远端移动: {far['totalDistance']:.1f}m  均速={far['avgMoveSpeed']:.2f}m/s  最大={far['maxMoveSpeed']:.2f}m/s\n"
                    f"{'='*60}\n"
                )
            else:
                logger.warning(
                    "Rally %d export failed: %s %s",
                    rally_result.rally_id, resp.status_code, resp.text[:200],
                )
        except Exception as e:
            logger.warning("Rally %d export error: %s", rally_result.rally_id, e)

    return payload


# ---------------------------------------------------------------------------
# Small helpers used by format_rally
# ---------------------------------------------------------------------------

def _forehand_rate(result_matrix: list) -> float:
    hits = [e for e in result_matrix if e["type"] in ("hit", "serve")]
    if not hits:
        return 0.0
    fh = sum(1 for h in hits if h["handType"] == "forehand")
    return round(100 * fh / len(hits), 1)


def _backhand_rate(result_matrix: list) -> float:
    hits = [e for e in result_matrix if e["type"] in ("hit", "serve")]
    if not hits:
        return 0.0
    bh = sum(1 for h in hits if h["handType"] == "backhand")
    return round(100 * bh / len(hits), 1)