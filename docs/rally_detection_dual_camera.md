# Rally Detection with Dual-Camera Bounce Detection

## Overview

`tools/eval_rally_gt.py` — GT-based rally analysis evaluation pipeline, using cam66 GT trajectory + cam68 motion+YOLO detections for bounce/serve/shot/rally detection.

### Results (cam66, frames 0-900, 259 GT frames)

| Event   | GT | Detected | Recall | FP | Method |
|---------|---:|--------:| ------:|---:|--------|
| Bounce  |  6 |       8 |   83%  |  3 | dual-camera cross-distance |
| Serve   |  3 |       3 |  100%  |  0 | gap + baseline impact scan |
| Shot    |  7 |       5 |   57%  |  1 | world_y reversal + NMS |

## Architecture

```
cam66 GT pixel positions
    |
    v
homography (pixel_to_world)
    |
    v
cam66 world trajectory  ─────────────┐
    |                                 |
    v                                 v
cam68 motion+YOLO detections ──> cross-camera world distance
    |                                 |
    v                                 v
bounce detection (dist minima)   serve detection (gap + baseline)
    |                                 |
    v                                 v
shot detection (wy reversal + NMS, exclude bounce/serve zones)
    |
    v
rally state machine (IDLE → PRE_SERVE → SERVING → RALLY → IDLE)
    |
    v
evaluation vs GT labels (frame_tolerance=±7)
```

## Key Algorithms

### 1. Dual-Camera Bounce Detection (`detect_bounces_dual_camera`)

**Core insight**: At Z≈0 (ball on ground/bounce), both cameras' ground-plane homographies project accurately to the same world point. When airborne, parallax causes projections to diverge.

```
At bounce (Z≈0):  cam66_world ≈ cam68_world  →  distance ~1-3m
Airborne (Z>0):   cam66_world ≠ cam68_world  →  distance ~5-50m
```

**Algorithm**:
1. For each overlapping frame (cam66 GT + cam68 detection exist):
   - Compute world coordinate distance: `hypot(wx66-wx68, wy66-wy68)`
2. Build contiguous segments (gap tolerance ≤3 frames)
3. Find local minima via `find_peaks(-dists, prominence=1.0, distance=5)`
4. Filter: `dist < 4.0m` + cooldown=8 frames

**Parameters**:
- `dist_threshold=4.0` — max cross-camera distance to qualify as bounce
- `prominence=1.0` — minimum peak prominence in the inverted distance signal
- `cooldown=8` — minimum frames between detections

**Why this works better than single-camera**:
- Single-cam pixel_y extrema: only detects near-end bounces (17% recall)
- Single-cam world_y extrema: misses same-direction bounces (17% recall)
- Dual-cam cross-distance: detects bounces regardless of ball direction (83% recall)

**Limitation**: Requires both cameras to detect the ball at the same frame. Missed f851 (too sparse for both cameras).

### 2. Serve Detection (`detect_serves`)

**Problem**: Simple gap-based detection finds the first frame after a gap (ball toss start), not the actual serve impact point.

**Solution**: After finding a gap, scan forward up to 40 frames to find the **last frame in the baseline zone** before the ball leaves toward the net.

```
Gap detected at f118 (first frame after 77-frame gap)
  f118: world_y=2.1  ← baseline (toss start)
  f130: world_y=6.0  ← ball going up (toss)
  f140: world_y=4.7  ← coming back down
  f144: world_y=2.7  ← baseline again (impact!)  ← SERVE DETECTED HERE
  f145: world_y=3.9  ← ball leaving baseline
  f150: world_y=9.6  ← heading to net
```

**Parameters**:
- `gap_threshold=15` — minimum gap between trajectory segments
- Forward scan window: 40 frames max
- Baseline zone: `world_y < 5.0` (near) or `world_y > 18.77` (far)
- Exit condition: `world_y > baseline + 2.0m`

### 3. Shot Detection (`detect_shots`)

**Method**: Detect world_y velocity direction reversals with NMS (Non-Maximum Suppression).

**Algorithm**:
1. For each frame with valid neighbors (gap ≤5):
   - Compute `vy_before` and `vy_after` (world_y velocity)
   - If `vy_before * vy_after < 0` (direction reversal):
     - `wy_change = |vy_after - vy_before|`
     - Add to candidates if `wy_change > 0.4`
2. Exclude frames near detected bounces (±8 frames) and serves (±15 frames)
3. NMS: sort by wy_change descending, suppress within cooldown=15 frames

**Why world_y instead of pixel velocity**:
- Pixel velocity depends on distance to camera (far-end shots appear slow in pixels)
- World velocity is camera-independent
- Example: f190 shot has pixel_vel=5.1 px/frame but world wy_change=0.95 m/frame

**Limitations**:
- Cannot detect shots where ball continues same world_y direction (f801, f839)
- These are volleys/returns where the ball doesn't reverse — requires audio or player tracking

### 4. Rally State Machine (`detect_rallies`)

```
States: IDLE → PRE_SERVE → SERVING → RALLY → IDLE

IDLE:
  - Detect gap → check for nearby serve (within 40 frames)
    - Yes: → PRE_SERVE (toss phase)
    - No:  → RALLY (direct start)
  - Detect serve frame directly: → SERVING

PRE_SERVE:
  - Serve frame reached: → SERVING (update rally_start to impact)
  - Net crossing: → RALLY

SERVING:
  - Net crossing: → RALLY

RALLY:
  - Bounce outside court: → IDLE (end_reason="out")
  - Double bounce same side: → IDLE (end_reason="double_bounce")
  - Timeout (75 frames / 3s): → IDLE (end_reason="timeout")
  - Net crossing: stroke_count++

Key fix: Net crossing check skipped when frame gap > 5 to prevent false crossings
across trajectory gaps (e.g., f41→f118 would appear as net crossing).
```

## 2D Mini-Map Rendering

`draw_minimap()` renders a top-down court view overlay:

**Court layout** (world coords → map pixels):
- Green court lines (outer boundary, service lines, center line, doubles sidelines)
- White net line at `world_y = 11.885`
- Extended range: x=[-1, 9.23], y=[-2, 26] to show out-of-court balls

**Dynamic elements**:
- Green circle: cam66 ball position (GT)
- Orange circle: cam68 ball position (motion+YOLO detection)
- Gray line connecting cam66↔cam68 world positions (visual parallax)
- Cyan ▼: bounce markers (persist 30 frames, fade out)
- Yellow ★: serve markers (persist 20 frames)
- Green ◆: shot markers (persist 20 frames)
- Trail: last 30 frames, color fading

**Cross-camera distance bar** (below minimap):
- Green (<4m): ball near ground, potential bounce
- Yellow (4-8m): intermediate height
- Red (>8m): ball high in air
- White vertical line: 4m threshold marker

## Usage

```bash
# Evaluate only (default: dual-camera, frame_tolerance=5)
python -m tools.eval_rally_gt

# With video rendering
python -m tools.eval_rally_gt --render --frame-tolerance 7

# Single camera only (no cam68)
python -m tools.eval_rally_gt --single-cam

# Custom parameters
python -m tools.eval_rally_gt --cam-dist-threshold 5.0 --bounce-cooldown 10
```

## Output

```
exports/eval_rally_gt/
  rally_eval_results.json   # evaluation metrics + rally details
  rally_gt_eval.mp4         # rendered video (if --render)
```

## Dependencies

| Component | Source | Usage |
|-----------|--------|-------|
| GT loading | `tools/eval_rally_gt.py:load_match_ball_gt()` | LabelMe JSON → positions + event labels |
| cam68 detections | `exports/eval_motion_yolo/cam68_detections.json` | Motion+YOLO pipeline output |
| Homography | `app/pipeline/homography.py:HomographyTransformer` | `pixel_to_world()` for both cameras |
| Court constants | `app/analytics.py` | NET_Y, COURT_X, COURT_Y, baselines |

## Known Limitations

1. **Bounce f851 missed**: Only 2 GT frames (f848, f851) with no cam68 detection nearby → insufficient data for cross-distance signal
2. **Shots f801, f839 missed**: Ball continues same direction after shot (no world_y reversal) → requires audio/player tracking
3. **Shot f190 classified as bounce**: Cross-camera distance is low (2.91m) because ball is near ground level during the shot → bounce/shot disambiguation needs additional features
4. **Homography Z=0 assumption**: World coordinates from `pixel_to_world()` are only accurate at ground level; airborne positions are systematically biased → this is exactly what makes the cross-camera distance signal work for bounce detection
