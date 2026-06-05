# YOLO match-ball false positives on cam68 clip11

## Symptom

GT comparison for cam68 clip11 showed that dashboard YOLO replay could match
all GT match-ball bounces after tuning the HIT angle threshold, but still
reported many extra bounces.

GT source:

```text
D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min
```

GT match-ball bounce frames:

```text
554, 604, 635, 878, 932, 980, 1017, 1080, 1129, 1195,
1844, 2014, 2047, 2697, 2738, 2889
```

## Findings

Baseline with GT match-ball trajectory only:

```text
offline_from_gt_match_ball:
matches: 12
misses: 4
false positives: 5
```

This baseline has no player poses, so several GT HIT turns are naturally
misclassified as BOUNCE. It is useful for isolating event-shape behavior, not
for final deployment quality.

Dashboard YOLO replay before angle tuning:

```text
hit_angle_thresh effectively used by realtime: 45 deg
dashboard matches: 15
misses: 1
false positives: 20
missed GT bounce: 635
```

Root cause for missed 635:

```text
frame 636 was classified as bottom_reversal_player_anchor HIT
player_distance_px: 100.68
player_threshold_px: 250.0
angle: 86.42
```

GT says frame 635 is a match-ball bounce. YOLO detection existed around
620-650, so this was not a detector miss. It was a false HIT caused by an overly
wide bottom-half HIT angle rule.

Dashboard YOLO replay after `hit_angle_thresh=110` and passing config into the
dashboard event call:

```text
dashboard matches: 16
misses: 0
false positives: 21
```

The remaining false positives are mostly not match-ball events. Checking the GT
JSON at these frames shows `is_match_ball=false` for most of them:

```text
21, 527, 696, 1238, 1271, 1295, 1329, 1508, 1541, 1564,
1370, 1860, 1960, 1978, 2056, 2221, 2248, 2278, 2677
```

Examples:

```text
frame 21:  is_match_ball=false
frame 527: is_match_ball=false
frame 696: is_match_ball=false
frame 1564: ball_event=bounce; is_match_ball=false
frame 2248: ball_event=bounce; is_match_ball=false
```

## Root Causes

1. Dashboard realtime did not pass `config.hit_bounce_refiner` parameters into
   `detect_single_camera_events()`. It silently used default `hit_angle_thresh=45`.
2. The bottom-half direct HIT angle threshold was too permissive for this clip.
   A bounce at frame 635/636 was suppressed as a false HIT.
3. The remaining false positives are primarily match-ball selection errors:
   YOLO detects a real moving ball, but it is not the current rally/match ball.

## Fix Applied

- `app/orchestrator.py`
  - Passes `hit_angle_thresh`, bottom HIT pixel thresholds, lookback frames,
    HIT suppression window, and bounce clean parameters from
    `config.hit_bounce_refiner` into `detect_single_camera_events()`.
- `app/config.py` and `config.yaml`
  - Changed default/config `hit_angle_thresh` from `45.0` to `110.0`.
- `tools/compare_cam68_clip11_gt.py`
  - Added GT parser and frame-tolerance comparison report.
  - Uses the same `hit_bounce_refiner` config for offline comparison.

## Remaining Work

The extra bounces should not be tuned away only with bounce thresholds. The
next fix should add a match-ball selector/trajectory lock for YOLO:

- Prefer a continuous rally trajectory over isolated moving balls.
- Reject candidate trajectories outside the current rally window.
- Use player HIT/net-crossing context to start or continue the active match-ball
  track.
- Consider reusing the existing `ball_selector`/candidate-reselection work for
  YOLO candidates, or add a lighter YOLO match-ball continuity scorer.
