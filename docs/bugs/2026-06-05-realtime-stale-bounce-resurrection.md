# Realtime YOLO Stale Bounce Resurrection

## Status

Fixed in working tree, pending longer offline/realtime parity validation.

## Summary

Dashboard realtime YOLO event replay could publish old BOUNCE candidates again after the live sliding buffer moved forward. These BOUNCE events were originally close to HIT candidates and should have been suppressed by the HIT +/- 3 frame rule.

This was not primarily a threshold issue. It was an event-state lifetime issue in the realtime path.

## Symptoms

- In cam68 clip11 1000-frame realtime replay, extra BOUNCE events appeared near HIT frames.
- Before the fix, realtime replay produced 15 BOUNCE events, including stale early frames:
  - 119
  - 128
  - 132
- These frames were near HIT/HIT-candidate suppression windows and should not have reached `_live_bounces` or the 3D push queue.

## Root Cause

The realtime YOLO path recomputes events from a bounded sliding buffer. HIT suppression worked only inside the current `detect_single_camera_events()` result.

When the buffer moved forward:

1. Early HIT candidate frames fell out of the current buffer.
2. Old raw BOUNCE candidates could be recomputed as final BOUNCE candidates.
3. The realtime publisher no longer knew those BOUNCE candidates had previously been suppressed by HIT logic.
4. The stale BOUNCE events were published again.

Offline full-window inference did not have the same failure mode because the suppression frames stayed available for the whole clip.

## Fix

The event refiner now exposes HIT suppression frames explicitly, and the realtime publisher persists them across sliding-window updates.

Key behavior:

- Raw BOUNCE candidates are kept first.
- HIT and HIT-candidate frames are identified before final BOUNCE publishing.
- `hit_suppression_frames` are returned from `detect_single_camera_events()`.
- `Orchestrator` stores per-camera persistent HIT suppression frames.
- Before publishing a realtime BOUNCE, the publisher checks whether it is within `hit_suppression_frames +/- 3`.
- Only after HIT suppression can a BOUNCE pass through the final `25 frames + 1.5m` strongest-signal cleanup.

## Files Changed

- `app/pipeline/yolo_bounce_filter.py`
  - Returns `hit_suppression_frames`.
  - Keeps HIT-first, BOUNCE-after-cleaning event order.
  - Uses strongest-signal BOUNCE selection inside duplicate time/space windows.
- `app/orchestrator.py`
  - Adds persistent per-camera `_yolo_fuzzy_hit_suppression_frames`.
  - Applies persistent HIT-window suppression before publishing realtime YOLO BOUNCE events.
- `tests/test_yolo_bounce_filter.py`
  - Verifies suppression frames are exposed.
- `tests/test_realtime_logic.py`
  - Verifies stale BOUNCE candidates are blocked after earlier HIT suppression frames leave the current buffer.

## Verification

Focused tests:

```text
pytest tests\test_yolo_bounce_filter.py tests\test_realtime_logic.py -q
19 passed
```

Realtime cam68 clip11 replay:

```text
python -m tools.replay_yolo_dashboard_events ^
  --frames-dir D:\tennis-dataset\1001\clip11\cam68_20260404_075325_2min ^
  --camera cam68 ^
  --max-frames 1000 ^
  --device cpu ^
  --enable-ws-queue ^
  --out reports\dashboard_yolo_replay_compare\cam68_1000_realtime_after_persistent_hit_suppression.json
```

Before fix:

```text
total_bounces = 15
stale bounce frames included: 119, 128, 132
```

After fix:

```text
total_bounces = 9
stale bounce frames removed: 119, 128, 132
skipped_persistent_hit_suppressed_bounces = 197
ws_pending_bounces = 9
```

## Remaining Validation

The stale-HIT-window bug is fixed, but full parity still needs a longer current-offline-vs-realtime comparison.

Remaining frames such as `384`, `527`, `553`, and `604` should be compared against the current offline event refiner output to determine whether they are valid strongest-signal final BOUNCE events or another realtime/offline mismatch.

## Acceptance Criteria

- A BOUNCE within HIT/HIT-candidate +/- 3 frames must never be published, even after the sliding buffer moves forward.
- `_live_bounces`, dashboard minimap, report data, and `_ws_bounce_queue` must all use the same final BOUNCE list.
- 3D push must receive final BOUNCE only.
- Realtime replay should match current offline refiner output within the configured release-delay window.
