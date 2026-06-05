# YOLO HIT-window shadow bounce resurrection

## Status

Algorithm change in working tree, pending replay/GT validation.

## Summary

cam68 clip11 realtime YOLO replay diverged from the offline dashboard-publish
filter after switching duplicate handling to keep the strongest bounce signal.
The stream published an extra OUT bounce at frame `903`.

The nearby offline candidate at frame `901` was correctly suppressed by a
top-half HIT lookback event around frame `897`, but the realtime sliding-window
analysis later produced a jittered candidate at frame `903`. Because that
jittered frame was just outside the configured HIT suppression window, it could
resurrect as a new bounce even though it represented the same physical event.

## Root Cause

The realtime path persisted HIT suppression frames, but it did not persist the
spatial location of bounce candidates already suppressed by HIT. A later
sliding-window result could therefore shift the same candidate by a few frames
and escape the pure frame-window check.

## Algorithm Change

After direct HIT-window suppression, remember the suppressed bounce candidate
per camera. Before publishing a later YOLO bounce, check whether it falls in the
same time/space duplicate window as any HIT-suppressed candidate:

```text
same bounce shadow = abs(frame_a - frame_b) <= clean_time_frames
                  and distance_m(a, b) <= max(clean_space_meters, 2.5m)
```

If true, suppress the later candidate as `hit_window_shadow` in realtime stats
and as `publish_suppression_reason=hit_window` in the offline dashboard publish
filter. This keeps the event order unchanged:

```text
raw bounce candidates -> HIT recognition -> HIT suppression
-> HIT-suppressed duplicate shadow suppression -> strongest bounce cleanup
-> publish final bounce
```

## Files Changed

- `app/orchestrator.py`
  - Added per-camera `_yolo_fuzzy_hit_suppressed_bounces`.
  - Added `_remember_yolo_hit_suppressed_bounce_locked()`.
  - Added `_yolo_hit_suppressed_duplicate_frame_locked()`.
  - Suppresses later realtime candidates that shadow a HIT-suppressed bounce.
- `app/pipeline/yolo_bounce_filter.py`
  - Applies the same HIT-suppressed shadow rule in
    `filter_dashboard_yolo_publishable_bounces()`.
- `tests/test_yolo_bounce_filter.py`
  - Adds a minimal `901 -> 903` style regression test.

## Rollback

To revert only this algorithm attempt:

1. Remove `_yolo_fuzzy_hit_suppressed_bounces` and its reset block from
   `app/orchestrator.py`.
2. Remove `_remember_yolo_hit_suppressed_bounce_locked()` and
   `_yolo_hit_suppressed_duplicate_frame_locked()`.
3. Remove the two realtime calls that record/check HIT-shadow bounces before
   `release_delay`.
4. Remove `hit_suppressed_candidates` handling from
   `filter_dashboard_yolo_publishable_bounces()`.
5. Remove
   `test_dashboard_yolo_publish_filter_suppresses_hit_window_shadow_candidate()`.

## Validation Plan

- Run targeted syntax and pytest checks.
- Replay cam68 clip11 for 3000 frames.
- Compare:
  - realtime dashboard bounces
  - offline dashboard-publishable bounces
  - GT match-ball bounce frames
