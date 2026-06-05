# Dashboard Preview Frame Starvation

## Status

Fixed in working tree, requires dashboard process restart to take effect.

## Summary

After the dashboard runs for a long time with fake RTSP streams, the main monitoring image at `http://localhost:8000/` can appear frozen or drop to a very low visual frame rate.

This was not caused by the fake stream publisher in the observed run. The fake stream ffmpeg publishers kept running at 25 FPS / 1x speed for more than 7 hours. The dashboard backend pipeline also still reported roughly 21-22 FPS. The visible freeze came from the dashboard preview output path being starved by heavier detection/event work.

## Symptoms

- Chrome page `http://localhost:8000/` main playback/monitoring area appears frozen.
- `/api/status` still reports the cam68 pipeline as running.
- cam68 pipeline FPS was still around 21-22 FPS during the freeze.
- MJPEG preview output returned repeated JPEG frames.

Observed MJPEG sample:

```text
/api/camera/cam68/stream?delay=0
frames_parsed = 45
unique = 1
fps = 5.62
unique_fps = 0.12
```

This means the stream endpoint was yielding frames, but they were mostly the same cached preview image.

## Fake Stream Evidence

Fake stream sender:

```text
tools/start_fake_streams.bat
ffmpeg -re -stream_loop -1 ... -r 25 -f rtsp rtsp://localhost:8554/cam66
ffmpeg -re -stream_loop -1 ... -r 25 -f rtsp rtsp://localhost:8554/cam68
```

Observed ffmpeg log tail after long runtime:

```text
fps=25
speed=1x
elapsed=7:09:xx
```

MediaMTX logs showed the publishers stayed up. The visible freeze therefore points more strongly to the dashboard consumer/preview path than to the fake RTSP sender.

## Root Cause

`Orchestrator._consume_loop()` consumed each camera result queue to exhaustion before reading the preview frame queue.

With the newer YOLO realtime event path, each cam68 detection can trigger heavier work:

- candidate queue event detection
- HIT/BOUNCE refinement
- persistent HIT suppression
- live event publication checks

When the result queue stayed busy, the consumer loop spent too long processing detection/event messages before it reached `handle.frame_queue`. The frame queue intentionally keeps only the freshest preview JPEG, but if the consumer does not drain it often enough, `_latest_frames[name]` stops updating frequently. The MJPEG endpoint then repeatedly serves the same cached JPEG, so the browser looks frozen.

Related long-run signal:

```text
single_cam_bounce_stats.cam68.pending_release_bounces increased past 1000
single_cam_bounce_stats.cam68.skipped_duplicate_live_bounces increased past 12000
some event detect_delay values reached 30-40 seconds
```

## Fix

`Orchestrator._consume_loop()` now prioritizes preview frames:

- Drain `handle.frame_queue` before result/event processing.
- Keep only the freshest preview JPEG, as before.
- Limit result queue draining to `_CONSUMER_MAX_RESULTS_PER_HANDLE_TICK = 8` per handle per loop.

The preview path was then hardened further:

- Replace `Queue.empty()` polling with `get_nowait()` until `queue.Empty`.
- Add preview frame metadata in the orchestrator: `seq`, `frame_id`, `capture_ts`, `updated_ts`, `age_ms`.
- Make the MJPEG endpoint wait for a new preview `seq` before yielding another frame.
- Keep a bounded delay buffer for `?delay=...` streams.
- Send only an occasional heartbeat frame when no new preview arrives, instead of continuously repeating the same cached JPEG.

This combines both sides of the fix: expensive event processing no longer starves preview consumption, and the HTTP endpoint no longer fabricates apparent FPS by repeatedly yielding an old cached frame.

## Files Changed

- `app/orchestrator.py`
  - Added `_CONSUMER_MAX_RESULTS_PER_HANDLE_TICK`.
  - Moved preview frame draining ahead of detection result processing.
  - Limited per-loop detection queue drain.
  - Added preview frame sequence/metadata and wait helpers.
- `app/api/routes.py`
  - Changed `/api/camera/{name}/stream` to yield new preview sequences instead of repeating cached frames every sleep tick.
- `app/schemas.py`
  - Added preview frame status fields to `PipelineStatus`.
- `app/pipeline/camera_pipeline.py`
  - Replaced `frame_queue.empty()` queue draining with `get_nowait()`/`queue.Empty`.
- `app/pipeline/video_pipeline.py`
  - Applied the same queue-drain hardening for video-test preview frames.

## Verification

Code checks:

```text
python -m py_compile app\orchestrator.py
pytest tests\test_realtime_logic.py -q
15 passed
pytest tests\test_realtime_logic.py tests\test_yolo_bounce_filter.py tests\test_player_detector.py tests\test_realtime_hit_bounce_refiner.py -q
27 passed
```

Runtime verification still requires restarting the dashboard process so the new consumer loop is active, then re-testing:

```text
python -m tools.replay_yolo_dashboard_events ...
```

or, for the live fake-stream page:

```text
http://localhost:8000/api/camera/cam68/stream?delay=0
```

Expected after restart:

- `/api/status` should show `latest_preview_seq` and `latest_preview_frame_id` increasing.
- MJPEG unique frame rate should match real preview producer updates rather than repeated cached frames.
- Main dashboard preview should continue updating while YOLO events run.
- `/api/status` pipeline FPS and browser-visible preview should no longer diverge badly.

Observed after restarting `python .\main.py` and starting cam68:

```text
cam68 state: running
fps: 24.3 -> 24.5
latest_preview_seq: 321 -> 370 in 4s
latest_preview_frame_id: 710 -> 812 in 4s
latest_preview_age_ms: 33.6 -> 15.9
MJPEG sample: frames=36, unique=36, fps=7.17, unique_fps=7.17
Browser page: cam68 image loaded at 960x540
```

## Remaining Validation

- Run the fake streams for a longer period after dashboard restart.
- Sample MJPEG output every 10-20 minutes and compare `unique_fps`.
- If unique FPS is still low while pipeline FPS is healthy, next suspect is the JPEG worker or `frame_queue` producer in `app/pipeline/camera_pipeline.py`.
- If both pipeline FPS and unique FPS drop, next suspect is RTSP decode or YOLO/player inference load.
