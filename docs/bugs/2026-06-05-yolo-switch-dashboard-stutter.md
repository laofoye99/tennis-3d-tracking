# YOLO Switch Dashboard Stutter

## Status

Fixed in working tree and verified after restarting the dashboard.

## Summary

Switching the live model from TrackNet to YOLO Roadmap could make the dashboard playback feel stuck or drop frames. The preview starvation fix kept `_latest_frames` healthier, but YOLO mode added enough extra load that the browser and backend could still lag.

## Symptoms

- TrackNet mode preview stayed near 25 FPS.
- After switching to YOLO Roadmap, cam68 previously dropped to roughly 20-21 FPS in observed runs.
- `single_cam_bounce_stats.cam68` showed growing delayed event counters.
- Some YOLO event `detect_delay` values grew to several seconds.

## Root Causes

1. `PlayerPoseDetector.detect()` returned cached player detections on skipped frames, and `camera_pipeline.py` still published those cached detections to `result_queue` every frame.
   - This created repeated `player_pose` queue messages even when no new player inference ran.
2. YOLO single-camera event analysis re-ran the full recent detection window on every frame.
   - The window can be 240 detections plus player poses.
   - This offline-style computation is valid, but too expensive to run every realtime frame.
3. `/api/status` returned up to 500 bounces and 500 hits, and the dashboard redrew the minimap every 500 ms.
   - YOLO mode creates more hit/bounce events than TrackNet, so the payload and draw cost grew.
4. During manual testing, old dashboard multiprocessing children from killed parent processes were still alive and could add extra CPU/GPU load.

## Fix

- `app/pipeline/player_detector.py`
  - Added `last_inference_ran`.
  - Skipped calls still return cached results, but now expose that no new inference happened.
- `app/pipeline/camera_pipeline.py`
  - Publishes `player_pose` messages only when player inference actually ran.
- `app/config.py`
  - YOLO player detector default changed to every 2 frames instead of every frame.
- `app/orchestrator.py`
  - Added `_YOLO_FUZZY_ANALYSIS_STRIDE = 3`.
  - Still buffers every YOLO detection, but only runs the expensive event analysis every 3 frames.
  - Filters player pose messages to the relevant buffered frame range.
  - Adds analysis telemetry: `analysis_stride`, `analysis_calls`, `skipped_analysis_stride`, `last_analysis_ms`.
  - Limits `/api/status.analytics.recent_bounces` and `recent_hits` to the latest 120 events while preserving total counts.
  - Resets live analytics and stale per-camera queues during model switch.
- `app/api/templates/dashboard.html`
  - Reduced dashboard minimap event draw cap from 500 to 120.

## Verification

Focused tests:

```text
python -m py_compile app\orchestrator.py app\pipeline\camera_pipeline.py app\pipeline\player_detector.py app\config.py app\api\routes.py app\schemas.py
pytest tests\test_realtime_logic.py tests\test_player_detector.py -q
19 passed
pytest tests\test_realtime_logic.py tests\test_yolo_bounce_filter.py tests\test_player_detector.py tests\test_realtime_hit_bounce_refiner.py -q
29 passed
```

Runtime replay with fake RTSP and cam68:

```text
TrackNet baseline:
fps: 24.36 -> 24.61
preview seq delta: 49 in 4s
frame_id delta: 100 in 4s

After switching to YOLO Roadmap:
fps: 25.14 -> 25.12
preview seq delta: 37 in 4s
frame_id delta: 102 in 4s
latest_preview_age_ms: 0.0-2.0
analysis_stride: 3
analysis_calls: 259
skipped_analysis_stride: 449
last_analysis_ms: 89.09
player_pose_buffered: 213
recent_event_limit: 120
```

MJPEG sample after YOLO switch:

```text
frames=37
unique=37
fps=7.08
unique_fps=7.08
```

Browser check:

```text
Dashboard model label: YOLO
cam68 image loaded: 960x540
YOLO static status visible
```

## Follow-up

- The first fix reduced complete freezes, but user-visible playback could still drop badly because YOLO event analysis held the Python GIL and delayed MJPEG serving.
- The second fix split preview consumption into a dedicated preview thread and reduced realtime YOLO analysis pressure:
  - `frame_queue -> _latest_frames` now runs outside the heavy result/event consumer.
  - YOLO event analysis runs every 10 frames instead of every 3 frames.
  - YOLO event buffer is 180 frames instead of 240 frames.
  - YOLO preview encoding uses `preview_stride=1`.
  - Dashboard live video uses `delay=0` for the main playback image.

Second-stage verification:

```text
YOLO running:
pipeline fps: 25.14 -> 25.40
preview seq fps: 21.83
preview frame fps: 25.33
latest_preview_age_ms: 2.5
analysis_stride: 10
buffered: 180
last_analysis_ms: 123.67

MJPEG:
delay=0 frames=87 unique=87 fps=14.49 unique_fps=14.49
delay=1 frames=69 unique=69 fps=11.43 unique_fps=11.43

Browser:
img src=/api/camera/cam68/stream?delay=0
cam68 image loaded: 960x540
model label: YOLO
```

- If long-run YOLO still accumulates event delay, move the YOLO event refiner to a separate worker/process.
- Expose queue depth and event-analysis age in `/api/status` for faster diagnosis.
- Add a safe dashboard restart script that terminates child pipeline processes before killing the parent.

## Third-stage fix: status/event-loop and preview encode pressure

After the preview thread fix, YOLO inference stayed alive but the browser page
could still feel choppy. The important finding was that `/api/status` and the
dashboard polling path were still too heavy during YOLO mode:

- Full status could take about 2 seconds while the analytics lock was busy.
- The async status route did synchronous work on the FastAPI event loop.
- The dashboard polled every 500 ms, so slow status calls could pile up.
- Preview frame production was close to 25 FPS, but JPEG preview encoding and
  queue handling still dropped a few frames.

Additional fixes:

- Added `/api/dashboard/status` as a compact live-dashboard payload.
- Changed `/api/status`, `/api/dashboard/status`, and `/api/analytics/live` to
  sync route handlers so FastAPI runs them in the threadpool instead of blocking
  the event loop.
- Added non-blocking compact analytics: if the analytics lock is busy, dashboard
  status returns the latest cached compact payload instead of waiting.
- Reduced compact dashboard events to 24 bounce/hit events and 12 speed events.
- Added dashboard poll in-flight protection and changed polling from 500 ms to
  1000 ms.
- Optimized preview JPEG generation:
  - preview width: 720 px
  - JPEG quality: 60
  - JPEG work queue keeps the newest frame instead of preserving stale queued
    frames
  - removed one extra full-frame copy before JPEG encoding
- Switched MJPEG endpoint to a sync streaming generator so streaming work is
  isolated from the main event loop.

Third-stage verification:

```text
Focused tests:
python -m py_compile app\api\routes.py app\orchestrator.py app\pipeline\camera_pipeline.py
pytest tests\test_realtime_logic.py tests\test_player_detector.py -q
19 passed

YOLO dashboard status:
payload_len: ~6.5 KB after compacting early run, ~13-18 KB with more events
status latency after startup: ~17-50 ms
pipeline fps: ~24.5-25.5
preview seq fps: ~24.8-25.1
latest_preview_age_ms: commonly 1-50 ms

MJPEG stream:
12 s backend preview production: 300 seq, about 24.8 FPS
12 s curl receive: 293 unique JPEGs, about 24.2 FPS
720 px preview stream-only sample: 294 unique JPEGs / 12 s, about 24.5 FPS
```

Current conclusion:

- The original YOLO switch freeze/stutter root cause was not YOLO inference
  alone; it was YOLO analytics/status work competing with MJPEG and preview
  encoding.
- The page no longer waits on the heavy full status payload.
- The preview path now produces close to the source stream rate. The remaining
  gap is small and mostly caused by MJPEG transport/client timing and multiple
  open clients. For a true 30 FPS display path, replace MJPEG with a WebSocket
  binary frame stream or WebRTC.

## Fourth-stage fix: move YOLO event analysis out of the dashboard process

User-visible stutter still reproduced after switching from TrackNet to YOLO:

- The YOLO camera process stayed near 25 FPS.
- Preview seq production stayed close to source rate.
- But dashboard status and MJPEG could still stutter because the realtime
  YOLO hit/bounce/speed analysis was running inside the main dashboard Python
  process.

Fix:

- Added `app/yolo_event_worker.py`.
- `app/orchestrator.py` now starts a per-camera YOLO event worker process for
  YOLO Roadmap pipelines.
- The main process still buffers every selected YOLO detection and player pose,
  but submits only the latest analysis window to a maxsize-1 worker queue.
- The worker runs `detect_single_camera_events()` with the same hit-first /
  bounce-after-cleaning parameters, then returns final event candidates.
- The main process remains the only publisher to `_live_bounces`,
  `_live_hits`, minimap/report state, and 3D push queues.
- Dashboard status reads pipeline snapshots from a background cache instead of
  synchronously polling the multiprocessing manager.
- `/api/dashboard/status` access logs are suppressed to avoid long-run log
  spam during dashboard polling.

Verification after worker split:

```text
Focused tests:
python -m py_compile app\orchestrator.py app\yolo_event_worker.py app\api\routes.py app\pipeline\camera_pipeline.py
pytest tests\test_realtime_logic.py tests\test_player_detector.py tests\test_yolo_bounce_filter.py tests\test_realtime_hit_bounce_refiner.py -q
29 passed

Runtime, cam68 YOLO:
pipeline fps: about 25 FPS
preview seq fps: about 24.5-25.9 FPS
dashboard status latency: p50 about 4 ms, max about 29 ms in 20 samples
MJPEG unique FPS: 288 unique frames / 12 s, about 24.0 FPS
worker telemetry: worker_enabled=true, submitted/result task ids advancing
```

## Fifth-stage fix: reduce browser main-thread overlay pressure

The backend stream path was healthy after the worker split, but the browser
could still look choppy after switching to YOLO. Current runtime evidence:

```text
YOLO dashboard status:
cam66 pipeline fps: 24.8, preview_fps: 25.2, latest_preview_age_ms: 14.8
cam68 pipeline fps: 24.7, preview_fps: 25.0, latest_preview_age_ms: 1.8

MJPEG unique FPS, 5 s:
cam66: 125 unique / 5.02 s = 24.88 FPS
cam68: 126 unique / 5.01 s = 25.13 FPS

MJPEG jitter, 10 s:
cam66: 250 frames, p95 49 ms, max 62.8 ms
cam68: 238 frames, p95 48.3 ms, max 641.7 ms, over 120 ms: 1

Dashboard APIs:
/api/dashboard/status avg 5.7-6.9 ms, about 2.2 KB
/api/dashboard/live avg 11.8-12.5 ms, about 18.5 KB
```

Conclusion:

- The camera subprocess and MJPEG endpoint can deliver about 25 unique FPS.
- The remaining user-visible stutter is likely browser-side contention:
  MJPEG image decode/paint competes with YOLO live overlay JSON parsing,
  minimap updates, and the ball overlay canvas repaint loop.

Additional frontend fix:

- `app/api/templates/dashboard.html`
  - Reduced live overlay/minimap polling from 250 ms to 500 ms.
  - Reduced ball overlay repaint cap from 40 ms to 67 ms.
  - Added a dirty/active-window gate so ball overlay canvases repaint only
    when a detection changed or the short trail is still fading.
  - Clears overlay canvases once after activity ends, then stops repainting.
  - Skips live overlay polling while the document is hidden.

Verification:

```text
Dashboard reload in browser:
JS error/warn logs: none
video-container streams: 1 MJPEG img
overlay canvases: 1
```

Rollback:

- Restore `_LIVE_OVERLAY_POLL_MS` to `250`.
- Restore `_OVERLAY_FRAME_MS` to `40`.
- Remove `_ballOverlayDirty`, `_ballOverlayActiveUntil`,
  `_ballOverlayNeedsClear`, `_clearBallOverlayCanvases()`, and
  `_hasActiveBallOverlayTrail()` gating if smoother overlay animation is more
  important than video playback stability.

## Sixth-stage fix: lazy-load the external 3D iframe

After reloading the dashboard in YOLO mode, backend evidence still showed the
video path was healthy:

```text
Model: yolo_roadmap
cam68 pipeline fps: 24.7-25.3
cam68 preview_fps: 24.8-25.3
latest_preview_age_ms: 8-47 ms
MJPEG unique FPS, 4 s: 100 unique / 4.02 s = 24.88 FPS
/api/dashboard/live: recent_bounces=24, recent_hits=24
Focused tests: 49 passed
```

The remaining browser-side pressure came from the dashboard's main 3D panel:

```html
<iframe id="court3d-iframe" src="https://tennis.motionrivalry.com/tennisAi/">
```

That external page loaded by default even when 3D Push was off. It could compete
with MJPEG decode/paint, ball overlay canvas drawing, minimap drawing, and live
JSON polling. This is especially visible after switching to YOLO because YOLO
produces a fresher selected ball point and therefore more overlay updates.

Fix:

- `app/api/templates/dashboard.html`
  - The 3D iframe now starts as `about:blank` and hidden.
  - A lightweight placeholder is shown instead.
  - Enabling `3D Push` loads the iframe.
  - Disabling `3D Push` unloads the iframe back to `about:blank`.
  - Manual `Load 3D` still allows opening the panel when needed.
  - Local Three.js rendering, where present, now caps pixel ratio and renders
    at low FPS when idle, with short high-FPS windows after new data or user
    interaction.

Verification:

```text
Dashboard reload after patch:
activeCam: cam68
imgSrc: /api/camera/cam68/stream?delay=0
imgNatural: 640x360
3D Push: off
court3d iframe src: about:blank
court3d iframe display: none
placeholder display: flex
overlay canvases: 1
```

Conclusion:

- `ultralytics.predict(stream=True)` is not the first fix for this observed
  stutter, because the YOLO camera process and MJPEG endpoint already sustain
  about 25 unique FPS.
- If future profiling shows the camera process itself falls below source FPS,
  then evaluate `predict(stream=True)` or a persistent YOLO generator. For the
  current issue, the highest-confidence fix is reducing browser/GPU contention.

Rollback:

- Restore the iframe `src` to `https://tennis.motionrivalry.com/tennisAi/`.
- Remove `loadCourt3DFrame()`, `unloadCourt3DFrame()`, and
  `setCourt3DFrameEnabled()`.
- Remove the `toggle3DPush()` and `syncToggles()` calls that load/unload the
  iframe.

## Seventh-stage fix: stabilize MJPEG streaming under multiple clients

After the current-state 3000-frame replay finished, the frame store recovered
to 25 FPS but a direct MJPEG probe briefly measured only about 20 FPS:

```text
cam68 status after replay:
pipeline fps: mostly 24.5-25.7
preview_fps: mostly 24.8-25.3

MJPEG probe before route change:
8 s: 163 unique frames, about 20.3 FPS
p95 gap: about 202 ms
max gap: about 323 ms
```

Connection inspection also showed several long-lived localhost clients from
Codex/Chrome and some stale `CLOSE_WAIT`/`FIN_WAIT_2` entries after repeated
page reloads and probes. The old MJPEG route used an async generator that called
`asyncio.to_thread(orch.wait_for_latest_frame, ...)` once per emitted frame and
per connected client. With several long-lived MJPEG clients, that creates
unnecessary threadpool churn.

Fix:

- `app/api/routes.py`
  - Changed `/api/camera/{name}/stream` to use a synchronous streaming
    generator that blocks directly on `orch.wait_for_latest_frame()`.
  - The route now wakes only on new frame seq changes, instead of sleeping and
    repeating cached JPEGs.
  - Added `Cache-Control: no-store`, `Pragma: no-cache`, and
    `X-Accel-Buffering: no`.
  - Keeps the existing delay buffer behavior for `?delay=N`.

Verification after service restart:

```text
Model: yolo_roadmap
cam68 status:
pipeline fps: 24.7-25.2
preview_fps: 24.8-25.1
latest_preview_age_ms: 14-47 ms

MJPEG probe after route change:
8 s: 200 unique frames, 24.96 unique FPS
p95 gap: about 155 ms
avg JPEG: about 19.7 KB

Dashboard API:
recent_bounces: 12
recent_hits: 18
show_hits_on_minimap: true
latest_detection_frame: 2258

Focused tests:
pytest tests\test_realtime_logic.py tests\test_realtime_hit_bounce_refiner.py tests\test_yolo_bounce_filter.py tests\test_player_detector.py -q
49 passed
```

30-second stability probe after the route change:

```text
MJPEG stream:
elapsed: 30.01 s
unique frames: 749
unique FPS: 24.96
avg JPEG: 19.6 KB

Dashboard status samples:
samples: 15
pipeline fps avg/min: 24.97 / 24.69
preview fps avg/min: 25.01 / 24.72
max latest_preview_age_ms: 46.9
```

Note:

- The synthetic MJPEG probe can report large inter-frame gap percentiles because
  `urllib` may read multiple multipart frames from one socket read and timestamp
  them as arriving together. For stream health, unique frame count over wall
  time and dashboard `latest_preview_age_ms` are more reliable in this probe.
- Repeated reloads from Chrome/Codex/probe clients can still leave short-lived
  TCP `TIME_WAIT`/`FIN_WAIT_2` entries. After the synchronous generator change,
  concurrent MJPEG clients still received about 25 unique FPS each.

Current-state replay/GT verification:

```text
Replay summary:
reports\cam68_clip11_current_replay_20260605_171126.json

GT compare summary:
reports\cam68_clip11_current_gt_compare_20260605_171350\summary.json

GT match-ball bounces:
[554, 604, 635, 878, 932, 980, 1017, 1080, 1129, 1195, 1844, 2014, 2047, 2697, 2738, 2889]

Dashboard publishable bounces:
[553, 604, 636, 877, 932, 980, 1017, 1081, 1129, 1196, 1846, 2014, 2056, 2696, 2736, 2887]

stream_vs_offline_dashboard:
match_count: 16
miss_count: 0
false_positive_count: 0

dashboard_vs_gt frame+space:
match_count: 16
miss_count: 0
false_positive_count: 0
```

Rollback:

- Restore the async MJPEG generator if future ASGI behavior requires it.
- Keep the seq/event wait mechanism either way; do not go back to fixed
  `sleep(1/15)` plus repeated cached JPEGs.
