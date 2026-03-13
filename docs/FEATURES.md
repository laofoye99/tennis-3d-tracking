# Tennis 3D Tracking System — Feature Guide

## System Overview

A dual-camera tennis ball 3D tracking system that uses TrackNetV3 deep learning detection combined with stereo triangulation to reconstruct ball trajectories in real-time 3D. The system provides:

- **Dual-camera tracking** using cameras at opposite baselines (cam66 at y=0, cam68 at y=23.77m)
- **Real-time 3D triangulation** from two camera angles via homography-based ray intersection
- **Physics-based trajectory fitting** with RANSAC spatial parabolic modeling
- **Automated rally segmentation** detecting serve/return boundaries
- **Bounce detection** using Z-axis V-shape analysis
- **Interactive calibration** with manual point correspondence marking

---

## Camera Setup & Calibration

### Camera Placement
- **cam66**: Positioned behind the near baseline (y=0), looking toward the far end (y=23.77m)
- **cam68**: Positioned behind the far baseline (y=23.77m), looking toward the near end
- Both cameras should be elevated (5-7m) and roughly centered (x ≈ 4.1m) above the court

### Interactive Calibration UI (Dashboard → Calibration tab)

The calibration page allows you to fine-tune camera positions by marking corresponding ground-level points visible in both camera images:

1. **Select videos** for cam66 and cam68 from the dropdown menus
2. **Load frames** at a specific timestamp
3. **Click matching points**: Click a point on the cam66 image, then click the same physical point on the cam68 image. The system records pixel coordinates for both cameras.
4. **Minimum 4 point pairs** required (6+ recommended for better accuracy)
5. **Run Calibration**: Uses PnP (Perspective-n-Point) solving with the marked correspondences
6. **Apply to Config**: Updates `config.yaml` with calibrated camera positions and enables `use_calibrated_positions: true`
7. **Export JSON**: Download calibration results as a JSON file

**Zoom & Pan for precise marking:**
- **Scroll wheel** on either camera image to zoom in/out (1x–20x), centered on cursor
- **Ctrl+drag** or **middle-click drag** to pan the zoomed image
- Zoom indicator in bottom-right corner shows current zoom level with a Reset button
- Markers scale inversely with zoom so they stay readable at any zoom level

**Best practices for marking points:**
- Use ground-level stationary objects (court line intersections, corners, fixed equipment)
- Spread points across the entire court for better coverage
- Zoom in (5x+) for sub-pixel accuracy when marking court line intersections
- Avoid points at extreme image edges where lens distortion is highest

### Auto Calibration (Labelme-based)

For higher precision with 12 labeled court keypoints per camera:
- Use Labelme to annotate court keypoints in `src/cam66.json` and `src/cam68.json`
- Run `POST /api/calibration/run` or `python -m app.calibration`
- Uses Zhang's calibration method + iterative PnP refinement

---

## Video Test Workflow

### 1. Upload Videos
- Drag-and-drop or click to upload video files (.mp4, .avi, .mkv, etc.)
- Videos appear in the video list on the left panel

### 2. Select Clips (Dual-View Setup)
- Choose cam66 video in **View 1** and cam68 video in **View 2**
- Use range sliders to select the clip segment for processing
- View 1 is the anchor (sets start and end); View 2 sets start only (duration matches View 1)
- Frame-step buttons (◀ ▶) allow fine-grained frame alignment

### 3. Run Parallel Processing
Click **Run** to start the pipeline:

| Phase | Description |
|-------|-------------|
| **Phase 1-2** | YOLO + TrackNet ball detection on both cameras simultaneously |
| **Phase 3** | 3D triangulation: auto time-offset search, frame matching, ray intersection |
| **Phase 4** | Trajectory fitting: RANSAC parabolic model, rally segmentation |
| **Phase 5** | Analytics: V-shape bounce detection, net crossing, landing points |

### Real-Time Detection Display

During processing (Phases 1-2), the system streams detection results live via Server-Sent Events (SSE):

- **Live 2D minimap**: Shows a court diagram with detection dots appearing in real-time (blue for cam66, red for cam68)
- **Live detection table**: Displays the latest 30 detections with camera, frame, position, and confidence
- **Detection counters**: Shows per-camera detection counts updated at ~3Hz
- Detections stream at 10Hz from the backend; the UI throttles updates to avoid rendering overhead
- When processing completes, the live view transitions to the 3D trajectory visualization

### 4. Processing Pipeline Details

- **Detection**: TrackNetV3 (3-frame temporal model) detects ball positions in each camera
- **Auto Sync**: Sweeps time offsets to find optimal frame alignment between cameras
- **Triangulation**: Projects each camera's detection to 3D via homography rays, finds closest point
- **Rally Segmentation**: Splits trajectory by time gaps >1s or spatial jumps >80m/s
- **Trajectory Fit**: RANSAC spatial parabola (X linear in Y, Z quadratic in Y)
- **Bounce Detection**: V-shape fitting on Z-axis values within sliding windows

---

## Playback & Visualization

### Continuous Frame-Based Playback

After processing completes, the system enters a **presentation mode** with continuous video playback:

- **Slider scrubs through all video frames** (not just detection points)
- Wherever 3D data exists, overlays appear on camera frames and the minimap
- Frames without ball detection still show camera feeds (with stats showing "--")
- Time-based scrubbing for smooth continuous playback

### Presentation Grid Layout

The playback uses a TV-broadcast-style integrated layout:

```
┌──────────────┬──────────────┬──────────────┐
│  Camera 66   │  Camera 68   │  Minimap     │
│  (live frame │  (live frame │  (animated   │
│   + ball     │   + ball     │   court map) │
│   marker)    │   marker)    │              │
├──────────────┴──────────────┤  Speed KM/H  │
│  3D Trajectory View         │  Height M    │
│  (Three.js interactive)     │  Position    │
│                             │  Rally info  │
│                             │  Landing     │
└─────────────────────────────┴──────────────┘
```

### Minimap Features
- **Trajectory trail**: Orange line showing ball path from start to current time
- **Fading dots**: Recent 10 detection points with progressive opacity
- **Numbered bounce markers**: "B1", "B2", etc. appear as they're reached
- **Landing point**: Target circle with IN/OUT label and distance from nearest line
- **Ball position**: Red glowing dot at current detection (grayed out when no detection)

### Timeline
- **Rally bands**: Colored segments showing rally boundaries
- **Event markers**: Clickable dots for bounces (green/red), net crossing (blue), landing (orange)
- **Click-to-seek**: Click any marker to jump to that event

### Per-Frame Stats
- **Speed**: Instantaneous velocity computed from consecutive 3D points (km/h)
- **Height**: Ball Z coordinate (meters above ground)
- **Position**: Court coordinates (mirrored X for display consistency)
- **Rally**: Current rally number and shot count
- **Landing**: Progressive bounce count with distance from nearest line

### Playback Controls
- **Play/Pause**: Space bar or button
- **Step**: Arrow keys (← →) for frame-by-frame
- **Speed**: 0.25x, 0.5x, 1x, 2x, 4x

---

## Rally Analysis

### Automatic Rally Segmentation
Rallies are detected automatically based on:
- **Time gaps** > 1 second between consecutive detections
- **Spatial jumps** > 80 m/s (impossible ball velocity indicates different rally)
- **Quality filters**: Minimum 15 points, 0.5s duration, 2m displacement

### Rally Navigation
- **Rally selector**: Buttons to switch between "All" and individual rallies (R1, R2, ...)
- **Timeline bands**: Visual rally boundaries on the playback timeline
- **Click-to-jump**: Click a rally band to jump to its start

### Landing Point Accuracy
- Each landing/bounce shows distance from the nearest court line (cm)
- Numbered markers on the minimap track landing sequence: B1, B2, B3...
- IN/OUT determination based on court boundary check

---

## Bounce Detection

### V-Shape Z-Axis Algorithm
- Slides a window (size=10) along Z values looking for V-shaped patterns
- Fits both a straight line and a V-shape (two-segment piecewise linear)
- Bounce detected when V-shape fit is significantly better (improvement ratio > 0.3)
- Ground threshold: z < 0.5m (configurable)

### Dependency on Z Accuracy
Bounce detection quality depends directly on Z-axis accuracy, which requires:
1. **Accurate camera positions** (height especially)
2. **Good calibration** (use the Calibration tab)
3. **Ground plane validation**: The system checks Z distribution on playback enter and warns if values appear offset

### Z Validation
After calibration, re-run the video test. The system checks:
- Z minimum should be near 0 (ground level)
- Z median should be between 0-3m (normal trajectory range)
- If offset detected, a warning appears suggesting re-calibration

---

## 3D Trajectory

### Triangulation
- Each camera's ball detection (pixel) is projected to a 3D ray using homography
- The closest point between the two rays gives the 3D ball position
- **Ray distance** metric indicates triangulation quality (lower = better)

### Auto Time-Offset Synchronization
- Cameras may not start recording at exactly the same time
- The system sweeps ±100 frame offsets looking for minimum mean ray distance
- Uses robust trimmed-mean (central 60%) to ignore outliers

### RANSAC Parabolic Fitting
- Models trajectory as: X(Y) = linear, Z(Y) = quadratic (gravity parabola)
- RANSAC rejects outlier points (wrong detections, noise)
- Detects bounces as Z discontinuities in the fitted curve

### Net Crossing Detection
- Finds where trajectory crosses Y = 11.885m (net position)
- Reports: speed at net, height above net, clear/fault classification

---

## Technical Architecture

### Backend (Python)
- **Flask/FastAPI** web server with REST API
- **OpenCV** for video processing, frame extraction, calibration
- **TrackNetV3** (PyTorch) for ball detection
- **NumPy/SciPy** for 3D math, trajectory fitting, analytics

### Frontend (JavaScript)
- **Vanilla JS** — no framework dependencies
- **Three.js** (r128) for 3D trajectory visualization with OrbitControls
- **Canvas API** for minimap, court drawing, trajectory overlay
- **CSS Grid** for presentation layout

### Key Files

| File | Purpose |
|------|---------|
| `app/api/templates/dashboard.html` | Main dashboard UI (HTML + CSS + JS) |
| `app/api/routes.py` | REST API endpoints |
| `app/orchestrator.py` | Pipeline coordination, 3D computation |
| `app/trajectory.py` | Detection cleaning, triangulation, trajectory fitting, rally segmentation |
| `app/calibration.py` | Camera calibration (PnP, stereo validation) |
| `app/analytics.py` | Bounce detection (V-shape), rally tracking |
| `config.yaml` | Camera positions, model paths, settings |
| `src/homography_matrices.json` | Per-camera image↔world projection matrices |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML |
| `POST` | `/api/video-test/run-parallel` | Start dual-camera processing |
| `POST` | `/api/video-test/compute-trajectory` | Compute 3D trajectory |
| `GET` | `/api/video-preview/frame` | Extract frame at timestamp |
| `POST` | `/api/calibration/run-from-points` | Run calibration from marked points |
| `POST` | `/api/calibration/apply` | Apply calibration to config |
| `GET` | `/api/calibration/status` | Get calibration results |
| `GET` | `/api/video-test/detections` | Get stored detections |

---

## Court Coordinate System

Standard ITF singles court dimensions (meters):

```
(0,23.77) ─────────────────── (8.23,23.77)
│                                        │
│              Far baseline              │
│                                        │
│ (0,18.285) ── Far service ── (8.23,18.285) │
│        │                    │          │
│        │   (4.115,18.285)   │          │
│        │        │           │          │
│ ═══════╪════════╪═══════════╪══════════ │  ← Net (y=11.885)
│        │        │           │          │
│        │   (4.115,5.485)    │          │
│        │                    │          │
│ (0,5.485) ── Near service ── (8.23,5.485) │
│                                        │
│              Near baseline             │
│                                        │
(0,0) ───────────────────────── (8.23,0)
```

- **cam66** at near baseline (y < 0), looking toward y = 23.77
- **cam68** at far baseline (y > 23.77), looking toward y = 0
- **Z-axis**: Height above ground (0 = ground, positive = up)
