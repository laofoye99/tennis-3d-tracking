# Tracking Video Pipeline — Detection Optimization

## Overview

This document describes the multi-layer filtering pipeline implemented in
`tools/render_tracking_video.py` to produce clean dual-camera tracking videos
with accurate bounce detection. The pipeline addresses the primary challenge:
**false positive detections on white shoes and hats during dead-ball periods**.

---

## Problem

TrackNet detects bright white objects as candidate balls. During dead-ball
periods (between rallies), players walk around and their white shoes / hats
generate persistent false positives. Traditional approaches fail because:

- **Pixel displacement filtering** doesn't work — shoes and hats *move*.
- **3D velocity filtering** doesn't work — when each camera sees a different
  moving object, the triangulated "position" jumps randomly, creating apparent
  velocity that mimics a real ball.

## Solution: Three-Layer Filtering Pipeline

### Layer 1 — TrackNet Heatmap Threshold (0.3)

Raised from the default 0.1 to 0.3 in `config.yaml`. This eliminates
low-confidence blobs while retaining true ball detections.

**Impact**: Significantly reduces raw detection count; ray distance median
improved from 0.65 m → 0.49 m.

### Layer 2 — 8-Frame Temporal Consistency

TrackNet processes 8 frames at a time (`seq_len=8`). Within each window we
require:

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Minimum detections | ≥ 4 of 8 frames | Real ball visible in most frames |
| Max consecutive pixel jump | < 120 px | Smooth trajectory within window |

Windows failing either check have all 8 frames' detections discarded.

### Layer 3 — Ray Distance from Stereo Triangulation

The key insight: when **both cameras see the real ball**, the triangulated rays
converge tightly (ray distance < 2 m). When cameras see **different objects**
(one sees a shoe, the other sees a hat), the rays diverge wildly.

```
Real ball:  cam66 sees ball → ray₁ ──╲
                                       ╳ ← intersection (ray_dist ≈ 0.5m)
            cam68 sees ball → ray₂ ──╱

False positive: cam66 sees shoe → ray₁ ────────── (diverge, ray_dist >> 2m)
                cam68 sees hat  → ray₂ ──────────
```

Combined with court-bounds filtering (0 < x < 8.23, 0 < y < 23.77, z < 8),
this layer removes the vast majority of dead-ball false positives.

**Pass rate**: ~84.5% of matched frames pass (threshold=0.3).

### Post-Filter: Savitzky-Golay Smoothing

After filtering, the surviving 3D trajectory is smoothed with a Savitzky-Golay
filter (`window_length=11, polyorder=3`) applied independently per continuous
segment (gap > 3 frames = new segment). This:

- Reduces high-frequency noise from triangulation imprecision
- Produces cleaner Z-axis profiles for bounce detection
- Reduced false bounce count from ~40 → ~20

---

## Bounce Detection

V-shape fitting on Z-axis values within a sliding window (size=10):

1. Extract Z values in window
2. Fit straight line (baseline) and V-shape (two-segment piecewise linear)
3. If V-shape fit improvement ratio > 0.3 and minimum Z < 0.5 m → bounce

Bounce markers appear on the minimap as numbered circles (B1, B2, ...) with
green (IN) or red (OUT) coloring based on court boundary check.

---

## Video Output Layout

```
┌──────────────┬──────────────┬────────────┐
│   Camera 66  │   Camera 68  │  Minimap   │
│   840×472    │   840×472    │  240×472   │
│              │              │            │
│  Ball marker │  Ball marker │ Court view │
│  + trail     │  + trail     │ + bounces  │
│  (cyan dot)  │  (cyan dot)  │ (B1,B2..) │
└──────────────┴──────────────┴────────────┘
                   1920 × 472
```

- Ball markers: cyan circles with fading 10-frame trail
- Minimap: green court background, white lines, bounce markers
- Frame counter overlay in top-left corner

---

## Configuration

Key parameters in `config.yaml`:

```yaml
model:
  threshold: 0.3          # Heatmap confidence threshold (was 0.1)
```

Pipeline parameters in `tools/render_tracking_video.py`:

```python
RAY_DIST_THRESHOLD = 2.0        # Max ray distance for valid triangulation
SG_WINDOW = 11                  # Savitzky-Golay window length
SG_POLYORDER = 3                # Savitzky-Golay polynomial order
TEMPORAL_MIN_DETECTIONS = 4     # Min detections per 8-frame window
TEMPORAL_MAX_JUMP_PX = 120      # Max pixel jump within window
BOUNCE_Z_THRESHOLD = 0.5        # Max Z for bounce candidate
BOUNCE_IMPROVEMENT_RATIO = 0.3  # V-shape vs line fit improvement
```

## Usage

```bash
python tools/render_tracking_video.py
```

Output: `output/tracking_1800f.mp4` (1920×472, 30fps, 1800 frames)
