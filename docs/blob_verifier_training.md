# Blob Verifier YOLO Training Guide

## Overview

Blob Verifier is a YOLO-based secondary verification module for filtering TrackNet false positives.
When TrackNet detects multiple blobs (candidate ball positions) in a single frame, the verifier
crops each blob region and runs YOLO detection to confirm whether it contains a real tennis ball.

**Pipeline**: TrackNet (high recall) → YOLO Blob Verifier (high precision) → court-X filtering

## Architecture Decisions

### Why YOLO detect (not classify)?

- **detect mode** outputs bbox + confidence, which can refine ball center position
- **cls mode** only returns probability, no spatial information
- detect mode with `single_cls=True` treats all classes as one (ball), simplifying training

### Why crop-based detection?

Tennis balls are 5-15px in 1920x1080 frames. YOLO's receptive field cannot detect objects this
small in full-resolution images. Cropping 128x128 regions around each blob makes the ball
proportionally larger (~10% of crop), within YOLO's detection capability.

### Why not use COCO pretrained "sports ball" class?

Zero-shot baseline test showed COCO pretrained YOLO11n detected "sports ball" in only 6/61
crops (9.8% recall). Tennis balls at 5-15px are too small for COCO pretrained models.
Fine-tuning on domain-specific crops is mandatory.

## Training Data Preparation

### Tools

```bash
python -m tools.prepare_yolo_crops <dir1> <dir2> ... [--crop-size 128] [--neg-per-frame 1]
```

### Annotation Format

Supports LabelMe JSON with both point and rectangle annotations:
- **Rectangle** (preferred): 4-corner points, parsed as `pts[0]` (top-left) and `pts[2]` (bottom-right)
- **Point** (legacy): single point, bbox generated with `--box-radius` (default 10px)

**Critical bug found**: LabelMe rectangles store 4 corner points `[TL, TR, BR, BL]`.
Initial code used `pts[0]` and `pts[1]` (both on top edge, same Y), producing **bbox height = 0**.
Fix: use `pts[0]` and `pts[2]` (diagonal corners) to get correct width and height.

### Positive Samples

- Crop 128x128 centered on each GT ball annotation
- YOLO label: `0 0.5 0.5 <w_norm> <h_norm>` (ball always at crop center)
- bbox size from actual rectangle annotation, or fallback to `box_radius` for point annotations

### Negative Samples

- Random 128x128 crops from background regions
- **Only from frames with box annotations** (not point annotations) — frames without box labels
  may contain unlabeled balls
- Minimum distance 100px from any ball position (GT or model)
- Skip OSD area (top 41px)
- YOLO label: empty file (no objects)
- Ratio: ~1 negative per GT frame (`--neg-per-frame 1`)

### Labeling Guidelines

1. **Use rectangle (box) annotations**, not points — provides actual ball size for training
2. **Only label balls you can identify in a single frame** — if you need consecutive frames to
   confirm, skip it (avoids training the model to detect shoe/clothing as ball)
3. **Label dead balls too** — dead ball vs match ball distinction is handled by trajectory analysis,
   not the verifier
4. **Draw tight bboxes** around the ball edge, keep consistent style
5. **Cover diverse scenarios**: serve, rally, net play, different blur levels, different distances
6. **Don't over-label consecutive frames** — adjacent frames are nearly identical, add little value
7. **Prioritize far-court balls** (5-8px) — these are hardest to detect and most valuable for training

### Dataset Statistics (Current)

| Source | Annotated Frames | Boxes |
|--------|-----------------|-------|
| cam68_2min | 410 | 496 |
| cam66_2min | 450 | 507 |
| cam68_clip | 33 | 33 |
| cam66_clip | 33 | 33 |
| **Total** | | **~1003 boxes** |

Generated crops: 1393 positive + 860 negative = 2253 total (80/20 train/val split)

## Training

### Tool

```bash
python -m tools.train_blob_verifier [--device 0] [--epochs 100] [--model yolo11n.pt]
```

### Experiment History

| Version | Epochs | freeze | mosaic | bbox bug | P | R | mAP50 | mAP50-95 |
|---------|--------|--------|--------|----------|------|------|-------|----------|
| v2 | 50 | 10 | 1.0 | height=0 | 0.404 | 0.269 | 0.173 | 0.122 |
| v3 | 100 | 0 | 0.0 | height=0 | 0.856 | 0.217 | 0.314 | 0.241 |
| **v4** | **100** | **0** | **0.0** | **fixed** | **0.659** | **0.736** | **0.771** | **0.443** |

### Key Findings

1. **bbox height=0 bug was catastrophic** — v2/v3 had zero-height labels, making it impossible
   for YOLO to learn ball detection. Fixing this single bug improved mAP50 from 0.31 to 0.77.

2. **freeze=0 is better than freeze=10** — COCO pretrained backbone features don't transfer well
   to 128x128 tennis ball crops. All layers need fine-tuning.

3. **mosaic=0.0 is critical** — Mosaic augmentation creates 4-image grids, shrinking each sub-image
   to 64x64. A 10px ball becomes ~5px, nearly invisible. Disabling mosaic preserves ball visibility.

4. **scale=0.2, translate=0.05** — Light augmentation. Heavy scale/translate moves the ball
   (always at center) too far, confusing the model.

### Recommended Training Parameters

```python
model.train(
    data="data/blob_crops/data.yaml",
    epochs=100,
    imgsz=128,
    batch=32,
    freeze=0,          # fine-tune all layers
    single_cls=True,   # single class detection
    device=0,          # CUDA GPU
    mosaic=0.0,        # disable mosaic (critical for small objects)
    scale=0.2,         # light scale augmentation
    translate=0.05,    # light translate augmentation
)
```

### Validation Results (v4 best model)

- **mAP50**: 0.771
- **Precision**: 0.659
- **Recall**: 0.736
- **Practical test**: 20/20 positive crops detected (100%), 0/20 false positives (0%)

## Pipeline Integration

### Config (`config.yaml`)

```yaml
blob_verifier:
  enabled: true
  model_path: model_weight/blob_verifier_yolo.pt
  crop_size: 128
  conf: 0.25
```

### Verification Logic (`app/pipeline/blob_verifier.py`)

1. **Short-circuit**: if `len(blobs) <= 1`, return immediately (no verification needed)
2. **Crop**: extract 128x128 crops around each blob from the raw (unmasked) frame
3. **Detect**: batch YOLO inference on all crops
4. **Filter**: remove blobs where YOLO found no ball
5. **Re-rank**: remaining blobs scored by `yolo_conf * blob_sum`
6. **Fallback**: if YOLO filters ALL blobs, fall back to TrackNet top-1

### Performance

- Inference: ~0.3ms per crop (RTX 5070 Ti)
- Typically 2-5 crops per multi-blob frame
- Total overhead: <2ms per frame (negligible)

## Troubleshooting

### Model not detecting balls

- Check bbox labels are not zero-height (LabelMe rectangle parsing bug)
- Verify `mosaic=0.0` during training
- Lower `conf` threshold in config (try 0.15)
- Check crop size matches training (`imgsz=128`)

### Too many false positives

- Increase `conf` threshold (try 0.4)
- Add more diverse negative samples
- Check negative samples don't accidentally contain balls

### Poor generalization to new camera angles

- Add training data from the new camera
- Ensure diverse lighting/blur conditions in training set
- Consider increasing training data to 2000+ boxes
