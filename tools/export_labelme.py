"""Export TrackNet detections as LabelMe JSON + frame JPG images.

For each frame, saves:
  - NNNNN.jpg  (the video frame)
  - NNNNN.json (LabelMe-format annotation)

Frames without detections get an empty shapes list.

Usage:
    python -m tools.export_labelme \
        --video66 D:/tennis-dataset/1001/999/cam66_20260307_173403.mp4 \
        --video68 D:/tennis-dataset/1001/999/cam68_20260307_173403.mp4 \
        --max-frames 3000
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_labelme_json(frame_idx, shapes, img_h=1080, img_w=1920):
    """Create a LabelMe-format dict."""
    return {
        "version": "5.1.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{frame_idx:05d}.jpg",
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def make_shape(pixel_x, pixel_y, blob_sum):
    """Create a single LabelMe shape for a ball detection."""
    return {
        "label": "ball",
        "points": [[float(pixel_x), float(pixel_y)]],
        "group_id": None,
        "description": f"ball,conf={blob_sum:.1f}",
        "shape_type": "point",
        "flags": {},
        "mask": None,
    }


def export_camera(video_path, output_dir, detector, postproc, max_frames):
    """Run detection on a single camera video and export LabelMe files.

    Uses the same inference logic as render_tracking_video but without
    temporal consistency filtering — we want raw detections for labeling.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, max_frames)
    seq_len = detector.frames_in
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    os.makedirs(output_dir, exist_ok=True)

    # Compute background median
    logger.info("Computing background median for %s ...", video_path)
    detector.compute_video_median(cap, 0, n_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read all frames into memory
    logger.info("Reading %d frames from %s ...", n_frames, video_path)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n = len(frames)
    logger.info("Read %d frames.", n)

    # Run detection in windows (no temporal filtering for LabelMe export)
    logger.info("Running TrackNet on %d frames (seq_len=%d) ...", n, seq_len)
    detections = {}  # {frame_idx: list[dict]}

    for start in range(0, n, seq_len):
        end = min(start + seq_len, n)
        batch = frames[start:end]
        actual_len = end - start
        if len(batch) < seq_len:
            batch += [batch[-1]] * (seq_len - len(batch))

        heatmaps = detector.infer(batch)

        for i in range(actual_len):
            blobs = postproc.process_heatmap_multi(heatmaps[i], max_blobs=1)
            if blobs:
                detections[start + i] = blobs

        if start % (seq_len * 50) == 0:
            logger.info("  Detection progress: %d/%d", min(start + seq_len, n), n)

    logger.info("Detected ball in %d/%d frames.", len(detections), n)

    # Write LabelMe JSON + JPG for every frame
    logger.info("Writing LabelMe files to %s ...", output_dir)
    for fi in range(n):
        fname = f"{fi:05d}"

        # Save JPG
        jpg_path = os.path.join(output_dir, f"{fname}.jpg")
        cv2.imwrite(jpg_path, frames[fi], [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Build shapes
        shapes = []
        if fi in detections:
            for blob in detections[fi]:
                shapes.append(make_shape(
                    blob["pixel_x"], blob["pixel_y"], blob["blob_sum"]
                ))

        # Save JSON
        labelme_data = make_labelme_json(fi, shapes, img_h, img_w)
        json_path = os.path.join(output_dir, f"{fname}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2)

        if fi % 500 == 0:
            logger.info("  Written %d/%d frames", fi, n)

    logger.info("Done. %d frames exported to %s", n, output_dir)
    return n, len(detections)


def main():
    parser = argparse.ArgumentParser(description="Export TrackNet detections as LabelMe JSON")
    parser.add_argument("--video66", required=True, help="Path to cam66 video")
    parser.add_argument("--video68", required=True, help="Path to cam68 video")
    parser.add_argument("--max-frames", type=int, default=3000, help="Max frames to process")
    args = parser.parse_args()

    cfg = load_config()

    # Import and build detector using the shared builder
    from tools.render_tracking_video import build_detector
    detector, postproc = build_detector(cfg)

    video66_dir = str(Path(args.video66).parent / "cam66_labelme")
    video68_dir = str(Path(args.video68).parent / "cam68_labelme")

    logger.info("=== Processing cam66 ===")
    n1, d1 = export_camera(args.video66, video66_dir, detector, postproc, args.max_frames)

    logger.info("=== Processing cam68 ===")
    n2, d2 = export_camera(args.video68, video68_dir, detector, postproc, args.max_frames)

    logger.info("=== Summary ===")
    logger.info("cam66: %d frames, %d detections (%.1f%%)", n1, d1, 100 * d1 / max(n1, 1))
    logger.info("cam68: %d frames, %d detections (%.1f%%)", n2, d2, 100 * d2 / max(n2, 1))


if __name__ == "__main__":
    main()
