"""Phase 0: Zero-shot YOLO baseline test on multi-blob frames.

Tests whether COCO-pretrained YOLO26n can detect tennis balls in 128x128 crops
extracted from TrackNet blob positions, without any fine-tuning.

Usage:
    python -m tools.test_yolo_baseline <camera_dir> [--crop-size 128] [--conf 0.15]

Example:
    python -m tools.test_yolo_baseline uploads/cam68_20260307_173403_2min --crop-size 128
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_frame_annotations(ann_dir: Path) -> dict[int, list[dict]]:
    """Load LabelMe JSON annotations, grouped by frame index.

    Returns:
        dict mapping frame_index -> list of blob dicts with pixel_x, pixel_y, score.
    """
    frames: dict[int, list[dict]] = {}
    for jf in sorted(ann_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract frame index from filename (e.g., "000123.json" -> 123)
        try:
            fi = int(jf.stem)
        except ValueError:
            continue

        blobs = []
        for shape in data.get("shapes", []):
            pts = shape.get("points", [])
            if not pts:
                continue
            px, py = pts[0]
            score = shape.get("score")
            blobs.append({
                "pixel_x": float(px),
                "pixel_y": float(py),
                "score": score,
                "label": shape.get("label", "ball"),
            })
        frames[fi] = blobs

    return frames


def find_multi_blob_frames(
    frames: dict[int, list[dict]],
) -> list[int]:
    """Find frames with multiple model-output blobs (score is not None)."""
    multi = []
    for fi, blobs in frames.items():
        model_blobs = [b for b in blobs if b["score"] is not None]
        if len(model_blobs) > 1:
            multi.append(fi)
    return sorted(multi)


def run_baseline_test(
    camera_dir: str,
    crop_size: int = 128,
    conf: float = 0.15,
    model_name: str = "yolo11n.pt",
    max_frames: int = 50,
) -> None:
    """Run zero-shot YOLO on multi-blob frame crops."""
    cam_path = Path(camera_dir)

    # Find image files
    image_files = sorted(cam_path.glob("*.jpg")) + sorted(cam_path.glob("*.png"))
    if not image_files:
        print(f"No images found in {cam_path}")
        sys.exit(1)

    # Build frame index -> image path mapping
    frame_images: dict[int, Path] = {}
    for img_path in image_files:
        try:
            fi = int(img_path.stem)
            frame_images[fi] = img_path
        except ValueError:
            continue

    # Load annotations
    annotations = load_frame_annotations(cam_path)
    if not annotations:
        print(f"No JSON annotations found in {cam_path}")
        sys.exit(1)

    # Find multi-blob frames
    multi_frames = find_multi_blob_frames(annotations)
    print(f"Found {len(multi_frames)} multi-blob frames out of {len(annotations)} annotated frames")

    if not multi_frames:
        print("No multi-blob frames to test")
        return

    # Load YOLO model
    from ultralytics import YOLO
    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    # COCO sports ball class
    SPORTS_BALL = 32

    # Process multi-blob frames
    test_frames = multi_frames[:max_frames]
    print(f"\nTesting {len(test_frames)} frames with crop_size={crop_size}, conf={conf}")
    print("-" * 70)

    stats = {"total_crops": 0, "detected": 0, "gt_crops": 0, "gt_detected": 0}
    half = crop_size // 2

    # Collect GT blobs for each test frame
    gt_blobs_per_frame: dict[int, list[dict]] = {}
    for fi in test_frames:
        gt_blobs_per_frame[fi] = [b for b in annotations[fi] if b["score"] is None]

    for fi in test_frames:
        if fi not in frame_images:
            continue

        frame = cv2.imread(str(frame_images[fi]))
        if frame is None:
            continue

        blobs = annotations[fi]
        model_blobs = [b for b in blobs if b["score"] is not None]
        gt_blobs = gt_blobs_per_frame.get(fi, [])

        # Extract crops for all model blobs
        h, w = frame.shape[:2]
        crops = []
        for blob in model_blobs:
            cx = int(round(blob["pixel_x"]))
            cy = int(round(blob["pixel_y"]))
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(w, cx + half), min(h, cy + half)
            dx0, dy0 = x0 - (cx - half), y0 - (cy - half)
            dx1, dy1 = dx0 + (x1 - x0), dy0 + (y1 - y0)
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop[dy0:dy1, dx0:dx1] = frame[y0:y1, x0:x1]
            crops.append(crop)

        if not crops:
            continue

        # Run YOLO on all crops
        results = model(crops, conf=conf, verbose=False)

        frame_results = []
        for idx, (blob, result) in enumerate(zip(model_blobs, results)):
            boxes = result.boxes
            detected = False
            det_conf = 0.0
            det_cls = -1

            if boxes is not None and len(boxes) > 0:
                for j in range(len(boxes)):
                    cls_id = int(boxes.cls[j].item())
                    c = float(boxes.conf[j].item())
                    if cls_id == SPORTS_BALL and c > det_conf:
                        detected = True
                        det_conf = c
                        det_cls = cls_id

            # Check if this blob matches a GT point (within 15px)
            is_gt = False
            for gt in gt_blobs:
                dist = ((blob["pixel_x"] - gt["pixel_x"])**2 +
                        (blob["pixel_y"] - gt["pixel_y"])**2) ** 0.5
                if dist < 15:
                    is_gt = True
                    break

            stats["total_crops"] += 1
            if detected:
                stats["detected"] += 1
            if is_gt:
                stats["gt_crops"] += 1
                if detected:
                    stats["gt_detected"] += 1

            status = "DETECT" if detected else "  miss"
            gt_tag = " [GT]" if is_gt else ""
            frame_results.append(
                f"  blob{idx}: ({blob['pixel_x']:.0f},{blob['pixel_y']:.0f}) "
                f"score={blob['score']:.3f} → {status} conf={det_conf:.3f}{gt_tag}"
            )

        print(f"Frame {fi:>5d} ({len(model_blobs)} blobs, {len(gt_blobs)} GT):")
        for line in frame_results:
            print(line)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Total crops tested:     {stats['total_crops']}")
    print(f"  YOLO detected:          {stats['detected']} ({100*stats['detected']/max(1,stats['total_crops']):.1f}%)")
    print(f"  GT crops (true balls):  {stats['gt_crops']}")
    print(f"  GT detected by YOLO:    {stats['gt_detected']} ({100*stats['gt_detected']/max(1,stats['gt_crops']):.1f}%)")
    print(f"  False positive crops:   {stats['total_crops'] - stats['gt_crops']}")
    fp_detected = stats['detected'] - stats['gt_detected']
    print(f"  FP detected by YOLO:    {fp_detected} ({100*fp_detected/max(1,stats['total_crops']-stats['gt_crops']):.1f}%)")

    # Save visualization of a few crops
    viz_dir = Path("exports/yolo_baseline_viz")
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving crop visualizations to {viz_dir}/")

    viz_count = 0
    for fi in test_frames[:10]:
        if fi not in frame_images:
            continue
        frame = cv2.imread(str(frame_images[fi]))
        if frame is None:
            continue
        model_blobs = [b for b in annotations[fi] if b["score"] is not None]
        for idx, blob in enumerate(model_blobs):
            cx = int(round(blob["pixel_x"]))
            cy = int(round(blob["pixel_y"]))
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(w, cx + half), min(h, cy + half)
            dx0, dy0 = x0 - (cx - half), y0 - (cy - half)
            dx1, dy1 = dx0 + (x1 - x0), dy0 + (y1 - y0)
            crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
            crop[dy0:dy1, dx0:dx1] = frame[y0:y1, x0:x1]

            # Run YOLO and draw results
            res = model(crop, conf=conf, verbose=False)[0]
            annotated = res.plot()

            out_path = viz_dir / f"frame{fi:05d}_blob{idx}.jpg"
            cv2.imwrite(str(out_path), annotated)
            viz_count += 1

    print(f"Saved {viz_count} visualizations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot YOLO baseline test")
    parser.add_argument("camera_dir", help="Directory with images and LabelMe JSON annotations")
    parser.add_argument("--crop-size", type=int, default=128, help="Crop size (default: 128)")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold (default: 0.15)")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model name (default: yolo11n.pt)")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to test (default: 50)")
    args = parser.parse_args()

    run_baseline_test(
        camera_dir=args.camera_dir,
        crop_size=args.crop_size,
        conf=args.conf,
        model_name=args.model,
        max_frames=args.max_frames,
    )
