"""Generate YOLO-format training data from existing LabelMe annotations.

Reads GT annotations + original frame images, extracts crops around each blob,
and labels them with YOLO bbox format for fine-tuning.

Supports multiple input directories — crops are merged into one dataset.

Usage:
    python -m tools.prepare_yolo_crops dir1 dir2 dir3 [--crop-size 128] [--box-radius 10]

Example:
    python -m tools.prepare_yolo_crops uploads/cam66_clip uploads/cam68_clip \
        uploads/cam66_20260307_173403_2min uploads/cam68_20260307_173403_2min

Output structure:
    data/blob_crops/
        images/train/  images/val/
        labels/train/  labels/val/
        data.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def load_annotations(ann_dir: Path) -> dict[int, list[dict]]:
    """Load LabelMe JSON annotations grouped by frame index.

    Supports both point and rectangle (box) annotations.
    For rectangles, computes center and stores bbox width/height.
    """
    frames: dict[int, list[dict]] = {}
    for jf in sorted(ann_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            fi = int(jf.stem)
        except ValueError:
            continue

        blobs = []
        for shape in data.get("shapes", []):
            pts = shape.get("points", [])
            if not pts:
                continue
            label = shape.get("label", "ball")
            shape_type = shape.get("shape_type", "point")

            if shape_type == "rectangle" and len(pts) >= 2:
                # Rectangle: use first and third corner (top-left, bottom-right)
                x1, y1 = pts[0]
                x2, y2 = pts[2] if len(pts) >= 3 else pts[1]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                blobs.append({
                    "pixel_x": float(cx),
                    "pixel_y": float(cy),
                    "box_w": float(bw),
                    "box_h": float(bh),
                    "score": shape.get("score"),
                    "label": label,
                })
            else:
                # Point annotation (legacy)
                px, py = pts[0]
                blobs.append({
                    "pixel_x": float(px),
                    "pixel_y": float(py),
                    "box_w": None,
                    "box_h": None,
                    "score": shape.get("score"),
                    "label": label,
                })
        frames[fi] = blobs
    return frames


def _extract_and_save_crop(
    frame: np.ndarray,
    cx: int,
    cy: int,
    crop_size: int,
    out_path: Path,
    split: str,
    crop_name: str,
    label_line: str | None,
) -> None:
    """Extract a crop and save image + label file."""
    h, w = frame.shape[:2]
    half = crop_size // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half), min(h, cy + half)
    dx0, dy0 = x0 - (cx - half), y0 - (cy - half)
    dx1, dy1 = dx0 + (x1 - x0), dy0 + (y1 - y0)
    crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    crop[dy0:dy1, dx0:dx1] = frame[y0:y1, x0:x1]

    cv2.imwrite(str(out_path / "images" / split / f"{crop_name}.jpg"), crop)
    label_path = out_path / "labels" / split / f"{crop_name}.txt"
    if label_line:
        with open(label_path, "w") as f:
            f.write(label_line + "\n")
    else:
        label_path.touch()  # empty = no object (background)


def _process_one_dir(
    cam_path: Path,
    dir_prefix: str,
    out_path: Path,
    crop_size: int,
    box_radius: int,
    val_ratio: float,
    neg_per_frame: int = 1,
    min_ball_dist: int = 100,
) -> tuple[int, int]:
    """Process a single directory, return (positive_count, negative_count).

    Positive: crops centered on GT ball points.
    Negative: random crops far from ANY ball position (GT or model blob).
    """
    # Find images
    image_files: dict[int, Path] = {}
    for img_path in sorted(cam_path.glob("*.jpg")) + sorted(cam_path.glob("*.png")):
        try:
            fi = int(img_path.stem)
            image_files[fi] = img_path
        except ValueError:
            continue

    # Load annotations
    annotations = load_annotations(cam_path)
    if not annotations:
        print(f"  No annotations found in {cam_path}, skipping")
        return 0, 0

    gt_count = sum(1 for blobs in annotations.values()
                   if any(b["score"] is None for b in blobs))
    print(f"  {cam_path.name}: {len(annotations)} frames, {len(image_files)} images, {gt_count} GT frames")

    half = crop_size // 2
    positive_count = 0
    negative_count = 0
    box_w_norm = (2 * box_radius) / crop_size
    box_h_norm = (2 * box_radius) / crop_size

    frame_indices = sorted(annotations.keys())
    random.shuffle(frame_indices)

    for fi in frame_indices:
        if fi not in image_files:
            continue

        frame = cv2.imread(str(image_files[fi]))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        blobs = annotations[fi]
        gt_blobs = [b for b in blobs if b["score"] is None and b["label"] == "ball"]
        model_blobs = [b for b in blobs if b["score"] is not None]

        # Collect ALL ball positions (GT + model) for negative avoidance
        all_ball_pts = [(b["pixel_x"], b["pixel_y"]) for b in blobs]

        # --- Positive crops: from GT annotation points/boxes ---
        for idx, gt in enumerate(gt_blobs):
            split = "val" if random.random() < val_ratio else "train"
            crop_name = f"{dir_prefix}_frame{fi:05d}_pos{idx}"
            # Use actual bbox if available, otherwise fallback to box_radius
            if gt["box_w"] is not None and gt["box_h"] is not None:
                bw_norm = gt["box_w"] / crop_size
                bh_norm = gt["box_h"] / crop_size
            else:
                bw_norm = box_w_norm
                bh_norm = box_h_norm
            label_line = f"0 0.500000 0.500000 {bw_norm:.6f} {bh_norm:.6f}"
            _extract_and_save_crop(
                frame, int(round(gt["pixel_x"])), int(round(gt["pixel_y"])),
                crop_size, out_path, split, crop_name, label_line,
            )
            positive_count += 1

        # --- Negative crops: random positions far from ANY ball ---
        # Only generate negatives from frames where user labeled boxes (not point annotations)
        # because unlabeled frames or point-annotated frames may have unlabeled balls
        gt_has_boxes = [b for b in gt_blobs if b["box_w"] is not None]
        if not gt_has_boxes:
            continue
        for neg_idx in range(neg_per_frame):
            # Try up to 20 times to find a random position far from all balls
            for _ in range(20):
                rx = random.randint(half, w - half - 1)
                ry = random.randint(half + 41, h - half - 1)  # skip OSD area (top 41px)
                # Check distance to all balls
                too_close = False
                for bx, by in all_ball_pts:
                    if abs(rx - bx) < min_ball_dist and abs(ry - by) < min_ball_dist:
                        too_close = True
                        break
                if not too_close:
                    break
            else:
                continue  # couldn't find a valid position

            split = "val" if random.random() < val_ratio else "train"
            crop_name = f"{dir_prefix}_frame{fi:05d}_neg{neg_idx}"
            _extract_and_save_crop(
                frame, rx, ry, crop_size, out_path, split, crop_name, None,
            )
            negative_count += 1

    return positive_count, negative_count


def generate_crops(
    camera_dirs: list[str],
    crop_size: int = 128,
    box_radius: int = 10,
    val_ratio: float = 0.2,
    output_dir: str = "data/blob_crops",
    neg_per_frame: int = 1,
) -> None:
    """Generate YOLO-format crops from multiple annotated directories.

    Positive crops come from GT annotation points.
    Negative crops are randomly sampled from positions far from any ball.
    """
    out_path = Path(output_dir)

    # Create output directories
    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_pos = 0
    total_neg = 0

    for camera_dir in camera_dirs:
        cam_path = Path(camera_dir)
        if not cam_path.is_dir():
            print(f"  WARNING: {camera_dir} is not a directory, skipping")
            continue

        dir_prefix = cam_path.name

        pos, neg = _process_one_dir(
            cam_path, dir_prefix, out_path, crop_size, box_radius, val_ratio,
            neg_per_frame=neg_per_frame,
        )
        total_pos += pos
        total_neg += neg

    # Write data.yaml
    data_yaml = out_path / "data.yaml"
    abs_path = out_path.resolve()
    with open(data_yaml, "w") as f:
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['ball']\n")

    print(f"\nDataset generated at {out_path}/")
    print(f"  Positive crops (ball):     {total_pos}")
    print(f"  Negative crops (no ball):  {total_neg}")
    print(f"  Total:                     {total_pos + total_neg}")
    print(f"  data.yaml: {data_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO training crops")
    parser.add_argument(
        "camera_dirs", nargs="+",
        help="One or more directories with images and LabelMe JSON annotations",
    )
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--box-radius", type=int, default=10, help="Ball bbox half-size in pixels")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--neg-per-frame", type=int, default=1,
                        help="Random background negatives per frame (default: 1)")
    parser.add_argument("--output", default="data/blob_crops")
    args = parser.parse_args()

    generate_crops(
        camera_dirs=args.camera_dirs,
        crop_size=args.crop_size,
        box_radius=args.box_radius,
        val_ratio=args.val_ratio,
        output_dir=args.output,
        neg_per_frame=args.neg_per_frame,
    )
