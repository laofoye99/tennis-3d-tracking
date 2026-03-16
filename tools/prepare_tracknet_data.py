"""Prepare unified TrackNet training data from in-house GT + public dataset.

Converts all data sources into TrackNet training format:
  - In-house LabelMe GT → CSV (Frame, Visibility, X, Y)
  - Dead ball negative mining from existing TrackNet detections
  - Public dataset integration
  - Train/val split files

Usage:
    python -m tools.prepare_tracknet_data
    python -m tools.prepare_tracknet_data --skip-public
    python -m tools.prepare_tracknet_data --skip-dead-ball-mining
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Input/output paths
CAM_CONFIGS = {
    "cam66": {
        "ann_dir": "uploads/cam66_20260307_173403_2min",
        "video": "uploads/cam66_20260307_173403_2min.mp4",
    },
    "cam68": {
        "ann_dir": "uploads/cam68_20260307_173403_2min",
        "video": "uploads/cam68_20260307_173403_2min.mp4",
    },
}
PUBLIC_DIR = "data/tracknet_public"
OUTPUT_DIR = "data/tracknet_training"

# TrackNet input resolution
TRACKNET_H, TRACKNET_W = 288, 512


def load_gt_annotations(ann_dir: Path) -> dict[int, dict]:
    """Load GT-only annotations from LabelMe JSON (score=None, label containing 'ball').

    Returns: {frame_index: {"pixel_x": float, "pixel_y": float}} for GT frames.
    """
    gt = {}
    for jf in sorted(ann_dir.glob("*.json")):
        try:
            fi = int(jf.stem)
        except ValueError:
            continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            pts = shape.get("points", [])
            if not pts:
                continue

            label = (shape.get("label") or "").lower()
            desc = (shape.get("description") or "").lower()
            score = shape.get("score")

            # GT = no score (human annotation), label contains ball-related keywords
            is_gt = score is None and ("ball" in label or "match_ball" in desc)
            if not is_gt:
                continue

            shape_type = shape.get("shape_type", "point")
            if shape_type == "rectangle" and len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[2] if len(pts) >= 3 else pts[1]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            else:
                cx, cy = pts[0][0], pts[0][1]

            gt[fi] = {"pixel_x": float(cx), "pixel_y": float(cy)}
            break  # One ball per frame

    return gt


def get_total_frames(video_path: str) -> int:
    """Get total frame count from video."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def get_frame_resolution(video_path: str) -> tuple[int, int]:
    """Get frame (width, height) from video."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def mine_dead_ball_frames(
    video_path: str,
    gt: dict[int, dict],
    max_frames: int = 3000,
    static_window: int = 5,
    static_threshold: float = 3.0,
) -> set[int]:
    """Mine dead ball frames using existing TrackNet model.

    Runs TrackNet inference and identifies frames where:
    1. Model detects a ball (heatmap peak above threshold)
    2. No GT annotation exists
    3. Detection position is nearly static for `static_window` consecutive frames

    Returns set of frame indices that are dead ball frames.
    """
    try:
        from app.pipeline.inference import TrackNetDetector
        from app.pipeline.postprocess import BallTracker
    except ImportError:
        log.warning("Cannot import TrackNet inference modules, skipping dead ball mining")
        return set()

    log.info("Mining dead ball frames from %s (up to %d frames)...", video_path, max_frames)

    detector = TrackNetDetector(
        model_path="model_weight/TrackNet_best.pt",
        input_size=(288, 512),
        frames_in=8,
        frames_out=8,
        device="cuda",
        bg_mode="concat",
    )
    tracker = BallTracker(original_size=(1920, 1080), threshold=0.35)

    cap = cv2.VideoCapture(video_path)
    detector.compute_video_median(cap, 0, min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Collect detections for non-GT frames
    non_gt_detections = {}  # {fi: (px, py)}
    raw_buffer = []
    seq_len = 8

    for fi in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        raw_buffer.append(frame.copy())

        if len(raw_buffer) < seq_len:
            continue

        if len(raw_buffer) > seq_len:
            raw_buffer.pop(0)

        # Run inference
        heatmaps = detector.infer(raw_buffer)
        if heatmaps is None:
            continue

        # Process last heatmap (corresponds to current frame)
        h_idx = seq_len - 1
        if h_idx < len(heatmaps):
            hm = heatmaps[h_idx]
            # Resize heatmap to original resolution
            orig_h, orig_w = frame.shape[:2]
            hm_full = cv2.resize(hm, (orig_w, orig_h))
            result = tracker.process_heatmap(hm_full)

            if result is not None and fi not in gt:
                px, py, conf = result
                non_gt_detections[fi] = (px, py)

    cap.release()
    log.info("Non-GT detections: %d frames", len(non_gt_detections))

    # Find static detection sequences (dead balls)
    dead_ball_frames = set()
    sorted_frames = sorted(non_gt_detections.keys())

    for i in range(len(sorted_frames)):
        window_frames = []
        window_positions = []

        for j in range(i, min(i + static_window, len(sorted_frames))):
            fj = sorted_frames[j]
            # Check consecutive (allow gap of 1)
            if window_frames and fj - window_frames[-1] > 2:
                break
            window_frames.append(fj)
            window_positions.append(non_gt_detections[fj])

        if len(window_frames) >= static_window:
            positions = np.array(window_positions)
            std_x = np.std(positions[:, 0])
            std_y = np.std(positions[:, 1])

            if std_x < static_threshold and std_y < static_threshold:
                dead_ball_frames.update(window_frames)

    log.info("Dead ball frames identified: %d", len(dead_ball_frames))
    return dead_ball_frames


def convert_inhouse_camera(
    cam_name: str,
    ann_dir: str,
    video_path: str,
    output_dir: Path,
    mine_dead_balls: bool = True,
    max_frames: int = 3000,
) -> dict:
    """Convert in-house camera data to TrackNet training format."""
    ann_path = Path(ann_dir)
    match_name = f"match_{cam_name}"
    match_dir = output_dir / match_name
    frame_dir = match_dir / "frame"
    csv_dir = match_dir / "csv"
    frame_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Load GT
    gt = load_gt_annotations(ann_path)
    log.info("[%s] GT frames: %d", cam_name, len(gt))

    # Get video info
    src_w, src_h = get_frame_resolution(video_path)
    total_frames = min(get_total_frames(video_path), max_frames)
    log.info("[%s] Video: %dx%d, %d frames", cam_name, src_w, src_h, total_frames)

    # Mine dead ball frames
    dead_ball_frames = set()
    if mine_dead_balls:
        dead_ball_frames = mine_dead_ball_frames(video_path, gt, max_frames)

    # Extract frames from video (only those not already extracted as JPGs)
    cap = cv2.VideoCapture(video_path)
    extracted = 0
    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Check if already exists as annotation JPG
        src_jpg = ann_path / f"{fi:05d}.jpg"
        dst_jpg = frame_dir / f"{fi}.jpg"

        if not dst_jpg.exists():
            if src_jpg.exists():
                # Symlink or copy from annotation dir
                import shutil
                shutil.copy2(src_jpg, dst_jpg)
            else:
                cv2.imwrite(str(dst_jpg), frame)
            extracted += 1

    cap.release()
    log.info("[%s] Extracted/copied %d frames to %s", cam_name, extracted, frame_dir)

    # Scale coordinates from source resolution to TrackNet resolution
    scale_x = TRACKNET_W / src_w
    scale_y = TRACKNET_H / src_h

    # Write CSV
    csv_path = csv_dir / f"{match_name}.csv"
    gt_count = 0
    neg_count = 0
    dead_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])

        for fi in range(total_frames):
            if fi in gt:
                # GT positive frame
                px = gt[fi]["pixel_x"] * scale_x
                py = gt[fi]["pixel_y"] * scale_y
                writer.writerow([fi, 2, f"{px:.1f}", f"{py:.1f}"])
                gt_count += 1
            elif fi in dead_ball_frames:
                # Dead ball negative (explicit hard negative)
                writer.writerow([fi, 0, 0, 0])
                dead_count += 1
            else:
                # Unknown frame (no GT) — mark as not visible
                writer.writerow([fi, 0, 0, 0])
                neg_count += 1

    stats = {
        "name": match_name,
        "total_frames": total_frames,
        "gt_frames": gt_count,
        "dead_ball_frames": dead_count,
        "negative_frames": neg_count,
        "source_resolution": f"{src_w}x{src_h}",
    }
    log.info("[%s] CSV written: %d GT, %d dead ball, %d other negative",
             cam_name, gt_count, dead_count, neg_count)
    return stats


def integrate_public_dataset(public_dir: Path, output_dir: Path) -> list[dict]:
    """Copy/link public dataset matches into the training directory."""
    if not public_dir.exists():
        log.warning("Public dataset not found: %s", public_dir)
        return []

    stats = []
    for match_dir in sorted(public_dir.iterdir()):
        if not match_dir.is_dir() or match_dir.name == "raw":
            continue

        labels_csv = match_dir / "labels.csv"
        frame_subdir = match_dir / "frame"

        if not labels_csv.exists() or not frame_subdir.exists():
            continue

        # Copy to output
        dst = output_dir / match_dir.name
        if dst.exists():
            log.info("Public match already exists: %s", dst)
        else:
            import shutil
            shutil.copytree(match_dir, dst)

        # Also copy CSV to csv/ subdirectory for consistency
        csv_dir = dst / "csv"
        csv_dir.mkdir(exist_ok=True)
        dst_csv = csv_dir / f"{match_dir.name}.csv"
        if not dst_csv.exists():
            import shutil
            shutil.copy2(labels_csv, dst_csv)

        # Count
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        n_labeled = sum(1 for r in rows if int(float(r.get("Visibility", 0))) > 0)

        s = {"name": match_dir.name, "frames": len(rows), "labeled": n_labeled}
        stats.append(s)
        log.info("  Public %s: %d frames (%d labeled)", match_dir.name, len(rows), n_labeled)

    return stats


def generate_split_files(
    output_dir: Path,
    val_ratio: float = 0.2,
) -> tuple[list[str], list[str]]:
    """Generate train.txt and val.txt split files.

    Strategy:
    - In-house data: temporal split (first 80% frames train, last 20% val)
    - Public data: random match-level split
    """
    import random
    random.seed(42)

    all_matches = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and (d / "csv").exists():
            all_matches.append(d.name)

    inhouse = [m for m in all_matches if m.startswith("match_cam")]
    public = [m for m in all_matches if not m.startswith("match_cam")]

    train_matches = []
    val_matches = []

    # In-house: all go to train (temporal split handled in Dataset class)
    train_matches.extend(inhouse)

    # Public: random match-level split
    random.shuffle(public)
    n_val = max(1, int(len(public) * val_ratio))
    val_matches.extend(public[:n_val])
    train_matches.extend(public[n_val:])

    # Write split files
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    with open(train_path, "w") as f:
        for m in sorted(train_matches):
            f.write(m + "\n")

    with open(val_path, "w") as f:
        for m in sorted(val_matches):
            f.write(m + "\n")

    log.info("Split: %d train matches, %d val matches", len(train_matches), len(val_matches))
    log.info("Train: %s", train_path)
    log.info("Val: %s", val_path)

    return train_matches, val_matches


def main():
    parser = argparse.ArgumentParser(description="Prepare TrackNet training data")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=3000,
                        help="Max frames per in-house camera")
    parser.add_argument("--skip-public", action="store_true",
                        help="Skip public dataset integration")
    parser.add_argument("--skip-dead-ball-mining", action="store_true",
                        help="Skip dead ball negative mining (faster)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert in-house cameras
    log.info("\n=== Converting in-house cameras ===")
    inhouse_stats = []
    for cam_name, cfg in CAM_CONFIGS.items():
        stats = convert_inhouse_camera(
            cam_name, cfg["ann_dir"], cfg["video"],
            output_dir,
            mine_dead_balls=not args.skip_dead_ball_mining,
            max_frames=args.max_frames,
        )
        inhouse_stats.append(stats)

    # Step 2: Integrate public dataset
    public_stats = []
    if not args.skip_public:
        log.info("\n=== Integrating public dataset ===")
        public_dir = Path(PUBLIC_DIR)
        public_stats = integrate_public_dataset(public_dir, output_dir)

    # Step 3: Generate train/val splits
    log.info("\n=== Generating splits ===")
    train_matches, val_matches = generate_split_files(output_dir, args.val_ratio)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("  DATA PREPARATION SUMMARY")
    log.info("=" * 60)

    total_gt = sum(s.get("gt_frames", 0) for s in inhouse_stats)
    total_dead = sum(s.get("dead_ball_frames", 0) for s in inhouse_stats)
    total_inhouse = sum(s.get("total_frames", 0) for s in inhouse_stats)
    total_public = sum(s.get("frames", 0) for s in public_stats)
    total_public_labeled = sum(s.get("labeled", 0) for s in public_stats)

    log.info("\n  In-house data:")
    for s in inhouse_stats:
        log.info("    %s: %d total, %d GT, %d dead ball",
                 s["name"], s["total_frames"], s["gt_frames"], s["dead_ball_frames"])

    if public_stats:
        log.info("\n  Public data:")
        log.info("    Matches: %d", len(public_stats))
        log.info("    Total frames: %d (%d labeled)", total_public, total_public_labeled)

    log.info("\n  Totals:")
    log.info("    In-house frames: %d (%d GT, %d dead ball neg)",
             total_inhouse, total_gt, total_dead)
    log.info("    Public frames:   %d (%d labeled)", total_public, total_public_labeled)
    log.info("    Grand total:     %d frames", total_inhouse + total_public)
    log.info("    Train matches:   %d", len(train_matches))
    log.info("    Val matches:     %d", len(val_matches))
    log.info("\n  Output: %s", output_dir)

    # Save summary
    import json as json_mod
    summary = {
        "inhouse": inhouse_stats,
        "public": [{"name": s["name"], "frames": s["frames"], "labeled": s["labeled"]}
                    for s in public_stats],
        "train_matches": train_matches,
        "val_matches": val_matches,
        "total_frames": total_inhouse + total_public,
    }
    with open(output_dir / "data_summary.json", "w") as f:
        json_mod.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
