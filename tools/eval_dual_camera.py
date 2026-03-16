"""Evaluate dual-camera 3D tracking with temporal continuity filtering.

Runs TrackNet on both cam66 and cam68, then compares:
  - No temporal filtering (ray_distance only)
  - With temporal filtering (ray_distance + 3D distance from prediction)

Uses GT annotations from both cameras to evaluate 3D triangulation quality.

Usage:
    python -m tools.eval_dual_camera
    python -m tools.eval_dual_camera --max-frames 1800
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Defaults
MODEL_PATH = "model_weight/TrackNet_finetuned.pt"
CAM66_VIDEO = "uploads/cam66_20260307_173403_2min.mp4"
CAM68_VIDEO = "uploads/cam68_20260307_173403_2min.mp4"
CAM66_LABELS = "uploads/cam66_20260307_173403_2min"
CAM68_LABELS = "uploads/cam68_20260307_173403_2min"
OUT_DIR = Path("exports/eval_dual_camera")

# Camera positions from config
CAM66_POS = [3.7657, -15.7273, 7.0]
CAM68_POS = [5.8058, 45.0351, 7.0]

# Homography path
HOMOGRAPHY_PATH = "src/homography_matrices.json"


@dataclass
class Detection:
    frame: int
    pixel_x: float
    pixel_y: float
    confidence: float
    world_x: float = 0.0
    world_y: float = 0.0
    candidates: list = field(default_factory=list)


def load_gt(label_dir: str, max_frame: int = 9999) -> dict[int, tuple[float, float]]:
    """Load GT ball positions from LabelMe annotations."""
    from tools.prepare_tracknet_data import load_gt_annotations

    gt_raw = load_gt_annotations(Path(label_dir))
    return {fi: (e["pixel_x"], e["pixel_y"]) for fi, e in gt_raw.items() if fi <= max_frame}


def load_homography(cam_key: str) -> np.ndarray:
    """Load homography matrix for a camera."""
    with open(HOMOGRAPHY_PATH, "r") as f:
        data = json.load(f)
    return np.array(data[cam_key]["H_image_to_world"])


def pixel_to_world(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """Convert pixel coords to world coords using homography."""
    pt = H @ np.array([px, py, 1.0])
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def run_tracknet_multi(
    model_path: str,
    video_path: str,
    max_frames: int,
    threshold: float = 0.35,
    cam_key: str = "cam66",
    label: str = "model",
) -> list[Detection]:
    """Run TrackNet with multi-blob output and world coordinate conversion."""
    from app.pipeline.inference import TrackNetDetector
    from app.pipeline.postprocess import BallTracker

    log.info("Running %s on %s: %s", label, cam_key, model_path)

    detector = TrackNetDetector(
        model_path=model_path,
        input_size=(288, 512),
        frames_in=8,
        frames_out=8,
        device="cuda",
        bg_mode="concat",
    )
    tracker = BallTracker(original_size=(1920, 1080), threshold=threshold)
    H = load_homography(cam_key)

    cap = cv2.VideoCapture(video_path)
    total = min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    detector.compute_video_median(cap, 0, total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detections = []
    raw_buffer = []
    seq_len = 8

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        raw_buffer.append(frame.copy())
        if len(raw_buffer) < seq_len:
            continue
        if len(raw_buffer) > seq_len:
            raw_buffer.pop(0)

        heatmaps = detector.infer(raw_buffer)
        if heatmaps is None:
            continue

        h_idx = seq_len - 1
        if h_idx < len(heatmaps):
            hm = heatmaps[h_idx]
            orig_h, orig_w = frame.shape[:2]
            hm_full = cv2.resize(hm, (orig_w, orig_h))
            hm_full[:41, :603] = 0  # OSD mask

            blobs = tracker.process_heatmap_multi(hm_full, max_blobs=3)
            if not blobs:
                continue

            # Convert all blobs to world coordinates
            candidates = []
            for blob in blobs:
                wx, wy = pixel_to_world(blob["pixel_x"], blob["pixel_y"], H)
                candidates.append({
                    "pixel_x": blob["pixel_x"],
                    "pixel_y": blob["pixel_y"],
                    "world_x": wx,
                    "world_y": wy,
                    "blob_sum": blob["blob_sum"],
                    "blob_max": blob["blob_max"],
                    "blob_area": blob["blob_area"],
                })

            best = candidates[0]
            detections.append(Detection(
                frame=fi,
                pixel_x=best["pixel_x"],
                pixel_y=best["pixel_y"],
                confidence=best["blob_sum"],
                world_x=best["world_x"],
                world_y=best["world_y"],
                candidates=candidates,
            ))

    cap.release()
    log.info("  %s %s: %d detections in %d frames", label, cam_key, len(detections), total)
    return detections


def run_matching(
    cam66_dets: list[Detection],
    cam68_dets: list[Detection],
    temporal_weight: float = 0.0,
    label: str = "baseline",
) -> tuple[list[dict], dict]:
    """Run MultiBlobMatcher on paired detections.

    Returns (3d_points, matcher_stats).
    """
    from app.pipeline.multi_blob_matcher import MultiBlobMatcher

    matcher = MultiBlobMatcher(
        CAM66_POS, CAM68_POS,
        temporal_weight=temporal_weight,
    )

    # Build frame-indexed dicts
    cam66_map = {}
    for d in cam66_dets:
        cam66_map[d.frame] = {
            "frame_index": d.frame,
            "x": d.world_x,
            "y": d.world_y,
            "pixel_x": d.pixel_x,
            "pixel_y": d.pixel_y,
            "confidence": d.confidence,
            "candidates": d.candidates,
        }

    cam68_map = {}
    for d in cam68_dets:
        cam68_map[d.frame] = {
            "frame_index": d.frame,
            "x": d.world_x,
            "y": d.world_y,
            "pixel_x": d.pixel_x,
            "pixel_y": d.pixel_y,
            "confidence": d.confidence,
            "candidates": d.candidates,
        }

    common_frames = sorted(set(cam66_map.keys()) & set(cam68_map.keys()))
    log.info("  %s: %d common frames", label, len(common_frames))

    results = []
    for fi in common_frames:
        match = matcher.match(cam66_map[fi], cam68_map[fi])
        if match is not None:
            results.append(match)

    stats = matcher.get_stats()
    log.info(
        "  %s: %d matched, %d non-top1 (%.1f%%), %d temporal assists (%.1f%%)",
        label,
        stats["matched_frames"],
        stats["non_top1_picks"],
        stats["non_top1_rate"] * 100,
        stats.get("temporal_assists", 0),
        stats.get("temporal_assist_rate", 0) * 100,
    )
    return results, stats


def evaluate_3d_points(
    points: list[dict],
    cam66_gt: dict[int, tuple[float, float]],
    cam68_gt: dict[int, tuple[float, float]],
    dist_threshold: float = 15.0,
) -> dict:
    """Evaluate 3D triangulation results against per-camera GT.

    Uses pixel-level comparison on cam66 to measure accuracy.
    """
    # Frames where BOTH cameras have GT
    both_gt_frames = set(cam66_gt.keys()) & set(cam68_gt.keys())

    pt_map = {p["frame_index"]: p for p in points}

    tp = 0
    errors_cam66 = []

    for fi in both_gt_frames:
        if fi not in pt_map:
            continue
        p = pt_map[fi]
        gx66, gy66 = cam66_gt[fi]

        # Compare cam66 pixel output to GT
        if "cam1_pixel" in p:
            px, py = p["cam1_pixel"]
        else:
            continue

        dist = np.hypot(px - gx66, py - gy66)
        if dist < dist_threshold:
            tp += 1
            errors_cam66.append(dist)

    # Count detections on frames with no GT (FP)
    fp = sum(1 for p in points if p["frame_index"] not in both_gt_frames)

    # Count GT frames with no detection (FN)
    detected_frames = set(pt_map.keys())
    fn = sum(1 for fi in both_gt_frames if fi not in detected_frames)

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    mean_err = float(np.mean(errors_cam66)) if errors_cam66 else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt": len(both_gt_frames),
        "total_detections": len(points),
        "mean_pixel_error": mean_err,
    }


def evaluate_stability(points: list[dict], jump_threshold: float = 5.0) -> dict:
    """Evaluate 3D trajectory stability (frame-to-frame jumps in 3D space)."""
    if len(points) < 2:
        return {"jumps": 0, "jump_rate": 0.0, "mean_displacement": 0.0}

    jumps = 0
    displacements = []

    for i in range(1, len(points)):
        p1 = points[i - 1]
        p2 = points[i]

        # Only check consecutive or near-consecutive frames
        gap = p2["frame_index"] - p1["frame_index"]
        if gap > 3:
            continue

        d3d = np.sqrt(
            (p2["x"] - p1["x"]) ** 2
            + (p2["y"] - p1["y"]) ** 2
            + (p2["z"] - p1["z"]) ** 2
        )
        displacements.append(d3d)
        if d3d > jump_threshold:
            jumps += 1

    n = len(displacements) or 1
    return {
        "jumps_3d": jumps,
        "jump_rate_3d": jumps / n,
        "mean_displacement_3d": float(np.mean(displacements)) if displacements else 0.0,
        "median_displacement_3d": float(np.median(displacements)) if displacements else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual-camera 3D tracking")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--output", type=str, default=str(OUT_DIR))
    parser.add_argument("--temporal-weights", type=str, default="0.0,0.2,0.3,0.5",
                        help="Comma-separated temporal weights to test")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT for both cameras
    cam66_gt = load_gt(CAM66_LABELS, args.max_frames)
    cam68_gt = load_gt(CAM68_LABELS, args.max_frames)
    both_gt = set(cam66_gt.keys()) & set(cam68_gt.keys())
    log.info("GT: cam66=%d, cam68=%d, both=%d frames", len(cam66_gt), len(cam68_gt), len(both_gt))

    # Run TrackNet inference on both cameras
    cam66_dets = run_tracknet_multi(
        args.model, CAM66_VIDEO, args.max_frames,
        threshold=args.threshold, cam_key="cam66", label="Finetuned",
    )
    cam68_dets = run_tracknet_multi(
        args.model, CAM68_VIDEO, args.max_frames,
        threshold=args.threshold, cam_key="cam68", label="Finetuned",
    )

    # Test different temporal weights
    weights = [float(w) for w in args.temporal_weights.split(",")]
    all_results = {}

    log.info("\n" + "=" * 70)
    log.info("  DUAL-CAMERA 3D TRACKING EVALUATION")
    log.info("=" * 70)

    for tw in weights:
        label = f"tw={tw:.1f}" if tw > 0 else "baseline (no temporal)"
        log.info("\n--- %s ---", label)

        points, m_stats = run_matching(cam66_dets, cam68_dets, temporal_weight=tw, label=label)
        metrics = evaluate_3d_points(points, cam66_gt, cam68_gt)
        stability = evaluate_stability(points)

        log.info("  Recall:       %.1f%% (%d/%d)", metrics["recall"] * 100, metrics["tp"], metrics["total_gt"])
        log.info("  Precision:    %.1f%%", metrics["precision"] * 100)
        log.info("  F1:           %.3f", metrics["f1"])
        log.info("  Mean px err:  %.1f px", metrics["mean_pixel_error"])
        log.info("  Detections:   %d", metrics["total_detections"])
        log.info("  FP:           %d", metrics["fp"])
        log.info("  3D jumps:     %d (%.1f%%)", stability["jumps_3d"], stability["jump_rate_3d"] * 100)
        log.info("  Mean 3D disp: %.2f m", stability["mean_displacement_3d"])
        log.info("  Temporal assists: %d", m_stats.get("temporal_assists", 0))

        all_results[f"tw_{tw}"] = {
            **metrics,
            **stability,
            "matcher_stats": m_stats,
            "temporal_weight": tw,
        }

    # Summary comparison table
    log.info("\n" + "=" * 70)
    log.info("  COMPARISON SUMMARY")
    log.info("=" * 70)
    log.info("  %-25s %8s %8s %8s %8s %8s %8s", "Config", "Recall", "Prec", "F1", "FP", "3D Jump", "Assists")
    log.info("  " + "-" * 73)
    for key, r in all_results.items():
        tw = r["temporal_weight"]
        label = f"tw={tw:.1f}" if tw > 0 else "baseline"
        log.info(
            "  %-25s %7.1f%% %7.1f%% %8.3f %8d %8d %8d",
            label,
            r["recall"] * 100,
            r["precision"] * 100,
            r["f1"],
            r["fp"],
            r["jumps_3d"],
            r["matcher_stats"].get("temporal_assists", 0),
        )

    # Save results
    with open(output_dir / "dual_camera_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("\nResults saved to %s", output_dir / "dual_camera_results.json")


if __name__ == "__main__":
    main()
