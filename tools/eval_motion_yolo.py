"""Evaluate motion-detection + YOLO pipeline for tennis ball detection.

Replaces TrackNet with frame differencing + background subtraction + YOLO verification.
Motion detection naturally filters out dead balls (stationary objects).

Usage:
    python -m tools.eval_motion_yolo
    python -m tools.eval_motion_yolo --cam cam66
    python -m tools.eval_motion_yolo --sweep
    python -m tools.eval_motion_yolo --bg-thresh 30 --min-area 20
"""

import argparse
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

CAMERAS = {
    "cam66": {
        "video": "uploads/cam66_20260307_173403_2min.mp4",
        "annotations": "uploads/cam66_20260307_173403_2min",
        "homography_key": "cam66",
    },
    "cam68": {
        "video": "uploads/cam68_20260307_173403_2min.mp4",
        "annotations": "uploads/cam68_20260307_173403_2min",
        "homography_key": "cam68",
    },
}

VERIFIER_PATH = "model_weight/blob_verifier_yolo.pt"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
DIST_THRESHOLD = 15.0
MAX_FRAMES = 3000
OSD_MASK = (0, 0, 620, 41)  # x0, y0, x1, y1
OUT_DIR = Path("exports/eval_motion_yolo")


@dataclass
class MotionParams:
    bg_diff_thresh: int = 25
    temporal_diff_thresh: int = 20
    morph_kernel: int = 3
    min_area: int = 15
    max_area: int = 5000
    max_blobs: int = 20
    verifier_conf: float = 0.15
    crop_size: int = 128
    temporal_gap: int = 2


# ── GT Loading ──────────────────────────────────────────────────────────────


def load_gt(ann_dir: str) -> dict[int, list[dict]]:
    """Load GT rectangle annotations, return {frame_index: [blob_dicts]}."""
    from tools.prepare_yolo_crops import load_annotations

    all_ann = load_annotations(Path(ann_dir))
    gt = {}
    for fi, blobs in all_ann.items():
        gt_blobs = [b for b in blobs if b["score"] is None and b["label"] == "ball"]
        if gt_blobs:
            gt[fi] = gt_blobs
    return gt


# ── Background Median ───────────────────────────────────────────────────────


def compute_background_median(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    max_samples: int = 200,
) -> np.ndarray:
    """Compute pixel-wise median background in grayscale."""
    total = end_frame - start_frame
    step = max(1, total // max_samples)
    frames = []

    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    log.info("Background median from %d samples", len(frames))
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


# ── Motion Detection ────────────────────────────────────────────────────────


def compute_motion_mask(
    frame_gray: np.ndarray,
    bg_gray: np.ndarray,
    prev_gray: np.ndarray | None,
    params: MotionParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binary motion mask and temporal intensity map.

    Mask: union of bg diff + temporal diff (max recall).
    Intensity: temporal diff only (for ranking — fast motion = match ball).

    Returns:
        mask: Binary motion mask (uint8, 0 or 255)
        intensity: Temporal motion intensity map (uint8) for ranking
    """
    # Background difference — catches balls different from median
    bg_diff = cv2.absdiff(frame_gray, bg_gray)
    _, bg_mask = cv2.threshold(bg_diff, params.bg_diff_thresh, 255, cv2.THRESH_BINARY)

    # Temporal difference — catches actually moving objects
    if prev_gray is not None:
        temporal_diff = cv2.absdiff(frame_gray, prev_gray)
        _, temporal_mask = cv2.threshold(
            temporal_diff, params.temporal_diff_thresh, 255, cv2.THRESH_BINARY
        )
        combined = cv2.bitwise_or(bg_mask, temporal_mask)
        # Intensity from temporal diff ONLY — dead balls have near-zero temporal motion
        intensity = temporal_diff
    else:
        combined = bg_mask
        intensity = bg_diff

    # Morphological close (fill gaps) then open (remove noise)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.morph_kernel, params.morph_kernel)
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Zero out OSD region
    x0, y0, x1, y1 = OSD_MASK
    combined[y0:y1, x0:x1] = 0
    intensity[y0:y1, x0:x1] = 0

    return combined, intensity


def extract_motion_blobs(
    motion_mask: np.ndarray,
    intensity: np.ndarray,
    homography,
    params: MotionParams,
) -> list[dict]:
    """Extract filtered connected components from motion mask."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(motion_mask)

    blobs = []
    for i in range(1, n_labels):  # skip background (label 0)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < params.min_area or area > params.max_area:
            continue

        px, py = float(centroids[i][0]), float(centroids[i][1])

        # Court-X filter
        if homography is not None and not homography.is_in_court_x(px, py):
            continue

        # Compute mean motion intensity within this component
        component_mask = labels == i
        mean_intensity = float(np.mean(intensity[component_mask]))

        blobs.append({
            "pixel_x": px,
            "pixel_y": py,
            "area": area,
            "motion_intensity": mean_intensity,
        })

    # Sort by area ascending (ball is smaller than player)
    blobs.sort(key=lambda b: b["area"])
    return blobs[: params.max_blobs]


# ── YOLO Verification ───────────────────────────────────────────────────────


def verify_motion_blobs(
    frame: np.ndarray,
    blobs: list[dict],
    verifier,
    params: MotionParams,
) -> list[dict]:
    """Run YOLO on motion blob crops, return verified detections."""
    if not blobs:
        return []

    from app.pipeline.blob_verifier import extract_crops

    crops = extract_crops(frame, blobs, params.crop_size)
    detections = verifier.detect_crops(crops)

    half = params.crop_size // 2
    verified = []
    for blob, det in zip(blobs, detections):
        if det is None or det["yolo_conf"] < params.verifier_conf:
            continue
        refined_px = blob["pixel_x"] + det["crop_cx"] - half
        refined_py = blob["pixel_y"] + det["crop_cy"] - half
        verified.append(
            {
                "pixel_x": refined_px,
                "pixel_y": refined_py,
                "blob_cx": blob["pixel_x"],
                "blob_cy": blob["pixel_y"],
                "yolo_conf": det["yolo_conf"],
                "area": blob["area"],
                "motion_intensity": blob.get("motion_intensity", 0),
            }
        )

    # Sort by motion_intensity * yolo_conf / log(area+1)
    # Fast-moving + high-confidence + small = most likely match ball
    verified.sort(
        key=lambda b: b["motion_intensity"] * b["yolo_conf"] / np.log(b["area"] + 1),
        reverse=True,
    )
    return verified


class StaticBlobFilter:
    """Track detection positions over time and suppress persistent (dead ball) locations."""

    def __init__(self, grid_size: int = 40, min_count: int = 30, decay: float = 0.98):
        self.grid_size = grid_size
        self.min_count = min_count
        self.decay = decay
        self._counts: dict[tuple[int, int], float] = {}

    def _grid_key(self, px: float, py: float) -> tuple[int, int]:
        return (int(px / self.grid_size), int(py / self.grid_size))

    def update_and_filter(self, blobs: list[dict]) -> list[dict]:
        """Update grid counts and remove blobs at persistent locations."""
        # Decay all counts
        for key in self._counts:
            self._counts[key] *= self.decay

        # Update counts for current blobs
        for b in blobs:
            key = self._grid_key(b["pixel_x"], b["pixel_y"])
            self._counts[key] = self._counts.get(key, 0.0) + 1.0

        # Filter: remove blobs at persistent positions
        filtered = []
        for b in blobs:
            key = self._grid_key(b["pixel_x"], b["pixel_y"])
            if self._counts.get(key, 0) < self.min_count:
                filtered.append(b)

        return filtered


def _select_best_blob(
    verified: list[dict],
    prev_det: dict | None,
    fi: int,
    prev_fi: int,
    max_speed_px: float = 80.0,
) -> dict:
    """Select best blob using temporal continuity + confidence.

    If previous detection exists and gap is small, prefer the blob closest
    to predicted position. Otherwise fall back to confidence/area ranking.
    """
    if not verified:
        return verified[0]

    # No history or large gap → use ranking (yolo_conf / log(area))
    if prev_det is None or (fi - prev_fi) > 5:
        return verified[0]

    # Predict position: assume constant velocity isn't available,
    # just use proximity to previous detection
    gap = fi - prev_fi
    max_dist = max_speed_px * gap  # max reasonable displacement

    # Score each blob: proximity to prev + yolo_conf
    best = None
    best_score = -1.0
    for v in verified:
        dist = float(
            np.hypot(
                v["pixel_x"] - prev_det["pixel_x"],
                v["pixel_y"] - prev_det["pixel_y"],
            )
        )
        if dist > max_dist:
            proximity_score = 0.0
        else:
            proximity_score = 1.0 - dist / max_dist

        # Combined: proximity dominates, confidence breaks ties
        score = proximity_score * 0.7 + v["yolo_conf"] * 0.3
        if score > best_score:
            best_score = score
            best = v

    return best if best is not None else verified[0]


# ── Main Pipeline ───────────────────────────────────────────────────────────


def run_motion_yolo_pipeline(
    cam_name: str,
    cam_cfg: dict,
    params: MotionParams,
    max_frames: int = MAX_FRAMES,
    return_all_candidates: bool = False,
) -> dict[int, dict] | tuple[dict[int, dict], dict[int, list[dict]]]:
    """Process video with motion + YOLO, return {frame_index: detection_dict}.

    If return_all_candidates=True, also returns {frame_index: [all_verified_blobs]}.
    """
    from app.pipeline.blob_verifier import BlobVerifier
    from app.pipeline.homography import HomographyTransformer

    cap = cv2.VideoCapture(cam_cfg["video"])
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    # Background median
    log.info("%s: computing background median...", cam_name)
    bg_gray = compute_background_median(cap, 0, total)

    # Homography for court-X filtering
    homography = HomographyTransformer(HOMOGRAPHY_PATH, cam_cfg["homography_key"])

    # YOLO verifier
    verifier = BlobVerifier(model_path=VERIFIER_PATH, crop_size=params.crop_size, conf=params.verifier_conf)

    # Frame buffer for temporal differencing
    gray_buffer: deque[np.ndarray] = deque(maxlen=params.temporal_gap + 1)

    results: dict[int, dict] = {}
    all_candidates: dict[int, list[dict]] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    stats = {"total_motion_blobs": 0, "total_verified": 0, "empty_frames": 0}
    static_filter = StaticBlobFilter(grid_size=30, min_count=20, decay=0.97)

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_buffer.append(frame_gray)

        # Temporal reference: frame[t - temporal_gap]
        prev_gray = gray_buffer[0] if len(gray_buffer) > params.temporal_gap else None

        # Motion mask + intensity
        motion_mask, motion_intensity = compute_motion_mask(frame_gray, bg_gray, prev_gray, params)

        # Extract motion blobs
        motion_blobs = extract_motion_blobs(motion_mask, motion_intensity, homography, params)
        stats["total_motion_blobs"] += len(motion_blobs)

        if not motion_blobs:
            stats["empty_frames"] += 1
            continue

        # YOLO verification
        verified = verify_motion_blobs(frame, motion_blobs, verifier, params)

        # Filter out persistent (dead ball) positions
        verified = static_filter.update_and_filter(verified)
        stats["total_verified"] += len(verified)

        if not verified:
            stats["empty_frames"] += 1
            continue

        # Store all candidates with world coords
        if return_all_candidates:
            cands = []
            for v in verified:
                wx, wy = homography.pixel_to_world(v["pixel_x"], v["pixel_y"])
                cands.append({**v, "world_x": wx, "world_y": wy})
            all_candidates[fi] = cands

        # Take top-1 detection (ranked by yolo_conf / log(area))
        top = verified[0]
        wx, wy = homography.pixel_to_world(top["pixel_x"], top["pixel_y"])

        results[fi] = {
            "pixel_x": top["pixel_x"],
            "pixel_y": top["pixel_y"],
            "world_x": wx,
            "world_y": wy,
            "confidence": float(top["area"]),
            "yolo_conf": top["yolo_conf"],
            "n_blobs_before": len(motion_blobs),
            "n_blobs_after": len(verified),
            "used_verifier": True,
        }

        if (fi + 1) % 500 == 0:
            log.info("  %s: %d/%d frames, %d detections", cam_name, fi + 1, total, len(results))

    cap.release()

    processed = fi + 1 if total > 0 else 0
    avg_blobs = stats["total_motion_blobs"] / max(processed, 1)
    avg_verified = stats["total_verified"] / max(processed, 1)
    log.info(
        "%s done: %d detections / %d frames (%.1f%%), avg motion=%.1f, avg verified=%.1f",
        cam_name, len(results), processed,
        100 * len(results) / max(processed, 1),
        avg_blobs, avg_verified,
    )

    if return_all_candidates:
        return results, all_candidates
    return results


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_detections(
    detections: dict[int, dict],
    gt: dict[int, list[dict]],
    dist_threshold: float = DIST_THRESHOLD,
) -> dict:
    """Compare detections vs GT, return metrics."""
    tp = 0
    fp = 0
    fn = 0
    errors = []

    for fi, gt_blobs in gt.items():
        det = detections.get(fi)
        if det is None:
            fn += 1
            continue

        min_dist = min(
            float(
                np.hypot(
                    det["pixel_x"] - gb["pixel_x"],
                    det["pixel_y"] - gb["pixel_y"],
                )
            )
            for gb in gt_blobs
        )

        if min_dist <= dist_threshold:
            tp += 1
            errors.append(min_dist)
        else:
            fp += 1

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    return {
        "gt_frames": len(gt),
        "detected_frames": len(detections),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "mean_error": float(np.mean(errors)) if errors else 0.0,
        "median_error": float(np.median(errors)) if errors else 0.0,
    }


def print_summary(cam_name: str, metrics: dict, total_frames: int) -> None:
    """Print evaluation summary."""
    print(f"\n--- {cam_name} ---")
    print(f"  GT frames:        {metrics['gt_frames']}")
    print(f"  Detected frames:  {metrics['detected_frames']} / {total_frames} "
          f"({100 * metrics['detected_frames'] / max(total_frames, 1):.1f}%)")
    print(f"  Recall:           {100 * metrics['recall']:.1f}% (TP={metrics['tp']}, FN={metrics['fn']})")
    print(f"  Precision:        {100 * metrics['precision']:.1f}% (TP={metrics['tp']}, FP={metrics['fp']})")
    print(f"  Mean error:       {metrics['mean_error']:.1f}px")
    print(f"  Median error:     {metrics['median_error']:.1f}px")


# ── Parameter Sweep ─────────────────────────────────────────────────────────


def run_sweep(cam_cfg: dict, gt: dict[int, list[dict]]) -> None:
    """Sweep key parameters on cam66, 1000 frames."""
    print("\n" + "=" * 60)
    print("  PARAMETER SWEEP (cam66, 1000 frames)")
    print("=" * 60)

    sweep_grid = [
        {"bg_diff_thresh": bg, "temporal_diff_thresh": td, "min_area": ma}
        for bg in [15, 20, 25, 30]
        for td in [15, 20, 25]
        for ma in [10, 15, 25]
    ]

    best_recall = 0.0
    best_params = None

    for i, overrides in enumerate(sweep_grid):
        params = MotionParams(**overrides)
        detections = run_motion_yolo_pipeline("cam66", cam_cfg, params, max_frames=1000)
        metrics = evaluate_detections(detections, gt)

        tag = ""
        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best_params = overrides
            tag = " *** BEST"

        print(
            f"  [{i + 1:2d}/{len(sweep_grid)}] "
            f"bg={overrides['bg_diff_thresh']:2d} td={overrides['temporal_diff_thresh']:2d} "
            f"min_area={overrides['min_area']:2d} → "
            f"recall={100 * metrics['recall']:.1f}% "
            f"prec={100 * metrics['precision']:.1f}% "
            f"err={metrics['mean_error']:.1f}px{tag}"
        )

    print(f"\n  Best: recall={100 * best_recall:.1f}%, params={best_params}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Motion + YOLO evaluation")
    parser.add_argument("--cam", choices=["cam66", "cam68", "both"], default="both")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--bg-thresh", type=int, default=25)
    parser.add_argument("--temporal-thresh", type=int, default=20)
    parser.add_argument("--min-area", type=int, default=15)
    parser.add_argument("--max-area", type=int, default=5000)
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    args = parser.parse_args()

    params = MotionParams(
        bg_diff_thresh=args.bg_thresh,
        temporal_diff_thresh=args.temporal_thresh,
        min_area=args.min_area,
        max_area=args.max_area,
        verifier_conf=args.yolo_conf,
    )

    cams = ["cam66", "cam68"] if args.cam == "both" else [args.cam]

    # Sweep mode
    if args.sweep:
        gt = load_gt(CAMERAS["cam66"]["annotations"])
        run_sweep(CAMERAS["cam66"], gt)
        return

    # Normal evaluation
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    print("\n" + "=" * 60)
    print("  MOTION + YOLO EVALUATION")
    print("=" * 60)

    for cam_name in cams:
        cam_cfg = CAMERAS[cam_name]
        gt = load_gt(cam_cfg["annotations"])
        log.info("%s: %d GT frames loaded", cam_name, len(gt))

        detections = run_motion_yolo_pipeline(cam_name, cam_cfg, params, args.max_frames)

        # Save detections
        det_path = OUT_DIR / f"{cam_name}_detections.json"
        with open(det_path, "w") as f:
            json.dump({str(k): v for k, v in detections.items()}, f, indent=1)
        log.info("Saved %s", det_path)

        # Evaluate
        metrics = evaluate_detections(detections, gt)
        all_metrics[cam_name] = metrics
        print_summary(cam_name, metrics, args.max_frames)

    # Save metrics
    metrics_path = OUT_DIR / "eval_results.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
