"""Evaluate TrackNet + YOLO blob verifier: recall, YOLO impact, stability, dead ball.

Runs TrackNet inference once, stores all intermediate blob data, then runs four tests:
  Test 1: TrackNet raw recall at top-1/3/5
  Test 2: YOLO verification impact on multi-blob frames
  Test 3: Stability / drift analysis (frame-to-frame jumps)
  Test 4: Dead ball contamination (stationary detection windows)

Usage:
    python -m tools.eval_tracknet_yolo
"""

import json
import logging
import sys
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

MODEL_PATH = "model_weight/TrackNet_best.pt"
VERIFIER_PATH = "model_weight/blob_verifier_yolo.pt"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
INPUT_SIZE = (288, 512)
FRAMES_IN = 8
FRAMES_OUT = 8
THRESHOLD = 0.35
DEVICE = "cuda"
HEATMAP_MASK = [(0, 0, 620, 40)]
VERIFIER_CONF = 0.15
VERIFIER_CROP = 128
MAX_FRAMES = 3000

DIST_THRESHOLD = 15.0       # pixels — GT match threshold
JUMP_THRESHOLD = 100.0      # pixels — frame-to-frame jump
STATIONARY_WINDOW = 5       # consecutive frames
STATIONARY_MAX_MOVE = 8.0   # pixels — max spread for "stationary"

OUT_DIR = Path("exports/eval_tracknet_yolo")


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


def run_inference(cam_name: str, cam_cfg: dict) -> dict[int, dict]:
    """Run TrackNet + YOLO on video, return per-frame intermediate data.

    Each frame stores:
        raw_blobs: all blobs from process_heatmap_multi (max_blobs=5)
        yolo_blobs: blobs after verify_blobs (only for multi-blob frames)
        raw_top1: raw_blobs[0] position
        yolo_top1: yolo_blobs[0] position (or raw_top1 if single blob)
        used_verifier: whether YOLO was triggered
        n_raw: number of raw blobs
    """
    from app.pipeline.inference import create_detector
    from app.pipeline.postprocess import BallTracker
    from app.pipeline.blob_verifier import BlobVerifier, verify_blobs

    log.info("=== Inference: %s ===", cam_name)

    detector = create_detector(MODEL_PATH, INPUT_SIZE, FRAMES_IN, FRAMES_OUT, DEVICE)
    tracker = BallTracker(
        original_size=(1920, 1080),
        threshold=THRESHOLD,
        heatmap_mask=HEATMAP_MASK,
    )
    verifier = BlobVerifier(
        model_path=VERIFIER_PATH,
        crop_size=VERIFIER_CROP,
        conf=VERIFIER_CONF,
        device=DEVICE,
    )

    cap = cv2.VideoCapture(cam_cfg["video"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {cam_cfg['video']}")

    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)

    # Compute background median
    if hasattr(detector, "compute_video_median"):
        log.info("Computing video median...")
        detector.compute_video_median(cap, 0, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_buffer = []  # OSD-masked frames → TrackNet input
    raw_buffer = []    # original frames → YOLO crop source
    results = {}
    processed = 0

    while processed < total:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1
        raw_buffer.append(frame)

        masked = frame.copy()
        masked[0:41, 0:603] = 0
        frame_buffer.append(masked)

        if len(frame_buffer) < FRAMES_IN:
            continue

        heatmaps = detector.infer(frame_buffer)

        for i in range(min(FRAMES_OUT, len(heatmaps))):
            fi = processed - FRAMES_OUT + i
            raw_blobs = tracker.process_heatmap_multi(heatmaps[i], max_blobs=5)
            if not raw_blobs:
                continue

            n_raw = len(raw_blobs)
            raw_top1 = {"pixel_x": raw_blobs[0]["pixel_x"],
                        "pixel_y": raw_blobs[0]["pixel_y"],
                        "blob_sum": raw_blobs[0]["blob_sum"]}

            # YOLO verification on EVERY frame (not just multi-blob)
            # Even single-blob frames need YOLO to confirm it's actually a ball
            from app.pipeline.blob_verifier import extract_crops
            crops = extract_crops(raw_buffer[i], raw_blobs, verifier.crop_size)
            detections = verifier.detect_crops(crops)

            yolo_blobs = []
            for blob, det in zip(raw_blobs, detections):
                blob_copy = dict(blob)
                if det is not None and det["yolo_conf"] >= VERIFIER_CONF:
                    blob_copy["yolo_conf"] = det["yolo_conf"]
                    half = verifier.crop_size // 2
                    blob_copy["refined_pixel_x"] = blob["pixel_x"] + det["crop_cx"] - half
                    blob_copy["refined_pixel_y"] = blob["pixel_y"] + det["crop_cy"] - half
                    yolo_blobs.append(blob_copy)
                else:
                    blob_copy["yolo_conf"] = 0.0
                    blob_copy["refined_pixel_x"] = blob["pixel_x"]
                    blob_copy["refined_pixel_y"] = blob["pixel_y"]

            # Sort by yolo_conf * blob_sum
            yolo_blobs.sort(key=lambda b: b["yolo_conf"] * b["blob_sum"], reverse=True)

            # Fallback: if YOLO rejects everything, keep raw top-1
            if not yolo_blobs:
                fallback = dict(raw_blobs[0])
                fallback["yolo_conf"] = 0.0
                fallback["refined_pixel_x"] = raw_blobs[0]["pixel_x"]
                fallback["refined_pixel_y"] = raw_blobs[0]["pixel_y"]
                yolo_blobs = [fallback]

            used_verifier = True
            yolo_top1 = {"pixel_x": yolo_blobs[0]["pixel_x"],
                         "pixel_y": yolo_blobs[0]["pixel_y"],
                         "blob_sum": yolo_blobs[0]["blob_sum"],
                         "yolo_conf": yolo_blobs[0].get("yolo_conf", 0)}

            results[fi] = {
                "raw_blobs": [{"pixel_x": b["pixel_x"], "pixel_y": b["pixel_y"],
                               "blob_sum": b["blob_sum"]} for b in raw_blobs],
                "n_raw": n_raw,
                "raw_top1": raw_top1,
                "yolo_top1": yolo_top1,
                "used_verifier": used_verifier,
            }

        frame_buffer.clear()
        raw_buffer.clear()

        if processed % 500 == 0:
            log.info("  %s: %d/%d", cam_name, processed, total)

    cap.release()
    log.info("%s: %d frames with detections", cam_name, len(results))
    return results


def _dist(a: dict, b: dict) -> float:
    """Pixel distance between two points."""
    return float(np.hypot(a["pixel_x"] - b["pixel_x"],
                          a["pixel_y"] - b["pixel_y"]))


def _min_gt_dist(det: dict, gt_blobs: list[dict]) -> float:
    """Minimum distance from detection to any GT blob."""
    return min(_dist(det, gb) for gb in gt_blobs)


def test1_tracknet_raw_recall(data: dict[int, dict], gt: dict[int, list[dict]]) -> dict:
    """Test 1: TrackNet raw recall at top-1/3/5."""
    gt_total = len(gt)
    no_det = 0
    hits = {1: 0, 3: 0, 5: 0}
    top1_errors = []
    multi_blob_frames = 0

    for fi, gt_blobs in gt.items():
        if fi not in data:
            no_det += 1
            continue

        frame_data = data[fi]
        raw_blobs = frame_data["raw_blobs"]
        if frame_data["n_raw"] > 1:
            multi_blob_frames += 1

        for n in [1, 3, 5]:
            top_n = raw_blobs[:n]
            min_d = min(_dist(b, gb) for b in top_n for gb in gt_blobs)
            if min_d <= DIST_THRESHOLD:
                hits[n] += 1

        # Top-1 error
        top1_d = min(_dist(raw_blobs[0], gb) for gb in gt_blobs)
        top1_errors.append(top1_d)

    total_with_det = gt_total - no_det
    return {
        "gt_frames": gt_total,
        "no_detection": no_det,
        "recall_top1": hits[1] / gt_total if gt_total else 0,
        "recall_top3": hits[3] / gt_total if gt_total else 0,
        "recall_top5": hits[5] / gt_total if gt_total else 0,
        "mean_top1_error": float(np.mean(top1_errors)) if top1_errors else 0,
        "median_top1_error": float(np.median(top1_errors)) if top1_errors else 0,
        "multi_blob_rate": multi_blob_frames / total_with_det if total_with_det else 0,
    }


def test2_yolo_impact(data: dict[int, dict], gt: dict[int, list[dict]]) -> dict:
    """Test 2: YOLO verification impact on multi-blob GT frames."""
    improved = 0
    degraded = 0
    no_change = 0
    rescued = 0  # wrong→correct
    killed = 0   # correct→wrong
    raw_hits = 0
    yolo_hits = 0
    total = 0

    for fi, gt_blobs in gt.items():
        if fi not in data:
            continue
        fd = data[fi]
        if not fd["used_verifier"]:
            continue

        total += 1
        raw_d = _min_gt_dist(fd["raw_top1"], gt_blobs)
        yolo_d = _min_gt_dist(fd["yolo_top1"], gt_blobs)

        raw_ok = raw_d <= DIST_THRESHOLD
        yolo_ok = yolo_d <= DIST_THRESHOLD

        if raw_ok:
            raw_hits += 1
        if yolo_ok:
            yolo_hits += 1

        if yolo_d < raw_d - 1.0:
            improved += 1
        elif yolo_d > raw_d + 1.0:
            degraded += 1
        else:
            no_change += 1

        if not raw_ok and yolo_ok:
            rescued += 1
        if raw_ok and not yolo_ok:
            killed += 1

    return {
        "multi_blob_gt_frames": total,
        "improved": improved,
        "degraded": degraded,
        "no_change": no_change,
        "rescued": rescued,
        "killed": killed,
        "raw_recall_multi": raw_hits / total if total else 0,
        "yolo_recall_multi": yolo_hits / total if total else 0,
    }


def test3_stability(data: dict[int, dict], gt: dict[int, list[dict]]) -> dict:
    """Test 3: Frame-to-frame stability and drift analysis."""
    sorted_frames = sorted(data.keys())
    total_pairs = 0
    jumps = 0
    real_jumps = 0
    drift_events = 0
    unverifiable = 0
    displacements = []
    drift_frame_list = []

    for idx in range(1, len(sorted_frames)):
        fi_prev = sorted_frames[idx - 1]
        fi_curr = sorted_frames[idx]

        # Skip gaps > 2 frames
        if fi_curr - fi_prev > 2:
            continue

        total_pairs += 1
        prev_pos = data[fi_prev]["yolo_top1"]
        curr_pos = data[fi_curr]["yolo_top1"]
        disp = _dist(prev_pos, curr_pos)
        displacements.append(disp)

        if disp > JUMP_THRESHOLD:
            jumps += 1
            # Check GT to classify jump
            if fi_prev in gt and fi_curr in gt:
                gt_disp = min(
                    _dist(gb_prev, gb_curr)
                    for gb_prev in gt[fi_prev]
                    for gb_curr in gt[fi_curr]
                )
                if gt_disp > JUMP_THRESHOLD * 0.5:
                    real_jumps += 1
                else:
                    drift_events += 1
                    drift_frame_list.append(fi_curr)
            else:
                unverifiable += 1

    return {
        "total_pairs": total_pairs,
        "jumps": jumps,
        "real_jumps": real_jumps,
        "drift_events": drift_events,
        "unverifiable_jumps": unverifiable,
        "stability_rate": 1 - drift_events / total_pairs if total_pairs else 1,
        "mean_displacement": float(np.mean(displacements)) if displacements else 0,
        "median_displacement": float(np.median(displacements)) if displacements else 0,
        "drift_frames": drift_frame_list[:20],  # first 20 for debugging
    }


def test4_dead_ball(data: dict[int, dict], gt: dict[int, list[dict]]) -> dict:
    """Test 4: Dead ball contamination — stationary detection windows."""
    sorted_frames = sorted(data.keys())
    stationary_windows = 0
    stationary_frame_set = set()
    dead_ball_frames = set()
    correct_stationary = set()

    for start in range(len(sorted_frames) - STATIONARY_WINDOW + 1):
        window = sorted_frames[start:start + STATIONARY_WINDOW]

        # Check frames are actually consecutive (allow gap of 1)
        if window[-1] - window[0] > STATIONARY_WINDOW + 1:
            continue

        # Compute max pairwise spread
        positions = [data[fi]["yolo_top1"] for fi in window]
        max_spread = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = _dist(positions[i], positions[j])
                max_spread = max(max_spread, d)

        if max_spread < STATIONARY_MAX_MOVE:
            stationary_windows += 1
            for fi in window:
                stationary_frame_set.add(fi)
                if fi in gt:
                    d_to_gt = _min_gt_dist(data[fi]["yolo_top1"], gt[fi])
                    if d_to_gt > DIST_THRESHOLD:
                        dead_ball_frames.add(fi)
                    else:
                        correct_stationary.add(fi)

    total_det = len(data)
    return {
        "stationary_windows": stationary_windows,
        "stationary_frames": len(stationary_frame_set),
        "dead_ball_frames": len(dead_ball_frames),
        "correct_stationary": len(correct_stationary),
        "dead_ball_rate": len(dead_ball_frames) / total_det if total_det else 0,
    }


def run_camera_eval(cam_name: str, cam_cfg: dict) -> dict:
    """Run all four tests for one camera."""
    gt = load_gt(cam_cfg["annotations"])
    data = run_inference(cam_name, cam_cfg)

    t1 = test1_tracknet_raw_recall(data, gt)
    t2 = test2_yolo_impact(data, gt)
    t3 = test3_stability(data, gt)
    t4 = test4_dead_ball(data, gt)

    return {
        "test1_raw_recall": t1,
        "test2_yolo_impact": t2,
        "test3_stability": t3,
        "test4_dead_ball": t4,
    }


def print_summary(results: dict):
    """Print formatted console summary."""
    print("\n" + "=" * 65)
    print("  TRACKNET + YOLO EVALUATION")
    print("=" * 65)

    for cam, r in results.items():
        t1 = r["test1_raw_recall"]
        t2 = r["test2_yolo_impact"]
        t3 = r["test3_stability"]
        t4 = r["test4_dead_ball"]

        print(f"\n--- {cam} ---")

        print(f"\n  Test 1: TrackNet Raw Recall (threshold={THRESHOLD}, dist={DIST_THRESHOLD}px)")
        print(f"    GT frames:       {t1['gt_frames']}")
        print(f"    No detection:    {t1['no_detection']} ({t1['no_detection']/t1['gt_frames']*100:.1f}%)")
        print(f"    Recall @top-1:   {t1['recall_top1']:.1%}  (mean={t1['mean_top1_error']:.1f}px, median={t1['median_top1_error']:.1f}px)")
        print(f"    Recall @top-3:   {t1['recall_top3']:.1%}")
        print(f"    Recall @top-5:   {t1['recall_top5']:.1%}")
        print(f"    Multi-blob rate: {t1['multi_blob_rate']:.1%}")

        print(f"\n  Test 2: YOLO Verification Impact (multi-blob GT frames)")
        print(f"    Frames:          {t2['multi_blob_gt_frames']}")
        print(f"    Improved:        {t2['improved']} ({t2['improved']/max(t2['multi_blob_gt_frames'],1)*100:.1f}%)")
        print(f"    Degraded:        {t2['degraded']} ({t2['degraded']/max(t2['multi_blob_gt_frames'],1)*100:.1f}%)")
        print(f"    No change:       {t2['no_change']}")
        print(f"    Rescued:         {t2['rescued']}  (wrong->correct)")
        print(f"    Killed:          {t2['killed']}  (correct->wrong)")
        print(f"    Recall: raw={t2['raw_recall_multi']:.1%} -> yolo={t2['yolo_recall_multi']:.1%}")

        print(f"\n  Test 3: Stability / Drift")
        print(f"    Consecutive pairs: {t3['total_pairs']}")
        print(f"    Jumps (>{JUMP_THRESHOLD}px): {t3['jumps']}")
        print(f"      Real jumps:    {t3['real_jumps']}")
        print(f"      Drift events:  {t3['drift_events']}")
        print(f"      Unverifiable:  {t3['unverifiable_jumps']}")
        print(f"    Stability rate:  {t3['stability_rate']:.1%}")
        print(f"    Displacement:    mean={t3['mean_displacement']:.1f}px, median={t3['median_displacement']:.1f}px")
        if t3["drift_frames"]:
            print(f"    Drift frames:    {t3['drift_frames'][:10]}")

        print(f"\n  Test 4: Dead Ball Contamination")
        print(f"    Stationary windows ({STATIONARY_WINDOW}f, <{STATIONARY_MAX_MOVE}px): {t4['stationary_windows']}")
        print(f"    Stationary frames: {t4['stationary_frames']}")
        print(f"    Dead ball frames:  {t4['dead_ball_frames']} ({t4['dead_ball_rate']:.1%})")
        print(f"    Correct stationary: {t4['correct_stationary']}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for cam_name, cam_cfg in CAMERAS.items():
        all_results[cam_name] = run_camera_eval(cam_name, cam_cfg)

    # Save JSON (convert drift_frames lists for serialization)
    with open(OUT_DIR / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved %s", OUT_DIR / "eval_results.json")

    print_summary(all_results)


if __name__ == "__main__":
    main()
