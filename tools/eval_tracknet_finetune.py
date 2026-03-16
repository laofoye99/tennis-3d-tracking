"""Evaluate fine-tuned TrackNet vs original, side-by-side comparison.

Runs both models on in-house test data and compares:
  - Recall@15px (ball detection rate)
  - Dead ball false positive rate
  - Mean localization error
  - Frame-to-frame stability

Usage:
    python -m tools.eval_tracknet_finetune
    python -m tools.eval_tracknet_finetune --finetuned model_weight/tracknet_finetuned/best.pt
    python -m tools.eval_tracknet_finetune --render
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Defaults
ORIGINAL_MODEL = "model_weight/TrackNet_best.pt"
FINETUNED_MODEL = "model_weight/tracknet_finetuned/best.pt"
CAM66_VIDEO = "uploads/cam66_20260307_173403_2min.mp4"
CAM66_LABELS = "uploads/cam66_20260307_173403_2min"
OUT_DIR = Path("exports/eval_finetune")
DIST_THRESHOLD = 15.0
MAX_FRAMES = 900


@dataclass
class Detection:
    frame: int
    pixel_x: float
    pixel_y: float
    confidence: float


def load_gt(label_dir: str, max_frame: int = 9999) -> dict[int, tuple[float, float]]:
    """Load GT ball positions from LabelMe annotations."""
    from tools.prepare_tracknet_data import load_gt_annotations
    gt_raw = load_gt_annotations(Path(label_dir))
    return {fi: (e["pixel_x"], e["pixel_y"]) for fi, e in gt_raw.items() if fi <= max_frame}


def run_tracknet_inference(
    model_path: str,
    video_path: str,
    max_frames: int,
    threshold: float = 0.35,
    label: str = "model",
) -> list[Detection]:
    """Run TrackNet inference on video and return per-frame detections."""
    from app.pipeline.inference import TrackNetDetector
    from app.pipeline.postprocess import BallTracker

    log.info("Running %s: %s", label, model_path)

    detector = TrackNetDetector(
        model_path=model_path,
        input_size=(288, 512),
        frames_in=8,
        frames_out=8,
        device="cuda",
        bg_mode="concat",
    )
    tracker = BallTracker(original_size=(1920, 1080), threshold=threshold)

    cap = cv2.VideoCapture(video_path)
    total = min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Compute background median
    detector.compute_video_median(cap, 0, total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detections = []
    raw_buffer = []  # Store raw BGR frames for infer()
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

        # Process last heatmap
        h_idx = seq_len - 1
        if h_idx < len(heatmaps):
            hm = heatmaps[h_idx]
            orig_h, orig_w = frame.shape[:2]
            hm_full = cv2.resize(hm, (orig_w, orig_h))

            # OSD mask
            hm_full[:41, :603] = 0

            result = tracker.process_heatmap(hm_full)
            if result is not None:
                px, py, conf = result
                detections.append(Detection(fi, float(px), float(py), float(conf)))

    cap.release()
    log.info("  %s: %d detections in %d frames", label, len(detections), total)
    return detections


def evaluate_detections(
    detections: list[Detection],
    gt: dict[int, tuple[float, float]],
    dist_threshold: float = DIST_THRESHOLD,
) -> dict:
    """Evaluate detections against GT."""
    det_map = {d.frame: d for d in detections}

    tp = 0
    fn = 0
    errors = []

    for fi, (gx, gy) in gt.items():
        if fi in det_map:
            d = det_map[fi]
            dist = np.hypot(d.pixel_x - gx, d.pixel_y - gy)
            if dist < dist_threshold:
                tp += 1
                errors.append(dist)
            else:
                fn += 1
        else:
            fn += 1

    # FP: detections where no GT exists
    gt_frames = set(gt.keys())
    fp = sum(1 for d in detections if d.frame not in gt_frames)

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    mean_err = np.mean(errors) if errors else 0.0
    median_err = np.median(errors) if errors else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / max(recall + precision, 1e-7),
        "tp": tp, "fp": fp, "fn": fn,
        "mean_error": float(mean_err),
        "median_error": float(median_err),
        "total_detections": len(detections),
        "total_gt": len(gt),
    }


def evaluate_dead_ball(
    detections: list[Detection],
    gt: dict[int, tuple[float, float]],
    static_window: int = 5,
    static_threshold: float = 5.0,
) -> dict:
    """Evaluate dead ball false positive rate.

    Dead ball = detection exists for 5+ consecutive frames in nearly the same position
    with no GT annotation.
    """
    det_map = {d.frame: d for d in detections}
    sorted_frames = sorted(det_map.keys())

    dead_ball_count = 0
    dead_ball_sequences = 0
    non_gt_detections = [f for f in sorted_frames if f not in gt]

    # Find static sequences
    i = 0
    while i < len(non_gt_detections):
        window = [non_gt_detections[i]]
        j = i + 1
        while j < len(non_gt_detections):
            if non_gt_detections[j] - window[-1] <= 2:
                window.append(non_gt_detections[j])
                j += 1
            else:
                break

        if len(window) >= static_window:
            positions = np.array([[det_map[f].pixel_x, det_map[f].pixel_y] for f in window])
            std_x, std_y = np.std(positions[:, 0]), np.std(positions[:, 1])
            if std_x < static_threshold and std_y < static_threshold:
                dead_ball_count += len(window)
                dead_ball_sequences += 1

        i = j if j > i + 1 else i + 1

    total_non_gt = len(non_gt_detections)
    dead_ball_rate = dead_ball_count / max(total_non_gt, 1)

    return {
        "dead_ball_frames": dead_ball_count,
        "dead_ball_sequences": dead_ball_sequences,
        "non_gt_detections": total_non_gt,
        "dead_ball_rate": dead_ball_rate,
    }


def evaluate_stability(detections: list[Detection], max_jump: float = 100.0) -> dict:
    """Evaluate frame-to-frame detection stability.

    Counts "jumps" where detection moves > max_jump pixels between consecutive frames.
    """
    det_map = {d.frame: d for d in detections}
    sorted_frames = sorted(det_map.keys())

    jumps = 0
    total_pairs = 0
    frame_dists = []

    for i in range(1, len(sorted_frames)):
        f_prev = sorted_frames[i - 1]
        f_curr = sorted_frames[i]

        if f_curr - f_prev > 2:
            continue

        d_prev = det_map[f_prev]
        d_curr = det_map[f_curr]
        dist = np.hypot(d_curr.pixel_x - d_prev.pixel_x, d_curr.pixel_y - d_prev.pixel_y)
        frame_dists.append(dist)
        total_pairs += 1

        if dist > max_jump:
            jumps += 1

    return {
        "jumps": jumps,
        "total_pairs": total_pairs,
        "jump_rate": jumps / max(total_pairs, 1),
        "mean_frame_dist": float(np.mean(frame_dists)) if frame_dists else 0.0,
        "median_frame_dist": float(np.median(frame_dists)) if frame_dists else 0.0,
    }


def threshold_sweep(
    detections: list[Detection],
    video_path: str,
    model_path: str,
    gt: dict[int, tuple[float, float]],
    thresholds: list[float],
    max_frames: int = MAX_FRAMES,
) -> list[dict]:
    """Run evaluation at multiple heatmap thresholds."""
    results = []
    for thresh in thresholds:
        dets = run_tracknet_inference(model_path, video_path, max_frames, threshold=thresh,
                                      label=f"thresh={thresh}")
        metrics = evaluate_detections(dets, gt)
        metrics["threshold"] = thresh
        results.append(metrics)
        log.info("  thresh=%.2f: recall=%.3f prec=%.3f f1=%.3f",
                 thresh, metrics["recall"], metrics["precision"], metrics["f1"])
    return results


def render_comparison_video(
    video_path: str,
    original_dets: list[Detection],
    finetuned_dets: list[Detection],
    gt: dict[int, tuple[float, float]],
    output_path: Path,
    max_frames: int,
) -> None:
    """Render comparison video: red=original, blue=finetuned, green=GT."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    orig_map = {d.frame: d for d in original_dets}
    fine_map = {d.frame: d for d in finetuned_dets}

    for fi in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # GT (green circle, hollow)
        if fi in gt:
            gx, gy = int(gt[fi][0]), int(gt[fi][1])
            cv2.circle(frame, (gx, gy), 10, (0, 255, 0), 2)
            cv2.putText(frame, "GT", (gx + 12, gy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Original (red circle, filled)
        if fi in orig_map:
            ox, oy = int(orig_map[fi].pixel_x), int(orig_map[fi].pixel_y)
            cv2.circle(frame, (ox, oy), 6, (0, 0, 255), -1)
            cv2.putText(frame, "Orig", (ox + 12, oy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Fine-tuned (blue circle, filled)
        if fi in fine_map:
            fx, fy = int(fine_map[fi].pixel_x), int(fine_map[fi].pixel_y)
            cv2.circle(frame, (fx, fy), 6, (255, 100, 0), -1)
            cv2.putText(frame, "Fine", (fx + 12, fy - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

        # Info bar
        cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame {fi}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Green=GT  Red=Original  Blue=Finetuned", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(frame)

    cap.release()
    writer.release()
    log.info("Comparison video saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrackNet fine-tuning")
    parser.add_argument("--original", type=str, default=ORIGINAL_MODEL)
    parser.add_argument("--finetuned", type=str, default=FINETUNED_MODEL)
    parser.add_argument("--video", type=str, default=CAM66_VIDEO)
    parser.add_argument("--labels", type=str, default=CAM66_LABELS)
    parser.add_argument("--output", type=str, default=str(OUT_DIR))
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--render", action="store_true", help="Render comparison video")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check finetuned model exists
    if not Path(args.finetuned).exists():
        log.error("Fine-tuned model not found: %s", args.finetuned)
        log.info("Run tools.train_tracknet first to generate the fine-tuned model.")
        log.info("Evaluating original model only...")
        args.finetuned = None

    # Load GT
    gt = load_gt(args.labels, args.max_frames)
    log.info("GT: %d frames with ball annotations", len(gt))

    # Run inference
    original_dets = run_tracknet_inference(
        args.original, args.video, args.max_frames,
        threshold=args.threshold, label="Original",
    )

    finetuned_dets = None
    if args.finetuned:
        finetuned_dets = run_tracknet_inference(
            args.finetuned, args.video, args.max_frames,
            threshold=args.threshold, label="Fine-tuned",
        )

    # Evaluate
    log.info("\n" + "=" * 70)
    log.info("  TRACKNET EVALUATION: ORIGINAL vs FINE-TUNED")
    log.info("=" * 70)

    orig_metrics = evaluate_detections(original_dets, gt)
    orig_dead = evaluate_dead_ball(original_dets, gt)
    orig_stab = evaluate_stability(original_dets)

    log.info("\n  ── Original Model ──")
    log.info("  Recall:     %.1f%% (%d/%d)", orig_metrics["recall"] * 100,
             orig_metrics["tp"], orig_metrics["total_gt"])
    log.info("  Precision:  %.1f%%", orig_metrics["precision"] * 100)
    log.info("  F1:         %.3f", orig_metrics["f1"])
    log.info("  Mean error: %.1f px", orig_metrics["mean_error"])
    log.info("  FP:         %d", orig_metrics["fp"])
    log.info("  Dead ball:  %d frames (%d sequences, rate=%.1f%%)",
             orig_dead["dead_ball_frames"], orig_dead["dead_ball_sequences"],
             orig_dead["dead_ball_rate"] * 100)
    log.info("  Stability:  %d jumps (%.1f%%)",
             orig_stab["jumps"], orig_stab["jump_rate"] * 100)

    results = {"original": {**orig_metrics, **orig_dead, **orig_stab}}

    if finetuned_dets:
        fine_metrics = evaluate_detections(finetuned_dets, gt)
        fine_dead = evaluate_dead_ball(finetuned_dets, gt)
        fine_stab = evaluate_stability(finetuned_dets)

        log.info("\n  ── Fine-tuned Model ──")
        log.info("  Recall:     %.1f%% (%d/%d)", fine_metrics["recall"] * 100,
                 fine_metrics["tp"], fine_metrics["total_gt"])
        log.info("  Precision:  %.1f%%", fine_metrics["precision"] * 100)
        log.info("  F1:         %.3f", fine_metrics["f1"])
        log.info("  Mean error: %.1f px", fine_metrics["mean_error"])
        log.info("  FP:         %d", fine_metrics["fp"])
        log.info("  Dead ball:  %d frames (%d sequences, rate=%.1f%%)",
                 fine_dead["dead_ball_frames"], fine_dead["dead_ball_sequences"],
                 fine_dead["dead_ball_rate"] * 100)
        log.info("  Stability:  %d jumps (%.1f%%)",
                 fine_stab["jumps"], fine_stab["jump_rate"] * 100)

        # Delta
        log.info("\n  ── Improvement ──")
        dr = (fine_metrics["recall"] - orig_metrics["recall"]) * 100
        dd = orig_dead["dead_ball_frames"] - fine_dead["dead_ball_frames"]
        dj = orig_stab["jumps"] - fine_stab["jumps"]
        log.info("  Recall:     %+.1f%%", dr)
        log.info("  Dead ball:  %+d frames (%s)", -dd,
                 "improved" if dd > 0 else "worse")
        log.info("  Stability:  %+d jumps (%s)", -dj,
                 "improved" if dj > 0 else "worse")

        results["finetuned"] = {**fine_metrics, **fine_dead, **fine_stab}
        results["delta"] = {
            "recall": dr,
            "dead_ball_reduction": dd,
            "jump_reduction": dj,
        }

    # Save results
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("\nResults saved to %s", output_dir / "eval_results.json")

    # Threshold sweep
    if args.sweep:
        log.info("\n=== Threshold Sweep ===")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        log.info("\n  Original:")
        orig_sweep = threshold_sweep(original_dets, args.video, args.original, gt, thresholds,
                                     args.max_frames)
        if args.finetuned:
            log.info("\n  Fine-tuned:")
            fine_sweep = threshold_sweep(finetuned_dets, args.video, args.finetuned, gt, thresholds,
                                         args.max_frames)
            results["sweep"] = {"original": orig_sweep, "finetuned": fine_sweep}

    # Render comparison video
    if args.render and finetuned_dets:
        log.info("\nRendering comparison video...")
        render_comparison_video(
            args.video, original_dets, finetuned_dets, gt,
            output_dir / "comparison.mp4", args.max_frames,
        )


if __name__ == "__main__":
    main()
