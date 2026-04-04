"""
GT accuracy test: run TrackNet detection on GT video frames,
triangulate, detect bounces, compare with GT annotations frame by frame.

Uses the SAME frames the GT was annotated on.
Tests the full pipeline: detection → homography → triangulation → bounce.
"""

import json
import glob
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.inference import create_detector
from app.pipeline.postprocess import BallTracker
from app.pipeline.homography import HomographyTransformer
from app.triangulation import triangulate
from app.pipeline.bounce_detect import detect_bounces
from app.config import load_config

CAM66_DIR = "uploads/cam66_20260307_173403_2min"
CAM68_DIR = "uploads/cam68_20260307_173403_2min"
GT_BOUNCE_FILE = "D:/tennis/blob_frame_different/bounce_results.json"
MAX_FRAMES = 3000
RESULT_FILE = "debug_output/test_gt_accuracy.json"
RADIUS = 25  # pixel match radius


def load_gt_ball(gt_dir, max_frames):
    gt = {}
    for jf in sorted(glob.glob(os.path.join(gt_dir, "*.json")))[:max_frames]:
        idx = int(os.path.basename(jf).replace(".json", ""))
        with open(jf) as f:
            d = json.load(f)
        for s in d["shapes"]:
            if s["label"] == "ball" and s["shape_type"] == "rectangle":
                pts = s["points"]
                gt[idx] = ((pts[0][0] + pts[2][0]) / 2, (pts[0][1] + pts[2][1]) / 2)
                break
    return gt


def run_detection(gt_dir, cam_name, cfg, max_frames):
    """Run TrackNet detection on GT frames (JPEGs), return per-frame detections."""
    model_path = cfg.model.path
    input_size = tuple(cfg.model.input_size)
    frames_in = cfg.model.frames_in
    frames_out = cfg.model.frames_out
    threshold = cfg.model.threshold
    device = cfg.model.device
    heatmap_mask = [tuple(r) for r in cfg.model.heatmap_mask]

    detector = create_detector(model_path, input_size, frames_in, frames_out, device)
    tracker = BallTracker(
        original_size=(1920, 1080), threshold=threshold, heatmap_mask=heatmap_mask,
    )

    # Pre-compute video median from frames
    frame_paths = sorted(glob.glob(os.path.join(gt_dir, "*.jpg")))[:max_frames]
    if hasattr(detector, "compute_video_median"):
        # Use video file if available for median
        video_name = os.path.basename(gt_dir) + ".mp4"
        video_path = os.path.join("uploads", video_name)
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            detector.compute_video_median(cap, 0, min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
            cap.release()
            print(f"  {cam_name}: video median from {video_name}")
        else:
            # Fallback: compute from JPEG frames
            sample_frames = []
            step = max(1, len(frame_paths) // 200)
            for i in range(0, len(frame_paths), step):
                f = cv2.imread(frame_paths[i])
                if f is not None:
                    sample_frames.append(cv2.resize(f, (input_size[1], input_size[0])))
            if sample_frames:
                median = np.median(sample_frames, axis=0).astype(np.uint8)
                detector._bg_frame = median.astype(np.float32).transpose(2, 0, 1) / 255.0
                detector._video_median_computed = True
                print(f"  {cam_name}: median from {len(sample_frames)} JPEG frames")

    detections = {}  # frame_idx -> {pixel_x, pixel_y, confidence}
    buffer = []
    indices = []

    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            continue
        # OSD mask
        frame[0:41, 0:603] = 0
        buffer.append(frame)
        indices.append(i)

        if len(buffer) < frames_in:
            continue

        heatmaps = detector.infer(buffer)
        for j in range(min(frames_out, len(heatmaps))):
            fi = indices[j]
            blobs = tracker.process_heatmap_multi(heatmaps[j], max_blobs=3)
            if blobs:
                top = blobs[0]
                detections[fi] = {
                    "pixel_x": top["pixel_x"],
                    "pixel_y": top["pixel_y"],
                    "confidence": top["blob_sum"],
                }
        buffer.clear()
        indices.clear()

        if i % 200 == 0:
            print(f"  {cam_name}: {i}/{len(frame_paths)}")

    return detections


def main():
    cfg = load_config()
    result = {
        "test": "gt_accuracy_full_pipeline",
        "max_frames": MAX_FRAMES,
        "model": cfg.model.path,
    }

    # Load GT annotations
    gt66 = load_gt_ball(CAM66_DIR, MAX_FRAMES)
    gt68 = load_gt_ball(CAM68_DIR, MAX_FRAMES)
    print(f"GT: cam66={len(gt66)}, cam68={len(gt68)}")

    # Load GT bounce reference
    with open(GT_BOUNCE_FILE) as f:
        gt_ref = json.load(f)
    gt_bounces = gt_ref["bounces"]
    gt_traj = gt_ref["trajectory"]

    # Stage 1: Run detection on both cameras
    print("\n--- Stage 1: TrackNet Detection ---")
    dets66 = run_detection(CAM66_DIR, "cam66", cfg, MAX_FRAMES)
    dets68 = run_detection(CAM68_DIR, "cam68", cfg, MAX_FRAMES)

    # Detection recall vs GT
    for cam, dets, gt in [("cam66", dets66, gt66), ("cam68", dets68, gt68)]:
        tp, fn = 0, 0
        for fi, (gx, gy) in gt.items():
            if fi in dets:
                dx = dets[fi]["pixel_x"] - gx
                dy = dets[fi]["pixel_y"] - gy
                if dx * dx + dy * dy <= RADIUS * RADIUS:
                    tp += 1
                else:
                    fn += 1
            else:
                fn += 1
        recall = tp / (tp + fn) if (tp + fn) else 0
        result[f"detection_{cam}"] = {
            "gt_frames": len(gt),
            "detected": len(dets),
            "tp": tp, "fn": fn,
            "recall": round(recall, 3),
        }
        print(f"  {cam}: recall={recall:.1%} ({tp}/{tp+fn}), detected={len(dets)}")

    # Stage 2: Homography + Triangulation
    print("\n--- Stage 2: Homography + Triangulation ---")
    h66 = HomographyTransformer(cfg.homography.path, "cam66")
    h68 = HomographyTransformer(cfg.homography.path, "cam68")
    cam_pos = {n: cfg.cameras[n].position_3d for n in cfg.cameras}
    cams = [n for n in cam_pos if cam_pos[n] != [0, 0, 0]][:2]

    common_frames = sorted(set(dets66.keys()) & set(dets68.keys()))
    trajectory = []
    for fi in common_frames:
        px66 = dets66[fi]["pixel_x"]
        py66 = dets66[fi]["pixel_y"]
        px68 = dets68[fi]["pixel_x"]
        py68 = dets68[fi]["pixel_y"]
        w66 = h66.pixel_to_world(px66, py66)
        w68 = h68.pixel_to_world(px68, py68)
        x, y, z = triangulate(w66, w68, cam_pos[cams[0]], cam_pos[cams[1]])
        trajectory.append({
            "frame": fi, "x": x, "y": y, "z": z,
            "px66": round(px66, 1), "py66": round(py66, 1),
            "px68": round(px68, 1), "py68": round(py68, 1),
        })

    # Compare with GT trajectory
    gt_traj_map = {t["frame"]: t for t in gt_traj}
    traj_errors = []
    for pt in trajectory:
        ref = gt_traj_map.get(pt["frame"])
        if ref:
            dx = abs(pt["x"] - ref["x"])
            dy = abs(pt["y"] - ref["y"])
            dz = abs(pt["z"] - ref["z"])
            traj_errors.append({"frame": pt["frame"], "dx": dx, "dy": dy, "dz": dz,
                                "dist_2d": (dx**2 + dy**2)**0.5})

    if traj_errors:
        dxs = [e["dx"] for e in traj_errors]
        dys = [e["dy"] for e in traj_errors]
        dzs = [e["dz"] for e in traj_errors]
        d2ds = [e["dist_2d"] for e in traj_errors]
        result["triangulation"] = {
            "pipeline_points": len(trajectory),
            "gt_matched": len(traj_errors),
            "error_2d": {"mean": round(np.mean(d2ds), 3), "max": round(np.max(d2ds), 3),
                         "p95": round(np.percentile(d2ds, 95), 3)},
            "error_x": {"mean": round(np.mean(dxs), 3), "max": round(np.max(dxs), 3)},
            "error_y": {"mean": round(np.mean(dys), 3), "max": round(np.max(dys), 3)},
            "error_z": {"mean": round(np.mean(dzs), 3), "max": round(np.max(dzs), 3)},
        }
        print(f"  Trajectory: {len(trajectory)} points, GT matched: {len(traj_errors)}")
        print(f"  2D error: mean={np.mean(d2ds):.3f}m, max={np.max(d2ds):.3f}m")

    # Stage 3: Bounce detection
    print("\n--- Stage 3: Bounce Detection ---")
    traj_tuples = [(pt["frame"], pt["x"], pt["y"], pt["z"], 0) for pt in trajectory]
    pipe_bounces = detect_bounces(traj_tuples)

    gt_bounce_frames = sorted(b["frame"] for b in gt_bounces)
    pipe_bounce_frames = sorted(b["frame"] for b in pipe_bounces)

    # Match bounces
    matched, missed, fp = [], [], []
    used = set()
    for gb in gt_bounces:
        gf = gb["frame"]
        best_i, best_d = None, 5
        for i, pb in enumerate(pipe_bounces):
            if i in used:
                continue
            if abs(pb["frame"] - gf) <= 5:
                if best_i is None or abs(pb["frame"] - gf) < best_d:
                    best_i, best_d = i, abs(pb["frame"] - gf)
        if best_i is not None:
            pb = pipe_bounces[best_i]
            used.add(best_i)
            dx = abs(pb["x"] - gb["x"])
            dy = abs(pb["y"] - gb["y"])
            matched.append({
                "gt_frame": gf, "pipe_frame": pb["frame"],
                "gt_pos": {"x": round(gb["x"], 3), "y": round(gb["y"], 3)},
                "pipe_pos": {"x": round(pb["x"], 3), "y": round(pb["y"], 3)},
                "error_2d": round((dx**2 + dy**2)**0.5, 3),
            })
        else:
            missed.append(gf)
    for i, pb in enumerate(pipe_bounces):
        if i not in used:
            fp.append(pb["frame"])

    bounce_recall = len(matched) / len(gt_bounces) if gt_bounces else 0
    bounce_prec = len(matched) / len(pipe_bounces) if pipe_bounces else 0

    result["bounce_detection"] = {
        "gt_count": len(gt_bounces),
        "detected_count": len(pipe_bounces),
        "matched": len(matched),
        "missed": len(missed),
        "false_positives": len(fp),
        "recall": round(bounce_recall, 3),
        "precision": round(bounce_prec, 3),
        "missed_frames": missed,
        "fp_frames": fp[:20],
        "landing_errors": [m["error_2d"] for m in matched],
        "landing_error_mean": round(np.mean([m["error_2d"] for m in matched]), 3) if matched else 0,
    }
    print(f"  Bounce: recall={bounce_recall:.1%} ({len(matched)}/{len(gt_bounces)}), "
          f"precision={bounce_prec:.1%}, FP={len(fp)}")
    if matched:
        errs = [m["error_2d"] for m in matched]
        print(f"  Landing error: mean={np.mean(errs):.3f}m, max={np.max(errs):.3f}m")

    # Verdict
    det_recall_min = min(result.get("detection_cam66", {}).get("recall", 0),
                         result.get("detection_cam68", {}).get("recall", 0))
    result["verdict"] = {
        "detection_recall_min": det_recall_min,
        "bounce_recall": bounce_recall,
        "pass": bounce_recall >= 0.9 and det_recall_min >= 0.85,
    }

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"VERDICT: {'PASS' if result['verdict']['pass'] else 'FAIL'}")
    print(f"  Detection recall: cam66={result['detection_cam66']['recall']:.1%}, cam68={result['detection_cam68']['recall']:.1%}")
    print(f"  Bounce recall: {bounce_recall:.1%}")
    print(f"Saved: {RESULT_FILE}")

    return result


if __name__ == "__main__":
    main()
