"""
Diagnostic: feed same video through LIVE and OFFLINE code paths,
compare outputs at every stage to find where they diverge.

Uses cam66 video as single-camera test (detection stage only).
No actual cameras needed.
"""

import json
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.inference import create_detector
from app.pipeline.postprocess import BallTracker
from app.config import load_config

VIDEO = "uploads/cam66_20260307_173403_2min.mp4"
GT_DIR = "uploads/cam66_20260307_173403_2min"
MAX_FRAMES = 300
RESULT_FILE = "debug_output/test_live_vs_offline_diff.json"


def load_gt(gt_dir, max_frames):
    import glob
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


def run_test():
    cfg = load_config()
    model_path = cfg.model.path
    input_size = tuple(cfg.model.input_size)
    frames_in = cfg.model.frames_in
    frames_out = cfg.model.frames_out
    threshold = cfg.model.threshold
    device = cfg.model.device
    heatmap_mask = [tuple(r) for r in cfg.model.heatmap_mask]

    result = {
        "test": "live_vs_offline_detection_diff",
        "video": VIDEO,
        "max_frames": MAX_FRAMES,
        "config": {
            "model": model_path,
            "frames_in": frames_in,
            "threshold": threshold,
            "heatmap_mask": heatmap_mask,
        },
        "differences": [],
    }

    # Load video
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        result["error"] = f"Cannot open {VIDEO}"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(MAX_FRAMES, total)

    # Load GT
    gt = load_gt(GT_DIR, MAX_FRAMES)

    # ================================================================
    # Create TWO detectors (same model, different BG mode)
    # ================================================================
    print("Loading detectors...")
    det_offline = create_detector(model_path, input_size, frames_in, frames_out, device)
    det_live = create_detector(model_path, input_size, frames_in, frames_out, device)

    # OFFLINE: pre-compute video median
    if hasattr(det_offline, "compute_video_median"):
        print("  Offline: computing full-video median...")
        det_offline.compute_video_median(cap, 0, n_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # LIVE: uses running median (default, no pre-compute)
    print("  Live: will use running median")

    # TWO trackers (different max_blobs)
    tracker_offline = BallTracker(
        original_size=(vid_w, vid_h), threshold=threshold,
        heatmap_mask=heatmap_mask,
    )
    tracker_live = BallTracker(
        original_size=(vid_w, vid_h), threshold=threshold,
        # Live path does NOT pass heatmap_mask
    )

    # ================================================================
    # Process frames
    # ================================================================
    print(f"Processing {n_frames} frames...")
    frames_raw = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_raw.append(frame)
    cap.release()

    # Stage results
    stage_bg = {"desc": "Background median", "diffs": []}
    stage_det = {"desc": "Detection (heatmap peak)", "diffs": [], "stats": {}}
    stage_post = {"desc": "Postprocess (blob extraction)", "diffs": [], "stats": {}}

    offline_dets = {}  # frame_idx -> (px, py, conf)
    live_dets = {}

    offline_total = 0
    live_total = 0
    both_detect = 0
    only_offline = 0
    only_live = 0

    for batch_start in range(0, len(frames_raw), frames_in):
        batch_end = min(batch_start + frames_in, len(frames_raw))
        batch = frames_raw[batch_start:batch_end]
        if len(batch) < frames_in:
            break

        # Apply OSD mask (same for both)
        batch_offline = []
        batch_live = []
        for f in batch:
            fo = f.copy()
            fo[0:41, 0:603] = 0
            batch_offline.append(fo)

            fl = f.copy()
            fl[0:41, 0:603] = 0
            batch_live.append(fl)

        # ---- STAGE 1: Compare BG median ----
        if hasattr(det_offline, '_bg_frame') and hasattr(det_live, '_bg_frame'):
            if det_offline._bg_frame is not None and det_live._bg_frame is not None:
                bg_diff = np.abs(det_offline._bg_frame.astype(float) - det_live._bg_frame.astype(float))
                if bg_diff.mean() > 0.001:
                    stage_bg["diffs"].append({
                        "batch": batch_start,
                        "mean_diff": round(float(bg_diff.mean()), 4),
                        "max_diff": round(float(bg_diff.max()), 4),
                    })

        # ---- STAGE 2: Inference ----
        hm_offline = det_offline.infer(batch_offline)
        hm_live = det_live.infer(batch_live)

        # ---- STAGE 3: Postprocess ----
        for i in range(min(frames_out, len(hm_offline))):
            fi = batch_start + i

            # Heatmap difference
            hm_diff = np.abs(hm_offline[i].astype(float) - hm_live[i].astype(float))
            hm_mean_diff = float(hm_diff.mean())

            # Offline: max_blobs=3, with heatmap_mask
            blobs_off = tracker_offline.process_heatmap_multi(hm_offline[i], max_blobs=3)
            # Live: max_blobs=2, no heatmap_mask
            blobs_live = tracker_live.process_heatmap_multi(hm_live[i], max_blobs=2)

            off_top = blobs_off[0] if blobs_off else None
            live_top = blobs_live[0] if blobs_live else None

            if off_top:
                offline_dets[fi] = (off_top["pixel_x"], off_top["pixel_y"], off_top["blob_sum"])
                offline_total += 1
            if live_top:
                live_dets[fi] = (live_top["pixel_x"], live_top["pixel_y"], live_top["blob_sum"])
                live_total += 1

            if off_top and live_top:
                both_detect += 1
                dx = abs(off_top["pixel_x"] - live_top["pixel_x"])
                dy = abs(off_top["pixel_y"] - live_top["pixel_y"])
                if dx > 5 or dy > 5:
                    stage_det["diffs"].append({
                        "frame": fi,
                        "offline_px": [round(off_top["pixel_x"], 1), round(off_top["pixel_y"], 1)],
                        "live_px": [round(live_top["pixel_x"], 1), round(live_top["pixel_y"], 1)],
                        "diff_px": [round(dx, 1), round(dy, 1)],
                        "heatmap_mean_diff": round(hm_mean_diff, 5),
                        "offline_conf": round(off_top["blob_sum"], 2),
                        "live_conf": round(live_top["blob_sum"], 2),
                    })
            elif off_top and not live_top:
                only_offline += 1
                stage_post["diffs"].append({
                    "frame": fi, "type": "only_offline",
                    "offline_px": [round(off_top["pixel_x"], 1), round(off_top["pixel_y"], 1)],
                    "offline_conf": round(off_top["blob_sum"], 2),
                })
            elif live_top and not off_top:
                only_live += 1
                stage_post["diffs"].append({
                    "frame": fi, "type": "only_live",
                    "live_px": [round(live_top["pixel_x"], 1), round(live_top["pixel_y"], 1)],
                    "live_conf": round(live_top["blob_sum"], 2),
                })

        if batch_start % 80 == 0:
            print(f"  frame {batch_start}/{n_frames}")

    # ---- GT Comparison ----
    radius = 25
    off_tp, off_fn, live_tp, live_fn = 0, 0, 0, 0
    for fi, (gx, gy) in gt.items():
        if fi in offline_dets:
            dx = offline_dets[fi][0] - gx
            dy = offline_dets[fi][1] - gy
            if dx * dx + dy * dy <= radius * radius:
                off_tp += 1
            else:
                off_fn += 1
        else:
            off_fn += 1

        if fi in live_dets:
            dx = live_dets[fi][0] - gx
            dy = live_dets[fi][1] - gy
            if dx * dx + dy * dy <= radius * radius:
                live_tp += 1
            else:
                live_fn += 1
        else:
            live_fn += 1

    off_recall = off_tp / (off_tp + off_fn) if (off_tp + off_fn) else 0
    live_recall = live_tp / (live_tp + live_fn) if (live_tp + live_fn) else 0

    result["stages"] = {
        "bg_median": {
            "n_batches_with_diff": len(stage_bg["diffs"]),
            "first_5": stage_bg["diffs"][:5],
        },
        "detection": {
            "both_detect": both_detect,
            "only_offline": only_offline,
            "only_live": only_live,
            "position_diffs_gt_5px": len(stage_det["diffs"]),
            "first_10": stage_det["diffs"][:10],
        },
        "postprocess": {
            "only_offline_count": sum(1 for d in stage_post["diffs"] if d["type"] == "only_offline"),
            "only_live_count": sum(1 for d in stage_post["diffs"] if d["type"] == "only_live"),
            "first_10": stage_post["diffs"][:10],
        },
    }

    result["gt_comparison"] = {
        "gt_frames": len(gt),
        "offline": {"tp": off_tp, "fn": off_fn, "recall": round(off_recall, 3)},
        "live": {"tp": live_tp, "fn": live_fn, "recall": round(live_recall, 3)},
        "recall_gap": round(off_recall - live_recall, 3),
    }

    result["root_causes"] = []
    if len(stage_bg["diffs"]) > 0:
        result["root_causes"].append({
            "cause": "BG_MEDIAN_DIFFERENCE",
            "desc": "Offline uses full-video median; live uses running median from first N frames",
            "severity": "HIGH",
            "impact": f"{len(stage_bg['diffs'])} batches with different BG",
        })
    if only_offline > 0:
        result["root_causes"].append({
            "cause": "HEATMAP_MASK_MISSING_IN_LIVE",
            "desc": "Live BallTracker has no heatmap_mask; offline has mask for OSD region",
            "severity": "MEDIUM",
            "impact": f"{only_offline} frames detected only offline",
        })
    if len(stage_det["diffs"]) > 0:
        result["root_causes"].append({
            "cause": "DETECTION_POSITION_SHIFT",
            "desc": "Same frame, different detection position (>5px) due to BG difference",
            "severity": "HIGH",
            "impact": f"{len(stage_det['diffs'])} frames with shifted detections",
        })

    result["verdict"] = "IDENTICAL" if not result["root_causes"] else "DIVERGENT"

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_test()
