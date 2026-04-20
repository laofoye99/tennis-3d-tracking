"""Compare Hybrid vs Peak streaming detectors vs offline detect_bounces against GT.

Runs the full offline detection pipeline (cam66 + cam68 → 3D trajectory), then
feeds the same trajectory through three bounce detectors:

    1. Offline reference: tools.render_tracking_video.detect_bounces()
    2. Streaming Hybrid: app.analytics.HybridBounceDetector (with live params)
    3. Streaming Peak:   app.analytics.PeakBounceDetector

Matches each detector's output against GT `match_ball,bounce[,_out]` events.
Reports recall / precision / position error for each.

This isolates whether Hybrid's low live yield is (a) an algorithm issue
(fails on GT 3D data too) or (b) live-data-sparsity issue (succeeds on
dense trajectory but fails on live's intermittent one).

Usage:
    python -m tools.eval_detector_recall [--max-frames 3000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths — GT lives under the worktree; video copies are in main uploads/
GT_DIR = Path(".claude/worktrees/elastic-goldberg/uploads/cam66_20260307_173403_2min")
VIDEO_66 = "uploads/cam66_20260307_173403_2min.mp4"
VIDEO_68 = "uploads/cam68_20260307_173403_2min.mp4"

TOLERANCE_FRAMES = 10  # match GT↔detected within ±10 frames
POSITION_TOL_M = 2.0   # report errors above this as suspicious


def load_gt_bounces(gt_dir: Path, max_frames: int) -> list[tuple[int, float, float, str]]:
    """Return [(frame, pixel_x, pixel_y, label), ...] for match_ball bounces only."""
    out = []
    for fi in range(max_frames):
        p = gt_dir / f"{fi:05d}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for s in data.get("shapes", []):
            desc = (s.get("description") or "").strip().lower()
            if "match_ball" not in desc or "bounce" not in desc:
                continue
            pts = s.get("points") or []
            if not pts:
                continue
            if s.get("shape_type") == "rectangle" and len(pts) >= 2:
                px = (pts[0][0] + pts[1][0]) / 2
                py = (pts[0][1] + pts[1][1]) / 2
            else:
                px, py = pts[0][0], pts[0][1]
            out.append((fi, float(px), float(py), desc))
            break
    return out


def load_config_dict():
    with open("config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pixel_to_world(homography_path: str, cam_key: str, px: float, py: float):
    H = np.array(json.load(open(homography_path))[cam_key]["H_image_to_world"])
    p = np.array([px, py, 1.0])
    w = H @ p
    return float(w[0] / w[2]), float(w[1] / w[2])


def run_offline_pipeline(max_frames: int):
    """Reuse tools.render_tracking_video to produce a dense 3D trajectory."""
    from tools.render_tracking_video import (
        build_detector, run_detection_multi,
        triangulate_multi_blob, smooth_trajectory_sg,
        detect_bounces as offline_detect_bounces,
    )
    cfg = load_config_dict()
    detector, postproc = build_detector(cfg)

    logger.info("Running cam66 detection (max %d frames)...", max_frames)
    multi66, det66, n66 = run_detection_multi(VIDEO_66, detector, postproc, max_frames, top_k=2)
    logger.info("Running cam68 detection...")
    multi68, det68, n68 = run_detection_multi(VIDEO_68, detector, postproc, max_frames, top_k=2)

    logger.info("Triangulating...")
    points_3d, _, stats = triangulate_multi_blob(multi66, multi68, cfg)
    logger.info("Triangulation: %d 3D points", len(points_3d))

    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)

    return points_3d, smoothed_3d, offline_detect_bounces


def run_streaming_hybrid(smoothed_3d: dict, points_3d: dict | None = None,
                          **hybrid_kwargs) -> list[dict]:
    """Feed 3D trajectory frame-by-frame into HybridBounceDetector.

    Each emitted bounce includes:
        frame       — detector's claimed event frame (bounce_frame)
        emit_frame  — the frame at which update() actually returned it
        latency_f   = emit_frame - frame (detector's real report lag)

    If `points_3d` is supplied, cam_dets at every frame carries per-camera
    world coords (from top-1 blob of the offline pipeline) so _select_landing_coords
    Priority 1 activates — letting us observe the landing-coord correction.
    """
    from app.analytics import HybridBounceDetector

    det = HybridBounceDetector(**hybrid_kwargs)
    out = []
    frames = sorted(smoothed_3d.keys())
    for emit_idx, fi in enumerate(frames):
        x, y, z = smoothed_3d[fi]
        pt = {
            "x": float(x), "y": float(y), "z": float(z),
            "timestamp": emit_idx / 25.0,
            "frame_index": int(fi),
        }
        cam_dets = None
        if points_3d and fi in points_3d:
            # Offline points_3d has (x, y, z, ...). No per-camera world coords
            # are stored separately, so we synthesize a cam_dets with the same
            # world point under both cameras — the tie-break then falls to
            # blob_sum (also unavailable here), exercising the no-preference
            # code path. This is a proxy for live behavior.
            px, py, pz = points_3d[fi][:3]
            cam_dets = {
                "cam66": {"world_x": float(px), "world_y": float(py),
                          "blob_sum": 1.0},
                "cam68": {"world_x": float(px), "world_y": float(py),
                          "blob_sum": 1.0},
            }
        b = det.update(pt, cam_detections=cam_dets)
        if b is not None:
            bounce_frame = int(b.frame_index) if b.frame_index else int(fi)
            out.append({
                "frame": bounce_frame,
                "emit_frame": int(fi),
                "latency_f": int(fi) - bounce_frame,
                "x": float(b.x), "y": float(b.y), "z": float(b.z),
                "in_court": bool(b.in_court),
                "source_camera": b.source_camera,
            })
    return out


def run_streaming_peak(smoothed_3d: dict, **peak_kwargs) -> list[dict]:
    """Feed 3D trajectory through PeakBounceDetector, drain pop_pending after each update.

    Each emitted bounce includes:
        frame       — detector's claimed event frame (from BounceEvent.frame_index)
        emit_frame  — the frame at which pop_pending() returned it
        latency_f   = emit_frame - frame
    """
    from app.analytics import PeakBounceDetector

    det = PeakBounceDetector(**peak_kwargs)
    out = []
    frames = sorted(smoothed_3d.keys())
    for emit_idx, fi in enumerate(frames):
        x, y, z = smoothed_3d[fi]
        pt = {
            "x": float(x), "y": float(y), "z": float(z),
            "timestamp": emit_idx / 25.0,
            "frame_index": int(fi),
        }
        det.update(pt)
        if hasattr(det, "pop_pending"):
            for b in det.pop_pending():
                bounce_frame = int(b.frame_index)
                out.append({
                    "frame": bounce_frame,
                    "emit_frame": int(fi),
                    "latency_f": int(fi) - bounce_frame,
                    "x": float(b.x), "y": float(b.y), "z": float(b.z),
                    "in_court": bool(b.in_court),
                })
    return out


def match_against_gt(
    detected: list[dict], gt_world: list[tuple[int, float, float, str]], tol_frames: int
):
    """Greedy frame-first match. Return (matches, missed_gt_idx, spurious_det_idx)."""
    gt_used, det_used = set(), set()
    matches = []
    for di, db in enumerate(detected):
        df = db["frame"]
        dx, dy = db["x"], db["y"]
        best_fd = tol_frames + 1
        best_gi = -1
        for gi, (gf, gwx, gwy, _lbl) in enumerate(gt_world):
            if gi in gt_used:
                continue
            fd = abs(df - gf)
            if fd < best_fd:
                best_fd = fd
                best_gi = gi
        if best_gi >= 0 and best_fd <= tol_frames:
            gf, gwx, gwy, _lbl = gt_world[best_gi]
            matches.append({
                "gt_frame": gf, "det_frame": df,
                "gt_xy": (gwx, gwy), "det_xy": (dx, dy),
                "err_m": ((dx - gwx) ** 2 + (dy - gwy) ** 2) ** 0.5,
                "frame_delta": best_fd,
            })
            gt_used.add(best_gi)
            det_used.add(di)
    missed = [i for i in range(len(gt_world)) if i not in gt_used]
    spurious = [i for i in range(len(detected)) if i not in det_used]
    return matches, missed, spurious


def summarize(label: str, detected: list[dict], gt_world: list[tuple[int, float, float, str]],
              total_frames: int):
    matches, missed, spurious = match_against_gt(detected, gt_world, TOLERANCE_FRAMES)
    n_gt = len(gt_world)
    n_det = len(detected)
    tp = len(matches)
    fn = len(missed)
    fp = len(spurious)
    # Treat each frame as a trial: non-bounce frames where nothing fired → TN
    # Approximate TN = total_frames - (GT bounces) - (spurious detections)
    tn = max(0, total_frames - n_gt - fp)
    recall = tp / n_gt if n_gt else 0
    precision = tp / n_det if n_det else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0

    # in_court confusion (only on TP matches where we have both sides)
    # GT in/out inferred from label: 'bounce_out' → OUT, 'bounce' → IN
    gt_by_frame = {fi: ("out" not in lbl) for fi, _gx, _gy, lbl in gt_world}  # True = in
    # Re-pair TPs to read their detected in_court; index into detected via det_frame
    det_by_frame = {db["frame"]: db for db in detected}
    ic_tt = ic_tf = ic_ft = ic_ff = 0  # GT-in/det-in, GT-in/det-out, GT-out/det-in, GT-out/det-out
    for m in matches:
        gt_in = gt_by_frame.get(m["gt_frame"], True)
        db = det_by_frame.get(m["det_frame"], {})
        det_in = bool(db.get("in_court", True))
        if gt_in and det_in: ic_tt += 1
        elif gt_in and not det_in: ic_tf += 1
        elif not gt_in and det_in: ic_ft += 1
        else: ic_ff += 1

    errs = [m["err_m"] for m in matches]
    gt_deltas = [m["frame_delta"] for m in matches]
    det_by_frame = {db["frame"]: db for db in detected}
    real_latencies = [
        det_by_frame.get(m["det_frame"], {}).get("latency_f")
        for m in matches
        if det_by_frame.get(m["det_frame"], {}).get("latency_f") is not None
    ]

    print()
    print(f"=== {label} ===")
    print(f"  frames analyzed: {total_frames} | GT events: {n_gt} | detected: {n_det}")
    print(f"  +-----------------+---------+")
    print(f"  | TP (matched)    | {tp:7d} |")
    print(f"  | FN (missed GT)  | {fn:7d} |")
    print(f"  | FP (spurious)   | {fp:7d} |")
    print(f"  | TN (quiet frm)  | {tn:7d} |  (approx: total_frames - GT - FP)")
    print(f"  +-----------------+---------+")
    print(f"  recall    = TP/(TP+FN) = {recall*100:5.1f}%")
    print(f"  precision = TP/(TP+FP) = {precision*100:5.1f}%")
    print(f"  F1        = {f1*100:5.1f}%")
    if errs:
        print(f"  err median:       {np.median(errs):.2f} m   p90: {np.percentile(errs, 90):.2f} m")
        print(f"  GT frame delta:   median {int(np.median(gt_deltas))}   max {max(gt_deltas)}"
              f"   (diff between detector's claim and GT label; matching tolerance)")
    if real_latencies:
        print(f"  real report lag:  median {int(np.median(real_latencies))} frames"
              f"   max {max(real_latencies)}   (emit_frame - bounce_frame)")
    print(f"  in_court on TP:   GT_in/det_in={ic_tt}  GT_in/det_out={ic_tf}  GT_out/det_in={ic_ft}  GT_out/det_out={ic_ff}")
    if missed:
        print(f"  missed GT frames: {[gt_world[i][0] for i in missed]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=3000)
    args = ap.parse_args()

    # 1. Load GT (pixel space) and project to V2 world via cam66 homography
    logger.info("Loading GT bounces from %s...", GT_DIR)
    gt_px = load_gt_bounces(GT_DIR, args.max_frames)
    logger.info("GT bounces: %d", len(gt_px))

    cfg = load_config_dict()
    h_path = cfg["homography"]["path"]
    cam_key = cfg["cameras"]["cam66"]["homography_key"]
    gt_world = [
        (fi, *pixel_to_world(h_path, cam_key, px, py), lbl)
        for fi, px, py, lbl in gt_px
    ]
    logger.info("GT projected to world coords (cam66 homography)")

    # 2. Run offline pipeline for the 3D trajectory
    points_3d, smoothed_3d, offline_detect_bounces = run_offline_pipeline(args.max_frames)
    logger.info("Smoothed 3D trajectory: %d frames", len(smoothed_3d))

    # 3. Offline reference detector (V-shape + parabolic, the original logic Hybrid is based on)
    offline_bounces = offline_detect_bounces(smoothed_3d)
    logger.info("Offline detect_bounces: %d events", len(offline_bounces))

    # 4. Streaming Hybrid — once w/o cam_dets (3D fallback landing),
    # once WITH per-frame cam_dets (homography landing via _select_landing_coords)
    hybrid_default = run_streaming_hybrid(smoothed_3d)
    hybrid_with_cam = run_streaming_hybrid(smoothed_3d, points_3d=points_3d)

    # 5. Streaming Peak
    peak_bounces = run_streaming_peak(smoothed_3d, batch_size=10)

    # 6. Report
    summarize("OFFLINE detect_bounces (reference)", offline_bounces, gt_world, args.max_frames)
    summarize("STREAMING Hybrid (no cam_dets, 3D landing)", hybrid_default, gt_world, args.max_frames)
    summarize("STREAMING Hybrid (with cam_dets, homography landing)", hybrid_with_cam, gt_world, args.max_frames)
    summarize("STREAMING Peak", peak_bounces, gt_world, args.max_frames)


if __name__ == "__main__":
    main()
