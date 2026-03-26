"""Export all intermediate tracking data to JSON for analysis/visualization.

Runs the full pipeline (detection, matching, triangulation, smoothing,
bounce detection, net crossing) and saves ALL intermediate data.

Usage:
    python -m tools.export_tracking_data \
        --video66 uploads/cam66_20260307_173403_2min.mp4 \
        --video68 uploads/cam68_20260307_173403_2min.mp4 \
        --output exports/tracking_2min_data.json \
        --max-frames 1800 --mode viterbi
"""

import argparse
import cv2
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Reuse all functions from render_tracking_video
from tools.render_tracking_video import (
    load_config,
    build_detector,
    run_detection_multi,
    load_stereo_calibration,
    triangulate_stereo,
    triangulate_with_ray_dist,
    triangulate_detections,
    triangulate_multi_blob,
    build_flight_mask,
    smooth_trajectory_sg,
    detect_bounces,
    detect_bounces_2d,
    detect_net_crossings,
    SINGLES_X_MIN, SINGLES_X_MAX, COURT_L, NET_Y,
)


def main():
    parser = argparse.ArgumentParser(description="Export tracking pipeline data to JSON")
    parser.add_argument("--video66", default="uploads/cam66_20260307_173403_2min.mp4")
    parser.add_argument("--video68", default="uploads/cam68_20260307_173403_2min.mp4")
    parser.add_argument("--output", default="exports/tracking_2min_data.json")
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--mode", choices=["top1", "multi", "viterbi"], default="viterbi",
    )
    parser.add_argument("--no-rally", action="store_true")
    parser.add_argument("--ocr-align", action="store_true",
                        help="Use PaddleOCR to read OSD timestamps and align frames")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    export = {
        "meta": {
            "video66": args.video66,
            "video68": args.video68,
            "max_frames": args.max_frames,
            "mode": args.mode,
            "top_k": args.top_k,
            "fps": 25.0,
            "court": {
                "singles_x_min": SINGLES_X_MIN,
                "singles_x_max": SINGLES_X_MAX,
                "court_length": COURT_L,
                "net_y": NET_Y,
            },
        },
    }

    # ── Phase 1: Detection ──────────────────────────────────────────
    logger.info("=== Phase 1: TrackNet Detection (top_k=%d, mode=%s) ===", args.top_k, args.mode)
    detector, postproc = build_detector(cfg)

    logger.info("--- cam66 ---")
    multi66, det66, n66 = run_detection_multi(
        args.video66, detector, postproc, args.max_frames, top_k=args.top_k,
    )

    detector._bg_frame = None
    detector._video_median_computed = False

    logger.info("--- cam68 ---")
    multi68, det68, n68 = run_detection_multi(
        args.video68, detector, postproc, args.max_frames, top_k=args.top_k,
    )

    n_frames = min(n66, n68)

    # Save raw detections (top-1 for each camera)
    export["det66"] = {
        str(fi): {"px": float(v[0]), "py": float(v[1]), "blob_sum": float(v[2])}
        for fi, v in det66.items()
    }
    export["det68"] = {
        str(fi): {"px": float(v[0]), "py": float(v[1]), "blob_sum": float(v[2])}
        for fi, v in det68.items()
    }

    # Save multi-blob detections
    export["multi66"] = {
        str(fi): [
            {"px": float(b["pixel_x"]), "py": float(b["pixel_y"]),
             "blob_sum": float(b["blob_sum"]), "blob_max": float(b.get("blob_max", 0)),
             "blob_area": int(b.get("blob_area", 0))}
            for b in blobs
        ]
        for fi, blobs in multi66.items()
    }
    export["multi68"] = {
        str(fi): [
            {"px": float(b["pixel_x"]), "py": float(b["pixel_y"]),
             "blob_sum": float(b["blob_sum"]), "blob_max": float(b.get("blob_max", 0)),
             "blob_area": int(b.get("blob_area", 0))}
            for b in blobs
        ]
        for fi, blobs in multi68.items()
    }

    export["meta"]["n_frames"] = n_frames
    export["meta"]["det66_count"] = len(det66)
    export["meta"]["det68_count"] = len(det68)

    # ── Phase 1.5: Frame Alignment ─────────────────────────────────
    if getattr(args, 'ocr_align', False):
        import os as _os
        _os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        import re as _re
        logger.info("=== Phase 1.5: OCR Frame Alignment ===")

        del detector
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        import gc; gc.collect()

        from paddleocr import TextRecognition
        ocr_model = TextRecognition(
            model_name='PP-OCRv5_mobile_rec',
            model_dir='model_weight/PP-OCRv5_mobile_rec',
            enable_mkldnn=False,
        )

        def _read_osd_sec(frame):
            crop = frame[0:40, 430:615]
            out = ocr_model.predict(input=crop, batch_size=1)
            for r in out:
                m = _re.search(r'(\d{1,2}):(\d{2}):(\d{2})', r['rec_text'])
                if m:
                    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            return None

        import time as _time
        _t0 = _time.perf_counter()
        cap66_a = cv2.VideoCapture(args.video66)
        cap68_a = cv2.VideoCapture(args.video68)

        from collections import defaultdict as _ddict
        sec_f66 = _ddict(list)
        sec_f68 = _ddict(list)
        for fi in range(0, n_frames, 25):
            cap66_a.set(cv2.CAP_PROP_POS_FRAMES, fi)
            cap68_a.set(cv2.CAP_PROP_POS_FRAMES, fi)
            r66, f66 = cap66_a.read()
            r68, f68 = cap68_a.read()
            if not r66 or not r68: break
            s66 = _read_osd_sec(f66)
            s68 = _read_osd_sec(f68)
            if s66: sec_f66[s66].append(fi)
            if s68: sec_f68[s68].append(fi)
        cap66_a.release()
        cap68_a.release()

        import numpy as _np
        common = sorted(set(sec_f66.keys()) & set(sec_f68.keys()))
        _elapsed = _time.perf_counter() - _t0
        logger.info("  OCR: %d common seconds in %.1fs", len(common), _elapsed)

        if len(common) >= 2:
            a66 = _np.array([int(_np.median(sec_f66[s])) for s in common], dtype=float)
            a68 = _np.array([int(_np.median(sec_f68[s])) for s in common], dtype=float)
            new_det68 = {}
            new_multi68 = {}
            for fi in range(n_frames):
                fi68 = int(_np.interp(fi, a66, a68))
                fi68 = max(0, min(fi68, n68 - 1))
                if fi68 in det68: new_det68[fi] = det68[fi68]
                if fi68 in multi68: new_multi68[fi] = multi68[fi68]
            logger.info("  Remapped cam68: %d -> %d detections", len(det68), len(new_det68))
            det68 = new_det68
            multi68 = new_multi68
            # Update export
            export["det68"] = {
                str(fi): {"px": float(v[0]), "py": float(v[1]), "blob_sum": float(v[2])}
                for fi, v in det68.items()
            }
            export["meta"]["frame_alignment"] = "ocr_piecewise"
        else:
            logger.warning("  OCR failed: %d common seconds", len(common))
            export["meta"]["frame_alignment"] = "ocr_failed"
    else:
        logger.info("Phase 1.5: No frame alignment (use --ocr-align to enable)")
        export["meta"]["frame_alignment"] = "disabled"

    # ── Phase 2: Triangulation ─────────────────────────────────────
    chosen_pixels = {}

    if args.mode == "viterbi":
        logger.info("=== Phase 2: Viterbi Global Optimal Trajectory ===")
        from app.pipeline.homography import HomographyTransformer
        from app.pipeline.viterbi_tracker import ViterbiTracker

        homo_path = cfg["homography"]["path"]
        homo66 = HomographyTransformer(homo_path, "cam66")
        homo68 = HomographyTransformer(homo_path, "cam68")

        cam66_pos = cfg["cameras"]["cam66"]["position_3d"]
        cam68_pos = cfg["cameras"]["cam68"]["position_3d"]

        stereo_cal = load_stereo_calibration("src/camera_calibration.json")

        tracker = ViterbiTracker(
            cam1_pos=cam66_pos,
            cam2_pos=cam68_pos,
            max_ray_distance=2.5,
            valid_z_range=(0.0, 8.0),
            fps=25.0,
            gap_threshold=5,
            stereo_cal=stereo_cal,
        )

        points_3d_homo, chosen_pixels, viterbi_stats = tracker.track(
            multi66, multi68, homo66, homo68,
        )

        points_3d = {}
        stereo_ok = 0
        for fi, cp in chosen_pixels.items():
            px66, py66 = cp["cam66"]
            px68, py68 = cp["cam68"]
            result = triangulate_stereo(px66, py66, px68, py68, stereo_cal)
            if result is not None:
                x, y, z, rd = result
                if -2 < z < 10:
                    points_3d[fi] = (x, y, z, rd)
                    stereo_ok += 1
                else:
                    points_3d[fi] = points_3d_homo[fi]
            elif fi in points_3d_homo:
                points_3d[fi] = points_3d_homo[fi]

        export["meta"]["viterbi_stats"] = viterbi_stats if isinstance(viterbi_stats, dict) else {}
        export["meta"]["stereo_upgraded"] = stereo_ok

        filt66 = {}
        filt68 = {}
        matched_frames = set(points_3d.keys())
        for fi in matched_frames:
            if fi in chosen_pixels:
                cp = chosen_pixels[fi]
                filt66[fi] = (cp["cam66"][0], cp["cam66"][1], 1.0)
                filt68[fi] = (cp["cam68"][0], cp["cam68"][1], 1.0)
            else:
                if fi in det66:
                    filt66[fi] = det66[fi]
                if fi in det68:
                    filt68[fi] = det68[fi]

    elif args.mode == "multi":
        logger.info("=== Phase 2: Multi-Blob Matching ===")
        points_3d, chosen_pixels, matcher_stats = triangulate_multi_blob(multi66, multi68, cfg)
        matched_frames = set(points_3d.keys())
        filt66 = {fi: det66[fi] for fi in matched_frames if fi in det66}
        filt68 = {fi: det68[fi] for fi in matched_frames if fi in det68}
        export["meta"]["matcher_stats"] = matcher_stats

    else:
        logger.info("=== Phase 2: Top-1 Triangulation ===")
        points_3d = triangulate_detections(det66, det68, cfg)
        _, filt66, filt68 = build_flight_mask(points_3d, det66, det68, fps=25.0)

    # Save chosen pixels (which blob was selected per frame)
    export["chosen_pixels"] = {
        str(fi): {
            "cam66": [float(cp["cam66"][0]), float(cp["cam66"][1])],
            "cam68": [float(cp["cam68"][0]), float(cp["cam68"][1])],
        }
        for fi, cp in chosen_pixels.items()
    }

    # Save raw 3D points (before smoothing)
    export["points_3d"] = {
        str(fi): {"x": float(v[0]), "y": float(v[1]), "z": float(v[2]),
                  "ray_dist": float(v[3]) if len(v) > 3 else 0.0}
        for fi, v in points_3d.items()
    }

    # ── Phase 2.5: Rally Segmentation ──────────────────────────────
    rally_model_path = Path("model_weight/rally_segmentation.pkl")
    rally_frames_set = None
    if rally_model_path.exists() and not args.no_rally:
        logger.info("=== Phase 2.5: Rally Segmentation (ML model) ===")
        import pickle
        from tools.train_rally_model import extract_features, smooth_predictions

        with open(rally_model_path, "rb") as f:
            rally_bundle = pickle.load(f)

        rally_threshold = rally_bundle.get("threshold", 0.42)
        rally_scaler = rally_bundle.get("scaler", None)

        X_rally = extract_features(det66, multi66, multi68, n_frames)
        if rally_scaler is not None:
            X_rally = rally_scaler.transform(X_rally)

        models_to_avg = []
        for key in ["rf", "rf_model"]:
            if key in rally_bundle:
                models_to_avg.append(rally_bundle[key].predict_proba(X_rally)[:, 1])
                break
        for key in ["gbt", "gb_model"]:
            if key in rally_bundle:
                models_to_avg.append(rally_bundle[key].predict_proba(X_rally)[:, 1])
                break
        if "lstm_state" in rally_bundle:
            from tools.train_rally_model import BiLSTMClassifier
            import torch as _torch
            lstm_model = BiLSTMClassifier(X_rally.shape[1])
            lstm_model.load_state_dict(rally_bundle["lstm_state"])
            lstm_model.eval()
            with _torch.no_grad():
                seq = _torch.from_numpy(X_rally.astype(np.float32)).unsqueeze(0)
                lstm_probs = _torch.sigmoid(lstm_model(seq)).squeeze().numpy()
            models_to_avg.append(lstm_probs)

        if models_to_avg:
            rally_probs = np.mean(models_to_avg, axis=0)
            rally_preds = (rally_probs >= rally_threshold).astype(int)
            rally_preds = smooth_predictions(rally_preds, min_rally=5, min_gap=10)

            rally_frames_set = set(int(fi) for fi in range(n_frames) if rally_preds[fi] == 1)

            # Save rally probabilities
            export["rally_probs"] = [float(p) for p in rally_probs]
            export["rally_preds"] = [int(p) for p in rally_preds]

            pre_3d = len(points_3d)
            points_3d = {fi: v for fi, v in points_3d.items() if fi in rally_frames_set}
            filt66 = {fi: v for fi, v in filt66.items() if fi in rally_frames_set}
            filt68 = {fi: v for fi, v in filt68.items() if fi in rally_frames_set}

            logger.info("Rally filter: %d -> %d 3D points", pre_3d, len(points_3d))
    else:
        logger.info("No rally model or --no-rally, skipping rally segmentation")

    # ── Phase 3: Smoothing ─────────────────────────────────────────
    logger.info("=== Phase 3: SG Smoothing ===")
    flight_3d = {fi: points_3d[fi][:3] for fi in points_3d}
    smoothed_3d = smooth_trajectory_sg(flight_3d, window_length=11, polyorder=3)

    export["smoothed_3d"] = {
        str(fi): {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
        for fi, v in smoothed_3d.items()
    }

    # ── Phase 4: Bounce Detection ──────────────────────────────────
    logger.info("=== Phase 4: Bounce Detection ===")
    bounces = detect_bounces(smoothed_3d)

    # Net crossing detection
    net_crossings = detect_net_crossings(smoothed_3d, fps=25.0)

    # Net-crossing anchor filter
    NC_ANCHOR_WINDOW = 150
    nc_frames = [nc["frame"] for nc in net_crossings]
    if nc_frames:
        pre_filter = len(bounces)
        bounces = [
            b for b in bounces
            if any(abs(b["frame"] - ncf) <= NC_ANCHOR_WINDOW for ncf in nc_frames)
        ]
        removed = pre_filter - len(bounces)
        if removed:
            logger.info("Net-crossing anchor filter removed %d bounce(s)", removed)

    # Bounce IN/OUT refinement using homography
    from app.pipeline.homography import HomographyTransformer
    homo_path = "src/homography_matrices.json"
    homo66 = HomographyTransformer(homo_path, "cam66")
    homo68 = HomographyTransformer(homo_path, "cam68")

    MAX_Z_FOR_HOMO = 0.35
    SEARCH_RADIUS = 3

    for b in bounces:
        fi = b["frame"]
        best_fi = fi
        best_z = b["z"]
        for offset in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            check_fi = fi + offset
            if check_fi in smoothed_3d:
                cz = smoothed_3d[check_fi][2]
                if cz < best_z:
                    best_z = cz
                    best_fi = check_fi

        if best_z > MAX_Z_FOR_HOMO:
            b["cam_used"] = "3d"
            continue

        last_nc = None
        for nc in net_crossings:
            if nc["frame"] <= fi:
                last_nc = nc
            else:
                break

        if last_nc and last_nc.get("direction") == "near_to_far":
            use_cam68 = True
        elif last_nc and last_nc.get("direction") == "far_to_near":
            use_cam68 = False
        else:
            use_cam68 = b["y"] > NET_Y

        homo_fi = best_fi
        if use_cam68 and homo_fi in det68:
            px, py = det68[homo_fi][:2]
            wx, wy = homo68.pixel_to_world(px, py)
            cam_used = "cam68"
        elif not use_cam68 and homo_fi in det66:
            px, py = det66[homo_fi][:2]
            wx, wy = homo66.pixel_to_world(px, py)
            cam_used = "cam66"
        elif homo_fi in det66:
            px, py = det66[homo_fi][:2]
            wx, wy = homo66.pixel_to_world(px, py)
            cam_used = "cam66"
        elif homo_fi in det68:
            px, py = det68[homo_fi][:2]
            wx, wy = homo68.pixel_to_world(px, py)
            cam_used = "cam68"
        else:
            b["cam_used"] = "3d"
            continue

        old_in = b["in_court"]
        b["x_homo"] = float(wx)
        b["y_homo"] = float(wy)
        b["in_court"] = (SINGLES_X_MIN <= wx <= SINGLES_X_MAX and 0 <= wy <= COURT_L)
        b["cam_used"] = cam_used
        b["homo_z"] = float(best_z)

    # Shot/bounce disambiguation
    def _z_threshold_for_bounce(b):
        return 0.65 if b["y"] > NET_Y else 0.35

    bounces = [b for b in bounces
               if b["z"] <= _z_threshold_for_bounce(b)
               or b.get("homo_z", b["z"]) <= _z_threshold_for_bounce(b)]

    export["bounces"] = bounces
    export["net_crossings"] = net_crossings

    # ── 2D Pixel Bounce Detection ──────────────────────────────────
    logger.info("=== 2D Pixel Bounce Detection ===")
    bounces_2d = detect_bounces_2d(det66, det68, cfg, smoothed_3d=smoothed_3d)
    export["bounces_2d"] = bounces_2d

    # ── Compute speed over time ────────────────────────────────────
    logger.info("=== Computing speed profile ===")
    frames_sorted = sorted(smoothed_3d.keys())
    speed_profile = {}
    for i in range(1, len(frames_sorted)):
        fi_prev = frames_sorted[i - 1]
        fi_curr = frames_sorted[i]
        dt = (fi_curr - fi_prev) / 25.0
        if dt < 1e-6:
            continue
        p0 = np.array(smoothed_3d[fi_prev])
        p1 = np.array(smoothed_3d[fi_curr])
        dist = float(np.linalg.norm(p1 - p0))
        speed_ms = dist / dt
        speed_kmh = speed_ms * 3.6
        if speed_kmh < 300:  # sanity cap
            speed_profile[str(fi_curr)] = {"speed_ms": round(speed_ms, 2),
                                            "speed_kmh": round(speed_kmh, 1)}

    export["speed_profile"] = speed_profile

    # ── Summary stats ──────────────────────────────────────────────
    export["meta"]["total_3d_points"] = len(points_3d)
    export["meta"]["smoothed_points"] = len(smoothed_3d)
    export["meta"]["bounce_count"] = len(bounces)
    export["meta"]["net_crossing_count"] = len(net_crossings)
    export["meta"]["bounces_in_court"] = sum(1 for b in bounces if b["in_court"])
    export["meta"]["bounces_out"] = sum(1 for b in bounces if not b["in_court"])

    # ── Write JSON ─────────────────────────────────────────────────
    logger.info("Writing export to %s ...", args.output)
    with open(args.output, "w") as f:
        json.dump(export, f, indent=2, default=str)

    file_size_mb = Path(args.output).stat().st_size / 1024 / 1024
    logger.info("Export complete: %.1f MB, %d frames", file_size_mb, n_frames)
    logger.info("  det66=%d, det68=%d, 3D=%d, smoothed=%d",
                len(export["det66"]), len(export["det68"]),
                len(export["points_3d"]), len(export["smoothed_3d"]))
    logger.info("  bounces=%d (IN=%d, OUT=%d), net_crossings=%d",
                len(bounces),
                export["meta"]["bounces_in_court"],
                export["meta"]["bounces_out"],
                len(net_crossings))


if __name__ == "__main__":
    main()
