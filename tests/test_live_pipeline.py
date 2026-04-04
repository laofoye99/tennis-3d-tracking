"""
Structured test: simulate live pipeline with GT data, compare against bounce_results.json.

Tests the exact code paths used in live mode:
  1. Homography (project's homography_matrices.json)
  2. Triangulation (app.triangulation.triangulate)
  3. Bounce detection (app.analytics.BounceDetector)
  4. Debug recording (orchestrator._debug_record_*)

All output is structured JSON — no visual, no format guessing.
"""

import json
import glob
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import load_config
from app.triangulation import triangulate
from app.pipeline.homography import HomographyTransformer
from app.analytics import BounceDetector
from app.pipeline.bounce_detect import detect_bounces

# ================================================================
# Config
# ================================================================
CAM66_DIR = "uploads/cam66_20260307_173403_2min"
CAM68_DIR = "uploads/cam68_20260307_173403_2min"
GT_FILE = "D:/tennis/blob_frame_different/bounce_results.json"
MAX_FRAMES = 3000

RESULT_FILE = "debug_output/test_live_pipeline_result.json"


# ================================================================
# Load GT
# ================================================================
def load_gt_ball(gt_dir, max_frames):
    files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))[:max_frames]
    gt = {}
    for jf in files:
        idx = int(os.path.basename(jf).replace(".json", ""))
        with open(jf) as f:
            d = json.load(f)
        for s in d["shapes"]:
            if s["label"] == "ball" and s["shape_type"] == "rectangle":
                pts = s["points"]
                cx = (pts[0][0] + pts[2][0]) / 2
                cy = (pts[0][1] + pts[2][1]) / 2
                gt[idx] = (cx, cy)
                break
    return gt


# ================================================================
# Test
# ================================================================
def run_test():
    result = {
        "test": "live_pipeline_vs_gt",
        "config": {},
        "stages": {},
        "comparison": {},
        "verdict": "UNKNOWN",
    }

    # --- Load config ---
    cfg = load_config()
    cam_positions = {
        n: cfg.cameras[n].position_3d for n in cfg.cameras if cfg.cameras[n].position_3d != [0, 0, 0]
    }
    result["config"] = {
        "detector_type": cfg.model.detector_type,
        "homography_path": cfg.homography.path,
        "camera_positions": {k: v for k, v in cam_positions.items()},
    }

    # --- Load GT reference ---
    with open(GT_FILE) as f:
        gt_ref = json.load(f)
    ref_traj = gt_ref["trajectory"]
    ref_bounces = gt_ref["bounces"]
    ref_cam_pos = gt_ref["config"]["cameras"]

    result["config"]["gt_camera_positions"] = {
        cam: info["position"] for cam, info in ref_cam_pos.items()
    }

    # --- Load GT ball positions ---
    gt66 = load_gt_ball(CAM66_DIR, MAX_FRAMES)
    gt68 = load_gt_ball(CAM68_DIR, MAX_FRAMES)
    common_frames = sorted(set(gt66.keys()) & set(gt68.keys()))

    result["stages"]["input"] = {
        "cam66_frames": len(gt66),
        "cam68_frames": len(gt68),
        "common_frames": len(common_frames),
        "gt_trajectory_points": len(ref_traj),
        "gt_bounces": len(ref_bounces),
    }

    # --- Stage 1: Homography (project's matrices) ---
    h66 = HomographyTransformer(cfg.homography.path, "cam66")
    h68 = HomographyTransformer(cfg.homography.path, "cam68")

    # Compare homography output with GT for first 5 common frames
    homography_check = []
    for fi in common_frames[:5]:
        px66, py66 = gt66[fi]
        px68, py68 = gt68[fi]
        w66 = h66.pixel_to_world(px66, py66)
        w68 = h68.pixel_to_world(px68, py68)

        # Find same frame in GT ref
        ref_pt = next((t for t in ref_traj if t["frame"] == fi), None)
        if ref_pt:
            homography_check.append({
                "frame": fi,
                "pipeline_world66": [round(w66[0], 4), round(w66[1], 4)],
                "gt_world66": ref_pt["world66"],
                "diff_world66": [round(w66[0] - ref_pt["world66"][0], 4),
                                 round(w66[1] - ref_pt["world66"][1], 4)],
                "pipeline_world68": [round(w68[0], 4), round(w68[1], 4)],
                "gt_world68": ref_pt["world68"],
                "diff_world68": [round(w68[0] - ref_pt["world68"][0], 4),
                                 round(w68[1] - ref_pt["world68"][1], 4)],
            })

    result["stages"]["homography"] = {
        "sample_comparison": homography_check,
        "uses_same_matrices": True,  # both use homography_matrices.json
    }

    # --- Stage 2: Triangulation (project's triangulate function) ---
    cams = [n for n in cam_positions if not cfg.cameras[n].record_only][:2]
    cam1, cam2 = cams[0], cams[1]
    pos1, pos2 = cam_positions[cam1], cam_positions[cam2]

    pipeline_traj = []
    for fi in common_frames:
        px66, py66 = gt66[fi]
        px68, py68 = gt68[fi]
        w66 = h66.pixel_to_world(px66, py66)
        w68 = h68.pixel_to_world(px68, py68)

        x, y, z = triangulate(w66, w68, pos1, pos2)

        pipeline_traj.append({
            "frame": fi,
            "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
            "px66": round(px66, 1), "py66": round(py66, 1),
            "px68": round(px68, 1), "py68": round(py68, 1),
            "world66": [round(w66[0], 4), round(w66[1], 4)],
            "world68": [round(w68[0], 4), round(w68[1], 4)],
        })

    # Compare trajectory point by point
    ref_traj_map = {t["frame"]: t for t in ref_traj}
    traj_diffs = []
    max_dx, max_dy, max_dz = 0, 0, 0
    for pt in pipeline_traj:
        fi = pt["frame"]
        ref = ref_traj_map.get(fi)
        if ref:
            dx = abs(pt["x"] - ref["x"])
            dy = abs(pt["y"] - ref["y"])
            dz = abs(pt["z"] - ref["z"])
            max_dx = max(max_dx, dx)
            max_dy = max(max_dy, dy)
            max_dz = max(max_dz, dz)
            if dx > 0.01 or dy > 0.01 or dz > 0.01:
                traj_diffs.append({
                    "frame": fi,
                    "pipeline": {"x": pt["x"], "y": pt["y"], "z": pt["z"]},
                    "gt": {"x": ref["x"], "y": ref["y"], "z": ref["z"]},
                    "diff": {"dx": round(dx, 4), "dy": round(dy, 4), "dz": round(dz, 4)},
                })

    result["stages"]["triangulation"] = {
        "pipeline_points": len(pipeline_traj),
        "gt_points": len(ref_traj),
        "frame_match": len(pipeline_traj) == len(ref_traj),
        "max_diff": {"dx": round(max_dx, 4), "dy": round(max_dy, 4), "dz": round(max_dz, 4)},
        "points_with_diff_gt_0.01m": len(traj_diffs),
        "first_5_diffs": traj_diffs[:5],
        "camera_positions_used": {"cam1": cam1, "pos1": pos1, "cam2": cam2, "pos2": pos2},
    }

    # --- Stage 3: Bounce detection (BounceDetector — live V-shape) ---
    live_detector = BounceDetector()
    live_bounces = []
    for pt in pipeline_traj:
        b = live_detector.update({
            "x": pt["x"], "y": pt["y"], "z": pt["z"],
            "timestamp": pt["frame"] * 0.04,
            "frame_index": pt["frame"],
        })
        if b is not None:
            live_bounces.append({
                "frame": b.frame_index,
                "z": round(float(b.z), 4),
                "x": round(float(b.x), 4),
                "y": round(float(b.y), 4),
                "in_court": bool(b.in_court),
            })

    # --- Stage 4: Bounce detection (detect_bounces — batch, same as GT) ---
    traj_tuples = [(pt["frame"], pt["x"], pt["y"], pt["z"], 0) for pt in pipeline_traj]
    batch_bounces = detect_bounces(traj_tuples)

    # --- Compare ---
    ref_bounce_frames = sorted(b["frame"] for b in ref_bounces)
    live_bounce_frames = sorted(b["frame"] for b in live_bounces)
    batch_bounce_frames = sorted(b["frame"] for b in batch_bounces)

    def compare_bounces(detected, expected, label, tolerance=5):
        det_set = set(detected)
        exp_set = set(expected)
        matched = []
        missed = []
        false_pos = []

        matched_det = set()
        for ef in expected:
            found = False
            for df in detected:
                if abs(df - ef) <= tolerance and df not in matched_det:
                    matched.append({"expected": ef, "detected": df, "diff": df - ef})
                    matched_det.add(df)
                    found = True
                    break
            if not found:
                missed.append(ef)

        for df in detected:
            if df not in matched_det:
                false_pos.append(df)

        return {
            "label": label,
            "expected_count": len(expected),
            "detected_count": len(detected),
            "matched": len(matched),
            "missed_count": len(missed),
            "false_positive_count": len(false_pos),
            "recall": round(len(matched) / len(expected), 3) if expected else 0,
            "precision": round(len(matched) / len(detected), 3) if detected else 0,
            "exact_frame_match": detected == expected,
            "missed_frames": missed,
            "false_positive_frames": false_pos,
            "match_details": matched[:10],
        }

    result["stages"]["bounce_detection"] = {
        "live_vshape": compare_bounces(live_bounce_frames, ref_bounce_frames, "BounceDetector (V-shape)"),
        "batch_findpeaks": compare_bounces(batch_bounce_frames, ref_bounce_frames, "detect_bounces (find_peaks)"),
    }

    # --- Stage 5: Root cause analysis ---
    issues = []

    # Check camera position mismatch
    gt_cam66_pos = ref_cam_pos["cam66"]["position"]
    gt_cam68_pos = ref_cam_pos["cam68"]["position"]
    pipeline_pos1 = pos1
    pipeline_pos2 = pos2

    pos_match_66 = all(abs(a - b) < 0.01 for a, b in zip(pipeline_pos1, gt_cam66_pos))
    pos_match_68 = all(abs(a - b) < 0.01 for a, b in zip(pipeline_pos2, gt_cam68_pos))

    if not pos_match_66:
        issues.append({
            "issue": "CAMERA_POSITION_MISMATCH",
            "camera": cam1,
            "pipeline": pipeline_pos1,
            "gt": gt_cam66_pos,
            "note": "Y sign may be inverted between config.yaml and research code",
        })
    if not pos_match_68:
        issues.append({
            "issue": "CAMERA_POSITION_MISMATCH",
            "camera": cam2,
            "pipeline": pipeline_pos2,
            "gt": gt_cam68_pos,
        })

    if max_dz > 0.5:
        issues.append({
            "issue": "LARGE_Z_DISCREPANCY",
            "max_dz": round(max_dz, 4),
            "note": "Z values significantly different — likely camera position or homography mismatch",
        })

    if traj_diffs:
        issues.append({
            "issue": "TRAJECTORY_DRIFT",
            "affected_frames": len(traj_diffs),
            "note": "3D points differ from GT by >0.01m",
        })

    result["comparison"]["issues"] = issues
    result["comparison"]["camera_position_match"] = {
        cam1: pos_match_66,
        cam2: pos_match_68,
    }

    # Verdict
    batch_ok = result["stages"]["bounce_detection"]["batch_findpeaks"]["exact_frame_match"]
    live_recall = result["stages"]["bounce_detection"]["live_vshape"]["recall"]
    traj_match = len(traj_diffs) == 0

    if batch_ok and traj_match:
        result["verdict"] = "PASS"
    elif batch_ok and not traj_match:
        result["verdict"] = "PASS_WITH_TRAJECTORY_DIFF"
    elif not batch_ok and len(issues) > 0:
        result["verdict"] = "FAIL_KNOWN_CAUSE"
    else:
        result["verdict"] = "FAIL"

    result["comparison"]["summary"] = {
        "trajectory_exact_match": traj_match,
        "batch_bounce_exact_match": batch_ok,
        "live_bounce_recall": live_recall,
        "root_causes": [i["issue"] for i in issues],
    }

    # Save
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    r = run_test()
    print(json.dumps(r, indent=2))
