"""
Video-test bounce accuracy: run the pipeline on the recorded video with GT,
compare detected bounces against bounce_results.json.

This uses the SAME video files the GT was annotated on, so frame numbers match.
Tests the full pipeline: TrackNet detection → triangulation → bounce detection.
"""

import json
import os
import time
import sys
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "http://localhost:8000"
GT_FILE = "D:/tennis/blob_frame_different/bounce_results.json"

# The uploaded video files corresponding to the GT annotations
VIDEO_66 = "cam66_20260307_173403_2min.mp4"
VIDEO_68 = "cam68_20260307_173403_2min.mp4"

TOLERANCE_FRAMES = 5
TOLERANCE_XY = 1.5  # meters


def api(method, path, **kw):
    r = getattr(requests, method)(f"{BASE}{path}", timeout=120, **kw)
    r.raise_for_status()
    return r.json()


def compare_bounces(detected, gt_bounces):
    matched, missed, false_pos = [], [], []
    matched_det = set()

    for gb in gt_bounces:
        gf = gb["frame"]
        best_i, best_df = None, None
        for i, db in enumerate(detected):
            df = db.get("frame") or db.get("frame_index")
            if df is None or i in matched_det:
                continue
            if abs(df - gf) <= TOLERANCE_FRAMES:
                if best_i is None or abs(df - gf) < abs(best_df - gf):
                    best_i, best_df = i, df

        if best_i is not None:
            db = detected[best_i]
            matched_det.add(best_i)
            matched.append({
                "gt_frame": gf, "det_frame": best_df, "diff": best_df - gf,
                "gt": {"x": round(gb["x"], 3), "y": round(gb["y"], 3), "z": round(gb["z"], 3)},
                "det": {"x": round(db.get("x", 0), 3), "y": round(db.get("y", 0), 3), "z": round(db.get("z", 0), 3)},
                "gt_in_court": gb["in_court"],
            })
        else:
            missed.append({"frame": gf, "x": round(gb["x"], 3), "y": round(gb["y"], 3)})

    for i, db in enumerate(detected):
        if i not in matched_det:
            false_pos.append({
                "frame": db.get("frame") or db.get("frame_index"),
                "x": round(db.get("x", 0), 3), "y": round(db.get("y", 0), 3),
            })

    n_gt, n_det, n_m = len(gt_bounces), len(detected), len(matched)
    recall = n_m / n_gt if n_gt else 0
    precision = n_m / n_det if n_det else 0
    return {
        "gt_count": n_gt, "detected_count": n_det, "matched": n_m,
        "missed": len(missed), "false_positives": len(false_pos),
        "recall": round(recall, 3), "precision": round(precision, 3),
        "recall_pct": f"{recall:.1%}", "precision_pct": f"{precision:.1%}",
        "pass": recall >= 0.9,
        "match_details": matched, "missed_details": missed, "false_pos_details": false_pos[:20],
    }


def main():
    with open(GT_FILE) as f:
        gt = json.load(f)
    gt_bounces = gt["bounces"]
    print(f"GT: {len(gt_bounces)} bounces")

    # Step 1: Run parallel video test
    print("\n--- Starting video-test (parallel) ---")
    t0 = time.time()
    try:
        r = api("post", "/api/video-test/run-parallel", json={
            "cameras": [
                {"camera": "cam66", "filename": VIDEO_66, "start_time": 0, "end_time": 120},
                {"camera": "cam68", "filename": VIDEO_68, "start_time": 0, "end_time": 120},
            ]
        })
        print(f"  Started: {r}")
    except Exception as e:
        print(f"  Failed: {e}")
        sys.exit(1)

    # Step 2: Poll until done
    print("  Waiting for processing...")
    while True:
        time.sleep(5)
        elapsed = time.time() - t0
        try:
            st = api("get", "/api/video-test/status")
            state = st.get("state", "unknown")
            progress = st.get("progress", {})
            pct_str = ""
            for cam, p in progress.items():
                if isinstance(p, dict):
                    pct_str += f" {cam}:{p.get('processed_frames',0)}/{p.get('total_frames',0)}"
            print(f"  [{elapsed:.0f}s] state={state}{pct_str}")
            if state in ("done", "idle", "error", "stopped"):
                break
        except Exception as e:
            print(f"  [{elapsed:.0f}s] poll error: {e}")
        if elapsed > 600:
            print("  TIMEOUT after 10 min")
            break

    # Step 3: Compute 3D
    print("\n--- Computing 3D trajectory ---")
    try:
        r3d = api("post", "/api/video-test/compute-3d")
        n_pts = len(r3d.get("points", []))
        stats = r3d.get("stats", {})
        print(f"  3D points: {n_pts}")
        print(f"  Stats: common_frames={stats.get('common_frames')}")
    except Exception as e:
        print(f"  compute-3d failed: {e}")
        r3d = {"points": []}

    # Step 4: Save debug output
    print("\n--- Saving debug ---")
    dbg = api("post", "/api/debug/save")
    print(f"  Saved to: {dbg['path']}")

    # Step 5: Load debug bounces
    bounces_path = f"{dbg['path']}/bounces.json"
    try:
        with open(bounces_path) as f:
            debug_bounces = json.load(f)
    except Exception:
        debug_bounces = []

    # Also get live bounces from API
    st = api("get", "/api/status")
    api_bounces = st.get("analytics", {}).get("recent_bounces", [])

    # Step 6: Also run batch detect_bounces on the 3D points for comparison
    from app.pipeline.bounce_detect import detect_bounces
    pts = r3d.get("points", [])
    if pts:
        traj_tuples = [(p["frame_index"], p["x"], p["y"], p["z"], p.get("ray_distance", 0))
                       for p in pts if "frame_index" in p]
        batch_bounces = detect_bounces(traj_tuples)
    else:
        batch_bounces = []

    # Step 7: Compare
    print(f"\n{'='*60}")
    print(f"BOUNCE ACCURACY REPORT")
    print(f"{'='*60}")

    reports = {}

    if debug_bounces:
        reports["debug_output"] = compare_bounces(debug_bounces, gt_bounces)
        print(f"\n[debug_output] {reports['debug_output']['recall_pct']} recall "
              f"({reports['debug_output']['matched']}/{reports['debug_output']['gt_count']})")

    if api_bounces:
        reports["api_live"] = compare_bounces(api_bounces, gt_bounces)
        print(f"[api_live]     {reports['api_live']['recall_pct']} recall "
              f"({reports['api_live']['matched']}/{reports['api_live']['gt_count']})")

    if batch_bounces:
        reports["batch_findpeaks"] = compare_bounces(batch_bounces, gt_bounces)
        print(f"[batch_detect] {reports['batch_findpeaks']['recall_pct']} recall "
              f"({reports['batch_findpeaks']['matched']}/{reports['batch_findpeaks']['gt_count']})")

    best_report = max(reports.values(), key=lambda r: r["recall"]) if reports else {"recall": 0, "pass": False}

    result = {
        "test": "video_bounce_accuracy",
        "videos": {"cam66": VIDEO_66, "cam68": VIDEO_68},
        "elapsed_s": round(time.time() - t0, 1),
        "trajectory_points": len(pts),
        "reports": reports,
        "best_recall": best_report["recall"],
        "verdict": "PASS" if best_report["pass"] else "FAIL",
    }

    out_path = "debug_output/test_video_bounce_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nVerdict: {result['verdict']} (best recall={result['best_recall']:.1%})")
    print(f"Saved: {out_path}")

    return result


if __name__ == "__main__":
    main()
