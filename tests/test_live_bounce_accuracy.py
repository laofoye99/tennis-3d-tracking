"""
Automated live bounce accuracy test.

1. Start cameras via API
2. Collect data until enough bounces are detected
3. Save debug output
4. Compare against GT bounce_results.json
5. Output structured accuracy report

Runs headless — no visual, no browser.
"""

import json
import time
import sys
import requests

BASE = "http://localhost:8000"
GT_FILE = "D:/tennis/blob_frame_different/bounce_results.json"
MIN_BOUNCES = 5          # minimum bounces before evaluating
MAX_WAIT_S = 300         # max wait time (5 min)
POLL_INTERVAL_S = 3      # poll every 3s
TOLERANCE_FRAMES = 5     # frame tolerance for bounce matching
TOLERANCE_XY = 1.0       # meters tolerance for position matching


def api(method, path, **kw):
    r = getattr(requests, method)(f"{BASE}{path}", **kw)
    r.raise_for_status()
    return r.json()


def load_gt():
    with open(GT_FILE) as f:
        d = json.load(f)
    return d["bounces"], d["trajectory"]


def compare_bounces(detected, gt_bounces):
    """Compare detected bounces against GT. Returns structured result."""
    gt_frames = sorted(b["frame"] for b in gt_bounces)
    det_frames = sorted(b.get("frame") or b.get("frame_index") for b in detected)

    matched = []
    missed = []
    false_pos = []
    matched_det = set()

    for gb in gt_bounces:
        gf = gb["frame"]
        best_d, best_b = None, None
        for i, db in enumerate(detected):
            df = db.get("frame") or db.get("frame_index")
            if df is None:
                continue
            if i in matched_det:
                continue
            if abs(df - gf) <= TOLERANCE_FRAMES:
                if best_d is None or abs(df - gf) < abs(best_d - gf):
                    best_d = df
                    best_b = (i, db)

        if best_b is not None:
            i, db = best_b
            matched_det.add(i)
            dx = abs(gb["x"] - db.get("x", 0))
            dy = abs(gb["y"] - db.get("y", 0))
            pos_ok = dx <= TOLERANCE_XY and dy <= TOLERANCE_XY
            matched.append({
                "gt_frame": gf,
                "det_frame": best_d,
                "frame_diff": best_d - gf,
                "gt_pos": {"x": gb["x"], "y": gb["y"]},
                "det_pos": {"x": round(db.get("x", 0), 4), "y": round(db.get("y", 0), 4)},
                "dx": round(dx, 3),
                "dy": round(dy, 3),
                "position_ok": pos_ok,
                "gt_in_court": gb["in_court"],
                "det_in_court": db.get("in_court"),
            })
        else:
            missed.append({"frame": gf, "x": gb["x"], "y": gb["y"], "in_court": gb["in_court"]})

    for i, db in enumerate(detected):
        if i not in matched_det:
            false_pos.append({
                "frame": db.get("frame") or db.get("frame_index"),
                "x": round(db.get("x", 0), 4),
                "y": round(db.get("y", 0), 4),
            })

    n_gt = len(gt_bounces)
    n_det = len(detected)
    n_match = len(matched)
    recall = n_match / n_gt if n_gt else 0
    precision = n_match / n_det if n_det else 0

    return {
        "gt_count": n_gt,
        "detected_count": n_det,
        "matched": n_match,
        "missed": len(missed),
        "false_positives": len(false_pos),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "recall_pct": f"{recall:.1%}",
        "precision_pct": f"{precision:.1%}",
        "pass": recall >= 0.9,
        "match_details": matched,
        "missed_details": missed[:10],
        "false_pos_details": false_pos[:10],
    }


def main():
    gt_bounces, gt_traj = load_gt()
    print(f"GT: {len(gt_bounces)} bounces, {len(gt_traj)} trajectory points")

    # Start cameras
    print("\n--- Starting cameras ---")
    try:
        api("post", "/api/pipeline/cam66/start")
        api("post", "/api/pipeline/cam68/start")
    except Exception as e:
        print(f"Failed to start cameras: {e}")
        sys.exit(1)

    print("Cameras started. Waiting for bounces...")

    # Poll for bounces
    start_time = time.time()
    last_bounce_count = 0
    stale_count = 0

    while time.time() - start_time < MAX_WAIT_S:
        time.sleep(POLL_INTERVAL_S)
        elapsed = time.time() - start_time

        try:
            status = api("get", "/api/status")
        except Exception as e:
            print(f"  [{elapsed:.0f}s] Poll error: {e}")
            continue

        pipelines = status.get("pipelines", {})
        fps66 = pipelines.get("cam66", {}).get("fps", 0)
        fps68 = pipelines.get("cam68", {}).get("fps", 0)

        analytics = status.get("analytics", {})
        bounces = analytics.get("recent_bounces", [])
        rally = analytics.get("rally_state", {}).get("state", "?")
        ball3d = status.get("latest_ball_3d")

        n_bounces = len(bounces)
        ball_str = ""
        if ball3d:
            ball_str = f"ball=({ball3d['x']:.1f},{ball3d['y']:.1f},{ball3d['z']:.1f})"

        print(f"  [{elapsed:.0f}s] fps={fps66:.1f}/{fps68:.1f} "
              f"bounces={n_bounces} rally={rally} {ball_str}")

        if n_bounces == last_bounce_count:
            stale_count += 1
        else:
            stale_count = 0
            last_bounce_count = n_bounces

        # Check if we have enough bounces
        if n_bounces >= MIN_BOUNCES:
            print(f"\n  Got {n_bounces} bounces. Evaluating...")
            break

        # If stale for too long (60s no new bounces after first one), also evaluate
        if n_bounces > 0 and stale_count >= 20:
            print(f"\n  Stale for {stale_count * POLL_INTERVAL_S}s. Evaluating with {n_bounces} bounces...")
            break

    # Save debug output
    print("\n--- Saving debug output ---")
    dbg = api("post", "/api/debug/save")
    print(f"Saved to: {dbg['path']}")

    # Get final bounce data
    status = api("get", "/api/status")
    live_bounces = status.get("analytics", {}).get("recent_bounces", [])

    # Stop cameras
    print("\n--- Stopping cameras ---")
    api("post", "/api/pipeline/cam66/stop")
    api("post", "/api/pipeline/cam68/stop")

    # Compare with GT
    print(f"\n--- Accuracy Report ({len(live_bounces)} detected vs {len(gt_bounces)} GT) ---")
    report = compare_bounces(live_bounces, gt_bounces)

    # Load debug trajectory for additional analysis
    try:
        with open(f"{dbg['path']}/trajectory_3d.json") as f:
            debug_traj = json.load(f)
        with open(f"{dbg['path']}/summary.json") as f:
            debug_summary = json.load(f)
        report["debug"] = {
            "path": dbg["path"],
            "trajectory_points": len(debug_traj),
            "summary": debug_summary,
        }
    except Exception:
        pass

    # Full result
    result = {
        "test": "live_bounce_accuracy",
        "elapsed_s": round(time.time() - start_time, 1),
        "config": {
            "min_bounces": MIN_BOUNCES,
            "max_wait_s": MAX_WAIT_S,
            "tolerance_frames": TOLERANCE_FRAMES,
            "tolerance_xy_m": TOLERANCE_XY,
        },
        "accuracy": report,
        "verdict": "PASS" if report["pass"] else "FAIL",
    }

    # Save result
    out_path = "debug_output/test_live_bounce_accuracy.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
