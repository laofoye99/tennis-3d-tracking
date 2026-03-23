"""Compare real 3D tracking data vs Unity synthetic data.

Extracts statistical features from both and shows where they differ.
This tells you exactly what to fix in the simulation.

Usage:
    python -m tools.compare_real_vs_synth --real exports/tracking_cache.json --synth exports/unity_frames.json
"""
import argparse
import json
import numpy as np
from pathlib import Path


# ─── Court constants ───
NET_Y = 11.885
COURT_L = 23.77


def load_real_data(path: str) -> dict:
    """Load real tracking pipeline output."""
    with open(path) as f:
        data = json.load(f)

    points_3d = {}
    bounces = []

    for item in data.get("frames", []):
        fi = item["frame"]
        if item.get("x") is not None:
            points_3d[fi] = {
                "x": item["x"], "y": item["y"], "z": item["z"],
                "frame": fi,
            }

    for b in data.get("bounces", []):
        bounces.append(b)

    return {"points_3d": points_3d, "bounces": bounces, "fps": data.get("fps", 25)}


def load_synth_data(path: str) -> dict:
    """Load Unity simulation exported frames."""
    with open(path) as f:
        data = json.load(f)
    return data


# ─── Feature extraction ───

def extract_real_features(data: dict) -> dict:
    """Extract statistical features from real tracking data."""
    pts = data["points_3d"]
    fps = data["fps"]
    frames = sorted(pts.keys())

    features = {}

    # 1. Ball speed distribution (m/s)
    speeds = []
    for i in range(1, len(frames)):
        f0, f1 = frames[i-1], frames[i]
        if f1 - f0 > 3:  # skip gaps
            continue
        dt = (f1 - f0) / fps
        p0, p1 = pts[f0], pts[f1]
        dx = p1["x"] - p0["x"]
        dy = p1["y"] - p0["y"]
        dz = p1["z"] - p0["z"]
        spd = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        if 1 < spd < 80:  # filter outliers
            speeds.append(spd)

    features["speed_mean"] = float(np.mean(speeds)) if speeds else 0
    features["speed_std"] = float(np.std(speeds)) if speeds else 0
    features["speed_median"] = float(np.median(speeds)) if speeds else 0
    features["speed_p90"] = float(np.percentile(speeds, 90)) if speeds else 0

    # 2. Height distribution
    heights = [pts[f]["z"] for f in frames]
    features["height_mean"] = float(np.mean(heights))
    features["height_std"] = float(np.std(heights))
    features["height_max"] = float(np.max(heights))

    # 3. Bounce z-values
    bounce_zs = [b.get("z", 0) for b in data["bounces"]]
    features["bounce_z_mean"] = float(np.mean(bounce_zs)) if bounce_zs else 0
    features["bounce_count"] = len(data["bounces"])

    # 4. Net crossing speed (from consecutive frames crossing y=NET_Y)
    net_speeds = []
    for i in range(1, len(frames)):
        f0, f1 = frames[i-1], frames[i]
        if f1 - f0 > 3:
            continue
        y0, y1 = pts[f0]["y"], pts[f1]["y"]
        if (y0 < NET_Y and y1 > NET_Y) or (y0 > NET_Y and y1 < NET_Y):
            dt = (f1 - f0) / fps
            p0, p1 = pts[f0], pts[f1]
            dx = p1["x"] - p0["x"]
            dy = p1["y"] - p0["y"]
            dz = p1["z"] - p0["z"]
            spd = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            if 5 < spd < 80:
                net_speeds.append(spd)

    features["net_cross_speed_mean"] = float(np.mean(net_speeds)) if net_speeds else 0
    features["net_cross_speed_std"] = float(np.std(net_speeds)) if net_speeds else 0
    features["net_cross_count"] = len(net_speeds)

    # 5. Landing position distribution (from bounces)
    if data["bounces"]:
        bx = [b["x"] for b in data["bounces"]]
        by = [b["y"] for b in data["bounces"]]
        features["landing_x_mean"] = float(np.mean(bx))
        features["landing_x_std"] = float(np.std(bx))
        features["landing_y_mean"] = float(np.mean(by))
        features["landing_y_std"] = float(np.std(by))
    else:
        features["landing_x_mean"] = 0
        features["landing_x_std"] = 0
        features["landing_y_mean"] = 0
        features["landing_y_std"] = 0

    # 6. Trajectory arc height (max z between two net crossings)
    # Find segments between bounces
    arc_heights = []
    for i in range(len(data["bounces"]) - 1):
        b0 = data["bounces"][i]
        b1 = data["bounces"][i + 1]
        f0_frame = b0["frame"]
        f1_frame = b1["frame"]
        segment_heights = [
            pts[f]["z"] for f in frames
            if f0_frame <= f <= f1_frame and f in pts
        ]
        if segment_heights:
            arc_heights.append(max(segment_heights))

    features["arc_height_mean"] = float(np.mean(arc_heights)) if arc_heights else 0
    features["arc_height_std"] = float(np.std(arc_heights)) if arc_heights else 0

    # 7. Time between bounces (rally rhythm)
    if len(data["bounces"]) >= 2:
        bounce_frames = [b["frame"] for b in data["bounces"]]
        intervals = [(bounce_frames[i+1] - bounce_frames[i]) / fps
                      for i in range(len(bounce_frames) - 1)
                      if bounce_frames[i+1] - bounce_frames[i] < 100]
        features["bounce_interval_mean"] = float(np.mean(intervals)) if intervals else 0
        features["bounce_interval_std"] = float(np.std(intervals)) if intervals else 0
    else:
        features["bounce_interval_mean"] = 0
        features["bounce_interval_std"] = 0

    # 8. Horizontal speed (XY only, ignoring Z)
    h_speeds = []
    for i in range(1, len(frames)):
        f0, f1 = frames[i-1], frames[i]
        if f1 - f0 > 3:
            continue
        dt = (f1 - f0) / fps
        p0, p1 = pts[f0], pts[f1]
        dx = p1["x"] - p0["x"]
        dy = p1["y"] - p0["y"]
        spd = np.sqrt(dx**2 + dy**2) / dt
        if 1 < spd < 80:
            h_speeds.append(spd)

    features["h_speed_mean"] = float(np.mean(h_speeds)) if h_speeds else 0
    features["h_speed_std"] = float(np.std(h_speeds)) if h_speeds else 0

    return features


def print_comparison(real_feat: dict, synth_feat: dict = None):
    """Print features table, highlighting differences."""
    print(f"\n{'Feature':<30} {'Real':>10}", end="")
    if synth_feat:
        print(f" {'Synth':>10} {'Delta':>10} {'Status':>8}")
    else:
        print()

    print("-" * 70)

    for key in sorted(real_feat.keys()):
        rv = real_feat[key]
        print(f"{key:<30} {rv:>10.2f}", end="")

        if synth_feat and key in synth_feat:
            sv = synth_feat[key]
            delta = sv - rv
            # Determine if difference is significant
            ref = max(abs(rv), 1.0)
            pct = abs(delta) / ref * 100
            status = "OK" if pct < 20 else "WARN" if pct < 50 else "BAD"
            print(f" {sv:>10.2f} {delta:>+10.2f} {status:>8}")
        else:
            print()

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="exports/tracking_cache.json")
    parser.add_argument("--synth", default=None, help="Unity export JSON (optional)")
    args = parser.parse_args()

    print("=== Real Data Features ===")
    real_data = load_real_data(args.real)
    real_feat = extract_real_features(real_data)
    print_comparison(real_feat)

    print("\nThese are your TARGET values. Tune Unity simulation until synth matches.")
    print("\nKey parameters to adjust in Unity:")
    print(f"  Ball speed:     aim for mean={real_feat['speed_mean']:.1f} m/s, "
          f"median={real_feat['speed_median']:.1f} m/s")
    print(f"  Arc height:     aim for mean={real_feat['arc_height_mean']:.1f}m")
    print(f"  Net cross speed: aim for mean={real_feat['net_cross_speed_mean']:.1f} m/s "
          f"({real_feat['net_cross_speed_mean']*3.6:.0f} km/h)")
    print(f"  Bounce interval: aim for mean={real_feat['bounce_interval_mean']:.2f}s")
    print(f"  Landing spread:  x_std={real_feat['landing_x_std']:.1f}m, "
          f"y_std={real_feat['landing_y_std']:.1f}m")


if __name__ == "__main__":
    main()
