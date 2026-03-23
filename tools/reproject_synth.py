"""Re-project Unity synthetic trajectory data using real camera calibration.

Unity's WorldToScreenPoint doesn't match real cameras (different cx/cy, no distortion).
This script takes the 3D positions from Unity export and projects them using
the real PnP calibration (K, rvec, tvec, dist_coeffs) for accurate pixel coords.

Usage:
    python -m tools.reproject_synth --input trajectory_20260319.json --output trajectory_reprojected.json
"""
import argparse
import json
import numpy as np
import cv2


def load_calibration(path="src/camera_calibration.json"):
    with open(path) as f:
        cal = json.load(f)

    cams = {}
    for name in ["cam66", "cam68"]:
        c = cal[name]
        cams[name] = {
            "K": np.array(c["K"], dtype=np.float64),
            "dist": np.array(c["dist_coeffs"][0], dtype=np.float64),
            "rvec": np.array(c["rvec"], dtype=np.float64).reshape(3, 1),
            "tvec": np.array(c["tvec"], dtype=np.float64).reshape(3, 1),
        }
    return cams


def project_point(cam, world_x, world_y, world_z):
    """Project a 3D world point to pixel coords using real calibration.

    World coords: x = across court, y = along court, z = height.
    """
    pt = np.array([[world_x, world_y, world_z]], dtype=np.float64)
    px, _ = cv2.projectPoints(pt, cam["rvec"], cam["tvec"], cam["K"], cam["dist"])
    return float(px[0][0][0]), float(px[0][0][1])


def reproject(input_path, output_path, cal_path="src/camera_calibration.json"):
    cams = load_calibration(cal_path)

    with open(input_path) as f:
        data = json.load(f)

    frames = data["frames"]
    print(f"Re-projecting {len(frames)} frames using real PnP calibration...")

    for fr in frames:
        bx, by, bz = fr["pos"]

        # cam66
        px66, py66 = project_point(cams["cam66"], bx, by, bz)
        visible66 = 0 <= px66 <= 1920 and 0 <= py66 <= 1080
        fr["c66"] = [round(px66, 1), round(py66, 1), 1 if visible66 else 0]

        # cam68
        px68, py68 = project_point(cams["cam68"], bx, by, bz)
        visible68 = 0 <= px68 <= 1920 and 0 <= py68 <= 1080
        fr["c68"] = [round(px68, 1), round(py68, 1), 1 if visible68 else 0]

    data["pixel_note"] = "Re-projected using real PnP calibration (K, rvec, tvec, dist_coeffs)"

    with open(output_path, "w") as f:
        json.dump(data, f)

    # Verify with court corners
    print("\nVerification — court corners:")
    test_pts = [("near_left", 0, 0, 0), ("far_right", 8.23, 23.77, 0),
                ("net_center", 4.115, 11.885, 0), ("ball_at_1m", 4.115, 11.885, 1.0)]
    for name, x, y, z in test_pts:
        px66 = project_point(cams["cam66"], x, y, z)
        px68 = project_point(cams["cam68"], x, y, z)
        print(f"  {name:20s}  cam66=({px66[0]:7.1f}, {px66[1]:7.1f})  cam68=({px68[0]:7.1f}, {px68[1]:7.1f})")

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cal", default="src/camera_calibration.json")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".json", "_reprojected.json")

    reproject(args.input, args.output, args.cal)


if __name__ == "__main__":
    main()
