"""
Test: Load GT from both cameras → 3D trajectory → detect bounces → push to WebSocket 3D display.

Mapping: V2 coords (origin at court center, net at y=0)
  x: [-4.115, +4.115]  →  WS x: [0, 1]  where x=0 is left, x=1 is right
  y: [-11.89, +11.89]  →  WS y: [0, 1]  where y=0 is far baseline, y=1 is near baseline
"""

import asyncio
import glob
import json
import os
import ssl
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.bounce_detect import detect_bounces, detect_events

# ================================================================
# Config
# ================================================================
CAM66_DIR = "uploads/cam66_20260307_173403_2min"
CAM68_DIR = "uploads/cam68_20260307_173403_2min"
WS_URL = "wss://tennisserver.motionrivalry.com:8086/general"
MAX_FRAMES = 3000

# V2 court
HW = 4.115
HL = 11.89

# Camera positions (from bounce_detector.py research — verified correct)
CAM66_POS = np.array([0.165, -17.042, 6.217])
CAM68_POS = np.array([0.211, 17.156, 5.286])

# Homography keypoints
CAM66_KP = np.array([
    [760, 62],  [935, 58],  [1110, 59],
    [723, 122], [934, 117], [1145, 119],
    [529, 461], [928, 456], [1326, 456],
    [329, 860], [917, 881], [1507, 862]
], dtype=np.float32)
CAM68_KP = np.array([
    [1640, 949], [925, 954], [168, 959],
    [1362, 510], [922, 506], [466, 505],
    [1143, 182], [916, 180], [584, 176],
    [1103, 122], [914, 120], [722, 117]
], dtype=np.float32)
WORLD_KP = np.array([
    [-4.115, 11.89],  [0., 11.89], [4.115, 11.89],
    [-4.115, 6.4],    [0., 6.4],   [4.115, 6.4],
    [-4.115, -6.4],   [0., -6.4],  [4.115, -6.4],
    [-4.115, -11.89], [0., -11.89], [4.115, -11.89]
], dtype=np.float32)


# ================================================================
# Helpers
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


def pixel_to_world(px, py, H):
    p = np.array([px, py, 1.0])
    r = H @ p
    return float(r[0] / r[2]), float(r[1] / r[2])


def triangulate(w1, w2):
    cam1, cam2 = CAM66_POS, CAM68_POS
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
    d1, d2 = g1 - cam1, g2 - cam2
    w = cam1 - cam2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d_v, e = float(d1 @ w), float(d2 @ w)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return (0, 0, 0), 999.0
    s = np.clip((b * e - c * d_v) / denom, 0, 1)
    t = np.clip((a * e - b * d_v) / denom, 0, 1)
    p1 = cam1 + s * d1
    t = float((p1 - cam2) @ d2) / c if c > 1e-10 else t
    t = np.clip(t, 0, 1)
    p2 = cam2 + t * d2
    s = float((p2 - cam1) @ d1) / a if a > 1e-10 else s
    s = np.clip(s, 0, 1)
    p1, p2 = cam1 + s * d1, cam2 + t * d2
    mid = (p1 + p2) / 2
    rd = float(np.linalg.norm(p1 - p2))
    if mid[2] < 0:
        mid[2] = 0
    return (float(mid[0]), float(mid[1]), float(mid[2])), rd


def v2_to_ws(x, y):
    """V2 world coords → WebSocket values (×10).

    V2: x in [-4.115, +4.115], y in [-11.89, +11.89], net at y=0
    WS: x in [-41.15, 41.15], y in [-118.9, 118.9]
    """
    return round(x * 10, 4), round(y * 10, 4)


# ================================================================
# Main
# ================================================================

def main():
    print("=== 3D Push Test ===\n")

    # Step 1: Load GT
    gt66 = load_gt_ball(CAM66_DIR, MAX_FRAMES)
    gt68 = load_gt_ball(CAM68_DIR, MAX_FRAMES)
    print(f"GT: cam66={len(gt66)} frames, cam68={len(gt68)} frames")

    # Step 2: Build 3D trajectory
    H66, _ = cv2.findHomography(CAM66_KP, WORLD_KP, cv2.RANSAC, 5.0)
    H68, _ = cv2.findHomography(CAM68_KP, WORLD_KP, cv2.RANSAC, 5.0)

    common = sorted(set(gt66.keys()) & set(gt68.keys()))
    trajectory = []
    for fi in common:
        px66, py66 = gt66[fi]
        px68, py68 = gt68[fi]
        w66 = pixel_to_world(px66, py66, H66)
        w68 = pixel_to_world(px68, py68, H68)
        pt3d, rd = triangulate(w66, w68)
        trajectory.append((fi, pt3d[0], pt3d[1], pt3d[2], rd))

    print(f"3D trajectory: {len(trajectory)} points")
    if trajectory:
        zs = [t[3] for t in trajectory]
        xs = [t[1] for t in trajectory]
        ys = [t[2] for t in trajectory]
        print(f"  x: [{min(xs):.2f}, {max(xs):.2f}]")
        print(f"  y: [{min(ys):.2f}, {max(ys):.2f}]")
        print(f"  z: [{min(zs):.2f}, {max(zs):.2f}]")

    # Step 3: Detect bounces
    bounces = detect_bounces(trajectory)
    print(f"\nBounces: {len(bounces)}")
    for b in bounces[:5]:
        wx, wy = v2_to_ws(b["x"], b["y"])
        print(f"  frame={b['frame']:>5d}  V2=({b['x']:>6.2f}, {b['y']:>6.2f})  "
              f"WS=({wx:.4f}, {wy:.4f})  z={b['z']:.3f}  {'IN' if b['in_court'] else 'OUT'}")

    # Step 4: Verify mapping sanity
    print("\nMapping sanity checks:")
    for label, x, y in [("Center", 0, 0), ("Near baseline", 0, -HL), ("Far baseline", 0, HL),
                         ("Left", -HW, 0), ("Right", HW, 0)]:
        wx, wy = v2_to_ws(x, y)
        print(f"  {label:20s}: V2=({x:>6.2f}, {y:>6.2f}) → WS=({wx:.4f}, {wy:.4f})")

    # Step 5: Push to WebSocket
    print(f"\n--- Pushing {len(bounces)} bounces to {WS_URL} ---")
    asyncio.run(push_bounces(bounces))


async def push_bounces(bounces):
    import websockets

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        async with websockets.connect(WS_URL, ssl=ssl_ctx) as ws:
            print("Connected!")
            for b in bounces:
                wx, wy = v2_to_ws(b["x"], b["y"])
                msg = json.dumps({
                    "msg": {
                        "message": "bounce_data",
                        "data": {
                            "bounce": {
                                "timeStamp": int(time.time() * 1000),
                                "x": wx,
                                "y": wy,
                                "speed": 0,
                            }
                        }
                    }
                })
                await ws.send(msg)
                print(f"  Sent: frame={b['frame']} WS=({wx:.4f}, {wy:.4f}) "
                      f"{'IN' if b['in_court'] else 'OUT'}")
                await asyncio.sleep(0.3)  # pace for visual review
            print("Done!")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    main()
