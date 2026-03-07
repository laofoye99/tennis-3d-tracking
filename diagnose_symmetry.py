"""Diagnostic script: verify homography and triangulation for symmetry issue.

Run after compute-trajectory to analyze whether X coordinates are mirrored.
Usage: python diagnose_symmetry.py
"""

import json
import numpy as np

# Load homography matrices
with open("src/homography_matrices.json") as f:
    hdata = json.load(f)

H66 = np.array(hdata["cam66"]["H_image_to_world"])
H68 = np.array(hdata["cam68"]["H_image_to_world"])

cam66_pos = [4.094, -5.21, 6.2]
cam68_pos = [4.115, 28.97, 5.2]

def pixel_to_world(H, px, py):
    pt = H @ np.array([px, py, 1.0])
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def triangulate(w1, w2, cam1_pos, cam2_pos):
    c1 = np.asarray(cam1_pos, dtype=np.float64)
    c2 = np.asarray(cam2_pos, dtype=np.float64)
    g1 = np.array([w1[0], w1[1], 0.0])
    g2 = np.array([w2[0], w2[1], 0.0])
    d1, d2 = g1 - c1, g2 - c2
    w = c1 - c2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d_val = np.dot(d1, w)
    e = np.dot(d2, w)
    denom = a * c - b * b
    if abs(denom) < 1e-10:
        return None, 99.0
    s = (b * e - c * d_val) / denom
    t = (a * e - b * d_val) / denom
    s, t = np.clip(s, 0, 1), np.clip(t, 0, 1)
    p1f = c1 + s * d1
    t_val = float(np.dot(p1f - c2, d2)) / c if c > 1e-10 else t
    t_val = np.clip(t_val, 0, 1)
    p2f = c2 + t_val * d2
    s = float(np.dot(p2f - c1, d1)) / a if a > 1e-10 else s
    s = np.clip(s, 0, 1)
    p1 = c1 + s * d1
    p2 = c2 + t_val * d2
    mid = (p1 + p2) / 2.0
    rd = float(np.linalg.norm(p1 - p2))
    return mid, rd


print("=" * 70)
print("Test 1: Known court keypoints")
print("=" * 70)

# Test: left_bottom_serve (cam66 pixel -> world)
# cam66: pixel (536, 459) -> world (0, 5.485)
# cam68: pixel (472, 504) -> world (8.23, 18.285)  <- same physical point!
# Wait, these are NOT the same physical point.
# cam66's left_bottom_serve = world (0, 5.485) [near baseline, left]
# cam68's left_bottom_serve = world (8.23, 18.285) [from cam68's view: left=far-right of cam66, bottom=far end]

# Let's test with the center_bottom_serve which is on the center line
print("\ncenter_bottom_serve:")
wx66, wy66 = pixel_to_world(H66, 928.8, 454.6)
wx68, wy68 = pixel_to_world(H68, 922.5, 504.4)
print(f"  cam66 pixel(928.8, 454.6) -> world({wx66:.2f}, {wy66:.2f})  expected(4.115, 5.485)")
print(f"  cam68 pixel(922.5, 504.4) -> world({wx68:.2f}, {wy68:.2f})  expected(4.115, 18.285)")

print("\n" + "=" * 70)
print("Test 2: Simulate ball at different court positions")
print("=" * 70)

# For each test position, compute the ground plane projection through each camera,
# then verify the homography round-trip
test_positions = [
    (2.0, 8.0, 1.5, "left of center, near baseline"),
    (6.0, 8.0, 1.5, "right of center, near baseline"),
    (2.0, 16.0, 1.5, "left of center, far side"),
    (6.0, 16.0, 1.5, "right of center, far side"),
    (4.1, 12.0, 2.0, "center court, high"),
]

H66_w2i = np.array(hdata["cam66"]["H_world_to_image"])
H68_w2i = np.array(hdata["cam68"]["H_world_to_image"])

def world_to_pixel(H, wx, wy):
    pt = H @ np.array([wx, wy, 1.0])
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])

for bx, by, bz, desc in test_positions:
    print(f"\nBall at ({bx}, {by}, {bz}) — {desc}:")

    # Ground plane projection through cam66
    c66 = np.array(cam66_pos)
    ball = np.array([bx, by, bz])
    ray66 = ball - c66
    t66 = -c66[2] / ray66[2]
    g66 = c66 + t66 * ray66

    # Ground plane projection through cam68
    c68 = np.array(cam68_pos)
    ray68 = ball - c68
    t68 = -c68[2] / ray68[2]
    g68 = c68 + t68 * ray68

    # What pixel would the ball appear at?
    px66, py66 = world_to_pixel(H66_w2i, g66[0], g66[1])
    px68, py68 = world_to_pixel(H68_w2i, g68[0], g68[1])

    # Round trip: pixel -> world via homography
    rwx66, rwy66 = pixel_to_world(H66, px66, py66)
    rwx68, rwy68 = pixel_to_world(H68, px68, py68)

    # Triangulate from round-tripped world coords
    pt3d, rd = triangulate((rwx66, rwy66), (rwx68, rwy68), cam66_pos, cam68_pos)

    print(f"  cam66: ground_proj=({g66[0]:.2f},{g66[1]:.2f}) pixel=({px66:.0f},{py66:.0f}) H_roundtrip=({rwx66:.2f},{rwy66:.2f})")
    print(f"  cam68: ground_proj=({g68[0]:.2f},{g68[1]:.2f}) pixel=({px68:.0f},{py68:.0f}) H_roundtrip=({rwx68:.2f},{rwy68:.2f})")
    if pt3d is not None:
        print(f"  Triangulated: ({pt3d[0]:.2f}, {pt3d[1]:.2f}, {pt3d[2]:.2f})  ray_dist={rd:.3f}m")
        print(f"  X error: {abs(pt3d[0] - bx):.3f}m  Y error: {abs(pt3d[1] - by):.3f}m  Z error: {abs(pt3d[2] - bz):.3f}m")
    else:
        print(f"  Triangulation failed (parallel rays)")

print("\n" + "=" * 70)
print("Test 3: Sensitivity to pixel detection error")
print("=" * 70)

bx, by, bz = 2.0, 12.0, 1.0
c66 = np.array(cam66_pos)
c68 = np.array(cam68_pos)
ball = np.array([bx, by, bz])

# True ground projections
ray66 = ball - c66
t66 = -c66[2] / ray66[2]
g66 = c66 + t66 * ray66
ray68 = ball - c68
t68 = -c68[2] / ray68[2]
g68 = c68 + t68 * ray68

# True pixels
px66_true, py66_true = world_to_pixel(H66_w2i, g66[0], g66[1])
px68_true, py68_true = world_to_pixel(H68_w2i, g68[0], g68[1])

print(f"\nBall at ({bx}, {by}, {bz}), true pixels: cam66=({px66_true:.0f},{py66_true:.0f}), cam68=({px68_true:.0f},{py68_true:.0f})")

for pixel_err in [0, 5, 10, 20, 50]:
    # Add error to cam66 X pixel only
    px66_err = px66_true + pixel_err
    wx66, wy66 = pixel_to_world(H66, px66_err, py66_true)
    wx68, wy68 = pixel_to_world(H68, px68_true, py68_true)
    pt3d, rd = triangulate((wx66, wy66), (wx68, wy68), cam66_pos, cam68_pos)
    if pt3d is not None:
        print(f"  cam66 +{pixel_err:2d}px X error -> 3D=({pt3d[0]:.2f}, {pt3d[1]:.2f}, {pt3d[2]:.2f}) "
              f"Xerr={abs(pt3d[0]-bx):.2f}m ray_dist={rd:.3f}m")

print("\n" + "=" * 70)
print("Test 4: X baseline analysis")
print("=" * 70)
print(f"\ncam66 position: {cam66_pos}")
print(f"cam68 position: {cam68_pos}")
print(f"X baseline: {abs(cam66_pos[0] - cam68_pos[0]):.3f}m (!!)")
print(f"Y baseline: {abs(cam66_pos[1] - cam68_pos[1]):.2f}m")
print(f"Z baseline: {abs(cam66_pos[2] - cam68_pos[2]):.2f}m")
print()
print("With a 2.1cm X baseline, X triangulation depends entirely on the")
print("different viewing angles from Y=-5.21 vs Y=28.97. Small pixel errors")
print("in either camera can cause large X errors in the 3D reconstruction.")
print()
print("Expected accuracy (rough estimate):")
print(f"  Y direction: ~0.1-0.3m (34m baseline)")
print(f"  Z direction: ~0.1-0.5m (cameras at ~6m height)")
print(f"  X direction: ~0.5-2.0m (depends on ball position and detection accuracy)")
