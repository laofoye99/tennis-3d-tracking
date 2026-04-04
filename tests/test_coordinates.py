"""Comprehensive coordinate mapping tests.

Validates that all minimap/3D court drawing functions correctly map
V2 world coordinates to canvas pixel positions, using known bounce
positions from bounce_results.json as ground truth.
"""

import json
import sys
import math

sys.path.insert(0, ".")

# JS constants (from dashboard.html line 984)
HW = 4.115
HL = 11.89
NET_Y = 0.0
SRV_N = -6.4
SRV_F = 6.4
CX = 0.0


def load_expected():
    with open("D:/tennis/blob_frame_different/bounce_results.json") as f:
        d = json.load(f)
    return d["trajectory"], d["bounces"]


# ============================================================
# 1. PeakBounceDetector batch alignment
# ============================================================
def test_peak_bounce_detector():
    from app.analytics import PeakBounceDetector

    traj, expected = load_expected()
    expected_frames = sorted(b["frame"] for b in expected)

    det = PeakBounceDetector(
        batch_size=30, z_max=0.5, prominence=0.10, min_distance=5, smooth=3
    )
    for pt in traj:
        det.update({
            "x": pt["x"], "y": pt["y"], "z": pt["z"],
            "timestamp": pt["frame"] * 0.04,
            "frame_index": pt["frame"],
        })
    det._counter = det.batch_size
    det._run_batch({"timestamp": traj[-1]["frame"] * 0.04})

    detected_frames = sorted(b.frame_index for b in det.get_all_bounces())
    assert expected_frames == detected_frames, (
        f"Frame mismatch:\n  expected {expected_frames}\n  got      {detected_frames}"
    )

    # Compare x, y, in_court
    exp_map = {b["frame"]: b for b in expected}
    det_map = {b.frame_index: b for b in det.get_all_bounces()}
    for f in expected_frames:
        e, d = exp_map[f], det_map[f]
        assert abs(e["x"] - d.x) < 0.001, f"f={f} x: {e['x']} vs {d.x}"
        assert abs(e["y"] - d.y) < 0.001, f"f={f} y: {e['y']} vs {d.y}"
        assert e["in_court"] == d.in_court, f"f={f} in_court: {e['in_court']} vs {d.in_court}"

    print(f"[PASS] PeakBounceDetector: {len(expected_frames)}/{len(expected_frames)} frames, x/y/in_court match")


# ============================================================
# 2. Minimap drawCourt (tx/ty)
# ============================================================
def test_minimap_drawcourt():
    _, expected = load_expected()

    W, H = 300, 500
    PAD_X, PAD_Y = 2.0, 3.0
    VW = 2 * HW + 2 * PAD_X
    VH = 2 * HL + 2 * PAD_Y
    margin = 4
    dw = W - 2 * margin
    dh = H - 2 * margin

    def tx(x):
        return margin + ((HW + PAD_X - x) / VW) * dw

    def ty(y):
        return margin + ((HL + PAD_Y - y) / VH) * dh

    # Court corners
    left = tx(HW)
    right = tx(-HW)
    top = ty(HL)
    bot = ty(-HL)
    net = ty(NET_Y)

    assert 0 < left < right < W, f"Court X: left={left:.1f} right={right:.1f}"
    assert 0 < top < bot < H, f"Court Y: top={top:.1f} bot={bot:.1f}"
    assert top < net < bot, f"Net={net:.1f} not between top={top:.1f} bot={bot:.1f}"

    # Net at center of court
    center_y = (top + bot) / 2
    assert abs(net - center_y) < 0.5, f"Net={net:.1f} not at center={center_y:.1f}"

    # Service lines symmetric
    srv_n = ty(SRV_N)
    srv_f = ty(SRV_F)
    assert abs((srv_n - net) - (net - srv_f)) < 0.5, "Service lines not symmetric"

    # Center line at canvas center
    center_x = tx(CX)
    court_cx = (left + right) / 2
    assert abs(center_x - court_cx) < 0.5, f"Center={center_x:.1f} not at court_cx={court_cx:.1f}"

    # All bounces in canvas; IN bounces inside court rect
    for b in expected:
        bx = tx(b["x"])
        by = ty(b["y"])
        assert 0 <= bx <= W, f"f={b['frame']} x={b['x']} px={bx:.1f} outside canvas"
        assert 0 <= by <= H, f"f={b['frame']} y={b['y']} py={by:.1f} outside canvas"
        if b["in_court"]:
            assert left - 5 <= bx <= right + 5, (
                f"IN bounce f={b['frame']} px={bx:.1f} outside court [{left:.1f},{right:.1f}]"
            )
            assert top - 5 <= by <= bot + 5, (
                f"IN bounce f={b['frame']} py={by:.1f} outside court [{top:.1f},{bot:.1f}]"
            )

    # Specific bounces: verify direction
    # Bounce at x=+2.694 should be LEFT of center (mirror X)
    assert tx(2.694) < center_x, "Positive x should map left of center (mirror)"
    # Bounce at y=+6.441 should be ABOVE net
    assert ty(6.441) < net, "Positive y should map above net"
    # Bounce at y=-10.069 should be BELOW net
    assert ty(-10.069) > net, "Negative y should map below net"

    print("[PASS] Minimap drawCourt: court landmarks + 30 bounces correct")


# ============================================================
# 3. Video-test minimap toC
# ============================================================
def test_vt_minimap_toC():
    _, expected = load_expected()

    w, h = 268, 460
    pad = 8
    scaleX = (w - 2 * pad) / (2 * HW)
    scaleY = (h - 2 * pad) / (2 * HL)
    sc = min(scaleX, scaleY)
    cw = 2 * HW * sc
    ch = 2 * HL * sc
    ox = (w - cw) / 2
    oy = (h - ch) / 2

    def toC(wx, wy):
        return ox + (HW - wx) * sc, oy + (HL - wy) * sc

    # Court corners
    tl = toC(HW, HL)
    br = toC(-HW, -HL)
    net = toC(0, NET_Y)

    assert tl[0] < br[0], f"Court X: TL={tl[0]:.1f} BR={br[0]:.1f}"
    assert tl[1] < br[1], f"Court Y: TL={tl[1]:.1f} BR={br[1]:.1f}"

    # Net at center
    center_y = (tl[1] + br[1]) / 2
    assert abs(net[1] - center_y) < 0.5, f"Net={net[1]:.1f} not at center={center_y:.1f}"

    # Service lines symmetric
    srv_n = toC(0, SRV_N)
    srv_f = toC(0, SRV_F)
    assert abs((srv_n[1] - net[1]) - (net[1] - srv_f[1])) < 0.5, "Service lines not symmetric"

    # All bounces in canvas
    for b in expected:
        bx, by = toC(b["x"], b["y"])
        assert -5 <= bx <= w + 5, f"f={b['frame']} px={bx:.1f} outside"
        assert -5 <= by <= h + 5, f"f={b['frame']} py={by:.1f} outside"

    print("[PASS] Video-test minimap toC: court + 30 bounces correct")


# ============================================================
# 4. 3D Court ball/bounce positions
# ============================================================
def test_3d_court():
    _, expected = load_expected()

    for b in expected:
        # Live 3D: position.set(-ball.x, ball.y, ball.z)
        three_x = -b["x"]
        three_y = b["y"]
        assert abs(three_x) <= HW + 5, f"f={b['frame']} 3D x={three_x} out of range"
        assert abs(three_y) <= HL + 5, f"f={b['frame']} 3D y={three_y} out of range"

        # Video-test 3D: position.set(-p.x, p.z, p.y)
        vt_x = -b["x"]
        vt_y = b["z"]   # height
        vt_z = b["y"]   # court length
        assert abs(vt_x) <= HW + 5, f"f={b['frame']} VT3D x={vt_x} out of range"
        assert abs(vt_z) <= HL + 5, f"f={b['frame']} VT3D z={vt_z} out of range"
        assert vt_y >= -0.1, f"f={b['frame']} VT3D height={vt_y} below ground"

    print("[PASS] 3D Court: all 30 bounces map to valid positions")


# ============================================================
# 5. Cross-check: minimap and toC agree on direction
# ============================================================
def test_cross_consistency():
    """Verify all minimap functions agree on which direction is which."""
    # drawCourt tx/ty
    W, H = 300, 500
    PAD_X, PAD_Y = 2.0, 3.0
    VW = 2 * HW + 2 * PAD_X
    VH = 2 * HL + 2 * PAD_Y
    margin = 4
    dw = W - 2 * margin
    dh = H - 2 * margin

    def tx(x):
        return margin + ((HW + PAD_X - x) / VW) * dw

    def ty(y):
        return margin + ((HL + PAD_Y - y) / VH) * dh

    # toC
    w, h = 268, 460
    pad = 8
    sc = min((w - 2 * pad) / (2 * HW), (h - 2 * pad) / (2 * HL))
    cw = 2 * HW * sc
    ch = 2 * HL * sc
    ox = (w - cw) / 2
    oy = (h - ch) / 2

    def toC(wx, wy):
        return ox + (HW - wx) * sc, oy + (HL - wy) * sc

    # Both should agree: positive x → left, positive y → up
    # drawCourt: tx(+1) < tx(-1) (positive x goes left = mirror)
    assert tx(1) < tx(-1), "drawCourt: positive x should go left"
    # toC: toC(+1,0).x < toC(-1,0).x (positive x goes left = mirror)
    assert toC(1, 0)[0] < toC(-1, 0)[0], "toC: positive x should go left"

    # drawCourt: ty(+1) < ty(-1) (positive y goes up)
    assert ty(1) < ty(-1), "drawCourt: positive y should go up (smaller px)"
    # toC: toC(0,+1).y < toC(0,-1).y (positive y goes up)
    assert toC(0, 1)[1] < toC(0, -1)[1], "toC: positive y should go up"

    # Net at y=0 in both
    net_dc = ty(0)
    net_tc = toC(0, 0)[1]
    dc_top = ty(HL)
    dc_bot = ty(-HL)
    tc_top = toC(0, HL)[1]
    tc_bot = toC(0, -HL)[1]

    # Both: net is at 50% of court height
    dc_ratio = (net_dc - dc_top) / (dc_bot - dc_top)
    tc_ratio = (net_tc - tc_top) / (tc_bot - tc_top)
    assert abs(dc_ratio - 0.5) < 0.01, f"drawCourt net ratio {dc_ratio:.3f} != 0.5"
    assert abs(tc_ratio - 0.5) < 0.01, f"toC net ratio {tc_ratio:.3f} != 0.5"

    print("[PASS] Cross-consistency: drawCourt and toC agree on direction + net position")


# ============================================================
# 6. draw3DMap px/py
# ============================================================
def test_draw3DMap():
    _, expected = load_expected()

    # From dashboard.html draw3DMap:
    # px(x) = margin + (x + HW + 1.5) * sx
    # py(y) = H - margin - (y + HL + 2) * sy
    # sx = cw / (2*HW + 3), sy = ch / (2*HL + 4)
    # Ball: px(ball.x), py(ball.y) — no negation after fix

    W, H = 300, 300  # approximate 3D map canvas
    margin = 40
    cw = W - 2 * margin
    ch = H - 2 * margin
    sx = cw / (2 * HW + 3)
    sy = ch / (2 * HL + 4)

    def px(x):
        return margin + (x + HW + 1.5) * sx

    def py(y):
        return H - margin - (y + HL + 2) * sy

    # Court rect
    court_left = px(-HW)
    court_right = px(HW)
    court_top = py(HL)    # smaller y value = top
    court_bot = py(-HL)

    assert court_left < court_right, f"3DMap court X: L={court_left:.1f} R={court_right:.1f}"
    assert court_top < court_bot, f"3DMap court Y: T={court_top:.1f} B={court_bot:.1f}"

    # Net at center
    net_y = py(NET_Y)
    center = (court_top + court_bot) / 2
    assert abs(net_y - center) < 0.5, f"3DMap net={net_y:.1f} not at center={center:.1f}"

    # Ball positions (no negation after fix)
    for b in expected:
        bx = px(b["x"])
        by = py(b["y"])
        assert margin - 10 <= bx <= W - margin + 10, f"f={b['frame']} px={bx:.1f} outside"
        assert margin - 10 <= by <= H - margin + 10, f"f={b['frame']} py={by:.1f} outside"

    print("[PASS] draw3DMap: court layout + bounce positions correct (no X negation)")


if __name__ == "__main__":
    test_peak_bounce_detector()
    test_minimap_drawcourt()
    test_vt_minimap_toC()
    test_3d_court()
    test_cross_consistency()
    test_draw3DMap()
    print("\n=== ALL 6 TESTS PASSED ===")
