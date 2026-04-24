import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.analytics import RallyStateMachine
from app.config import load_config
from app.orchestrator import Orchestrator


@pytest.fixture
def orch():
    orch = Orchestrator(load_config("config.yaml"))
    try:
        yield orch
    finally:
        try:
            orch._manager.shutdown()
        except Exception:
            pass


def test_rally_state_machine_pending_start_used_for_serving():
    sm = RallyStateMachine(serve_confirm_frames=3)
    pts = [
        {"x": 0.0, "y": -8.0, "z": 1.2, "timestamp": 1.00, "frame_index": 100},
        {"x": 0.1, "y": -8.1, "z": 1.1, "timestamp": 1.04, "frame_index": 101},
        {"x": 0.2, "y": -8.2, "z": 1.0, "timestamp": 1.08, "frame_index": 102},
    ]

    for pt in pts:
        sm.update(pt)

    assert sm.get_state_dict()["state"] == "serving"
    assert sm._rally_start_time == pytest.approx(1.00)
    assert sm._rally_start_frame == 100


def test_rally_state_machine_pending_start_used_for_midflight_rally():
    sm = RallyStateMachine(serve_confirm_frames=3)
    pts = [
        {"x": 0.0, "y": -1.0, "z": 1.2, "timestamp": 2.00, "frame_index": 200},
        {"x": 0.1, "y": 0.8, "z": 1.1, "timestamp": 2.04, "frame_index": 201},
    ]

    for pt in pts:
        sm.update(pt)

    assert sm.get_state_dict()["state"] == "rally"
    assert sm._stroke_count == 1
    assert sm._rally_start_time == pytest.approx(2.00)
    assert sm._rally_start_frame == 200


def test_reset_live_analytics_clears_sg_buffer(orch):
    orch._sg_buffer.append(
        {
            "x": 1.0,
            "y": 2.0,
            "z": 0.5,
            "timestamp": 10.0,
            "capture_ts": 10.0,
            "frame_index": 1,
        }
    )

    orch.reset_live_analytics()

    assert orch._sg_buffer == []


def test_post_filter_f2_allows_quick_but_distant_bounce(orch):
    orch._live_bounces = [
        {"timestamp": 10.0, "x": -3.0, "y": -8.0, "side": "near", "in_court": True}
    ]

    ok, reason = orch._post_filter_bounce(
        {"timestamp": 10.2, "x": 3.0, "y": -8.0, "side": "near", "in_court": True}
    )

    assert ok is True
    assert reason == "accepted"


def test_post_filter_f2_rejects_quick_nearby_repeat(orch):
    orch._live_bounces = [
        {"timestamp": 10.0, "x": -3.0, "y": -8.0, "side": "near", "in_court": True}
    ]

    ok, reason = orch._post_filter_bounce(
        {"timestamp": 10.2, "x": -2.2, "y": -8.3, "side": "near", "in_court": True}
    )

    assert ok is False
    assert reason == "f2_min_interval"


def test_live_detectors_respect_bounce_toggle_and_reset_buffers(orch, monkeypatch):
    pt = {
        "x": 0.0,
        "y": -4.0,
        "z": 1.0,
        "timestamp": 20.0,
        "capture_ts": 20.0,
        "frame_index": 10,
    }
    calls = {"peak": 0, "hybrid": 0}

    def fake_peak_update(_point):
        calls["peak"] += 1
        return None

    def fake_pop_pending():
        return []

    def fake_smooth(point):
        return point

    def fake_hybrid_update(_point, _cam_dets):
        calls["hybrid"] += 1
        return None

    monkeypatch.setattr(orch._bounce_detector, "update", fake_peak_update)
    monkeypatch.setattr(orch._bounce_detector, "pop_pending", fake_pop_pending)
    monkeypatch.setattr(orch, "_smooth_latest", fake_smooth)
    monkeypatch.setattr(orch._hybrid_bounce, "update", fake_hybrid_update)

    orch._sg_buffer.append({"x": 9.0, "y": 9.0, "z": 9.0, "timestamp": 9.0})
    orch.set_bounce_detection_enabled(False)
    assert orch._sg_buffer == []

    with orch._analytics_lock:
        smoothed_pt, hbounce = orch._run_live_bounce_detectors_locked(pt, {})
    assert smoothed_pt == pt
    assert hbounce is None
    assert calls == {"peak": 0, "hybrid": 0}

    orch.set_bounce_detection_enabled(True)
    with orch._analytics_lock:
        smoothed_pt, hbounce = orch._run_live_bounce_detectors_locked(pt, {})
    assert smoothed_pt == pt
    assert hbounce is None
    assert calls == {"peak": 1, "hybrid": 1}
