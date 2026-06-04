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


def test_orchestrator_uses_bounce_detection_config():
    config = load_config("config.yaml")
    config.bounce_detection.hybrid.min_seg_len = 6
    config.bounce_detection.hybrid.min_dense = 5
    config.bounce_detection.hybrid.max_gap_s = 0.9
    config.bounce_detection.hybrid.z_max = 0.9
    config.bounce_detection.hybrid.min_speed = 2.0
    config.bounce_detection.hybrid.v_window = 6
    config.bounce_detection.hybrid.half_wins = [4, 6]
    config.bounce_detection.smoothing.max_frame_gap = 6
    config.bounce_detection.smoothing.max_gap_s = 0.9

    orch = Orchestrator(config)
    try:
        assert orch._hybrid_bounce._min_seg_len == 6
        assert orch._hybrid_bounce._min_dense == 5
        assert orch._hybrid_bounce._max_gap_s == pytest.approx(0.9)
        assert orch._hybrid_bounce._z_max == pytest.approx(0.9)
        assert orch._hybrid_bounce._min_speed == pytest.approx(2.0)
        assert orch._hybrid_bounce._v_window == 6
        assert orch._hybrid_bounce._half_wins == (4, 6)
        assert orch._sg_max_gap == 6
        assert orch._sg_max_gap_s == pytest.approx(0.9)
    finally:
        orch._manager.shutdown()


def test_live_bounce_history_keeps_true_total_after_rollover(orch):
    orch._LIVE_BOUNCE_HISTORY_LIMIT = 3

    for i in range(5):
        orch._record_live_bounce_locked({
            "timestamp": float(i),
            "x": float(i),
            "y": 0.0,
            "z": 0.0,
            "in_court": True,
            "frame_index": i,
        })

    analytics = orch.get_live_analytics()

    assert analytics["total_bounces"] == 5
    assert [b["frame_index"] for b in analytics["recent_bounces"]] == [2, 3, 4]
    assert [b["sequence"] for b in analytics["recent_bounces"]] == [3, 4, 5]


def test_live_hit_analytics_does_not_enqueue_3d_push(orch):
    orch._ws_enabled = True

    orch._record_live_hit_locked({
        "event_type": "hit",
        "frame_index": 10,
        "timestamp": 1.0,
        "x": 0.2,
        "y": 6.0,
        "source": "bottom_up_lookback",
    })

    analytics = orch.get_live_analytics()

    assert len(analytics["recent_hits"]) == 1
    assert analytics["recent_hits"][0]["source"] == "bottom_up_lookback"
    assert len(orch._ws_bounce_queue) == 0


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

    def fake_smooth(point, cam_dets=None):
        return point, cam_dets

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


def test_switch_model_sets_matching_detector_type_and_yolo_roadmap_weight(orch):
    tracknet = orch.switch_model("tracknet")

    assert tracknet["model"] == "tracknet"
    assert tracknet["path"] == "model_weight/TrackNet_finetuned.onnx"
    assert tracknet["frames_in"] == 8
    assert tracknet["frames_out"] == 8
    assert tracknet["detector_type"] == "tracknet"
    assert orch.get_current_model()["model"] == "tracknet"

    yolo = orch.switch_model("yolo_roadmap")

    assert yolo["model"] == "yolo_roadmap"
    assert yolo["path"] == "yolo_roadmap/best.pt"
    assert yolo["frames_in"] == 1
    assert yolo["frames_out"] == 1
    assert yolo["detector_type"] == "yolo_roadmap"
    assert orch.get_current_model()["model"] == "yolo_roadmap"
