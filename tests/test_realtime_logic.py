import os
import sys
from collections import deque

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


def test_yolo_roadmap_single_cam_chain_publishes_bounce(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 0

    def fake_events(_detections, **_kwargs):
        return {
            "bounces": [
                {
                    "frame_index": 10,
                    "x": 1.0,
                    "y": -5.0,
                    "pixel_x": 900.0,
                    "pixel_y": 500.0,
                    "in_court": True,
                    "confidence": 0.8,
                    "source": "test_yolo_single_cam",
                }
            ],
            "hits": [],
            "speed_events": [],
            "count": 1,
        }

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(22):
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": float(frame),
                    "capture_ts": float(frame),
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    analytics = orch.get_live_analytics()

    assert analytics["total_bounces"] == 1
    assert analytics["recent_bounces"][0]["source"] == "test_yolo_single_cam"
    assert analytics["recent_bounces"][0]["bounce_mode"] == "mono_cam68"
    assert analytics["single_cam_bounce_stats"]["cam68"]["accepted"] == 1


def test_yolo_roadmap_final_bounces_bypass_legacy_post_filter(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 0

    def fake_events(_detections, **_kwargs):
        return {
            "bounces": [
                {
                    "frame_index": 10,
                    "x": 0.0,
                    "y": -5.0,
                    "pixel_x": 900.0,
                    "pixel_y": 500.0,
                    "in_court": True,
                    "confidence": 0.8,
                    "source": "test_yolo_final",
                },
                {
                    "frame_index": 12,
                    "x": 1.7,
                    "y": -5.0,
                    "pixel_x": 920.0,
                    "pixel_y": 500.0,
                    "in_court": True,
                    "confidence": 0.8,
                    "source": "test_yolo_final",
                },
            ],
            "hits": [],
            "speed_events": [],
            "count": 2,
        }

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(22):
            ts = frame / 25.0
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": ts,
                    "capture_ts": ts,
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    analytics = orch.get_live_analytics()

    assert analytics["total_bounces"] == 2
    assert [b["frame_index"] for b in analytics["recent_bounces"]] == [10, 12]
    assert analytics["single_cam_bounce_stats"]["cam68"]["accepted"] == 2
    assert analytics["post_filter_stats"].get("f2_min_interval", 0) == 0


def test_yolo_roadmap_persistent_hit_suppression_blocks_stale_bounce(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 0

    def fake_events(detections, **_kwargs):
        latest = int(detections[-1]["frame_index"])
        if latest < 20:
            return {
                "bounces": [],
                "hits": [],
                "speed_events": [],
                "hit_suppression_frames": [12],
                "count": 0,
            }
        return {
            "bounces": [
                {
                    "frame_index": 10,
                    "x": 1.0,
                    "y": -5.0,
                    "pixel_x": 900.0,
                    "pixel_y": 500.0,
                    "in_court": True,
                    "confidence": 0.8,
                    "source": "test_stale_bounce_after_hit_candidate",
                }
            ],
            "hits": [],
            "speed_events": [],
            "hit_suppression_frames": [],
            "count": 1,
        }

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(30):
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": float(frame),
                    "capture_ts": float(frame),
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    analytics = orch.get_live_analytics()
    assert analytics["total_bounces"] == 0
    stats = analytics["single_cam_bounce_stats"]["cam68"]
    assert stats["hit_suppression_frames"] == 1
    assert stats["skipped_persistent_hit_suppressed_bounces"] > 0
    assert stats["last_reject_reason"] == "hit_window:12"


def test_yolo_roadmap_suppresses_shadow_of_hit_suppressed_bounce(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 0
    orch.config.hit_bounce_refiner.hit_suppression_frames = 5

    def fake_events(detections, **_kwargs):
        latest = int(detections[-1]["frame_index"])
        if latest < 10:
            return {
                "bounces": [],
                "hits": [],
                "speed_events": [],
                "hit_suppression_frames": [],
                "count": 0,
            }
        if latest < 20:
            return {
                "bounces": [],
                "hits": [
                    {
                        "frame_index": 5,
                        "x": 1.0,
                        "y": -10.0,
                        "pixel_x": 840.0,
                        "pixel_y": 120.0,
                        "confidence": 0.8,
                        "source": "test_top_hit",
                    }
                ],
                "speed_events": [],
                "hit_suppression_frames": [5],
                "count": 0,
            }
        if latest < 30:
            return {
                "bounces": [],
                "hits": [],
                "speed_events": [],
                "hit_suppression_frames": [5],
                "suppressed_bounces": [
                    {
                        "frame_index": 10,
                        "x": 1.0,
                        "y": -13.0,
                        "pixel_x": 840.0,
                        "pixel_y": 106.0,
                        "in_court": False,
                        "confidence": 0.5,
                        "source": "test_hit_suppressed_shadow_seed",
                        "suppression_reason": "hit_window",
                        "suppressed_by_hit_frame": 5,
                    }
                ],
                "count": 0,
            }
        return {
            "bounces": [
                {
                    "frame_index": 12,
                    "x": 1.1,
                    "y": -13.1,
                    "pixel_x": 845.0,
                    "pixel_y": 107.0,
                    "in_court": False,
                    "confidence": 0.5,
                    "source": "test_hit_suppressed_shadow_candidate",
                }
            ],
            "hits": [],
            "speed_events": [],
            "hit_suppression_frames": [],
            "count": 1,
        }

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(40):
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": float(frame),
                    "capture_ts": float(frame),
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    analytics = orch.get_live_analytics()
    assert analytics["total_bounces"] == 0
    stats = analytics["single_cam_bounce_stats"]["cam68"]
    assert stats["remembered_hit_suppressed_bounces"] > 0
    assert stats["skipped_hit_suppressed_duplicate_bounces"] > 0
    assert stats["last_reject_reason"] == "hit_window_shadow:10"


def test_yolo_duplicate_replacement_does_not_resurrect_hit_suppressed_bounce(orch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.hit_suppression_frames = 5
    seen = deque([901], maxlen=50)
    stats = {}
    previous = {
        "frame_index": 901,
        "x": 1.11,
        "y": -13.36,
        "camera": "cam68",
        "camera_name": "cam68",
        "type": "OUT",
        "source": "yolo_verify_queue_single_cam",
        "bounce_signal_score": 117.0,
    }
    candidate = {
        "frame_index": 903,
        "x": 1.19,
        "y": -13.15,
        "camera": "cam68",
        "camera_name": "cam68",
        "type": "OUT",
        "source": "yolo_verify_queue_single_cam",
        "bounce_signal_score": 148.0,
    }
    orch._live_hits.append(
        {
            "frame_index": 897,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "HIT",
        }
    )
    orch._live_bounces.append(previous)

    action = orch._replace_weaker_yolo_duplicate_bounce_locked(
        "cam68",
        candidate,
        stats=stats,
        seen_frames=seen,
    )

    assert action == "skip"
    assert orch._live_bounces == []
    assert list(seen) == []
    assert stats["retro_suppressed_duplicate_live_bounces_by_hit"] == 1
    assert stats["last_retro_duplicate_hit_frame"] == 897
    assert stats["last_retro_duplicate_bounce_frame"] == 901


def test_yolo_retracts_live_bounce_shadowing_hit_suppressed_candidate(orch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.hit_suppression_frames = 5
    stats = {}
    orch._live_hits.append(
        {
            "frame_index": 897,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "HIT",
        }
    )
    orch._remember_yolo_hit_suppressed_bounce_locked(
        "cam68",
        {
            "frame_index": 901,
            "x": 1.11,
            "y": -13.36,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "OUT",
            "source": "yolo_verify_queue_single_cam",
        },
        suppressing_hit_frame=897,
    )
    orch._live_bounces.append(
        {
            "frame_index": 903,
            "x": 1.19,
            "y": -13.15,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "OUT",
            "source": "yolo_verify_queue_single_cam",
        }
    )
    orch._yolo_fuzzy_emitted_frames["cam68"].append(903)

    retracted = orch._retract_yolo_live_bounces_shadowing_hit_suppressed_locked(
        "cam68",
        stats,
    )

    assert [event["frame_index"] for event in retracted] == [903]
    assert orch._live_bounces == []
    assert list(orch._yolo_fuzzy_emitted_frames["cam68"]) == []
    assert stats["retro_suppressed_hit_shadow_live_bounces"] == 1
    assert stats["last_retro_hit_shadow_live_bounce_frames"] == [903]
    assert stats["last_retro_hit_shadow_seed_frames"] == [901]


def test_yolo_does_not_retract_live_bounce_outside_hit_shadow_window(orch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.hit_suppression_frames = 5
    stats = {}
    orch._live_hits.append(
        {
            "frame_index": 945,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "HIT",
        }
    )
    orch._remember_yolo_hit_suppressed_bounce_locked(
        "cam68",
        {
            "frame_index": 944,
            "x": 1.12,
            "y": 8.2,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "IN",
            "source": "yolo_verify_queue_single_cam",
        },
        suppressing_hit_frame=945,
    )
    orch._live_bounces.append(
        {
            "frame_index": 932,
            "x": 1.06,
            "y": 8.13,
            "camera": "cam68",
            "camera_name": "cam68",
            "type": "IN",
            "source": "yolo_verify_queue_single_cam",
        }
    )
    orch._yolo_fuzzy_emitted_frames["cam68"].append(932)

    retracted = orch._retract_yolo_live_bounces_shadowing_hit_suppressed_locked(
        "cam68",
        stats,
    )

    assert retracted == []
    assert [event["frame_index"] for event in orch._live_bounces] == [932]
    assert list(orch._yolo_fuzzy_emitted_frames["cam68"]) == [932]
    assert stats == {}


def test_late_yolo_hit_retracts_already_published_bounce(orch):
    orch.switch_model("yolo_roadmap")
    orch._ws_enabled = True
    orch.config.hit_bounce_refiner.hit_suppression_frames = 3

    with orch._analytics_lock:
        orch._record_live_bounce_locked(
            {
                "frame_index": 10,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 1.0,
                "capture_ts": 1.0,
                "x": 1.0,
                "y": -5.0,
                "z": 0.0,
                "in_court": True,
                "source": "test_late_hit_order",
            }
        )
        orch._rally_raw_buffer.append(
            {
                "frame_index": 10,
                "is_bounce": True,
                "bounce_event": {"frame_index": 10},
                "event_ball": {"x": 1.0, "y": -5.0, "z": 0.0},
            }
        )
        assert len(orch._ws_bounce_queue) == 1

        orch._record_live_hit_locked(
            {
                "frame_index": 12,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 1.08,
                "capture_ts": 1.08,
                "x": 1.2,
                "y": -8.0,
                "source": "top_down_lookback",
            }
        )

    analytics = orch.get_live_analytics()

    assert analytics["total_bounces"] == 0
    assert analytics["recent_bounces"] == []
    assert len(analytics["recent_hits"]) == 1
    assert len(orch._ws_bounce_queue) == 0
    assert orch._rally_raw_buffer[-1]["is_bounce"] is False
    assert "bounce_event" not in orch._rally_raw_buffer[-1]
    stats = analytics["single_cam_bounce_stats"]["cam68"]
    assert stats["retro_suppressed_bounces_by_hit"] == 1
    assert stats["retro_suppressed_ws_bounces_by_hit"] == 1


def test_yolo_out_gate_blocks_until_bottom_hit_restart(orch):
    orch.switch_model("yolo_roadmap")

    with orch._analytics_lock:
        orch._record_live_bounce_locked(
            {
                "frame_index": 100,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 4.0,
                "x": 1.0,
                "y": -12.5,
                "z": 0.0,
                "in_court": False,
            }
        )

        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 150, "camera_name": "cam68"}
        ) is False

        orch._record_live_hit_locked(
            {
                "frame_index": 230,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 9.2,
                "x": 0.0,
                "y": -8.0,
                "source": "top_down_lookback",
            }
        )
        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 240, "camera_name": "cam68"}
        ) is False

        orch._record_live_hit_locked(
            {
                "frame_index": 260,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 10.4,
                "x": 0.0,
                "y": 8.0,
                "source": "bottom_up_lookback",
            }
        )
        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 240, "camera_name": "cam68"}
        ) is False
        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 270, "camera_name": "cam68"}
        ) is True


def test_yolo_gate_only_out_blocks_without_publishing_bounce(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 0

    def fake_events(detections, **_kwargs):
        latest = int(detections[-1]["frame_index"])
        result = {
            "bounces": [],
            "hits": [],
            "speed_events": [],
            "gate_only_bounces": [
                {
                    "frame_index": 10,
                    "x": 4.4,
                    "y": -12.7,
                    "pixel_x": 684.0,
                    "pixel_y": 107.0,
                    "in_court": False,
                    "type": "OUT",
                    "source": "yolo_verify_queue_single_cam",
                    "gate_only": True,
                    "publishable": False,
                }
            ],
            "gate_only_bounce_count": 1,
            "count": 0,
        }
        if latest >= 60:
            result["bounces"] = [
                {
                    "frame_index": 60,
                    "x": 0.0,
                    "y": -5.0,
                    "pixel_x": 900.0,
                    "pixel_y": 500.0,
                    "in_court": True,
                    "confidence": 0.8,
                    "source": "yolo_verify_queue_single_cam",
                }
            ]
            result["count"] = 1
        return result

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(75):
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": float(frame),
                    "capture_ts": float(frame),
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    analytics = orch.get_live_analytics()
    stats = analytics["single_cam_bounce_stats"]["cam68"]
    assert analytics["total_bounces"] == 0
    assert stats["gate_only_out_gate_bounces"] == 1
    assert stats["out_gate_suppressed_bounces"] >= 1


def test_yolo_out_gate_restart_uses_speed_before_display_release(orch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 50

    with orch._analytics_lock:
        orch._record_live_bounce_locked(
            {
                "frame_index": 901,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.04,
                "x": 1.0,
                "y": -13.0,
                "z": 0.0,
                "in_court": False,
            }
        )
        orch._prime_yolo_out_gate_restarts_locked(
            "cam68",
            hit_events=[],
            speed_events=[
                {
                    "frame_index": 923,
                    "camera": "cam68",
                    "camera_name": "cam68",
                    "speed_kmh": 42,
                    "source": "single_cam_verify_queue_speed_trap",
                }
            ],
            candidate_frame=932,
            latest_frame=940,
        )

        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 932, "camera_name": "cam68"}
        ) is True


def test_yolo_out_gate_retro_primes_speed_recorded_before_delayed_out(orch):
    orch.switch_model("yolo_roadmap")

    with orch._analytics_lock:
        orch._record_live_speed_event_locked(
            {
                "frame_index": 923,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.92,
                "capture_ts": 36.92,
                "speed_kmh": 42,
                "source": "single_cam_verify_queue_speed_trap",
            }
        )
        orch._record_live_bounce_locked(
            {
                "frame_index": 901,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.04,
                "capture_ts": 36.04,
                "x": 1.0,
                "y": -13.0,
                "z": 0.0,
                "in_court": False,
            }
        )

        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 910, "camera_name": "cam68"}
        ) is False
        assert orch._yolo_out_gate_allows_bounce_locked(
            "cam68", {"frame_index": 932, "camera_name": "cam68"}
        ) is True

    stats = orch.get_live_analytics()["single_cam_bounce_stats"]["cam68"]
    assert stats["out_gate_last_suppressed_interval"] == [901, 923]


def test_yolo_out_gate_releases_pending_bounce_after_late_speed_restart(orch):
    orch.switch_model("yolo_roadmap")
    orch._ws_enabled = True

    with orch._analytics_lock:
        orch._record_live_bounce_locked(
            {
                "frame_index": 901,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.04,
                "capture_ts": 36.04,
                "x": 1.0,
                "y": -13.0,
                "z": 0.0,
                "in_court": False,
            }
        )
        pending = orch._normalize_live_bounce_dict(
            {
                "frame_index": 932,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 37.28,
                "capture_ts": 37.28,
                "x": 0.2,
                "y": 6.0,
                "z": 0.0,
                "in_court": True,
                "source": "test_out_gate_pending",
            },
            fallback_ts=37.28,
            fallback_speed_kmh=0,
        )
        orch._stash_yolo_out_gate_pending_bounce_locked("cam68", pending)

        assert orch._total_live_bounces == 1

        orch._record_live_speed_event_locked(
            {
                "frame_index": 923,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.92,
                "capture_ts": 36.92,
                "speed_kmh": 42,
                "source": "single_cam_verify_queue_speed_trap",
            }
        )

    analytics = orch.get_live_analytics()
    assert analytics["total_bounces"] == 2
    assert [b["frame_index"] for b in analytics["recent_bounces"]] == [901, 932]
    stats = analytics["single_cam_bounce_stats"]["cam68"]
    assert stats["out_gate_released_pending_bounces"] == 1
    assert stats["out_gate_pending_bounces"] == 0
    assert len(orch._ws_bounce_queue) == 2


def test_yolo_out_gate_drops_pending_bounce_inside_closed_interval(orch):
    orch.switch_model("yolo_roadmap")

    with orch._analytics_lock:
        orch._record_live_bounce_locked(
            {
                "frame_index": 901,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.04,
                "capture_ts": 36.04,
                "x": 1.0,
                "y": -13.0,
                "z": 0.0,
                "in_court": False,
            }
        )
        pending = orch._normalize_live_bounce_dict(
            {
                "frame_index": 910,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.4,
                "capture_ts": 36.4,
                "x": 0.2,
                "y": 5.0,
                "z": 0.0,
                "in_court": True,
                "source": "test_out_gate_pending",
            },
            fallback_ts=36.4,
            fallback_speed_kmh=0,
        )
        orch._stash_yolo_out_gate_pending_bounce_locked("cam68", pending)
        orch._record_live_speed_event_locked(
            {
                "frame_index": 923,
                "camera": "cam68",
                "camera_name": "cam68",
                "timestamp": 36.92,
                "capture_ts": 36.92,
                "speed_kmh": 42,
                "source": "single_cam_verify_queue_speed_trap",
            }
        )

    analytics = orch.get_live_analytics()
    assert analytics["total_bounces"] == 1
    stats = analytics["single_cam_bounce_stats"]["cam68"]
    assert stats["out_gate_dropped_pending_closed_interval"] == 1
    assert stats["out_gate_pending_bounces"] == 0


def test_yolo_roadmap_analysis_stride_keeps_full_detection_buffer(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    calls = []

    def fake_events(detections, **_kwargs):
        calls.append([int(d["frame_index"]) for d in detections])
        return {
            "bounces": [],
            "hits": [],
            "speed_events": [],
            "count": 0,
        }

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(11):
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": float(frame),
                    "capture_ts": float(frame),
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    stats = orch.get_live_analytics()["single_cam_bounce_stats"]["cam68"]
    assert stats["detections"] == 11
    assert stats["analysis_stride"] == 5
    assert stats["analysis_calls"] == 3
    assert stats["skipped_analysis_stride"] == 8
    assert len(calls) == 3
    assert calls[-1] == list(range(11))


def test_preview_frame_store_exposes_seq_and_waits_for_new_frame(orch):
    orch._store_latest_frame("cam68", b"frame-a", frame_id=100, capture_ts=10.0)

    first = orch.get_latest_frame_info("cam68")
    assert first["jpeg"] == b"frame-a"
    assert first["seq"] == 1
    assert first["frame_id"] == 100
    assert first["capture_ts"] == pytest.approx(10.0)

    status = orch.get_pipeline_status("cam68")
    assert status.latest_preview_seq == 1
    assert status.latest_preview_frame_id == 100
    assert status.latest_preview_capture_ts == pytest.approx(10.0)
    assert status.latest_preview_age_ms is not None

    assert orch.wait_for_latest_frame("cam68", after_seq=1, timeout=0.0) is None

    orch._store_latest_frame("cam68", b"frame-b", frame_id=101, capture_ts=10.04)
    second = orch.wait_for_latest_frame("cam68", after_seq=1, timeout=0.0)
    assert second["jpeg"] == b"frame-b"
    assert second["seq"] == 2
    assert second["frame_id"] == 101


def test_pipeline_status_surfaces_preview_feeder_metrics(orch, monkeypatch):
    handle = orch._handles["cam68"]
    handle.status_dict = {
        "state": "running",
        "fps": 11.0,
        "preview_fps": 24.8,
        "preview_frame_id": 1234,
        "inference_enabled": True,
        "inference_ready": True,
    }
    monkeypatch.setattr(handle, "is_alive", lambda: True)

    status = orch.get_pipeline_status("cam68")

    assert status.fps == pytest.approx(11.0)
    assert status.preview_fps == pytest.approx(24.8)
    assert status.preview_frame_id == 1234


def test_live_analytics_recent_payload_is_limited_without_losing_totals(orch):
    orch._LIVE_ANALYTICS_EVENT_LIMIT = 2
    orch._LIVE_ANALYTICS_SPEED_LIMIT = 1

    with orch._analytics_lock:
        for frame in range(3):
            orch._record_live_bounce_locked({
                "timestamp": float(frame),
                "x": float(frame) * 2.0,
                "y": 0.0,
                "z": 0.0,
                "in_court": True,
                "frame_index": frame,
            })
            orch._record_live_hit_locked({
                "timestamp": float(frame),
                "x": float(frame) * 2.0,
                "y": 5.0,
                "frame_index": frame,
                "source": "test",
            })
            orch._record_live_speed_event_locked({
                "timestamp": float(frame),
                "frame_index": frame,
                "speed_kmh": 20 + frame,
            })

    analytics = orch.get_live_analytics()
    assert analytics["total_bounces"] == 3
    assert analytics["total_hits"] == 3
    assert analytics["total_speed_events"] == 3
    assert [b["frame_index"] for b in analytics["recent_bounces"]] == [1, 2]
    assert [h["frame_index"] for h in analytics["recent_hits"]] == [1, 2]
    assert [s["frame_index"] for s in analytics["recent_speed_events"]] == [2]


def test_yolo_roadmap_final_bounce_waits_for_release_delay(orch, monkeypatch):
    orch.switch_model("yolo_roadmap")
    orch.config.hit_bounce_refiner.release_delay_frames = 50

    def fake_events(_detections, **_kwargs):
        return {
            "bounces": [
                {
                    "frame_index": 10,
                    "x": 1.0,
                    "y": -5.0,
                    "pixel_x": 900.0,
                    "pixel_y": 500.0,
                    "in_court": True,
                    "confidence": 0.8,
                    "source": "test_yolo_delayed_final",
                }
            ],
            "hits": [],
            "speed_events": [],
            "count": 1,
        }

    monkeypatch.setattr(
        "app.pipeline.yolo_bounce_filter.detect_single_camera_events",
        fake_events,
    )

    with orch._analytics_lock:
        for frame in range(55):
            orch._run_yolo_fuzzy_single_cam_locked(
                "cam68",
                {
                    "camera_name": "cam68",
                    "frame_index": frame,
                    "timestamp": float(frame),
                    "capture_ts": float(frame),
                    "pixel_x": 800.0 + frame,
                    "pixel_y": 500.0,
                    "x": 0.0,
                    "y": -5.0,
                    "yolo_conf": 0.9,
                },
            )

    assert orch.get_live_analytics()["total_bounces"] == 0

    with orch._analytics_lock:
        orch._run_yolo_fuzzy_single_cam_locked(
            "cam68",
            {
                "camera_name": "cam68",
                "frame_index": 60,
                "timestamp": 60.0,
                "capture_ts": 60.0,
                "pixel_x": 860.0,
                "pixel_y": 500.0,
                "x": 0.0,
                "y": -5.0,
                "yolo_conf": 0.9,
            },
        )

    analytics = orch.get_live_analytics()
    assert analytics["total_bounces"] == 1
    assert analytics["recent_bounces"][0]["frame_index"] == 10
