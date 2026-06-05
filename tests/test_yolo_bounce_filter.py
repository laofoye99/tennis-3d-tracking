import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.yolo_bounce_filter import (
    _apply_out_rally_gate,
    _is_gate_only_weak_out_bounce,
    _select_strongest_bounces,
    dashboard_yolo_quality_reject_reason,
    detect_single_camera_events,
    filter_dashboard_yolo_publishable_bounces,
)


class FakeHomography:
    def pixel_to_world(self, x, y):
        return (x / 100.0 - 2.0, (y - 170.0) / 10.0)


class FakePixelTrapHomography(FakeHomography):
    def world_to_pixel(self, x, y):
        return (x * 100.0 + 200.0, y * 10.0 + 170.0)


def _det(frame, pixel_y, world_y, pixel_x=160.0):
    return {
        "frame_index": frame,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "x": pixel_x / 100.0 - 2.0,
        "y": world_y,
        "confidence": 0.9,
    }


def _player_pose(frame):
    return {
        "frame_id": frame,
        "detections": [
            {
                "bbox": [150.0, 130.0, 170.0, 230.0],
                "conf": 0.9,
                "foot_px": [160.0, 230.0],
            }
        ],
    }


def _top_player_pose(frame):
    return {
        "frame_id": frame,
        "detections": [
            {
                "bbox": [150.0, 80.0, 170.0, 120.0],
                "conf": 0.9,
                "foot_px": [160.0, 110.0],
            }
        ],
    }


def test_yolo_events_hit_first_suppresses_nearby_raw_bounce():
    detections = [
        _det(frame, pixel_y, 5.0 if frame == 4 else 4.0)
        for frame, pixel_y in enumerate([100, 120, 140, 160, 140, 120, 100], start=1)
    ]

    result = detect_single_camera_events(
        detections,
        camera_name="cam68",
        player_pose_messages=[_player_pose(4)],
        homography=FakeHomography(),
        speed_min_kmh=0,
        speed_max_kmh=10_000,
    )

    assert result["raw_bounce_candidate_count"] == 1
    assert result["count"] == 0
    assert result["suppressed_bounces_by_hit_window"] == 1
    assert result["hit_count"] == 1
    assert result["hit_suppression_frames"] == [4]

    hit = result["hits"][0]
    assert hit["source"] == "bottom_reversal_player_anchor"
    assert hit["x"] == pytest.approx(hit["ball_x"])
    assert hit["y"] == pytest.approx(hit["player_court_y"])
    assert hit["y"] == pytest.approx(6.0)


def test_yolo_events_bottom_up_crossing_fallback_adds_hit():
    detections = [
        _det(frame, 100 + frame * 5, world_y)
        for frame, world_y in enumerate([5, 4, 3, 2, 1, -1, -2], start=1)
    ]

    result = detect_single_camera_events(
        detections,
        camera_name="cam68",
        player_pose_messages=[_player_pose(frame) for frame in range(1, 7)],
        homography=FakeHomography(),
        speed_min_kmh=0,
        speed_max_kmh=10_000,
    )

    assert result["speed_events"][0]["direction"] == "bottom_up"
    assert result["hits"][0]["source"] == "bottom_up_lookback"
    assert result["hits"][0]["crossing_frame"] == 6
    assert result["hits"][0]["y"] == pytest.approx(6.0)


def test_yolo_events_top_down_crossing_lookback_adds_top_hit():
    detections = [
        _det(frame, 100 + frame * 5, world_y)
        for frame, world_y in enumerate([-5, -4, -3, -2, -1, 1, 2], start=1)
    ]

    result = detect_single_camera_events(
        detections,
        camera_name="cam68",
        player_pose_messages=[_top_player_pose(frame) for frame in range(1, 7)],
        homography=FakeHomography(),
        speed_min_kmh=0,
        speed_max_kmh=10_000,
    )

    assert result["speed_events"][0]["direction"] == "top_down"
    assert result["hits"][0]["source"] == "top_down_lookback"
    assert result["hits"][0]["crossing_frame"] == 6
    assert result["hits"][0]["y"] == pytest.approx(-6.0)


def test_yolo_events_pixel_speed_trap_matches_offline_crossing():
    detections = [
        _det(1, 80.0, -5.0),
        _det(2, 100.0, -4.0),
        _det(3, 110.0, -3.0),
        # Deliberately huge world jump: the old world-speed gate filtered this,
        # while verify_tennis.py accepts it via the pixel speed trap.
        _det(4, 130.0, 500.0),
        _det(5, 180.0, 501.0),
        _det(6, 270.0, 502.0),
        _det(7, 330.0, 503.0),
    ]

    result = detect_single_camera_events(
        detections,
        camera_name="cam68",
        player_pose_messages=[_top_player_pose(3)],
        homography=FakePixelTrapHomography(),
        speed_max_kmh=100.0,
    )

    assert result["speed_events"]
    crossing = result["speed_events"][0]
    assert crossing["source"] == "single_cam_pixel_speed_trap"
    assert crossing["direction"] == "top_down"
    assert crossing["frame"] == 4
    assert crossing["speed_kmh"] == pytest.approx(90.0)
    assert result["hits"][0]["source"] == "top_down_lookback"
    assert result["hits"][0]["frame"] == 3


def test_yolo_bounce_cleaning_keeps_strongest_signal_in_window():
    selected, dropped = _select_strongest_bounces(
        [
            {
                "frame_index": 10,
                "x": 0.0,
                "y": 0.0,
                "angle": 20.0,
                "delta_v": 4.0,
                "confidence": 0.9,
                "y_reversal": False,
            },
            {
                "frame_index": 12,
                "x": 0.2,
                "y": 0.2,
                "angle": 70.0,
                "delta_v": 30.0,
                "confidence": 0.6,
                "y_reversal": True,
            },
            {
                "frame_index": 50,
                "x": 3.0,
                "y": 3.0,
                "angle": 10.0,
                "delta_v": 2.0,
                "confidence": 0.5,
                "y_reversal": False,
            },
        ],
        clean_time_frames=25,
        clean_space_meters=1.5,
    )

    assert [event["frame_index"] for event in selected] == [12, 50]
    assert [event["frame_index"] for event in dropped] == [10]
    assert dropped[0]["deduped_by_frame"] == 12
    assert selected[0]["bounce_signal_score"] > dropped[0]["bounce_signal_score"]


def test_yolo_out_rally_gate_waits_for_bottom_hit_restart():
    bounces = [
        {"frame_index": 10, "x": 0.0, "y": -8.0, "in_court": False},
        {"frame_index": 40, "x": 0.2, "y": -7.0, "in_court": True},
        {"frame_index": 130, "x": 0.3, "y": 3.0, "in_court": True},
        {"frame_index": 180, "x": 0.4, "y": 5.0, "in_court": True},
    ]
    hits = [
        {"frame_index": 120, "source": "top_down_lookback"},
        {"frame_index": 170, "source": "bottom_up_lookback"},
    ]

    kept, suppressed = _apply_out_rally_gate(
        bounces,
        hit_events=hits,
        speed_events=[],
        restart_hit_gap_frames=100,
        restart_speed_kmh=20.0,
    )

    assert [event["frame_index"] for event in kept] == [10, 180]
    assert [event["frame_index"] for event in suppressed] == [40, 130]
    assert all(event["suppression_reason"] == "out_rally_gate" for event in suppressed)


def test_yolo_out_rally_gate_uses_gate_only_out_without_publishing():
    bounces = [
        {
            "frame_index": 10,
            "x": 4.2,
            "y": -12.2,
            "in_court": False,
            "publishable": False,
            "gate_only": True,
        },
        {"frame_index": 40, "x": 0.2, "y": -7.0, "in_court": True},
        {"frame_index": 180, "x": 0.4, "y": 5.0, "in_court": True},
    ]
    hits = [{"frame_index": 170, "source": "bottom_up_lookback"}]

    kept, suppressed = _apply_out_rally_gate(
        bounces,
        hit_events=hits,
        speed_events=[],
        restart_hit_gap_frames=100,
        restart_speed_kmh=20.0,
    )

    assert [event["frame_index"] for event in kept] == [180]
    assert [event["frame_index"] for event in suppressed] == [40]


def test_yolo_near_boundary_weak_out_is_gate_only_candidate():
    weak_out = {
        "frame_index": 2221,
        "x": 4.4194,
        "y": -12.7802,
        "in_court": False,
        "angle": 112.1,
        "delta_v": 2.27,
        "confidence": 0.3858,
        "y_reversal": True,
    }
    clear_out = {
        "frame_index": 2736,
        "x": 1.9542,
        "y": -18.7519,
        "in_court": False,
        "angle": 173.34,
        "delta_v": 3.32,
        "confidence": 0.1671,
        "y_reversal": True,
    }

    assert _is_gate_only_weak_out_bounce(weak_out)
    assert not _is_gate_only_weak_out_bounce(clear_out)


def test_yolo_out_rally_gate_restarts_on_speed():
    kept, suppressed = _apply_out_rally_gate(
        [
            {"frame_index": 10, "x": 0.0, "y": -8.0, "in_court": False},
            {"frame_index": 40, "x": 0.2, "y": -7.0, "in_court": True},
            {"frame_index": 80, "x": 0.3, "y": 3.0, "in_court": True},
        ],
        hit_events=[],
        speed_events=[{"frame_index": 60, "speed_kmh": 42}],
        restart_hit_gap_frames=100,
        restart_speed_kmh=20.0,
    )

    assert [event["frame_index"] for event in kept] == [10, 80]
    assert [event["frame_index"] for event in suppressed] == [40]


def test_dashboard_yolo_quality_reject_reason_matches_live_publish_gate():
    assert (
        dashboard_yolo_quality_reject_reason(
            {
                "frame_index": 21,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 25,
                "angle": 171.1,
                "bounce_signal_score": 208.5,
                "y_reversal": True,
            }
        )
        == "quality_warmup"
    )
    assert (
        dashboard_yolo_quality_reject_reason(
            {
                "frame_index": 696,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 7,
                "angle": 84.6,
                "bounce_signal_score": 139.4,
                "y_reversal": True,
            }
        )
        == "quality_short_track"
    )
    assert (
        dashboard_yolo_quality_reject_reason(
            {
                "frame_index": 527,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 29,
                "angle": 11.5,
                "bounce_signal_score": 52.2,
                "y_reversal": False,
            }
        )
        == "quality_weak_non_reversal"
    )


def test_dashboard_yolo_publish_filter_matches_live_final_layer():
    result = filter_dashboard_yolo_publishable_bounces(
        [
            {
                "frame_index": 21,
                "x": 0.0,
                "y": 0.0,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 25,
                "angle": 171.0,
                "bounce_signal_score": 208.0,
                "y_reversal": True,
            },
            {
                "frame_index": 100,
                "x": 0.0,
                "y": 0.0,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 30,
                "angle": 100.0,
                "bounce_signal_score": 150.0,
                "y_reversal": True,
            },
            {
                "frame_index": 114,
                "x": 0.2,
                "y": 0.1,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 30,
                "angle": 110.0,
                "bounce_signal_score": 160.0,
                "y_reversal": True,
            },
            {
                "frame_index": 150,
                "x": 3.0,
                "y": 3.0,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 30,
                "angle": 100.0,
                "bounce_signal_score": 150.0,
                "y_reversal": True,
            },
            {
                "frame_index": 196,
                "x": 5.0,
                "y": 5.0,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 30,
                "angle": 100.0,
                "bounce_signal_score": 150.0,
                "y_reversal": True,
            },
        ],
        hit_events=[{"frame_index": 151}],
        latest_frame=200,
        hit_suppress_frames=5,
        clean_time_frames=25,
        clean_space_meters=1.5,
        release_delay_frames=10,
    )

    assert [event["frame_index"] for event in result["bounces"]] == [114]
    assert result["suppression_counts"] == {
        "quality_warmup": 1,
        "hit_window": 1,
        "release_delay": 1,
        "duplicate_live_bounce": 1,
    }
    duplicate = [
        event
        for event in result["suppressed_bounces"]
        if event["publish_suppression_reason"] == "duplicate_live_bounce"
    ][0]
    assert duplicate["frame_index"] == 100
    assert duplicate["deduped_by_frame"] == 114


def test_dashboard_yolo_publish_filter_suppresses_hit_window_shadow_candidate():
    result = filter_dashboard_yolo_publishable_bounces(
        [
            {
                "frame_index": 901,
                "x": 1.11,
                "y": -13.36,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 10,
                "angle": 120.0,
                "bounce_signal_score": 117.0,
                "y_reversal": True,
            },
            {
                "frame_index": 903,
                "x": 1.19,
                "y": -13.15,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 10,
                "angle": 120.0,
                "bounce_signal_score": 148.0,
                "y_reversal": True,
            },
            {
                "frame_index": 932,
                "x": 1.06,
                "y": 8.13,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 29,
                "angle": 21.4,
                "bounce_signal_score": 103.0,
                "y_reversal": False,
            },
        ],
        hit_events=[{"frame_index": 897}],
        latest_frame=1000,
        hit_suppress_frames=5,
        clean_time_frames=25,
        clean_space_meters=1.5,
        release_delay_frames=10,
    )

    assert [event["frame_index"] for event in result["bounces"]] == [932]
    suppressed = [
        (event["frame_index"], event["publish_suppression_reason"], event.get("hit_window_shadow_frame"))
        for event in result["suppressed_bounces"]
    ]
    assert suppressed == [
        (901, "hit_window", None),
        (903, "hit_window", 901),
    ]


def test_dashboard_yolo_publish_filter_keeps_bounce_outside_hit_shadow_window():
    result = filter_dashboard_yolo_publishable_bounces(
        [
            {
                "frame_index": 932,
                "x": 1.06,
                "y": 8.13,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 29,
                "angle": 21.4,
                "bounce_signal_score": 103.0,
                "y_reversal": False,
            },
            {
                "frame_index": 944,
                "x": 1.12,
                "y": 8.2,
                "source": "yolo_verify_queue_single_cam",
                "queue_history_len": 29,
                "angle": 120.0,
                "bounce_signal_score": 120.0,
                "y_reversal": True,
            },
        ],
        hit_events=[{"frame_index": 945}],
        latest_frame=1000,
        hit_suppress_frames=5,
        clean_time_frames=25,
        clean_space_meters=1.5,
        release_delay_frames=10,
    )

    assert [event["frame_index"] for event in result["bounces"]] == [932]
    assert [
        (event["frame_index"], event["publish_suppression_reason"])
        for event in result["suppressed_bounces"]
    ] == [(944, "hit_window")]
