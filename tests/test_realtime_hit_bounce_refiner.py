from types import SimpleNamespace

import pytest

from app.realtime_hit_bounce import HitBounceRefiner


def _cfg(**overrides):
    defaults = {
        "enabled": True,
        "show_hits_on_minimap": True,
        "lookback_frames": 5,
        "release_delay_frames": 5,
        "hit_suppression_frames": 3,
        "hit_angle_thresh": 45.0,
        "top_hit_dist_px": 50.0,
        "bottom_hit_dist_px_net": 100.0,
        "bottom_hit_dist_px_base": 250.0,
        "top_hit_dist_m": 1.2,
        "bottom_hit_dist_m_net": 1.2,
        "bottom_hit_dist_m_base": 2.5,
        "clean_time_frames": 25,
        "clean_space_meters": 1.5,
        "history_frames": 30,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _point(frame, x=0.0, y=0.0, z=0.5):
    ts = frame / 25.0
    return {
        "x": x,
        "y": y,
        "z": z,
        "timestamp": ts,
        "capture_ts": ts,
        "frame_index": frame,
    }


def _bounce(frame, x=0.0, y=0.0):
    return {
        "x": x,
        "y": y,
        "z": 0.05,
        "timestamp": frame / 25.0,
        "capture_ts": frame / 25.0,
        "frame_index": frame,
        "in_court": True,
    }


def _far_player(anchor_y=3.0, foot_y=6.0):
    return {
        "camera_name": "cam68",
        "side": "far",
        "foot_court": [0.2, foot_y],
        "hit_anchor_court": [0.0, anchor_y],
        "hit_anchor_px": [100.0, 100.0],
    }


def test_hit_window_suppresses_pending_raw_bounce_after_hit_detection():
    refiner = HitBounceRefiner(_cfg())

    # Frame 3 is both the raw bounce candidate and the bottom-half reversal
    # candidate. Publication is delayed, so the HIT can suppress it first.
    refiner.update(_point(1, y=1.0), players=[_far_player()])
    refiner.update(_point(2, y=2.0), players=[_far_player()])
    refiner.update(_point(3, y=3.0), raw_bounce=_bounce(3, y=3.0), players=[_far_player()])
    refiner.update(_point(4, y=2.4), players=[_far_player()])
    out = refiner.update(_point(5, y=1.2), players=[_far_player()])

    assert out["new_hits"]
    hit = out["new_hits"][0]
    assert hit["source"] == "bottom_reversal_player_anchor"
    assert hit["x"] == pytest.approx(0.0)
    assert hit["y"] == pytest.approx(6.0)

    for frame in range(6, 8):
        out = refiner.update(_point(frame, y=1.0), players=[_far_player()])
        assert out["new_final_bounces"] == []

    out = refiner.update(_point(8, y=1.0), players=[_far_player()])
    assert out["new_final_bounces"] == []
    assert out["suppressed_bounces"][0]["frame_index"] == 3
    assert out["stats"]["raw_bounce_candidate_count"] == 1
    assert out["stats"]["suppressed_bounces_by_hit"] == 1


def test_bottom_up_lookback_adds_hit_when_direct_hit_was_missing():
    refiner = HitBounceRefiner(_cfg())

    for frame in range(1, 6):
        refiner.update(
            _point(frame, y=4.0),
            players=[_far_player(anchor_y=4.0, foot_y=7.0)],
        )

    out = refiner.update(
        _point(6, y=-0.5),
        players=[_far_player(anchor_y=4.0, foot_y=7.0)],
        net_crossing={
            "direction": "far_to_near",
            "frame_index": 6,
            "timestamp": 6 / 25.0,
        },
    )

    assert len(out["new_hits"]) == 1
    hit = out["new_hits"][0]
    assert hit["source"] == "bottom_up_lookback"
    assert hit["crossing_direction"] == "far_to_near"
    assert hit["y"] == pytest.approx(7.0)
    assert out["stats"]["bottom_up_lookback_hits"] == 1


def test_bounce_dedupe_happens_after_hit_stage():
    refiner = HitBounceRefiner(_cfg(release_delay_frames=0))

    out = refiner.update(_point(1), raw_bounce=_bounce(1, x=0.0, y=0.0), players=[])
    assert [b["frame_index"] for b in out["new_final_bounces"]] == [1]

    out = refiner.update(_point(2), raw_bounce=_bounce(2, x=0.4, y=0.2), players=[])
    assert out["new_final_bounces"] == []
    assert out["stats"]["raw_bounce_candidate_count"] == 2
    assert out["stats"]["deduped_bounces_after_hit"] == 1
    assert out["stats"]["final_bounce_count"] == 1


def test_crossing_lookback_can_use_pixel_thresholds_when_available():
    refiner = HitBounceRefiner(_cfg())
    cam_dets = {"cam66": {"pixel_x": 110.0, "pixel_y": 105.0}}
    player = {
        "camera_name": "cam66",
        "side": "near",
        "foot_court": [0.1, -6.0],
        "hit_anchor_court": [0.0, -2.0],
        "hit_anchor_px": [100.0, 100.0],
    }
    for frame in range(1, 5):
        refiner.update(_point(frame, y=-2.0), players=[player], cam_dets=cam_dets)

    out = refiner.update(
        _point(5, y=0.2),
        players=[player],
        cam_dets=cam_dets,
        net_crossing={
            "direction": "near_to_far",
            "frame_index": 5,
            "timestamp": 5 / 25.0,
        },
    )

    assert out["new_hits"][0]["source"] == "top_down_lookback"
    assert out["new_hits"][0]["distance_unit"] == "px"
    assert out["new_hits"][0]["distance"] < 50.0
