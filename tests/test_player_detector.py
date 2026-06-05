from types import SimpleNamespace
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.player_detector import PlayerPoseDetector


class _ArrayWrap:
    def __init__(self, value):
        self._value = np.asarray(value, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Boxes:
    def __init__(self):
        self.xyxy = _ArrayWrap([[10.0, 20.0, 30.0, 60.0]])
        self.conf = _ArrayWrap([0.9])

    def __len__(self):
        return 1


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.predict_calls = 0

    def to(self, _device):
        return self

    def predict(self, *_args, **_kwargs):
        self.predict_calls += 1
        return [SimpleNamespace(boxes=_Boxes(), keypoints=None)]


def test_player_detector_run_every_one_runs_each_call(monkeypatch):
    fake_model = _FakeModel()

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda _path: fake_model)

    detector = PlayerPoseDetector("fake.pt", device="cpu", conf=0.1, run_every_n=1)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.detect(frame)
    assert detector.last_inference_ran is True
    detector.detect(frame)
    assert detector.last_inference_ran is True
    detector.detect(frame)
    assert detector.last_inference_ran is True

    assert fake_model.predict_calls == 3


def test_player_detector_skipped_frames_return_cached_result(monkeypatch):
    fake_model = _FakeModel()

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda _path: fake_model)

    detector = PlayerPoseDetector("fake.pt", device="cpu", conf=0.1, run_every_n=2)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first = detector.detect(frame)
    assert detector.last_inference_ran is True
    second = detector.detect(frame)
    assert detector.last_inference_ran is False
    third = detector.detect(frame)
    assert detector.last_inference_ran is True

    assert fake_model.predict_calls == 2
    assert second == first
    assert third == first
