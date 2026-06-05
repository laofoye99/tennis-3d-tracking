"""YOLO pose-based player detector.

Wraps ultralytics YOLO pose model to detect tennis players and their
17-keypoint skeleton. Only the `person` class (class 0) is returned.

Foot position for court-plane homography is derived from ankle keypoints
(COCO kp 15 = left_ankle, kp 16 = right_ankle) when visible; otherwise
falls back to the bottom-center of the bounding box.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO keypoint indices used for foot position
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16
_KP_VIS_THRESH = 0.3  # minimum keypoint confidence to consider visible


class PlayerPoseDetector:
    """Detect players and their pose using a YOLO pose model.

    Args:
        model_path: Path to YOLO pose weights (e.g. yolo26x-pose.pt).
        device: "cuda" or "cpu".
        conf: Minimum detection confidence.
        run_every_n: Only run inference on every N-th call to ``detect()``.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf: float = 0.4,
        imgsz: int = 960,
        use_tracking: bool = False,
        run_every_n: int = 5,
    ):
        from ultralytics import YOLO

        logger.info("Loading player pose model: %s (device=%s)", model_path, device)
        self._model = YOLO(model_path)
        self._model.to(device)
        self._conf = conf
        self._imgsz = int(imgsz or 960)
        self._use_tracking = bool(use_tracking)
        self._run_every_n = run_every_n
        self._call_count = 0
        self._last_results: list[dict] = []
        self.last_inference_ran = False
        logger.info(
            "PlayerPoseDetector ready (conf=%.2f, imgsz=%d, tracking=%s, run_every_n=%d)",
            conf,
            self._imgsz,
            self._use_tracking,
            run_every_n,
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run player detection on a BGR frame.

        Returns the same result as the previous call when skipped (every-N
        logic), so callers always get the most recent valid result.

        Returns:
            List of dicts, one per detected person:
                bbox      – [x1, y1, x2, y2] in pixels
                conf      – detection confidence
                foot_px   – [px, py] best foot pixel position (for homography)
                keypoints – list of 17 × [px, py, conf]
        """
        self._call_count += 1
        if (self._call_count - 1) % self._run_every_n != 0:
            self.last_inference_ran = False
            return self._last_results
        self.last_inference_ran = True

        infer_kwargs = {
            "classes": [0],          # person only
            "conf": self._conf,
            "imgsz": self._imgsz,
            "verbose": False,
            "device": self._model.device,
        }
        if self._use_tracking:
            try:
                results = self._model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    **infer_kwargs,
                )
            except Exception:
                logger.debug("Player tracking failed; falling back to predict", exc_info=True)
                results = self._model.predict(frame, **infer_kwargs)
        else:
            results = self._model.predict(frame, **infer_kwargs)

        detections: list[dict] = []
        if not results:
            self._last_results = detections
            return detections

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            self._last_results = detections
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()       # (N, 4)
        confs = res.boxes.conf.cpu().numpy()       # (N,)
        track_ids = (
            res.boxes.id.int().cpu().numpy().tolist()
            if getattr(res.boxes, "id", None) is not None
            else [None] * len(boxes)
        )
        kps_arr = (
            res.keypoints.data.cpu().numpy()       # (N, 17, 3)  [x, y, conf]
            if res.keypoints is not None
            else None
        )

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            det_conf = float(confs[i])

            kps: list[list[float]] = []
            foot_px: list[float] = [float((x1 + x2) / 2), float(y2)]  # bbox fallback

            if kps_arr is not None:
                raw = kps_arr[i]  # (17, 3)
                kps = [[float(raw[j, 0]), float(raw[j, 1]), float(raw[j, 2])]
                       for j in range(17)]

                # Prefer ankle midpoint for court projection
                lv = kps[_LEFT_ANKLE][2] >= _KP_VIS_THRESH
                rv = kps[_RIGHT_ANKLE][2] >= _KP_VIS_THRESH
                if lv and rv:
                    foot_px = [
                        (kps[_LEFT_ANKLE][0] + kps[_RIGHT_ANKLE][0]) / 2,
                        (kps[_LEFT_ANKLE][1] + kps[_RIGHT_ANKLE][1]) / 2,
                    ]
                elif lv:
                    foot_px = kps[_LEFT_ANKLE][:2]
                elif rv:
                    foot_px = kps[_RIGHT_ANKLE][:2]

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": det_conf,
                "foot_px": foot_px,
                "keypoints": kps,
                "track_id": int(track_ids[i]) if track_ids[i] is not None else None,
            })

        self._last_results = detections
        return detections
