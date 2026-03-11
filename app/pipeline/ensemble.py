"""Ensemble detector: runs TrackNet + HRNet in parallel, cross-validates detections.

Architecture:
    - TrackNet (PyTorch): seq_len=8, processes 8 frames per batch, outputs 8 heatmaps
    - HRNet (ONNX): frames_in=3, processes 3 frames per batch, outputs 3 heatmaps
    - Shared frame buffer: TrackNet uses all 8 frames, HRNet uses the last 3
    - Cross-validation: per-frame comparison of pixel detections from both models
    - Output: fused detections with source tag and adjusted confidence
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.pipeline.inference import BallDetector, TrackNetDetector
from app.pipeline.postprocess import BallTracker

logger = logging.getLogger(__name__)


@dataclass
class EnsembleStats:
    """Running statistics for ensemble cross-validation."""
    agree: int = 0
    disagree: int = 0
    tracknet_only: int = 0
    hrnet_only: int = 0
    neither: int = 0

    def to_dict(self) -> dict:
        total = self.agree + self.disagree + self.tracknet_only + self.hrnet_only + self.neither
        return {
            "agree": self.agree,
            "disagree": self.disagree,
            "tracknet_only": self.tracknet_only,
            "hrnet_only": self.hrnet_only,
            "neither": self.neither,
            "total": total,
            "agree_rate": round(self.agree / total, 3) if total > 0 else 0.0,
        }


@dataclass
class FrameDetection:
    """A single detection from one model."""
    pixel_x: float
    pixel_y: float
    confidence: float


class EnsembleDetector:
    """Runs TrackNet and HRNet simultaneously with cross-validation.

    Usage in the video pipeline:
        1. Create EnsembleDetector with both model paths
        2. Feed masked frames one at a time via ``feed_frame()``
        3. When enough frames accumulate (8 for TrackNet), both models run
        4. Returns list of fused detections for the batch

    The ensemble handles the different batch sizes internally:
        - TrackNet needs 8 frames → runs when buffer reaches 8
        - HRNet needs 3 frames → runs on the last 3 frames of the buffer
        - Cross-validation aligns detections by their position within the batch
    """

    def __init__(
        self,
        tracknet_path: str,
        hrnet_path: str,
        input_size: tuple[int, int],
        device: str,
        threshold: float,
        original_size: tuple[int, int],
        agree_distance: float = 3.0,
        boost_factor: float = 1.2,
        penalty_factor: float = 0.6,
        single_factor: float = 0.8,
    ):
        self.input_size = input_size
        self.threshold = threshold
        self.agree_distance = agree_distance
        self.boost_factor = boost_factor
        self.penalty_factor = penalty_factor
        self.single_factor = single_factor

        # TrackNet: 8 frames in, 8 heatmaps out
        self.tracknet = TrackNetDetector(
            model_path=tracknet_path,
            input_size=input_size,
            frames_in=8,
            frames_out=8,
            device=device,
        )

        # HRNet: 3 frames in, 3 heatmaps out
        self.hrnet = BallDetector(
            model_path=hrnet_path,
            input_size=input_size,
            frames_in=3,
            frames_out=3,
            device=device,
        )

        # Each model gets its own BallTracker (stateful position history)
        self.tracker_tn = BallTracker(original_size=original_size, threshold=threshold)
        self.tracker_hr = BallTracker(original_size=original_size, threshold=threshold)

        # Frame buffer (accumulates until we have 8 for TrackNet)
        self._frame_buffer: list[np.ndarray] = []
        self._frame_count: int = 0  # total frames fed

        self.stats = EnsembleStats()

        logger.info(
            "EnsembleDetector ready: TrackNet(%s) + HRNet(%s), agree_dist=%.1fm",
            tracknet_path, hrnet_path, agree_distance,
        )

    @property
    def frames_in(self) -> int:
        """TrackNet batch size determines the overall batch cadence."""
        return 8

    @property
    def frames_out(self) -> int:
        return 8

    def compute_video_median(self, cap, start_frame: int, end_frame: int) -> None:
        """Delegate to TrackNet's median computation (HRNet doesn't need it)."""
        self.tracknet.compute_video_median(cap, start_frame, end_frame)

    def infer_ensemble(self, frames: list[np.ndarray]) -> list[Optional[tuple[float, float, float, str]]]:
        """Run both models on the frame batch and cross-validate.

        Args:
            frames: List of 8 BGR frames (masked).

        Returns:
            List of 8 results, one per frame. Each is either:
                (pixel_x, pixel_y, confidence, source) or None.
            source is one of: "ensemble_agree", "ensemble_disagree",
                "tracknet_only", "hrnet_only".
        """
        assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"

        # --- TrackNet inference (all 8 frames) ---
        try:
            tn_heatmaps = self.tracknet.infer(frames)  # (8, H, W)
        except Exception as e:
            logger.error("TrackNet inference error: %s", e)
            tn_heatmaps = None

        # --- HRNet inference (last 3 frames) ---
        hrnet_frames = frames[-3:]
        try:
            hr_heatmaps = self.hrnet.infer(hrnet_frames)  # (3, H, W)
        except Exception as e:
            logger.error("HRNet inference error: %s", e)
            hr_heatmaps = None

        # --- Post-process heatmaps to pixel detections ---
        tn_detections: list[Optional[FrameDetection]] = [None] * 8
        hr_detections: list[Optional[FrameDetection]] = [None] * 8

        if tn_heatmaps is not None:
            for i in range(min(8, len(tn_heatmaps))):
                result = self.tracker_tn.process_heatmap(tn_heatmaps[i])
                if result is not None:
                    px, py, conf = result
                    tn_detections[i] = FrameDetection(px, py, conf)

        if hr_heatmaps is not None:
            for i in range(min(3, len(hr_heatmaps))):
                # HRNet outputs correspond to frames[-3:], which are indices 5,6,7
                frame_idx = 5 + i
                result = self.tracker_hr.process_heatmap(hr_heatmaps[i])
                if result is not None:
                    px, py, conf = result
                    hr_detections[frame_idx] = FrameDetection(px, py, conf)

        # --- Cross-validate ---
        results: list[Optional[tuple[float, float, float, str]]] = []
        for i in range(8):
            tn = tn_detections[i]
            hr = hr_detections[i]
            fused = self._fuse_detection(tn, hr)
            results.append(fused)

        return results

    def _fuse_detection(
        self,
        tn: Optional[FrameDetection],
        hr: Optional[FrameDetection],
    ) -> Optional[tuple[float, float, float, str]]:
        """Fuse detections from both models for a single frame.

        Returns (pixel_x, pixel_y, confidence, source) or None.
        """
        if tn is None and hr is None:
            self.stats.neither += 1
            return None

        if tn is not None and hr is None:
            # TrackNet only — for frames 0-4 this is always the case (HRNet only covers 5-7)
            self.stats.tracknet_only += 1
            return (tn.pixel_x, tn.pixel_y, tn.confidence * self.single_factor, "tracknet_only")

        if tn is None and hr is not None:
            self.stats.hrnet_only += 1
            return (hr.pixel_x, hr.pixel_y, hr.confidence * self.single_factor, "hrnet_only")

        # Both detected — compute pixel distance
        assert tn is not None and hr is not None
        dx = tn.pixel_x - hr.pixel_x
        dy = tn.pixel_y - hr.pixel_y
        pixel_dist = math.sqrt(dx * dx + dy * dy)

        # Convert pixel distance to rough world distance for threshold comparison
        # Rough approximation: 1920px ≈ 24m court length → ~80px/m
        # We use a pixel threshold instead: agree_distance * 80
        pixel_threshold = self.agree_distance * 80

        if pixel_dist < pixel_threshold:
            # Models agree — weighted average biased toward higher confidence
            self.stats.agree += 1
            total_conf = tn.confidence + hr.confidence
            if total_conf > 0:
                w_tn = tn.confidence / total_conf
                w_hr = hr.confidence / total_conf
            else:
                w_tn = w_hr = 0.5
            fused_x = tn.pixel_x * w_tn + hr.pixel_x * w_hr
            fused_y = tn.pixel_y * w_tn + hr.pixel_y * w_hr
            fused_conf = max(tn.confidence, hr.confidence) * self.boost_factor
            return (fused_x, fused_y, fused_conf, "ensemble_agree")
        else:
            # Models disagree — pick higher confidence, apply penalty
            self.stats.disagree += 1
            if tn.confidence >= hr.confidence:
                return (tn.pixel_x, tn.pixel_y, tn.confidence * self.penalty_factor, "ensemble_disagree")
            else:
                return (hr.pixel_x, hr.pixel_y, hr.confidence * self.penalty_factor, "ensemble_disagree")
