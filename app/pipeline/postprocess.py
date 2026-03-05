"""Heatmap post-processing and ball tracking."""

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BallTracker:
    """Post-processes heatmaps to extract ball pixel coordinates with tracking."""

    def __init__(
        self,
        original_size: tuple[int, int] = (1920, 1080),
        threshold: float = 0.5,
        history_len: int = 3,
    ):
        self.orig_w, self.orig_h = original_size
        self.threshold = threshold
        self.prev_positions: deque[np.ndarray] = deque(maxlen=history_len)

    def predict_position(self) -> Optional[np.ndarray]:
        """Linear extrapolation from recent positions."""
        if len(self.prev_positions) < 2:
            return None
        positions = list(self.prev_positions)
        velocity = positions[-1] - positions[-2]
        return positions[-1] + velocity

    def process_heatmap(self, heatmap: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Extract ball (x, y, confidence) in original image coordinates from a single heatmap.

        Args:
            heatmap: 2D array (H_model, W_model) with values in [0, 1].

        Returns:
            (pixel_x, pixel_y, confidence) or None if not detected.
        """
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(
            heatmap, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR
        )
        # Threshold
        heatmap_filtered = np.where(
            heatmap_resized > self.threshold, heatmap_resized, 0.0
        ).astype(np.float32)

        binary = (heatmap_filtered > 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        blob_centers: list[tuple[float, float, float]] = []
        for j in range(1, num_labels):
            mask = labels_im == j
            blob_sum = float(heatmap_filtered[mask].sum())
            if blob_sum <= 0:
                continue
            cx = float(np.sum(np.where(mask)[1] * heatmap_filtered[mask]) / blob_sum)
            cy = float(np.sum(np.where(mask)[0] * heatmap_filtered[mask]) / blob_sum)
            blob_centers.append((cx, cy, blob_sum))

        if not blob_centers:
            return None

        predicted = self.predict_position()
        if predicted is not None:
            distances = [
                np.sqrt((c[0] - predicted[0]) ** 2 + (c[1] - predicted[1]) ** 2)
                for c in blob_centers
            ]
            best = blob_centers[int(np.argmin(distances))]
        else:
            blob_centers.sort(key=lambda c: c[2], reverse=True)
            best = blob_centers[0]

        cx, cy, conf = best
        self.prev_positions.append(np.array([cx, cy]))
        return cx, cy, conf
