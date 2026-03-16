"""Heatmap post-processing: extract ball blob candidates from TrackNet heatmaps."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BallTracker:
    """Post-processes heatmaps to extract ball pixel coordinates.

    Pure blob detection — no cross-frame prediction or history.
    Blob selection is left to downstream filters (e.g. court-X).
    """

    def __init__(
        self,
        original_size: tuple[int, int] = (1920, 1080),
        threshold: float = 0.5,
        heatmap_mask: Optional[list[tuple[int, int, int, int]]] = None,
        **_kwargs,
    ):
        self.orig_w, self.orig_h = original_size
        self.threshold = threshold
        self.heatmap_mask = heatmap_mask or []

    def _apply_mask(self, heatmap: np.ndarray) -> np.ndarray:
        """Zero out masked regions (e.g. camera OSD timestamp) on the heatmap.

        Mask rects are in original image coordinates (x0, y0, x1, y1).
        They are scaled to heatmap resolution before applying.
        """
        if not self.heatmap_mask:
            return heatmap
        hm_h, hm_w = heatmap.shape[:2]
        sx = hm_w / self.orig_w
        sy = hm_h / self.orig_h
        hm = heatmap.copy()
        for x0, y0, x1, y1 in self.heatmap_mask:
            mx0 = int(x0 * sx)
            my0 = int(y0 * sy)
            mx1 = int(x1 * sx + 0.5)
            my1 = int(y1 * sy + 0.5)
            hm[my0:my1, mx0:mx1] = 0.0
        return hm

    def _find_blobs(
        self,
        heatmap: np.ndarray,
        threshold: float,
        scale_x: float,
        scale_y: float,
        max_blobs: int,
    ) -> list[dict]:
        """Core blob detection at a given threshold.

        Returns list of blob dicts sorted by blob_sum descending.
        """
        heatmap_filtered = np.where(
            heatmap > threshold, heatmap, 0.0
        ).astype(np.float32)

        binary = (heatmap_filtered > 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        blobs: list[dict] = []
        for j in range(1, num_labels):
            mask = labels_im == j
            blob_sum = float(heatmap_filtered[mask].sum())
            if blob_sum <= 0:
                continue
            cx = float(np.sum(np.where(mask)[1] * heatmap_filtered[mask]) / blob_sum)
            cy = float(np.sum(np.where(mask)[0] * heatmap_filtered[mask]) / blob_sum)
            blob_max = float(heatmap[mask].max())
            blob_area = int(stats[j, cv2.CC_STAT_AREA])
            blobs.append({
                "pixel_x": cx * scale_x,
                "pixel_y": cy * scale_y,
                "blob_sum": blob_sum,
                "blob_max": blob_max,
                "blob_area": blob_area,
            })

        blobs.sort(key=lambda b: b["blob_sum"], reverse=True)
        return blobs[:max_blobs]

    def process_heatmap(self, heatmap: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Extract ball (x, y, confidence) from a single heatmap.

        Returns the top-1 blob by blob_sum, or None if nothing detected.
        """
        heatmap = self._apply_mask(heatmap)
        hm_h, hm_w = heatmap.shape[:2]
        scale_x = self.orig_w / hm_w
        scale_y = self.orig_h / hm_h

        blobs = self._find_blobs(heatmap, self.threshold, scale_x, scale_y, max_blobs=5)
        if not blobs:
            return None

        best = blobs[0]
        return best["pixel_x"], best["pixel_y"], best["blob_sum"]

    def process_heatmap_multi(
        self, heatmap: np.ndarray, max_blobs: int = 3
    ) -> list[dict]:
        """Extract up to max_blobs candidates from a heatmap.

        Returns blob dicts sorted by blob_sum descending.
        Each dict has: pixel_x, pixel_y, blob_sum, blob_max, blob_area.
        """
        heatmap = self._apply_mask(heatmap)
        hm_h, hm_w = heatmap.shape[:2]
        scale_x = self.orig_w / hm_w
        scale_y = self.orig_h / hm_h

        return self._find_blobs(heatmap, self.threshold, scale_x, scale_y, max_blobs)
