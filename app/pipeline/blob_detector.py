"""
Tennis ball blob detector using 30-frame median background subtraction.

Best method: MedianBG with threshold=10
- cam66 recall: 94.2% (1855/1969)
- cam68 recall: 94.0% (1482/1577)
- Average detections per frame: ~67-89

Input: 30 frames as a block (non-overlapping)
Output: Per-frame blob candidates (cx, cy) that include the ball with >90% recall.

Usage:
    detector = BallBlobDetector(thresh=10, min_area=2, max_area=600)
    detections = detector.detect_block(list_of_30_grayscale_frames)
    # detections[i] = list of (cx, cy) for frame i
"""

import cv2
import numpy as np


class BallBlobDetector:
    """
    30-frame block median background subtraction for tennis ball detection.

    Approach:
    1. Take a block of 30 consecutive grayscale frames
    2. Compute per-pixel median as background model
    3. For each frame, compute |frame - median|
    4. Threshold the difference image
    5. Morphological opening to remove noise
    6. Find contours as blob candidates

    The ball appears as a moving object that differs from the median background.
    """

    def __init__(self, thresh=10, min_area=2, max_area=600, morph_size=3):
        self.thresh = thresh
        self.min_area = min_area
        self.max_area = max_area
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )

    def compute_median_bg(self, frames):
        """Compute per-pixel median background from a stack of frames."""
        stack = np.array(frames, dtype=np.uint8)
        stack.sort(axis=0)
        return stack[len(frames) // 2]

    def detect_blobs(self, mask):
        """Extract blob centroids from binary mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w / 2, y + h / 2
            results.append((cx, cy))
        return results

    def detect_block(self, frames):
        """
        Detect blobs in a block of frames.

        Args:
            frames: List of grayscale numpy arrays (e.g., 30 frames).

        Returns:
            Dict mapping frame index (0-based within block) to list of (cx, cy).
        """
        if len(frames) < 2:
            return {0: []}

        median_bg = self.compute_median_bg(frames)

        results = {}
        for i, frame in enumerate(frames):
            diff = cv2.absdiff(frame, median_bg)
            _, binary = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            results[i] = self.detect_blobs(binary)

        return results
