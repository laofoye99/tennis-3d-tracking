"""YOLO-based secondary blob verification for multi-blob disambiguation.

When TrackNet detects multiple blobs (dead balls, reflections, false positives),
this module crops each blob's neighborhood and runs YOLO detection to verify
which crops actually contain a tennis ball, optionally refining the ball center.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class ID for "sports ball"
SPORTS_BALL_CLASS = 32


def extract_crops(
    frame: np.ndarray,
    blobs: list[dict],
    crop_size: int = 128,
) -> list[np.ndarray]:
    """Crop regions around each blob centroid from the original frame.

    Args:
        frame: Original BGR frame (H, W, 3).
        blobs: Blob dicts with pixel_x, pixel_y keys (original image coords).
        crop_size: Size of square crop around each blob center.

    Returns:
        List of BGR crop arrays, each (crop_size, crop_size, 3).
        Edge blobs are zero-padded.
    """
    h, w = frame.shape[:2]
    half = crop_size // 2
    crops = []

    for blob in blobs:
        cx = int(round(blob["pixel_x"]))
        cy = int(round(blob["pixel_y"]))

        # Compute source region (may be clipped at frame boundaries)
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)

        # Destination offsets within the zero-padded crop
        dx0 = x0 - (cx - half)
        dy0 = y0 - (cy - half)
        dx1 = dx0 + (x1 - x0)
        dy1 = dy0 + (y1 - y0)

        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        crop[dy0:dy1, dx0:dx1] = frame[y0:y1, x0:x1]
        crops.append(crop)

    return crops


class BlobVerifier:
    """YOLO-based secondary detector for blob verification.

    Loads a YOLO model (COCO pretrained or fine-tuned) and runs detection
    on cropped blob regions to verify ball presence and refine position.
    """

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        crop_size: int = 128,
        conf: float = 0.25,
        device: str = "cuda",
        target_classes: Optional[list[int]] = None,
    ):
        """Initialize the YOLO verifier.

        Args:
            model_path: Path to YOLO weights. Use "yolo26n.pt" for COCO pretrained.
            crop_size: Crop size for blob extraction.
            conf: Minimum detection confidence threshold.
            device: "cuda" or "cpu".
            target_classes: COCO class IDs to accept. Defaults to [32] (sports ball).
                Set to None for fine-tuned single-class models.
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.crop_size = crop_size
        self.conf = conf
        self.device = device
        self.target_classes = target_classes if target_classes is not None else [SPORTS_BALL_CLASS]
        self._is_single_cls = False

        # Detect if this is a single-class fine-tuned model
        if hasattr(self.model, "names") and len(self.model.names) == 1:
            self._is_single_cls = True
            self.target_classes = [0]

        logger.info(
            "BlobVerifier loaded: model=%s, crop=%d, conf=%.2f, classes=%s",
            model_path, crop_size, conf, self.target_classes,
        )

    def detect_crops(self, crops: list[np.ndarray]) -> list[Optional[dict]]:
        """Run YOLO detection on a batch of crops.

        Args:
            crops: List of BGR crop images.

        Returns:
            List of detection results (one per crop). Each is either None
            (no ball detected) or a dict with:
                - yolo_conf: detection confidence
                - crop_cx, crop_cy: ball center within the crop
        """
        if not crops:
            return []

        results = self.model(
            crops,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        detections: list[Optional[dict]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                detections.append(None)
                continue

            # Filter by target classes
            best = None
            best_conf = 0.0
            for j in range(len(boxes)):
                cls_id = int(boxes.cls[j].item())
                conf_val = float(boxes.conf[j].item())
                if not self._is_single_cls and cls_id not in self.target_classes:
                    continue
                if conf_val > best_conf:
                    best_conf = conf_val
                    # Box center in crop coordinates
                    x0, y0, x1, y1 = boxes.xyxy[j].tolist()
                    best = {
                        "yolo_conf": conf_val,
                        "crop_cx": (x0 + x1) / 2.0,
                        "crop_cy": (y0 + y1) / 2.0,
                    }

            detections.append(best)

        return detections


def verify_blobs(
    frame: np.ndarray,
    blobs: list[dict],
    verifier: BlobVerifier,
    threshold: float = 0.25,
) -> list[dict]:
    """Verify and re-rank blobs using YOLO detection.

    Short-circuits when only 0-1 blobs exist (no verification needed).

    Args:
        frame: Original BGR frame.
        blobs: Blob dicts from process_heatmap_multi().
        verifier: Initialized BlobVerifier instance.
        threshold: Minimum YOLO confidence to keep a blob.

    Returns:
        Filtered and re-ranked blob list. Each blob gains:
            - yolo_conf: YOLO detection confidence (0 if not detected)
            - refined_pixel_x/y: YOLO-refined ball center (or original if no detection)
    """
    if len(blobs) == 0:
        return blobs

    # Extract crops and run YOLO
    crops = extract_crops(frame, blobs, verifier.crop_size)
    detections = verifier.detect_crops(crops)

    half = verifier.crop_size // 2
    verified: list[dict] = []

    for blob, det in zip(blobs, detections):
        blob_copy = dict(blob)

        if det is not None and det["yolo_conf"] >= threshold:
            blob_copy["yolo_conf"] = det["yolo_conf"]
            # Convert crop-local detection back to original image coordinates
            cx_offset = det["crop_cx"] - half
            cy_offset = det["crop_cy"] - half
            blob_copy["refined_pixel_x"] = blob["pixel_x"] + cx_offset
            blob_copy["refined_pixel_y"] = blob["pixel_y"] + cy_offset
            verified.append(blob_copy)
        else:
            blob_copy["yolo_conf"] = 0.0
            blob_copy["refined_pixel_x"] = blob["pixel_x"]
            blob_copy["refined_pixel_y"] = blob["pixel_y"]
            # Don't add to verified — this blob failed verification

    # Re-sort by combined score: yolo_conf * blob_sum
    verified.sort(key=lambda b: b["yolo_conf"] * b["blob_sum"], reverse=True)

    # If YOLO filtered everything out, fall back to original top-1
    if not verified:
        logger.debug("YOLO filtered all blobs — falling back to TrackNet top-1")
        fallback = dict(blobs[0])
        fallback["yolo_conf"] = 0.0
        fallback["refined_pixel_x"] = blobs[0]["pixel_x"]
        fallback["refined_pixel_y"] = blobs[0]["pixel_y"]
        return [fallback]

    return verified
