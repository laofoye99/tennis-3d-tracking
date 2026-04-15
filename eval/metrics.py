"""Structured evaluation metrics for bounce detection."""

import math
from dataclasses import dataclass, field


@dataclass
class BounceMetrics:
    """Evaluation metrics for a single detection method."""

    method: str
    gt_count: int = 0
    detected_count: int = 0
    matched: int = 0
    missed: int = 0
    false_positives: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    landing_errors: list[float] = field(default_factory=list)
    landing_error_mean: float = 0.0
    landing_error_median: float = 0.0
    landing_error_p95: float = 0.0
    matched_details: list[dict] = field(default_factory=list)
    missed_frames: list[int] = field(default_factory=list)
    fp_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "method": self.method,
            "gt_count": self.gt_count,
            "detected_count": self.detected_count,
            "matched": self.matched,
            "missed": self.missed,
            "false_positives": self.false_positives,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "landing_error_mean": round(self.landing_error_mean, 4),
            "landing_error_median": round(self.landing_error_median, 4),
            "landing_error_p95": round(self.landing_error_p95, 4),
            "landing_errors": [round(e, 4) for e in self.landing_errors],
            "matched_details": self.matched_details,
            "missed_frames": self.missed_frames,
            "fp_frames": self.fp_frames,
        }


def compute_metrics(
    method_name: str,
    gt_bounces: list[dict],
    det_bounces: list[dict],
    frame_tolerance: int = 5,
    position_tolerance: float = 0.5,
) -> BounceMetrics:
    """Match detected bounces to GT and compute metrics.

    GT and detected bounces are dicts with keys: frame, x, y, z.
    Matching is greedy by frame proximity (within frame_tolerance).

    Args:
        method_name: label for this method combination
        gt_bounces: ground truth bounce list
        det_bounces: detected bounce list
        frame_tolerance: max frame difference for a match
        position_tolerance: not used for matching, but reported

    Returns:
        BounceMetrics with all fields populated
    """
    metrics = BounceMetrics(method=method_name)
    metrics.gt_count = len(gt_bounces)
    metrics.detected_count = len(det_bounces)

    if not gt_bounces:
        metrics.false_positives = len(det_bounces)
        metrics.fp_frames = [d["frame"] for d in det_bounces]
        return metrics

    if not det_bounces:
        metrics.missed = len(gt_bounces)
        metrics.missed_frames = [g["frame"] for g in gt_bounces]
        return metrics

    # Greedy matching: for each GT bounce, find closest unmatched detection
    gt_sorted = sorted(gt_bounces, key=lambda b: b["frame"])
    det_sorted = sorted(det_bounces, key=lambda b: b["frame"])
    det_used = set()
    matched_details = []
    matched_gt = set()

    for gi, gt in enumerate(gt_sorted):
        best_di = -1
        best_frame_diff = frame_tolerance + 1

        for di, det in enumerate(det_sorted):
            if di in det_used:
                continue
            frame_diff = abs(gt["frame"] - det["frame"])
            if frame_diff <= frame_tolerance and frame_diff < best_frame_diff:
                best_frame_diff = frame_diff
                best_di = di

        if best_di >= 0:
            det = det_sorted[best_di]
            det_used.add(best_di)
            matched_gt.add(gi)

            # Position error (Euclidean in x-y plane)
            dx = gt["x"] - det["x"]
            dy = gt["y"] - det["y"]
            error = math.sqrt(dx * dx + dy * dy)

            matched_details.append({
                "gt_frame": gt["frame"],
                "det_frame": det["frame"],
                "frame_diff": best_frame_diff,
                "gt_pos": [round(gt["x"], 4), round(gt["y"], 4)],
                "det_pos": [round(det["x"], 4), round(det["y"], 4)],
                "error_m": round(error, 4),
            })

    # Missed GT bounces
    missed_frames = []
    for gi, gt in enumerate(gt_sorted):
        if gi not in matched_gt:
            missed_frames.append(gt["frame"])

    # False positive detections
    fp_frames = []
    for di, det in enumerate(det_sorted):
        if di not in det_used:
            fp_frames.append(det["frame"])

    # Populate metrics
    metrics.matched = len(matched_details)
    metrics.missed = len(missed_frames)
    metrics.false_positives = len(fp_frames)
    metrics.matched_details = matched_details
    metrics.missed_frames = missed_frames
    metrics.fp_frames = fp_frames

    landing_errors = [d["error_m"] for d in matched_details]
    metrics.landing_errors = landing_errors

    if landing_errors:
        sorted_errors = sorted(landing_errors)
        metrics.landing_error_mean = sum(sorted_errors) / len(sorted_errors)
        n = len(sorted_errors)
        metrics.landing_error_median = (
            sorted_errors[n // 2]
            if n % 2 == 1
            else (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2.0
        )
        p95_idx = min(int(n * 0.95), n - 1)
        metrics.landing_error_p95 = sorted_errors[p95_idx]

    # Recall / Precision / F1
    if metrics.gt_count > 0:
        metrics.recall = metrics.matched / metrics.gt_count
    if metrics.detected_count > 0:
        metrics.precision = metrics.matched / metrics.detected_count
    if metrics.recall + metrics.precision > 0:
        metrics.f1 = 2 * metrics.recall * metrics.precision / (metrics.recall + metrics.precision)

    return metrics
