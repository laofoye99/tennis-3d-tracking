"""Pipeline evaluation: compare model detections against hand-labeled ground truth.

Loads LabelImg-format JSON annotations, runs TrackNet inference with the same
pipeline logic (OSD mask → blob detection → court-X filter), then computes
frame-by-frame pixel error and aggregate metrics.

Usage (CLI):
    python -m app.pipeline.evaluate                          # all cameras in config
    python -m app.pipeline.evaluate --camera cam68           # single camera
    python -m app.pipeline.evaluate --threshold 0.3          # custom threshold
    python -m app.pipeline.evaluate --output eval_result.json  # save JSON report

Usage (API):
    from app.pipeline.evaluate import run_evaluation, EvalConfig
    cfg = EvalConfig(camera="cam68", video="uploads/cam68_clip.mp4", ...)
    report = run_evaluation(cfg)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from app.pipeline.homography import HomographyTransformer
from app.pipeline.postprocess import BallTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Single-frame evaluation result."""

    frame_index: int
    gt_pixel: tuple[float, float]
    gt_world: tuple[float, float]
    det_pixel: Optional[tuple[float, float]] = None
    det_world: Optional[tuple[float, float]] = None
    det_confidence: float = 0.0
    pixel_error: Optional[float] = None
    world_x_error: Optional[float] = None
    status: str = "miss"  # "correct", "wrong", "miss"

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "gt_pixel": list(self.gt_pixel),
            "gt_world": list(self.gt_world),
            "det_pixel": list(self.det_pixel) if self.det_pixel else None,
            "det_world": list(self.det_world) if self.det_world else None,
            "det_confidence": round(self.det_confidence, 4),
            "pixel_error": round(self.pixel_error, 2) if self.pixel_error is not None else None,
            "world_x_error": round(self.world_x_error, 4) if self.world_x_error is not None else None,
            "status": self.status,
        }


@dataclass
class EvalReport:
    """Aggregate evaluation metrics for one camera."""

    camera: str
    total_gt_frames: int = 0
    correct: int = 0
    wrong: int = 0
    missed: int = 0
    correct_threshold_px: float = 30.0
    pixel_errors: list[float] = field(default_factory=list)
    frame_results: list[FrameResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total_gt_frames if self.total_gt_frames else 0.0

    @property
    def wrong_rate(self) -> float:
        return self.wrong / self.total_gt_frames if self.total_gt_frames else 0.0

    @property
    def miss_rate(self) -> float:
        return self.missed / self.total_gt_frames if self.total_gt_frames else 0.0

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.pixel_errors)) if self.pixel_errors else 0.0

    @property
    def median_error(self) -> float:
        return float(np.median(self.pixel_errors)) if self.pixel_errors else 0.0

    @property
    def max_error(self) -> float:
        return float(np.max(self.pixel_errors)) if self.pixel_errors else 0.0

    def summary_dict(self) -> dict:
        return {
            "camera": self.camera,
            "total_gt_frames": self.total_gt_frames,
            "correct": self.correct,
            "wrong": self.wrong,
            "missed": self.missed,
            "correct_rate": round(self.correct_rate, 4),
            "wrong_rate": round(self.wrong_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "correct_threshold_px": self.correct_threshold_px,
            "pixel_error_mean": round(self.mean_error, 2),
            "pixel_error_median": round(self.median_error, 2),
            "pixel_error_max": round(self.max_error, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def full_dict(self) -> dict:
        return {
            **self.summary_dict(),
            "frames": [fr.to_dict() for fr in self.frame_results],
        }

    def print_table(self) -> None:
        """Print a human-readable frame-by-frame table to stdout."""
        header = (
            f"{'Frm':>4} | {'GT pixel':>18} | {'GT wX':>6} | "
            f"{'Det pixel':>18} | {'Det wX':>6} | {'Err':>6} | Status"
        )
        print(f"\n{'=' * 90}")
        print(f"  {self.camera}  evaluation results")
        print(f"{'=' * 90}")
        print(header)
        print("-" * 90)

        for fr in self.frame_results:
            gt_str = f"({fr.gt_pixel[0]:7.1f},{fr.gt_pixel[1]:7.1f})"
            if fr.det_pixel is not None:
                det_str = f"({fr.det_pixel[0]:7.1f},{fr.det_pixel[1]:7.1f})"
                det_wx = f"{fr.det_world[0]:5.1f}m"
                err_str = f"{fr.pixel_error:5.0f}px"
            else:
                det_str = "--- NO DETECTION ---"
                det_wx = "     "
                err_str = "     "
            print(
                f"{fr.frame_index:4d} | {gt_str} | {fr.gt_world[0]:5.1f}m | "
                f"{det_str} | {det_wx} | {err_str} | {fr.status}"
            )

        # Summary
        print(f"\n{'=' * 70}")
        print(f"SUMMARY  {self.camera}  ({self.total_gt_frames} GT frames)")
        print(f"{'=' * 70}")
        print(f"  CORRECT  (<{self.correct_threshold_px:.0f}px):  {self.correct:>3} / {self.total_gt_frames}  ({self.correct_rate:.1%})")
        print(f"  WRONG    (>{self.correct_threshold_px:.0f}px):  {self.wrong:>3} / {self.total_gt_frames}  ({self.wrong_rate:.1%})")
        print(f"  MISSED   (no det):  {self.missed:>3} / {self.total_gt_frames}  ({self.miss_rate:.1%})")
        if self.pixel_errors:
            print(f"  Pixel error: mean={self.mean_error:.1f}px, median={self.median_error:.1f}px, max={self.max_error:.1f}px")
        print(f"  Time: {self.elapsed_seconds:.1f}s")
        print()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation configuration for a single camera."""

    camera: str
    video_path: str
    gt_dir: str
    homography_path: str
    homography_key: str
    model_path: str
    input_size: tuple[int, int] = (288, 512)
    seq_len: int = 8
    threshold: float = 0.5
    heatmap_mask: list[tuple[int, int, int, int]] = field(default_factory=lambda: [(0, 0, 620, 40)])
    correct_threshold_px: float = 30.0
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_dir: str | Path) -> dict[int, tuple[float, float]]:
    """Load LabelImg JSON annotations from a directory.

    Each JSON file should be named ``{frame_index:05d}.json`` and contain
    ``shapes[0].points[0] = [pixel_x, pixel_y]``.

    Returns:
        dict mapping frame_index → (pixel_x, pixel_y).
    """
    gt_dir = Path(gt_dir)
    gt: dict[int, tuple[float, float]] = {}

    for jf in sorted(gt_dir.glob("*.json")):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            if shapes and shapes[0].get("points"):
                pts = shapes[0]["points"][0]
                gt[int(jf.stem)] = (float(pts[0]), float(pts[1]))
        except Exception:
            continue

    logger.info("Loaded %d GT annotations from %s", len(gt), gt_dir)
    return gt


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tracknet(
    model_path: str,
    seq_len: int = 8,
    device: str = "cuda",
) -> tuple[torch.nn.Module, torch.device]:
    """Load TrackNet model weights.

    Returns:
        (model, device) tuple ready for inference.
    """
    from app.pipeline.tracknet import TrackNet

    in_dim = (seq_len + 1) * 3  # bg_mode='concat'
    model = TrackNet(in_dim=in_dim, out_dim=seq_len)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    logger.info("TrackNet loaded: %s → %s", model_path, dev)
    return model, dev


def _preprocess_frame(
    frame: np.ndarray, input_w: int, input_h: int,
) -> np.ndarray:
    """Preprocess a single BGR frame for TrackNet.

    Returns float32 (3, H, W) in [0, 1] — no ImageNet normalization.
    """
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


def _compute_median(
    cap: cv2.VideoCapture,
    input_w: int,
    input_h: int,
    n_samples: int = 200,
) -> np.ndarray:
    """Compute background median frame.

    Returns float32 (3, H, W) in [0, 1].
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_samples, total)
    indices = set(int(i * total / n) for i in range(n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            small = cv2.resize(frame, (input_w, input_h))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(small_rgb)
    median = np.median(frames, axis=0).astype(np.uint8)
    return median.astype(np.float32).transpose(2, 0, 1) / 255.0


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg: EvalConfig,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
) -> EvalReport:
    """Run full pipeline evaluation for a single camera.

    Simulates the exact pipeline flow:
    OSD mask → TrackNet inference → BallTracker blob detection → court-X filter.

    Compares each detection against GT and computes metrics.

    Args:
        cfg: Evaluation configuration.
        model: Pre-loaded TrackNet model (optional, loaded if not provided).
        device: Torch device (optional, inferred from model or cfg).

    Returns:
        EvalReport with per-frame results and aggregate metrics.
    """
    t0 = time.time()

    # Load model if not provided
    if model is None or device is None:
        model, device = load_tracknet(cfg.model_path, cfg.seq_len, cfg.device)

    # Load GT
    gt = load_ground_truth(cfg.gt_dir)
    if not gt:
        logger.warning("No GT annotations found in %s", cfg.gt_dir)
        return EvalReport(camera=cfg.camera)

    # Open video
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_h, input_w = cfg.input_size

    logger.info(
        "[%s] Video: %dx%d, %d frames | GT: %d annotations (frames %d-%d)",
        cfg.camera, vid_w, vid_h, total_frames, len(gt), min(gt), max(gt),
    )

    # Compute background median
    bg = _compute_median(cap, input_w, input_h)

    # Read all frames into memory
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames: dict[int, np.ndarray] = {}
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames[i] = frame
    cap.release()

    # Pipeline components
    tracker = BallTracker(
        original_size=(vid_w, vid_h),
        threshold=cfg.threshold,
        heatmap_mask=[tuple(r) for r in cfg.heatmap_mask],
    )
    homography = HomographyTransformer(cfg.homography_path, cfg.homography_key)

    # Run inference in batches
    pipeline_detections: dict[int, Optional[dict]] = {}

    for batch_start in range(0, total_frames, cfg.seq_len):
        batch_indices = list(range(batch_start, batch_start + cfg.seq_len))
        if not all(bi in all_frames for bi in batch_indices):
            break

        # OSD mask + preprocess
        frames_masked = []
        for fi in batch_indices:
            masked = all_frames[fi].copy()
            masked[0:41, 0:603] = 0
            frames_masked.append(masked)

        processed = [_preprocess_frame(f, input_w, input_h) for f in frames_masked]
        stacked = np.concatenate([bg] + processed, axis=0)
        inp = torch.from_numpy(stacked[np.newaxis]).to(device)

        with torch.no_grad():
            heatmaps = model(inp)[0].cpu().numpy()

        for i in range(cfg.seq_len):
            fi = batch_start + i
            blobs = tracker.process_heatmap_multi(heatmaps[i])
            if not blobs:
                pipeline_detections[fi] = None
                continue

            # Court-X filter (same as video_pipeline)
            candidates = []
            for blob in blobs:
                wx, wy = homography.pixel_to_world(blob["pixel_x"], blob["pixel_y"])
                if not (homography.court_x_min <= wx <= homography.court_x_max):
                    continue
                candidates.append({
                    "pixel_x": blob["pixel_x"],
                    "pixel_y": blob["pixel_y"],
                    "world_x": wx,
                    "world_y": wy,
                    "blob_sum": blob["blob_sum"],
                })

            pipeline_detections[fi] = candidates[0] if candidates else None

    # Compare detections vs GT
    report = EvalReport(
        camera=cfg.camera,
        total_gt_frames=len(gt),
        correct_threshold_px=cfg.correct_threshold_px,
    )

    for fi in sorted(gt.keys()):
        gt_px, gt_py = gt[fi]
        gt_wx, gt_wy = homography.pixel_to_world(gt_px, gt_py)
        det = pipeline_detections.get(fi)

        fr = FrameResult(
            frame_index=fi,
            gt_pixel=(gt_px, gt_py),
            gt_world=(gt_wx, gt_wy),
        )

        if det is None:
            fr.status = "miss"
            report.missed += 1
        else:
            dpx, dpy = det["pixel_x"], det["pixel_y"]
            dwx, dwy = det["world_x"], det["world_y"]
            err = float(np.sqrt((dpx - gt_px) ** 2 + (dpy - gt_py) ** 2))

            fr.det_pixel = (dpx, dpy)
            fr.det_world = (dwx, dwy)
            fr.det_confidence = det["blob_sum"]
            fr.pixel_error = err
            fr.world_x_error = abs(dwx - gt_wx)
            report.pixel_errors.append(err)

            if err <= cfg.correct_threshold_px:
                fr.status = "correct"
                report.correct += 1
            else:
                fr.status = "wrong"
                report.wrong += 1

        report.frame_results.append(fr)

    report.elapsed_seconds = time.time() - t0
    return report


# ---------------------------------------------------------------------------
# Multi-camera evaluation
# ---------------------------------------------------------------------------

def run_all_evaluations(
    configs: list[EvalConfig],
    shared_model: bool = True,
) -> list[EvalReport]:
    """Evaluate multiple cameras, optionally sharing one model instance.

    Args:
        configs: List of EvalConfig, one per camera.
        shared_model: If True, load model once and reuse across cameras.

    Returns:
        List of EvalReport, one per camera.
    """
    model, device = None, None
    if shared_model and configs:
        model, device = load_tracknet(
            configs[0].model_path, configs[0].seq_len, configs[0].device,
        )

    reports = []
    for cfg in configs:
        logger.info("Evaluating %s ...", cfg.camera)
        report = run_evaluation(cfg, model=model, device=device)
        reports.append(report)
    return reports


def print_summary(reports: list[EvalReport]) -> None:
    """Print a combined summary table for all cameras."""
    for rpt in reports:
        rpt.print_table()

    if len(reports) > 1:
        print(f"\n{'=' * 70}")
        print("COMBINED SUMMARY")
        print(f"{'=' * 70}")
        total_gt = sum(r.total_gt_frames for r in reports)
        total_correct = sum(r.correct for r in reports)
        total_wrong = sum(r.wrong for r in reports)
        total_missed = sum(r.missed for r in reports)
        all_errors = []
        for r in reports:
            all_errors.extend(r.pixel_errors)

        print(f"  Cameras: {', '.join(r.camera for r in reports)}")
        print(f"  Total GT frames: {total_gt}")
        print(f"  CORRECT: {total_correct}/{total_gt} ({total_correct / total_gt:.1%})")
        print(f"  WRONG:   {total_wrong}/{total_gt} ({total_wrong / total_gt:.1%})")
        print(f"  MISSED:  {total_missed}/{total_gt} ({total_missed / total_gt:.1%})")
        if all_errors:
            ea = np.array(all_errors)
            print(f"  Pixel error: mean={ea.mean():.1f}px, median={np.median(ea):.1f}px, max={ea.max():.1f}px")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_configs_from_yaml(
    config_path: str = "config.yaml",
    camera: Optional[str] = None,
    threshold: Optional[float] = None,
    correct_threshold_px: float = 30.0,
) -> list[EvalConfig]:
    """Build EvalConfig list from the project config.yaml.

    Auto-discovers GT directories at ``uploads/{camera}_clip/``.
    """
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    homo_cfg = cfg["homography"]

    # Default GT/video path convention
    gt_video_map = {
        "cam68": ("uploads/cam68_clip", "uploads/cam68_clip.mp4", "cam68"),
        "cam66": ("uploads/cam66_clip", "uploads/cam66_clip.mp4", "cam66"),
    }

    cameras = cfg.get("cameras", {})
    configs = []

    for cam_name, cam_data in cameras.items():
        if camera and cam_name != camera:
            continue
        if cam_name not in gt_video_map:
            logger.warning("No GT data mapping for camera %s, skipping", cam_name)
            continue

        gt_dir, video_path, homo_key = gt_video_map[cam_name]

        # Only include cameras that have GT annotations
        gt_path = Path(gt_dir)
        if not gt_path.exists() or not list(gt_path.glob("*.json")):
            logger.warning("No GT annotations for %s at %s, skipping", cam_name, gt_dir)
            continue

        heatmap_mask = model_cfg.get("heatmap_mask", [[0, 0, 620, 40]])

        configs.append(EvalConfig(
            camera=cam_name,
            video_path=video_path,
            gt_dir=gt_dir,
            homography_path=homo_cfg["path"],
            homography_key=cam_data.get("homography_key", homo_key),
            model_path=model_cfg["path"],
            input_size=tuple(model_cfg["input_size"]),
            seq_len=model_cfg.get("frames_in", 8),
            threshold=threshold if threshold is not None else model_cfg.get("threshold", 0.5),
            heatmap_mask=[tuple(m) for m in heatmap_mask],
            correct_threshold_px=correct_threshold_px,
            device=model_cfg.get("device", "cuda"),
        ))

    return configs


def main() -> None:
    """CLI entry point for pipeline evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate TrackNet pipeline against hand-labeled ground truth.",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to project config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--camera", default=None,
        help="Evaluate a single camera (e.g. cam68). Default: all cameras with GT.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Heatmap blob threshold (overrides config.yaml).",
    )
    parser.add_argument(
        "--correct-threshold", type=float, default=30.0,
        help="Pixel error threshold for 'correct' classification (default: 30px).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save JSON evaluation report to this path.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress frame-by-frame table output.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [eval] %(levelname)s %(message)s",
    )

    configs = _build_configs_from_yaml(
        config_path=args.config,
        camera=args.camera,
        threshold=args.threshold,
        correct_threshold_px=args.correct_threshold,
    )

    if not configs:
        print("No cameras with GT data found. Check config.yaml and uploads/ directory.")
        return

    print(f"Evaluating {len(configs)} camera(s): {', '.join(c.camera for c in configs)}")

    reports = run_all_evaluations(configs)

    if not args.quiet:
        print_summary(reports)

    if args.output:
        out = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cameras": [r.full_dict() for r in reports],
            "combined": {
                "total_gt_frames": sum(r.total_gt_frames for r in reports),
                "correct": sum(r.correct for r in reports),
                "wrong": sum(r.wrong for r in reports),
                "missed": sum(r.missed for r in reports),
            },
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
