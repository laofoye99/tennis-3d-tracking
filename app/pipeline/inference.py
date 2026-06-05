"""Ball detection inference with GPU/CPU fallback.

Supports three detector backends:
    - BallDetector:      ONNX-based HRNet (frames_in=3, frames_out=3)
    - TrackNetDetector:  ONNX/PyTorch TrackNet (seq_len=8, bg_mode='concat')
    - MedianBGDetector:  Median background subtraction (frames_in=30, no GPU)
    - YoloRoadmapDetector: Ultralytics YOLO roadmap detector

Use ``create_detector()`` factory to select backend.  Default auto-selects by
model file extension; pass ``detector_type="median_bg"`` to use MedianBGDetector.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# ONNX HRNet detector (original)
# ---------------------------------------------------------------------------

class BallDetector:
    """ONNX-based tennis ball detector using HRNet heatmap model."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (288, 512),
        frames_in: int = 3,
        frames_out: int = 3,
        device: str = "cuda",
    ):
        import onnxruntime as ort

        self.input_h, self.input_w = input_size
        self.frames_in = frames_in
        self.frames_out = frames_out

        providers = self._get_providers(device)
        logger.info("Loading ONNX model: %s (providers=%s)", model_path, providers)
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self._use_cuda = "CUDAExecutionProvider" in self.session.get_providers()
        logger.info("BallDetector ready (CUDA=%s)", self._use_cuda)

    @staticmethod
    def _get_providers(device: str) -> list[str]:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" not in available and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")
        return ["CPUExecutionProvider"]

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, HWC→CHW. Returns float32 array of shape (3, H, W)."""
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return img.transpose(2, 0, 1)  # CHW

    def infer(self, frames: list[np.ndarray]) -> np.ndarray:
        """Run inference on a list of frames.

        Args:
            frames: list of BGR frames (length == frames_in).

        Returns:
            Raw output array of shape (frames_out, H, W) after sigmoid.
        """
        processed = [self.preprocess_frame(f) for f in frames]
        # Stack channels: (frames_in * 3, H, W)
        stacked = np.concatenate(processed, axis=0)
        # Add batch dim: (1, frames_in*3, H, W)
        input_tensor = stacked[np.newaxis].astype(np.float32)

        # session.run() uses CUDA EP internally when available — no need for
        # io_binding (which has known output corruption issues with some models).
        output = self.session.run(
            [self.output_name], {self.input_name: input_tensor}
        )[0]

        # output shape: (1, frames_out, H, W)
        output = torch.sigmoid(torch.from_numpy(output[0])).numpy()
        return output  # (frames_out, H, W)


# ---------------------------------------------------------------------------
# PyTorch TrackNet detector
# ---------------------------------------------------------------------------

class TrackNetDetector:
    """PyTorch-based TrackNet ball detector (seq_len=8, bg_mode='concat').

    Uses the original author's TrackNet architecture and preprocessing:
        - Native PyTorch inference (not ONNX)
        - Normalization: simple /255.0 (NO ImageNet mean/std)
        - Background median frame prepended BEFORE sequence frames
        - Sigmoid is included in the model's forward pass
        - Outputs ``seq_len`` heatmaps per inference call
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (288, 512),
        frames_in: int = 8,
        frames_out: int = 8,
        device: str = "cuda",
        bg_mode: str = "concat",
    ):
        self.input_h, self.input_w = input_size
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.bg_mode = bg_mode

        # Select device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            if device == "cuda":
                logger.warning("CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")

        # Load model — prefer ONNX if available (much faster on RTX 50xx)
        logger.info("Loading TrackNet model: %s (device=%s)", model_path, self.device)
        if bg_mode == "concat":
            in_dim = (frames_in + 1) * 3  # 27 for seq_len=8
        else:
            in_dim = frames_in * 3

        import os

        is_onnx_model = model_path.lower().endswith(".onnx")
        is_pt_model = model_path.lower().endswith(".pt")
        onnx_path = model_path if is_onnx_model else model_path.replace(".pt", ".onnx")
        self._use_onnx = False
        self.model = None

        if is_onnx_model or os.path.exists(onnx_path):
            try:
                import onnxruntime as ort

                available = ort.get_available_providers()
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if device == "cuda" and "CUDAExecutionProvider" in available
                    else ["CPUExecutionProvider"]
                )
                self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                self._use_onnx = True
                actual = self._ort_session.get_providers()
                logger.info("TrackNet using ONNX Runtime (%s): %s", actual[0], onnx_path)
            except ImportError as e:
                message = (
                    "onnxruntime is required for ONNX TrackNet model "
                    f"({onnx_path}). Install onnxruntime-gpu or onnxruntime."
                )
                if is_onnx_model:
                    raise RuntimeError(message) from e
                logger.warning("%s Falling back to PyTorch checkpoint: %s", message, model_path)
            except Exception as e:
                if is_onnx_model:
                    raise RuntimeError(
                        f"Failed to load ONNX TrackNet model with ONNX Runtime ({onnx_path}): {e}"
                    ) from e
                logger.warning("ONNX Runtime failed, falling back to PyTorch: %s", e)

        if not self._use_onnx:
            if not is_pt_model:
                raise RuntimeError(f"TrackNet model path must be .onnx or .pt, got: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"TrackNet PyTorch checkpoint not found: {model_path}")
            from app.pipeline.tracknet import TrackNet

            # PyTorch fallback
            self.model = TrackNet(in_dim=in_dim, out_dim=frames_in)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.model.to(self.device)

        # Background (median) frame — (3, H, W) float32 in [0, 1]
        self._bg_frame: Optional[np.ndarray] = None
        self._video_median_computed = False
        self._static_median_loaded = False

        # Try loading pre-computed median from src/bg_median_{camera}.png
        self._try_load_static_median()

        # Running median: static background is the warm start; live updates are
        # intentionally conservative because first-line court conditions matter
        # more than offline GT sweeps for this parameter.
        self._bg_buffer: list[np.ndarray] = []
        self._bg_max_frames: int = 200
        self._bg_update_interval: int = 60  # seconds
        self._bg_last_update: float = 0
        self._bg_thread = None

        if self.model is not None:
            n_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "TrackNetDetector ready (PyTorch, CUDA=%s, params=%s, seq_len=%d, bg=%s)",
                self.device.type == "cuda", f"{n_params:,}", frames_in, bg_mode,
            )
        else:
            logger.info(
                "TrackNetDetector ready (ONNX Runtime, seq_len=%d, bg=%s)",
                frames_in, bg_mode,
            )

    def _try_load_static_median(self):
        """Load pre-computed median background from src/bg_median_*.png if available.

        Tries all matching files. In multi-camera setups, call
        ``load_static_median(camera_name)`` after construction to load
        the correct per-camera median.
        """
        import glob
        for path in sorted(glob.glob("src/bg_median_*.png")):
            self._load_median_file(path)
            return  # load first match as default

    def load_static_median(self, camera_name: str) -> bool:
        """Load median for a specific camera: src/bg_median_{camera_name}.png"""
        path = f"src/bg_median_{camera_name}.png"
        return self._load_median_file(path)

    def _load_median_file(self, path: str) -> bool:
        import os
        if not os.path.exists(path):
            return False
        try:
            img = cv2.imread(path)
            if img is None:
                return False
            img = cv2.resize(img, (self.input_w, self.input_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._bg_frame = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            # Static medians are a warm start for live cameras; keep the
            # running median enabled so lighting changes can adapt.
            self._static_median_loaded = True
            logger.info("Loaded static median background: %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load median from %s: %s", path, e)
            return False

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize, BGR→RGB, HWC→CHW, /255.  Returns float32 (3, H, W) in [0, 1].

        Matches the author's preprocessing: no ImageNet mean/std normalization.
        """
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)  # CHW

    def compute_video_median(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        max_samples: int = 200,
    ) -> None:
        """Compute background median from video frames (author's approach).

        Samples up to ``max_samples`` frames evenly across the range.  Frames
        are read sequentially (no seeking per frame) and resized to model
        resolution before computing the median for speed.

        Result is stored as (3, H, W) float32 in [0, 1].
        """
        video_seg_len = end_frame - start_frame
        if video_seg_len <= 0:
            logger.warning("Invalid frame range for median: %d-%d", start_frame, end_frame)
            return

        # Determine which frames to sample (evenly spaced, capped at max_samples)
        n_samples = min(max_samples, video_seg_len)
        sample_indices = set(
            int(start_frame + i * video_seg_len / n_samples)
            for i in range(n_samples)
        )

        # Read sequentially (much faster than seeking per frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_list = []
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if i in sample_indices:
                small = cv2.resize(frame, (self.input_w, self.input_h))
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                frame_list.append(small_rgb)
                if len(frame_list) >= n_samples:
                    break

        if not frame_list:
            logger.warning("No frames sampled for median computation")
            return

        # Pixel-wise median at model resolution → CHW float32 [0, 1]
        median = np.median(frame_list, axis=0).astype(np.uint8)
        self._bg_frame = median.astype(np.float32).transpose(2, 0, 1) / 255.0
        self._video_median_computed = True

        logger.info("Video median computed from %d sampled frames", len(frame_list))

    def _update_running_median(self, preprocessed: np.ndarray) -> None:
        """Accumulate frames for background median. Recompute in background thread every N seconds."""
        import time as _time
        import threading

        self._bg_buffer.append(preprocessed)
        if len(self._bg_buffer) > self._bg_max_frames:
            self._bg_buffer.pop(0)

        now = _time.time()

        # First call: compute immediately (need BG before first inference)
        if self._bg_frame is None and len(self._bg_buffer) >= 2:
            self._bg_frame = np.median(self._bg_buffer, axis=0).astype(np.float32)
            self._bg_last_update = now
            return

        # Periodic update in background thread (every _bg_update_interval seconds)
        if now - self._bg_last_update >= self._bg_update_interval:
            if self._bg_thread is None or not self._bg_thread.is_alive():
                buf_copy = list(self._bg_buffer)  # snapshot
                def _compute():
                    if len(buf_copy) < 10:
                        return
                    new_bg = np.median(buf_copy, axis=0).astype(np.float32)
                    self._bg_frame = new_bg  # atomic replace
                    logger.info("Background median updated from %d frames", len(buf_copy))
                self._bg_thread = threading.Thread(target=_compute, daemon=True)
                self._bg_thread.start()
                self._bg_last_update = now

    def infer(self, frames: list[np.ndarray]) -> np.ndarray:
        """Run inference on a list of BGR frames.

        Args:
            frames: list of BGR frames (length == frames_in / seq_len).

        Returns:
            Heatmap array of shape (frames_out, H, W) in [0, 1].
            Sigmoid is already applied inside the model.
        """
        processed = [self.preprocess_frame(f) for f in frames]

        # Update running median for live camera (skip if video median was computed)
        if not self._video_median_computed:
            for p in processed:
                self._update_running_median(p)

        # Build input: median FIRST, then seq_len frames (author's channel order)
        if self.bg_mode == "concat":
            bg = self._bg_frame if self._bg_frame is not None else processed[0]
            all_channels = [bg] + processed  # median prepended before frames
        else:
            all_channels = processed

        # Stack: ((seq_len+1)*3, H, W) for concat, (seq_len*3, H, W) otherwise
        stacked = np.concatenate(all_channels, axis=0)

        if self._use_onnx:
            input_np = stacked[np.newaxis].astype(np.float32)
            outputs = self._ort_session.run(None, {self._ort_input_name: input_np})
            return outputs[0][0]  # (seq_len, H, W)
        else:
            input_tensor = torch.from_numpy(stacked[np.newaxis]).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                output = output[0].cpu().numpy()
            return output


# ---------------------------------------------------------------------------
# Median background subtraction detector (no GPU required)
# ---------------------------------------------------------------------------

class MedianBGDetector:
    """Median background subtraction ball detector (30 frames per block).

    Returns ALL raw pixel (cx, cy) blobs per frame — no filtering, no limit.
    Downstream tracker (track_single_camera) handles blob linking and filtering.

    Recall ~94% with thresh=10, ~67-89 candidates per frame.
    """

    # Flag: camera_pipeline sends raw blob_block instead of per-frame detections.
    returns_blobs = True

    def __init__(
        self,
        input_size: tuple[int, int] = (288, 512),
        frames_in: int = 30,
        frames_out: int = 30,
        device: str = "cuda",
        thresh: int = 10,
        min_area: int = 2,
        max_area: int = 600,
        **_kwargs,
    ):
        from app.pipeline.blob_detector import BallBlobDetector

        self.input_h, self.input_w = input_size
        self.frames_in = frames_in
        self.frames_out = frames_in
        self._detector = BallBlobDetector(
            thresh=thresh, min_area=min_area, max_area=max_area,
        )
        logger.info(
            "MedianBGDetector ready (thresh=%d, area=%d-%d, block=%d)",
            thresh, min_area, max_area, frames_in,
        )

    def infer(self, frames: list[np.ndarray]) -> dict[int, list[tuple]]:
        """Run median-BG blob detection on a block of BGR frames.

        Returns:
            Dict mapping frame index (0-based in block) to list of (cx, cy).
            ALL blobs returned — no limit, no ranking.
        """
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        return self._detector.detect_block(gray_frames)


# ---------------------------------------------------------------------------
# BallSelector detector (TrackNet + CandidateTransformer)
# ---------------------------------------------------------------------------

class BallSelectorDetector:
    """TrackNet + CandidateTransformer ball detector.

    Uses TrackNet for heatmap generation, then a lightweight Transformer
    (85K params) to select the correct ball from top-K candidates.
    Reduces false positives from 9% to 3-6% at the cost of lower recall.

    Same interface as TrackNetDetector: infer(frames) → heatmaps-like output.
    But returns_blobs=True since it outputs pixel coordinates directly.
    """

    returns_blobs = True  # camera_pipeline skips BallTracker postprocessing

    def __init__(
        self,
        model_path: str = "",
        input_size: tuple[int, int] = (288, 512),
        frames_in: int = 8,
        frames_out: int = 8,
        device: str = "cuda",
        selector_weights: str = "model_weight/ball_selector_v2.2.pt",
        **_kwargs,
    ):
        from app.pipeline.ball_selector.inference_api import BallSelector
        from app.pipeline.ball_selector.utils import compute_median_bg

        self.input_h, self.input_w = input_size
        self.frames_in = 8  # BallSelector always uses 8 frames
        self.frames_out = 8
        self._compute_median = compute_median_bg

        tracknet_path = model_path or "model_weight/TrackNet_finetuned.pt"
        self._selector = BallSelector(
            tracknet_weights=tracknet_path,
            selector_weights=selector_weights,
            device=device,
            exist_threshold=0.5,
        )
        self._median_bg = None
        self._median_computed = False

        logger.info("BallSelectorDetector ready (TrackNet + CandidateTransformer)")

    def load_static_median(self, camera_name: str) -> bool:
        """Load pre-computed median background for a camera."""
        import os
        path = f"src/bg_median_{camera_name}.png"
        if not os.path.exists(path):
            return False
        try:
            img = cv2.imread(path)
            if img is None:
                return False
            img = cv2.resize(img, (self.input_w, self.input_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._median_bg = img.astype(np.float32).transpose(2, 0, 1) / 255.0
            self._median_computed = True
            logger.info("BallSelector: loaded static median from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load median: %s", e)
            return False

    def infer(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Run BallSelector on a batch of BGR frames.

        Returns:
            List of length len(frames). Each element is a list of blob dicts:
            [{"pixel_x": float, "pixel_y": float, "blob_sum": float}] or []
        """
        # Compute median from frames if not pre-loaded
        if self._median_bg is None:
            from app.pipeline.ball_selector.utils import preprocess_frame
            processed = [preprocess_frame(f) for f in frames]
            self._median_bg = np.median(processed, axis=0).astype(np.float32)

        # Pad to 8 frames if needed
        batch = list(frames)
        while len(batch) < 8:
            batch.append(batch[-1])

        results = self._selector.detect(batch[:8], self._median_bg)

        output: list[list[dict]] = []
        for i in range(len(frames)):
            r = results[i] if i < len(results) else {"px": None, "py": None, "conf": 0}
            if r["px"] is not None:
                output.append([{
                    "pixel_x": float(r["px"]),
                    "pixel_y": float(r["py"]),
                    "blob_sum": float(r["conf"]),
                    "blob_max": float(r["conf"]),
                    "blob_area": 0,
                }])
            else:
                output.append([])

        return output


# ---------------------------------------------------------------------------
# YOLO Roadmap detector
# ---------------------------------------------------------------------------

@dataclass
class _YoloTrackState:
    key: int | str
    last_seen: int = 0
    last_x: float | None = None
    last_y: float | None = None
    last_w: float = 0.0
    last_h: float = 0.0
    static_count: int = 0
    speed_streak: int = 0
    recent_step: float = 0.0
    recent_displacement: float = 0.0
    positions: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))


@dataclass
class _StaticZone:
    zone_id: int
    x: float
    y: float
    radius: float
    created_frame: int
    last_seen_frame: int
    source_track: int | str
    hits: int = 0
    blocked: int = 0
    released: int = 0

    def distance_to(self, x: float, y: float) -> float:
        return float(np.hypot(x - self.x, y - self.y))

    def contains(self, x: float, y: float) -> bool:
        return self.distance_to(x, y) <= self.radius

    def to_dict(self, current_frame: int) -> dict[str, Any]:
        return {
            "id": self.zone_id,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "radius": round(self.radius, 1),
            "age_frames": max(0, current_frame - self.created_frame),
            "last_seen_age": max(0, current_frame - self.last_seen_frame),
            "source_track": self.source_track,
            "hits": self.hits,
            "blocked": self.blocked,
            "released": self.released,
        }


class YoloRoadmapDetector:
    """Ultralytics YOLO detector from ``yolo_roadmap/best.pt``.

    This backend mirrors the prototype in ``yolo_roadmap``: run YOLO directly
    on each frame, keep active tracked boxes, and suppress objects that stay
    nearly static for several consecutive frames.
    """

    returns_blobs = True
    already_verified = True

    def __init__(
        self,
        model_path: str = "yolo_roadmap/best.pt",
        input_size: tuple[int, int] = (288, 512),
        frames_in: int = 1,
        frames_out: int = 1,
        device: str = "cuda",
        conf: float = 0.25,
        imgsz: int = 960,
        move_threshold: float = 5.0,
        static_frame_limit: int = 7,
        static_zone_radius: float = 28.0,
        static_zone_ttl_frames: int = 150,
        static_zone_max: int = 8,
        static_release_speed: float = 8.0,
        static_release_displacement: float = 24.0,
        static_release_frames: int = 2,
        static_starvation_frames: int = 90,
        use_yolo_track: bool = False,
        **_kwargs,
    ):
        from ultralytics import YOLO

        self.input_h, self.input_w = input_size
        self.frames_in = max(1, int(frames_in or 1))
        self.frames_out = self.frames_in
        self.conf = conf
        self.imgsz = imgsz
        self.move_threshold = move_threshold
        self.static_frame_limit = static_frame_limit
        self.static_zone_radius = static_zone_radius
        self.static_zone_ttl_frames = static_zone_ttl_frames
        self.static_zone_max = static_zone_max
        self.static_release_speed = static_release_speed
        self.static_release_displacement = static_release_displacement
        self.static_release_frames = static_release_frames
        self.static_starvation_frames = static_starvation_frames
        self.use_yolo_track = use_yolo_track
        self.device = self._resolve_device(device)

        self.model = YOLO(model_path)
        self._target_classes = None

        self._track_available = True
        self._frame_counter = 0
        self._track_history: dict[int | str, _YoloTrackState] = {}
        self._last_seen: dict[int | str, int] = {}
        self._static_zones: dict[int, _StaticZone] = {}
        self._next_static_zone_id = 1
        self._next_pseudo_track_id = 1
        self._static_starvation_count = 0
        self._static_fail_open_until = 0
        self._static_stats: dict[str, int] = {
            "raw_detections": 0,
            "kept_detections": 0,
            "static_blocked": 0,
            "static_zones_created": 0,
            "static_zones_expired": 0,
            "motion_released": 0,
            "fail_open_kept": 0,
            "untracked_kept": 0,
            "pseudo_tracked": 0,
        }

        logger.info(
            "YoloRoadmapDetector ready (model=%s, conf=%.2f, imgsz=%d, device=%s, static=%d frames, ttl=%d)",
            model_path,
            conf,
            imgsz,
            self.device,
            static_frame_limit,
            static_zone_ttl_frames,
        )

    @staticmethod
    def _resolve_device(device: str):
        if isinstance(device, str) and device.startswith("cuda"):
            if torch.cuda.is_available():
                return 0 if device == "cuda" else device
            logger.warning("CUDA not available, falling back to CPU for YOLO")
            return "cpu"
        return device

    def infer(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """Run YOLO on each frame and return dashboard-compatible blob lists."""
        outputs: list[list[dict]] = []
        for frame in frames:
            self._frame_counter += 1
            if self.use_yolo_track and self._track_available:
                try:
                    results = self.model.track(
                        frame,
                        persist=True,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
                except Exception as exc:
                    self._track_available = False
                    logger.warning(
                        "YOLO tracking unavailable, falling back to predict(): %s",
                        exc,
                    )
                    results = self.model.predict(
                        frame,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False,
                    )
            else:
                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

            result = results[0] if results else None
            outputs.append(self._result_to_blobs(result))
            self._prune_track_history()

        return outputs

    def _result_to_blobs(self, result) -> list[dict]:
        if result is None or result.boxes is None or len(result.boxes) == 0:
            self._expire_static_zones()
            self._update_static_starvation(raw_count=0, kept_count=0)
            return []

        boxes = result.boxes
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xywh))
        track_ids = (
            boxes.id.int().cpu().numpy().tolist()
            if boxes.id is not None
            else [None] * len(xywh)
        )

        blobs: list[dict] = []
        raw_blobs: list[dict] = []
        raw_count = 0
        claimed_pseudo_tracks: set[int | str] = set()
        for box, conf, cls_id, track_id in zip(xywh, confs, classes, track_ids):
            if self._target_classes is not None and int(cls_id) not in self._target_classes:
                continue

            x, y, w, h = [float(v) for v in box]
            raw_count += 1
            has_real_track = track_id is not None
            track_key = (
                int(track_id)
                if has_real_track
                else self._assign_pseudo_track(x, y, claimed_pseudo_tracks)
            )
            if not has_real_track:
                self._static_stats["pseudo_tracked"] += 1
            state = self._update_track_state(track_key, x, y, w, h)
            keep, static_status, zone = self._apply_static_gate(state, x, y)

            blob = {
                "pixel_x": x,
                "pixel_y": y,
                "blob_sum": float(conf),
                "blob_max": float(conf),
                "blob_area": int(max(1.0, w * h)),
                "yolo_conf": float(conf),
                "bbox": [
                    x - w / 2.0,
                    y - h / 2.0,
                    x + w / 2.0,
                    y + h / 2.0,
                ],
                "track_id": int(track_id) if has_real_track else None,
                "pseudo_track_id": track_key if not has_real_track else None,
                "static_count": state.static_count if state is not None else 0,
                "static_status": static_status,
                "static_zone_id": zone.zone_id if zone is not None else None,
                "source": "yolo_roadmap",
            }
            if keep:
                blobs.append(blob)
            raw_blobs.append({
                **blob,
                "static_blocked": not keep,
            })

        blobs.sort(key=lambda b: b["yolo_conf"], reverse=True)
        raw_blobs.sort(key=lambda b: b["yolo_conf"], reverse=True)
        if blobs:
            blobs[0]["raw_candidates"] = raw_blobs
        elif raw_blobs:
            blobs.append({
                **raw_blobs[0],
                "raw_candidates": raw_blobs,
                "event_only_raw_candidates": True,
                "static_status": "event_only_raw",
            })
        self._expire_static_zones()
        self._update_static_starvation(raw_count=raw_count, kept_count=len(blobs))
        self._static_stats["raw_detections"] += raw_count
        self._static_stats["kept_detections"] += len(blobs)
        return blobs

    def _assign_pseudo_track(
        self,
        x: float,
        y: float,
        claimed: set[int | str],
    ) -> str:
        best_key = None
        best_dist = 45.0
        for key, state in self._track_history.items():
            if not isinstance(key, str) or not key.startswith("u"):
                continue
            if key in claimed or self._frame_counter - state.last_seen > 5:
                continue
            if state.last_x is None or state.last_y is None:
                continue
            dist = float(np.hypot(x - state.last_x, y - state.last_y))
            if dist < best_dist:
                best_dist = dist
                best_key = key

        if best_key is None:
            best_key = f"u{self._next_pseudo_track_id}"
            self._next_pseudo_track_id += 1
        claimed.add(best_key)
        return best_key

    def _update_track_state(
        self,
        track_key: int | str,
        x: float,
        y: float,
        w: float,
        h: float,
    ) -> _YoloTrackState:
        state = self._track_history.get(track_key)
        if state is None:
            state = _YoloTrackState(key=track_key)
            self._track_history[track_key] = state

        if state.last_x is None or state.last_y is None:
            step = 0.0
        else:
            step = float(np.hypot(x - state.last_x, y - state.last_y))

        state.recent_step = step
        state.static_count = state.static_count + 1 if step < self.move_threshold else 0
        state.speed_streak = state.speed_streak + 1 if step >= self.static_release_speed else 0
        state.last_seen = self._frame_counter
        state.last_x = x
        state.last_y = y
        state.last_w = w
        state.last_h = h
        state.positions.append((x, y))
        if len(state.positions) >= 3:
            px, py = state.positions[-3]
            state.recent_displacement = float(np.hypot(x - px, y - py))
        else:
            state.recent_displacement = step

        if state.static_count >= self.static_frame_limit:
            self._upsert_static_zone(state)

        self._last_seen[track_key] = self._frame_counter
        return state

    def _zone_radius_for_track(self, state: _YoloTrackState) -> float:
        box_radius = max(state.last_w, state.last_h) * 0.75 + 10.0
        return float(max(self.static_zone_radius, min(70.0, box_radius)))

    def _find_static_zone(self, x: float, y: float) -> _StaticZone | None:
        matches = [zone for zone in self._static_zones.values() if zone.contains(x, y)]
        if not matches:
            return None
        return min(matches, key=lambda zone: zone.distance_to(x, y))

    def _upsert_static_zone(self, state: _YoloTrackState) -> _StaticZone | None:
        if state.last_x is None or state.last_y is None:
            return None

        radius = self._zone_radius_for_track(state)
        zone = self._find_static_zone(state.last_x, state.last_y)
        if zone is None:
            if len(self._static_zones) >= self.static_zone_max:
                oldest_id = min(
                    self._static_zones,
                    key=lambda zid: self._static_zones[zid].last_seen_frame,
                )
                self._static_zones.pop(oldest_id, None)
                self._static_stats["static_zones_expired"] += 1
            zone = _StaticZone(
                zone_id=self._next_static_zone_id,
                x=state.last_x,
                y=state.last_y,
                radius=radius,
                created_frame=self._frame_counter,
                last_seen_frame=self._frame_counter,
                source_track=state.key,
                hits=1,
            )
            self._static_zones[zone.zone_id] = zone
            self._next_static_zone_id += 1
            self._static_stats["static_zones_created"] += 1
            return zone

        zone.x = float(0.85 * zone.x + 0.15 * state.last_x)
        zone.y = float(0.85 * zone.y + 0.15 * state.last_y)
        zone.radius = max(zone.radius, radius)
        zone.last_seen_frame = self._frame_counter
        zone.hits += 1
        return zone

    def _has_static_release_evidence(self, state: _YoloTrackState, zone: _StaticZone) -> bool:
        if state.static_count >= max(2, self.static_frame_limit // 2):
            return False
        if state.speed_streak >= self.static_release_frames:
            return True
        if state.recent_displacement >= self.static_release_displacement:
            return True
        if state.last_x is not None and state.last_y is not None:
            if zone.distance_to(state.last_x, state.last_y) >= zone.radius + self.move_threshold:
                return True
        return False

    def _apply_static_gate(
        self,
        state: _YoloTrackState | None,
        x: float,
        y: float,
    ) -> tuple[bool, str, _StaticZone | None]:
        if state is None:
            self._static_stats["untracked_kept"] += 1
            return True, "untracked", None

        zone = self._find_static_zone(x, y)
        if zone is None:
            return True, "moving" if state.static_count < self.static_frame_limit else "static_candidate", None

        if self._frame_counter < self._static_fail_open_until:
            self._static_stats["fail_open_kept"] += 1
            return True, "fail_open", zone

        if self._has_static_release_evidence(state, zone):
            zone.released += 1
            self._static_stats["motion_released"] += 1
            return True, "motion_released", zone

        zone.last_seen_frame = self._frame_counter
        zone.blocked += 1
        self._static_stats["static_blocked"] += 1
        return False, "static_blocked", zone

    def _expire_static_zones(self) -> None:
        expired = [
            zone_id
            for zone_id, zone in self._static_zones.items()
            if self._frame_counter - zone.last_seen_frame > self.static_zone_ttl_frames
        ]
        for zone_id in expired:
            self._static_zones.pop(zone_id, None)
        if expired:
            self._static_stats["static_zones_expired"] += len(expired)

    def _update_static_starvation(self, raw_count: int, kept_count: int) -> None:
        if raw_count > 0 and kept_count == 0 and self._static_zones:
            self._static_starvation_count += 1
        else:
            self._static_starvation_count = 0

        if self._static_starvation_count >= self.static_starvation_frames:
            # Fail open briefly instead of deleting masks. This prevents a
            # full detector blackout while keeping static zones intact.
            self._static_fail_open_until = self._frame_counter + 10
            self._static_starvation_count = 0

    def get_runtime_stats(self) -> dict[str, Any]:
        return {
            "type": "yolo_roadmap",
            "track_available": self._track_available,
            "frame": self._frame_counter,
            "active_static_zones": len(self._static_zones),
            "static_starvation": self._static_starvation_count,
            "static_fail_open_remaining": max(0, self._static_fail_open_until - self._frame_counter),
            **self._static_stats,
            "zones": [
                zone.to_dict(self._frame_counter)
                for zone in sorted(self._static_zones.values(), key=lambda z: z.zone_id)
            ],
        }

    def _prune_track_history(self) -> None:
        stale_after = 150
        stale = [
            tid for tid, last_seen in self._last_seen.items()
            if self._frame_counter - last_seen > stale_after
        ]
        for tid in stale:
            self._last_seen.pop(tid, None)
            self._track_history.pop(tid, None)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_detector(
    model_path: str,
    input_size: tuple[int, int] = (288, 512),
    frames_in: int = 3,
    frames_out: int = 3,
    device: str = "cuda",
    detector_type: str = "auto",
) -> "BallDetector | TrackNetDetector | MedianBGDetector | BallSelectorDetector | YoloRoadmapDetector":
    """Select detector backend.

    Args:
        detector_type: ``"auto"`` selects TrackNet/HRNet by extension,
            ``"median_bg"`` uses MedianBGDetector,
            ``"yolo_roadmap"`` uses the Ultralytics YOLO prototype backend,
            ``"ball_selector"`` uses TrackNet + CandidateTransformer.
    """
    normalized_path = model_path.replace("\\", "/").lower()
    if detector_type == "median_bg":
        logger.info("Using MedianBGDetector (median background subtraction)")
        return MedianBGDetector(
            input_size=input_size,
            frames_in=frames_in,
            frames_out=frames_in,
            device=device,
        )
    if detector_type == "ball_selector":
        logger.info("Using BallSelectorDetector (TrackNet + CandidateTransformer)")
        return BallSelectorDetector(
            model_path=model_path,
            input_size=input_size,
            frames_in=8,
            frames_out=8,
            device=device,
        )
    if detector_type == "tracknet":
        logger.info("Using TrackNetDetector")
        return TrackNetDetector(
            model_path=model_path,
            input_size=input_size,
            frames_in=frames_in,
            frames_out=frames_out,
            device=device,
        )
    if (
        detector_type in {"yolo", "yolo_roadmap"}
        or (detector_type == "auto" and "/yolo_roadmap/" in normalized_path)
    ):
        logger.info("Using YoloRoadmapDetector")
        return YoloRoadmapDetector(
            model_path=model_path or "yolo_roadmap/best.pt",
            input_size=input_size,
            frames_in=frames_in,
            frames_out=frames_out,
            device=device,
            conf=0.15,
            imgsz=960,
            static_frame_limit=7,
        )
    # Auto-select by file extension
    if model_path.endswith(".pt"):
        logger.info("Auto-detected PyTorch model → TrackNetDetector")
        return TrackNetDetector(
            model_path=model_path,
            input_size=input_size,
            frames_in=frames_in,
            frames_out=frames_out,
            device=device,
        )
    else:
        logger.info("Auto-detected ONNX model → BallDetector")
        return BallDetector(
            model_path=model_path,
            input_size=input_size,
            frames_in=frames_in,
            frames_out=frames_out,
            device=device,
        )
