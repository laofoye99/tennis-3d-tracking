"""Ball detection inference with GPU/CPU fallback.

Supports three detector backends:
    - BallDetector:      ONNX-based HRNet (frames_in=3, frames_out=3)
    - TrackNetDetector:  PyTorch-based TrackNet (seq_len=8, bg_mode='concat')
    - MedianBGDetector:  Median background subtraction (frames_in=30, no GPU)

Use ``create_detector()`` factory to select backend.  Default auto-selects by
model file extension; pass ``detector_type="median_bg"`` to use MedianBGDetector.
"""

import logging
from typing import Optional

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
        from app.pipeline.tracknet import TrackNet

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

        onnx_path = model_path.replace('.pt', '.onnx')
        self._use_onnx = False
        self.model = None

        import os
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
                self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                self._use_onnx = True
                actual = self._ort_session.get_providers()
                logger.info("TrackNet using ONNX Runtime (%s): %s", actual[0], onnx_path)
            except Exception as e:
                logger.warning("ONNX Runtime failed, falling back to PyTorch: %s", e)

        if not self._use_onnx:
            # PyTorch fallback
            self.model = TrackNet(in_dim=in_dim, out_dim=frames_in)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.model.to(self.device)

        # Background (median) frame — (3, H, W) float32 in [0, 1]
        self._bg_frame: Optional[np.ndarray] = None
        self._video_median_computed = False

        # Try loading pre-computed median from src/bg_median_{camera}.png
        self._try_load_static_median()

        # Running median buffer for live camera use (keep small for speed)
        self._bg_buffer: list[np.ndarray] = []
        self._bg_max_frames: int = 20  # smaller = faster median computation

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
            self._video_median_computed = True
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
        """Update running background median for live camera use."""
        self._bg_buffer.append(preprocessed)
        if len(self._bg_buffer) > self._bg_max_frames:
            self._bg_buffer.pop(0)
        # Recompute median every 10 frames (or on first frame)
        if self._bg_frame is None or len(self._bg_buffer) % 20 == 0:
            self._bg_frame = np.median(self._bg_buffer, axis=0).astype(np.float32)

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
# Factory function
# ---------------------------------------------------------------------------

def create_detector(
    model_path: str,
    input_size: tuple[int, int] = (288, 512),
    frames_in: int = 3,
    frames_out: int = 3,
    device: str = "cuda",
    detector_type: str = "auto",
) -> "BallDetector | TrackNetDetector | MedianBGDetector":
    """Select detector backend.

    Args:
        detector_type: ``"auto"`` (default) selects by file extension,
            ``"median_bg"`` uses MedianBGDetector (no model file needed).
    """
    if detector_type == "median_bg":
        logger.info("Using MedianBGDetector (median background subtraction)")
        return MedianBGDetector(
            input_size=input_size,
            frames_in=frames_in,
            frames_out=frames_in,
            device=device,
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
