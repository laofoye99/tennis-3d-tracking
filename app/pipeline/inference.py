"""Ball detection inference with GPU/CPU fallback.

Supports two model backends:
    - BallDetector:      ONNX-based HRNet (frames_in=3, frames_out=3)
    - TrackNetDetector:  PyTorch-based TrackNet (seq_len=8, bg_mode='concat')

Use ``create_detector()`` factory to auto-select backend based on model file
extension (.onnx → BallDetector, .pt → TrackNetDetector).
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

        # Load model (author's original architecture: TrackNet(in_dim, out_dim))
        logger.info("Loading TrackNet model: %s (device=%s)", model_path, self.device)
        if bg_mode == "concat":
            in_dim = (frames_in + 1) * 3  # 27 for seq_len=8
        else:
            in_dim = frames_in * 3
        self.model = TrackNet(in_dim=in_dim, out_dim=frames_in)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.to(self.device)

        # Background (median) frame — (3, H, W) float32 in [0, 1]
        self._bg_frame: Optional[np.ndarray] = None
        self._video_median_computed = False

        # Running median buffer for live camera use
        self._bg_buffer: list[np.ndarray] = []
        self._bg_max_frames: int = 50

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "TrackNetDetector ready (CUDA=%s, params=%s, seq_len=%d, bg=%s)",
            self.device.type == "cuda", f"{n_params:,}", frames_in, bg_mode,
        )

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
        if self._bg_frame is None or len(self._bg_buffer) % 10 == 0:
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
        input_tensor = torch.from_numpy(stacked[np.newaxis]).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)  # (1, seq_len, H, W) — sigmoid included
            output = output[0].cpu().numpy()   # (seq_len, H, W)

        return output


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_detector(
    model_path: str,
    input_size: tuple[int, int] = (288, 512),
    frames_in: int = 3,
    frames_out: int = 3,
    device: str = "cuda",
) -> BallDetector | TrackNetDetector:
    """Auto-select detector backend based on model file extension.

    - ``.onnx`` → BallDetector (ONNX Runtime)
    - ``.pt``   → TrackNetDetector (PyTorch native)
    """
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
