"""ONNX ball detection inference with GPU/CPU fallback."""

import logging

import cv2
import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        self.input_h, self.input_w = input_size
        self.frames_in = frames_in
        self.frames_out = frames_out

        providers = self._get_providers(device)
        logger.info("Loading ONNX model: %s (providers=%s)", model_path, providers)
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self._use_cuda = "CUDAExecutionProvider" in self.session.get_providers()
        if self._use_cuda:
            self.io_binding = self.session.io_binding()
        logger.info("BallDetector ready (CUDA=%s)", self._use_cuda)

    @staticmethod
    def _get_providers(device: str) -> list[str]:
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

        if self._use_cuda:
            input_torch = torch.from_numpy(input_tensor).cuda()
            self.io_binding.bind_input(
                name=self.input_name,
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(input_torch.shape),
                buffer_ptr=input_torch.data_ptr(),
            )
            self.io_binding.bind_output(name=self.output_name)
            self.session.run_with_iobinding(self.io_binding)
            output = self.io_binding.copy_outputs_to_cpu()[0]
        else:
            output = self.session.run(
                [self.output_name], {self.input_name: input_tensor}
            )[0]

        # output shape: (1, frames_out, H, W)
        output = torch.sigmoid(torch.from_numpy(output[0])).numpy()
        return output  # (frames_out, H, W)
