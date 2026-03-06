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
        """Run inference on a list of frames."""
        processed = [self.preprocess_frame(f) for f in frames]
        stacked = np.concatenate(processed, axis=0)
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

        output = torch.sigmoid(torch.from_numpy(output[0])).numpy()
        return output

class PlayerDetector:
    """YOLO-based person detector for identifying players."""
    def __init__(self, model_name='yolo11n.pt', device='cuda'):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')

    def detect(self, frame):
        results = self.model(frame, classes=[0], verbose=False)
        persons = []
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cx, cy = (x1 + x2) / 2, y2
                persons.append({"px": cx, "py": cy, "bbox": [x1, y1, x2, y2], "conf": conf})
        
        persons.sort(key=lambda x: x['conf'], reverse=True)
        top_2 = persons[:2]
        top_2.sort(key=lambda x: x['py'], reverse=True) # Near is larger y
        
        return {
            "near": top_2[0] if len(top_2) > 0 else None,
            "far": top_2[1] if len(top_2) > 1 else None
        }
