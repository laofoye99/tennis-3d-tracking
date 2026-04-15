"""Ball Selector V2.2 — inference API for integration with test platform.

Pipeline:
    1. TrackNet (frozen) → heatmap
    2. Extract top-3 peaks from heatmap
    3. CandidateTransformer → select best candidate + existence prediction

Usage:
    from src.inference_api import BallSelector

    selector = BallSelector(
        tracknet_weights="model_weights/TrackNet_finetuned.pt",
        selector_weights="model_weights/ball_selector_v2.2.pt",
        device="cuda",
        exist_threshold=0.5,  # adjust for recall/precision trade-off
    )

    # Per-clip: must call reset() at the start of each new video/clip
    selector.reset()

    # Process 8 frames at a time (TrackNet seq_len=8)
    for frames_8 in sliding_window(video, size=8, stride=8):
        results = selector.detect(frames_8, median_bg)
        # results = [
        #   {"frame": 0, "px": 945.2, "py": 312.1, "conf": 0.87},  # ball detected
        #   {"frame": 1, "px": None, "py": None, "conf": 0.12},     # no ball
        #   ...
        # ]

Requirements:
    - TrackNet weights: model_weights/TrackNet_finetuned.pt (130MB, frozen)
    - Selector weights: model_weights/ball_selector_v2.2.pt (1.1MB)
    - Input frames: BGR numpy arrays (any resolution, resized internally to 288x512)
    - Median background: precomputed per camera/clip (see utils.compute_median_bg)
"""

import numpy as np
import torch

from .model import CandidateTransformer, K, NONE_CLASS
from .precompute import load_tracknet, extract_topk_peaks
from .utils import ORIG_H, ORIG_W, preprocess_frame


class BallSelector:
    """End-to-end ball detection: TrackNet + CandidateTransformer.

    Args:
        tracknet_weights: path to TrackNet_finetuned.pt
        selector_weights: path to ball_selector_v2.pt
        device: "cuda" or "cpu"
    """

    def __init__(self, tracknet_weights, selector_weights, device="cuda",
                 exist_threshold=0.5):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.exist_threshold = exist_threshold

        # Load TrackNet (frozen)
        self.tracknet = load_tracknet(tracknet_weights, self.device)

        # Load selector
        self.selector = CandidateTransformer()
        ckpt = torch.load(selector_weights, map_location="cpu", weights_only=False)
        self.selector.load_state_dict(ckpt["model"])
        self.selector.eval().to(self.device)

        # Memory state (reset per clip)
        self._memory_coords = []  # list of (K, 3) tensors
        self._memory_meta = []    # list of (2,) tensors
        self._prev_coord = np.array([0.5, 0.5])

    def reset(self):
        """Call at the start of each new video/clip."""
        self._memory_coords.clear()
        self._memory_meta.clear()
        self._prev_coord = np.array([0.5, 0.5])

    @torch.no_grad()
    def detect(self, frames, median_bg):
        """Detect ball in 8 consecutive frames.

        Args:
            frames: list of 8 BGR numpy arrays (original resolution)
            median_bg: (3, 288, 512) float32 [0,1], precomputed median background

        Returns:
            list of 8 dicts: {"frame": int, "px": float|None, "py": float|None, "conf": float}
            px, py are in original pixel space (1920x1080).
            None means no ball detected (NONE selected).
        """
        assert len(frames) == 8, f"Expected 8 frames, got {len(frames)}"

        # 1. Preprocess + TrackNet
        processed = [preprocess_frame(f) for f in frames]
        inp = np.concatenate([median_bg] + processed, axis=0)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        heatmaps = self.tracknet(inp_t)[0].cpu().numpy()  # (8, 288, 512)

        # 2. Extract top-K peaks + build features
        all_peaks = []
        cand_features = np.zeros((8, K, 7), dtype=np.float32)

        for t in range(8):
            peaks = extract_topk_peaks(heatmaps[t])
            all_peaks.append(peaks)
            max_conf = max(peaks[:, 2].max(), 1e-6)
            for i in range(K):
                px, py, conf = peaks[i]
                cand_features[t, i] = [
                    px, py, conf,
                    i / max(K - 1, 1),
                    conf / max_conf,
                    px - self._prev_coord[0],
                    py - self._prev_coord[1],
                ]
            if peaks[0, 2] > 0.3:
                self._prev_coord = peaks[0, :2].copy()

        # 3. Build memory tensors
        mem_len = self.selector.memory_len
        mem_c = torch.zeros(1, mem_len, K, 3, device=self.device)
        mem_m = torch.zeros(1, mem_len, 2, device=self.device)
        mem_mask = torch.zeros(1, mem_len, dtype=torch.bool, device=self.device)

        n = min(len(self._memory_coords), mem_len)
        if n > 0:
            mem_c[0, mem_len - n:] = torch.stack(self._memory_coords[-n:])
            mem_m[0, mem_len - n:] = torch.stack(self._memory_meta[-n:])
            mem_mask[0, mem_len - n:] = True

        # 4. Selector forward (V2.2: decoupled select + exist)
        cands_t = torch.from_numpy(cand_features).unsqueeze(0).to(self.device)
        sel_logits, exist_logits = self.selector(cands_t, mem_c, mem_m, mem_mask)
        sel_idx, sel_coords, conf = self.selector.get_predictions(
            sel_logits, exist_logits, cands_t, exist_threshold=self.exist_threshold,
        )

        # 5. Update memory
        mc, mm = self.selector.build_memory_entry(cands_t, sel_logits, exist_logits)
        for t in range(8):
            self._memory_coords.append(mc[0, t])
            self._memory_meta.append(mm[0, t])
        # Keep bounded
        max_buf = mem_len * 10
        if len(self._memory_coords) > max_buf:
            self._memory_coords = self._memory_coords[-max_buf:]
            self._memory_meta = self._memory_meta[-max_buf:]

        # 6. Build results
        results = []
        for t in range(8):
            idx = sel_idx[0, t].item()
            c = conf[0, t].item()

            if idx < K and all_peaks[t][idx, 2] > 0:
                px = float(all_peaks[t][idx, 0] * ORIG_W)
                py = float(all_peaks[t][idx, 1] * ORIG_H)
            else:
                px, py = None, None

            results.append({"frame": t, "px": px, "py": py, "conf": c})

        return results
