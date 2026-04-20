"""Offline TrackNet peak extraction.

Runs frozen TrackNet on all clips, extracts top-K peaks per frame, saves as .npz.

Usage:
    python -m src.precompute
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from app.pipeline.tracknet import TrackNet
from .utils import INPUT_H, INPUT_W, preprocess_frame, get_or_compute_median_bg

K = 3  # top-K peaks per frame
SEQ_LEN = 8
PEAK_THRESHOLD = 0.1
MIN_PEAK_DIST = 20  # minimum pixel distance between peaks in heatmap space


def load_tracknet(weights_path, device="cuda"):
    model = TrackNet(in_dim=27, out_dim=8)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def extract_topk_peaks(heatmap, k=K, threshold=PEAK_THRESHOLD, min_dist=MIN_PEAK_DIST):
    """Extract top-K peaks from a single heatmap (H, W).

    Returns:
        peaks: (K, 3) array of (px_norm, py_norm, conf). Padded with zeros if < K peaks.
    """
    H, W = heatmap.shape
    peaks = []
    hm = heatmap.copy()

    for _ in range(k):
        max_val = hm.max()
        if max_val < threshold:
            break

        idx = hm.argmax()
        y = idx // W
        x = idx % W

        # Normalize to [0, 1] in original image space
        px_norm = x / W  # already in heatmap space ratio
        py_norm = y / H

        peaks.append((px_norm, py_norm, float(max_val)))

        # Suppress this peak region
        y_min = max(0, y - min_dist)
        y_max = min(H, y + min_dist + 1)
        x_min = max(0, x - min_dist)
        x_max = min(W, x + min_dist + 1)
        hm[y_min:y_max, x_min:x_max] = 0

    # Pad to K
    while len(peaks) < k:
        peaks.append((0.0, 0.0, 0.0))

    return np.array(peaks, dtype=np.float32)  # (K, 3)


def precompute_clip(model, clip_dir, output_path, device, cache_dir="data/median_bg_cache"):
    """Run TrackNet on a clip and save top-K peaks per frame.

    Saves: {frame_idx: (K, 3)} as .npz
    """
    clip_dir = Path(clip_dir)
    median_bg = get_or_compute_median_bg(str(clip_dir), cache_dir=cache_dir)

    jpg_files = sorted(clip_dir.glob("*.jpg"))
    frame_indices = [int(f.stem) for f in jpg_files]
    frame_indices.sort()

    all_peaks = {}  # frame_idx → (K, 3)

    for start in range(0, len(frame_indices) - SEQ_LEN + 1, SEQ_LEN):
        window_indices = frame_indices[start: start + SEQ_LEN]

        frames = []
        for fi in window_indices:
            img = cv2.imread(str(clip_dir / f"{fi:05d}.jpg"))
            if img is None:
                break
            frames.append(preprocess_frame(img))

        if len(frames) < SEQ_LEN:
            break

        input_np = np.concatenate([median_bg] + frames, axis=0)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmaps = model(input_tensor)[0].cpu().numpy()  # (8, H, W)

        for t, fi in enumerate(window_indices):
            peaks = extract_topk_peaks(heatmaps[t])
            all_peaks[fi] = peaks

    # Also handle remaining frames (last incomplete window)
    remaining = len(frame_indices) % SEQ_LEN
    if remaining > 0 and len(frame_indices) >= SEQ_LEN:
        # Process the last full window that overlaps
        start = len(frame_indices) - SEQ_LEN
        window_indices = frame_indices[start: start + SEQ_LEN]
        frames = []
        for fi in window_indices:
            img = cv2.imread(str(clip_dir / f"{fi:05d}.jpg"))
            if img is None:
                break
            frames.append(preprocess_frame(img))

        if len(frames) == SEQ_LEN:
            input_np = np.concatenate([median_bg] + frames, axis=0)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(device)
            with torch.no_grad():
                heatmaps = model(input_tensor)[0].cpu().numpy()
            for t, fi in enumerate(window_indices):
                if fi not in all_peaks:
                    all_peaks[fi] = extract_topk_peaks(heatmaps[t])

    # Save as npz: keys are string frame indices
    save_dict = {str(fi): peaks for fi, peaks in all_peaks.items()}
    np.savez_compressed(output_path, **save_dict)

    return len(all_peaks)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="model_weights/TrackNet_finetuned.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="data/peaks")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_tracknet(args.weights, device)
    print(f"TrackNet loaded on {device}")

    # All clip directories
    clips = []

    # cam66_184338/train
    base = "D:/tennis/blob_frame_different/GT/cam66_20260323_184338/train"
    for clip_name in sorted(os.listdir(base)):
        p = os.path.join(base, clip_name)
        if os.path.isdir(p):
            clips.append((p, f"cam66_184338_train_{clip_name}"))

    # cam66_184338/test
    base_test = "D:/tennis/blob_frame_different/GT/cam66_20260323_184338/test"
    for clip_name in sorted(os.listdir(base_test)):
        p = os.path.join(base_test, clip_name)
        if os.path.isdir(p):
            clips.append((p, f"cam66_184338_test_{clip_name}"))

    # Other directories
    clips.append(("D:/tennis/blob_frame_different/GT/cam66_20260307_173403_2min", "cam66_173403"))
    clips.append(("D:/tennis/blob_frame_different/GT/cam68_20260307_173403_2min", "cam68_173403"))
    clips.append(("D:/tennis/blob_frame_different/GT/cam66", "cam66_multi"))
    clips.append(("D:/tennis/blob_frame_different/GT/cam68", "cam68_multi"))

    total = 0
    for clip_dir, name in clips:
        output_path = os.path.join(args.output_dir, f"{name}.npz")
        n = precompute_clip(model, clip_dir, output_path, device)
        total += n
        print(f"  {name}: {n} frames → {output_path}")

    print(f"\nDone. Total: {total} frames across {len(clips)} clips.")


if __name__ == "__main__":
    main()
