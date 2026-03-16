"""Run TrackNet on a video and export ALL blob detections as LabelMe JSON per frame.

No top-1 filtering, no court-X filtering — every detected blob is saved.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from app.pipeline.postprocess import BallTracker
from app.pipeline.tracknet import TrackNet


def preprocess_frame(frame, input_w, input_h):
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


def compute_median(cap, input_w, input_h, n_samples=200):
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


def blobs_to_labelme(blobs, frame_index, img_w, img_h):
    """Convert blob list to LabelMe JSON dict."""
    shapes = []
    for i, blob in enumerate(blobs):
        shapes.append({
            "label": "ball",
            "score": round(blob["blob_sum"], 4),
            "points": [[blob["pixel_x"], blob["pixel_y"]]],
            "group_id": i if len(blobs) > 1 else None,
            "description": f"blob_sum={blob['blob_sum']:.2f} blob_max={blob['blob_max']:.3f} area={blob['blob_area']}",
            "difficult": False,
            "shape_type": "point",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })

    return {
        "version": "2.5.4",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{frame_index:05d}.jpg",
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "uploads/cam66_20260307_173403_2min.mp4"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "exports/cam66_2min_blobs"
    threshold_override = float(sys.argv[3]) if len(sys.argv) > 3 else None

    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    input_h, input_w = model_cfg["input_size"]
    seq_len = model_cfg.get("frames_in", 8)
    threshold = threshold_override if threshold_override is not None else model_cfg.get("threshold", 0.1)
    heatmap_mask = [tuple(m) for m in model_cfg.get("heatmap_mask", [[0, 0, 620, 40]])]
    max_blobs = 10  # 多给一些，记录所有 blob

    # Load model
    in_dim = (seq_len + 1) * 3
    model = TrackNet(in_dim=in_dim, out_dim=seq_len)
    ckpt = torch.load(model_cfg["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded: {model_cfg['path']} -> {device}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path} ({vid_w}x{vid_h}, {total_frames} frames)")

    # BallTracker (no filtering, just blob detection)
    tracker = BallTracker(
        original_size=(vid_w, vid_h),
        threshold=threshold,
        heatmap_mask=heatmap_mask,
    )

    # Background median
    print("Computing background median ...")
    bg = compute_median(cap, input_w, input_h)

    # Read all frames
    print("Reading frames ...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = {}
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames[i] = frame
    cap.release()
    print(f"Read {len(all_frames)} frames")

    # Output dir
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Inference + export
    n_with_blobs = 0
    n_total_blobs = 0

    for batch_start in range(0, total_frames, seq_len):
        batch_indices = list(range(batch_start, batch_start + seq_len))
        if not all(bi in all_frames for bi in batch_indices):
            break

        # Preprocess with OSD mask on raw frame
        processed = []
        for fi in batch_indices:
            frame = all_frames[fi].copy()
            frame[0:41, 0:603] = 0  # OSD mask on raw frame
            processed.append(preprocess_frame(frame, input_w, input_h))

        stacked = np.concatenate([bg] + processed, axis=0)
        inp = torch.from_numpy(stacked[np.newaxis]).to(device)

        with torch.no_grad():
            heatmaps = model(inp)[0].cpu().numpy()

        for i in range(seq_len):
            fi = batch_start + i
            blobs = tracker.process_heatmap_multi(heatmaps[i], max_blobs=max_blobs)

            if blobs:
                n_with_blobs += 1
                n_total_blobs += len(blobs)

            # Save LabelMe JSON (even if no blobs — shapes will be empty)
            labelme_data = blobs_to_labelme(blobs, fi, vid_w, vid_h)
            json_path = out / f"{fi:05d}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

        done = min(batch_start + seq_len, total_frames)
        print(f"\r  {done}/{total_frames} frames processed ...", end="", flush=True)

    print(f"\n\nDone! Output: {out}")
    print(f"  Frames with blobs: {n_with_blobs}/{total_frames}")
    print(f"  Total blobs: {n_total_blobs}")
    print(f"  Avg blobs/frame (when detected): {n_total_blobs / max(n_with_blobs, 1):.1f}")


if __name__ == "__main__":
    main()
