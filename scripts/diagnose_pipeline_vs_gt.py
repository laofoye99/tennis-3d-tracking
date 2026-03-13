"""Simulate the full video_pipeline flow (BallTracker + court-X filter) and compare to GT.

Reproduces the exact logic: BallTracker with history/prediction/fallback → homography → court-X filter.
Shows frame-by-frame what the pipeline outputs vs what GT expects.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from app.pipeline.postprocess import BallTracker
from app.pipeline.homography import HomographyTransformer

MODEL_PATH = "model_weight/TrackNet_best.pt"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
INPUT_H, INPUT_W = 288, 512
SEQ_LEN = 8
THRESHOLD = 0.5
HEATMAP_MASK = [(0, 0, 620, 40)]


def load_gt(ann_dir):
    gt = {}
    for jf in sorted(ann_dir.glob("*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
            shapes = data.get("shapes", [])
            if shapes and shapes[0].get("points"):
                pts = shapes[0]["points"][0]
                gt[int(jf.stem)] = (float(pts[0]), float(pts[1]))
        except Exception:
            continue
    return gt


def load_tracknet(model_path, device="cuda"):
    from app.pipeline.tracknet import TrackNet
    in_dim = (SEQ_LEN + 1) * 3
    model = TrackNet(in_dim=in_dim, out_dim=SEQ_LEN)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return model, dev


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


def compute_median(cap, n_samples=200):
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
            small = cv2.resize(frame, (INPUT_W, INPUT_H))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(small_rgb)
    median = np.median(frames, axis=0).astype(np.uint8)
    return median.astype(np.float32).transpose(2, 0, 1) / 255.0


def simulate_pipeline(cam_name, video_path, ann_dir, homo_key, model, device):
    print(f"\n{'='*100}")
    print(f"  {cam_name}:  simulating full pipeline (BallTracker + court-X filter)")
    print(f"{'='*100}")

    gt = load_gt(ann_dir)
    print(f"  GT: {len(gt)} frames ({min(gt)}-{max(gt)})")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total_frames} frames, {vid_w}x{vid_h}")

    bg = compute_median(cap)
    homography = HomographyTransformer(HOMOGRAPHY_PATH, homo_key)

    # Create BallTracker exactly like video_pipeline does
    tracker = BallTracker(
        original_size=(vid_w, vid_h),
        threshold=THRESHOLD,
        heatmap_mask=[tuple(r) for r in HEATMAP_MASK],
    )

    # Read all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = {}
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames[i] = frame
    cap.release()

    # Process in batches of 8 (same as pipeline)
    # Pipeline processes sequentially: batch 0-7, 8-15, 16-23, ...
    pipeline_detections = {}  # frame_index -> detection dict or None

    for batch_start in range(0, total_frames, SEQ_LEN):
        batch_end = min(batch_start + SEQ_LEN, total_frames)
        batch_indices = list(range(batch_start, batch_end))

        if len(batch_indices) < SEQ_LEN:
            break  # incomplete batch

        # Mask OSD + preprocess (same as _prefetch_thread)
        frames_masked = []
        for fi in batch_indices:
            masked = all_frames[fi].copy()
            masked[0:41, 0:603] = 0
            frames_masked.append(masked)

        processed = [preprocess(f) for f in frames_masked]
        stacked = np.concatenate([bg] + processed, axis=0)
        inp = torch.from_numpy(stacked[np.newaxis]).to(device)

        with torch.no_grad():
            heatmaps = model(inp)[0].cpu().numpy()

        for i in range(SEQ_LEN):
            fi = batch_start + i
            # process_heatmap_multi (same as video_pipeline)
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
                    "fallback": blob.get("fallback", False),
                })

            if not candidates:
                pipeline_detections[fi] = None
                continue

            top = candidates[0]
            pipeline_detections[fi] = top

    # ── Compare pipeline output vs GT ───────────────────────────
    gt_frames = sorted(gt.keys())

    print(f"\n{'Frm':>4} | {'GT pixel':>18} | {'GT wX':>6} | {'Pipeline pixel':>18} | {'Pipe wX':>6} | {'Err':>6} | {'fallback':>8} | Notes")
    print("-" * 130)

    correct = 0
    missed = 0
    wrong = 0
    errors = []

    for fi in gt_frames:
        gt_px, gt_py = gt[fi]
        gt_wx, gt_wy = homography.pixel_to_world(gt_px, gt_py)
        gt_str = f"({gt_px:7.1f},{gt_py:7.1f})"

        det = pipeline_detections.get(fi)
        if det is None:
            missed += 1
            print(f"{fi:4d} | {gt_str} | {gt_wx:5.1f}m | {'--- NO DETECTION ---':>18} |        |        |          | MISS")
            continue

        dpx, dpy = det["pixel_x"], det["pixel_y"]
        dwx = det["world_x"]
        err = np.sqrt((dpx - gt_px)**2 + (dpy - gt_py)**2)
        errors.append(err)
        det_str = f"({dpx:7.1f},{dpy:7.1f})"
        fb = "YES" if det["fallback"] else ""

        if err <= 30:
            correct += 1
            note = "OK"
        else:
            wrong += 1
            note = f"WRONG (err={err:.0f}px, wX diff={abs(dwx-gt_wx):.1f}m)"

        print(f"{fi:4d} | {gt_str} | {gt_wx:5.1f}m | {det_str} | {dwx:5.1f}m | {err:5.0f}px | {fb:>8} | {note}")

    # Summary
    total = len(gt_frames)
    print(f"\n{'='*80}")
    print(f"SUMMARY  {cam_name}  ({total} GT frames)")
    print(f"{'='*80}")
    print(f"  CORRECT  (<30px):  {correct:>3} / {total}  ({100*correct/total:.1f}%)")
    print(f"  WRONG    (>30px):  {wrong:>3} / {total}  ({100*wrong/total:.1f}%)")
    print(f"  MISSED   (no det): {missed:>3} / {total}  ({100*missed/total:.1f}%)")
    if errors:
        ea = np.array(errors)
        print(f"  Pixel error (detected frames): mean={ea.mean():.1f}px, median={np.median(ea):.1f}px, max={ea.max():.1f}px")


if __name__ == "__main__":
    print("Loading TrackNet...")
    model, device = load_tracknet(MODEL_PATH)

    configs = [
        ("cam68", "uploads/cam68_clip.mp4", Path("uploads/cam68_clip"), "cam68"),
        ("cam66", "uploads/cam66_clip.mp4", Path("uploads/cam66_clip"), "cam66"),
    ]
    for cam_name, vpath, ann_dir, hkey in configs:
        simulate_pipeline(cam_name, vpath, ann_dir, hkey, model, device)

    print("\nDone!")
