"""Side-by-side comparison: heatmap export (per-frame, no tracker) vs pipeline (with tracker state).

Reads results.json from debug_heatmaps/ and re-runs pipeline simulation.
Highlights discrepancies.
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


def compare_camera(cam_name, video_path, ann_dir, homo_key, model, device):
    print(f"\n{'='*130}")
    print(f"  {cam_name}: heatmap export (独立) vs pipeline (有tracker状态)")
    print(f"{'='*130}")

    gt = load_gt(ann_dir)
    hm_json_path = Path("debug_heatmaps") / cam_name / "results.json"
    with open(hm_json_path) as f:
        hm_results = json.load(f)

    homography = HomographyTransformer(HOMOGRAPHY_PATH, homo_key)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg = compute_median(cap)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = {}
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames[i] = frame
    cap.release()

    tracker = BallTracker(
        original_size=(vid_w, vid_h),
        threshold=THRESHOLD,
        heatmap_mask=[tuple(r) for r in HEATMAP_MASK],
    )

    # Run pipeline sequentially
    pipeline_results = {}
    for batch_start in range(0, total_frames, SEQ_LEN):
        batch_indices = list(range(batch_start, batch_start + SEQ_LEN))
        if not all(bi in all_frames for bi in batch_indices):
            break

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
            blobs = tracker.process_heatmap_multi(heatmaps[i])
            if not blobs:
                pipeline_results[fi] = None
                continue

            candidates = []
            for blob in blobs:
                wx, wy = homography.pixel_to_world(blob["pixel_x"], blob["pixel_y"])
                if not (homography.court_x_min <= wx <= homography.court_x_max):
                    continue
                candidates.append({
                    "pixel_x": blob["pixel_x"],
                    "pixel_y": blob["pixel_y"],
                    "world_x": wx,
                    "blob_sum": blob["blob_sum"],
                    "fallback": blob.get("fallback", False),
                })

            pipeline_results[fi] = candidates[0] if candidates else None

    # ── Side-by-side comparison ─────────────────────────────────
    gt_frames = sorted(gt.keys())

    print(f"\n{'Frm':>4} | {'GT pixel':>18} |  Heatmap export (独立)          | Pipeline (有tracker)           | {'Match?':>6}")
    print(f"     |                    |  blobs@0.5  top1→GT  top1_wX    | pixel          →GT  wX  fb     |")
    print("-" * 130)

    for fi in gt_frames:
        gt_px, gt_py = gt[fi]
        gt_wx, _ = homography.pixel_to_world(gt_px, gt_py)
        gt_str = f"({gt_px:7.1f},{gt_py:7.1f})"

        # Heatmap export result
        hm_r = hm_results.get(str(fi), {})
        hm_blobs_05 = hm_r.get("blobs", {}).get("0.5", {}).get("after_court_x_filter", [])
        if hm_blobs_05:
            hb = hm_blobs_05[0]
            hm_dist = hb["dist_to_gt_px"]
            hm_str = f"  {len(hm_blobs_05):>1} blob  {hm_dist:>5.0f}px  wX={hb['world_x']:.1f}m"
        else:
            hm_all = hm_r.get("blobs", {}).get("0.5", {}).get("all", [])
            hm_str = f"  0 blob (raw={len(hm_all)})     ---     "

        # Pipeline result
        p = pipeline_results.get(fi)
        if p is not None:
            p_dist = np.sqrt((p["pixel_x"] - gt_px)**2 + (p["pixel_y"] - gt_py)**2)
            fb = "fb" if p["fallback"] else "  "
            p_str = f"({p['pixel_x']:7.1f},{p['pixel_y']:7.1f}) {p_dist:>5.0f}px {p['world_x']:4.1f}m {fb}"
        else:
            p_dist = None
            p_str = "--- NO DETECTION ---         "

        # Check if they match
        if hm_blobs_05 and p is not None:
            hm_px = hm_blobs_05[0]["cx_orig"]
            hm_py = hm_blobs_05[0]["cy_orig"]
            diff = np.sqrt((hm_px - p["pixel_x"])**2 + (hm_py - p["pixel_y"])**2)
            if diff < 5:
                match = "SAME"
            else:
                match = f"DIFF {diff:.0f}px"
        elif not hm_blobs_05 and p is None:
            match = "SAME"
        elif not hm_blobs_05 and p is not None:
            match = f"PIPE+{p_dist:.0f}" if p_dist else "PIPE+"
        else:
            match = "HM+"

        print(f"{fi:4d} | {gt_str} | {hm_str} | {p_str} | {match}")

    print()


if __name__ == "__main__":
    print("Loading TrackNet...")
    model, device = load_tracknet(MODEL_PATH)

    configs = [
        ("cam68", "uploads/cam68_clip.mp4", Path("uploads/cam68_clip"), "cam68"),
        ("cam66", "uploads/cam66_clip.mp4", Path("uploads/cam66_clip"), "cam66"),
    ]
    for c in configs:
        compare_camera(*c, model, device)

    print("Done!")
