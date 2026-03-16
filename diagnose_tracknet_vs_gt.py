"""Compare TrackNet detections against ground-truth LabelImg annotations.

Runs TrackNet on cam66_clip.mp4 and cam68_clip.mp4 for frames 51-83,
then compares pixel-level detections against hand-labeled annotations.
Outputs heatmap visualizations and error statistics.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = "model_weight/TrackNet_best.pt"
VIDEOS = {
    "cam66": "uploads/cam66_clip.mp4",
    "cam68": "uploads/cam68_clip.mp4",
}
ANNOTATION_DIRS = {
    "cam66": Path("uploads/cam66_clip"),
    "cam68": Path("uploads/cam68_clip"),
}
GT_FRAME_RANGE = (51, 83)  # inclusive
INPUT_H, INPUT_W = 288, 512
SEQ_LEN = 8
THRESHOLD = 0.5
OUTPUT_DIR = Path("debug_gt_vs_tracknet")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Load ground truth ───────────────────────────────────────────────
def load_gt(ann_dir: Path) -> dict[int, tuple[float, float]]:
    """Load LabelImg annotations → {frame: (pixel_x, pixel_y)}."""
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


# ── TrackNet inference ──────────────────────────────────────────────
def load_tracknet(model_path: str, device: str = "cuda"):
    from app.pipeline.tracknet import TrackNet

    in_dim = (SEQ_LEN + 1) * 3  # 27 (bg concat)
    model = TrackNet(in_dim=in_dim, out_dim=SEQ_LEN)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return model, dev


def preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR → RGB, resize, /255, CHW."""
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


def compute_median(cap, start, end, n_samples=200):
    total = end - start
    n = min(n_samples, total)
    indices = set(int(start + i * total / n) for i in range(n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for i in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            small = cv2.resize(frame, (INPUT_W, INPUT_H))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            frames.append(small_rgb)
    median = np.median(frames, axis=0).astype(np.uint8)
    return median.astype(np.float32).transpose(2, 0, 1) / 255.0


def extract_blob(heatmap: np.ndarray, threshold: float = 0.5):
    """Extract all blobs from a heatmap at model resolution.
    Returns list of (cx_model, cy_model, blob_sum, blob_max, blob_area)."""
    hm = np.where(heatmap > threshold, heatmap, 0.0).astype(np.float32)
    binary = (hm > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for j in range(1, n_labels):
        mask = labels == j
        blob_sum = float(hm[mask].sum())
        if blob_sum <= 0:
            continue
        cx = float(np.sum(np.where(mask)[1] * hm[mask]) / blob_sum)
        cy = float(np.sum(np.where(mask)[0] * hm[mask]) / blob_sum)
        blob_max = float(heatmap[mask].max())
        blob_area = int(stats[j, cv2.CC_STAT_AREA])
        blobs.append((cx, cy, blob_sum, blob_max, blob_area))
    blobs.sort(key=lambda b: b[2], reverse=True)
    return blobs


def scale_to_orig(cx_model, cy_model, orig_w=1920, orig_h=1080):
    return cx_model * (orig_w / INPUT_W), cy_model * (orig_h / INPUT_H)


# ── Main analysis ───────────────────────────────────────────────────
def analyze_camera(cam_name: str, video_path: str, ann_dir: Path, model, device):
    print(f"\n{'='*70}")
    print(f"  Analyzing {cam_name}: {video_path}")
    print(f"{'='*70}")

    gt = load_gt(ann_dir)
    print(f"  Ground truth: {len(gt)} frames ({min(gt):05d}-{max(gt):05d})")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total_frames} frames, {fps:.1f} fps, {orig_w}x{orig_h}")

    # Compute background median
    bg = compute_median(cap, 0, total_frames)
    print(f"  Background median computed")

    # We need to process frames in batches of 8
    # GT frames are 51-83, so we need batches covering those frames
    # Batch starts: frames are 0-indexed from video start
    # batch 0: frames 0-7, batch 1: frames 8-15, ..., batch 6: frames 48-55, batch 7: frames 56-63, ...

    # Read ALL frames we need (some buffer around GT range)
    start_read = max(0, GT_FRAME_RANGE[0] - SEQ_LEN)  # need some before for context
    end_read = min(total_frames, GT_FRAME_RANGE[1] + SEQ_LEN + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = {}
    for fi in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if start_read <= fi <= end_read:
            all_frames[fi] = frame
    cap.release()
    print(f"  Read {len(all_frames)} frames ({start_read}-{end_read})")

    # Process in batches of 8 that cover GT frames
    # The pipeline processes frames_in=8 at a time, producing 8 heatmaps
    # heatmap[i] corresponds to frame (batch_start + i)
    results = {}  # frame_idx → {heatmap, blobs, detection, gt}

    # Find batch ranges that cover GT frames
    # The pipeline starts from frame 0 and processes every 8 frames
    # batch_start = processed_count - 8 + start_frame
    # Let's just process overlapping batches to cover all GT frames
    for batch_start in range(start_read, end_read - SEQ_LEN + 1):
        # Check if this batch covers any GT frame
        batch_frames_idx = list(range(batch_start, batch_start + SEQ_LEN))
        has_gt = any(fi in gt for fi in batch_frames_idx)
        if not has_gt:
            continue

        # Collect frames for this batch
        frames_bgr = []
        valid = True
        for fi in batch_frames_idx:
            if fi in all_frames:
                frames_bgr.append(all_frames[fi])
            else:
                valid = False
                break
        if not valid:
            continue

        # Preprocess
        processed = [preprocess(f) for f in frames_bgr]

        # Build input: [bg, f0, f1, ..., f7] → (1, 27, H, W)
        all_channels = [bg] + processed
        stacked = np.concatenate(all_channels, axis=0)
        inp = torch.from_numpy(stacked[np.newaxis]).to(device)

        # Inference
        with torch.no_grad():
            output = model(inp)
            heatmaps = output[0].cpu().numpy()  # (8, 288, 512)

        # Process each heatmap
        for i in range(SEQ_LEN):
            fi = batch_start + i
            if fi not in gt:
                continue
            if fi in results:
                continue  # already processed by an earlier batch

            hm = heatmaps[i]
            blobs = extract_blob(hm, THRESHOLD)

            # Also try lower thresholds
            blobs_03 = extract_blob(hm, 0.3)
            blobs_01 = extract_blob(hm, 0.1)

            # Best detection at threshold 0.5
            detection = None
            if blobs:
                cx_m, cy_m = blobs[0][0], blobs[0][1]
                px, py = scale_to_orig(cx_m, cy_m, orig_w, orig_h)
                detection = (px, py, blobs[0][2], blobs[0][3], blobs[0][4])

            gt_px, gt_py = gt[fi]

            # GT position in model coords
            gt_mx = gt_px / (orig_w / INPUT_W)
            gt_my = gt_py / (orig_h / INPUT_H)

            # Heatmap value at GT position
            gt_mx_int = int(np.clip(round(gt_mx), 0, INPUT_W - 1))
            gt_my_int = int(np.clip(round(gt_my), 0, INPUT_H - 1))
            hm_at_gt = float(hm[gt_my_int, gt_mx_int])

            # Heatmap max in a 10px radius around GT (model coords)
            y_lo = max(0, gt_my_int - 10)
            y_hi = min(INPUT_H, gt_my_int + 11)
            x_lo = max(0, gt_mx_int - 10)
            x_hi = min(INPUT_W, gt_mx_int + 11)
            hm_near_gt = float(hm[y_lo:y_hi, x_lo:x_hi].max()) if (y_hi > y_lo and x_hi > x_lo) else 0

            results[fi] = {
                "heatmap": hm,
                "blobs_05": blobs,
                "blobs_03": blobs_03,
                "blobs_01": blobs_01,
                "detection": detection,
                "gt_pixel": (gt_px, gt_py),
                "gt_model": (gt_mx, gt_my),
                "hm_at_gt": hm_at_gt,
                "hm_near_gt": hm_near_gt,
                "frame_bgr": frames_bgr[i],
                "batch_start": batch_start,
                "position_in_batch": i,
            }

    # ── Print per-frame comparison ──────────────────────────────────
    print(f"\n  {'Frame':>5} | {'GT pixel':>16} | {'Det pixel':>16} | {'Error':>7} | {'HM@GT':>6} | {'HM near':>7} | {'HM max':>6} | {'#blob05':>7} | {'#blob03':>7} | {'#blob01':>7} | {'BlobMax':>7}")
    print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*16}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    errors = []
    missed = 0

    for fi in sorted(results.keys()):
        r = results[fi]
        gt_str = f"({r['gt_pixel'][0]:7.1f},{r['gt_pixel'][1]:6.1f})"
        hm_max = float(r['heatmap'].max())

        if r['detection']:
            dx = r['detection'][0] - r['gt_pixel'][0]
            dy = r['detection'][1] - r['gt_pixel'][1]
            err = np.sqrt(dx ** 2 + dy ** 2)
            errors.append(err)
            det_str = f"({r['detection'][0]:7.1f},{r['detection'][1]:6.1f})"
            err_str = f"{err:6.1f}px"
        else:
            missed += 1
            det_str = "      MISSED      "
            err_str = "   ---"

        n05 = len(r['blobs_05'])
        n03 = len(r['blobs_03'])
        n01 = len(r['blobs_01'])
        bmax = r['blobs_05'][0][3] if r['blobs_05'] else 0

        print(f"  {fi:5d} | {gt_str} | {det_str} | {err_str} | {r['hm_at_gt']:.4f} | {r['hm_near_gt']:.5f} | {hm_max:.4f} | {n05:>7d} | {n03:>7d} | {n01:>7d} | {bmax:>7.4f}")

    # ── Summary statistics ──────────────────────────────────────────
    print(f"\n  === Summary for {cam_name} ===")
    print(f"  Total GT frames: {len(results)}")
    print(f"  Detected (thr=0.5): {len(errors)}/{len(results)} ({100*len(errors)/max(len(results),1):.0f}%)")
    print(f"  Missed: {missed}")
    if errors:
        errors = np.array(errors)
        print(f"  Pixel error: mean={errors.mean():.1f}px, median={np.median(errors):.1f}px, max={errors.max():.1f}px, std={errors.std():.1f}px")
        print(f"  Error < 10px: {np.sum(errors < 10)}/{len(errors)}")
        print(f"  Error < 30px: {np.sum(errors < 30)}/{len(errors)}")
        print(f"  Error < 50px: {np.sum(errors < 50)}/{len(errors)}")

    # ── Heatmap value analysis at GT locations ──────────────────────
    hm_at_gt_vals = [results[fi]['hm_at_gt'] for fi in sorted(results.keys())]
    hm_near_gt_vals = [results[fi]['hm_near_gt'] for fi in sorted(results.keys())]
    hm_max_vals = [float(results[fi]['heatmap'].max()) for fi in sorted(results.keys())]
    print(f"\n  Heatmap value at exact GT pixel:  mean={np.mean(hm_at_gt_vals):.4f}, max={np.max(hm_at_gt_vals):.4f}")
    print(f"  Heatmap max within 10px of GT:    mean={np.mean(hm_near_gt_vals):.4f}, max={np.max(hm_near_gt_vals):.4f}")
    print(f"  Heatmap global max:               mean={np.mean(hm_max_vals):.4f}, max={np.max(hm_max_vals):.4f}")

    # ── Save visualizations for a few key frames ────────────────────
    cam_dir = OUTPUT_DIR / cam_name
    cam_dir.mkdir(exist_ok=True)

    for fi in sorted(results.keys()):
        r = results[fi]
        hm = r['heatmap']
        frame = r['frame_bgr']

        # 1. Raw heatmap (grayscale, amplified)
        hm_vis = (hm * 255).astype(np.uint8)
        cv2.imwrite(str(cam_dir / f"f{fi:03d}_1_heatmap_raw.png"), hm_vis)

        # 2. Heatmap colormap overlay on frame
        frame_small = cv2.resize(frame, (INPUT_W, INPUT_H))
        hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_small, 0.5, hm_color, 0.5, 0)

        # Draw GT position (green cross)
        gt_mx, gt_my = int(round(r['gt_model'][0])), int(round(r['gt_model'][1]))
        cv2.drawMarker(overlay, (gt_mx, gt_my), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(overlay, "GT", (gt_mx + 8, gt_my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw detection (red circle)
        if r['detection']:
            det_mx = r['detection'][0] / (frame.shape[1] / INPUT_W)
            det_my = r['detection'][1] / (frame.shape[0] / INPUT_H)
            cv2.circle(overlay, (int(det_mx), int(det_my)), 8, (0, 0, 255), 2)
            cv2.putText(overlay, "DET", (int(det_mx) + 8, int(det_my) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Frame info text
        hm_max = float(hm.max())
        info = f"f{fi} | hmMax={hm_max:.3f} | hmAtGT={r['hm_at_gt']:.3f} | blobs@0.5={len(r['blobs_05'])} @0.3={len(r['blobs_03'])} @0.1={len(r['blobs_01'])}"
        cv2.putText(overlay, info, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        if r['detection']:
            err = np.sqrt((r['detection'][0] - r['gt_pixel'][0])**2 + (r['detection'][1] - r['gt_pixel'][1])**2)
            cv2.putText(overlay, f"err={err:.1f}px", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

        cv2.imwrite(str(cam_dir / f"f{fi:03d}_2_overlay.jpg"), overlay)

        # 3. Full-res frame with GT and detection marked
        frame_full = frame.copy()
        gtx, gty = int(round(r['gt_pixel'][0])), int(round(r['gt_pixel'][1]))
        cv2.drawMarker(frame_full, (gtx, gty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame_full, f"GT({gtx},{gty})", (gtx + 12, gty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if r['detection']:
            dx, dy = int(round(r['detection'][0])), int(round(r['detection'][1]))
            cv2.circle(frame_full, (dx, dy), 12, (0, 0, 255), 2)
            cv2.putText(frame_full, f"DET({dx},{dy})", (dx + 12, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(cam_dir / f"f{fi:03d}_3_fullres.jpg"), frame_full)

    print(f"\n  Saved visualizations → {cam_dir}/")
    return results


# ── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading TrackNet model...")
    model, device = load_tracknet(MODEL_PATH)
    print(f"Model loaded on {device}")

    all_results = {}
    for cam, vpath in VIDEOS.items():
        all_results[cam] = analyze_camera(cam, vpath, ANNOTATION_DIRS[cam], model, device)

    # ── Cross-camera comparison ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Cross-camera frame-by-frame comparison")
    print(f"{'='*70}")
    common = sorted(set(all_results["cam66"].keys()) & set(all_results["cam68"].keys()))
    print(f"  Common frames with GT: {len(common)}")

    for fi in common:
        r66 = all_results["cam66"][fi]
        r68 = all_results["cam68"][fi]
        d66 = "MISS" if not r66['detection'] else f"({r66['detection'][0]:7.1f},{r66['detection'][1]:6.1f})"
        d68 = "MISS" if not r68['detection'] else f"({r68['detection'][0]:7.1f},{r68['detection'][1]:6.1f})"
        e66 = "---" if not r66['detection'] else f"{np.sqrt((r66['detection'][0]-r66['gt_pixel'][0])**2+(r66['detection'][1]-r66['gt_pixel'][1])**2):.0f}px"
        e68 = "---" if not r68['detection'] else f"{np.sqrt((r68['detection'][0]-r68['gt_pixel'][0])**2+(r68['detection'][1]-r68['gt_pixel'][1])**2):.0f}px"
        hm66 = float(r66['heatmap'].max())
        hm68 = float(r68['heatmap'].max())
        print(f"  f{fi:03d} | cam66: {d66} err={e66:>5} hmMax={hm66:.3f} | cam68: {d68} err={e68:>5} hmMax={hm68:.3f}")

    print(f"\nDone! Check {OUTPUT_DIR}/ for heatmap visualizations.")
