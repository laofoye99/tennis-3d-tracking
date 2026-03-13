"""Diagnose: does TrackNet heatmap detect the ball even when blob selection fails?

Uses LabelImg ground-truth annotations (cam68_clip/, cam66_clip/) for frames 51-83.
For each GT frame, checks:
  1. Heatmap value at GT location (exact pixel + nearby max)
  2. All blobs at multiple thresholds
  3. Whether the CLOSEST blob to GT matches vs the SELECTED blob (by blob_sum)
  4. Categorizes: correct_select / wrong_select / no_blob_but_heatmap / no_heatmap

Usage:
    python diagnose_heatmap_vs_blob.py
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
    "cam68": "uploads/cam68_clip.mp4",
    "cam66": "uploads/cam66_clip.mp4",
}
ANNOTATION_DIRS = {
    "cam68": Path("uploads/cam68_clip"),
    "cam66": Path("uploads/cam66_clip"),
}
INPUT_H, INPUT_W = 288, 512
SEQ_LEN = 8
OSD_MASK = (0, 0, 620, 40)  # original coords
CLOSE_PX = 30  # "correct" if blob within 30 original pixels of GT
THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05]


# ── Load GT (LabelImg format) ──────────────────────────────────────
def load_gt(ann_dir: Path) -> dict[int, tuple[float, float]]:
    """Load LabelImg annotations -> {frame_index: (pixel_x, pixel_y)}."""
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


# ── Model ───────────────────────────────────────────────────────────
def load_tracknet(model_path: str, device: str = "cuda"):
    from app.pipeline.tracknet import TrackNet
    in_dim = (SEQ_LEN + 1) * 3
    model = TrackNet(in_dim=in_dim, out_dim=SEQ_LEN)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)
    return model, dev


def preprocess(frame: np.ndarray) -> np.ndarray:
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


def apply_osd_mask(heatmap: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    hm = heatmap.copy()
    hm_h, hm_w = hm.shape
    sx, sy = hm_w / orig_w, hm_h / orig_h
    x0, y0, x1, y1 = OSD_MASK
    hm[int(y0 * sy):int(y1 * sy + 0.5), int(x0 * sx):int(x1 * sx + 0.5)] = 0.0
    return hm


def extract_blobs(heatmap: np.ndarray, threshold: float):
    """Returns list of (cx_model, cy_model, blob_sum, blob_max, blob_area)."""
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


# ── Analyze one camera ──────────────────────────────────────────────
def analyze_camera(cam_name: str, video_path: str, ann_dir: Path, model, device):
    print(f"\n{'=' * 80}")
    print(f"  {cam_name}: {video_path}  |  GT: {ann_dir}")
    print(f"{'=' * 80}")

    gt = load_gt(ann_dir)
    if not gt:
        print("  No GT annotations found!")
        return
    print(f"  GT: {len(gt)} frames ({min(gt)}-{max(gt)})")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {total_frames} frames, {orig_w}x{orig_h}")

    bg = compute_median(cap)
    scale_x = orig_w / INPUT_W
    scale_y = orig_h / INPUT_H

    # Read all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = {}
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames[i] = frame
    cap.release()

    gt_frames = sorted(gt.keys())

    categories = {
        "correct_top1": [],
        "wrong_top1": [],
        "no_blob_hm_yes": [],
        "no_heatmap": [],
    }

    # Header
    print(f"\n{'Frm':>4} | {'GT pixel':>18} | {'HM@GT':>6} | {'HMnr':>6} | {'HMmx':>6} |", end="")
    for t in THRESHOLDS:
        print(f"@{t:<4}|", end="")
    print(f" {'Category':>14} | {'Top1→GT':>7} | Detail")
    print("-" * 170)

    for fi in gt_frames:
        gt_px, gt_py = gt[fi]
        gt_mx = gt_px / scale_x
        gt_my = gt_py / scale_y

        # Build batch: 8 frames ending at fi
        batch_start = max(0, fi - SEQ_LEN + 1)
        batch_indices = list(range(batch_start, batch_start + SEQ_LEN))
        if not all(bi in all_frames for bi in batch_indices):
            continue

        frames_bgr = [all_frames[bi] for bi in batch_indices]
        processed = [preprocess(f) for f in frames_bgr]
        stacked = np.concatenate([bg] + processed, axis=0)
        inp = torch.from_numpy(stacked[np.newaxis]).to(device)

        with torch.no_grad():
            output = model(inp)
            heatmaps = output[0].cpu().numpy()

        hm_idx = fi - batch_start
        hm_raw = heatmaps[hm_idx]
        hm = apply_osd_mask(hm_raw, orig_w, orig_h)

        # Heatmap at GT
        gt_mx_i = int(np.clip(round(gt_mx), 0, INPUT_W - 1))
        gt_my_i = int(np.clip(round(gt_my), 0, INPUT_H - 1))
        hm_at_gt = float(hm[gt_my_i, gt_mx_i])

        # Heatmap max in 10px model-coord radius
        r = 10
        y0 = max(0, gt_my_i - r)
        y1 = min(INPUT_H, gt_my_i + r + 1)
        x0 = max(0, gt_mx_i - r)
        x1 = min(INPUT_W, gt_mx_i + r + 1)
        hm_near = float(hm[y0:y1, x0:x1].max())
        hm_max = float(hm.max())

        # Extract blobs at all thresholds
        blobs_by_thresh = {}
        for t in THRESHOLDS:
            blobs_by_thresh[t] = extract_blobs(hm, t)

        # Find if any blob at any threshold is near GT
        found_any_near = False
        best_near_thresh = None
        for t in THRESHOLDS:
            for b in blobs_by_thresh[t]:
                bx = b[0] * scale_x
                by = b[1] * scale_y
                dist = np.sqrt((bx - gt_px) ** 2 + (by - gt_py) ** 2)
                if dist <= CLOSE_PX:
                    found_any_near = True
                    if best_near_thresh is None:
                        best_near_thresh = t
                    break

        blobs_05 = blobs_by_thresh[0.5]
        top1_dist_str = "   ---"

        if blobs_05:
            top1 = blobs_05[0]
            top1_x = top1[0] * scale_x
            top1_y = top1[1] * scale_y
            top1_dist = np.sqrt((top1_x - gt_px) ** 2 + (top1_y - gt_py) ** 2)
            top1_dist_str = f"{top1_dist:5.0f}px"

            # Find closest blob to GT at threshold=0.5
            closest_dist = float('inf')
            closest_idx = 0
            for idx, b in enumerate(blobs_05):
                bx = b[0] * scale_x
                by = b[1] * scale_y
                d = np.sqrt((bx - gt_px) ** 2 + (by - gt_py) ** 2)
                if d < closest_dist:
                    closest_dist = d
                    closest_idx = idx

            if top1_dist <= CLOSE_PX:
                categories["correct_top1"].append(fi)
                cat = "CORRECT"
                detail = f"top1=({top1_x:.0f},{top1_y:.0f})"
            elif closest_dist <= CLOSE_PX:
                categories["wrong_top1"].append(fi)
                cat = "WRONG_SELECT"
                cb = blobs_05[closest_idx]
                detail = (f"top1=({top1_x:.0f},{top1_y:.0f}) sum={top1[2]:.1f} | "
                          f"correct=blob#{closest_idx} ({cb[0]*scale_x:.0f},{cb[1]*scale_y:.0f}) "
                          f"sum={cb[2]:.1f} dist={closest_dist:.0f}px")
            else:
                if found_any_near:
                    categories["wrong_top1"].append(fi)
                    cat = "WRONG+LOW"
                    detail = f"top1=({top1_x:.0f},{top1_y:.0f}), GT blob needs thresh={best_near_thresh}"
                elif hm_near > 0.05:
                    categories["no_blob_hm_yes"].append(fi)
                    cat = "NO_NEAR_BLOB"
                    detail = f"hm_near={hm_near:.3f}"
                else:
                    categories["no_heatmap"].append(fi)
                    cat = "NO_HEATMAP"
                    detail = f"hm_near={hm_near:.4f}"
        else:
            if found_any_near:
                categories["no_blob_hm_yes"].append(fi)
                cat = "FAINT_BLOB"
                detail = f"GT blob appears at thresh={best_near_thresh}"
            elif hm_near > 0.05:
                categories["no_blob_hm_yes"].append(fi)
                cat = "HM_SIGNAL"
                detail = f"hm_near={hm_near:.3f}"
            else:
                categories["no_heatmap"].append(fi)
                cat = "NO_HEATMAP"
                detail = f"hm_near={hm_near:.4f}"

        gt_str = f"({gt_px:7.1f},{gt_py:7.1f})"
        print(f"{fi:4d} | {gt_str} | {hm_at_gt:.4f} | {hm_near:.4f} | {hm_max:.4f} |", end="")
        for t in THRESHOLDS:
            n = len(blobs_by_thresh[t])
            print(f" {n:>3} |", end="")
        print(f" {cat:>14} | {top1_dist_str} | {detail}")

    # ── Summary ─────────────────────────────────────────────────────
    total = len(gt_frames)
    print(f"\n{'=' * 80}")
    print(f"SUMMARY  {cam_name}  ({total} GT frames, match_radius={CLOSE_PX}px)")
    print(f"{'=' * 80}")
    n_correct = len(categories['correct_top1'])
    n_wrong = len(categories['wrong_top1'])
    n_faint = len(categories['no_blob_hm_yes'])
    n_none = len(categories['no_heatmap'])
    print(f"  CORRECT    (top-1 blob near GT):     {n_correct:>3} / {total}  ({100*n_correct/total:.1f}%)")
    print(f"  WRONG_SEL  (GT blob exists, wrong pick): {n_wrong:>3} / {total}  ({100*n_wrong/total:.1f}%)")
    print(f"  FAINT/SIG  (heatmap signal, no blob@0.5): {n_faint:>3} / {total}  ({100*n_faint/total:.1f}%)")
    print(f"  NO_HEATMAP (no signal at GT at all):  {n_none:>3} / {total}  ({100*n_none/total:.1f}%)")
    print()
    print(f"  Heatmap detection rate (有信号): {n_correct + n_wrong + n_faint}/{total} = {100*(n_correct+n_wrong+n_faint)/total:.1f}%")
    print(f"  Blob selection accuracy (选对):  {n_correct}/{n_correct+n_wrong} = {100*n_correct/max(1,n_correct+n_wrong):.1f}% (of frames with blob@0.5)")

    if categories["wrong_top1"]:
        print(f"\n  WRONG_SELECT frames: {categories['wrong_top1']}")
    if categories["no_blob_hm_yes"]:
        print(f"  FAINT/SIGNAL frames: {categories['no_blob_hm_yes']}")
    if categories["no_heatmap"]:
        print(f"  NO_HEATMAP   frames: {categories['no_heatmap']}")

    return categories


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading TrackNet model...")
    model, device = load_tracknet(MODEL_PATH)
    print(f"Model on {device}")

    for cam in ["cam68", "cam66"]:
        analyze_camera(cam, VIDEOS[cam], ANNOTATION_DIRS[cam], model, device)

    print("\nDone!")
