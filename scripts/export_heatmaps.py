"""Export TrackNet heatmaps + detection JSON for GT frames 51-83.

Blob filtering: OSD mask + homography X-coordinate must be within court width (0~8.23m).

Saves to debug_heatmaps/{cam}/:
  - f{NN}_heatmap.png      : raw heatmap (grayscale, 0-255)
  - f{NN}_overlay.jpg       : heatmap colormap overlaid on frame (model res)
  - f{NN}_fullres.jpg       : full-res frame with GT(green) + blobs marked
  - results.json            : per-frame blob detection data at all thresholds
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_PATH = "model_weight/TrackNet_best.pt"
VIDEOS = {
    "cam68": "uploads/cam68_clip.mp4",
    "cam66": "uploads/cam66_clip.mp4",
}
ANNOTATION_DIRS = {
    "cam68": Path("uploads/cam68_clip"),
    "cam66": Path("uploads/cam66_clip"),
}
HOMOGRAPHY_PATH = "src/homography_matrices.json"
INPUT_H, INPUT_W = 288, 512
SEQ_LEN = 8
OSD_MASK = (0, 0, 620, 40)
OUTPUT_DIR = Path("debug_heatmaps")
THRESHOLDS = [0.5, 0.3, 0.2, 0.1]

# Court X range (meters)
COURT_X_MIN = -1.0   # small margin
COURT_X_MAX = 9.23   # 8.23 + small margin


def load_homography(cam_name):
    """Load H_image_to_world for a camera."""
    with open(HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    H = np.array(data[cam_name]["H_image_to_world"], dtype=np.float64)
    return H


def pixel_to_world_x(H, px, py):
    """Project pixel (px, py) through homography, return world X only."""
    pt = np.array([px, py, 1.0])
    result = H @ pt
    if abs(result[2]) < 1e-10:
        return None
    return float(result[0] / result[2])


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


def apply_osd_mask(hm, orig_w, orig_h):
    hm = hm.copy()
    hm_h, hm_w = hm.shape
    sx, sy = hm_w / orig_w, hm_h / orig_h
    x0, y0, x1, y1 = OSD_MASK
    hm[int(y0 * sy):int(y1 * sy + 0.5), int(x0 * sx):int(x1 * sx + 0.5)] = 0.0
    return hm


def extract_blobs(heatmap, threshold):
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
        blobs.append({
            "cx_model": round(cx, 2), "cy_model": round(cy, 2),
            "blob_sum": round(blob_sum, 3), "blob_max": round(blob_max, 4),
            "blob_area": blob_area,
        })
    blobs.sort(key=lambda b: b["blob_sum"], reverse=True)
    return blobs


def filter_blobs_by_court_x(blobs, H, scale_x, scale_y):
    """Keep only blobs whose world X falls within court range."""
    kept = []
    for b in blobs:
        px = b["cx_model"] * scale_x
        py = b["cy_model"] * scale_y
        wx = pixel_to_world_x(H, px, py)
        if wx is not None and COURT_X_MIN <= wx <= COURT_X_MAX:
            b["world_x"] = round(wx, 2)
            kept.append(b)
        else:
            b["world_x"] = round(wx, 2) if wx is not None else None
            b["filtered_out"] = True
    return kept, blobs  # kept, all_with_annotations


def process_camera(cam_name, video_path, ann_dir, model, device):
    print(f"\n=== {cam_name} ===")
    gt = load_gt(ann_dir)
    print(f"GT: {len(gt)} frames ({min(gt)}-{max(gt)})")

    H = load_homography(cam_name)
    print(f"Homography loaded, court X filter: [{COURT_X_MIN}, {COURT_X_MAX}]m")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {orig_w}x{orig_h}")

    bg = compute_median(cap)
    scale_x = orig_w / INPUT_W
    scale_y = orig_h / INPUT_H

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames = {}
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        all_frames[i] = frame
    cap.release()

    out_dir = OUTPUT_DIR / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for fi in sorted(gt.keys()):
        gt_px, gt_py = gt[fi]
        gt_mx = gt_px / scale_x
        gt_my = gt_py / scale_y

        batch_start = max(0, fi - SEQ_LEN + 1)
        batch_indices = list(range(batch_start, batch_start + SEQ_LEN))
        if not all(bi in all_frames for bi in batch_indices):
            continue

        frames_bgr = [all_frames[bi] for bi in batch_indices]
        processed = [preprocess(f) for f in frames_bgr]
        stacked = np.concatenate([bg] + processed, axis=0)
        inp = torch.from_numpy(stacked[np.newaxis]).to(device)

        with torch.no_grad():
            heatmaps = model(inp)[0].cpu().numpy()

        hm_idx = fi - batch_start
        hm_raw = heatmaps[hm_idx]
        hm = apply_osd_mask(hm_raw, orig_w, orig_h)

        # GT world X
        gt_world_x = pixel_to_world_x(H, gt_px, gt_py)

        # ── Save raw heatmap ────────────────────────────────────
        hm_vis = (hm * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"f{fi:02d}_heatmap.png"), hm_vis)

        # ── Save overlay at model resolution ────────────────────
        frame_small = cv2.resize(all_frames[fi], (INPUT_W, INPUT_H))
        hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame_small, 0.5, hm_color, 0.5, 0)

        gx, gy = int(round(gt_mx)), int(round(gt_my))
        cv2.drawMarker(overlay, (gx, gy), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(overlay, "GT", (gx + 8, gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Top-1 blob at 0.5 after court-X filtering (red circle)
        blobs_05_all = extract_blobs(hm, 0.5)
        blobs_05_kept, _ = filter_blobs_by_court_x(
            [b.copy() for b in blobs_05_all], H, scale_x, scale_y)
        if blobs_05_kept:
            b = blobs_05_kept[0]
            bx, by = int(round(b["cx_model"])), int(round(b["cy_model"]))
            cv2.circle(overlay, (bx, by), 8, (0, 0, 255), 2)
            cv2.putText(overlay, "SEL", (bx + 8, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        gt_my_i = int(np.clip(round(gt_my), 0, INPUT_H - 1))
        gt_mx_i = int(np.clip(round(gt_mx), 0, INPUT_W - 1))
        hm_at_gt = float(hm[gt_my_i, gt_mx_i])
        info = f"f{fi} hmMax={float(hm.max()):.3f} hmAtGT={hm_at_gt:.3f} GT_wX={gt_world_x:.1f}m"
        cv2.putText(overlay, info, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"f{fi:02d}_overlay.jpg"), overlay)

        # ── Save full-res with all blobs ────────────────────────
        frame_full = all_frames[fi].copy()
        gtx_i, gty_i = int(round(gt_px)), int(round(gt_py))
        cv2.drawMarker(frame_full, (gtx_i, gty_i), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
        cv2.putText(frame_full, f"GT({gtx_i},{gty_i}) wX={gt_world_x:.1f}m",
                    (gtx_i + 15, gty_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw blobs: red=@0.5 kept, gray=@0.5 filtered, yellow=@0.3 kept
        for t, color_kept, color_reject, radius, thick in [
            (0.5, (0, 0, 255), (128, 128, 128), 15, 2),
            (0.3, (0, 200, 255), (80, 80, 80), 10, 1),
        ]:
            blobs_all = extract_blobs(hm, t)
            blobs_kept_list, _ = filter_blobs_by_court_x(
                [b.copy() for b in blobs_all], H, scale_x, scale_y)
            kept_set = set()
            for b in blobs_kept_list:
                kept_set.add((b["cx_model"], b["cy_model"]))

            for idx, b in enumerate(blobs_all):
                bx_o = int(round(b["cx_model"] * scale_x))
                by_o = int(round(b["cy_model"] * scale_y))
                wx = pixel_to_world_x(H, bx_o, by_o)
                is_kept = (b["cx_model"], b["cy_model"]) in kept_set
                color = color_kept if is_kept else color_reject
                cv2.circle(frame_full, (bx_o, by_o), radius, color, thick)
                wx_str = f"{wx:.1f}" if wx is not None else "?"
                status = "" if is_kept else "[OUT]"
                label = f"#{idx}@{t} s={b['blob_sum']:.1f} wX={wx_str}{status}"
                cv2.putText(frame_full, label, (bx_o + 18, by_o + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(out_dir / f"f{fi:02d}_fullres.jpg"), frame_full)

        # ── JSON data ───────────────────────────────────────────
        r = 10
        y0 = max(0, gt_my_i - r)
        y1 = min(INPUT_H, gt_my_i + r + 1)
        x0 = max(0, gt_mx_i - r)
        x1 = min(INPUT_W, gt_mx_i + r + 1)

        frame_data = {
            "frame": fi,
            "gt_pixel": {"x": round(gt_px, 1), "y": round(gt_py, 1)},
            "gt_model": {"x": round(gt_mx, 2), "y": round(gt_my, 2)},
            "gt_world_x": round(gt_world_x, 2) if gt_world_x is not None else None,
            "heatmap_at_gt": round(hm_at_gt, 4),
            "heatmap_near_gt_max": round(float(hm[y0:y1, x0:x1].max()), 4),
            "heatmap_global_max": round(float(hm.max()), 4),
            "blobs": {},
        }
        for t in THRESHOLDS:
            blobs_all = extract_blobs(hm, t)
            blobs_kept, _ = filter_blobs_by_court_x(
                [b.copy() for b in blobs_all], H, scale_x, scale_y)

            frame_data["blobs"][str(t)] = {
                "all": [],
                "after_court_x_filter": [],
            }
            for b in blobs_all:
                bx_o = b["cx_model"] * scale_x
                by_o = b["cy_model"] * scale_y
                wx = pixel_to_world_x(H, bx_o, by_o)
                frame_data["blobs"][str(t)]["all"].append({
                    "cx_orig": round(bx_o, 1),
                    "cy_orig": round(by_o, 1),
                    "blob_sum": b["blob_sum"],
                    "blob_max": b["blob_max"],
                    "blob_area": b["blob_area"],
                    "world_x": round(wx, 2) if wx is not None else None,
                    "dist_to_gt_px": round(float(np.sqrt(
                        (bx_o - gt_px) ** 2 + (by_o - gt_py) ** 2
                    )), 1),
                })
            for b in blobs_kept:
                bx_o = b["cx_model"] * scale_x
                by_o = b["cy_model"] * scale_y
                frame_data["blobs"][str(t)]["after_court_x_filter"].append({
                    "cx_orig": round(bx_o, 1),
                    "cy_orig": round(by_o, 1),
                    "blob_sum": b["blob_sum"],
                    "blob_max": b["blob_max"],
                    "world_x": b.get("world_x"),
                    "dist_to_gt_px": round(float(np.sqrt(
                        (bx_o - gt_px) ** 2 + (by_o - gt_py) ** 2
                    )), 1),
                })

        all_results[str(fi)] = frame_data

        # Print summary line
        n_all = len(blobs_05_all)
        n_kept = len(blobs_05_kept)
        sel_str = "---"
        if blobs_05_kept:
            sel = blobs_05_kept[0]
            sel_x = sel["cx_model"] * scale_x
            sel_y = sel["cy_model"] * scale_y
            sel_dist = np.sqrt((sel_x - gt_px) ** 2 + (sel_y - gt_py) ** 2)
            sel_str = f"{sel_dist:.0f}px wX={sel.get('world_x', '?')}m"
        print(f"  f{fi:02d}: hmMax={float(hm.max()):.3f} GT_wX={gt_world_x:.1f}m "
              f"blobs@0.5={n_all}→{n_kept}(filtered) sel→GT: {sel_str}")

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_dir}/")


if __name__ == "__main__":
    print("Loading TrackNet...")
    model, device = load_tracknet(MODEL_PATH)
    print(f"Model on {device}")

    for cam in ["cam68", "cam66"]:
        process_camera(cam, VIDEOS[cam], ANNOTATION_DIRS[cam], model, device)

    print("\nDone!")
