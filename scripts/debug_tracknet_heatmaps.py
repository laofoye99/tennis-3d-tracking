"""Debug script: run TrackNet on cam68_clip.mp4, save heatmap visualizations.

Outputs saved to debug_heatmaps/ directory:
  - raw heatmap (grayscale)
  - colormap overlay on original frame
  - thresholded binary + blob detection results
  - multi-blob analysis (all blobs, not just the "best" one)
"""

import os
import sys
import cv2
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from app.pipeline.tracknet import TrackNet

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = "model_weight/TrackNet_best.pt"
VIDEO_PATH = "uploads/cam66_clip.mp4"
OUTPUT_DIR = "debug_heatmaps_cam66_clip"
INPUT_H, INPUT_W = 288, 512
SEQ_LEN = 8
BG_MODE = "concat"
THRESHOLD = 0.5

# Frame range to analyze
START_FRAME = 0
END_FRAME = 200
NUM_BATCHES = (END_FRAME - START_FRAME + SEQ_LEN - 1) // SEQ_LEN  # ceil

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load model ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

in_dim = (SEQ_LEN + 1) * 3  # 27 for bg_mode=concat
model = TrackNet(in_dim=in_dim, out_dim=SEQ_LEN)
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(device)
print(f"Model loaded: {MODEL_PATH}")


# ── Open video ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {vid_w}x{vid_h} @ {fps:.1f} fps, {total_frames} frames")


# ── Compute background median ──────────────────────────────────────────
print("Computing background median...")
n_samples = min(200, total_frames)
sample_indices = set(int(i * total_frames / n_samples) for i in range(n_samples))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
sampled = []
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break
    if i in sample_indices:
        small = cv2.resize(frame, (INPUT_W, INPUT_H))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        sampled.append(small_rgb)
        if len(sampled) >= n_samples:
            break

median = np.median(sampled, axis=0).astype(np.uint8)
bg_frame = median.astype(np.float32).transpose(2, 0, 1) / 255.0  # (3, H, W)
print(f"Background median computed from {len(sampled)} frames")

# Save background median visualization
bg_vis = cv2.cvtColor(median, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(OUTPUT_DIR, "background_median.jpg"), bg_vis)


# ── Preprocess helper ───────────────────────────────────────────────────
def preprocess(frame_bgr):
    """BGR → resize → RGB → CHW float32 [0,1]"""
    img = cv2.resize(frame_bgr, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


# ── Run inference and save heatmaps ─────────────────────────────────────
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

for batch_idx in range(NUM_BATCHES):
    # Read 8 frames
    raw_frames = []
    processed = []
    for _ in range(SEQ_LEN):
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        processed.append(preprocess(frame))

    if len(processed) < SEQ_LEN:
        print(f"Not enough frames at batch {batch_idx}")
        break

    # Build input: bg + 8 frames → (1, 27, H, W)
    all_channels = [bg_frame] + processed
    stacked = np.concatenate(all_channels, axis=0)
    input_tensor = torch.from_numpy(stacked[np.newaxis]).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)  # (1, 8, H, W) — sigmoid included
        heatmaps = output[0].cpu().numpy()  # (8, H, W)

    batch_start_frame = START_FRAME + batch_idx * SEQ_LEN
    print(f"\n{'='*70}")
    print(f"Batch {batch_idx}: frames {batch_start_frame}-{batch_start_frame + SEQ_LEN - 1}")
    print(f"Heatmap shape: {heatmaps.shape}, dtype: {heatmaps.dtype}")
    print(f"Heatmap range: [{heatmaps.min():.6f}, {heatmaps.max():.6f}]")

    for i in range(SEQ_LEN):
        hm = heatmaps[i]  # (H, W)
        frame_idx = batch_start_frame + i
        raw = raw_frames[i]

        print(f"\n  Frame {frame_idx}:")
        print(f"    heatmap max={hm.max():.6f}, mean={hm.mean():.6f}, "
              f"nonzero(>0.01)={np.sum(hm > 0.01)}, nonzero(>0.1)={np.sum(hm > 0.1)}, "
              f"nonzero(>0.5)={np.sum(hm > 0.5)}")

        # ── 1. Raw heatmap (grayscale) ──
        hm_vis = (hm * 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"b{batch_idx}_f{frame_idx}_1_raw_heatmap.png"),
            hm_vis,
        )

        # ── 2. Heatmap colormap overlay on original ──
        hm_color = cv2.applyColorMap(hm_vis, cv2.COLORMAP_JET)
        hm_color_resized = cv2.resize(hm_color, (vid_w, vid_h))
        overlay = cv2.addWeighted(raw, 0.6, hm_color_resized, 0.4, 0)
        # Scale down for reasonable file size
        overlay_small = cv2.resize(overlay, (960, int(vid_h * 960 / vid_w)))
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"b{batch_idx}_f{frame_idx}_2_overlay.jpg"),
            overlay_small,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        # ── 3. Multi-threshold analysis ──
        for thresh in [0.3, 0.5, 0.7]:
            filtered = np.where(hm > thresh, hm, 0.0).astype(np.float32)
            binary = (filtered > 0).astype(np.uint8)
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            scale_x = vid_w / INPUT_W
            scale_y = vid_h / INPUT_H

            blobs = []
            for j in range(1, num_labels):
                mask = labels_im == j
                blob_sum = float(filtered[mask].sum())
                blob_max = float(hm[mask].max())
                blob_area = int(stats[j, cv2.CC_STAT_AREA])
                # Weighted centroid
                if blob_sum > 0:
                    cx = float(np.sum(np.where(mask)[1] * filtered[mask]) / blob_sum)
                    cy = float(np.sum(np.where(mask)[0] * filtered[mask]) / blob_sum)
                else:
                    cx, cy = centroids[j]
                # Scale to original coordinates
                cx_orig = cx * scale_x
                cy_orig = cy * scale_y
                blobs.append({
                    "cx": cx_orig, "cy": cy_orig,
                    "sum": blob_sum, "max": blob_max,
                    "area": blob_area,
                    "cx_model": cx, "cy_model": cy,
                })

            if blobs:
                print(f"    thresh={thresh:.1f}: {len(blobs)} blob(s)")
                for b in sorted(blobs, key=lambda x: -x["sum"]):
                    print(f"      pos=({b['cx']:.1f}, {b['cy']:.1f}) "
                          f"sum={b['sum']:.2f} max={b['max']:.4f} area={b['area']}px")

        # ── 4. Binary blob visualization (threshold=0.5) ──
        filtered_05 = np.where(hm > 0.5, hm, 0.0).astype(np.float32)
        binary_05 = (filtered_05 > 0).astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            binary_05, connectivity=8
        )
        # Draw on resized original frame
        draw = cv2.resize(raw.copy(), (INPUT_W, INPUT_H))
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        for j in range(1, num_labels):
            c = colors[(j - 1) % len(colors)]
            mask = labels_im == j
            blob_sum = float(filtered_05[mask].sum())
            if blob_sum > 0:
                cx = float(np.sum(np.where(mask)[1] * filtered_05[mask]) / blob_sum)
                cy = float(np.sum(np.where(mask)[0] * filtered_05[mask]) / blob_sum)
            else:
                cx, cy = centroids[j]
            cv2.circle(draw, (int(cx), int(cy)), 8, c, 2)
            cv2.putText(draw, f"#{j} sum={blob_sum:.1f}", (int(cx)+10, int(cy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        # Also show the heatmap as semi-transparent overlay
        hm_alpha = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        draw = cv2.addWeighted(draw, 0.7, hm_alpha, 0.3, 0)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"b{batch_idx}_f{frame_idx}_3_blobs.jpg"),
            draw,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        # ── 5. Save raw heatmap as numpy for further analysis ──
        if batch_idx == 0:  # Only save numpy for first batch
            np.save(
                os.path.join(OUTPUT_DIR, f"b{batch_idx}_f{frame_idx}_heatmap.npy"),
                hm,
            )

cap.release()

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"Done! Saved outputs to {OUTPUT_DIR}/")
print(f"\nFile naming convention:")
print(f"  b{{batch}}_f{{frame}}_1_raw_heatmap.png  — grayscale heatmap")
print(f"  b{{batch}}_f{{frame}}_2_overlay.jpg      — colormap overlay on video frame")
print(f"  b{{batch}}_f{{frame}}_3_blobs.jpg        — blob detection + heatmap overlay")
print(f"  b{{batch}}_f{{frame}}_heatmap.npy        — raw numpy array (batch 0 only)")
