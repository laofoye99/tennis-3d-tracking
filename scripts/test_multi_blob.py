"""Test multi-blob matching with dual-camera TrackNet inference.

Processes cam66 + cam68 videos, extracts multi-blob candidates per frame,
runs cross-camera matching via ray_distance, and compares with top-1 baseline.
"""

import os
import sys
import json
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from app.pipeline.tracknet import TrackNet
from app.pipeline.postprocess import BallTracker
from app.pipeline.homography import HomographyTransformer
from app.pipeline.multi_blob_matcher import MultiBlobMatcher, _triangulate_with_distance

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = "model_weight/TrackNet_best.pt"
CAM66_VIDEO = "uploads/cam66_20260307_173403_2min.mp4"
CAM68_VIDEO = "uploads/cam68_20260307_173403_2min.mp4"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
INPUT_H, INPUT_W = 288, 512
SEQ_LEN = 8
THRESHOLD = 0.5

START_FRAME = 0
END_FRAME = 300

# Camera positions from config
CAM66_POS = [3.7657, -15.7273, 7.4776]
CAM68_POS = [5.8058, 45.0351, 10.2461]


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = (SEQ_LEN + 1) * 3
    model = TrackNet(in_dim=in_dim, out_dim=SEQ_LEN)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model, device


def compute_bg_median(cap, start_frame, end_frame):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_samples = min(200, total)
    sample_indices = set(int(i * total / n_samples) for i in range(n_samples))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sampled = []
    for i in range(total):
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
    bg = median.astype(np.float32).transpose(2, 0, 1) / 255.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    return bg


def preprocess(frame_bgr):
    img = cv2.resize(frame_bgr, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32).transpose(2, 0, 1) / 255.0


def run_inference(model, device, cap, bg, start_frame, end_frame, tracker, homography):
    """Run TrackNet inference and extract multi-blob candidates per frame."""
    total_frames = end_frame - start_frame
    num_batches = (total_frames + SEQ_LEN - 1) // SEQ_LEN
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_detections = {}  # frame_index -> detection dict

    for batch_idx in range(num_batches):
        raw_frames = []
        processed = []
        for _ in range(SEQ_LEN):
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            # Mask timestamp overlay
            masked = frame.copy()
            masked[0:41, 0:603] = 0
            processed.append(preprocess(masked))

        if len(processed) < SEQ_LEN:
            break

        all_channels = [bg] + processed
        stacked = np.concatenate(all_channels, axis=0)
        input_tensor = torch.from_numpy(stacked[np.newaxis]).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            heatmaps = output[0].cpu().numpy()

        batch_start = start_frame + batch_idx * SEQ_LEN
        for i in range(SEQ_LEN):
            fi = batch_start + i
            blobs = tracker.process_heatmap_multi(heatmaps[i])
            if not blobs:
                continue

            candidates = []
            for blob in blobs:
                wx, wy = homography.pixel_to_world(blob["pixel_x"], blob["pixel_y"])
                candidates.append({
                    "pixel_x": blob["pixel_x"],
                    "pixel_y": blob["pixel_y"],
                    "world_x": wx,
                    "world_y": wy,
                    "blob_sum": blob["blob_sum"],
                    "blob_max": blob["blob_max"],
                    "blob_area": blob["blob_area"],
                })

            top = candidates[0]
            all_detections[fi] = {
                "frame_index": fi,
                "candidates": candidates,
                # top-1 for baseline comparison
                "pixel_x": top["pixel_x"],
                "pixel_y": top["pixel_y"],
                "x": top["world_x"],
                "y": top["world_y"],
                "confidence": top["blob_sum"],
            }

    return all_detections


def main():
    print("Loading model...")
    model, device = load_model()

    print("Processing cam66...")
    cap66 = cv2.VideoCapture(CAM66_VIDEO)
    vid_w66 = int(cap66.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h66 = int(cap66.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg66 = compute_bg_median(cap66, START_FRAME, END_FRAME)
    tracker66 = BallTracker(original_size=(vid_w66, vid_h66), threshold=THRESHOLD)
    homo66 = HomographyTransformer(HOMOGRAPHY_PATH, "cam66")
    dets66 = run_inference(model, device, cap66, bg66, START_FRAME, END_FRAME, tracker66, homo66)
    cap66.release()
    print(f"  cam66: {len(dets66)} frames with detections")

    print("Processing cam68...")
    cap68 = cv2.VideoCapture(CAM68_VIDEO)
    vid_w68 = int(cap68.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h68 = int(cap68.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg68 = compute_bg_median(cap68, START_FRAME, END_FRAME)
    tracker68 = BallTracker(original_size=(vid_w68, vid_h68), threshold=THRESHOLD)
    homo68 = HomographyTransformer(HOMOGRAPHY_PATH, "cam68")
    dets68 = run_inference(model, device, cap68, bg68, START_FRAME, END_FRAME, tracker68, homo68)
    cap68.release()
    print(f"  cam68: {len(dets68)} frames with detections")

    # Find common frames
    common = sorted(set(dets66.keys()) & set(dets68.keys()))
    print(f"\nCommon frames: {len(common)}")

    # ── Multi-blob matching ──
    matcher = MultiBlobMatcher(CAM66_POS, CAM68_POS)
    multi_results = []
    baseline_results = []

    multi_blob_frames = 0  # frames where either camera had >1 blob
    different_pick = 0  # frames where multi-blob picked different from top-1

    for fi in common:
        d1 = dets66[fi]
        d2 = dets68[fi]

        n_cands1 = len(d1["candidates"])
        n_cands2 = len(d2["candidates"])
        if n_cands1 > 1 or n_cands2 > 1:
            multi_blob_frames += 1

        # Multi-blob match
        match = matcher.match(d1, d2)
        if match:
            multi_results.append(match)

        # Baseline: top-1 from each camera
        x, y, z = _triangulate_with_distance(
            (d1["x"], d1["y"]),
            (d2["x"], d2["y"]),
            CAM66_POS, CAM68_POS,
        )[:3]
        baseline_results.append({
            "frame_index": fi,
            "x": x, "y": y, "z": z,
        })

        # Check if multi-blob picked differently
        if match and (match["cam1_idx"] != 0 or match["cam2_idx"] != 0):
            different_pick += 1

    # ── Statistics ──
    print(f"\n{'='*60}")
    print("MULTI-BLOB MATCHING RESULTS")
    print(f"{'='*60}")
    print(f"Frames with >1 blob (either camera): {multi_blob_frames}/{len(common)} "
          f"({100*multi_blob_frames/len(common):.1f}%)")
    print(f"Matched frames (valid ray_distance): {len(multi_results)}/{len(common)}")
    print(f"Non-top-1 picks: {different_pick} "
          f"({100*different_pick/max(len(multi_results),1):.1f}%)")

    # Ray distance comparison
    if multi_results:
        ray_dists = [r["ray_distance"] for r in multi_results]
        print(f"\nRay distance (multi-blob):")
        print(f"  mean={np.mean(ray_dists):.4f}, median={np.median(ray_dists):.4f}")
        print(f"  min={np.min(ray_dists):.4f}, max={np.max(ray_dists):.4f}")

    # Baseline ray distances
    baseline_ray_dists = []
    for fi in common:
        d1 = dets66[fi]
        d2 = dets68[fi]
        _, _, _, rd = _triangulate_with_distance(
            (d1["x"], d1["y"]),
            (d2["x"], d2["y"]),
            CAM66_POS, CAM68_POS,
        )
        baseline_ray_dists.append(rd)

    if baseline_ray_dists:
        print(f"\nRay distance (baseline top-1):")
        print(f"  mean={np.mean(baseline_ray_dists):.4f}, median={np.median(baseline_ray_dists):.4f}")
        print(f"  min={np.min(baseline_ray_dists):.4f}, max={np.max(baseline_ray_dists):.4f}")

    # Show frames where the pick changed
    print(f"\n{'='*60}")
    print("FRAMES WHERE MULTI-BLOB PICKED DIFFERENTLY FROM TOP-1:")
    print(f"{'='*60}")
    for fi in common:
        d1 = dets66[fi]
        d2 = dets68[fi]
        match = matcher.match(d1, d2) if False else None  # Already processed above

    # Re-run matcher for detailed per-frame output (first 10 different picks)
    matcher2 = MultiBlobMatcher(CAM66_POS, CAM68_POS)
    shown = 0
    for fi in common:
        d1 = dets66[fi]
        d2 = dets68[fi]
        match = matcher2.match(d1, d2)
        if match and (match["cam1_idx"] != 0 or match["cam2_idx"] != 0):
            shown += 1
            _, _, _, baseline_rd = _triangulate_with_distance(
                (d1["x"], d1["y"]), (d2["x"], d2["y"]),
                CAM66_POS, CAM68_POS,
            )
            print(f"\nFrame {fi}:")
            print(f"  cam66 candidates: {len(d1['candidates'])}, cam68 candidates: {len(d2['candidates'])}")
            print(f"  Multi-blob pick: cam1_idx={match['cam1_idx']}, cam2_idx={match['cam2_idx']}")
            print(f"    3D=({match['x']:.2f}, {match['y']:.2f}, {match['z']:.2f}), ray_dist={match['ray_distance']:.4f}")
            print(f"  Baseline (top-1):")
            print(f"    3D=({baseline_results[common.index(fi)]['x']:.2f}, "
                  f"{baseline_results[common.index(fi)]['y']:.2f}, "
                  f"{baseline_results[common.index(fi)]['z']:.2f}), ray_dist={baseline_rd:.4f}")
            if shown >= 15:
                print(f"  ... ({different_pick - shown} more)")
                break

    print(f"\nMatcher stats: {matcher2.get_stats()}")


if __name__ == "__main__":
    main()
