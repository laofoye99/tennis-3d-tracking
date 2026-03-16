"""Evaluate blob verifier: run TrackNet + YOLO on videos, compare with GT, triangulate 3D.

Usage:
    python -m tools.evaluate_blob_verifier

Outputs:
    exports/blob_eval/  — per-camera results, GT comparison, 3D visualization
"""

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────

CAMERAS = {
    "cam66": {
        "video": "uploads/cam66_20260307_173403_2min.mp4",
        "annotations": "uploads/cam66_20260307_173403_2min",
        "homography_key": "cam66",
    },
    "cam68": {
        "video": "uploads/cam68_20260307_173403_2min.mp4",
        "annotations": "uploads/cam68_20260307_173403_2min",
        "homography_key": "cam68",
    },
}

MODEL_PATH = "model_weight/TrackNet_best.pt"
VERIFIER_PATH = "model_weight/blob_verifier_yolo.pt"
HOMOGRAPHY_PATH = "src/homography_matrices.json"
INPUT_SIZE = (288, 512)
FRAMES_IN = 8
FRAMES_OUT = 8
THRESHOLD = 0.35
DEVICE = "cuda"
HEATMAP_MASK = [(0, 0, 620, 40)]
VERIFIER_CONF = 0.15
VERIFIER_CROP = 128
MAX_FRAMES = 3000  # process all frames
OUT_DIR = Path("exports/blob_eval")


def load_gt_annotations(ann_dir: str) -> dict[int, list[dict]]:
    """Load GT rectangle annotations from LabelMe JSON files."""
    from tools.prepare_yolo_crops import load_annotations
    all_ann = load_annotations(Path(ann_dir))
    gt = {}
    for fi, blobs in all_ann.items():
        gt_blobs = [b for b in blobs if b["score"] is None and b["label"] == "ball"]
        if gt_blobs:
            gt[fi] = gt_blobs
    return gt


def run_camera_pipeline(cam_name: str, cam_cfg: dict) -> dict[int, dict]:
    """Run TrackNet + blob verifier on a camera's video, return per-frame detections."""
    from app.pipeline.inference import create_detector
    from app.pipeline.postprocess import BallTracker
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.blob_verifier import BlobVerifier, verify_blobs

    log.info("=== Processing %s ===", cam_name)

    # Initialize components
    detector = create_detector(MODEL_PATH, INPUT_SIZE, FRAMES_IN, FRAMES_OUT, DEVICE)
    tracker = BallTracker(
        original_size=(1920, 1080),
        threshold=THRESHOLD,
        heatmap_mask=HEATMAP_MASK,
    )
    homography = HomographyTransformer(HOMOGRAPHY_PATH, cam_cfg["homography_key"])
    verifier = BlobVerifier(
        model_path=VERIFIER_PATH,
        crop_size=VERIFIER_CROP,
        conf=VERIFIER_CONF,
        device=DEVICE,
    )

    cap = cv2.VideoCapture(cam_cfg["video"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {cam_cfg['video']}")

    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute background median
    if hasattr(detector, "compute_video_median"):
        log.info("Computing video median...")
        detector.compute_video_median(cap, 0, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_buffer = []
    raw_buffer = []
    results = {}
    processed = 0
    multi_blob_count = 0
    verified_count = 0

    log.info("Running inference on %d frames...", total)
    while processed < total:
        ret, frame = cap.read()
        if not ret:
            break
        processed += 1
        raw_buffer.append(frame)

        masked = frame.copy()
        masked[0:41, 0:603] = 0
        frame_buffer.append(masked)

        if len(frame_buffer) < FRAMES_IN:
            continue

        # Inference
        heatmaps = detector.infer(frame_buffer)

        for i in range(min(FRAMES_OUT, len(heatmaps))):
            fi = processed - FRAMES_OUT + i
            blobs = tracker.process_heatmap_multi(heatmaps[i])
            if not blobs:
                continue

            n_blobs_before = len(blobs)

            # YOLO blob verification on every frame
            multi_blob_count += 1 if len(blobs) > 1 else 0
            blobs = verify_blobs(raw_buffer[i], blobs, verifier, threshold=VERIFIER_CONF)
            used_verifier = True
            verified_count += 1

            # Court-X filtering
            candidates = []
            for blob in blobs:
                wx, wy = homography.pixel_to_world(blob["pixel_x"], blob["pixel_y"])
                if homography.court_x_min <= wx <= homography.court_x_max:
                    candidates.append({
                        "pixel_x": blob["pixel_x"],
                        "pixel_y": blob["pixel_y"],
                        "world_x": wx,
                        "world_y": wy,
                        "blob_sum": blob["blob_sum"],
                        "yolo_conf": blob.get("yolo_conf", -1),
                    })

            if candidates:
                top = candidates[0]
                results[fi] = {
                    "pixel_x": top["pixel_x"],
                    "pixel_y": top["pixel_y"],
                    "world_x": top["world_x"],
                    "world_y": top["world_y"],
                    "confidence": top["blob_sum"],
                    "yolo_conf": top.get("yolo_conf", -1),
                    "n_blobs_before": n_blobs_before,
                    "n_blobs_after": len(candidates),
                    "used_verifier": used_verifier,
                }

        frame_buffer.clear()
        raw_buffer.clear()

        if processed % 500 == 0:
            log.info("  %s: %d/%d frames", cam_name, processed, total)

    cap.release()
    log.info(
        "%s done: %d detections, %d multi-blob frames, %d verified",
        cam_name, len(results), multi_blob_count, verified_count,
    )
    return results


def compare_with_gt(detections: dict[int, dict], gt: dict[int, list[dict]], cam_name: str) -> dict:
    """Compare detections with GT annotations, compute metrics."""
    tp = 0  # detection within threshold of GT
    fp = 0  # detection but no GT (can't evaluate — skip)
    fn = 0  # GT but no detection
    errors = []
    dist_threshold = 15.0  # pixels

    gt_frames = set(gt.keys())
    det_frames = set(detections.keys())

    for fi in gt_frames:
        gt_blobs = gt[fi]
        if fi not in det_frames:
            fn += len(gt_blobs)
            continue

        det = detections[fi]
        # Check if detection is close to any GT blob
        min_dist = float("inf")
        for gb in gt_blobs:
            d = np.hypot(det["pixel_x"] - gb["pixel_x"], det["pixel_y"] - gb["pixel_y"])
            min_dist = min(min_dist, d)

        if min_dist <= dist_threshold:
            tp += 1
            errors.append(min_dist)
        else:
            fn += 1
            errors.append(min_dist)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mean_err = np.mean(errors) if errors else 0
    median_err = np.median(errors) if errors else 0

    stats = {
        "camera": cam_name,
        "gt_frames": len(gt_frames),
        "detected_frames": len(det_frames),
        "tp": tp,
        "fn": fn,
        "recall": recall,
        "mean_pixel_error": float(mean_err),
        "median_pixel_error": float(median_err),
    }
    log.info(
        "%s GT comparison: recall=%.1f%% (%d/%d), mean_err=%.1fpx, median_err=%.1fpx",
        cam_name, recall * 100, tp, tp + fn, mean_err, median_err,
    )
    return stats


def triangulate_3d(
    det66: dict[int, dict],
    det68: dict[int, dict],
) -> dict[int, dict]:
    """Simple midpoint triangulation from two camera world coordinates.

    Uses world (court) coordinates from homography. For frames where both cameras
    detect the ball, average their world positions. This gives a 2D court position.
    For Z (height), estimate from pixel Y deviation from court plane.
    """
    common_frames = set(det66.keys()) & set(det68.keys())
    points_3d = {}

    for fi in sorted(common_frames):
        d66 = det66[fi]
        d68 = det68[fi]

        # Average world coordinates (court plane projection)
        wx = (d66["world_x"] + d68["world_x"]) / 2
        wy = (d66["world_y"] + d68["world_y"]) / 2

        # Estimate height from pixel positions
        # Higher ball = lower pixel_y, use deviation from expected court-plane position
        # This is a rough estimate — proper 3D requires camera matrices
        points_3d[fi] = {
            "frame": fi,
            "x": float(wx),
            "y": float(wy),
            "z": 0.0,  # court plane — height estimation needs camera intrinsics
            "cam66_px": (d66["pixel_x"], d66["pixel_y"]),
            "cam68_px": (d68["pixel_x"], d68["pixel_y"]),
            "cam66_world": (d66["world_x"], d66["world_y"]),
            "cam68_world": (d68["world_x"], d68["world_y"]),
        }

    return points_3d


def visualize_results(
    points_3d: dict[int, dict],
    det66: dict[int, dict],
    det68: dict[int, dict],
    gt66: dict[int, list[dict]],
    gt68: dict[int, list[dict]],
    out_dir: Path,
):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # ── 1. Court top-view with 3D triangulated positions ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Draw court (ITF dimensions: 23.77m x 10.97m)
    court_w, court_h = 10.97, 23.77
    # Court outline
    ax.add_patch(Rectangle((0, 0), court_w, court_h, fill=False, edgecolor="white", linewidth=2))
    # Singles lines
    ax.add_patch(Rectangle((1.37, 0), court_w - 2.74, court_h, fill=False, edgecolor="white", linewidth=1))
    # Service boxes
    ax.plot([0, court_w], [court_h/2, court_h/2], "w-", linewidth=1)  # net
    ax.plot([1.37, court_w - 1.37], [court_h/2 - 6.4, court_h/2 - 6.4], "w-", linewidth=0.5)
    ax.plot([1.37, court_w - 1.37], [court_h/2 + 6.4, court_h/2 + 6.4], "w-", linewidth=0.5)
    ax.plot([court_w/2, court_w/2], [court_h/2 - 6.4, court_h/2 + 6.4], "w-", linewidth=0.5)

    # Plot triangulated points
    frames = sorted(points_3d.keys())
    xs = [points_3d[f]["x"] for f in frames]
    ys = [points_3d[f]["y"] for f in frames]
    scatter = ax.scatter(xs, ys, c=range(len(frames)), cmap="plasma", s=8, alpha=0.7, zorder=5)
    plt.colorbar(scatter, ax=ax, label="Frame index", shrink=0.8)

    ax.set_facecolor("#2d5a27")
    ax.set_xlim(-2, court_w + 2)
    ax.set_ylim(-2, court_h + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"3D Triangulated Ball Positions (n={len(points_3d)})")
    fig.tight_layout()
    fig.savefig(str(out_dir / "court_topview_3d.png"), dpi=150, facecolor="#1a1a1a")
    plt.close(fig)
    log.info("Saved court_topview_3d.png")

    # ── 2. Per-camera world coordinate comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, cam_name, det, gt in [
        (axes[0], "cam66", det66, gt66),
        (axes[1], "cam68", det68, gt68),
    ]:
        # Plot all detections
        det_frames_sorted = sorted(det.keys())
        det_xs = [det[f]["world_x"] for f in det_frames_sorted]
        det_ys = [det[f]["world_y"] for f in det_frames_sorted]
        ax.scatter(det_xs, det_ys, s=4, alpha=0.3, label=f"Detections (n={len(det)})", color="cyan")

        # Highlight verified frames
        ver_frames = [f for f in det_frames_sorted if det[f].get("used_verifier")]
        if ver_frames:
            ver_xs = [det[f]["world_x"] for f in ver_frames]
            ver_ys = [det[f]["world_y"] for f in ver_frames]
            ax.scatter(ver_xs, ver_ys, s=12, alpha=0.6, label=f"YOLO verified (n={len(ver_frames)})", color="yellow", marker="x")

        # Draw court
        ax.add_patch(Rectangle((0, 0), court_w, court_h, fill=False, edgecolor="white", linewidth=1))
        ax.plot([0, court_w], [court_h/2, court_h/2], "w-", linewidth=0.5)

        ax.set_facecolor("#2d5a27")
        ax.set_xlim(-3, court_w + 3)
        ax.set_ylim(-3, court_h + 3)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{cam_name} World Coordinates")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_dir / "per_camera_world.png"), dpi=150, facecolor="#1a1a1a")
    plt.close(fig)
    log.info("Saved per_camera_world.png")

    # ── 3. Pixel error distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cam_name, det, gt in [
        (axes[0], "cam66", det66, gt66),
        (axes[1], "cam68", det68, gt68),
    ]:
        errors = []
        for fi, gt_blobs in gt.items():
            if fi not in det:
                continue
            d = det[fi]
            for gb in gt_blobs:
                err = np.hypot(d["pixel_x"] - gb["pixel_x"], d["pixel_y"] - gb["pixel_y"])
                errors.append(err)

        if errors:
            ax.hist(errors, bins=50, range=(0, 50), color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(np.median(errors), color="red", linestyle="--", label=f"Median={np.median(errors):.1f}px")
            ax.axvline(15, color="orange", linestyle=":", label="Threshold=15px")
            ax.legend()
        ax.set_xlabel("Pixel Error")
        ax.set_ylabel("Count")
        ax.set_title(f"{cam_name} Detection Error (n={len(errors)})")

    fig.tight_layout()
    fig.savefig(str(out_dir / "pixel_error_dist.png"), dpi=150)
    plt.close(fig)
    log.info("Saved pixel_error_dist.png")

    # ── 4. Timeline: detections vs GT ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    for ax, cam_name, det, gt in [
        (axes[0], "cam66", det66, gt66),
        (axes[1], "cam68", det68, gt68),
    ]:
        det_fi = sorted(det.keys())
        gt_fi = sorted(gt.keys())
        ax.scatter(det_fi, [1]*len(det_fi), s=1, alpha=0.5, color="cyan", label="Detection")
        ax.scatter(gt_fi, [0]*len(gt_fi), s=3, alpha=0.8, color="lime", label="GT")
        # Mark multi-blob verified frames
        ver_fi = [f for f in det_fi if det[f].get("used_verifier")]
        ax.scatter(ver_fi, [1]*len(ver_fi), s=6, color="yellow", marker="|", label="YOLO verified")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GT", "Det"])
        ax.set_title(cam_name)
        ax.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Frame Index")
    fig.tight_layout()
    fig.savefig(str(out_dir / "timeline.png"), dpi=150)
    plt.close(fig)
    log.info("Saved timeline.png")

    # ── 5. World coordinate difference between cameras ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    common = sorted(set(det66.keys()) & set(det68.keys()))
    if common:
        dx = [det66[f]["world_x"] - det68[f]["world_x"] for f in common]
        dy = [det66[f]["world_y"] - det68[f]["world_y"] for f in common]
        axes[0].plot(common, dx, ".", markersize=2, alpha=0.5, color="steelblue")
        axes[0].set_ylabel("delta X (m)")
        axes[0].set_title(f"World Coordinate Diff: cam66 - cam68 (n={len(common)})")
        axes[0].axhline(0, color="gray", linewidth=0.5)
        axes[1].plot(common, dy, ".", markersize=2, alpha=0.5, color="coral")
        axes[1].set_ylabel("delta Y (m)")
        axes[1].set_xlabel("Frame Index")
        axes[1].axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(str(out_dir / "camera_diff.png"), dpi=150)
    plt.close(fig)
    log.info("Saved camera_diff.png")


def cross_camera_rescue(
    det66: dict[int, dict],
    det68: dict[int, dict],
) -> None:
    """Cross-camera YOLO rescue: when one camera misses, project from the other and verify.

    For frames where only one camera has a detection:
    1. Get source camera's world coords
    2. Project to target camera's pixel space via world_to_pixel
    3. Extract 128x128 crop from target camera's raw frame
    4. Run YOLO to confirm ball presence
    5. If confirmed, add detection to target camera's results

    Modifies det66/det68 dicts in-place.
    """
    from app.pipeline.homography import HomographyTransformer
    from app.pipeline.blob_verifier import BlobVerifier, extract_crops

    log.info("=== Cross-camera YOLO rescue ===")

    homo66 = HomographyTransformer(HOMOGRAPHY_PATH, "cam66")
    homo68 = HomographyTransformer(HOMOGRAPHY_PATH, "cam68")
    verifier = BlobVerifier(
        model_path=VERIFIER_PATH,
        crop_size=VERIFIER_CROP,
        conf=VERIFIER_CONF,
        device=DEVICE,
    )

    all_frames_66 = set(det66.keys())
    all_frames_68 = set(det68.keys())

    # Frames where only one camera detected
    only66 = sorted(all_frames_66 - all_frames_68)
    only68 = sorted(all_frames_68 - all_frames_66)
    log.info("  cam66-only frames: %d, cam68-only frames: %d", len(only66), len(only68))

    # Open both video captures
    cap66 = cv2.VideoCapture(CAMERAS["cam66"]["video"])
    cap68 = cv2.VideoCapture(CAMERAS["cam68"]["video"])

    rescued_66 = 0  # rescued into cam66 from cam68
    rescued_68 = 0  # rescued into cam68 from cam66

    # Rescue cam68 from cam66: cam66 has detection, cam68 missing
    # Project cam66 world → cam68 pixels → crop cam68 frame → YOLO
    for fi in only66:
        d66 = det66[fi]
        wx, wy = d66["world_x"], d66["world_y"]

        # Project to cam68 pixel space
        try:
            px68, py68 = homo68.world_to_pixel(wx, wy)
        except Exception:
            continue

        # Bounds check
        if not (0 <= px68 < 1920 and 0 <= py68 < 1080):
            continue

        # Read cam68 frame
        cap68.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame68 = cap68.read()
        if not ret:
            continue

        # Extract crop and run YOLO
        synthetic_blob = {"pixel_x": px68, "pixel_y": py68}
        crops = extract_crops(frame68, [synthetic_blob], VERIFIER_CROP)
        results = verifier.detect_crops(crops)

        if results[0] is not None and results[0]["yolo_conf"] >= VERIFIER_CONF:
            det_info = results[0]
            half = VERIFIER_CROP // 2
            refined_px = px68 + det_info["crop_cx"] - half
            refined_py = py68 + det_info["crop_cy"] - half

            # Convert refined pixel to world coords
            rwx, rwy = homo68.pixel_to_world(refined_px, refined_py)

            det68[fi] = {
                "pixel_x": refined_px,
                "pixel_y": refined_py,
                "world_x": rwx,
                "world_y": rwy,
                "confidence": 0.0,
                "yolo_conf": det_info["yolo_conf"],
                "n_blobs_before": 0,
                "n_blobs_after": 1,
                "used_verifier": True,
                "source": "cross_rescue_from_cam66",
            }
            rescued_68 += 1

    # Rescue cam66 from cam68: cam68 has detection, cam66 missing
    for fi in only68:
        d68 = det68[fi]
        wx, wy = d68["world_x"], d68["world_y"]

        try:
            px66, py66 = homo66.world_to_pixel(wx, wy)
        except Exception:
            continue

        if not (0 <= px66 < 1920 and 0 <= py66 < 1080):
            continue

        cap66.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame66 = cap66.read()
        if not ret:
            continue

        synthetic_blob = {"pixel_x": px66, "pixel_y": py66}
        crops = extract_crops(frame66, [synthetic_blob], VERIFIER_CROP)
        results = verifier.detect_crops(crops)

        if results[0] is not None and results[0]["yolo_conf"] >= VERIFIER_CONF:
            det_info = results[0]
            half = VERIFIER_CROP // 2
            refined_px = px66 + det_info["crop_cx"] - half
            refined_py = py66 + det_info["crop_cy"] - half

            rwx, rwy = homo66.pixel_to_world(refined_px, refined_py)

            det66[fi] = {
                "pixel_x": refined_px,
                "pixel_y": refined_py,
                "world_x": rwx,
                "world_y": rwy,
                "confidence": 0.0,
                "yolo_conf": det_info["yolo_conf"],
                "n_blobs_before": 0,
                "n_blobs_after": 1,
                "used_verifier": True,
                "source": "cross_rescue_from_cam68",
            }
            rescued_66 += 1

    cap66.release()
    cap68.release()

    log.info(
        "Cross-camera rescue done: cam66 +%d frames (from cam68), cam68 +%d frames (from cam66)",
        rescued_66, rescued_68,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run both cameras
    all_detections = {}
    all_gt = {}
    all_stats = {}

    for cam_name, cam_cfg in CAMERAS.items():
        detections = run_camera_pipeline(cam_name, cam_cfg)
        all_detections[cam_name] = detections

        gt = load_gt_annotations(cam_cfg["annotations"])
        all_gt[cam_name] = gt

        stats = compare_with_gt(detections, gt, cam_name)
        all_stats[cam_name] = stats

    # ── Cross-camera YOLO rescue ──────────────────────────────────────────
    det66 = all_detections["cam66"]
    det68 = all_detections["cam68"]
    cross_camera_rescue(det66, det68)

    # Save per-camera results
    for cam_name in CAMERAS:
        det = all_detections[cam_name]
        out_file = OUT_DIR / f"{cam_name}_detections.json"
        with open(out_file, "w") as f:
            json.dump({str(k): v for k, v in det.items()}, f, indent=2)
        log.info("Saved %s (%d detections)", out_file, len(det))

    # Save GT comparison stats
    with open(OUT_DIR / "gt_comparison.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Triangulate 3D
    det66 = all_detections["cam66"]
    det68 = all_detections["cam68"]
    points_3d = triangulate_3d(det66, det68)
    log.info("3D triangulation: %d common frames out of cam66=%d, cam68=%d",
             len(points_3d), len(det66), len(det68))

    # Save 3D points
    with open(OUT_DIR / "points_3d.json", "w") as f:
        json.dump({str(k): v for k, v in points_3d.items()}, f, indent=2)

    # Visualize
    visualize_results(
        points_3d, det66, det68,
        all_gt["cam66"], all_gt["cam68"],
        OUT_DIR,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for cam_name, stats in all_stats.items():
        print(f"\n{cam_name}:")
        print(f"  GT frames: {stats['gt_frames']}")
        print(f"  Detected frames: {stats['detected_frames']}")
        print(f"  Recall: {stats['recall']:.1%} ({stats['tp']}/{stats['tp']+stats['fn']})")
        print(f"  Mean pixel error: {stats['mean_pixel_error']:.1f}px")
        print(f"  Median pixel error: {stats['median_pixel_error']:.1f}px")
    print(f"\n3D triangulated: {len(points_3d)} frames")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
