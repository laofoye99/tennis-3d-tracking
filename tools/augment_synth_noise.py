"""Augment synthetic trajectory data with realistic noise matching real pipeline errors.

Real pipeline noise profile (measured from cam66 2-min clip):
  - Detection dropout: 37% of frames have no detection
  - Consecutive tracking: 96.4% of detected frames are gap=1
  - Tracking gaps >30 frames: 0.6% (dead ball / rally transitions)
  - Z offset at bounce: real=0.34m vs ideal=0.0m (homography error)
  - Z noise std: ~0.2m per frame
  - XY noise: ~0.1-0.3m from homography reprojection error
  - Bounce z in real data: mean=0.34m (should be 0)

Augmentation pipeline:
  1. Resample synth 100fps → 25fps (match real camera)
  2. Add per-frame position noise (XY: σ=0.15m, Z: σ=0.2m + bias=0.25m)
  3. Random frame dropout (37% overall, clustered)
  4. Rally gap injection (insert 30-130 frame gaps between rallies)
  5. Occasional drift (ball locks to wrong object for 5-20 frames)
  6. Keep event labels on surviving frames

Usage:
    python -m tools.augment_synth_noise
    python -m tools.augment_synth_noise --num-variants 20 --base-seed 100
"""

import argparse
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REAL_FPS = 25

# ── Position noise (from real data analysis) ──
XY_NOISE_STD = 0.15       # meters
Z_NOISE_STD = 0.20        # meters
Z_BIAS_MEAN = 0.25        # systematic upward bias from homography
Z_BIAS_STD = 0.10         # bias varies per segment
JITTER_AUTOCORR = 0.7     # temporal correlation

# ── Dropout (37% overall, clustered) ──
DROPOUT_RATE = 0.37
DROPOUT_CLUSTER_P = 0.85   # prob dropout continues

# ── Tracking gaps (dead ball) ──
GAP_INJECT_PROB = 0.3
GAP_MIN_FRAMES = 30
GAP_MAX_FRAMES = 130

# ── Drift (locks to wrong object) ──
DRIFT_PROB_PER_RALLY = 0.15
DRIFT_MIN_FRAMES = 5
DRIFT_MAX_FRAMES = 20
DRIFT_OFFSET_RANGE = (1.0, 4.0)

# ── Outliers (wrong blob selection) ──
OUTLIER_RATE = 0.04
OUTLIER_MAGNITUDE = 2.0


def resample_to_25fps(frames: list, synth_fps: int) -> list:
    """Downsample from synth_fps to 25fps, preserving event frames."""
    ratio = synth_fps / REAL_FPS
    event_indices = {i for i, f in enumerate(frames) if f["evt"] != "fly"}

    resampled = []
    for i in range(0, len(frames), int(ratio)):
        window = set(range(i, min(i + int(ratio), len(frames))))
        evt_in_window = window & event_indices
        idx = min(evt_in_window) if evt_in_window else i

        if idx < len(frames):
            f = deepcopy(frames[idx])
            f["f"] = len(resampled)
            f["t"] = round(len(resampled) / REAL_FPS, 4)
            resampled.append(f)

    return resampled


def add_position_noise(frames: list, rng: np.random.Generator) -> list:
    """Add realistic position noise with Z bias and temporal correlation."""
    n = len(frames)

    def ar1_noise(std, length):
        white = rng.normal(0, std, length)
        corr = np.zeros(length)
        corr[0] = white[0]
        for i in range(1, length):
            corr[i] = JITTER_AUTOCORR * corr[i - 1] + (1 - JITTER_AUTOCORR) * white[i]
        if corr.std() > 0:
            corr = corr / corr.std() * std
        return corr

    nx = ar1_noise(XY_NOISE_STD, n)
    ny = ar1_noise(XY_NOISE_STD, n)
    nz = ar1_noise(Z_NOISE_STD, n)

    # Per-segment Z bias (changes every 50-200 frames)
    seg_len = int(rng.integers(50, 200))
    z_bias = float(rng.normal(Z_BIAS_MEAN, Z_BIAS_STD))

    for i, f in enumerate(frames):
        if i % seg_len == 0:
            z_bias = float(rng.normal(Z_BIAS_MEAN, Z_BIAS_STD))
            seg_len = int(rng.integers(50, 200))

        # Distance-dependent Z noise (worse at mid-court)
        y = f["pos"][1]
        dist_factor = 1.0 + 0.3 * min(abs(y), abs(y - 23.77)) / 12.0

        pos = f["pos"]
        f["pos"] = [
            round(pos[0] + nx[i], 4),
            round(pos[1] + ny[i], 4),
            round(max(0, pos[2] + nz[i] * dist_factor + z_bias), 4),
        ]

    # Recompute velocities from noisy positions
    for i in range(1, len(frames)):
        dt = frames[i]["t"] - frames[i - 1]["t"]
        if dt > 0.001:
            frames[i]["vel"] = [
                round((frames[i]["pos"][j] - frames[i - 1]["pos"][j]) / dt, 2)
                for j in range(3)
            ]
            frames[i]["spd"] = round(
                sum(v ** 2 for v in frames[i]["vel"]) ** 0.5, 1
            )

    return frames


def add_outliers(frames: list, rng: np.random.Generator) -> list:
    """Add occasional large position jumps (wrong blob)."""
    n_outliers = int(len(frames) * OUTLIER_RATE)
    indices = rng.choice(len(frames), n_outliers, replace=False)

    for idx in indices:
        f = frames[idx]
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0.5, OUTLIER_MAGNITUDE)
        f["pos"][0] = round(f["pos"][0] + dist * np.cos(angle), 4)
        f["pos"][1] = round(f["pos"][1] + dist * np.sin(angle), 4)
        f["pos"][2] = round(max(0, f["pos"][2] + rng.normal(0, 0.3)), 4)

    return frames


def inject_drift(frames: list, rng: np.random.Generator) -> list:
    """Ball locks to wrong object for short bursts."""
    rallies = set(f.get("rid", -1) for f in frames if f.get("rid", -1) >= 0)

    for rid in rallies:
        if rng.random() > DRIFT_PROB_PER_RALLY:
            continue

        rally_idx = [i for i, f in enumerate(frames) if f.get("rid") == rid]
        if len(rally_idx) < 30:
            continue

        drift_len = int(rng.integers(DRIFT_MIN_FRAMES, DRIFT_MAX_FRAMES))
        max_start = len(rally_idx) - drift_len
        if max_start <= 0:
            continue

        start = int(rng.integers(0, max_start))
        offset = float(rng.uniform(*DRIFT_OFFSET_RANGE))
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d) * offset
        d[2] = abs(d[2]) * 0.3  # mostly XY

        for j in range(drift_len):
            fi = rally_idx[start + j]
            blend = np.sin(j / drift_len * np.pi)
            frames[fi]["pos"] = [
                round(frames[fi]["pos"][0] + d[0] * blend, 4),
                round(frames[fi]["pos"][1] + d[1] * blend, 4),
                round(max(0, frames[fi]["pos"][2] + d[2] * blend), 4),
            ]

    return frames


def apply_dropout(frames: list, rng: np.random.Generator) -> list:
    """Clustered frame dropout matching real detection failure patterns."""
    result = []
    in_dropout = False

    for f in frames:
        if in_dropout:
            if rng.random() < DROPOUT_CLUSTER_P:
                continue
            in_dropout = False

        speed = f.get("spd", 0)
        start_prob = 0.05 if speed > 5.0 else 0.12 if speed > 2.0 else 0.18

        if rng.random() < start_prob:
            in_dropout = True
            continue

        result.append(f)

    return result


def inject_rally_gaps(frames: list, rng: np.random.Generator) -> list:
    """Insert dead-ball gaps between rallies."""
    prev_rid = frames[0].get("rid", -1) if frames else -1
    boundaries = []
    for i, f in enumerate(frames):
        rid = f.get("rid", -1)
        if rid != prev_rid and rid >= 0:
            boundaries.append(i)
        prev_rid = rid

    # Remove frames at some rally boundaries
    to_remove = set()
    for b in boundaries:
        if rng.random() < GAP_INJECT_PROB:
            gap = int(rng.integers(GAP_MIN_FRAMES, GAP_MAX_FRAMES))
            half = gap // 2
            for j in range(max(0, b - half), min(len(frames), b + half)):
                to_remove.add(j)

    return [f for i, f in enumerate(frames) if i not in to_remove]


def renumber(frames: list) -> list:
    """Sequential frame numbers after dropout."""
    for i, f in enumerate(frames):
        f["f"] = i
        f["t"] = round(i / REAL_FPS, 4)
    return frames


def compute_stats(frames: list) -> dict:
    bounces = [f for f in frames if f.get("evt") == "bounce"]
    hits = [f for f in frames if f.get("evt") == "hit"]
    serves = [f for f in frames if f.get("evt") == "serve"]
    rallies = set(f.get("rid", -1) for f in frames if f.get("rid", -1) >= 0)
    bz = [f["pos"][2] for f in bounces] if bounces else [0]
    speeds = [f["spd"] for f in frames if f.get("spd", 0) > 0.5]

    return {
        "total_frames": len(frames),
        "fps": REAL_FPS,
        "duration_sec": round(len(frames) / REAL_FPS, 2),
        "total_rallies": len(rallies),
        "total_serves": len(serves),
        "total_hits": len(hits),
        "total_bounces": len(bounces),
        "bounce_z_mean": round(float(np.mean(bz)), 4),
        "bounce_z_std": round(float(np.std(bz)), 4),
        "speed_mean": round(float(np.mean(speeds)), 2) if speeds else 0,
    }


def augment_one(raw_frames: list, synth_fps: int, seed: int) -> tuple:
    """Full noise pipeline. Returns (noisy_frames, stats)."""
    rng = np.random.default_rng(seed)

    frames = resample_to_25fps(raw_frames, synth_fps)
    n0 = len(frames)

    frames = add_position_noise(frames, rng)
    frames = add_outliers(frames, rng)
    frames = inject_drift(frames, rng)
    frames = inject_rally_gaps(frames, rng)
    frames = apply_dropout(frames, rng)
    frames = renumber(frames)

    stats = compute_stats(frames)
    stats["dropout_pct"] = round((1 - len(frames) / n0) * 100, 1)
    stats["seed"] = seed
    return frames, stats


def main():
    parser = argparse.ArgumentParser(description="Augment synthetic data with realistic noise")
    parser.add_argument("--input", default="data/synth/raw/trajectory_20260320_012911.json")
    parser.add_argument("--output", default="data/synth/noisy/")
    parser.add_argument("--num-variants", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.input) as f:
        raw = json.load(f)
    raw_frames = raw["frames"]
    synth_fps = raw.get("fps", 100)
    logger.info("Loaded %d frames at %dfps, %.0fs", len(raw_frames), synth_fps, raw["duration_sec"])

    logger.info("Generating %d noisy variants...", args.num_variants)
    all_stats = []

    for i in range(args.num_variants):
        seed = args.base_seed + i
        frames, stats = augment_one(raw_frames, synth_fps, seed)
        path = os.path.join(args.output, f"noisy_{i:03d}.json")
        with open(path, "w") as f:
            json.dump({**stats, "frames": frames}, f, separators=(",", ":"))
        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info("  [%d] %d frames, %d bounces, %d hits, dropout=%.1f%%, bz=%.3f (%.1fMB)",
                     i, stats["total_frames"], stats["total_bounces"],
                     stats["total_hits"], stats["dropout_pct"],
                     stats["bounce_z_mean"], size_mb)
        all_stats.append(stats)

    # Summary
    bounces = [s["total_bounces"] for s in all_stats]
    hits = [s["total_hits"] for s in all_stats]
    logger.info("=== DONE ===")
    logger.info("Total: %d bounces, %d hits across %d variants",
                sum(bounces), sum(hits), len(all_stats))
    logger.info("Bounces/variant: mean=%.0f range=[%d, %d]",
                np.mean(bounces), min(bounces), max(bounces))
    logger.info("Bounce z mean: %.3f (target ~0.25-0.35)",
                np.mean([s["bounce_z_mean"] for s in all_stats]))

    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump({"variants": all_stats}, f, indent=2)


if __name__ == "__main__":
    main()
