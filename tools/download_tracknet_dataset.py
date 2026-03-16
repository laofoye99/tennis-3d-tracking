"""Download and convert public TrackNet tennis ball dataset.

Downloads the TrackNet tennis dataset from Google Drive and converts it
to a unified format for training.

Usage:
    python -m tools.download_tracknet_dataset --output data/tracknet_public
    python -m tools.download_tracknet_dataset --skip-download --input /path/to/raw --output data/tracknet_public
"""

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Google Drive folder ID for the TrackNet tennis dataset (19,835 frames)
GDRIVE_FOLDER_ID = "11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut"


def download_dataset(output_dir: Path) -> Path:
    """Download TrackNet dataset from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        log.error("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    log.info("Downloading TrackNet dataset from %s ...", url)
    log.info("Output: %s", raw_dir)

    gdown.download_folder(url, output=str(raw_dir), quiet=False)

    log.info("Download complete: %s", raw_dir)
    return raw_dir


def find_csv_files(raw_dir: Path) -> list[Path]:
    """Find all CSV label files in the downloaded dataset."""
    csv_files = sorted(raw_dir.rglob("*.csv"))
    log.info("Found %d CSV files in %s", len(csv_files), raw_dir)
    return csv_files


def parse_tracknet_csv(csv_path: Path) -> list[dict]:
    """Parse a TrackNet-format CSV file.

    Expected columns: Frame, Visibility, X, Y
    Visibility: 0=not visible, 1=visible but blurry, 2=visible and clear
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows

        # Normalize header names
        header = [h.strip().lower() for h in header]

        # Find column indices
        frame_idx = None
        vis_idx = None
        x_idx = None
        y_idx = None

        for i, h in enumerate(header):
            if "frame" in h:
                frame_idx = i
            elif "vis" in h:
                vis_idx = i
            elif h == "x":
                x_idx = i
            elif h == "y":
                y_idx = i

        if frame_idx is None or x_idx is None or y_idx is None:
            # Try positional: Frame, Visibility, X, Y
            if len(header) >= 4:
                frame_idx, vis_idx, x_idx, y_idx = 0, 1, 2, 3
            else:
                log.warning("Cannot parse CSV header: %s in %s", header, csv_path)
                return rows

        for row in reader:
            if len(row) < max(frame_idx, x_idx, y_idx) + 1:
                continue
            try:
                frame_name = row[frame_idx].strip()
                vis = int(float(row[vis_idx].strip())) if vis_idx is not None else 2
                x = float(row[x_idx].strip())
                y = float(row[y_idx].strip())

                # Extract frame number from filename or raw number
                if "." in frame_name:
                    frame_num = int(Path(frame_name).stem)
                else:
                    frame_num = int(frame_name)

                rows.append({
                    "frame": frame_num,
                    "visibility": vis,
                    "x": x,
                    "y": y,
                    "frame_name": frame_name,
                })
            except (ValueError, IndexError):
                continue

    return rows


def find_frame_dir(csv_path: Path) -> Path | None:
    """Find the frame directory corresponding to a CSV file."""
    parent = csv_path.parent

    # Common patterns: frames/, frame/, same dir as CSV
    for name in ["frame", "frames", "Frame", "Frames"]:
        candidate = parent / name
        if candidate.is_dir():
            return candidate

    # Check if parent itself contains image files
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for f in parent.iterdir():
        if f.suffix.lower() in img_exts:
            return parent

    # Go up one level and check
    for name in ["frame", "frames", "Frame", "Frames"]:
        candidate = parent.parent / name
        if candidate.is_dir():
            return candidate

    return None


def convert_match(
    csv_path: Path,
    frame_dir: Path,
    output_dir: Path,
    match_name: str,
) -> dict:
    """Convert a single match from raw TrackNet format to unified format."""
    match_out = output_dir / match_name
    match_frame_out = match_out / "frame"
    match_frame_out.mkdir(parents=True, exist_ok=True)

    # Parse CSV
    rows = parse_tracknet_csv(csv_path)
    if not rows:
        log.warning("No valid rows in %s", csv_path)
        return {"name": match_name, "frames": 0, "labeled": 0}

    # Copy/link frames and write standardized CSV
    csv_out_path = match_out / "labels.csv"
    labeled_count = 0

    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Visibility", "X", "Y"])

        for row in rows:
            fi = row["frame"]
            vis = row["visibility"]
            x = row["x"]
            y = row["y"]

            # Find source frame
            src_frame = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                for pattern in [f"{fi}{ext}", f"{fi:05d}{ext}", f"{fi:04d}{ext}",
                                row["frame_name"]]:
                    candidate = frame_dir / pattern
                    if candidate.exists():
                        src_frame = candidate
                        break
                if src_frame:
                    break

            if src_frame is None:
                continue

            # Copy frame
            dst_frame = match_frame_out / f"{fi}.jpg"
            if not dst_frame.exists():
                if src_frame.suffix.lower() in [".jpg", ".jpeg"]:
                    shutil.copy2(src_frame, dst_frame)
                else:
                    img = cv2.imread(str(src_frame))
                    if img is not None:
                        cv2.imwrite(str(dst_frame), img)

            writer.writerow([fi, vis, x, y])

            if vis > 0:
                labeled_count += 1

    stats = {
        "name": match_name,
        "frames": len(rows),
        "labeled": labeled_count,
        "negative": len(rows) - labeled_count,
    }
    log.info("  %s: %d frames (%d labeled, %d negative)",
             match_name, stats["frames"], stats["labeled"], stats["negative"])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and convert TrackNet dataset")
    parser.add_argument("--output", type=str, default="data/tracknet_public",
                        help="Output directory")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only convert existing data")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to already-downloaded raw data (with --skip-download)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download
    if args.skip_download:
        raw_dir = Path(args.input) if args.input else output_dir / "raw"
        if not raw_dir.exists():
            log.error("Raw data not found: %s", raw_dir)
            sys.exit(1)
        log.info("Skipping download, using: %s", raw_dir)
    else:
        raw_dir = download_dataset(output_dir)

    # Step 2: Find CSV files
    csv_files = find_csv_files(raw_dir)
    if not csv_files:
        log.error("No CSV files found in %s", raw_dir)
        log.info("Expected structure: <raw>/<match>/<csv_file>.csv")
        sys.exit(1)

    # Step 3: Convert each match
    log.info("\n=== Converting matches ===")
    all_stats = []
    for i, csv_path in enumerate(csv_files):
        match_name = f"public_{i:03d}_{csv_path.parent.name}"
        frame_dir = find_frame_dir(csv_path)
        if frame_dir is None:
            log.warning("No frame directory found for %s, skipping", csv_path)
            continue

        stats = convert_match(csv_path, frame_dir, output_dir, match_name)
        all_stats.append(stats)

    # Summary
    total_frames = sum(s["frames"] for s in all_stats)
    total_labeled = sum(s.get("labeled", 0) for s in all_stats)
    log.info("\n=== Summary ===")
    log.info("Matches converted: %d", len(all_stats))
    log.info("Total frames: %d", total_frames)
    log.info("Labeled (ball visible): %d", total_labeled)
    log.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
