"""Simulate RTSP camera streams from MP4 files for testing.

Starts two ffmpeg processes that stream MP4 files as RTSP,
mimicking cam66 and cam68 for testing the real-time dashboard pipeline.

Requires: ffmpeg + mediamtx (RTSP server)

Quick setup:
    1. Download mediamtx: https://github.com/bluenviron/mediamtx/releases
    2. Extract to tools/mediamtx/
    3. Run: python -m tools.fake_rtsp_server

Or use FFmpeg's built-in RTSP output (no mediamtx needed):
    python -m tools.fake_rtsp_server --mode tcp

Usage:
    python -m tools.fake_rtsp_server
    python -m tools.fake_rtsp_server --video66 path/to/cam66.mp4 --video68 path/to/cam68.mp4
    python -m tools.fake_rtsp_server --loop    # loop videos indefinitely
"""

import argparse
import subprocess
import shutil
import signal
import sys
import time
import os
import threading


def find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg not found in PATH")
        sys.exit(1)
    return ffmpeg


def stream_rtsp_via_tcp(ffmpeg, video_path, port, path_name, loop=False, fps=25):
    """Stream MP4 as RTSP using ffmpeg's built-in RTSP server (TCP mode).

    This creates an RTSP server at rtsp://localhost:{port}/{path_name}
    """
    cmd = [ffmpeg]

    if loop:
        cmd += ["-stream_loop", "-1"]

    cmd += [
        "-re",  # read at real-time speed
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", "2M",
        "-r", str(fps),
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        f"rtsp://localhost:{port}/{path_name}",
    ]

    return cmd


def stream_via_http_mjpeg(ffmpeg, video_path, port, loop=False, fps=25):
    """Stream MP4 as MJPEG over HTTP (simplest, no RTSP server needed).

    Access at: http://localhost:{port}/
    Compatible with OpenCV VideoCapture.
    """
    cmd = [ffmpeg]

    if loop:
        cmd += ["-stream_loop", "-1"]

    cmd += [
        "-re",
        "-i", video_path,
        "-c:v", "mjpeg",
        "-q:v", "5",
        "-r", str(fps),
        "-f", "mpjpeg",
        "-an",
        f"http://localhost:{port}",
    ]

    return cmd


def stream_via_udp(ffmpeg, video_path, port, loop=False, fps=25):
    """Stream MP4 via UDP (low latency, OpenCV compatible with udp:// URL).

    Access at: udp://localhost:{port}
    """
    cmd = [ffmpeg]

    if loop:
        cmd += ["-stream_loop", "-1"]

    cmd += [
        "-re",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "mpegts",
        "-r", str(fps),
        f"udp://localhost:{port}?pkt_size=1316",
    ]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Simulate RTSP streams from MP4 files")
    parser.add_argument("--video66", default="uploads/cam66_20260323_184338.mp4",
                        help="Path to cam66 MP4 file")
    parser.add_argument("--video68", default="uploads/cam68_20260323_184338.mp4",
                        help="Path to cam68 MP4 file")
    parser.add_argument("--mode", choices=["udp", "rtsp", "http"], default="udp",
                        help="Stream mode: udp (simplest), rtsp (needs mediamtx), http (MJPEG)")
    parser.add_argument("--loop", action="store_true", help="Loop videos indefinitely")
    parser.add_argument("--fps", type=int, default=25, help="Output FPS")
    parser.add_argument("--port66", type=int, default=8554, help="Port for cam66 stream")
    parser.add_argument("--port68", type=int, default=8555, help="Port for cam68 stream")
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()

    # Verify video files exist
    for path in [args.video66, args.video68]:
        if not os.path.exists(path):
            print(f"ERROR: Video file not found: {path}")
            sys.exit(1)

    if args.mode == "udp":
        cmd66 = stream_via_udp(ffmpeg, args.video66, args.port66, args.loop, args.fps)
        cmd68 = stream_via_udp(ffmpeg, args.video68, args.port68, args.loop, args.fps)
        url66 = f"udp://localhost:{args.port66}"
        url68 = f"udp://localhost:{args.port68}"
    elif args.mode == "rtsp":
        cmd66 = stream_rtsp_via_tcp(ffmpeg, args.video66, args.port66, "cam66", args.loop, args.fps)
        cmd68 = stream_rtsp_via_tcp(ffmpeg, args.video68, args.port68, "cam68", args.loop, args.fps)
        url66 = f"rtsp://localhost:{args.port66}/cam66"
        url68 = f"rtsp://localhost:{args.port68}/cam68"
    elif args.mode == "http":
        cmd66 = stream_via_http_mjpeg(ffmpeg, args.video66, args.port66, args.loop, args.fps)
        cmd68 = stream_via_http_mjpeg(ffmpeg, args.video68, args.port68, args.loop, args.fps)
        url66 = f"http://localhost:{args.port66}"
        url68 = f"http://localhost:{args.port68}"

    print(f"Starting fake camera streams ({args.mode} mode)...")
    print(f"  cam66: {args.video66}")
    print(f"    -> {url66}")
    print(f"  cam68: {args.video68}")
    print(f"    -> {url68}")
    print()
    print(f"Update config.yaml to use these URLs:")
    print(f"  cam66 rtsp_url: {url66}")
    print(f"  cam68 rtsp_url: {url68}")
    print()
    print("Press Ctrl+C to stop")
    print()

    procs = []

    def start_stream(cmd, name):
        print(f"[{name}] Starting: {' '.join(cmd[:6])}...")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append((name, proc))

        # Monitor stderr in background
        def monitor(proc, name):
            for line in proc.stderr:
                line = line.decode("utf-8", errors="replace").strip()
                if "error" in line.lower() or "fail" in line.lower():
                    print(f"[{name}] {line}")

        t = threading.Thread(target=monitor, args=(proc, name), daemon=True)
        t.start()

    start_stream(cmd66, "cam66")
    time.sleep(0.5)
    start_stream(cmd68, "cam68")

    print(f"\nBoth streams running. Waiting...")

    # Wait for Ctrl+C
    try:
        while True:
            # Check if processes are still running
            for name, proc in procs:
                if proc.poll() is not None:
                    print(f"[{name}] Process exited with code {proc.returncode}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping streams...")
        for name, proc in procs:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[{name}] Stopped")


if __name__ == "__main__":
    main()
