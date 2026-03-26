"""Thread-based RTSP camera frame reader."""

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """Reads frames from an RTSP camera in a background thread."""

    def __init__(self, url: str, name: str, reconnect_delay: float = 3.0):
        self.url = url
        self.name = name
        self.reconnect_delay = reconnect_delay

        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_id: int = 0
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def frame_id(self) -> int:
        return self._frame_id

    def start(self) -> "CameraStream":
        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _open(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                # Set buffer size and timeouts for stable RTSP reading
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                # Use TCP transport for RTSP (more reliable than UDP)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                self._cap = cap
                logger.info("[%s] Connected to %s", self.name, self.url)
                return True
            cap.release()
        except Exception as e:
            logger.error("[%s] Failed to open stream: %s", self.name, e)
        return False

    def _run(self) -> None:
        while not self._stopped.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open():
                    logger.warning(
                        "[%s] Reconnecting in %.1fs...",
                        self.name,
                        self.reconnect_delay,
                    )
                    self._stopped.wait(self.reconnect_delay)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("[%s] Frame read failed, reconnecting...", self.name)
                self._cap.release()
                self._cap = None
                self._stopped.wait(self.reconnect_delay)
                continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._timestamp = time.time()

    def read(self) -> tuple[np.ndarray | None, int, float]:
        """Return (frame, frame_id, timestamp). Thread-safe."""
        with self._lock:
            if self._frame is None:
                return None, 0, 0.0
            return self._frame.copy(), self._frame_id, self._timestamp

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("[%s] Stream stopped", self.name)
