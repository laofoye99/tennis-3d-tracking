"""Entry point for the Tennis 3D Ball Tracking system."""

import logging
import signal
import socket
import sys

import uvicorn
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, set_orchestrator
from app.config import load_config
from app.orchestrator import Orchestrator

import datetime as _dt

# Log to both console and file
_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"tennis_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_log_file), encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")
logger.info("Log file: %s", _log_file)


class _AccessNoiseFilter(logging.Filter):
    """Suppress very chatty dashboard polling from uvicorn access logs."""

    _suppressed_fragments = (
        '"GET /api/status HTTP/1.1" 200',
        '"GET /api/dashboard/status HTTP/1.1" 200',
        '"GET /api/recording/status HTTP/1.1" 200',
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(fragment in msg for fragment in self._suppressed_fragments)


logging.getLogger("uvicorn.access").addFilter(_AccessNoiseFilter())


def create_app() -> FastAPI:
    config = load_config("config.yaml")
    orch = Orchestrator(config)
    set_orchestrator(orch)

    app = FastAPI(title="Tennis 3D Tracker")
    app.include_router(router)
    app.state.orchestrator = orch

    # Serve uploaded videos
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

    # Serve generated reports
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    app.mount("/reports", StaticFiles(directory=str(reports_dir), html=True), name="reports")

    @app.on_event("shutdown")
    def on_shutdown():
        logger.info("Shutting down orchestrator...")
        orch.shutdown()

    return app


def main() -> None:
    config = load_config("config.yaml")
    app = create_app()

    def handle_signal(sig, frame):
        logger.info("Received signal %s, shutting down...", sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    host = config.server.host
    port = config.server.port
    if host == "0.0.0.0" and socket.has_dualstack_ipv6():
        try:
            sock = socket.create_server(
                ("::", port),
                family=socket.AF_INET6,
                backlog=2048,
                dualstack_ipv6=True,
            )
            logger.info(
                "Starting dual-stack server on [::]:%d (localhost IPv6 + IPv4)",
                port,
            )
            server = uvicorn.Server(uvicorn.Config(app))
            server.run(sockets=[sock])
            return
        except OSError:
            logger.warning(
                "Dual-stack bind failed; falling back to %s:%d",
                host,
                port,
                exc_info=True,
            )

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
