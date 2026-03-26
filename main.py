"""Entry point for the Tennis 3D Ball Tracking system."""

import logging
import signal
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

    logger.info("Starting server on %s:%d", config.server.host, config.server.port)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
