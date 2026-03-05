"""Entry point for the Tennis 3D Ball Tracking system."""

import logging
import signal
import sys

import uvicorn
from fastapi import FastAPI

from app.api.routes import router, set_orchestrator
from app.config import load_config
from app.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("main")


def create_app() -> FastAPI:
    config = load_config("config.yaml")
    orch = Orchestrator(config)
    set_orchestrator(orch)

    app = FastAPI(title="Tennis 3D Tracker")
    app.include_router(router)
    app.state.orchestrator = orch

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
