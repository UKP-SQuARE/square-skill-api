import logging
import os
from logging.config import fileConfig
from typing import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import square_skill_api.api.routes.query
from square_skill_api.api.routes.router import api_router
from square_skill_api.core.event_handlers import start_app_handler, stop_app_handler

logger = logging.getLogger(__name__)


def get_app(
    predict_fn: Callable,
    app_name: str = "square-skill-api",
    api_prefix: str = "",
    version: str = "0.1.0",
    add_cors: bool = True
) -> FastAPI:
    try:
        fileConfig(
            os.environ.get("LOGGING_CONFIG", "logging.conf"),
            disable_existing_loggers=False,
        )
    except:
        logger.info(
            "Failed to load 'logging.conf'. Continuing without configuring the server logger"
        )
    app = FastAPI(title=app_name, version=version)
    app.include_router(api_router, prefix=api_prefix)

    # Nothing done here currently.
    app.add_event_handler("startup", start_app_handler(app))
    app.add_event_handler("shutdown", stop_app_handler(app))

    app.dependency_overrides[
        square_skill_api.api.routes.query.predict
    ] = lambda: predict_fn

    if add_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app
