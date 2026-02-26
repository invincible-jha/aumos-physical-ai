"""AumOS Physical AI service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.observability import get_logger

from aumos_physical_ai.settings import Settings

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage service lifecycle — startup and shutdown."""
    logger.info(
        "Starting aumos-physical-ai",
        version="0.1.0",
        environment=settings.environment,
    )
    init_database(settings.database)
    # TODO: Initialize Kafka publisher
    # TODO: Initialize BlenderProc client
    # TODO: Initialize sensor simulator client
    yield
    logger.info("Shutting down aumos-physical-ai")


app = create_app(
    service_name="aumos-physical-ai",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[],
)

# Import and include routers
from aumos_physical_ai.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
