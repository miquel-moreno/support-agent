"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from support_agent import __version__
from support_agent.adapters.db import make_engine, make_session_factory
from support_agent.api import demo, drafts, health
from support_agent.api.middleware import request_id_middleware
from support_agent.core.config import get_settings
from support_agent.core.errors import register_error_handlers
from support_agent.core.logging import configure_logging


async def open_checkpointer(database_url: str, stack: AsyncExitStack) -> BaseCheckpointSaver[str]:
    """Where paused runs are saved: PostgreSQL, or memory when the app runs on SQLite (tests)."""
    if not database_url.startswith("postgresql"):
        return InMemorySaver()
    plain_url = "postgresql://" + database_url.split("://", 1)[1]  # psycopg, not asyncpg
    saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(plain_url))
    await saver.setup()  # creates or upgrades LangGraph's own checkpoint tables
    return saver


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # One connection pool for the whole process, closed on shutdown.
    database_url = get_settings().database_url
    engine = make_engine(database_url)
    app.state.session_factory = make_session_factory(engine)
    async with AsyncExitStack() as stack:
        app.state.checkpointer = await open_checkpointer(database_url, stack)
        yield
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.middleware("http")(request_id_middleware)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(drafts.router)
    app.include_router(demo.router)
    return app


app = create_app()
