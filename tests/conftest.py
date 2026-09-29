from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from support_agent.adapters.db import Base, make_session_factory
from support_agent.core.config import get_settings
from support_agent.main import create_app


@pytest.fixture
def database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A fresh SQLite file with every table, used by the app through DATABASE_URL."""
    path = (tmp_path / "app.sqlite").as_posix()
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()
    url = f"sqlite+aiosqlite:///{path}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


@pytest.fixture
def client(database_url: str) -> Iterator[TestClient]:
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A fresh in-memory SQLite database per test (no PostgreSQL needed in CI)."""
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with make_session_factory(engine)() as s:
        yield s
    await engine.dispose()
