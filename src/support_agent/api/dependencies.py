"""FastAPI dependencies. Tests override them to use SQLite and a scripted chat model."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from support_agent.adapters.llm import LLMError, build_chat_model
from support_agent.core.config import get_settings
from support_agent.core.errors import ServiceUnavailableError
from support_agent.services.agent import Agent, build_agent


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


def get_chat_model() -> BaseChatModel:
    # Built per request (it is cheap) so a missing key is a clear 503, not a crash at startup.
    try:
        return build_chat_model(get_settings())
    except LLMError as exc:
        raise ServiceUnavailableError(str(exc)) from exc


def get_agent(request: Request, model: Annotated[BaseChatModel, Depends(get_chat_model)]) -> Agent:
    # Compiling the graph is cheap; the checkpointer is shared, so any request can resume
    # a run that another request paused.
    return build_agent(
        model, request.app.state.session_factory, checkpointer=request.app.state.checkpointer
    )
