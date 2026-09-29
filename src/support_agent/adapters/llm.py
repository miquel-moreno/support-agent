"""LLM access through LangChain chat models, chosen by LLM_PROVIDER in .env.

OpenAI and Ollama both speak the OpenAI Chat Completions API, so ChatOpenAI serves
both: only the base URL, key and model change. Tests use ScriptedChatModel, so CI
never calls a real LLM.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from support_agent.core.config import Settings

PLACEHOLDER_KEY = "change-me"


class LLMError(Exception):
    """The provider failed, is misconfigured or returned something unusable."""


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Pick the chat model from LLM_PROVIDER (temperature 0: same email, same draft)."""
    if settings.llm_provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if key in ("", PLACEHOLDER_KEY):
            raise LLMError("OPENAI_API_KEY is not set in .env")
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=SecretStr(key),
            base_url=settings.openai_base_url,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "ollama":
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=SecretStr("ollama"),  # Ollama ignores it, but the header must be present
            base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
            temperature=0,
            timeout=settings.llm_timeout_seconds,
        )
    raise LLMError(f"LLM provider {settings.llm_provider!r} is not implemented yet")


class ScriptedChatModel(BaseChatModel):
    """Plays back scripted replies in order and records every prompt. For tests only.

    Each reply is an AIMessage (plain answer or tool calls) or, for
    `with_structured_output`, the Pydantic object the real model would return.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    replies: list[Any]  # AIMessage | BaseModel; Any so Pydantic keeps them as they are
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self, expected: type) -> Any:
        if not self.replies:
            raise LLMError("ScriptedChatModel ran out of replies")
        reply = self.replies.pop(0)
        if not isinstance(reply, expected):
            raise LLMError(f"expected a scripted {expected.__name__}, got {reply!r}")
        return reply

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next(AIMessage))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def with_structured_output(  # type: ignore[override]
        self, schema: type[BaseModel], **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseModel]:
        def reply(messages: Any) -> BaseModel:
            self.prompts.append(list(messages))
            result: BaseModel = self._next(schema)
            return result

        return RunnableLambda(reply)
