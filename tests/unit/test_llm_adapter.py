import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from support_agent.adapters.llm import LLMError, ScriptedChatModel, build_chat_model
from support_agent.core.config import Settings


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def test_openai_needs_a_real_key() -> None:
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_chat_model(settings(llm_provider="openai"))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_chat_model(settings(llm_provider="openai", openai_api_key=SecretStr("change-me")))


def test_openai_model_from_settings() -> None:
    model = build_chat_model(
        settings(llm_provider="openai", llm_model="gpt-test", openai_api_key=SecretStr("k"))
    )
    assert isinstance(model, ChatOpenAI)
    assert (model.model_name, model.temperature) == ("gpt-test", 0)


def test_ollama_uses_its_openai_compatible_endpoint() -> None:
    model = build_chat_model(
        settings(llm_provider="ollama", llm_model="qwen2.5:3b", ollama_base_url="http://h:11434/")
    )
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://h:11434/v1"


def test_unknown_provider_is_an_error() -> None:
    with pytest.raises(LLMError, match="not implemented"):
        build_chat_model(settings(llm_provider="anthropic"))


class Answer(BaseModel):
    value: int


async def test_scripted_model_plays_back_replies_and_records_prompts() -> None:
    model = ScriptedChatModel(replies=[AIMessage("hello"), Answer(value=3)])

    first = await model.bind_tools([]).ainvoke([HumanMessage("hi")])
    second = await model.with_structured_output(Answer).ainvoke([HumanMessage("n?")])

    assert first.content == "hello"
    assert second == Answer(value=3)
    assert [[m.content for m in p] for p in model.prompts] == [["hi"], ["n?"]]
    with pytest.raises(LLMError, match="ran out"):
        await model.ainvoke([HumanMessage("more")])


async def test_scripted_model_rejects_a_reply_of_the_wrong_kind() -> None:
    model = ScriptedChatModel(replies=[AIMessage("text")])
    with pytest.raises(LLMError, match="expected a scripted Answer"):
        await model.with_structured_output(Answer).ainvoke([HumanMessage("n?")])
