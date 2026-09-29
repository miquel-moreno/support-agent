import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import Generation, LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from support_agent.adapters.db import Base, make_session_factory, replace_shop
from support_agent.adapters.llm import ScriptedChatModel
from support_agent.services.agent import Category, EmailClassification, build_agent
from support_agent.services.shop_data import generate_shop
from support_agent.services.tracing import MAX_TEXT, TraceRecorder

SHOP = generate_shop()
ORDER = SHOP.orders[0]
USAGE = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = make_session_factory(engine)
    async with factory() as session:
        await replace_shop(session, SHOP)
    yield factory
    await engine.dispose()


def reply(content: str = "", **kwargs: Any) -> AIMessage:
    return AIMessage(
        content,
        usage_metadata=USAGE,  # type: ignore[arg-type]
        response_metadata={"model_name": "fake-model"},
        **kwargs,
    )


async def test_a_run_leaves_every_step_in_order(sessions: Any) -> None:
    model = ScriptedChatModel(
        replies=[
            EmailClassification(category=Category.ORDER_STATUS),
            reply(
                tool_calls=[
                    {"name": "find_order", "args": {"order_number": ORDER.number}, "id": "1"}
                ]
            ),
            reply("done"),
            reply("Hola, tu pedido va en camino."),
        ]
    )
    recorder = TraceRecorder()
    agent = build_agent(model, sessions, checkpointer=InMemorySaver())
    await agent.ainvoke(
        {
            "sender": SHOP.customers[ORDER.customer_id - 1].email,
            "subject": "",
            "body": ORDER.number,
        },
        {"configurable": {"thread_id": "t"}, "callbacks": [recorder]},
    )
    steps = recorder.steps

    assert [s.seq for s in steps] == list(range(1, len(steps) + 1))
    assert [s.name for s in steps if s.kind == "node"] == [
        *("classify", "lookup", "tools", "lookup", "resolve", "policy"),
        *("draft", "review", "approval"),
    ]
    tool = next(s for s in steps if s.kind == "tool")
    assert (tool.node, tool.name) == ("tools", "find_order")
    assert json.loads(tool.input or "") == {"order_number": ORDER.number}  # no email data
    assert ORDER.invoice_number in (tool.output or "")
    llm = [s for s in steps if s.kind == "llm"]
    assert [s.node for s in llm] == ["lookup", "lookup", "draft"]
    assert all((s.input_tokens, s.output_tokens, s.model) == (120, 30, "fake-model") for s in llm)
    assert "find_order" in (llm[0].output or "") and "Hola" in (llm[2].output or "")
    assert steps[-1].output == "paused: waiting for a person to approve or reject"
    assert all(s.latency_ms is not None and s.error is None for s in steps)


async def test_errors_and_long_texts_are_recorded() -> None:
    recorder = TraceRecorder(first_seq=10)
    node, llm, tool = uuid4(), uuid4(), uuid4()

    await recorder.on_chain_start(None, {}, run_id=node, metadata={"langgraph_node": "x"}, name="x")
    await recorder.on_chain_start(None, {}, run_id=uuid4(), metadata={"langgraph_node": "x"})
    await recorder.on_chat_model_start(None, [[HumanMessage("a" * (MAX_TEXT + 50))]], run_id=llm)
    await recorder.on_llm_error(TimeoutError("slow"), run_id=llm)
    await recorder.on_tool_start({"name": "t"}, "raw input", run_id=tool)
    await recorder.on_tool_end("plain output", run_id=tool)
    await recorder.on_chain_error(ValueError("boom"), run_id=node)
    await recorder.on_llm_end(LLMResult(generations=[[Generation(text="x")]]), run_id=uuid4())

    first, model_call, tool_call = recorder.steps
    assert [s.seq for s in recorder.steps] == [10, 11, 12]
    assert "boom" in (first.error or "")
    assert "slow" in (model_call.error or "") and (model_call.input or "").endswith("…")
    assert (tool_call.input, tool_call.output) == ("raw input", "plain output")


async def test_plain_text_generations_and_tool_errors() -> None:
    recorder = TraceRecorder()
    llm, tool, node = uuid4(), uuid4(), uuid4()

    await recorder.on_chat_model_start(None, [[HumanMessage("hi")]], run_id=llm)
    await recorder.on_llm_end(LLMResult(generations=[[Generation(text="plain")]]), run_id=llm)
    await recorder.on_tool_start(None, "x", run_id=tool, name="find_my_orders")
    await recorder.on_tool_error(RuntimeError("db down"), run_id=tool)
    await recorder.on_chain_start(None, {}, run_id=node, metadata={"langgraph_node": "n"}, name="n")
    await recorder.on_chain_error(GraphInterrupt(), run_id=node)
    await recorder.on_chain_end({}, run_id=uuid4())
    await recorder.on_tool_end("late", run_id=uuid4())
    await recorder.on_llm_error(RuntimeError("late"), run_id=uuid4())
    await recorder.on_tool_error(RuntimeError("late"), run_id=uuid4())

    model_call, tool_call, paused = recorder.steps
    assert (model_call.output, model_call.input_tokens) == ("plain", None)
    assert tool_call.name == "find_my_orders" and "db down" in (tool_call.error or "")
    assert paused.output is not None and paused.output.startswith("paused")
