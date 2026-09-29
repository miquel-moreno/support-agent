"""Traces: what the agent did with one email, step by step.

A LangChain callback handler listens to the graph run and records every node, LLM
call (with tokens and latency) and tool call, in the order they started. The steps
are stored with the draft, so it is possible to see why the agent did what it did.
A small, self-hosted alternative to Langfuse.
"""

import json
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import LLMResult
from langgraph.errors import GraphInterrupt

MAX_TEXT = 4_000  # characters kept per input/output: enough to debug, bounded storage


@dataclass
class TraceStep:
    seq: int
    node: str
    kind: str  # "node", "llm" or "tool"
    name: str
    input: str | None = None
    output: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    error: str | None = None


def _message(message: BaseMessage) -> dict[str, Any]:
    data: dict[str, Any] = {"type": message.type, "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        data["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return _message(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(_jsonable(value), ensure_ascii=False)
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "…"


class TraceRecorder(AsyncCallbackHandler):
    """Collects the steps of one graph invocation (pass it in `config["callbacks"]`)."""

    def __init__(self, first_seq: int = 1) -> None:
        self.steps: list[TraceStep] = []
        self._open: dict[UUID, tuple[TraceStep, float]] = {}
        self._next_seq = first_seq

    def _start(self, run_id: UUID, **fields: Any) -> None:
        step = TraceStep(seq=self._next_seq, **fields)
        self._next_seq += 1
        self.steps.append(step)
        self._open[run_id] = (step, time.perf_counter())

    def _finish(self, run_id: UUID) -> TraceStep | None:
        if run_id not in self._open:
            return None
        step, started = self._open.pop(run_id)
        step.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return step

    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        if node and kwargs.get("name") == node:  # the node itself, not what runs inside it
            self._start(run_id, node=node, kind="node", name=node)

    async def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._finish(run_id)
        if step:
            step.output = _text(outputs)

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._finish(run_id)
        if step and isinstance(error, GraphInterrupt):
            step.output = "paused: waiting for a person to approve or reject"
        elif step:
            step.error = repr(error)[:MAX_TEXT]

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node", "")
        self._start(run_id, node=node, kind="llm", name="chat_model", input=_text(messages[0]))

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._finish(run_id)
        if step is None:
            return
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        step.output = _text(_message(message) if message is not None else generation.text)
        usage = getattr(message, "usage_metadata", None) or {}
        step.input_tokens = usage.get("input_tokens")
        step.output_tokens = usage.get("output_tokens")
        metadata = getattr(message, "response_metadata", None) or {}
        step.model = metadata.get("model_name") or metadata.get("model")

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._finish(run_id)
        if step:
            step.error = repr(error)[:MAX_TEXT]

    async def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node", "")
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._start(run_id, node=node, kind="tool", name=name, input=_text(inputs or input_str))

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._finish(run_id)
        if step:
            step.output = _text(output.content if isinstance(output, ToolMessage) else output)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._finish(run_id)
        if step:
            step.error = repr(error)[:MAX_TEXT]
