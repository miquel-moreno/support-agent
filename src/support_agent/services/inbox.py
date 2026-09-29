"""The review queue: an email comes in, the agent drafts a reply and pauses; a person
approves (optionally editing it) or rejects it, and the agent resumes.

The draft id is the LangGraph thread id, so the checkpointer can resume exactly that
run, even after a restart. The drafts table is what people see and query, and every
run leaves its trace (see services/tracing.py).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from support_agent.adapters.db import (
    DraftRecord,
    DraftStatus,
    TraceStepRecord,
    count_trace_steps,
    get_draft,
)
from support_agent.core.errors import ConflictError, NotFoundError
from support_agent.services.agent import Agent, ReviewDecision
from support_agent.services.tracing import TraceRecorder, TraceStep


@dataclass(frozen=True)
class IncomingEmail:
    sender: str
    subject: str
    body: str


def _config(draft_id: str, recorder: TraceRecorder) -> RunnableConfig:
    return {"configurable": {"thread_id": draft_id}, "callbacks": [recorder]}


def _store_trace(session: AsyncSession, draft_id: str, steps: list[TraceStep]) -> None:
    session.add_all(TraceStepRecord(draft_id=draft_id, **vars(step)) for step in steps)


async def submit_email(agent: Agent, session: AsyncSession, email: IncomingEmail) -> DraftRecord:
    """Run the agent until it pauses for approval and store the draft as pending."""
    draft_id = str(uuid4())
    recorder = TraceRecorder()
    state = await agent.ainvoke(
        {"sender": email.sender, "subject": email.subject, "body": email.body},
        _config(draft_id, recorder),
    )
    order = state.get("order") or {}
    record = DraftRecord(
        id=draft_id,
        created_at=datetime.now(UTC),
        status=DraftStatus.PENDING.value,
        sender=email.sender,
        subject=email.subject,
        body=email.body,
        category=str(state["category"]),
        situation=str(state["situation"]) if state.get("situation") else None,
        order_number=order.get("number"),
        return_decision=(state.get("policy") or {}).get("decision"),
        draft=state["draft"],
        issues=state["issues"],
    )
    session.add(record)
    await session.flush()  # the draft row must exist before its trace steps
    _store_trace(session, draft_id, recorder.steps)
    await session.commit()
    return record


async def decide(
    agent: Agent, session: AsyncSession, draft_id: str, decision: ReviewDecision
) -> DraftRecord:
    """Resume the paused run with the person's decision and record the outcome."""
    record = await get_draft(session, draft_id)
    if record is None:
        raise NotFoundError(f"draft {draft_id} not found")
    if record.status != DraftStatus.PENDING:
        raise ConflictError(f"draft {draft_id} is already {record.status}")

    recorder = TraceRecorder(first_seq=await count_trace_steps(session, draft_id) + 1)
    state = await agent.ainvoke(Command(resume=decision), _config(draft_id, recorder))
    record.decided_at = datetime.now(UTC)
    if state.get("outcome") == "sent":
        record.status = DraftStatus.SENT.value
        record.final_text = state["final_text"]
        record.edited = state["final_text"].strip() != record.draft.strip()
    else:
        record.status = DraftStatus.REJECTED.value
        record.reject_reason = decision["reason"]
    _store_trace(session, draft_id, recorder.steps)
    await session.commit()
    return record
