"""Emails in, drafts out, and the human decision on each draft."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from support_agent.adapters.db import DraftStatus, get_draft, get_trace, list_drafts
from support_agent.api.dependencies import get_agent, get_session
from support_agent.core.errors import NotFoundError
from support_agent.services.agent import Agent, ReviewDecision
from support_agent.services.inbox import IncomingEmail, decide, submit_email

router = APIRouter(tags=["drafts"])

Session = Annotated[AsyncSession, Depends(get_session)]
AgentDep = Annotated[Agent, Depends(get_agent)]


class EmailIn(BaseModel):
    sender: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=200)
    subject: str = Field(default="", max_length=300)
    body: str = Field(min_length=1, max_length=10_000)


class DraftOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: DraftStatus
    created_at: datetime
    sender: str
    subject: str
    body: str
    category: str
    situation: str | None
    order_number: str | None
    draft: str
    issues: list[str] = Field(description="Data in the draft that the order does not back")
    final_text: str | None
    edited: bool | None
    reject_reason: str | None
    decided_at: datetime | None

    @field_validator("created_at", "decided_at")
    @classmethod
    def in_utc(cls, value: datetime | None) -> datetime | None:
        # Stored in UTC; SQLite (tests) drops the time zone on the way back.
        return value.replace(tzinfo=UTC) if value and value.tzinfo is None else value


class ApproveIn(BaseModel):
    text: str | None = Field(
        default=None, min_length=1, max_length=10_000, description="Edited reply, if any"
    )


class RejectIn(BaseModel):
    reason: str = Field(min_length=1, max_length=1_000)


@router.post("/emails", status_code=status.HTTP_201_CREATED)
async def receive_email(email: EmailIn, session: Session, agent: AgentDep) -> DraftOut:
    """The agent drafts a reply and leaves it pending for a person to approve."""
    record = await submit_email(agent, session, IncomingEmail(**email.model_dump()))
    return DraftOut.model_validate(record)


@router.get("/drafts")
async def drafts(session: Session, status: DraftStatus | None = None) -> list[DraftOut]:
    """The review queue, newest first; `?status=pending` for the ones waiting."""
    return [DraftOut.model_validate(r) for r in await list_drafts(session, status)]


@router.get("/drafts/{draft_id}")
async def draft(draft_id: str, session: Session) -> DraftOut:
    record = await get_draft(session, draft_id)
    if record is None:
        raise NotFoundError(f"draft {draft_id} not found")
    return DraftOut.model_validate(record)


@router.post("/drafts/{draft_id}/approve")
async def approve(
    draft_id: str, session: Session, agent: AgentDep, body: ApproveIn | None = None
) -> DraftOut:
    """Send the reply (simulated): the draft as it is, or the edited text."""
    decision = ReviewDecision(action="approve", text=body.text if body else None, reason=None)
    return DraftOut.model_validate(await decide(agent, session, draft_id, decision))


@router.post("/drafts/{draft_id}/reject")
async def reject(draft_id: str, body: RejectIn, session: Session, agent: AgentDep) -> DraftOut:
    decision = ReviewDecision(action="reject", text=None, reason=body.reason)
    return DraftOut.model_validate(await decide(agent, session, draft_id, decision))


class TraceStepOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    seq: int
    node: str
    kind: str = Field(description="node, llm or tool")
    name: str
    input: str | None
    output: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    error: str | None


class TraceTotals(BaseModel):
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    seconds: float = Field(description="Time spent in graph nodes (the person's wait excluded)")


class TraceOut(BaseModel):
    draft_id: str
    totals: TraceTotals
    steps: list[TraceStepOut]


@router.get("/drafts/{draft_id}/trace")
async def trace(draft_id: str, session: Session) -> TraceOut:
    """Why the agent did what it did: every node, LLM call and tool call, in order."""
    if await get_draft(session, draft_id) is None:
        raise NotFoundError(f"draft {draft_id} not found")
    steps = [TraceStepOut.model_validate(s) for s in await get_trace(session, draft_id)]
    llm = [s for s in steps if s.kind == "llm"]
    totals = TraceTotals(
        llm_calls=len(llm),
        tool_calls=sum(s.kind == "tool" for s in steps),
        input_tokens=sum(s.input_tokens or 0 for s in llm),
        output_tokens=sum(s.output_tokens or 0 for s in llm),
        seconds=round(sum(s.latency_ms or 0 for s in steps if s.kind == "node") / 1000, 2),
    )
    return TraceOut(draft_id=draft_id, totals=totals, steps=steps)
