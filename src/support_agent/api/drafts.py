"""Emails in, drafts out, and the human decision on each draft."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from support_agent.adapters.db import DraftStatus, get_draft, list_drafts
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
