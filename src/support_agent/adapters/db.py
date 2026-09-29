"""Database access: SQLAlchemy 2 async engine and the declarative base.

PostgreSQL in production (asyncpg), SQLite (aiosqlite) in tests: models should use
portable column types so both work. Every model change needs an Alembic migration
(tests/integration/test_migrations.py checks it).
"""

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    delete,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from support_agent.services.shop_data import Shop


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


# --- Shop ------------------------------------------------------------------------


class CustomerRecord(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)


class OrderRecord(Base):
    __tablename__ = "orders"

    number: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String(20))
    ordered_on: Mapped[date] = mapped_column(Date)
    shipped_on: Mapped[date | None] = mapped_column(Date)
    delivered_on: Mapped[date | None] = mapped_column(Date)
    carrier: Mapped[str | None] = mapped_column(String(50))
    tracking_code: Mapped[str | None] = mapped_column(String(50))
    invoice_number: Mapped[str] = mapped_column(String(30))
    total: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON)


async def replace_shop(session: AsyncSession, shop: Shop) -> None:
    """Load a generated shop, replacing any previous one (idempotent)."""
    await session.execute(delete(OrderRecord))
    await session.execute(delete(CustomerRecord))
    session.add_all(CustomerRecord(id=c.id, name=c.name, email=c.email) for c in shop.customers)
    await session.flush()
    session.add_all(
        OrderRecord(
            number=o.number,
            customer_id=o.customer_id,
            status=o.status.value,
            ordered_on=o.ordered_on,
            shipped_on=o.shipped_on,
            delivered_on=o.delivered_on,
            carrier=o.carrier,
            tracking_code=o.tracking_code,
            invoice_number=o.invoice_number,
            total=o.total,
            items=[
                {"name": i.name, "quantity": i.quantity, "unit_price": str(i.unit_price)}
                for i in o.items
            ],
        )
        for o in shop.orders
    )
    await session.commit()


async def get_order(session: AsyncSession, number: str) -> OrderRecord | None:
    return await session.get(OrderRecord, number.strip().upper())


async def find_orders_by_email(session: AsyncSession, email: str) -> Sequence[OrderRecord]:
    result = await session.execute(
        select(OrderRecord)
        .join(CustomerRecord, CustomerRecord.id == OrderRecord.customer_id)
        .where(func.lower(CustomerRecord.email) == email.strip().lower())
        .order_by(OrderRecord.ordered_on.desc(), OrderRecord.number)
    )
    return result.scalars().all()


async def get_customer(session: AsyncSession, customer_id: int) -> CustomerRecord | None:
    return await session.get(CustomerRecord, customer_id)


# --- Drafts: the human review queue -----------------------------------------------


class DraftStatus(StrEnum):
    PENDING = "pending"  # waiting for a person
    SENT = "sent"  # approved; sending is simulated
    REJECTED = "rejected"


class DraftRecord(Base):
    """One incoming email and its reply. `id` is also the LangGraph thread id."""

    __tablename__ = "drafts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), index=True)
    sender: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))
    situation: Mapped[str | None] = mapped_column(String(30))
    order_number: Mapped[str | None] = mapped_column(String(20))
    draft: Mapped[str] = mapped_column(Text)
    issues: Mapped[list[str]] = mapped_column(JSON)
    final_text: Mapped[str | None] = mapped_column(Text)
    edited: Mapped[bool | None] = mapped_column(Boolean)  # approved with changes?
    reject_reason: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


async def get_draft(session: AsyncSession, draft_id: str) -> DraftRecord | None:
    return await session.get(DraftRecord, draft_id)


async def list_drafts(
    session: AsyncSession, status: DraftStatus | None = None
) -> Sequence[DraftRecord]:
    query = select(DraftRecord).order_by(DraftRecord.created_at.desc())
    if status is not None:
        query = query.where(DraftRecord.status == status.value)
    return (await session.execute(query)).scalars().all()


# --- Traces: what the agent did, step by step --------------------------------------


class TraceStepRecord(Base):
    __tablename__ = "trace_steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    draft_id: Mapped[str] = mapped_column(ForeignKey("drafts.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(String(50))
    kind: Mapped[str] = mapped_column(String(10))  # node, llm or tool
    name: Mapped[str] = mapped_column(String(100))
    input: Mapped[str | None] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(100))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)


async def count_trace_steps(session: AsyncSession, draft_id: str) -> int:
    query = select(func.count()).where(TraceStepRecord.draft_id == draft_id)
    return int(await session.scalar(query) or 0)


async def get_trace(session: AsyncSession, draft_id: str) -> Sequence[TraceStepRecord]:
    query = (
        select(TraceStepRecord)
        .where(TraceStepRecord.draft_id == draft_id)
        .order_by(TraceStepRecord.seq)
    )
    return (await session.execute(query)).scalars().all()
