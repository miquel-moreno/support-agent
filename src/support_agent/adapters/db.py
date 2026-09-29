"""Database access: SQLAlchemy 2 async engine and the declarative base.

PostgreSQL in production (asyncpg), SQLite (aiosqlite) in tests: models should use
portable column types so both work. Every model change needs an Alembic migration
(tests/integration/test_migrations.py checks it).
"""

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
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
