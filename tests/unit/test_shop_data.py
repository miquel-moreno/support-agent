from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest
from scripts import seed_shop
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from support_agent.adapters.db import (
    Base,
    find_orders_by_email,
    get_customer,
    get_order,
    replace_shop,
)
from support_agent.core.config import get_settings
from support_agent.services.policy import OrderStatus
from support_agent.services.shop_data import REFERENCE_DATE, generate_shop

SHOP = generate_shop()


def test_same_seed_same_shop() -> None:
    assert generate_shop() == SHOP
    assert generate_shop(seed=8) != SHOP


def test_shop_size_and_unique_invented_emails() -> None:
    emails = [c.email for c in SHOP.customers]

    assert (len(SHOP.customers), len(SHOP.orders)) == (30, 80)
    assert len(set(emails)) == len(emails)
    # Faker's safe_email only uses example.com/org/net: never a real inbox.
    assert all(e.split("@")[1] in {"example.com", "example.org", "example.net"} for e in emails)


def test_every_status_appears() -> None:
    assert set(Counter(o.status for o in SHOP.orders)) == set(OrderStatus)


@pytest.mark.parametrize("order", SHOP.orders, ids=lambda o: o.number)
def test_dates_are_consistent_with_the_status(order: object) -> None:
    from support_agent.services.shop_data import Order

    assert isinstance(order, Order)
    assert order.ordered_on <= REFERENCE_DATE
    shipped = order.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED, OrderStatus.RETURNED)
    delivered = order.status in (OrderStatus.DELIVERED, OrderStatus.RETURNED)
    assert (order.shipped_on is not None) == shipped
    assert (order.carrier is not None) == shipped
    assert (order.delivered_on is not None) == delivered
    if order.shipped_on:
        assert order.ordered_on <= order.shipped_on <= REFERENCE_DATE
    if order.delivered_on and order.shipped_on:
        assert order.shipped_on <= order.delivered_on <= REFERENCE_DATE
    assert order.items and order.total == sum(i.unit_price * i.quantity for i in order.items)
    assert all(i.unit_price == i.unit_price.quantize(Decimal("0.01")) for i in order.items)


async def test_repository_loads_and_finds_orders(session: AsyncSession) -> None:
    await replace_shop(session, SHOP)
    first = SHOP.orders[0]
    customer = SHOP.customers[first.customer_id - 1]

    found = await get_order(session, f"  {first.number.lower()} ")  # tolerant lookup
    by_email = await find_orders_by_email(session, customer.email.upper())

    assert found is not None and found.total == first.total
    assert found.status == first.status.value
    assert first.number in [o.number for o in by_email]
    assert all(o.customer_id == customer.id for o in by_email)
    assert await get_order(session, "PED-99999") is None
    assert await find_orders_by_email(session, "nobody@example.com") == []
    stored_customer = await get_customer(session, customer.id)
    assert stored_customer is not None and stored_customer.email == customer.email


async def test_replacing_the_shop_twice_does_not_duplicate(session: AsyncSession) -> None:
    await replace_shop(session, SHOP)
    await replace_shop(session, SHOP)

    customer = SHOP.customers[0]
    orders = await find_orders_by_email(session, customer.email)
    assert len(orders) == sum(o.customer_id == customer.id for o in SHOP.orders)


async def test_seed_script_only_loads_an_empty_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'db.sqlite').as_posix()}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        first = await seed_shop.run(if_empty=True)
        second = await seed_shop.run(if_empty=True)
    finally:
        get_settings.cache_clear()

    assert (first, second) == (80, 0)
