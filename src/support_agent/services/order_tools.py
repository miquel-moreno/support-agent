"""Tools the agent calls to look up orders.

They only ever return rows that exist in the database, so the model cannot invent an
order: whatever it writes about an order has to come from here. The sender's email is
injected by the graph (the model never sees or chooses it), and an order placed with
another email address is never disclosed.
"""

import json
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any, TypedDict

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from support_agent.adapters.db import (
    OrderRecord,
    find_orders_by_email,
    get_customer,
    get_order,
)
from support_agent.services.policy import OrderStatus

ORDER_NUMBER = re.compile(r"(?:PED)?\W*(\d{5})\b", re.IGNORECASE)
STATUS_ES = {
    OrderStatus.PENDING: "pendiente de envío",
    OrderStatus.SHIPPED: "enviado (en camino)",
    OrderStatus.DELIVERED: "entregado",
    OrderStatus.CANCELLED: "cancelado",
    OrderStatus.RETURNED: "devuelto",
}


class OrderFacts(TypedDict):
    """An order as the agent sees it: already formatted the way the reply should show it."""

    number: str
    status: str
    status_es: str
    ordered_on: str
    shipped_on: str | None
    delivered_on: str | None
    carrier: str | None
    tracking_code: str | None
    invoice_number: str
    total: str
    items: list[dict[str, Any]]


class LookupResult(TypedDict):
    """What a lookup found, for the graph's code (the model gets a text version)."""

    orders: list[OrderFacts]
    hidden: list[str]  # order numbers that exist but belong to another email address


def es_date(value: Any) -> str | None:
    return value.strftime("%d/%m/%Y") if value is not None else None


def parse_es_date(text: str | None) -> date | None:
    return datetime.strptime(text, "%d/%m/%Y").date() if text else None


def es_money(value: Decimal) -> str:
    return f"{value:.2f}".replace(".", ",") + " €"


def normalize_order_number(text: str) -> str | None:
    """'ped-10023', 'PED 10023' or '10023' -> 'PED-10023'."""
    match = ORDER_NUMBER.search(text)
    return f"PED-{match.group(1)}" if match else None


def order_facts(order: OrderRecord) -> OrderFacts:
    status = OrderStatus(order.status)
    return OrderFacts(
        number=order.number,
        status=status.value,
        status_es=STATUS_ES[status],
        ordered_on=es_date(order.ordered_on) or "",
        shipped_on=es_date(order.shipped_on),
        delivered_on=es_date(order.delivered_on),
        carrier=order.carrier,
        tracking_code=order.tracking_code,
        invoice_number=order.invoice_number,
        total=es_money(order.total),
        items=[
            {
                "name": i["name"],
                "quantity": i["quantity"],
                "unit_price": es_money(Decimal(i["unit_price"])),
            }
            for i in order.items
        ],
    )


def make_order_tools(sessions: async_sessionmaker[AsyncSession]) -> list[BaseTool]:
    @tool(response_format="content_and_artifact")
    async def find_order(
        order_number: str, sender: Annotated[str, InjectedState("sender")]
    ) -> tuple[str, LookupResult]:
        """Look up one order by the number the customer wrote (for example PED-10023)."""
        number = normalize_order_number(order_number)
        if number is None:
            return (
                f"{order_number!r} is not a valid order number (format PED-XXXXX).",
                LookupResult(orders=[], hidden=[]),
            )
        async with sessions() as session:
            order = await get_order(session, number)
            customer = await get_customer(session, order.customer_id) if order else None
        if order is None or customer is None:
            return f"There is no order {number}.", LookupResult(orders=[], hidden=[])
        if customer.email.lower() != sender.strip().lower():
            # Privacy: never reveal someone else's order, not even its status.
            return (
                f"Order {number} was placed with a different email address. "
                "Do not share any detail about it.",
                LookupResult(orders=[], hidden=[number]),
            )
        facts = order_facts(order)
        return json.dumps([facts], ensure_ascii=False), LookupResult(orders=[facts], hidden=[])

    @tool(response_format="content_and_artifact")
    async def find_my_orders(
        sender: Annotated[str, InjectedState("sender")],
    ) -> tuple[str, LookupResult]:
        """List the orders placed with the sender's email address, most recent first."""
        async with sessions() as session:
            orders = await find_orders_by_email(session, sender)
        if not orders:
            return (
                "There are no orders placed with the sender's email address.",
                LookupResult(orders=[], hidden=[]),
            )
        facts = [order_facts(o) for o in orders]
        return json.dumps(facts, ensure_ascii=False), LookupResult(orders=facts, hidden=[])

    return [find_order, find_my_orders]
