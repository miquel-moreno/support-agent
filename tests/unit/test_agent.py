"""The agent graph end to end, with a scripted model and the synthetic shop in SQLite."""

from collections import Counter
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from support_agent.adapters.db import Base, make_session_factory, replace_shop
from support_agent.adapters.llm import ScriptedChatModel
from support_agent.services.agent import (
    MAX_LOOKUP_ROUNDS,
    AgentState,
    Category,
    EmailClassification,
    Situation,
    build_agent,
)
from support_agent.services.order_tools import es_date, es_money
from support_agent.services.policy import RETURN_WINDOW_DAYS, OrderStatus
from support_agent.services.shop_data import REFERENCE_DATE, Order, generate_shop

SHOP = generate_shop()
ORDERS_PER_CUSTOMER = Counter(o.customer_id for o in SHOP.orders)


def email_of(order: Order) -> str:
    return SHOP.customers[order.customer_id - 1].email


def delivered(*, returnable: bool) -> Order:
    window = timedelta(days=RETURN_WINDOW_DAYS)
    return next(
        o
        for o in SHOP.orders
        if o.status == OrderStatus.DELIVERED
        and o.delivered_on is not None
        and (o.delivered_on + window >= REFERENCE_DATE) == returnable
    )


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = make_session_factory(engine)
    async with factory() as session:
        await replace_shop(session, SHOP)
    yield factory
    await engine.dispose()


def classify(category: Category) -> EmailClassification:
    return EmailClassification(category=category)


def call(tool: str, n: int = 1, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": tool, "args": args, "id": f"call-{tool}-{n}"}])


async def run(
    sessions: async_sessionmaker[AsyncSession], replies: list[Any], sender: str, body: str
) -> tuple[AgentState, ScriptedChatModel]:
    model = ScriptedChatModel(replies=replies)
    agent = build_agent(model, sessions)
    state = await agent.ainvoke({"sender": sender, "subject": "Consulta", "body": body})
    assert not model.replies, "the graph did not use every scripted reply"
    return state, model


def draft_prompt(model: ScriptedChatModel, n: int = 1) -> str:
    """Text of the n-th prompt sent to the drafting step."""
    drafts = [p for p in model.prompts if "Facts (JSON)" in str(p[1].content)]
    return "\n".join(str(m.content) for m in drafts[n - 1])


async def test_return_within_the_window_is_allowed(sessions: Any) -> None:
    order = delivered(returnable=True)
    deadline = es_date(order.delivered_on + timedelta(days=RETURN_WINDOW_DAYS))  # type: ignore[operator]
    draft = f"Hola. Puedes devolver el pedido {order.number} hasta el {deadline}."

    state, model = await run(
        sessions,
        [
            classify(Category.RETURN_REQUEST),
            call("find_order", order_number=order.number.lower()),
            AIMessage("done"),
            AIMessage(draft),
        ],
        sender=email_of(order),
        body=f"Quiero devolver el pedido {order.number}.",
    )

    assert state["situation"] == Situation.ORDER_FOUND
    assert state["order"] is not None and state["order"]["number"] == order.number
    assert state["policy"] is not None and state["policy"]["decision"] == "allowed"
    assert (state["draft"], state["issues"], state["drafts"]) == (draft, [], 1)
    prompt = draft_prompt(model)
    assert '"decision": "allowed"' in prompt and order.invoice_number in prompt


async def test_return_after_the_window_is_refused_by_the_policy(sessions: Any) -> None:
    order = delivered(returnable=False)

    state, _ = await run(
        sessions,
        [
            classify(Category.RETURN_REQUEST),
            call("find_order", order_number=order.number),
            AIMessage("done"),
            AIMessage("Lo sentimos, el plazo de devolución ha terminado."),
        ],
        sender=email_of(order).upper(),  # the email match ignores case
        body=f"Devolución del {order.number}",
    )

    assert state["policy"] is not None and state["policy"]["decision"] == "window_expired"


async def test_someone_elses_order_is_never_disclosed(sessions: Any) -> None:
    order = SHOP.orders[0]
    stranger = next(c.email for c in SHOP.customers if c.id != order.customer_id)

    state, model = await run(
        sessions,
        [
            classify(Category.ORDER_STATUS),
            call("find_order", order_number=order.number),
            AIMessage("done"),
            AIMessage("Escríbenos desde el email con el que hiciste la compra."),
        ],
        sender=stranger,
        body=f"¿Dónde está el pedido {order.number}?",
    )

    assert (state["situation"], state["order"]) == (Situation.FOREIGN_ORDER, None)
    tool_reply = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    everything_the_model_saw = str(model.prompts) + str(tool_reply.content)
    assert order.invoice_number not in everything_the_model_saw
    assert es_money(order.total) not in everything_the_model_saw


async def test_several_orders_and_no_number_asks_which_one(sessions: Any) -> None:
    customer_id = next(c for c, n in ORDERS_PER_CUSTOMER.items() if n > 1)
    order = next(o for o in SHOP.orders if o.customer_id == customer_id)

    state, model = await run(
        sessions,
        [
            classify(Category.ORDER_STATUS),
            call("find_my_orders"),
            AIMessage("done"),
            AIMessage("¿A cuál de tus pedidos te refieres?"),
        ],
        sender=email_of(order),
        body="¿Dónde está mi pedido?",
    )

    assert state["situation"] == Situation.CHOOSE_ORDER
    assert len(state["candidates"]) == ORDERS_PER_CUSTOMER[customer_id]
    assert order.number in draft_prompt(model)


async def test_a_single_order_by_email_is_used_directly(sessions: Any) -> None:
    customer_id = next(c for c, n in ORDERS_PER_CUSTOMER.items() if n == 1)
    order = next(o for o in SHOP.orders if o.customer_id == customer_id)

    state, _ = await run(
        sessions,
        [
            classify(Category.INVOICE_PROBLEM),
            call("find_my_orders"),
            AIMessage("done"),
            AIMessage(f"Revisaremos la factura {order.invoice_number}."),
        ],
        sender=email_of(order),
        body="La factura tiene mal el nombre.",
    )

    assert state["situation"] == Situation.ORDER_FOUND
    assert state["policy"] is None  # only returns go through the return policy
    assert state["issues"] == []


async def test_unknown_order_asks_for_the_number(sessions: Any) -> None:
    state, _ = await run(
        sessions,
        [
            classify(Category.ORDER_STATUS),
            call("find_order", order_number="PED-99999"),
            call("find_my_orders", n=2),
            AIMessage("done"),
            AIMessage("¿Nos indicas el número de pedido (PED-XXXXX)?"),
        ],
        sender="nobody@example.com",
        body="¿Dónde está el pedido PED-99999?",
    )

    assert (state["situation"], state["order"], state["candidates"]) == (
        Situation.ASK_ORDER_NUMBER,
        None,
        [],
    )
    assert state["issues"] == []  # "PED-XXXXX" is the format example, not an order number


async def test_a_number_the_customer_did_not_write_is_not_looked_up(sessions: Any) -> None:
    # Seen with a real 3B model: no number in the email, so it made one up.
    customer_id = next(c for c, n in ORDERS_PER_CUSTOMER.items() if n == 1)
    own = next(o for o in SHOP.orders if o.customer_id == customer_id)
    invented = next(o for o in SHOP.orders if o.customer_id != customer_id)

    state, _ = await run(
        sessions,
        [
            classify(Category.ORDER_STATUS),
            call("find_order", order_number=invented.number),
            call("find_my_orders", n=2),
            AIMessage("done"),
            AIMessage(f"Tu pedido {own.number} está {own.status.value}."),
        ],
        sender=email_of(own),
        body="Hice un pedido hace unos días y no me ha llegado.",
    )

    refusal = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert "did not write" in str(refusal.content)
    assert state["situation"] == Situation.ORDER_FOUND
    assert state["order"] is not None and state["order"]["number"] == own.number


async def test_other_emails_skip_the_lookup(sessions: Any) -> None:
    state, model = await run(
        sessions,
        [classify(Category.OTHER), AIMessage("Un compañero te responderá.")],
        sender="nobody@example.com",
        body="¿Vendéis neumáticos de invierno?",
    )

    assert "messages" not in state or state["messages"] == []
    assert len(model.prompts) == 2  # classify + draft


async def test_invented_data_gets_one_rewrite(sessions: Any) -> None:
    order = SHOP.orders[0]
    fixed = f"Tu pedido {order.number} está {order.status.value}."

    state, model = await run(
        sessions,
        [
            classify(Category.ORDER_STATUS),
            call("find_order", order_number=order.number),
            AIMessage("done"),
            AIMessage("Tu pedido llegará el 31/12/2026."),
            AIMessage(fixed),
        ],
        sender=email_of(order),
        body=f"¿Y mi {order.number}?",
    )

    assert (state["draft"], state["issues"], state["drafts"]) == (fixed, [], 2)
    assert "date 31/12/2026 does not come from" in draft_prompt(model, 2)


async def test_issues_that_survive_the_rewrite_are_kept_for_the_reviewer(sessions: Any) -> None:
    order = SHOP.orders[0]
    invented = "Te devolvemos 999,99 €."

    state, _ = await run(
        sessions,
        [
            classify(Category.ORDER_STATUS),
            call("find_order", order_number=order.number),
            AIMessage("done"),
            AIMessage(invented),
            AIMessage(invented),
        ],
        sender=email_of(order),
        body=f"¿Y mi {order.number}?",
    )

    assert state["drafts"] == 2
    assert state["issues"] == ["amount 999,99 € does not come from the order"]


async def test_the_lookup_loop_is_capped(sessions: Any) -> None:
    order = SHOP.orders[0]
    loops = [call("find_my_orders", n=i) for i in range(MAX_LOOKUP_ROUNDS + 1)]

    state, _ = await run(
        sessions,
        [classify(Category.ORDER_STATUS), *loops, AIMessage("Hola.")],
        sender=email_of(order),
        body="¿Mi pedido?",
    )

    tool_replies = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_replies) == MAX_LOOKUP_ROUNDS
