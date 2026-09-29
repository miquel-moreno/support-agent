"""Run the agent on one email against the synthetic shop, with the LLM from .env.

The shop is loaded into an in-memory SQLite database, so no PostgreSQL is needed.

uv run python -m scripts.try_agent --order PED-10001 "Quiero devolver mi pedido PED-10001"
uv run python -m scripts.try_agent --sender someone@example.com "¿Dónde está mi pedido?"
"""

import argparse
import asyncio
import json
import time

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from support_agent.adapters.db import Base, make_session_factory, replace_shop
from support_agent.adapters.llm import build_chat_model
from support_agent.core.config import get_settings
from support_agent.services.agent import build_agent
from support_agent.services.shop_data import generate_shop


async def run(body: str, *, sender: str | None, order: str | None) -> None:
    shop = generate_shop()
    if sender is None:
        # Write "as" the customer who placed the order (or the first customer).
        customer_id = next((o.customer_id for o in shop.orders if o.number == order), 1)
        sender = shop.customers[customer_id - 1].email

    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = make_session_factory(engine)
    async with sessions() as session:
        await replace_shop(session, shop)

    settings = get_settings()
    agent = build_agent(build_chat_model(settings), sessions)
    start = time.perf_counter()
    state = await agent.ainvoke({"sender": sender, "subject": "Consulta", "body": body})
    seconds = time.perf_counter() - start
    await engine.dispose()

    tool_calls = [c for m in state.get("messages", []) for c in getattr(m, "tool_calls", [])]
    summary = {
        "model": f"{settings.llm_provider}/{settings.llm_model}",
        "sender": sender,
        "category": state["category"],
        "tool_calls": [f"{c['name']}({json.dumps(c['args'])})" for c in tool_calls],
        "situation": state.get("situation"),
        "order": (state.get("order") or {}).get("number"),
        "policy": state.get("policy"),
        "drafts": state["drafts"],
        "issues": state["issues"],
        "seconds": round(seconds, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n--- draft ---\n" + state["draft"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("body")
    parser.add_argument("--sender")
    parser.add_argument("--order", help="write as the customer who placed this order")
    args = parser.parse_args()
    asyncio.run(run(args.body, sender=args.sender, order=args.order))


if __name__ == "__main__":
    main()
