"""Load the synthetic shop (customers and orders) into the database.

uv run python -m scripts.seed_shop              # replaces the shop
uv run python -m scripts.seed_shop --if-empty   # only if there are no customers yet
"""

import argparse
import asyncio

from sqlalchemy import func, select

from support_agent.adapters.db import (
    CustomerRecord,
    make_engine,
    make_session_factory,
    replace_shop,
)
from support_agent.core.config import get_settings
from support_agent.services.shop_data import generate_shop


async def run(*, if_empty: bool = False) -> int:
    engine = make_engine(get_settings().database_url)
    try:
        async with make_session_factory(engine)() as session:
            stored = await session.scalar(select(func.count()).select_from(CustomerRecord))
            if if_empty and stored:
                print(f"{stored} customers already loaded: nothing to do")
                return 0
            shop = generate_shop()
            await replace_shop(session, shop)
            print(f"{len(shop.customers)} customers and {len(shop.orders)} orders loaded")
            return len(shop.orders)
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--if-empty", action="store_true")
    asyncio.run(run(if_empty=parser.parse_args().if_empty))


if __name__ == "__main__":
    main()
