"""Synthetic shop: customers and orders generated with Faker and a fixed seed.

Everything is invented (no real people or orders). The same seed and reference
date always give the same shop, so the evaluation is reproducible.
"""

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from faker import Faker

from support_agent.services.policy import OrderStatus

DEFAULT_SEED = 7
CENT = Decimal("0.01")
REFERENCE_DATE = date(2026, 9, 28)  # the shop's "today" for the demo and the evaluation
CARRIERS = ("SEUR", "MRW", "Correos Express", "GLS")
PRODUCTS: tuple[tuple[str, int, int], ...] = (
    ("Filtro de aceite", 8, 25),
    ("Pastillas de freno", 25, 90),
    ("Batería 12V 70Ah", 80, 160),
    ("Neumático 205/55 R16", 55, 130),
    ("Aceite motor 5W30 (5 L)", 25, 60),
    ("Escobillas limpiaparabrisas", 8, 30),
    ("Lámpara H7", 3, 15),
    ("Kit de embrague", 120, 350),
    ("Amortiguador trasero", 40, 110),
    ("Correa de distribución", 30, 120),
)
# Relative weights of each status, roughly how a small shop's orders look.
STATUS_WEIGHTS = {
    OrderStatus.DELIVERED: 55,
    OrderStatus.SHIPPED: 15,
    OrderStatus.PENDING: 10,
    OrderStatus.CANCELLED: 10,
    OrderStatus.RETURNED: 10,
}


@dataclass(frozen=True)
class Customer:
    id: int
    name: str
    email: str


@dataclass(frozen=True)
class OrderItem:
    name: str
    quantity: int
    unit_price: Decimal


@dataclass(frozen=True)
class Order:
    number: str
    customer_id: int
    status: OrderStatus
    ordered_on: date
    shipped_on: date | None
    delivered_on: date | None
    carrier: str | None
    tracking_code: str | None
    invoice_number: str
    items: list[OrderItem] = field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((i.unit_price * i.quantity for i in self.items), Decimal("0.00"))


@dataclass(frozen=True)
class Shop:
    customers: list[Customer]
    orders: list[Order]


def generate_shop(
    *, seed: int = DEFAULT_SEED, customers: int = 30, orders: int = 80, today: date = REFERENCE_DATE
) -> Shop:
    rng = random.Random(seed)  # noqa: S311 - invented data, not security
    fake = Faker("es_ES")
    fake.seed_instance(seed)

    people = []
    for customer_id in range(1, customers + 1):
        name = fake.name()
        email = fake.unique.safe_email()  # example.com/org/net: never a real inbox
        people.append(Customer(customer_id, name, email))

    result = []
    statuses = list(STATUS_WEIGHTS)
    weights = list(STATUS_WEIGHTS.values())
    for index in range(orders):
        status = rng.choices(statuses, weights=weights)[0]
        ordered_on = today - timedelta(days=rng.randint(1, 75))
        shipped_on = delivered_on = None
        carrier = tracking = None
        if status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED, OrderStatus.RETURNED):
            shipped_on = min(ordered_on + timedelta(days=rng.randint(1, 3)), today)
            carrier = rng.choice(CARRIERS)
            tracking = f"{carrier[:3].upper()}{rng.randint(10**8, 10**9 - 1)}"
        if status in (OrderStatus.DELIVERED, OrderStatus.RETURNED) and shipped_on is not None:
            delivered_on = min(shipped_on + timedelta(days=rng.randint(1, 4)), today)
        items = [
            OrderItem(
                name,
                rng.randint(1, 4),
                (Decimal(rng.randint(low * 100, high * 100)) / 100).quantize(CENT),
            )
            for name, low, high in rng.sample(PRODUCTS, rng.randint(1, 3))
        ]
        result.append(
            Order(
                number=f"PED-{10001 + index}",
                customer_id=rng.randint(1, customers),
                status=status,
                ordered_on=ordered_on,
                shipped_on=shipped_on,
                delivered_on=delivered_on,
                carrier=carrier,
                tracking_code=tracking,
                invoice_number=f"F-{ordered_on.year}-{5001 + index:05d}",
                items=items,
            )
        )
    return Shop(people, result)
