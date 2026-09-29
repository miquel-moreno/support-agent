"""Minimal demo page (the one in the README GIF). It only calls the public API."""

import json
from functools import cache
from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from support_agent.services.shop_data import generate_shop

router = APIRouter(include_in_schema=False)

_PAGE = files("support_agent").joinpath("static/index.html")

# (button label, write as the customer of this order (or None: a stranger), subject, body)
_EXAMPLES = [
    (
        "Devolución en plazo",
        "PED-10001",
        "Devolución",
        "Hola, me equivoqué de pieza. Quiero devolver el pedido PED-10001.",
    ),
    (
        "Fuera de plazo",
        "PED-10007",
        "Devolución",
        "Buenas, quiero devolver el pedido PED-10007, ya no lo necesito.",
    ),
    (
        "¿Dónde está mi pedido?",
        "PED-10004",
        "Mi pedido",
        "Hola, ¿por dónde va mi pedido PED-10004? Gracias.",
    ),
    ("Pedido de otra persona", None, "Pedido", "Hola, ¿dónde está el pedido PED-10026?"),
]


@cache
def examples() -> list[dict[str, str]]:
    """Demo emails written as real customers of the synthetic shop."""
    shop = generate_shop()
    owners = {o.number: shop.customers[o.customer_id - 1].email for o in shop.orders}
    orders_by_customer = {o.customer_id for o in shop.orders}
    stranger = next(c.email for c in shop.customers if c.id not in orders_by_customer)
    return [
        {
            "label": label,
            "sender": owners[order] if order else stranger,
            "subject": subject,
            "body": body,
        }
        for label, order, subject, body in _EXAMPLES
    ]


@router.get("/")
async def demo_page() -> HTMLResponse:
    page = _PAGE.read_text(encoding="utf-8")
    data = json.dumps(examples(), ensure_ascii=False).replace("</", "<\\/")  # safe in <script>
    return HTMLResponse(page.replace("__EXAMPLES__", data))
