from datetime import date

import pytest

from support_agent.services.guardrails import check_draft
from support_agent.services.order_tools import OrderFacts, es_money, normalize_order_number

ORDER = OrderFacts(
    number="PED-10023",
    status="delivered",
    status_es="entregado",
    ordered_on="01/09/2026",
    shipped_on="02/09/2026",
    delivered_on="04/09/2026",
    carrier="SEUR",
    tracking_code="SEU123456789",
    invoice_number="F-2026-05023",
    total="1.234,50 €",
    items=[
        {"name": "Kit de embrague", "quantity": 3, "unit_price": "345,50 €"},
        {"name": "Lámpara H7", "quantity": 1, "unit_price": "198,00 €"},
    ],
)


def test_a_draft_that_only_uses_real_data_passes() -> None:
    draft = (
        "Tu pedido PED-10023 (factura F-2026-05023) se entregó el 04/09/2026 con SEUR, "
        "seguimiento SEU123456789. Total: 1.234,50 €; el kit fueron 3 x 345,50 € = 1036,50 €. "
        "Puedes devolverlo hasta el 04/10/2026. La lámpara costó 198 euros."
    )
    assert check_draft(draft, [ORDER], extra_dates=[date(2026, 10, 4)]) == []


@pytest.mark.parametrize(
    ("draft", "issue"),
    [
        ("Tu pedido PED-10024 está en camino.", "order number PED-10024"),
        ("Seguimiento: MRW987654321.", "tracking code MRW987654321"),
        ("Revisaremos la factura F-2026-05999.", "invoice F-2026-05999"),
        ("Llegará el 30/09/2026.", "date 30/09/2026"),
        ("Te devolvemos 50,00 €.", "amount 50,00 €"),
        ("Te devolvemos 1.500 euros.", "amount 1.500 euros"),
        ("Fecha rara: 31/02/2026.", "date 31/02/2026"),
    ],
)
def test_invented_values_are_reported(draft: str, issue: str) -> None:
    issues = check_draft(draft, [ORDER])
    assert len(issues) == 1 and issue in issues[0]


def test_no_orders_means_no_order_data_is_allowed() -> None:
    issues = check_draft("Tu pedido PED-10023 del 01/09/2026.", [])
    assert len(issues) == 2


def test_each_issue_is_reported_once() -> None:
    assert len(check_draft("PED-10999 ... PED-10999", [ORDER])) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PED-10023", "PED-10023"),
        ("ped 10023", "PED-10023"),
        ("pedido nº 10023", "PED-10023"),
        ("PED10023", "PED-10023"),
        ("sin número", None),
    ],
)
def test_order_numbers_are_normalized(text: str, expected: str | None) -> None:
    assert normalize_order_number(text) == expected


def test_money_is_written_the_spanish_way() -> None:
    from decimal import Decimal

    assert es_money(Decimal("1234.5")) == "1234,50 €"
