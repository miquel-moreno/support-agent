"""Checks on a draft reply before a person sees it: no invented order data.

Every order number, tracking code, invoice number, date and amount in the draft must
come from the orders the lookup returned (or the policy). Anything else is reported
as an issue: the agent gets one chance to fix it, then the issue is shown to the
person who approves the reply.
"""

import re
from collections.abc import Iterable
from datetime import date
from decimal import Decimal, InvalidOperation

from support_agent.services.order_tools import OrderFacts, parse_es_date

ORDER_NUMBERS = re.compile(r"\bPED-?\s?\d{5}\b", re.IGNORECASE)
TRACKING_CODES = re.compile(r"\b[A-Z]{3}\d{9}\b")
INVOICE_NUMBERS = re.compile(r"\bF-\d{4}-\d{5}\b")
DATES = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
MONTHS = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]
# "11 de octubre de 2026", or "11 de octubre" (then any year is accepted)
WORD_DATES = re.compile(
    rf"\b(\d{{1,2}}) de ({'|'.join(MONTHS)})(?: de(?:l)? (\d{{4}}))?", re.IGNORECASE
)
AMOUNTS = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?\s?(?:€|euros?\b)", re.IGNORECASE)


def _money(text: str) -> Decimal | None:
    match = AMOUNTS.fullmatch(text.strip())
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(".", "") + "." + (match.group(2) or "00"))
    except InvalidOperation:  # pragma: no cover - the regex only lets digits through
        return None


def _date(day: str, month: str, year: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def check_draft(
    draft: str,
    orders: Iterable[OrderFacts],
    *,
    extra_dates: Iterable[date] = (),
) -> list[str]:
    """Return one plain-English issue per value in the draft that the data does not back."""
    orders = list(orders)
    numbers = {o["number"] for o in orders}
    tracking = {o["tracking_code"] for o in orders if o["tracking_code"]}
    invoices = {o["invoice_number"] for o in orders}
    dates = set(extra_dates)
    amounts: set[Decimal] = set()
    for o in orders:
        for field in ("ordered_on", "shipped_on", "delivered_on"):
            parsed = parse_es_date(o[field])
            if parsed:
                dates.add(parsed)
        total = _money(o["total"])
        if total is not None:
            amounts.add(total)
        for item in o["items"]:
            unit = _money(item["unit_price"])
            if unit is not None:
                amounts.update({unit, unit * item["quantity"]})

    issues: list[str] = []
    for match in ORDER_NUMBERS.finditer(draft):
        number = "PED-" + match.group()[-5:]
        if number not in numbers:
            issues.append(f"order number {match.group()} is not one of the customer's orders")
    for code in TRACKING_CODES.findall(draft):
        if code not in tracking:
            issues.append(f"tracking code {code} does not match the order")
    for invoice in INVOICE_NUMBERS.findall(draft):
        if invoice not in invoices:
            issues.append(f"invoice {invoice} does not match the order")
    for match in DATES.finditer(draft):
        if _date(*match.groups()) not in dates:
            issues.append(f"date {match.group()} does not come from the order or the policy")
    for match in WORD_DATES.finditer(draft):
        day, month, year = int(match[1]), MONTHS.index(match[2].lower()) + 1, match[3]
        if not any(
            (d.day, d.month) == (day, month) and (year is None or d.year == int(year))
            for d in dates
        ):
            issues.append(f"date {match.group()} does not come from the order or the policy")
    for match in AMOUNTS.finditer(draft):
        if _money(match.group()) not in amounts:
            issues.append(f"amount {match.group()} does not come from the order")
    return list(dict.fromkeys(issues))  # unique, in order of appearance
