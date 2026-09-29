"""The shop's policy, as code. The agent explains decisions; it never makes them.

Keeping the rules here (not in the prompt) means a refund promise cannot depend on
the model reading the prompt correctly, and every rule has a test.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

RETURN_WINDOW_DAYS = 30
REFUND_DAYS = 14  # refund within 14 days of receiving the returned product
INVOICE_REVIEW_DAYS = 2  # working days for the accounts team to review an invoice

# What the drafting step may tell customers. Built from the constants, so the text
# and the rules cannot drift apart.
POLICY_SUMMARY = (
    f"- Returns: up to {RETURN_WINDOW_DAYS} days after delivery. The refund is made within "
    f"{REFUND_DAYS} days of receiving the product back.\n"
    "- Orders not shipped yet can be cancelled instead of returned.\n"
    "- Orders on their way can be returned once delivered.\n"
    "- Invoice problems are passed to the accounts team, which reviews them within "
    f"{INVOICE_REVIEW_DAYS} working days. Support never changes an invoice itself.\n"
    "- Anything else is passed to a colleague who will answer personally."
)


class OrderStatus(StrEnum):
    PENDING = "pending"  # paid, not shipped yet
    SHIPPED = "shipped"  # with the carrier
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class ReturnDecision(StrEnum):
    ALLOWED = "allowed"
    CANCEL_INSTEAD = "cancel_instead"  # not shipped yet: it can be cancelled
    WAIT_FOR_DELIVERY = "wait_for_delivery"
    WINDOW_EXPIRED = "window_expired"
    NOT_APPLICABLE = "not_applicable"  # cancelled or already returned


@dataclass(frozen=True)
class ReturnCheck:
    decision: ReturnDecision
    reason: str  # plain explanation the agent can reuse
    deadline: date | None = None  # last day to request the return


def check_return(status: OrderStatus, delivered_on: date | None, today: date) -> ReturnCheck:
    if status == OrderStatus.PENDING:
        return ReturnCheck(
            ReturnDecision.CANCEL_INSTEAD,
            "The order has not been shipped yet, so it can be cancelled instead of returned.",
        )
    if status == OrderStatus.SHIPPED:
        return ReturnCheck(
            ReturnDecision.WAIT_FOR_DELIVERY,
            "The order is on its way; a return can be requested once it is delivered.",
        )
    if status in (OrderStatus.CANCELLED, OrderStatus.RETURNED):
        return ReturnCheck(
            ReturnDecision.NOT_APPLICABLE,
            f"The order is already {status.value}; there is nothing to return.",
        )
    if delivered_on is None:  # delivered orders always have a date; be safe anyway
        return ReturnCheck(ReturnDecision.NOT_APPLICABLE, "The delivery date is unknown.")

    deadline = delivered_on + timedelta(days=RETURN_WINDOW_DAYS)
    if today <= deadline:
        return ReturnCheck(
            ReturnDecision.ALLOWED,
            f"Returns are accepted until {deadline.isoformat()} ({RETURN_WINDOW_DAYS} days after "
            f"delivery); the refund is made within {REFUND_DAYS} days of receiving the product.",
            deadline,
        )
    return ReturnCheck(
        ReturnDecision.WINDOW_EXPIRED,
        f"The {RETURN_WINDOW_DAYS}-day return period ended on {deadline.isoformat()}.",
        deadline,
    )
