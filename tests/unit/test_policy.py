from datetime import date

import pytest

from support_agent.services.policy import (
    RETURN_WINDOW_DAYS,
    OrderStatus,
    ReturnDecision,
    check_return,
)

TODAY = date(2026, 9, 28)


def test_return_allowed_within_the_window() -> None:
    check = check_return(OrderStatus.DELIVERED, date(2026, 9, 1), TODAY)

    assert check.decision == ReturnDecision.ALLOWED
    assert check.deadline == date(2026, 10, 1)
    assert "2026-10-01" in check.reason


def test_last_day_of_the_window_is_still_allowed() -> None:
    delivered = date(2026, 8, 29)  # + 30 days = 2026-09-28 = today

    assert check_return(OrderStatus.DELIVERED, delivered, TODAY).decision == ReturnDecision.ALLOWED


def test_return_refused_after_the_window() -> None:
    check = check_return(OrderStatus.DELIVERED, date(2026, 8, 28), TODAY)

    assert check.decision == ReturnDecision.WINDOW_EXPIRED
    assert str(RETURN_WINDOW_DAYS) in check.reason


@pytest.mark.parametrize(
    ("status", "decision"),
    [
        (OrderStatus.PENDING, ReturnDecision.CANCEL_INSTEAD),
        (OrderStatus.SHIPPED, ReturnDecision.WAIT_FOR_DELIVERY),
        (OrderStatus.CANCELLED, ReturnDecision.NOT_APPLICABLE),
        (OrderStatus.RETURNED, ReturnDecision.NOT_APPLICABLE),
    ],
)
def test_orders_not_delivered_cannot_be_returned(
    status: OrderStatus, decision: ReturnDecision
) -> None:
    check = check_return(status, None, TODAY)

    assert check.decision == decision
    assert check.deadline is None


def test_delivered_without_a_date_is_not_returnable() -> None:
    assert (
        check_return(OrderStatus.DELIVERED, None, TODAY).decision == ReturnDecision.NOT_APPLICABLE
    )
