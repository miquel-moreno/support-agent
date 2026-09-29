"""The evaluation itself: a trustworthy answer key and correct scoring (no LLM)."""

from collections import Counter
from typing import Any

import pytest
from evals.run import (
    EvalCase,
    JudgeVerdict,
    cost_usd,
    judge,
    load_cases,
    review_sheet,
    score,
    sender_for,
    summarize,
)

from support_agent.adapters.llm import ScriptedChatModel
from support_agent.services.agent import AgentState
from support_agent.services.order_tools import mentioned_order_numbers
from support_agent.services.policy import OrderStatus, check_return
from support_agent.services.shop_data import REFERENCE_DATE, generate_shop
from support_agent.services.tracing import TraceRecorder

CASES = load_cases()
SHOP = generate_shop()
ORDERS = {o.number: o for o in SHOP.orders}


def test_forty_emails_with_unique_ids_and_the_planned_mix() -> None:
    assert len(CASES) == 40
    assert len({c.id for c in CASES}) == 40
    assert Counter(c.expected["category"] for c in CASES) == {
        "order_status": 15,
        "return_request": 13,
        "invoice_problem": 8,
        "other": 4,
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_answer_key_matches_the_shop(case: EvalCase) -> None:
    sender = sender_for(case, SHOP)
    assert sender.split("@")[1].startswith(("example.", "proveedor.example."))
    expected = case.expected
    if expected["order"] is None:
        assert "decision" not in expected
        return
    order = ORDERS[expected["order"]]
    assert SHOP.customers[order.customer_id - 1].email == sender  # the customer's own order
    mentioned = mentioned_order_numbers(case.subject + " " + case.body)
    customer_orders = [o for o in SHOP.orders if o.customer_id == order.customer_id]
    # Findable: either the email names it, or it is the customer's only order.
    assert order.number in mentioned or (not mentioned and len(customer_orders) == 1)
    if "decision" in expected:
        decision = check_return(OrderStatus(order.status), order.delivered_on, REFERENCE_DATE)
        assert decision.decision.value == expected["decision"]


def state(**overrides: Any) -> AgentState:
    base: dict[str, Any] = {
        "category": "return_request",
        "situation": "order_found",
        "order": {"number": "PED-10001"},
        "policy": {"decision": "allowed"},
        "issues": [],
        "draft": "Hola.",
    }
    return AgentState(**{**base, **overrides})  # type: ignore[typeddict-item]


CASE = EvalCase(
    id="x",
    subject="",
    body="",
    expected={
        "category": "return_request",
        "situation": ["order_found"],
        "order": "PED-10001",
        "decision": "allowed",
    },
)


def test_scoring_passes_the_expected_behaviour() -> None:
    assert score(CASE, state()) == {
        "category": True,
        "order_handling": True,
        "decision": True,
        "no_invented_data": True,
    }


def test_scoring_catches_each_mistake() -> None:
    wrong = state(
        category="other",
        order={"number": "PED-10002"},
        policy={"decision": "window_expired"},
        issues=["date 01/01/2026 does not come from the order or the policy"],
    )
    assert set(score(CASE, wrong).values()) == {False}


def test_checks_that_do_not_apply_are_none() -> None:
    other = EvalCase(
        id="o",
        subject="",
        body="",
        expected={"category": "other", "situation": ["no_order_needed"], "order": None},
    )
    result = score(other, AgentState(category="other", issues=[], draft="Hola."))  # type: ignore[typeddict-item]
    assert result["decision"] is None and result["order_handling"] is True


async def test_the_judge_sees_the_email_the_facts_and_the_draft() -> None:
    verdict = JudgeVerdict(ready_to_send=False, breaks_policy=True, problems=["promete 500 €"])
    model = ScriptedChatModel(replies=[verdict])

    result = await judge(model, CASE, state(draft="Te devolvemos 500 €."), TraceRecorder())

    assert result == verdict
    prompt = str(model.prompts[0][1].content)
    assert "Te devolvemos 500 €." in prompt and '"decision": "allowed"' in prompt


def result(i: int, *, ready: bool) -> dict[str, Any]:
    return {
        "id": f"case-{i}",
        "email": {"subject": "", "body": f"email {i}"},
        "got": {"order": None, "decision": None},
        "checks": {
            "category": True,
            "order_handling": i % 2 == 0,
            "decision": None,
            "no_invented_data": True,
        },
        "draft": "Hola.\nAdiós.",
        "verdict": {"ready_to_send": ready, "breaks_policy": False, "problems": []},
        "seconds": 2.0,
        "tokens": {"agent": (1000, 100), "judge": (500, 50)},
    }


def test_summary_counts_only_the_checks_that_apply() -> None:
    results = [result(i, ready=i < 3) for i in range(4)]

    summary = summarize(results, "gpt-4.1-mini")

    assert summary["order_handling"] == "2/4"
    assert summary["return_decision"] == "0/0"
    assert summary["ready_to_send_judge"] == "3/4"
    assert summary["tokens_agent"] == [4000, 400]
    assert summary["cost_usd_agent_per_email"] == pytest.approx(0.00056)
    assert summarize(results, "qwen2.5:3b")["cost_usd_total"] is None


def test_cost_uses_the_price_per_million_tokens() -> None:
    assert cost_usd("gpt-4.1-mini", 1_000_000, 1_000_000) == 2.0
    assert cost_usd("unknown", 1, 1) is None


def test_review_sheet_samples_ten_drafts() -> None:
    sheet = review_sheet([result(i, ready=True) for i in range(40)], "test")

    assert sheet.count("## case-") == 10
    assert "> Adiós." in sheet and "**Yo:** ___" in sheet


def test_result_files_get_a_safe_name() -> None:
    from evals.run import result_stem

    assert result_stem("2026-09-29", "gpt-4.1-mini", tag=None, limit=None) == (
        "2026-09-29_gpt-4.1-mini"
    )
    assert result_stem("2026-09-29", "qwen2.5:3b", tag="v2", limit=3) == (
        "2026-09-29_qwen2.5-3b_v2_limit3"
    )
    assert "/" not in result_stem("d", "openai/gpt", tag="a/b", limit=None)
