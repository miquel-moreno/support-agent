"""Evaluation with 40 synthetic customer emails (`make eval`).

Runs the agent on every email of evals/emails.json against the synthetic shop, with
the LLM from .env, and scores it:

- by code: category, order handling (situation + order), return decision, and
  invented data left after the review step;
- by an LLM judge: would a careful supervisor send the draft unchanged? (plus
  policy violations). A person checks a sample of the judge's verdicts by hand.

Real LLM calls, so it runs locally and never in CI. Results go to
evals/results/<date>_<model>.json, and a sheet for the manual check to
evals/results/<date>_<model>_review.md.

uv run python -m evals.run              # all 40
uv run python -m evals.run --limit 3    # a quick smoke run
uv run python -m evals.run --tag v2     # results saved as <date>_<model>_v2.json
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from support_agent.adapters.db import Base, make_session_factory, replace_shop
from support_agent.adapters.llm import build_chat_model
from support_agent.core.config import get_settings
from support_agent.services.agent import AgentState, build_agent
from support_agent.services.policy import POLICY_SUMMARY
from support_agent.services.shop_data import Shop, generate_shop
from support_agent.services.tracing import TraceRecorder, TraceStep

ROOT = Path(__file__).parent
DATASET = ROOT / "emails.json"
RESULTS = ROOT / "results"
# USD per million tokens (input, output), checked on openai.com on 2026-09-28.
PRICES = {"gpt-4.1-mini": (0.40, 1.60)}
REVIEW_SAMPLE = 10  # drafts a person checks by hand against the judge


@dataclass(frozen=True)
class EvalCase:
    id: str
    subject: str
    body: str
    expected: dict[str, Any]
    as_customer_of: str | None = None  # write as the customer who placed this order
    as_customer_id: int | None = None
    sender: str | None = None  # or an explicit (invented) address


def load_cases(path: Path = DATASET) -> list[EvalCase]:
    return [EvalCase(**raw) for raw in json.loads(path.read_text(encoding="utf-8"))]


def sender_for(case: EvalCase, shop: Shop) -> str:
    if case.sender:
        return case.sender
    customer_id = case.as_customer_id or next(
        o.customer_id for o in shop.orders if o.number == case.as_customer_of
    )
    return shop.customers[customer_id - 1].email


def score(case: EvalCase, state: AgentState) -> dict[str, bool | None]:
    """Checks decided by code. None means the check does not apply to this email."""
    expected = case.expected
    order = (state.get("order") or {}).get("number")
    situation = str(state.get("situation", "no_order_needed"))
    decision = (state.get("policy") or {}).get("decision")
    return {
        "category": str(state["category"]) == expected["category"],
        "order_handling": situation in expected["situation"] and order == expected["order"],
        "decision": decision == expected["decision"] if "decision" in expected else None,
        "no_invented_data": not state["issues"],
    }


class JudgeVerdict(BaseModel):
    ready_to_send: bool = Field(description="A careful supervisor would send it unchanged")
    breaks_policy: bool = Field(description="Promises or grants something the facts do not allow")
    problems: list[str] = Field(description="Short list of what is wrong; empty if nothing")


JUDGE_PROMPT = f"""You review reply drafts written by a support assistant of a car parts shop.
Decide whether a careful supervisor would send the draft exactly as it is.
It is ready to send only if ALL of these hold:
- it answers what the customer asked, or asks for exactly what is missing;
- every fact it states (orders, dates, amounts, statuses, deadlines) is in the facts given;
  the assistant has no product catalogue, stock, prices or opening hours, so any claim
  about them (e.g. "yes, we sell X") is invented and the draft is not ready to send;
- it does not say an action is done (cancelled, refunded) when it only can be offered;
- it follows the return policy decision and promises nothing beyond it;
- it reveals nothing about orders that are not the customer's;
- it is clear, polite, coherent Spanish, with no contradictions.
Mark breaks_policy if it promises or grants something the facts or policy do not allow.
List problems briefly (in Spanish). Be strict but fair: style preferences are not problems.
Shop policy:
{POLICY_SUMMARY}"""


async def judge(
    model: BaseChatModel, case: EvalCase, state: AgentState, recorder: TraceRecorder
) -> JudgeVerdict:
    facts = {
        "situation": str(state.get("situation", "no_order_needed")),
        "order": state.get("order"),
        "candidate_orders": [o["number"] for o in state.get("candidates", [])],
        "return_policy_decision": state.get("policy"),
    }
    prompt = [
        SystemMessage(JUDGE_PROMPT),
        HumanMessage(
            f"Customer email:\n{case.subject}\n{case.body}\n\n"
            f"Facts the assistant had (JSON):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"Draft:\n{state['draft']}"
        ),
    ]
    verdict = await model.with_structured_output(JudgeVerdict).ainvoke(
        prompt, {"callbacks": [recorder]}
    )
    assert isinstance(verdict, JudgeVerdict)  # noqa: S101 - narrows the type
    return verdict


def tokens(steps: list[TraceStep]) -> tuple[int, int]:
    llm = [s for s in steps if s.kind == "llm"]
    return sum(s.input_tokens or 0 for s in llm), sum(s.output_tokens or 0 for s in llm)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def summarize(results: list[dict[str, Any]], model: str) -> dict[str, Any]:
    def count(check: str) -> str:
        applicable = [r["checks"][check] for r in results if r["checks"][check] is not None]
        return f"{sum(applicable)}/{len(applicable)}"

    agent_in = sum(r["tokens"]["agent"][0] for r in results)
    agent_out = sum(r["tokens"]["agent"][1] for r in results)
    judge_in = sum(r["tokens"]["judge"][0] for r in results)
    judge_out = sum(r["tokens"]["judge"][1] for r in results)
    agent_cost = cost_usd(model, agent_in, agent_out)
    return {
        "emails": len(results),
        "category": count("category"),
        "order_handling": count("order_handling"),
        "return_decision": count("decision"),
        "no_invented_data": count("no_invented_data"),
        "ready_to_send_judge": f"{sum(r['verdict']['ready_to_send'] for r in results)}/"
        f"{len(results)}",
        "breaks_policy_judge": sum(r["verdict"]["breaks_policy"] for r in results),
        "avg_seconds_per_email": round(sum(r["seconds"] for r in results) / len(results), 1),
        "tokens_agent": [agent_in, agent_out],
        "tokens_judge": [judge_in, judge_out],
        "cost_usd_agent_per_email": round(agent_cost / len(results), 5) if agent_cost else None,
        "cost_usd_total": (
            round(agent_cost + (cost_usd(model, judge_in, judge_out) or 0), 4)
            if agent_cost is not None
            else None
        ),
    }


def review_sheet(results: list[dict[str, Any]], title: str) -> str:
    """Every 4th email, for a person to check the judge by hand."""
    lines = [
        f"# Revisión manual · {title}",
        "",
        "Para cada borrador: ¿lo enviarías tal cual? Anota sí/no y compáralo con la IA.",
        "",
    ]
    for r in results[:: max(1, len(results) // REVIEW_SAMPLE)][:REVIEW_SAMPLE]:
        verdict = "sí" if r["verdict"]["ready_to_send"] else "no"
        problems = "; ".join(r["verdict"]["problems"]) or "—"
        lines += [
            f"## {r['id']}",
            f"**Email:** {r['email']['body']}",
            "",
            f"**Pedido:** {r['got']['order'] or '—'} · **Decisión:** {r['got']['decision'] or '—'}",
            "",
            "**Borrador:**",
            "",
            *[f"> {line}".rstrip() for line in r["draft"].splitlines()],
            "",
            f"**IA:** {verdict} ({problems}) · **Yo:** ___",
            "",
        ]
    return "\n".join(lines)


def result_stem(day: str, model: str, *, tag: str | None, limit: int | None) -> str:
    """File name without extension: <date>_<model>[_<tag>][_limit<N>], safe on any OS."""
    stem = f"{day}_{model}" + (f"_{tag}" if tag else "") + (f"_limit{limit}" if limit else "")
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in stem)


async def run(limit: int | None = None, tag: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    model = build_chat_model(settings)
    shop = generate_shop()
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = make_session_factory(engine)
    async with sessions() as session:
        await replace_shop(session, shop)
    agent = build_agent(model, sessions, checkpointer=InMemorySaver())

    results = []
    for case in load_cases()[:limit]:
        agent_trace, judge_trace = TraceRecorder(), TraceRecorder()
        start = time.perf_counter()
        state = await agent.ainvoke(
            {"sender": sender_for(case, shop), "subject": case.subject, "body": case.body},
            {"configurable": {"thread_id": case.id}, "callbacks": [agent_trace]},
        )
        seconds = time.perf_counter() - start
        verdict = await judge(model, case, state, judge_trace)
        checks = score(case, state)
        results.append(
            {
                "id": case.id,
                "email": {"subject": case.subject, "body": case.body},
                "expected": case.expected,
                "got": {
                    "category": str(state["category"]),
                    "situation": str(state.get("situation", "no_order_needed")),
                    "order": (state.get("order") or {}).get("number"),
                    "decision": (state.get("policy") or {}).get("decision"),
                    "tools": [s.input for s in agent_trace.steps if s.kind == "tool"],
                },
                "checks": checks,
                "issues": state["issues"],
                "draft": state["draft"],
                "verdict": verdict.model_dump(),
                "seconds": round(seconds, 1),
                "tokens": {"agent": tokens(agent_trace.steps), "judge": tokens(judge_trace.steps)},
            }
        )
        failed = [k for k, v in checks.items() if v is False]
        mark = "ok " if not failed and verdict.ready_to_send else "-- "
        print(
            f"{mark}{case.id:<22} {seconds:5.1f}s  fallos={failed}  enviar={verdict.ready_to_send}"
        )
    await engine.dispose()

    label = f"{settings.llm_provider}/{settings.llm_model}"
    report = {
        "date": date.today().isoformat(),
        "model": label,
        "judge": label,
        "summary": summarize(results, settings.llm_model),
        "cases": results,
    }
    RESULTS.mkdir(exist_ok=True)
    stem = result_stem(report["date"], settings.llm_model, tag=tag, limit=limit)
    (RESULTS / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (RESULTS / f"{stem}_review.md").write_text(
        review_sheet(results, f"{report['date']} · {label}"),  # already ends with "\n"
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--limit", type=int, help="only the first N emails")
    parser.add_argument("--tag", help="suffix for the result files, e.g. v2")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.tag))


if __name__ == "__main__":
    main()
