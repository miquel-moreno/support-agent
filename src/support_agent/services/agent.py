"""The support agent as a LangGraph state graph.

    classify -> lookup <-> tools -> resolve -> policy -> draft -> review -> (draft | end)

The model does three things: label the email, decide which lookups to make (tool
calling) and write the reply. Everything with consequences is code: which order the
reply is about, whether a return is allowed, and whether the draft's data is real.
"""

import json
from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from support_agent.services.guardrails import check_draft
from support_agent.services.order_tools import (
    LookupResult,
    OrderFacts,
    es_date,
    make_order_tools,
    parse_es_date,
)
from support_agent.services.policy import POLICY_SUMMARY, OrderStatus, check_return
from support_agent.services.shop_data import REFERENCE_DATE

MAX_LOOKUP_ROUNDS = 3  # model turns in the lookup loop, so it can never spin forever
MAX_DRAFTS = 2  # first draft + one fix after the review


class Category(StrEnum):
    ORDER_STATUS = "order_status"  # where is my order?
    RETURN_REQUEST = "return_request"  # I want to return it
    INVOICE_PROBLEM = "invoice_problem"  # the invoice is wrong
    OTHER = "other"


class Situation(StrEnum):
    ORDER_FOUND = "order_found"
    CHOOSE_ORDER = "choose_order"  # several orders and no number: ask which one
    ASK_ORDER_NUMBER = "ask_order_number"  # nothing found: ask for the number
    FOREIGN_ORDER = "foreign_order"  # the number belongs to another email address
    NO_ORDER_NEEDED = "no_order_needed"


class EmailClassification(BaseModel):
    category: Category = Field(description="What the customer is asking for")


class AgentState(TypedDict, total=False):
    sender: str
    subject: str
    body: str
    messages: Annotated[list[AnyMessage], add_messages]  # the lookup conversation
    category: Category
    situation: Situation
    order: OrderFacts | None
    candidates: list[OrderFacts]
    policy: dict[str, str | None] | None
    draft: str
    drafts: int
    issues: list[str]


CLASSIFY_PROMPT = """You sort customer emails for a car parts shop.
Categories:
- order_status: where the order is, when it arrives, tracking.
- return_request: returning a product or getting a refund.
- invoice_problem: a wrong or missing invoice, wrong amount or details on it.
- other: anything else (product questions, complaints without an order, spam...)."""

LOOKUP_PROMPT = """You find the order a customer email is about, using the tools.
- If the email mentions an order number, call find_order with exactly that number.
  Never make a number up.
- If it does not, or find_order finds nothing, call find_my_orders.
- Only call tools. When you have looked up what you need, answer with the word: done."""

DRAFT_PROMPT = f"""You write replies to customers of a car parts shop, in Spanish (tú).
Rules:
- Use ONLY the facts given (order, policy decision). Never invent order numbers,
  dates, amounts, tracking codes or invoice numbers; copy them exactly as given.
- Never promise anything the policy decision does not allow.
- situation "ask_order_number": we could not find the order; ask for its number
  (format PED-XXXXX) and say nothing about any order.
- situation "choose_order": list the candidate orders (number, date, status) and ask
  which one they mean.
- situation "foreign_order": the order was placed with another email address; do not
  share anything about it and ask them to write from the email used to buy.
- Short and friendly: greeting, answer, next step. Sign as "Equipo de atención al cliente".
- Write only the body of the email.
Shop policy:
{POLICY_SUMMARY}"""


def _lookup_results(messages: list[AnyMessage]) -> tuple[list[OrderFacts], list[OrderFacts], bool]:
    """(orders found by number, orders found by sender email, a hidden order was asked)."""
    by_number: dict[str, OrderFacts] = {}
    by_email: dict[str, OrderFacts] = {}
    hidden = False
    for message in messages:
        if not isinstance(message, ToolMessage) or not message.artifact:
            continue
        result: LookupResult = message.artifact
        target = by_number if message.name == "find_order" else by_email
        target.update({o["number"]: o for o in result["orders"]})
        hidden = hidden or bool(result["hidden"])
    return list(by_number.values()), list(by_email.values()), hidden


def build_agent(
    model: BaseChatModel,
    sessions: async_sessionmaker[AsyncSession],
    *,
    today: date = REFERENCE_DATE,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    tools = make_order_tools(sessions)
    lookup_model = model.bind_tools(tools)
    classifier = model.with_structured_output(EmailClassification)

    def email_text(state: AgentState) -> str:
        return f"From: {state['sender']}\nSubject: {state['subject']}\n\n{state['body']}"

    async def classify(state: AgentState) -> AgentState:
        result = await classifier.ainvoke(
            [SystemMessage(CLASSIFY_PROMPT), HumanMessage(email_text(state))]
        )
        assert isinstance(result, EmailClassification)  # noqa: S101 - narrows the type
        return {"category": result.category}

    async def lookup(state: AgentState) -> AgentState:
        prompt: list[AnyMessage] = [SystemMessage(LOOKUP_PROMPT), HumanMessage(email_text(state))]
        reply = await lookup_model.ainvoke(prompt + state.get("messages", []))
        return {"messages": [reply]}

    def resolve(state: AgentState) -> AgentState:
        by_number, by_email, hidden = _lookup_results(state.get("messages", []))
        if len(by_number) == 1:
            return {"situation": Situation.ORDER_FOUND, "order": by_number[0], "candidates": []}
        if hidden and not by_number:
            return {"situation": Situation.FOREIGN_ORDER, "order": None, "candidates": []}
        candidates = by_number or by_email
        if len(candidates) == 1:
            return {"situation": Situation.ORDER_FOUND, "order": candidates[0], "candidates": []}
        if candidates:
            return {"situation": Situation.CHOOSE_ORDER, "order": None, "candidates": candidates}
        return {"situation": Situation.ASK_ORDER_NUMBER, "order": None, "candidates": []}

    def policy(state: AgentState) -> AgentState:
        order = state.get("order")
        if state["category"] != Category.RETURN_REQUEST or order is None:
            return {"policy": None}
        check = check_return(
            OrderStatus(order["status"]), parse_es_date(order["delivered_on"]), today
        )
        return {
            "policy": {
                "decision": check.decision.value,
                "reason": check.reason,
                "deadline": es_date(check.deadline),
            }
        }

    async def draft(state: AgentState) -> AgentState:
        facts = {
            "today": es_date(today),
            "category": state["category"],
            "situation": state.get("situation", Situation.NO_ORDER_NEEDED),
            "order": state.get("order"),
            "candidate_orders": [
                {"number": o["number"], "ordered_on": o["ordered_on"], "status_es": o["status_es"]}
                for o in state.get("candidates", [])
            ],
            "return_policy_decision": state.get("policy"),
        }
        messages: list[AnyMessage] = [
            SystemMessage(DRAFT_PROMPT),
            HumanMessage(
                f"Customer email:\n{email_text(state)}\n\n"
                f"Facts (JSON):\n{json.dumps(facts, ensure_ascii=False, indent=1)}"
            ),
        ]
        if state.get("issues"):
            messages.append(AIMessage(state["draft"]))
            messages.append(
                HumanMessage(
                    "That draft contains data that is not in the facts: "
                    + "; ".join(state["issues"])
                    + ". Rewrite it using only the facts."
                )
            )
        reply = await model.ainvoke(messages)
        return {"draft": reply.text.strip(), "drafts": state.get("drafts", 0) + 1}

    def review(state: AgentState) -> AgentState:
        orders = [o for o in [state.get("order")] if o] + state.get("candidates", [])
        deadline = parse_es_date((state.get("policy") or {}).get("deadline"))
        extra = [today] + ([deadline] if deadline else [])
        return {"issues": check_draft(state["draft"], orders, extra_dates=extra)}

    def after_classify(state: AgentState) -> Literal["lookup", "draft"]:
        return "draft" if state["category"] == Category.OTHER else "lookup"

    def after_lookup(state: AgentState) -> Literal["tools", "resolve"]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        rounds = sum(isinstance(m, AIMessage) for m in messages)
        if isinstance(last, AIMessage) and last.tool_calls and rounds <= MAX_LOOKUP_ROUNDS:
            return "tools"
        return "resolve"

    def after_review(state: AgentState) -> Literal["draft", "__end__"]:
        return "draft" if state["issues"] and state["drafts"] < MAX_DRAFTS else "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("classify", classify)
    graph.add_node("lookup", lookup)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("resolve", resolve)
    graph.add_node("policy", policy)
    graph.add_node("draft", draft)
    graph.add_node("review", review)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify)
    graph.add_conditional_edges("lookup", after_lookup)
    graph.add_edge("tools", "lookup")
    graph.add_edge("resolve", "policy")
    graph.add_edge("policy", "draft")
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", after_review)
    return graph.compile(checkpointer=checkpointer)
