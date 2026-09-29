"""The review queue through the HTTP API, with a scripted model and the synthetic shop."""

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from scripts import seed_shop

from support_agent.adapters.llm import ScriptedChatModel
from support_agent.api.dependencies import get_chat_model
from support_agent.main import create_app
from support_agent.services.agent import Category, EmailClassification
from support_agent.services.shop_data import generate_shop

ORDER = generate_shop().orders[0]
SENDER = generate_shop().customers[ORDER.customer_id - 1].email
DRAFT = f"Hola. Tu pedido {ORDER.number} está {ORDER.status.value}."


def replies() -> list[Any]:
    return [
        EmailClassification(category=Category.ORDER_STATUS),
        AIMessage(
            "",
            tool_calls=[{"name": "find_order", "args": {"order_number": ORDER.number}, "id": "1"}],
        ),
        AIMessage("done"),
        AIMessage(DRAFT),
    ]


@pytest.fixture
def api(database_url: str) -> Iterator[tuple[TestClient, ScriptedChatModel]]:
    asyncio.run(seed_shop.run())
    model = ScriptedChatModel(replies=[])
    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: model
    with TestClient(app) as client:
        yield client, model


def submit(api: tuple[TestClient, ScriptedChatModel]) -> dict[str, Any]:
    client, model = api
    model.replies.extend(replies())
    response = client.post(
        "/emails", json={"sender": SENDER, "subject": "Pedido", "body": f"¿Y el {ORDER.number}?"}
    )
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def test_an_email_becomes_a_pending_draft(api: Any) -> None:
    client, _ = api
    draft = submit(api)

    assert draft["status"] == "pending"
    assert (draft["category"], draft["situation"]) == ("order_status", "order_found")
    assert (draft["order_number"], draft["draft"], draft["issues"]) == (ORDER.number, DRAFT, [])
    assert draft["final_text"] is None
    pending = client.get("/drafts", params={"status": "pending"}).json()
    assert [d["id"] for d in pending] == [draft["id"]]
    assert client.get(f"/drafts/{draft['id']}").json() == draft


def test_approving_sends_the_draft_unchanged(api: Any) -> None:
    client, _ = api
    draft_id = submit(api)["id"]

    sent = client.post(f"/drafts/{draft_id}/approve").json()

    assert (sent["status"], sent["final_text"], sent["edited"]) == ("sent", DRAFT, False)
    assert sent["decided_at"] is not None
    assert client.get("/drafts", params={"status": "pending"}).json() == []


def test_approving_an_edited_reply_records_the_change(api: Any) -> None:
    client, _ = api
    draft_id = submit(api)["id"]

    sent = client.post(f"/drafts/{draft_id}/approve", json={"text": "Hola, ya sale hoy."}).json()

    assert (sent["final_text"], sent["edited"]) == ("Hola, ya sale hoy.", True)


def test_rejecting_keeps_the_reason_and_sends_nothing(api: Any) -> None:
    client, _ = api
    draft_id = submit(api)["id"]

    rejected = client.post(f"/drafts/{draft_id}/reject", json={"reason": "Tono frío"}).json()

    assert (rejected["status"], rejected["reject_reason"]) == ("rejected", "Tono frío")
    assert rejected["final_text"] is None


def test_a_draft_is_decided_only_once(api: Any) -> None:
    client, _ = api
    draft_id = submit(api)["id"]
    client.post(f"/drafts/{draft_id}/approve")

    again = client.post(f"/drafts/{draft_id}/reject", json={"reason": "tarde"})

    assert again.status_code == 409
    assert again.json()["error"]["code"] == "conflict"


def test_unknown_draft_is_404(api: Any) -> None:
    client, _ = api
    assert client.get("/drafts/nope").status_code == 404
    assert client.post("/drafts/nope/approve").status_code == 404


@pytest.mark.parametrize(
    "email",
    [
        {"sender": "not-an-email", "body": "hola"},
        {"sender": "a@example.com", "body": ""},
    ],
)
def test_invalid_emails_are_rejected(api: Any, email: dict[str, str]) -> None:
    client, _ = api
    assert client.post("/emails", json=email).status_code == 422


def test_a_missing_llm_key_is_a_clear_503(database_url: str, monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "change-me")
    from support_agent.core.config import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        response = client.post("/emails", json={"sender": "a@example.com", "body": "hola"})
    assert response.status_code == 503
