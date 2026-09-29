# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Project scaffold: FastAPI app, health endpoint, JSON logging, CI, Docker.
- Return policy as code (30 days from delivery; cancel if not shipped).
- Synthetic shop (Faker, fixed seed): customers and orders tables, lookups and seed script.
- LangGraph agent: classify, look up the order with tool calling, apply the return policy, draft the reply and review it for invented data.
- `scripts/try_agent.py` to run one email against the synthetic shop.
- Human approval: the graph pauses before sending (`interrupt` + PostgreSQL checkpointer). API: `POST /emails`, `GET /drafts`, `GET /drafts/{id}`, `POST /drafts/{id}/approve` (optionally edited) and `/reject`. Sending is simulated.
- Docker loads the demo shop on first start.
- Traces: every node, LLM call (tokens, model, latency) and tool call of each email, stored in `trace_steps`; `GET /drafts/{id}/trace`.
- Evaluation with 40 synthetic emails: code checks, an LLM judge and a manual review sample; results in `evals/results/`.

### Fixed
- Drafts no longer make claims about products, stock, prices or hours, and offer cancellations instead of announcing them (found in the manual review).
- The review step accepts order numbers the customer wrote.

### Changed
- The LLM layer uses LangChain chat models (`ChatOpenAI` for OpenAI and Ollama).
