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

### Changed
- The LLM layer uses LangChain chat models (`ChatOpenAI` for OpenAI and Ollama).
