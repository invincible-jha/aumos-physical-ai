# Contributing to aumos-physical-ai

Thank you for your interest in contributing to AumOS Physical AI.

## Development Setup

```bash
git clone <repo-url>
cd aumos-physical-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

## Running Tests

```bash
make test          # Full test suite with coverage
make test-quick    # Fast run, stop on first failure
```

## Code Style

```bash
make lint          # Check linting and formatting
make format        # Auto-fix formatting
make typecheck     # Run mypy type checks
```

## Architecture

This service follows hexagonal architecture:

- `api/` — FastAPI routers and Pydantic schemas (input/output only)
- `core/` — Domain models, services, and Protocol interfaces (no I/O)
- `adapters/` — External system clients (DB repositories, Kafka, BlenderProc, sensor simulator)

**Rules:**
- Services depend only on Protocol interfaces, never on concrete adapters
- No business logic in routers — delegate to services
- All external I/O goes through adapters
- Pydantic validation at API boundary; domain models are ORM types

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/my-change`
2. Write tests alongside your implementation
3. Run `make all` to verify linting, types, and tests pass
4. Open a pull request with a clear description of the change

## Commit Message Format

Follow conventional commits:
```
feat: add ultrasonic sensor synthesis support
fix: correct temporal alignment window calculation
docs: update sensor fusion strategy examples
test: add integration tests for domain randomization
```
