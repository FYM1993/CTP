# Design Rules

## Table Of Contents

1. Module boundaries
2. Data models and typing
3. State and side effects
4. Exceptions and error translation
5. Configuration and secrets
6. Logging and observability
7. Persistence and external APIs
8. Concurrency and async
9. Testing boundaries

## Module Boundaries

- Give each module one primary job.
- Keep dependency direction inward toward stable business logic.
- Introduce an interface or protocol when two implementations must satisfy the same contract.
- Avoid circular imports by moving shared concepts into a smaller lower-level module.
- Do not create a `utils.py` for unrelated helpers. Name helpers by domain or concern.

## Data Models And Typing

- Add type hints to all public functions and methods.
- Use `dataclass` for structured in-memory records with clear fields.
- Use `TypedDict` only for dictionary-shaped data that must stay dict-like.
- Use `Protocol` for behavior contracts at dependency boundaries.
- Use `Enum` or `Literal` when string modes are constrained.
- Prefer explicit return types over tuples with unclear positional meaning.
- Avoid passing `dict[str, Any]` through multiple layers unless the boundary is truly dynamic.

## State And Side Effects

- Keep pure computation separate from I/O and mutable state.
- Inject clocks, random generators, sessions, and clients when they affect behavior.
- Prefer immutable intermediate values over in-place mutation across long functions.
- Contain caches behind a module or object boundary with documented invalidation rules.
- Make retry, timeout, and backoff behavior visible at the call site or in config.

## Exceptions And Error Translation

- Raise domain-specific errors for domain failures.
- Translate infrastructure exceptions when crossing a public boundary.
- Preserve the original exception as context when it matters for debugging.
- Do not swallow exceptions silently.
- Avoid `except Exception` unless you are adding context, logging, cleanup, or fallback behavior deliberately.

## Configuration And Secrets

- Load configuration once near startup.
- Validate configuration early and fail fast on missing required values.
- Pass a typed config object instead of reading environment variables throughout the codebase.
- Keep secrets out of logs and repr output.
- Separate runtime configuration from user input and business state.

## Logging And Observability

- Use the `logging` module for reusable code and long-running programs.
- Keep log messages stable and informative. Include identifiers, counts, durations, and result states when useful.
- Log at boundaries: startup, external requests, retries, failures, and major state transitions.
- Avoid noisy logs inside hot loops unless they are debug-only and gated.
- Prefer structured fields or consistent message patterns over ad-hoc prose.

## Persistence And External APIs

- Convert raw external payloads into internal models before applying business rules.
- Keep ORM models, HTTP payloads, and database rows from leaking through the whole system.
- Make network and storage clients thin adapters rather than business-logic containers.
- Define idempotency rules for operations that can be retried.
- Surface pagination, rate limits, and partial failure behavior explicitly.

## Concurrency And Async

- Use async only when the surrounding stack benefits from it.
- Keep sync and async interfaces separate unless there is a strong reason to support both.
- Do not call blocking I/O from async code without an explicit bridge.
- Bound task fan-out and concurrency. Unlimited `gather()` over untrusted input is a design bug.
- Make cancellation, timeout, and cleanup behavior explicit for background work.

## Testing Boundaries

- Unit-test pure transforms and decision logic directly.
- Integration-test adapters with the real filesystem, database, or HTTP boundary when practical.
- Prefer fakes or narrow protocols over deep mocks of implementation details.
- Test public behavior and important invariants, not private helper call counts.
- If testing is painful, redesign the seam before adding more patching.
