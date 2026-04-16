# Review Checklist

## Table Of Contents

1. Architecture pass
2. Interface pass
3. Operational safety pass
4. Testability pass
5. Refactor heuristics

## Architecture Pass

Ask these questions first:

- Does each module have one primary responsibility?
- Does dependency direction point from entrypoints toward stable core logic?
- Are I/O adapters separated from business rules?
- Is there any framework object, request object, or database model leaking too far inward?
- Is the entrypoint composing the system, or is it carrying the business logic itself?

## Interface Pass

Ask these questions next:

- Are public functions and methods typed?
- Are important inputs modeled explicitly instead of passed around as loose dicts?
- Are return values stable and easy to consume?
- Are exceptions translated into meaningful domain or application errors?
- Are configuration and collaborators injected explicitly instead of read from globals?

## Operational Safety Pass

Check the edges:

- Are retries, timeouts, and partial failures defined where external calls happen?
- Are logs useful enough to debug a production issue?
- Is cleanup of files, sessions, pools, or temp resources explicit?
- Are secrets protected from accidental logging?
- Does async or concurrent code bound fan-out and handle cancellation?

## Testability Pass

Look for friction:

- Can the core behavior run without patching internals?
- Are pure transforms isolated enough for small unit tests?
- Are adapters narrow enough for targeted integration tests?
- Do tests rely on private implementation details rather than behavior?
- Would a small protocol or helper extraction reduce mocking pressure?

## Refactor Heuristics

Use these rules to turn smells into a plan:

- If one file mixes orchestration, business rules, and I/O, split by responsibility first.
- If many functions pass the same loose dict, introduce a typed model first.
- If environment reads appear deep in the call graph, extract configuration loading upward first.
- If tests patch multiple layers to reach core logic, introduce a seam before changing behavior.
- If several modes share one function behind flag parameters, split the modes into separate functions or strategies.
