---
name: python-program-design
description: "Design, review, or refactor Python programs for clear boundaries, typed interfaces, configuration handling, observability, and testability. Use when Codex needs guidance on Python architecture, package or module layout, CLI or service structure, data modeling, exceptions, logging, configuration, testing strategy, code review, 程序设计, 设计规范, 代码规范, 重构, 目录结构, 类型注解, 异常处理, 配置管理, or 测试策略."
---

# Python Program Design

## Overview

Use this skill to design Python code before implementation or to review existing code for maintainability. Optimize for explicit boundaries, isolated side effects, typed contracts, and code that stays easy to test and evolve.

## Route The Request First

Read only the references needed for the current request.

1. Read [references/project-shapes.md](references/project-shapes.md) for package layout, entrypoint structure, CLI or service decomposition, and repository organization.
2. Read [references/design-rules.md](references/design-rules.md) for module boundaries, typing, state, exceptions, configuration, logging, persistence, and concurrency rules.
3. Read [references/review-checklist.md](references/review-checklist.md) for review prompts, refactor triage, and common design smells.

## Design Workflow

1. Identify the execution surface first: library, CLI app, HTTP service, scheduled job, data pipeline, or one-off script.
2. Define stable boundaries before writing code: orchestration, domain logic, infrastructure, persistence, and presentation or CLI concerns.
3. Model important data explicitly with typed structures. Prefer dataclasses, TypedDict, Protocol, Enum, and small value objects over passing shapeless dicts across the whole system.
4. Isolate side effects. Keep filesystem, network, clock, environment, subprocess, and database interactions at the edges.
5. Choose the smallest useful abstraction. Use functions by default, classes when state or lifecycle matters, and inheritance only for true substitutability.
6. Make failure modes explicit. Translate low-level exceptions at module boundaries and expose errors with domain context.
7. Design observability up front. Decide configuration loading, logging, retries, timeouts, and progress reporting before wiring the runtime.
8. Plan tests around seams. Unit-test pure logic, integration-test boundaries, and avoid designs that require patching internals to exercise the core behavior.

## Core Rules

1. Prefer modules with one reason to change. Split files that mix parsing, business rules, I/O, and presentation.
2. Prefer explicit dependency injection over global singletons or hidden imports with behavior.
3. Prefer standard library and simple composition before framework-heavy abstractions.
4. Prefer return values and typed result objects over hidden mutation.
5. Keep application bootstrapping near the edge. Let `main()` compose collaborators and then hand off to domain code.
6. Keep adapters thin. Convert external payloads into internal models early and convert back late.
7. Add type hints to public functions, return values, protocols, and important internal seams. Do not leave core contracts implicit.
8. Treat configuration as validated input. Parse it once, normalize it, and pass a typed config object.
9. Use consistent logging for long-running workflows and services. Avoid `print()` in reusable library code.
10. Make async boundaries deliberate. Do not hide blocking I/O inside async code, and do not mix sync and async call chains casually.
11. Make resource ownership explicit. Sessions, files, pools, temp directories, and sockets need clear creation and cleanup points.
12. Design with tests in mind. If a design requires extensive monkeypatching just to run core logic, the boundaries are wrong.

## Smells To Fix Early

- God modules that orchestrate, compute, persist, and format output together
- Boolean flags that switch unrelated behavior inside one function
- Functions that return differently shaped data depending on hidden modes
- Deep business logic that reads environment variables or current time directly
- Large `utils.py` files that become dependency dumping grounds
- Catch-all `except Exception` without translation, logging, or re-raise strategy
- Repeated raw dict keys instead of typed models or constants
- Tests that mock most of the system because there is no clean seam to call

## Answering Style

- Start with structure and responsibility boundaries before line-level style advice.
- Propose concrete package or module names when describing a design.
- Recommend the smallest change set that improves maintainability without speculative overengineering.
- Preserve intentional local conventions unless they materially harm correctness or testability.
- When reviewing code, call out dependency direction and where side effects should move.

## Reference Map

- [references/project-shapes.md](references/project-shapes.md) -> choose a repository or package shape for libraries, CLI apps, services, scheduled jobs, and pipelines.
- [references/design-rules.md](references/design-rules.md) -> apply detailed rules for interfaces, typing, configuration, errors, logging, persistence, and concurrency.
- [references/review-checklist.md](references/review-checklist.md) -> run a review pass, identify structural risks, and turn smells into a refactor plan.
