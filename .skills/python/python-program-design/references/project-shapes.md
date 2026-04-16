# Project Shapes

## Table Of Contents

1. Choose the smallest shape
2. Library package
3. CLI application
4. HTTP service
5. Scheduled job or data pipeline
6. Mixed repository rules

## Choose The Smallest Shape

Pick the repository or package shape from the runtime surface, not from vague ambition.

- Choose a library package when other code imports the core logic.
- Choose a CLI application when the main interaction is a command.
- Choose an HTTP service when the system exposes network endpoints and request lifecycles.
- Choose a scheduled job or pipeline when the system pulls data, transforms it, and writes outputs on a cadence.
- Do not add extra layers only to look enterprise. Every layer must own a distinct responsibility.

## Library Package

Use this when the code is mainly imported.

```text
my_package/
  __init__.py
  models.py
  errors.py
  services.py
  ports.py
  adapters/
tests/
```

Rules:

- Keep public imports small and stable in `__init__.py`.
- Put domain models and public service functions near the top level.
- Put infrastructure adapters behind ports or narrow helper functions.
- Keep command-line behavior out of the reusable package.

## CLI Application

Use this when the primary interface is terminal commands.

```text
app/
  __main__.py
  cli.py
  config.py
  commands/
  domain/
  infra/
tests/
```

Rules:

- Parse arguments in `cli.py` or `commands/`, not inside domain logic.
- Keep each command thin: validate input, call a use case, format output.
- Put reusable business logic in `domain/` or `services/`, not in command handlers.
- Keep side-effect setup such as logging, config, and clients near the entrypoint.

## HTTP Service

Use this when the main surface is request handling.

```text
service/
  app.py
  config.py
  api/
  domain/
  use_cases/
  infra/
tests/
```

Rules:

- Treat `api/` as translation code between HTTP and internal models.
- Keep request validation and serialization separate from business rules.
- Put transaction or workflow orchestration in `use_cases/` when it spans more than one dependency.
- Keep persistence, queues, and external APIs in `infra/`.

## Scheduled Job Or Data Pipeline

Use this when the process is started by time, events, or batch input.

```text
job/
  main.py
  config.py
  orchestration.py
  sources.py
  transforms.py
  sinks.py
tests/
```

Rules:

- Keep `sources.py` responsible for fetching data.
- Keep `transforms.py` deterministic when possible.
- Keep `sinks.py` responsible for writes and side effects.
- Keep `orchestration.py` responsible for ordering the steps and handling retries.
- Separate checkpointing or idempotency rules from the core transform logic.

## Mixed Repository Rules

Some repositories contain a library plus one or more delivery surfaces. Keep the direction clear.

- Let CLI, service, and job entrypoints depend on shared domain code.
- Do not let shared domain code depend on the CLI or web framework.
- Keep each surface in its own package when the startup logic differs materially.
- Prefer one `config.py` per surface only if their runtime settings truly differ.
- Split a package only after there is a clear ownership boundary, not just because the tree looks large.
