# Scripts

All first-party Experiment 12 code is here.

- Top-level `*12.py`: runnable experiment, validation, and analysis modules.
- `core/`: artifact, budget, schema, and provider transport infrastructure.
- `domains/`: benchmark adapters.
- `monitors/`: active/passive observation implementations.
- `tests/`: provider-free unit and integration tests.
- `posthoc/`: one-off audits, recovery tools, and paper-material builders.
- `integrations/`: code that must be copied into an external pinned checkout.

Run modules from the repository root with their stable names, for example:

```bash
python -m experiments12.cli12 selftest
python -m experiments12.prepare_deployment12 --help
```

`paths12.py` is the single source of truth for the repository layout.
