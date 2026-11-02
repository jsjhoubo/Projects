# RecordFlow — Business Entity Ingestion (Mock Interview)

You joined a data products team that ingests supplier updates about local businesses. Supplier records are unreliable: attributes may be missing, identifiers may be malformed, and different suppliers may report overlapping businesses.

This repository is a reduced production-like service. It has existing domain models, adapters, validators, a storage abstraction, and an ingestion coordinator. **Do not replace the repository with a standalone solution.** Work within the existing interfaces.

## Interview conditions
- Language: Python 3.10+; standard library only for application code.
- Timebox: 60 minutes for progressive tasks. Start with **LEVEL_1.md only**.
- Inspect existing code, run tests, and ask clarifying questions as needed.
- Do not change existing test expectations, public API signatures, or fixture files.
- You may add your own tests.
- Future levels are deliberately not described here. Ask the interviewer to unlock the next level after Level 1 passes.

## Setup

```bash
python -m pip install pytest
python -m pytest -q
```

## Repository layout

- `recordflow/domain.py` — immutable domain data types
- `recordflow/validation.py` — parsing and normalization rules
- `recordflow/storage.py` — in-memory storage adapter
- `recordflow/suppliers.py` — supplier payload adapters
- `recordflow/ingestion.py` — ingestion coordinator (**your task touches this**)
- `recordflow/metrics.py` — ingest counters
- `recordflow/query.py` — business lookup
- `recordflow/audit.py` — audit events
- `recordflow/clock.py` — deterministic timestamps
- `tests/` — visible unit tests

The repository intentionally contains more code than the Level 1 change requires. Read the relevant contracts and tests instead of every file in full.
