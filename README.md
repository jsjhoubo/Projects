# AegisLab — Research-to-Production Guardrail Service

> **Interview simulation**. A fictional, LATTICE-inspired research prototype. Not Distyl's internal code or actual interview question.

A customer wants the guardrail team to ship a safe policy rollout workflow for multiple customer assistants. Your research team has handed you a working-but-incomplete service.

## Your assignment (45-minute live exercise)

**Customer escalation:** We have occasionally observed an updated guardrail policy blocking legitimate requests after deployment. Build a **safe promotion workflow** that evaluates a candidate against representative cases, prevents regressions against the active version, and allows rollback. A legitimate client request should never accidentally be evaluated using another client's policy.

The customer needs a demo in 45 minutes. You may use an AI coding assistant. You are judged on correctness, prioritization, test coverage, and how you defend your decisions—not on implementing everything.

**Requirements deliberately incomplete.** Ask the interviewer about acceptance criteria, error handling, and production constraints. You can propose reasonable assumptions.

### Suggested first actions

```bash
python -m pip install -e '.[dev]'
pytest -q
uvicorn aegislab.api:app --reload
```

No API key, network, or paid model is required. A deterministic local model is included.

### Basic flow

1. `POST /v1/{tenant}/policies/{policy_id}/candidates` — propose candidate rules.
2. `POST /v1/{tenant}/policies/{policy_id}/evaluate/{version}` — measure performance on cases.
3. `POST /v1/{tenant}/policies/{policy_id}/activate/{version}` — activate a version.
4. `POST /v1/{tenant}/check` — enforce current policy against a message.
5. `GET /v1/{tenant}/policies/{policy_id}` — inspect versions.

### Constraints

- Keep the public API backwards-compatible unless you explicitly explain why not.
- Treat test inputs as untrusted; a case can include user-controlled strings.
- The existing `evaluate` endpoint is not itself an approval to deploy.
- Existing tenants may have different policies with identical policy IDs.
- Unit tests should run without remote LLMs or databases.

### During the interview

You will receive a **change request** after your first working implementation. Do not ask for it ahead of time.

### What is already here?

A FastAPI app, policy registry, in-memory research storage, evaluation metrics, generator, runtime safety classifier, a batch job helper, test suite, and sample data. Some components are intentionally imperfect. It is your job to decide what needs to change.

### Deliverables

Working code, targeted tests, a brief explanation of why the rollout is safe and what still needs productionization.


## Expanded repository

The `aegislab.enterprise` package contains deterministic risk-signal extraction and infrastructure primitives used to make realistic navigation, test discovery, and dependency selection possible. These are **not** all wired into the request path. Treat them as existing research-team components that may or may not be relevant to the customer incident.

The supplied `data/enterprise` samples are **synthetic** and not intended to represent real-world safety distributions or fair evaluation labels. Do not use them as proof of guardrail quality.

Run the full suite with `pytest -q`, or begin with the narrow `tests/test_api.py tests/test_service.py`.
