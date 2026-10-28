# Architecture notes from the research team

The prototype processes a tenant-specific policy and a user message via a deterministic stand-in for a model. The goal of this simulation is to test production-engineering judgment rather than benchmark a language model.

```
Client -> FastAPI -> AegisService -> GuardrailRuntime -> KeywordGuard
                              |-> PolicyRegistry -> PolicyGenerator
                              |-> ResearchStore
                              |-> Evaluate -> JSONL dataset
```

The system intentionally uses process-local storage so the exercise requires no external services. A production implementation would replace that piece with durable state, auditing, authn/authz, and transactional writes.

Note: The research team did not provide a formal rollout safety specification. Please discuss acceptance thresholds and failure modes with the interviewer.
