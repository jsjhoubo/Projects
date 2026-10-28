from __future__ import annotations
from .datasets import load_cases
from .metrics import evaluate
from .models import CandidateRequest, EvaluationResult, PolicyVersion
from .registry import PolicyRegistry
from .runtime import GuardrailRuntime
from .storage import ResearchStore


class AegisService:
    def __init__(self, store: ResearchStore | None = None):
        self.store = store or ResearchStore()
        self.registry = PolicyRegistry(self.store)
        self.runtime = GuardrailRuntime(self.store)
        self.evaluations: dict[tuple[str, str, int], EvaluationResult] = {}

    def propose(self, tenant: str, policy_id: str, req: CandidateRequest) -> PolicyVersion:
        return self.registry.propose(tenant, policy_id, req)

    def evaluate(self, tenant: str, policy_id: str, version: int, dataset: str = "regression") -> EvaluationResult:
        policy = self.store.get(tenant, policy_id, version)
        result = evaluate(policy, load_cases(tenant, dataset))
        self.evaluations[(tenant, policy_id, version)] = result
        return result

    def activate(self, tenant: str, policy_id: str, version: int) -> PolicyVersion:
        return self.store.activate(tenant, policy_id, version)

    def check(self, tenant: str, policy_id: str, message: str):
        return self.runtime.check(tenant, policy_id, message)
