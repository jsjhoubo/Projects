from __future__ import annotations
from .models import CandidateRequest, PolicyVersion, Rule
from .research import DeterministicResearchLLM, PolicyGenerator
from .storage import ResearchStore


class PolicyRegistry:
    def __init__(self, store: ResearchStore):
        self.store = store
        self.generator = PolicyGenerator(DeterministicResearchLLM())

    def propose(self, tenant: str, policy_id: str, req: CandidateRequest) -> PolicyVersion:
        versions = self.store.list_versions(tenant, policy_id)
        current_rules = versions[-1].rules if versions else []
        rules = req.rules or self.generator.propose(current_rules, req.failures)
        policy = PolicyVersion(tenant=tenant, policy_id=policy_id,
                               version=self.store.next_version(tenant, policy_id), rules=rules)
        return self.store.add(policy)
