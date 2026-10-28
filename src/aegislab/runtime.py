from __future__ import annotations
from .models import PolicyVersion
from .storage import ResearchStore


class KeywordGuard:
    """Local deterministic model, intentionally simple."""

    def classify(self, policy: PolicyVersion, message: str) -> tuple[bool, str | None]:
        text = message.lower()
        for rule in sorted(policy.rules, key=lambda r: r.priority):
            if rule.keywords and any(keyword.lower() in text for keyword in rule.keywords):
                return rule.action == "block", rule.rule_id
        return False, None


class GuardrailRuntime:
    def __init__(self, store: ResearchStore, classifier: KeywordGuard | None = None):
        self.store = store
        self.classifier = classifier or KeywordGuard()
        self._cache: dict[str, PolicyVersion] = {}

    def check(self, tenant: str, policy_id: str, message: str) -> tuple[bool, int, str | None]:
        # Notebook optimization: avoid repeated reads during inference.
        policy = self._cache.get(policy_id)
        if policy is None:
            policy = self.store.active(tenant, policy_id)
            self._cache[policy_id] = policy
        blocked, rule_id = self.classifier.classify(policy, message)
        return blocked, policy.version, rule_id

    def clear_cache(self, policy_id: str | None = None) -> None:
        if policy_id is None:
            self._cache.clear()
        else:
            self._cache.pop(policy_id, None)
