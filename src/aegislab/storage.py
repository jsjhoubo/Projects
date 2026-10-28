from __future__ import annotations
from collections import defaultdict
from threading import RLock
from .models import PolicyStatus, PolicyVersion


class MissingPolicy(KeyError):
    pass


class ResearchStore:
    """Notebook-era storage. Separate instances can be used in tests."""

    def __init__(self):
        self._policies: dict[tuple[str, str], list[PolicyVersion]] = defaultdict(list)
        self._lock = RLock()

    def add(self, policy: PolicyVersion) -> PolicyVersion:
        with self._lock:
            key = (policy.tenant, policy.policy_id)
            if any(p.version == policy.version for p in self._policies[key]):
                raise ValueError("duplicate policy version")
            self._policies[key].append(policy.model_copy(deep=True))
        return policy

    def list_versions(self, tenant: str, policy_id: str) -> list[PolicyVersion]:
        with self._lock:
            return [p.model_copy(deep=True) for p in self._policies[(tenant, policy_id)]]

    def get(self, tenant: str, policy_id: str, version: int) -> PolicyVersion:
        for policy in self.list_versions(tenant, policy_id):
            if policy.version == version:
                return policy
        raise MissingPolicy(f"{tenant}/{policy_id}/{version}")

    def active(self, tenant: str, policy_id: str) -> PolicyVersion:
        matches = [p for p in self.list_versions(tenant, policy_id) if p.status == PolicyStatus.ACTIVE]
        if not matches:
            raise MissingPolicy(f"no active policy: {tenant}/{policy_id}")
        return matches[-1]

    def next_version(self, tenant: str, policy_id: str) -> int:
        versions = self.list_versions(tenant, policy_id)
        return max((p.version for p in versions), default=0) + 1

    def activate(self, tenant: str, policy_id: str, version: int) -> PolicyVersion:
        with self._lock:
            key = (tenant, policy_id)
            versions = self._policies[key]
            target = next((p for p in versions if p.version == version), None)
            if target is None:
                raise MissingPolicy(f"{tenant}/{policy_id}/{version}")
            for p in versions:
                if p.status == PolicyStatus.ACTIVE:
                    p.status = PolicyStatus.ARCHIVED
            target.status = PolicyStatus.ACTIVE
            return target.model_copy(deep=True)
