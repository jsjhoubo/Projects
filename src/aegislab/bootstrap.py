from .models import PolicyStatus, PolicyVersion, Rule
from .service import AegisService


def seeded_service() -> AegisService:
    service = AegisService()
    service.store.add(PolicyVersion(
        tenant="acme", policy_id="support", version=1, status=PolicyStatus.ACTIVE,
        rules=[Rule(rule_id="secret", description="Block credentials", keywords=["password", "api key"])]))
    service.store.add(PolicyVersion(
        tenant="globex", policy_id="support", version=1, status=PolicyStatus.ACTIVE,
        rules=[Rule(rule_id="finance", description="Block confidential financial data", keywords=["routing number"])]))
    return service
