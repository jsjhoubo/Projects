from aegislab.bootstrap import seeded_service
from aegislab.models import CandidateRequest, Rule


def test_candidate_is_not_activated_on_creation():
    svc = seeded_service()
    p = svc.propose("acme", "support", CandidateRequest(rules=[
        Rule(rule_id="new", description="Block blue", keywords=["blue"])
    ]))
    assert p.version == 2
    assert svc.store.active("acme", "support").version == 1


def test_evaluation_uses_correct_dataset():
    svc = seeded_service()
    result = svc.evaluate("acme", "support", 1)
    assert result.case_count == 6
    assert result.accuracy == 1.0


def test_store_returns_copies():
    svc = seeded_service()
    version = svc.store.get("acme", "support", 1)
    version.rules.clear()
    assert len(svc.store.get("acme", "support", 1).rules) == 1
