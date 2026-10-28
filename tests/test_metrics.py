from aegislab.metrics import evaluate
from aegislab.models import EvaluationCase, PolicyVersion, Rule


def test_accuracy_basic():
    policy = PolicyVersion(tenant="acme", policy_id="p", version=1,
                           rules=[Rule(rule_id="r", description="deny password", keywords=["password"])])
    cases = [EvaluationCase(case_id="1", message="password", expected_block=True),
             EvaluationCase(case_id="2", message="weather", expected_block=False)]
    result = evaluate(policy, cases)
    assert result.accuracy == 1
    assert result.precision == 1
    assert result.recall == 1


def test_empty_cases_return_zero():
    policy = PolicyVersion(tenant="acme", policy_id="p", version=1, rules=[])
    assert evaluate(policy, []).accuracy == 0
