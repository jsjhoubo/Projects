from __future__ import annotations
from .models import EvaluationCase, EvaluationResult, PolicyVersion
from .runtime import KeywordGuard


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def evaluate(policy: PolicyVersion, cases: list[EvaluationCase], guard: KeywordGuard | None = None) -> EvaluationResult:
    guard = guard or KeywordGuard()
    tp = tn = fp = fn = 0
    details: list[dict] = []
    for case in cases:
        predicted, matched_rule = guard.classify(policy, case.message)
        if predicted and case.expected_block:
            tp += 1
        elif predicted and not case.expected_block:
            fp += 1
        elif not predicted and case.expected_block:
            fn += 1
        else:
            tn += 1
        details.append({"case_id": case.case_id, "predicted_block": predicted,
                        "expected_block": case.expected_block, "rule_id": matched_rule,
                        "category": case.category})
    total = len(cases)
    return EvaluationResult(
        tenant=policy.tenant, policy_id=policy.policy_id, version=policy.version,
        case_count=total, accuracy=safe_div(tp + tn, total),
        precision=safe_div(tp, tp + fp), recall=safe_div(tp, tp + fn),
        false_positive_rate=safe_div(fp, fp + tn),
        false_negative_rate=safe_div(fn, fn + tp), details=details,
    )
