from __future__ import annotations
import json
from collections.abc import Callable
from .models import Rule


class PolicyGenerator:
    """Research interface. LLM supplied by caller; no external dependency."""

    def __init__(self, llm: Callable[[str], str]):
        self.llm = llm

    def propose(self, old_rules: list[Rule], failures: list[str]) -> list[Rule]:
        prompt = (
            "Rewrite guardrail rules as JSON array of {rule_id,description,keywords,action,priority}.\n"
            f"Current rules: {json.dumps([r.model_dump() for r in old_rules])}\n"
            f"Observed failures: {json.dumps(failures)}"
        )
        output = self.llm(prompt)
        raw = json.loads(output)
        return [Rule.model_validate(item) for item in raw]


class DeterministicResearchLLM:
    """Stub for local demonstrations; not a safety-trained model."""

    def __call__(self, prompt: str) -> str:
        return json.dumps([
            {"rule_id": "deny-secrets", "description": "Block requests for secrets",
             "keywords": ["password", "api key"], "action": "block", "priority": 10}
        ])
