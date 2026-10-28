from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PolicyStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"


class Rule(BaseModel):
    rule_id: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    action: Literal["block", "allow"] = "block"
    priority: int = 100


class PolicyVersion(BaseModel):
    tenant: str
    policy_id: str
    version: int
    rules: list[Rule]
    status: PolicyStatus = PolicyStatus.DRAFT
    created_at: datetime = Field(default_factory=utcnow)
    source: str = "research"


class EvaluationCase(BaseModel):
    case_id: str
    message: str
    expected_block: bool
    category: str = "general"
    weight: float = Field(default=1.0, gt=0)


class EvaluationResult(BaseModel):
    tenant: str
    policy_id: str
    version: int
    case_count: int
    accuracy: float
    precision: float
    recall: float
    false_positive_rate: float
    false_negative_rate: float
    passed: bool = False
    details: list[dict] = Field(default_factory=list)


class CheckRequest(BaseModel):
    policy_id: str
    message: str


class CheckResponse(BaseModel):
    blocked: bool
    policy_id: str
    version: int
    rule_id: str | None = None


class CandidateRequest(BaseModel):
    failures: list[str] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
