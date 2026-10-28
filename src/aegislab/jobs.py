from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4
from .service import AegisService


@dataclass
class Job:
    job_id: str
    tenant: str
    policy_id: str
    version: int
    state: str = "queued"
    output: dict | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchJobRunner:
    def __init__(self, service: AegisService, workers: int = 2):
        self.service = service
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: dict[str, Job] = {}
        self.lock = Lock()

    def submit_evaluation(self, tenant: str, policy_id: str, version: int) -> Job:
        job = Job(job_id=uuid4().hex, tenant=tenant, policy_id=policy_id, version=version)
        with self.lock:
            self.jobs[job.job_id] = job
        self.pool.submit(self._run, job)
        return job

    def _run(self, job: Job):
        job.state = "running"
        try:
            result = self.service.evaluate(job.tenant, job.policy_id, job.version)
            job.output = result.model_dump(mode="json")
            job.state = "completed"
        except Exception as exc:
            job.error = repr(exc)
            job.state = "failed"

    def get(self, job_id: str) -> Job:
        return self.jobs[job_id]
