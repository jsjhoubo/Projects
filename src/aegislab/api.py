from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from .models import CandidateRequest, CheckRequest, CheckResponse
from .service import AegisService
from .storage import MissingPolicy


def create_app(service: AegisService | None = None) -> FastAPI:
    app = FastAPI(title="AegisLab Research Handoff", version="0.1")
    svc = service or AegisService()
    app.state.service = svc

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/{tenant}/policies/{policy_id}/candidates")
    def propose(tenant: str, policy_id: str, req: CandidateRequest):
        try:
            return svc.propose(tenant, policy_id, req)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/{tenant}/policies/{policy_id}")
    def versions(tenant: str, policy_id: str):
        return svc.store.list_versions(tenant, policy_id)

    @app.post("/v1/{tenant}/policies/{policy_id}/evaluate/{version}")
    def evaluation(tenant: str, policy_id: str, version: int, dataset: str = Query("regression")):
        try:
            return svc.evaluate(tenant, policy_id, version, dataset)
        except MissingPolicy as e:
            raise HTTPException(status_code=404, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/v1/{tenant}/policies/{policy_id}/activate/{version}")
    def activate(tenant: str, policy_id: str, version: int):
        try:
            return svc.activate(tenant, policy_id, version)
        except MissingPolicy as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/v1/{tenant}/check", response_model=CheckResponse)
    def check(tenant: str, req: CheckRequest):
        try:
            blocked, version, rule_id = svc.check(tenant, req.policy_id, req.message)
            return CheckResponse(blocked=blocked, policy_id=req.policy_id, version=version, rule_id=rule_id)
        except MissingPolicy as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app


app = create_app()
