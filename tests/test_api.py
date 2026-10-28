from fastapi.testclient import TestClient
from aegislab.api import create_app
from aegislab.bootstrap import seeded_service


def test_health():
    client = TestClient(create_app(seeded_service()))
    assert client.get("/health").json() == {"status": "ok"}


def test_check_baseline():
    client = TestClient(create_app(seeded_service()))
    response = client.post("/v1/acme/check", json={"policy_id": "support", "message": "password"})
    assert response.status_code == 200
    assert response.json()["blocked"] is True
    assert response.json()["version"] == 1


def test_missing_policy_404():
    client = TestClient(create_app(seeded_service()))
    response = client.post("/v1/unknown/check", json={"policy_id": "support", "message": "password"})
    assert response.status_code == 404
