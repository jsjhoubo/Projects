# Curl examples

```bash
uvicorn aegislab.api:app --reload
curl localhost:8000/health
curl -X POST localhost:8000/v1/acme/policies/support/candidates \
  -H 'content-type: application/json' \
  -d '{"rules":[{"rule_id":"deny-secret","description":"Block secrets","keywords":["password"]}]}'
curl -X POST localhost:8000/v1/acme/policies/support/evaluate/1
```

The default `app` starts with empty storage. Unit tests and scripts demonstrate seeding.
