from aegislab.bootstrap import seeded_service
from aegislab.models import CandidateRequest, Rule


def main():
    svc = seeded_service()
    candidate = svc.propose("acme", "support", CandidateRequest(rules=[
        Rule(rule_id="secret", description="Deny credentials", keywords=["password", "api key"]),
        Rule(rule_id="medical", description="Deny health records", keywords=["medical record"]),
    ]))
    print("candidate", candidate.version)
    print("evaluation", svc.evaluate("acme", "support", candidate.version).model_dump_json(indent=2))
    print("before", svc.check("acme", "support", "send medical record"))
    print("activated", svc.activate("acme", "support", candidate.version).version)
    print("after", svc.check("acme", "support", "send medical record"))


if __name__ == "__main__":
    main()
