# AegisLab — Distyl-inspired systems mock

This is a **separate GitHub branch**, not a change to the existing Projects default branch.

## Status

The complete 15,851-line, 160-Python-file exercise is available in the ChatGPT conversation as `aegislab-distyl-mock-10k.zip`. **The complete ZIP has not been committed to this branch** because the available GitHub connector cannot access the generated container archive bytes.

This landing page records the intended exercise, not a completed source upload. Do not treat the repository branch as runnable until the local ZIP is pushed.

## Exercise

Build a safe multi-tenant policy promotion workflow, evaluate against regression datasets, enforce atomic activation/rollback, and handle production incidents. Begin with tests, then trace active policy resolution through storage, cache and API. The project includes FastAPI, pytest, synthetic enterprise datasets, and a large set of research/infrastructure components.

## Upload locally

Unzip the archive, then from the extracted project root:

```bash
git init
git remote add origin https://github.com/jsjhoubo/Projects.git
git fetch origin distyl-lattice-systems-mock
git checkout -b distyl-lattice-systems-mock origin/distyl-lattice-systems-mock
git add .
git commit -m "Add Distyl research-to-production mock"
git push -u origin distyl-lattice-systems-mock
```

The exercise is fictional and not a Distyl internal codebase.
