# Customer incident — interview handout

The enterprise customer has reported inconsistent allow/block decisions after frequent policy promotions. They operate several assistants under distinct tenants and need confident rollback when a regression is detected. Some policies are large, and ingestion and evaluation happen asynchronously.

You have 45 minutes. Investigate the code, make a narrow working change, prove it using tests, and tell the customer what is still unsafe. You may use AI coding assistants. The interviewer may change the requirements mid-exercise.

Do **not** assume a passing baseline test suite proves concurrency, tenancy, or rollback correctness. The task statement intentionally leaves policy approval thresholds and fallback behavior undefined. Ask for the highest-risk acceptance criteria.

This is a fictional training repository, not Distyl interview IP.
