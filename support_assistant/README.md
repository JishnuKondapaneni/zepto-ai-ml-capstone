\## Verification



The Support Assistant was tested locally with FastAPI and Docker.



Verified:

\- Policy question returns HTTP 200 with retrieved sources.

\- General question returns the required policy-only response.

\- `MOCK\_LLM=1` provides deterministic graded-baseline responses.

\- Docker image builds successfully.

\- Docker container starts successfully on port 7860.

