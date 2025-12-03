# FastAPI + LangGraph Streaming Sample

This is a minimal FastAPI service that composes a single-node LangGraph graph and streams its output back to the client as chunked responses.

## Prerequisites

- Python 3.10+
- An OpenAI API key exported as `OPENAI_API_KEY`

## Setup

```bash
cd /Users/rohanpatankar/a
poetry install
```

## Run the API

```bash
poetry run uvicorn app.main:app --reload
```

## Streaming Example

Use `curl` to stream model output as it is generated:

```bash
curl -N -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain LangGraph in one paragraph."}'
```

Each chunk arrives as soon as it is available, giving you a live view of LangGraph's streaming capability.


