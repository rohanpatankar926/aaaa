from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .graph import stream_prompt

app = FastAPI(title="FastAPI LangGraph Streaming Demo")


class ChatRequest(BaseModel):
    user_query: str
    thread_id: str


@app.get("/")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(body: ChatRequest) -> StreamingResponse:
    def iterator():
        sample_prompt = f"Thread ID: {body.thread_id}\nUser Query: {body.user_query}"
        for chunk in stream_prompt(sample_prompt):
            yield chunk

    return StreamingResponse(iterator(), media_type="text/plain")

