"""OpenAI-compatible HTTP server.

One background task drives ``engine.step()``; HTTP handlers only put requests into the
scheduler and read from a per-request queue. Nothing in a request handler ever touches
the model, which is what keeps one slow client from stalling the batch.

Run with: ``python -m mini_vllm.server --model Qwen/Qwen2.5-0.5B-Instruct``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mini_vllm.detokenizer import IncrementalDetokenizer
from mini_vllm.engine import LLMEngine
from mini_vllm.request import SamplingParams

# How long the engine loop sleeps when there is nothing to do. Short enough that a new
# request is picked up promptly, long enough not to spin a core.
IDLE_POLL_SECONDS = 0.005


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = -1
    seed: int | None = None
    stop: list[str] | str | None = None
    stream: bool = False

    def to_sampling_params(self, eos_token_id: int | None) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop_token_ids=[eos_token_id] if eos_token_id is not None else [],
            seed=self.seed,
        )

    @property
    def stop_strings(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class AsyncEngine:
    """Async shell around the synchronous engine."""

    def __init__(self, engine: LLMEngine) -> None:
        self.engine = engine
        self.queues: dict[str, asyncio.Queue] = {}
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _run(self) -> None:
        while True:
            if not self.engine.has_unfinished_requests():
                await asyncio.sleep(IDLE_POLL_SECONDS)
                continue
            for output in self.engine.step():
                queue = self.queues.get(output.request_id)
                if queue is not None:
                    queue.put_nowait(output)
            # Yield between steps so handlers can enqueue new work. The step itself is
            # blocking; this is the seam where a production server would move the
            # forward pass onto its own thread.
            await asyncio.sleep(0)

    async def generate(self, prompt_token_ids, params, stop_strings) -> AsyncIterator:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        queue: asyncio.Queue = asyncio.Queue()
        self.queues[request_id] = queue
        detok = IncrementalDetokenizer(self.engine.get_tokenizer(), stop_strings=stop_strings)
        self.engine.add_request(request_id, prompt_token_ids, params)

        try:
            while True:
                output = await queue.get()
                for token_id in output.new_token_ids:
                    text = detok.add_token(token_id)
                    if text:
                        yield text, None
                if detok.stopped:
                    self.engine.abort(request_id)
                    yield "", "stop"
                    return
                if output.finished:
                    tail = detok.flush()
                    if tail:
                        yield tail, None
                    yield "", output.finish_reason
                    return
        finally:
            # Covers the client-disconnect case: Starlette closes this generator when
            # the connection drops, and without the abort the request would keep
            # occupying a batch slot and burning compute until it hit max_tokens.
            self.queues.pop(request_id, None)
            self.engine.abort(request_id)


def build_app(engine: LLMEngine, served_model_name: str) -> FastAPI:
    async_engine = AsyncEngine(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async_engine.start()
        yield
        await async_engine.stop()

    app = FastAPI(title="mini-vllm", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        return engine.stats()

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": served_model_name, "object": "model", "owned_by": "mini-vllm"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        tokenizer = engine.get_tokenizer()
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        text = tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_token_ids = tokenizer(text, add_special_tokens=False).input_ids
        params = request.to_sampling_params(tokenizer.eos_token_id)
        stream = async_engine.generate(prompt_token_ids, params, request.stop_strings)
        created = int(time.time())
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if request.stream:
            return StreamingResponse(
                _stream_chunks(stream, request_id, created, served_model_name),
                media_type="text/event-stream",
            )

        content, finish_reason = "", "stop"
        async for delta, reason in stream:
            content += delta
            if reason is not None:
                finish_reason = reason
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(tokenizer(content).input_ids),
                    "total_tokens": len(prompt_token_ids) + len(tokenizer(content).input_ids),
                },
            }
        )

    return app


async def _stream_chunks(stream, request_id: str, created: int, model: str) -> AsyncIterator[str]:
    def chunk(delta: dict, finish_reason: str | None) -> str:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield chunk({"role": "assistant", "content": ""}, None)
    async for text, finish_reason in stream:
        if text:
            yield chunk({"content": text}, None)
        if finish_reason is not None:
            yield chunk({}, finish_reason)
    yield "data: [DONE]\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-vllm OpenAI-compatible server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()

    import uvicorn

    engine = LLMEngine(
        args.model,
        device=args.device,
        dtype=args.dtype,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_cache=not args.no_prefix_cache,
    )
    engine.get_tokenizer()  # load before serving so the first request is not slow
    print(repr(engine.runner))
    uvicorn.run(build_app(engine, args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
