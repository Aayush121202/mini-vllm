"""The HTTP surface, driven through FastAPI's test client."""

from __future__ import annotations

import json
import time

import pytest

from mini_vllm.engine import LLMEngine
from mini_vllm.server import build_app

pytestmark = pytest.mark.slow

MODEL_NAME = "test-model"


@pytest.fixture(scope="module")
def client(model_path):
    from fastapi.testclient import TestClient

    engine = LLMEngine(model_path, device="cpu", dtype="float32", num_blocks=128)
    engine.get_tokenizer()
    with TestClient(build_app(engine, MODEL_NAME)) as client:
        yield client


def chat_body(content: str, **kwargs) -> dict:
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 16, **kwargs}


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/v1/models").json()
    assert body["data"][0]["id"] == MODEL_NAME


def test_stats_reports_scheduler_state(client):
    stats = client.get("/stats").json()
    assert {"running", "waiting", "kv_utilization", "preemptions"} <= stats.keys()


def test_chat_completion(client):
    body = client.post("/v1/chat/completions", json=chat_body("Name a colour.")).json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"].strip()
    assert choice["finish_reason"] in ("stop", "length")
    assert body["usage"]["prompt_tokens"] > 0


def test_streaming_matches_the_non_streaming_answer(client):
    """Same seed, same prompt: streaming must not change the text, only how it arrives."""
    body = chat_body("Name a colour.", temperature=0.0)
    whole = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]

    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as response:
        assert response.status_code == 200
        chunks = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert chunks[-1] == "data: [DONE]"
    streamed = ""
    for line in chunks[:-1]:
        delta = json.loads(line[len("data: ") :])["choices"][0]["delta"]
        streamed += delta.get("content", "")
    assert streamed == whole["content"]


def test_stop_string_truncates_the_response(client):
    body = chat_body("Count: one two three four five", temperature=0.0, stop=["three"])
    content = client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"][
        "content"
    ]
    assert "three" not in content


def test_empty_messages_rejected(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_concurrent_requests_are_batched(client):
    """Three clients at once must all get answers, and the engine must batch them
    rather than serialising."""
    from concurrent.futures import ThreadPoolExecutor

    prompts = ["Name a colour.", "Name a fruit.", "Name a country."]
    with ThreadPoolExecutor(max_workers=3) as pool:
        responses = list(
            pool.map(
                lambda p: client.post("/v1/chat/completions", json=chat_body(p)),
                prompts,
            )
        )
    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["choices"][0]["message"]["content"].strip() for r in responses)


def disconnect_midway(client, prompt: str) -> dict:
    body = chat_body(prompt, max_tokens=256, stream=True)
    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        for _ in zip(range(2), response.iter_lines()):
            pass  # read a couple of chunks, then bail out of the context manager

    deadline = time.time() + 5
    while client.get("/stats").json()["running"] and time.time() < deadline:
        pass
    return client.get("/stats").json()


def test_disconnecting_mid_stream_frees_the_slot(client):
    """A client that hangs up should stop costing compute immediately, not keep a batch
    slot until max_tokens.

    Blocks retained afterwards are the prefix cache's, not a leak -- it deliberately
    holds a reference so the next request with that prefix can reuse it. The invariant
    that actually matters is that the request's own blocks come back, so repeating the
    same disconnect does not grow the pool.
    """
    first = disconnect_midway(client, "Write a long story about a lighthouse.")
    assert first["running"] == 0 and first["waiting"] == 0

    second = disconnect_midway(client, "Write a long story about a lighthouse.")
    assert second["blocks_used"] == first["blocks_used"], "blocks leaked across disconnects"
