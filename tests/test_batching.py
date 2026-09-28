"""Batch invariance: the reason to trust every performance number that follows.

If running four requests together changes any of their outputs, then continuous
batching is not an optimisation -- it is a bug that happens to be faster.
"""

from __future__ import annotations

import pytest

from mini_vllm import LLMEngine, SamplingParams

pytestmark = pytest.mark.slow

PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The three primary colours are",
    "In one word, the opposite of hot is",
]


def drain(engine: LLMEngine) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            outputs.setdefault(out.request_id, []).extend(out.new_token_ids)
    return outputs


def submit(engine: LLMEngine, prompts: list[str], params: SamplingParams) -> None:
    for i, prompt in enumerate(prompts):
        engine.add_request(f"r{i}", engine.encode(prompt, chat=False), params)


@pytest.fixture(scope="module")
def engine(model_path):
    return LLMEngine(model_path, device="cpu", dtype="float32", num_blocks=128)


def run_solo(engine: LLMEngine, prompts: list[str], params: SamplingParams) -> dict:
    merged = {}
    for i, prompt in enumerate(prompts):
        engine.add_request(f"r{i}", engine.encode(prompt, chat=False), params)
        merged.update(drain(engine))
    return merged


def test_greedy_output_is_independent_of_batching(engine):
    params = SamplingParams(temperature=0.0, max_tokens=16, stop_token_ids=[])
    solo = run_solo(engine, PROMPTS, params)

    submit(engine, PROMPTS, params)
    batched = drain(engine)

    assert set(solo) == set(batched)
    for request_id in solo:
        assert batched[request_id] == solo[request_id], f"{request_id} changed under batching"


def test_seeded_sampling_is_independent_of_batching(engine):
    """Seeded requests draw from their own generator, so batch composition cannot move
    them. This is why the sampler does not simply call multinomial once for the batch."""
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=12, seed=7, stop_token_ids=[])
    solo = run_solo(engine, PROMPTS[:2], params)

    submit(engine, PROMPTS[:2], params)
    batched = drain(engine)

    for request_id in solo:
        assert batched[request_id] == solo[request_id]


def test_output_survives_preemption(model_path):
    """A pool too small to hold every request forces preempt-and-recompute. Recomputing
    must land on exactly the same tokens."""
    params = SamplingParams(temperature=0.0, max_tokens=20, stop_token_ids=[])

    roomy = LLMEngine(model_path, device="cpu", dtype="float32", num_blocks=128)
    submit(roomy, PROMPTS, params)
    expected = drain(roomy)

    # 4 blocks x 16 tokens = 64 token slots; the four requests need roughly 104 at
    # once, so the pool cannot hold them and the scheduler has to evict and recompute.
    cramped = LLMEngine(model_path, device="cpu", dtype="float32", num_blocks=4, block_size=16)
    submit(cramped, PROMPTS, params)
    actual = drain(cramped)

    assert cramped.scheduler.num_preemptions > 0, "test did not exercise preemption"
    assert actual == expected


def test_prefix_cache_does_not_change_output(model_path):
    """The whole point: reusing a cached prefix must be indistinguishable from
    recomputing it."""
    params = SamplingParams(temperature=0.0, max_tokens=16, stop_token_ids=[])
    shared = "You are a helpful assistant. Answer in one short sentence. Question: "
    prompts = [shared + q for q in ("Name a colour.", "Name a fruit.")]

    off = LLMEngine(
        model_path, device="cpu", dtype="float32", num_blocks=128, enable_prefix_cache=False
    )
    submit(off, prompts, params)
    expected = drain(off)

    on = LLMEngine(model_path, device="cpu", dtype="float32", num_blocks=128)
    submit(on, prompts, params)
    actual = drain(on)

    assert on.scheduler.num_cached_tokens > 0, "test did not exercise the prefix cache"
    assert off.scheduler.num_cached_tokens == 0
    assert actual == expected
