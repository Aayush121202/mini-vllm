"""End-to-end engine behaviour. Requires weights."""

from __future__ import annotations

import pytest

from mini_vllm import LLMEngine, SamplingParams

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine(model_path):
    return LLMEngine(model_path, device="cpu", dtype="float32")


def test_generate_returns_text(engine):
    text = engine.generate("Name one colour.", SamplingParams(max_tokens=16))
    assert isinstance(text, str) and text.strip()
    assert not engine.has_unfinished_requests()


def test_step_is_one_token_per_call(engine):
    engine.add_request("r1", [40, 1079, 264], SamplingParams(max_tokens=4, stop_token_ids=[]))
    outputs = engine.step()
    assert len(outputs) == 1
    assert len(outputs[0].new_token_ids) == 1
    assert not outputs[0].finished

    steps = 1
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    assert steps == 4  # one prefill step + three decode steps, max_tokens=4


def test_abort_drops_request(engine):
    engine.add_request("r2", [40, 1079, 264], SamplingParams(max_tokens=32))
    engine.step()
    engine.abort("r2")
    assert not engine.has_unfinished_requests()


def test_seeded_sampling_is_reproducible(engine):
    def run():
        engine.add_request(
            "seeded", [40, 1079, 264], SamplingParams(temperature=0.9, max_tokens=12, seed=7)
        )
        out: list[int] = []
        while engine.has_unfinished_requests():
            for o in engine.step():
                out.extend(o.new_token_ids)
        return out

    assert run() == run()


def accelerator() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return ""


@pytest.fixture(scope="module")
def accel_engine(model_path):
    """An engine on whatever accelerator this machine has.

    The rest of the suite runs on CPU in fp32, which structurally cannot catch two whole
    classes of bug: non-parameter buffers (rope tables) left behind when the weights
    move, and anything that only misbehaves in fp16. These tests are the CI for
    'does it work on the hardware people actually serve on'.
    """
    device = accelerator()
    if not device:
        pytest.skip("no accelerator available")
    return LLMEngine(model_path, device=device, num_blocks=128)


ACCEL_PROMPTS = ["The capital of France is", "Water boils at", "The opposite of hot is"]

# Longer than two 16-token blocks (38 tokens), so the shared region spans whole
# blocks regardless of what follows it. A prefix shorter than block_size would make
# these tests pass or fail on whether the first question token happened to match.
SHARED_PREFIX = (
    "You are a helpful assistant. Answer every question in exactly one short sentence, "
    "using plain language, with no preamble, no lists, and no restatement of the "
    "question itself. Question: "
)


def collect(engine, prompts, params) -> dict[str, list[int]]:
    for i, prompt in enumerate(prompts):
        engine.add_request(f"a{i}", engine.encode(prompt, chat=False), params)
    outputs: dict[str, list[int]] = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            outputs.setdefault(out.request_id, []).extend(out.new_token_ids)
    return outputs


def test_accelerator_generates(accel_engine):
    """Catches device-placement regressions -- notably non-parameter buffers left on CPU
    while the weights move to the accelerator."""
    params = SamplingParams(max_tokens=4, stop_token_ids=[])
    engine = accel_engine
    engine.add_request("dev", [40, 1079, 264], params)
    tokens: list[int] = []
    while engine.has_unfinished_requests():
        for out in engine.step():
            tokens.extend(out.new_token_ids)
    assert len(tokens) == 4


def test_accelerator_batch_invariance(accel_engine):
    """Batching must not change output on the accelerator either -- half the point of
    the paged gather is that it behaves identically to running one sequence at a time."""
    params = SamplingParams(temperature=0.0, max_tokens=12, stop_token_ids=[])
    engine = accel_engine

    solo: dict[str, list[int]] = {}
    for i, prompt in enumerate(ACCEL_PROMPTS):
        engine.add_request(f"a{i}", engine.encode(prompt, chat=False), params)
        while engine.has_unfinished_requests():
            for out in engine.step():
                solo.setdefault(out.request_id, []).extend(out.new_token_ids)

    assert collect(engine, ACCEL_PROMPTS, params) == solo


def test_accelerator_prefix_cache(accel_engine):
    """The scatter/gather indexing that backs prefix reuse is device-specific code."""
    params = SamplingParams(temperature=0.0, max_tokens=8, stop_token_ids=[])
    engine = accel_engine
    prompts = [SHARED_PREFIX + q for q in ("Name a colour.", "Name a fruit.")]

    before = engine.scheduler.num_cached_tokens
    collect(engine, prompts, params)
    assert engine.scheduler.num_cached_tokens > before


def test_accelerator_seeded_sampling_is_reproducible(accel_engine):
    """Per-row generators have to be created on the right device; a CPU generator with
    CUDA probs raises, and a shared generator would break batch independence."""
    params = SamplingParams(temperature=0.9, max_tokens=8, seed=13, stop_token_ids=[])
    first = collect(accel_engine, ACCEL_PROMPTS[:2], params)
    second = collect(accel_engine, ACCEL_PROMPTS[:2], params)
    assert first == second
