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


def test_runs_on_accelerator(model_path):
    """Catches device-placement regressions -- notably non-parameter buffers (rope
    tables) being left on CPU while the weights move to the accelerator. The CPU-only
    parity suite cannot see that class of bug."""
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        pytest.skip("no accelerator available")

    engine = LLMEngine(model_path, device=device)
    engine.add_request("dev", [40, 1079, 264], SamplingParams(max_tokens=4, stop_token_ids=[]))
    tokens: list[int] = []
    while engine.has_unfinished_requests():
        for out in engine.step():
            tokens.extend(out.new_token_ids)
    assert len(tokens) == 4
