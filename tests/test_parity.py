"""Phase 0 gate: this engine must agree with HuggingFace, or nothing else matters.

Run with: pytest -m slow  (downloads ~1GB of weights on first run)

Everything is fp32 on CPU. Parity thresholds are dtype-dependent and fp16 on MPS drifts
enough over 24 layers to make a tight bound meaningless -- performance dtypes get
validated by the perplexity benchmark instead, not by logit parity.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.config import ModelConfig
from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.engine.sequence import SamplingParams
from mini_vllm.runner.model_runner import ModelRunner

pytestmark = pytest.mark.slow

NUM_GREEDY_TOKENS = 32
LOGIT_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def hf_model(model_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def test_prefill_logits_match_hf(model_path, hf_model, tokenizer, prompt):
    """Full-sequence logit parity on a single prefill."""
    token_ids = tokenizer(prompt).input_ids

    runner = ModelRunner(
        ModelConfig.from_json(model_path), device="cpu", dtype="float32"
    )
    ours = runner.execute(token_ids, start_pos=0, last_token_only=False)

    with torch.inference_mode():
        theirs = hf_model(torch.tensor([token_ids])).logits.squeeze(0)

    assert ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < LOGIT_TOLERANCE, f"max |Δlogit| = {max_diff:.3e}"


def test_decode_logits_match_prefill(model_path, tokenizer, prompt):
    """The KV cache must be transparent: decoding token-by-token has to reproduce the
    logits from a single full prefill. This is the test that catches cache-offset and
    RoPE-position bugs, and it needs no reference implementation."""
    token_ids = tokenizer(prompt).input_ids
    runner = ModelRunner(
        ModelConfig.from_json(model_path), device="cpu", dtype="float32"
    )

    prefill_logits = runner.execute(token_ids, start_pos=0, last_token_only=False)

    runner.reset()
    incremental = []
    for i, tok in enumerate(token_ids):
        incremental.append(runner.execute([tok], start_pos=i)[-1])
    incremental_logits = torch.stack(incremental)

    max_diff = (prefill_logits - incremental_logits).abs().max().item()
    assert max_diff < LOGIT_TOLERANCE, f"prefill vs decode max |Δlogit| = {max_diff:.3e}"


def test_greedy_decode_matches_hf(model_path, hf_model, tokenizer, prompt):
    """Exact token-ID match. Logit parity can hide a bug that only bites after several
    steps; this catches drift."""
    token_ids = tokenizer(prompt).input_ids

    with torch.inference_mode():
        reference = hf_model.generate(
            torch.tensor([token_ids]),
            max_new_tokens=NUM_GREEDY_TOKENS,
            do_sample=False,
        )[0].tolist()[len(token_ids) :]

    engine = LLMEngine(model_path, device="cpu", dtype="float32")
    engine.add_request(
        "parity",
        token_ids,
        SamplingParams(temperature=0.0, max_tokens=NUM_GREEDY_TOKENS, stop_token_ids=[]),
    )
    ours: list[int] = []
    while engine.has_unfinished_requests():
        for out in engine.step():
            ours.extend(out.new_token_ids)

    assert ours[: len(reference)] == reference, (
        f"diverged at position {next(i for i, (a, b) in enumerate(zip(ours, reference)) if a != b)}"
    )
