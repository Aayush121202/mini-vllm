"""Phase 0 gate: this engine must agree with HuggingFace, or nothing else matters.

Run with: pytest -m slow  (downloads ~1GB of weights on first run)

Everything is fp32 on CPU. Parity thresholds are dtype-dependent, and fp16 on MPS drifts
enough over 24 layers to make a tight bound meaningless -- performance dtypes get
validated by the perplexity benchmark instead, not by logit parity.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.config import ModelConfig
from mini_vllm.engine import LLMEngine
from mini_vllm.request import SamplingParams
from mini_vllm.engine import ModelRunner

pytestmark = pytest.mark.slow

NUM_GREEDY_TOKENS = 32

# Same forward pass, same weights: only kernel-level rounding separates us from HF.
LOGIT_TOLERANCE = 1e-4
# Paged decode attends over a padded block table and masks the surplus columns. Those
# columns contribute exactly zero weight mathematically, but they widen sdpa's softmax
# reduction and so change its rounding. Measured on Qwen2.5-0.5B in fp32: 1.81e-5 at 32
# padded columns, 2.29e-5 at 128 -- it tracks the padding, which is the signature of
# reduction order rather than a bug. The behavioural oracle is the exact greedy match
# below; this bound just has to be loose enough not to flag float32.
PAGED_DECODE_TOLERANCE = 1e-4
# Prefill and incremental decode take *different* BLAS paths (GEMM over the whole
# sequence vs GEMV per token), so their accumulation orders differ. HF exhibits exactly
# the same 3.03e-4 delta between its own two paths, so this bound measures float32, not
# correctness. The real oracle is test_decode_logits_match_hf_incremental below, which
# compares like with like and holds to within float32 exactness.
CACHE_PATH_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def hf_model(model_path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="module")
def runner(model_path):
    return ModelRunner(
        ModelConfig.from_json(model_path), device="cpu", dtype="float32", num_blocks=64
    )


def prefill(runner, token_ids: list[int]) -> torch.Tensor:
    """One prefill through the real paged path: allocate blocks, then run."""
    runner.reset()
    table = runner.cache.allocator.allocate(_blocks_for(runner, len(token_ids)))
    return runner.execute([token_ids], [0], [table], last_token_only=False)[0]


def incremental(runner, token_ids: list[int]) -> torch.Tensor:
    """Feed one token at a time, as a decode loop would."""
    runner.reset()
    table = runner.cache.allocator.allocate(_blocks_for(runner, len(token_ids)))
    rows = [runner.execute([[t]], [i], [table])[0, -1] for i, t in enumerate(token_ids)]
    return torch.stack(rows)


def _blocks_for(runner, num_tokens: int) -> int:
    size = runner.cache.block_size
    return (num_tokens + size - 1) // size


def hf_incremental_logits(hf_model, token_ids: list[int]) -> torch.Tensor:
    """Feed HF one token at a time through its own KV cache."""
    from transformers import DynamicCache

    cache = DynamicCache()
    rows = []
    with torch.inference_mode():
        for token in token_ids:
            out = hf_model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            rows.append(out.logits[0, -1])
            cache = out.past_key_values
    return torch.stack(rows)


def test_prefill_logits_match_hf(runner, hf_model, tokenizer, prompt):
    """Full-sequence logit parity on a single prefill."""
    token_ids = tokenizer(prompt).input_ids
    ours = prefill(runner, token_ids)

    with torch.inference_mode():
        theirs = hf_model(torch.tensor([token_ids])).logits.squeeze(0)

    assert ours.shape == theirs.shape
    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < LOGIT_TOLERANCE, f"max |Δlogit| = {max_diff:.3e}"


def test_decode_logits_match_hf_incremental(runner, hf_model, tokenizer, prompt):
    """The strongest oracle in the suite: our KV-cached decode against HF's KV-cached
    decode, token for token. Same shapes, same kernels, so any difference here is a real
    bug rather than rounding."""
    token_ids = tokenizer(prompt).input_ids
    ours = incremental(runner, token_ids)
    theirs = hf_incremental_logits(hf_model, token_ids)

    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < PAGED_DECODE_TOLERANCE, f"max |Δlogit| vs HF incremental = {max_diff:.3e}"


def test_decode_logits_match_prefill(runner, tokenizer, prompt):
    """The KV cache must be transparent: decoding token-by-token has to reproduce the
    logits from a single full prefill. This is the test that catches cache-offset and
    RoPE-position bugs, and it needs no reference implementation.

    Bounded by float32 accumulation order, not by correctness -- see CACHE_PATH_TOLERANCE.
    """
    token_ids = tokenizer(prompt).input_ids
    prefill_logits = prefill(runner, token_ids)
    incremental_logits = incremental(runner, token_ids)

    max_diff = (prefill_logits - incremental_logits).abs().max().item()
    assert max_diff < CACHE_PATH_TOLERANCE, f"prefill vs decode max |Δlogit| = {max_diff:.3e}"


def test_greedy_decode_matches_hf(model_path, hf_model, tokenizer, prompt):
    """Exact token-ID match. Logit parity can hide a bug that only bites after several
    steps; this catches drift.

    The explicit GenerationConfig is load-bearing. Qwen2.5's checkpoint ships
    ``repetition_penalty: 1.1`` in generation_config.json, and HF applies logits
    processors even when do_sample=False -- so a bare ``generate(do_sample=False)`` is
    not greedy decoding and diverges from an honest argmax within a few tokens.
    """
    from transformers import GenerationConfig

    token_ids = tokenizer(prompt).input_ids
    greedy = GenerationConfig(
        do_sample=False,
        repetition_penalty=1.0,
        max_new_tokens=NUM_GREEDY_TOKENS,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.inference_mode():
        reference = hf_model.generate(
            torch.tensor([token_ids]), generation_config=greedy
        )[0].tolist()[len(token_ids) :]

    engine = LLMEngine(model_path, device="cpu", dtype="float32", num_blocks=64)
    engine.add_request(
        "parity",
        token_ids,
        SamplingParams(temperature=0.0, max_tokens=NUM_GREEDY_TOKENS, stop_token_ids=[]),
    )
    ours: list[int] = []
    while engine.has_unfinished_requests():
        for out in engine.step():
            ours.extend(out.new_token_ids)

    assert len(reference) == NUM_GREEDY_TOKENS
    assert ours == reference
