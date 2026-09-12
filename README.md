# mini-vllm

A transformer **inference engine** built from scratch: paged KV cache, continuous
batching, iteration-level scheduling, and an OpenAI-compatible serving layer.

PyTorch does the matmuls. Everything around them — memory management, scheduling,
batching, kernels — is the project. See [DESIGN.md](DESIGN.md) for the architecture and
phase plan.

> **Status: Phase 0.** Single sequence, contiguous KV cache, greedy decode, validated
> against HuggingFace. Not yet fast, and not trying to be. Phases 2–3 are where the
> throughput work lives.

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

```python
from mini_vllm import LLMEngine, SamplingParams

engine = LLMEngine("Qwen/Qwen2.5-0.5B-Instruct")
print(engine.generate("Explain KV caching in one sentence.", SamplingParams(max_tokens=64)))
```

Or drive the engine loop yourself — this is the real API, and the one that becomes
continuous batching in Phase 3:

```python
engine.add_request("req-1", prompt_token_ids, SamplingParams(temperature=0.7, seed=0))
while engine.has_unfinished_requests():
    for out in engine.step():
        print(out.new_token_ids, out.finished)
```

## Tests

```bash
pytest                 # fast: sampler, cache, sequence bookkeeping
pytest -m slow         # parity vs HuggingFace (downloads ~1GB on first run)
```

The parity suite is the Phase 0 gate:

| Test | What it catches |
|---|---|
| `test_prefill_logits_match_hf` | wrong RoPE convention, RMSNorm precision, GQA head mapping |
| `test_decode_logits_match_prefill` | KV cache offset and position bugs, with no reference needed |
| `test_greedy_decode_matches_hf` | drift that only appears after several steps |

## Layout

```
mini_vllm/
  engine/    llm_engine.py  sequence.py          # the contract: add_request / step / abort
  cache/     interface.py   contiguous.py        # phase 2: block_manager.py
  layers/    attention.py   rotary.py  norm.py  sampler.py
  models/    qwen2.py       loader.py  registry.py
  runner/    model_runner.py  backend.py
  serving/                                       # phase 1: api_server.py, detokenizer.py
bench/                                           # phase 2+: throughput, TTFT/TPOT curves
```
