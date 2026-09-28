# mini-vllm

An LLM inference server built from scratch: **paged KV cache**, **continuous batching**,
**prefix caching**, and an OpenAI-compatible HTTP API. ~1,600 lines of Python across ten
files.

PyTorch does the matmuls. Everything that makes a server a server — memory management,
scheduling, batching, streaming — is the project.

## The whole thing in 60 seconds

Ten files, one job each.

| File | Job |
|---|---|
| `config.py` | Parse a HuggingFace `config.json` |
| `request.py` | What a request is: prompt, sampling params, generated tokens, status |
| `model.py` | Qwen2 decoder — RMSNorm, RoPE, GQA, SwiGLU — plus weight loading |
| `cache.py` | **Paged KV cache**: block pool, allocator, prefix cache |
| `scheduler.py` | **Continuous batching**: who runs this iteration, who gets evicted |
| `engine.py` | The loop: schedule → batch → forward → sample |
| `sampler.py` | Batched sampling with per-request temperature / top-k / top-p |
| `detokenizer.py` | Tokens back into text, incrementally, without breaking UTF-8 |
| `server.py` | FastAPI, OpenAI-compatible, SSE streaming |

Three ideas carry it:

**Paged KV cache.** A sequence's KV does not live in one contiguous span. It lives in
16-token blocks scattered through a pool, and each request carries a block table mapping
logical positions to physical blocks. That is a page table — virtual memory, applied to
attention. No request has to reserve a worst-case contiguous span, so far more of them fit.

**Continuous batching.** The scheduling decision is remade every iteration, not once per
batch. A request that finishes at step 40 frees its slot at step 40; a request that arrives
at step 41 joins the very next forward pass. Almost all of the throughput comes from this.

**Prefix caching.** Requests sharing a system prompt shouldn't compute it twice. Each full
block of tokens gets a chained hash, which makes the prefix addressable, and the block
refcounts already in the allocator make it shareable.

The one design rule that holds it together: **attention never allocates cache memory.** It
receives the cache and metadata as arguments. That is why the cache implementation could be
swapped from contiguous to paged without touching a line of `model.py`.

## Results

Qwen2.5-0.5B-Instruct, fp16, Apple M5 (MPS). Reproduce with `bench/benchmark.py`.

**Continuous vs static batching** — 32 requests × 32 tokens:

| scheduling | tokens/s | wall time | forward passes |
|---|---:|---:|---:|
| continuous | 307.2 | 3.3s | 63 |
| static | 46.4 | 22.1s | 1024 |

**6.6× throughput**, and it scales with concurrency (2.4× at 12 concurrent requests). The
mechanism is visible in the last column: static batching spends 1024 forward passes to do
what continuous batching does in 63, because it decodes one sequence at a time.

**Latency under load** — 24 requests, Poisson arrivals at 6/s:

| metric | p50 | p99 |
|---|---:|---:|
| TTFT | 107 ms | 1816 ms |
| TPOT | 55.4 ms | 63.7 ms |

**Prefix caching** — 8 requests sharing a 317-token system prompt:

| prefix cache | cold TTFT (1st) | warm TTFT (rest) | prompt tokens reused |
|---|---:|---:|---:|
| on | 92 ms | 28 ms | 2,128 |
| off | 59 ms | 45 ms | 0 |

**1.63× faster TTFT** once warm. The first request is *slower* with caching on — it pays to
populate the cache and benefits from nothing. Averaging it in hides the entire effect, so
it's reported separately.

One number that is deliberately **not** claimed: paged attention is not faster per step than
a contiguous cache. Gathering blocks in pure PyTorch copies K/V every step, where a fused
kernel would read them in place. Paging buys **concurrency per gigabyte**, not raw speed,
and those are different claims.

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,serving]"
```

Serve it:

```bash
python -m mini_vllm.server --model Qwen/Qwen2.5-0.5B-Instruct --dtype float16
```

It speaks OpenAI, so any existing client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain paging in one sentence."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Or drive the engine directly. This is the real API — `step()` runs exactly one forward pass
and advances every live request:

```python
from mini_vllm import LLMEngine, SamplingParams

engine = LLMEngine("Qwen/Qwen2.5-0.5B-Instruct")
engine.add_request("a", engine.encode("Name a colour."), SamplingParams(max_tokens=32))
engine.add_request("b", engine.encode("Name a fruit."), SamplingParams(max_tokens=32))

while engine.has_unfinished_requests():
    for out in engine.step():
        print(out.request_id, out.new_token_ids, out.finished)
```

`GET /stats` reports live scheduler state: running, waiting, KV utilisation, preemptions,
and prompt tokens reused.

## Tests

```bash
pytest            # 49 fast tests: allocator, scheduler, sampler, detokenizer
pytest -m slow    # 22 more against real weights (~1GB on first run)
```

The ones that matter:

| Test | What it protects |
|---|---|
| `test_greedy_output_is_independent_of_batching` | Batching must not change any request's output. Without this, no benchmark below means anything. |
| `test_output_survives_preemption` | Evicting a request mid-generation and recomputing it lands on identical tokens. |
| `test_prefix_cache_does_not_change_output` | Reusing a cached prefix is indistinguishable from recomputing it. |
| `test_prefill_logits_match_hf` | Logits are bit-identical to HuggingFace (fp32). |
| `test_greedy_decode_matches_hf` | 32 greedy tokens match HF exactly. |
| `test_eviction_skips_blocks_a_request_still_holds` | Eviction can never pull memory out from under a live sequence. |
| `test_stop_string_straddling_a_token_boundary` | Stop strings that span two tokens are still caught. |

## Deliberately not here

Knowing what you chose not to build is part of the design:

- **Fused attention kernels (Triton / FlashAttention).** The gather is the obvious next
  optimisation. FlashAttention-2 also needs Ampere or newer, and the free GPU tiers are T4
  (sm_75), so a hand-written Triton kernel is the real path — a project of its own.
- **CUDA graphs.** Worth ~1.5–2× at small batch sizes by removing launch overhead, but it
  requires static input buffers throughout.
- **Chunked prefill.** A long prompt currently stalls decoding for one step. The attention
  mask already supports it (`start_pos > 0` works); the scheduler does not.
- **Batched prefill.** One prefill per step. A known limitation, not an oversight.
- **Swap-to-CPU.** Preemption recomputes instead. Re-running a prefill is usually cheaper
  than a round trip over PCIe, and it is far less code.
- **Speculative decoding, tensor parallelism, quantisation.** Each is a project.

## Design notes

[DESIGN.md](DESIGN.md) covers the internals and the traps found along the way — including
why `generate(do_sample=False)` is not greedy decoding, and why paged decode can never be
bit-identical to a contiguous cache.
