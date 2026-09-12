# mini-vllm — Design

A from-scratch transformer **inference engine**. PyTorch does the matmuls; this project is
everything around them: KV cache memory management, iteration-level scheduling, continuous
batching, and an OpenAI-compatible serving layer.

Explicitly **not** a training framework, and not an attempt to beat vLLM. It is a
reimplementation of vLLM's core architectural ideas with measured ablations for each one.

---

## 1. The engine contract

Everything conforms to this. It is fixed before anything else is written.

```python
engine.add_request(request_id, prompt_token_ids, sampling_params) -> None
engine.step() -> list[RequestOutput]   # ONE forward pass; advances every live request
engine.abort(request_id) -> None
engine.has_unfinished_requests() -> bool
```

`step()` advancing *all* in-flight requests by exactly one iteration — rather than a
`generate()` that owns the loop — is the decision that makes continuous batching possible
later. Nothing above the engine is allowed to own a decode loop.

Corollary rule, load-bearing for Phase 2: **attention never owns its KV cache.**

```python
def forward(self, hidden, positions, kv_cache: KVCache, md: AttnMetadata) -> Tensor
```

If an attention module allocates or holds its own cache tensor, paging is impossible and the
model layer gets rewritten. `md` (sequence lengths, offsets, later block tables) is built once
per step by the runner and threaded down to every layer.

---

## 2. Components

### 2.1 Model layer — `mini_vllm/models/`, `mini_vllm/layers/`

Llama-style decoder: RMSNorm, RoPE, grouped-query attention, SwiGLU MLP. Module names mirror
the HuggingFace checkpoint exactly, so weight loading is an identity mapping plus tied-embedding
handling. (Phase 2 fuses q/k/v into one projection, at which point the loader grows a real
name-mapping table.)

Architectures live in a registry keyed on `config.json["architectures"][0]`. Phase 0 ships
Qwen2 only — Llama 3.x is *nearly* identical but has rope scaling that would break parity
silently, so it gets added deliberately with its own parity test, not by assumption.

Tokenization uses HF `tokenizers`. Writing a BPE implementation is a different project.

### 2.2 KV cache — `mini_vllm/cache/`

**Phase 0 (`contiguous.py`):** one `[batch, max_len, n_kv, head_dim]` tensor per layer.
Deliberately naive. Its purpose is to be the correctness oracle and to *measure* how much memory
is wasted on padding — that measurement is the motivation for the paged version.

**Phase 2 (`block_manager.py`):**

- Physical pool: `[num_layers, 2, num_blocks, block_size, n_kv, head_dim]`, allocated once,
  `block_size = 16`.
- Per-sequence **block table**: list of physical block IDs. This is a page table.
- **Free-list allocator** with refcounts: `allocate`, `append_slot`, `fork` (copy-on-write),
  `free`.
- **Sizing**: profile one forward pass at startup, measure peak activation memory, then
  `num_blocks = (total_mem * util - weights - activations) / bytes_per_block`.

Target metric: internal fragmentation falls from "padded to `max_len`" to "≤15 tokens per
sequence." Quantify it in the README.

### 2.3 Attention backends — `mini_vllm/kernels/`

Two paths, for one reason: **prefill is compute-bound, decode is memory-bandwidth-bound.**

| | Prefill | Decode |
|---|---|---|
| Query tokens | many | exactly 1 per sequence |
| Keys | contiguous, new | scattered across physical blocks |
| Bound by | FLOPs | HBM bandwidth |
| Implementation | varlen packing + `cu_seqlens` | Triton kernel walking the block table |

The decode kernel is the one piece `torch.sdpa` cannot express, because the gather through the
block table has no PyTorch equivalent. ~100 lines of Triton: one program per (sequence, head),
walk the block table, online-softmax accumulate.

**Free-tier constraint:** FlashAttention-2 requires Ampere (sm_80+). Kaggle/Colab give T4
(sm_75) and P100. So FA2 is off the table there — which is why the decode kernel is hand-written
Triton (works fine on T4) rather than a library call.

A naive PyTorch reference implementation stays behind the same interface forever: it is the
`assert_close` oracle for the Triton kernel, and the fallback that keeps the project runnable on
Mac/MPS and CPU.

### 2.4 Scheduler — `mini_vllm/engine/scheduler.py`

Three queues (`waiting`, `running`, `swapped`); policy runs once per `step()`:

1. Can every running sequence get its next block? If the allocator is exhausted, **preempt** —
   evict newest-first, recompute on resume (simpler than swap-to-CPU; that comes later, if ever).
2. Admit from `waiting` while under budget: `max_num_seqs` AND `max_num_batched_tokens`.
3. **Chunked prefill**: a 4k-token prompt would stall every decoding request for one long step.
   Split it across iterations and mix prefill chunks with decode tokens in one batch.

The interesting tradeoff, and the thing to plot: prioritizing prefill improves **TTFT** for new
arrivals but adds inter-token jitter (**TPOT**) for everyone already streaming.
`max_num_batched_tokens` is the knob that moves along that curve.

### 2.5 Model runner — `mini_vllm/runner/`

Scheduler output → tensors → outputs.

- Flatten the batch: concatenated token IDs, position IDs, `cu_seqlens`, block tables. No padding.
- **CUDA graph capture** for the decode path, one graph per batch-size bucket (1,2,4,8,16,…).
  At small batch sizes Python and kernel-launch overhead is a large fraction of step time;
  typically a 1.5–2× win. Requires static input buffers you copy into — a real design constraint.
- Device abstraction is one thin `Backend` protocol: `cuda` (Triton + graphs) vs `mps`/`cpu`
  (reference kernels, no graphs).

### 2.6 Sampler — `mini_vllm/layers/sampler.py`

On-device, whole batch at once, with **per-request parameters** — different temperature/top-p/
top-k per row of the same batch. The naive per-request Python loop works and then dominates step
time at batch 64; vectorizing it (sort once, build masks from per-row thresholds) is a clean,
measurable ablation.

Hard rule: no `.item()` or `.cpu()` inside the sampler. Each one is a device sync that
serializes the pipeline.

Determinism comes from per-request `torch.Generator`, never a global seed.

### 2.7 Detokenizer — `mini_vllm/serving/detokenizer.py`

Subtle, and where most hobby engines are visibly wrong:

- BPE tokens are not 1:1 with characters. Decoding each new token independently corrupts
  multi-byte UTF-8 and word-boundary prefixes. Keep a per-request incremental buffer that only
  emits complete, stable text.
- **Stop strings straddle token boundaries.** Buffer enough trailing text to detect the longest
  stop string, then truncate at the exact character offset.
- Stop conditions: EOS, stop strings, `max_tokens`, client disconnect.

### 2.8 Server — `mini_vllm/serving/api_server.py`

FastAPI, OpenAI-compatible `/v1/chat/completions` with SSE streaming, so any existing OpenAI
client can point at it. An async engine loop runs `step()` continuously in the background; HTTP
handlers push into the waiting queue and pull from a per-request `asyncio.Queue` and never touch
the model.

Prometheus metrics: TTFT, TPOT, throughput, queue depth, **KV cache utilization**, preemption
count.

### 2.9 Benchmarks — `bench/`

**Correctness first — without this, no performance number means anything.**

- Logit parity vs HF `transformers`, same prompt: max abs diff < 1e-4 (fp32)
- Greedy decode: exact token-ID match vs HF for 50+ tokens
- Paged vs contiguous cache: identical outputs
- Triton kernel vs PyTorch reference: `assert_close`
- Perplexity on a WikiText slice vs the published number

**Then performance.**

- Offline: total throughput (tok/s) on a fixed ShareGPT-style workload
- Online: Poisson arrivals at rising request rate → TTFT/TPOT p50 and p99 vs rate. The knee in
  that curve is the result.
- Baseline: HF `generate()` with naive batching.
- **Ablation table, one row each**: paged off · chunked prefill off · CUDA graphs off ·
  prefix caching off · vectorized sampler off.

Run every config 3× and report medians; free-tier GPUs are shared.

---

## 3. Phases

Each ends with something demoable and a test. Do not start the next until the current one is green.

| Phase | Deliverable | Hardware |
|---|---|---|
| **0** | Single sequence, contiguous cache, greedy decode, **logit parity vs HF passing** | Mac / CPU |
| **1** | Static batching, batched sampler, streaming detokenizer, FastAPI SSE — it is a server | Mac |
| **2** | Paged KV cache, block manager, Triton decode kernel; first real benchmark | Kaggle T4 |
| **3** | **Continuous batching**, preemption, chunked prefill; TTFT/TPOT curves | Kaggle T4 |
| **4** | Pick 1–2: prefix caching (radix tree) · CUDA graphs · INT8 KV cache · speculative decoding | Lightning A10G |

Phase 0 runs a 0.5B model on the Mac on purpose: a 10-second iteration loop while the hard parts
are unstable, and parity green *before* optimizing anything.

**Phase 3 is the resume project.** 0–2 are setup, 4 is garnish. Stopping after 3 still leaves a
complete, defensible system.

### Phase 0 checklist

- [x] Engine contract: `add_request` / `step` / `abort`
- [x] `SamplingParams`, `Sequence`, `RequestOutput`
- [x] Qwen2 model: RMSNorm, RoPE, GQA, SwiGLU
- [x] Contiguous KV cache behind the `KVCache` interface
- [x] Safetensors loader with tied-embedding handling
- [x] Sampler: greedy + temperature/top-k/top-p, seeded
- [x] Parity test vs HF (logits + greedy token match)
- [ ] `pytest -m slow` green on your machine

---

## 4. Non-goals and traps

- **No training.** No backward pass, no optimizers, ever. Inference-only is a feature.
- **No tensor parallelism.** One GPU means it cannot be tested; untested distributed code in a
  portfolio repo is a liability, not a credential.
- **No hand-written BPE or GEMM.** Different project; dilutes the story.
- **dtype policy is explicit, not incidental.** T4 has no hardware bf16 → fp16 there; bf16 on
  Ampere+; fp32 on CPU/MPS for parity tests. Parity thresholds differ per dtype and live in the
  test.
- **Use `apply_chat_template`.** Hand-rolled prompt formatting silently wrecks output quality.
- **Chunked prefill needs a real attention mask** (`start_pos > 0` with multiple query tokens).
  Phase 0's `is_causal` shortcut is only valid for a full prefill from position 0. Marked TODO in
  `layers/attention.py`.
