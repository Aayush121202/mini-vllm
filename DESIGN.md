# Design notes

Internals and the things that bit. See [README.md](README.md) for what the project is and
what it measures.

## The contract

```python
engine.add_request(request_id, prompt_token_ids, sampling_params)
engine.step() -> list[RequestOutput]   # ONE forward pass, advances every live request
engine.abort(request_id)
```

`step()` advancing all in-flight requests by exactly one iteration — rather than a
`generate()` that owns the loop — is what makes continuous batching possible. Nothing above
the engine owns a decode loop; the server's background task just calls `step()` forever.

## Attention is one code path

Write K/V into the pool first, then gather the whole context back:

```python
kv_cache.write(layer, k, v, md.slot_mapping)
k_all, v_all = kv_cache.gather(layer, md.block_tables)
```

Doing it in that order collapses prefill and decode into the same code. A prefill whose
prefix is already cached simply finds more context waiting for it, so the two differ only in
how many tokens they contribute.

The mask is a single inequality: **key `j` is visible to the query at absolute position `p`
iff `j <= p`**. Causality within a prefill, each row's context boundary in a mixed decode
batch, and the padding in short block tables all fall out of that one comparison. There is no
separate padding mask, and no row can ever be fully masked (position 0 is always visible), so
softmax can't produce NaN.

## Scheduling

Each step is either one prefill or a decode over everything running. Admission is bounded by
`max_num_seqs` and by block availability.

When the pool is exhausted, the **newest** running request is evicted and recomputed later.
Newest-first keeps older requests moving, so the requests that have already waited longest
don't wait longer. Recompute rather than swap-to-CPU: re-running a prefill is usually cheaper
than a PCIe round trip, and it is a great deal less code.

`num_computed_tokens` means *tokens whose KV is in the cache* — nothing else. In particular
`append_token` does not touch it: a sampled token has no KV until the next forward pass
writes it. Prefix cache hits advance it before any forward pass runs at all.

## Sampling

Per-row temperature, top-k and top-p, vectorised over the batch — one sort, per-row
thresholds. Seeded requests draw from their own `torch.Generator` one row at a time, so their
output cannot depend on which other requests happen to share the batch. That is what makes
`test_seeded_sampling_is_independent_of_batching` pass; a single `multinomial` over the batch
would fail it.

## Traps found along the way

- **`generate(do_sample=False)` is not greedy decoding.** Qwen2.5 ships
  `repetition_penalty: 1.1` in `generation_config.json`, and HF applies logits processors
  regardless of `do_sample`. A parity test against the bare default diverges within ~3 tokens
  and looks exactly like a KV cache bug. Pass an explicit `GenerationConfig`.

- **Prefill and incremental decode are never bit-identical**, even in fp32: GEMM over a whole
  sequence and GEMV per token accumulate in different orders. Measured 3.03e-4 max logit
  delta on Qwen2.5-0.5B — and HF shows *the same* delta between its own two paths. Compare
  like with like instead.

- **Masked padding changes the answer slightly.** Paged decode attends over a padded block
  table; the surplus columns contribute exactly zero weight but widen sdpa's softmax
  reduction. The delta tracks the padding — 1.81e-5 at 32 padded columns, 2.29e-5 at 128 —
  which is the signature of reduction order, not a bug. The behavioural oracle is the exact
  greedy token match.

- **Non-parameter buffers do not follow the weights.** Rope tables built with an explicit
  `device=` stay there while `nn.Parameter`s move, crashing on MPS in a way the CPU-only
  parity suite cannot see. Build them from ambient factory calls under `with torch.device()`,
  and keep an accelerator smoke test.

- **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` in transformers 5.x**, not
  a list of ids. Render to text, then tokenize.

- **A fully cached prompt has nothing to run.** If prefix matching consumes every token there
  are no new tokens to forward, so no logits to sample from. Always leave at least one token
  to recompute.

- **An ablation can be a strawman without anyone noticing.** The `static_batching` switch
  originally blocked admission whenever anything was running. Because a prefill step also
  samples a token, the wave sealed after the first request and the "baseline" degenerated
  into running one request at a time — `tokens_per_step = 1.0`. It produced a headline 6.6×
  that meant nothing. The tell was in the output the whole time: a batching baseline whose
  batch size is 1. Real static batching fills a wave to the concurrency limit, seals it on
  first decode, and drains before admitting more.

- **An unwarmed benchmark invents results.** Throughput varied close to 2× between
  back-to-back runs, and TTFT p50 moved more than 10× between a cold and a warm run — enough
  to report a 23.7× improvement that was really 2×, and to make continuous batching look
  *worse* at p99 than static when warmed runs show it 22× better. Warm up, run several
  trials, take the median, and print the spread so a reader can see how much to trust it.

- **Averaging hides prefix caching.** The first request populates the cache and benefits from
  nothing, so it is *slower* with caching on. Mixed into a mean over 8 requests, a real 1.63×
  win reads as 1.00×. Report cold and warm separately.

## dtype policy

fp32 on CPU and MPS by default, bf16 on Ampere+, fp16 on Turing (T4 has no hardware bf16).
Parity thresholds are dtype-dependent and live in the tests; fp16 drifts enough over 24
layers that a tight logit bound is meaningless, so performance dtypes are validated
behaviourally instead.
