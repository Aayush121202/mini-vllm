"""Benchmarks for mini-vllm.

Three modes, because the project makes three separate claims and they should not be
blurred into one number:

  offline  continuous vs static batching, optionally vs HF -> throughput (tokens/s)
  serving  Poisson arrivals at a fixed rate        -> TTFT / TPOT percentiles
  prefix   requests sharing a system prompt        -> prefill work avoided

Usage:
  python bench/benchmark.py offline --num-requests 32 --baseline hf
  python bench/benchmark.py serving --rate 4
  python bench/benchmark.py prefix
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from mini_vllm import LLMEngine, SamplingParams

WORDS = ["system", "model", "token", "cache", "block", "request", "latency", "batch", "kernel", "memory", "server", "engine", "attention", "throughput", "schedule", "prefill", "decode", "paging", "stream"]


def make_prompt(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(num_words))


def build_engine(args, **overrides) -> LLMEngine:
    kwargs = {
        "device": args.device,
        "dtype": args.dtype,
        "num_blocks": args.num_blocks,
        "block_size": args.block_size,
        "max_num_seqs": args.max_num_seqs,
    }
    kwargs.update(overrides)
    return LLMEngine(args.model, **kwargs)


def summarise(results: list[dict]) -> dict:
    """Median trial, plus the observed spread.

    A single timing on a laptop is not a measurement -- throughput here has been seen to
    vary close to 2x between back-to-back runs on the same machine, depending on thermal
    state and whatever else is resident. Report the median and show the range so a reader
    can see how much to trust the middle number.
    """
    ordered = sorted(results, key=lambda r: r["throughput"])
    median = ordered[len(ordered) // 2]
    rates = [r["throughput"] for r in ordered]
    return {**median, "low": rates[0], "high": rates[-1], "trials": len(rates)}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


# ----------------------------------------------------------------- hf baseline


def run_hf_baseline(args, prompts: list[str], lengths: list[int], trials: int) -> dict:
    """The same work through HuggingFace ``generate()``, batched as well as it can be.

    Held to the same concurrency limit as the engine (``--max-num-seqs``), since batch
    size is a memory budget and handing one side a larger one measures nothing. Prompts
    go in in chunks of that size, left-padded -- which is what a real HF serving script
    does. Both sides are greedy with repetition_penalty disabled and no early EOS, so
    the comparison is equal useful work.

    What HF cannot do is admit a request mid-batch or free a finished one. It therefore
    runs the whole batch to the longest requested length, so throughput is reported over
    *useful* tokens -- the ones actually asked for -- not the padded total it computed.
    That difference is the gap the number is measuring.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from mini_vllm.config import parse_dtype
    from mini_vllm.engine import get_device
    from mini_vllm.model import resolve_model_path

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = get_device(args.device)
    dtype = parse_dtype(args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()

    greedy = GenerationConfig(
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    size = args.max_num_seqs
    chunks = [
        (prompts[i : i + size], lengths[i : i + size]) for i in range(0, len(prompts), size)
    ]

    def once(scale: float = 1.0) -> dict:
        computed, steps = 0, 0
        start = time.perf_counter()
        for chunk_prompts, chunk_lengths in chunks:
            # Each chunk runs to its own longest member -- a finished sequence cannot
            # free its slot, so the batch keeps paying for it.
            new_tokens = max(1, int(max(chunk_lengths) * scale))
            config = GenerationConfig(
                **{**greedy.to_dict(), "max_new_tokens": new_tokens,
                   "min_new_tokens": new_tokens}
            )
            encoded = tokenizer(chunk_prompts, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                generated = model.generate(**encoded, generation_config=config)
            computed += (generated.shape[1] - encoded.input_ids.shape[1]) * len(chunk_prompts)
            steps += new_tokens
        elapsed = time.perf_counter() - start
        useful = sum(lengths)
        return {
            "elapsed_s": elapsed,
            "output_tokens": useful,
            "throughput": useful / elapsed,
            "steps": steps,
            "wasted": 1 - useful / computed,
        }

    once(scale=0.05)  # warm up
    # No `del model` here: `once` closes over it, and unbinding the name would leave
    # the closure referencing a deleted local. It goes out of scope on return anyway.
    return summarise([once() for _ in range(trials)])


# ------------------------------------------------------------------ offline mode


def make_lengths(rng: random.Random, count: int, max_tokens: int, uniform: bool) -> list[int]:
    """Per-request output lengths.

    Real traffic does not produce the same number of tokens for every request, and that
    variance is the whole reason continuous batching exists: a static batch runs until
    its *longest* member finishes, paying for every slot that finished early. With
    uniform lengths there is nothing to reclaim and static batching is optimal, so
    benchmarking only the uniform case would flatter the wrong design.
    """
    if uniform:
        return [max_tokens] * count
    return [rng.randint(max(1, max_tokens // 4), max_tokens) for _ in range(count)]


def run_offline(engine: LLMEngine, prompts: list[str], lengths: list[int]) -> dict:
    for i, (prompt, length) in enumerate(zip(prompts, lengths)):
        params = SamplingParams(temperature=0.0, max_tokens=length, stop_token_ids=[])
        engine.add_request(f"r{i}", engine.encode(prompt, chat=False), params)

    num_tokens, steps = 0, 0
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        num_tokens += sum(len(o.new_token_ids) for o in engine.step())
        steps += 1
    elapsed = time.perf_counter() - start

    return {
        "elapsed_s": elapsed,
        "output_tokens": num_tokens,
        "throughput": num_tokens / elapsed,
        "steps": steps,
        "tokens_per_step": num_tokens / steps if steps else 0.0,
    }


def offline(args) -> None:
    rng = random.Random(args.seed)
    prompts = [make_prompt(rng, args.prompt_words) for _ in range(args.num_requests)]
    lengths = make_lengths(rng, args.num_requests, args.max_tokens, args.uniform_lengths)

    rows = []
    for label, static in (("mini-vllm, continuous", False), ("mini-vllm, static", True)):
        engine = build_engine(args, static_batching=static, enable_prefix_cache=False)
        if not rows:
            print(repr(engine.runner))
        run_offline(engine, prompts[:2], [8, 8])  # warm up kernels and allocator
        results = [run_offline(engine, prompts, lengths) for _ in range(args.trials)]
        rows.append((label, summarise(results)))
        del engine

    if args.baseline == "hf":
        rows.append(
            ("HF generate(), batched", run_hf_baseline(args, prompts, lengths, args.trials))
        )

    shape = ("uniform" if args.uniform_lengths
             else f"{min(lengths)}-{max(lengths)} tokens, mean {sum(lengths) // len(lengths)}")
    print(f"\n{args.num_requests} requests, {sum(lengths)} useful output tokens "
          f"({shape}), median of {args.trials}\n")
    print("| engine | tokens/s | range | wall time | forward passes |")
    print("|---|---:|---:|---:|---:|")
    for label, r in rows:
        print(
            f"| {label} | {r['throughput']:.1f} | {r['low']:.0f}-{r['high']:.0f} | "
            f"{r['elapsed_s']:.1f}s | {r['steps']} |"
        )
    for label, other in rows[1:]:
        ratio = rows[0][1]["throughput"] / other["throughput"]
        print(f"\nvs {label}: {ratio:.2f}x")


# ------------------------------------------------------------------ serving mode


def run_serving(args, static: bool, engine: LLMEngine | None = None) -> dict:
    rng = random.Random(args.seed)
    if engine is None:
        engine = build_engine(args, enable_prefix_cache=False, static_batching=static)
    engine.scheduler.static_batching = static
    lengths = make_lengths(rng, args.num_requests, args.max_tokens, args.uniform_lengths)

    arrivals, clock = [], 0.0
    for i in range(args.num_requests):
        clock += rng.expovariate(args.rate)
        arrivals.append((clock, f"r{i}", make_prompt(rng, args.prompt_words), lengths[i]))

    submitted: dict[str, float] = {}
    first_token: dict[str, float] = {}
    last_token: dict[str, float] = {}
    counts: dict[str, int] = {}

    # Warm up on its own prompts. Without this the first trial pays kernel compilation
    # inside the measurement, and TTFT p50 has been seen to move more than 10x between
    # a cold and a warm run -- enough to invent a result that does not exist.
    warm_rng = random.Random(0)
    warm = SamplingParams(temperature=0.0, max_tokens=8, stop_token_ids=[])
    for i in range(4):
        prompt = make_prompt(warm_rng, args.prompt_words)
        engine.add_request(f"warm{i}", engine.encode(prompt, chat=False), warm)
    while engine.has_unfinished_requests():
        engine.step()

    pending = list(arrivals)
    start = time.perf_counter()
    while pending or engine.has_unfinished_requests():
        now = time.perf_counter() - start
        while pending and pending[0][0] <= now:
            _, request_id, prompt, length = pending.pop(0)
            params = SamplingParams(temperature=0.0, max_tokens=length, stop_token_ids=[])
            engine.add_request(request_id, engine.encode(prompt, chat=False), params)
            submitted[request_id] = now
        if not engine.has_unfinished_requests():
            time.sleep(0.001)
            continue
        for output in engine.step():
            now = time.perf_counter() - start
            first_token.setdefault(output.request_id, now)
            last_token[output.request_id] = now
            counts[output.request_id] = counts.get(output.request_id, 0) + len(
                output.new_token_ids
            )
    elapsed = time.perf_counter() - start

    return {
        "ttft": [first_token[r] - submitted[r] for r in submitted],
        "tpot": [
            (last_token[r] - first_token[r]) / max(1, counts[r] - 1) for r in submitted
        ],
        "throughput": sum(counts.values()) / elapsed,
        "elapsed": elapsed,
        "preemptions": engine.scheduler.num_preemptions,
    }


def serving(args) -> None:
    """Poisson arrivals against the engine, measuring per-request latency.

    Driven in-process rather than over HTTP so the numbers describe the engine rather
    than the web stack. This is where continuous batching earns its keep: a request that
    arrives mid-flight joins the next forward pass instead of waiting for the running
    batch to drain, which shows up in TTFT long before it shows up in throughput.
    """
    engine = build_engine(args, enable_prefix_cache=False)
    print(repr(engine.runner))

    rows = []
    for label, static in (("continuous", False), ("static", True)):
        trials = [run_serving(args, static=static, engine=engine) for _ in range(args.trials)]
        # Median trial by TTFT p50 -- the metric being compared picks the representative
        # run, rather than cherry-picking a different trial per column.
        trials.sort(key=lambda t: percentile(t["ttft"], 50))
        rows.append((label, trials[len(trials) // 2], trials))

    print(f"\n{args.num_requests} requests, Poisson arrivals at {args.rate}/s, "
          f"median of {args.trials}\n")
    print("| scheduling | TTFT p50 | TTFT p99 | TPOT p50 | tokens/s |")
    print("|---|---:|---:|---:|---:|")
    for label, r, _ in rows:
        print(
            f"| {label} | {percentile(r['ttft'], 50) * 1000:.0f} ms | "
            f"{percentile(r['ttft'], 99) * 1000:.0f} ms | "
            f"{percentile(r['tpot'], 50) * 1000:.1f} ms | {r['throughput']:.1f} |"
        )
    spread = [
        f"{label} p50 {min(percentile(t['ttft'], 50) for t in ts) * 1000:.0f}"
        f"-{max(percentile(t['ttft'], 50) for t in ts) * 1000:.0f} ms"
        for label, _, ts in rows
    ]
    ratio = percentile(rows[1][1]["ttft"], 50) / percentile(rows[0][1]["ttft"], 50)
    print(f"\nmedian TTFT improvement = {ratio:.2f}x")
    print("across trials: " + ", ".join(spread))


# ------------------------------------------------------------------- prefix mode


def prefix(args) -> None:
    """Every request carries the same long system prompt -- the multi-tenant case."""
    rng = random.Random(args.seed)
    shared = make_prompt(rng, args.prefix_words)
    prompts = [f"{shared} {make_prompt(rng, 8)}" for _ in range(args.num_requests)]
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[])

    rows = []
    for label, enabled in (("prefix cache on", True), ("prefix cache off", False)):
        engine = build_engine(args, enable_prefix_cache=enabled)
        if not rows:
            print(repr(engine.runner))
            print(f"shared prefix: {len(engine.encode(shared, chat=False))} tokens")

        ttfts, submitted, seen = [], {}, set()
        start = time.perf_counter()
        for i, prompt in enumerate(prompts):
            request_id = f"r{i}"
            engine.add_request(request_id, engine.encode(prompt, chat=False), params)
            submitted[request_id] = time.perf_counter() - start
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.request_id not in seen:
                        seen.add(output.request_id)
                        ttfts.append(time.perf_counter() - start - submitted[output.request_id])
        elapsed = time.perf_counter() - start
        rows.append((label, ttfts, elapsed, engine.scheduler.num_cached_tokens))
        del engine

    # The first request always pays full price -- it is the one that fills the cache.
    # Averaging it together with the rest hides the entire effect, so report it apart.
    print(f"\n{args.num_requests} requests sharing one prefix\n")
    print("| prefix cache | cold TTFT (1st) | warm TTFT (rest) | wall time | tokens reused |")
    print("|---|---:|---:|---:|---:|")
    for label, ttfts, elapsed, cached in rows:
        warm = statistics.mean(ttfts[1:]) if len(ttfts) > 1 else float("nan")
        print(
            f"| {label} | {ttfts[0] * 1000:.0f} ms | {warm * 1000:.0f} ms | "
            f"{elapsed:.1f}s | {cached} |"
        )
    warm_on = statistics.mean(rows[0][1][1:])
    warm_off = statistics.mean(rows[1][1][1:])
    print(f"\nwarm TTFT improvement = {warm_off / warm_on:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["offline", "serving", "prefix"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-words", type=int, default=24)
    parser.add_argument("--prefix-words", type=int, default=200)
    parser.add_argument("--rate", type=float, default=4.0, help="arrivals per second")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3, help="timed runs per config")
    parser.add_argument(
        "--uniform-lengths", action="store_true",
        help="give every request the same output length (the case static batching wins)",
    )
    parser.add_argument(
        "--baseline", choices=["none", "hf"], default="none",
        help="offline mode: also run HuggingFace generate() for an external comparison",
    )
    args = parser.parse_args()

    {"offline": offline, "serving": serving, "prefix": prefix}[args.mode](args)


if __name__ == "__main__":
    main()
