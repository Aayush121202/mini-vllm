"""Benchmarks for mini-vllm.

Three modes, because the project makes three separate claims and they should not be
blurred into one number:

  offline  continuous batching vs static batching  -> throughput (tokens/s)
  serving  Poisson arrivals at a fixed rate        -> TTFT / TPOT percentiles
  prefix   requests sharing a system prompt        -> prefill work avoided

Usage:
  python bench/benchmark.py offline --num-requests 32
  python bench/benchmark.py serving --rate 4
  python bench/benchmark.py prefix
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from mini_vllm import LLMEngine, SamplingParams

WORDS = (
    "system model token cache block request latency batch kernel memory server engine "
    "attention throughput schedule prefill decode paging stream"
).split()


def make_prompt(rng: random.Random, num_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(num_words))


def build_engine(args, **overrides) -> LLMEngine:
    kwargs = dict(
        device=args.device,
        dtype=args.dtype,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
    )
    kwargs.update(overrides)
    return LLMEngine(args.model, **kwargs)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


# ------------------------------------------------------------------ offline mode


def run_offline(engine: LLMEngine, prompts: list[str], max_tokens: int) -> dict:
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=[])
    for i, prompt in enumerate(prompts):
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

    rows = []
    for label, static in (("continuous batching", False), ("static batching", True)):
        engine = build_engine(args, static_batching=static, enable_prefix_cache=False)
        if not rows:
            print(repr(engine.runner))
        result = run_offline(engine, prompts, args.max_tokens)
        rows.append((label, result))
        del engine

    print(f"\n{args.num_requests} requests x {args.max_tokens} tokens\n")
    print("| scheduling | tokens/s | wall time | steps | tokens/step |")
    print("|---|---:|---:|---:|---:|")
    for label, r in rows:
        print(
            f"| {label} | {r['throughput']:.1f} | {r['elapsed_s']:.1f}s | "
            f"{r['steps']} | {r['tokens_per_step']:.1f} |"
        )
    speedup = rows[0][1]["throughput"] / rows[1][1]["throughput"]
    print(f"\ncontinuous / static = {speedup:.2f}x")


# ------------------------------------------------------------------ serving mode


def serving(args) -> None:
    """Poisson arrivals against the engine, measuring per-request latency.

    Driven in-process rather than over HTTP so the numbers describe the engine rather
    than the web stack.
    """
    rng = random.Random(args.seed)
    engine = build_engine(args, enable_prefix_cache=False)
    print(repr(engine.runner))

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=[])
    arrivals, clock = [], 0.0
    for i in range(args.num_requests):
        clock += rng.expovariate(args.rate)
        arrivals.append((clock, f"r{i}", make_prompt(rng, args.prompt_words)))

    submitted: dict[str, float] = {}
    first_token: dict[str, float] = {}
    last_token: dict[str, float] = {}
    counts: dict[str, int] = {}

    pending = list(arrivals)
    start = time.perf_counter()
    while pending or engine.has_unfinished_requests():
        now = time.perf_counter() - start
        while pending and pending[0][0] <= now:
            _, request_id, prompt = pending.pop(0)
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

    ttft = [first_token[r] - submitted[r] for r in submitted]
    tpot = [
        (last_token[r] - first_token[r]) / max(1, counts[r] - 1) for r in submitted
    ]
    total_tokens = sum(counts.values())

    print(f"\n{args.num_requests} requests, Poisson arrivals at {args.rate}/s\n")
    print("| metric | p50 | p99 |")
    print("|---|---:|---:|")
    print(f"| TTFT | {percentile(ttft, 50) * 1000:.0f} ms | {percentile(ttft, 99) * 1000:.0f} ms |")
    print(f"| TPOT | {percentile(tpot, 50) * 1000:.1f} ms | {percentile(tpot, 99) * 1000:.1f} ms |")
    print(f"\nthroughput {total_tokens / elapsed:.1f} tok/s over {elapsed:.1f}s")
    print(f"preemptions {engine.scheduler.num_preemptions}")


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
    args = parser.parse_args()

    {"offline": offline, "serving": serving, "prefix": prefix}[args.mode](args)


if __name__ == "__main__":
    main()
