"""Throughput against the concurrency limit, continuous vs static batching.

Emits a light and a dark PNG. Dark is a selected variant -- its own steps from the same
ramps against the dark surface -- not an inverted copy of the light one.

Usage: python bench/plot.py --dtype float16
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.benchmark import (
    build_engine,
    make_lengths,
    make_prompt,
    run_offline,
    summarise,
)

OUT_DIR = Path(__file__).parent / "results"
CONCURRENCY = [1, 2, 4, 8, 16, 32, 64]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926"),
    },
}
LABELS = ("Continuous", "Static")


def measure(args) -> dict[str, list[float]]:
    """One engine, two knobs. Rebuilding per point would spend most of the run loading
    weights, and the knobs are the only thing that differs between points."""
    rng = random.Random(args.seed)
    prompts = [make_prompt(rng, args.prompt_words) for _ in range(args.num_requests)]
    lengths = make_lengths(rng, args.num_requests, args.max_tokens, uniform=False)

    engine = build_engine(args, enable_prefix_cache=False)
    print(repr(engine.runner))
    run_offline(engine, prompts[:2], [8, 8])  # warm up

    results: dict[str, list[float]] = {"Continuous": [], "Static": []}
    for limit in CONCURRENCY:
        engine.scheduler.max_num_seqs = limit
        for label, static in (("Continuous", False), ("Static", True)):
            engine.scheduler.static_batching = static
            trials = [run_offline(engine, prompts, lengths) for _ in range(args.trials)]
            rate = summarise(trials)["throughput"]
            results[label].append(rate)
            print(f"  max_num_seqs={limit:3d}  {label:<11} {rate:7.1f} tok/s")
    return results


def draw(results: dict[str, list[float]], mode: str, subtitle: str) -> Path:
    theme = THEMES[mode]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    series = list(results.items())
    # Label where the lines are furthest apart, not at the end. These two curves meet at
    # both extremes -- one slot is no batching either way, and once every request fits in
    # a single wave static batching is continuous batching -- so end-anchored labels land
    # on top of each other and render as mush.
    gaps = [abs(series[0][1][i] - series[1][1][i]) for i in range(len(CONCURRENCY))]
    at = gaps.index(max(gaps))
    upper = 0 if series[0][1][at] >= series[1][1][at] else 1

    for index, ((label, values), colour) in enumerate(zip(series, theme["series"])):
        ax.plot(
            CONCURRENCY, values, color=colour, linewidth=2, marker="o", markersize=5,
            markeredgecolor=theme["surface"], markeredgewidth=1.5, label=label, zorder=3,
        )
        # Direct label in text ink -- the coloured mark beside it carries identity.
        ax.annotate(
            label, (CONCURRENCY[at], values[at]), textcoords="offset points",
            xytext=(-4, 11 if index == upper else -19), ha="right",
            fontsize=9, color=theme["secondary"],
        )

    ax.set_title(
        "Throughput vs concurrency limit", loc="left", fontsize=12,
        color=theme["ink"], fontweight="bold", pad=14,
    )
    ax.text(
        0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=theme["muted"],
    )
    ax.set_xlabel("max concurrent requests", fontsize=9, color=theme["secondary"])
    ax.set_ylabel("tokens / second", fontsize=9, color=theme["secondary"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY)
    ax.set_xticklabels([str(c) for c in CONCURRENCY])
    ax.set_xlim(CONCURRENCY[0] * 0.85, CONCURRENCY[-1] * 1.12)
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", color=theme["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"throughput-{mode}.png"
    fig.savefig(path, facecolor=theme["surface"], bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--prompt-words", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = measure(args)
    subtitle = f"{args.num_requests} requests, variable output lengths, median of {args.trials}"
    for mode in ("light", "dark"):
        print("wrote", draw(results, mode, subtitle))


if __name__ == "__main__":
    main()
