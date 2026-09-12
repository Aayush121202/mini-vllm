"""Token sampling.

Phase 0 keeps a per-request Python loop for the non-greedy path: correct, readable, and
slow in a way that becomes a measurable ablation once batches get large. Phase 1 replaces
it with a vectorised version (one sort, per-row threshold masks).

Hard rule, already enforced here: no ``.item()`` or ``.cpu()`` inside this module. Each
one is a device sync that serialises the pipeline.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_vllm.engine.sequence import SamplingParams


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Keep the first token that crosses the threshold, drop everything after it.
    remove = cumulative - probs > top_p
    remove_unsorted = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
    return logits.masked_fill(remove_unsorted, float("-inf"))


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        params: list[SamplingParams],
        generators: list[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        """logits: (num_seqs, vocab_size) -> token ids: (num_seqs,)."""
        if logits.shape[0] != len(params):
            raise ValueError(f"got {logits.shape[0]} rows of logits for {len(params)} requests")
        generators = generators or [None] * len(params)

        logits = logits.to(torch.float32)
        out = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)

        for i, p in enumerate(params):
            row = logits[i : i + 1]
            if p.is_greedy:
                out[i] = row.argmax(dim=-1).squeeze(0)
                continue
            row = row / p.temperature
            row = apply_top_k(row, p.top_k)
            row = apply_top_p(row, p.top_p)
            probs = torch.softmax(row, dim=-1)
            out[i] = torch.multinomial(probs, num_samples=1, generator=generators[i]).squeeze()
        return out
