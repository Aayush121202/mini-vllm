"""Batched sampling.

Every request in a batch can carry different temperature, top_k and top_p, so the
processing is vectorised over rows rather than looped: one sort, per-row thresholds.
Looping in Python works and then quietly dominates step time once the batch is large.

Hard rule: nothing in here reads a tensor value on the host. That means no ``.item()``,
no ``.cpu()``, and no ``bool(tensor)`` -- the last one is easy to write by accident and
syncs just as hard. Every branch is decided from ``params``, which is already Python.
"""

from __future__ import annotations

import torch
from torch import nn

from mini_vllm.request import SamplingParams


def apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    """Mask all but the top k per row. top_k: (N,) with -1 meaning disabled.

    A disabled row is handled arithmetically rather than branched on: k becomes the vocab
    size, so the threshold is the smallest logit and nothing is masked. Branching would
    mean reading a tensor on the host, which is a device sync.
    """
    vocab = logits.shape[-1]
    k = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab)).clamp(max=vocab)
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    threshold = sorted_logits.gather(1, (k - 1).unsqueeze(1))
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Nucleus filter per row. top_p: (N,) with 1.0 meaning disabled.

    top_p = 1.0 masks nothing on its own, since cumulative - probs never exceeds 1.
    """
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Subtracting probs keeps the token that crosses the threshold, so a row is never
    # left entirely masked even when one token holds more than top_p of the mass.
    remove = (cumulative - probs) > top_p.unsqueeze(1)
    remove = torch.zeros_like(remove).scatter_(-1, sorted_idx, remove)
    return logits.masked_fill(remove, float("-inf"))


class Sampler(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        params: list[SamplingParams],
        generators: list[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        """logits: (N, vocab_size) -> token ids: (N,)."""
        if logits.shape[0] != len(params):
            raise ValueError(f"got {logits.shape[0]} rows of logits for {len(params)} requests")
        generators = generators or [None] * len(params)
        device = logits.device
        logits = logits.to(torch.float32)

        def col(values, dtype):
            return torch.tensor(values, dtype=dtype, device=device)

        # Every fast-path decision is made from `params`, which is already host-side
        # Python. Deriving them from the logits instead would mean a `bool(tensor.all())`
        # per call -- correct, but a pipeline stall on every decode step.
        if all(p.is_greedy for p in params):
            return logits.argmax(dim=-1)

        temperature = col([p.temperature for p in params], torch.float32)
        greedy = temperature == 0.0

        # Divide by 1.0 for greedy rows so they are unaffected; their result is
        # overwritten by argmax at the end.
        logits = logits / torch.where(greedy, torch.ones_like(temperature), temperature).unsqueeze(1)
        if any(p.top_k > 0 for p in params):
            logits = apply_top_k(logits, col([p.top_k for p in params], torch.long))
        if any(p.top_p < 1.0 for p in params):
            logits = apply_top_p(logits, col([p.top_p for p in params], torch.float32))

        probs = torch.softmax(logits, dim=-1)
        out = torch.empty(logits.shape[0], dtype=torch.long, device=device)

        # Seeded requests draw from their own generator one row at a time, so their
        # output does not depend on which other requests happen to share the batch.
        # Everything else is drawn in a single call.
        unseeded = [i for i, g in enumerate(generators) if g is None]
        for i, gen in enumerate(generators):
            if gen is not None:
                out[i] = torch.multinomial(probs[i], num_samples=1, generator=gen).squeeze()
        if unseeded:
            idx = col(unseeded, torch.long)
            out[idx] = torch.multinomial(probs[idx], num_samples=1).squeeze(1)

        return torch.where(greedy, logits.argmax(dim=-1), out)
