from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF's half-split convention (NOT the interleaved one from the RoFormer paper).

    Getting this wrong produces a model that still emits fluent-looking text while being
    subtly wrong, which is why parity tests exist.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables. Buffers are non-persistent, so state_dict stays clean
    and the module can be shared across layers without affecting weight loading.

    Tables are built by ambient factory calls, so they land on whatever device the model
    is being constructed under, and are pinned to fp32 regardless of the model dtype --
    fp16 rope tables measurably degrade long-context positions.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (batch, seq, num_heads, head_dim); positions: (seq,)."""
        cos = self.cos_cached[positions].to(q.dtype).unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[positions].to(q.dtype).unsqueeze(0).unsqueeze(2)
        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out
