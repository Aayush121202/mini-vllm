from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mini_vllm.cache.interface import AttnMetadata, KVCache
from mini_vllm.config import ModelConfig
from mini_vllm.layers.rotary import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, s, n_kv, d) -> (b, s, n_kv * n_rep, d) for grouped-query attention.

    Phase 2's Triton kernel does this gather implicitly and never materialises the
    expanded tensor; this version exists because sdpa needs matching head counts.
    """
    if n_rep == 1:
        return x
    b, s, n_kv, d = x.shape
    return x[:, :, :, None, :].expand(b, s, n_kv, n_rep, d).reshape(b, s, n_kv * n_rep, d)


class Attention(nn.Module):
    """Grouped-query attention.

    Note the signature: the cache is passed *in*. This module never allocates cache
    memory, which is what allows Phase 2 to swap a paged cache underneath it without
    touching the model definition.
    """

    def __init__(self, config: ModelConfig, layer_idx: int, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = config.num_queries_per_kv
        self.scaling = self.head_dim**-0.5
        self.rotary = rotary

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        md: AttnMetadata,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape

        q = self.q_proj(hidden).view(batch, seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim)

        q, k = self.rotary(q, k, positions)
        k_all, v_all = kv_cache.append(self.layer_idx, k, v, md)

        # sdpa wants (batch, heads, seq, head_dim).
        q = q.transpose(1, 2)
        k_all = repeat_kv(k_all, self.num_queries_per_kv).transpose(1, 2)
        v_all = repeat_kv(v_all, self.num_queries_per_kv).transpose(1, 2)

        # TODO(phase 3): `is_causal` is only correct for a full prefill starting at
        # position 0, or for single-token decode. Chunked prefill (start_pos > 0 with
        # seq > 1) needs an explicit mask over [start_pos, end_pos) x [0, end_pos).
        if md.is_prefill and md.start_pos != 0:
            raise NotImplementedError("chunked prefill needs an explicit mask; see phase 3")

        out = F.scaled_dot_product_attention(
            q, k_all, v_all, is_causal=md.is_prefill, scale=self.scaling
        )
        out = out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim)
        return self.o_proj(out)
