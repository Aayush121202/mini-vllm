"""Turns scheduler decisions into forward passes.

Phase 0 handles exactly one sequence at a time. Phase 1 batches; Phase 3 flattens the
batch into varlen form. The signatures here are already shaped for that: ``execute``
takes token ids plus positions, not a Sequence.
"""

from __future__ import annotations

import torch

from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.interface import AttnMetadata
from mini_vllm.config import CacheConfig, ModelConfig, parse_dtype
from mini_vllm.models.loader import build_model
from mini_vllm.runner.backend import describe, get_device


class ModelRunner:
    def __init__(
        self,
        config: ModelConfig,
        cache_config: CacheConfig | None = None,
        device: str | torch.device = "auto",
        dtype: str | torch.dtype = "auto",
    ) -> None:
        self.config = config
        self.cache_config = cache_config or CacheConfig()
        self.device = get_device(device)
        self.dtype = parse_dtype(dtype, self.device)
        self.model = build_model(config, self.device, self.dtype)
        self.kv_cache = ContiguousKVCache(
            num_layers=config.num_hidden_layers,
            max_batch_size=self.cache_config.max_batch_size,
            max_seq_len=self.cache_config.max_seq_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def __repr__(self) -> str:
        return (
            f"ModelRunner({self.config.architecture} on {describe(self.device)}, "
            f"dtype={self.dtype})"
        )

    @torch.inference_mode()
    def execute(
        self,
        token_ids: list[int],
        start_pos: int,
        *,
        last_token_only: bool = True,
    ) -> torch.Tensor:
        """Run one forward pass over ``token_ids`` starting at ``start_pos``.

        Returns logits of shape (seq, vocab), or (1, vocab) when last_token_only.
        """
        is_prefill = len(token_ids) > 1
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + len(token_ids), dtype=torch.long, device=self.device
        )
        md = AttnMetadata(
            is_prefill=is_prefill, start_pos=start_pos, num_new_tokens=len(token_ids)
        )
        logits = self.model(
            input_ids, positions, self.kv_cache, md, last_token_only=last_token_only
        )
        return logits.squeeze(0)

    def reset(self) -> None:
        self.kv_cache.reset()
