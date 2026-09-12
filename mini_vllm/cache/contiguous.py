"""Phase 0 KV cache: one dense ``[batch, max_seq_len, ...]`` tensor per layer.

Deliberately naive. It exists to be (a) the correctness oracle for the paged cache and
(b) the measurement that motivates it: a sequence of length L in a cache sized for
max_seq_len wastes ``(max_seq_len - L) / max_seq_len`` of its allocation. Run
``memory_report()`` before and after Phase 2 and put the delta in the README.
"""

from __future__ import annotations

import torch

from mini_vllm.cache.interface import AttnMetadata


class ContiguousKVCache:
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        shape = (num_layers, max_batch_size, max_seq_len, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, md: AttnMetadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = k.shape[0]
        if md.end_pos > self.max_seq_len:
            raise ValueError(
                f"sequence length {md.end_pos} exceeds cache capacity {self.max_seq_len}"
            )
        self.k_cache[layer_idx, :batch, md.start_pos : md.end_pos] = k
        self.v_cache[layer_idx, :batch, md.start_pos : md.end_pos] = v
        return (
            self.k_cache[layer_idx, :batch, : md.end_pos],
            self.v_cache[layer_idx, :batch, : md.end_pos],
        )

    def reset(self) -> None:
        # Zeroing is not required for correctness (reads are bounded by end_pos) but
        # makes debugging far less confusing.
        self.k_cache.zero_()
        self.v_cache.zero_()

    def memory_report(self, used_len: int) -> dict[str, float]:
        total = (self.k_cache.numel() + self.v_cache.numel()) * self.k_cache.element_size()
        used_frac = used_len / self.max_seq_len
        return {
            "allocated_mib": total / 1024**2,
            "useful_mib": total * used_frac / 1024**2,
            "wasted_mib": total * (1 - used_frac) / 1024**2,
            "utilization": used_frac,
        }
