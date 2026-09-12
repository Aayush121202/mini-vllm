"""The KV cache interface every attention implementation talks to.

This indirection is the whole point of Phase 0: attention must never allocate or own
cache memory, or paging (Phase 2) would require rewriting the model layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class AttnMetadata:
    """Built once per engine step by the runner, threaded down to every layer.

    Phase 2 adds ``block_tables`` and ``slot_mapping`` here; Phase 3 adds ``cu_seqlens``
    for varlen packing. Nothing in the model layer changes when that happens.
    """

    is_prefill: bool
    start_pos: int  # position of the first new token within the sequence
    num_new_tokens: int

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_new_tokens


class KVCache(Protocol):
    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, md: AttnMetadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's K/V and return all K/V visible to the current query.

        Shapes are ``(batch, seq, num_kv_heads, head_dim)`` in and out.
        """
        ...

    def reset(self) -> None: ...
