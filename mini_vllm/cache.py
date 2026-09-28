"""Paged KV cache -- virtual memory, applied to attention.

A sequence's KV does not live in one contiguous span. It lives in fixed-size *blocks*
scattered through one big pool, and each sequence carries a *block table* mapping its
logical positions to physical block ids. That is exactly a page table, and it buys the
same thing paging buys an OS: no need to reserve a contiguous worst-case span per
sequence, so far more sequences fit in the same memory.

Attention never allocates cache memory -- it receives the cache and metadata as
arguments. That rule is what let this file be rewritten without touching model.py.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class AttnMetadata:
    """Built once per engine step and threaded down to every layer.

    Uniform ``num_new_tokens`` across the batch: a step is either one prefill (N=1, many
    tokens) or a decode over every running request (N=many, one token each).
    """

    block_tables: torch.Tensor  # (N, max_blocks) int64, padded with block 0
    slot_mapping: torch.Tensor  # (N * num_new_tokens,) int64, flat index into the pool
    positions: torch.Tensor  # (N, num_new_tokens) int64, absolute position of each token
    attn_mask: torch.Tensor  # (N, 1, num_new_tokens, max_blocks * block_size) bool
    num_new_tokens: int


class BlockAllocator:
    """Free list over physical block ids, with refcounts.

    Refcounts exist so two sequences can share a block. Nothing shares blocks yet --
    that is what prefix caching will use them for.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: list[int] = list(reversed(range(num_blocks)))
        self._refcount: dict[int, int] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise MemoryError(f"out of KV blocks: wanted {n}, {len(self._free)} free")
        blocks = [self._free.pop() for _ in range(n)]
        for b in blocks:
            self._refcount[b] = 1
        return blocks

    def incref(self, block_id: int) -> None:
        self._refcount[block_id] = self._refcount.get(block_id, 0) + 1

    def refcount(self, block_id: int) -> int:
        return self._refcount.get(block_id, 0)

    def free(self, block_ids: list[int]) -> None:
        """Drop one reference per id, returning blocks that hit zero to the free list."""
        for b in block_ids:
            count = self._refcount.get(b, 0)
            if count <= 1:
                self._refcount.pop(b, None)
                self._free.append(b)
            else:
                self._refcount[b] = count - 1


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.allocator = BlockAllocator(num_blocks)

    @property
    def bytes_per_block(self) -> int:
        per = self.block_size * self.num_kv_heads * self.head_dim * self.k_cache.element_size()
        return 2 * self.num_layers * per  # K and V

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def utilization(self) -> float:
        return self.allocator.num_used / self.num_blocks

    def write(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor):
        """Scatter new K/V into the pool.

        k, v: (num_tokens, num_kv_heads, head_dim). slot_mapping holds, for each token, a
        flat index ``block_id * block_size + offset`` into the layer's pool.
        """
        self.k_cache[layer_idx].view(-1, self.num_kv_heads, self.head_dim)[slot_mapping] = k
        self.v_cache[layer_idx].view(-1, self.num_kv_heads, self.head_dim)[slot_mapping] = v

    def gather(self, layer_idx: int, block_tables: torch.Tensor):
        """Collect each sequence's context into a dense tensor for attention.

        Returns (N, max_blocks * block_size, num_kv_heads, head_dim). Slots past a
        sequence's real context hold whatever those physical blocks contain; the
        attention mask excludes them.

        This copy is the price of doing paged attention in pure PyTorch. A fused kernel
        would read the blocks in place -- see the out-of-scope notes in the README.
        """
        k = self.k_cache[layer_idx][block_tables]  # (N, max_blocks, block_size, kv, d)
        v = self.v_cache[layer_idx][block_tables]
        return k.flatten(1, 2), v.flatten(1, 2)

    def reset(self) -> None:
        self.allocator = BlockAllocator(self.num_blocks)
        self.k_cache.zero_()
        self.v_cache.zero_()


def build_attn_metadata(
    block_tables: list[list[int]],
    start_positions: list[int],
    num_new_tokens: int,
    block_size: int,
    device: torch.device,
) -> AttnMetadata:
    """Assemble the per-step tensors once, rather than per layer.

    The mask rule is a single inequality: key j is visible to the query at absolute
    position p iff ``j <= p``. Causality within a prefill, the boundary at each
    sequence's context length, and the padding in short block tables all fall out of
    that one comparison -- there is no separate padding mask.
    """
    n = len(block_tables)
    max_blocks = max(len(t) for t in block_tables)
    padded = [t + [0] * (max_blocks - len(t)) for t in block_tables]
    block_tables_t = torch.tensor(padded, dtype=torch.long, device=device)

    start = torch.tensor(start_positions, dtype=torch.long, device=device).unsqueeze(1)
    offsets = torch.arange(num_new_tokens, dtype=torch.long, device=device).unsqueeze(0)
    positions = start + offsets  # (N, num_new_tokens)

    # Flat slot index per new token, in the same (N, num_new_tokens) row-major order the
    # model will flatten its K/V into.
    block_of = block_tables_t.gather(1, positions // block_size)
    slot_mapping = (block_of * block_size + positions % block_size).flatten()

    keys = torch.arange(max_blocks * block_size, dtype=torch.long, device=device)
    attn_mask = keys.view(1, 1, 1, -1) <= positions.view(n, 1, num_new_tokens, 1)

    return AttnMetadata(
        block_tables=block_tables_t,
        slot_mapping=slot_mapping,
        positions=positions,
        attn_mask=attn_mask,
        num_new_tokens=num_new_tokens,
    )


class PrefixCache:
    """Reuse KV blocks across requests that start with the same tokens.

    Two requests sharing a system prompt compute it twice for no reason. Hashing each
    *full* block of tokens -- chained, so a hash identifies a position as well as its
    contents -- makes that prefix addressable, and the block refcounts already in the
    allocator make it shareable.

    Only full blocks are cached: a partial block would be ambiguous, since its later
    slots are still unwritten. Blocks are registered after a prefill, so the win shows up
    on the *second* request with a given prefix.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        self.allocator = allocator
        self.block_size = block_size
        self.blocks: dict[int, int] = {}  # block hash -> physical block id
        self.order: OrderedDict[int, None] = OrderedDict()  # LRU over hashes
        self.num_hits = 0
        self.num_misses = 0

    def block_hashes(self, token_ids: list[int]) -> list[int]:
        """Chained hash per full block: position matters, not just content."""
        hashes, parent = [], 0
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            parent = hash((parent, tuple(token_ids[start : start + self.block_size])))
            hashes.append(parent)
        return hashes

    def match(self, token_ids: list[int]) -> list[int]:
        """Longest cached prefix, as block ids with a reference taken on each.

        The caller owns those references and must free them if it then fails to allocate
        the rest of the sequence.
        """
        matched = []
        for h in self.block_hashes(token_ids):
            block_id = self.blocks.get(h)
            if block_id is None:
                break
            self.allocator.incref(block_id)
            self.order.move_to_end(h)
            matched.append(block_id)
        if matched:
            self.num_hits += 1
        else:
            self.num_misses += 1
        return matched

    def register(self, token_ids: list[int], block_table: list[int]) -> None:
        """Publish the full blocks of a freshly computed prefix."""
        for i, h in enumerate(self.block_hashes(token_ids)):
            if i >= len(block_table):
                break
            if h in self.blocks:
                self.order.move_to_end(h)
                continue
            self.blocks[h] = block_table[i]
            self.allocator.incref(block_table[i])  # the cache itself holds a reference
            self.order[h] = None

    def evict(self, num_blocks: int) -> int:
        """Release up to ``num_blocks`` cached blocks, least recently used first.

        Blocks still referenced by a live request are skipped -- the refcount is the
        source of truth, so eviction can never pull memory out from under a sequence.
        """
        freed = 0
        for h in list(self.order):
            if freed >= num_blocks:
                break
            block_id = self.blocks[h]
            if self.allocator.refcount(block_id) > 1:
                continue
            del self.blocks[h]
            del self.order[h]
            self.allocator.free([block_id])
            freed += 1
        return freed
