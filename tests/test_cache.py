"""Paged KV cache: allocator, scatter/gather round trip, and metadata."""

from __future__ import annotations

import pytest
import torch

from mini_vllm.cache import BlockAllocator, PagedKVCache, build_attn_metadata


def make_cache(num_blocks: int = 8, block_size: int = 4) -> PagedKVCache:
    return PagedKVCache(
        num_layers=2,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


# -- allocator ------------------------------------------------------------------


def test_allocate_and_free_round_trips():
    alloc = BlockAllocator(4)
    assert alloc.num_free == 4
    blocks = alloc.allocate(3)
    assert len(set(blocks)) == 3
    assert alloc.num_free == 1 and alloc.num_used == 3
    alloc.free(blocks)
    assert alloc.num_free == 4


def test_allocation_beyond_capacity_raises():
    alloc = BlockAllocator(2)
    assert not alloc.can_allocate(3)
    with pytest.raises(MemoryError, match="out of KV blocks"):
        alloc.allocate(3)


def test_shared_block_survives_one_free():
    """Refcounts are what will let two sequences share a cached prefix."""
    alloc = BlockAllocator(2)
    (block,) = alloc.allocate(1)
    alloc.incref(block)
    alloc.free([block])
    assert alloc.num_free == 1  # still held by the second reference
    alloc.free([block])
    assert alloc.num_free == 2


# -- cache ----------------------------------------------------------------------


def test_write_then_gather_returns_what_was_written():
    cache = make_cache()
    table = cache.allocator.allocate(2)  # 8 token slots
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    slots = torch.tensor(
        [table[i // 4] * 4 + i % 4 for i in range(5)], dtype=torch.long
    )
    cache.write(0, k, v, slots)

    k_all, v_all = cache.gather(0, torch.tensor([table], dtype=torch.long))
    assert k_all.shape == (1, 8, 2, 4)
    torch.testing.assert_close(k_all[0, :5], k)
    torch.testing.assert_close(v_all[0, :5], v)


def test_layers_are_independent():
    cache = make_cache()
    table = torch.tensor([cache.allocator.allocate(1)], dtype=torch.long)
    slots = torch.tensor([table[0, 0].item() * 4], dtype=torch.long)
    cache.write(0, torch.ones(1, 2, 4), torch.ones(1, 2, 4), slots)
    k_all, _ = cache.gather(1, table)
    assert k_all.abs().sum() == 0


def test_blocks_need_not_be_contiguous():
    """The point of paging: a sequence's KV can be scattered anywhere in the pool."""
    cache = make_cache()
    a = cache.allocator.allocate(3)
    cache.allocator.free([a[1]])  # punch a hole
    b = cache.allocator.allocate(1)
    table = [a[0], b[0], a[2]]
    assert sorted(table) != table or len(set(table)) == 3

    k = torch.randn(12, 2, 4)
    slots = torch.tensor(
        [table[i // 4] * 4 + i % 4 for i in range(12)], dtype=torch.long
    )
    cache.write(0, k, k, slots)
    k_all, _ = cache.gather(0, torch.tensor([table], dtype=torch.long))
    torch.testing.assert_close(k_all[0], k)


# -- metadata -------------------------------------------------------------------


def test_slot_mapping_follows_the_block_table():
    md = build_attn_metadata(
        block_tables=[[5, 2]], start_positions=[0], num_new_tokens=6, block_size=4, device="cpu"
    )
    # positions 0-3 land in block 5, positions 4-5 in block 2
    assert md.slot_mapping.tolist() == [20, 21, 22, 23, 8, 9]


def test_mask_is_causal_during_prefill():
    md = build_attn_metadata(
        block_tables=[[0]], start_positions=[0], num_new_tokens=3, block_size=4, device="cpu"
    )
    assert md.attn_mask.shape == (1, 1, 3, 4)
    assert md.attn_mask[0, 0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
    ]


def test_mask_bounds_each_row_at_its_own_context_length():
    """Decode step: two sequences at different lengths share one batch."""
    md = build_attn_metadata(
        block_tables=[[0, 1], [2, 3]],
        start_positions=[2, 6],
        num_new_tokens=1,
        block_size=4,
        device="cpu",
    )
    assert md.attn_mask[0, 0, 0].tolist() == [True] * 3 + [False] * 5
    assert md.attn_mask[1, 0, 0].tolist() == [True] * 7 + [False]


def test_mask_never_leaves_a_row_fully_masked():
    """An all-False row makes softmax produce NaN. Position 0 must always be visible."""
    md = build_attn_metadata(
        block_tables=[[0, 1]], start_positions=[0], num_new_tokens=1, block_size=4, device="cpu"
    )
    assert bool(md.attn_mask.any(dim=-1).all())


def test_shorter_block_tables_are_padded():
    md = build_attn_metadata(
        block_tables=[[1], [2, 3]], start_positions=[0, 4], num_new_tokens=1,
        block_size=4, device="cpu",
    )
    assert md.block_tables.shape == (2, 2)
    assert md.block_tables[0].tolist() == [1, 0]  # padded with block 0, masked out
    assert not bool(md.attn_mask[0, 0, 0, 4:].any())
