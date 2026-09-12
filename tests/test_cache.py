from __future__ import annotations

import pytest
import torch

from mini_vllm.cache.contiguous import ContiguousKVCache
from mini_vllm.cache.interface import AttnMetadata


def make_cache(max_seq_len: int = 16) -> ContiguousKVCache:
    return ContiguousKVCache(
        num_layers=2,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def test_append_then_decode_returns_full_history():
    cache = make_cache()
    prefill_k = torch.randn(1, 5, 2, 4)
    prefill_v = torch.randn(1, 5, 2, 4)
    k_all, _ = cache.append(0, prefill_k, prefill_v, AttnMetadata(True, 0, 5))
    assert k_all.shape == (1, 5, 2, 4)

    step_k = torch.randn(1, 1, 2, 4)
    k_all, v_all = cache.append(0, step_k, torch.randn(1, 1, 2, 4), AttnMetadata(False, 5, 1))
    assert k_all.shape == (1, 6, 2, 4)
    torch.testing.assert_close(k_all[:, :5], prefill_k)
    torch.testing.assert_close(k_all[:, 5:], step_k)


def test_layers_are_independent():
    cache = make_cache()
    md = AttnMetadata(True, 0, 3)
    k0 = torch.ones(1, 3, 2, 4)
    cache.append(0, k0, k0, md)
    k1, _ = cache.append(1, torch.zeros(1, 3, 2, 4), torch.zeros(1, 3, 2, 4), md)
    assert k1.abs().sum() == 0


def test_overflow_raises():
    cache = make_cache(max_seq_len=4)
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        cache.append(0, torch.randn(1, 5, 2, 4), torch.randn(1, 5, 2, 4), AttnMetadata(True, 0, 5))


def test_memory_report_quantifies_waste():
    """The number that motivates Phase 2: a 128-token sequence in a 2048-slot cache
    wastes 94% of its allocation."""
    cache = make_cache(max_seq_len=2048)
    report = cache.memory_report(used_len=128)
    assert report["utilization"] == pytest.approx(0.0625)
    assert report["wasted_mib"] > report["useful_mib"] * 10
