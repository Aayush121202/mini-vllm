"""Continuous batching policy. Pure logic -- no model, no weights."""

from __future__ import annotations

import torch

from mini_vllm.cache import PagedKVCache
from mini_vllm.request import RequestStatus, Request, SamplingParams
from mini_vllm.scheduler import Scheduler


def make_scheduler(num_blocks: int = 8, block_size: int = 4, **kwargs) -> Scheduler:
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return Scheduler(cache, **kwargs)


def make_request(request_id: str, num_prompt_tokens: int = 4, max_tokens: int = 32) -> Request:
    return Request(request_id, list(range(num_prompt_tokens)), SamplingParams(max_tokens=max_tokens))


def complete_step(output) -> None:
    """Mimic what the engine does after a forward pass."""
    for req in output.requests:
        req.num_computed_tokens = req.num_tokens
        req.append_token(99)
        req.check_stop()


def test_first_schedule_is_a_prefill_for_one_request():
    sched = make_scheduler()
    sched.add(make_request("a"))
    out = sched.schedule()
    assert out.is_prefill and [r.request_id for r in out.requests] == ["a"]
    assert sched.running[0].status is RequestStatus.RUNNING
    assert len(sched.block_tables["a"]) == 1  # 4 prompt tokens, block_size 4


def test_decode_batches_every_running_request():
    sched = make_scheduler()
    for name in ("a", "b"):
        sched.add(make_request(name))
    complete_step(sched.schedule())  # prefill a
    complete_step(sched.schedule())  # prefill b
    out = sched.schedule()
    assert not out.is_prefill
    assert [r.request_id for r in out.requests] == ["a", "b"]


def test_new_request_joins_mid_generation():
    """The point of continuous batching: 'b' does not wait for 'a' to finish."""
    sched = make_scheduler()
    sched.add(make_request("a", max_tokens=50))
    complete_step(sched.schedule())
    complete_step(sched.schedule())  # 'a' is several tokens in

    sched.add(make_request("b"))
    out = sched.schedule()
    assert out.is_prefill and out.requests[0].request_id == "b"
    complete_step(out)
    assert [r.request_id for r in sched.schedule().requests] == ["a", "b"]


def test_static_batching_makes_arrivals_wait():
    sched = make_scheduler(static_batching=True)
    sched.add(make_request("a", max_tokens=50))
    complete_step(sched.schedule())
    sched.add(make_request("b"))

    out = sched.schedule()
    assert not out.is_prefill  # 'b' is not admitted while 'a' runs
    assert [r.request_id for r in out.requests] == ["a"]


def test_max_num_seqs_caps_the_batch():
    sched = make_scheduler(max_num_seqs=1)
    sched.add(make_request("a", max_tokens=50))
    sched.add(make_request("b"))
    complete_step(sched.schedule())
    assert not sched.schedule().is_prefill
    assert len(sched.running) == 1


def test_exhausted_pool_preempts_the_newest_request():
    sched = make_scheduler(num_blocks=2, block_size=4)
    sched.add(make_request("a"))
    sched.add(make_request("b"))
    complete_step(sched.schedule())  # 'a' takes the first block
    complete_step(sched.schedule())  # 'b' takes the second; pool is now full

    # Both now need a 5th token slot, and there is nothing left to hand out.
    out = sched.schedule()
    assert [r.request_id for r in out.preempted] == ["b"]
    assert [r.request_id for r in out.requests] == ["a"]
    assert sched.num_preemptions == 1


def test_preempted_request_is_requeued_for_full_recompute():
    sched = make_scheduler(num_blocks=2, block_size=4)
    sched.add(make_request("a"))
    sched.add(make_request("b"))
    complete_step(sched.schedule())
    complete_step(sched.schedule())
    sched.schedule()

    victim = sched.waiting[0]
    assert victim.request_id == "b"
    assert victim.status is RequestStatus.PREEMPTED
    assert victim.num_computed_tokens == 0  # KV was dropped; recompute on resume
    assert victim.output_token_ids == [99]  # generated tokens are kept
    assert "b" not in sched.block_tables


def test_finishing_returns_blocks_to_the_pool():
    sched = make_scheduler()
    sched.add(make_request("a", max_tokens=1))
    out = sched.schedule()
    complete_step(out)
    assert out.requests[0].status.is_finished

    used_before = sched.cache.allocator.num_used
    sched.finish(out.requests[0])
    assert sched.cache.allocator.num_used == used_before - 1
    assert not sched.has_unfinished()


def test_second_request_with_the_same_prompt_skips_prefill_work():
    sched = make_scheduler(num_blocks=16, block_size=4)
    first = make_request("a", num_prompt_tokens=12)
    sched.add(first)
    out = sched.schedule()
    complete_step(out)
    sched.register_prefix(first)

    second = Request("b", list(range(12)), SamplingParams())
    sched.add(second)
    sched.schedule()
    # 12 prompt tokens = 3 full blocks, but one token must be left to recompute,
    # so 2 blocks are reused and 8 tokens of prefill are skipped.
    assert second.num_computed_tokens == 8
    assert sched.num_cached_tokens == 8


def test_prefix_cache_can_be_disabled():
    sched = make_scheduler(num_blocks=16, block_size=4, enable_prefix_cache=False)
    first = make_request("a", num_prompt_tokens=12)
    sched.add(first)
    complete_step(sched.schedule())
    sched.register_prefix(first)

    second = Request("b", list(range(12)), SamplingParams())
    sched.add(second)
    sched.schedule()
    assert second.num_computed_tokens == 0
