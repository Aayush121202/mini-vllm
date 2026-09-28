"""Continuous batching.

The scheduling decision is remade every iteration rather than once per batch. A request
that finishes at step 40 frees its slot at step 40, and a request that arrives at step 41
joins the very next forward pass -- it does not wait for the rest of the batch to drain.
That is the entire idea, and it is where almost all of the throughput comes from.

Each step is either one prefill or a decode over every running request. Batched and
chunked prefill are deliberately not implemented; see the README.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mini_vllm.cache import PagedKVCache, PrefixCache
from mini_vllm.request import Request, RequestStatus


@dataclass
class SchedulerOutput:
    requests: list[Request]
    is_prefill: bool
    preempted: list[Request] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.requests)


class Scheduler:
    def __init__(
        self,
        cache: PagedKVCache,
        max_num_seqs: int = 32,
        static_batching: bool = False,
        enable_prefix_cache: bool = True,
    ) -> None:
        self.cache = cache
        self.block_size = cache.block_size
        self.max_num_seqs = max_num_seqs
        self.prefix_cache = (
            PrefixCache(cache.allocator, cache.block_size) if enable_prefix_cache else None
        )
        self.num_cached_tokens = 0
        # Ablation switch: when true, no new request is admitted until the running batch
        # has fully drained. This is the "before" picture in the benchmark.
        self.static_batching = static_batching

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.block_tables: dict[str, list[int]] = {}
        self.num_preemptions = 0

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, request_id: str) -> None:
        for req in list(self.running):
            if req.request_id == request_id:
                req.status = RequestStatus.FINISHED_ABORTED
                self.finish(req)
        self.waiting = deque(r for r in self.waiting if r.request_id != request_id)

    def finish(self, request: Request) -> None:
        if request in self.running:
            self.running.remove(request)
        blocks = self.block_tables.pop(request.request_id, [])
        # Blocks published to the prefix cache carry an extra reference, so freeing here
        # returns only the request's own claim and leaves the cached prefix intact.
        self.cache.allocator.free(blocks)

    def register_prefix(self, request: Request) -> None:
        """Publish a completed prefill so the next request with this prefix reuses it."""
        if self.prefix_cache is None:
            return
        table = self.block_tables.get(request.request_id)
        if table:
            self.prefix_cache.register(request.token_ids[: request.num_computed_tokens], table)

    def schedule(self) -> SchedulerOutput:
        admitted = self._try_admit()
        if admitted is not None:
            return SchedulerOutput([admitted], is_prefill=True)

        preempted = self._make_room()
        if not self.running:
            return SchedulerOutput([], is_prefill=False, preempted=preempted)
        return SchedulerOutput(list(self.running), is_prefill=False, preempted=preempted)

    # -- internals -----------------------------------------------------------------

    def _try_admit(self) -> Request | None:
        if not self.waiting:
            return None
        if self.static_batching and self.running:
            return None
        if len(self.running) >= self.max_num_seqs:
            return None

        req = self.waiting[0]
        matched = self._match_prefix(req)
        needed = self._blocks_for(req.num_tokens) - len(matched)
        if not self._reserve(needed):
            self.cache.allocator.free(matched)  # give the prefix references back
            return None

        self.waiting.popleft()
        self.block_tables[req.request_id] = matched + self.cache.allocator.allocate(needed)
        req.num_computed_tokens = len(matched) * self.block_size
        self.num_cached_tokens += req.num_computed_tokens
        req.status = RequestStatus.RUNNING
        self.running.append(req)
        return req

    def _match_prefix(self, request: Request) -> list[int]:
        if self.prefix_cache is None:
            return []
        matched = self.prefix_cache.match(request.token_ids)
        # Always leave at least one token to recompute: a forward pass over zero tokens
        # produces no logits, so there would be nothing to sample from.
        limit = (request.num_tokens - 1) // self.block_size
        if len(matched) > limit:
            self.cache.allocator.free(matched[limit:])
            matched = matched[:limit]
        return matched

    def _reserve(self, num_blocks: int) -> bool:
        """Make room for ``num_blocks``, evicting unreferenced cached prefixes first."""
        if self.cache.allocator.can_allocate(num_blocks):
            return True
        if self.prefix_cache is None:
            return False
        shortfall = num_blocks - self.cache.allocator.num_free
        self.prefix_cache.evict(shortfall)
        return self.cache.allocator.can_allocate(num_blocks)

    def _make_room(self) -> list[Request]:
        """Grow every running request's block table by one token's worth of space.

        When the pool is exhausted, evict the newest request and recompute it later.
        Newest-first keeps older requests moving, so latency stays bounded for the
        requests that have already waited longest. Recompute rather than swap-to-CPU:
        re-running a prefill is usually cheaper than a round trip over PCIe, and it is a
        great deal less code.
        """
        preempted: list[Request] = []
        i = 0
        while i < len(self.running):
            req = self.running[i]
            if self._ensure_capacity(req):
                i += 1
                continue
            victim = self.running.pop()
            self._preempt(victim)
            preempted.append(victim)
        return preempted

    def _ensure_capacity(self, request: Request) -> bool:
        table = self.block_tables[request.request_id]
        needed = self._blocks_for(request.num_tokens)
        if needed <= len(table):
            return True
        extra = needed - len(table)
        if not self._reserve(extra):
            return False
        table.extend(self.cache.allocator.allocate(extra))
        return True

    def _preempt(self, request: Request) -> None:
        self.cache.allocator.free(self.block_tables.pop(request.request_id, []))
        request.num_computed_tokens = 0  # everything must be recomputed
        request.status = RequestStatus.PREEMPTED
        self.waiting.appendleft(request)
        self.num_preemptions += 1

    def _blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size
