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

from mini_vllm.cache import PagedKVCache
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
        self, cache: PagedKVCache, max_num_seqs: int = 32, static_batching: bool = False
    ) -> None:
        self.cache = cache
        self.block_size = cache.block_size
        self.max_num_seqs = max_num_seqs
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
        self.cache.allocator.free(blocks)

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
        needed = self._blocks_for(req.num_tokens)
        if not self.cache.allocator.can_allocate(needed):
            return None

        self.waiting.popleft()
        self.block_tables[req.request_id] = self.cache.allocator.allocate(needed)
        req.status = RequestStatus.RUNNING
        self.running.append(req)
        return req

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
        if not self.cache.allocator.can_allocate(extra):
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
