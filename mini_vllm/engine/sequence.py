"""Request state. Deliberately free of any tensor or device concepts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    # Phase 3: SWAPPED / PREEMPTED land here.
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class SamplingParams:
    temperature: float = 0.0  # 0.0 means greedy
    top_p: float = 1.0
    top_k: int = -1  # -1 disables
    max_tokens: int = 64
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or a positive integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    # Number of tokens whose KV is already in the cache. Phase 3 (chunked prefill)
    # advances this by less than the full prompt length.
    num_computed_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def check_stop(self) -> None:
        """Token-level stop conditions. String-level stops live in the detokenizer (Phase 1)."""
        if self.output_token_ids and self.last_token_id in self.sampling_params.stop_token_ids:
            self.status = SequenceStatus.FINISHED_STOPPED
        elif len(self.output_token_ids) >= self.sampling_params.max_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH


@dataclass
class RequestOutput:
    """One iteration's worth of output for one request."""

    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]  # everything generated so far
    new_token_ids: list[int]  # delta produced by this step
    finished: bool
    finish_reason: str | None = None

    @classmethod
    def from_seq(cls, seq: Sequence, new_token_ids: list[int]) -> "RequestOutput":
        reason = None
        if seq.status.is_finished:
            reason = {
                SequenceStatus.FINISHED_STOPPED: "stop",
                SequenceStatus.FINISHED_LENGTH: "length",
                SequenceStatus.FINISHED_ABORTED: "abort",
            }[seq.status]
        return cls(
            request_id=seq.request_id,
            prompt_token_ids=list(seq.prompt_token_ids),
            token_ids=list(seq.output_token_ids),
            new_token_ids=new_token_ids,
            finished=seq.status.is_finished,
            finish_reason=reason,
        )
